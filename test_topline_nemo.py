import os
os.environ["HF_HOME"] = "/home/s2522924/.cache/huggingface"
import re
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from timing_nemo import force_align, filter_attention
from metrics import eval_n1, eval_n1_strict, get_seg_metrics, dtw_timestamp
from retokenize_nemo import normalizer, encode, remove_punctuation
from dataset import TIMIT, LibriSpeech, Collate
from whisper.timing import median_filter
from nemo.collections.asr.models import EncDecMultiTaskModel

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def infer_dataset(args):
    tolerance = args.tolerance

    # load model
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b', map_location=DEVICE)
        
    # update decode params
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    tokenizer = model.tokenizer
    
    # dataset
    dataset = eval(args.dataset)(args.scp, n_mels=args.n_mels, device=model.device)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    corrects = 0
    total_preds = 0
    total_gts = 0
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        results = model.transcribe(
                audio=audios[:durations].to(model.device),
                batch_size=1,  # batch size to run the inference with,
                source_lang="en",
                target_lang="en",
                task="asr",
                pnc="no"
        )[0]

        # results = model.transcribe(
        #     audio=fids,
        #     batch_size=1,  # batch size to run the inference with,
        #     taskname="asr",
        #     pnc="no"
        # )[0]
        transcription = results.text
        text_tokens = results.y_sequence        
                
        print(transcription)
        print(text_tokens)
        print(tokenizer.ids_to_tokens(text_tokens.detach().cpu().tolist()))
        # print(tokenizer.ids_to_tokens([ 3,  4,  8,  4, 10, 2]))
        # # ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|en|>', '<|nopnc|>', '<|endoftext|>']

        texts = remove_punctuation(texts)
        transcription = remove_punctuation(transcription)
        if len(transcription) == '':
            transcription = ' '

        QKs = [torch.tensor([]).to(model.device) for _ in  range(len(model.transf_decoder._decoder.layers))]
        
        def get_attn(index):
            def hook(module, ins, outs):
                QKs[index] = torch.cat((QKs[index], outs[-1]), dim=-2).to(model.device)
            return hook
        hooks = [
            block.second_sub_layer.register_forward_hook(get_attn(i))
            for i, block in enumerate(model.transf_decoder._decoder.layers)
        ]
        
        input_signal = audios[:durations].unsqueeze(0).to(model.device)
        input_signal_length = torch.tensor([input_signal.shape[1]]).to(model.device)
        prompts = [3,  4,  8,  4, 10]
        input_tokens = encode(transcription, tokenizer, args.aligned_unit_type)
        transcript = torch.tensor(prompts + input_tokens + [2]).unsqueeze(0).to(model.device)
        transcript_length = torch.tensor([transcript.shape[1]]).to(model.device)
        
        model.freeze()
        res = model(input_signal=input_signal, 
                    input_signal_length=input_signal_length,
                    transcript=transcript,
                    transcript_length=transcript_length
                    )
        model.unfreeze()

        qks = torch.cat(QKs)
        print(qks.shape)

        for hook in hooks:
            hook.remove()

        weights = median_filter(qks.detach().cpu(), args.medfilt_width)
        weights = weights.softmax(dim=-1)
        weights = weights / weights.norm(dim=-2, keepdim=True)
        
        ws, scores = filter_attention(weights, topk=180)
        candidates = []
        best_score = -1
        # best_score = float('inf')
        best_ends_hat = None
        for w, score in zip(ws, scores):
            results = force_align(w.unsqueeze(0), input_tokens, tokenizer, 
                    aligned_unit_type=args.aligned_unit_type, aggregation="mean", topk=15)
            # cost, _ = dtw_timestamp(ends, results[2])
            # collect predicted boundaries
            ends_hat = results[2]
            words = ' '.join(results[0][:-1]).split()
            tp, fp, fn  = eval_n1_strict(ends, ends_hat, texts.split(), words, tolerance)
            correct_pred = tp
            correct_pred = tp
            total_gt = (tp + fn)
            total_pred = (tp + fp)
            precision, recall, f1, r_value, os = \
                get_seg_metrics(correct_pred, correct_pred, total_pred, total_gt)
            # print(score, f1)
            if f1 > best_score:
                best_score = f1
                best_ends_hat = results[2]

            # not used now but maybe useful for topk
            # candidates.append(ends_hat)
            # if cost < best_score:
            #     best_score = cost
            #     best_ends_hat = results[2]
            #     best_head = score
        
        # eval
        # total_gts += len(ends)
        # total_preds += len(best_ends_hat)
        # correct_pred, _ = eval_n1(ends, best_ends_hat, tolerance)
        # corrects += correct_pred
        tp, fp, fn = eval_n1_strict(ends, best_ends_hat, texts.split(), words, tolerance)
        total_gts += (tp + fn)
        total_preds += (tp + fp)
        corrects += tp

    precision, recall, f1, r_value, os = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    print(precision, recall, f1, r_value)

def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for NeMo model forced alignments")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Path to the output directory", required=True)
    parser.add_argument('--dataset', type=str, default="TIMIT", choices=["TIMIT", "LibriSpeech"])
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--scp', type=str, default="scp/test.wav.scp")
    parser.add_argument('--medfilt_width', type=int, default=3)
    parser.add_argument('--aggr', type=str, default="mean", choices=["mean", "topk"])
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--aligned_unit_type', type=str, default='subword', choices=["subword", "char"])
    parser.add_argument('--tolerance', type=float, default=0.02)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    infer_dataset(args)
