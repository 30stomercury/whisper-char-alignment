import os
os.environ["HF_HOME"] = "/home/s2522924/.cache/huggingface/"
import re
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from timing_nemo import force_align, filter_attention
from metrics import eval_n1, eval_n1_strict, get_seg_metrics
from retokenize_nemo import remove_punctuation, encode
from dataset import TIMIT, LibriSpeech, Collate
from whisper.timing import median_filter
from nemo.collections.asr.models import EncDecMultiTaskModel
from collections import defaultdict
import joblib

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

DATASET = {"TIMIT": TIMIT, "LibriSpeech": LibriSpeech}

def infer_dataset(args):
    tolerance = args.tolerance

    # load model
    model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b', map_location=DEVICE)
    model.eval()
    print(model.transf_decoder)
    print(model.device)
        
    # update dcode params
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)
    tokenizer = model.tokenizer

    dataset = DATASET[args.dataset](args.scp, n_mels=args.n_mels, device=model.device)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    corrects = 0
    total_preds = 0
    total_gts = 0
    all_predictions = defaultdict(int)
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
        #     batch_size=1,
        #     pnc="no"
        # )[0]
        transcription = results.text
        text_tokens = results.y_sequence      
        
        print(f"ground truth: {texts}") 
        # print(results)
        print(f"hypothesis: {transcription}")
        print(text_tokens)
        print(tokenizer.ids_to_tokens(text_tokens.detach().cpu().tolist()))
        # print(tokenizer.ids_to_tokens([ 3,  4,  8,  4, 10, 2]))
        # # ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|en|>', '<|nopnc|>', '<|endoftext|>']
        
        QKs = [torch.tensor([]).to(model.device) for _ in  range(len(model.transf_decoder._decoder.layers))]
        
        def get_attn(index):
            def hook(module, ins, outs):
                print(outs[-1].shape)
                # QKs[index] = torch.cat((QKs[index], outs[-1]), dim=-2).to(model.device)
                QKs[index] = outs[-1]
            return hook
        hooks = [
            # second_sub_layer is for cross_attn
            block.second_sub_layer.register_forward_hook(get_attn(i))
            for i, block in enumerate(model.transf_decoder._decoder.layers)
        ]

        input_signal = audios[:durations].unsqueeze(0).to(model.device)
        input_signal_length = torch.tensor([input_signal.shape[1]]).to(model.device)
        prompts = [3,  4,  8,  4, 10]

        texts = remove_punctuation(texts)
        transcription = remove_punctuation(transcription)
        if len(transcription) == '':
            transcription = ' '
        input_tokens = encode(transcription, tokenizer, args.aligned_unit_type)
        print(len(input_tokens))
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
                
        words, start_times, end_times, ws, scores = force_align(weights, input_tokens, tokenizer, 
                   aligned_unit_type=args.aligned_unit_type, aggregation=args.aggr, topk=args.topk)


        ends_hat = end_times
        print(ends_hat)
        print(ends)
        if args.save_prediction:
            all_predictions[n] = dict(starts=starts, ends=ends, texts=texts.split(), 
                starts_hat=start_times, ends_hat=ends_hat, predwords=words, fids=fids)
            
        # eval
        if not args.strict:
            correct_pred, _ = eval_n1(ends, ends_hat, tolerance)
            total_gts += len(ends)
            total_preds += len(ends_hat)
            corrects += correct_pred
        else:
            #tp, fp, fn = eval_n1_strict(ends, ends_hat, texts.split(), words[:-1], tolerance)
            words = ' '.join(words[:-1]).split()
            tp, fp, fn = eval_n1_strict(ends, ends_hat, texts.split(), words, tolerance)
            corrects += tp
            total_gts += (tp + fn)
            total_preds += (tp + fp)
        
        if n == 10:
            break
        
    precision, recall, f1, r_value, _ = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    results = dict(precision=precision, recall=recall, f1=f1, r_value=r_value)
    print(precision, recall, f1, r_value)

    # dump results
    ts = time.time()
    filename = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    results = {**vars(args), **results}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f"{args.output_dir}/canary-{filename}.json", 'w') as f:
        json.dump(results, f)
    if args.save_prediction:
        if args.aggr == 'topk':
            joblib.dump(all_predictions, f"{args.output_dir}/canary-{args.dataset}-{args.aligned_unit_type}-top{args.topk}.pkl")
        else:
            joblib.dump(all_predictions, f"{args.output_dir}/canary-{args.dataset}-{args.aligned_unit_type}-{args.aggr}.pkl")


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
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--strict', action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    infer_dataset(args)
