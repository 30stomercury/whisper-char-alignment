import os
import numpy as np
from tqdm import tqdm
import torch
import argparse
from metrics import eval_n1, get_seg_metrics, dtw_timestamp
from dataset import TIMIT, LibriSpeech, AMI, Collate
from timing import get_attentions, force_align, filter_attention
from retokenize import encode, remove_punctuation

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

MAX_FRAMES = 1500
MAX_LENGTH = 448

DATASET = {"TIMIT": TIMIT, "LibriSpeech": LibriSpeech, "AMI": AMI}

def infer_dataset(args):
    tolerance = args.tolerance

    # model
    model = whisper.load_model(args.model)
    model.to(DEVICE)

    # decode the audio
    options = whisper.DecodingOptions(language="en")
    tokenizer = get_tokenizer(model.is_multilingual, language='English')

    # basically paremeters to do denoising
    medfilt_width = args.medfilt_width
    qk_scale = 1.0
    dataset = DATASET[args.dataset](args.scp, n_mels=args.n_mels, device=model.device)

    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    # decode the audio
    options = whisper.DecodingOptions(language="en")

    corrects = 0
    total_preds = 0
    total_gts = 0
    selected = []
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        # print the recognized text
        #result = whisper.decode(model, mels, options)
        #transcription = result.text
        mels = mels.to(model.device)
        result = whisper.decode(model, mels, options)
        transcription = result.text
        print(texts)
        print(transcription)
        
        transcription = remove_punctuation(transcription)
        if len(transcription) == '':
            transcription = ' '

        text_tokens = encode(transcription, tokenizer, args.aligned_unit_type)
        tokens = torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.no_timestamps,
                *text_tokens,
                tokenizer.eot,
            ]
        ).to(model.device)

        # Get attention maps
        max_frames = durations // AUDIO_SAMPLES_PER_TOKEN
        if max_frames > MAX_FRAMES or len(tokens) > MAX_LENGTH:
            print(fids)
            continue

        w, logits = get_attentions(mels, tokens, model, tokenizer, max_frames, medfilt_width, qk_scale)

        ws, scores = filter_attention(w, topk=150)
        print(texts.split())
        print(transcription.split())
        candidates = []
        # best_score = -1
        best_score = float('inf')
        best_ends_hat = None
        best_head = None
        for w, score in zip(ws, scores):
            results = force_align(w.unsqueeze(0), text_tokens, tokenizer,
                    aligned_unit_type=args.aligned_unit_type, aggregation="mean", topk=15)
            cost, _ = dtw_timestamp(ends, results[2])
            # # collect predicted boundaries
            # ends_hat = results[2]
            # correct_pred, _ = eval_n1(ends, ends_hat, tolerance)
            # precision, recall, f1, r_value, os = \
            #         get_seg_metrics(correct_pred, correct_pred, len(ends_hat), len(ends))
            # if f1 > best_score:
            #     best_score = f1
            #     best_ends_hat = results[2]
            #     best_head = score + (f1, )

            # not used now but maybe useful for topk
            # candidates.append(ends_hat)
            if cost < best_score:
                best_score = cost
                best_ends_hat = results[2]
                best_head = score
        selected.append(str(best_head))
        # eval
        total_gts += len(ends)
        total_preds += len(best_ends_hat)
        correct_pred, _ = eval_n1(ends, best_ends_hat, tolerance)
        corrects += correct_pred

        print(ends)
        print(best_ends_hat)

    precision, recall, f1, r_value, os = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    print(precision, recall, f1, r_value)

    with open(f"{args.output_dir}/{args.dataset}-{args.aligned_unit_type}-best.txt", 'w') as f:
        f.write('\n'.join(selected))
    f.close()


def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for whisper-based forced alignments")
    parser.add_argument('--model', type=str, default='medium')
    parser.add_argument('--dataset', type=str, default="TIMIT", choices=["TIMIT", "LibriSpeech", "AMI"])
    parser.add_argument('--scp', type=str, default="scp/test.wav.scp")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Path to the output directory", required=True)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--medfilt_width', type=int, default=7)
    parser.add_argument('--aggr', type=str, default="mean", choices=["mean", "topk"])
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--aligned_unit_type', type=str, default='subword', choices=["subword", "char"])
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--plot', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
    AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

    infer_dataset(args)
