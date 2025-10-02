import os
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch

from metrics import eval_n1, get_seg_metrics, eval_n1_strict
from dataset import TIMIT, LibriSpeech, Collate
from timing import get_attentions, force_align, filter_attention
from retokenize import encode, remove_punctuation
from plot import plot_attns

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

MAX_FRAMES = 1500
MAX_LENGTH = 448

DATASET = {"TIMIT": TIMIT, "LibriSpeech": LibriSpeech}

def infer_dataset(args):
    print(args)
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
    if_include_best = 0
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        if len(texts.split()) < 18:
            continue

        mels = mels.to(model.device)
        result = whisper.decode(model, mels, options)
        transcription = result.text

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
        ws, scores = filter_attention(w, topk=360)
        candidates = []
        best_score = -1
        best_ends_hat = None
        best_head = None
        for w, score in zip(ws, scores):
            results = force_align(w.unsqueeze(0), text_tokens, tokenizer,
                    aggregation="mean", topk=1, aligned_unit_type=args.aligned_unit_type)

            # collect predicted boundaries
            ends_hat = results[2]
            words = results[0]
            words = ' '.join(words[:-1]).split()
            tp, fp, fn = eval_n1_strict(ends, best_ends_hat, texts.split(), words, tolerance)
            precision, recall, f1, r_value, _ = \
                    get_seg_metrics(correct_pred, correct_pred, len(ends_hat), len(ends))

            if f1 >= best_score:
                best_score = f1
                best_ends_hat = results[2]
                best_head = score[0]

            # not used now but maybe useful for topk
            candidates.append(ends_hat)

        if best_head > scores[-args.if_include_within][0]:
            if_include_best += 1
        # eval
        if not args.strict:
            correct_pred, _ = eval_n1(ends, best_ends_hat, tolerance)
            total_gts += len(ends)
            total_preds += len(best_ends_hat)
            corrects += correct_pred
        else:
            words = results[0]
            words = ' '.join(words[:-1]).split()
            tp, fp, fn = eval_n1_strict(ends, best_ends_hat, texts.split(), words, tolerance)
            corrects += tp
            total_gts += (tp + fn)
            total_preds += (tp + fp)

    precision, recall, f1, r_value, _ = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    results = dict(
            precision=precision, 
            recall=recall, f1=f1, r_value=r_value, 
            include_best_rate=if_include_best/len(loader))

    # dump results
    ts = time.time()
    filename = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    results = {**vars(args), **results}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f"{args.output_dir}/{filename}.json", 'w') as f:
        json.dump(results, f)


def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for whisper-based forced alignments")
    parser.add_argument('--model', type=str, default='medium')
    parser.add_argument('--dataset', type=str, default="TIMIT", choices=["TIMIT", "LibriSpeech"])
    parser.add_argument('--scp', type=str, default="scp/test.wav.scp")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Path to the output directory", required=True)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--medfilt_width', type=int, default=7)
    parser.add_argument('--if_include_within', type=int, default=10)
    parser.add_argument('--aggr', type=str, default="mean", choices=["mean", "topk"])
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--aligned_unit_type', type=str, default='subword', choices=["subword", "char"])
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--strict', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
    AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

    infer_dataset(args)
