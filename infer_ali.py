import os
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from collections import defaultdict
import joblib

from metrics import eval_n1, get_seg_metrics, eval_n1_strict
from dataset import TIMIT, LibriSpeech, Collate
from timing import get_attentions, force_align, filter_attention, default_find_alignment
from retokenize import encode, remove_punctuation
from plot import plot_attn

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

MAX_FRAMES = 1500
MAX_LENGTH = 448

# We remove AMI in the main branch and keep that part in the dev branch.
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

    options = whisper.DecodingOptions(language="en")

    # decode the audio
    corrects = 0
    total_preds = 0
    total_gts = 0
    all_predictions = defaultdict(int)
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):

        mels = mels.to(model.device)
        result = whisper.decode(model, mels, options)
        transcription = result.text

        texts = remove_punctuation(texts)
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

        max_frames = durations // AUDIO_SAMPLES_PER_TOKEN
        if max_frames > MAX_FRAMES or len(tokens) > MAX_LENGTH:
            print(fids)
            continue
        
        if args.default_whisper_timing:
            words, start_times, end_times, ws, scores = default_find_alignment(
                    model, tokenizer, text_tokens, mels, max_frames)
        else:
            kwargs = dict(
                w_colnorm=args.w_colnorm,
                w_rownorm=args.w_rownorm,
                w_coverage=args.w_coverage
            )
            # Get attention maps
            attn_w, logits = get_attentions(mels, tokens, model, tokenizer, max_frames, medfilt_width, qk_scale)
            words, start_times, end_times, ws, scores = force_align(
                attn_w, text_tokens, 
                tokenizer, 
                aligned_unit_type=args.aligned_unit_type, 
                aggregation=args.aggr, 
                topk=args.topk, 
                **kwargs
            )
        if args.plot:
            plot_attn(
                ws,
                text_tokens,
                tokenizer,
                gt_alignment=ends,
                pred_alignment=end_times,
                fid=fids, 
                aligned_unit_type=args.aligned_unit_type,
                path=f'{args.output_dir}/imgs/{args.dataset}'
            )


        # predicted boundaries
        ends_hat = end_times
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
            words = ' '.join(words[:-1]).split()
            tp, fp, fn = eval_n1_strict(ends, ends_hat, texts.split(), words, tolerance)
            corrects += tp
            total_gts += (tp + fn)
            total_preds += (tp + fp)

    precision, recall, f1, r_value, _ = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    results = dict(precision=precision, recall=recall, f1=f1, r_value=r_value)
    print(results)

    # dump results
    ts = time.time()
    filename = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    results = {**vars(args), **results}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f"{args.output_dir}/{filename}.json", 'w') as f:
        json.dump(results, f)
    if args.save_prediction:
        joblib.dump(all_predictions, f"{args.output_dir}/{filename}-predictions.pkl")


def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for whisper-based forced alignments")
    parser.add_argument('--model', type=str, default='medium')
    parser.add_argument('--dataset', type=str, default="TIMIT", choices=["TIMIT", "LibriSpeech"])
    parser.add_argument('--scp', type=str, default="scp/test.wav.scp")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Path to the output directory", required=True)
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--medfilt_width', type=int, default=7)
    parser.add_argument('--aggr', type=str, default="mean", choices=["mean", "topk"])
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--aligned_unit_type', type=str, default='subword', choices=["subword", "char"])
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--w_colnorm', type=float, default=1.0)
    parser.add_argument('--w_rownorm', type=float, default=1.0)
    parser.add_argument('--w_coverage', type=float, default=0.0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--default_whisper_timing', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
    AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

    infer_dataset(args)
