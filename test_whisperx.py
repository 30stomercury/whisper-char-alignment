import os
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch

from metrics import eval_n1, eval_n1_strict, get_seg_metrics
from dataset import TIMIT, LibriSpeech, AMI, Collate
from retokenize import encode, remove_punctuation

import whisperx


DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

DATASET = {"TIMIT": TIMIT, "LibriSpeech": LibriSpeech, "AMI": AMI}

def infer_dataset(args):
    tolerance = args.tolerance

    # model
    device = "cuda"
    batch_size = 1 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(args.model, device, compute_type=compute_type, language='en')
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)

    dataset = DATASET[args.dataset](args.scp, n_mels=args.n_mels, device=model.device)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)


    corrects = 0
    total_preds = 0
    total_gts = 0
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        if texts.split() < args.min_num_words:
            continue
        
        audios = audios.cpu().detach().numpy()
        result = model.transcribe(audios, batch_size=batch_size)
        for segment in result['segments']:
            segment['text'] = remove_punctuation(segment['text'])

        # 2. Align whisper output
        ends_hat = []
        words = []
        if len(result["segments"]) > 0:
            result = whisperx.align(
                    result["segments"], model_a, metadata, audios, device, return_char_alignments=False)
            total_words = len(result["segments"][0]['words'])
            for i in range(total_words):
                if 'end' in result["segments"][0]['words'][i]:
                    ends_hat.append(result["segments"][0]['words'][i]['end'])
                    words.append(result["segments"][0]['words'][i]['word'])

        texts = remove_punctuation(texts)
        words = remove_punctuation(' '.join(words))
        # eval
        if not args.strict:
            correct_pred, _ = eval_n1(ends, ends_hat, tolerance)
            total_gts += len(ends)
            total_preds += len(ends_hat)
            corrects += correct_pred
        else:
            tp, fp, fn = eval_n1_strict(ends, ends_hat, texts.split(), words.split(), tolerance)
            corrects += tp
            total_gts += (tp + fn)
            total_preds += (tp + fp)

    print("corrects:", corrects)
    print("gt:", total_gts)
    precision, recall, f1, r_value, _ = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    results = dict(precision=precision, recall=recall, 
            f1=f1, r_value=r_value, corrects=corrects, total_gts=total_gts)
    print(precision, recall, f1, r_value)

    # dump results
    ts = time.time()
    filename = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    results = {**vars(args), **results}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f"{args.output_dir}/whisperx-{filename}.json", 'w') as f:
        json.dump(results, f)


def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for whisper-based forced alignments")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Path to the output directory", required=True)
    parser.add_argument('--model', type=str, default='medium')
    parser.add_argument('--dataset', type=str, default="TIMIT", choices=["TIMIT", "LibriSpeech", "AMI"])
    parser.add_argument('--n_mels', type=int, default=80)
    parser.add_argument('--scp', type=str, default="scp/test.wav.scp")
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--min_num_words', type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    infer_dataset(args)
