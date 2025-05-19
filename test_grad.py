import os
import datetime
import time
import json
import argparse
import re
import numpy as np
from tqdm import tqdm
import torch

from metrics import eval_n1, get_seg_metrics
from dataset import TIMIT, LibriSpeech, AMI, Collate
from timing import get_attentions, force_align, filter_attention, default_find_alignment
from retokenize import encode, remove_punctuation
from plot import plot_attns

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from whisper.timing import median_filter

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
    model.requires_grad_(False)
    model.eval()

    # decode the audio
    options = whisper.DecodingOptions(language="en")
    tokenizer = get_tokenizer(model.is_multilingual, language='English')

    # basically paremeters to do denoising
    # medfilt_width = args.medfilt_width
    qk_scale = 1.0
    dataset = DATASET[args.dataset](args.scp, n_mels=args.n_mels, device=model.device)

    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    # decode the audio
    options = whisper.DecodingOptions(language="en")

    corrects = 0
    total_preds = 0
    total_gts = 0
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        # print the recognized text
        mels = mels.to(model.device)
        result = whisper.decode(model, mels, options)
        transcription = result.text
        print(f"ground truth: {texts}")
        #transcription = texts
        #transcription = transcription[0].upper() + transcription[1:]

        transcription = remove_punctuation(transcription)
        print(f"hypothesis: {transcription}")

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

        input_feat = [None]
        def enc_out_hook(module, input, output):
            input_feat[0] = output
        
        def gradient_hook(module, grad_input, grad_output):
            input_feat[0] = grad_output[0]
        
        hook1 = model.encoder.ln_post.register_forward_hook(enc_out_hook)
        # hook1 = model.encoder.ln_post.register_full_backward_hook(gradient_hook)
        mels = mels.unsqueeze(0)
        mels.requires_grad = True
        logits = model(mels, tokens.unsqueeze(0))[0]
        hook1.remove()

        w =  []
        for i in range(logits.shape[0]):
            # option 1. forward hook + torch.autograd
            grad = torch.autograd.grad(logits[i, tokens[i]], input_feat[0], retain_graph=True)[0]
            # option 2. backward hook
            # logits[i, tokens[i]].backward(retain_graph=True)
            # grad = input_feat[0]
            grad_norm = grad.norm(dim=2)
            w.append(grad_norm[:, :max_frames])
        w = torch.cat(w)
        # w = median_filter(w, args.medfilt_width)
        w = w.softmax(dim=-1)
        std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
        w = (w - mean) / std
        # print(w)
        # print(w.shape)

        words, start_times, end_times, ws, scores = force_align(w, text_tokens, tokenizer, 
                    aligned_unit_type=args.aligned_unit_type, aggregation="grad_norm")
        if args.plot:
            plot_attns(ws, scores, wrd_pos=ends, path=f'{args.output_dir}/imgs')

        # predicted boundaries
        ends_hat = end_times
        print(ends_hat)
        print(ends)

        # eval
        total_gts += len(ends)
        total_preds += len(ends_hat)
        correct_pred, _ = eval_n1(ends, ends_hat, tolerance)
        corrects += correct_pred

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
    with open(f"{args.output_dir}/grad-{filename}.json", 'w') as f:
        json.dump(results, f)

def parse_args():

    parser = argparse.ArgumentParser(description="Arguments for whisper-based forced alignments")
    parser.add_argument('--model', type=str, default='medium')
    parser.add_argument('--dataset', type=str, default="TIMIT", choices=["TIMIT", "LibriSpeech", "AMI"])
    parser.add_argument('--scp', type=str, default="scp/test.wav.scp")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Path to the output directory", required=True)
    parser.add_argument('--n_mels', type=int, default=80)
    # parser.add_argument('--medfilt_width', type=int, default=7)
    parser.add_argument('--aligned_unit_type', type=str, default='subword', choices=["subword", "char"])
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--plot', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
    AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

    infer_dataset(args)


