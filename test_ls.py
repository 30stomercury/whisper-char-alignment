import os
import numpy as np
from tqdm import tqdm
import torch
from metrics import eval_n1, get_seg_metrics
from dataset import LibriSpeech, Collate
from timing_ls import get_attentions, force_align, filter_attention
from retokenize_ls import char_tokenizer_encode, remove_punctuation, normalizer

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def infer_dataset(model, tokenizer, split='dev-clean', tolerance=0.02):
    # basically parameters to do denoising
    medfilt_width = 7
    qk_scale = 1.0
    dataset = LibriSpeech(split=split, n_mels=model.dims.n_mels, device=DEVICE)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    # decode the audio
    options = whisper.DecodingOptions(language="en")

    corrects = 0
    total_preds = 0
    total_gts = 0
    sep_space=False
    for n, (mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        mels = mels.to(DEVICE)
        # print the recognized text
        #result = whisper.decode(model, mels, options)
        #transcription = result.text
        transcription = texts
        transcription = transcription[0].upper() + transcription[1:]
        transcription = remove_punctuation(transcription)

        with torch.no_grad():
            results = model.decode(mels.unsqueeze(0), options)
        hypothesis = normalizer(results[0].text)

        text_tokens = char_tokenizer_encode(hypothesis, tokenizer, sep_space=sep_space)
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
        w, logits = get_attentions(mels, tokens, model, tokenizer, max_frames, medfilt_width, qk_scale)

        results = force_align(w, text_tokens, hypothesis, tokenizer,
                aggregation="topk", topk=10, plot=False, wrd_pos=ends, sep_space=sep_space)

        # predicted boundaries
        ends_hat = results[2]

        # eval
        total_gts += len(ends)
        total_preds += len(ends_hat)
        correct_pred, _ = eval_n1(ends, ends_hat, tolerance)
        corrects += correct_pred

        if n == 10:
            break

    precision, recall, f1, r_value, os = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    print(precision, recall, f1, r_value)


AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

model = whisper.load_model("medium")
model.to(DEVICE)
model.requires_grad_(False)
model.eval()

# decode the audio
options = whisper.DecodingOptions(language="en")

tokenizer = get_tokenizer(model.is_multilingual, language='English')
infer_dataset(model, tokenizer)