import os
import numpy as np
from tqdm import tqdm
import torch
from metrics import eval_n1, get_seg_metrics
from dataset import TIMIT, Collate
from timing import get_attentions, force_align, filter_attention
from retokenize import char_tokenizer_encode, remove_punctuation

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND


def infer_dataset(model, tokenizer, scp_file="scp/test-subset.wav.scp", tolerance=0.02):

    # basically paremeters to do denoising
    medfilt_width = 7
    qk_scale = 1.0
    dataset = TIMIT(scp_file, n_mels=80, device=model.device)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    # decode the audio
    options = whisper.DecodingOptions(language="en")

    corrects = 0
    total_preds = 0
    total_gts = 0
    for n, (mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
        # print the recognized text
        #result = whisper.decode(model, mels, options)
        #transcription = result.text
        transcription = texts
        transcription = transcription[0].upper() + transcription[1:]
        transcription = remove_punctuation(transcription)

        text_tokens = char_tokenizer_encode(transcription, tokenizer)
        #text_tokens = tokenizer.encode(transcription)
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

        ws, scores = filter_attention(w, topk=30)
        candidates = []
        best_score = -1
        best_ends_hat = None
        for w, score in zip(ws, scores):
            results = force_align(w.unsqueeze(0), text_tokens, tokenizer,
                    aggregation="mean", topk=15, plot=False, wrd_pos=ends)

            # collect predicted boundaries
            ends_hat = results[2]
            correct_pred, _ = eval_n1(ends, ends_hat, tolerance)
            precision, recall, f1, r_value, os = \
                    get_seg_metrics(correct_pred, correct_pred, len(ends_hat), len(ends))
            print(score, f1)
            if f1 > best_score:
                best_score = f1
                best_ends_hat = results[2]

            # not used now but maybe useful for topk
            candidates.append(ends_hat)

        import pdb; pdb.set_trace()
        # eval
        total_gts += len(ends)
        total_preds += len(best_ends_hat)
        correct_pred, _ = eval_n1(ends, best_ends_hat, tolerance)
        corrects += correct_pred

    precision, recall, f1, r_value, os = \
             get_seg_metrics(corrects, corrects, total_preds, total_gts)
    print(precision, recall, f1, r_value)



AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

model = whisper.load_model("medium")

# decode the audio
options = whisper.DecodingOptions(language="en")

tokenizer = get_tokenizer(model.is_multilingual, language='English')
infer_dataset(model, tokenizer)
