import os
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch

from metrics import eval_n1, get_seg_metrics
from dataset import TIMIT, LibriSpeech, AMI, Collate
from retokenize import encode, remove_punctuation
from timing import get_attentions, force_align, filter_attention, default_find_alignment
from plot import plot_attns

import whisper
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output

DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

MAX_FRAMES = 1500
MAX_LENGTH = 448

DATASET = {"TIMIT": TIMIT, "LibriSpeech": LibriSpeech, "AMI": AMI}





def infer_dataset(args):
    print(args)
    tolerance = args.tolerance

    # model
    model_whisper = whisper.load_model(args.model)
    model_whisper.to(DEVICE)
    tokenizer = get_tokenizer(model_whisper.is_multilingual, language='English')
    """
    faster_whisper_model = 'nyrahealth/CrisperWhisper'
    # Initialize the Whisper model
    model = WhisperModel(faster_whisper_model, device=DEVICE, compute_type="float32")
    """

    # basically paremeters to do denoising
    medfilt_width = args.medfilt_width
    qk_scale = 1.0

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
    )
    model.to(DEVICE)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps='word',
        torch_dtype=torch_dtype,
        device=DEVICE,
    )


    # basically paremeters to do denoising
    dataset = DATASET[args.dataset](args.scp, n_mels=args.n_mels, device=DEVICE)

    loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)

    # decode the audio
    options = whisper.DecodingOptions(language="en")

    corrects = 0
    total_preds = 0
    total_gts = 0
    for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):

        sample = audios.numpy()[:durations]
        hf_pipeline_output = pipe(sample)
        text = []
        for chunk in hf_pipeline_output['chunks']:
            text.append(chunk['text'])

        transcription = remove_punctuation(' '.join(text))
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
        
        w, logits = get_attentions(mels, tokens, model_whisper, tokenizer, max_frames, medfilt_width, qk_scale)
        words, start_times, end_times, ws, scores = force_align(w, text_tokens, tokenizer, 
                aligned_unit_type=args.aligned_unit_type, aggregation=args.aggr, topk=args.topk)
        ends_hat = end_times
        print(ends)
        print(end_times)

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
    with open(f"{args.output_dir}/{filename}.json", 'w') as f:
        json.dump(results, f)


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
