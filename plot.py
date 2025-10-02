import os
import datetime
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import joblib
import matplotlib.pyplot as plt
from retokenize import encode, remove_punctuation, split_tokens_on_spaces
import whisper
from whisper.tokenizer import get_tokenizer
from whisper.timing import median_filter, dtw
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND

MAX_FRAMES = 1500
MAX_LENGTH = 448
AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

def plot_attn(
        weights,
        text_tokens,
        tokenizer,
        gt_alignment,
        pred_alignment,
        fid, 
        aligned_unit_type,
        path
    ):

    os.makedirs(path, exist_ok=True)
    color = 'cyan' if aligned_unit_type == 'subword' else 'red'

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.imshow(weights.detach().cpu().numpy(), aspect='auto')
    N = len(weights)

    if gt_alignment is not None:
        for e in gt_alignment:
            ax.axvline(int(e / 0.02), linewidth=2, color='white')
    for e in pred_alignment:
        ax.axvline(int(e / 0.02) , linewidth=3, color=color, ls='dotted')

    words, word_tokens = split_tokens_on_spaces(text_tokens + [tokenizer.eot], tokenizer, aligned_unit_type)
    token_boundaries = np.cumsum([len(w) for w in word_tokens[:-1]])
    for b in token_boundaries:
        ax.axhline(b-0.5, linewidth=1.5, color='gray', ls='--')
    ax.set_yticks(np.arange(len(weights)-1, -1, -1))
    ylabels = [tokenizer.decode([t]) for t in text_tokens] + ['']
    ax.set_yticklabels(ylabels[::-1], fontsize=9)
    ax.set_xticks([])
    
    plt.xlabel(r'${time} (\rightarrow)$', fontsize=18)
    plt.tight_layout()  
    plt.savefig(os.path.join(path, f'{fid}.png'), bbox_inches='tight', dpi=400)
    plt.close()