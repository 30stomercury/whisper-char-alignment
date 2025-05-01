import os
import sys
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
from glob import glob
import string
import whisper
from whisper.tokenizer import get_tokenizer
from whisper.model import disable_sdpa
from whisper.timing import median_filter, dtw
from whisper.normalizers import EnglishTextNormalizer
from metrics import AlignmentMetric

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
AUDIO_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2 
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN

def entropy(prob, eps=1e-15):
    # compute mean entropy
    prob = prob / torch.sum(prob, dim=-1).unsqueeze(-1)
    ent = torch.zeros(prob.size(0))
    logprob = torch.log(prob + eps)
    ent = torch.sum(-(prob * logprob), dim=-1)
    ent = torch.mean(ent)
    return ent

def coverage_penalty(attn, threshold=0.5):
    """
    attn : torch.tensor in (tokens, frames)
    """
    coverage = torch.sum(attn, dim=0)

    # Compute coverage penalty
    penalty = torch.max(
        coverage, coverage.clone().fill_(threshold)
    ).sum(-1)
    penalty = penalty - coverage.size(-1) * threshold
    return penalty

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", n_mels=80, device=DEVICE):
        root = '/disk/scratch/s2522924/LibriSpeech'
        file_list = sorted(glob(os.path.join(root, split, "**/*.flac"), recursive=True))
        trans_list = sorted(glob(os.path.join(root, split, "**/*.trans.txt"), recursive=True))
        label_dict = {}
        for trans in trans_list:
            lines = open(trans, 'r').readlines()
            for l in lines:
                fid, text = l.split(' ', 1)
                label_dict[fid] = text
        self.dataset = []
        print('collecting audio...')
        for file in tqdm(file_list):
            fid = file.split('/')[-1].split('.')[0]
            audio, sample_rate = torchaudio.load(file)
            audio = audio.squeeze()
            text = label_dict[file.split('/')[-1].split('.')[0]]
            self.dataset.append((audio, sample_rate, text, fid))
        self.n_mels = n_mels
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, fid = self.dataset[item]
        assert sample_rate == 16000
        duration = len(audio.flatten())
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, self.n_mels)

        return (mel, duration, text, fid)


def get_attentions(
        mel, 
        tokens, 
        model, 
        tokenizer, 
        max_frames, 
        aggregation='mean', 
        topk=-1, 
        medfilt_width=7, 
        qk_scale=1.0
    ):
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad(), disable_sdpa():
        logits = model(mels, tokens.unsqueeze(0))[0]
    for hook in hooks:
        hook.remove()

    weights = torch.stack(QKs)
    weights = weights[:, :, :, : max_frames]
    weights = (weights * qk_scale).softmax(dim=-1)
    L, H, N, T = weights.shape
    # print(weights.shape)

    all_heads = torch.zeros(
        L, H, dtype=torch.bool
    )

    if aggregation == "mean":
        # whisper implementation:
        all_heads[model.dims.n_text_layer // 2 :, :] = True
        all_heads = all_heads.to_sparse()

    elif aggregation == "topk":
        assert topk > 0
        # select attentions:
        weight_norm = weights / weights.norm(dim=-2, keepdim=True)
        ent_mat = []
        for l in range(L):
            for h in range(H):
                # score = entropy(weight_norm[l, h])
                score = coverage_penalty(weight_norm[l, h])
                ent_mat.append((score.item(), (l, h)))
        scores_sorted = sorted(ent_mat)[:topk]
        # print(scores_sorted)
        for _, (_l, _h) in scores_sorted:
            all_heads[_l, _h] = True
        
    all_heads = all_heads.to_sparse()
    weights = torch.stack([weights[_l][_h] for _l, _h in all_heads.indices().T])
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)
    weights = weights.mean(axis=0)

    return weights, logits

def plot_alignment(
        weights, 
        hypothesis,
        gt_alignment,
        fid, 
        text_tokens, 
        words, 
        word_tokens, 
        start_times, 
        end_times,
        ali_type='char'
    ):
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(weights, aspect='auto')
    
    N = len(weights)
    for i, (word, s, e) in enumerate(gt_alignment):
        if word == '':
            word = '<>'
        b = int(e / 0.02) 
        value = b
        ax.axvline(x = value-1, linewidth=0.8, color='white')
        if i == 0:
            ax.text(b//2, N * 0.65, word, va='center', ha='center', color='white', fontsize=10, clip_on=True)
        else:
            if i % 2 == 1:
                ax.text((int(ali[i-1][2] / 0.02) + b) // 2, N * 0.7, word, va='center', ha='center', color='white', fontsize=10, clip_on=True)
            else:
                ax.text((int(ali[i-1][2] / 0.02) + b) // 2, N * 0.65, word, va='center', ha='center', color='white', fontsize=10, clip_on=True)
    
    if ali_type == 'word':
        for i, (word, tokens, start, end) in enumerate(zip(
                words, word_tokens, start_times, end_times
            )):
                b = int(end / 0.02) 
                value = b
                ax.axvline(x = value-1, linewidth=1, color='red', ls='dotted')
                if i == 0:
                    ax.text(b//2, N * 0.25, word, va='center', ha='center', color='red', fontsize=10, clip_on=True)
                else:
                    if i % 2 == 1:
                        ax.text((int(end_times[i-1] / 0.02) + b) // 2, N * 0.2, word, va='center', ha='center', color='red', fontsize=10, clip_on=True)
                    else:
                        ax.text((int(end_times[i-1] / 0.02) + b) // 2, N * 0.25, word, va='center', ha='center', color='red', fontsize=10, clip_on=True)
    elif ali_type == 'char':
        s = 0
        count = 0
        for word in hypothesis.split():
            b = int(end_times[s + len(word)-1] / 0.02) 
            value = b
            ax.axvline(x = value-1, linewidth=1, color='red', ls='dotted')
            if count == 0:
                ax.text(b//2, N * 0.25, word, va='center', ha='center', color='red', fontsize=10, clip_on=True)
            else:
                if count % 2 == 1:
                    ax.text((prev + b) // 2, N * 0.2, word, va='center', ha='center', color='red', fontsize=10, clip_on=True)
                else:
                    ax.text((prev + b) // 2, N * 0.25, word, va='center', ha='center', color='red', fontsize=10, clip_on=True)
            prev = b
            s += len(word)
            count += 1
    else:
        raise NotImplementedError

    ax.set_ylim((len(text_tokens)-1, 0))
    ax.set_yticks(np.arange(len(text_tokens)-1, -1, -1))
    ylabels = [tokenizer.decode([t]) for t in text_tokens]
    ax.set_yticklabels(ylabels[::-1], fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fid}.png'), bbox_inches='tight', dpi=400)
    plt.close()

def force_align(
        weights, 
        text_tokens, 
        tokenizer, 
        hypothesis=None,
        gt_alignment=None,
        fid=None, 
        aggregation="mean", 
        topk=-1, 
        plot=False,
        path='alignment_sample/dev-clean/nedium',
        ali_type='char'
    ):

        matrix = weights[len(tokenizer.sot_sequence):-1, :]
        text_indices, time_indices = dtw(-matrix)

        words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] / TOKENS_PER_SECOND
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]

        if plot == True:
            plot_alignment(matrix.cpu().numpy(), hypothesis, gt_alignment, fid, text_tokens, words, word_tokens, start_times, end_times, ali_type)

        return words, word_tokens, start_times, end_times
    

if __name__ == '__main__':
    # load model
    model_name = sys.argv[1] # model name (ex. medium)
    model = whisper.load_model(model_name, download_root='./checkpoints', device='cpu')
    model.to(DEVICE)
    model.requires_grad_(False)
    model.eval()
    # set dataloader
    split = "dev-clean" 
    dataset = LibriSpeech(split=split, n_mels=model.dims.n_mels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # load ground truth alignment (mfa alignments for librispeecdh)
    alignment_dict = {}
    raw = open('ls_alignment_dev-clean.txt', 'r').readlines()
    for line in raw:
        fname = line.split(' ', 1)[0]
        alignment_dict[fname] = eval(line.split(' ', 1)[1])
    # settings
    decode_options = whisper.DecodingOptions(language="en", without_timestamps=True)
    language: str = decode_options.language
    task: str = decode_options.task
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )
    normalizer = EnglishTextNormalizer()

    output_dir = os.path.join('alignment_sample', split, model_name)
    os.makedirs(output_dir, exist_ok=True)

    stats = {}
    L, H = model.dims.n_text_layer, model.dims.n_text_head
    for _l in range(L):
        for _h in range(H):
            stats[f'l{_l+1}_h{_h+1}'] = 0

    # set alignment type and eval metric
    ali_type = 'word'
    agg_type = 'mean'
    k = 50
    metric = AlignmentMetric(tolerance=0.02)

    for n, (mels, durations, texts, fids) in enumerate(tqdm(loader)):
        # forced alignment for librispeech
        ali = alignment_dict[fids[0]]
        
        mels = mels.to(DEVICE)
        with torch.no_grad():
            results = model.decode(mels, decode_options)

        hypothesis = results[0].text
        hypothesis = hypothesis.replace(",", "").replace(".", "") 
        
        if ali_type == 'word':
            text_tokens = tokenizer.encode(hypothesis)
        elif ali_type == 'char':
            text_tokens = tokenizer.encode(' '.join([c.upper() for c in hypothesis if c != ' ']))
        # ground truth transcript
        transcript = tokenizer.encode(" ".join([w[0] for w in ali[1:]]))
        tokens = torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.no_timestamps,
                *text_tokens,
                tokenizer.eot,
            ]
        ).to(model.device)

        # raw attention score (before softmax)
        max_frames = durations[0] // AUDIO_SAMPLES_PER_TOKEN
        weights, logits = get_attentions(mels, tokens, model, tokenizer, max_frames, aggregation=agg_type, topk=k) 
        words, word_tokens, start_times, end_times = force_align(weights, text_tokens, tokenizer, hypothesis, gt_alignment=ali, fid=fids[0], plot=False, path=output_dir, ali_type=ali_type)
        # print(words)
        
        # eval alignment
        tg_boundaries = [item[2] for item in ali if item[0] != '']
        if ali_type == 'word':
            pred_boundaries = end_times
        elif ali_type == 'char':
            pred_boundaries = []
            s = 0
            for word in hypothesis.split():
                pred_boundaries.append(end_times[s + len(word)-1])
                s += len(word)
            # print(tg_boundaries)
            # print(pred_boundaries)

        metric.update(tg_boundaries, pred_boundaries)

    if agg_type == 'topk':
        print(f"unit: {ali_type} | aggregation: {agg_type[:-1]}{k}")
    else:
        print(f"unit: {ali_type} | aggregation: {agg_type}") 
    P, R, F1, Rval = metric.get_final_metrics()
    print('================================')
    print(f"{'Precsion:':<10} {P:>10.6f}")
    print(f"{'Recall:':<10} {R:>10.6f}")
    print(f"{'F1:':<10} {F1:>10.6f}")
    print(f"{'R-value:':<10} {Rval:>10.6f}")
    print('================================')
