import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
from glob import glob
import whisper
from whisper.tokenizer import get_tokenizer
from whisper.model import disable_sdpa
from whisper.timing import median_filter, dtw

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

def entropy(prob, eps=1e-15):
    ent = 0
    for p in prob:
        ent -= p * math.log(p + eps, math.e)
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

model_name = 'medium'
model = whisper.load_model(model_name, device='cpu')
model.to(DEVICE)
model.requires_grad_(False)
model.eval()

split = "dev-clean" 
dataset = LibriSpeech(split=split, n_mels=model.dims.n_mels)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

alignment_dict = {}
raw = open('ls_alignment_dev-clean.txt', 'r').readlines()
for line in raw:
    fname = line.split(' ', 1)[0]
    alignment_dict[fname] = eval(line.split(' ', 1)[1])

decode_options = whisper.DecodingOptions(language="en", without_timestamps=True)
language: str = decode_options.language
task: str = decode_options.task
tokenizer = get_tokenizer(
    model.is_multilingual,
    num_languages=model.num_languages,
    language=language,
    task=task,
)

output_dir = os.path.join('alignment_sample', split, model_name)
os.makedirs(output_dir, exist_ok=True)

for n, (mels, durations, texts, fids) in enumerate(tqdm(loader)):
    mels = mels.to(DEVICE)
    with torch.no_grad():
        results = model.decode(mels, decode_options)

    print(f"hypothesis: {results[0].text}")
    text_tokens = results[0].tokens
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    ).to(model.device)

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

    qk = torch.stack(QKs)
    qk = qk[:, :, :, : durations[0] // AUDIO_SAMPLES_PER_TOKEN]
    attn = F.softmax(qk, dim=-1)
    print(attn.shape)
    
    # np.save(os.path.join(output_dir, f'{fids[0]}.npy'), attn.detach().cpu().numpy())
    
    ali = alignment_dict[fids[0]]
    print(ali)

    L, H, N, T = attn.shape
    attn = attn / attn.norm(dim=-2, keepdim=True)

    ent_mat = []
    for l in range(L):
        for h in range(H):
            # score = np.mean([entropy(c / sum(c)) for c in attn[l][h]])
            score = coverage_penalty(attn[l, h])
            ent_mat.append((score.item(), (l, h)))
            # print(f'layer {l+1}, head {h+1}: {score}')

    all_heads = torch.zeros(
        L, H, dtype=torch.bool
    )
    topk = 10
    scores_sorted = sorted(ent_mat)[:topk]
    print(scores_sorted)
    # all_heads[-6:, :] = True
    for _, (_l, _h) in scores_sorted:
        all_heads[_l, _h] = True
    all_heads = all_heads.to_sparse()

    weights = torch.stack([attn[_l][_h] for _l, _h in all_heads.indices().T])
    print(weights.shape)
    # std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    # weights = (weights - mean) / std
    # weights = median_filter(weights, 7)
    attn_pool = weights.mean(axis=0)
    attn_pool = attn_pool[len(tokenizer.sot_sequence) : -1]
    # text_indices, time_indices = dtw(-attn_pool)
    # print(text_indices, time_indices)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(attn_pool.detach().cpu().numpy())

    words = []
    N = len(attn_pool)
    for i, (word, s, e) in enumerate(ali):
        if word == '':
            word = '<sil>'
        words.append(word)
        b = int(e / 0.02) 
        value = b
        ax.axvline(x = value-1, linewidth=1, color='white', ls='dotted')
        if i == 0:
            ax.text(b//2, N * 0.7, word, va='center', ha='center', color='white', fontsize=6, clip_on=True)
        else:
            if i % 2 == 1:
                ax.text((int(ali[i-1][2] / 0.02) + b) // 2, N * 0.8, word, va='center', ha='center', color='white', fontsize=6, clip_on=True)
            else:
                ax.text((int(ali[i-1][2] / 0.02) + b) // 2, N * 0.7, word, va='center', ha='center', color='white', fontsize=6, clip_on=True)
    
    ax.set_xlim(0, min(durations[0] // AUDIO_SAMPLES_PER_TOKEN - 1, 500))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fids[0]}.png'), bbox_inches='tight', dpi=400)

    if n == 10:
        break