import os
import numpy as np
import string
import torch
import whisper
from whisper.timing import dtw
from whisper.tokenizer import get_tokenizer
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt


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

def count_transitions(x):
    count = 0
    positions = []
    prev = x[0]
    for i in range(1, len(x)):
        if x[i] != x[i-1]: 
            positions.append(i)
            count += 1

    return count, positions

def filter_attention(w, topk=20, plot=True, path="imgs", wrd_pos=None):
    """
    w : torch.tensor in (layers, heads, tokens, frames)
    """
    if not os.path.exists(path):
        os.makedirs(path)
    scores = []
    for l in range(w.size(0)):
        for n_h in range(w.size(1)):
            score = coverage_penalty(w[l, n_h])
            name = f"sample_layer{l}_head{n_h}"
            scores.append((score, (l, n_h), name))

    scores_sorted = sorted(scores)[:topk]
    ws = []
    for _, (l, n_h), name in scores_sorted:

        name = f"sample_layer{l}_head{n_h}"
        ws.append(w[l, n_h].unsqueeze(0))
        if plot:
            assert wrd_pos is not None
            plt.vlines(wrd_pos, ymin=-1, ymax=w[l, n_h].size(0), colors="white")
            plt.imshow(w[l, n_h], aspect="auto")
            plt.savefig(f"{path}/{name}.png")

    if plot:
        ws_ = torch.cat(ws, 0).mean(0)
        plt.vlines(wrd_pos, ymin=-1, ymax=w[l, n_h].size(0), colors="white")
        plt.imshow(ws_, aspect="auto")
        plt.savefig(f"{path}/sample_ave.png")
    
    return ws

def get_attentions(mel, tokens, model, tokenizer, medfilt_width=7, qk_scale=1.0):
    # install hooks on the cross attention layers to retrieve the attention weights
    # NOTE: make sure MultiHeadAttention.use_sdpa = False
    QKs = [None] * model.dims.n_text_layer

    for i, block in enumerate(model.decoder.blocks):
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )

    with torch.no_grad():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]

    weights = torch.cat(QKs)  # layers * heads * tokens * frames    
    weights = weights[:, :, len(tokenizer.sot_sequence):-1, 
                      :duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
    weights = median_filter(weights, (1, 1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    w = weights / weights.norm(dim=-2, keepdim=True)
    return w, logits

def force_align(
        w, tokens, 
        tokenizer, 
        aggregation="mean", 
        topk=-1, 
        plot=False,
        path="img_topk", 
        wrd_pos=None
    ):
    """
    w : torch.tensor in (layers, heads, tokens, frames)
    tokens : tokens of texts, without bot, eot
    """
    if aggregation == "mean":
        # whisper implementation:
        matrix = w.mean(axis=(0, 1))

    elif aggregation == "topk":
        assert topk > 0
        # select attentions:
        matrix = filter_attention(w, topk=topk, plot=plot, path=path, wrd_pos=wrd_pos)
        matrix = torch.cat(matrix, 0).mean(0)

    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    return words, start_times, end_times




AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE
print(TOKENS_PER_SECOND)

# basically paremeters to do denoising
medfilt_width = 7
qk_scale = 1.0

model = whisper.load_model("medium")

# decode the audio
options = whisper.DecodingOptions()


# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("/home/s2196654/dataset/timit/timit-wav/train/dr1/fcjf0/sa1.wav")
duration = len(audio)
audio = whisper.pad_or_trim(audio)

# audio
mel = whisper.log_mel_spectrogram(audio).to(model.device)


# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# print the recognized text
result = whisper.decode(model, mel, options)
print(result.text)

# tokens
tokenizer = get_tokenizer(model.is_multilingual, language='English')
transcription = result.text
transcription = transcription.translate(str.maketrans('', '', string.punctuation))
#transcription = ' '.join([i for i in transcription])

wrd_pos = []
f = open("/home/s2196654/dataset/timit/timit-wav/train/dr1/fcjf0/sa1.wrd", 'r')
for line in f.readlines():
    wrd_pos.append(int(line.split()[1])/SAMPLE_RATE)

text_tokens = tokenizer.encode(transcription)
tokens = torch.tensor(
    [
        *tokenizer.sot_sequence,
        tokenizer.no_timestamps,
        *text_tokens,
        tokenizer.eot,
    ]
).to(model.device)
print(tokens)

# Get attention maps
print(wrd_pos)
wrd_pos = [int(i/0.02) for i in wrd_pos]
w, logits = get_attentions(mel, tokens, model, tokenizer, medfilt_width, qk_scale)
results = force_align(w, text_tokens, tokenizer, aggregation="mean", topk=20, plot=False, wrd_pos=wrd_pos)
import pdb; pdb.set_trace()
