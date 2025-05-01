import os
import torch
import numpy as np
from metrics import coverage_penalty
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from retokenize import split_chars_on_spaces

import whisper
from whisper.timing import dtw
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND


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

    #for i, block in enumerate(model.decoder.blocks):
    #    block.cross_attn.register_forward_hook(
    #        lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
    #    )
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]

    for hook in hooks:
        hook.remove()

    weights = torch.cat(QKs)  # layers * heads * tokens * frames    
    weights = weights.cpu()
    weights = median_filter(weights, (1, 1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
    w = weights / weights.norm(dim=-2, keepdim=True)
    return w, logits

def force_align(
        w, tokens, 
        tokenizer, 
        max_frames,
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
    w = w[..., :max_frames]
    wrd_pos = [int(i/0.02) for i in wrd_pos]
    if aggregation == "mean":
        # whisper implementation:
        matrix = w.mean(axis=(0, 1))

    elif aggregation == "topk":
        assert topk > 0
        # select attentions:
        matrix = filter_attention(w, topk=topk, plot=plot, path=path, wrd_pos=wrd_pos)
        matrix = torch.cat(matrix, 0).mean(0)

    matrix = matrix[len(tokenizer.sot_sequence):-1].cpu()
    text_indices, time_indices = dtw(-matrix)

    #words, word_tokens = tokenizer.split_to_word_tokens(tokens + [tokenizer.eot])
    words, word_tokens = split_chars_on_spaces(tokens + [tokenizer.eot], tokenizer)
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

