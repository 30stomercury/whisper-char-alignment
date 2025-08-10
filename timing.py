import os
import torch
import numpy as np
from metrics import coverage_penalty, entropy
from retokenize import split_tokens_on_spaces

import whisper
from whisper.model import disable_sdpa
from whisper.timing import median_filter, dtw
from whisper.audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND


def filter_attention(attns, topk=20, w_colnorm=1, w_rownorm=1, w_coverage=0):
    """
    attns : torch.tensor in (layers, heads, tokens, frames)
    """
    n_layers = attns.size(0)
    n_heads = attns.size(1)
    score_matix = torch.zeros(n_layers, n_heads, device=attns.device)
    if w_colnorm > 0:
        col_norm_sum = attns.norm(dim=-2).sum(-1)
        score_matix += w_colnorm * col_norm_sum
    if w_rownorm > 0:
        row_norm_sum = attns.norm(dim=-1).sum(-1)
        score_matix += w_rownorm * row_norm_sum
    scores = []
    for l in range(attns.size(0)):
        for n_h in range(attns.size(1)):
            score = score_matix[l, n_h]
            if w_coverage > 0:
                penalty = w_coverage * coverage_penalty(attns[l, n_h])
                score -= penalty
            name = f"sample_layer{l}_head{n_h}"
            scores.append((score.item(),(l, n_h), name))

    scores_sorted = sorted(scores)[-topk:]

    selected_attns = []
    for score, (l, n_h), name in scores_sorted:
        name = f"sample_layer{l}_head{n_h}"
        selected_attns.append(attns[l, n_h].unsqueeze(0))

    return selected_attns, scores_sorted

def get_attentions(mel, tokens, model, tokenizer, max_frames, medfilt_width=7, qk_scale=1.0):
    # install hooks on the cross attention layers to retrieve the attention weights
    # NOTE: make sure MultiHeadAttention.use_sdpa = False
    QKs = [None] * model.dims.n_text_layer

    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad(), disable_sdpa():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]

    for hook in hooks:
        hook.remove()

    weights = torch.cat(QKs)  # layers * heads * tokens * frames    
    weights = weights[..., :max_frames]
    weights = median_filter(weights, medfilt_width)
    weights = (weights * qk_scale).softmax(dim=-1)
    return weights, logits

def force_align(
        ws, tokens, 
        tokenizer, 
        aligned_unit_type='subword',
        aggregation="mean", 
        topk=-1, 
        w_colnorm=1.0,
        w_rownorm=1.0,
        w_coverage=0.0
    ):
    """
    w : torch.tensor in (layers, heads, tokens, frames)
    tokens : tokens of texts, without bot, eot
    """

    scores = None
    if aggregation == "mean":
        # whisper implementation:
        ws = ws / ws.norm(dim=-2, keepdim=True)
        n_layers = ws.size(0)
        ws = ws[n_layers//2:]
        matrix = ws.mean(axis=(0, 1))

    elif aggregation == "topk":
        assert topk > 0
        # select attentions:
        ws, scores = filter_attention(ws, topk, w_colnorm, w_rownorm, w_coverage)
        matrix = torch.cat(ws, 0)
        col_norm = matrix.norm(dim=-2, keepdim=True)
        matrix = torch.mean(matrix / col_norm, 0)

    elif aggregation == 'grad_norm':
        matrix = ws

    matrix = matrix[len(tokenizer.sot_sequence):-1].cpu()
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = split_tokens_on_spaces(tokens + [tokenizer.eot], tokenizer, aligned_unit_type)
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return [[], [], [], [], None]
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    return words, start_times, end_times, ws, scores

def default_find_alignment(
    model: "Whisper",
    tokenizer,
    text_tokens,
    mel,
    max_frames,
    *,
    medfilt_width=7,
    qk_scale=1.0,
):

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    ).to(model.device)

    # install hooks on the cross attention layers to retrieve the attention weights
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad(), disable_sdpa():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
        sampled_logits = logits[len(tokenizer.sot_sequence) :, : tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
        text_token_probs = text_token_probs.tolist()

    for hook in hooks:
        hook.remove()

    # heads * tokens * frames
    weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
    weights = weights[:, :, : max_frames]
    weights = median_filter(weights, medfilt_width)
    weights = (weights * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std

    matrix = weights.mean(axis=0)
    matrix = matrix[len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return [[], [], [], [], None]
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return words, start_times, end_times, weights, None
