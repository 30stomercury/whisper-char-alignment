import numpy as np
import torch
import string

def dtw_timestamp(gt_ends, pred_ends):
    n, m = len(gt_ends), len(pred_ends)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.abs(gt_ends[i - 1] - pred_ends[j - 1]) 
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    
                dtw_matrix[i, j - 1],    
                dtw_matrix[i - 1, j - 1]
            )

    distance = dtw_matrix[n, m]
    return distance, dtw_matrix

def eval_n1(y, yhat, tolerance=1):
    def is_match(i, j, tolerance):
        return (1 if abs(i-j) <= tolerance else 0)
    n_match = 0

    if len(yhat) == 0:
        return 0, 0

    i, j = 0, 0
    while i < len(y) and j < len(yhat):
        if is_match(y[i], yhat[j], tolerance):
            i += 1
            j += 1
            n_match += 1

        elif y[i] < yhat[j]:
            i += 1

        elif y[i] > yhat[j]:
            j += 1

    return n_match, n_match

def eval_n1_strict(y, y_hat, words, words_hat, tolerance=1):
    words = [w.lower().strip(string.punctuation) for w in words]
    words_hat = [w.lower().strip(string.punctuation) for w in words_hat]
    def is_match(y_i, yhat_j, w_i, what_j, tolerance):
        return (
            w_i == what_j and abs(y_i - yhat_j) <= tolerance
        )

    i, j = 0, 0
    tp = 0
    used_refs = set()

    while i < len(y_hat):
        matched = False
        for j in range(len(y)):
            if j in used_refs:
                continue
            if is_match(y[j], y_hat[i], words[j], words_hat[i], tolerance):
                tp += 1
                used_refs.add(j)
                matched = True
                break
        i += 1

    fp = len(y_hat) - tp
    fn = len(y) - len(used_refs)

    return tp, fp, fn

def get_seg_metrics(correct_predict, correct_retrieve, total_predict, total_gold):
    EPS = 1e-7
    
    precision = correct_predict / (total_predict + EPS)
    recall = correct_retrieve / (total_gold + EPS)
    f1 = 2 * (precision * recall) / (precision + recall + EPS)
    
    os = recall / (precision + EPS) - 1
    r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (np.sqrt(2))
    r_value = 1 - (abs(r1) + abs(r2)) / 2

    return precision, recall, f1, r_value, os

def count_transitions(x):
    count = 0
    positions = []
    prev = x[0]
    for i in range(1, len(x)):
        if x[i] != x[i-1]: 
            positions.append(i)
            count += 1

    return count, positions

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

def entropy(prob, eps=1e-15):
    # compute mean entropy
    prob = prob / torch.sum(prob, dim=-1).unsqueeze(-1)
    ent = torch.zeros(prob.size(0))
    logprob = torch.log(prob + eps)
    ent = torch.sum(-(prob * logprob), dim=-1)
    ent = torch.mean(ent)
    return -ent
