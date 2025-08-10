import glob
import numpy as np
import os
from tqdm import tqdm
import torch
from dataset import TIMIT, LibriSpeech, AMI, Collate
from metrics import eval_n1, get_seg_metrics, eval_n1_strict, eval_n1_c
import joblib

mfa_ali = joblib.load('timit_alignment_train.pkl')
dataset = TIMIT('/home/s2522924/scp/train.wav.scp', n_mels=80, device='cpu')
loader = torch.utils.data.DataLoader(dataset, collate_fn=Collate(), batch_size=1)
corrects = 0
total_preds = 0
total_gts = 0
for n, (audios, mels, durations, texts, starts, ends, fids) in enumerate(tqdm(loader)):
    fname = fids
    starts_hat = mfa_ali[fname]['starts']
    ends_hat = mfa_ali[fname]['ends']
    words = mfa_ali[fname]['words']
    print(ends)
    print(ends_hat)
    print(texts.split())
    print(words)
    tp, fp, fn = eval_n1_strict(ends, ends_hat, texts.split(), words, tolerance=0.05)
    corrects += tp
    total_gts += (tp + fn)
    total_preds += (tp + fp)

precision, recall, f1, r_value, _ = \
            get_seg_metrics(corrects, corrects, total_preds, total_gts)
results = dict(precision=precision, recall=recall, f1=f1, r_value=r_value)
print(precision, recall, f1, r_value)
