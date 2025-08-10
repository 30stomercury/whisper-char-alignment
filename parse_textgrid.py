import glob
import numpy as np
import os
from textgrid import TextGrid
from tqdm import tqdm
from collections import defaultdict
import joblib

root = '/home/s2522924/datasets/out_train_whisper'
files = glob.glob(os.path.join(root, '**', '*.TextGrid'), recursive=True)

output = defaultdict(str)
for n, fpath in enumerate(tqdm(files)):
    starts = []
    ends = []
    words = []
    tg = TextGrid.fromFile(fpath)
    fname = fpath.split('/')[-1].split('.')[0]
    label = []
    for e in tg[0]:
        start = e.minTime
        end = e.maxTime
        # phn = phn_map(e.mark)
        word = e.mark
        if word == '':
            continue
        starts.append(start)
        ends.append(end)
        words.append(word)
    output[fname] = dict(starts=starts, ends=ends, words=words)

joblib.dump(output, 'timit_alignment_train_whisper.pkl')