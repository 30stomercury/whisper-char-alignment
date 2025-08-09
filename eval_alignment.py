import os
import joblib
import argparse
from tqdm import tqdm
from retokenize import remove_punctuation
from metrics import eval_n1, get_seg_metrics, eval_n1_strict
import pickle

def eval_ali(args):
    preds = joblib.load(args.pred)
    ami_kaldi = joblib.load("ami_kaldi.pkl")

    pred_ali = {}
    htk_ali = {}
    for i in range(len(preds)):
        if not preds[i]:
            continue
        fid = preds[i]['fids'].replace('eval_', '').upper()
        pred_ali[fid] = {
            'starts': preds[i]['starts_hat'], 
            'ends': preds[i]['ends_hat'], 
            'words': [remove_punctuation(word) for word in preds[i]['predwords']]
        }
        htk_ali[fid] = {
            'starts': preds[i]['starts'], 
            'ends': preds[i]['ends'], 
            'words': [remove_punctuation(word) for word in preds[i]['texts']]
        }
    print(len(list(htk_ali.keys())))

    kaldi_ali = {}
    for key in ami_kaldi.keys():
        fid = key.upper()
        kaldi_ali[fid] = {
            'starts': [l[1] for l in ami_kaldi[key]], 
            'ends': [l[2] for l in ami_kaldi[key]],             'words': [remove_punctuation(l[0])for l in ami_kaldi[key]]
        }

    # testing
    corrects = 0
    total_preds = 0
    total_gts = 0
    for k in tqdm(htk_ali.keys()):
        if not kaldi_ali.get(k):
            continue
        htk_ends = htk_ali[k]['ends']
        htk_words = htk_ali[k]['words']
        kaldi_ends = kaldi_ali[k]['ends']
        kaldi_words = kaldi_ali[k]['words']
        pred_ends = pred_ali[k]['ends']
        pred_words = pred_ali[k]['words']

        print(f"htk: {htk_ends}")
        print(f"kaldi: {kaldi_ends}")
        print(f"pred: {pred_ends}")

        gt_ends = eval(f"{args.gt}_ends")
        gt_words = eval(f"{args.gt}_words")
        tp, fp, fn = eval_n1_strict(gt_ends, pred_ends, gt_words, pred_words, tolerance=args.tolerance)
        corrects += tp
        total_gts += (tp + fn)
        total_preds += (tp + fp)

    precision, recall, f1, r_value, _ = get_seg_metrics(corrects, corrects, total_preds, total_gts)
    results = dict(precision=precision, recall=recall, f1=f1, r_value=r_value)
    print(precision, recall, f1, r_value)

def parse_args():
    parser = argparse.ArgumentParser(description="eval alignment")
    parser.add_argument('--pred', type=str, required=True) # /path/to/*-prediction.pkl
    parser.add_argument('--gt', type=str, default='htk', choices=["htk", "kaldi"])
    parser.add_argument('--tolerance', type=float, default=0.05)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval_ali(args)
