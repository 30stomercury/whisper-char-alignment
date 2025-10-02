import os
import joblib
import argparse
from tqdm import tqdm
from retokenize import remove_punctuation
from metrics import eval_n1, get_seg_metrics, eval_n1_strict
import pickle

def run_eval(args):
    preds = joblib.load(args.pred)
    pred_ali = {}
    gt_ali = {}
    for i in range(len(preds)):
        if not preds[i]:
            continue
        fid = preds[i]['fids'].replace('eval_', '').upper()
        pred_ali[fid] = {
            'starts': preds[i]['starts_hat'], 
            'ends': preds[i]['ends_hat'], 
            'words': [remove_punctuation(word) for word in preds[i]['predwords']]
        }
        gt_ali[fid] = {
            'starts': preds[i]['starts'], 
            'ends': preds[i]['ends'], 
            'words': [remove_punctuation(word) for word in preds[i]['texts']]
        }

    # testing
    corrects = 0
    total_preds = 0
    total_gts = 0
    for k in tqdm(gt_ali.keys()):
        gt_ends = gt_ali[k]['ends']
        gt_words = gt_ali[k]['words']
        pred_ends = pred_ali[k]['ends']
        pred_words = pred_ali[k]['words']

        print(f"gt: {gt_ends}")
        print(f"pred: {pred_ends}")

        tp, fp, fn = eval_n1_strict(gt_ends, pred_ends, gt_words, pred_words, tolerance=args.tolerance)
        corrects += tp
        total_gts += (tp + fn)
        total_preds += (tp + fp)

    precision, recall, f1, r_value, _ = get_seg_metrics(corrects, corrects, total_preds, total_gts)
    results = dict(precision=precision, recall=recall, f1=f1, r_value=r_value)
    print("-----------------")
    print(f"precision: {precision:.2f}")
    print(f"recall: {recall:.2f}")
    print(f"f1: {f1:.2f}")
    print(f"r value: {r_value:.2f}")
    print("-----------------")


def parse_args():
    parser = argparse.ArgumentParser(description="eval alignment")
    parser.add_argument('--pred', type=str, required=True) # /path/to/*-prediction.pkl
    parser.add_argument('--tolerance', type=float, default=0.05)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
