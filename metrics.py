import os
import yaml
import glob
import numpy as np
import torch
import torchaudio

class AlignmentMetric:
    def __init__(self, tolerance):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0
        self.tolerance = tolerance
        self.eps = 1e-5

    def cal_R_val(self, P, R, eps=1e-10):
        OS = R / (P + eps) - 1
        r1 = np.sqrt((1 - R) ** 2 + OS ** 2)
        r2 = (-OS + R - 1) / np.sqrt(2)
        R_val = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return R_val

    def get_metrics(self, precision_counter, recall_counter, pred_counter, gt_counter):
        precision = precision_counter / (pred_counter + self.eps)
        recall = recall_counter / (gt_counter + self.eps)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)
        rval = self.cal_R_val(precision, recall)
        return precision, recall, f1, rval

    def get_final_metrics(self):
        return self.get_metrics(self.precision_counter, self.recall_counter, self.pred_counter, self.gt_counter)

    def zero(self):
        self.precision_counter = 0
        self.recall_counter = 0
        self.pred_counter = 0
        self.gt_counter = 0

    def update(self, gt, pred):
        precision_counter = 0
        recall_counter = 0
        pred_counter = 0
        gt_counter = 0

        gt, pred = np.array(gt), np.array(pred)
        for pred_i in pred:
            min_dist = np.abs(gt - pred_i).min()
            precision_counter += (min_dist <= self.tolerance)
        for gt_i in gt:
            min_dist = np.abs(pred - gt_i).min()
            recall_counter += (min_dist <= self.tolerance)
        pred_counter += len(pred)
        gt_counter += len(gt)

        self.precision_counter += precision_counter
        self.recall_counter += recall_counter
        self.pred_counter += pred_counter
        self.gt_counter += gt_counter

        return self.get_metrics(precision_counter, recall_counter, pred_counter, gt_counter)


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

def get_seg_metrics(correct_predict, correct_retrieve, total_predict, total_gold):
    EPS = 1e-7
    
    precision = correct_predict / (total_predict + EPS)
    recall = correct_retrieve / (total_gold + EPS)
    f1 = 2 * (precision * recall) / (precision + recall + EPS)
    
    os = recall / (precision + EPS) - 1
    r1 = math.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (math.sqrt(2))
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
