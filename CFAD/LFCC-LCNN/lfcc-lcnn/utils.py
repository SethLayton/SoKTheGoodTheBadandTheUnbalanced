import os
import numpy as np
import torch
from torch.utils import data

def datatransform(labelfile, db_path):
    feat_list = []
    label_list = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, label = line.strip().split()
            feat_path = os.path.join(db_path, utt_id)
            feat_list.append([utt_id,feat_path])
            if label == "bonafide":
                label = 0
            else:
                label = 1
            label_list.append(int(label))

    return feat_list, label_list

def datatransform_plus_balance_classes(labelfile, database_path, np_seed):
    '''
    Balance number of sample per class.
    Designed for Binary(two-class) classification.
    '''
    feat_list = []
    label_list = []
    list_0 = []
    list_1 = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, label = line.strip().split()
            feat_path = os.path.join(database_path, utt_id)
            if(label=='bonafide'):
                list_0.append([utt_id, feat_path, 0])
            elif(label=='spoof'):
                list_1.append([utt_id, feat_path, 1])

    lines_small, lines_big = list_0, list_1
    if(len(list_0)>len(list_1)):
        lines_small, lines_big = list_1, list_0

    len_small_lines = len(lines_small)
    np.random.seed(np_seed)
    np.random.shuffle(lines_big)
    # new_lines = lines_small + lines_big[:len_small_lines]
    new_lines = lines_small + lines_big

    for line in new_lines:
        utt_id, feat_path, label = line
        feat_list.append([utt_id,feat_path])
        label_list.append(int(label))

    return feat_list, label_list
