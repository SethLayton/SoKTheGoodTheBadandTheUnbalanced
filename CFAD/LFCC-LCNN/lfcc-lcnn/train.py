import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.nn import functional as F
import os
import yaml
import numpy as np
from torch import cuda
from torch import optim
import torch
import torch.nn as nn
from torch.utils import data
from lcnn import LCNN
import scipy.io as sio

REALID = NEGID = 0
FAKEID = POSID = 1
from utils import datatransform, datatransform_plus_balance_classes
from dataset import NumpyDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution', type=str, default='25-75')
    parser.add_argument('--trainlabelfile', type=str, default='../../CFADTrain/protocols')
    parser.add_argument('--database_path', type=str, default='../../../../DataSets/CFAD/trn')
    
    args = parser.parse_args()
    # load yaml file & set comet_ml config
    _abspath = os.path.abspath(__file__)
    dir_yaml = 'train.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.safe_load(f_yaml)

    # device setting
    cuda = cuda.is_available()
    device = torch.device('cuda:%s' % parser['gpu_idx'][0] if cuda else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser['gpu_idx'][0])

    # set save directory
    save_dir = '../models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # define model
    model = LCNN()
    model = model.to(device)

    # set ojbective funtions
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    params = list(model.parameters())
    if parser['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(params,
                                    lr=parser['lr'],
                                    momentum=parser['opt_mom'],
                                    weight_decay=parser['wd'],
                                    nesterov=bool(parser['nesterov']))

    elif parser['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(params,
                                     lr=parser['lr'],
                                     weight_decay=parser['wd'],
                                     betas=[0.9, 0.98],
                                     eps=1.0e-9,
                                     amsgrad=False)

    ##########################################
    # train/val################################
    ##########################################
    trn_label_file = os.path.join(args.trainlabelfile, f"{args.distribution}.txt")
    for epoch in tqdm(range(parser['epoch'])):

        # define dataset generators
        x_train, y_train = datatransform_plus_balance_classes(trn_label_file, args.database_path, epoch)
        trnset = NumpyDataset(parser, x_train, y_train, is_eval=False)         
        
        trnset_gen = data.DataLoader(trnset,
                                     batch_size=parser['batch_size'],
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=parser['num_workers'])

        # train phase
        model.train()
        with tqdm(total=len(trnset_gen), ncols=70) as pbar:
            for m_batch, m_label in trnset_gen:
                m_batch, m_label = m_batch.to(device=device,dtype=torch.float), m_label.to(device)

                logits, _ = model(m_batch)
                loss = criterion(logits, m_label)
                optimizer.zero_grad()
                loss.backward()
                model.parameters()
                optimizer.step()

                pbar.set_description('epoch%d:\t loss_ce:%.3f' % (epoch, loss))
                pbar.update(1)
                


    torch.save(model.state_dict(), os.path.join(save_dir, f"{args.distribution}.pt"))

