import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.nn import functional as F

import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

from lcnn import LCNN
import scipy.io as sio
REALID = NEGID = 0
FAKEID = POSID = 1
from utils import datatransform
from dataset import NumpyDataset


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--distribution', type=str, default='90-10')
        parser.add_argument('--testlabelfile', type=str, default='../../CFADEval/protocols/cfad_eval.txt')
        parser.add_argument('--database_path', type=str, default='../../../../DataSets/CFAD/tst/')
        parser.add_argument('--designator', type=str, default="retrained")
        parser.add_argument('--real_only', type=bool, default=False)
        args = parser.parse_args()
        #load yaml file & set comet_ml config
        dir_yaml = 'test.yaml'
        with open(dir_yaml, 'r') as f_yaml:
                parser = yaml.safe_load(f_yaml)

        #device setting
        
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')
        # torch.cuda.set_device(device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser['gpu_idx'][0])

        #define model
        model = LCNN()

        if args.designator == "pretrained":
                tmp = (f'../models/{args.distribution}.pt').split(".pt")
                mod_name = f"{tmp[0]}_p.pt{tmp[1]}"
        else:
                mod_name = f'../models/{args.distribution}.pt'

        model.load_state_dict(torch.load(mod_name, map_location=device))
        model=model.to(device)
        #get utt

        test_label=args.testlabelfile
        test_wavlist, test_labellist= datatransform(test_label, args.database_path)

        evalset = NumpyDataset(parser, test_wavlist, test_labellist, is_eval=True)
        evalset_gen = data.DataLoader(evalset,
                batch_size = parser['batch_size'],
                shuffle = False,
                drop_last = False,
                num_workers = parser['num_workers'])
                
        model.eval()

        if args.real_only:
                scores_file = f'../results/{args.distribution}_ro-eval.scores'
        else:
                scores_file = f'../results/{args.distribution}_cfad-eval.scores'

        if args.designator == "pretrained":
                tmp = scores_file.split(".score")
                scores_file = f"{tmp[0]}_p.score{tmp[1]}"
        else:
                scores_file = scores_file


        

        with torch.set_grad_enabled(False):
                with tqdm(total = len(evalset_gen), ncols = 70) as pbar:
                        y_score1 = [] # score for each sample
                        y1 = [] # label for each sample
                        y1_proba = []
                        y1_probb = []
                        fnames = []
                        for m_batch, m_label, m_fnames in evalset_gen:
                                m_batch = m_batch.to(device=device,dtype=torch.float)
                                y1.extend(list(m_label))
                                logits1, out1 = model(m_batch)
                                probs = F.softmax(logits1, dim=-1)
                                y_score1.extend([probs[i, FAKEID].item() for i in range(probs.size(0))])
                                y1_proba.extend(list(probs[:,0]))
                                y1_probb.extend(list(probs[:,1]))
                                fnames.extend(list(m_fnames))
                
                                pbar.update(1)
                #calculate EER
                f_res = open(scores_file, 'w')
                for _t, _s, _p1, _p2, _f in zip(y1, y_score1, y1_proba, y1_probb, fnames):
                    f_res.write('{name} {score} {target} [{proba}, {probb}]\n'.format(score=_s,target=_t, name=_f, proba=_p1, probb=_p2))
                f_res.close()
