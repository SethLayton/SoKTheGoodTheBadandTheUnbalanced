#!/usr/bin/env python
"""
main.py

The default training/inference process wrapper
Requires model.py and config.py

Usage: $: python main.py [options]
"""
from __future__ import absolute_import
import argparse
import os
import sys
import torch
import importlib
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import pickle as pkl

sys.path.insert(0,'../../LFCC-LCNN')
sys.path.insert(0,'../LFCC-LCNN')

from core_scripts.other_tools import display as nii_warn
from core_scripts.data_io import default_data_io as nii_dset
from core_scripts.data_io import conf as nii_dconf
from core_scripts.other_tools import list_tools as nii_list_tool
# from core_scripts.config_parse import config_parse as nii_config_parse
from core_scripts.config_parse import arg_parse as nii_arg_parse
from core_scripts.op_manager import op_manager as nii_op_wrapper
from core_scripts.nn_manager import nn_manager as nii_nn_wrapper
from core_scripts import startup_config as nii_startup
import core_scripts.data_io.io_tools as nii_io_tk
import core_scripts.data_io.seq_info as nii_seqinfo

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

def main():
    """ main(): the default wrapper for training and inference process
    Please prepare config.py and model.py
    """

    args = nii_arg_parse.f_args_parsed()

    nii_warn.f_print_w_date("Start program", level='h')
    nii_warn.f_print("Load module: %s" % (args.module_config))
    nii_warn.f_print("Load module: %s" % (args.module_model))
    prj_conf = importlib.import_module(args.module_config)
    prj_model = importlib.import_module(args.module_model)
    nii_startup.set_random_seed(args.seed, args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

        
    params = {'batch_size':  args.batch_size,
            'shuffle': False,
            'num_workers': args.num_workers}

    if args.real_only == True:
        prj_conf.test_set_name = prj_conf.test_set_name_ro
        prj_conf.test_list = prj_conf.test_list_ro
        prj_conf.test_input_dirs = prj_conf.test_input_dirs_ro

    if args.designator == "pretrained":
        tmp = (args.pretrained_model).split(".pt")
        model_name = f"{tmp[0]}_p.pt{tmp[1]}"
    else:
        model_name = args.pretrained_model
    
    checkpoint = torch.load(model_name,map_location=device)
    
    if args.designator == "pretrained":
        tmp = (args.scores_file).split(".score")
        scores_file = f"{tmp[0]}_p.score{tmp[1]}"
    else:
        scores_file = args.scores_file

    prj_conf.optional_argument = [""]
    
    t_lst = nii_list_tool.read_list_from_text(prj_conf.test_list)

    test_set = nii_dset.NIIDataSetLoader(
        prj_conf.test_set_name, \
        t_lst, \
        prj_conf.test_input_dirs,
        prj_conf.input_exts, 
        prj_conf.input_dims, 
        prj_conf.input_reso, 
        prj_conf.input_norm,
        prj_conf.test_output_dirs, 
        prj_conf.output_exts, 
        prj_conf.output_dims, 
        prj_conf.output_reso, 
        prj_conf.output_norm,
        './',
        params = params,
        truncate_seq= None,
        min_seq_len = None,
        save_mean_std = False,
        wav_samp_rate = prj_conf.wav_samp_rate,
        global_arg = args)
    
    
    model = prj_model.Model(test_set.get_in_dim(),test_set.get_out_dim(),args, prj_conf)


    with open(scores_file, "w+") as f:
        nii_nn_wrapper.f_inference_wrapper(args, model, device,test_set, checkpoint, f)
    
    lines = []
    with open(scores_file, "r") as f:
        lines = f.readlines()

    scr = []
    for x in lines:
        _,name,_,score = x.split(", ")
        scr.append((name, score))


    scr = np.array(scr)
    np_scores = scr[:,1].astype(np.float64)
    t = torch.from_numpy(np_scores)


    prob = (torch.sigmoid(t)).data.cpu().numpy()
    prob = np.vstack((np.ones(prob.shape) - prob, prob)).T
    pred = np.argmax(prob, axis=1)

    with open(scores_file, "w") as fh:
        for f, cm, p, pr in zip(scr[:,0],np_scores, pred, prob):
            fh.write('{} {} {} {}\n'.format(f, cm, p, pr))
    return

if __name__ == "__main__":
    main()

