# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from collections import namedtuple, defaultdict
import numpy as np
import os
import sys
from copy import copy
import warnings
warnings.filterwarnings('ignore')
try:
    import ipdb
except:
    pass

from utils import *
from data import TravelDataset, random_split, KGDataset
from ARmodel import ARModel
from model import SPOTModel
import metrics
from trainer import *

import pickle

def main():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    dataset_name = 'Foursquare'
    parser.add_argument('--dataset_name', type=str, default='Foursquare')
    parser.add_argument('--ori_data', type=str, default=f'../{dataset_name}/home.txt')
    parser.add_argument('--dst_data', type=str, default=f'../{dataset_name}/oot.txt')
    parser.add_argument('--trans_data', type=str, default=f'../{dataset_name}/travel.txt')
    parser.add_argument('--save_path', type=str, default=f'../{dataset_name}/model_save')
    parser.add_argument("--best_save", action="store_true")
    parser.add_argument("--kg_path", type=str, default=f'../{dataset_name}/kg.txt')
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--data_split_path', type=str, default=f'../{dataset_name}/data_split.pkl')

    # Training Configurations
    parser.add_argument('--model', type=str, default='SPOT-Trip')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_batch', type=int, default=4)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--test_batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument("--projection_dim", type=int, default=64)

    parser.add_argument('--margin', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr_dc', type=float, default=0.2)
    parser.add_argument('--lr_dc_step', type=int, default=4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=2050)
    parser.add_argument('--log_path', type=str, default='../')
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--name', type=str, default="default")
    # parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--stop_epoch", type=int, default=8) # early stopping
    parser.add_argument("--fine_stop", type=int, default=12)

    # Knowledge Graph (KG) Arguments
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--kg", action="store_true")
    parser.add_argument("--entity_num_per_poi", type=int, default=2) # Note: F 2 For Yelp, use 10
    parser.add_argument("--train_trans", action="store_true")
    parser.add_argument('--trans', type=str, default="transe")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--kgcn", type=str, default="RGAT")
    parser.add_argument("--kg_p_drop", type=float, default=0.5)
    parser.add_argument("--ui_p_drop", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.2)

    # AR-Trip
    parser.add_argument("--drifting", action="store_true")
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--repetition_beta", type=float, default=1.0)
    parser.add_argument("--train_type", type=str, default='Penalty')
    parser.add_argument('--confidence', type=float, default=0.5)
    # ODE
    parser.add_argument("--ode", action="store_true")
    parser.add_argument("--t_unif_res", type=int, default=10, help="Number of point in unfirom temporal grid used for intepolation.")
    parser.add_argument("--solver", type=str, default="dopri5", help="Name of the ODE solver (see torchdiffeq).")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for ODE solver.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for ODE solver.")
    parser.add_argument("--dyn_hid_layers", type=int, default=3, help="Number of hidden layers in dynamics function.")
    parser.add_argument("--dyn_latent_dim", type=int, default=128, help="Hidden layer dimension in dynamics function.")
    # Model (lm).
    parser.add_argument("--lm_hid_layers", type=int, default=3, help="Number of hidden layers in intensity function.")
    parser.add_argument("--lm_latent_dim", type=int, default=128, help="Hidden layer dimension in intensity function.")
    parser.add_argument("--sig_v", type=float, default=0.6, help="Observation variance.")  # Note: F 0.6 For Yelp, use 0.4

    parser.add_argument("--s_infer", action="store_true")

    # Parsing command-line arguments
    args = parser.parse_args()
    set_seeds(args.seed)
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)


    # Initializing a Logger instance for recording various metrics during the training process
    # args.log_path: Path where the log file is saved
    # args.name: Name of the model, used in the log
    # args.seed: Random seed value, also recorded in the log
    # args.log: A boolean value indicating whether to output logs to the console
    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))
    logger.log("Experiment name: %s" % args.name)


    # Loading the travel dataset with parameters and data paths specified in args
    data = TravelDataset(args, args.ori_data, args.dst_data, args.trans_data)

    # Checking if the knowledge graph (KG) option is enabled and loading KG data accordingly
    if args.kg:
        kg_data = KGDataset(args)
    else:
        kg_data = None
    train_data, valid_data, test_data = random_split(data, dataset_name=dataset_name, split_path=args.data_split_path)

    train_loader = DataLoader(train_data, args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, args.test_batch, shuffle=False, collate_fn=collate_fn)

    n_region = len(data.region_idx)
    max_d_length = max(len(seq) for seq in data.dsts)

    if args.model == 'SPOT-Trip':
        model = SPOTModel(args, len(data.poi_idx) + 1, data.region_poi, max_d_length,
                          d_model=args.hidden_size, n_head=4, num_encoder_layers=1, d_z=args.hidden_size, kg_dataset=kg_data).to(args.device)
        train_am = None
        train_pm = None
    # Training or testing the model based on the mode specified in args
    elif args.model == 'AR-Trip':
        train_am = poi_adjacent(train_data, len(data.poi_idx) + 1)
        train_pm, confidence = poi_position(train_data, len(data.poi_idx) + 1, max_d_length)
        train_am = torch.tensor(train_am).to(args.device)
        train_pm = torch.tensor(train_pm).to(args.device)
        args.confidence = confidence
        model = ARModel(args, len(data.poi_idx) + 1, 25, args.drifting, args.guiding, data.region_poi,
                         args.repetition_beta,
                         max_d_length, d_model=args.hidden_size, n_head=4, num_encoder_layers=1).to(args.device)

    if args.mode == 'train':
        best = train_single_phase(model, train_loader, valid_loader, args, logger, kg_data, train_am, train_pm)

        test(model, os.path.join(args.save_path, "model_{}.xhr".format(best)), test_loader, args, logger, n_region, train_am, train_pm)
        print("################## current exp done ##################")
    elif args.mode == 'test':
        test(model, os.path.join(args.save_path, "model_best.xhr"), test_loader, args, logger, n_region, train_am, train_pm)

    logger.close_log()
    
if __name__ == "__main__":
    main()