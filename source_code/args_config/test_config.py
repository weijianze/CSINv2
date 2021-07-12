#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import os.path as osp

cp_dir   = './checkpoint/'

def test_args():

    parser = argparse.ArgumentParser(description='PyTorch for CSINPlus')

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--workers', type=int,  default=0)  

    parser.add_argument('--in_feats',   type=int,   default=256)
    parser.add_argument('--batch_size',  type=int,   default=50)
    parser.add_argument('--img_size', type=int,   default=128) 
    
    args = parser.parse_args()

    return args
