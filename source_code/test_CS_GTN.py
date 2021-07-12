#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pdb
import argparse
import time
import math
import torch
import shutil
import random
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import model as mlib
from args_config import test_args
from data_config import CSIR_config, NDCS_config
from data_load import ImageList
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy import io 

torch.backends.cudnn.bencmark = True

from IPython import embed

def get_tpr_at_fpr(tpr, fpr, thr):
    idx = np.argwhere(fpr > thr)
    return tpr[idx[0]][0]

def OneHot(x):
    # get one hot vectors
    n_class = int(x.max() + 1)
    onehot = torch.eye(n_class)[x.long()]
    return onehot # N X D

def get_eer(tpr, fpr):
    for i, fpr_point in enumerate(fpr):
        if (tpr[i] >= 1 - fpr_point):
            idx = i
            break
    if (tpr[idx] == tpr[idx+1]):
        return 1 - tpr[idx]
    else:
        return fpr[idx]

class GTN_Tester(object):

    def __init__(self, args, config, check_path,):
        self.args    = args
        self.model   = dict()
        self.data    = dict()
        self.result  = dict()
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.config = config

        self.check_path = check_path


    def _report_settings(self):
        ''' Report the settings '''

        str = '-' * 16
        print('%sTesting Setting%s' % (str, str))
        print("- Dataset   : {}".format(self.config.data_name))
        print("- Protocol  : {}".format(self.config.test_type))
        print('-' * 52)
        
    def _model_loader(self):

        self.model['backbone'] = mlib.GaborTridentNet(feature_dim=self.args.in_feats)
        if self.use_gpu:
            self.model['backbone']  = self.model['backbone'].cuda()

        if self.use_gpu and len(self.args.gpu_ids) > 1:
            self.model['backbone'] = torch.nn.DataParallel(self.model['backbone'], device_ids=self.args.gpu_ids)
            print('Parallel mode was going ...')
        elif self.use_gpu:
            print('Single-gpu mode was going ...')
        else:
            print('CPU mode was going ...')
       
        pretrained_dict = torch.load(self.check_path)
        self.model['backbone'].load_state_dict(pretrained_dict)
        print('Model loading was finished ...')
       
    def _data_loader(self):
        gallery_loader_param = self.config.gallery_loaderGet()
        self.data['gallery'] = torch.utils.data.DataLoader(
            ImageList(root=gallery_loader_param[0], fileList=gallery_loader_param[1], 
            transform=transforms.Compose([ 
                transforms.Resize((self.args.img_size,self.args.img_size)),
                transforms.ToTensor(),
            ])),batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)

        probe_loader_param = self.config.probe_loaderGet()
        self.data['probe'] = torch.utils.data.DataLoader(
            ImageList(root=probe_loader_param[0], fileList=probe_loader_param[1], 
            transform=transforms.Compose([ 
                transforms.Resize((self.args.img_size,self.args.img_size)),
                transforms.ToTensor(),
            ])),batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)
        print('Data loading was finished ...')
   
    def _test_model(self):
        self.model['backbone'].eval()
        gallery_feature = 1       
        with torch.no_grad():
            for data, label in self.data['gallery']:           
                data.requires_grad = False
                label.requires_grad = False

                if self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                
                feature,_ = self.model['backbone'](data,data,label)
                
                data_feature = feature.squeeze()
                if torch.is_tensor(gallery_feature):
                    gallery_feature = torch.cat((gallery_feature,data_feature),0)
                    gallery_label = torch.cat((gallery_label,label),0)            
                else:
                    gallery_feature = data_feature
                    gallery_label = label
        
        probe_feature = 1        
        with torch.no_grad():
            for data, label in self.data['probe']:           
                data.requires_grad = False
                label.requires_grad = False

                if self.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                    
                _, feature = self.model['backbone'](data, data, label)
                
                data_feature = feature.squeeze()
                if torch.is_tensor(probe_feature):
                    probe_feature = torch.cat((probe_feature,data_feature),0)
                    probe_label = torch.cat((probe_label,label),0)            
                else:
                    probe_feature = data_feature
                    probe_label = label
    
        gallery_feature = gallery_feature/gallery_feature.norm(dim=1,keepdim=True)
        gallery_onehot = OneHot(gallery_label)

        probe_feature = probe_feature/probe_feature.norm(dim=1,keepdim=True)
        probe_onehot = OneHot(probe_label)

        sim_mat = gallery_feature.mm(probe_feature.t())
        sig_mat = torch.mm(gallery_onehot, probe_onehot.t())
        scores = sim_mat.contiguous().view(-1)
        signals = sig_mat.contiguous().view(-1)

        score_matrix = scores.reshape((-1, ))
        label_matrix = signals.reshape((-1, ))
        # pdb.set_trace()
        fpr, tpr, _ = roc_curve(label_matrix.cpu(), score_matrix.cpu(), pos_label=1)
        eer = get_eer(tpr,fpr)
        prec1 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-1)
        prec2 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-3)
        prec3 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-4)
        prec4 = 1.-get_tpr_at_fpr(tpr, fpr, 1e-5)
        print('RESULTS: EER {:.4f} | R@A1e-1 {:.4f} | R@A1e-3 {:.4f} | R@A1e-4 {:.4f} | R@A1e-5 {:.4f}\n'.format(eer,prec1,prec2,prec3,prec4))

    def runner(self):

        self._report_settings()
        self._model_loader()
        self._data_loader()
        self._test_model()

if __name__ == "__main__":  
    input_args = test_args() 
    # # --------------------------------------- NDCS CROSS -----------------------------------------------
    config = NDCS_config()    
    checkpoint_path='./checkpoint/NDCS/GTN4NDCS.pth'  
    # ---------------------------------------- CSIR CROSS -----------------------------------------------
    # config = CSIR_config()   
    # checkpoint_path='./checkpoint/CSIR/GTN4CSIR.pth'
    # ---------------------------------------------------------------------------------------------------
    GTN_tester = GTN_Tester(input_args, config,checkpoint_path)
    GTN_tester.runner()

