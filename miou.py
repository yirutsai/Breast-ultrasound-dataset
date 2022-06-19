import os
from posixpath import dirname
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchsummary import summary
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import cv2
import wandb

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--seed",type =int, default = 3030)
parser.add_argument("--batch_size",type = int, default= 4)
parser.add_argument("--lr", type = float, default = 1e-2)
parser.add_argument("--n_epochs",type = int, default= 80)
parser.add_argument("--clip_grad",type = float,default = 5)
parser.add_argument("--wandb",action="store_true")
parser.add_argument("--loss_type",type=str,default="dice")
parser.add_argument("--output_dir",type =str,default="output")

args = parser.parse_args()
def MIOU(output,target):
    assert torch.max(output).item()<=1
    assert torch.max(target).item()<=1,f"get torch.max(target.item()={torch.max(target).item()}"
    output = torch.round(output)
    target = torch.round(target)
    union = torch.logical_or(output,target)
    intersection = torch.logical_and(output,target)
    if(union.float().sum()==0):
        return 1
    return intersection.float().sum()/union.float().sum()
ct = 0
train_base_mious = []
train_aid_mious = []
test_base_mious = []
test_aid_mious = []
types = ["benign","malignant","normal"]
for t in types:
    for filename in os.listdir(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{t}")):
        if(re.match("^train-.*-mask.png$",filename)):
            mask = cv2.imread(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{t}/{filename}")),cv2.IMREAD_GRAYSCALE)
            output = cv2.imread(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{t}/{filename.replace('mask','pred')}")),cv2.IMREAD_GRAYSCALE)
            aid_output = cv2.imread(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"train/{t}/{filename.replace('mask','pred_aid')}")),cv2.IMREAD_GRAYSCALE)
            mask = torch.tensor(mask/255,dtype=float)
            output = torch.tensor(output/255,dtype=float)
            aid_output = torch.tensor(aid_output/255,dtype = float)
            train_base_mious.append(MIOU(output,mask))
            train_aid_mious.append(MIOU(aid_output,mask))
            ct +=1
for t in types:
    for filename in os.listdir(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{t}")):
        if(re.match("^test-.*-mask.png$",filename)):
            mask = cv2.imread(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{t}/{filename}")),cv2.IMREAD_GRAYSCALE)
            output = cv2.imread(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{t}/{filename.replace('mask','pred')}")),cv2.IMREAD_GRAYSCALE)
            aid_output = cv2.imread(str(Path(os.path.dirname(__file__))/Path(args.output_dir)/Path(f"test/{t}/{filename.replace('mask','pred_aid')}")),cv2.IMREAD_GRAYSCALE)
            mask = torch.tensor(mask/255,dtype=float)
            output = torch.tensor(output/255,dtype=float)
            aid_output = torch.tensor(aid_output/255,dtype = float)
            test_base_mious.append(MIOU(output,mask))
            test_aid_mious.append(MIOU(aid_output,mask))
            ct+=1
print(ct)
print(f"{args.loss_type}: train_base_mious = {sum(train_base_mious)/len(train_base_mious)}   train_aid_mious = {sum(train_aid_mious)/len(train_aid_mious)}")
print(f"{args.loss_type}: test_base_mious = {sum(test_base_mious)/len(test_base_mious)}   test_aid_mious = {sum(test_aid_mious)/len(test_aid_mious)}")