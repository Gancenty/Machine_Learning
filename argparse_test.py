import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument("--seed",type=int,default=5319)

    parser.add_argument("--dataset_dir",type=str,default=r"./DATASETS/classify_leaves")
    parser.add_argument("--check_point_dir",type=str,default=r"./CHECK_POINT/classify_leaves.pt")

    parser.add_argument("--epoch",type=int,default=10)
    parser.add_argument("--device",type=str,default="cuda")
    
    arg = parser.parse_args(args=[])
    return arg

args = parse_arg()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


print(torch.cuda.device_count())