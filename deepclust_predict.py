
import time

import os
import sys
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from glob import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable



# custom
from util import load_model
from util import AverageMeter, Logger, UnifLabelSampler

np.random.seed(6)
lbl_cmap = random_label_cmap()

# restriction to float for argument '--example_prob'

# arguments
parser = argparse.ArgumentParser(
    description='''A module to infer classes of images based on deepcluster trained net''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Infer classes wisely''')
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--model', default=os.path.join(os.getcwd(), 'models'),
                    help='directory with models. Default - WD/models')
parser.add_argument('--out', default=os.path.join(os.getcwd(), 'clust_infer'), help='output dir. Default - WD/clust_infer')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()

def main():
    # load model
    model = load_model(argsP.model)
    model.cuda()
    model.eval()



    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          ])





    #### Random chunks #####



if __name__ == "__main__":




