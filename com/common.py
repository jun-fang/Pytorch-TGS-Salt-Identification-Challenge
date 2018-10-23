## numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg
# print(matplotlib.get_backend())
#print(matplotlib.__version__)


## torch libs
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel


## std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools

import os
import csv
import pandas as pd
import pickle
import glob
from  glob import glob as gb

import sys
from distutils.dir_util import copy_tree
import time
from time import gmtime, strftime
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## constant #
PI  = np.pi
INF = np.inf
EPS = 1e-12

## project path
PROJECT_DIR = os.path.abspath(__file__).replace('/com/common.py', '')

## data path
DATA_DIR = os.path.abspath('..') + os.sep + 'data'

## time
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

## other defined libs
from utils.file import *
from net.rate import *
from net.loss   import *
from net.metric import *
from net.sync_batchnorm.batchnorm import *

from net.lovasz_losses import *
import net.lovasz_losses as L

from data.transform import *
from data.load_data import *

#---------------------------------------------------------------------------------

class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

#---------------------------------------------------------------------------------

print('@%s:  ' % os.path.basename(__file__))

if 1:
    SEED = 35202  #123  #35202   #int(time.time()) #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)

if 1:
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


print('')

#---------------------------------------------------------------------------------
