from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score




class ImagePairDataset(Dataset):
    def __init__(self, root_dir, path_list, transform):
        self.root_dir = root_dir
        self.path_list = path_list
        self.transform = transform
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        anc_path, pos_path, fips = self.path_list[idx]
        anc_path1 = os.path.join(self.root_dir, anc_path)
        anc_image = Image.open(anc_path1)
        pos_path1 = os.path.join(self.root_dir, pos_path)
        pos_image = Image.open(pos_path1)
        if not pos_image.mode == 'RGB':
            pos_image = pos_image.convert('RGB')
        if not anc_image.mode == 'RGB':
            anc_image = anc_image.convert('RGB')
        anc_image = self.transform(anc_image)
        pos_image = self.transform(pos_image)
        sample = [anc_image, pos_image, fips]
        return sample
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        #pad_sequence
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
        

class ImageDataset(Dataset):
    def __init__(self, root_dir, path_list, transform):
        self.root_dir = root_dir
        self.path_list = path_list
        self.transform = transform
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        #anc_path, fips = self.path_list[idx]
        fips, anc_path = self.path_list[idx]
        anc_path1 = os.path.join(self.root_dir, anc_path)
        anc_image = Image.open(anc_path1)
        if not anc_image.mode == 'RGB':
            anc_image = anc_image.convert('RGB')
        anc_image = self.transform(anc_image)
        sample = [anc_image, fips]
        return sample