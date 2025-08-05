# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
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

from image_dataset import ImageDataset
from ImageEmb import PlaceImageEmb 


data_dir = "./居住区visual"
train_path_list_path = './pair-2-train.pkl'
val_path_list_path = './pair-2-test.pkl'



image_path = ''  #path of images

old_ckpt_path = '' # saved model path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 256
batch_size = 32
threshold = 0.5
embedding_dim = 200

def test_embedding(model, dataloader):
    model.eval()
    fips_list=[]
    embedding=[]
    for images, fips in tqdm(dataloader,ascii=True):   
        images = images.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(images)
        embedding.extend(np.array(outputs.cpu()))
        fips_list.extend(np.array(fips))
    embedding=np.array(embedding)
    return embedding,fips_list


transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

if __name__ == '__main__':
    # data
    with open(image_path, 'rb') as f:
        image_path_list = pickle.load(f)
    
    test_image_set={"all":set()}
    dataset_test = ImageDataset(data_dir, image_path_list, transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    # model 
    model = PlaceImageEmb(embedding_dim=embedding_dim)
    model = model.to(device)
    # load old parameters
    checkpoint = torch.load(old_ckpt_path, map_location=device)
    if old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print('Old checkpoint loaded: ' + old_ckpt_path)
   
    embedding, fips_list = test_embedding(model, dataloader_test)
    
    with open("embeddingny_bj_sv.pickle","wb") as file:
        pickle.dump(embedding,file)
    with open("fipsny_bj_sv.pickle","wb") as file:
        pickle.dump(fips_list,file)
    
    
