import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
from torch.utils.data import DataLoader
import numpy as np

class UbranModalDataset(Dataset):

    def __init__(self, train =True):
        
        path ='/path/dataset.pkl'
        self.areaIDs, self.HouseText, self.HouseImage,\
        self.HousePOIText, self.Labels1, self.Labels2, self.Labels3, self.Labels4,\
        self.HousePrice, self.streets = pickle.load(open(path, 'rb'))
        self.keys = [x for x in self.streets]
        self.len = len(self.areaIDs)

    def __getitem__(self, index):
        vid = self.keys[index]
        text = torch.FloatTensor(self.HouseText[vid])
        image = torch.FloatTensor(self.HouseImage[vid])
        poitext = torch.FloatTensor(self.HousePOIText[vid])
        Labellen = torch.FloatTensor([1]*len(self.Labels1[vid]))
        MCIlabel = torch.FloatTensor(self.Labels1[vid]) #MCI
        Hypertensionlabel = torch.FloatTensor(self.Labels2[vid])
        diabetes = torch.FloatTensor(self.Labels3[vid])
        MDD = torch.FloatTensor(self.Labels4[vid])

        return text, image, poitext, Labellen, MCIlabel, Hypertensionlabel,\
            diabetes, MDD, vid
               

    def __len__(self):
        return self.len
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<8 else dat[i].tolist() for i in dat]
    

