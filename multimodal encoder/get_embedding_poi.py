
# -*- coding: utf-8 -*-
import random
import time
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from transformers import BertConfig, BertModel, BertTokenizer

EPOCHS = 2
SAMPLES = 10000
BATCH_SIZE = 128
LR = 5e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

ROBERTA = '/nlp_models/chinese_roberta_wwm_ext_pytorch'#
model_path = ROBERTA 

SAVE_PATH = ''
train_path_unsp = ''


def load_sts_data_unsup(path):

    df = pd.read_pickle(path)
    house_ids = df.ID_left.values.tolist() #小区id
    poi_ids = df.ID_right.values.tolist() #poi id
    texts = df['内容'].values.tolist()  #文本
    poi_type0s = df['type_0'].values.tolist() 
    poi_type0s_counts = df['count_poitype'].values.tolist()
    
    return house_ids, poi_ids, texts, poi_type0s, poi_type0s_counts

class TrainDataset(Dataset):
    def __init__(self, data: List):
        self.house_ids, self.poi_ids, self.texts, self.poi_type0s,\
            self.poi_type0s_counts = data
      
    def __len__(self):
        return len(self.house_ids)
    
    def text_2_id(self, text):
        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.house_ids[index], self.poi_ids[index], self.text_2_id(self.texts[index]), \
            self.poi_type0s[index], self.poi_type0s_counts[index]



class SimcseModel(nn.Module):
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)       
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT           
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
    


if __name__ == '__main__':
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {SAVE_PATH}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    HouseIDs=[]
    PoiIDs=[]
    PoiTypes=[]
    counts=[]
    embedding=[]
    train_data_source = load_sts_data_unsup(train_path_unsp)

    train_dataloader = DataLoader(TrainDataset(train_data_source), batch_size=BATCH_SIZE)
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)  
    model = model.to(DEVICE)

    model.load_state_dict(torch.load(SAVE_PATH))

    for data in tqdm(train_dataloader):
        HouseID, PoiID, source, PoiType, count = [d for d in data]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num , -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num , -1).to(DEVICE)
        
        out = model(input_ids, attention_mask, token_type_ids)   
        embedding.extend(np.array(out.cpu().detach().numpy()))
        HouseIDs.extend(np.array(HouseID))
        PoiIDs.extend(np.array(PoiID)), 
        PoiTypes.extend(np.array(PoiType)), 
        counts.extend(np.array(count))
        dict ={"HouseID": HouseIDs, "PoiID": PoiIDs,'Emb': embedding, 'PoiType': PoiTypes,'count': counts}#将列表a，b转换成字典
        df =pd.DataFrame(dict)
        df_b = df.groupby(['HouseID', 'PoiID'])['Emb'].mean()  
        df_b = b.reset_index() 
        df_b.to_csv('poi_text_mean.csv')

        

       

