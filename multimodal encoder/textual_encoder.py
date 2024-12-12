# -*- encoding: utf-8 -*-

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
from transformers import BertConfig, BertModel, BertTokenizer

EPOCHS = 2
SAMPLES = 10000
BATCH_SIZE = 64
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu') 

ROBERTA = '/nlp_models/chinese_roberta_wwm_ext_pytorch'
model_path = ROBERTA 

SAVE_PATH = ''

train_path_unsp = '/postive_pairs_text.csv'

def load_sts_data_unsup(path):
    data = pd.read_csv(path)
    text_lines1 = data['tese'].values.tolist()
    text_lines2 = data['tese_right'].values.tolist()
    return text_lines1, text_lines2

class TrainDataset(Dataset):
    def __init__(self, data: List):
        self.anchor, self.positive = data
      
    def __len__(self):
        return len(self.anchor)
    
    def text_2_id(self, text1: str, text2: str):
        return tokenizer([text1, text2], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.anchor[index],self.positive[index])

class SimcseModel(nn.Module):
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)       
        config.attention_probs_dropout_prob = DROPOUT
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)
    
        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]
        
        if self.pooling == 'pooler':
            return out.pooler_output
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    sim = sim / 0.05
    loss = F.cross_entropy(sim, y_true)
    return loss

def eval(model, dataloader) -> float:
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
            return out.pooler_output            # [batch*2, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch*2, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
    
    
def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    sim = sim / 0.05
    loss = F.cross_entropy(sim, y_true)
    return loss


def eval(model, dataloader) -> float:
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
            label_array = np.append(label_array, np.array(label))
    # corrcoef 
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation

            
def train(model, train_dl, optimizer) -> None:

    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
        
        out = model(input_ids, attention_mask, token_type_ids)   

        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        loss = loss.item()
        if batch_idx % 10 == 0:
            logger.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")
        if batch_idx % 1 == 0:
            #torch.save(model.state_dict(), SAVE_PATH)
            torch.save(model.state_dict(), SAVE_PATH)


if __name__ == '__main__':

    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_data = load_sts_data_unsup(train_path_unsp)
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
 
        train(model, train_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    model.load_state_dict(torch.load(SAVE_PATH))
 