# -*- encoding: utf-8 -*-

import random
import time
import os
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

EPOCHS = 15
SAMPLES = 10000
BATCH_SIZE = 32
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'pooler'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']

Temperature = 0.05
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

ROBERTA = '/nlp_models/chinese_roberta_wwm_ext_pytorch'#
model_path = ROBERTA 

SAVE_PATH = './saved_model/'
train_path_unsp = '/data/lijinlin/kdd文件/ljl/KDD模型实验/对比学习/poi文本/poi文本_train_2.csv'


def load_sts_data_unsup(path):

    train_data = pd.read_csv(train_path_unsp)
    ids = train_data['ID'].values.tolist()
    text_lines1 = train_data['内容'].values.tolist()
    labels = train_data['评分'].values.tolist()
    return ids, text_lines1, labels

def load_test_unsup(path):
    test_data = pd.read_csv(train_path_unsp)
    labels = test_data['评分'].unique().tolist()
    id =[]
    review =[]
    label =[]
    for x in labels:
        df = test_data[test_data['评分'] == x]
        id.append(list(df['ID']))
        review.append(list(df['内容']))
        label.append(list(df['评分']))
    return id, review, labels


class TrainDataset(Dataset):
    def __init__(self, data: List):
        self.id, self.anchor, self.label = data
      
    def __len__(self):
        return len(self.anchor)

    def text_2_id(self, text: str):

        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.id[index], self.text_2_id(self.anchor[index]), self.label[index]
    
    


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
            f = out.last_hidden_state[:, 0]  # [batch*2, 768]
            return F.normalize(f,dim=1) 
        

        if self.pooling == 'pooler':
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




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(DEVICE)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
    
            mask = torch.eq(labels, labels.T).float().to(DEVICE)
      
        else:
            mask = mask.float().to(DEVICE)

        contrast_count = features.shape[1] 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
      
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) 
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(DEVICE),
            0
        )
        #print(logits_mask.shape)  #torch.Size([128, 128])
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

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

            
def train(model, train_dl, criterion_supcon, optimizer) -> None:
    model.train()
    global best
    for batch_idx, data in enumerate(tqdm(train_dl), start=1):
        ids, source, labels=[d for d in data]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num, -1).to(DEVICE)
        
        features1 = model(input_ids, attention_mask, token_type_ids) 
        features2 = model(input_ids, attention_mask, token_type_ids) 
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss_sup = criterion_supcon(features, labels)
        optimizer.zero_grad()
        loss_sup.backward()
        optimizer.step()
        

        loss = loss_sup.item()
        if batch_idx % 10 == 0:
            logger.info(f"Epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss:>10f}")
        if batch_idx % 100 == 0:
            filepath = os.path.join(SAVE_PATH, "epoch_{0}-batch_{1}-loss_{2:.6f}".format(epoch, batch_idx, loss))#
            torch.save(model.state_dict(), filepath)

if __name__ == '__main__':

    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_sts_data_unsup(train_path_unsp)
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE, shuffle =True)

    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)  
    criterion_supcon = SupConLoss(temperature = Temperature).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train

    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, criterion_supcon, optimizer)


    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')

