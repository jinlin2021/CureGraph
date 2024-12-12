import torch 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import scale
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import Parameter
import numpy as np, itertools, random, copy, math
import pandas as pd
import scipy.sparse as sp
from gcn import GCNII_lyc
from tasks import adj_space
import ipdb
import pickle

  


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output






def adjConcat(a, b):

    lena,_ = a.shape
    lenb,_ = b.shape
    left = torch.row_stack((a, torch.zeros((lenb, lena)).cuda()))  
    right = torch.row_stack((torch.zeros((lena, lenb)).cuda(), b))  
    result = torch.hstack((left, right))  
    return result

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (mu - data) / sigma


class SM_GCN(nn.Module):
    def __init__(self, t_dim, v_dim, p_dim, n_dim, nlayers, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph='full', modals=['t','v','p'], use_modal=False):
        super(SM_GCN, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
                               return_feature=return_feature, use_residue=use_residue)
        
        self.t_fc = nn.Linear(300, 128) 
        self.v_fc = nn.Linear(300, 128)
        self.p_fc = nn.Linear(300, 128)
        if self.use_residue:
            self.feature_fc = nn.Linear(n_dim*3+nhidden*3, nhidden) 
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden) 
        self.final_fc = nn.Linear(nhidden*3, 4)
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.dropout_ = nn.Dropout(self.dropout)
        self.use_modal = use_modal

    def forward(self, t, v, p, dia_len,vid_index):
        print(t.shape) 
        print(v.shape) 
        print(p.shape) 
        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx) 
            if 't' in self.modals:
                t += emb_vector[0].reshape(1, -1).expand(t.shape[0], t.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'p' in self.modals:
                p += emb_vector[2].reshape(1, -1).expand(p.shape[0], p.shape[1])


        adj_la = self.create_big_adj(t, v, p, dia_len, self.modals,vid_index)

        if len(self.modals) == 3:
            features = torch.cat([t, v, p], dim=0).cuda()
           
        elif 't' in self.modals and 'v' in self. modals:
            features = torch.cat([t, v], dim=0).cuda()
        elif 'v' in self.modals and 'p' in self.modals:
            features = torch.cat([v, p], dim=0).cuda()
        elif 't' in self.modals and 'p' in self.modals:
            features = torch.cat([t, p], dim=0).cuda()
        else:
            return NotImplementedError
        features_e = self.graph_net(features, None, adj_la)
        all_length = t.shape[0] if len(t)!=0 else v.shape[0] if len(v) != 0 else p.shape[0]
        if len(self.modals) == 3: 
            neighbor_emb = torch.cat([features_e[:all_length], features_e[all_length:all_length * 2], features_e[all_length * 2:all_length * 3]], dim=-1)
        if self.return_feature:
            ft = features_e[:all_length]
            fv = features_e[all_length:all_length * 2]
            fp = features_e[all_length * 2:all_length * 3]

            return features_e, adj_la, neighbor_emb, ft, fv, fp
        else:
            return F.softmax(self.final_fc(features), dim=-1)


    def create_big_adj(self, t, v, p, dia_len, modals, vid_index):
        modal_num = len(modals)
        all_length = t.shape[0] if len(t)!=0 else v.shape[0] if len(v) != 0 else p.shape[0]
        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).cuda()
        if len(modals) == 3:
            features = [t, v, p] 
        elif 't' in modals and 'v' in modals:
            features = [t, v]
        elif 'v' in modals and 'p' in modals:
            features = [v, p]
        elif 't' in modals and 'p' in modals:
            features = [t, p]
        else:
            return NotImplementedError


        sp = []
        distance_s =[]
        for index in vid_index:
            distance,_ = adj_space(index)
            distance = torch.FloatTensor(distance)
            distance_s.append(distance)

        start = 0
        for i in range(len(dia_len)):
            #Calculate modality similarity.
            sub_adjs = []
            for j, x in enumerate(features): 
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) 
                    temp = x[start:start + dia_len[i]] 
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1)) 
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi 
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix  
  
                sub_adjs.append(sub_adj) 
            dia_idx = np.array(np.diag_indices(dia_len[i]))  
            #For all neighborhoods within the same street, calculate the edge weights between different modalities and create a fully connected network.
            for m in range(modal_num): 
                for n in range(modal_num): 
                    m_start = start + all_length*m 
                    n_start = start + all_length*n
                    if m == n:
                        sub_adj_spital = torch.div(sub_adjs[m], torch.log(distance_s[i]+1)+0.0001) 
                        _, number = sub_adj_spital.shape
                        k = 20
                        for e in range(number):
                            sub_adj_spital[torch.argsort(sub_adj_spital[:, e])[:-k], e] = 0
                            sub_adj_spital[e, torch.argsort(sub_adj_spital[e, :])[:-k]] = 0
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adj_spital
                    
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx] = dia_sim 
                        print(adj.shape) 

            start += dia_len[i]

        
        d = adj.sum(1)
        eps = 1e-12
        d = torch.where(d == 0.0, torch.tensor(eps).cuda(), d)
        x= torch.pow((d), -0.5)  
        D = torch.diag(torch.pow((d), -0.5)) 
        adj_la = D.mm(adj).mm(D) 
        return adj_la



def symmetric_normalized_laplacian(adjacency_matrix):
    
    degree = adjacency_matrix.sum(dim=1)
    degree_inv_sqrt = torch.sqrt(1.0 / degree)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    normalized_laplacian = torch.eye(adjacency_matrix.size(0)).to(adjacency_matrix.device) - \
                           degree_inv_sqrt.view(-1, 1) * adjacency_matrix * degree_inv_sqrt.view(1, -1)
    return normalized_laplacian

