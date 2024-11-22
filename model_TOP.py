import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import scale
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
import pandas as pd
from mm_TOP import SM_GCN
import ipdb






class DeepFc(nn.Module):  
    def __init__(self, input_dim = 900, output_dim = 96):

        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            )

    def forward(self, x):
        output = self.model(x) 
        return output



def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [m, d]
    Returns:
        dist: pytorch Variable, with shape
    """
    dist = torch.sum(F.pairwise_distance(x, y, p=2))
    return dist


def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape  
    mat_expand = torch.unsqueeze(mat_2, 0) 
    mat_expand = mat_expand.expand(n, n, m)  
    mat_expand = mat_expand.permute(1, 0, 2)  
    inner_prod = torch.mul(mat_expand, mat_1)  
    inner_prod = torch.sum(inner_prod, axis=-1)  
    return inner_prod


def _adj_loss(embeddings, adj):
    inner_prod = pairwise_inner_product(embeddings, embeddings)
    loss= torch.nn.MSELoss()
    return loss(inner_prod, adj)


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, adj, emb, featuret, featurev, featurep):
        loss = _adj_loss(emb, adj) + 0.9*euclidean_dist(featuret,featurev) + 0.9*euclidean_dist(featuret,featurep) 
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1)
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss



class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score




class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, amask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=amask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions



def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
    

def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features, None, None, None, None

        
def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = [] 
    
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])
  
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))
    
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()
        
            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
    
    node_features = torch.cat(node_features, dim=0) 
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()
    
    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths 




class SpatialGCNModel(nn.Module):
 
    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a, graph_hidden_size, dropout_rec=0.5, dropout=0.5, nodal_attention=True,
                 no_cuda=False, graph_type='SMGCN', alpha=0.2, multiheads=6, graph_construct='direct',use_residue=True,
                 D_m_v=200,D_m_p=768,modals='tvp',att_type='gated',av_using_lstm=False,Deep_GCN_nlayers = 4, dataset='UbranModal',use_modal=False):


        super(SpatialGCNModel, self).__init__()

        self.base_model = base_model
        self.no_cuda = no_cuda
        self.graph_type=graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.dropout = dropout
        self.use_residue = use_residue
        self.return_feature = True
        self.modals = [x for x in modals] 
        self.use_modal = use_modal
        self.att_type = att_type
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently':
            self.multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset
        self.fc = DeepFc(900, 128)
        

        if self.base_model == 'L':
            if not self.multi_modal:
                if len(self.modals) == 3:
                    hidden_ = 250
                elif ''.join(self.modals) == 'tv':
                    hidden_ = 150
                elif ''.join(self.modals) == 'tp':
                    hidden_ = 150
                else:
                    hidden_ = 100
                self.linear_ = nn.Linear(D_m, hidden_)
                self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            else:
                if 't' in self.modals:
                    hidden_l = 200 
                    self.linear_t = nn.Linear(D_m, hidden_l)
                    self.lstm_t = nn.LSTM(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
                if 'v' in self.modals:
                    hidden_v = 200
                    self.linear_v = nn.Linear(D_m_v, hidden_v)
                    if self.av_using_lstm:
                        self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
                if 'p' in self.modals:
                    hidden_a = 200
                    self.linear_p = nn.Linear(D_m_p, hidden_a)
                    self.lstm_p = nn.LSTM(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
                    if self.av_using_lstm:
                        self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            raise NotImplementedError 

        if self.graph_type=='SMGCN' or self.graph_type=='SMGCN2':
            if self.graph_type == 'SMGCN':
                self.graph_model = SM_GCN(t_dim=2*D_e, v_dim=2*D_e, p_dim=2*D_e, n_dim=2*D_e, nlayers=64, nhidden=graph_hidden_size, dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, use_residue=self.use_residue, modals=self.modals, use_modal=self.use_modal)
            print("construct "+self.graph_type)
        elif self.graph_type=='None':
            print("There are no such kind of graph")

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            if self.att_type == 'concat_subsequently':
                self.smax_fc = nn.Linear(300, 4)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), 4)
                else:
                    self.smax_fc = nn.Linear(100, 4)
            else:
                self.smax_fc = nn.Linear(2*D_e+graph_hidden_size*len(self.modals), 4)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, amask, lengths, U_v=None, U_p=None, vid_area=None):
        #U :文本， u-a：poi，u-v:visual 

        if self.base_model == 'L':
            if not self.multi_modal:
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)
            else:
                if 't' in self.modals:
                    emotions_t = U

                if 'v' in self.modals:
                    emotions_v = U_v
                    if self.av_using_lstm:
                        emotions_v, hidden_v = self.lstm_v(U_v)
                    else:
                        emotions_v = U_v
                if 'p' in self.modals:
                    emotions_p = self.linear_p(U_p)
                   
                    if self.av_using_lstm:
                        emotions_p, hidden_p = self.lstm_a(U_p)
    
               
        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        

        features_t, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_t, lengths, self.no_cuda)
        features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v, lengths, self.no_cuda)
        features_p, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_p, lengths, self.no_cuda)

        if self.graph_type=='SMGCN' or self.graph_type=='SMGCN2':
            features_e, adj, emb, t, v, p = self.graph_model(features_t, features_v, features_p, lengths, vid_area)
            space_smi,_ = adj_space(vid_area)
            space_smi = torch.FloatTensor(space_smi).cuda()
            emb = self.fc(emb)
            # t = F.normalize(t, dim = 1)
            # v = F.normalize(v, dim = 1)
       
        else:
            print("There are no such kind of graph")        
        
        return space_smi, emb, t, v, p


def adj_space(vid):
    
    path = ''
    df_voroni = pd.read_pickle(path)
    com =[]
    for item in list(vid):
        #print(item)
        df = df_voroni[df_voroni['street'] == item] 
        disct = list(df.index) 
        com.extend(disct)
    df_a = df_voroni.loc[com, com] 
    disct_m = np.array(df_a.values)
    disct_m = scale(disct_m) 
   
    return disct_m, com

