import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parameter import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
import ipdb
 

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant   #variant=True,
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
        hi = torch.mm(adj, input)
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



class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False):
        super(GCNII, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat+nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel):
        if self.new_graph:
            adj = self.message_passing_directed_speaker(x, dia_len, topicLabel)
        else:
            adj = self.create_big_adj(x, dia_len)
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner
        

    def create_big_adj(self, x, dia_len):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            temp_len = torch.sqrt(torch.bmm(temp.unsqueeze(1),temp.unsqueeze(2)).squeeze(-1).squeeze(-1))
            temp_len_matrix = temp_len.unsqueeze(1)*temp_len.unsqueeze(0)
            cos_sim_matrix = torch.matmul(temp,temp.permute(1,0))/temp_len_matrix
            sim_matrix = torch.acos(cos_sim_matrix*0.99999)
            sim_matrix = 1 - sim_matrix/math.pi

            sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix

            m_start = start
            n_start = start
            adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adj

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def message_passing_wo_speaker(self, x,dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))+torch.eye(x.shape[0])
        start = 0
        for i in range(len(dia_len)): #
            for j in range(dia_len[i]-1):
                for pin in range(dia_len[i] - 1-j):
                    xz=start+j
                    yz=xz+pin+1
                    f = self.cossim(x[xz],x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f >1 and f <1.05:
            f = 1
        elif f< -1 and f>-1.05:
            f = -1
        elif f>=1.05 or f<=-1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        start = 0
        use_utterance_edge=False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_-1):
                    f = self.atom_calculate_edge_weight(x[start+j], x[start+j+1])
                    Aij = 1-math.acos(f) / math.pi
                    adj[start+j][start+j+1] = Aij
                    adj[start+j+1][start+j] = Aij
            for k in range(len(speaker0)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker0[k]], x[start+speaker0[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker0[k]][start+speaker0[k+1]] = Aij
                adj[start+speaker0[k+1]][start+speaker0[k]] = Aij
            for k in range(len(speaker1)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker1[k]], x[start+speaker1[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker1[k]][start+speaker1[k+1]] = Aij
                adj[start+speaker1[k+1]][start+speaker1[k]] = Aij

            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()
        
        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k-window_size)
                right = min(len_-1, k+window_size)
                edge_set = edge_set + [str(i)+'_'+str(j) for i in range(left, right) for j in range(i+1, right+1)]
            edge_set = [[start+int(str_.split('_')[0]),start+int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1-math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()
        
        return adj.cuda()

class GCNII_lyc(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False):
        super(GCNII_lyc, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue   #variant=True,
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            #nhidden =100
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))  #(200,100)

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha  # lamda=0.5, alpha=0.1
        self.lamda = lamda

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, adj=None):
        adj = adj
        _layers = []
        xdropout = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](xdropout)) 
        
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training) 
            
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)  


        if self.use_residue:  
            layer_inner = torch.cat([x, layer_inner], dim=-1)

        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def message_passing_wo_speaker(self, x,dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start+dia_len[i]]
            vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
            norm_temp = (temp.permute(1, 0) / vec_length) 
            cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
            cos_sim_matrix = cos_sim_matrix * 0.99999
            sim_matrix = torch.acos(cos_sim_matrix)

            d = sim_matrix.sum(1)
            D = torch.diag(torch.pow(d, -0.5))

            sub_adj[:dia_len[i], :dia_len[i]] = D.mm(sim_matrix).mm(D)
            adj[start:start+dia_len[i], start:start+dia_len[i]] = sub_adj
            start+=dia_len[i]

        adj = adj.cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f >1 and f <1.05:
            f = 1
        elif f< -1 and f>-1.05:
            f = -1
        elif f>=1.05 or f<=-1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        start = 0
        use_utterance_edge=False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_-1):
                    f = self.atom_calculate_edge_weight(x[start+j], x[start+j+1])
                    Aij = 1-math.acos(f) / math.pi
                    adj[start+j][start+j+1] = Aij
                    adj[start+j+1][start+j] = Aij
            for k in range(len(speaker0)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker0[k]], x[start+speaker0[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker0[k]][start+speaker0[k+1]] = Aij
                adj[start+speaker0[k+1]][start+speaker0[k]] = Aij
            for k in range(len(speaker1)-1):
                f = self.atom_calculate_edge_weight(x[start+speaker1[k]], x[start+speaker1[k+1]])
                Aij = 1-math.acos(f) / math.pi
                adj[start+speaker1[k]][start+speaker1[k+1]] = Aij
                adj[start+speaker1[k+1]][start+speaker1[k]] = Aij

            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()
        
        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len))+torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k-window_size)
                right = min(len_-1, k+window_size)
                edge_set = edge_set + [str(i)+'_'+str(j) for i in range(left, right) for j in range(i+1, right+1)]
            edge_set = [[start+int(str_.split('_')[0]),start+int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1-math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start+=dia_len[i]

        d = adj.sum(1)
        D=torch.diag(torch.pow(d,-0.5))
        adj = D.mm(adj).mm(D).cuda()
        
        return adj.cuda()