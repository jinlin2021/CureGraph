import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import UbranModalDataset
from model_TOP import MaskedNLLLoss, LSTMModel, SpatialGCNModel, MaskedMSELoss, FocalLoss, SimLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import ipdb
import torch.nn.functional as F
from tasks import predict_regression, adj_space

# We use seed = 10 for reproduction of the results reported in the paper.
seed = 10
 
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1, dataset='UbranModal'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


# data loader
def get_data_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = UbranModalDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    return train_loader, valid_loader


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='UbranModal'):
    losses, preds= [], []
    MCIlabels, Hypertensionlabels, Diabeteslabels, MDDlabels =[], [], [], []
    embs =[]
    community =[]
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visuf, poiidf, amask, MCIlabel, Hypertensionlabel,\
            Diabeteslabel, MDDlabel, vid_area= [d for d in data] 
        textf = textf.cuda()
        visuf =visuf.cuda()
        poiidf = poiidf.cuda()
        amask =  amask.cuda()
        MCIlabel = MCIlabel.cuda()
        Hypertensionlabel = Hypertensionlabel.cuda()
        Diabeteslabel = Diabeteslabel.cuda()
        MDDlabel = MDDlabel.cuda()
        
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat': 
                if modals == 'tvp':
                    textf = torch.cat([textf, visuf, poiidf],dim=-1) 
                elif modals == 'tv':
                    textf = torch.cat([textf, visuf],dim=-1) 
                elif modals == 'vp':
                    textf = torch.cat([visuf, poiidf],dim=-1)
                elif modals == 'tp':
                    textf = torch.cat([textf, poiidf],dim=-1) 
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
                textf = textf
        else:
            if modals == 't':
                textf = textf
            elif modals == 'v':
                textf = visuf
            elif modals == 'p':
                textf = poiidf
            else:
                raise NotImplementedError

        lengths = [(amask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(amask))]
        _, community_list = adj_space(vid_area)
        if args.multi_modal and args.mm_fusion_mthd=='gated':
           features, adj, emb, featuret, featurev, featurep = model(textf, amask, lengths, visuf, poiidf, vid_area)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':
            adj, emb, featuret, featurev, featurep  = model(textf, amask, lengths, visuf, poiidf, vid_area)
        MCIlabel = torch.cat([MCIlabel[j][:lengths[j]] for j in range(len(MCIlabel))])
        Hypertensionlabel = torch.cat([Hypertensionlabel[j][:lengths[j]] for j in range(len(Hypertensionlabel))])
        Diabeteslabel = torch.cat([Diabeteslabel[j][:lengths[j]] for j in range(len(Diabeteslabel))])
        MDDlabel =torch.cat([MDDlabel[j][:lengths[j]] for j in range(len(MDDlabel))])
        loss = loss_function(adj, emb, featuret, featurev, featurep) 
        MCIlabels.append(MCIlabel.cpu().numpy())
        Hypertensionlabels.append(Hypertensionlabel.cpu().numpy())
        Diabeteslabels.append(Diabeteslabel.cpu().numpy())
        MDDlabels.append(MDDlabel.cpu().numpy())
        embs.append(emb.detach().cpu().numpy())
        community.append(community_list)

        if train:
            loss.backward()
            optimizer.step()

    embs  = np.concatenate(embs)
    MCIlabels = np.concatenate(MCIlabels)
    Hypertensionlabels = np.concatenate(Hypertensionlabels)
    Diabeteslabels = np.concatenate(Diabeteslabels)
    MDDlabels = np.concatenate(MDDlabels)
    community = np.concatenate(community)


    return loss, embs, MCIlabels, Hypertensionlabels, Diabeteslabels, MDDlabels, community




if __name__ == '__main__':
    path = './saved/ '

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='L', help='base recurrent model, must be one of DialogRNN/L/GRU')

    parser.add_argument('--graph-model', action='store_true', default=False, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/SMGCN')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for SMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=True, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False, help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=256, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='UbranModal', help='dataset to train and test')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    args = parser.parse_args()
    today = datetime.datetime.now()
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset

    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda

    #if args.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('log')

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'text':200, 'image':200,'poi':768}

    D_text= feat2dim['text']
    D_visual = feat2dim['image'] 
    D_poitf = feat2dim['poi'] 

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'tvp':
                D_m = D_text+D_visual+D_poitf
            elif modals == 'tv':
                D_m = D_text+D_visual
            elif modals == 'vp':
                D_m = D_visual+D_poitf
            elif modals == 'tp':
                D_m = D_text+D_poitf
            else:
                raise NotImplementedError
        else:
            D_m = D_text
   
    else:
        if modals == 't':
            D_m = D_text

        elif modals == 'v':
            D_m = D_visual
        elif modals == 'p':
            D_m = D_poitf
            
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100

    graph_h = 100
    n_classes  = 4 if args.Dataset=='UbranModal' else 1

    if args.graph_model:
        seed_everything()
        model = SpatialGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 dropout=args.dropout,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_p = D_poitf,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 Deep_GCN_nlayers=args.Deep_GCN_nlayers,
                                 dataset=args.Dataset,
                                 use_modal=args.use_modal)

        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        if args.base_model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h, 
                              n_classes=n_classes, 
                              dropout=args.dropout)

            print ('Basic LSTM Model.')

        else:
            print ('Base model must be one of DialogRNN/L/GRU/Transformer')
            #raise NotImplementedError

        name = 'Base'

    if cuda:
        model.cuda()
   
    loss_function = SimLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'UbranModal':
        train_loader, valid_loader = get_data_loaders(valid=0, batch_size=batch_size, num_workers=0)                                                          
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    embedding, embs = [],[]

    test_label = False
    if test_label:
        state = torch.load('best_model_UbranModal/model.pth')  
        model.load_state_dict(state['net'])
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)

    for e in range(n_epochs):
        start_time = time.time()
       
        if args.graph_model:
            loss, embs, MCIlabels, Hypertensionlabels, Diabeteslabels, MDDlabels,community_list = train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset)
            mae, rmse, r2 = predict_regression(embs, MCIlabels, display=True)
            mae, rmse, r2 = predict_regression(embs, Hypertensionlabels, display=True)
            mae, rmse, r2 = predict_regression(embs, Diabeteslabels, display=True)
            mae, rmse, r2 = predict_regression(embs, MDDlabels, display=True)
        if (e+1)%1 == 0:
            print('epoch: {}, train_loss: {},  time: {} sec'.\
                format(e+1, loss.item(), round(time.time()-start_time, 2)))
np.save('embeddings_proposed_model.npy',embs)
np.save('MCIlabels.npy',MCIlabels)
np.save('community_list.npy',community_list)

