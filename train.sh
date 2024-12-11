python train.py --base-model 'L' --graph-model --nodal-attention --dropout 0.3 --lr 0.001 --batch-size 32 --l2 0.0003 --graph_type='SMGCN' --epochs=60 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_subsequently' --modals='tvp' --Dataset='UbranModal' --Deep_GCN_nlayers 3 

