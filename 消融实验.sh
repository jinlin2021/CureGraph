#没有空间自相关


nohup python train.py --base-model 'L' --graph-model --nodal-attention --dropout 0.3 --lr 0.0005 --batch-size 32 --l2 0.003 --graph_type='SMGCN' --epochs=30 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_subsequently' --modals='tvp' --Dataset='UbranModal' --Deep_GCN_nlayers 3 >新实验2.log 2>&1