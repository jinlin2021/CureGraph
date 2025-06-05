## CureGraph: Contrastive Multi-Modal Graph Representation Learning for Urban Living Circle Health Profiling and Prediction 


we propose CureGraph, a contrastive multi-modal representation learning framework for urban health prediction that employs graph-based techniques to infer the prevalence of common chronic
diseases among the elderly within the urban living circles of each neighborhood.


## Installation

### Environment
- Tested OS: Linux
- Python == 3.9
- PyTorch == 2.2.2



**Step-1** Contrastive learning to get multi-modal representation within the urban living circles:
```
cd ./multimodal encoder 

python xx_encoder.py

python get_embedding_xx.py

```


**Step-2** Train SMGCN model, and conduct downstream elderly health prediction:


```
bash train.sh
```


## Overall Framework 
![OverallFramework](./CureGraph_framework.png "Overall framework")


## Note

The implementation of multimodal encoder is based on *[SimCLR] (https://github.com/google-research/simclr)* and *[SupCon] (https://github.com/google-research/google-research/tree/master/supcon)*

## Citation

Please consider citing our paper in your publications, if our findings help your research.
```bibtex
@article{li2025curegraph,
  title={CureGraph: Contrastive multi-modal graph representation learning for urban living circle health profiling and prediction},
  author={Li, Jinlin and Zhou, Xiao},
  journal={Artificial Intelligence},
  volume={340},
  pages={104278},
  year={2025},
  publisher={Elsevier}
}
```


