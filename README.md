## CureGraph: Contrastive Multi-Modal Graph Representation Learning for Urban Living Circle Health Profiling and Prediction 


we propose CureGraph, a contrastive multi-modal representation learning framework for urban health prediction that employs graph-based techniques to infer the prevalence of common chronic
diseases among the elderly within the urban living circles of each neighborhood.


## Installation

### Environment
- Tested OS: Linux
- Python == 3.9
- PyTorch == 2.2.2


### Dependencies
1. Install PyTorch 2.2.0 with the correct CUDA version.
2. Use the ``pip install -r requirements.txt`` command to install all of the Python modules and packages used in this project.



**Step-1** Contrastive learning to get multi-modal representation within the urban living circles:
```
cd ./multimodal encoder 

python xx_encoder.py

python xx_embedding.py

```


**Step-2** Train SMGCN model on downstream elderly health prediction:


```
python train.sh
```


## Note

The implementation of multimodal encoder is based on *[SimCLR] (https://github.com/google-research/simclr)* and *[SupCon] (https://github.com/google-research/google-research/tree/master/supcon)*





![OverallFramework](./CureGraph_framework.png "Overall framework")


