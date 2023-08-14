# Graph Clustering Network with Structure Embedding Enhanced

An official code for the paper [Graph Clustering Network with Structure Embedding Enhanced](https://doi.org/10.1016/j.patcog.2023.109833) based on [A-Unified-Framework-for-Deep-Attribute-Graph-Clustering](https://github.com/Marigoldwu/A-Unified-Framework-for-Deep-Attribute-Graph-Clustering). The codes of comparison methods are also shared at [A-Unified-Framework-for-Deep-Attribute-Graph-Clustering](https://github.com/Marigoldwu/A-Unified-Framework-for-Deep-Attribute-Graph-Clustering). If this project and the unified framework are useful for you, please consider staring them or citing our paper.
## Abstract
Recently, deep clustering utilizing Graph Neural Networks has shown good performance in the graph clustering. However, the structure information of graph was underused in existing deep clustering methods. Particularly, the lack of concern on mining different types structure information simultaneously. To tackle with the problem, this paper proposes a **G**raph **C**lustering Network with **S**tructure **E**mbedding **E**nhanced (GC-SEE) which extracts nodes importance-based and attributes importance-based structure information via a feature attention fusion graph convolution module and a graph attention encoder module respectively. Additionally, it captures different orders-based structure information through multi-scale feature fusion. Finally, a self-supervised learning module has been designed to integrate different types structure information and guide the updates of the GC-SEE. The comprehensive experiments on benchmark datasets commonly used demonstrate the superiority of the GC-SEE. The results showcase the effectiveness of the GC-SEE in exploiting multiple types of structure for deep clustering.

## Quick Start

### Requirements

```
python >= 3.7

matplotlib==3.5.3
munkres==1.1.4
numpy==1.21.5
scikit_learn==1.0.2
torch==1.11.0
```

### Commands

- Train GC-SEE for 10 experiments (:exclamation: Before training, you should download the dataset from \[ [dataset-Google Drive](https://drive.google.com/drive/folders/1TlpGNU9miqJtGYs6hDBJfqlicyZLpJ8F?usp=sharing) \] and the pretrained parameter files from \[ [pretrain-Google Drive](https://drive.google.com/drive/folders/1-WS0Snb7sjCtvn9dG-fClutEmWPjbeub?usp=sharing) \].) Then run the followed command:

```bash
python main.py -M GCSEE -D cora -A npy -T 2 -LS 10 -S 325
```

- Or you want to use other datasets, please add the hyper-parameters in the all three train.py in model folder. Firstly, pretrain ae and gat separately.

```shell
python main.py -P -M pretrain_ae_for_gcsee -D cora
python main.py -P -M pretrain_gat_for_gcsee -D cora -T 2
```

## Citation

We are truly grateful for citing our paper! The BibTex entry of our paper is:

```
@article{ding2023graph,
title = {Graph clustering network with structure embedding enhanced},
journal = {Pattern Recognition},
volume = {144},
pages = {109833},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109833},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323005319},
author = {Shifei Ding and Benyu Wu and Xiao Xu and Lili Guo and Ling Ding},
}
```

