# DyMamba
This repository is an official implementation of DyMamba.

*Code will be released soon.*

## Installation
```bash
git clone https://github.com/cbqBit/dymamba.git
cd dymamba

conda create -n dymamba python==3.10
conda activate dymamba

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 pytorch-cuda=11.8 -c pytorch

pip install causal-conv1d==1.1.1
pip install mamba-ssm

pip install -r requirements.txt
```


## Datasets
- Public datasets
    - NeurIPS-CellSeg https://neurips22-cellseg.grand-challenge.org/
    - Cellpose https://www.cellpose.org/dataset
    - Lucchi https://www.epfl.ch/labs/cvlab/data/data-em/
    - LiveCell https://sartorius-research.github.io/LIVECell/
    - MitoEM-R https://mitoem.grand-challenge.org/

- The datasets directories under the root should the following structure:
```
  Root
  ├── Datasets
  │   ├── images 
  │   │    ├── train_00001.png
  │   │    ├── train_00002.png
  │   │    ├── train_00003.png
  │   │    ├── ...  
  │   └── labels (labels must have .tiff extension.)
  │   │    ├── train_00001_label.png 
  │   │    ├── train_00002.label.png
  │   │    ├── train_00003.label.png
  │   │    ├── ...
  └── ...
```
