# Not Just Oversmoothing: Detecting Echo Chamber Effect in Graph Neural Networks

This repository contains the implementation of **Community-Aware Split Propagation (CASP)**, a lightweight model-agnostic plugin that decouples intra- and inter-community aggregation in Graph Neural Networks to address the echo chamber effect.

## Requirements

```
Python          3.11.5
torch           2.3.1
torchvision     0.18.1
torchaudio      2.3.1
torch-geometric 2.7.0
torch-cluster   1.6.3
torch-scatter   2.0.9
torch-sparse    0.6.18
```

## Running Experiments

For binary heterophilic benchmarks (**Minesweeper**, **Tolokers**, **Questions**), which are evaluated using ROC-AUC:

```bash
python train_auc.py --dataset minesweeper
python train_auc.py --dataset tolokers
python train_auc.py --dataset questions
```

For all other datasets, which are evaluated using accuracy:

```bash
python train.py --dataset pubmed
python train.py --dataset citeseer
python train.py --dataset cora_full
```
