# HetMS-AMRGNN

Heterogeneous Multi-Scale Graph Neural Network for Antimicrobial Drug Recommendation in Electronic Health Records

## Overview

This repository contains the implementation of HetMS-AMRGNN, a novel graph neural network model for antimicrobial drug recommendation in intensive care units (ICUs). The model effectively addresses:

- Complex combination therapy requirements
- Heterogeneous electronic health records (EHRs) data
- Drug-drug interactions and resistance patterns
- Patient-specific characteristics

## Key Features

- Multi-view feature extraction
- Multi-scale graph convolution
- Metapath-based aggregation
- Patient history encoding
- Drug-drug interaction integration

## Requirements

```
torch>=1.8.0
torch-geometric>=2.0.0
numpy>=1.19.2
pandas>=1.2.0
scikit-learn>=0.24.0
```

## Installation

```bash
git clone https://github.com/username/HetMS-AMRGNN.git
cd HetMS-AMRGNN
pip install -r requirements.txt
```

## Usage

### Data Preparation

```python
python utils/data_utils.py --data_dir /path/to/data
```

### Training

```python
python experiments/train.py --model HetMS-AMRGNN --hidden_dim 256 --num_views 2
```

### Evaluation

```python
python experiments/evaluate.py --model_path /path/to/model --test_data /path/to/test
```

## Model Architecture

The model incorporates multiple components:
- Heterogeneous graph construction
- Multi-view feature extraction
- Multi-scale graph convolution
- Metapath-based aggregation
- Patient history encoding
- Drug-drug interaction integration

## Citation

If you use this code in your research, please cite our paper
