# HetMS-AMRGNN: Heterogeneous Multi-Scale Graph Neural Network for Antimicrobial Drug Recommendation

## 📋 Overview

HetMS-AMRGNN (Heterogeneous Multi-Scale Antimicrobial Recommendation Graph Neural Network) is a novel deep learning framework designed for antimicrobial drug recommendation in Intensive Care Units (ICUs). Our model effectively addresses the complexities of combination therapy and heterogeneous Electronic Health Records (EHRs) data through an advanced graph representation learning approach.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/leanqon/HetMS-AMRGNN.git
cd HetMS-AMRGNN

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

The model has been tested on MIMIC-IV and eICU datasets. Due to data use agreements, we cannot directly provide the datasets. You can obtain access by:

1. Completing the required training and applying for access at [PhysioNet](https://physionet.org/)
2. Follow our preprocessing steps

## 🚀 Usage

### Training

```bash
python hetms.py --mode train \
                --model HetMSAMRGNN \
                --dataset eicu \
                --hidden_dim 256 \
                --num_views 3 \
                --num_scales 3 \
                --num_heads 4 \
                --num_layers 2 \
                --lr 0.0001 \
                --dropout 0.2 \
                --weight_decay 1e-4 \
                --num_epochs 300 \
                --batch_size 256 \
                --output_dir ./output
```

### Evaluation

```bash
python hetms.py --mode evaluate \
                --model HetMSAMRGNN \
                --dataset mimic \
                --model_path ./output \
                --output_dir ./output
```

### Case Study

```bash
python hetms.py --mode case_study \
                --model HetMSAMRGNN \
                --model_path ./output \
                --case_study_patient_id \
                --explanation_output_dir ./output/explanations
```
## 📚 Directory Structure

```
HetMS-AMRGNN/
├── base_models.py         # Core model architecture components
├── case_study.py          # Functions for case study analysis
├── explainer.py           # Explainability module
├── hetms.py               # Main training and evaluation script
├── integration.py         # Integration utilities
├── data/                  # Directory for datasets (not included)
├── output/                # Directory for output files
└── requirements.txt       # Required Python packages
```

## 📚 Citation

## 🤝 Contributing

We welcome contributions to improve HetMS-AMRGNN! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Contact

For any questions or feedback, please contact:
- Zhengqiu Yu - [email]
- Xiangrong Liu - xrliu@xmu.edu.cn
