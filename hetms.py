import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import GCNConv
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ndcg_score
from sklearn.metrics import jaccard_score, roc_auc_score, average_precision_score
from tqdm import tqdm
import copy
from collections import defaultdict
from itertools import product
import logging
logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
from het10_baselines import * 
from case_study import *
import gzip
from base_models import HetMSAMRGNN, MultiViewFeatureExtraction, BiChannelExtraction, MetapathAggregation, FusionModule, HistoryEncoder
from ablation_models import HetUMSGNN_NoMultiView, HetUMSGNN_NoMultiScale, HetUMSGNN_NoFrequency, HetUMSGNN_NoMetapath, HetUMSGNN_NoDDI, HetUMSGNN_NoHistory
from torch.utils.data import Subset, DataLoader
import math
from integration import *

def run_ablation_study(args, dataset, metapaths, device):
    all_models = {
        "Full": HetUMSGNN,
        "NoMultiView": HetUMSGNN_NoMultiView,
        "NoMultiScale": HetUMSGNN_NoMultiScale,
        "NoFrequency": HetUMSGNN_NoFrequency,
        "NoMetapath": HetUMSGNN_NoMetapath,
        "NoDDI": HetUMSGNN_NoDDI,
        "NoHistory": HetUMSGNN_NoHistory
    }

    models_to_run = ["Full"] 
    if args.ablation:
        models_to_run.extend(args.ablation)

    results = {}

    for name in models_to_run:
        print(f"\nRunning ablation study for: {name}")
        
        model_class = all_models[name]
        model = model_class(
            node_types=dataset.node_types,
            in_dims=dataset.node_type_dims,
            hidden_dim=args.hidden_dim,
            num_drugs=dataset.node_type_count['c'],
            num_views=args.num_views,
            num_scales=args.num_scales,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            metapaths=metapaths,
            node_id_ranges=dataset.get_node_id_ranges(),
            dropout_rate=args.dropout,
            use_history=args.use_history,
            num_diagnoses=dataset.get_num_diagnoses()
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,           
            T_mult=2,         
            eta_min=1e-6      
        )
        criterion = DrugRecommendationLoss(alpha=1.0, beta=0.1)

        train_mask, val_mask, test_mask = dataset.split_data()

        best_model, _ = train(model, dataset, train_mask, val_mask, optimizer, 
                              scheduler, criterion, num_epochs=args.num_epochs, device=device, args=args)
        
        test_metrics = evaluate(best_model, dataset, test_mask, device, args)
        results[name] = test_metrics

        print(f"Results for {name}:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    return results

def get_ablation_model(args, dataset, metapaths, ablation_type): 
    base_params = {
        'node_types': dataset.node_types,
        'in_dims': dataset.node_type_dims,
        'hidden_dim': args.hidden_dim,
        'num_drugs': dataset.node_type_count['c'],
        'num_views': args.num_views,
        'num_scales': args.num_scales,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'metapaths': metapaths,
        'node_id_ranges': dataset.get_node_id_ranges(),
        'dropout_rate': args.dropout,
        'use_history': args.use_history,
        'num_diagnoses': dataset.get_num_diagnoses()
    }

    if ablation_type == 'NoMultiView':
        return HetUMSGNN_NoMultiView(**base_params)
    elif ablation_type == 'NoMultiScale':
        return HetUMSGNN_NoMultiScale(**base_params)
    elif ablation_type == 'NoFrequency':
        return HetUMSGNN_NoFrequency(**base_params)
    elif ablation_type == 'NoMetapath':
        return HetUMSGNN_NoMetapath(**base_params)
    elif ablation_type == 'NoDDI':
        return HetUMSGNN_NoDDI(**base_params)
    elif ablation_type == 'NoHistory':
        return HetUMSGNN_NoHistory(**base_params)
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

def load_icd_mapping(file_path):
    with gzip.open(file_path, 'rt') as f:
        df = pd.read_csv(f)
    return dict(zip(df['icd_code'].astype(str), df['long_title']))

def create_weighted_sampler(labels):
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    binary_labels = (labels > 0.5).long()
    positive_counts = binary_labels.sum(dim=1)
    sample_weights = 1.0 / (positive_counts + 1) 
    num_samples = len(sample_weights)
    return WeightedRandomSampler(sample_weights, num_samples, replacement=True)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_dataset_stats(dataset):
    stats = {
        '# patients': dataset.node_type_count['p'],
        '# disease': dataset.node_type_count['d'],
        '# drug': dataset.node_type_count['c'],
        '# of patient-disease links': sum(dataset.edge_index['p_d'][0] < dataset.node_type_count['p']),
        '# of patient-drug links': sum(dataset.edge_index['p_c'][0] < dataset.node_type_count['p']),
        'avg. # of diseases per patient': sum(dataset.edge_index['p_d'][0] < dataset.node_type_count['p']) / dataset.node_type_count['p'],
        'avg. # of drugs per patient': sum(dataset.edge_index['p_c'][0] < dataset.node_type_count['p']) / dataset.node_type_count['p'],
    }
    return stats

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    BCE_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    focal_loss = alpha * (1-pt)**gamma * BCE_loss
    return focal_loss.mean()

class DrugRecommendationLoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super(DrugRecommendationLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        predictions = torch.clamp(predictions, -10, 10) 
        pos_weight = torch.tensor([self.pos_weight], device=predictions.device)
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, pos_weight=pos_weight, reduction='none'
        )

        loss = torch.clamp(loss, 0, 50) 
        
        return loss.mean()

with open('./data/patient_history_sequences.pkl', 'rb') as f:
    patient_history = pickle.load(f)

with open('./data/diagnosis_index_map.pkl', 'rb') as f:
    diagnosis_index_map = pickle.load(f)

class DrugRecommendationDataset:
    def __init__(self, data, patient_history=None):
        self.data = data
        self.node_types = ['p', 'd', 'c', 's']
        self.edge_types = list(data['edge_index'].keys())

        if patient_history is None:
            if 'patient_history' in globals():
                self.patient_history = patient_history
            else:
                self.patient_history = {}
                for (node_type, node_id), idx in data['unified_map'].items():
                    if node_type == 'p':
                        self.patient_history[node_id] = []
        else:
            self.patient_history = patient_history
        
        self.subject_ids = [key[1] for key in data['unified_map'].keys() if key[0] == 'p']
        self.index_to_subject_id = {idx: subject_id for idx, subject_id in enumerate(self.subject_ids)}
        self.process_data()
        self.compute_statistics()

        if 'm' in self.node_features and 't' in self.node_features:
            self.preprocess_specimen_nodes()

        self.ddi_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/ddi_matrix.pt')
        self.load_or_create_ddi_matrix()
        self.create_ddi_edge_index()
        self.compute_statistics()
        self.create_type_specific_adj_matrices()       
        self.create_global_to_local_mapping() 
    
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        features = {k: v[subject_id] for k, v in self.data['node_features'].items()}
        history = self.patient_history.get(subject_id, [])
        labels = self.data['labels'][subject_id]
        return features, history, labels
    
    def get_subject_id(self, index):
        return self.index_to_subject_id[index]
    
    def get_num_diagnoses(self):
        diagnosis_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/diagnosis_index_map.pkl')
        try:
            with open(diagnosis_index_path, 'rb') as f:
                diagnosis_index_map = pickle.load(f)
            return len(diagnosis_index_map)
        except FileNotFoundError:
            return len(self.node_features['d'])
    
    def create_global_to_local_mapping(self):
        """Create a complete mapping from global to local indices for all node types"""
        self.global_to_local = {}
        node_offsets = {}
        offset = 0
        for nt in self.node_types:
            if nt in self.node_type_count:
                node_offsets[nt] = offset
                offset += self.node_type_count[nt]
        for nt in self.node_types:
            if nt in self.node_type_count:
                self.global_to_local[nt] = {}
                for i in range(self.node_type_count[nt]):
                    global_idx = i + node_offsets[nt]
                    self.global_to_local[nt][global_idx] = i
                for edge_type, indices in self.edge_index.items():
                    if '_' in edge_type:
                        src_type, dst_type = edge_type.split('_')
                        if src_type == nt:
                            for idx in indices[0].tolist():
                                if idx not in self.global_to_local[nt]:
                                    self.global_to_local[nt][idx] = idx - node_offsets[nt]
                        if dst_type == nt:
                            for idx in indices[1].tolist():
                                if idx not in self.global_to_local[nt]:
                                    self.global_to_local[nt][idx] = idx - node_offsets[nt]

        for nt in self.global_to_local:
            for global_idx, local_idx in list(self.global_to_local[nt].items()):
                if local_idx < 0 or local_idx >= self.node_type_count[nt]:
                    self.global_to_local[nt][global_idx] = min(max(0, local_idx), self.node_type_count[nt] - 1)

        for nt in self.node_types:
            if nt in self.global_to_local:
                indices = list(self.global_to_local[nt].keys())
                if indices:
                    print(f"Global index range for {nt}: {min(indices)} to {max(indices)}")

    def create_ddi_edge_index(self):
        ddi_edges = torch.nonzero(self.ddi_matrix, as_tuple=False)
        self.ddi_edge_index = ddi_edges.t().contiguous()
        print(f"Created DDI edge index with shape {self.ddi_edge_index.shape}")

    def get_ddi_info(self):
        return self.ddi_matrix, self.ddi_edge_index

    def create_ca_ca_edges(self):
        ca_ca_edges = torch.nonzero(self.ddi_matrix, as_tuple=False)
        self.edge_index['c-c'] = ca_ca_edges.t().contiguous()
        print(f"Number of ca-ca edges: {ca_ca_edges.shape[0]}")

    def load_or_create_ddi_matrix_old(self):
        if os.path.exists(self.ddi_matrix_path):
            self.ddi_matrix = torch.load(self.ddi_matrix_path)
            print("Loaded DDI matrix from file.")
        else:
            print("DDI matrix file not found. Creating new DDI matrix.")
            os.makedirs(os.path.dirname(self.ddi_matrix_path), exist_ok=True)
            
            try:
                self.load_drug_info()
                self.create_ddi_matrix()
                torch.save(self.ddi_matrix, self.ddi_matrix_path)
                print("Created and saved DDI matrix.")
            except Exception as e:
                print(f"Error creating DDI matrix: {e}")
                num_drugs = self.node_type_count['c']
                self.ddi_matrix = torch.zeros((num_drugs, num_drugs))
                print(f"Created default zero DDI matrix of size {num_drugs}x{num_drugs}")

    def load_or_create_ddi_matrix(self):
        if os.path.exists(self.ddi_matrix_path):
            self.ddi_matrix = torch.load(self.ddi_matrix_path)
            print("Loaded DDI matrix from file.")
        else:
            print("DDI matrix file not found. Creating new DDI matrix.")
            os.makedirs(os.path.dirname(self.ddi_matrix_path), exist_ok=True)
            
            try:
                if 'c' not in self.node_type_count:
                    print("No clinical antimicrobials found in data. Creating empty DDI matrix.")
                    self.ddi_matrix = torch.zeros((0, 0))
                    return
                    
                self.load_drug_info()
                self.create_ddi_matrix()
                torch.save(self.ddi_matrix, self.ddi_matrix_path)
                print("Created and saved DDI matrix.")
            except Exception as e:
                print(f"Error creating DDI matrix: {e}")
                num_drugs = self.node_type_count.get('c', 0)
                self.ddi_matrix = torch.zeros((num_drugs, num_drugs))
                print(f"Created default zero DDI matrix of size {num_drugs}x{num_drugs}")

    def load_drug_info(self):
        drug_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drugbank/drug.csv')
        
        try:
            drug_df = pd.read_csv(drug_csv_path)
            self.drug_to_db = {}
            for _, row in drug_df.iterrows():
                drug_name = row['drug']
                db_ids = [db for db in str(row['db']).split() if db.startswith('DB')]
                for db_id in db_ids:
                    self.drug_to_db[drug_name] = db_id

            self.db_to_index = {db: idx for idx, (drug, db) in enumerate(self.drug_to_db.items())}
        except FileNotFoundError:
            print(f"Drug info file not found: {drug_csv_path}")
            self.drug_to_db = {}
            self.db_to_index = {}

    def create_ddi_matrix(self):
        num_drugs = self.node_type_count['c']
        self.ddi_matrix = np.zeros((num_drugs, num_drugs))
        ddi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drugbank/multi_ddi_name.txt')
        try:
            with open(ddi_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        db1 = parts[0].split('::')[1]  
                        db2 = parts[1].split('::')[1] 
                        if db1 in self.db_to_index and db2 in self.db_to_index:
                            idx1, idx2 = self.db_to_index[db1], self.db_to_index[db2]
                            self.ddi_matrix[idx1, idx2] = self.ddi_matrix[idx2, idx1] = 1
        except FileNotFoundError:
            print(f"DDI file not found: {ddi_path}")

        drug_features = self.node_features['c'].numpy()
        for i in range(num_drugs):
            if i < drug_features.shape[0] and drug_features.shape[1] > 1:
                if drug_features[i, 0] == 1: 
                    involved_drugs = np.where(drug_features[i, 1:] == 1)[0]
                    for j in range(num_drugs):
                        if i != j:
                            if j < drug_features.shape[0]:
                                if drug_features[j, 0] == 0:  
                                    if j in involved_drugs:
                                        self.ddi_matrix[i, j] = self.ddi_matrix[j, i] = 1
                                else: 
                                    if j < drug_features.shape[0]:
                                        other_drugs = np.where(drug_features[j, 1:] == 1)[0]
                                        if np.any(np.isin(involved_drugs, other_drugs)):
                                            self.ddi_matrix[i, j] = self.ddi_matrix[j, i] = 1

        self.ddi_matrix = torch.FloatTensor(self.ddi_matrix)

    def get_node_id_ranges(self):
        ranges = {}
        start = 0
        for ntype in self.node_types:
            count = self.node_type_count.get(ntype, 0)
            ranges[ntype] = (start, start + count)
            start += count
        return ranges

    def preprocess_specimen_nodes(self):
        print("Preprocessing specimen nodes...")
        try:
            s_features = self.node_features['s']  # Specimen features
            m_features = self.node_features['m']  # Microorganism features
            t_features = self.node_features['t']  # Test antimicrobial features
            
            s_m_adj = self.edge_index.get('s_m', torch.zeros((2, 0), dtype=torch.long))
            m_t_adj = self.edge_index.get('m_t', torch.zeros((2, 0), dtype=torch.long))
            m_t_features = self.edge_features.get('m_t', {})

            s_global_to_local = {}
            m_global_to_local = {}
            t_global_to_local = {}
            
            for (node_type, node_id), idx in self.data['unified_map'].items():
                if node_type == 's':
                    s_idx = idx - (self.node_type_count.get('p', 0) + self.node_type_count.get('d', 0) + self.node_type_count.get('c', 0))
                    if s_idx >= 0 and s_idx < len(s_features):
                        s_global_to_local[idx] = s_idx
                elif node_type == 'm':
                    m_idx = idx - (self.node_type_count.get('p', 0) + self.node_type_count.get('d', 0) + 
                                  self.node_type_count.get('c', 0) + self.node_type_count.get('s', 0))
                    if m_idx >= 0 and m_idx < len(m_features):
                        m_global_to_local[idx] = m_idx
                elif node_type == 't':
                    t_idx = idx - (self.node_type_count.get('p', 0) + self.node_type_count.get('d', 0) + 
                                  self.node_type_count.get('c', 0) + self.node_type_count.get('s', 0) + 
                                  self.node_type_count.get('m', 0))
                    if t_idx >= 0 and t_idx < len(t_features):
                        t_global_to_local[idx] = t_idx

            if len(s_m_adj) > 0 and s_m_adj.shape[1] > 0:
                s_m_adj_sparse = torch.sparse_coo_tensor(s_m_adj, torch.ones(s_m_adj.shape[1]), dtype=torch.float32)
            else:
                s_m_adj_sparse = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), 
                                                         torch.tensor([], dtype=torch.float32),
                                                         dtype=torch.float32)

            if len(m_t_adj) > 0 and m_t_adj.shape[1] > 0:
                m_t_adj_sparse = torch.sparse_coo_tensor(m_t_adj, torch.ones(m_t_adj.shape[1]), dtype=torch.float32)
            else:
                m_t_adj_sparse = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), 
                                                         torch.tensor([], dtype=torch.float32),
                                                         dtype=torch.float32)

            s_features_new = s_features.clone()
            for i, j in zip(s_m_adj[0], s_m_adj[1]):
                i_item, j_item = i.item(), j.item()
                if i_item in s_global_to_local and j_item in m_global_to_local:
                    s_local = s_global_to_local[i_item]
                    m_local = m_global_to_local[j_item]
                    if s_local < len(s_features_new) and m_local < len(m_features):
                        s_features_new[s_local] += m_features[m_local]

            for i, j in zip(m_t_adj[0], m_t_adj[1]):
                i_item, j_item = i.item(), j.item()
                if i_item in m_global_to_local and j_item in t_global_to_local:
                    m_local = m_global_to_local[i_item]
                    t_local = t_global_to_local[j_item]
                    m_t_feature = torch.tensor([0.0]) 
                    if isinstance(m_t_features, dict):
                        feature_key = f"{i_item},{j_item}"
                        if feature_key in m_t_features:
                            m_t_feature = torch.tensor([float(m_t_features[feature_key])])
                    elif isinstance(m_t_features, torch.Tensor) and j_item < len(m_t_features):
                        m_t_feature = m_t_features[j_item].unsqueeze(0)
                    if t_local < len(t_features):
                        combined_feature = torch.cat([t_features[t_local], m_t_feature], dim=0)
                        for s_global, s_local in s_global_to_local.items():
                            if s_m_adj_sparse.indices().shape[1] > 0:
                                connected = False
                                for idx in range(s_m_adj_sparse.indices().shape[1]):
                                    if s_m_adj_sparse.indices()[0, idx] == s_global and s_m_adj_sparse.indices()[1, idx] == i_item:
                                        connected = True
                                        break
                                
                                if connected and s_local < len(s_features_new):
                                    if combined_feature.shape[0] <= s_features_new.shape[1]:
                                        padding = torch.zeros(s_features_new.shape[1] - combined_feature.shape[0])
                                        padded_feature = torch.cat([combined_feature, padding])
                                        s_features_new[s_local] += padded_feature
                                    else:
                                        s_features_new[s_local] += combined_feature[:s_features_new.shape[1]]

            self.node_features['s'] = s_features_new

            if 'm' in self.node_features:
                del self.node_features['m']
            if 't' in self.node_features:
                del self.node_features['t']

            for edge_type in ['s_m', 'm_s', 'm_t', 't_m']:
                if edge_type in self.edge_index:
                    del self.edge_index[edge_type]

            if 'm_t' in self.edge_features:
                del self.edge_features['m_t']

            print(f"Updated specimen feature shape: {self.node_features['s'].shape}")
        except Exception as e:
            print(f"Error during specimen node preprocessing: {e}")
            print("Continuing with original specimen features.")

    def process_data(self):
        self.node_features = {}
        for k, v in self.data['node_features'].items():
            if isinstance(v, np.ndarray):
                self.node_features[k] = torch.tensor(v, dtype=torch.float)
            else:
                self.node_features[k] = v

        self.edge_index = {}
        for k, v in self.data['edge_index'].items():
            if isinstance(v, np.ndarray):
                self.edge_index[k] = torch.tensor(v, dtype=torch.long)
            else:
                self.edge_index[k] = v
        
        # Process edge features
        self.edge_features = {}
        if 'edge_features' in self.data:
            for k, v in self.data['edge_features'].items():
                if k == 'c_p' and isinstance(v, dict):
                    values_list = list(v.values())
                    if values_list:
                        self.edge_features[k] = torch.tensor(values_list, dtype=torch.float)
                    else:
                        self.edge_features[k] = torch.tensor([], dtype=torch.float)
                elif k == 'm_t' and isinstance(v, dict):
                    self.edge_features[k] = v
                else:
                    if isinstance(v, list) or isinstance(v, np.ndarray):
                        self.edge_features[k] = torch.tensor(v, dtype=torch.float)
                    else:
                        self.edge_features[k] = v
        
        self.unified_map = self.data['unified_map']
        self.drug_mapping = self.data.get('ordered_drug_mapping', {})
        self.labels = self.data.get('labels', {})

    def compute_statistics(self):
        self.node_type_count = {t: self.node_features[t].shape[0] for t in self.node_types if t in self.node_features}
        self.node_type_dims = {t: self.node_features[t].shape[1] for t in self.node_types if t in self.node_features}
        
        print("Node type counts:")
        for t, count in self.node_type_count.items():
            print(f"  {t}: {count} nodes")
        
        print("Edge counts:")
        for edge_type, indices in self.edge_index.items():
            if isinstance(indices, torch.Tensor):
                print(f"  {edge_type}: {indices.shape[1]} edges")
            else:
                print(f"  {edge_type}: {len(indices[0])} edges")

    def create_adj_matrices(self):
        self.adj_matrices = {}
        for edge_type, indices in self.edge_index.items():
            if isinstance(indices, torch.Tensor) and indices.shape[1] > 0:
                src, dst = indices
                num_nodes = max(src.max(), dst.max()) + 1
                adj = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (num_nodes, num_nodes))
                self.adj_matrices[edge_type] = adj
            else:
                print(f"Skipping empty edge type: {edge_type}")

    def create_type_specific_adj_matrices(self):
        self.type_specific_adj = {}
        for edge_type, indices in self.edge_index.items():
            src_type, dst_type = edge_type.split('_')
            src, dst = indices
            num_src = self.node_type_count[src_type]
            num_dst = self.node_type_count[dst_type]
            adj_matrix = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (num_src, num_dst)).coalesce()
            if edge_type in self.edge_features:
                edge_dim = self.edge_features[edge_type].shape[1] if self.edge_features[edge_type].dim() > 1 else 1
            else:
                edge_dim = 0
            
            self.type_specific_adj[f"{src_type}-{dst_type}"] = (adj_matrix, edge_dim)

    def get_edge_features(self, edge_type):
        return self.edge_features.get(edge_type, None)

    def get_type_specific_adj(self):
        return self.type_specific_adj

    def get_features(self, device):
        return {t: self.node_features[t].to(device) for t in self.node_types if t in self.node_features}

    def split_data(self, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42):
        num_patients = self.node_type_count['p']

        np.random.seed(random_seed)
        indices = np.random.permutation(num_patients)
        
        train_size = int(num_patients * train_ratio)
        val_size = int(num_patients * val_ratio)
        
        self.train_mask = indices[:train_size]
        self.val_mask = indices[train_size:train_size+val_size]
        self.test_mask = indices[train_size+val_size:]

        print(f"Split data into {len(self.train_mask)} train, {len(self.val_mask)} validation, {len(self.test_mask)} test samples")
        return self.train_mask, self.val_mask, self.test_mask
    
    def preprocess_and_cache_data(self, device):
        self.processed_features = {k: v.to(device) for k, v in self.node_features.items()}
        self.processed_adj_matrices = {k: v[0].to(device) for k, v in self.type_specific_adj.items()}
        self.processed_edge_index = {k: v.to(device) for k, v in self.edge_index.items()}
        
        print(f"Cached data to device: {device}")

    def get_processed_data(self):
        return self.processed_features, self.processed_adj_matrices, self.processed_edge_index
    
    def get_node_type_offset(self, node_type):
        offset = 0
        for t in self.node_types:
            if t == node_type:
                return offset
            if t in self.node_type_count:
                offset += self.node_type_count[t]
        raise ValueError(f"Invalid node type: {node_type}")
    
    @staticmethod
    def load_from_file(data_path, history_path=None):
        """Load dataset from files"""
        print(f"Loading data from: {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if history_path and os.path.exists(history_path):
            print(f"Loading patient history from: {history_path}")
            with open(history_path, 'rb') as f:
                patient_history = pickle.load(f)
        else:
            patient_history = None
        
        return DrugRecommendationDataset(data, patient_history)

def preprocess_data(data):
    self.node_features = {}
    for k, v in self.data['node_features'].items():
        if isinstance(v, np.ndarray):
            self.node_features[k] = torch.tensor(v, dtype=torch.float)
        else:
            self.node_features[k] = v

    self.edge_index = {}
    for k, v in self.data['edge_index'].items():
        if isinstance(v, np.ndarray):
            self.edge_index[k] = torch.tensor(v, dtype=torch.long)
        else:
            self.edge_index[k] = v

    self.edge_features = {}
    if 'edge_features' in self.data:
        for k, v in self.data['edge_features'].items():
            if k not in self.edge_index:
                continue 
                
            if isinstance(v, dict):
                if not v: 
                    continue

                if all(isinstance(val, (int, float)) for val in v.values()):
                    values_list = list(v.values())
                    self.edge_features[k] = torch.tensor(values_list, dtype=torch.float).reshape(-1, 1)
                else:
                    self.edge_features[k] = v
            elif isinstance(v, list):
                self.edge_features[k] = torch.tensor(v, dtype=torch.float)
            elif isinstance(v, np.ndarray):
                self.edge_features[k] = torch.tensor(v, dtype=torch.float)
            else:
                self.edge_features[k] = v

    unified_map = data['unified_map']
    ordered_drug_mapping = data.get('ordered_drug_mapping', None)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'unified_map': unified_map,
        'ordered_drug_mapping': ordered_drug_mapping
    }

def create_adj_matrices(edge_index, num_nodes):
    adj_matrices = {}
    for edge_type, edges in edge_index.items():
        src_type, dst_type = edge_type.split('-')
        adj = sparse.coo_matrix(
            (np.ones(edges.shape[1]), (edges[0], edges[1])),
            shape=(num_nodes[src_type], num_nodes[dst_type])
        )
        adj_matrices[edge_type] = adj.tocsr()
    return adj_matrices

def generate_metapaths(node_types, edge_types, hop_lengths):
    metapaths = {ntype: [] for ntype in node_types}
    
    def dfs(start_type, current_type, path, target_length):
        if len(path) == target_length and path[-1] == start_type:
            metapaths[start_type].append(path)
            return
        
        if len(path) >= target_length:
            return
        
        for edge in edge_types:
            if edge[0] == current_type:
                new_path = path + [edge[1]]
                dfs(start_type, edge[1], new_path, target_length)
    
    for ntype in node_types:
        target_length = hop_lengths[ntype] + 1  # +1 因为路径长度包括起始节点
        for start_type in node_types:
            dfs(ntype, start_type, [start_type], target_length)
    
    return metapaths

def compute_ndcg(predictions, labels, k=10):
    # 确保预测和标签是 numpy 数组
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 对每个样本计算 NDCG@k
    ndcgs = []
    for pred, label in zip(predictions, labels):
        # 获取前k个预测的索引
        top_k_indices = np.argsort(pred)[::-1][:k]
        
        # 计算 DCG@k
        dcg = np.sum((2**label[top_k_indices] - 1) / np.log2(np.arange(2, k+2)))
        
        # 计算 IDCG@k (理想 DCG)
        ideal_ranking = np.argsort(label)[::-1]
        idcg = np.sum((2**label[ideal_ranking[:k]] - 1) / np.log2(np.arange(2, k+2)))
        
        # 计算 NDCG@k
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs)

def compute_recall(predictions, labels, k=10):
    # 确保预测和标签是 numpy 数组
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 对每个样本计算 Recall@k
    recalls = []
    for pred, label in zip(predictions, labels):
        # 获取前k个预测的索引
        top_k_indices = np.argsort(pred)[::-1][:k]
        # 计算 Recall@k
        recall = np.sum(label[top_k_indices]) / np.sum(label)
        recalls.append(recall)
    
    return np.mean(recalls)

def split_data(data, test_size=0.4, val_size=0.1):
    num_patients = data['node_features']['p'].shape[0]
    all_indices = np.arange(num_patients)
    train_val_indices, test_indices = train_test_split(all_indices, test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size/(1-test_size), random_state=42)
    
    return train_indices, val_indices, test_indices

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def save_results(result, filename):
    with open(filename, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=4)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
   
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def get_type_specific_adj(self):
    print("Type specific adj contents:")
    for key, value in self.type_specific_adj.items():
        print(f"{key}: {type(value)}, {len(value) if isinstance(value, tuple) else 'not tuple'}")
    return self.type_specific_adj

def prepare_histories(batch_histories, device, max_len=100, num_diagnoses=None):
    """
    Process patient histories for the model.
    Compatible with both MIMIC (nested lists) and eICU (flat lists) formats.
    """
    if not batch_histories:
        return None
    
    if num_diagnoses is None:
        num_diagnoses = 100
    
    flat_histories = []
    for history in batch_histories:
        if history and not isinstance(history[0], list) and isinstance(history[0], int):
            flat = history
        else:
            try:
                flat = [diag for visit in history for diag in visit]
            except TypeError:
                flat = history

        flat = [min(diag, num_diagnoses-1) for diag in flat if isinstance(diag, int)]

        if len(flat) > max_len:
            flat = flat[-max_len:] 
        else:
            flat = [0] * (max_len - len(flat)) + flat 
        
        flat_histories.append(flat)
    
    return torch.LongTensor(flat_histories).to(device)

def create_patient_subgraph(adj_matrices):
    p_c = adj_matrices['p-c']
    c_p = adj_matrices['c-p']
    p_p = torch.mm(p_c, c_p)
    p_p.fill_diagonal_(0)
    
    return p_p

def train(model, dataset, train_mask, val_mask, optimizer, scheduler, criterion, num_epochs, device, args):
    """Train the model and return the best model"""
    # === Initialize data ===
    model.train()
    features = dataset.get_features(device)
    adj_dict = dataset.get_type_specific_adj()
    node_id_ranges = dataset.get_node_id_ranges()
    model_filename = get_model_filename(args)
    
    # Check and handle NaN in input features
    for ntype, feat in features.items():
        if torch.isnan(feat).any():
            print(f"Warning: NaN detected in input features for node type {ntype}")
            features[ntype] = torch.nan_to_num(feat, nan=0.0)
    
    # Move adjacency matrices to device
    adj_matrices = {}
    for key, value in adj_dict.items():
        if isinstance(value, tuple):
            adj, edge_dim = value
            adj_matrices[key] = adj.to(device)
        else:
            adj_matrices[key] = value.to(device)
    adj_dict = adj_matrices
    
    # Get drug-drug interaction information
    _, ddi_edge_index = dataset.get_ddi_info()
    ddi_edge_index = ddi_edge_index.to(device)
    
    # === Prepare training data ===
    train_indices = torch.tensor(list(train_mask))
    train_labels = torch.zeros(len(train_mask), dataset.node_type_count['c'], dtype=torch.float) 
    for i, patient_idx in enumerate(train_mask):
        positive_drugs = dataset.edge_index['p_c'][1][dataset.edge_index['p_c'][0] == patient_idx]
        positive_drugs_local = positive_drugs - node_id_ranges['c'][0]
        train_labels[i, positive_drugs_local] = 1.0 

    train_dataset = TensorDataset(train_indices, train_labels)
    sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    # === Initialize training parameters ===
    patience = args.patience * 2
    patience_counter = 0
    best_model = None
    best_performance = 0
    
    # Learning rate warmup setup
    warmup_epochs = 5
    
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs * 0.1 
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    
    # Contrastive learning weight setup
    contrast_epochs = int(num_epochs * 0.5)
    contrast_weight_init = 0.5
    contrast_weight_final = 0.1

    # === Main training loop ===
    for epoch in range(num_epochs):
        # Set to training mode
        model.train()
        total_loss = 0
        
        # Calculate current contrastive weight
        if epoch < contrast_epochs:
            progress = epoch / contrast_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            contrast_weight = contrast_weight_init * cosine_decay + contrast_weight_final * (1 - cosine_decay)
        else:
            contrast_weight = contrast_weight_final
        
        # Batch loop
        for batch_idx, (batch_indices, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_histories = [dataset.patient_history.get(dataset.get_subject_id(idx.item()), []) 
                              for idx in batch_indices]
            history_tensor = prepare_histories(batch_histories, device, max_len=100, 
                                              num_diagnoses=dataset.get_num_diagnoses())
    
            try:
                if args.model == 'HetMSAMRGNN' or args.model.startswith('HetMSAMRSGNN'):
                    batch_output, contrastive_loss = model(
                        features, adj_matrices, adj_dict, dataset, 
                        batch_indices, history=history_tensor, 
                        return_contrastive_loss=args.contrastive_loss
                    )
                    if torch.isnan(batch_output).any():
                        print(f"Warning: NaN in model output at epoch {epoch}, batch {batch_idx}")
                        batch_output = torch.nan_to_num(batch_output, nan=0.0)
                    if isinstance(contrastive_loss, torch.Tensor):
                        if torch.isnan(contrastive_loss).any():
                            print(f"Warning: NaN in contrastive loss at epoch {epoch}, batch {batch_idx}")
                            contrastive_loss = torch.tensor(0.0, device=device)
                    else:
                        if math.isnan(contrastive_loss):
                            print(f"Warning: NaN in contrastive loss at epoch {epoch}, batch {batch_idx}")
                            contrastive_loss = torch.tensor(0.0, device=device)
                        else:
                            contrastive_loss = torch.tensor(contrastive_loss, device=device)

                    base_loss = criterion(batch_output, batch_labels.to(device))
                    loss = base_loss + (contrast_weight * contrastive_loss if contrast_weight > 0 else 0)
                else:
                    batch_output = get_model_output(model, args.model, features, adj_matrices, 
                                                  adj_dict, history_tensor, ddi_edge_index, 
                                                  batch_indices)
                    loss = criterion(batch_output, batch_labels.to(device))
 
                if torch.isnan(loss).any():
                    print(f"Warning: NaN in loss at epoch {epoch}, batch {batch_idx}")
                    continue 

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except RuntimeError as e:
                print(f"Error in batch {batch_idx} of epoch {epoch}: {e}")
                continue

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        
        # === Validation phase ===
        model.eval()
        try:
            val_metrics = evaluate(model, dataset, val_mask, device, args)
            log_training_results(epoch, num_epochs, avg_loss, contrast_weight, val_metrics)
            current_performance = (val_metrics['ndcg@10'] * 0.3 + 
                                  val_metrics['avg_precision'] * 0.3 + 
                                  val_metrics['auc_roc'] * 0.4)
            scheduler.step(current_performance)
            if current_performance > best_performance:
                best_performance = current_performance
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), os.path.join(args.output_dir, model_filename))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        except Exception as e:
            print(f"Error during validation at epoch {epoch}: {e}")
            continue
  
    return best_model, model_filename

def get_model_output(model, model_name, features, adj_matrices, adj_dict, 
                    history_tensor, ddi_edge_index, batch_indices):
    """Get output from different model types"""
    if model_name in ['GCN', 'GAT']:
        output_dict = model(features, adj_matrices)
        return output_dict['p'][batch_indices]
    elif model_name == 'HAN':
        output_dict = model(features, adj_dict)
        return output_dict['p'][batch_indices]
    elif model_name == 'RETAIN':
        return model(features, history_tensor, batch_indices)['p']
    elif model_name in 'GAMENet':
        output_dict = model(features, history_tensor, batch_indices)
        return output_dict['p']
    elif model_name == 'VITA':
        output_dict = model(features, adj_dict, history_tensor, batch_indices)
        return output_dict['p']
    elif model_name == 'KGDNet':
        output_dict = model(features, adj_dict, ddi_edge_index, batch_indices)
        return output_dict['p']
    elif model_name == 'MoleRec':
        output_dict = model(features, adj_dict, history_tensor, batch_indices)
        return output_dict['p']
    elif model_name in ['SafeDrug', 'LEAP']:
        output_dict = model(features, history_tensor, ddi_edge_index, batch_indices)
        return output_dict['p']
    elif model_name in ['Deepwalk', 'Node2Vec']:
        output_dict = model(features, history_tensor, batch_indices)
        return output_dict['p']
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def log_training_results(epoch, num_epochs, avg_loss, contrast_weight, val_metrics):
    """Log training results"""
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Contrast Weight: {contrast_weight:.4f}')
    logging.info(f'Validation Metrics:')
    logging.info(f'  NDCG@10: {val_metrics["ndcg@10"]:.4f}')
    logging.info(f'  AUC-ROC: {val_metrics["auc_roc"]:.4f}')
    logging.info(f'  Average Precision: {val_metrics["avg_precision"]:.4f}')
    logging.info(f"  Jaccard Index: {val_metrics['jaccard']:.4f}")

    for k in val_metrics['precision@k']:
        logging.info(f'  Precision@{k}: {val_metrics["precision@k"][k]:.4f}')
        logging.info(f'  Recall@{k}: {val_metrics["recall@k"][k]:.4f}')
        logging.info(f'  F1@{k}: {val_metrics["f1@k"][k]:.4f}')
        logging.info(f'  MAP@{k}: {val_metrics["map@k"][k]:.4f}')

def get_model_filename(args):
    lr_str = f"{args.lr:.4f}".split(".")[1]  

    return f'best_model_{args.model}_hd{args.hidden_dim}_nv{args.batch_size}_lr{lr_str}.pth'

def evaluate(model, dataset, mask, device, args, k_list=[1, 3, 5, 10]):
    model.eval()
    features = dataset.get_features(device)
    adj_dict = dataset.get_type_specific_adj()
    node_id_ranges = dataset.get_node_id_ranges()

    for ntype, feat in features.items():
        if torch.isnan(feat).any():
            print(f"Warning: NaN detected in input features for node type {ntype} during evaluation")
            features[ntype] = torch.nan_to_num(feat, nan=0.0)

    _, ddi_edge_index = dataset.get_ddi_info()
    ddi_edge_index = ddi_edge_index.to(device)

    adj_matrices = {}
    for key, value in adj_dict.items():
        if isinstance(value, tuple):
            adj, edge_dim = value
            adj_matrices[key] = adj.to(device)
        else:
            adj_matrices[key] = value.to(device)
    adj_dict = adj_matrices

    with torch.no_grad():
        indices = torch.tensor(list(mask))

        histories = [dataset.patient_history.get(dataset.get_subject_id(idx.item()), []) for idx in indices]
        history_tensor = prepare_histories(histories, device, max_len=100, num_diagnoses=dataset.get_num_diagnoses())
    
        try:
            if args.model == 'HetMSAMRGNN' or args.model.startswith('HetMSAMRGNN'):
                output, _ = model(
                    features, adj_matrices, adj_dict, dataset, 
                    indices, history=history_tensor
                )
            else:
                if args.model in ['GCN', 'GAT']:
                    output_dict = model(features, adj_matrices)
                    output = output_dict['p'][indices]
                elif args.model == 'HAN':
                    output_dict = model(features, adj_dict)
                    output = output_dict['p'][indices]
                elif args.model == 'RETAIN':
                    output = model(features, history_tensor, indices)['p']
                elif args.model in ['GAMENet']:
                    output_dict = model(features, history_tensor, indices)
                    output = output_dict['p']
                elif args.model in ['SafeDrug', 'LEAP']:
                    output_dict = model(features, history_tensor, ddi_edge_index, indices)
                    output = output_dict['p']
                elif args.model in ['Deepwalk', 'Node2Vec']:
                    output_dict = model(features, history_tensor, indices)
                    output = output_dict['p']

            if torch.isnan(output).any():
                print("Warning: NaN detected in model output during evaluation")
                output = torch.nan_to_num(output, nan=0.0)
       
            labels = torch.zeros(len(mask), dataset.node_type_count['c'])
            for i, patient_idx in enumerate(mask):
                positive_drugs = dataset.edge_index['p_c'][1][dataset.edge_index['p_c'][0] == patient_idx]
                positive_drugs_local = positive_drugs - node_id_ranges['c'][0]
                labels[i, positive_drugs_local] = 1

            output = torch.sigmoid(output)
            output_np = output.cpu().numpy()
            labels_np = labels.cpu().numpy()
            if np.isnan(output_np).any():
                print("Warning: NaN detected in output_np during evaluation")
                output_np = np.nan_to_num(output_np, nan=0.0)
            predictions = (output_np > 0.5).astype(float)
            
            try:
                accuracy = accuracy_score(labels_np.flatten(), predictions.flatten())
                precision = precision_score(labels_np.flatten(), predictions.flatten(), average='macro', zero_division=0)
                recall = recall_score(labels_np.flatten(), predictions.flatten(), average='macro', zero_division=0)
                f1 = f1_score(labels_np.flatten(), predictions.flatten(), average='macro', zero_division=0)
                jaccard = jaccard_score(labels_np.flatten(), predictions.flatten(), average='macro', zero_division=0)
            except Exception as e:
                print(f"Error calculating classification metrics: {e}")
                accuracy = precision = recall = f1 = jaccard = 0.0
            try:
                ndcg = ndcg_score(labels_np, output_np, k=10)
                auc_roc = roc_auc_score(labels_np.flatten(), output_np.flatten())
                avg_precision = average_precision_score(labels_np.flatten(), output_np.flatten())
            except Exception as e:
                print(f"Error calculating ranking metrics: {e}")
                ndcg = auc_roc = avg_precision = 0.0

            precision_at_k = {}
            recall_at_k = {}
            f1_at_k = {}
            for k in k_list:
                try:
                    top_k_indices = np.argsort(-output_np, axis=1)[:, :k]
                    precision_at_k[k] = np.mean([np.sum(labels_np[i, top_k_indices[i]]) / k for i in range(len(mask))])
                    recall_at_k[k] = np.mean([np.sum(labels_np[i, top_k_indices[i]]) / max(np.sum(labels_np[i]), 1e-10) for i in range(len(mask))])
                    sample_f1_at_k = []
                    for i in range(len(mask)):
                        precision_i = np.sum(labels_np[i, top_k_indices[i]]) / k if k > 0 else 0
                        recall_i = np.sum(labels_np[i, top_k_indices[i]]) / max(np.sum(labels_np[i]), 1e-10)
                        if precision_i + recall_i > 0:
                            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i)
                        else:
                            f1_i = 0
                        sample_f1_at_k.append(f1_i)
                    f1_at_k[k] = np.mean(sample_f1_at_k)
                except Exception as e:
                    print(f"Error calculating metrics for k={k}: {e}")
                    precision_at_k[k] = recall_at_k[k] = f1_at_k[k] = 0.0

            def apk(actual, predicted, k=10):
                if len(predicted) > k:
                    predicted = predicted[:k]
                score = 0.0
                num_hits = 0.0
                for i, p in enumerate(predicted):
                    if p in actual and p not in predicted[:i]:
                        num_hits += 1.0
                        score += num_hits / (i + 1.0)
                return score / min(len(actual), k) if len(actual) > 0 else 0.0

            def mapk(actual, predicted, k=10):
                return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

            try:
                actual = [set(np.where(l == 1)[0]) for l in labels_np]
                predicted = [np.argsort(-o)[:max(k_list)] for o in output_np]
                map_k = {k: mapk(actual, predicted, k=k) for k in k_list}
            except Exception as e:
                print(f"Error calculating MAP@k: {e}")
                map_k = {k: 0.0 for k in k_list}
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            accuracy = precision = recall = f1 = ndcg = auc_roc = avg_precision = jaccard = 0.0
            precision_at_k = recall_at_k = f1_at_k = map_k = {k: 0.0 for k in k_list}
            output_np = np.zeros((len(mask), dataset.node_type_count['c']))
            labels_np = np.zeros((len(mask), dataset.node_type_count['c']))

    return {
        'accuracy': accuracy,
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1,
        'ndcg@10': ndcg,
        'auc_roc': auc_roc,
        'avg_precision': avg_precision,
        'precision@k': precision_at_k,
        'recall@k': recall_at_k,
        'f1@k': f1_at_k,
        'map@k': map_k,
        'jaccard': jaccard,
        'output_np': output_np,
        'labels_np': labels_np
    }

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Loading and preprocessing data...")

    if not hasattr(args, 'dataset'):
        args.dataset = 'mimic'  
    
    if not hasattr(args, 'eicu_path'):
        args.eicu_path = None

    if not hasattr(args, 'save_processed_data'):
        args.save_processed_data = False
        
    if not hasattr(args, 'use_processed_data'):
        args.use_processed_data = False

    if not hasattr(args, 'generate_explanations'):
        args.generate_explanations = False
    
    if not hasattr(args, 'explanation_mode'):
        args.explanation_mode = 'case_study'
    
    if not hasattr(args, 'explanation_output_dir'):
        args.explanation_output_dir = os.path.join(args.output_dir, 'explanations')
    
    if not hasattr(args, 'num_explanations'):
        args.num_explanations = 1
    
    if not hasattr(args, 'paper_figures'):
        args.paper_figures = False

    if args.dataset == 'mimic':
        data = load_data(f'{args.root}/data/modified_data.pkl')
        patient_history_path = f'{args.root}/data/patient_history_sequences.pkl'
        if os.path.exists(patient_history_path):
            with open(patient_history_path, 'rb') as f:
                patient_history = pickle.load(f)
        else:
            patient_history = None

    elif args.dataset == 'eicu':
        if args.use_processed_data:
            eicu_processed_path = os.path.join(args.eicu_path, 'eicu_processed_final.pkl')
            eicu_history_path = os.path.join(args.eicu_path, 'eicu_patient_history_final.pkl')
            
            if os.path.exists(eicu_processed_path) and os.path.exists(eicu_history_path):
                logging.info(f"Loading pre-processed eICU data from {eicu_processed_path}")
                with open(eicu_processed_path, 'rb') as f:
                    data = pickle.load(f)
                
                with open(eicu_history_path, 'rb') as f:
                    patient_history = pickle.load(f)
                
                print(f"Loaded pre-processed eICU data with {len(data['node_features']['p'])} patients")
            else:
                logging.warning("Pre-processed eICU data not found. Processing from scratch.")
                args.use_processed_data = False

        if not args.use_processed_data:
            if args.eicu_path:
                data_path = os.path.join(args.eicu_path, 'eicu_modified_data.pkl')
                history_path = os.path.join(args.eicu_path, 'eicu_patient_history.pkl')
                
                if os.path.exists(data_path) and os.path.exists(history_path):
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    with open(history_path, 'rb') as f:
                        patient_history = pickle.load(f)
                    
                    print(f"Loaded eICU data with {len(data['node_features']['p'])} patients")
                else:
                    raise ValueError(f"eICU data files not found at {args.eicu_path}")
            else:
                raise ValueError("eicu_path must be specified when using eICU dataset")
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset = DrugRecommendationDataset(data, patient_history)
    dataset_stats = get_dataset_stats(dataset)
    print("Dataset Statistics:")
    for key, value in dataset_stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    if args.save_processed_data and args.dataset == 'eicu' and not args.use_processed_data:
        logging.info("Saving processed eICU data for future use...")
        processed_data = {
            'node_features': {k: v.cpu().numpy() for k, v in dataset.node_features.items()},
            'edge_index': {k: v.cpu().numpy() for k, v in dataset.edge_index.items()},
            'edge_features': dataset.edge_features, 
            'unified_map': dataset.unified_map,
            'ordered_drug_mapping': dataset.drug_mapping,
            'labels': dataset.labels
        }

        if hasattr(dataset, 'type_specific_adj'):
            processed_data['type_specific_adj'] = {
                k: (v[0].to_dense().cpu().numpy() if isinstance(v[0], torch.Tensor) else v[0], v[1]) 
                for k, v in dataset.type_specific_adj.items()
            }

        save_dir = os.path.join(args.eicu_path, 'final')
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'eicu_processed_final.pkl'), 'wb') as f:
            pickle.dump(processed_data, f)

        with open(os.path.join(save_dir, 'eicu_patient_history_final.pkl'), 'wb') as f:
            pickle.dump(dataset.patient_history, f)
        
        print(f"Saved processed eICU data to {save_dir}")
    
    train_mask, val_mask, test_mask = dataset.split_data()

    print(f"Number of nodes: {sum(dataset.node_type_count.values())}")    
    print(f"Number of training samples: {len(train_mask)}")
    print(f"Number of validation samples: {len(val_mask)}")
    print(f"Number of test samples: {len(test_mask)}")
    
    metapaths = {
        'p': [['c', 'p'], ['s', 'p'], ['p', 'c', 'p'], ['p', 's', 'p'], ['p', 'd', 'p']],
        'c': [['p', 'c'], ['c', 'p', 'c']],
        's': [['p', 's'], ['s', 'p', 's']],
        'd': [['p', 'd'], ['d', 'p', 'd']]
    }

    if args.mode in ['train', 'evaluate']:
        models_to_run = [args.model]
        
        for model_name in models_to_run:
            print(f"\nRunning model: {model_name}")
            model = get_model(args, dataset, metapaths=metapaths)
            
            model.to(device)

            if args.mode == 'train' or args.model_path is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience // 2, verbose=True)
                criterion = DrugRecommendationLoss()
                train_func = modify_train_function(train)
                best_model, model_filename = train_func(
                    model, dataset, train_mask, val_mask, optimizer, 
                    scheduler, criterion, num_epochs=args.num_epochs, device=device, args=args
                )
                
                save_path = os.path.join(args.output_dir, model_filename)
                torch.save(best_model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")
            else:
                lr_str = f"{args.lr:.4f}".split(".")[1]
                model_file = f'best_model_{model_name}_hd{args.hidden_dim}_nv{args.batch_size}_lr{lr_str}.pth'
                model_path = os.path.join(args.output_dir, model_file)
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"Loaded pre-trained model for {model_name}")
                    best_model = model
                else:
                    print(f"Pre-trained model not found for {model_name}. Training from scratch.")
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience // 2, verbose=True)
                    criterion = DrugRecommendationLoss()

                    best_model, model_filename = train(model, dataset, train_mask, val_mask, optimizer, 
                               scheduler, criterion, num_epochs=args.num_epochs, device=device, args=args)
                    
                    save_path = os.path.join(args.output_dir, model_filename)
                    torch.save(best_model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")

            if args.mode in ['train', 'evaluate']:
                evaluate_func = modify_evaluate_function(evaluate)
                test_metrics = evaluate_func(
                    best_model, dataset, test_mask, device, args, 
                    generate_explanations_flag=(args.generate_explanations and args.explanation_mode == 'test')
                )
                
                results = {model_name: test_metrics}

                logging.info(f"Results for {model_name}:")
                logging.info(f"Test NDCG@10: {test_metrics['ndcg@10']:.4f}")
                logging.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
                logging.info(f"Test Average Precision: {test_metrics['avg_precision']:.4f}")
                
                for k in [1, 3, 5, 10]:  
                    logging.info(f"Test Precision@{k}: {test_metrics['precision@k'][k]:.4f}")
                    logging.info(f"Test Recall@{k}: {test_metrics['recall@k'][k]:.4f}")
                    logging.info(f"Test F1@{k}: {test_metrics['f1@k'][k]:.4f}")
                    logging.info(f"Test MAP@{k}: {test_metrics['map@k'][k]:.4f}")
    
    elif args.mode == 'case_study':
        logging.info("Starting case study with explanations...")
        model = get_model(args, dataset, metapaths=metapaths)
        model.to(device)
        
        if args.model_path:
            lr_str = f"{args.lr:.4f}".split(".")[1]
            model_file = os.path.join(args.model_path, f'best_model_{args.model}_hd{args.hidden_dim}_nv{args.batch_size}_lr{lr_str}.pth')
            if os.path.exists(model_file):
                model.load_state_dict(torch.load(model_file, map_location=device))
                print(f"Loaded model from {model_file}")
            else:
                print(f"Model file not found: {model_file}")
                return
        else:
            print("Model path not specified. Please provide --model_path")
            return

        icd_mapping = load_icd_mapping('./mimiciv/3.0/hosp/d_icd_diagnoses.csv.gz')
        original_data = load_data(f'{args.root}/data/modified_data.pkl')
        index_diagnosis_map = {v: k for k, v in diagnosis_index_map.items()}

        if args.generate_explanations:
            features = dataset.get_features(device)
            adj_dict = dataset.get_type_specific_adj()
            adj_matrices = {}
            for key, value in adj_dict.items():
                if isinstance(value, tuple):
                    adj, edge_dim = value
                    adj_matrices[key] = adj.to(device)
                else:
                    adj_matrices[key] = value.to(device)

            case_studies = add_case_study_code(args, model, dataset, device, test_mask)
            case_study_results = {
                'model': args.model,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'num_patients': len(case_studies),
                'patient_ids': [dataset.get_subject_id(idx) for idx in case_studies.keys()]
            }
            
            results_path = os.path.join(args.explanation_output_dir, 'case_study_results.json')
            with open(results_path, 'w') as f:
                json.dump(case_study_results, f, indent=2)
            
            logging.info(f"Case study results saved to {results_path}")
        else:
            run_case_study(args, model, dataset, device, original_data, icd_mapping, patient_history, index_diagnosis_map)
    
    elif args.mode == 'paper_figures':
        logging.info("Generating figures for the paper...")
        model = get_model(args, dataset, metapaths=metapaths)
        model.to(device)
        
        if args.model_path:
            lr_str = f"{args.lr:.4f}".split(".")[1]
            model_file = os.path.join(args.model_path, f'best_model_{args.model}_hd{args.hidden_dim}_nv{args.batch_size}_lr{lr_str}.pth')
            if os.path.exists(model_file):
                model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
                print(f"Loaded model from {model_file}")
            else:
                print(f"Model file not found: {model_file}")
                return
        else:
            print("Model path not specified. Please provide --model_path")
            return
        args.generate_explanations = True
        args.explanation_mode = 'paper_figures'
        args.paper_figures = True
        features = dataset.get_features(device)
        adj_dict = dataset.get_type_specific_adj()
        adj_matrices = {}
        for key, value in adj_dict.items():
            if isinstance(value, tuple):
                adj, edge_dim = value
                adj_matrices[key] = adj.to(device)
            else:
                adj_matrices[key] = value.to(device)

        test_indices = torch.tensor(list(test_mask), device=device)
        args.use_full_test_set = True
        explanations = generate_explanations(
            args, model, dataset, device, test_mask,
            features, adj_matrices, adj_dict
        )
        
        logging.info(f"Paper figures generated and saved to {args.explanation_output_dir}")
    
    elif args.mode == 'ablation':
        if args.model != 'HetMSAMRGNN':
            raise ValueError("Ablation study is only available for HetMSAMRGNN model.")
        if args.ablation:
            models_to_run = [args.ablation]
        else:
            models_to_run = ['NoMultiView', 'NoMultiScale', 'NoFrequency', 'NoMetapath', 'NoDDI', 'NoHistory']
        
        results = {}
        
        for ablation_type in models_to_run:
            print(f"\nRunning ablation model: Het-MSAMRGNN_{ablation_type}")
            model = get_ablation_model(args, dataset, metapaths, ablation_type)
            model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience // 2, verbose=True)
            criterion = DrugRecommendationLoss()
            
            lr_str = f"{args.lr:.4f}".split(".")[1]
            model_file = f'best_model_HetMSAMRGNN_{ablation_type}_hd{args.hidden_dim}_nv{args.batch_size}_lr{lr_str}.pth'
            model_path = os.path.join(args.output_dir, model_file)
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded pre-trained model for Het-MSAMRGNN_{ablation_type}")
                best_model = model
            else:
                print(f"Pre-trained model not found for Het-MSAMRGNN_{ablation_type}. Training from scratch.")
                best_model, _ = train(model, dataset, train_mask, val_mask, optimizer, 
                                    scheduler, criterion, num_epochs=args.num_epochs, device=device, args=args)
                
                torch.save(best_model.state_dict(), model_path)
                print(f"Best model saved to {model_path}")
            
            test_metrics = evaluate(best_model, dataset, test_mask, device, args)
            results[f"Het-MSAMRGNN_{ablation_type}"] = test_metrics
            
            for k in [1, 3, 5, 10]:
                logging.info(f"Test Precision@{k}: {test_metrics['precision@k'][k]:.4f}")
                logging.info(f"Test Recall@{k}: {test_metrics['recall@k'][k]:.4f}")
                logging.info(f"Test F1@{k}: {test_metrics['f1@k'][k]:.4f}")
                logging.info(f"Test MAP@{k}: {test_metrics['map@k'][k]:.4f}")
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    if args.mode in ['train', 'evaluate', 'ablation']:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.root}/results/result_{timestamp}_model_{args.model}_mode_{args.mode}"
        if args.mode == 'ablation':
            filename += f"_ablation_{'-'.join(models_to_run)}"
        filename += f"_hd{args.hidden_dim}_nv{args.num_views}_ns{args.num_scales}_nh{args.num_heads}_nl{args.num_layers}.json"
        
        save_results(results, filename)
        logging.info(f"Results saved to {filename}")

def get_args():
    parser = argparse.ArgumentParser(description='HetMS-AMRGNN for Drug Recommendation')
    parser = add_explainability_args(parser)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the best model')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'case_study', 'ablation', 'paper_figures'], 
                        default='train', help='Execution mode: train, evaluate, case_study, or ablation')
    parser.add_argument('--model_path', type=str, default='./output', 
                        help='Path to a pre-trained model for evaluation or case study')
    parser.add_argument('--case_study_patient_id', type=str, default='3283', 
                        help='Patient ID for case study (if None, a random patient will be selected)')
    parser.add_argument('--model', type=str, default='HetMSAMRGNN', 
                        choices=['HetMSAMRGNN', 'GCN', 'GAT', 'HAN', 'RETAIN', 'GAMENet', 'SafeDrug', 'LEAP', 
                                 'Deepwalk', 'Node2Vec'],
                        help='Model to use for drug recommendation')
    parser.add_argument('--ablation', type=str, nargs='+', default='NoMultiScale',
                        choices=['NoMultiView', 'NoMultiScale', 'NoFrequency', 'NoMetapath', 'NoDDI', 'NoHistory'],
                        help='Specify which ablation models to run')
    parser.add_argument('--out_dim', type=int, default=4, help='number of classes')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_views', type=int, default=3, help='Number of views')
    parser.add_argument('--num_scales', type=int, default=3, help='Number of scales')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--use_history', type=bool, default=True, help='Use patient history')
    parser.add_argument('--contrastive_loss', type=bool, default=True, help='Use contrastive_loss')
    parser.add_argument('--memory_size', type=int, default=10, help='Memory size for GAMENet')
    parser.add_argument('--dataset', type=str, default='eicu', choices=['mimic', 'eicu'],
                        help='Dataset to use: mimic or eicu')
    parser.add_argument('--eicu_path', type=str, default='./data/eicu/output/final', 
                        help='Path to processed eICU data directory')
    parser.add_argument('--save_processed_data', action='store_true',
                        help='Save processed data for future use')
    parser.add_argument('--use_processed_data', action='store_true',
                        help='Use previously processed and saved data')

    return parser.parse_args()

def get_model(args, dataset, metapaths=None):
    edge_types = [('p', 'c'), ('c', 'p'), ('p', 's'), ('s', 'p'), ('p', 'd'), ('d', 'p')]
    num_diagnoses = dataset.get_num_diagnoses()
    node_id_ranges = dataset.get_node_id_ranges()
    in_sizes = dataset.node_type_dims
    input_dim = dataset.node_type_dims['p']
    output_dim = dataset.node_type_count['c']
    if args.model == 'HetMSAMRGNN':
        return HetMSAMRGNN(dataset.node_types, dataset.node_type_dims, args.hidden_dim, 
                         dataset.node_type_count['c'], args.num_views, args.num_scales, 
                         args.num_heads, args.num_layers, metapaths=metapaths, 
                         node_id_ranges=node_id_ranges, dropout_rate=args.dropout, 
                         use_history=args.use_history, num_diagnoses=num_diagnoses)
    elif args.model == 'GCN':
        return HeteroGCN(
            node_types=dataset.node_types,
            edge_types=edge_types,
            in_channels=dataset.node_type_dims,
            hidden_channels=args.hidden_dim,
            out_channels=dataset.node_type_count['c'],
            metapaths=metapaths
        )
    elif args.model == 'GAT':
        return HeteroGAT(
            node_types=dataset.node_types,
            edge_types=edge_types,
            in_channels=dataset.node_type_dims,
            hidden_channels=args.hidden_dim,
            out_channels=dataset.node_type_count['c'],
            num_heads=args.num_heads,
            metapaths=metapaths
        )
    elif args.model == 'HAN':
        return HAN(
            node_types=dataset.node_types,
            in_dims=dataset.node_type_dims,
            hidden_dim=args.hidden_dim,
            out_dim=dataset.node_type_count['c'],
            num_heads=args.num_heads,
            dropout=args.dropout,
            metapaths=metapaths
        )
    elif args.model == 'RETAIN':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        num_diagnoses = dataset.get_num_diagnoses()
        return RETAIN(node_types, in_dims, hidden_dim, out_dim, num_diagnoses)
    
    elif args.model == 'GAMENet':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        memory_size = args.memory_size  
        num_diagnoses = dataset.get_num_diagnoses()
        return GAMENet(node_types, in_dims, hidden_dim, out_dim, memory_size, num_diagnoses)
    
    elif args.model == 'SafeDrug':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        memory_size = args.memory_size
        num_diagnoses = dataset.get_num_diagnoses()
        return SafeDrug(node_types, in_dims, hidden_dim, out_dim, memory_size, num_diagnoses)
    
    elif args.model == 'VITA':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        num_diagnoses = dataset.get_num_diagnoses()
        return VITA(node_types, in_dims, hidden_dim, out_dim, num_diagnoses, 
                   dropout=args.dropout)
    
    elif args.model == 'KGDNet':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        num_diagnoses = dataset.get_num_diagnoses()
        kg_dim = args.kg_dim if hasattr(args, 'kg_dim') else hidden_dim
        return KGDNet(node_types, in_dims, hidden_dim, out_dim, kg_dim=kg_dim,
                     num_diagnoses=num_diagnoses, dropout=args.dropout)
    
    elif args.model == 'MoleRec':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        num_diagnoses = dataset.get_num_diagnoses()
        return MoleRec(node_types, in_dims, hidden_dim, out_dim, 
                      num_diagnoses=num_diagnoses, dropout=args.dropout)

    elif args.model == 'LEAP':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        num_diagnoses = dataset.get_num_diagnoses()
        return LEAP(node_types, in_dims, hidden_dim, out_dim, num_diagnoses)
    
    elif args.model == 'Deepwalk':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        adj_dict = dataset.get_type_specific_adj()
        deepwalk = DeepWalk(adj_dict, walk_length=3, num_walks=3, dimensions=args.hidden_dim)
        deepwalk_embeddings = deepwalk.get_embeddings()

        return DeepWalkModel(node_types, in_dims, hidden_dim, out_dim, deepwalk_embeddings)

    elif args.model == 'Node2Vec':
        node_types = dataset.node_types
        in_dims = dataset.node_type_dims
        hidden_dim = args.hidden_dim
        out_dim = dataset.node_type_count['c']
        adj_dict = dataset.get_type_specific_adj()
        node2vec = Node2Vec(adj_dict, walk_length=3, num_walks=3, dimensions=args.hidden_dim, p=1, q=1)
        node2vec_embeddings = node2vec.get_embeddings()

        return Node2VecModel(node_types, in_dims, hidden_dim, out_dim, node2vec_embeddings)

    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == "__main__":
    main()
