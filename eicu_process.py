import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Final processing of eICU data')
    parser.add_argument('--eicu_path', type=str, default='./data/eicu/output',
                        help='Path to processed eICU data')
    parser.add_argument('--original_data_path', type=str, default ='./eicu/physionet.org/files/eicu-crd/2.0',
                        help='Path to original eICU data (for diagnosis statistics)')
    parser.add_argument('--output_dir', type=str, default='./eicu/output5/processed_updated_history',
                        help='Output directory for processed data')
    parser.add_argument('--top_n_diagnoses', type=int, default=50,
                        help='Number of top diagnoses to keep')
    return parser.parse_args()

def load_data(eicu_path):
    """Load processed eICU data"""
    graph_path = os.path.join(eicu_path, 'eicu_hetms_graph.pkl')
    metadata_path = os.path.join(eicu_path, 'eicu_hetms_metadata.pkl')
    
    print(f"Loading graph data from: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)
    
    print(f"Loading metadata from: {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return graph_data, metadata

def load_original_diagnoses(original_data_path):
    """Load original diagnosis data for statistics"""
    if not original_data_path:
        print("No original data path provided, skipping diagnosis statistics")
        return None
    
    diagnosis_path = os.path.join(original_data_path, 'diagnosis.csv')
    print(f"Loading original diagnosis data from: {diagnosis_path}")
    
    try:
        diagnosis_df = pd.read_csv(diagnosis_path)
        return diagnosis_df
    except Exception as e:
        print(f"Error loading original diagnosis data: {e}")
        return None

def filter_unknown_patients(graph_data, metadata):
    """Remove patients with unknown infection category"""
    infection_categories = metadata.get('infection_categories', {})
    known_patients = [
        int(patient_id) for patient_id, category in infection_categories.items()
        if category != 'unknown' and isinstance(patient_id, (int, float, str))
    ]
    known_patients_set = set()
    for pid in known_patients:
        try:
            known_patients_set.add(int(pid))
        except (ValueError, TypeError):
            known_patients_set.add(pid)

    category_counts = Counter([cat for pat_id, cat in infection_categories.items() 
                              if pat_id in known_patients_set])
    
    print(f"Found {len(known_patients_set)} patients with known infection categories")
    print("Category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} patients")

    patient_mask = np.zeros(len(graph_data['node_features']['p']), dtype=bool)
    patient_index_map = {}  
    
    for (node_type, node_id), idx in graph_data['unified_map'].items():
        if node_type == 'p':
            try:
                int_node_id = int(node_id)
                patient_index_map[int_node_id] = idx
            except (ValueError, TypeError):
                patient_index_map[node_id] = idx
    
    for patient_id in known_patients_set:
        if patient_id in patient_index_map:
            idx = patient_index_map[patient_id]
            if idx < len(patient_mask):
                patient_mask[idx] = True
    
    print(f"Keeping {patient_mask.sum()} of {len(patient_mask)} patient nodes")
    
    return patient_mask, known_patients_set, category_counts

def get_top_diagnoses(diagnosis_df, patient_ids, top_n=50, patient_id_col='patientunitstayid', diagnosis_col='diagnosisstring'):
    """Get top diagnoses from the original diagnosis data"""
    if diagnosis_df is None:
        print("No diagnosis data provided, cannot extract top diagnoses")
        return None, None
    patient_diagnoses = diagnosis_df[diagnosis_df[patient_id_col].isin(patient_ids)]
    diagnosis_counts = patient_diagnoses[diagnosis_col].value_counts()
    top_diagnoses = diagnosis_counts.head(top_n)
    top_diagnosis_set = set(top_diagnoses.index)
    
    print(f"Found {len(diagnosis_counts)} unique diagnoses")
    print(f"Selected top {top_n} diagnoses which cover {top_diagnoses.sum() / diagnosis_counts.sum():.2%} of all diagnoses")

    diagnosis_to_idx = {diag: i for i, diag in enumerate(top_diagnoses.index)}
    
    return top_diagnoses, diagnosis_to_idx

def create_filtered_data(graph_data, patient_mask, diagnosis_to_idx=None, original_diagnosis_df=None):
    """Create filtered dataset with only selected patients and top diagnoses"""
    filtered_data = {
        'node_features': {},
        'edge_index': {},
        'edge_features': {},
        'unified_map': {},
        'ordered_drug_mapping': {}
    }

    filtered_data['node_features']['p'] = graph_data['node_features']['p'][patient_mask]
    patient_idx_map = {}  
    current_idx = 0
    for i, mask_value in enumerate(patient_mask):
        if mask_value:
            patient_idx_map[i] = current_idx
            current_idx += 1

    diagnosis_nodes = []
    for (node_type, node_id), idx in graph_data['unified_map'].items():
        if node_type == 'p' and idx in patient_idx_map:
            filtered_data['unified_map'][(node_type, node_id)] = patient_idx_map[idx]
        elif node_type == 'd':
            diagnosis_nodes.append((node_id, idx))

    if diagnosis_to_idx is not None and original_diagnosis_df is not None:
        patient_ids = [node_id for (node_type, node_id), _ in filtered_data['unified_map'].items() 
                     if node_type == 'p']

        top_n = len(diagnosis_to_idx)
        diagnosis_features = np.zeros((top_n, top_n))
        np.fill_diagonal(diagnosis_features, 1)  
        filtered_data['node_features']['d'] = diagnosis_features
        for diagnosis_text, new_idx in diagnosis_to_idx.items():
            filtered_data['unified_map'][('d', diagnosis_text)] = current_idx + new_idx

        print("Creating new patient-diagnosis edges from original data")
        p_d_source = []
        p_d_target = []
        d_p_source = []
        d_p_target = []

        for _, row in original_diagnosis_df.iterrows():
            patient_id = row['patientunitstayid']
            diagnosis_text = row['diagnosisstring']
            if patient_id not in patient_ids or diagnosis_text not in diagnosis_to_idx:
                continue

            patient_idx = None
            for (node_type, node_id), idx in filtered_data['unified_map'].items():
                if node_type == 'p' and node_id == patient_id:
                    patient_idx = idx
                    break
            
            diagnosis_idx = current_idx + diagnosis_to_idx[diagnosis_text]
            
            if patient_idx is not None:
                p_d_source.append(patient_idx)
                p_d_target.append(diagnosis_idx)
                d_p_source.append(diagnosis_idx)
                d_p_target.append(patient_idx)
        
        filtered_data['edge_index']['p_d'] = np.array([p_d_source, p_d_target])
        filtered_data['edge_index']['d_p'] = np.array([d_p_source, d_p_target])
        current_idx += top_n
    else:
        filtered_data['node_features']['d'] = graph_data['node_features']['d']
        for node_id, idx in diagnosis_nodes:
            original_d_idx = idx - len(graph_data['node_features']['p'])
            if 0 <= original_d_idx < len(graph_data['node_features']['d']):
                filtered_data['unified_map'][('d', node_id)] = current_idx + original_d_idx
        
        current_idx += len(graph_data['node_features']['d'])

        if 'p_d' in graph_data['edge_index']:
            p_d_source = []
            p_d_target = []
            for src, dst in zip(graph_data['edge_index']['p_d'][0], graph_data['edge_index']['p_d'][1]):
                if src in patient_idx_map:
                    new_src = patient_idx_map[src]
                    d_offset = len(graph_data['node_features']['p'])
                    new_dst = current_idx - len(graph_data['node_features']['d']) + (dst - d_offset)
                    
                    p_d_source.append(new_src)
                    p_d_target.append(new_dst)
            
            filtered_data['edge_index']['p_d'] = np.array([p_d_source, p_d_target])

        if 'd_p' in graph_data['edge_index']:
            d_p_source = []
            d_p_target = []
            for src, dst in zip(graph_data['edge_index']['d_p'][0], graph_data['edge_index']['d_p'][1]):
                if dst in patient_idx_map:
                    d_offset = len(graph_data['node_features']['p'])
                    new_src = current_idx - len(graph_data['node_features']['d']) + (src - d_offset)
                    new_dst = patient_idx_map[dst]
                    
                    d_p_source.append(new_src)
                    d_p_target.append(new_dst)
            
            filtered_data['edge_index']['d_p'] = np.array([d_p_source, d_p_target])

    for node_type in ['c', 's', 'm', 't']:
        if node_type in graph_data['node_features']:
            filtered_data['node_features'][node_type] = graph_data['node_features'][node_type]

            for (nt, node_id), idx in graph_data['unified_map'].items():
                if nt == node_type:
                    type_offset = 0
                    for t in ['p', 'd']:
                        if t in graph_data['node_features']:
                            type_offset += len(graph_data['node_features'][t])
                    
                    original_idx = idx - type_offset
                    
                    if 0 <= original_idx < len(graph_data['node_features'][node_type]):
                        filtered_data['unified_map'][(nt, node_id)] = current_idx + original_idx
            
            current_idx += len(graph_data['node_features'][node_type])

    for edge_type in graph_data['edge_index']:
        if edge_type in ['p_d', 'd_p']:
            continue
        
        if '_' not in edge_type:
            continue
        
        source_type, dest_type = edge_type.split('_')

        if source_type == 'p' or dest_type == 'p':
            source_idx, dest_idx = graph_data['edge_index'][edge_type]
            new_source_idx = []
            new_dest_idx = []
            
            for src, dst in zip(source_idx, dest_idx):
                valid_edge = False
                new_src = src
                new_dst = dst
                
                if source_type == 'p' and src in patient_idx_map:
                    valid_edge = True
                    new_src = patient_idx_map[src]
                elif dest_type == 'p' and dst in patient_idx_map:
                    valid_edge = True
                    new_dst = patient_idx_map[dst]
                
                if valid_edge:
                    if source_type != 'p':
                        type_offset = 0
                        for t in ['p', 'd']:
                            if t in graph_data['node_features']:
                                type_offset += len(graph_data['node_features'][t])

                        if source_type == 'd':
                            if diagnosis_to_idx is not None:
                                continue
                            type_offset = len(graph_data['node_features']['p'])

                        original_idx = src - type_offset
                        if source_type == 'd':
                            new_src = len(filtered_data['node_features']['p']) + original_idx
                        else:
                            node_offset = len(filtered_data['node_features']['p'])
                            if 'd' in filtered_data['node_features']:
                                node_offset += len(filtered_data['node_features']['d'])
                            
                            for t in ['c', 's', 'm', 't']:
                                if t == source_type:
                                    break
                                if t in filtered_data['node_features']:
                                    node_offset += len(filtered_data['node_features'][t])
                            
                            new_src = node_offset + original_idx
                    
                    if dest_type != 'p':
                        type_offset = 0
                        for t in ['p', 'd']:
                            if t in graph_data['node_features']:
                                type_offset += len(graph_data['node_features'][t])
                        if dest_type == 'd':
                            if diagnosis_to_idx is not None:
                                continue
                            type_offset = len(graph_data['node_features']['p'])
                        original_idx = dst - type_offset
                        if dest_type == 'd':
                            new_dst = len(filtered_data['node_features']['p']) + original_idx
                        else:
                            node_offset = len(filtered_data['node_features']['p'])
                            if 'd' in filtered_data['node_features']:
                                node_offset += len(filtered_data['node_features']['d'])
                            
                            for t in ['c', 's', 'm', 't']:
                                if t == dest_type:
                                    break
                                if t in filtered_data['node_features']:
                                    node_offset += len(filtered_data['node_features'][t])
                            
                            new_dst = node_offset + original_idx
                    
                    new_source_idx.append(new_src)
                    new_dest_idx.append(new_dst)
            
            if new_source_idx:
                filtered_data['edge_index'][edge_type] = np.array([new_source_idx, new_dest_idx])
        else:
            filtered_data['edge_index'][edge_type] = graph_data['edge_index'][edge_type]

    for edge_type, features in graph_data.get('edge_features', {}).items():
        if edge_type in filtered_data['edge_index']:
            filtered_data['edge_features'][edge_type] = features

    filtered_data['ordered_drug_mapping'] = graph_data.get('ordered_drug_mapping', {})
    
    return filtered_data

def prepare_for_drug_recommendation_dataset(filtered_data):
    """Prepare data for DrugRecommendationDataset by creating labels and patient history"""
    labels = {}
    for (node_type, node_id), idx in filtered_data['unified_map'].items():
        if node_type == 'p':
            drug_count = len(filtered_data['node_features']['c'])
            labels[node_id] = np.zeros(drug_count, dtype=np.float32)
            if 'p_c' in filtered_data['edge_index']:
                patient_idx = filtered_data['unified_map'][(node_type, node_id)]
                for src, dst in zip(filtered_data['edge_index']['p_c'][0], filtered_data['edge_index']['p_c'][1]):
                    if src == patient_idx:
                        drug_offset = len(filtered_data['node_features']['p'])
                        if 'd' in filtered_data['node_features']:
                            drug_offset += len(filtered_data['node_features']['d'])
                        
                        drug_idx = dst - drug_offset
                        if 0 <= drug_idx < drug_count:
                            labels[node_id][drug_idx] = 1

    return filtered_data

def calculate_dataset_statistics(filtered_data):
    """Calculate statistics for the filtered dataset"""
    stats = {}
    stats['# patients'] = len(filtered_data['node_features']['p'])
    stats['# disease (top)'] = len(filtered_data['node_features']['d'])
    stats['# antimicrobial drugs'] = len(filtered_data['node_features']['c'])
    stats['# specimen (top)'] = len(filtered_data['node_features']['s']) if 's' in filtered_data['node_features'] else 0
    if 'p_d' in filtered_data['edge_index']:
        stats['# of patient-disease links'] = len(filtered_data['edge_index']['p_d'][0])
    else:
        stats['# of patient-disease links'] = 0

    if 'p_c' in filtered_data['edge_index']:
        stats['# of patient-drug links'] = len(filtered_data['edge_index']['p_c'][0])
    else:
        stats['# of patient-drug links'] = 0

    if 'p_s' in filtered_data['edge_index']:
        stats['# of patient-specimen links'] = len(filtered_data['edge_index']['p_s'][0])
    else:
        stats['# of patient-specimen links'] = 0
    patient_diagnosis_counts = defaultdict(int)
    patient_drug_counts = defaultdict(int)
    patient_specimen_counts = defaultdict(int)

    if 'p_d' in filtered_data['edge_index']:
        for src, _ in zip(filtered_data['edge_index']['p_d'][0], filtered_data['edge_index']['p_d'][1]):
            patient_diagnosis_counts[src] += 1

    if 'p_c' in filtered_data['edge_index']:
        for src, _ in zip(filtered_data['edge_index']['p_c'][0], filtered_data['edge_index']['p_c'][1]):
            patient_drug_counts[src] += 1

    if 'p_s' in filtered_data['edge_index']:
        for src, _ in zip(filtered_data['edge_index']['p_s'][0], filtered_data['edge_index']['p_s'][1]):
            patient_specimen_counts[src] += 1
    
    stats['avg. # of diseases per patient'] = sum(patient_diagnosis_counts.values()) / max(1, len(patient_diagnosis_counts))
    stats['avg. # of drugs per patient'] = sum(patient_drug_counts.values()) / max(1, len(patient_drug_counts))
    stats['avg. # of specimens per patient'] = sum(patient_specimen_counts.values()) / max(1, len(patient_specimen_counts))
    
    return stats

def save_dataset_statistics(stats, output_dir):
    """Save dataset statistics as a table"""
    os.makedirs(output_dir, exist_ok=True)
    stat_items = [(k, v) for k, v in stats.items()]
    df = pd.DataFrame(stat_items, columns=['Statistics', 'Value'])

    stats_path = os.path.join(output_dir, 'dataset_statistics.csv')
    df.to_csv(stats_path, index=False)
    print(f"Saved dataset statistics to: {stats_path}")

    with open(os.path.join(output_dir, 'dataset_statistics.md'), 'w') as f:
        f.write("# Dataset Statistics\n\n")
        f.write("| Statistics | Value |\n")
        f.write("|------------|-------|\n")
        for stat, value in stat_items:
            f.write(f"| {stat} | {value} |\n")

    plt.figure(figsize=(12, 8))
    count_stats = {k: v for k, v in stats.items() if "avg" not in k}
    plt.subplot(2, 1, 1)
    plt.bar(range(len(count_stats)), list(count_stats.values()))
    plt.xticks(range(len(count_stats)), list(count_stats.keys()), rotation=45, ha='right')
    plt.title('Count Statistics')
    plt.tight_layout()

    avg_stats = {k: v for k, v in stats.items() if "avg" in k}
    plt.subplot(2, 1, 2)
    plt.bar(range(len(avg_stats)), list(avg_stats.values()))
    plt.xticks(range(len(avg_stats)), list(avg_stats.keys()), rotation=45, ha='right')
    plt.title('Average Statistics')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'dataset_statistics.png'))
    plt.close()

def create_patient_history(filtered_data, diagnosis_df, diagnosis_to_idx):
    """Create patient diagnosis history from the diagnosis data"""
    patient_history = {}
    for (node_type, patient_id), _ in filtered_data['unified_map'].items():
        if node_type == 'p':
            patient_history[patient_id] = []
    if diagnosis_df is None or diagnosis_to_idx is None:
        print("Warning: No diagnosis data available. Patient history will be empty.")
        return patient_history
    for patient_id in patient_history.keys():
        patient_diagnoses = diagnosis_df[diagnosis_df['patientunitstayid'] == patient_id]
        if 'diagnosisoffset' in patient_diagnoses.columns:
            patient_diagnoses = patient_diagnoses.sort_values('diagnosisoffset')
        for _, row in patient_diagnoses.iterrows():
            diagnosis_text = row['diagnosisstring']
            if diagnosis_text in diagnosis_to_idx:
                diagnosis_idx = diagnosis_to_idx[diagnosis_text]
                patient_history[patient_id].append(diagnosis_idx)

    history_counts = [len(history) for history in patient_history.values()]
    avg_history_len = sum(history_counts) / len(history_counts) if history_counts else 0
    max_history_len = max(history_counts) if history_counts else 0
    
    print(f"Created patient history with average length: {avg_history_len:.2f}, max length: {max_history_len}")
    print(f"Patients with non-empty history: {sum(1 for h in history_counts if h > 0)}/{len(history_counts)}")
    
    return patient_history

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    graph_data, metadata = load_data(args.eicu_path)
    diagnosis_df = load_original_diagnoses(args.original_data_path)
    patient_mask, known_patients, category_counts = filter_unknown_patients(graph_data, metadata)
    top_diagnoses, diagnosis_to_idx = get_top_diagnoses(diagnosis_df, known_patients, args.top_n_diagnoses)
    filtered_data = create_filtered_data(graph_data, patient_mask, diagnosis_to_idx, diagnosis_df)
    modified_data = prepare_for_drug_recommendation_dataset(filtered_data)
    stats = calculate_dataset_statistics(modified_data)

    save_dataset_statistics(stats, args.output_dir)

    modified_data_path = os.path.join(args.output_dir, 'eicu_modified_data.pkl')
    with open(modified_data_path, 'wb') as f:
        pickle.dump(modified_data, f)
    print(f"Saved modified data to: {modified_data_path}")

    history_path = os.path.join(args.output_dir, 'eicu_patient_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(patient_history, f)
    print(f"Saved patient history to: {history_path}")

    if diagnosis_to_idx:
        diagnosis_map_path = os.path.join(args.output_dir, 'diagnosis_index_map.pkl')
        with open(diagnosis_map_path, 'wb') as f:
            pickle.dump(diagnosis_to_idx, f)
        print(f"Saved diagnosis mapping to: {diagnosis_map_path}")

    filtered_categories = {
        patient_id: category for patient_id, category in metadata['infection_categories'].items()
        if patient_id in known_patients
    }
    categories_path = os.path.join(args.output_dir, 'infection_categories.pkl')
    with open(categories_path, 'wb') as f:
        pickle.dump(filtered_categories, f)
    print(f"Saved infection categories to: {categories_path}")
    
    print("\nData processing complete!")
    print("Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nInfection category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} patients ({count/sum(category_counts.values())*100:.1f}%)")

if __name__ == "__main__":
    main()