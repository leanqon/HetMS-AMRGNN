import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter


pd.set_option('display.max_columns', None)

def process_data(final_data, filtered_patients, micro_event):
    patient_info = filtered_patients[['subject_id', 'gender', 'anchor_age', 'icd_code']].drop_duplicates()
    diagnoses = patient_info['icd_code'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('icd_code')
    diagnoses = diagnoses.join(patient_info[['subject_id']])
    diagnoses = diagnoses.drop_duplicates()
    final_data['start'] = pd.to_datetime(final_data['start'])
    final_data['end'] = pd.to_datetime(final_data['end'])
    final_data['itemid'] = pd.to_numeric(final_data['itemid'], errors='coerce').astype('Int64')
    return patient_info, diagnoses, final_data, micro_event

def standardize_drug_name(drug):
    return drug.strip().lower() if isinstance(drug, str) else drug

def get_drug_combo(row):
    drug1 = standardize_drug_name(row['drug1'])
    drug2 = standardize_drug_name(row['drug2'])
    if pd.isna(drug2):
        return drug1
    return tuple(sorted([drug1, drug2]))

def create_drug_features(final_data):
    # 获取所有药物组合
    final_data['drug_combo'] = final_data.apply(get_drug_combo, axis=1)

    # 提取所有单个药物和组合药物
    all_combos = set(final_data['drug_combo'])
    all_drug1 = set(final_data['drug1'].apply(standardize_drug_name))
    all_combos.update(all_drug1)
    single_drugs = sorted(set(drug for combo in all_combos for drug in (combo if isinstance(combo, tuple) else [combo])))

    # 创建 drug_features DataFrame
    drug_features = pd.DataFrame(index=[', '.join(combo) if isinstance(combo, tuple) else combo for combo in all_combos], 
                                 columns=['is_combo'] + single_drugs)

    # 初始化所有值为 False
    drug_features.loc[:, :] = False

    # 填充特征
    for combo in all_combos:
        if isinstance(combo, tuple):
            combo_str = ', '.join(combo)
            drug_features.loc[combo_str, 'is_combo'] = True
            for drug in combo:
                drug_features.loc[combo_str, drug] = True
        else:
            drug_features.loc[combo, 'is_combo'] = False
            drug_features.loc[combo, combo] = True

    return drug_features, all_combos

def create_node_features(patient_info, diagnoses, final_data, micro_event, top_k_dict):
    # 患者特征
    patient_features = patient_info[['subject_id', 'gender', 'anchor_age']]
    patient_features['gender'] = (patient_features['gender'] == 'F').astype(int)
    scaler = StandardScaler()
    patient_features['anchor_age'] = scaler.fit_transform(patient_features[['anchor_age']])

    # 诊断特征 - 选择出现频率最高的前k种诊断
    top_k_diagnoses = diagnoses['icd_code'].value_counts().nlargest(top_k_dict['diagnosis']).index
    diagnosis_features = pd.get_dummies(pd.Series(top_k_diagnoses, name='icd_code'))

    # 药物特征
    drug_features, drug_combos = create_drug_features(final_data)

    # 标本特征
    top_k_specimen_types = micro_event['spec_type_desc'].value_counts().nlargest(top_k_dict['specimen']).index
    specimen_features = pd.get_dummies(pd.Series(top_k_specimen_types, name='spec_type_desc'))

    # 微生物特征 - 选择出现频率最高的前k种微生物
    top_k_microorganisms = micro_event['org_name'].value_counts().nlargest(top_k_dict['microorganism']).index
    microorganism_features = pd.get_dummies(pd.Series(top_k_microorganisms, name='org_name'))

    # 测试抗生素特征 - 选择出现频率最高的前k种测试抗生素
    top_k_test_antibiotics = micro_event['ab_name'].value_counts().nlargest(top_k_dict['test_antibiotic']).index
    test_antibiotic_features = pd.get_dummies(pd.Series(top_k_test_antibiotics, name='ab_name'))

    return patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features, top_k_diagnoses, top_k_specimen_types, top_k_microorganisms, top_k_test_antibiotics, drug_combos

def create_unified_node_mapping(patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features):
    node_types = ['p', 'd', 'c', 's', 'm', 't']
    unified_map = {}
    current_id = 0

    for node_type in node_types:
        if node_type == 'p':
            nodes = patient_features['subject_id']
        elif node_type == 'd':
            nodes = diagnosis_features.columns
        elif node_type == 'c':
            nodes = drug_features.index
        elif node_type == 's':
            nodes = specimen_features.columns
        elif node_type == 'm':
            nodes = microorganism_features.columns
        elif node_type == 't':
            nodes = test_antibiotic_features.columns

        for node in nodes:
            unified_map[(node_type, node)] = current_id
            current_id += 1

    return unified_map

def create_edge_list(unified_map, final_data, diagnoses, micro_event):
    edge_list = {
        'ca_p': [], 'p_ca': [],
        'p_d': [], 'd_p': [],
        'p_s': [], 's_p': [],
        's_m': [], 'm_s': [],
        'm_ta': [], 'ta_m': []
    }

    # p-d 和 d-p edges
    for _, row in diagnoses.iterrows():
        if ('p', row['subject_id']) in unified_map and ('d', row['icd_code']) in unified_map:
            p_id = unified_map[('p', row['subject_id'])]
            d_id = unified_map[('d', row['icd_code'])]
            edge_list['p_d'].append((p_id, d_id))
            edge_list['d_p'].append((d_id, p_id))

    # ca-p 和 p-ca edges
    for _, row in final_data.iterrows():
        drug_combo = get_drug_combo(row)
        if ('p', row['subject_id']) in unified_map and ('c', drug_combo) in unified_map:
            p_id = unified_map[('p', row['subject_id'])]
            ca_id = unified_map[('c', drug_combo)]
            edge_list['ca_p'].append((ca_id, p_id))
            edge_list['p_ca'].append((p_id, ca_id))

    # p-s, s-p, s-m, m-s, m-ta, ta-m edges
    for _, row in micro_event.iterrows():
        if all(key in unified_map for key in [('p', row['subject_id']), ('s', row['spec_type_desc']), ('m', row['org_name']), ('t', row['ab_name'])]):
            p_id = unified_map[('p', row['subject_id'])]
            s_id = unified_map[('s', row['spec_type_desc'])]
            m_id = unified_map[('m', row['org_name'])]
            ta_id = unified_map[('t', row['ab_name'])]
            
            edge_list['p_s'].append((p_id, s_id))
            edge_list['s_p'].append((s_id, p_id))
            edge_list['s_m'].append((s_id, m_id))
            edge_list['m_s'].append((m_id, s_id))
            edge_list['m_ta'].append((m_id, ta_id))
            edge_list['ta_m'].append((ta_id, m_id))

    return edge_list

def process_edge_features(final_data, micro_event, unified_map):
    # ca-p 边特征
    ca_p_features = final_data.groupby(['subject_id', 'drug1', 'drug2', 'itemid']).agg({
        'valuenum_mean': ['mean', 'min', 'max'],
        'valuenum_count': 'sum'
    }).reset_index()
    ca_p_features.columns = ['subject_id', 'drug1', 'drug2', 'itemid', 'mean', 'min', 'max', 'count']
    
    # 计算每个 itemid 的缺失率
    missing_rates = ca_p_features.groupby('itemid').apply(lambda x: x[['mean', 'min', 'max', 'count']].isnull().mean())
    valid_itemids = missing_rates[missing_rates.max(axis=1) < 0.2].index
    
    # 只保留缺失率低于 20% 的 itemid
    ca_p_features = ca_p_features[ca_p_features['itemid'].isin(valid_itemids)]

    # 对保留的数据进行插补
    numeric_columns = ['mean', 'min', 'max', 'count']
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    def impute_and_scale(group):
        imputed = imputer.fit_transform(group[numeric_columns])
        scaled = scaler.fit_transform(imputed)
        return pd.DataFrame(scaled, columns=numeric_columns, index=group.index)
    
    ca_p_features[numeric_columns] = ca_p_features.groupby('itemid').apply(impute_and_scale).reset_index(level=0, drop=True)

    ca_p_edge_features = {}
    for _, row in ca_p_features.iterrows():
        drug_combo = get_drug_combo(row)
        drug_combo_str = ', '.join(drug_combo) if isinstance(drug_combo, tuple) else drug_combo
        if ('p', row['subject_id']) in unified_map and ('c', drug_combo_str) in unified_map:
            p_id = unified_map[('p', row['subject_id'])]
            ca_id = unified_map[('c', drug_combo_str)]
            edge_key = (ca_id, p_id)
            if edge_key not in ca_p_edge_features:
                ca_p_edge_features[edge_key] = {}
            ca_p_edge_features[edge_key][row['itemid']] = row[numeric_columns].tolist()

    # 将特征转换为固定长度向量
    for edge_key in ca_p_edge_features:
        feature_vector = []
        for itemid in valid_itemids:
            if itemid in ca_p_edge_features[edge_key]:
                feature_vector.extend(ca_p_edge_features[edge_key][itemid])
            else:
                feature_vector.extend([0] * len(numeric_columns))
        ca_p_edge_features[edge_key] = np.array(feature_vector)
    
    # m-ta 边特征
    m_ta_edge_features = {}
    for _, row in micro_event.iterrows():
        if ('m', row['org_name']) in unified_map and ('t', row['ab_name']) in unified_map:
            m_id = unified_map[('m', row['org_name'])]
            ta_id = unified_map[('t', row['ab_name'])]
            edge_key = (m_id, ta_id)
            m_ta_edge_features[edge_key] = row['interpretation']
    
    return {'ca_p': ca_p_edge_features, 'm_ta': m_ta_edge_features}



def create_adjacency_matrices(final_data, diagnoses, micro_event, type_specific_maps):
    adj_matrices = {
        'p_d': np.zeros((len(type_specific_maps['patient']), len(type_specific_maps['diagnosis'])), dtype=np.float32),
        'p_ca': np.zeros((len(type_specific_maps['patient']), len(type_specific_maps['drug'])), dtype=np.float32),
        'p_s': np.zeros((len(type_specific_maps['patient']), len(type_specific_maps['specimen'])), dtype=np.float32),
        's_m': np.zeros((len(type_specific_maps['specimen']), len(type_specific_maps['microorganism'])), dtype=np.float32),
        'm_ta': np.zeros((len(type_specific_maps['microorganism']), len(type_specific_maps['test_antibiotic'])), dtype=np.float32)
    }

    # p-d edges
    for _, row in diagnoses.iterrows():
        p_id = type_specific_maps['patient'][row['subject_id']]
        d_id = type_specific_maps['diagnosis'][row['icd_code']]
        adj_matrices['p_d'][p_id, d_id] = 1

    # p-ca edges
    for _, row in final_data.iterrows():
        p_id = type_specific_maps['patient'][row['subject_id']]
        ca_id = type_specific_maps['drug'][row['drug_combo']]
        adj_matrices['p_ca'][p_id, ca_id] = 1

    # p-s, s-m, m-ta edges
    for _, row in micro_event.iterrows():
        p_id = type_specific_maps['patient'][row['subject_id']]
        s_id = type_specific_maps['specimen'][row['spec_type_desc']]
        m_id = type_specific_maps['microorganism'][row['org_name']]
        ta_id = type_specific_maps['test_antibiotic'][row['ab_name']]
        adj_matrices['p_s'][p_id, s_id] = 1
        adj_matrices['s_m'][s_id, m_id] = 1
        adj_matrices['m_ta'][m_id, ta_id] = 1

    return adj_matrices

def process_edge_features_1(final_data, micro_event, type_specific_maps):
    scaler = StandardScaler()
    
    # 处理 ca-p 边特征
    ca_p_features = final_data.groupby(['subject_id', 'drug1', 'drug2', 'itemid']).agg({
        'valuenum_mean': 'mean',
        'valuenum_min': 'min',
        'valuenum_max': 'max',
        'valuenum_count': 'sum'
    }).reset_index()
    ca_p_features.columns = ['subject_id', 'drug1', 'drug2', 'itemid', 'mean', 'min', 'max', 'count']
    
    # 标准化数值特征
    numeric_columns = ['mean', 'min', 'max', 'count']
    ca_p_features[numeric_columns] = scaler.fit_transform(ca_p_features[numeric_columns])
    
    ca_p_edge_features = {}
    for _, row in ca_p_features.iterrows():
        p_id = type_specific_maps[('p', row['subject_id'])]
        drug_combo = get_drug_combo(row)
        ca_id = type_specific_maps[('c', drug_combo)]
        edge_key = (p_id, ca_id)
        if edge_key not in ca_p_edge_features:
            ca_p_edge_features[edge_key] = {}
        ca_p_edge_features[edge_key][row['itemid']] = row[numeric_columns].tolist()

    # 处理 m-ta 边特征
    m_ta_edge_features = {}
    for _, row in micro_event.iterrows():
        m_id = type_specific_maps[('m', row['org_name'])]
        ta_id = type_specific_maps[('t', row['ab_name'])]
        edge_key = (m_id, ta_id)
        m_ta_edge_features[edge_key] = row['interpretation']
 
    return {'ca_p': ca_p_edge_features, 'm_ta': m_ta_edge_features}

def preprocess_patient_features(patient_features):
    # 去除重复行
    patient_features = patient_features.drop_duplicates(subset=['subject_id'])
    
    # 将性别列合并为一列，并用0和1表示
    patient_features['gender'] = patient_features['gender_F'].astype(int)
    patient_features = patient_features.drop(columns=['gender_F', 'gender_M'])
    
    # 确保 anchor_age 是数值类型
    patient_features['anchor_age'] = pd.to_numeric(patient_features['anchor_age'])
    
    return patient_features

def preprocess_boolean_features(features):
    # 将布尔值转换为整数（0和1）
    return features.astype(int)

def create_dataset(patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features, edge_index, edge_features, unified_map, edge_counts):
    dataset = {
        'node_features': {
            'p': patient_features.drop('subject_id', axis=1).values,
            'd': diagnosis_features.values,
            'c': drug_features.values,
            's': specimen_features.values,
            'm': microorganism_features.values,
            't': test_antibiotic_features.values
        },
        'feature_names': {
            'p': patient_features.drop('subject_id', axis=1).columns.tolist(),
            'd': diagnosis_features.columns.tolist(),
            'c': drug_features.columns.tolist(),
            's': specimen_features.columns.tolist(),
            'm': microorganism_features.columns.tolist(),
            't': test_antibiotic_features.columns.tolist()
        },
        'edge_index': edge_index,
        'edge_features': edge_features,
        'unified_map': unified_map,
        'edge_counts': edge_counts
    }
    return dataset

def create_ordered_drug_mapping(drug_features, unified_map):
    # 获取药物特征的列名（不包括 'is_combo'）
    drug_columns = [col for col in drug_features.columns if col != 'is_combo']
    
    # 从 unified_map 中提取单药映射
    drug_mapping = {k[1]: v for k, v in unified_map.items() if k[0] == 'c' and ',' not in k[1]}
    
    # 创建有序的单药映射，保持特征矩阵中的列顺序
    ordered_drug_mapping = {drug: drug_mapping[drug] for drug in drug_columns if drug in drug_mapping}
    
    return ordered_drug_mapping

def create_streamlined_dataset(full_dataset):
    streamlined_dataset = {
        'node_features': {},
        'edge_index': full_dataset['edge_index'],
        'edge_features': full_dataset['edge_features'],
        'unified_map': full_dataset['unified_map'],
        'ordered_drug_mapping': {}
    }

    # 处理节点特征
    for node_type, features in full_dataset['node_features'].items():
        if node_type == 'p':
            streamlined_dataset['node_features'][node_type] = features.drop('subject_id', axis=1).values
        elif node_type == 'c':
            streamlined_dataset['node_features'][node_type] = features.values
            streamlined_dataset['ordered_drug_mapping'] = create_ordered_drug_mapping(features, full_dataset['unified_map'])
        else:
            num_nodes = len(set(key[1] for key in full_dataset['unified_map'] if key[0] == node_type))
            streamlined_dataset['node_features'][node_type] = np.eye(num_nodes)

    return streamlined_dataset

def main():
    save_path = '/home/yzq/paper/HetMS-AMRGNN/het/data2/'
    os.makedirs(save_path, exist_ok=True)
    path = '/home/yzq/paper/HetMS-AMRGNN/het/data2'

    final_data = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/final_clinical_data.csv')
    filtered_patients = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/filtered_patients.csv')
    micro_event = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/relevant_micro.csv')

    top_k_dict = {
        'diagnosis': 50,
        'specimen': 30,
        'microorganism': 30,
        'test_antibiotic': 30
    }

    patient_info, diagnoses, final_data, micro_event = process_data(final_data, filtered_patients, micro_event)
    #patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features, top_k_diagnoses, top_k_specimen_types, top_k_microorganisms, top_k_test_antibiotics, all_combos = create_node_features(patient_info, diagnoses, final_data, micro_event, top_k_dict)
    patient_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/patient_features.csv')
    diagnosis_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/diagnosis_features.csv')
    drug_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/drug_features.csv')
    specimen_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/specimen_features.csv')
    microorganism_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/microorganism_features.csv')
    test_antibiotic_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/test_antibiotic_features.csv')

    patient_features = preprocess_patient_features(patient_features)
    diagnosis_features = preprocess_boolean_features(diagnosis_features)
    drug_features = preprocess_boolean_features(drug_features)
    specimen_features = preprocess_boolean_features(specimen_features)
    microorganism_features = preprocess_boolean_features(microorganism_features)
    test_antibiotic_features = preprocess_boolean_features(test_antibiotic_features)

    scaler = StandardScaler()
    patient_features['anchor_age'] = scaler.fit_transform(patient_features[['anchor_age']])

    if os.path.exists(path):
        with open(f'{path}/edge_list.pkl', 'rb') as f:
            edge_list = pickle.load(f)
        with open(f'{path}/unified_map.pkl', 'rb') as f:
            unified_map = pickle.load(f)
    else: 
        unified_map = create_unified_node_mapping(patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features)
        edge_list = create_edge_list(unified_map, final_data, diagnoses, micro_event)
        for edge_type, edges in edge_list.items():
            print(f"Created {len(edges)} {edge_type} edges")
        with open(path, 'wb') as f:
            pickle.dump(edge_list, f)

    edge_features = process_edge_features(final_data, micro_event, unified_map)

    def deduplicate_and_count_edges(edge_list):
        deduplicated_edges = {edge_type: list(set(edges)) for edge_type, edges in edge_list.items()}
        edge_counts = {edge_type: len(edges) for edge_type, edges in deduplicated_edges.items()}
        return deduplicated_edges, edge_counts

    deduplicated_edges, edge_counts = deduplicate_and_count_edges(edge_list)

    for edge_type, count in edge_counts.items():
        print(f"Number of unique {edge_type} edges: {count}")

    ca_p_edges = set(deduplicated_edges['ca_p'])
    existing_edges = set(edge_features['ca_p'].keys())

    feature_length = len(next(iter(edge_features['ca_p'].values())))
    for edge in ca_p_edges - existing_edges:
        edge_features['ca_p'][edge] = np.zeros(feature_length)
    
    # 移除多余的边
    for edge in existing_edges - ca_p_edges:
        del edge_features['ca_p'][edge]

    dataset = {
        'node_features': {
            'p': patient_features,
            'd': diagnosis_features,
            'c': drug_features,
            's': specimen_features,
            'm': microorganism_features,
            't': test_antibiotic_features
        },
        'edge_index': {edge_type: np.array(edges).T for edge_type, edges in deduplicated_edges.items()},
        'edge_features': edge_features,
        'unified_map': unified_map,
        'edge_counts': edge_counts
    }

    # 保存处理后的数据
    with open(os.path.join(save_path, 'full_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    print("Data processing completed. Dataset has been saved.")

    streamlined_dataset = create_streamlined_dataset(dataset)

    with open(os.path.join(save_path,'streamlined_dataset.pkl'), 'wb') as f:
        pickle.dump(streamlined_dataset, f)

    print("Streamlined dataset has been created and saved.")



if __name__ == "__main__":
    main()