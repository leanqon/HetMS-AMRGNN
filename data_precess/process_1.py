import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import pickle
# 设置显示的最大列数为 None，这样就能显示所有的列
pd.set_option('display.max_columns', None)

def process_data(final_data, filtered_patients, micro_event):
    # 处理 filtered_patients 数据
    patient_info = filtered_patients[['subject_id', 'gender', 'anchor_age', 'icd_code']].drop_duplicates()
    
    # 处理诊断数据
    diagnoses = patient_info['icd_code'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('icd_code')
    diagnoses = diagnoses.join(patient_info[['subject_id']])
    diagnoses = diagnoses.drop_duplicates()
    
    # 处理 final_data
    final_data['start'] = pd.to_datetime(final_data['start'])
    final_data['end'] = pd.to_datetime(final_data['end'])
    
    # 将 itemid 转换为整数类型，忽略 NaN 值
    final_data['itemid'] = pd.to_numeric(final_data['itemid'], errors='coerce').astype('Int64')

    # 分析每种 itemid 的缺失情况
    itemid_missing = final_data.groupby('itemid').apply(lambda x: x['valuenum_mean'].isnull().sum() / len(x) * 100)
    print("Percentage of missing values for each itemid:")
    print(itemid_missing)
    
    return patient_info, diagnoses, final_data, micro_event

def standardize_drug_name(drug):
    return drug.strip().lower() if isinstance(drug, str) else drug

def get_drug_combo(row):
    drug1 = standardize_drug_name(row['drug1'])
    drug2 = standardize_drug_name(row['drug2'])
    if pd.isna(drug2):
        return drug1  # 单药情况
    return tuple(sorted([drug1, drug2]))  # 组合用药情况

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
    patient_features = pd.get_dummies(patient_features, columns=['gender'])
    
    # 诊断特征 - 选择出现频率最高的前k种诊断
    top_k_diagnoses = diagnoses['icd_code'].value_counts().nlargest(top_k_dict['diagnosis']).index
    diagnosis_features = pd.get_dummies(pd.Series(top_k_diagnoses, name='icd_code'))
    
    # 创建药物特征
    drug_features, drug_combos = create_drug_features(final_data)

    # 标本特征
    top_k_sepimen_types = micro_event['spec_type_desc'].value_counts().nlargest(top_k_dict['specimen']).index
    specimen_features = pd.get_dummies(pd.Series(top_k_sepimen_types, name='spec_type_desc'))
    #specimen_features = pd.get_dummies(micro_event['spec_type_desc'].unique())
    
    # 微生物特征 - 选择出现频率最高的前k种微生物
    top_k_microorganisms = micro_event['org_name'].value_counts().nlargest(top_k_dict['microorganism']).index
    microorganism_features = pd.get_dummies(pd.Series(top_k_microorganisms, name='org_name'))
    
    # 测试抗生素特征 - 选择出现频率最高的前k种测试抗生素
    top_k_test_antibiotics = micro_event['ab_name'].value_counts().nlargest(top_k_dict['test_antibiotic']).index
    test_antibiotic_features = pd.get_dummies(pd.Series(top_k_test_antibiotics, name='ab_name'))
    
    return patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features, top_k_diagnoses, top_k_microorganisms, top_k_test_antibiotics, drug_combos

def create_adjacency_matrices(final_data, diagnoses, micro_event, top_k_diagnoses, top_k_microorganisms, top_k_test_antibiotics, drug_combos):
    top_specimen_types = micro_event['spec_type_desc'].unique()

    # 创建映射字典
    patient_map = {id: i for i, id in enumerate(final_data['subject_id'].unique())}
    drug_map = {combo: i for i, combo in enumerate(drug_combos)}
    diagnosis_map = {code: i for i, code in enumerate(top_k_diagnoses)}
    specimen_map = {type: i for i, type in enumerate(top_specimen_types)}
    microorganism_map = {org: i for i, org in enumerate(top_k_microorganisms)}
    test_antibiotic_map = {ab: i for i, ab in enumerate(top_k_test_antibiotics)}

    # 创建邻接矩阵
    ca_p = np.zeros((len(drug_map), len(patient_map)))
    d_p = np.zeros((len(diagnosis_map), len(patient_map)))
    ta_m = np.zeros((len(test_antibiotic_map), len(microorganism_map)))
    m_s = np.zeros((len(microorganism_map), len(specimen_map)))
    s_p = np.zeros((len(specimen_map), len(patient_map)))

    # 填充 ca-p 邻接矩阵
    for _, row in final_data.iterrows():
        drug_combo = get_drug_combo(row)
        if drug_combo in drug_map:
            drug_idx = drug_map[drug_combo]
            patient_idx = patient_map[row['subject_id']]
            ca_p[drug_idx, patient_idx] = 1

    # 填充 d-p 邻接矩阵
    for _, row in diagnoses.iterrows():
        if row['icd_code'] in diagnosis_map:
            diagnosis_idx = diagnosis_map[row['icd_code']]
            patient_idx = patient_map[row['subject_id']]
            d_p[diagnosis_idx, patient_idx] = 1

    # 填充 ta-m, m-s, s-p 邻接矩阵
    for _, row in micro_event.iterrows():
        if row['ab_name'] in test_antibiotic_map and row['org_name'] in microorganism_map:
            ta_idx = test_antibiotic_map[row['ab_name']]
            m_idx = microorganism_map[row['org_name']]
            ta_m[ta_idx, m_idx] = 1

        if row['org_name'] in microorganism_map and row['spec_type_desc'] in specimen_map:
            m_idx = microorganism_map[row['org_name']]
            s_idx = specimen_map[row['spec_type_desc']]
            m_s[m_idx, s_idx] = 1

        if row['spec_type_desc'] in specimen_map and row['subject_id'] in patient_map:
            s_idx = specimen_map[row['spec_type_desc']]
            p_idx = patient_map[row['subject_id']]
            s_p[s_idx, p_idx] = 1

    return ca_p, d_p, ta_m, m_s, s_p, patient_map, drug_map, diagnosis_map, specimen_map, microorganism_map, test_antibiotic_map


def create_edge_data(final_data, diagnoses, micro_event):
    # 患者-诊断边 (p-d)
    patient_diagnosis_edges = diagnoses[['subject_id', 'icd_code']]
    
    # 患者-临床抗生素边 (p-ca)
    patient_drug_edges = final_data[['subject_id', 'hadm_id', 'drug1', 'drug2', 'start', 'end', 'is_combo']]
    
    # 患者-标本边 (p-s)
    patient_specimen_edges = micro_event[['subject_id', 'hadm_id', 'spec_type_desc']].drop_duplicates()
    
    # 标本-微生物边 (s-m)
    specimen_microorganism_edges = micro_event[['spec_type_desc', 'org_name']].drop_duplicates()
    
    # 微生物-测试抗生素边 (m-ta)
    microorganism_test_antibiotic_edges = micro_event[['org_name', 'ab_name', 'interpretation']]
    
    # 临床抗生素-临床抗生素边 (ca-ca)
    drug_drug_edges = final_data[final_data['is_combo'] == True][['drug1', 'drug2']]
    
    return patient_diagnosis_edges, patient_drug_edges, patient_specimen_edges, specimen_microorganism_edges, microorganism_test_antibiotic_edges, drug_drug_edges

def process_edge_features(final_data):
    # 将 itemid 转换为整数类型，忽略 NaN 值
    final_data['itemid'] = pd.to_numeric(final_data['itemid'], errors='coerce').astype('Int64')

    # 分析每种 itemid 的缺失情况
    itemid_missing = final_data.groupby('itemid').apply(lambda x: x['valuenum_mean'].isnull().sum() / len(x) * 100)
    print("Percentage of missing values for each itemid:")
    print(itemid_missing)
    
    # 删除缺失严重的 itemid（例如，缺失率超过 20%）
    itemids_to_keep = itemid_missing[itemid_missing < 20].index
    final_data = final_data[final_data['itemid'].isin(itemids_to_keep)]
    
    # 对缺失值进行插补
    edge_features = final_data[['subject_id', 'hadm_id', 'drug1', 'itemid', 'valuenum_mean', 'valuenum_min', 'valuenum_max', 'valuenum_count']]
    
    numeric_features = ['valuenum_mean', 'valuenum_min', 'valuenum_max', 'valuenum_count']
    
    # 对每个 itemid 分别进行插补
    for itemid in edge_features['itemid'].unique():
        mask = edge_features['itemid'] == itemid
        imputer = SimpleImputer(strategy='mean')
        edge_features.loc[mask, numeric_features] = imputer.fit_transform(edge_features.loc[mask, numeric_features])

    # 将 edge_features 转换为宽格式
    edge_features_wide = edge_features.pivot_table(
        index=['subject_id', 'hadm_id', 'drug1'],
        columns='itemid',
        values=numeric_features
    )
    edge_features_wide.columns = [f'{col[0]}_{col[1]}' for col in edge_features_wide.columns]
    edge_features_wide = edge_features_wide.reset_index()
    
    # 对所有数值型变量进行标准化
    scaler = StandardScaler()
    numeric_columns = edge_features_wide.select_dtypes(include=[np.number]).columns
    edge_features_wide[numeric_columns] = scaler.fit_transform(edge_features_wide[numeric_columns])
    
    return edge_features_wide


def main():     
    final_data = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/final_clinical_data.csv')
    filtered_patients = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/filtered_patients.csv')
    micro_event = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/relevant_micro.csv')

    # 处理诊断数据
    diagnoses = filtered_patients['icd_code'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('icd_code')
    diagnoses = diagnoses.join(filtered_patients[['subject_id']])
    diagnoses = diagnoses.drop_duplicates()

    # 定义 top k 的字典
    top_k_dict = {
        'diagnosis': 50,
        'specimen': 30,
        'microorganism': 30,
        'test_antibiotic': 30
    }
    
    # 创建节点特征
    patient_features, diagnosis_features, drug_features, specimen_features, microorganism_features, test_antibiotic_features, top_k_diagnoses, top_k_microorganisms, top_k_test_antibiotics, drug_combos = create_node_features(filtered_patients, diagnoses, final_data, micro_event, top_k_dict)

    # 创建邻接矩阵
    ca_p, d_p, ta_m, m_s, s_p, patient_map, drug_map, diagnosis_map, specimen_map, microorganism_map, test_antibiotic_map = create_adjacency_matrices(final_data, diagnoses, micro_event, top_k_diagnoses, top_k_microorganisms, top_k_test_antibiotics, drug_combos)

    # 创建边数据
    patient_diagnosis_edges, patient_drug_edges, patient_specimen_edges, specimen_microorganism_edges, microorganism_test_antibiotic_edges, drug_drug_edges = create_edge_data(final_data, diagnoses, micro_event)
    
    # 处理边特征
    edge_features = process_edge_features(final_data)

    # 保存处理后的数据
    save_path = '/home/yzq/paper/HetMS-AMRGNN/het/data2/'
    os.makedirs(save_path, exist_ok=True)

    # 保存邻接矩阵
    np.save(os.path.join(save_path, 'ca_p.npy'), ca_p)
    np.save(os.path.join(save_path, 'd_p.npy'), d_p)
    np.save(os.path.join(save_path, 'ta_m.npy'), ta_m)
    np.save(os.path.join(save_path, 'm_s.npy'), m_s)
    np.save(os.path.join(save_path, 's_p.npy'), s_p)

    # 保存映射字典
    maps = {
        'patient_map': patient_map,
        'drug_map': drug_map,
        'diagnosis_map': diagnosis_map,
        'specimen_map': specimen_map,
        'microorganism_map': microorganism_map,
        'test_antibiotic_map': test_antibiotic_map
    }

    with open(os.path.join(save_path, 'maps.pickle'), 'wb') as f:
        pickle.dump(maps, f)

    patient_features.to_csv(f'{save_path}patient_features.csv', index=False)
    diagnosis_features.to_csv(f'{save_path}diagnosis_features.csv', index=False)
    drug_features.to_csv(f'{save_path}drug_features.csv', index=False)
    specimen_features.to_csv(f'{save_path}specimen_features.csv', index=False)
    microorganism_features.to_csv(f'{save_path}microorganism_features.csv', index=False)
    test_antibiotic_features.to_csv(f'{save_path}test_antibiotic_features.csv', index=False)
    
    patient_diagnosis_edges.to_csv(f'{save_path}patient_diagnosis_edges.csv', index=False)
    patient_drug_edges.to_csv(f'{save_path}patient_drug_edges.csv', index=False)
    patient_specimen_edges.to_csv(f'{save_path}patient_specimen_edges.csv', index=False)
    specimen_microorganism_edges.to_csv(f'{save_path}specimen_microorganism_edges.csv', index=False)
    microorganism_test_antibiotic_edges.to_csv(f'{save_path}microorganism_test_antibiotic_edges.csv', index=False)
    drug_drug_edges.to_csv(f'{save_path}drug_drug_edges.csv', index=False)
    
    edge_features.to_csv(f'{save_path}edge_features.csv', index=False)
    
    print("Data processing completed. All files have been saved.")

if __name__ == "__main__":
    main()
