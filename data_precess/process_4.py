import pickle
import pandas as pd
import numpy as np
import gzip
from collections import Counter

# Loading the data from the file
with open('/home/yzq/paper/HetMS-AMRGNN/het/data2/unified_map.pkl', 'rb') as f:
    unified_map = pickle.load(f)

# 获取需要处理的 subject_id 列表
subject_ids = [id for (type, id), _ in unified_map.items() if type == 'p']

# 从 diagnosis_features.csv 获取已有的诊断特征
diagnosis_features = pd.read_csv('/home/yzq/paper/HetMS-AMRGNN/het/data/diagnosis_features.csv')
existing_diagnoses = diagnosis_features.columns.tolist()
diagnosis_features = diagnosis_features.astype(int)  # 将 True/False 转换为 1/0

# 创建诊断到全局索引的映射
diagnosis_to_global_index = {diag: unified_map[('d', diag)] for diag in existing_diagnoses if ('d', diag) in unified_map}

# 创建诊断到索引的映射
diagnosis_to_index = {diag: idx for idx, diag in enumerate(existing_diagnoses)}

# 读取 admissions 数据以获取住院时间信息
with gzip.open('/home/yzq/mimiciv/3.0/hosp/admissions.csv.gz', 'rt') as f:
    admissions = pd.read_csv(f)
admissions = admissions[admissions['subject_id'].isin(subject_ids)]
admissions = admissions.sort_values(['subject_id', 'admittime'])

# 读取历史诊断数据
with gzip.open('/home/yzq/mimiciv/3.0/hosp/diagnoses_icd.csv.gz', 'rt') as f:
    historical_diagnoses = pd.read_csv(f)

# 合并 admissions 和 diagnoses 数据
historical_diagnoses = historical_diagnoses.merge(admissions[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id'])

# 过滤和排序历史诊断
historical_diagnoses = historical_diagnoses[historical_diagnoses['subject_id'].isin(subject_ids)]
historical_diagnoses = historical_diagnoses.sort_values(['subject_id', 'admittime'])

# 获取每个患者的最后一次住院 ID
last_admissions = admissions.groupby('subject_id').last()['hadm_id'].to_dict()

# 处理每个患者的历史诊断
patient_history = {subject_id: [] for subject_id in subject_ids}
all_diagnoses = []

for subject_id, group in historical_diagnoses.groupby('subject_id'):
    last_hadm_id = last_admissions[subject_id]
    patient_diagnoses = []
    for _, row in group.iterrows():
        if row['hadm_id'] != last_hadm_id:  # 排除最后一次住院
            icd_code = row['icd_code']
            patient_diagnoses.append(icd_code)
            all_diagnoses.append(icd_code)
    patient_history[subject_id] = patient_diagnoses

# 计算诊断频次
diagnosis_freq = Counter(all_diagnoses)

# 选择高频诊断（例如，选择前50个高频诊断）
top_diagnoses = [diag for diag, _ in diagnosis_freq.most_common(50) if diag not in existing_diagnoses]

# 更新诊断到索引的映射
for diag in top_diagnoses:
    if diag not in diagnosis_to_index:
        diagnosis_to_index[diag] = len(diagnosis_to_index)

# 创建患者历史诊断序列
patient_history_sequences = {}
for subject_id, diagnoses in patient_history.items():
    sequence = [diagnosis_to_index[d] for d in diagnoses if d in diagnosis_to_index]
    patient_history_sequences[subject_id] = sequence

# 保存患者历史诊断序列
import pickle
with open('patient_history_sequences.pkl', 'wb') as f:
    pickle.dump(patient_history_sequences, f)

# 保存诊断索引映射
with open('diagnosis_index_map.pkl', 'wb') as f:
    pickle.dump(diagnosis_to_index, f)