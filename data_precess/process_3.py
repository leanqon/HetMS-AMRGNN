import pandas as pd
import pickle
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import re

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class Drug:
    def __init__(self, data):
        self.data = data
        self.process_data()

    def process_data(self):
        self.drug_mapping = self.data['ordered_drug_mapping']
    
root_path='/home/yzq/paper/HetMS-AMRGNN/het'

data = load_data(f'{root_path}/data2/modified_data.pkl')

dataset = Drug(data)

c_items = {k: v for k, v in dataset.drug_mapping.items()}

df = pd.DataFrame(list(c_items.items()), columns=['drug', 'id'])

with gzip.open('/home/yzq/mimiciv/3.0/hosp/prescriptions.csv.gz', 'rt') as f:
    prescriptions = pd.read_csv(f)

data_path = '/home/yzq/MK-GNN/data1/'
ndc2atc = pd.read_csv(f'{data_path}ndc2atc_level4.csv')
drug_atc = pd.read_csv(f'{data_path}drug-atc.csv', sep='\t')
drug_ddi = pd.read_csv(f'{data_path}drug-DDI.csv')

# 创建药品名称到 NDC 的映射
drug_to_ndc = prescriptions.groupby('drug')['ndc'].first().to_dict()

def match_drug(name):
    # 尝试直接匹配
    if name in drug_to_ndc:
        return drug_to_ndc[name]
    # 尝试小写匹配
    lower_name = name.lower()
    if lower_name in drug_to_ndc:
        return drug_to_ndc[lower_name]
    # 尝试首字母大写匹配
    capitalized_name = name.capitalize()
    if capitalized_name in drug_to_ndc:
        return drug_to_ndc[capitalized_name]
    return None

# 匹配 NDC
df['ndc'] = df['drug'].apply(match_drug)

# 格式化 NDC
def format_ndc(ndc):
    if pd.isna(ndc):
        return ndc
    ndc = str(int(ndc))  # 移除前导零并转换为字符串
    if len(ndc) == 11:
        return f"{ndc[:5]}-{ndc[5:9]}-{ndc[9:]}"
    return ndc

df['formatted_ndc'] = df['ndc'].apply(format_ndc)

# 匹配 ATC4 代码
ndc_to_atc4 = ndc2atc.set_index('NDC')['ATC4'].drop_duplicates()
df['atc4'] = df['formatted_ndc'].map(ndc_to_atc4)

# 匹配 CID
atc4_to_cid = drug_atc.set_index('B')['A'].str.strip('CID').drop_duplicates()
df['cid'] = df['atc4'].map(atc4_to_cid)

# 移除缺失 CID 的行
df_with_cid = df.dropna(subset=['cid'])
df_with_cid['cid'] = df_with_cid['cid'].astype(int)

# 创建 CID 到索引的映射
cid_to_index = {cid: i for i, cid in enumerate(df_with_cid['cid'])}

# 创建 DDI 邻接矩阵
n = len(df)
ddi_matrix = np.zeros((n, n), dtype=int)

for _, row in tqdm(drug_ddi.iterrows(), total=len(drug_ddi)):
    cid1, cid2 = int(row['STITCH 1'].strip('CID')), int(row['STITCH 2'].strip('CID'))
    if cid1 in cid_to_index and cid2 in cid_to_index:
        i, j = cid_to_index[cid1], cid_to_index[cid2]
        ddi_matrix[i, j] = ddi_matrix[j, i] = 1

# 创建带有药品 ID 作为索引和列的 DataFrame
ddi_df = pd.DataFrame(ddi_matrix, index=df['id'], columns=df['id'])

# 保存结果
ddi_df.to_csv('ddi_adjacency_matrix.csv')
