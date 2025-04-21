import torch
import numpy as np
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import json
from datetime import datetime
import matplotlib.pyplot as plt
from het10 import prepare_histories

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    else:
        return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)
    
def get_node_info(node_id, node_type, data):
    unified_map = data['unified_map']
    node_features = data['node_features'][node_type]
    type_ranges = {
        'p': (0, data['node_features']['p'].shape[0]),
        'd': (data['node_features']['p'].shape[0], data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0]),
        'c': (data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0], 
              data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0]),
        's': (data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0],
              data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0] + data['node_features']['s'].shape[0]),
        'm': (data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0] + data['node_features']['s'].shape[0],
              data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0] + data['node_features']['s'].shape[0] + data['node_features']['m'].shape[0]),
        't': (data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0] + data['node_features']['s'].shape[0] + data['node_features']['m'].shape[0],
              data['node_features']['p'].shape[0] + data['node_features']['d'].shape[0] + data['node_features']['c'].shape[0] + data['node_features']['s'].shape[0] + data['node_features']['m'].shape[0] + data['node_features']['t'].shape[0])
    }

    if node_id < type_ranges[node_type][0] or node_id >= type_ranges[node_type][1]:
        logging.warning(f"Node ID {node_id} is out of range for type {node_type}")
        return None, None

    original_id = next((key for key, value in unified_map.items() if value == node_id), None)
    if original_id is None:
        logging.warning(f"No matching entry found for node_id {node_id} of type {node_type}")
        return None, None

    try:
        features = node_features[node_id - type_ranges[node_type][0]]
    except IndexError:
        logging.warning(f"No features found for node_id {node_id} of type {node_type}")
        features = None

    return original_id, features

def get_edge_info(src_id, dst_id, edge_type, data):
    edge_index = data['edge_index'][edge_type]
    edge_features = data.get('edge_features', {}).get(edge_type)
    edge_idx = np.where((edge_index[0] == src_id) & (edge_index[1] == dst_id))[0]
    
    if len(edge_idx) > 0 and edge_features is not None:
        return edge_features[edge_idx[0]]
    return None

def case_study(model, patient_id, dataset, device, args, original_data, icd_mapping, patient_history, index_diagnosis_map):
    model.eval()
    features = dataset.get_features(device)
    adj_dict = dataset.get_type_specific_adj()
    node_id_ranges = dataset.get_node_id_ranges()

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
        index = torch.tensor([patient_id]).to(device)
        history = dataset.patient_history.get(dataset.get_subject_id(patient_id), [])
        history_tensor = prepare_histories([history], device, max_len=100, num_diagnoses=dataset.get_num_diagnoses())

        if args.model == 'Het-UMSGNN':
            output = model(features, adj_matrices, adj_dict, dataset, index, history=history_tensor)
        elif args.model in ['GCN', 'GAT']:
            output_dict = model(features, adj_matrices)
            output = output_dict['p'][index]
        elif args.model == 'HAN':
            output_dict = model(features, adj_dict)
            output = output_dict['p'][index]
        elif args.model == 'RETAIN':
            output = model(features, history_tensor, index)['p']
        elif args.model in 'GAMENet':
            output_dict = model(features, history_tensor, index)
            output = output_dict['p']
        elif args.model == 'VITA':
            output_dict = model(features, adj_dict, history_tensor, index)
            return output_dict['p']
        elif args.model == 'KGDNet':
            output_dict = model(features, adj_dict, ddi_edge_index, index)
            return output_dict['p']
        elif args.model == 'MoleRec':
            output_dict = model(features, adj_dict, history_tensor, index)
            return output_dict['p']
        elif args.model in ['SafeDrug']:
            output_dict = model(features, history_tensor, ddi_edge_index, index)
            output = output_dict['p']
        elif args.model in ['LEAP']:
            output_dict = model(features, history_tensor, ddi_edge_index, index)
            output = output_dict['p']
        elif args.model in ['Deepwalk', 'Node2Vec']:
            output_dict = model(features, history_tensor, index)
            output = output_dict['p']

        output = torch.sigmoid(output).squeeze()

        k = 10
        top_k_values, top_k_indices = torch.topk(output, k)

        c_offset = dataset.get_node_type_offset('c')
        p_offset = dataset.get_node_type_offset('p')
        d_offset = dataset.get_node_type_offset('d')
        s_offset = dataset.get_node_type_offset('s')
        c_offset = dataset.get_node_type_offset('c')

        actual_prescription = dataset.edge_index['p_c'][1][dataset.edge_index['p_c'][0] == patient_id].tolist()
        global_patient_id = patient_id + p_offset
        patient_original_id, patient_features = get_node_info(global_patient_id, 'p', original_data)

        history_diagnoses = []
        if patient_original_id[1] in patient_history:
            for visit in patient_history[patient_original_id[1]]:
                visit_diagnoses = []
                for diagnosis_index in visit:
                    icd_code = index_diagnosis_map.get(diagnosis_index, "Unknown")
                    diagnosis_label = icd_mapping.get(icd_code, "Unknown")
                    visit_diagnoses.append((icd_code, diagnosis_label))
                history_diagnoses.append(visit_diagnoses)

        diagnoses = dataset.edge_index['p_d'][1][dataset.edge_index['p_d'][0] == patient_id].tolist()
        diagnosis_info = []
        for d in diagnoses:
            d_info = get_node_info(d, 'd', original_data)
            if d_info[0] is not None:
                icd_code = d_info[0][1]  
                diagnosis_label = icd_mapping.get(icd_code, "Unknown")
                diagnosis_info.append((icd_code, diagnosis_label, d_info[1]))

        specimens = dataset.edge_index['p_s'][1][dataset.edge_index['p_s'][0] == patient_id].tolist()
        specimen_info = []
        for s in specimens:
            s_info = get_node_info(s, 's', original_data)
            if s_info[0] is not None:
                microorganisms = original_data['edge_index']['s_m'][1][original_data['edge_index']['s_m'][0] == s].tolist()
                m_info = []
                for m in microorganisms:
                    m_data = get_node_info(m, 'm', original_data)
                    if m_data[0] is not None:
                        tests = original_data['edge_index']['m_t'][1][original_data['edge_index']['m_t'][0] == m].tolist()
                        t_info = []
                        for t in tests:
                            t_data = get_node_info(t, 't', original_data)
                            if t_data[0] is not None:
                                result = original_data['edge_features']['m_t'].get((m, t), 'Unknown')
                                t_info.append((t_data[0], result))  
                        m_info.append((m_data[0], t_info))
                specimen_info.append((s_info[0], m_info))

        recommended_drugs = []
        for i, score in zip(top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()):
            global_drug_id = i + c_offset 
            drug_info = get_node_info(global_drug_id, 'c', original_data)
            if drug_info[0] is not None:
                recommended_drugs.append((drug_info[0], float(score)))
            else:
                logging.warning(f"Skipping recommended drug with id {global_drug_id} due to missing information")

        actual_drugs = []
        for d in actual_prescription:
            global_drug_id = d 
            drug_info = get_node_info(global_drug_id, 'c', original_data)
            if drug_info[0] is not None:
                actual_drugs.append(drug_info[0])
            else:
                logging.warning(f"Skipping actual drug with id {global_drug_id} due to missing information")
        
        return {
            'patient': str(patient_original_id),
            'history_diagnoses': history_diagnoses,
            'current_diagnoses': [(str(icd_code), str(diagnosis_label)) for icd_code, diagnosis_label, _ in diagnosis_info],
            'specimens': [(str(s), [(str(m), [(str(t), str(r)) for t, r in tests]) for m, tests in microorganisms]) for s, microorganisms in specimen_info],
            'recommended_drugs': recommended_drugs,
            'actual_drugs': actual_drugs
        }    
        
def run_case_study(args, model, dataset, device, original_data, icd_mapping, patient_history, index_diagnosis_map):
    logging.info("Starting case study...")

    if args.case_study_patient_id:
        patient_id = int(args.case_study_patient_id)
    else:
        patient_id = random.choice(range(dataset.node_type_count['p']))

    results = case_study(model, patient_id, dataset, device, args, original_data, icd_mapping, patient_history, index_diagnosis_map)

    log_output = []

    log_output.append(f"Case Study for Patient {results['patient']}:")

    log_output.append("\nHistorical Diagnoses:")
    for i, visit in enumerate(results['history_diagnoses'], 1):
        log_output.append(f"  Visit {i}:")
        for icd_code, diagnosis_label in visit:
            log_output.append(f"    {icd_code} - {diagnosis_label}")

    log_output.append("\nCurrent Diagnoses:")
    for icd_code, diagnosis_label in results['current_diagnoses']:
        log_output.append(f"  {icd_code} - {diagnosis_label}")

    log_output.append("\nSpecimens:")
    for specimen, microorganisms in results['specimens']:
        log_output.append(f"  Specimen: {specimen}")
        for microorganism, tests in microorganisms:
            log_output.append(f"    Microorganism: {microorganism}")
            for test, result in tests:
                log_output.append(f"      Test: {test}, Result: {result}")

    log_output.append("\nTop 10 Recommended Antibiotics:")
    for drug, score in results['recommended_drugs']:
        log_output.append(f"  {drug}: {score:.4f}")

    log_output.append("\nActual Prescription:")
    for drug in results['actual_drugs']:
        log_output.append(f"  {drug}")

    recommended_set = set(drug for drug, _ in results['recommended_drugs'])
    actual_set = set(results['actual_drugs'])
    correct_recommendations = recommended_set & actual_set
    
    log_output.append("\nAnalysis:")
    log_output.append(f"Correctly recommended drugs: {', '.join(str(drug) for drug in correct_recommendations)}")
    log_output.append(f"Missed drugs: {', '.join(str(drug) for drug in actual_set - recommended_set)}")
    log_output.append(f"Extra recommended drugs: {', '.join(str(drug) for drug in recommended_set - actual_set)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.output_dir}/case_study_log_{args.case_study_patient_id}_{timestamp}.txt"
    with open(log_filename, 'w') as f:
        for line in log_output:
            f.write(f"INFO:root:{line}\n")
            logging.info(line)

    logging.info(f"Case study log saved to {log_filename}")

def analyze_case_study_results(recommendations, actual_prescription):
    recommended_set = set(drug[0] for drug, _ in recommendations)
    actual_set = set(drug[0] for drug in actual_prescription)
    
    correct_recommendations = recommended_set & actual_set
    precision = len(correct_recommendations) / len(recommended_set) if recommended_set else 0
    recall = len(correct_recommendations) / len(actual_set) if actual_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nAnalysis:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Correctly recommended drugs: {', '.join(str(drug) for drug in correct_recommendations)}")
    print(f"Missed drugs: {', '.join(str(drug) for drug in actual_set - recommended_set)}")
    print(f"Extra recommended drugs: {', '.join(str(drug) for drug in recommended_set - actual_set)}")