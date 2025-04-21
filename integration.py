import os
import numpy as np
import torch
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from explainer import HetMSAMRGNNExplainer

def add_explainability_args(parser):
    exp_group = parser.add_argument_group('Explainability')
    exp_group.add_argument('--generate_explanations', action='store_true',
                          help='Generate explanations for model predictions')
    exp_group.add_argument('--explanation_mode', type=str, choices=['train', 'test', 'case_study', 'paper_figures'],
                          default='case_study', help='When to generate explanations')
    exp_group.add_argument('--explanation_output_dir', type=str, default=None,
                          help='Directory to save explanations (defaults to output_dir/explanations)')
    exp_group.add_argument('--explain_samples', type=str, default=None,
                          help='Comma-separated list of patient IDs to explain')
    exp_group.add_argument('--num_explanations', type=int, default=1,
                          help='Number of random samples to explain if not specified')
    exp_group.add_argument('--paper_figures', action='store_true',
                          help='Generate high-quality figures for the paper')
    return parser

def select_explanation_samples(dataset, mask, num_samples=5, specified_ids=None):
    if specified_ids:
        patient_indices = []
        for pid in specified_ids:
            for idx in mask:
                if str(dataset.get_subject_id(idx)) == pid:
                    patient_indices.append(idx)
                    break

        if not patient_indices:
            logging.warning(f"No matches found for specified patient IDs: {specified_ids}. Using random selection.")
            patient_indices = np.random.choice(mask, min(num_samples, len(mask)), replace=False).tolist()
    else:
        patient_indices = np.random.choice(mask, min(num_samples, len(mask)), replace=False).tolist()
    
    return patient_indices

def select_diverse_samples(dataset, mask, model, features, adj_matrices, adj_dict, device, num_samples=5):
    """
    Select diverse samples for explanation based on model predictions and ground truth
    """
    model.eval()
    batch_indices = torch.tensor(list(mask), device=device)
    
    processed_adj_dict = {}
    for key, value in adj_dict.items():
        if isinstance(value, tuple):
            processed_adj_dict[key] = value[0].to(device)
        else:
            processed_adj_dict[key] = value.to(device)

    with torch.no_grad():
        scores, _ = model(features, adj_matrices, processed_adj_dict, dataset, batch_indices)
    
    scores = torch.sigmoid(scores).cpu().numpy()

    categories = {
        'high_confidence_correct': [],   
        'high_confidence_wrong': [],     
        'low_confidence_correct': [],   
        'combo_therapy': [],          
        'complex_case': []           
    }

    node_id_ranges = dataset.get_node_id_ranges()
    for i, patient_idx in enumerate(mask):
        if i >= len(scores):
            continue
        ground_truth = torch.zeros(dataset.node_type_count.get('c', 0))
        if hasattr(dataset, 'edge_index') and 'p_c' in dataset.edge_index:
            positive_drugs = dataset.edge_index['p_c'][1][dataset.edge_index['p_c'][0] == patient_idx]
            positive_drugs_local = positive_drugs - node_id_ranges['c'][0]
            ground_truth[positive_drugs_local] = 1
        
        ground_truth = ground_truth.numpy()
        patient_scores = scores[i]
        predictions = (patient_scores > 0.5).astype(float)
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        false_positives = np.sum((predictions == 1) & (ground_truth == 0))
        false_negatives = np.sum((predictions == 0) & (ground_truth == 1)) 
        precision = true_positives / max(true_positives + false_positives, 1e-10)
        recall = true_positives / max(true_positives + false_negatives, 1e-10)
        top_drugs = np.argsort(-patient_scores)[:min(5, len(patient_scores))]
        avg_confidence = np.mean(patient_scores[top_drugs])
        num_positive = np.sum(ground_truth)
        is_combo = num_positive > 1
        is_complex = num_positive > 3

        if avg_confidence > 0.8 and precision > 0.7 and recall > 0.7:
            categories['high_confidence_correct'].append(patient_idx)
        elif avg_confidence > 0.8 and (precision < 0.5 or recall < 0.5):
            categories['high_confidence_wrong'].append(patient_idx)
        elif avg_confidence < 0.6 and precision > 0.7 and recall > 0.7:
            categories['low_confidence_correct'].append(patient_idx)
        
        if is_combo:
            categories['combo_therapy'].append(patient_idx)
        
        if is_complex:
            categories['complex_case'].append(patient_idx)

    selected_indices = []
    for category, indices in categories.items():
        if indices and len(selected_indices) < num_samples:
            selected_index = np.random.choice(indices)
            if selected_index not in selected_indices:
                selected_indices.append(selected_index)
    remaining = num_samples - len(selected_indices)
    if remaining > 0:
        mask_list = list(mask)
        eligible = [idx for idx in mask_list if idx not in selected_indices]
        if eligible:
            additional = np.random.choice(eligible, min(remaining, len(eligible)), replace=False)
            selected_indices.extend(additional)
    
    return selected_indices

def generate_explanations(args, model, dataset, device, mask, features, adj_matrices, adj_dict):
    """
    Generate explanations for the specified samples
    """
    explainer = HetMSAMRGNNExplainer(model, dataset, device)
    output_dir = args.explanation_output_dir
    if output_dir is None:
        output_dir = os.path.join(args.output_dir, 'explanations', 
                                 datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_adj_matrices = {}
    for key, value in adj_matrices.items():
        processed_adj_matrices[key] = value.to(device)
    
    processed_adj_dict = {}
    for key, value in adj_dict.items():
        if isinstance(value, tuple):
            processed_adj_dict[key] = value[0].to(device)
        else:
            processed_adj_dict[key] = value.to(device)

    processed_features = {}
    for key, value in features.items():
        processed_features[key] = value.to(device)
    full_test_indices = torch.tensor(list(mask), device=device)

    specified_ids = None
    if args.explain_samples:
        specified_ids = [id.strip() for id in args.explain_samples.split(',')]
    
    if args.explanation_mode == 'paper_figures' or args.paper_figures:
        patient_indices = select_diverse_samples(
            dataset, mask, model, features, processed_adj_matrices, processed_adj_dict,
            device, num_samples=args.num_explanations
        )
    else:
        patient_indices = select_explanation_samples(
            dataset, mask, num_samples=args.num_explanations, specified_ids=specified_ids
        )

    case_studies = explainer.run_case_study(
        patient_indices, processed_features, processed_adj_matrices, processed_adj_dict, output_dir
    )

    if args.explanation_mode == 'paper_figures' or args.paper_figures:
        paper_figures_dir = os.path.join(output_dir, 'paper_figures')
        os.makedirs(paper_figures_dir, exist_ok=True)
        generate_paper_figures(
            explainer, model, dataset, device, patient_indices,
            processed_features, processed_adj_matrices, processed_adj_dict, 
            paper_figures_dir, full_test_indices
        )
    
    return case_studies

def generate_paper_figures(explainer, model, dataset, device, patient_indices, 
                          features, adj_matrices, adj_dict, output_dir, full_test_indices=None):
    """
    Generate high-quality figures for the academic paper with neutral labeling
    """
    test_indices = full_test_indices
    if test_indices is None:
        test_indices = torch.arange(min(1000, dataset.node_type_count['p']), device=device)
    elif len(test_indices) > 5000: 
        indices = torch.randperm(len(test_indices))[:5000]
        test_indices = test_indices[indices]

    num_views = model.num_views if hasattr(model, 'num_views') else 3
    print(f"Model has {num_views} views")

    fig_path = os.path.join(output_dir, 'fig1_multi_view_embeddings.png')
    explainer.visualize_multi_view_embeddings(features, adj_matrices, adj_dict, fig_path, test_indices)
    sample_indices = test_indices
    if len(test_indices) > 500:
        indices = torch.randperm(len(test_indices))[:500]
        sample_indices = test_indices[indices]
        
    fig_path = os.path.join(output_dir, 'fig2_contrastive_learning_effect.png')
    explainer.visualize_contrastive_learning_effect(features, adj_matrices, adj_dict, fig_path, sample_indices)
    patient_idx = patient_indices[0] if patient_indices else 0
    explanation = explainer.explain_prediction(patient_idx, features, adj_matrices, adj_dict)
    fig_path = os.path.join(output_dir, 'fig3_multi_scale_importance.png')
    explainer.visualize_scale_importance(explanation, fig_path)
    fig_path = os.path.join(output_dir, 'fig4_metapath_importance.png')
    explainer.visualize_metapath_importance(explanation, 'p', fig_path)
    fig_path = os.path.join(output_dir, 'fig5_contrastive_learning.png')
    explainer.visualize_contrastive_learning(explanation, 'p', fig_path)
    fig_path = os.path.join(output_dir, 'fig6_clinical_case_analysis.png')
    num_patients = min(4, len(patient_indices))
    rows = (num_patients + 1) // 2  
    plt.figure(figsize=(18, 6 * rows))

    for i, patient_idx in enumerate(patient_indices[:num_patients]):
        patient_id = dataset.get_subject_id(patient_idx)
        patient_explanation = explainer.explain_prediction(patient_idx, features, adj_matrices, adj_dict)
        plt.subplot(rows, 2, i+1)
        recommendations = patient_explanation['top_recommendations'][:10]  
      
        if recommendations:
            drug_indices = [rec[0] for rec in recommendations]
            scores = [rec[1] for rec in recommendations]
            y_pos = np.arange(len(drug_indices))
            bars = plt.barh(y_pos, scores, color='cornflowerblue')
            plt.yticks(y_pos, [f"Drug {idx}" for idx in drug_indices])
            plt.xlabel('Confidence Score')
            plt.title(f'Patient {patient_id}: Top Recommendations')
            for j, drug_info in enumerate(patient_explanation['drug_analysis'][:5]):
                if drug_info.get('is_combination', False):
                    if j < len(bars):  # Safety check
                        bars[j].set_color('indianred')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='cornflowerblue', label='Single-agent'),
                Patch(facecolor='indianred', label='Combination')
            ]
            plt.legend(handles=legend_elements, loc='lower right', fontsize='small')
            clinical_text = ""
            num_diagnoses = len(patient_explanation['clinical_context'].get('diagnoses', []))
            num_specimens = len(patient_explanation['clinical_context'].get('specimens', []))
            
            clinical_text += f"Diagnoses: {num_diagnoses}, Specimens: {num_specimens}"
            row_idx = i // 2
            col_idx = i % 2
            plt.figtext(0.125 + col_idx*0.5, 0.85 - row_idx*0.5/rows, clinical_text, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def modify_train_function(original_train_function):
    """
    Modify the original train function to include explainability
    """
    def train_with_explainability(model, dataset, train_mask, val_mask, optimizer, 
                                 scheduler, criterion, num_epochs, device, args):
        best_model, model_filename = original_train_function(
            model, dataset, train_mask, val_mask, optimizer, 
            scheduler, criterion, num_epochs, device, args
        )
        if args.generate_explanations and args.explanation_mode == 'train':
            features = dataset.get_features(device)
            adj_dict = dataset.get_type_specific_adj()

            adj_matrices = {}
            for key, value in adj_dict.items():
                if isinstance(value, tuple):
                    adj, edge_dim = value
                    adj_matrices[key] = adj.to(device)
                else:
                    adj_matrices[key] = value.to(device)
            
            generate_explanations(
                args, best_model, dataset, device, val_mask,
                features, adj_matrices, adj_dict
            )
        
        return best_model, model_filename
    
    return train_with_explainability

def modify_evaluate_function(original_evaluate_function):
    """
    Modify the original evaluate function to include explainability
    """
    def evaluate_with_explainability(model, dataset, mask, device, args, generate_explanations_flag=False):
        test_metrics = original_evaluate_function(model, dataset, mask, device, args)
        if ((args.generate_explanations and args.explanation_mode == 'test') or 
            generate_explanations_flag):
            features = dataset.get_features(device)
            adj_dict = dataset.get_type_specific_adj()

            adj_matrices = {}
            for key, value in adj_dict.items():
                if isinstance(value, tuple):
                    adj, edge_dim = value
                    adj_matrices[key] = adj.to(device)
                else:
                    adj_matrices[key] = value.to(device)

            explanations = generate_explanations(
                args, model, dataset, device, mask,
                features, adj_matrices, adj_dict
            )

            test_metrics['explanations'] = explanations
        
        return test_metrics
    
    return evaluate_with_explainability

def add_case_study_code(args, model, dataset, device, test_mask):
    """
    Add code to run case study with explainability
    """
    features = dataset.get_features(device)
    adj_dict = dataset.get_type_specific_adj()
    adj_matrices = {}
    for key, value in adj_dict.items():
        if isinstance(value, tuple):
            adj, edge_dim = value
            adj_matrices[key] = adj.to(device)
        else:
            adj_matrices[key] = value.to(device)

    case_studies = generate_explanations(
        args, model, dataset, device, test_mask,
        features, adj_matrices, adj_dict
    )
    
    return case_studies