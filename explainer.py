import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import json

class HetMSAMRGNNExplainer:
    def __init__(self, model, dataset, device):
        """
        Initialize the explainer
        
        Args:
            model: Trained HetMS-AMRGNN model
            dataset: Dataset object
            device: PyTorch device
        """
        self.model = model
        self.dataset = dataset
        self.device = device

        num_views = model.num_views if hasattr(model, 'num_views') else 3
        num_scales = model.num_scales if hasattr(model, 'num_scales') else 3
        view_meanings = {}
        for i in range(num_views):
            view_meanings[i] = f"View {i+1}"

        scale_meanings = {}
        for i in range(num_scales):
            scale_meanings[i] = f"Scale {i+1}"

        self.clinical_mappings = {
            'view_meanings': view_meanings,
            'scale_meanings': scale_meanings
        }
        
        print(f"Explainer initialized with {num_views} views and {num_scales} scales")
        
    def prepare_patient_histories(self, batch_histories, max_len=100, num_diagnoses=None):
        """
        Process patient histories for the model
        
        Args:
            batch_histories: List of patient history sequences
            max_len: Maximum sequence length
            num_diagnoses: Total number of possible diagnoses
            
        Returns:
            Tensor of processed histories
        """
        if not batch_histories:
            return None
        
        if num_diagnoses is None:
            num_diagnoses = self.dataset.get_num_diagnoses()
        
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
        
        return torch.LongTensor(flat_histories).to(self.device)
        
    def explain_prediction(self, patient_idx, features, adj_matrices, adj_dict, history=None):
        """
        Generate comprehensive explanation for a patient prediction
        
        Args:
            patient_idx: Index of the patient
            features: Node features
            adj_matrices: Adjacency matrices
            adj_dict: Dictionary of adjacency matrices by type
            history: Patient history sequence (optional)
            
        Returns:
            Explanation dictionary with all components needed for visualization
        """
        self.model.eval()
        batch_indices = torch.tensor([patient_idx], device=self.device)
        if history is not None:
            batch_histories = [history]
            history_tensor = self.prepare_patient_histories(batch_histories)
        elif self.model.use_history:
            patient_id = self.dataset.get_subject_id(patient_idx)
            history_data = self.dataset.patient_history.get(patient_id, [])
            if history_data:
                batch_histories = [history_data]
                history_tensor = self.prepare_patient_histories(batch_histories)
            else:
                history_tensor = None
        else:
            history_tensor = None

        with torch.no_grad():
            scores, contrastive_loss, intermediates = self.model(
                features, adj_matrices, adj_dict, self.dataset,
                batch_indices, history=history_tensor,
                return_contrastive_loss=True, return_intermediates=True
            )

        clinical_context = self._get_clinical_context(patient_idx)
        drug_scores = scores[0].cpu().numpy()
        top_indices = np.argsort(drug_scores)[::-1][:10]
        top_recommendations = [(int(idx), float(drug_scores[idx])) for idx in top_indices]
        explanation = {
            'patient_id': self.dataset.get_subject_id(patient_idx),
            'clinical_context': clinical_context,
            'top_recommendations': top_recommendations,
            'intermediates': intermediates,
            'drug_analysis': self._analyze_recommended_drugs(top_recommendations, clinical_context),
            'visualizations': {}
        }
        
        return explanation
    
    def _get_clinical_context(self, patient_idx):
        """Extract clinical information for the patient"""
        context = {
            'diagnoses': [],
            'medications': [],
            'specimens': []
        }

        if hasattr(self.dataset, 'edge_index') and 'p_d' in self.dataset.edge_index:
            diag_indices = self.dataset.edge_index['p_d'][1][
                self.dataset.edge_index['p_d'][0] == patient_idx
            ].tolist()
            context['diagnoses'] = diag_indices

        if hasattr(self.dataset, 'edge_index') and 'p_c' in self.dataset.edge_index:
            med_indices = self.dataset.edge_index['p_c'][1][
                self.dataset.edge_index['p_c'][0] == patient_idx
            ].tolist()
            context['medications'] = med_indices

        if hasattr(self.dataset, 'edge_index') and 'p_s' in self.dataset.edge_index:
            spec_indices = self.dataset.edge_index['p_s'][1][
                self.dataset.edge_index['p_s'][0] == patient_idx
            ].tolist()
            context['specimens'] = spec_indices
            
        return context
    
    def _analyze_recommended_drugs(self, recommendations, clinical_context):
        """Analyze recommended drugs and provide clinical interpretation"""
        analysis = []
        
        for drug_idx, score in recommendations:
            drug_info = {
                'index': drug_idx,
                'confidence_score': score,
                'is_combination': False
            }

            if hasattr(self.dataset, 'node_features') and 'c' in self.dataset.node_features:
                if self.dataset.node_features['c'].shape[1] > 1 and drug_idx < self.dataset.node_features['c'].shape[0]:
                    try:
                        is_combination = bool(self.dataset.node_features['c'][drug_idx, 0].item())
                        drug_info['is_combination'] = is_combination
                        if is_combination:
                            components = []
                            for i in range(1, self.dataset.node_features['c'].shape[1]):
                                if drug_idx < self.dataset.node_features['c'].shape[0] and i < self.dataset.node_features['c'].shape[1]:
                                    if bool(self.dataset.node_features['c'][drug_idx, i].item()):
                                        components.append(i-1)
                            drug_info['components'] = components
                    except IndexError:
                        pass

            diagnoses = clinical_context.get('diagnoses', [])
            if diagnoses:
                drug_info['diagnosis_relevance'] = self._get_diagnosis_drug_relevance(drug_idx, diagnoses)
            
            analysis.append(drug_info)
        
        return analysis
    
    def visualize_view_importance(self, explanation, output_path=None):
        """Visualize multi-view importance weights with neutral labeling"""
        plt.figure(figsize=(10, 6))
        view_weights = None
        for key, value in explanation['intermediates'].items():
            if key == 'view_attention_p':
                patient_idx = 0 
                view_weights = value[patient_idx].squeeze().cpu().numpy()
                break
        
        if view_weights is None or len(view_weights) == 0:
            plt.text(0.5, 0.5, "No view weights available", 
                    fontsize=14, ha='center', va='center')
        else:
            view_indices = np.arange(len(view_weights))
            bars = plt.bar(view_indices, view_weights, color='skyblue')
            labels = [f"View {i+1}" for i in view_indices]
            
            plt.xticks(view_indices, labels, rotation=45 if len(view_weights) > 4 else 0, 
                    ha='right' if len(view_weights) > 4 else 'center')
            plt.ylabel('Attention Weight')
            plt.title(f'Multi-View Importance for Patient {explanation["patient_id"]}')

            dominant_idx = np.argmax(view_weights)
            dominant_bar = bars[dominant_idx]
            dominant_bar.set_color('coral')
        
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            return output_path
        else:
            return plt.gcf()

    def visualize_scale_importance(self, explanation, output_path=None):
        """Visualize the importance of different graph scales with neutral labels"""
        plt.figure(figsize=(10, 6))
        scale_weights = None
        for key, value in explanation['intermediates'].items():
            if key.startswith('scale_weights_layer0'):
                scale_weights = value.cpu().numpy()
                break
        
        if scale_weights is None or len(scale_weights) == 0:
            plt.text(0.5, 0.5, "No scale weights available", 
                    fontsize=14, ha='center', va='center')
        else:
            scale_indices = np.arange(len(scale_weights))
            bars = plt.bar(scale_indices, scale_weights, color='lightgreen')
            labels = [f"Scale {i+1}" for i in scale_indices]
            
            plt.xticks(scale_indices, labels)
            plt.ylabel('Importance Weight')
            plt.title('Multi-Scale Graph Convolution: Scale Importance')

            dominant_idx = np.argmax(scale_weights)
            dominant_bar = bars[dominant_idx]
            dominant_bar.set_color('coral')
        
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            return output_path
        else:
            return plt.gcf()

    def visualize_metapath_importance(self, explanation, node_type='p', output_path=None):
        """Visualize the importance of different metapaths with neutral labels"""
        plt.figure(figsize=(12, 8))
        metapath_weights = None
        metapath_names = []
        
        for key, value in explanation['intermediates'].items():
            if key.startswith('metapath_weights_layer0'):
                if node_type in value:
                    metapath_weights = value[node_type]
                    if isinstance(metapath_weights, torch.Tensor):
                        metapath_weights = metapath_weights.cpu().numpy()
                    for mp_key, mp_value in explanation['intermediates'].items():
                        if mp_key.startswith('metapath_embeddings_layer0'):
                            if node_type in mp_value:
                                metapath_names = mp_value[node_type]['names']
                                break
                    break
        
        if metapath_weights is None or len(metapath_weights) == 0:
            plt.text(0.5, 0.5, f"No metapath weights available for {node_type} nodes", 
                    fontsize=14, ha='center', va='center')
            if output_path:
                plt.savefig(output_path, dpi=300)
                plt.close()
                return output_path
            else:
                return plt.gcf()
        
        if not metapath_names:
            if metapath_weights.ndim <= 2: 
                metapath_names = [f"Metapath {i+1}" for i in range(metapath_weights.shape[-1])]
            else:
                metapath_names = [f"Metapath {i+1}" for i in range(metapath_weights.shape[-1])]
        
        if metapath_weights.ndim == 2: 
            sns.heatmap(metapath_weights, cmap='YlOrRd', 
                        xticklabels=metapath_names, yticklabels=metapath_names,
                        annot=True, fmt=".2f")
            plt.title(f'Metapath Attention Weights for {node_type.upper()} Nodes')
        elif metapath_weights.ndim == 3: 
            num_views = self.model.num_views if hasattr(self.model, 'num_views') else None
            
            if num_views is not None and metapath_weights.shape[0] == num_views:
                fig, axes = plt.subplots(1, num_views, figsize=(5*num_views, 5))
                if num_views == 1:
                    axes = [axes]
                for view_idx in range(num_views):
                    if view_idx < metapath_weights.shape[0]:
                        view_weight = metapath_weights[view_idx]
                        sns.heatmap(view_weight, cmap='YlOrRd', 
                                    xticklabels=metapath_names, yticklabels=metapath_names,
                                    annot=True, fmt=".2f", ax=axes[view_idx])
                        axes[view_idx].set_title(f'View {view_idx+1}')

                plt.suptitle(f'Metapath Attention Weights Across Views ({node_type.upper()} Nodes)', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])  
            else:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                avg_weights = metapath_weights.mean(axis=0)
                sns.heatmap(avg_weights, cmap='YlOrRd', 
                            xticklabels=metapath_names, yticklabels=metapath_names,
                            annot=True, fmt=".2f", ax=axes[0])
                axes[0].set_title(f'Average Metapath Attention Weights ({node_type.upper()} Nodes)')
                if metapath_weights.shape[0] > 0:
                    example_idx = 0  
                    sns.heatmap(metapath_weights[example_idx], cmap='YlOrRd',
                                xticklabels=metapath_names, yticklabels=metapath_names,
                                annot=True, fmt=".2f", ax=axes[1])
                    axes[1].set_title(f'Example Node Metapath Attention Weights ({node_type.upper()} Node #{example_idx})')
        elif metapath_weights.ndim >= 4: 
            plt.text(0.5, 0.5, f"Complex metapath weights with {metapath_weights.ndim} dimensions detected.\nShowing average across all dimensions except the last two.", 
                    fontsize=14, ha='center', va='center')
            reduced_dims = tuple(range(metapath_weights.ndim - 2))
            avg_weights = metapath_weights.mean(axis=reduced_dims)
            sns.heatmap(avg_weights, cmap='YlOrRd', 
                        xticklabels=metapath_names, yticklabels=metapath_names,
                        annot=True, fmt=".2f")
            plt.title(f'Average Metapath Attention Weights ({node_type.upper()} Nodes)')
        else:
            plt.text(0.5, 0.5, f"Unexpected metapath weights dimension: {metapath_weights.ndim}", 
                    fontsize=14, ha='center', va='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            return output_path
        else:
            return plt.gcf()
    
    def visualize_contrastive_learning(self, explanation, node_type='p', output_path=None):
        """Visualize contrastive learning effects using actual embeddings data with t-SNE"""
        if 'contrastive_scores' not in explanation['intermediates'] or len(explanation['intermediates']['contrastive_scores']) == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No contrastive learning data available", 
                    fontsize=14, ha='center', va='center')
            if output_path:
                plt.savefig(output_path, dpi=300)
                plt.close()
                return output_path
            else:
                return plt.gcf()

        node_keys = [k for k in explanation['intermediates']['contrastive_scores'].keys() if k.startswith(node_type)]
        
        if not node_keys:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"No contrastive learning data for {node_type} nodes", 
                    fontsize=14, ha='center', va='center')
            if output_path:
                plt.savefig(output_path, dpi=300)
                plt.close()
                return output_path
            else:
                return plt.gcf()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        pos_scores = []
        neg_scores = []
        display_keys = node_keys[:min(5, len(node_keys))]
        
        for key in display_keys:
            scores = explanation['intermediates']['contrastive_scores'][key]
            pos_score = min(scores['positive'], 10.0) 
            neg_score = min(scores['negative'], 10.0)
            pos_scores.append(pos_score)
            neg_scores.append(neg_score)
        
        x = np.arange(len(display_keys))
        width = 0.35
        
        ax1.bar(x - width/2, pos_scores, width, label='Positive similarity', color='skyblue')
        ax1.bar(x + width/2, neg_scores, width, label='Negative similarity', color='salmon')
        
        ax1.set_xlabel('Node ID')
        ax1.set_ylabel('Similarity Score (Normalized)')
        ax1.set_title('Contrastive Learning: Positive vs. Negative Similarity')
        ax1.set_xticks(x)

        x_labels = []
        for k in display_keys:
            parts = k.split('_')
            if len(parts) >= 2:
                x_labels.append(parts[1]) 
            else:
                x_labels.append(k)
        ax1.set_xticklabels(x_labels)
        ax1.legend()

        view_embeddings = None
        for key, value in explanation['intermediates'].items():
            if key == f'extracted_features_{node_type}':
                view_embeddings = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                break

        num_views = self.model.num_views if hasattr(self.model, 'num_views') else 3
        
        if view_embeddings is None or view_embeddings.shape[1] < 2:
            final_embeddings = None
            for key, value in explanation['intermediates'].items():
                if key == f'final_embeddings_{node_type}':
                    final_embeddings = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
                    break
            
            if final_embeddings is not None and len(final_embeddings) > 0:
                from sklearn.manifold import TSNE
                if len(final_embeddings) >= 2:
                    max_samples = 200
                    if len(final_embeddings) > max_samples:
                        indices = np.random.choice(len(final_embeddings), max_samples, replace=False)
                        sample_embeddings = final_embeddings[indices]
                    else:
                        sample_embeddings = final_embeddings
                    perplexity = min(30, max(5, len(sample_embeddings) // 3))
                    
                    try:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                                learning_rate='auto', init='pca')
                        reduced_embeddings = tsne.fit_transform(sample_embeddings)
                        
                        ax2.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, c='green')
                        ax2.set_title('Final Embeddings (t-SNE)')
                        ax2.set_xlabel('t-SNE Dimension 1')
                        ax2.set_ylabel('t-SNE Dimension 2')
                    except Exception as e:
                        print(f"t-SNE error: {e}")
                        ax2.text(0.5, 0.5, "t-SNE failed for final embeddings", 
                                ha='center', va='center', transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, "Not enough samples for t-SNE", 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, "No embedding data available", 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            cmap = plt.cm.get_cmap('tab10', num_views)
            all_embeddings = []
            view_indices = [] 
            for v in range(min(num_views, view_embeddings.shape[1])):
                if v < view_embeddings.shape[1]:
                    max_samples_per_view = min(250, view_embeddings.shape[0]) 
                    if view_embeddings.shape[0] > max_samples_per_view:
                        indices = np.random.choice(view_embeddings.shape[0], max_samples_per_view, replace=False)
                        view_data = view_embeddings[indices, v, :]
                    else:
                        view_data = view_embeddings[:, v, :]
                    
                    all_embeddings.append(view_data)
                    view_indices.extend([v] * len(view_data))
            
            combined_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            view_indices = np.array(view_indices)

            if len(combined_embeddings) >= 10: 
                from sklearn.manifold import TSNE
                perplexity = min(30, max(5, len(combined_embeddings) // 5))
                
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                            learning_rate='auto', init='pca', n_iter=2000)
                    reduced_data = tsne.fit_transform(combined_embeddings)
                    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

                    for v in range(num_views):
                        if v in view_indices:
                            mask = (view_indices == v)
                            if np.any(mask):
                                color = cmap(v)
                                marker = markers[v % len(markers)]
                                jitter = np.random.normal(0, 0.05, reduced_data[mask].shape)
                                ax2.scatter(
                                    reduced_data[mask, 0] + jitter[:, 0], 
                                    reduced_data[mask, 1] + jitter[:, 1],
                                    alpha=0.7, c=[color], marker=marker, label=f"View {v+1}", s=30
                                )
                    
                    ax2.set_title('Multi-View Embeddings (t-SNE)')
                    ax2.set_xlabel('t-SNE Dimension 1')
                    ax2.set_ylabel('t-SNE Dimension 2')
                    ax2.legend(title="Views")
                    ax2.grid(True, linestyle='--', alpha=0.3)
                    
                except Exception as e:
                    print(f"t-SNE error: {e}")
                    ax2.text(0.5, 0.5, f"t-SNE failed: {str(e)}", 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, "Not enough samples for t-SNE", 
                        ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            return output_path
        else:
            return plt.gcf()

    def visualize_recommendations(self, explanation, output_path=None):
        """Visualize top drug recommendations with neutral labeling"""
        plt.figure(figsize=(12, 6))
        recommendations = explanation['top_recommendations'][:10] 
        
        if not recommendations:
            plt.text(0.5, 0.5, "No recommendations available", 
                    fontsize=14, ha='center', va='center')
        else:
            drug_indices = [rec[0] for rec in recommendations]
            scores = [rec[1] for rec in recommendations]
            y_pos = np.arange(len(drug_indices))
            bars = plt.barh(y_pos, scores, color='cornflowerblue')
            plt.yticks(y_pos, [f"Drug {idx}" for idx in drug_indices])
            plt.xlabel('Confidence Score')
            plt.title(f'Top Antimicrobial Recommendations for Patient {explanation["patient_id"]}')
            for i, drug_info in enumerate(explanation['drug_analysis'][:10]):
                if drug_info.get('is_combination', False):
                    bars[i].set_color('indianred')
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='cornflowerblue', label='Single-agent therapy'),
                Patch(facecolor='indianred', label='Combination therapy')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            return plt.gcf()

    def visualize_embeddings_umap(self, features, adj_matrices, adj_dict, node_type='p', output_path=None):
        self.model.eval()
        
        with torch.no_grad():
            batch_size = min(1000, self.dataset.node_type_count[node_type])
            batch_indices = torch.arange(batch_size, device=self.device)
            
            print(f"Getting embeddings for {batch_size} {node_type} nodes...")
            _, _, intermediates = self.model(
                features, adj_matrices, adj_dict, self.dataset,
                batch_indices if node_type == 'p' else None,
                return_contrastive_loss=False, return_intermediates=True
            )

            view_embeddings = None
            for key, value in intermediates.items():
                if key == f'extracted_features_{node_type}':
                    view_embeddings = value.cpu().numpy()
                    break

            final_embeddings = None
            for key, value in intermediates.items():
                if key == f'final_embeddings_{node_type}':
                    final_embeddings = value.cpu().numpy()
                    break
        
        if view_embeddings is None or final_embeddings is None:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"No embeddings available for {node_type} nodes", 
                    fontsize=14, ha='center', va='center')
            if output_path:
                plt.savefig(output_path, dpi=300)
                plt.close()
                return output_path
            else:
                return plt.gcf()

        num_views = view_embeddings.shape[1] if view_embeddings.ndim > 2 else 1
        fig_width = min(18, 4.5 * (num_views + 1)) 
        fig, axes = plt.subplots(1, num_views + 1, figsize=(fig_width, 6))
        if num_views + 1 == 1:
            axes = [axes]
        colors = plt.cm.tab10(np.linspace(0, 1, num_views))
        
        for view_idx in range(num_views):
            if view_embeddings.shape[1] > view_idx:  
                view_data = view_embeddings[:, view_idx, :]
                view_data_sample = view_data
                try:
                    n_neighbors = min(30, max(15, view_data_sample.shape[0] // 10))
                    min_dist = 0.1  
                    
                    reducer = UMAP(
                        n_components=2, 
                        random_state=42, 
                        min_dist=min_dist, 
                        n_neighbors=n_neighbors, 
                        metric='cosine',
                        low_memory=False
                    )
                    embedding = reducer.fit_transform(view_data_sample)
                    color = colors[view_idx]

                    from sklearn.neighbors import NearestNeighbors
                    try:
                        k_neighbors = min(15, len(embedding)-1)
                        if k_neighbors > 0:
                            nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(embedding)
                            distances, _ = nbrs.kneighbors(embedding)
                            avg_distances = distances.mean(axis=1)
                            point_sizes = 50 / (1 + avg_distances * 3)  
                            point_sizes = np.clip(point_sizes, 10, 80) 
                        else:
                            point_sizes = 30 
                    except Exception as e:
                        print(f"Error calculating point sizes: {e}")
                        point_sizes = 30  

                    scatter = axes[view_idx].scatter(
                        embedding[:, 0], embedding[:, 1], 
                        alpha=0.7, 
                        c=[color],
                        s=point_sizes
                    )

                    axes[view_idx].set_title(f'View {view_idx+1} Embeddings', pad=10)
                    axes[view_idx].set_xlabel('UMAP Dimension 1')
                    axes[view_idx].set_ylabel('UMAP Dimension 2')

                    axes[view_idx].grid(True, linestyle='--', alpha=0.3)
                    
                except Exception as e:
                    print(f"Error applying UMAP to view {view_idx}: {e}")
                    axes[view_idx].text(0.5, 0.5, f"UMAP failed for View {view_idx+1}: {str(e)}",
                                    ha='center', va='center', transform=axes[view_idx].transAxes)
            else:
                axes[view_idx].text(0.5, 0.5, f"View {view_idx+1} not available",
                                ha='center', va='center', transform=axes[view_idx].transAxes)
        try:
            final_embeddings_sample = final_embeddings
                
            n_neighbors = min(30, max(15, final_embeddings_sample.shape[0] // 10))
            reducer_final = UMAP(
                n_components=2, 
                random_state=42, 
                min_dist=0.1, 
                n_neighbors=n_neighbors, 
                metric='cosine',
                low_memory=False
            )
            embedding_final = reducer_final.fit_transform(final_embeddings_sample)
            from sklearn.neighbors import NearestNeighbors
            try:
                k_neighbors = min(15, len(embedding_final)-1)
                if k_neighbors > 0:
                    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(embedding_final)
                    distances, _ = nbrs.kneighbors(embedding_final)
                    avg_distances = distances.mean(axis=1)
                    point_sizes = 50 / (1 + avg_distances * 3)
                    point_sizes = np.clip(point_sizes, 10, 80)
                else:
                    point_sizes = 30
            except Exception as e:
                print(f"Error calculating point sizes for final embeddings: {e}")
                point_sizes = 30
            
            scatter = axes[num_views].scatter(
                embedding_final[:, 0], embedding_final[:, 1], 
                alpha=0.7, 
                c='green',
                s=point_sizes
            )
            
            axes[num_views].set_title('Final Fused Embeddings', pad=10)
            axes[num_views].set_xlabel('UMAP Dimension 1')
            axes[num_views].set_ylabel('UMAP Dimension 2')
            axes[num_views].grid(True, linestyle='--', alpha=0.3)
            
        except Exception as e:
            print(f"Error applying UMAP to final embeddings: {e}")
            axes[num_views].text(0.5, 0.5, f"UMAP failed for Final Embeddings: {str(e)}",
                            ha='center', va='center', transform=axes[num_views].transAxes)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            return output_path
        else:
            return plt.gcf()
    
    def create_explanation_report(self, explanation, features, adj_matrices, adj_dict, output_dir):
        """
        Create a comprehensive PDF report with all visualizations
        
        Args:
            explanation: The explanation dictionary
            features: Node features
            adj_matrices: Adjacency matrices
            adj_dict: Dictionary of adjacency matrices
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated PDF report
        """
        os.makedirs(output_dir, exist_ok=True)
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        visualizations = {}
        vis_path = os.path.join(vis_dir, 'view_importance.png')
        visualizations['view_importance'] = self.visualize_view_importance(explanation, vis_path)
        vis_path = os.path.join(vis_dir, 'scale_importance.png')
        visualizations['scale_importance'] = self.visualize_scale_importance(explanation, vis_path)
        vis_path = os.path.join(vis_dir, 'metapath_importance.png')
        visualizations['metapath_importance'] = self.visualize_metapath_importance(explanation, 'p', vis_path)
        vis_path = os.path.join(vis_dir, 'contrastive_learning.png')
        visualizations['contrastive_learning'] = self.visualize_contrastive_learning(explanation, 'p', vis_path)
        vis_path = os.path.join(vis_dir, 'recommendations.png')
        visualizations['recommendations'] = self.visualize_recommendations(explanation, vis_path)
        vis_path = os.path.join(vis_dir, 'embeddings_umap.png')
        visualizations['embeddings_umap'] = self.visualize_embeddings_umap(features, adj_matrices, adj_dict, 'p', vis_path)
        explanation['visualizations'] = visualizations
        pdf_path = os.path.join(output_dir, f'explanation_report_patient_{explanation["patient_id"]}.pdf')
        
        with PdfPages(pdf_path) as pdf:
            plt.figure(figsize=(12, 10))
            plt.axis('off')
            plt.text(0.5, 0.8, "HetMS-AMRGNN Explanation Report", fontsize=24, ha='center')
            plt.text(0.5, 0.7, f"Patient ID: {explanation['patient_id']}", fontsize=20, ha='center')
            plt.text(0.5, 0.6, f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", fontsize=16, ha='center')
            plt.text(0.5, 0.4, "This report provides a comprehensive explanation of the antimicrobial\n" +
                              "drug recommendations generated by the HetMS-AMRGNN model.\n" +
                              "It includes visualization of model internals and decision rationale.",
                    fontsize=14, ha='center')
            pdf.savefig()
            plt.close()
            plt.figure(figsize=(12, 10))
            plt.axis('off')
            plt.text(0.5, 0.95, "Patient Clinical Context", fontsize=20, ha='center')

            clinical_context = explanation['clinical_context']
            y_pos = 0.85

            if 'diagnoses' in clinical_context and clinical_context['diagnoses']:
                plt.text(0.1, y_pos, "Diagnoses:", fontsize=16)
                for i, diag_idx in enumerate(clinical_context['diagnoses'][:10]):
                    plt.text(0.1, y_pos - 0.03*(i+1), f"• Diagnosis {diag_idx}", fontsize=12)
                y_pos -= 0.03 * (len(clinical_context['diagnoses'][:10]) + 2)

            if 'medications' in clinical_context and clinical_context['medications']:
                plt.text(0.1, y_pos, "Medication History:", fontsize=16)
                for i, med_idx in enumerate(clinical_context['medications'][:10]):
                    plt.text(0.1, y_pos - 0.03*(i+1), f"• Medication {med_idx}", fontsize=12)
                y_pos -= 0.03 * (len(clinical_context['medications'][:10]) + 2)

            if 'specimens' in clinical_context and clinical_context['specimens']:
                plt.text(0.1, y_pos, "Clinical Specimens:", fontsize=16)
                for i, spec_idx in enumerate(clinical_context['specimens'][:10]):
                    plt.text(0.1, y_pos - 0.03*(i+1), f"• Specimen {spec_idx}", fontsize=12)
            
            pdf.savefig()
            plt.close()
            
            for name, path in visualizations.items():
                if os.path.exists(path):
                    plt.figure(figsize=(12, 10))
                    img = plt.imread(path)
                    plt.imshow(img)
                    plt.axis('off')
                    pdf.savefig()
                    plt.close()
            
            # Conclusion page
            plt.figure(figsize=(12, 10))
            plt.axis('off')
            plt.text(0.5, 0.95, "Summary and Clinical Interpretation", fontsize=20, ha='center')
            
            pdf.savefig()
            plt.close()

        json_path = os.path.join(output_dir, f'explanation_data_patient_{explanation["patient_id"]}.json')

        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, np.number)):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)

        serializable_explanation = {}
        for key, value in explanation.items():
            if key != 'intermediates': 
                serializable_explanation[key] = convert_for_json(value)

        if 'intermediates' in explanation:
            serializable_explanation['intermediates'] = {}
            keys_to_keep = ['view_attention_p', 'scale_weights_layer0']
            for key in keys_to_keep:
                if key in explanation['intermediates']:
                    serializable_explanation['intermediates'][key] = convert_for_json(explanation['intermediates'][key])
        
        with open(json_path, 'w') as f:
            json.dump(serializable_explanation, f, indent=2)
        
        return pdf_path
    
    def visualize_multi_view_embeddings(self, features, adj_matrices, adj_dict, output_path=None, sample_indices=None):
        """Visualize embeddings from different views with neutral labeling"""
        self.model.eval()
        
        with torch.no_grad():
            if sample_indices is None:
                batch_indices = torch.arange(min(1000, self.dataset.node_type_count['p']), device=self.device)
            else:
                batch_indices = sample_indices
            
            print(f"Generating embeddings for {len(batch_indices)} patients...")

            chunk_size = 500 
            all_view_embeddings = []
            
            for i in range(0, len(batch_indices), chunk_size):
                chunk_indices = batch_indices[i:i+chunk_size]
                print(f"Processing embedding chunk {i//chunk_size + 1}/{(len(batch_indices) + chunk_size - 1)//chunk_size}")
                _, _, intermediates = self.model(
                    features, adj_matrices, adj_dict, self.dataset,
                    chunk_indices, return_contrastive_loss=False, return_intermediates=True
                )

                chunk_embeddings = None
                for key, value in intermediates.items():
                    if key == 'extracted_features_p':
                        chunk_embeddings = value.cpu().numpy()
                        all_view_embeddings.append(chunk_embeddings)
                        break

            if all_view_embeddings:
                view_embeddings = np.concatenate(all_view_embeddings, axis=0)
            else:
                view_embeddings = None
            
            if view_embeddings is None:
                plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, "Multi-view embedding data unavailable", fontsize=14, ha='center', va='center')
                if output_path:
                    plt.savefig(output_path, dpi=300)
                    plt.close()
                    return output_path
                else:
                    return plt.gcf()

            num_views = view_embeddings.shape[1] if view_embeddings.ndim > 2 else 1
            
            print(f"Performing t-SNE on {view_embeddings.shape[0]} samples with {num_views} views...")

            max_tsne_samples = 3000  
            if view_embeddings.shape[0] > max_tsne_samples:
                print(f"Subsampling to {max_tsne_samples} points for t-SNE...")
                indices = np.random.choice(view_embeddings.shape[0], max_tsne_samples, replace=False)
                view_embeddings = view_embeddings[indices]

            colors = plt.cm.tab10(np.linspace(0, 1, num_views))

            fig_width = min(20, 5 * num_views) 
            fig, axes = plt.subplots(1, num_views, figsize=(fig_width, 6))

            if num_views == 1:
                axes = [axes]

            for view_idx in range(num_views):
                view_data = view_embeddings[:, view_idx, :]
                print(f"Running t-SNE for view {view_idx+1}...")
                perplexity = min(30, len(view_data)-1)
                
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                            learning_rate='auto', init='pca', n_jobs=-1)
                    embedding_2d = tsne.fit_transform(view_data)
                    color = colors[view_idx]
                    axes[view_idx].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5, s=5, c=[color])
                    axes[view_idx].set_title(f'View {view_idx+1} Embeddings')
                    axes[view_idx].set_xlabel('t-SNE Dimension 1')
                    axes[view_idx].set_ylabel('t-SNE Dimension 2')
                    axes[view_idx].grid(True, linestyle='--', alpha=0.6)
                except Exception as e:
                    print(f"Error in t-SNE for view {view_idx+1}: {e}")
                    axes[view_idx].text(0.5, 0.5, f"t-SNE failed for View {view_idx+1}",
                                    ha='center', va='center', transform=axes[view_idx].transAxes)
            
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return output_path
            else:
                return plt.gcf()

    def visualize_contrastive_learning_effect(self, features, adj_matrices, adj_dict, output_path=None, sample_indices=None):
        """Visualize the effect of contrastive learning with before/after comparison"""
        self.model.eval()
        
        with torch.no_grad():
            if sample_indices is None:
                batch_indices = torch.arange(min(200, self.dataset.node_type_count['p']), device=self.device)
            else:
                if len(sample_indices) > 1000:
                    indices = torch.randperm(len(sample_indices))[:1000]
                    batch_indices = sample_indices[indices]
                else:
                    batch_indices = sample_indices
            
            print(f"Analyzing contrastive learning effect on {len(batch_indices)} patients...")

            chunk_size = 500

            original_state = {}
            for name, module in self.model.named_modules():
                if 'contrastive_module' in name:
                    for param_name, param in module.named_parameters():
                        original_state[f"{name}.{param_name}"] = param.clone()                      
                        param.data.fill_(0.0)

            all_embeddings_without = []
            for i in range(0, len(batch_indices), chunk_size):
                chunk_indices = batch_indices[i:i+chunk_size]
                print(f"Processing 'without contrastive learning' chunk {i//chunk_size + 1}/{(len(batch_indices) + chunk_size - 1)//chunk_size}")
                
                _, _, intermediates = self.model(
                    features, adj_matrices, adj_dict, self.dataset,
                    chunk_indices, return_contrastive_loss=True, return_intermediates=True
                )
                
                for key, value in intermediates.items():
                    if key == 'extracted_features_p':
                        all_embeddings_without.append(value.cpu().numpy())
                        break

            for name, module in self.model.named_modules():
                if 'contrastive_module' in name:
                    for param_name, param in module.named_parameters():
                        key = f"{name}.{param_name}"
                        if key in original_state:
                            param.data = original_state[key]

            all_embeddings_with = []
            for i in range(0, len(batch_indices), chunk_size):
                chunk_indices = batch_indices[i:i+chunk_size]
                print(f"Processing 'with contrastive learning' chunk {i//chunk_size + 1}/{(len(batch_indices) + chunk_size - 1)//chunk_size}")
                
                _, _, intermediates = self.model(
                    features, adj_matrices, adj_dict, self.dataset,
                    chunk_indices, return_contrastive_loss=True, return_intermediates=True
                )
                
                for key, value in intermediates.items():
                    if key == 'extracted_features_p':
                        all_embeddings_with.append(value.cpu().numpy())
                        break

            if all_embeddings_without:
                view_embeddings_without = np.concatenate(all_embeddings_without, axis=0)
            else:
                view_embeddings_without = None
                
            if all_embeddings_with:
                view_embeddings_with = np.concatenate(all_embeddings_with, axis=0)
            else:
                view_embeddings_with = None
            
            if view_embeddings_without is None or view_embeddings_with is None:
                plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, "Unable to retrieve contrastive learning effect data", fontsize=14, ha='center', va='center')
                if output_path:
                    plt.savefig(output_path, dpi=300)
                    plt.close()
                    return output_path
                else:
                    return plt.gcf()

        num_views = view_embeddings_without.shape[1] if view_embeddings_without.ndim > 2 else 1

        def calculate_view_similarity(embeddings, num_views):
            print(f"Calculating similarity matrix for {embeddings.shape[0]} samples...")
            num_nodes = embeddings.shape[0]

            similarity_matrix = np.zeros((num_nodes, num_views, num_views))
            
            for i in range(num_nodes):
                for v1 in range(num_views):
                    for v2 in range(num_views):
                        vec1 = embeddings[i, v1]
                        vec2 = embeddings[i, v2]
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        if norm1 > 1e-8 and norm2 > 1e-8:
                            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                        else:
                            similarity = 0
                        similarity_matrix[i, v1, v2] = similarity

            return np.mean(similarity_matrix, axis=0)

        similarity_without = calculate_view_similarity(view_embeddings_without, num_views)
        similarity_with = calculate_view_similarity(view_embeddings_with, num_views)
        fig = plt.figure(figsize=(20, 7))
        gs = plt.GridSpec(1, 5, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0:2])  
        ax2 = fig.add_subplot(gs[0, 2:4])  
        ax3 = fig.add_subplot(gs[0, 4])   

        im1 = ax1.imshow(similarity_without, cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title('Without Contrastive Learning: Cross-View Similarity')
        ax1.set_xticks(np.arange(similarity_without.shape[1]))
        ax1.set_yticks(np.arange(similarity_without.shape[0]))
        ax1.set_xticklabels([f'View {i+1}' for i in range(similarity_without.shape[1])])
        ax1.set_yticklabels([f'View {i+1}' for i in range(similarity_without.shape[0])])
        
        im2 = ax2.imshow(similarity_with, cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title('With Contrastive Learning: Cross-View Similarity')
        ax2.set_xticks(np.arange(similarity_with.shape[1]))
        ax2.set_yticks(np.arange(similarity_with.shape[0]))
        ax2.set_xticklabels([f'View {i+1}' for i in range(similarity_with.shape[1])])
        ax2.set_yticklabels([f'View {i+1}' for i in range(similarity_with.shape[0])])
        fig.colorbar(im1, ax=ax1, label='Similarity')
        fig.colorbar(im2, ax=ax2, label='Similarity')

        for i in range(similarity_without.shape[0]):
            for j in range(similarity_without.shape[1]):
                ax1.text(j, i, f"{similarity_without[i, j]:.2f}", 
                    ha="center", va="center", color="black" if similarity_without[i, j] < 0.7 else "white")
                
                ax2.text(j, i, f"{similarity_with[i, j]:.2f}", 
                    ha="center", va="center", color="black" if similarity_with[i, j] < 0.7 else "white")
        
        samples_per_view = min(50, view_embeddings_without.shape[0])
        all_data = []
        point_colors = []
        point_markers = []
        point_labels = []
        
        marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

        for v in range(num_views):
            if view_embeddings_without.shape[0] > samples_per_view:
                indices = np.random.choice(view_embeddings_without.shape[0], samples_per_view, replace=False)
                sampled_data = view_embeddings_without[indices, v, :]
            else:
                sampled_data = view_embeddings_without[:, v, :]
            
            all_data.append(sampled_data)
            point_colors.extend(['blue'] * len(sampled_data))
            point_markers.extend([marker_styles[v % len(marker_styles)]] * len(sampled_data))
            point_labels.extend([f'Before: View {v+1}'] * len(sampled_data))

        for v in range(num_views):
            if view_embeddings_with.shape[0] > samples_per_view:
                indices = np.random.choice(view_embeddings_with.shape[0], samples_per_view, replace=False)
                sampled_data = view_embeddings_with[indices, v, :]
            else:
                sampled_data = view_embeddings_with[:, v, :]
            
            all_data.append(sampled_data)
            point_colors.extend(['red'] * len(sampled_data))  # Red for 'after'
            point_markers.extend([marker_styles[v % len(marker_styles)]] * len(sampled_data))
            point_labels.extend([f'After: View {v+1}'] * len(sampled_data))

        combined_data = np.vstack(all_data)

        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(combined_data) // 5))
        
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                    learning_rate='auto', init='pca', n_iter=2000)
            reduced_data = tsne.fit_transform(combined_data)

            grouped_data = {}
            for i, (color, marker, label) in enumerate(zip(point_colors, point_markers, point_labels)):
                key = (color, marker, label)
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(i)

            for (color, marker, label), indices in grouped_data.items():
                ax3.scatter(
                    reduced_data[indices, 0], 
                    reduced_data[indices, 1],
                    c=color, marker=marker, alpha=0.7, label=label, s=30
                )

            if num_views <= 3: 
                ax3.legend(fontsize='small')
            
            ax3.set_title('Before/After Contrastive Learning (t-SNE)')
            ax3.set_xlabel('t-SNE Dimension 1')
            ax3.set_ylabel('t-SNE Dimension 2')
            ax3.grid(True, linestyle='--', alpha=0.3)
            
        except Exception as e:
            print(f"t-SNE error in before/after visualization: {e}")
            ax3.text(0.5, 0.5, f"t-SNE failed: {str(e)}", 
                    ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            return plt.gcf()

    def calculate_view_similarity(embeddings, num_views):
        print(f"Calculating similarity matrix for {embeddings.shape[0]} samples...")
        num_nodes = embeddings.shape[0]
        similarity_matrix = np.zeros((num_nodes, num_views, num_views))
        
        for i in range(num_nodes):
            for v1 in range(num_views):
                for v2 in range(num_views):
                    vec1 = embeddings[i, v1]
                    vec2 = embeddings[i, v2]
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 1e-8 and norm2 > 1e-8:  
                        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    else:
                        similarity = 0
                    similarity_matrix[i, v1, v2] = similarity

        return np.mean(similarity_matrix, axis=0)

        similarity_without = calculate_view_similarity(view_embeddings_without, num_views)
        similarity_with = calculate_view_similarity(view_embeddings_with, num_views)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        im1 = ax1.imshow(similarity_without, cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title('Without Contrastive Learning: Cross-View Similarity')
        ax1.set_xticks(np.arange(similarity_without.shape[1]))
        ax1.set_yticks(np.arange(similarity_without.shape[0]))
        ax1.set_xticklabels([f'View {i+1}' for i in range(similarity_without.shape[1])])
        ax1.set_yticklabels([f'View {i+1}' for i in range(similarity_without.shape[0])])
        
        im2 = ax2.imshow(similarity_with, cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title('With Contrastive Learning: Cross-View Similarity')
        ax2.set_xticks(np.arange(similarity_with.shape[1]))
        ax2.set_yticks(np.arange(similarity_with.shape[0]))
        ax2.set_xticklabels([f'View {i+1}' for i in range(similarity_with.shape[1])])
        ax2.set_yticklabels([f'View {i+1}' for i in range(similarity_with.shape[0])])

        fig.colorbar(im1, ax=ax1, label='Similarity')
        fig.colorbar(im2, ax=ax2, label='Similarity')

        for i in range(similarity_without.shape[0]):
            for j in range(similarity_without.shape[1]):
                ax1.text(j, i, f"{similarity_without[i, j]:.2f}", 
                    ha="center", va="center", color="black" if similarity_without[i, j] < 0.7 else "white")
                
                ax2.text(j, i, f"{similarity_with[i, j]:.2f}", 
                    ha="center", va="center", color="black" if similarity_with[i, j] < 0.7 else "white")
        
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            return plt.gcf()

    def run_case_study(self, patient_indices, features, adj_matrices, adj_dict, output_dir):
        """
        Run a comprehensive case study for multiple patients
        
        Args:
            patient_indices: List of patient indices to analyze
            features: Node features
            adj_matrices: Adjacency matrices
            adj_dict: Dictionary of adjacency matrices
            output_dir: Directory to save results
            
        Returns:
            Dictionary with explanations for each patient
        """
        case_studies = {}
        os.makedirs(output_dir, exist_ok=True)
        for patient_idx in patient_indices:
            patient_id = self.dataset.get_subject_id(patient_idx)
            patient_dir = os.path.join(output_dir, f'patient_{patient_id}')
            os.makedirs(patient_dir, exist_ok=True)
            explanation = self.explain_prediction(patient_idx, features, adj_matrices, adj_dict)
            report_path = self.create_explanation_report(
                explanation, features, adj_matrices, adj_dict, patient_dir
            )

            case_studies[patient_idx] = {
                'patient_id': patient_id,
                'report_path': report_path,
                'top_recommendations': explanation['top_recommendations']
            }
            
            print(f"Generated explanation for Patient {patient_id} (report: {report_path})")

        summary_path = os.path.join(output_dir, 'case_study_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"HetMS-AMRGNN Case Study Summary\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Number of patients analyzed: {len(patient_indices)}\n\n")
            
            for patient_idx, study in case_studies.items():
                f.write(f"Patient {study['patient_id']}:\n")
                f.write(f"  Report: {study['report_path']}\n")
                f.write(f"  Top recommendations:\n")
                for i, (drug_idx, score) in enumerate(study['top_recommendations'][:5]):
                    f.write(f"    {i+1}. Drug {drug_idx}: {score:.4f}\n")
                f.write("\n")
        
        return case_studies