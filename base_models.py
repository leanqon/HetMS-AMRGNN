import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
import numpy as np
import math

def detect_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def get_node_embeddings(self):
    return self.final_embeddings

class MultiViewFeatureExtraction(nn.Module):
    def __init__(self, in_dim, out_dim, num_views):
        super(MultiViewFeatureExtraction, self).__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),  
                nn.LeakyReLU(0.1)       
            ) for _ in range(num_views)
        ])
        self.initialize_weights()
    
    def initialize_weights(self):
        for proj in self.projections:
            if isinstance(proj[0], nn.Linear):
                nn.init.kaiming_normal_(proj[0].weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(proj[0].bias)
    
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        
        results = []
        for proj in self.projections:
            x_clipped = torch.clamp(x_normalized, -10.0, 10.0)
            view_result = proj(x_clipped)
           
            if torch.isnan(view_result).any() or torch.isinf(view_result).any():
                view_result = torch.nan_to_num(view_result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            results.append(view_result)
        
        output = torch.stack(results, dim=1)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output

class BiChannelExtraction(nn.Module):
    def __init__(self, node_types, in_dim, out_dim, num_scales, num_views):
        super(BiChannelExtraction, self).__init__()
        self.spatial_conv = MultiScaleGraphConvolution(node_types, in_dim, out_dim, num_scales, num_views)
        self.out_proj = nn.ModuleDict({
            ntype: nn.Linear(out_dim, out_dim) for ntype in node_types
        })
    
    def forward(self, x, adj_dict):
        for ntype, feat in x.items():
            if torch.isnan(feat).any():
                x[ntype] = torch.nan_to_num(feat, nan=0.0)

        h_spatial = self.spatial_conv(x, adj_dict)
        
        return h_spatial
    
class GlobalCapture(nn.Module):
    def __init__(self, node_types, in_dim, hidden_dim, num_heads):
        super(GlobalCapture, self).__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, adj_matrices):
        outputs = {}
        for ntype in self.node_types:
            num_nodes = x[ntype].shape[0]
            query = x[ntype]  # [num_nodes, hidden_dim]
            
            neighbor_info = torch.cat([
                torch.sparse.mm(adj_matrices[f"{ntype}-{other_type}"], x[other_type])
                for other_type in self.node_types
            ], dim=0)
            
            if neighbor_info.shape[0] > 1000:
                indices = torch.randperm(neighbor_info.shape[0])[:1000]
                neighbor_info = neighbor_info[indices]
            
            key = value = neighbor_info  # [num_neighbors, hidden_dim]
            
            query = query.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            key = key.unsqueeze(1)  # [num_neighbors, 1, hidden_dim]
            value = value.unsqueeze(1)  # [num_neighbors, 1, hidden_dim]
            
            batch_size = 1000
            attn_outputs = []
            for i in range(0, num_nodes, batch_size):
                batch_query = query[i:i+batch_size]
                batch_output, _ = self.attention(batch_query, key, value)
                attn_outputs.append(batch_output)
            
            attn_output = torch.cat(attn_outputs, dim=0)
            outputs[ntype] = self.out_proj(attn_output.squeeze(1))
        
        return outputs

@torch.jit.script
def optimized_sparse_mm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(adj, x)

class MetapathAggregation(nn.Module):
    def __init__(self, node_types, in_dim, out_dim, metapaths, num_heads):
        super(MetapathAggregation, self).__init__()
        self.node_types = node_types
        self.metapaths = metapaths

        # Create projections for each metapath in each node type
        self.projections = nn.ModuleDict({
            ntype: nn.ModuleDict({
                '->'.join(metapath): nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU()
                )
                for metapath in metapaths[ntype]
            }) for ntype in node_types if ntype in metapaths and metapaths[ntype]
        })
        
        # Create attention layers for each node type
        self.attention = nn.ModuleDict({
            ntype: nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
            for ntype in node_types if ntype in metapaths and metapaths[ntype]
        })
        
        # Layer normalization for each node type
        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(out_dim)
            for ntype in node_types if ntype in metapaths and metapaths[ntype]
        })
        
        # Store attention weights for explainability
        self.attention_weights = {}
        self.metapath_embeddings = {}

    def forward(self, features, adj_dict):
        h_global = {}
        self.attention_weights = {}  # Reset attention weights
        self.metapath_embeddings = {}  # Reset metapath embeddings
        
        for ntype in self.node_types:
            if ntype not in self.metapaths or not self.metapaths[ntype]:
                h_global[ntype] = features[ntype]
                continue

            metapath_embeddings = []
            metapath_names = []
            
            # Process each metapath for this node type
            for metapath in self.metapaths[ntype]:
                metapath_names.append('->'.join(metapath))
                h = features[metapath[0]]
                
                # Follow the metapath
                for i in range(len(metapath) - 1):
                    src, dst = metapath[i], metapath[i+1]
                    adj_key = f"{src}-{dst}"
                
                    # Check if the adjacency matrix exists
                    if adj_key not in adj_dict:
                        continue
                    
                    # Get and handle the adjacency matrix (handle the case it's a tuple)
                    adj = adj_dict[adj_key]
                    if isinstance(adj, tuple):
                        adj = adj[0]  # Extract the sparse tensor from the tuple
                    try:
                        h = torch.stack([torch.sparse.mm(adj_dict[f"{src}-{dst}"].t(), h[:, v, :]) for v in range(h.size(1))], dim=1)
                        # Add L2 normalization after each matrix multiplication
                        h = F.normalize(h, p=2, dim=-1)
                    except Exception as e:
                        # If there's an error, try to continue with the current h
                        continue
                
                # Apply projection
                h = self.projections[ntype]['->'.join(metapath)](h)
                metapath_embeddings.append(h)
            
            # Store intermediate embeddings for explainability
            if metapath_embeddings:
                self.metapath_embeddings[ntype] = {
                    'names': metapath_names,
                    'embeddings': [emb.detach().clone() for emb in metapath_embeddings]
                }
                
                # Stack metapath embeddings for attention
                metapath_embeddings_stacked = torch.stack(metapath_embeddings, dim=2)
                attn_output = []
                attn_weights_list = []  # Store attention weights for all views
                
                # Process each view separately
                for v in range(metapath_embeddings_stacked.size(1)):
                    view_emb = metapath_embeddings_stacked[:, v, :, :]
                    attn_out, attn_weights = self.attention[ntype](view_emb, view_emb, view_emb)
                    attn_weights_list.append(attn_weights.detach().clone())
                    
                    # Apply residual connection and layer norm
                    attn_out = attn_out + view_emb  
                    attn_out = self.layer_norms[ntype](attn_out)
                    attn_output.append(attn_out.mean(dim=1))
                
                # Store attention weights for explainability
                self.attention_weights[ntype] = torch.stack(attn_weights_list, dim=0)
                
                # Combine outputs from all views
                h_global[ntype] = torch.stack(attn_output, dim=1)  # [num_nodes, num_views, out_dim]
            else:
                h_global[ntype] = features[ntype]

        return h_global

class MultiScaleGraphConvolution(nn.Module):
    def __init__(self, node_types, in_dim, out_dim, num_scales, num_views):
        super(MultiScaleGraphConvolution, self).__init__()
        self.node_types = node_types
        self.num_scales = num_scales
        self.num_views = num_views
        self.out_dim = out_dim

        # Linear transformations
        self.linear_transforms = nn.ModuleDict({
            ntype: nn.ModuleList([
                nn.Linear(in_dim, out_dim) for _ in range(num_scales)
            ]) for ntype in node_types
        })

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Output projection
        self.out_proj = nn.ModuleDict({
            ntype: nn.Linear(out_dim * num_scales, out_dim) for ntype in node_types
        })
        
        # Cache computation results
        self.adj_power_cache = {}
        
        # For explainability
        self.scale_outputs = {}

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Use more conservative initialization"""
        for ntype in self.node_types:
            for scale in range(self.num_scales):
                nn.init.xavier_uniform_(self.linear_transforms[ntype][scale].weight, gain=0.1)
                if self.linear_transforms[ntype][scale].bias is not None:
                    nn.init.zeros_(self.linear_transforms[ntype][scale].bias)
            
            nn.init.xavier_uniform_(self.out_proj[ntype].weight, gain=0.1)
            if self.out_proj[ntype].bias is not None:
                nn.init.zeros_(self.out_proj[ntype].bias)
        
        # Use smaller initial scale weights
        with torch.no_grad():
            self.scale_weights.fill_(0.1)

    def compute_power(self, adj, k, starts_from_d=False):
        # Ensure adjacency matrix is coalesced
        adj = adj.coalesce()
        
        # Create cache key
        cache_key = (adj._nnz(), adj.size(0), adj.size(1), k, starts_from_d)
        
        # Check cache
        if cache_key in self.adj_power_cache:
            return self.adj_power_cache[cache_key]
            
        # For scale 0, return original adjacency or its transpose
        if k == 0:
            if starts_from_d:
                result = adj
            else:
                result = adj.t().coalesce()
            
            self.adj_power_cache[cache_key] = result
            return result
        
        # Set numerical stability parameters
        max_value = -1.0
        min_nnz = 1.0
        
        try:
            with torch.no_grad():
                if starts_from_d:
                    # Relation starts from destination node
                    adj_t = adj.t().coalesce()
                    power_base = torch.sparse.mm(adj, adj_t).coalesce()
                    
                    # Check numerical stability
                    if power_base._nnz() < min_nnz or torch.isnan(power_base.values()).any() or torch.isinf(power_base.values()).any():
                        return adj
                    
                    # Limit values to prevent overflow
                    values = power_base.values()
                    clipped_values = torch.clamp(values, -max_value, max_value)
                    if not torch.allclose(values, clipped_values):
                        power_base = torch.sparse_coo_tensor(
                            power_base.indices(), clipped_values, power_base.size()
                        ).coalesce()
                    
                    # Compute (A_r A^T_r)^⌊k/2⌋
                    power = power_base
                    for i in range(k//2 - 1):
                        if power._nnz() == 0:
                            break
                        
                        next_power = torch.sparse.mm(power, power_base).coalesce()
                        
                        # Check numerical stability
                        if next_power._nnz() < min_nnz or torch.isnan(next_power.values()).any() or torch.isinf(next_power.values()).any():
                            break
                        
                        # Limit values
                        values = next_power.values()
                        clipped_values = torch.clamp(values, -max_value, max_value)
                        if not torch.allclose(values, clipped_values):
                            next_power = torch.sparse_coo_tensor(
                                next_power.indices(), clipped_values, next_power.size()
                            ).coalesce()
                        
                        power = next_power
                    
                    # Finally multiply by A_r
                    if k//2 > 0 and power._nnz() > 0:
                        result = torch.sparse.mm(power, adj).coalesce()
                        
                        # Check result
                        if result._nnz() < min_nnz or torch.isnan(result.values()).any() or torch.isinf(result.values()).any():
                            result = adj
                        
                        # Limit result values
                        values = result.values()
                        clipped_values = torch.clamp(values, -max_value, max_value)
                        if not torch.allclose(values, clipped_values):
                            result = torch.sparse_coo_tensor(
                                result.indices(), clipped_values, result.size()
                            ).coalesce()
                    else:
                        result = adj
                else:
                    # Relation ends at destination node
                    adj_t = adj.t().coalesce()
                    power_base = torch.sparse.mm(adj_t, adj).coalesce()
                    
                    # Check numerical stability
                    if power_base._nnz() < min_nnz or torch.isnan(power_base.values()).any() or torch.isinf(power_base.values()).any():
                        return adj_t
                    
                    # Limit values to prevent overflow
                    values = power_base.values()
                    clipped_values = torch.clamp(values, -max_value, max_value)
                    if not torch.allclose(values, clipped_values):
                        power_base = torch.sparse_coo_tensor(
                            power_base.indices(), clipped_values, power_base.size()
                        ).coalesce()
                    
                    # Compute (A^T_r A_r)^⌊k/2⌋
                    power = power_base
                    for i in range(k//2 - 1):
                        if power._nnz() == 0:
                            break
                        
                        next_power = torch.sparse.mm(power, power_base).coalesce()
                        
                        # Check numerical stability
                        if next_power._nnz() < min_nnz or torch.isnan(next_power.values()).any() or torch.isinf(next_power.values()).any():
                            break
                        
                        # Limit values
                        values = next_power.values()
                        clipped_values = torch.clamp(values, -max_value, max_value)
                        if not torch.allclose(values, clipped_values):
                            next_power = torch.sparse_coo_tensor(
                                next_power.indices(), clipped_values, next_power.size()
                            ).coalesce()
                        
                        power = next_power
                    
                    # Finally multiply by A^T_r
                    if k//2 > 0 and power._nnz() > 0:
                        result = torch.sparse.mm(power, adj_t).coalesce()
                        
                        # Check result
                        if result._nnz() < min_nnz or torch.isnan(result.values()).any() or torch.isinf(result.values()).any():
                            result = adj_t
                        
                        # Limit result values
                        values = result.values()
                        clipped_values = torch.clamp(values, -max_value, max_value)
                        if not torch.allclose(values, clipped_values):
                            result = torch.sparse_coo_tensor(
                                result.indices(), clipped_values, result.size()
                            ).coalesce()
                    else:
                        result = adj_t
        except RuntimeError as e:
            # Fallback to original matrix if computation fails
            if starts_from_d:
                result = adj
            else:
                result = adj.t().coalesce()
        
        # Cache result
        self.adj_power_cache[cache_key] = result
        
        # Cache management
        if len(self.adj_power_cache) > 100:
            keys_to_remove = list(self.adj_power_cache.keys())[:len(self.adj_power_cache)//2]
            for key in keys_to_remove:
                del self.adj_power_cache[key]
        
        return result

    def forward(self, x, adj_matrices):
        # Reset scale outputs
        self.scale_outputs = {ntype: [] for ntype in self.node_types}
        
        # Handle NaN inputs
        for ntype, feat in x.items():
            if torch.isnan(feat).any():
                x[ntype] = torch.nan_to_num(feat, nan=0.0)
        
        results = {}
        
        for ntype in self.node_types:
            # For each node type, process all scales and views
            node_results = []
            
            for scale in range(self.num_scales):
                scale_result = torch.zeros(
                    x[ntype].size(0), x[ntype].size(1), self.out_dim, 
                    device=x[ntype].device
                )
                
                for rel, adj in adj_matrices.items():
                    src_type, dst_type = rel.split('-')
                    
                    # Process relations where current node type is destination
                    if dst_type == ntype:
                        adj_power = self.compute_power(adj, scale, starts_from_d=False)
                        
                        for view in range(self.num_views):
                            try:
                                # Get source node features
                                src_input = x[src_type][:, view, :]
                                if torch.isnan(src_input).any():
                                    src_input = torch.nan_to_num(src_input, nan=0.0)
                                
                                # Transform source node features
                                src_transformed = self.linear_transforms[src_type][scale](src_input)
                                
                                # Check for NaN
                                if torch.isnan(src_transformed).any():
                                    src_transformed = torch.zeros_like(src_transformed)
                                
                                # Apply sparse matrix multiplication
                                message = torch.sparse.mm(adj_power, src_transformed)
                                
                                # Check for NaN
                                if torch.isnan(message).any():
                                    message = torch.zeros_like(message)
                                
                                scale_result[:, view, :] += message
                            except RuntimeError as e:
                                pass
                    
                    # Process relations where current node type is source
                    elif src_type == ntype:
                        adj_power = self.compute_power(adj, scale, starts_from_d=True)
                        
                        for view in range(self.num_views):
                            try:
                                # Get destination node features
                                dst_input = x[dst_type][:, view, :]
                                if torch.isnan(dst_input).any():
                                    dst_input = torch.nan_to_num(dst_input, nan=0.0)
                                
                                # Transform destination node features
                                dst_transformed = self.linear_transforms[dst_type][scale](dst_input)
                                
                                # Check for NaN
                                if torch.isnan(dst_transformed).any():
                                    dst_transformed = torch.zeros_like(dst_transformed)
                                
                                # Apply sparse matrix multiplication
                                message = torch.sparse.mm(adj_power, dst_transformed)
                                
                                # Check for NaN
                                if torch.isnan(message).any():
                                    message = torch.zeros_like(message)
                                
                                scale_result[:, view, :] += message
                            except RuntimeError as e:
                                pass
                
                # Check scale weights for NaN
                if torch.isnan(self.scale_weights[scale]).any():
                    with torch.no_grad():
                        self.scale_weights[scale] = torch.tensor(0.1, device=self.scale_weights.device)
                
                # Apply scale weight
                weighted_result = scale_result * self.scale_weights[scale]
                
                # Check for NaN
                if torch.isnan(weighted_result).any():
                    weighted_result = torch.zeros_like(weighted_result)
                
                # Store scale result for explainability
                self.scale_outputs[ntype].append(weighted_result.detach().clone())
                
                # Add to node results
                node_results.append(weighted_result)
            
            # Concatenate all scale results
            try:
                node_result = torch.cat(node_results, dim=-1)
                
                # Check for NaN
                if torch.isnan(node_result).any():
                    node_result = torch.zeros_like(node_result)
                
                # Apply output projection
                output = self.out_proj[ntype](node_result)
                
                # Apply ReLU activation
                results[ntype] = F.relu(output)
                
                # Final NaN check
                if torch.isnan(results[ntype]).any():
                    results[ntype] = torch.zeros_like(results[ntype])
            except RuntimeError as e:
                results[ntype] = x[ntype].clone()

        return results

class ViewAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(ViewAttention, self).__init__()
        self.query = nn.Parameter(torch.Tensor(hidden_dim))
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()
        self.weights = None  # Store attention weights for explainability
        self.input_embeddings = None  # Store input embeddings for explainability
        
    def reset_parameters(self):
        init.xavier_uniform_(self.query.unsqueeze(0))
        init.xavier_uniform_(self.projection.weight)
        init.zeros_(self.projection.bias)
        
    def forward(self, x):
        # x shape: [num_nodes, num_views, hidden_dim]
        num_nodes, num_views, hidden_dim = x.shape
        
        # Store input embeddings for later visualization
        self.input_embeddings = x.detach().clone()
        
        # Project views
        projected = torch.tanh(self.projection(x))  # [num_nodes, num_views, hidden_dim]
        
        # Calculate attention scores
        scores = torch.matmul(projected, self.query)  # [num_nodes, num_views]
        
        # Normalize attention weights
        weights = F.softmax(scores, dim=1).unsqueeze(2)  # [num_nodes, num_views, 1]
        
        # Store weights for explainability
        self.weights = weights.detach().clone()
        
        # Weighted combination of views
        output = torch.sum(x * weights, dim=1)  # [num_nodes, hidden_dim]
        
        return output
    
    def get_input_embeddings(self):
        """Return the stored input embeddings"""
        if self.input_embeddings is None:
            raise ValueError("Input embeddings not available. Run forward pass first.")
        return self.input_embeddings

class FusionModule(nn.Module):
    def __init__(self, hidden_dim, num_views):
        super(FusionModule, self).__init__()
        self.gate = nn.Linear(hidden_dim * 2, 2)
        self.transform = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, h_local, h_global):
        # h_local and h_global shape: [num_nodes, num_views, hidden_dim]
        h_concat = torch.cat([h_local, h_global], dim=-1)
        gate = F.softmax(self.gate(h_concat), dim=-1)
        h_fused = self.transform(h_concat)
        return h_fused * gate.sum(dim=-1, keepdim=True)

class ContrastiveLearningModule(nn.Module):
    def __init__(self, hidden_dim, temperature=0.5):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # For explainability
        self.positive_pairs = {}
        self.negative_pairs = {}
        self.contrast_scores = {}
        
    def forward(self, node_embeddings, node_types, num_views, num_layers):
        """
        Implement hierarchical contrastive learning
        with additional tracking for explainability
        """
        contrastive_loss = 0.0
        total_samples = 0
        
        # Reset tracking dictionaries
        self.positive_pairs = {}
        self.negative_pairs = {}
        self.contrast_scores = {}
        
        # Priority for patient and drug nodes
        priority_types = ['p', 'c']
        remaining_types = [t for t in node_types if t not in priority_types]
        processing_order = priority_types + remaining_types
        
        # Process each node type
        for ntype in processing_order:
            # Limit sample size for non-priority types
            max_nodes = 100 if ntype in priority_types else 20
            
            # Process each view and layer
            for view_i in range(num_views):
                for layer_i in range(num_layers):
                    key = (ntype, view_i, layer_i)
                    if key not in node_embeddings:
                        continue
                    
                    # Get current embeddings
                    embedding = node_embeddings[key]
                    num_nodes = embedding.size(0)
                    
                    # Sample nodes for efficiency
                    sample_size = min(max_nodes, num_nodes)
                    if sample_size < 5:  # Skip if too few nodes
                        continue
                        
                    indices = torch.randperm(num_nodes, device=embedding.device)[:sample_size]
                    sampled_nodes = embedding[indices]
                    
                    # Process each sampled node
                    for idx, node_emb in enumerate(sampled_nodes):
                        # Get current node embedding
                        z = self.projection(node_emb.unsqueeze(0))
                        z = F.normalize(z, p=2, dim=1)
                        
                        # Collect positive samples
                        pos_samples = []
                        
                        # P1: Same node, different views, same layer
                        for view_j in range(num_views):
                            if view_j != view_i and (ntype, view_j, layer_i) in node_embeddings:
                                pos_emb = node_embeddings[(ntype, view_j, layer_i)][indices[idx]].unsqueeze(0)
                                pos_proj = F.normalize(self.projection(pos_emb), p=2, dim=1)
                                pos_samples.append(pos_proj)
                        
                        # P2: Same node, same view, different layers
                        for layer_j in range(num_layers):
                            if layer_j != layer_i and (ntype, view_i, layer_j) in node_embeddings:
                                pos_emb = node_embeddings[(ntype, view_i, layer_j)][indices[idx]].unsqueeze(0)
                                pos_proj = F.normalize(self.projection(pos_emb), p=2, dim=1)
                                pos_samples.append(pos_proj)
                        
                        # P3: Same node, different views, different layers
                        for view_j in range(num_views):
                            for layer_j in range(num_layers):
                                if view_j != view_i and layer_j != layer_i and (ntype, view_j, layer_j) in node_embeddings:
                                    pos_emb = node_embeddings[(ntype, view_j, layer_j)][indices[idx]].unsqueeze(0)
                                    pos_proj = F.normalize(self.projection(pos_emb), p=2, dim=1)
                                    pos_samples.append(pos_proj)
                        
                        # Skip if no positive samples
                        if not pos_samples:
                            continue
                        
                        # Collect negative samples
                        neg_samples = []
                        
                        # N1: Same node type and view, different nodes
                        other_indices = [i for i in range(len(indices)) if i != idx]
                        if other_indices:
                            neg_indices = torch.tensor(other_indices, device=embedding.device)
                            neg_embs = sampled_nodes[neg_indices]
                            neg_projs = F.normalize(self.projection(neg_embs), p=2, dim=1)
                            neg_samples.append(neg_projs)
                        
                        # N3: Different node types
                        for other_type in node_types:
                            if other_type != ntype and (other_type, view_i, layer_i) in node_embeddings:
                                other_emb = node_embeddings[(other_type, view_i, layer_i)]
                                other_num_nodes = other_emb.size(0)
                                neg_sample_size = min(max_nodes // 2, other_num_nodes)
                                if neg_sample_size > 0:
                                    other_indices = torch.randperm(other_num_nodes, device=embedding.device)[:neg_sample_size]
                                    neg_embs = other_emb[other_indices]
                                    neg_projs = F.normalize(self.projection(neg_embs), p=2, dim=1)
                                    neg_samples.append(neg_projs)
                        
                        # Skip if no negative samples
                        if not neg_samples:
                            continue
                        
                        # Store samples for explainability
                        node_key = f"{ntype}_{indices[idx].item()}_v{view_i}_l{layer_i}"
                        self.positive_pairs[node_key] = [p.detach().cpu() for p in pos_samples]
                        self.negative_pairs[node_key] = [n.detach().cpu() for n in neg_samples]
                        
                        # Calculate contrastive loss
                        pos_similarity = 0.0
                        for pos in pos_samples:
                            sim = torch.mm(z, pos.t()) / self.temperature
                            pos_similarity += torch.exp(sim).sum()
                        
                        neg_similarity = 0.0
                        for neg in neg_samples:
                            sim = torch.mm(z, neg.t()) / self.temperature
                            neg_similarity += torch.exp(sim).sum()
                        
                        # Store similarity scores for explainability
                        self.contrast_scores[node_key] = {
                            'positive': pos_similarity.item(),
                            'negative': neg_similarity.item()
                        }
                        
                        # Compute the loss
                        if pos_similarity > 0 and neg_similarity > 0:
                            node_loss = -torch.log(pos_similarity / (pos_similarity + neg_similarity))
                            contrastive_loss += node_loss
                            total_samples += 1
        
        # Compute final loss
        if total_samples > 0:
            contrastive_loss = contrastive_loss / total_samples
        
        return contrastive_loss

class HistoryEncoder(nn.Module):
    def __init__(self, num_diagnoses, hidden_dim):
        super(HistoryEncoder, self).__init__()
        self.embedding = nn.Embedding(num_diagnoses, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention_query = nn.Parameter(torch.Tensor(hidden_dim))
        self.attention_projection = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.attention_query.unsqueeze(0))
        init.xavier_uniform_(self.attention_projection.weight)
        init.zeros_(self.attention_projection.bias)
        
    def forward(self, history_sequences, mask=None):
        """
        Process patient history according to Equations 16-18
        
        Args:
            history_sequences: Tensor of shape [batch_size, max_seq_length]
            mask: Optional mask for padding
        """
        batch_size, max_seq_length = history_sequences.shape
        embedded = self.embedding(history_sequences) 
        gru_output, _ = self.gru(embedded)
        projected = torch.tanh(self.attention_projection(gru_output)) 
        scores = torch.matmul(projected, self.attention_query)         
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=1).unsqueeze(2)  
        context = torch.sum(gru_output * attention_weights, dim=1) 
        
        return context
    
class NodeEnhancement(nn.Module):
    def __init__(self, hidden_dim):
        super(NodeEnhancement, self).__init__()
        self.history_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.ddi_alpha = nn.Parameter(torch.FloatTensor([0.5]))
        self.ddi_gcn = GCNConv(hidden_dim, hidden_dim)
        
    def enhance_patient(self, patient_emb, history_emb):
        concat = torch.cat([patient_emb, history_emb], dim=1)
        gate = self.history_gate(concat)
        enhanced = gate * patient_emb + (1 - gate) * history_emb
        
        return enhanced
    
    def enhance_drug(self, drug_emb, ddi_edge_index):
        ddi_emb = self.ddi_gcn(drug_emb, ddi_edge_index)
        enhanced = self.ddi_alpha * drug_emb + (1 - self.ddi_alpha) * ddi_emb
        
        return enhanced

class LinkPrediction(nn.Module):
    def __init__(self, hidden_dim):
        super(LinkPrediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, patient_emb, drug_emb):
        num_patients = patient_emb.size(0)
        num_drugs = drug_emb.size(0)
        hidden_dim = patient_emb.size(1)
        scores = torch.zeros((num_patients, num_drugs), device=patient_emb.device)
        
        batch_size = 32
        for i in range(0, num_patients, batch_size):
            end_i = min(i + batch_size, num_patients)
            p_batch = patient_emb[i:end_i]  
            p_expanded = p_batch.unsqueeze(1).expand(-1, num_drugs, -1)  
            d_expanded = drug_emb.unsqueeze(0).expand(end_i - i, -1, -1) 
            interaction = p_expanded * d_expanded  
            concat = torch.cat([p_expanded, d_expanded, interaction], dim=2)  
            flat_concat = concat.view(-1, hidden_dim * 3) 
            flat_scores = self.mlp(flat_concat) 
            batch_scores = flat_scores.view(end_i - i, num_drugs) 
            batch_scores = torch.sigmoid(batch_scores)
            scores[i:end_i] = batch_scores
        
        return scores

class HetMSAMRGNN(nn.Module):
    def __init__(self, node_types, in_dims, hidden_dim, num_drugs, num_views, 
                 num_scales, num_heads, num_layers, metapaths, node_id_ranges, 
                 dropout_rate=0.2, use_history=False, num_diagnoses=None,
                 contrastive_temp=0.5):
        super(HetMSAMRGNN, self).__init__()
        self.node_types = node_types
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_history = use_history
        self.num_views = num_views
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-view feature extraction
        self.feature_extractions = nn.ModuleDict({
            ntype: MultiViewFeatureExtraction(in_dims[ntype], hidden_dim, num_views)
            for ntype in node_types
        })

        # Multi-scale graph convolution for different layers
        self.multi_scale_convs = nn.ModuleList([MultiScaleGraphConvolution(
            node_types=node_types,
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_scales=num_scales,
            num_views=num_views
        )
            for _ in range(num_layers)
        ])
        
        # Metapath aggregation for different layers
        self.metapath_aggregations = nn.ModuleList([
            MetapathAggregation(node_types, hidden_dim, hidden_dim, metapaths, num_heads)
            for _ in range(num_layers)
        ])
        
        # Fusion modules to combine local and global information
        self.fusion_modules = nn.ModuleList([
            nn.ModuleDict({
                ntype: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                for ntype in node_types
            })
            for _ in range(num_layers)
        ])
        
        # Layer normalization for residual connections
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in node_types})
            for _ in range(num_layers)
        ])
        
        # Multi-view attention
        self.view_attention = nn.ModuleDict({
            ntype: ViewAttention(hidden_dim) for ntype in node_types
        })
        
        # Contrastive learning module
        self.contrastive_module = ContrastiveLearningModule(hidden_dim, temperature=contrastive_temp)
        
        # Node enhancement
        self.node_enhancement = NodeEnhancement(hidden_dim)
        
        # Link prediction
        self.link_prediction = LinkPrediction(hidden_dim)
        
        # History encoder for patient history
        if use_history:
            if num_diagnoses is None:
                raise ValueError("num_diagnoses must be provided when use_history is True")
            self.history_encoder = HistoryEncoder(num_diagnoses, hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def create_global_to_local_mapping(self, features):
        """Create mapping between global and local node indices"""
        global_to_local = {}
        local_to_global = {}
        global_counter = 0
        for node_type, feature in features.items():
            num_nodes = feature.shape[0]
            global_to_local[node_type] = {i: local_i for local_i, i in enumerate(range(global_counter, global_counter + num_nodes))}
            local_to_global[node_type] = {local_i: i for local_i, i in enumerate(range(global_counter, global_counter + num_nodes))}
            global_counter += num_nodes
        return global_to_local, local_to_global

    def forward(self, features, adj_matrices, adj_dict, dataset, batch_indices=None, history=None, 
                return_contrastive_loss=True, return_intermediates=False):
        # Create mappings between global and local indices
        global_to_local, local_to_global = self.create_global_to_local_mapping(features)

        intermediates = {} if return_intermediates else None

        original_features = {}
        for ntype, feat in features.items():
            original_features[ntype] = feat.clone()

        # Feature extraction
        h = {ntype: self.feature_extractions[ntype](feat) for ntype, feat in features.items()}

        # Store feature extraction results for explainability
        if return_intermediates:
            for ntype, extracted in h.items():
                intermediates[f'extracted_features_{ntype}'] = extracted.detach().clone()

        # Convert adjacency matrices to local indices
        local_adj_matrices = {}
        for adj_key, adj in adj_matrices.items():
            src_type, dst_type = adj_key.split('-')
            edge_index = adj.indices()
            new_edge_index = edge_index.clone()
            
            # Safely map source indices
            for i, idx in enumerate(edge_index[0]):
                idx_val = idx.item()
                if idx_val in global_to_local[src_type]:
                    new_edge_index[0, i] = global_to_local[src_type][idx_val]
                else:
                    closest_idx = min(global_to_local[src_type].keys(), 
                                    key=lambda x: abs(x - idx_val)) if global_to_local[src_type] else 0
                    new_edge_index[0, i] = global_to_local[src_type].get(closest_idx, 0)
            
            # Safely map destination indices
            for i, idx in enumerate(edge_index[1]):
                idx_val = idx.item()
                if idx_val in global_to_local[dst_type]:
                    new_edge_index[1, i] = global_to_local[dst_type][idx_val]
                else:
                    closest_idx = min(global_to_local[dst_type].keys(), 
                                    key=lambda x: abs(x - idx_val)) if global_to_local[dst_type] else 0
                    new_edge_index[1, i] = global_to_local[dst_type].get(closest_idx, 0)
            
            local_adj = torch.sparse_coo_tensor(
                new_edge_index, adj.values(), 
                size=(len(global_to_local[src_type]), len(global_to_local[dst_type])), 
                device=adj.device
            )
            local_adj_matrices[adj_key] = local_adj

        # Store intermediate embeddings for contrastive learning
        node_embeddings = {}
        
        # Multiple layers of message passing
        for layer in range(self.num_layers):
            # Local structural information via multi-scale graph convolution
            h_local = self.multi_scale_convs[layer](h, local_adj_matrices)
            
            # Store multi-scale weights for explainability
            if return_intermediates:
                intermediates[f'scale_weights_layer{layer}'] = self.multi_scale_convs[layer].scale_weights.detach().clone()
                
                # Store scale-specific outputs
                for ntype, scale_outputs in self.multi_scale_convs[layer].scale_outputs.items():
                    for scale_idx, scale_output in enumerate(scale_outputs):
                        intermediates[f'scale{scale_idx}_layer{layer}_{ntype}'] = scale_output.mean(dim=1).detach().clone()

            # Global semantic information via metapath aggregation
            h_global = self.metapath_aggregations[layer](h, adj_dict)
            
            # Store metapath attention weights for explainability
            if return_intermediates:
                intermediates[f'metapath_weights_layer{layer}'] = {}
                for ntype, weights in self.metapath_aggregations[layer].attention_weights.items():
                    intermediates[f'metapath_weights_layer{layer}'][ntype] = weights.detach().clone()
                
                # Also store metapath embeddings
                intermediates[f'metapath_embeddings_layer{layer}'] = {}
                for ntype, mp_data in self.metapath_aggregations[layer].metapath_embeddings.items():
                    intermediates[f'metapath_embeddings_layer{layer}'][ntype] = {
                        'names': mp_data['names'],
                        'embeddings': [emb.mean(dim=1).detach().clone() for emb in mp_data['embeddings']]
                    }

            # Fuse local and global information
            for ntype in self.node_types:
                # Concatenate local and global features
                concat_feat = torch.cat([h_local[ntype], h_global[ntype]], dim=-1)
                
                # Apply fusion module
                fused = self.fusion_modules[layer][ntype](concat_feat)
                
                # Store embeddings for each view for contrastive learning
                if return_contrastive_loss and ntype in ['p', 'c']:
                    for view in range(self.num_views):
                        node_embeddings[(ntype, view, layer)] = fused[:, view, :]
                
                # Apply dropout
                h[ntype] = self.dropout(fused)
            
            # Layer normalization with residual connection
            for ntype in self.node_types:
                h[ntype] = self.layer_norms[layer][ntype](h[ntype] + h_local[ntype])

                if ntype in original_features:
                    if original_features[ntype].dim() != h[ntype].dim():
                        num_nodes = original_features[ntype].shape[0]
                        num_views = h[ntype].shape[1]
                        hidden_dim = h[ntype].shape[2]
                        
                        if not hasattr(self, f'orig_proj_{ntype}'):
                            self.register_buffer(
                                f'orig_proj_{ntype}', 
                                torch.randn(original_features[ntype].shape[1], hidden_dim).to(h[ntype].device) * 0.1
                            )
                        
                        projected = torch.matmul(original_features[ntype], getattr(self, f'orig_proj_{ntype}'))
                        expanded = projected.unsqueeze(1).expand(-1, num_views, -1)
                        
                        alpha = 0.2
                        h[ntype] = (1 - alpha) * h[ntype] + alpha * expanded
                    else:
                        alpha = 0.2
                        h[ntype] = (1 - alpha) * h[ntype] + alpha * original_features[ntype]

                # Store intermediate layer representations for explainability
                if return_intermediates:
                    intermediates[f'layer{layer}_embeddings_{ntype}'] = h[ntype].detach().clone()
        
        # Calculate contrastive loss
        if return_contrastive_loss:
            contrastive_loss = self.contrastive_module(node_embeddings, self.node_types, self.num_views, self.num_layers)

            # Store contrastive learning information for explainability
            if return_intermediates:
                intermediates['contrastive_positive_pairs'] = self.contrastive_module.positive_pairs
                intermediates['contrastive_negative_pairs'] = self.contrastive_module.negative_pairs
                intermediates['contrastive_scores'] = self.contrastive_module.contrast_scores
        else:
            contrastive_loss = torch.tensor(0.0, device=features['p'].device, requires_grad=True)
    
        # Multi-view fusion
        h_fused = {}
        for ntype in self.node_types:
            # Calculate view attention weights
            attention_module = self.view_attention[ntype]
            h_fused[ntype] = attention_module(h[ntype])
            h_fused[ntype] = self.dropout(h_fused[ntype])

            # Store view attention weights for explainability
            if return_intermediates and hasattr(attention_module, 'weights'):
                intermediates[f'view_attention_{ntype}'] = attention_module.weights.detach().clone()

        # Store final embeddings for all nodes
        self.final_embeddings = h_fused

        # Store final embeddings for explainability
        if return_intermediates:
            for ntype, emb in h_fused.items():
                intermediates[f'final_embeddings_{ntype}'] = emb.detach().clone()
            
        # Get drug-drug interaction information
        _, ddi_edge_index = dataset.get_ddi_info()
        ddi_edge_index = ddi_edge_index.to(h_fused['c'].device)
        
        # Enhance drug representation with DDI information
        h_fused['c'] = self.node_enhancement.enhance_drug(h_fused['c'], ddi_edge_index)
        
        # Filter patient embeddings for batch if needed
        h_p = h_fused['p']
        if batch_indices is not None:
            h_p = h_p[batch_indices]
        
        # Enhance patient representation with historical diagnosis information
        if self.use_history and history is not None:
            # Encode patient history
            history_encoded = self.history_encoder(history)
            
            # Enhance patient representation
            h_p = self.node_enhancement.enhance_patient(h_p, history_encoded)

            # Store history information for explainability
            if return_intermediates:
                intermediates['history_encoding'] = history_encoded.detach().clone()
                intermediates['enhanced_patient_embeddings'] = h_p.detach().clone()
       
        scores = self.link_prediction(h_p, h_fused['c']) 

        if return_intermediates:
            return scores, contrastive_loss, intermediates
        else:
            return scores, contrastive_loss
        
        return scores, contrastive_loss 
