import os
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from typing import Dict, Optional


class TemplateManager:
    """Manages loading and applying TPL-ONAVG templates, including accurate template generation."""

    def __init__(self, hierarchical_data: Dict, data_dir="tpl_onavg_templates"):
        self.templates = {}
        self.data_dir = data_dir
        self.hierarchical_data = hierarchical_data
        self.levels = list(hierarchical_data.keys())

        # Store number of vertices for each level
        self.node_counts = {level: data['n_vertices'] for level, data in hierarchical_data.items()}

        # Ensure directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Delete old templates and regenerate
        self.regenerate_all_templates()

        print("\n=== Level Node Information ===")
        for level, count in self.node_counts.items():
            print(f"Level {level}: {count} nodes")

    def regenerate_all_templates(self):
        """Delete all old templates and regenerate them."""
        print("Deleting all old templates and regenerating...")

        # Remove all existing template files
        for filename in os.listdir(self.data_dir):
            if filename.startswith("tpl_") and filename.endswith(".npz"):
                os.remove(os.path.join(self.data_dir, filename))

        # Regenerate all templates
        for source_level in self.levels:
            for target_level in self.levels:
                if source_level == target_level:
                    continue
                self.generate_accurate_template(source_level, target_level)

    def generate_accurate_template(self, source_level: int, target_level: int):
        """Generate accurate TPL-ONAVG template using distance-based weighted interpolation."""

        # Verify level existence
        if source_level not in self.hierarchical_data or target_level not in self.hierarchical_data:
            print(f"Cannot generate template, missing level data: {source_level} -> {target_level}")
            return None

        # Get source and target meshes
        source_data = self.hierarchical_data[source_level]
        target_data = self.hierarchical_data[target_level]

        # Ensure vertices are fully loaded
        source_vertices = np.array(source_data['vertices'])
        target_vertices = np.array(target_data['vertices'])

        # Validate vertex count
        if source_vertices.shape[0] != source_data['n_vertices']:
            print(
                f"Warning: Level {source_level} vertex count mismatch! "
                f"Declared: {source_data['n_vertices']}, Actual: {source_vertices.shape[0]}"
            )
            source_data['n_vertices'] = source_vertices.shape[0]

        if target_vertices.shape[0] != target_data['n_vertices']:
            print(
                f"Warning: Level {target_level} vertex count mismatch! "
                f"Declared: {target_data['n_vertices']}, Actual: {target_vertices.shape[0]}"
            )
            target_data['n_vertices'] = target_vertices.shape[0]

        # Check vertex validity
        if source_vertices.shape[0] == 0:
            print(f"Error: Level {source_level} has 0 vertices, skipping template generation")
            return None

        if target_vertices.shape[0] == 0:
            print(f"Error: Level {target_level} has 0 vertices, skipping template generation")
            return None

        print(
            f"Generating template: Source Level {source_level} "
            f"({source_vertices.shape[0]} vertices) -> "
            f"Target Level {target_level} ({target_vertices.shape[0]} vertices)"
        )

        # Compute distance matrix
        try:
            dist_matrix = cdist(target_vertices, source_vertices)
        except MemoryError:
            print(f"Out of memory computing distance matrix {source_level}->{target_level}, using fallback interpolation")
            return None

        # Find k nearest neighbors
        k = min(4, source_vertices.shape[0])
        knn_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        knn_dists = np.take_along_axis(dist_matrix, knn_indices, axis=1)

        # Compute inverse-distance weights
        weights = 1.0 / (knn_dists + 1e-6)
        weights /= np.sum(weights, axis=1, keepdims=True)

        # Build sparse matrix
        rows = np.repeat(np.arange(target_vertices.shape[0]), k)
        cols = knn_indices.flatten()
        data = weights.flatten()

        shape = (target_vertices.shape[0], source_vertices.shape[0])
        matrix = sp.coo_matrix((data, (rows, cols)), shape=shape)

        # Save template
        filename = f"tpl_{source_level}_{target_level}.npz"
        filepath = os.path.join(self.data_dir, filename)

        np.savez(
            filepath,
            data=matrix.data,
            row=matrix.row,
            col=matrix.col,
            shape=matrix.shape
        )

        self.templates[(source_level, target_level)] = matrix
        print(f"Generated accurate template: {source_level} -> {target_level}, shape: {matrix.shape}")
        return matrix

    def get_template(self, source_level: int, target_level: int) -> Optional[sp.coo_matrix]:
        """Get template between specified levels."""
        key = (source_level, target_level)
        return self.templates.get(key, None)

    def apply_template(self, source_data: torch.Tensor, template: sp.coo_matrix,
                       device: torch.device, target_level: int) -> torch.Tensor:
        """
        Apply TPL-ONAVG template for feature propagation (supports batch).

        Args:
            source_data: Source level data [batch_size, source_nodes, features]
            template: Sparse template matrix
            device: Compute device
            target_level: Target level ID
        """

        if template is None:
            # Use simple interpolation if template is missing
            target_size = self.node_counts.get(target_level, 1000)  # default 1000 nodes
            return F.interpolate(
                source_data.permute(0, 2, 1),
                size=target_size,
                mode='linear'
            ).permute(0, 2, 1)

        if source_data.dtype != torch.float32:
            source_data = source_data.to(torch.float32)

        indices = np.vstack((template.row, template.col))
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        values = torch.tensor(template.data, dtype=torch.float32, device=device)
        shape = template.shape  # [target_nodes, source_nodes]

        if shape[0] <= 0 or shape[1] <= 0:
            print(f"Error: Invalid template shape {shape}, using fallback interpolation")
            target_size = self.node_counts.get(target_level, 1000)
            return F.interpolate(
                source_data.permute(0, 2, 1),
                size=target_size,
                mode='linear'
            ).permute(0, 2, 1)

        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)

        batch_size, source_nodes, features = source_data.shape
        target_nodes = shape[0]

        if source_nodes != shape[1]:
            print(f"Critical Warning: Template dimension mismatch!")
            print(f"Source nodes: {source_nodes}, Template expects: {shape[1]}")
            print(f"Template shape: {shape}, Input shape: {source_data.shape}")

            if source_nodes < shape[1]:
                padding = torch.zeros(
                    batch_size,
                    shape[1] - source_nodes,
                    features,
                    device=device,
                    dtype=source_data.dtype
                )
                source_data = torch.cat([source_data, padding], dim=1)
                source_nodes = shape[1]
            else:
                source_data = source_data[:, :shape[1], :]
                source_nodes = shape[1]

        result = torch.zeros(batch_size, target_nodes, features,
                             device=device, dtype=source_data.dtype)

        try:
            dense_tensor = sparse_tensor.to_dense()
            input_reshaped = source_data.permute(0, 2, 1)
            result = torch.matmul(input_reshaped, dense_tensor.t())
            result = result.permute(0, 2, 1)

        except RuntimeError as e:
            print(f"Dense matrix multiplication failed: {str(e)}, falling back to loop method")
            for b in range(batch_size):
                input_batch = source_data[b]
                result_b = torch.sparse.mm(sparse_tensor, input_batch)
                result[b] = result_b

        if source_data.dtype != torch.float32:
            result = result.to(source_data.dtype)

        return result