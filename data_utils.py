import numpy as np
import os
import h5py
from scipy.sparse import coo_matrix, csr_matrix
from typing import Dict, List, Tuple, Optional


def build_adjacency_matrix(triangles: np.ndarray, n_vertices: int) -> np.ndarray:
    """Build a symmetrically normalized adjacency matrix from triangular faces"""
    if triangles.size == 0:
        return np.eye(n_vertices, dtype=np.float32)

    edges = set()
    for tri in triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            if v1 != v2:
                if v1 < v2:
                    edges.add((v1, v2))
                else:
                    edges.add((v2, v1))

    edges = list(edges)
    if not edges:
        return np.eye(n_vertices, dtype=np.float32)

    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.float32)

    # Add self-loops
    self_loops = np.arange(n_vertices)
    rows = np.concatenate([rows, self_loops])
    cols = np.concatenate([cols, self_loops])
    data = np.concatenate([data, np.ones(n_vertices, dtype=np.float32)])

    # Build adjacency matrix
    adj = coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    adj = adj.tocsr()

    # Symmetric normalization
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero
    sqrt_degrees = np.sqrt(degrees)
    inv_sqrt_degrees = 1.0 / sqrt_degrees

    adj = adj.multiply(inv_sqrt_degrees[:, np.newaxis])
    adj = adj.multiply(inv_sqrt_degrees[np.newaxis, :])

    return adj.toarray().astype(np.float32)


def load_raw_data(selected_levels: List[int] = [1, 2, 3]) -> Dict[int, np.ndarray]:
    """Load raw hierarchical data and explicitly print shapes"""
    data_dir = "tpl_onavg_hierarchy"
    raw_data = {}
    h5_path = os.path.join(data_dir, "data.h5")

    if not os.path.exists(h5_path):
        print(f"Error: Data file not found: {h5_path}")
        return raw_data

    with h5py.File(h5_path, 'r') as hf:
        for group_name in hf.keys():
            if group_name.startswith("level_"):
                try:
                    level = int(group_name.split("_")[1])
                    if level not in selected_levels:
                        continue
                except (ValueError, IndexError):
                    continue

                if "function" in hf[group_name]:
                    func_data = hf[group_name]["function"][:]

                    if func_data.ndim == 2:
                        func_data = func_data[..., np.newaxis]

                    if len(func_data.shape) == 3:
                        timesteps, nodes, features = func_data.shape
                        print(
                            f"Raw data - Level {level}: "
                            f"shape=(timesteps={timesteps}, nodes={nodes}, features={features})"
                        )
                        print(
                            f"           Stats: mean={np.mean(func_data):.2f}, "
                            f"std={np.std(func_data):.2f}, "
                            f"min={np.min(func_data):.2f}, "
                            f"max={np.max(func_data):.2f}"
                        )
                        raw_data[level] = func_data
                    else:
                        print(f"Warning: Invalid data shape at level {level}: {func_data.shape}")

    missing = [l for l in selected_levels if l not in raw_data]
    if missing:
        print(f"Warning: Data not found for levels {missing}")

    return raw_data


def add_enhanced_features(data: np.ndarray,
                          lags: List[int] = [1, 2],
                          window_sizes: List[int] = [3]) -> np.ndarray:
    """Add temporal enhancement features to time-series data"""
    timesteps, nodes, features = data.shape
    enhanced_features = [data]

    # Add lag features
    for lag in lags:
        if lag < timesteps:
            lagged = np.zeros_like(data)
            lagged[lag:] = data[:-lag]
            lagged[:lag] = data[0]
            enhanced_features.append(lagged)

    # Add rolling statistics
    for window in window_sizes:
        if window < timesteps:
            rolling_mean = np.zeros_like(data)
            for t in range(timesteps):
                start = max(0, t - window + 1)
                rolling_mean[t] = data[start:t + 1].mean(axis=0)
            enhanced_features.append(rolling_mean)

            rolling_std = np.zeros_like(data)
            for t in range(timesteps):
                start = max(0, t - window + 1)
                window_data = data[start:t + 1]
                rolling_std[t] = window_data.std(axis=0) + 1e-8
            enhanced_features.append(rolling_std)

    result = np.concatenate(enhanced_features, axis=-1)
    print(f"Feature enhancement: {features} → {result.shape[-1]} dimensions")
    return result


def normalize_data(hierarchical_data: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Normalize data and correctly record node counts"""
    normalized_data = {}
    metadata = {}

    for level, data in hierarchical_data.items():
        timesteps, nodes, features = data.shape
        print(f"Normalization - Level {level}: "
              f"processing shape=(timesteps={timesteps}, nodes={nodes}, features={features})")

        data_reshaped = data.reshape(-1, features)
        q1 = np.percentile(data_reshaped, 25, axis=0, keepdims=True)
        q3 = np.percentile(data_reshaped, 75, axis=0, keepdims=True)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        lower_bound = q1 - 3 * iqr
        data_clipped = np.clip(data, lower_bound, upper_bound)

        mean = data_clipped.mean(axis=(0, 1), keepdims=True)
        std = data_clipped.std(axis=(0, 1), keepdims=True) + 1e-8

        normalized = (data - mean) / std

        norm_mean = np.mean(normalized)
        norm_std = np.std(normalized)
        print(f"Normalization check - Level {level}: "
              f"mean={norm_mean:.4f}, std={norm_std:.4f}")

        metadata[level] = {
            'mean': mean.squeeze(),
            'std': std.squeeze(),
            'min': data_clipped.min(axis=(0, 1)),
            'max': data_clipped.max(axis=(0, 1)),
            'n_vertices': nodes,
            'n_features': features,
            'timesteps': timesteps
        }

        print(f"Metadata - Level {level}: nodes={nodes}, features={features}")
        normalized_data[level] = normalized.astype(np.float32)

    return normalized_data, metadata


def create_memory_efficient_batches(
        hierarchical_data: Dict[int, np.ndarray],
        input_duration: int,
        predict_duration: int,
        batch_size: int = 8,
        target_level: int = 2,
        shuffle: bool = True,
        sample_ratio: float = 1.0
) -> List[Tuple[Dict[int, np.ndarray], np.ndarray]]:
    """Generate memory-efficient batches with consistent node counts"""

    level_info = {}
    for level, data in hierarchical_data.items():
        timesteps, nodes, features = data.shape
        level_info[level] = {
            'timesteps': timesteps,
            'nodes': nodes,
            'features': features
        }

    print(f"Level info: { {k: {'timesteps': v['timesteps'], 'nodes': v['nodes']} for k, v in level_info.items()} }")

    target_nodes = level_info[target_level]['nodes']
    print(f"Target level {target_level} node count: {target_nodes}")

    target_timesteps = level_info[target_level]['timesteps']
    max_start_target = target_timesteps - input_duration - predict_duration + 1
    if max_start_target <= 0:
        raise ValueError("Insufficient timesteps for target level")

    start_indices = np.arange(max_start_target)
    if shuffle:
        np.random.shuffle(start_indices)

    if sample_ratio < 1.0:
        n_samples = int(len(start_indices) * sample_ratio)
        start_indices = start_indices[:n_samples]

    valid_indices = {}
    for level in hierarchical_data:
        max_start = level_info[level]['timesteps'] - input_duration - predict_duration + 1
        if max_start <= 0:
            print(f"Warning: Level {level} has insufficient timesteps and will be skipped")
            valid_indices[level] = []
        else:
            valid_indices[level] = start_indices[start_indices < max_start]
    if not valid_indices:
        raise ValueError("Error: No valid indices found")

    common_indices = valid_indices[next(iter(valid_indices.keys()))]
    for level in valid_indices:
        common_indices = np.intersect1d(common_indices, valid_indices[level])

    if len(common_indices) == 0:
        raise ValueError("No common valid time indices found across all levels")

    print(f"Valid common time indices: {len(common_indices)}")

    batches = []
    n_batches = len(common_indices) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        batch_indices = common_indices[start:end]

        input_batch = {}
        for level, data in hierarchical_data.items():
            if len(valid_indices[level]) == 0:
                continue

            batch_data = []
            for idx in batch_indices:
                slice_data = data[idx:idx + input_duration, :, :]
                batch_data.append(slice_data)

            stacked = np.stack(batch_data, axis=0)
            input_batch[level] = stacked.transpose(0, 2, 1, 3)   # (batch, nodes, timesteps, features)

        target_batch_data = []
        target_data = hierarchical_data[target_level]
        for idx in batch_indices:
            slice_data = target_data[
                idx + input_duration:
                idx + input_duration + predict_duration, :, :
            ]
            target_batch_data.append(slice_data)

        target_batch = np.stack(target_batch_data, axis=0)
        target_batch = target_batch.transpose(0, 2, 1, 3)   # (batch, nodes, timesteps, features)

        if b == 0:
            print(f"Batch check - input nodes: "
                  f"{ {k: v.shape[1] for k, v in input_batch.items()} }")
            print(f"             target nodes: {target_batch.shape[1]}")
            if target_batch.shape[1] != target_nodes:
                print(f"Warning: Target batch node count does not match target level!")

        batches.append((input_batch, target_batch))

    print(f"Generated batches: {len(batches)} (batch size={batch_size})")
    return batches


def load_data(
        selected_levels: List[int] = [1, 2, 3],
        input_duration: int = 4,
        predict_duration: int = 2,
        batch_size: int = 8,
        target_level: int = 2,
        sample_ratio: float = 1.0
) -> Tuple[List[Tuple[Dict[int, np.ndarray], np.ndarray]], Dict[int, dict]]:
    """Full data loading pipeline"""

    raw_data = load_raw_data(selected_levels)
    if not raw_data:
        raise ValueError("No data loaded. Please check the dataset.")

    enhanced_data = {level: add_enhanced_features(data) for level, data in raw_data.items()}
    normalized_data, metadata = normalize_data(enhanced_data)

    train_batches = create_memory_efficient_batches(
        normalized_data,
        input_duration=input_duration,
        predict_duration=predict_duration,
        batch_size=batch_size,
        target_level=target_level,
        shuffle=True,
        sample_ratio=sample_ratio
    )

    return train_batches, metadata