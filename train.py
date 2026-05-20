import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
from torch.amp import autocast
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from typing import Tuple, Optional, List

# -------------------------- Data Utilities --------------------------
from data_utils import (
    load_raw_data,
    normalize_data,
    add_enhanced_features,
    build_adjacency_matrix,
    create_memory_efficient_batches as data_utils_create_batches
)

try:
    from torch_optimizer import RAdam
except ImportError:
    RAdam = torch.optim.AdamW


# -------------------------- Data Loading & Preprocessing --------------------------
def load_and_preprocess_data(selected_levels: List[int] = [1, 2, 3]):
    raw_data = load_raw_data(selected_levels)
    if not raw_data:
        raise ValueError("No data loaded")

    for level, data in raw_data.items():
        print(f"Raw data - Level {level} shape: {data.shape} (time_steps, nodes, features)")

    enhanced_data = {
        level: add_enhanced_features(
            data,
            lags=[1, 2, 4],
            window_sizes=[3, 5]
        )
        for level, data in raw_data.items()
    }

    normalized_data, metadata = normalize_data(enhanced_data)

    try:
        with open(os.path.join("tpl_onavg_hierarchy", "metadata.json"), 'r') as f:
            hierarchy_meta = json.load(f)
            for level in normalized_data.keys():
                level_str = str(level)
                if 'level_resolutions' in hierarchy_meta and level_str in hierarchy_meta['level_resolutions']:
                    metadata[level]['resolution'] = hierarchy_meta['level_resolutions'][level_str]
    except Exception as e:
        print(f"Warning: Failed to load metadata.json - {e}")

    print("\n===== Final Metadata Check =====")
    for level in metadata:
        print(f"Level {level}:")
        print(f"  Number of vertices: {metadata[level]['n_vertices']}")
        print(f"  Number of features: {metadata[level]['n_features']}")
        print(f"  Mean: {np.mean(metadata[level]['mean']):.2f}, Std: {np.mean(metadata[level]['std']):.2f}")

    return normalized_data, metadata


# -------------------------- Data Augmentation --------------------------
def augment_data(inputs, noise_scale=0.05):
    augmented = {}
    for level, data in inputs.items():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        batch_size, nodes, timesteps, features = data.shape
        aug_data = data.clone()

        if noise_scale > 0:
            noise = torch.randn_like(aug_data) * noise_scale * torch.std(aug_data, dim=(1, 2, 3), keepdim=True)
            aug_data += noise

        if timesteps > 3:
            shift = torch.randint(-1, 2, (1,)).item()
            if shift != 0:
                aug_data = aug_data.roll(shift, dims=2)
                if shift > 0:
                    aug_data[:, :, :shift, :] = data[:, :, :shift, :]
                else:
                    aug_data[:, :, shift:, :] = data[:, :, shift:, :]

        augmented[level] = aug_data
    return augmented


# -------------------------- Metrics & Denormalization --------------------------
def denormalize_data(data: torch.Tensor, metadata: dict, level: int, epoch: int) -> torch.Tensor:
    data = data.to(dtype=torch.float32)
    mean = torch.tensor(metadata[level]['mean'], device=data.device, dtype=torch.float32)
    std = torch.tensor(metadata[level]['std'], device=data.device, dtype=torch.float32)

    if mean.ndim == 1 and data.ndim == 4:
        mean = mean.reshape(1, 1, 1, -1)
        std = std.reshape(1, 1, 1, -1)

    std = torch.clamp(std, min=1e-4)
    denorm = data * std + mean

    if epoch < 3:
        print(f"Denormalization check - Level {level}:")
        print(f"  Restored mean: {denorm.mean().item():.2f} (Original mean: {mean.mean().item():.2f})")
        print(f"  Restored std: {denorm.std().item():.2f} (Original std: {std.mean().item():.2f})")

    raw_min = torch.tensor(metadata[level]['min'].min(), device=data.device)
    raw_max = torch.tensor(metadata[level]['max'].max(), device=data.device)
    return torch.clamp(denorm, min=raw_min * 0.8, max=raw_max * 1.2)


def calculate_metrics(outputs, targets, metadata, target_level, epoch):
    if outputs.shape[-1] != targets.shape[-1]:
        min_feat = min(outputs.shape[-1], targets.shape[-1])
        outputs = outputs[..., :min_feat]
        targets = targets[..., :min_feat]

    target_std = torch.tensor(metadata[target_level]['std'], device=outputs.device).mean()
    clip_min, clip_max = (-3.0, 3.0) if target_std <= 10000 else (-2.0, 2.0)
    outputs = torch.clamp(outputs, min=clip_min, max=clip_max)

    if outputs.shape[1] != targets.shape[1]:
        min_nodes = min(outputs.shape[1], targets.shape[1])
        outputs = outputs[:, :min_nodes, ...]
        targets = targets[:, :min_nodes, ...]

    if outputs.shape[2] != targets.shape[2]:
        min_time = min(outputs.shape[2], targets.shape[2])
        outputs = outputs[..., :min_time, :]
        targets = targets[..., :min_time, :]

    try:
        outputs_denorm = denormalize_data(outputs, metadata, target_level, epoch)
        targets_denorm = denormalize_data(targets, metadata, target_level, epoch)
    except Exception as e:
        print(f"Denormalization error: {e}")
        return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf'), 'mape': float('inf')}

    outputs_flat = outputs_denorm.reshape(-1)
    targets_flat = targets_denorm.reshape(-1)

    mae = torch.mean(torch.abs(outputs_flat - targets_flat)).item()
    mse = torch.mean((outputs_flat - targets_flat) ** 2).item()
    rmse = np.sqrt(mse) if mse >= 0 else 0.0

    ss_total = torch.sum((targets_flat - torch.mean(targets_flat)) ** 2).item()
    ss_residual = torch.sum((targets_flat - outputs_flat) ** 2).item()
    r2 = 1 - (ss_residual / ss_total) if ss_total > 1e-6 else 0.0
    r2 = max(min(r2, 1.0), -5.0)

    mask = targets_flat.abs() > 1e-3
    valid_count = mask.sum().item()
    mape = torch.mean(
        torch.abs(outputs_flat[mask] - targets_flat[mask]) / (targets_flat[mask].abs() + 1e-6)
    ).item() * 100 if valid_count > 10 else None
    mape = min(mape, 500.0) if mape is not None else None

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


# -------------------------- model structure --------------------------
class GraphMessagePassing(nn.Module):
    def __init__(self, latent_size, num_heads=2, dropout_rate=0.2):
        super().__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.update_net = nn.Sequential(
            nn.Linear(latent_size * 2, latent_size),
            nn.LayerNorm(latent_size),
            nn.SiLU(),
            nn.Linear(latent_size, latent_size),
            nn.Dropout(dropout_rate)
        )

        self.residual_norm = nn.LayerNorm(latent_size)

    def forward(self, x, adj_matrix):
        batch_size, nodes, _ = x.shape
        attn_mask = (adj_matrix == 0).bool()
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)

        aggregated = torch.cat([x, attn_output], dim=-1)
        updated = self.update_net(aggregated)

        return self.residual_norm(x + updated)


class AdaptiveGraphCast(nn.Module):
    def __init__(self, model_cfg, task_cfg, metadata):
        super().__init__()
        self.model_cfg = model_cfg
        self.task_cfg = task_cfg
        self.mesh_levels = model_cfg.mesh_levels
        self.target_level = task_cfg.target_level
        self.metadata = metadata

        self.target_nodes = metadata[task_cfg.target_level]['n_vertices']
        self.target_timesteps = task_cfg.input_duration
        self.latent_size = model_cfg.latent_size
        self.n_features = metadata[task_cfg.target_level]['n_features']

        print(
            f"Initializing model - Target nodes: {self.target_nodes}, Latent size: {self.latent_size}, "
            f"GNN steps: {model_cfg.gnn_msg_steps}, LSTM layers: {model_cfg.lstm_layers}"
        )

        self.feature_proj = nn.ModuleDict({
            str(level): nn.Linear(self.metadata[level]['n_features'], self.latent_size)
            for level in model_cfg.mesh_levels
        })

        fusion_input_dim = len(model_cfg.mesh_levels) * self.latent_size
        self.level_fusion = nn.Linear(fusion_input_dim, self.latent_size)

        self.gnn_layers = nn.ModuleList([
            GraphMessagePassing(
                latent_size=self.latent_size,
                num_heads=model_cfg.num_heads,
                dropout_rate=model_cfg.dropout_rate
            ) for _ in range(model_cfg.gnn_msg_steps)
        ])

        self.temporal_encoder = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.latent_size,
            num_layers=model_cfg.lstm_layers,
            batch_first=True,
            dropout=model_cfg.dropout_rate,
            bidirectional=False
        )

        self.output_proj = nn.Linear(
            self.latent_size,
            self.n_features * task_cfg.predict_duration
        )

        self.aux_proj = nn.ModuleDict({
            str(level): nn.Linear(self.latent_size, metadata[level]['n_features'] * task_cfg.predict_duration)
            for level in model_cfg.mesh_levels if level != task_cfg.target_level
        })

    def forward(self, inputs):
        level_features = []
        for level in self.mesh_levels:
            x = inputs[level]
            batch_size, nodes, timesteps, features = x.shape

            x_reshaped = x.reshape(-1, features)
            x_proj = self.feature_proj[str(level)](x_reshaped)
            x = x_proj.reshape(batch_size, nodes, timesteps, self.latent_size)

            if timesteps != self.target_timesteps:
                x_reshaped = x.reshape(batch_size * nodes, timesteps, self.latent_size)
                x_reshaped = F.interpolate(
                    x_reshaped.unsqueeze(1),
                    size=(self.target_timesteps, self.latent_size),
                    mode='bilinear',
                    align_corners=False
                )
                x = x_reshaped.squeeze(1).reshape(batch_size, nodes, self.target_timesteps, self.latent_size)
                timesteps = self.target_timesteps

            if nodes != self.target_nodes:
                x_reshaped = x.permute(0, 2, 1, 3)
                x_reshaped = F.interpolate(
                    x_reshaped,
                    size=(self.target_nodes, self.latent_size),
                    mode='bilinear',
                    align_corners=False
                )
                x = x_reshaped.permute(0, 2, 1, 3)
                nodes = self.target_nodes

            level_features.append(x)

        fused = torch.cat(level_features, dim=-1)
        fused = self.level_fusion(fused)

        batch_size, nodes, timesteps, latent = fused.shape
        temporal_input = fused.permute(0, 2, 1, 3).reshape(batch_size * timesteps, nodes, latent)

        adj_matrix = torch.ones(nodes, nodes, device=fused.device) - torch.eye(nodes, device=fused.device)

        graph_feat = temporal_input
        for gnn in self.gnn_layers:
            graph_feat = gnn(graph_feat, adj_matrix)

        graph_feat = graph_feat.reshape(batch_size, timesteps, nodes, latent)
        temporal_input = graph_feat.permute(0, 2, 1, 3).reshape(batch_size * nodes, timesteps, latent)
        _, (hidden, _) = self.temporal_encoder(temporal_input)
        last_hidden = hidden[-1]

        outputs = self.output_proj(last_hidden)
        outputs = outputs.reshape(batch_size, nodes, self.task_cfg.predict_duration, self.n_features)

        aux_outputs = {}
        for level in self.aux_proj:
            level_int = int(level)
            level_nodes = self.metadata[level_int]['n_vertices']
            level_features = self.metadata[level_int]['n_features']
            aux_feat = graph_feat

            if aux_feat.shape[1] != level_nodes:
                aux_feat_reshaped = aux_feat.permute(0, 2, 1, 3)
                aux_feat_reshaped = F.interpolate(
                    aux_feat_reshaped,
                    size=(level_nodes, latent),
                    mode='bilinear',
                    align_corners=False
                )
                aux_feat = aux_feat_reshaped.permute(0, 2, 1, 3)

            b_size, n_nodes, t_steps, l_dim = aux_feat.shape
            aux_feat_reshaped = aux_feat.reshape(b_size * n_nodes * t_steps, l_dim)
            aux_out = self.aux_proj[level](aux_feat_reshaped)
            aux_out = aux_out.reshape(b_size, n_nodes, t_steps, self.task_cfg.predict_duration, level_features)
            aux_out = aux_out.mean(dim=2)
            aux_outputs[level_int] = aux_out

        return outputs, aux_outputs


# -------------------------- loss functions --------------------------
def weighted_huber_loss(y_pred, y_true, metadata, target_level, delta=0.5):
    weights = 1.0 / (torch.abs(y_true) + 1e-6)
    weights = weights / weights.mean()
    weights = torch.clamp(weights, min=0.1, max=10.0)

    target_mean = torch.tensor(metadata[target_level]['mean'].mean(), device=y_true.device)
    large_value_mask = torch.abs(y_true) > target_mean
    weights = torch.where(large_value_mask, weights * 1.5, weights)

    error = y_pred - y_true
    abs_error = torch.abs(error)

    loss = torch.where(
        abs_error <= delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )

    return torch.mean(loss * weights)


def multi_scale_loss(outputs, aux_outputs, targets, metadata, alpha=0.7, target_level=2):
    target_batch, target_nodes, target_timesteps, target_features = targets.shape
    output_batch, output_nodes, output_timesteps, output_features = outputs.shape

    if output_features != target_features:
        min_feat = min(output_features, target_features)
        outputs = outputs[..., :min_feat]
        targets = targets[..., :min_feat]

    if output_nodes != target_nodes:
        outputs_reshaped = outputs.permute(0, 3, 1, 2)
        outputs_reshaped = F.interpolate(
            outputs_reshaped,
            size=(target_nodes, output_timesteps),
            mode='bilinear',
            align_corners=False
        )
        outputs = outputs_reshaped.permute(0, 2, 3, 1)

    if outputs.shape[2] != target_timesteps:
        outputs = outputs[:, :, :target_timesteps, :]

    main_loss = weighted_huber_loss(outputs, targets, metadata, target_level)

    aux_loss = 0.0
    aux_count = 0
    total_vertices = sum(metadata[l]['n_vertices'] for l in metadata)

    for level, pred in aux_outputs.items():
        if level not in metadata:
            continue
        n_vertices = metadata[level]['n_vertices']
        level_features = metadata[level]['n_features']

        target_reshaped = targets.permute(0, 3, 1, 2)
        downsampled = F.adaptive_avg_pool2d(
            target_reshaped,
            (n_vertices, target_timesteps)
        )
        downsampled = downsampled.permute(0, 2, 3, 1)

        if pred.shape[-1] != downsampled.shape[-1]:
            min_feat = min(pred.shape[-1], downsampled.shape[-1])
            pred = pred[..., :min_feat]
            downsampled = downsampled[..., :min_feat]

        if pred.shape != downsampled.shape:
            pred = F.adaptive_avg_pool2d(
                pred.permute(0, 3, 1, 2),
                (downsampled.shape[1], downsampled.shape[2])
            ).permute(0, 2, 3, 1)

        level_weight = n_vertices / total_vertices
        aux_loss += weighted_huber_loss(pred, downsampled, metadata, level) * level_weight
        aux_count += 1

    if aux_count > 0:
        return alpha * main_loss + (1 - alpha) * (aux_loss / aux_count)
    else:
        return main_loss


# -------------------------- config classes --------------------------
class GraphCastConfig:
    def __init__(
            self,
            mesh_levels: Tuple[int, ...] = (1, 2, 3),
            latent_size: int = 128,
            gnn_msg_steps: int = 3,
            lstm_layers: int = 2,
            num_heads: int = 2,
            dropout_rate: float = 0.2
    ):
        self.mesh_levels = mesh_levels
        self.latent_size = latent_size
        self.gnn_msg_steps = gnn_msg_steps
        self.lstm_layers = lstm_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate


class TaskConfig:
    def __init__(
            self,
            input_duration: int = 4,
            predict_duration: int = 2,
            target_level: int = 2,
            alpha_loss: float = 0.8
    ):
        self.input_duration = input_duration
        self.predict_duration = predict_duration
        self.target_level = target_level
        self.alpha_loss = alpha_loss


# -------------------------- main function for training (support continue training) --------------------------
def train_model(continue_training: bool = True):  # new argument: continue training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. Load data
        selected_levels = [1, 2, 3]
        hierarchical_data, metadata = load_and_preprocess_data(selected_levels)
        available_levels = sorted(hierarchical_data.keys())
        target_level = 2 if 2 in available_levels else available_levels[0]

        print(f"\nTarget level: {target_level} | Number of nodes: {metadata[target_level]['n_vertices']}")
        for level in available_levels:
            print(f"Level {level}: Number of nodes={metadata[level]['n_vertices']}, Number of features={metadata[level]['n_features']}")

        # 2. Configure parameters
        model_cfg = GraphCastConfig(
            mesh_levels=available_levels,
            latent_size=128,
            gnn_msg_steps=3,
            lstm_layers=2,
            num_heads=2,
            dropout_rate=0.2
        )

        task_cfg = TaskConfig(
            input_duration=4,
            predict_duration=2,
            target_level=target_level,
            alpha_loss=0.8
        )

        # 3. Initialize model + Load weights (core modification)
        model = AdaptiveGraphCast(model_cfg, task_cfg, metadata).to(device)
        start_epoch = 0  # Record start epoch (0 for training from scratch, last epoch for continue training)

        if continue_training and os.path.exists("best_model.pt"):
            try:
                # Load previously saved best model
                model.load_state_dict(torch.load("best_model.pt", map_location=device))
                print("✅ Successfully loaded previous model, will continue training")

                # Try to read the last trained epoch (requires saving history)
                if os.path.exists("training_history.npy"):
                    history = np.load("training_history.npy", allow_pickle=True).item()
                    start_epoch = len(history['train_loss'])  # Continue from last finished epoch
                    print(f"📌 Continue training from Epoch {start_epoch + 1}")
            except Exception as e:
                print(f"❌ Failed to load model, will start training from scratch: {e}")
        else:
            print("ℹ️ No model found to load, will start training from scratch")

        # 4. Configure optimizer and scheduler (reduce initial learning rate for continue training)
        initial_lr = 5e-5 if continue_training else 1e-4  # Reduce learning rate by half for continue training
        optimizer = RAdam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

        # 5. Generate training batches
        train_batches = data_utils_create_batches(
            hierarchical_data,
            input_duration=task_cfg.input_duration,
            predict_duration=task_cfg.predict_duration,
            batch_size=8,
            target_level=target_level,
            shuffle=True,
            sample_ratio=0.6
        )

        val_batches = data_utils_create_batches(
            hierarchical_data,
            input_duration=task_cfg.input_duration,
            predict_duration=task_cfg.predict_duration,
            batch_size=8,
            target_level=target_level,
            shuffle=False,
            sample_ratio=0.3
        )

        train_batches = [b for b in train_batches if next(iter(b[0].values())).shape[0] >= 2]
        val_batches = val_batches if val_batches else [train_batches[0]]
        print(f"Training batches: {len(train_batches)} | Validation batches: {len(val_batches)}")

        # 6. Training settings (load history when continuing training)
        max_epochs = 80
        early_stop_patience = 15
        best_r2 = -float('inf')

        # Load history (when continuing training)
        if continue_training and os.path.exists("training_history.npy"):
            try:
                history = np.load("training_history.npy", allow_pickle=True).item()
                best_r2 = max([m['r2'] for m in history['metrics'] if m['r2'] is not None], default=-float('inf'))
                print(f"📊 Loaded history, current best R²: {best_r2:.4f}")
            except:
                history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        else:
            history = {'train_loss': [], 'val_loss': [], 'metrics': []}

        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

        # 7. Training loop (start from start_epoch)
        print("\n=== Start Training ===")
        for epoch in range(start_epoch, max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            loop = tqdm(train_batches, desc=f"Epoch {epoch + 1}/{max_epochs}")

            for inputs, targets in loop:
                inputs_aug = augment_data(inputs)
                inputs_aug = {k: v.to(device) for k, v in inputs_aug.items()}
                targets = torch.from_numpy(targets).to(device)

                optimizer.zero_grad()
                with autocast('cuda', enabled=device.type == 'cuda'):
                    outputs, aux_outputs = model(inputs_aug)
                    loss = multi_scale_loss(
                        outputs, aux_outputs, targets, metadata,
                        alpha=task_cfg.alpha_loss, target_level=target_level
                    )

                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # Learning rate scheduler update
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            # Validation phase
            model.eval()
            val_loss = 0.0
            metrics_list = []
            with torch.no_grad():
                for inputs, targets in val_batches:
                    inputs_tensor = {k: torch.from_numpy(v).to(device) for k, v in inputs.items()}
                    targets_tensor = torch.from_numpy(targets).to(device)

                    with autocast('cuda', enabled=device.type == 'cuda'):
                        outputs, aux_outputs = model(inputs_tensor)
                        val_loss += multi_scale_loss(
                            outputs, aux_outputs, targets_tensor, metadata,
                            alpha=task_cfg.alpha_loss, target_level=target_level
                        ).item()

                    metrics = calculate_metrics(outputs, targets_tensor, metadata, target_level, epoch)
                    metrics_list.append(metrics)

            # Calculate average metrics
            avg_train_loss = train_loss / len(train_batches) if train_batches else 0
            avg_val_loss = val_loss / len(val_batches) if val_batches else 0

            avg_metrics = {}
            for k in metrics_list[0]:
                values = [m[k] for m in metrics_list if m[k] is not None]
                avg_metrics[k] = np.mean(values) if values else None

            # Record history and save
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['metrics'].append(avg_metrics)
            np.save("training_history.npy", history)  # Save history

            # Gradient checking
            if (epoch + 1) % 5 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Epoch {epoch + 1} Gradient norm: {total_norm:.6f} (Target range: 1e-4 ~ 1e-2)")

            # Save best model
            current_r2 = avg_metrics['r2'] if avg_metrics['r2'] is not None else -float('inf')
            if current_r2 > best_r2:
                best_r2 = current_r2
                torch.save(model.state_dict(), "best_model.pt")
                improved = "✅"
            else:
                improved = "❌"

            # Print results
            print(f"\nEpoch {epoch + 1} | Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")
            print(f"Metrics: MAE={avg_metrics['mae']:.2f}, RMSE={avg_metrics['rmse']:.2f}, "
                  f"R²={avg_metrics['r2']:.4f}, MAPE={avg_metrics['mape']:.2f}% {improved}")

            # Early stopping check
            if len(history['metrics']) > early_stop_patience and \
                    max([m['r2'] for m in history['metrics'][-early_stop_patience:] if m['r2'] is not None],
                        default=-float('inf')) < best_r2:
                print(f"Early stopping: R² has not improved for {early_stop_patience} rounds")
                break

        print(f"\nTraining complete | Best R²: {best_r2:.4f}")

    except Exception as e:
        print(f"Training exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # model continues training by default, to train from scratch, change to train_model(continue_training=False)
    train_model(continue_training=True)