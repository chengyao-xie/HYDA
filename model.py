import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
import os
from scipy.sparse import coo_matrix


# 自定义维度转置层（兼容低版本PyTorch）
class PermuteLayer(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(self.order)


# 从数据加载代码中导入邻接矩阵构建函数
from data_loader import build_adjacency_matrix  # 确保data_loader.py在同一目录


# 物理模块（保持原逻辑，适配邻接矩阵输入）
class EncoderDecoderWilsonCowanLayer(nn.Module):
    """物理约束的图卷积层（使用邻接矩阵）"""

    def __init__(self, n_nodes: int, adj_matrix: coo_matrix, device: torch.device,
                 config, hidden_dim: int):
        super().__init__()
        self.num_nodes = n_nodes
        self.device = device
        self.hidden_dim = hidden_dim

        # 将稀疏邻接矩阵转换为PyTorch稀疏张量
        adj = adj_matrix.tocoo()
        indices = torch.LongTensor(np.vstack((adj.row, adj.col))).to(device)
        values = torch.FloatTensor(adj.data).to(device)
        self.adj = torch.sparse_coo_tensor(indices, values, adj.shape, device=device)

        # 图卷积层
        self.conv = nn.Linear(hidden_dim, hidden_dim, device=device)
        self.norm = nn.BatchNorm1d(hidden_dim, device=device)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x形状: (batch, nodes, hidden_dim)
        batch_size = x.shape[0]

        # 图卷积: 邻接矩阵 × 特征
        x_agg = torch.sparse.mm(self.adj, x.permute(1, 0, 2).reshape(self.num_nodes, -1))
        x_agg = x_agg.reshape(self.num_nodes, batch_size, self.hidden_dim).permute(1, 0, 2)

        # 线性变换+激活
        x_out = self.conv(x_agg)
        x_out = self.norm(x_out.permute(0, 2, 1)).permute(0, 2, 1)  # 节点维度归一化
        return self.activation(x_out)


class PhysicallyConstrainedEncoder(nn.Module):
    """增强版编码器：保持原结构"""

    def __init__(self, input_steps: int, latent_size: int, device: torch.device, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_steps, latent_size * 4),  # 加宽第一层
            nn.BatchNorm1d(latent_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 4, latent_size * 2),
            nn.BatchNorm1d(latent_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size * 2, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size, 4)  # 输出4维特征
        ).to(device)

        # 权重初始化优化
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='gelu')
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape  # (batch, nodes, steps*features)
        x = x.reshape(B * N, T)  # 展平为 (B*N, T)
        x = self.encoder(x)
        return x.reshape(B, N, 4)  # 恢复为 (batch, nodes, 4)


class TemplateUpsampler(nn.Module):
    """修改上采样器：适配自定义TemplateManager的apply_template方法"""

    def __init__(self, source_level: int, target_level: int, template_manager,
                 device: torch.device, log_enabled: bool = False, log_freq: int = 50):
        super().__init__()
        self.source_level = source_level
        self.target_level = target_level
        self.template_manager = template_manager  # 使用用户提供的TemplateManager
        self.device = device
        self.log_enabled = log_enabled
        self.log_freq = log_freq
        self.call_count = 0

        # 从模板管理器获取预生成的模板
        self.template = self.template_manager.get_template(source_level, target_level)
        if self.template is None:
            print(f"警告: 未找到 {source_level}→{target_level} 的模板，将在运行时生成")
            # 触发模板生成
            self.template = self.template_manager.generate_accurate_template(source_level, target_level)

        self._validate_template()

    def _validate_template(self):
        """验证模板形状是否正确"""
        if self.template is None:
            return

        source_nodes = self.template_manager.node_counts[self.source_level]
        target_nodes = self.template_manager.node_counts[self.target_level]
        if self.template.shape != (target_nodes, source_nodes):
            raise ValueError(
                f"模板 {self.source_level}→{self.target_level} 形状无效: "
                f"实际 {self.template.shape}, 期望 ({target_nodes}, {source_nodes})"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.call_count += 1

        if self.log_enabled and (self.call_count % self.log_freq == 0):
            print(f"\n[模板应用] 源层级 {self.source_level} → 目标层级 {self.target_level}")
            print(f"输入形状: {x.shape} | 模板形状: {self.template.shape if self.template else 'None'}")

        # 直接调用TemplateManager的apply_template方法处理上采样
        return self.template_manager.apply_template(
            source_data=x,
            template=self.template,
            device=self.device,
            target_level=self.target_level
        )


class EnhancedHierarchicalProcessor(nn.Module):
    """增强版层级处理器：适配自定义TemplateManager"""

    def __init__(self, config, hierarchical_data: Dict, device: torch.device,
                 template_manager, log_enabled: bool = False, log_freq: int = 50,
                 dropout: float = 0.4):
        super().__init__()
        self.device = device
        self.config = config
        self.template_manager = template_manager  # 使用用户提供的TemplateManager
        self.hierarchical_data = hierarchical_data
        self.log_enabled = log_enabled
        self.log_freq = log_freq
        self.dropout = nn.Dropout(dropout)

        # 初始化处理器和采样器
        self.processors = nn.ModuleDict()
        self.upsamplers = nn.ModuleDict()
        self.downsamplers = nn.ModuleDict()

        levels = sorted(config.mesh_levels)
        for level in levels:
            if level not in hierarchical_data:
                continue

            # 物理处理器（使用从数据加载的面片构建邻接矩阵）
            level_data = hierarchical_data[level]
            n_vertices = level_data['n_vertices']
            faces = level_data['faces']  # 从数据加载的面片信息
            adj_matrix = build_adjacency_matrix(faces, n_vertices)  # 使用数据加载中的函数

            self.processors[str(level)] = EncoderDecoderWilsonCowanLayer(
                n_vertices, adj_matrix, device, config,
                hidden_dim=config.hidden_dim * 2  # 加宽隐藏层
            )

            # 跨层级采样器（使用用户的TemplateManager）
            for target_level in levels:
                if target_level > level:
                    key = f"{level}_{target_level}"
                    self.upsamplers[key] = TemplateUpsampler(
                        level, target_level, template_manager, device,
                        log_enabled=log_enabled, log_freq=log_freq
                    )

            for source_level in levels:
                if source_level > level:
                    key = f"{source_level}_{level}"
                    self.downsamplers[key] = TemplateUpsampler(
                        source_level, level, template_manager, device,
                        log_enabled=log_enabled, log_freq=log_freq
                    )

        # 改进残差门控（增加非线性）
        self.gate_layers = nn.ModuleDict()
        for level in levels:
            if level in hierarchical_data:
                self.gate_layers[str(level)] = nn.Sequential(
                    nn.Linear(4, 4),  # 适配4维输入
                    nn.GELU(),
                    nn.Linear(4, 4, bias=False),
                    nn.Sigmoid()
                ).to(device)

    def forward(self, encoded_states: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        processed_states = {}
        previous_states = {k: v for k, v in encoded_states.items()}

        # 层级内计算
        for level, state in encoded_states.items():
            level_str = str(level)
            if level_str in self.processors:
                expected_nodes = self.processors[level_str].num_nodes
                actual_nodes = state.shape[1]
                if actual_nodes != expected_nodes:
                    if actual_nodes < expected_nodes:
                        padding = torch.zeros(
                            state.shape[0], expected_nodes - actual_nodes, state.shape[2],
                            device=self.device, dtype=state.dtype
                        )
                        state = torch.cat([state, padding], dim=1)
                    else:
                        state = state[:, :expected_nodes, :]

                processed = self.processors[level_str](state)
                processed = self.dropout(processed)
                processed_states[level] = processed

        # 跨层级交互（增加消息传递步数）
        for step in range(self.config.gnn_msg_steps):
            new_states = {}
            for level, state in processed_states.items():
                aggregated = state.clone()

                # 聚合低层级特征（加权融合）
                for source_level in processed_states.keys():
                    if source_level < level and f"{source_level}_{level}" in self.upsamplers:
                        upsampled = self.upsamplers[f"{source_level}_{level}"](processed_states[source_level])
                        if upsampled.shape[1] == state.shape[1]:
                            # 动态权重（基于特征方差）
                            var_source = torch.var(upsampled, dim=(0, 1), keepdim=True)
                            var_target = torch.var(state, dim=(0, 1), keepdim=True)
                            weight = var_source / (var_source + var_target + 1e-8)
                            aggregated += upsampled * weight

                # 聚合高层级特征（加权融合）
                for source_level in processed_states.keys():
                    if source_level > level and f"{source_level}_{level}" in self.downsamplers:
                        downsampled = self.downsamplers[f"{source_level}_{level}"](processed_states[source_level])
                        if downsampled.shape[1] == state.shape[1]:
                            var_source = torch.var(downsampled, dim=(0, 1), keepdim=True)
                            var_target = torch.var(state, dim=(0, 1), keepdim=True)
                            weight = var_source / (var_source + var_target + 1e-8)
                            aggregated += downsampled * weight

                # 再次处理并应用残差门控
                level_str = str(level)
                if level_str in self.processors:
                    new_state = self.processors[level_str](aggregated)
                    new_state = self.dropout(new_state)

                    # 改进残差连接（动态门控）
                    if level in previous_states:
                        gate = self.gate_layers[level_str](aggregated)
                        new_state = gate * new_state + (1 - gate) * previous_states[level]

                    new_states[level] = new_state

            processed_states = new_states

        return processed_states


class AdaptiveGraphCast(nn.Module):
    """优化后的主模型：适配自定义TemplateManager"""

    def __init__(self, config, task_cfg, hierarchical_data: Dict, device: torch.device,
                 log_template: bool = False, template_log_freq: int = 50):
        super().__init__()
        self.device = device
        self.target_level = task_cfg.target_level
        self.predict_duration = task_cfg.predict_duration
        self.latent_size = config.latent_size
        self.hierarchical_data = hierarchical_data

        # 初始化用户提供的TemplateManager（不再自己实现）
        self.template_manager = TemplateManager(hierarchical_data)  # 使用用户的TemplateManager
        self.validate_templates()

        # 编码器（适配增强特征）
        self.encoders = nn.ModuleDict()
        for level in config.mesh_levels:
            if level in hierarchical_data:
                # 输入维度 = 输入时长 × 增强后的特征数（由data_utils控制）
                self.encoders[str(level)] = PhysicallyConstrainedEncoder(
                    task_cfg.input_duration * hierarchical_data[level]['n_features'],
                    config.latent_size,
                    device,
                    dropout=0.4
                )

        # 层级处理器
        self.processor = EnhancedHierarchicalProcessor(
            config, hierarchical_data, device, self.template_manager,
            log_enabled=log_template, log_freq=template_log_freq,
            dropout=0.4
        )

        # 增强特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(8, config.latent_size),  # 4（目标）+4（上下文）=8维输入
            PermuteLayer((0, 2, 1)),  # (batch, nodes, latent) → (batch, latent, nodes)
            nn.BatchNorm1d(config.latent_size),
            PermuteLayer((0, 2, 1)),  # 转回原形状
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(config.latent_size, config.latent_size),  # 增加一层特征转换
            nn.GELU()
        ).to(device)

        # 单层级适配器
        self.single_level_adapter = nn.Sequential(
            nn.Linear(4, config.latent_size),
            PermuteLayer((0, 2, 1)),
            nn.BatchNorm1d(config.latent_size),
            PermuteLayer((0, 2, 1)),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(config.latent_size, config.latent_size),
            nn.GELU()
        ).to(device)

        # 增强解码器
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_size, config.latent_size * 4),  # 加宽解码器
            nn.BatchNorm1d(config.latent_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_size * 4, config.latent_size * 2),
            nn.BatchNorm1d(config.latent_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.latent_size * 2, task_cfg.predict_duration)
        ).to(device)

        # 最终上采样器
        self.final_upsampler = None
        max_level = max(config.mesh_levels)
        if max_level > self.target_level:
            self.final_upsampler = TemplateUpsampler(
                self.target_level, max_level, self.template_manager, device,
                log_enabled=log_template, log_freq=template_log_freq
            )

        # 多尺度辅助解码器（增强正则化）
        self.aux_decoders = nn.ModuleDict()
        for level in config.mesh_levels:
            if level != self.target_level and level in hierarchical_data:
                self.aux_decoders[str(level)] = nn.Sequential(
                    nn.Linear(4, config.latent_size),
                    nn.GELU(),
                    nn.Linear(config.latent_size, config.latent_size // 2),
                    nn.GELU(),
                    nn.Linear(config.latent_size // 2, task_cfg.predict_duration)
                ).to(device)

        # 模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"模型总参数: {total_params:,}")

    def validate_templates(self):
        """验证所有必要的模板是否存在"""
        print("\n=== 模板验证 ===")
        valid = True
        levels = sorted(self.hierarchical_data.keys())
        for source_level in levels:
            for target_level in levels:
                if source_level == target_level:
                    continue
                template = self.template_manager.get_template(source_level, target_level)
                if template is None:
                    print(f"警告: 缺少 {source_level}→{target_level} 的模板，将在首次使用时生成")
                else:
                    source_nodes = self.template_manager.node_counts[source_level]
                    target_nodes = self.template_manager.node_counts[target_level]
                    if template.shape != (target_nodes, source_nodes):
                        print(
                            f"错误: 模板 {source_level}→{target_level} 形状不匹配: {template.shape} vs ({target_nodes}, {source_nodes})")
                        valid = False
        if valid:
            print("所有模板验证通过")
        else:
            print("模板验证存在问题，请检查数据或重新生成模板")

    def forward(self, x_dict: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        # 编码所有层级
        encoded_states = {}
        for level, x in x_dict.items():
            if str(level) in self.encoders:
                encoded_states[level] = self.encoders[str(level)](x)

        # 层级处理
        processed = self.processor(encoded_states)

        # 提取目标层级特征
        if self.target_level in processed:
            target_state = processed[self.target_level]  # (batch, nodes, 4)
        else:
            target_state = processed[max(processed.keys())]

        # 多尺度特征融合（增强上下文利用）
        if len(processed) > 1:
            context_list = []
            for level, state in processed.items():
                if level != self.target_level:
                    # 上下文特征池化（保留空间分布）
                    pooled = F.adaptive_avg_pool1d(
                        state.permute(0, 2, 1), target_state.shape[1]
                    ).permute(0, 2, 1)  # 对齐目标层级节点数
                    context_list.append(pooled)
            context = torch.mean(torch.stack(context_list), dim=0)  # 多尺度平均
            fused = torch.cat([target_state, context], dim=-1)  # (batch, nodes, 8)
            fused = self.fusion_layer(fused)  # (batch, nodes, latent_size)
        else:
            fused = self.single_level_adapter(target_state)

        # 解码器处理
        B, N, F = fused.shape
        fused_flat = fused.reshape(B * N, F)
        decoded_flat = self.decoder(fused_flat)
        decoded = decoded_flat.reshape(B, N, -1)

        # 最终上采样
        if self.final_upsampler:
            decoded = self.final_upsampler(decoded)

        # 辅助输出（多尺度监督）
        aux_outputs = {}
        for level, state in processed.items():
            if level != self.target_level and str(level) in self.aux_decoders:
                aux_outputs[level] = self.aux_decoders[str(level)](state)

        return decoded, aux_outputs
