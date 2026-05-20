from typing import Tuple


class GraphCastConfig:
    def __init__(
            self,
            mesh_levels: Tuple[int, ...] = (2, 3),
            latent_size: int = 128,
            gnn_msg_steps: int = 4,
            num_heads: int = 1,
            dropout_rate: float = 0.3,
            use_tpl_onavg: bool = True,
            coupling_init: float = 0.8,
            coupling_min: float = 0.1,
            coupling_max: float = 1.5,
            hidden_dim: int = 32  # New parameter: hidden dimension for TinyEncoder
    ):
        self.mesh_levels = mesh_levels
        self.latent_size = latent_size
        self.gnn_msg_steps = gnn_msg_steps
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_tpl_onavg = use_tpl_onavg
        self.coupling_init = coupling_init
        self.coupling_min = coupling_min
        self.coupling_max = coupling_max
        self.hidden_dim = hidden_dim  # New: store the parameter as a class attribute


class TaskConfig:
    def __init__(
            self,
            input_duration: int = 4,
            predict_duration: int = 2,
            target_level: int = 2,
            alpha_loss: float = 0.7  # Primary loss weight
    ):
        self.input_duration = input_duration
        self.predict_duration = predict_duration
        self.target_level = target_level
        self.alpha_loss = alpha_loss