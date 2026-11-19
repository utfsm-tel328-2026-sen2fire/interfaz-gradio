from __future__ import annotations
import torch, torch.nn as nn
from torch import Tensor
from burn_scar.models.convlstm import ConvLSTMCell
from burn_scar.models.segformer import MiniSegFormer

class BurnScarPropagationModel(nn.Module):
    def __init__(self, olci_channels: int = 5, hidden_ch: int = 32, scar_corruption_prob: float = 0.15) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.scar_corruption_prob = scar_corruption_prob
        self.temporal_encoder = ConvLSTMCell(in_channels=olci_channels, hidden_ch=hidden_ch, kernel_size=3)
        self.spatial_head = MiniSegFormer(in_channels=hidden_ch + 4, num_classes=1, embed_dims=(32, 64, 128), num_heads=(1, 2, 4), sr_ratios=(4, 2, 1), depths=(2, 2, 2), decoder_dim=64)

    def forward(self, x_olci: Tensor, x_dem: Tensor, x_worldcover: Tensor, x_scar: Tensor, horizon_days: Tensor) -> Tensor:
        B, T, _, H, W = x_olci.shape
        h = torch.zeros(B, self.hidden_ch, H, W, device=x_olci.device, dtype=x_olci.dtype)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.temporal_encoder(x_olci[:, t], h, c)
        if horizon_days.ndim == 1:
            horizon_days = horizon_days.unsqueeze(1)
        h_scaled = torch.log1p(horizon_days.clamp(min=0.0)) / 2.5
        h_map = h_scaled.view(B, 1, 1, 1).expand(B, 1, H, W)
        return self.spatial_head(torch.cat([h, x_dem, x_worldcover, x_scar, h_map], dim=1))
