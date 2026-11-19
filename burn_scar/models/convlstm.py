"""ConvLSTM temporal encoder + 2D backbone wrapper.

:class:`ConvLSTMCell`
    Single convolutional LSTM cell operating on spatial feature maps.

:class:`TemporalContextModel`
    Wraps any 2D segmentation backbone with a ConvLSTM temporal encoder.
    Processes the full image sequence to produce per-timestep hidden states,
    then concatenates each hidden state with the corresponding frame before
    the backbone.

    Input:  ``(B, T, C, H, W)``
    Output: ``(B, T, num_classes, H, W)``

    Use with ``mode="3d_sequence"`` in :class:`~bafep.modules.MeanTeacherModule`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    Args:
        in_channels: Input feature channels.
        hidden_ch:   Hidden/cell state channels.
        kernel_size: Spatial kernel size (default 3).
    """

    def __init__(self, in_channels: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(
            in_channels + hidden_ch,
            4 * hidden_ch,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """Single recurrent step.

        Args:
            x: ``(B, in_channels, H, W)``
            h: ``(B, hidden_ch, H, W)`` — previous hidden state
            c: ``(B, hidden_ch, H, W)`` — previous cell state

        Returns:
            ``(h_new, c_new)``
        """
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch: int, h: int, w: int, device: torch.device) -> tuple[Tensor, Tensor]:
        z = torch.zeros(batch, self.hidden_ch, h, w, device=device)
        return z, z.clone()


class TemporalContextModel(nn.Module):
    """ConvLSTM temporal encoder + 2D segmentation backbone.

    Runs a ConvLSTM over the image sequence to produce per-timestep hidden
    states ``(B, T, D, H, W)``.  For each timestep ``t`` the hidden state is
    concatenated with the original frame and passed to the 2D backbone.

    The backbone's ``in_channels`` must equal ``in_channels + hidden_ch``.

    Args:
        backbone:    2D segmentation model accepting ``(B, C+D, H, W)``.
        in_channels: Spectral bands in the input (C).
        hidden_ch:   ConvLSTM hidden channels (D).
        kernel_size: ConvLSTM conv kernel size.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 22,
        hidden_ch: int = 64,
        kernel_size: int = 3,
        use_pre_post_flag: bool = False,
        use_day_night_flag: bool = False,
        use_change_features: bool = False,
    ) -> None:
        super().__init__()
        n_flags = int(use_pre_post_flag) + int(use_day_night_flag)
        self.cell = ConvLSTMCell(in_channels + n_flags, hidden_ch, kernel_size)
        self.backbone = backbone
        self.use_pre_post_flag = use_pre_post_flag
        self.use_day_night_flag = use_day_night_flag
        self.use_change_features = use_change_features

    def forward(
        self,
        x: Tensor,
        flag: Tensor | None = None,
        day_night_flag: Tensor | None = None,
        change: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:              ``(B, T, C, H, W)`` or ``(B, C, H, W)`` for single-frame
            flag:           ``(B, T)`` float — 0 = pre-fire, 1 = post-fire.
            day_night_flag: ``(B, T)`` float — 0 = day, 1 = night.
            change:         ``(B, C, H, W)`` mean(post) − mean(pre) difference image.

        Returns:
            logits: ``(B, T, num_classes, H, W)`` or ``(B, num_classes, H, W)``
        """
        squeeze = x.dim() == 4
        if squeeze:
            x = x.unsqueeze(1)
            if flag is not None:
                flag = flag.unsqueeze(1)
            if day_night_flag is not None:
                day_night_flag = day_night_flag.unsqueeze(1)
        b, t, c, h, w = x.shape
        hid, cell = self.cell.init_hidden(b, h, w, x.device)

        hidden_states: list[Tensor] = []
        for step in range(t):
            frame = x[:, step]
            flag_chs: list[Tensor] = []
            if self.use_pre_post_flag and flag is not None:
                flag_chs.append(flag[:, step].view(b, 1, 1, 1).expand(b, 1, h, w))
            if self.use_day_night_flag and day_night_flag is not None:
                flag_chs.append(day_night_flag[:, step].view(b, 1, 1, 1).expand(b, 1, h, w))
            if flag_chs:
                frame = torch.cat([frame] + flag_chs, dim=1)
            hid, cell = self.cell(frame, hid, cell)
            hidden_states.append(hid)

        hiddens = torch.stack(hidden_states, dim=1)

        frames = x.view(b * t, c, h, w)
        ctx = hiddens.reshape(b * t, hiddens.shape[2], h, w)
        backbone_in = torch.cat([frames, ctx], dim=1)
        if self.use_change_features and change is not None:
            change_t = change.unsqueeze(1).expand(b, t, -1, h, w)
            backbone_in = torch.cat(
                [backbone_in, change_t.reshape(b * t, c, h, w)],
                dim=1,
            )
        out = self.backbone(backbone_in)

        out = out.view(b, t, out.shape[1], h, w)
        return out[:, 0] if squeeze else out
