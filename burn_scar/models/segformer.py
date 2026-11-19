"""SegFormer-inspired hierarchical ViT for small-image segmentation.

Adapted from the Mix Transformer (MiT) architecture with modifications for
small input sizes (e.g. 64×64 S3OLCI patches):

* Three stages, all with stride-2 overlapping patch embeddings (vs. the
  original stride-4 first stage), so a 64×64 input produces token grids of
  32×32 → 16×16 → 8×8 — avoiding excessive spatial collapse.
* Efficient Self-Attention (SR-Attention) with configurable sequence-reduction
  ratios per stage.
* Mix-FFN with a 3×3 depth-wise conv for local context (no positional encoding).
* All-MLP decoder: each scale is projected to ``decoder_dim``, upsampled to the
  finest grid, concatenated, and mapped to class logits via a linear head, then
  bilinearly upsampled to the original resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _OverlapPatchEmbed(nn.Module):
    """Strided convolution patch embedding with LayerNorm."""

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)  # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'W', C)
        return self.norm(x), H, W


class _EfficientAttention(nn.Module):
    """Multi-head self-attention with sequence reduction (SR-Attention).

    key/value sequences are spatially pooled by ``sr_ratio`` before attending,
    reducing cost from O(N²) to O(N·N/sr²).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(embed_dim, embed_dim, sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _MixFFN(nn.Module):
    """Feed-forward network with a 3×3 depth-wise conv for local positional context."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        x = self.drop(self.fc1(x))
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dw(x)
        x = F.gelu(x.reshape(B, -1, N).transpose(1, 2))
        x = self.drop(self.fc2(x))
        return self.drop(x)


class _MixTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _EfficientAttention(embed_dim, num_heads, sr_ratio, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = _MixFFN(embed_dim, mlp_ratio, drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class _MixTransformerStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        depth: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = _OverlapPatchEmbed(in_ch, embed_dim, patch_size=3, stride=2)
        self.blocks = nn.ModuleList(
            [_MixTransformerBlock(embed_dim, num_heads, sr_ratio, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x, H, W = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        return self.norm(x), H, W


class _TemporalAttention(nn.Module):
    """Per-location self-attention across T frames.

    Takes the token tensor ``(B*T, N, C)`` produced by an encoder stage and
    mixes information across the time axis.  Each of the N spatial positions
    attends independently over its T temporal counterparts — O(T²) cost,
    independent of N.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, drop: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)

    def forward(self, x: Tensor, T: int) -> Tensor:
        BT, N, C = x.shape
        B = BT // T
        # (B*T, N, C) → (B*N, T, C): each spatial token attends over time
        x_t = x.view(B, T, N, C).permute(0, 2, 1, 3).reshape(B * N, T, C)
        normed = self.norm(x_t)
        attn_out, _ = self.attn(normed, normed, normed)
        x_t = x_t + attn_out
        # (B*N, T, C) → (B*T, N, C)
        return x_t.reshape(B, N, T, C).permute(0, 2, 1, 3).reshape(BT, N, C)


class MiniSegFormer(nn.Module):
    """SegFormer-inspired hierarchical ViT for small-image segmentation.

    Three transformer stages with stride-2 overlapping patch embeddings
    produce token grids of H/2 × H/4 × H/8 (e.g. 32×32 → 16×16 → 8×8 for
    a 64×64 input) — avoiding the spatial collapse of the original stride-4
    first stage on sub-100 px imagery.

    The All-MLP decoder follows SegFormer: each scale is linearly projected to
    ``decoder_dim``, upsampled to the finest grid (H/2), concatenated, fused
    by a single linear layer, and mapped to class logits.  The output is
    bilinearly upsampled to the original input resolution.

    When ``temporal_attention=True`` the model additionally accepts a sequence
    input ``(B, T, in_channels, H, W)`` and returns ``(B, T, num_classes, H, W)``.
    A lightweight per-stage temporal self-attention block mixes information
    across frames after each encoder stage, sharing all spatial weights across
    time.  Use with ``mode="3d_sequence"`` in the training module.
    When ``temporal_attention=False`` (default) the interface is unchanged:
    ``(B, in_channels, H, W)`` → ``(B, num_classes, H, W)``.

    Args:
        in_channels:          Number of input spectral bands.
        num_classes:          Number of output classes.
        embed_dims:           Channel dims for the three encoder stages
                              (default: ``(32, 64, 160)`` — MiT-B0 style).
        num_heads:            Attention heads per stage (must divide embed_dim).
        sr_ratios:            Sequence-reduction ratio per stage; 1 = no reduction.
        depths:               Transformer blocks per stage.
        decoder_dim:          Projection dim in the all-MLP decoder.
        drop_rate:            Dropout applied in FFN and decoder.
        attn_drop:            Dropout on attention weights.
        temporal_attention:   Enable per-stage temporal self-attention.
        temporal_num_heads:   Attention heads for the temporal blocks (default 1).
    """

    def __init__(
        self,
        in_channels: int = 22,
        num_classes: int = 3,
        embed_dims: tuple[int, ...] = (32, 64, 160),
        num_heads: tuple[int, ...] = (1, 2, 5),
        sr_ratios: tuple[int, ...] = (4, 2, 1),
        depths: tuple[int, ...] = (2, 2, 2),
        decoder_dim: int = 128,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
        temporal_attention: bool = False,
        temporal_num_heads: int = 1,
    ) -> None:
        super().__init__()
        assert len(embed_dims) == len(num_heads) == len(sr_ratios) == len(depths), (
            "embed_dims, num_heads, sr_ratios, depths must all have the same length"
        )
        self._num_stages = len(embed_dims)
        self.temporal_attention = temporal_attention

        self.stages = nn.ModuleList()
        in_ch = in_channels
        for i in range(self._num_stages):
            self.stages.append(
                _MixTransformerStage(
                    in_ch=in_ch,
                    embed_dim=embed_dims[i],
                    num_heads=num_heads[i],
                    sr_ratio=sr_ratios[i],
                    depth=depths[i],
                    drop=drop_rate,
                    attn_drop=attn_drop,
                )
            )
            in_ch = embed_dims[i]

        if temporal_attention:
            self.temp_attn = nn.ModuleList(
                [_TemporalAttention(d, num_heads=temporal_num_heads, drop=drop_rate) for d in embed_dims]
            )

        # All-MLP decoder
        self.decode_proj = nn.ModuleList([nn.Linear(d, decoder_dim) for d in embed_dims])
        self.decode_fuse = nn.Sequential(
            nn.Linear(decoder_dim * self._num_stages, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )
        self.seg_head = nn.Linear(decoder_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if self.temporal_attention and x.dim() == 5:
            return self._forward_temporal(x)
        return self._forward_spatial(x)

    def _forward_spatial(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape

        feats: list[tuple[Tensor, int, int]] = []
        for stage in self.stages:
            x, h, w = stage(x)
            feats.append((x, h, w))
            x = x.transpose(1, 2).reshape(B, -1, h, w)

        h0, w0 = feats[0][1], feats[0][2]
        decoded: list[Tensor] = []
        for i, (feat, h, w) in enumerate(feats):
            feat = self.decode_proj[i](feat)  # (B, N, decoder_dim)
            feat = feat.transpose(1, 2).reshape(B, -1, h, w)
            if h != h0 or w != w0:
                feat = F.interpolate(feat, size=(h0, w0), mode="bilinear", align_corners=False)
            decoded.append(feat)

        fused = torch.cat(decoded, dim=1)  # (B, decoder_dim * S, h0, w0)
        fused = fused.flatten(2).transpose(1, 2)  # (B, h0*w0, decoder_dim * S)
        fused = self.decode_fuse(fused)  # (B, h0*w0, decoder_dim)
        logits = self.seg_head(fused)  # (B, h0*w0, num_classes)
        logits = logits.transpose(1, 2).reshape(B, -1, h0, w0)

        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

    def _forward_temporal(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        BT = B * T
        x = x.view(BT, C, H, W)

        feats: list[tuple[Tensor, int, int]] = []
        for i, stage in enumerate(self.stages):
            x, h, w = stage(x)  # (BT, N, C)
            x = self.temp_attn[i](x, T)  # mix across frames
            feats.append((x, h, w))
            x = x.transpose(1, 2).reshape(BT, -1, h, w)

        h0, w0 = feats[0][1], feats[0][2]
        decoded: list[Tensor] = []
        for i, (feat, h, w) in enumerate(feats):
            feat = self.decode_proj[i](feat)  # (BT, N, decoder_dim)
            feat = feat.transpose(1, 2).reshape(BT, -1, h, w)
            if h != h0 or w != w0:
                feat = F.interpolate(feat, size=(h0, w0), mode="bilinear", align_corners=False)
            decoded.append(feat)

        fused = torch.cat(decoded, dim=1)
        fused = fused.flatten(2).transpose(1, 2)
        fused = self.decode_fuse(fused)
        logits = self.seg_head(fused)
        logits = logits.transpose(1, 2).reshape(BT, -1, h0, w0)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits.view(B, T, -1, H, W)
