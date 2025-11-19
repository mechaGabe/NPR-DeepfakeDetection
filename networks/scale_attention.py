"""
Scale Attention Module for Multi-Scale NPR
Adapted from HW1 2D_UNet.py attention mechanisms

This module learns to weight different NPR scales based on input characteristics.
For H2: Attention-Based Scale Selection hypothesis
"""

import torch
import torch.nn as nn
from typing import List


class ScaleAttention(nn.Module):
    """
    Lightweight attention to weight NPR scales

    Architecture:
    1. Global average pooling to get image-level features
    2. MLP to predict scale importance weights
    3. Softmax to normalize weights to sum to 1

    Input: Original image [B, 3, H, W]
    Output: Weights for each scale [B, num_scales]

    This is simpler than the full LinearAttention from HW1, but captures
    the key idea: let the network learn which scales are important for
    each input image.
    """

    def __init__(self, num_scales: int = 3, hidden_dim: int = 16):
        """
        Args:
            num_scales: Number of NPR scales to weight (e.g., 3 for [0.25, 0.5, 0.75])
            hidden_dim: Hidden layer dimension for the attention MLP
        """
        super().__init__()

        self.num_scales = num_scales

        # Simple MLP attention network
        # Takes image -> global features -> scale weights
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # [B, 3, H, W] -> [B, 3, 1, 1]
            nn.Flatten(),                    # [B, 3, 1, 1] -> [B, 3]
            nn.Linear(3, hidden_dim),        # [B, 3] -> [B, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales), # [B, hidden_dim] -> [B, num_scales]
            nn.Softmax(dim=1)                 # Weights sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for NPR scales

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            weights: Scale attention weights [B, num_scales]
                    Each row sums to 1.0
        """
        return self.attention_net(x)


class MultiScaleNPRFusion(nn.Module):
    """
    Combines multiple NPR representations using learned attention weights

    This module:
    1. Computes NPR at multiple scales
    2. Gets attention weights from ScaleAttention module
    3. Performs weighted fusion of NPR maps

    This is the core of H2 hypothesis testing.
    """

    def __init__(self, npr_scales: List[float] = [0.25, 0.5, 0.75]):
        """
        Args:
            npr_scales: List of NPR interpolation factors to use
        """
        super().__init__()

        self.npr_scales = npr_scales
        self.num_scales = len(npr_scales)

        # Attention module to weight scales
        self.scale_attention = ScaleAttention(num_scales=self.num_scales)

    def compute_npr(self, x: torch.Tensor, factor: float) -> torch.Tensor:
        """
        Compute NPR at a given scale
        Same as original NPR paper implementation

        Args:
            x: Input image [B, 3, H, W]
            factor: Interpolation factor (e.g., 0.5)

        Returns:
            NPR map [B, 3, H, W]
        """
        import torch.nn.functional as F

        # Downsample and upsample (same as networks/resnet.py:153-154)
        x_down = F.interpolate(
            x,
            scale_factor=factor,
            mode='nearest',
            recompute_scale_factor=True
        )
        x_up = F.interpolate(
            x_down,
            scale_factor=1/factor,
            mode='nearest',
            recompute_scale_factor=True
        )

        # Compute residual
        return x - x_up

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass: compute multi-scale NPR and fuse with attention

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            npr_fused: Weighted NPR map [B, 3, H, W]
            weights: Attention weights [B, num_scales] (for visualization)
        """
        # Step 1: Compute NPR at each scale
        npr_maps = []
        for scale in self.npr_scales:
            npr = self.compute_npr(x, scale)
            npr_maps.append(npr)

        # Step 2: Get attention weights based on input
        weights = self.scale_attention(x)  # [B, num_scales]

        # Step 3: Weighted fusion
        # Stack NPR maps: [B, num_scales, 3, H, W]
        stacked_nprs = torch.stack(npr_maps, dim=1)

        # Reshape weights for broadcasting: [B, num_scales, 1, 1, 1]
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Weighted sum: [B, 3, H, W]
        npr_fused = (stacked_nprs * weights_expanded).sum(dim=1)

        return npr_fused, weights

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper to get just the attention weights (for analysis)

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            weights: Scale attention weights [B, num_scales]
        """
        return self.scale_attention(x)
