"""
Multi-Scale NPR with Attention-Based Fusion
For H2: Attention-Based Scale Selection

This architecture computes NPR at multiple scales [0.25, 0.5, 0.75] and uses
an attention mechanism to automatically weight them based on input characteristics.

Hypothesis: Attention fusion will improve detection accuracy by 3-7% over fixed-scale NPR.

Author: [Your Name]
Date: November 2024
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import numpy as np


__all__ = ['ResNetMultiScale', 'resnet50_multiscale']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ScaleAttention(nn.Module):
    """
    Attention mechanism to weight different NPR scales

    Architecture:
        Input: Original image (3 channels)
        → Global Average Pooling
        → FC layers (3 → 16 → num_scales)
        → Softmax
        Output: Attention weights for each scale

    This learns which scales are most informative for a given input.
    """
    def __init__(self, num_scales=3, hidden_dim=16):
        super(ScaleAttention, self).__init__()
        self.num_scales = num_scales

        # Attention network: learns to weight scales based on input
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Flatten(),
            nn.Linear(3, hidden_dim),  # 3 input channels (RGB)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=1)  # Output: [batch_size, num_scales]
        )

    def forward(self, x):
        """
        Args:
            x: input image [B, 3, H, W]
        Returns:
            weights: attention weights [B, num_scales, 1, 1, 1]
        """
        weights = self.attention(x)  # [B, num_scales]
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, num_scales, 1, 1, 1]
        return weights


class ResNetMultiScale(nn.Module):
    """
    Multi-Scale NPR with Attention Fusion

    Key Innovation:
    1. Compute NPR at multiple scales (0.25, 0.5, 0.75)
    2. Use attention to weight scales based on input
    3. Fuse weighted NPR maps
    4. Feed through ResNet backbone

    This allows the network to adaptively focus on the most discriminative scale
    for each input, rather than using a fixed scale.
    """

    def __init__(self, block, layers, num_classes=1, npr_scales=[0.25, 0.5, 0.75],
                 fusion_mode='attention', zero_init_residual=False):
        """
        Args:
            block: BasicBlock or Bottleneck
            layers: list of layer sizes
            num_classes: number of output classes
            npr_scales: list of NPR interpolation factors
            fusion_mode: 'attention', 'concat', or 'average'
                - attention: learn to weight scales (H2)
                - concat: concatenate all scales (baseline)
                - average: simple average (baseline)
            zero_init_residual: zero-initialize residual connections
        """
        super(ResNetMultiScale, self).__init__()

        self.npr_scales = npr_scales
        self.fusion_mode = fusion_mode

        print(f"[INFO] Multi-Scale NPR with scales: {npr_scales}")
        print(f"[INFO] Fusion mode: {fusion_mode}")

        # Attention mechanism (for H2)
        if fusion_mode == 'attention':
            self.scale_attention = ScaleAttention(num_scales=len(npr_scales))
            input_channels = 3  # Attention-weighted fusion → 3 channels
        elif fusion_mode == 'concat':
            input_channels = 3 * len(npr_scales)  # Concatenate all scales
        elif fusion_mode == 'average':
            input_channels = 3  # Simple average → 3 channels
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize

        self.inplanes = 64

        # Modified first conv to accept multi-scale input
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        """
        Downsample and upsample to compute NPR

        Args:
            img: input image tensor [B, C, H, W]
            factor: downsampling factor
        Returns:
            reconstructed image after down-up sampling
        """
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
            scale_factor=1/factor,
            mode='nearest',
            recompute_scale_factor=True
        )

    def compute_multi_scale_npr(self, x):
        """
        Compute NPR at multiple scales

        Args:
            x: input image [B, 3, H, W]
        Returns:
            npr_maps: list of NPR maps, one per scale [B, 3, H, W] each
        """
        npr_maps = []
        for scale in self.npr_scales:
            npr = x - self.interpolate(x, scale)
            npr_maps.append(npr)
        return npr_maps

    def fuse_npr_maps(self, x, npr_maps):
        """
        Fuse NPR maps using the specified fusion mode

        Args:
            x: original image [B, 3, H, W] (for attention)
            npr_maps: list of NPR maps [B, 3, H, W] each
        Returns:
            fused NPR map [B, C, H, W] where C depends on fusion_mode
        """
        if self.fusion_mode == 'attention':
            # Learn to weight scales based on input
            weights = self.scale_attention(x)  # [B, num_scales, 1, 1, 1]

            # Stack NPR maps: [B, num_scales, 3, H, W]
            stacked_nprs = torch.stack(npr_maps, dim=1)

            # Weighted sum: [B, 3, H, W]
            fused = torch.sum(stacked_nprs * weights, dim=1)

            return fused

        elif self.fusion_mode == 'concat':
            # Concatenate all scales: [B, 3*num_scales, H, W]
            return torch.cat(npr_maps, dim=1)

        elif self.fusion_mode == 'average':
            # Simple average: [B, 3, H, W]
            return torch.stack(npr_maps, dim=0).mean(dim=0)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def forward(self, x):
        """
        Forward pass with multi-scale NPR and attention fusion

        Pipeline:
        1. Compute NPR at multiple scales
        2. Fuse using attention weights
        3. Feed through ResNet backbone
        4. Binary classification (real vs fake)
        """
        # Step 1: Compute NPR at each scale
        npr_maps = self.compute_multi_scale_npr(x)

        # Step 2: Fuse NPR maps (attention-weighted or baseline)
        fused_npr = self.fuse_npr_maps(x, npr_maps)

        # Step 3: Feed through ResNet backbone
        out = self.conv1(fused_npr * 2.0/3.0)  # Same normalization as original
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

    def get_attention_weights(self, x):
        """
        Get attention weights for visualization/analysis

        Args:
            x: input image [B, 3, H, W]
        Returns:
            weights: [B, num_scales] attention weights
        """
        if self.fusion_mode != 'attention':
            return None

        with torch.no_grad():
            weights = self.scale_attention(x).squeeze(-1).squeeze(-1).squeeze(-1)
        return weights


def resnet50_multiscale(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model with multi-scale NPR and attention

    Usage:
        # H2: Attention-based fusion (recommended)
        model = resnet50_multiscale(fusion_mode='attention')

        # Baseline: Concatenation
        model = resnet50_multiscale(fusion_mode='concat')

        # Baseline: Simple average
        model = resnet50_multiscale(fusion_mode='average')
    """
    model = ResNetMultiScale(Bottleneck, [3, 4, 6, 3], **kwargs)

    # Note: Can't load ImageNet pretrained weights directly since input channels differ
    # Would need to train from scratch or use transfer learning approach

    return model


# Example usage for debugging
if __name__ == '__main__':
    print("Testing Multi-Scale NPR with Attention...")

    # Create model with attention fusion
    model = resnet50_multiscale(num_classes=1, fusion_mode='attention')

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"✓ Output shape: {output.shape}")  # Should be [2, 1]

    # Test attention weights
    weights = model.get_attention_weights(x)
    print(f"✓ Attention weights shape: {weights.shape}")  # Should be [2, 3]
    print(f"✓ Attention weights (should sum to 1):\n{weights}")

    # Test different fusion modes
    for mode in ['concat', 'average']:
        model = resnet50_multiscale(num_classes=1, fusion_mode=mode)
        output = model(x)
        print(f"✓ {mode} mode - Output shape: {output.shape}")

    print("\n✓ All tests passed!")
