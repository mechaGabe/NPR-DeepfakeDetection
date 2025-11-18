"""
Multi-Scale Attention-Based NPR for Deepfake Detection

This module implements an attention-based fusion strategy that combines
NPR (Neural Pixel Rethinking) artifacts extracted at multiple scales.

Key Features:
- Extracts NPR at 3 different scales (0.25x, 0.5x, 0.75x)
- Uses separate ResNet branches for each scale
- Attention mechanism learns to weight different scales adaptively
- Returns attention weights for visualization and analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from .resnet import BasicBlock, Bottleneck, conv1x1


class AttentionFusionModule(nn.Module):
    """
    Attention module that learns to weight features from different scales.

    Args:
        feature_dim: Dimension of input features from each branch
        num_scales: Number of scales to fuse (default: 3)
        reduction: Reduction ratio for the hidden layer (default: 4)
    """
    def __init__(self, feature_dim=128, num_scales=3, reduction=4):
        super(AttentionFusionModule, self).__init__()
        self.num_scales = num_scales

        # Attention network: learns to weight different scales
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // reduction, num_scales),
            nn.Softmax(dim=1)
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of [B, feature_dim] tensors, one per scale

        Returns:
            fused_features: [B, feature_dim] weighted combination
            attention_weights: [B, num_scales] attention scores
        """
        # Concatenate all features
        all_features = torch.cat(features_list, dim=1)  # [B, feature_dim * num_scales]

        # Compute attention weights
        attention_weights = self.attention_net(all_features)  # [B, num_scales]

        # Weighted fusion
        fused = torch.zeros_like(features_list[0])
        for i, features in enumerate(features_list):
            fused += attention_weights[:, i:i+1] * features

        return fused, attention_weights


class NPRExtractor(nn.Module):
    """
    Extracts NPR (Neural Pixel Rethinking) artifacts at a specific scale.

    NPR is computed as: NPR = x - interpolate(interpolate(x, scale), 1/scale)
    This captures upsampling artifacts left by generative models.
    """
    def __init__(self, scale_factor=0.5):
        super(NPRExtractor, self).__init__()
        self.scale_factor = scale_factor

    def interpolate(self, img, factor):
        """Downsample then upsample to extract artifacts."""
        downsampled = F.interpolate(img, scale_factor=factor, mode='nearest',
                                    recompute_scale_factor=True)
        upsampled = F.interpolate(downsampled, scale_factor=1/factor, mode='nearest',
                                 recompute_scale_factor=True)
        return upsampled

    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            NPR artifacts [B, 3, H, W]
        """
        reconstructed = self.interpolate(x, self.scale_factor)
        npr = x - reconstructed
        return npr


class MultiScaleResNet(nn.Module):
    """
    Multi-scale ResNet branch for processing NPR at a specific scale.

    This is a lighter version of ResNet that serves as a feature extractor
    for each scale. Uses only 2 layers to keep model size manageable.
    """
    def __init__(self, block, layers, feature_dim=128):
        super(MultiScaleResNet, self).__init__()
        self.inplanes = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (lighter than original - only 2 layers)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # Global average pooling and feature projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(128 * block.expansion, feature_dim)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, x):
        """
        Args:
            x: NPR artifacts [B, 3, H, W]

        Returns:
            features: [B, feature_dim]
        """
        # Apply 2/3 scaling as in original NPR paper
        x = x * (2.0 / 3.0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_proj(x)

        return x


class AttentionMultiScaleNPR(nn.Module):
    """
    Complete Attention-Based Multi-Scale NPR Model for Deepfake Detection.

    Architecture:
        Input Image
            ↓
        [NPR@0.25x, NPR@0.5x, NPR@0.75x]  ← Extract artifacts at 3 scales
            ↓           ↓          ↓
        ResNet₁     ResNet₂    ResNet₃    ← Separate feature extractors
            ↓           ↓          ↓
        feat₁       feat₂      feat₃      ← Features [B, 128]
            ↓           ↓          ↓
            → Attention Module ←           ← Learn scale weights
                    ↓
            Fused Features [B, 128]
                    ↓
              Classifier → Real/Fake

    Args:
        num_classes: Number of output classes (default: 1 for binary)
        scales: List of scale factors for NPR extraction
        feature_dim: Dimension of features from each branch
    """
    def __init__(self, num_classes=1, scales=[0.25, 0.5, 0.75], feature_dim=128):
        super(AttentionMultiScaleNPR, self).__init__()

        self.num_classes = num_classes
        self.scales = scales
        self.num_scales = len(scales)
        self.feature_dim = feature_dim

        # NPR extractors for each scale
        self.npr_extractors = nn.ModuleList([
            NPRExtractor(scale_factor=scale) for scale in scales
        ])

        # Feature extraction branches (one ResNet per scale)
        # Using BasicBlock with [2, 2] layers (lighter than ResNet18's [2,2,2,2])
        self.feature_branches = nn.ModuleList([
            MultiScaleResNet(BasicBlock, [2, 2], feature_dim=feature_dim)
            for _ in range(self.num_scales)
        ])

        # Attention fusion module
        self.attention_fusion = AttentionFusionModule(
            feature_dim=feature_dim,
            num_scales=self.num_scales,
            reduction=4
        )

        # Final classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input images [B, 3, H, W]
            return_attention: If True, return attention weights for visualization

        Returns:
            output: Classification logits [B, num_classes]
            attention_weights: (optional) [B, num_scales] attention scores
        """
        # Extract NPR at each scale
        npr_features = []
        for i, npr_extractor in enumerate(self.npr_extractors):
            npr = npr_extractor(x)  # [B, 3, H, W]
            npr_features.append(npr)

        # Extract features from each branch
        branch_features = []
        for i, branch in enumerate(self.feature_branches):
            features = branch(npr_features[i])  # [B, feature_dim]
            branch_features.append(features)

        # Attention-based fusion
        fused_features, attention_weights = self.attention_fusion(branch_features)

        # Classification
        output = self.classifier(fused_features)

        if return_attention:
            return output, attention_weights
        else:
            return output

    def get_scale_contributions(self, x):
        """
        Analyze which scales contribute most to the prediction.
        Useful for understanding model behavior on different generators.

        Returns:
            Dictionary with NPR visualizations and attention weights
        """
        self.eval()
        with torch.no_grad():
            # Extract NPR at each scale
            npr_maps = {}
            for i, (scale, npr_extractor) in enumerate(zip(self.scales, self.npr_extractors)):
                npr = npr_extractor(x)
                npr_maps[f'npr_{scale}'] = npr

            # Get attention weights
            output, attention_weights = self.forward(x, return_attention=True)

            return {
                'npr_maps': npr_maps,
                'attention_weights': attention_weights,
                'prediction': output
            }


def attention_multiscale_npr18(**kwargs):
    """
    Constructs an Attention-based Multi-Scale NPR model with ResNet18-style branches.

    Default configuration:
    - 3 scales: 0.25x, 0.5x, 0.75x
    - Feature dimension: 128
    - Lightweight branches: BasicBlock with [2, 2] layers
    """
    model = AttentionMultiScaleNPR(**kwargs)
    return model


def attention_multiscale_npr18_custom(scales=[0.25, 0.5, 0.75], **kwargs):
    """
    Custom multi-scale NPR with user-defined scales.

    Example:
        # Test different scale combinations
        model = attention_multiscale_npr18_custom(scales=[0.33, 0.5, 0.67])
    """
    model = AttentionMultiScaleNPR(scales=scales, **kwargs)
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing Attention-Based Multi-Scale NPR Model...")

    model = attention_multiscale_npr18(num_classes=1)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output, attention = model(dummy_input, return_attention=True)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention weights (sample):\n{attention}")

    # Test scale contribution analysis
    analysis = model.get_scale_contributions(dummy_input)
    print(f"\nScale contribution analysis keys: {analysis.keys()}")
    print("Model test successful!")
