"""
Multi-Scale NPR ResNet with Attention
For H2: Attention-Based Scale Selection hypothesis

Combines:
- NPR ResNet backbone (from networks/resnet.py)
- Multi-scale NPR processing
- Scale attention (from networks/scale_attention.py)

Key innovation: Instead of using a fixed NPR scale (0.5), we compute
NPR at multiple scales and learn to weight them adaptively using attention.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from networks.resnet import BasicBlock, Bottleneck, conv1x1
from networks.scale_attention import MultiScaleNPRFusion


class ResNetMultiScale(nn.Module):
    """
    Multi-scale NPR ResNet with attention fusion

    Architecture Flow:
    1. Input image [B, 3, H, W]
    2. Multi-scale NPR computation at [0.25, 0.5, 0.75]
    3. Attention-based fusion of NPR maps
    4. ResNet backbone for classification
    5. Output: fake/real prediction [B, 1]

    This tests H2: Can attention-based scale selection improve
    detection over fixed single-scale NPR?
    """

    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1,
        npr_scales: List[float] = [0.25, 0.5, 0.75],
        zero_init_residual: bool = False
    ):
        """
        Args:
            block: ResNet block type (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer
            num_classes: Number of output classes (1 for binary classification)
            npr_scales: List of NPR interpolation factors
            zero_init_residual: Whether to zero-init residual connections
        """
        super(ResNetMultiScale, self).__init__()

        self.npr_scales = npr_scales

        # Multi-scale NPR fusion module (with attention)
        self.npr_fusion = MultiScaleNPRFusion(npr_scales=npr_scales)

        # ResNet backbone (same as original NPR paper)
        # We reuse the exact same architecture, just without the NPR preprocessing
        self.inplanes = 64

        # Initial convolution (input: fused NPR map)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights(zero_init_residual)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        """
        Create a ResNet layer (copied from networks/resnet.py)
        """
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

    def _initialize_weights(self, zero_init_residual: bool):
        """
        Initialize model weights (same as original ResNet)
        """
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

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with multi-scale NPR and attention

        Args:
            x: Input image [B, 3, H, W]
            return_attention: If True, also return attention weights

        Returns:
            output: Classification logits [B, 1]
            (optional) attention_weights: Scale weights [B, num_scales]
        """
        # Step 1: Multi-scale NPR with attention fusion
        npr_fused, attention_weights = self.npr_fusion(x)

        # Step 2: Apply NPR normalization (same as original: *2.0/3.0)
        npr_normalized = npr_fused * (2.0 / 3.0)

        # Step 3: ResNet backbone
        x = self.conv1(npr_normalized)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc1(x)

        if return_attention:
            return output, attention_weights
        return output

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper to get attention weights for a batch of images

        Useful for:
        - Visualizing which scales the model focuses on
        - Analyzing whether GANs and diffusion get different weights
        - Understanding model behavior

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            weights: Attention weights [B, num_scales]
        """
        return self.npr_fusion.get_attention_weights(x)


# Factory functions (same pattern as networks/resnet.py)

def resnet18_multiscale(num_classes: int = 1, npr_scales: List[float] = [0.25, 0.5, 0.75], **kwargs):
    """ResNet-18 with multi-scale NPR attention"""
    return ResNetMultiScale(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, npr_scales=npr_scales, **kwargs)


def resnet34_multiscale(num_classes: int = 1, npr_scales: List[float] = [0.25, 0.5, 0.75], **kwargs):
    """ResNet-34 with multi-scale NPR attention"""
    return ResNetMultiScale(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, npr_scales=npr_scales, **kwargs)


def resnet50_multiscale(num_classes: int = 1, npr_scales: List[float] = [0.25, 0.5, 0.75], **kwargs):
    """
    ResNet-50 with multi-scale NPR attention

    This is the main model for H2 hypothesis testing.
    Same architecture as original NPR paper, but with attention-based
    multi-scale NPR instead of fixed single-scale.
    """
    return ResNetMultiScale(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, npr_scales=npr_scales, **kwargs)


def resnet101_multiscale(num_classes: int = 1, npr_scales: List[float] = [0.25, 0.5, 0.75], **kwargs):
    """ResNet-101 with multi-scale NPR attention"""
    return ResNetMultiScale(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, npr_scales=npr_scales, **kwargs)
