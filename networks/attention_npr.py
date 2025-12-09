import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Import BasicBlock and Bottleneck from papers resnet.py 
from .resnet import BasicBlock, Bottleneck, conv3x3, conv1x1


class MultiScaleNPR(nn.Module):
    """
    Extract NPR at multiple scales
    """
    def __init__(self, scales = [0.25, 0.5, 0.75, 1.0]):
        super().__init__()
        self.scales = scales

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Input image (B, 3, H, W)
        Returns:
            Multi-scale NPR (B, 12, H, W) - 4 scales × 3 RGB
        """
        npr_list = []


        # Feature extraction loop
        for scale in self.scales:
            if scale == 1.0:
                # Identity with no downsampling fior baseline reference
                npr = x
            else:
                # Computes residual
                x_down = F.interpolate(x, scale_factor=scale, mode='nearest', recompute_scale_factor=True)
                x_up = F.interpolate(x_down, scale_factor=1/scale, mode='nearest', recompute_scale_factor=True)
                npr = x - x_up  # Artifacts = Original - Reconstructed

            npr_list.append(npr)

        # Concatenate **needs work** try to implement like skipconnections  
        return torch.cat(npr_list, dim=1)  # (B, 12, H, W)


class ChannelAttention(nn.Module):
    """
    Channel attention module based on SENet decided against attention from HW 1 as it is pixelwise, 
    might use for extension for a pixel-wise spatially aware attention with einsum? 

    SNET is simple to implement, may change if results are not good
    """
    def __init__(self, in_channels: int):
        super().__init__()

        # Squeeze for Global Context
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Excitation Two FC layers 
        # Compress by factor of 2
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.sigmoid = nn.Sigmoid()  #

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Input features (B, C, H, W)
        Returns:
            Attention weights (B, C, 1, 1)
        """
        b, c, _, _ = x.shape

        # Squeeze for global context
        y = self.gap(x)          # (B, C, H, W) => (B, C, 1, 1)
        y = y.view(b, c)         # (B, C)

        # Excitation Learns importance weights
        y = self.fc1(y)          # (B, C) => (B, C/2) - Compression
        y = self.act(y)
        y = self.fc2(y)          # (B, C/2) => (B, C) - Expansion 
        y = self.sigmoid(y)      # (B, C) 

        # Reshape
        return y.view(b, c, 1, 1)


class AttentionNPRResNet(nn.Module):
    """
    Combines Multi-Scale NPR with Channel Attention and ResNet

    """
    def __init__(self, block, layers, num_classes=1):
        super().__init__()

        # Multi-scale NPR extraction
        self.scales = [0.25, 0.5, 0.75, 1.0]
        self.multi_scale_npr = MultiScaleNPR(scales=self.scales)

        # Channel attention
        self.attention = ChannelAttention(in_channels=12)  # 4 scales × 3 RGB

        # ResNet backbone same as in the papers ResNet.py
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """ResNet layer"""
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward Process:
        - Extracts NPR at multiple scales
        - Computes attention weights
        - Combines scales using attetion
        - Feeds combination into classifier
        """
        # Multi-scale NPR extraction
        multi_npr = self.multi_scale_npr(x)  # (B, 12, H, W)

        # Compute attention weights
        attn_weights = self.attention(multi_npr)  # (B, 12, 1, 1)

        # Weighting and fuse
        weighted_npr = multi_npr * attn_weights  # Element-wise multiply

        # Reshape - Sum scales => (B, 3, H, W)
        b, _, h, w = weighted_npr.shape
        weighted_npr = weighted_npr.view(b, 4, 3, h, w)  # 4 scales, 3 RGB
        fused_npr = weighted_npr.sum(dim=1)  # (B, 3, H, W)

        # ResNet classification 
        x = self.conv1(fused_npr * 2.0 / 3.0)  # Same scaling as ResNet.py
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def attention_npr_resnet18(**kwargs):
    """ResNet18 with attention-NPR"""
    return AttentionNPRResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def attention_npr_resnet34(**kwargs):
    """ResNet34 with attention-NPR"""
    return AttentionNPRResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def attention_npr_resnet50(**kwargs):
    """ResNet50 with attention-NPR (recommended)"""
    return AttentionNPRResNet(Bottleneck, [3, 4, 6, 3], **kwargs)



if __name__ == '__main__':
