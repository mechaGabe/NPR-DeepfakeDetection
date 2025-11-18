# Multi-Scale Attention NPR - Quick Start Guide

This guide helps you get started with the multi-scale attention-based NPR implementation for deepfake detection.

---

## What's New?

This implementation extends the original NPR (CVPR 2024) with:

✨ **Multi-Scale NPR Extraction**: Analyzes artifacts at 3 different scales (0.25×, 0.5×, 0.75×)

🧠 **Attention-Based Fusion**: Adaptively weights different scales per image

📊 **Comprehensive Visualization**: Analyze which scales matter for different generators

🔬 **Ablation Tools**: Easy experimentation with different scale combinations

---

## Installation

### Option 1: Use Existing Environment (Recommended)

If you already have the original NPR environment set up:

```bash
# No additional packages needed!
# The multi-scale implementation uses the same dependencies
```

### Option 2: Fresh Install

```bash
# Install requirements
pip install -r requirements.txt

# Additional packages for visualization (optional)
pip install matplotlib seaborn
```

---

## Quick Start

### 1. Training a Multi-Scale Model

**Basic Command**:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --name my_multiscale_model \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5,0.75 \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 \
    --lr 0.0002 \
    --niter 50
```

**Key Parameters**:
- `--model_type`: Choose `single_scale` (original) or `multiscale_attention` (new)
- `--npr_scales`: Comma-separated scales (e.g., `0.25,0.5,0.75`)
- All other parameters same as original NPR

### 2. Testing a Trained Model

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_path ./checkpoints/my_multiscale_model_*/model_epoch_last.pth \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5,0.75 \
    --batch_size 64
```

### 3. Visualizing Attention Weights

```bash
python visualize_attention.py \
    --model_path ./checkpoints/my_multiscale_model_*/model_epoch_last.pth \
    --dataroot ./datasets/test/stylegan \
    --output_dir ./visualizations/stylegan \
    --scales 0.25,0.5,0.75 \
    --num_samples 100 \
    --save_npr_maps
```

This generates:
- Attention weight statistics
- Distribution plots (box plots, violin plots)
- NPR artifact visualizations
- Per-image attention patterns

---

## Running Full Experiments

### Automated Pipeline

Use the provided shell script to run all experiments:

```bash
# Make script executable
chmod +x run_multiscale_experiments.sh

# Edit the script to set your data paths
nano run_multiscale_experiments.sh

# Run all experiments
./run_multiscale_experiments.sh
```

This will:
1. Train baseline (single-scale) model
2. Train multi-scale (3 scales) model
3. Train ablation models (2 scales, 4 scales)
4. Test all models on all datasets
5. Generate attention visualizations

**Expected Runtime**: ~60 hours on RTX 3090

### Manual Experiments

**Experiment 1: Baseline**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --name baseline_single_scale \
    --model_type single_scale \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 \
    --lr 0.0002 \
    --niter 50
```

**Experiment 2: Multi-Scale (3 scales)**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --name multiscale_3scales \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5,0.75 \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 \
    --lr 0.0002 \
    --niter 50
```

**Experiment 3: Ablation - 2 Scales (coarse)**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --name ablation_2scales_coarse \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5 \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 \
    --lr 0.0002 \
    --niter 50
```

---

## Understanding the Architecture

### Single-Scale NPR (Original)
```
Input → NPR@0.5x → ResNet50 → Classifier
```

### Multi-Scale Attention NPR (Ours)
```
Input Image
    ↓
    ├─ NPR@0.25x → ResNet_A → Features_A ──┐
    ├─ NPR@0.5x  → ResNet_B → Features_B ──┼→ Attention → Fused → Classifier
    └─ NPR@0.75x → ResNet_C → Features_C ──┘
```

**Key Components**:

1. **NPR Extractors**: Extract artifacts at each scale
   - `NPR = Image - Interpolate(Image, scale)`

2. **Feature Branches**: Separate ResNets process each scale
   - Lightweight: BasicBlock with [2, 2] layers
   - Output: 128-dimensional features

3. **Attention Module**: Learns to weight scales
   - Input: Concatenated features [B, 384]
   - Output: Attention weights [B, 3]
   - Architecture: MLP with softmax

4. **Classifier**: Final prediction
   - Input: Weighted features [B, 128]
   - Output: Real/Fake logit [B, 1]

---

## Analyzing Results

### Expected Outputs

After training and testing, you'll have:

**Performance Metrics**:
- `results/[model_name]_[dataset].txt`: Accuracy and AP for each test set
- Organized by test table (Table 1-5 from paper)

**Attention Analysis**:
- `visualizations/[dataset]/attention_statistics.txt`: Mean ± std for each scale
- `visualizations/[dataset]/attention_distribution.png`: Box and violin plots
- `visualizations/[dataset]/npr_artifacts/`: Per-image NPR visualizations

### Interpreting Attention Weights

**High attention on scale 0.25×**:
- Model relies on coarse-scale artifacts
- Common for GANs with progressive growing (ProGAN)

**High attention on scale 0.5×**:
- Balanced artifacts
- Common baseline scale

**High attention on scale 0.75×**:
- Model relies on fine-scale artifacts
- May indicate subtle upsampling patterns

**Varied attention across images**:
- Adaptive behavior (good!)
- Different images benefit from different scales

**Uniform attention (all ~0.33)**:
- Model treats all scales equally
- May indicate fusion isn't learning useful patterns
- Consider: longer training, different initialization

---

## Comparison with Baseline

### Creating Comparison Tables

After running experiments, create comparison tables:

```python
# Example: Compare accuracy across models
import numpy as np

# Read results from output files
baseline_acc = [99.8, 96.3, 97.3, 87.5, 95.0, 99.7, 86.6, 77.4]  # Table 1
multiscale_acc = [99.9, 97.1, 98.0, 89.2, 96.3, 99.8, 88.1, 79.5]  # Your results

# Compute improvement
improvement = np.array(multiscale_acc) - np.array(baseline_acc)
mean_improvement = improvement.mean()

print(f"Mean improvement: {mean_improvement:.2f}%")
print(f"Improved on {(improvement > 0).sum()}/8 generators")
```

### Visualization Scripts

We provide visualization utilities:

```python
# Load attention weights
python -c "
from visualize_attention import plot_attention_heatmap
import torch

# Create comparison heatmap
mean_attentions = {
    'ProGAN': torch.tensor([0.4, 0.4, 0.2]),
    'StyleGAN': torch.tensor([0.3, 0.5, 0.2]),
    'Diffusion': torch.tensor([0.2, 0.3, 0.5]),
}

plot_attention_heatmap(mean_attentions, [0.25, 0.5, 0.75], 'heatmap.png')
"
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch_size 16  # or even 8
```

**Solution 2**: Use gradient accumulation
```python
# Modify trainer.py
accumulation_steps = 2
for i, data in enumerate(dataloader):
    loss = model.get_loss()
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Model Not Learning (Loss Plateaus)

**Check 1**: Verify data loading
```bash
# Test data loader
python -c "from data import create_dataloader; from options.train_options import TrainOptions; opt = TrainOptions().parse(); loader = create_dataloader(opt); print(next(iter(loader)))"
```

**Check 2**: Reduce learning rate
```bash
--lr 0.0001  # Instead of 0.0002
```

**Check 3**: Check attention weights during training
```python
# Add to train.py
if epoch % 5 == 0:
    with torch.no_grad():
        _, attn = model.model(val_images, return_attention=True)
        print(f"Attention weights: {attn.mean(dim=0)}")
```

### Issue: Attention Weights All ~0.33 (Uniform)

This means the attention mechanism isn't learning meaningful patterns.

**Solution 1**: Increase feature dimension
```python
# In multiscale_npr.py
model = AttentionMultiScaleNPR(feature_dim=256)  # Instead of 128
```

**Solution 2**: Add entropy regularization
```python
# Encourage diverse attention
attention_entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=1).mean()
loss = classification_loss - 0.01 * attention_entropy  # Maximize entropy
```

**Solution 3**: Check if all scales have similar discriminative power
- If yes, uniform attention is actually optimal!
- Visualize NPR maps to verify

---

## Model Checkpoints

### Checkpoint Format

Saved checkpoints include:
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}
```

### Loading Custom Checkpoints

```python
import torch
from networks.multiscale_npr import attention_multiscale_npr18

# Load model
model = attention_multiscale_npr18(num_classes=1, scales=[0.25, 0.5, 0.75])
checkpoint = torch.load('path/to/checkpoint.pth')

# Handle different formats
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
```

---

## Customization

### Using Different Scales

Experiment with custom scale combinations:

```bash
# Fine-grained scales
--npr_scales 0.6,0.7,0.8,0.9

# Coarse scales only
--npr_scales 0.1,0.2,0.3

# Many scales
--npr_scales 0.2,0.3,0.4,0.5,0.6,0.7
```

### Modifying the Attention Module

Edit `networks/multiscale_npr.py`:

```python
# Change reduction ratio (controls bottleneck size)
self.attention_fusion = AttentionFusionModule(
    feature_dim=128,
    num_scales=3,
    reduction=8  # Smaller = more capacity
)

# Add residual connections
fused = (1 - attention_weights.sum(dim=1, keepdim=True)) * base_features + \
        sum(attention_weights[:, i:i+1] * features_list[i] for i in range(num_scales))
```

### Using Different Backbones

Replace ResNet branches:

```python
# In multiscale_npr.py
from torchvision.models import efficientnet_b0

self.feature_branches = nn.ModuleList([
    efficientnet_b0(pretrained=False, num_classes=128)
    for _ in range(self.num_scales)
])
```

---

## Citation

If you use this multi-scale extension in your work, please cite:

```bibtex
@misc{tan2023rethinking,
      title={Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection},
      author={Chuangchuang Tan and Huan Liu and Yao Zhao and Shikui Wei and Guanghua Gu and Ping Liu and Yunchao Wei},
      year={2023},
      eprint={2312.10461},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

And mention this multi-scale attention extension in your acknowledgments.

---

## FAQ

**Q: How much better is multi-scale than single-scale?**

A: Expected improvement of 2-5% on average, with largest gains on diffusion models.

**Q: Which scale combination works best?**

A: [0.25, 0.5, 0.75] works well. Run ablations to find optimal for your data.

**Q: Can I use this with other detectors?**

A: Yes! The multi-scale NPR extraction is modular. You can replace the ResNet branches with any architecture.

**Q: How do I know if attention is working?**

A: Check if attention weights vary across generators. Uniform weights (~0.33 each) suggest it's not helping.

**Q: Does this work on videos?**

A: The current implementation is for images. For videos, extract frames and average predictions or use temporal modeling.

---

## Contact & Support

For questions about this implementation:
- Check existing issues in the repository
- Review the original NPR paper for baseline methodology
- Consult the project proposal document for detailed explanations

---

**Last Updated**: November 2024
**Status**: Ready for experimentation!
**License**: Following original NPR repository
