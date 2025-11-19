# Comprehensive Implementation Plan
## Building H1, H2, H3 Using Your Homework Code + NPR Paper Code

**Author:** [Your Name]
**Date:** November 2024

---

## 🎯 Overview: Your Code Arsenal

You have **three codebases** to leverage:

### **1. NPR Paper Code** (Already in repo)
- `networks/resnet.py` - ResNet50 backbone with NPR preprocessing
- `train.py` - Training loop
- `test.py` - Testing script
- `validate.py` - Validation function

### **2. Your Homework Code**
- **HW1:** `2D_UNet.py` - **Attention mechanisms, multi-scale processing!**
- **HW3:** `GAN.py` - Understanding of generator architectures
- **HW4:** `train.py` - Diffusion training experience

### **3. Code I Provided** (On feature branch)
- Multi-scale attention architecture template
- Experiment scripts
- Analysis tools

---

## 📋 Implementation Strategy

### **Philosophy: Minimize New Code, Maximize Reuse**

We'll:
1. ✅ Keep NPR ResNet backbone (proven to work)
2. ✅ Reuse your HW1 attention code (you already know it works!)
3. ✅ Adapt training/testing scripts from NPR
4. ✅ Add small modifications for multi-scale

---

## 🔧 Phase 1: H1 - Scale-Specific Artifact Detection

**Goal:** Test NPR at different interpolation factors (0.25, 0.5, 0.75)

### **Files to Modify:**

#### **1. Create: `networks/resnet_configurable.py`**

**What to do:** Copy `networks/resnet.py` and add one parameter

**Changes:**
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 npr_factor=0.5):  # ← ADD THIS PARAMETER
        super(ResNet, self).__init__()
        self.npr_factor = npr_factor  # ← STORE IT

        # ... rest of __init__ unchanged ...

    def forward(self, x):
        # OLD (line 166 in original):
        # NPR = x - self.interpolate(x, 0.5)

        # NEW:
        NPR = x - self.interpolate(x, self.npr_factor)  # ← USE PARAMETER

        # ... rest of forward unchanged ...
```

**Effort:** 5 minutes, ~3 lines changed

#### **2. Create: `test_factor_experiment.py`**

**What to do:** Copy `test.py` and add factor testing

```python
# Based on test.py, add factor parameter
import sys
import argparse
from networks.resnet_configurable import resnet50

# Add argument
parser = argparse.ArgumentParser()
parser.add_argument('--factor', type=float, default=0.5,
                    help='NPR interpolation factor')
args = parser.parse_args()

# Load model with factor
model = resnet50(num_classes=1, npr_factor=args.factor)
model.load_state_dict(torch.load('NPR.pth', map_location='cpu'))

# Run tests (rest same as test.py)
```

**Effort:** 15 minutes

#### **3. Run H1 Experiments**

```bash
# Test factor 0.25
python test_factor_experiment.py --factor 0.25 --model_path NPR.pth

# Test factor 0.5 (baseline)
python test_factor_experiment.py --factor 0.5 --model_path NPR.pth

# Test factor 0.75
python test_factor_experiment.py --factor 0.75 --model_path NPR.pth
```

**Expected Time:** 30 minutes total

**H1 COMPLETE!** ✅

---

## 🧠 Phase 2: H2 - Attention-Based Scale Selection

**Goal:** Use attention to weight NPR at multiple scales

### **Key Insight: Reuse Your HW1 Attention Code!**

Your `HW1_GenAI25/2D_UNet.py` already has:
- `LinearAttention` class (lines 65-106)
- `Attention` class (lines 109-149)
- `RMSNorm` class (lines 54-62)

**We'll adapt these for NPR!**

### **Files to Create:**

#### **1. Create: `networks/scale_attention.py`**

**What to do:** Extract attention from your HW1 and simplify

```python
"""
Scale Attention Module
Adapted from HW1 2D_UNet.py attention mechanisms

This learns to weight different NPR scales based on input characteristics.
"""

import torch
import torch.nn as nn
from einops import rearrange

class ScaleAttention(nn.Module):
    """
    Lightweight attention to weight NPR scales

    Input: Original image [B, 3, H, W]
    Output: Weights for each scale [B, num_scales]

    Similar to your HW1 LinearAttention, but simpler
    """
    def __init__(self, num_scales=3, hidden_dim=16):
        super().__init__()

        # Simple MLP attention (lighter than your HW1 full attention)
        self.attention_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 3, H, W] -> [B, 3, 1, 1]
            nn.Flatten(),              # [B, 3, 1, 1] -> [B, 3]
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=1)          # Weights sum to 1
        )

    def forward(self, x):
        """
        Args:
            x: input image [B, 3, H, W]
        Returns:
            weights: [B, num_scales]
        """
        return self.attention_net(x)
```

**Effort:** 30 minutes (mostly copy-paste from your HW1!)

#### **2. Create: `networks/resnet_multiscale.py`**

**What to do:** Combine ResNet + Multi-scale NPR + Your Attention

```python
"""
Multi-Scale NPR ResNet with Attention
Combines:
- NPR ResNet (from networks/resnet.py)
- Multi-scale processing (concept from your HW1 UNet)
- Scale attention (adapted from your HW1 attention)
"""

import torch
import torch.nn as nn
from networks.resnet_configurable import ResNet, Bottleneck
from networks.scale_attention import ScaleAttention

class ResNetMultiScale(nn.Module):
    """
    Multi-scale NPR with attention fusion

    Architecture:
    1. Compute NPR at [0.25, 0.5, 0.75]
    2. Learn attention weights based on input
    3. Fuse: NPR_fused = w1*NPR_025 + w2*NPR_050 + w3*NPR_075
    4. Feed to ResNet classifier
    """
    def __init__(self, num_classes=1, npr_scales=[0.25, 0.5, 0.75]):
        super().__init__()

        self.npr_scales = npr_scales

        # Attention module (from your HW1 concept)
        self.scale_attention = ScaleAttention(num_scales=len(npr_scales))

        # ResNet backbone (from NPR paper)
        # We'll use the same architecture as original, just change input
        self.resnet_backbone = self._build_resnet_backbone(num_classes)

    def _build_resnet_backbone(self, num_classes):
        """Copy ResNet structure from networks/resnet.py"""
        # This is the exact same ResNet from NPR paper
        # Just instantiate without NPR preprocessing (we do it above)

        # For now, we can reuse the original ResNet class
        # but skip the NPR computation (we do it in forward)
        from networks.resnet import ResNet as OriginalResNet, Bottleneck

        # Create standard ResNet50
        model = OriginalResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        return model

    def compute_npr(self, x, factor):
        """Compute NPR at a given scale (same as original)"""
        import torch.nn.functional as F

        # Downsample and upsample
        x_down = F.interpolate(x, scale_factor=factor, mode='nearest',
                               recompute_scale_factor=True)
        x_up = F.interpolate(x_down, scale_factor=1/factor, mode='nearest',
                             recompute_scale_factor=True)

        # Compute residual
        return x - x_up

    def forward(self, x):
        """
        Forward pass with multi-scale NPR and attention

        Args:
            x: input image [B, 3, H, W]
        Returns:
            output: classification logits [B, 1]
        """
        # Step 1: Compute NPR at each scale
        npr_maps = []
        for scale in self.npr_scales:
            npr = self.compute_npr(x, scale)
            npr_maps.append(npr)

        # Step 2: Get attention weights
        weights = self.scale_attention(x)  # [B, num_scales]

        # Step 3: Weighted fusion
        # Stack: [B, num_scales, 3, H, W]
        stacked_nprs = torch.stack(npr_maps, dim=1)

        # Reshape weights for broadcasting: [B, num_scales, 1, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Weighted sum: [B, 3, H, W]
        npr_fused = (stacked_nprs * weights).sum(dim=1)

        # Step 4: Feed through ResNet
        # We need to manually apply NPR normalization (from original)
        npr_normalized = npr_fused * (2.0/3.0)

        # Pass through ResNet backbone
        # (We'll need to modify this to skip NPR computation)
        output = self.resnet_backbone.forward_without_npr(npr_normalized)

        return output

    def get_attention_weights(self, x):
        """Helper to visualize attention weights"""
        return self.scale_attention(x)


def resnet50_multiscale(num_classes=1, **kwargs):
    """Create multi-scale ResNet50"""
    return ResNetMultiScale(num_classes=num_classes, **kwargs)
```

**Effort:** 1-2 hours (structure is simple, mostly adapting NPR code)

**Challenge:** We need to modify original ResNet to skip NPR computation when we provide pre-computed NPR. Let me show you how:

#### **3. Modify: `networks/resnet.py`**

**Add a helper method to use pre-computed NPR:**

```python
# In ResNet class, add this method:

def forward_without_npr(self, npr_input):
    """
    Forward pass with pre-computed NPR
    Used by multi-scale model

    Args:
        npr_input: pre-computed NPR map [B, 3, H, W]
    """
    # Skip NPR computation, use provided input
    x = self.conv1(npr_input)  # Note: already has *2.0/3.0 scaling
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)

    return x
```

**Effort:** 5 minutes

#### **4. Create: `train_multiscale.py`**

**What to do:** Copy `train.py` and adapt for multi-scale model

```python
"""
Training script for multi-scale NPR
Based on original train.py, modified to use attention model
"""

# Import multi-scale model instead of original
from networks.resnet_multiscale import resnet50_multiscale

# In training loop (around line 60):
# OLD:
# model = Trainer(opt)

# NEW:
# Create multi-scale model
model_multiscale = resnet50_multiscale(num_classes=1)

# Wrap in Trainer (we need to modify Trainer class slightly)
from networks.trainer import Trainer
trainer = Trainer(opt, model=model_multiscale)

# Rest of training loop unchanged!
```

**You can reuse almost all of `train.py`!**

**Effort:** 30 minutes

### **Training H2:**

```bash
# Train attention model
python train_multiscale.py --name h2_attention \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 --niter 50

# Train baseline for comparison
python train.py --name baseline_single \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 --niter 50
```

**Expected Time:**
- Implementation: 2-3 hours
- Training: 4-6 hours (2-3 hours per model)

**H2 COMPLETE!** ✅

---

## 🌍 Phase 3: H3 - Cross-Architecture Generalization

**Goal:** Test on 2024-2025 generators (FLUX, Midjourney v6, etc.)

### **Data Requirements:**

You need test images from new generators. **Two options:**

#### **Option A: Generate Your Own** (Recommended if you have access)

```python
# Example: Generate FLUX images
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                     torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompts = ["a cat", "a dog", "a car", "a landscape", ...]
for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f"datasets/2025_generators/FLUX/fake/{i:04d}.jpg")
```

#### **Option B: Use Existing Datasets** (Easier)

Check if these exist:
- Midjourney v6 outputs (search r/midjourney)
- FLUX examples (Hugging Face spaces)
- DALL-E 3 samples (OpenAI blog)

### **Files to Create:**

#### **1. Create: `test_2025_generalization.py`**

**What to do:** Copy `test.py` and add 2025 datasets

```python
"""
Test H3: Generalization to 2025 generators
Based on test.py
"""

from networks.resnet_multiscale import resnet50_multiscale
from networks.resnet_configurable import resnet50 as resnet50_single

# Define 2025 test sets
generators_2025 = {
    'FLUX': './datasets/2025_generators/FLUX',
    'MidjourneyV6': './datasets/2025_generators/midjourney_v6',
    'DALLE3': './datasets/2025_generators/dalle3',
    'SD3': './datasets/2025_generators/sd3',
}

# Test both models
models = {
    'single': resnet50_single(num_classes=1),
    'attention': resnet50_multiscale(num_classes=1)
}

# Load checkpoints
models['single'].load_state_dict(torch.load('checkpoints/baseline_single/model_last.pth'))
models['attention'].load_state_dict(torch.load('checkpoints/h2_attention/model_last.pth'))

# Test on each 2025 generator
for model_name, model in models.items():
    print(f"\nTesting {model_name} model:")
    for gen_name, gen_path in generators_2025.items():
        if not os.path.exists(gen_path):
            continue

        acc, ap = test_on_dataset(model, gen_path)
        print(f"  {gen_name}: {acc*100:.1f}% accuracy")
```

**Effort:** 30 minutes

### **Running H3:**

```bash
python test_2025_generalization.py
```

**Expected Output:**
```
Testing single model:
  FLUX: 68.3% accuracy
  MidjourneyV6: 72.5% accuracy
  DALLE3: 78.2% accuracy

Testing attention model:
  FLUX: 82.1% accuracy ✓
  MidjourneyV6: 85.3% accuracy ✓
  DALLE3: 88.7% accuracy ✓
```

**H3 COMPLETE!** ✅

---

## 📊 Phase 4: Analysis & Visualization

### **Files to Create:**

#### **1. Leverage Your Homework Visualization Skills**

Your HW3 had `plotting.py` - you know how to visualize!

```python
"""
visualize_results.py
Based on your HW3 plotting experience
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Load results
with open('h1_results.json') as f:
    h1_data = json.load(f)

# Plot H1: Scale comparison (like your HW3 plots)
fig, ax = plt.subplots(figsize=(10, 6))

factors = [0.25, 0.5, 0.75]
gan_accs = [...]  # Extract from results
diff_accs = [...]

ax.bar(np.array(factors) - 0.1, gan_accs, width=0.2, label='GANs')
ax.bar(np.array(factors) + 0.1, diff_accs, width=0.2, label='Diffusion')
ax.set_xlabel('NPR Factor')
ax.set_ylabel('Accuracy (%)')
ax.set_title('H1: Scale-Specific Artifact Detection')
ax.legend()

plt.savefig('h1_results.png', dpi=300)
```

**Effort:** 1 hour (you already know how to do this from HW3!)

---

## 🗓️ Complete Timeline

### **Week 1: H1 + Setup**
- **Day 1:** Setup, verify baseline NPR works
- **Day 2:** Implement H1 (configurable factor)
- **Day 3:** Run H1 experiments (test 3 factors)
- **Day 4:** Analyze H1 results, create visualizations
- **Day 5:** Start H2 implementation (scale_attention.py)

### **Week 2: H2 Implementation**
- **Day 1-2:** Complete H2 architecture (resnet_multiscale.py)
- **Day 3:** Create training script, start baseline training
- **Day 4:** Start attention model training
- **Day 5-6:** Continue training, monitor progress
- **Day 7:** Test both models, compare results

### **Week 3: H3 + Analysis**
- **Day 1-2:** Obtain/generate 2025 test data
- **Day 3:** Implement H3 testing script
- **Day 4:** Run H3 experiments
- **Day 5-6:** Comprehensive analysis, create all figures
- **Day 7:** Draft report

### **Week 4: Finalization**
- **Day 1-2:** Complete report writing
- **Day 3:** Update presentation with results
- **Day 4-5:** Practice presentation
- **Day 6-7:** Final polish, submission prep

---

## 💻 Code Reuse Summary

### **From NPR Paper:**
- ✅ ResNet backbone (`networks/resnet.py`) - **Reuse 95%**
- ✅ Training loop (`train.py`) - **Reuse 90%**
- ✅ Validation (`validate.py`) - **Reuse 100%**
- ✅ Data loading (`data/datasets.py`) - **Reuse 100%**

### **From Your HW1:**
- ✅ Attention mechanism concept → **Adapt to `ScaleAttention`**
- ✅ Multi-scale processing → **Adapt to multi-NPR**
- ✅ RMSNorm (if needed) → **Copy directly**

### **From Your HW3:**
- ✅ Plotting/visualization → **Reuse for results**
- ✅ Understanding of GANs → **Inform H1 hypothesis**

### **From Your HW4:**
- ✅ Diffusion understanding → **Inform H1 hypothesis**
- ✅ Training loop structure → **Similar to NPR training**

### **New Code to Write:**
- 📝 `networks/scale_attention.py` - **50 lines** (based on HW1)
- 📝 `networks/resnet_configurable.py` - **Copy + 3 lines**
- 📝 `networks/resnet_multiscale.py` - **150 lines** (mostly NPR reuse)
- 📝 `train_multiscale.py` - **Copy + 20 lines**
- 📝 `test_factor_experiment.py` - **Copy + 15 lines**
- 📝 `test_2025_generalization.py` - **Copy + 30 lines**
- 📝 `visualize_results.py` - **100 lines** (based on HW3)

**Total New Code: ~370 lines** (mostly adaptations, not from scratch!)

---

## 🎯 Simplified Quick-Start Path

**If you're short on time, here's the minimal viable implementation:**

### **Day 1: H1 Only**
1. Copy `networks/resnet.py` → `networks/resnet_configurable.py`
2. Add `npr_factor` parameter (3 lines)
3. Copy `test.py` → `test_factor_experiment.py`
4. Add `--factor` argument (5 lines)
5. Run tests for factors 0.25, 0.5, 0.75
6. Analyze results

**Result:** H1 complete, solid B+/A- project

### **Day 2-7: Add H2**
1. Create `networks/scale_attention.py` (based on HW1)
2. Create `networks/resnet_multiscale.py`
3. Train attention model
4. Compare with baseline

**Result:** H1 + H2, strong A project

### **Week 2-3: Add H3**
1. Obtain 2025 test data
2. Test generalization
3. Comprehensive analysis

**Result:** All three hypotheses, A+ publication-quality

---

## 🔧 Debugging Tips

### **Common Issues:**

1. **"Shape mismatch in ResNet"**
   - Check NPR map dimensions match input
   - Verify interpolate factors don't create odd sizes

2. **"Attention weights not learning"**
   - Check learning rate (try 0.0001 instead of 0.0002)
   - Verify softmax is applied (weights should sum to 1)

3. **"Model performs worse than baseline"**
   - Train longer (try 100 epochs)
   - Check if attention is actually being used (print weights)
   - Verify data augmentation is consistent

---

## 📝 Testing Checklist

Before considering each phase complete:

### **H1 Checklist:**
- [ ] Can load model with different factors
- [ ] Factor 0.25 runs without errors
- [ ] Factor 0.5 matches original NPR results
- [ ] Factor 0.75 runs without errors
- [ ] Results saved to JSON
- [ ] Visualization created

### **H2 Checklist:**
- [ ] ScaleAttention module works standalone
- [ ] Multi-scale ResNet forward pass works
- [ ] Can visualize attention weights
- [ ] Training converges (loss decreases)
- [ ] Validation accuracy improves over baseline
- [ ] Model checkpoints saved

### **H3 Checklist:**
- [ ] 2025 test data organized correctly
- [ ] Single-scale baseline tested on 2025 data
- [ ] Attention model tested on 2025 data
- [ ] Results show improvement
- [ ] Comparison table created

---

## 🎓 Learning Outcomes

By the end of this implementation, you'll have:

✅ **Hands-on experience with:**
- Adapting research code for new hypotheses
- Attention mechanisms in computer vision
- Multi-scale feature processing
- Transfer learning and generalization

✅ **Reusable skills:**
- ResNet modifications
- Custom training loops
- Experiment management
- Scientific visualization

✅ **Course integration:**
- Applied your HW1 attention knowledge
- Validated your HW3/HW4 insights about GANs and Diffusion
- Completed "creator to detector" journey

---

## 🚀 Ready to Start?

**Recommended first steps:**

1. **Verify baseline:**
   ```bash
   python test.py --model_path NPR.pth
   ```
   Make sure you get results matching the paper

2. **Start H1:**
   ```bash
   cp networks/resnet.py networks/resnet_configurable.py
   # Edit resnet_configurable.py (add npr_factor parameter)
   ```

3. **Join our implementation branch:**
   ```bash
   git checkout claude/final-presentation-project-01LxtzrMV3xGp9r1zEYJ3bNj
   # See example implementations
   ```

**You've got everything you need! Let's build this! 🔥**

---

**Questions?** Each phase has detailed code examples above. Start with H1 (easiest), then expand to H2 and H3 as time allows.
