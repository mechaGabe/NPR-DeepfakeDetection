# Code Changes & Adaptations - Slide Version

---

## Slide: "Implementation Overview"

### **Starting Point: NPR CVPR 2024 Codebase**
```
✅ Single-scale NPR extraction (0.5×)
✅ ResNet-50 backbone
✅ Training/testing infrastructure
✅ ~2,000 lines of existing code
```

---

### **Our Modifications: Minimal Changes**

**3 Files Modified** (~30 lines total):

```python
# 1. networks/trainer.py (15 lines)
if opt.model_type == 'multiscale_attention':
    model = attention_multiscale_npr18(scales=[0.25, 0.5, 0.75])
else:
    model = resnet50()  # Original

# 2. options/base_options.py (8 lines)
parser.add_argument('--model_type', choices=['single_scale', 'multiscale_attention'])
parser.add_argument('--npr_scales', default='0.25,0.5,0.75')

# 3. test.py (10 lines)
# Added multi-scale model loading support
```

---

### **Our Additions: New Code**

**3 New Files** (~950 lines total):

| File | Lines | Purpose |
|------|-------|---------|
| `networks/multiscale_npr.py` | 350 | Multi-scale architecture |
| `visualize_attention.py` | 400 | Attention analysis |
| `run_multiscale_experiments.sh` | 200 | Automated pipeline |

---

## Slide: "Homework Concepts Applied"

### **Core Components from Course:**

#### **1. Attention Mechanism** → Scale Fusion
```python
# Learned from: [Homework on Attention/Transformers]
attention_weights = Softmax(MLP(features))  # [0.4, 0.3, 0.3]
output = Σ(weight_i × feature_i)            # Weighted combination
```
**Applied**: Adaptive weighting of 3 scales

---

#### **2. Multi-Branch CNN** → Parallel Processing
```python
# Learned from: [Homework on CNN Architectures]
features_1 = ResNet_branch1(NPR_scale1)
features_2 = ResNet_branch2(NPR_scale2)
features_3 = ResNet_branch3(NPR_scale3)
```
**Applied**: 3 specialized feature extractors

---

#### **3. ResNet Blocks** → Lightweight Branches
```python
# Learned from: [Homework on ResNets/Skip Connections]
class MultiScaleResNet:
    self.layer1 = BasicBlock([2, 2])  # Residual connections
    self.layer2 = BasicBlock([2, 2])
```
**Applied**: Efficient feature extraction per scale

---

#### **4. Binary Classification** → Real/Fake Detection
```python
# Learned from: [Standard Classification Homework]
loss = BCEWithLogitsLoss(predictions, labels)
optimizer = Adam(lr=0.0002)
```
**Applied**: Training pipeline for deepfake detection

---

## Slide: "What's Novel vs. What's Adapted"

### **Adapted from Course** 📚
- ✅ Attention mechanism (softmax weighting)
- ✅ Multi-branch architecture (parallel CNNs)
- ✅ ResNet building blocks (BasicBlock, BatchNorm)
- ✅ Standard training practices (Adam, BCE loss)

### **Novel Contributions** ✨
- 🆕 Multi-scale NPR extraction (3 scales: 0.25×, 0.5×, 0.75×)
- 🆕 Attention-based scale fusion (adaptive per image)
- 🆕 Lightweight multi-branch design (5M params/branch)
- 🆕 Interpretability analysis (which scales matter?)

---

## Slide: "Code Breakdown"

### **Total Implementation Effort:**

```
Original NPR Code:        2,000 lines (unchanged)
Modified Files:              30 lines (3 files)
New Code:                   950 lines (3 files)
─────────────────────────────────────────────
Total:                    2,980 lines
```

### **Work Distribution:**
- 98% of original code unchanged
- 2% modified for multi-scale support
- ~950 new lines implementing multi-scale + attention

---

## Slide: "Architecture: Original vs. Ours"

### **Original NPR (Given):**
```
Image → NPR@0.5× → ResNet-50 → Classifier
        (11M parameters)
```

### **Our Multi-Scale NPR (Implemented):**
```
        NPR@0.25× → ResNet₁ (5M) ↘
Image → NPR@0.5×  → ResNet₂ (5M) → Attention → Fused → Classifier
        NPR@0.75× → ResNet₃ (5M) ↗    (0.1M)

        (15M parameters total, +36%)
```

**Key Difference**: 1 deep network → 3 shallow specialized branches

---

## Quick Talking Points

### **When Asked "What Did You Implement?"**

**Short Answer**:
> "We extended the NPR codebase by implementing a multi-scale architecture with attention-based fusion. We modified 30 lines to support model switching and added 950 lines of new code for the multi-scale network, applying attention mechanisms and multi-branch concepts from our coursework."

### **When Asked "What's From Homework?"**

**Short Answer**:
> "The attention mechanism (softmax weighting) and multi-branch CNN structure come from course homework. The novel part is applying these to multi-scale NPR extraction—using attention to learn which scales matter for different generators."

### **When Asked "How Much Work Was This?"**

**Short Answer**:
> "About 1,000 lines of code: minimal modifications to the given codebase plus new implementations of the multi-scale architecture, attention module, and analysis tools. Built on homework concepts but applied to a novel problem."

---

## Visual for Slide

### **Implementation Pyramid:**

```
         Novel Application
              (Top)
                ↑
    ┌───────────────────────┐
    │   Multi-Scale NPR     │  ← Our Novel Contribution
    │   with Attention      │
    └───────────────────────┘
                ↑
    ┌───────────────────────┐
    │  Course Concepts:     │  ← From Homework
    │  • Attention          │
    │  • Multi-Branch CNNs  │
    │  • ResNet Blocks      │
    └───────────────────────┘
                ↑
    ┌───────────────────────┐
    │   NPR Codebase        │  ← Given Starting Point
    │   (CVPR 2024)         │
    └───────────────────────┘
          Foundation
           (Base)
```

---

## One Slide Summary (Copy-Paste Ready)

### **Implementation Overview**

**Starting Point**: NPR CVPR 2024 codebase (2,000 lines)

**Our Work**:
- **Modified**: 30 lines across 3 files (model selection)
- **Added**: 950 new lines (multi-scale + attention + visualization)
- **Applied**: Course concepts on attention and multi-branch CNNs
- **Novel**: Multi-scale NPR extraction with adaptive fusion

**Homework Concepts**:
- Attention mechanisms → Scale weighting
- Multi-branch CNNs → Parallel feature extraction
- ResNet blocks → Lightweight branches

**Result**: 15M parameter model (+36% vs baseline), real-time inference (83 img/s)

---

## TODO: Specify Your Homework

**Replace [Homework X] with actual assignment:**

- [ ] Homework __ on Attention Mechanisms
- [ ] Homework __ on CNN Architectures
- [ ] Homework __ on ResNets
- [ ] Lab __ on PyTorch Implementation

**Then update slides with**:
> "We adapted the attention mechanism from Homework [X] and the multi-branch architecture from Homework [Y]..."
