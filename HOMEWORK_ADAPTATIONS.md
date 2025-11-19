# Code Modifications & Homework Adaptations - Slide Content

---

## What We're Changing from Original NPR Code

### **Files Modified** ✏️

1. **`networks/trainer.py`**
   - Added model selection logic
   - Support for loading multi-scale vs single-scale models
   - **Lines changed**: 4 (imports) + 15 (model selection logic)

2. **`options/base_options.py`**
   - Added `--model_type` argument (single_scale vs multiscale_attention)
   - Added `--npr_scales` argument (comma-separated scales)
   - Added scale parsing logic
   - **Lines changed**: 8 new lines

3. **`test.py`**
   - Added multi-scale model loading
   - Support for testing both architectures
   - **Lines changed**: 10 lines

### **What Stays the Same** ✅
- Training loop (`train.py`) - unchanged
- Data loading (`data/datasets.py`) - unchanged
- Original ResNet-50 architecture - unchanged
- NPR extraction concept - unchanged
- Loss function, optimizer - unchanged

---

## What We're Adding (New Code)

### **New Files Created** ⭐

1. **`networks/multiscale_npr.py`** (~350 lines)
   - `NPRExtractor` class (20 lines)
   - `MultiScaleResNet` class (80 lines)
   - `AttentionFusionModule` class (30 lines)
   - `AttentionMultiScaleNPR` class (100 lines)
   - Helper functions (120 lines)

2. **`visualize_attention.py`** (~400 lines)
   - Attention weight analysis
   - Visualization generation
   - Statistical analysis

3. **`run_multiscale_experiments.sh`** (~200 lines)
   - Automated experiment pipeline
   - Training all models
   - Testing and visualization

**Total New Code**: ~950 lines

---

## Homework Adaptations

### **Which Homework Assignment?**

**Please specify which homework(s) you're adapting from:**
- [ ] Homework on CNNs / ResNets?
- [ ] Homework on Attention Mechanisms?
- [ ] Homework on GANs / Generative Models?
- [ ] Homework on Multi-Scale Architectures?
- [ ] Other: ___________

---

## Typical Course Concepts Applied

### **Common Homework Topics Likely Used:**

#### **1. Attention Mechanisms** 🎯
```python
# Concept: Softmax attention over features
# Typical homework: "Implement self-attention for..."

# Our adaptation:
attention_weights = Softmax(MLP(concatenated_features))
fused = Σ(weight_i × feature_i)
```
**From**: Homework on Transformers/Attention
**Applied to**: Multi-scale feature fusion

#### **2. Multi-Branch CNNs** 🌲
```python
# Concept: Parallel processing paths
# Typical homework: "Build a multi-stream network..."

# Our adaptation:
Branch1 → Features1 ↘
Branch2 → Features2 → Fusion → Output
Branch3 → Features3 ↗
```
**From**: Homework on CNN architectures
**Applied to**: Scale-specific feature extraction

#### **3. ResNet Building Blocks** 🧱
```python
# Concept: Residual connections, BasicBlock
# Typical homework: "Implement ResNet from scratch"

# Our adaptation:
class MultiScaleResNet(nn.Module):
    # Uses BasicBlock, residual connections
    # Lighter than standard ResNet
```
**From**: Homework on ResNets/Skip connections
**Applied to**: Lightweight feature extractors

#### **4. Custom Loss Functions** 📉
```python
# Concept: Binary classification with BCE
# Typical homework: "Train a binary classifier"

# Our adaptation:
loss = BCEWithLogitsLoss(predictions, labels)
# Could add: entropy regularization on attention
```
**From**: Standard classification homework
**Applied to**: Real/Fake detection

---

## Architecture Design Decisions

### **What's Novel (Not from Homework):**

1. **Multi-Scale NPR Extraction**
   - Original contribution
   - Applying NPR at multiple downsampling factors
   - Not a standard homework concept

2. **Attention-Based Scale Fusion**
   - Combining attention with multi-scale
   - Adaptive weighting per image
   - More advanced than typical homework

3. **Domain-Specific Design**
   - Specialized for upsampling artifacts
   - Lightweight branches for efficiency
   - Interpretability focus

### **What's Adapted from Course:**

1. **Attention Implementation**
   - Standard softmax attention pattern
   - MLP-based weight computation
   - Common in homework assignments

2. **CNN Architecture**
   - BasicBlock structure
   - Batch normalization
   - Standard building blocks

3. **Training Pipeline**
   - Adam optimizer
   - Learning rate scheduling
   - Standard practices from homework

---

## Slide-Ready Summary

### **Code Changes:**

| Category | Files | Lines Changed | Complexity |
|----------|-------|--------------|------------|
| **Modified** | 3 files | ~30 lines | Easy |
| **New Code** | 3 files | ~950 lines | Medium |
| **Unchanged** | 10+ files | ~2000 lines | N/A |

**Total Effort**: ~1000 new/modified lines

### **Homework Concepts Applied:**

```
Attention Mechanisms (HW X)
    ↓
Multi-Branch CNNs (HW Y)
    ↓
ResNet Building Blocks (HW Z)
    ↓
Our Multi-Scale NPR Architecture
```

### **Novel Contributions:**

1. ✨ Multi-scale NPR extraction (not from homework)
2. ✨ Attention for scale fusion (adapted from homework)
3. ✨ Lightweight multi-branch design (novel)
4. ✨ Interpretable attention patterns (novel)

---

## For Your Presentation

### **How to Present This:**

**Slide Title**: "Implementation: Building on Course Foundations"

**Content**:
```
Original NPR Code (Given):
├─ ResNet-50 backbone
├─ Single-scale NPR (0.5×)
└─ Training/testing infrastructure

Our Modifications:
├─ ✏️ Modified 3 files (~30 lines)
│   └─ Added multi-scale model selection
└─ ⭐ Added 3 new files (~950 lines)
    ├─ Multi-scale architecture
    ├─ Attention fusion
    └─ Visualization tools

Course Concepts Applied:
├─ Attention mechanisms (HW X)
├─ Multi-branch CNNs (HW Y)
└─ ResNet building blocks (HW Z)

Novel Contributions:
└─ Multi-scale NPR with adaptive fusion
```

---

## Specific Homework Mapping Template

**Fill this in based on your actual course:**

| Concept | Homework # | How We Used It |
|---------|-----------|----------------|
| **Attention** | HW ___ | Scale weighting in fusion module |
| **ResNets** | HW ___ | Lightweight branches (BasicBlock) |
| **Multi-branch** | HW ___ | 3 parallel processing streams |
| **CNNs** | HW ___ | Feature extraction architecture |
| **Binary Classification** | HW ___ | Real/Fake detection head |

---

## Questions to Clarify with Professor

Before your presentation, confirm:

1. **"We adapted the attention mechanism from Homework X - is this the correct approach?"**

2. **"Our multi-branch design builds on concepts from Homework Y - should we discuss this more?"**

3. **"We're adding ~1000 lines of new code - is this the right scope for a final project?"**

---

## Conservative Estimate (If No Specific Homework)

**If you don't have specific homework to cite:**

### **General Course Concepts Applied:**

```
✅ CNN Architectures (Weeks 1-3)
   → ResNet branches, conv layers, pooling

✅ Attention Mechanisms (Week X)
   → Softmax attention, learned weighting

✅ Multi-Task Learning Concepts (Week Y)
   → Parallel branches with shared fusion

✅ Training Deep Networks (Throughout course)
   → Optimizers, loss functions, regularization
```

**Present it as**:
> "We apply core concepts from the course—CNNs, attention mechanisms, and multi-branch architectures—to build a novel multi-scale deepfake detector."

---

## Key Message for Slides

### **One-Liner Summary:**

> **"We extend the given NPR codebase by implementing a multi-scale architecture with attention-based fusion, applying concepts from course homework on attention mechanisms and multi-branch CNNs."**

### **Visual Diagram for Slide:**

```
Original NPR Paper          Course Homework          Our Project
      (Given)                  (Learned)              (Novel)
         ↓                         ↓                      ↓
    Single-Scale     +      Attention Fusion    =   Multi-Scale NPR
    NPR @ 0.5×             Multi-Branch CNNs        with Attention
                           ResNet Blocks
```

---

## What to Say in Presentation

**Opening**:
> "Our project builds on the NPR codebase from CVPR 2024. We made minimal modifications to the original code—just 30 lines across 3 files—while adding 950 lines of new code implementing concepts we learned in homework assignments."

**Key Points**:
1. **Modified**: Model loading and configuration (minimal changes)
2. **Added**: Multi-scale architecture with attention (new implementation)
3. **Applied**: Course concepts on attention and multi-branch CNNs
4. **Novel**: Multi-scale NPR extraction and adaptive fusion strategy

**Closing**:
> "By combining the proven NPR approach with multi-scale analysis and attention mechanisms from our coursework, we create a more powerful and interpretable deepfake detector."

---

**Action Item**: Please tell me which specific homework assignment(s) you're adapting from, and I'll update this with exact references!
