# Methodology Quick Reference - One Pager

## 🎯 Core Methodology

**Research Question**: Do different generators leave artifacts at different scales?

**Approach**: Multi-Scale NPR with Attention Fusion
- Extract artifacts at 3 scales: 0.25×, 0.5×, 0.75×
- Process each with separate ResNet branches
- Attention module learns adaptive weights
- Compare vs. single-scale baseline (0.5×)

---

## 📊 Data at a Glance

| Aspect | Details |
|--------|---------|
| **Training** | ForenSynths: 40K images (ProGAN, 4 classes) |
| **Testing** | 50K+ images across 25+ generators (5 tables) |
| **Input** | 224×224 RGB images |
| **Splits** | 80% train / 20% validation |

**Test Diversity**:
- Table 1: 8 GANs (ProGAN, StyleGAN, etc.)
- Table 2: 9 GANs (AttGAN, BEGAN, etc.)
- Table 3: 8 Diffusion (DDPM, LDM, SDv1/2)
- Table 4-5: Advanced (DALL-E, Midjourney, Glide)

---

## 🏗️ Architecture Summary

**Baseline**: `Image → NPR@0.5× → ResNet-50 → Classifier` (11M params)

**Ours**:
```
Image → [NPR@0.25×, NPR@0.5×, NPR@0.75×]
          ↓         ↓         ↓
      ResNet₁   ResNet₂   ResNet₃
          ↓         ↓         ↓
        [128-dim features each]
                  ↓
          Attention Module
          (learns weights)
                  ↓
          Weighted Fusion
                  ↓
            Classifier
```
**Parameters**: 15M (+36% vs. baseline)

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.0002) |
| Batch Size | 32 |
| Epochs | 50 |
| LR Decay | ×0.9 every 10 epochs |
| Loss | Binary Cross-Entropy |
| Hardware | RTX 3090 (24GB) |
| Training Time | 12.5 hours (multi-scale) |

---

## 🧪 Experiments

1. **Baseline**: Single scale (0.5×) - 8 hours
2. **Multi-Scale**: 3 scales (0.25×, 0.5×, 0.75×) - 12 hours ⭐
3. **Ablation 1**: 2 scales coarse (0.25×, 0.5×) - 10 hours
4. **Ablation 2**: 2 scales fine (0.5×, 0.75×) - 10 hours
5. **Ablation 3**: 4 scales (0.2×, 0.4×, 0.6×, 0.8×) - 14 hours

**Total GPU Time**: 56 hours (~2.5 days)

---

## 📈 Evaluation Metrics

- **Accuracy**: % correct classifications
- **Average Precision (AP)**: Area under PR curve
- **Per-Generator**: Individual performance analysis
- **Attention Analysis**: Weight patterns per generator

**Success Criteria**: ≥2% improvement on ≥2 test tables

---

## 🔬 NPR Extraction Process

```python
For each scale s ∈ {0.25, 0.5, 0.75}:
  1. x_down = Downsample(image, scale=s, mode='nearest')
  2. x_recon = Upsample(x_down, scale=1/s, mode='nearest')
  3. NPR_s = image - x_recon  # Extract artifact
  4. features_s = ResNet_branch(NPR_s)  # 128-D
```

**Why Nearest-Neighbor?** Creates blocky artifacts different from GAN upsampling

---

## 📅 Timeline

- **Week 1** (Nov 11-17): ✅ Implementation complete
- **Week 2-3** (Nov 18-Dec 1): Training + testing (56 GPU hours)
- **Week 4** (Dec 2-8): Analysis + visualization + report
- **Dec 9**: Final submission

---

## 🎯 Expected Results

**Hypothesis**:
- **GANs**: Higher attention on coarse scales (0.25×, 0.5×)
  - Reason: Progressive upsampling
- **Diffusion**: Higher attention on fine scales (0.5×, 0.75×)
  - Reason: U-Net subtle artifacts

**Performance Target**: +2-5% accuracy improvement over baseline

---

## 💾 Computational Resources

| Resource | Requirement | Available |
|----------|-------------|-----------|
| GPU | RTX 3090 (24GB) | ✅ Yes |
| Training Time | 56 hours | ✅ Feasible |
| Storage | 150GB | ✅ Sufficient |
| Memory | 11.2GB (batch=32) | ✅ Fits |

**Inference Speed**: 83 images/sec (real-time capable)

---

## 📊 Key Deliverables

1. **Models**: Baseline + Multi-scale + 3 ablations
2. **Results**: Performance tables (5 test sets × 5 models)
3. **Visualizations**:
   - Attention heatmaps per generator
   - NPR artifact comparisons
   - Statistical distribution plots
4. **Analysis**: Which scales matter for which generators?
5. **Code**: Fully reproducible with documentation

---

## 🔑 Key Talking Points

1. **Innovation**: First work to explore multi-scale NPR with attention
2. **Efficiency**: Only +36% parameters for 3× scale coverage
3. **Interpretability**: Attention weights reveal generator characteristics
4. **Generalization**: Test on 25+ unseen generators
5. **Practical**: Real-time inference (83 img/s)

---

## ❓ Anticipated Questions & Answers

**Q: Why not just train 3 separate models?**
> Attention allows joint learning and cross-scale information sharing. More efficient than 3 independent models.

**Q: What if multi-scale doesn't help?**
> Still valuable! Negative results inform future work. Attention patterns still provide insights.

**Q: How did you choose scales 0.25, 0.5, 0.75?**
> 0.5× is baseline. We test coarser (0.25×) and finer (0.75×). Ablations explore alternatives.

**Q: Computational cost too high?**
> Only 50% more training time than baseline. Inference is 83 img/s—still real-time.

---

## 🎨 Visual Elements Needed

For your slides, prepare:
- ✅ Architecture diagram (3 branches with attention)
- ✅ NPR extraction illustration
- ✅ Data distribution chart
- ✅ Timeline Gantt chart
- ✅ Expected attention heatmap (mock-up)
- ✅ Performance comparison bar chart

---

## 📝 One-Sentence Summary

> **"We extend NPR deepfake detection by extracting upsampling artifacts at multiple scales and using an attention mechanism to adaptively weight which scales matter most for each generator type."**

---

**Use this for**: Quick reference during presentation prep, answering questions, or creating summary slides.
