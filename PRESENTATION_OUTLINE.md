# Attention-Weighted Multi-Scale NPR for Deepfake Detection
## Graduate AI Project Presentation

---

## 1. PROJECT TOPIC

**Title:** Adaptive Multi-Scale Natural Pattern Region Detection with Attention Mechanisms for Generalizable Deepfake Detection

**Core Concept:** Enhancing NPR-based deepfake detection by introducing learned attention mechanisms to automatically weigh multiple NPR scales, improving detection accuracy and generalization to emerging 2025 generators.

**Key Innovation:** Moving from fixed-scale NPR extraction (0.5x scale) to adaptive, attention-weighted multi-scale NPR that learns optimal scale combinations based on input characteristics.

---

## 2. BACKGROUND AND CONTEXT

### 2.1 Natural Pattern Region (NPR) Detection

**Original NPR Method** (CVPR 2024):
- Exploits upsampling artifacts inherent to generative models
- Computation: NPR = Original_Image - Downsample_Upsample(Image)
- Current limitation: Fixed 0.5x downsampling scale
- Achieves 91.7% average accuracy across multiple generators

**Key Insight:** Generative models (GANs, Diffusion) introduce detectable artifacts during upsampling operations. NPR captures these artifacts as residual patterns.

### 2.2 Why Fixed-Scale NPR is Limiting

**Generator Diversity:**
- Different generators operate at different scales
- ProGAN: Progressive generation from 4x4 → 1024x1024
- Diffusion Models: Multi-resolution noise scheduling
- FLUX, Midjourney v6: Unknown proprietary architectures

**Fixed 0.5x Scale Issues:**
- May miss artifacts at other scales (0.25x, 0.75x)
- Suboptimal for generators with different upsampling patterns
- Cannot adapt to input-specific characteristics

### 2.3 Related Work

**Multi-Scale Approaches:**
- Feature Pyramid Networks (FPN) for object detection
- Multi-resolution analysis in frequency-domain deepfake detection

**Attention Mechanisms:**
- Squeeze-and-Excitation Networks (SENet) - channel attention
- Convolutional Block Attention Module (CBAM) - spatial + channel
- Self-attention in Vision Transformers

**Gap in Literature:** No prior work applies learned attention to weight multi-scale NPR features for deepfake detection.

---

## 3. HYPOTHESES / EXPLORATIVE ELEMENT

### H2: Scale-Adaptive Attention Improves Detection Accuracy

**Hypothesis:** An attention mechanism can learn to automatically weigh NPR scales based on input characteristics, improving detection accuracy over fixed-scale approaches.

**Rationale:**
- Different generators leave artifacts at different scales
- Input-specific characteristics (texture, frequency content) should inform optimal scale selection
- Learned attention can capture these relationships better than fixed scales

**Experimental Question:** Can a neural attention module learn to assign higher weights to NPR scales that contain more discriminative artifacts for a given input?

### H3: Multi-Scale NPR Enhances Generalization

**Hypothesis:** Attention-weighted multi-scale NPR will generalize better to unseen 2025 generators (FLUX, Midjourney v6, DALL-E 3) than single-scale NPR.

**Rationale:**
- Emerging generators use novel upsampling strategies
- Multi-scale representation captures broader artifact spectrum
- Attention-based weighting adapts to unseen artifact patterns

**Experimental Question:** Does multi-scale NPR with attention maintain higher accuracy on out-of-distribution generators compared to single-scale baseline?

---

## 4. EXPERIMENTAL SETUP

### 4.1 Proposed Architecture

**Architecture Name:** Attention-NPR-ResNet50

**Pipeline:**
```
Input Image (3×H×W)
    ↓
Multi-Scale NPR Extraction (Parallel)
    ├→ NPR_0.25 (0.25x downsample → upsample)
    ├→ NPR_0.50 (0.5x downsample → upsample)  [original]
    ├→ NPR_0.75 (0.75x downsample → upsample)
    └→ NPR_1.00 (no downsampling, identity)
    ↓
Channel-wise Concatenation → (12×H×W)
    ↓
Attention Module (SENet-style)
    ├→ Global Average Pooling → (12×1×1)
    ├→ FC1: 12 → 6 → ReLU
    ├→ FC2: 6 → 12 → Sigmoid
    └→ Scale Weights: [w1, w2, w3, w4]
    ↓
Weighted NPR Fusion
    NPR_fused = w1*NPR_0.25 + w2*NPR_0.50 + w3*NPR_0.75 + w4*NPR_1.00
    ↓
ResNet50 Backbone
    ├→ Conv1 (3×3, stride=2): 3→64
    ├→ Layer1: BasicBlock×3
    ├→ Layer2: BasicBlock×4 (stride=2)
    ├→ Layer3: BasicBlock×6 (stride=2)
    ├→ Layer4: BasicBlock×3 (stride=2)
    ├→ Global Average Pool
    └→ FC: 512→1 (Binary Classification)
    ↓
Output: Real/Fake Prediction
```

**Key Components:**
1. **Multi-Scale NPR Extractor:** Generates 4 NPR representations at different scales
2. **Attention Module:** Learns scale importance weights (channel attention)
3. **Fusion Layer:** Combines NPR scales using learned weights
4. **ResNet50 Classifier:** Extracts features and performs binary classification

### 4.2 Code Implementation Plan

**Base Code:** Adaptation of existing `networks/resnet.py`

**New Modules:**
1. `MultiScaleNPR` class (lines 153-180 in resnet.py)
   - Parallel NPR extraction at 4 scales
   - Concatenation layer

2. `ScaleAttention` class (new, ~50 lines)
   - Squeeze: Global Average Pooling
   - Excitation: Two FC layers with reduction ratio=2
   - Output: Per-scale weights

3. Modified `NPRResNet` class
   - Integration of multi-scale extraction + attention
   - Weighted fusion before ResNet backbone

**Estimated Development Time:** 2-3 days for core implementation

### 4.3 Training Data

**Primary Training Set:** ForenSynths (4-class)
- **Size:** ~240,000 images (60K per class)
- **Classes:** car, cat, chair, horse
- **Generators:** ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake
- **Resolution:** 256×256, center crop to 224×224

**Rationale:**
- Diverse generator coverage (GANs + Deepfakes)
- Large enough for attention module to learn meaningful scale weights
- Established baseline from original NPR paper

**Data Split:**
- Train: 80% (~192K images)
- Validation: 20% (~48K images)

**Augmentation:**
- Random horizontal flip (p=0.5)
- Random crop (224×224)
- Optional: Gaussian blur (σ=0.5), JPEG compression (quality 90-100)

### 4.4 Generalization Test Sets

**Seen Generators (In-Distribution):**
- ForenSynths test set (8 generators)
- GANGen-Detection (9 GAN architectures)

**Unseen 2025 Generators (Out-of-Distribution):**
- **FLUX:** Stable Diffusion successor (open-source, 2024)
- **Midjourney v6:** Proprietary diffusion model (released Feb 2024)
- **DALL-E 3:** OpenAI's latest text-to-image (Sept 2023)

**Collection Strategy:**
- FLUX: Generate 5,000 images using official model
- Midjourney v6: Collect 5,000 images from public datasets
- DALL-E 3: Use existing UniversalFakeDetect dataset samples

### 4.5 Computational Resources

**Available Hardware:**
- NVIDIA RTX 3090 (24GB VRAM) or A100 (40GB)
- 32-core CPU, 128GB RAM

**Training Requirements:**

| Component | Memory | Time | Notes |
|-----------|--------|------|-------|
| Baseline NPR-ResNet50 | 12GB | 18h (50 epochs) | Existing implementation |
| Attention-NPR-ResNet50 | 18GB | 24h (50 epochs) | +4 NPR scales, attention |
| Ablation Studies | 18GB | 72h | 3 configurations |
| Testing/Validation | 8GB | 4h | All test sets |
| **Total Estimated** | **24GB peak** | **~120 hours** | ~5 days continuous |

**Batch Size:** 32 (fits in 24GB VRAM)
**Total Training Time:** 5-7 days including experiments

**Resource Availability:** ✓ Confirmed sufficient

### 4.6 Training Configuration

**Optimizer:** Adam (β1=0.9, β2=0.999)
**Learning Rate:** 2e-4 (decay 0.9 every 10 epochs)
**Loss Function:** BCEWithLogitsLoss
**Epochs:** 50
**Early Stopping:** Patience=10 epochs on validation loss

**Attention Module Specifics:**
- Reduction ratio: 2 (12 → 6 → 12)
- Activation: ReLU (hidden), Sigmoid (output)
- Initialization: Kaiming normal (weights), uniform 0.25 (biases for equal scale weighting initially)

### 4.7 Experimental Protocol

**Phase 1: Baseline Reproduction** (Week 1)
- Train single-scale NPR-ResNet50 (0.5x)
- Validate on ForenSynths test set
- Establish baseline metrics

**Phase 2: Multi-Scale Attention Implementation** (Week 2)
- Implement 4-scale NPR extraction
- Add attention module
- Train Attention-NPR-ResNet50

**Phase 3: Ablation Studies** (Week 3)
- Fixed equal weights (no attention): 0.25 each scale
- Fixed learned weights (non-adaptive)
- Adaptive attention (full model)

**Phase 4: Generalization Testing** (Week 4)
- Test on seen generators (GANGen, DiffusionForensics)
- Test on 2025 unseen generators (FLUX, Midjourney v6, DALL-E 3)
- Statistical significance testing (McNemar's test)

### 4.8 Expected Outputs

**Quantitative Metrics:**
1. Accuracy, Average Precision (AP) per generator
2. Attention weight distributions (visualizations)
3. ROC curves for baseline vs. attention model
4. Generalization gap: (Acc_seen - Acc_unseen)

**Qualitative Outputs:**
1. Attention heatmaps showing which scales activate for different generators
2. Failure case analysis (which images are misclassified)
3. t-SNE embeddings of NPR features (baseline vs. multi-scale)

---

## 5. SUCCESS CRITERIA

### 5.1 Primary Success Metrics

**Hypothesis H2 (Scale-Adaptive Attention):**
- **Criterion:** Attention-NPR achieves ≥2% accuracy improvement over baseline on ForenSynths test set
- **Baseline:** 91.7% (original NPR paper)
- **Target:** ≥93.7%
- **Statistical Test:** Paired t-test, p<0.05

**Hypothesis H3 (Generalization to 2025 Generators):**
- **Criterion:** Attention-NPR shows smaller generalization gap than baseline
- **Metric:** Δ_Acc = Acc_seen - Acc_unseen
- **Target:** Δ_Acc < 5% (vs. baseline Δ_Acc ≥ 8%)
- **Generators:** FLUX, Midjourney v6, DALL-E 3

### 5.2 Secondary Success Indicators

**Interpretability:**
- Attention weights demonstrate meaningful scale selection
  - Example: Diffusion models → higher weight on 0.25x scale
  - Example: GANs → higher weight on 0.5x scale

**Efficiency:**
- Inference time increase < 30% vs. baseline
- Memory footprint < 2x baseline

**Robustness:**
- Maintains performance under JPEG compression (quality 75-100)
- Stable across different image resolutions (224×224, 512×512)

### 5.3 Minimal Acceptable Outcome

If full hypotheses are not confirmed:
- Attention mechanism learns non-uniform scale weights (interpretability gain)
- Maintains baseline performance (no regression)
- Provides insights for future multi-scale fusion strategies

---

## 6. DELIVERABLES

### 6.1 Code and Models

1. **Source Code Repository**
   - `networks/attention_npr.py` - Multi-scale NPR + attention module
   - `train_attention.py` - Training script with attention-specific logging
   - `test_attention.py` - Testing with attention weight visualization
   - Documentation: README_ATTENTION.md

2. **Trained Model Checkpoints**
   - `attention_npr_best.pth` - Best validation accuracy model
   - `baseline_npr_reproduced.pth` - Baseline reproduction for fair comparison

3. **Configuration Files**
   - `configs/attention_npr.yaml` - Hyperparameters, scales, attention settings

### 6.2 Experimental Results

1. **Quantitative Results**
   - `results/metrics.csv` - Accuracy, AP, F1 per generator and test set
   - `results/attention_weights.npz` - Learned attention weights per test image
   - `results/confusion_matrices/` - Per-generator confusion matrices

2. **Visualizations**
   - `figures/roc_curves.pdf` - ROC comparison (baseline vs. attention)
   - `figures/attention_heatmaps.pdf` - Scale weight distributions by generator
   - `figures/generalization_plot.pdf` - Seen vs. unseen accuracy comparison
   - `figures/tsne_embeddings.pdf` - Feature space visualization

3. **Statistical Analysis**
   - `analysis/significance_tests.ipynb` - Jupyter notebook with statistical tests
   - `analysis/ablation_study.pdf` - Ablation study results table

### 6.3 Documentation

1. **Final Report** (8-10 pages, IEEE format)
   - Introduction & Related Work
   - Methodology (architecture, training)
   - Experiments & Results
   - Discussion & Limitations
   - Conclusion & Future Work
   - References

2. **Presentation Slides** (15-20 minutes)
   - Clear architecture diagram
   - Key results with visualizations
   - Demo: Real-time attention weight visualization (if time permits)

3. **Supplementary Materials**
   - `SUPPLEMENTARY.pdf` - Additional ablation studies, failure cases
   - `DATASET_DETAILS.md` - Dataset statistics, collection process for 2025 generators

### 6.4 Reproducibility Package

- `requirements.txt` - Python dependencies
- `scripts/reproduce_results.sh` - One-click reproduction script
- `REPRODUCTION_GUIDE.md` - Step-by-step instructions
- Pre-trained models hosted on Google Drive/HuggingFace

---

## 7. TIMELINE

### Week 1 (Nov 18-24): Baseline & Implementation
- **Day 1-2:** Reproduce baseline NPR-ResNet50, validate metrics
- **Day 3-5:** Implement multi-scale NPR extraction (4 scales)
- **Day 6-7:** Implement attention module, integrate with ResNet50

### Week 2 (Nov 25-Dec 1): Training & Initial Testing
- **Day 1-3:** Train Attention-NPR-ResNet50 on ForenSynths
- **Day 4-5:** Ablation study (fixed weights, no attention)
- **Day 6-7:** Test on seen generators (GANGen, DiffusionForensics)

### Week 3 (Dec 2-8): Generalization & Analysis
- **Day 1-2:** Collect/prepare 2025 generator datasets (FLUX, Midjourney v6, DALL-E 3)
- **Day 3-4:** Test on unseen generators
- **Day 5-7:** Statistical analysis, visualization generation

### Week 4 (Dec 9 - Submission): Documentation & Refinement
- **Day 1-2:** Write final report
- **Day 3:** Create/polish presentation slides
- **Day 4:** Code cleanup, documentation
- **Day 5:** Final review, reproducibility testing
- **Dec 9, 10:00 AM:** **Submission Deadline**

**Critical Milestones:**
- ✓ Nov 17, 5:00 PM: Presentation submission
- ✓ Nov 24: Baseline reproduced + multi-scale implemented
- ✓ Dec 1: Training complete, initial results
- ✓ Dec 8: All experiments done, analysis complete
- ✓ Dec 9, 10:00 AM: Final project submission

---

## 8. LIMITATIONS & RISKS

### 8.1 Technical Limitations

**Computational Constraints:**
- Limited to 4 NPR scales due to memory constraints (12GB → 24GB VRAM)
- Cannot test all possible scale combinations (would require grid search)

**Dataset Limitations:**
- 2025 generator images may be limited in availability
- Midjourney v6 images lack ground truth generation parameters
- FLUX is very recent (limited diverse samples)

**Model Limitations:**
- Attention module is relatively simple (SENet-style channel attention)
- Does not explore spatial attention or self-attention mechanisms
- Fixed ResNet50 backbone (no architectural search)

### 8.2 Experimental Risks

**Risk 1: Attention Module Fails to Learn**
- *Mitigation:* Carefully initialize biases to 0.25 (equal weighting)
- *Mitigation:* Monitor attention weights during training
- *Fallback:* Manual scale selection based on generator type

**Risk 2: Generalization Gap Remains Large**
- *Mitigation:* Augment training with partial 2025 generator data (few-shot)
- *Mitigation:* Ensemble approach if single model fails

**Risk 3: Training Time Exceeds Estimates**
- *Mitigation:* Reduce epochs to 30 if needed
- *Mitigation:* Prioritize H2 over H3 if time-constrained

### 8.3 Validity Threats

**Internal Validity:**
- Different training runs may yield varying results (random seed dependency)
- *Mitigation:* Report mean ± std over 3 random seeds

**External Validity:**
- Results may not generalize beyond tested generators
- Real-world deepfakes may include post-processing (compression, filtering)
- *Mitigation:* Test with JPEG-compressed inputs

**Construct Validity:**
- Accuracy may not fully capture detection utility (false positive cost)
- *Mitigation:* Report precision, recall, F1 alongside accuracy

---

## 9. EXPECTED CONTRIBUTIONS

### 9.1 Scientific Contributions

1. **First application** of learned attention to multi-scale NPR deepfake detection
2. **Empirical evidence** on optimal NPR scales for different generator types
3. **Generalization benchmark** for 2025 state-of-the-art generators (FLUX, Midjourney v6)

### 9.2 Practical Contributions

1. **Improved detector** with minimal architectural overhead (~10% parameter increase)
2. **Interpretable weights** reveal which scales matter for each generator
3. **Open-source implementation** enabling future multi-scale NPR research

---

## 10. REFERENCES

1. Chai, L., Bau, D., Lim, S. N., & Isola, P. (2024). "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." CVPR 2024.

2. Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." CVPR 2018.

3. Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). "CBAM: Convolutional Block Attention Module." ECCV 2018.

4. Wang, S. Y., Wang, O., Zhang, R., Owens, A., & Efros, A. A. (2020). "CNN-Generated Images Are Surprisingly Easy to Spot... for Now." CVPR 2020.

5. Corvi, R., Cozzolino, D., Poggi, G., Nagano, K., & Verdoliva, L. (2023). "Intriguing Properties of Synthetic Images: From Generative Adversarial Networks to Diffusion Models." CVPR 2023.

---

## APPENDIX: ARCHITECTURE DIAGRAM (Text Representation)

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE (3×224×224)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │  MULTI-SCALE NPR BLOCK  │
                └────────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    ┌───▼───┐           ┌───▼───┐           ┌───▼───┐
    │ NPR   │           │ NPR   │           │ NPR   │
    │ 0.25x │           │ 0.50x │           │ 0.75x │
    └───┬───┘           └───┬───┘           └───┬───┘
        │                   │                   │
        │     (3×224×224 each)                 │
        └───────────┬───────┴───────┬───────────┘
                    │               │
              ┌─────▼───────────────▼─────┐
              │   CHANNEL CONCATENATION   │
              │       (12×224×224)        │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │    ATTENTION MODULE       │
              │  ┌─────────────────────┐  │
              │  │ Global Avg Pool     │  │
              │  │    (12×1×1)         │  │
              │  └──────────┬──────────┘  │
              │  ┌──────────▼──────────┐  │
              │  │ FC: 12→6 (ReLU)    │  │
              │  └──────────┬──────────┘  │
              │  ┌──────────▼──────────┐  │
              │  │ FC: 6→12 (Sigmoid)  │  │
              │  └──────────┬──────────┘  │
              │  ┌──────────▼──────────┐  │
              │  │ Reshape: (4,3,1,1)  │  │
              │  │ Scale Weights       │  │
              │  │ [w1,w2,w3,w4]×3ch  │  │
              │  └─────────────────────┘  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   WEIGHTED FUSION         │
              │ NPR_fused = Σ(wi × NPRi) │
              │      (3×224×224)          │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   RESNET50 BACKBONE       │
              │  ┌─────────────────────┐  │
              │  │ Conv1: 3→64, s=2    │  │
              │  │ MaxPool: s=2        │  │
              │  │ Layer1: 64→256      │  │
              │  │ Layer2: 256→512, s=2│  │
              │  │ Layer3: 512→1024,s=2│  │
              │  │ Layer4: 1024→2048,s=2│ │
              │  │ Global Avg Pool     │  │
              │  │ FC: 2048→1          │  │
              │  └─────────────────────┘  │
              └─────────────┬─────────────┘
                            │
                ┌───────────▼───────────┐
                │  BCEWithLogitsLoss    │
                │  Output: Real/Fake    │
                └───────────────────────┘
```

**Key Innovation (Highlighted):**
- Multi-scale NPR extraction (0.25x, 0.50x, 0.75x) captures artifacts at different frequencies
- Attention module learns input-specific scale weights
- Weighted fusion creates adaptive NPR representation

---

## PRESENTATION FLOW SUGGESTIONS

**Slide 1:** Title + Team
**Slide 2:** Motivation - Why deepfake detection matters (2025 generators)
**Slide 3:** Background - NPR concept (visual: upsampling artifacts)
**Slide 4:** Problem - Fixed-scale limitation
**Slide 5:** Our Approach - Multi-scale + Attention (architecture diagram)
**Slide 6:** Hypotheses H2 & H3 (clear, testable statements)
**Slide 7:** Experimental Design - Training data, baselines
**Slide 8:** Timeline (Gantt chart visual)
**Slide 9:** Hardware & Computational Plan
**Slide 10:** Success Criteria (table: baseline vs. target metrics)
**Slide 11:** Expected Results (mock attention heatmap visualization)
**Slide 12:** Limitations & Risks
**Slide 13:** Deliverables (checklist)
**Slide 14:** Questions & Discussion Points
**Slide 15:** References

**Total:** 15 slides, ~1 min per slide = 15 min + 5 min Q&A

---

*This presentation outline is designed for graduate-level academic rigor while maintaining clarity and professional focus on the explorative hypotheses.*
