# Multi-Scale Attention NPR for Generalizable Deepfake Detection
## Final Project Proposal - Generative AI

---

## 1. Project Topic

**Enhancing Generalizable Deepfake Detection Through Multi-Scale Neural Pixel Rethinking with Attention-Based Fusion**

This project extends the state-of-the-art NPR (Neural Pixel Rethinking) method from CVPR 2024 by:
- Extracting upsampling artifacts at **multiple scales** (0.25×, 0.5×, 0.75×) instead of a single scale
- Using an **attention mechanism** to adaptively weight different scales based on the input
- Investigating which scales are most discriminative for **different generative models** (GANs vs. Diffusion models)

---

## 2. Background and Context

### 2.1 Base Paper: NPR (CVPR 2024)
- **Title**: "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection"
- **Key Insight**: Generative models (GANs, Diffusion) leave distinctive artifacts in upsampling operations
- **Method**: Extracts "NPR artifacts" by computing: `NPR = Image - Downsample-then-Upsample(Image)`
- **Limitation**: Uses only a **single downsampling factor (0.5×)**

### 2.2 Motivation for Multi-Scale Approach
Different generative architectures use different upsampling strategies:
- **ProGAN**: Progressive growing with nearest-neighbor upsampling
- **StyleGAN**: Transposed convolutions
- **Diffusion Models**: Learned upsampling in U-Net architecture

**Hypothesis**: These different strategies may leave artifacts at **different spatial scales**. A single scale may miss important patterns.

### 2.3 Related Work
- **CNNDetection (CVPR 2020)**: First to detect GAN artifacts using CNNs
- **FreqNet (AAAI 2024)**: Uses frequency domain analysis
- **UniversalFakeDetect (CVPR 2023)**: Focuses on diffusion model detection
- **Multi-scale architectures**: FPN, U-Net have proven effective in computer vision

**Gap**: No prior work has explored multi-scale NPR extraction with adaptive fusion.

---

## 3. Hypothesis / Explorative Element

### 3.1 Central Hypothesis
> **"Different generative models leave upsampling artifacts at different spatial scales. A multi-scale NPR approach with attention-based fusion will achieve better generalization across diverse GAN and diffusion architectures compared to single-scale NPR."**

### 3.2 Specific Research Questions

**a) What do we expect to observe?**
1. **Scale Preferences by Generator Type**:
   - GAN-based models (ProGAN, StyleGAN) will show stronger artifacts at **coarser scales (0.25×, 0.5×)** due to progressive upsampling
   - Diffusion models will show stronger artifacts at **finer scales (0.5×, 0.75×)** due to U-Net architecture

2. **Performance Improvements**:
   - Multi-scale NPR will achieve **≥2% higher accuracy** on average across test sets
   - Largest improvements on **diffusion model detection** (Table 3-5 in original paper)

3. **Attention Patterns**:
   - Attention weights will show **distinct patterns** for different generators
   - Within-generator consistency: images from the same generator will use similar scale weights

**b) Why does this question require an experiment?**
- Cannot be determined analytically - upsampling artifacts are complex and data-dependent
- Need empirical evaluation on diverse test sets (8 GANs, 8 Diffusion models)
- Attention weights must be learned from data
- Requires ablation studies to identify optimal scale combinations

---

## 4. Experimental Setup

### 4.1 Code Implementation

**Architecture Overview**:
```
Input Image (224×224)
    ↓
[NPR@0.25x, NPR@0.5x, NPR@0.75x]  ← Extract artifacts at 3 scales
    ↓           ↓          ↓
ResNet₁     ResNet₂    ResNet₃    ← Separate feature extractors (128-dim)
    ↓           ↓          ↓
    → Attention Module ←           ← Learn adaptive weights [3]
            ↓
    Fused Features [128]
            ↓
      Classifier → Real/Fake
```

**Key Components**:
1. **NPR Extractors**: 3 modules computing artifacts at different scales
2. **Feature Branches**: 3 lightweight ResNets (BasicBlock, [2,2] layers each)
3. **Attention Module**: MLP that learns scale weights per image
4. **Classifier**: Final linear layer for binary classification

**Code Files** (all implemented):
- `networks/multiscale_npr.py`: Main model architecture
- `networks/trainer.py`: Modified to support multi-scale model
- `options/base_options.py`: Added `--model_type` and `--npr_scales` arguments
- `visualize_attention.py`: Attention weight analysis and visualization
- `run_multiscale_experiments.sh`: Automated experiment pipeline

### 4.2 Training Data

**Train Set**: ForenSynths (CNNDetection CVPR 2020)
- **Source**: ProGAN-generated images
- **Classes**: 4 object categories (car, cat, chair, horse)
- **Size**: ~40,000 images (20,000 real + 20,000 fake)
- **Split**: Train/Val following original paper

**Why appropriate**:
- Same training set as baseline for fair comparison
- ProGAN is challenging for generalization (tests on 7 other GANs + diffusion models)
- Sufficient size for training attention mechanism

### 4.3 Test Data (Generalization)

**Table 1: ForenSynths Test** (8 GAN architectures)
- ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, Deepfake
- ~1,000 images per generator

**Table 2: GANGen-Detection** (9 additional GANs)
- AttGAN, BEGAN, CramerGAN, InfoMaxGAN, MMDGAN, RelGAN, S3GAN, SNGAN, STGAN

**Table 3: DiffusionForensics** (8 Diffusion models)
- ADM, DDPM, IDDPM, LDM, PNDM, SDv1, SDv2, VQ-Diffusion

**Table 4: UniversalFakeDetect** (DALL-E, Glide, Guided Diffusion, LDM variants)

**Table 5: Diffusion1kStep** (DALL-E, DDPM, Guided-Diffusion, IDDPM, Midjourney)

### 4.4 Computational Resources

**Hardware**:
- **Minimum**: 1× NVIDIA RTX 3090 (24GB VRAM)
- **Preferred**: 1× NVIDIA A100 (40GB) for faster training

**Training Time Estimates**:
| Model | Parameters | Batch Size | Epochs | Est. Time |
|-------|-----------|------------|--------|-----------|
| Baseline (Single-Scale) | ~11M | 32 | 50 | 8 hours |
| Multi-Scale (3 scales) | ~15M | 32 | 50 | 12 hours |
| Ablation (2 scales) | ~13M | 32 | 50 | 10 hours |
| Ablation (4 scales) | ~17M | 32 | 50 | 14 hours |

**Total Training Time**: ~54 hours for all experiments
**Total Testing Time**: ~6 hours (5 models × 25 test sets × ~3 min)

**Storage Requirements**:
- Datasets: ~100GB
- Checkpoints: ~5GB (5 models × 1GB each)
- Results/Visualizations: ~2GB

**Availability**: ✅ Confirmed access to RTX 3090 GPU with sufficient time allocation

### 4.5 Planned Outputs

**Quantitative Results**:
1. **Performance Tables**: Accuracy & AP for all 5 test tables
2. **Ablation Study**: Comparison of different scale combinations
3. **Statistical Analysis**: Mean/std across generators, significance tests

**Qualitative Outputs**:
4. **Attention Weight Heatmaps**: Show which scales matter for each generator
5. **NPR Artifact Visualizations**: Side-by-side comparison of artifacts at different scales
6. **Attention Distribution Plots**: Box plots, violin plots showing weight patterns

**Analysis**:
7. **Scale Contribution Analysis**: Per-generator breakdown
8. **Failure Case Analysis**: When does multi-scale fail?

---

## 5. Success Criteria

### Measurable Success (Primary)
1. **Performance Improvement**: Multi-scale NPR achieves **≥2% higher average accuracy** than baseline on at least **2 out of 5 test tables**
2. **Generalization**: Improved performance on **diffusion models** (Tables 3-5), which are challenging for original NPR

### Qualitative Success (Secondary)
3. **Interpretability**: Attention weights show **clear, interpretable patterns** distinguishing GANs from Diffusion models
4. **Consistency**: Scale preferences are **consistent within generator families** (e.g., all StyleGAN variants)

### Reproducibility
5. **Code Quality**: Clean, documented code with clear README
6. **Checkpoints**: All trained models publicly available
7. **Results**: All experiments reproducible with provided scripts

### Presentation Quality
8. **Visualizations**: Compelling figures showing attention patterns
9. **Analysis**: Clear explanation of why certain scales matter for certain generators

---

## 6. Deliverables

### Code & Models
1. **Complete Codebase**:
   - `networks/multiscale_npr.py`: Multi-scale attention model
   - `visualize_attention.py`: Visualization tools
   - `run_multiscale_experiments.sh`: Automated experiment pipeline
   - Modified training/testing scripts

2. **Trained Models**:
   - Baseline (single-scale NPR)
   - Multi-scale attention (3 scales: 0.25×, 0.5×, 0.75×)
   - Ablation models (2 scales, 4 scales, different combinations)

### Results & Analysis
3. **Performance Tables**:
   - Tables 1-5: Accuracy/AP comparison for all test sets
   - Ablation table: Different scale combinations

4. **Visualizations**:
   - Attention weight heatmap across generators
   - NPR artifact visualizations (10 samples per generator)
   - Attention distribution plots

5. **Statistical Analysis**:
   - Significance tests (t-tests, paired comparisons)
   - Per-generator breakdown
   - Correlation analysis (attention weights vs. accuracy)

### Documentation
6. **Final Report** (4-6 pages):
   - Introduction & Related Work
   - Method: Multi-scale NPR with Attention
   - Experiments & Results
   - Analysis & Discussion
   - Conclusions & Future Work

7. **README**:
   - Installation instructions
   - Usage examples
   - Reproduction guide

### Presentation
8. **Presentation Slides** (15-20 minutes):
   - Problem motivation
   - Method overview (architecture diagram)
   - Key results (tables + visualizations)
   - Attention analysis (most compelling part!)
   - Conclusions & takeaways

9. **Speaker Notes**:
   - Detailed explanations for each slide
   - Answers to anticipated questions

---

## 7. Timeline

### Week 1 (Nov 18-24): Setup & Initial Training
- ✅ Implementation complete (already done!)
- Run baseline experiments
- Train multi-scale model (3 scales)
- Preliminary testing

### Week 2 (Nov 25-Dec 1): Ablation Studies
- Train 2-scale models (different combinations)
- Train 4-scale model
- Comprehensive testing on all datasets
- Collect performance metrics

### Week 3 (Dec 2-8): Analysis & Visualization
- Generate all visualizations
- Attention weight analysis
- Statistical significance testing
- Identify patterns and insights

### Week 4 (Dec 9): Final Deliverables
- Write final report
- Create presentation slides
- Package code and models
- Final submission

---

## 8. Anticipated Challenges & Mitigation

### Challenge 1: Computational Cost
- **Issue**: Multi-scale model has 3× more branches
- **Mitigation**: Use lightweight ResNet branches (only 2 layers), batch size optimization

### Challenge 2: Attention Mechanism Training
- **Issue**: Attention might not learn meaningful patterns
- **Mitigation**: Careful initialization, visualization during training, ablation with fixed weights

### Challenge 3: Data Imbalance Across Generators
- **Issue**: Some test sets are smaller
- **Mitigation**: Report both accuracy and AP, use weighted metrics

### Challenge 4: Overfitting to ProGAN Training Data
- **Issue**: Model might specialize to ProGAN artifacts
- **Mitigation**: Same issue as baseline - focus on relative improvement

---

## 9. Potential Extensions (If Time Permits)

1. **Cross-Modal Attention**: Attention across both scales and frequency domains
2. **Spatial Attention**: Which image regions contribute most at each scale?
3. **Learnable Scales**: Instead of fixed [0.25, 0.5, 0.75], learn optimal scale values
4. **Ensemble Methods**: Combine multi-scale with other detectors

---

## 10. References

1. Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection", CVPR 2024
2. Wang et al., "CNNDetection: CNN-Generated Images Are Surprisingly Easy to Spot... For Now", CVPR 2020
3. Li et al., "Universal Fake Image Detector", CVPR 2023
4. Ojha et al., "Towards Universal Fake Image Detectors that Generalize Across Generative Models", CVPR 2023

---

## Appendix A: Example Training Command

```bash
# Train multi-scale attention model
CUDA_VISIBLE_DEVICES=0 python train.py \
    --name multiscale_attention_3scales \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5,0.75 \
    --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse \
    --batch_size 32 \
    --lr 0.0002 \
    --niter 50 \
    --delr_freq 10
```

## Appendix B: Example Testing Command

```bash
# Test on StyleGAN
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_path ./checkpoints/multiscale/model_epoch_last.pth \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5,0.75 \
    --dataroot ./datasets/Generalization_Test/ForenSynths_test/stylegan \
    --batch_size 64
```

## Appendix C: Attention Visualization Command

```bash
# Visualize attention for StyleGAN
python visualize_attention.py \
    --model_path ./checkpoints/multiscale/model_epoch_last.pth \
    --dataroot ./datasets/test/stylegan \
    --output_dir ./visualizations/stylegan \
    --scales 0.25,0.5,0.75 \
    --num_samples 100 \
    --save_npr_maps
```

---

**Project Team**: [Your Name]
**Course**: Generative AI
**Submission Date**: December 9, 2024
**Presentation Date**: November 17-18, 2024
