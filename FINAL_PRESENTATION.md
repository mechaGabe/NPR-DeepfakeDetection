# NPR: Rethinking Up-Sampling Operations for Generalizable Deepfake Detection
## CVPR 2024

**Final Project Presentation - Generative AI Class**

---

## 1. Project Topic

**Detecting AI-Generated Images by Analyzing Upsampling Artifacts in Generative Networks**

- Focus: Generalizable deepfake detection across different generative models (GANs, Diffusion Models)
- Key Innovation: Neural Perceptual Residual (NPR) - exploits upsampling artifacts left by generative models
- Paper: "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection" (CVPR 2024)

**Why This Matters:**
- Most detection methods fail to generalize across different AI generators
- NPR achieves 91.7% average accuracy across 16 different generative models
- Critical for combating misinformation, deepfakes, and synthetic media

---

## 2. Background and Context

### 2.1 How Generative Models Create Fake Images

**GANs (Generative Adversarial Networks):**
- Use upsampling operations (nearest neighbor, bilinear) to generate high-resolution images
- Common architectures: ProGAN, StyleGAN, BigGAN, CycleGAN

**Diffusion Models:**
- Generate images through iterative denoising process
- Examples: Stable Diffusion, DALL-E 2, Midjourney, ADM

### 2.2 The Upsampling Problem

All these models must upsample features during generation:
```
Low Resolution → [Upsampling Operation] → High Resolution
  (e.g., 64×64)                             (e.g., 256×256)
```

**Key Insight:** Upsampling operations leave consistent, detectable artifacts!

### 2.3 Related Work Limitations

| Method | Limitation |
|--------|-----------|
| CNNDetection (CVPR 2020) | Poor generalization to unseen generators |
| UniversalFakeDetect (CVPR 2023) | Struggles with diffusion models |
| DIRE (ICCV 2023) | Requires reconstruction, computationally expensive |
| Frequency-based methods | Limited to specific artifact patterns |

**Gap:** Need a method that captures fundamental artifacts shared across ALL generative models.

---

## 3. Methodology: Neural Perceptual Residual (NPR)

### 3.1 Core Innovation

**The NPR Operation:**
```python
# Step 1: Downsample image by 50%
x_half = Downsample(x, factor=0.5, mode='nearest')

# Step 2: Upsample back to original size
x_reconstructed = Upsample(x_half, factor=2.0, mode='nearest')

# Step 3: Compute residual (the "fingerprint")
NPR = x - x_reconstructed
```

**Intuition:**
- Real images: Small residuals (natural images are information-rich)
- Fake images: Large residuals (upsampling artifacts amplified)

### 3.2 Architecture

```
Input Image (224×224×3)
    ↓
[NPR Computation] ← Key Innovation!
    ↓
NPR Map (224×224×3)
    ↓
[ResNet-50 Backbone]
    ↓  (conv layers)
    ↓  (feature extraction)
    ↓
[Classification Head]
    ↓
Real or Fake?
```

**Key Details:**
- Backbone: ResNet-50 (pre-trained on ImageNet, then fine-tuned)
- Training: 4-class ProGAN dataset (car, cat, chair, horse)
- Input: NPR scaled by 2.0/3.0 for normalization

### 3.3 Why NPR Works

1. **Universality:** All generative models use upsampling → all leave similar artifacts
2. **Simplicity:** Single operation captures the core discriminative signal
3. **Efficiency:** No reconstruction needed (unlike DIRE)
4. **Generalization:** Trained on ProGAN, works on unseen GANs and diffusion models

---

## 4. Hypothesis / Explorative Element

### 4.1 Original Paper Hypothesis

**"Upsampling operations in generative models leave consistent artifacts that can be amplified through downsampling-upsampling residual computation, enabling detection across diverse generators."**

### 4.2 Our Experimental Extension

**Hypothesis:** *The interpolation factor (currently 0.5) affects the detection performance differently for GAN-generated vs. Diffusion-generated images.*

**Research Questions:**
1. Does factor = 0.25 capture different artifacts than factor = 0.5?
2. Are GAN artifacts more visible at certain scales vs. diffusion artifacts?
3. Can we improve generalization by using multiple scales?

**Why This Requires Experimentation:**
- Different generators may leave artifacts at different frequency scales
- Optimal scale for ProGAN may not be optimal for Stable Diffusion
- Multi-scale approach could capture complementary information

**What We Expect:**
- Hypothesis: Smaller factors (0.25) may better detect diffusion models (finer artifacts)
- Hypothesis: Larger factors (0.75) may better preserve GAN artifacts (coarser patterns)
- Hypothesis: Multi-scale fusion could outperform single-scale NPR

---

## 5. Experimental Setup

### 5.1 Code Adaptations

**Baseline Implementation:**
- Original NPR code from https://github.com/chuangchuangtan/NPR-DeepfakeDetection
- ResNet-50 architecture with NPR preprocessing

**Our Modifications:**
```python
# Original (in networks/resnet.py, line 166)
NPR = x - self.interpolate(x, 0.5)

# Our Experiment: Test multiple factors
factors = [0.25, 0.5, 0.75]
# Compare single-scale performance

# Optional: Multi-scale fusion
NPR_025 = x - self.interpolate(x, 0.25)
NPR_050 = x - self.interpolate(x, 0.50)
NPR_075 = x - self.interpolate(x, 0.75)
NPR_multi = concat([NPR_025, NPR_050, NPR_075])
```

**Implementation Complexity:** ~10 lines of code change + experiment loop

### 5.2 Training Data

| Dataset | Size | Content | Usage |
|---------|------|---------|-------|
| ProGAN-4class | ~360K images | car, cat, chair, horse | Training |
| ForenSynths Val | ~40K images | Same classes | Validation |
| ForenSynths Test | ~80K images | 8 GAN types | Test (Table 1) |
| GANGen-Detection | ~90K images | 9 novel GANs | Test (Table 2) |
| DiffusionForensics | ~60K images | 8 diffusion models | Test (Table 3) |

**Why These Datasets:**
- **ProGAN:** Standard training set, enables comparison with baseline
- **ForenSynths:** Tests generalization to unseen GAN architectures
- **DiffusionForensics:** Critical test - models never trained on diffusion!

### 5.3 Computational Resources

**Requirements:**
- **GPU:** 1× NVIDIA GPU with ≥ 8GB VRAM (RTX 3090, V100, or similar)
- **Training Time:** ~2-3 hours per factor (50 epochs)
- **Testing Time:** ~30 minutes per test set
- **Storage:** ~100GB for datasets

**Feasibility:** ✅ Available via Google Colab Pro / University cluster

### 5.4 Planned Outputs

**Quantitative Results:**
```
Table: Single-Scale NPR Performance
-----------------------------------------
Factor | ProGAN | StyleGAN | Diffusion | Mean
-----------------------------------------
0.25   |   ?    |    ?     |     ?     |  ?
0.50   | 99.8%  |  96.1%   |  84.9%    | 92.5% (baseline)
0.75   |   ?    |    ?     |     ?     |  ?
```

**Qualitative Analysis:**
- Visualize NPR maps at different scales
- Heatmaps showing where artifacts are detected
- Per-generator breakdown (GAN vs. Diffusion)

**Statistical Validation:**
- Accuracy, Average Precision (AP), AUC per generator
- T-tests to confirm statistical significance
- Confusion matrices

---

## 6. Success Criteria

### 6.1 Minimum Success (C/B Grade)

✅ Successfully run baseline NPR model on test sets
✅ Implement factor modification and train 3 variants
✅ Generate comparative results table
✅ Document observations

### 6.2 Target Success (A Grade)

✅ All minimum criteria +
✅ Statistically significant findings (p < 0.05)
✅ Clear insights on scale-dependent artifact detection
✅ Visualization of NPR maps showing differences
✅ Analysis of GAN vs. Diffusion performance gaps

### 6.3 Outstanding Success (A+ Grade)

✅ All target criteria +
✅ Multi-scale fusion implementation showing improvement
✅ Theory explaining why certain scales work better
✅ Actionable recommendations for future deepfake detection
✅ Publication-quality figures and analysis

**Measurable Metrics:**
- **Primary:** Mean accuracy across all test sets (target: ≥ 92.5% baseline)
- **Secondary:** Per-generator improvement on diffusion models
- **Insight:** Identification of scale-artifact relationship

---

## 7. Deliverables

### 7.1 Code

- ✅ Modified `networks/resnet.py` with configurable interpolation factor
- ✅ Experiment script: `train_multiscale_experiment.py`
- ✅ Results analysis notebook: `analyze_results.ipynb`
- ✅ All code pushed to GitHub repository with clear documentation

### 7.2 Results

- ✅ CSV files with all experimental results
- ✅ Trained model checkpoints for each factor
- ✅ Visualization plots (NPR maps, accuracy curves, confusion matrices)

### 7.3 Documentation

- ✅ Final report (5-10 pages) documenting:
  - Hypothesis and motivation
  - Experimental methodology
  - Results and analysis
  - Conclusions and future work
- ✅ This presentation with speaker notes
- ✅ README with instructions to reproduce experiments

### 7.4 Presentation Materials

- ✅ Slides (this document, convertible to PowerPoint/PDF)
- ✅ Demo: Live detection on sample images
- ✅ Video (optional): 3-minute summary of findings

---

## 8. Timeline

| Week | Tasks |
|------|-------|
| Week 1 | Setup environment, download datasets, verify baseline |
| Week 2 | Implement multi-factor training, run experiments |
| Week 3 | Analyze results, create visualizations, write report |
| Week 4 | Finalize presentation, prepare for Q&A |

**Current Status:** Setup complete, ready to begin experiments

---

## 9. Expected Results & Discussion Points

### 9.1 Predicted Outcomes

**Hypothesis 1 (GAN Detection):**
- Factors 0.5-0.75 will perform best on GAN-generated images
- GANs use nearest-neighbor upsampling at 2× factors

**Hypothesis 2 (Diffusion Detection):**
- Smaller factors (0.25) may improve diffusion model detection
- Diffusion artifacts may be finer-grained

**Hypothesis 3 (Multi-Scale):**
- Concatenating multiple scales will improve overall accuracy by 2-5%

### 9.2 Potential Challenges

1. **Training Time:** Mitigate by using smaller batch sizes or fewer epochs
2. **Overfitting:** Use data augmentation and early stopping
3. **No Improvement:** Still valuable negative result - documents optimal scale

### 9.3 Discussion Questions for Feedback

1. Should we test interpolation modes (bilinear vs. nearest)?
2. Is multi-scale fusion worth the added computational cost?
3. Should we focus on one generator family (GAN vs. Diffusion)?

---

## 10. Key Takeaways

### Why This Project is Important

1. **Practical Impact:** Deepfake detection is critical for media authentication
2. **Scientific Contribution:** Understanding scale-dependent artifacts advances detection theory
3. **Accessibility:** Simple modification enables beginner-level contribution

### What Makes This Project Strong

✅ **Clear Hypothesis:** Testable prediction about scale effects
✅ **Solid Foundation:** Building on CVPR 2024 paper
✅ **Achievable Scope:** Realistic for course timeline
✅ **Measurable Success:** Quantitative metrics for validation

### Expected Learning Outcomes

- Understanding of GAN/Diffusion architectures
- Experience with deep learning training pipelines
- Scientific experiment design and execution
- Academic presentation and writing skills

---

## 11. Backup Plan

**If Primary Experiment Fails:**

1. **Alternative 1:** Test different interpolation modes (nearest, bilinear, bicubic)
2. **Alternative 2:** Apply NPR in different color spaces (RGB, YCbCr, LAB)
3. **Alternative 3:** Analyze failure cases - which generators are hardest to detect?

**All alternatives require minimal code changes and provide valuable insights.**

---

## References

1. Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection," CVPR 2024
2. Wang et al., "CNNDetection: CNN-Generated Images Are Surprisingly Easy to Spot... For Now," CVPR 2020
3. Ojha et al., "Towards Universal Fake Image Detectors," CVPR 2023
4. Wang et al., "DIRE: Diffusion-Generated Image Detection," ICCV 2023
5. Corvi et al., "Detection of Synthetic Images: A Survey," arXiv 2023

---

## Thank You!

**Questions & Feedback Welcome**

*Contact: [Your Email]*
*GitHub: https://github.com/chuangchuangtan/NPR-DeepfakeDetection*
*Paper: https://arxiv.org/abs/2312.10461*

---

# SPEAKER NOTES

## Slide 1 - Title
- Introduce yourself and team
- Mention this is a CVPR 2024 paper (top-tier conference)
- Hook: "Can you tell if an image is real or AI-generated?"

## Slide 2 - Project Topic
- Emphasize "generalizable" - this is the key challenge
- Show example images (real vs. fake) if possible
- Mention real-world applications: news verification, legal evidence, social media

## Slide 3 - Background
- Briefly explain GAN generator architecture (noise → upsampling → image)
- Key point: ALL generators must upsample
- Connect to your coursework on generative models

## Slide 4 - Methodology
- Walk through NPR computation step-by-step
- Show visual: original image → NPR map (should show artifacts)
- Analogy: "Like JPEG compression artifacts, but for AI generation"

## Slide 5 - Hypothesis
- Clearly state this is YOUR experimental contribution
- Explain why scale matters (different artifacts at different frequencies)
- Connect to signal processing concepts if audience is technical

## Slide 6 - Experimental Setup
- Emphasize feasibility: "only 10 lines of code change"
- Show you've thought through practical constraints
- Mention you have access to computational resources

## Slide 7 - Success Criteria
- Show you have realistic expectations
- Even negative results are valuable (science!)
- Tiered approach shows planning

## Slide 8 - Deliverables
- Demonstrate completeness and organization
- Mention open-source contribution aspect
- Reproducibility is key for scientific work

## Slide 9 - Expected Results
- Don't oversell - acknowledge uncertainty
- Present multiple hypotheses to show critical thinking
- Invite feedback on experimental design

## Slide 10 - Backup Plan
- Shows you've thought about risk mitigation
- Demonstrates flexibility and problem-solving
- All alternatives are tractable

## Q&A Strategy
**Likely Questions:**
1. "How long does training take?" → 2-3 hours per model on single GPU
2. "What if factor doesn't matter?" → Still valuable negative result, try alternative experiments
3. "Can this detect future AI models?" → Hypothesis: yes, because upsampling is fundamental
4. "What about computational cost?" → NPR adds minimal overhead (one operation)
5. "Why not test more factors?" → Time constraints, but could do in future work

**Tips:**
- Be honest about limitations
- Emphasize learning process over perfect results
- Show enthusiasm for the problem space
- Connect back to course concepts (GANs, diffusion, overfitting, etc.)
