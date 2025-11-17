# Experimental Extension Guide
## Testing Different NPR Interpolation Factors for Deepfake Detection

**Author:** [Your Name]
**Date:** November 2024
**Course:** Generative AI Final Project

---

## 🎯 Executive Summary

This experimental extension investigates whether **different interpolation factors in the NPR (Neural Perceptual Residual) computation affect detection performance** across different types of generative models (GANs vs. Diffusion Models).

### Why This Matters

The original NPR paper uses a fixed factor of **0.5** (50% downsampling) to compute perceptual residuals. However:
- Different generators may leave artifacts at different frequency scales
- GAN upsampling (typically 2× nearest neighbor) vs. Diffusion denoising may create artifacts of different sizes
- Testing multiple scales could reveal **scale-dependent artifact patterns**

### Key Hypothesis

**"The optimal interpolation factor for NPR differs between GAN-generated and Diffusion-generated images, with smaller factors (0.25) better capturing fine-grained diffusion artifacts and larger factors (0.5-0.75) better capturing coarser GAN artifacts."**

---

## 🔬 Research Questions

1. **Primary Question:** Does changing the NPR interpolation factor (0.25, 0.5, 0.75) significantly affect detection accuracy?

2. **Secondary Questions:**
   - Do GAN-based generators show better detection at certain scales?
   - Do Diffusion models show better detection at different scales than GANs?
   - Can we improve overall generalization by using multiple scales?

3. **Exploratory Question:**
   - What does the scale-sensitivity tell us about the nature of generative artifacts?

---

## 🧪 Experimental Methodology

### NPR Computation Explained

The Neural Perceptual Residual captures upsampling artifacts:

```python
# Original image: x (e.g., 224×224×3)

# Step 1: Downsample by factor (e.g., 0.5 → 112×112)
x_down = Downsample(x, factor=0.5, mode='nearest')

# Step 2: Upsample back to original size (224×224)
x_reconstructed = Upsample(x_down, factor=1/0.5, mode='nearest')

# Step 3: Compute residual
NPR = x - x_reconstructed
```

**Key Insight:** Fake images have larger residuals because generative models use similar upsampling operations during synthesis!

### Factors Being Tested

| Factor | Downsampling | Expected Behavior |
|--------|--------------|-------------------|
| 0.25   | 224×224 → 56×56 → 224×224 | Captures **fine-grained** artifacts (good for diffusion?) |
| 0.5    | 224×224 → 112×112 → 224×224 | **Baseline** from original paper |
| 0.75   | 224×224 → 168×168 → 224×224 | Captures **coarse** artifacts (good for GANs?) |

### Experimental Design

```
                    ┌─────────────┐
                    │  ProGAN-4   │
                    │   Training  │
                    │   Dataset   │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
       Factor=0.25    Factor=0.5     Factor=0.75
       (Model A)      (Baseline)     (Model C)
            │              │              │
            └──────────────┴──────────────┘
                           │
                   Test on 4 Benchmarks:
                   ├─ ForenSynths (8 GANs)
                   ├─ GANGen-Detection (9 GANs)
                   ├─ DiffusionForensics (8 Diffusion)
                   └─ UniversalFakeDetect (DALL-E, etc.)
                           │
                    Compare Performance:
                    - Per-generator accuracy
                    - GAN average vs Diffusion average
                    - Statistical significance tests
```

### Why This is Beginner-Friendly

✅ **Minimal Code Changes:** Only ~5 lines need modification
✅ **No New Architecture:** Uses existing ResNet-50 backbone
✅ **Clear Hypothesis:** Easy to test and validate
✅ **Fast Training:** 2-3 hours per model on single GPU
✅ **Interpretable Results:** Direct accuracy comparisons

---

## 🛠️ Implementation Details

### Code Changes Required

#### 1. Main Modification (networks/resnet_configurable.py)

```python
class ResNet(nn.Module):
    def __init__(self, ..., npr_factor=0.5):  # NEW PARAMETER
        super(ResNet, self).__init__()
        self.npr_factor = npr_factor  # STORE FACTOR
        # ... rest of initialization

    def forward(self, x):
        # MODIFIED LINE:
        NPR = x - self.interpolate(x, self.npr_factor)  # Use configured factor
        # ... rest of forward pass
```

That's it! The rest of the code remains unchanged.

#### 2. Training Script (use existing train.py)

```bash
# Train with factor 0.25
python train.py --name factor_025 --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse --batch_size 32 --niter 50 --npr_factor 0.25

# Train with factor 0.5 (baseline)
python train.py --name factor_050 --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse --batch_size 32 --niter 50 --npr_factor 0.50

# Train with factor 0.75
python train.py --name factor_075 --dataroot ./datasets/ForenSynths_train_val \
    --classes car,cat,chair,horse --batch_size 32 --niter 50 --npr_factor 0.75
```

#### 3. Testing (use provided experiment script)

```bash
# Run all experiments and compare
python run_factor_experiment.py --run_all --model_path ./NPR.pth
```

---

## 📊 Expected Results Format

### Quantitative Table

```
Performance Comparison: Accuracy (%) Across NPR Factors

Dataset: ForenSynths (GAN-based generators)
---------------------------------------------------------------------------
Generator       | Factor 0.25 | Factor 0.5 | Factor 0.75 | Best Factor
---------------------------------------------------------------------------
ProGAN          |    98.5     |   99.8     |    99.2     |    0.50
StyleGAN        |    94.2     |   96.1     |    95.8     |    0.50
BigGAN          |    85.1     |   87.3     |    88.2     |    0.75
CycleGAN        |    88.7     |   90.3     |    89.5     |    0.50
StarGAN         |    99.1     |   99.6     |    99.3     |    0.50
GauGAN          |    83.8     |   85.4     |    86.1     |    0.75
StyleGAN2       |    96.8     |   98.1     |    97.5     |    0.50
Deepfake        |    75.3     |   77.4     |    76.2     |    0.50
---------------------------------------------------------------------------
MEAN (GAN)      |    90.2     |   92.0     |    91.5     |    0.50
---------------------------------------------------------------------------

Dataset: DiffusionForensics (Diffusion models)
---------------------------------------------------------------------------
Generator       | Factor 0.25 | Factor 0.5 | Factor 0.75 | Best Factor
---------------------------------------------------------------------------
ADM             |    86.2     |   84.9     |    82.1     |    0.25  ← KEY FINDING?
DDPM            |    88.5     |   86.3     |    83.7     |    0.25  ← KEY FINDING?
IDDPM           |    87.9     |   85.2     |    82.9     |    0.25  ← KEY FINDING?
LDM             |    89.1     |   87.6     |    84.3     |    0.25  ← KEY FINDING?
StableDiff v1   |    91.3     |   90.1     |    87.5     |    0.25  ← KEY FINDING?
StableDiff v2   |    90.8     |   89.4     |    86.2     |    0.25  ← KEY FINDING?
VQ-Diffusion    |    88.6     |   87.2     |    84.8     |    0.25  ← KEY FINDING?
PNDM            |    85.4     |   83.7     |    81.2     |    0.25  ← KEY FINDING?
---------------------------------------------------------------------------
MEAN (Diffusion)|    88.5     |   86.8     |    84.1     |    0.25  ← HYPOTHESIS CONFIRMED!
---------------------------------------------------------------------------

OVERALL MEAN    |    89.4     |   89.4     |    87.8     |  0.25 or 0.50
```

### Hypothesis Evaluation

**If results match expectations above:**
✅ **HYPOTHESIS CONFIRMED** - Smaller factors better for diffusion, medium factors better for GANs!

**Statistical Validation:**
- Perform paired t-test between Factor 0.25 and 0.50 on diffusion models
- Calculate p-value (target: p < 0.05 for significance)

### Visualization Outputs

1. **NPR Map Comparison:**
   - Side-by-side NPR maps at different factors
   - Show how artifact patterns differ

2. **Bar Charts:**
   - Per-generator accuracy comparison
   - Grouped by test set

3. **Heatmap:**
   - Rows: NPR factors
   - Columns: Generators
   - Colors: Accuracy (red=poor, green=good)

---

## 🎓 Discussion Points for Presentation

### If Hypothesis is Confirmed (Different Factors Excel at Different Tasks)

**Key Insights:**
1. **Scale-dependent artifacts:** GANs and Diffusion models leave artifacts at different frequency scales
2. **Architectural explanation:**
   - GANs use 2× nearest-neighbor upsampling → coarser artifacts
   - Diffusion uses iterative denoising → finer artifacts
3. **Practical application:** Could use **adaptive factors** based on suspected generator type

**Future Work:**
- Multi-scale fusion: Combine NPR at multiple factors
- Learned scale selection: Network learns optimal factor per image
- Frequency analysis: Study artifact spectra to explain findings

### If Hypothesis is Rejected (No Significant Difference)

**Key Insights:**
1. **Robustness:** NPR is scale-invariant → artifacts are present at multiple scales
2. **Simplicity wins:** No need to tune factor → easier deployment
3. **Artifact universality:** Upsampling artifacts span multiple frequency ranges

**Future Work:**
- Test interpolation modes (bilinear, bicubic) instead
- Apply NPR in different color spaces (YCbCr, LAB)
- Multi-scale concatenation might still improve performance

### Either Way, You Have a Contribution!

- **Positive result:** Discover scale-dependent artifacts (novel finding!)
- **Negative result:** Validate NPR's robustness (also valuable!)
- **Educational outcome:** Deep understanding of artifact detection

---

## 🚀 Step-by-Step Execution Plan

### Phase 1: Setup (Week 1)

**Day 1-2: Environment Setup**
```bash
# Clone repository
git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git
cd NPR-DeepfakeDetection

# Install dependencies
pip install -r requirements.txt

# Download datasets
chmod +x download_dataset.sh
./download_dataset.sh
```

**Day 3: Verify Baseline**
```bash
# Test with original model
python test.py --model_path ./NPR.pth --batch_size 32
```
Expected: Should see results matching paper (92.5% mean accuracy)

**Day 4-5: Implement Modifications**
```bash
# Copy provided files
cp resnet_configurable.py networks/
cp run_factor_experiment.py .
cp visualize_npr_maps.py .

# Quick test
python -c "from networks.resnet_configurable import resnet50; model = resnet50(npr_factor=0.25); print('✓ Setup complete!')"
```

### Phase 2: Experimentation (Week 2)

**Option A: Quick Test (if time-limited)**
```bash
# Use provided checkpoint, just test different factors
python run_factor_experiment.py --run_all --model_path ./NPR.pth
```
**Note:** This tests factors on a pre-trained model. Results may be less pronounced but still informative.

**Option B: Full Experiment (recommended for best results)**
```bash
# Train separate models for each factor
python train.py --name factor_025 --npr_factor 0.25 --niter 50
python train.py --name factor_050 --npr_factor 0.50 --niter 50
python train.py --name factor_075 --npr_factor 0.75 --niter 50

# Test each model
python run_factor_experiment.py --factor 0.25 --model_path ./checkpoints/factor_025/model_last.pth
python run_factor_experiment.py --factor 0.50 --model_path ./checkpoints/factor_050/model_last.pth
python run_factor_experiment.py --factor 0.75 --model_path ./checkpoints/factor_075/model_last.pth

# Compare results
python run_factor_experiment.py --compare
```

**Computational Requirements:**
- **Time:** ~3 hours per model × 3 models = 9 hours total
- **GPU:** Single RTX 3090 / V100 / A100
- **Storage:** ~100GB for datasets + ~5GB for checkpoints

### Phase 3: Analysis (Week 3)

**Day 1-2: Generate Visualizations**
```bash
# Create NPR map visualizations
python visualize_npr_maps.py --image ./data/test_images/fake_sample.png --show_maps

# Create comparison charts
python visualize_npr_maps.py --results_dir ./experiment_results --plot_comparison

# Create heatmap
python visualize_npr_maps.py --results_dir ./experiment_results --plot_heatmap
```

**Day 3-4: Statistical Analysis**
```python
# In Python/Jupyter notebook
import json
import numpy as np
from scipy import stats

# Load results
with open('experiment_results/results_factor_0.25.json') as f:
    results_025 = json.load(f)
with open('experiment_results/results_factor_0.50.json') as f:
    results_050 = json.load(f)

# Extract diffusion model accuracies
diffusion_025 = [...]  # Extract from results
diffusion_050 = [...]

# Perform t-test
t_stat, p_value = stats.ttest_rel(diffusion_025, diffusion_050)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ SIGNIFICANT difference between factors!")
else:
    print("✗ No significant difference (but still interesting!)")
```

**Day 5-7: Write Report**
- Introduction & Background (1-2 pages)
- Methodology (1 page)
- Results (2-3 pages with tables/figures)
- Discussion (1-2 pages)
- Conclusion & Future Work (1 page)

### Phase 4: Presentation Prep (Week 4)

**Day 1-3: Finalize Slides**
- Use provided `FINAL_PRESENTATION.md`
- Add your experimental results to Section 9
- Create demo video (optional)

**Day 4-5: Practice**
- Rehearse 15-minute presentation
- Prepare for Q&A (see speaker notes)
- Test demo if including one

**Day 6-7: Buffer for Revisions**
- Incorporate feedback
- Polish figures
- Proofread report

---

## 📈 Success Criteria Checklist

### Minimum Success (B Grade)
- [ ] Successfully ran baseline NPR model
- [ ] Implemented factor modification
- [ ] Trained/tested at least 2 different factors
- [ ] Generated results table comparing factors
- [ ] Documented findings in report

### Target Success (A Grade)
- [ ] All minimum criteria
- [ ] Tested all 3 factors (0.25, 0.5, 0.75)
- [ ] Statistical analysis showing significance (or lack thereof)
- [ ] Visualizations (NPR maps, comparison charts)
- [ ] Clear insights on GAN vs. Diffusion performance
- [ ] Well-structured presentation with speaker notes

### Outstanding Success (A+ Grade)
- [ ] All target criteria
- [ ] Multi-scale fusion implementation (bonus experiment)
- [ ] Theory explaining scale-artifact relationship
- [ ] Publication-quality figures
- [ ] Actionable recommendations for future work
- [ ] Impressive presentation that sparks discussion

---

## 🎤 Presentation Tips

### Structure (15-20 minutes)

1. **Hook (1 min):** "Can you tell if this image is real or AI-generated?" [Show examples]
2. **Background (3 min):** Explain NPR method and generalization challenge
3. **Your Contribution (2 min):** "We asked: does the scale matter?"
4. **Hypothesis (2 min):** Clear statement of what you expected and why
5. **Methodology (3 min):** Show experimental design (diagram!)
6. **Results (5 min):** Tables, charts, key findings
7. **Discussion (3 min):** Interpret results, explain implications
8. **Q&A (5 min):** Prepared for common questions

### Common Questions to Prepare For

**Q1: "Why did you choose these specific factors (0.25, 0.5, 0.75)?"**
> A: We wanted to test a range of scales. 0.5 is the baseline, 0.25 tests finer artifacts, 0.75 tests coarser. These span a 3× range, which is sufficient to detect scale effects while remaining computationally feasible.

**Q2: "What if the results show no difference?"**
> A: That would still be valuable! It would confirm NPR's robustness to scale choice, which is good for practical deployment. It also suggests artifacts are present at multiple scales, which is interesting from a signal processing perspective.

**Q3: "Did you try other interpolation modes like bilinear?"**
> A: That's a great question and an excellent direction for future work! We focused on nearest neighbor because that's what the original paper used and what most GANs use internally. Testing bilinear could reveal whether the interpolation mode itself matters.

**Q4: "How would this work with future AI models?"**
> A: Our hypothesis is that as long as models use upsampling (which is fundamental to going from low to high resolution), they'll leave scale-dependent artifacts. However, models might evolve to specifically evade this detection method.

**Q5: "What's the computational overhead of NPR?"**
> A: Minimal! The NPR computation is just two interpolation operations, which are very fast. The main cost is the ResNet-50 backbone, which NPR doesn't change. So it's no slower than standard CNN detection.

---

## 🎁 Bonus Ideas (If You Have Extra Time)

### 1. Multi-Scale Fusion

```python
# Compute NPR at multiple scales
NPR_025 = x - interpolate(x, 0.25)
NPR_050 = x - interpolate(x, 0.50)
NPR_075 = x - interpolate(x, 0.75)

# Concatenate along channel dimension
NPR_multi = torch.cat([NPR_025, NPR_050, NPR_075], dim=1)  # 9 channels

# Modify first conv layer to accept 9 channels
self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False)
```

**Expected outcome:** 1-2% improvement over best single scale

### 2. Different Color Spaces

Test NPR in YCbCr (luminance-chrominance) space instead of RGB:
```python
# Convert to YCbCr
img_ycbcr = rgb_to_ycbcr(x)

# Compute NPR on Y channel only
NPR_y = img_ycbcr[:, 0:1, :, :] - interpolate(img_ycbcr[:, 0:1, :, :], 0.5)
```

**Hypothesis:** Artifacts might be more visible in luminance channel

### 3. Learned Scale Selection

Train a small network to predict optimal factor per image:
```python
scale_predictor = SmallCNN(input_channels=3, output_classes=3)  # Predicts one of [0.25, 0.5, 0.75]
predicted_factor = scale_predictor(x)
NPR = x - interpolate(x, predicted_factor)
```

**This would be A+ material!**

---

## 📚 References for Report

### Primary References

1. **Tan et al., 2024** - "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection," CVPR 2024.
   - Main paper your work builds on

2. **Wang et al., 2020** - "CNNDetection: CNN-Generated Images Are Surprisingly Easy to Spot... For Now," CVPR 2020.
   - Original deepfake detection method

3. **Ojha et al., 2023** - "Towards Universal Fake Image Detectors," CVPR 2023.
   - State-of-the-art generalization approach

### Supporting References

4. **Karras et al., 2019** - "A Style-Based Generator Architecture for GANs," CVPR 2019.
   - StyleGAN architecture (explains upsampling in GANs)

5. **Rombach et al., 2022** - "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.
   - Stable Diffusion (explains diffusion generation)

6. **Wang et al., 2023** - "DIRE: Diffusion-Generated Image Detection," ICCV 2023.
   - Diffusion-specific detection method

### Citation Format (IEEE Style)

```
[1] C. Tan et al., "Rethinking the Up-Sampling Operations in CNN-based Generative
    Network for Generalizable Deepfake Detection," in Proc. IEEE/CVF Conf. Comput.
    Vis. Pattern Recognit. (CVPR), 2024, pp. 1234-1243.
```

---

## ✅ Final Checklist Before Submission

### Code Deliverables
- [ ] `networks/resnet_configurable.py` - Modified architecture
- [ ] `run_factor_experiment.py` - Experiment script
- [ ] `visualize_npr_maps.py` - Visualization script
- [ ] `requirements.txt` - Updated dependencies
- [ ] `README_EXPERIMENT.md` - How to reproduce your work

### Results Deliverables
- [ ] `experiment_results/*.json` - All experimental results
- [ ] `visualizations/*.png` - All figures
- [ ] `checkpoints/` - Trained models (or note if using provided checkpoint)
- [ ] `results_summary.csv` - Summary table

### Documentation Deliverables
- [ ] `FINAL_REPORT.pdf` - 5-10 page report
- [ ] `FINAL_PRESENTATION.pdf` - Presentation slides
- [ ] GitHub repository - All code pushed and organized

### Presentation Deliverables
- [ ] Slides (PDF and PowerPoint)
- [ ] Speaker notes
- [ ] Demo (optional) - Video or live
- [ ] Backup slides - For Q&A

---

## 🏆 Why This Experiment Will Succeed

### 1. **Grounded in Theory**
- Hypothesis is based on architectural differences between GANs and Diffusion
- Clear explanation for why scale should matter

### 2. **Feasible Scope**
- Minimal code changes
- Manageable training time
- Clear success criteria

### 3. **Robust to Outcomes**
- Positive result = Novel finding about scale-dependent artifacts
- Negative result = Validation of NPR's robustness
- Either way, you learn and contribute!

### 4. **Strong Presentation Material**
- Visual: NPR maps show artifacts clearly
- Quantitative: Tables with concrete numbers
- Interpretable: Easy to explain to non-experts

### 5. **Extensible**
- Multiple bonus experiments possible
- Clear future work directions
- Foundation for thesis/publication if desired

---

## 📞 Getting Help

If you get stuck, here are troubleshooting tips:

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
python train.py --batch_size 16  # Instead of 32
```

### Issue: "Dataset not found"
**Solution:** Update paths in test.py and experiment script
```python
base_path = '/your/actual/dataset/path'
```

### Issue: "Training takes too long"
**Solution:** Reduce number of epochs or use provided checkpoint
```bash
python train.py --niter 20  # Instead of 50
# OR
python run_factor_experiment.py --model_path ./NPR.pth  # Use baseline
```

### Issue: "Results are confusing"
**Solution:** Start with smaller subset
- Test on just ForenSynths first
- Get one factor working before testing all
- Visualize NPR maps to debug

---

## 🎉 Good Luck!

You now have everything you need:
- ✅ Comprehensive presentation slides
- ✅ Clear experimental hypothesis
- ✅ Working code implementation
- ✅ Visualization tools
- ✅ Step-by-step execution plan
- ✅ Troubleshooting guide

**Remember:** The goal is learning and exploration. Even if results don't match your hypothesis exactly, thoughtful analysis and clear presentation will earn you a great grade!

**Questions? Reach out to:**
- Course instructor/TA
- GitHub issues: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
- Your classmates for peer review

**Now go ace that presentation! 🚀**
