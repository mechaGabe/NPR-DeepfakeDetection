# Final Project: NPR Deepfake Detection - Scale Factor Experiment

## 🎓 Quick Start Guide for Final Presentation

This project explores **whether different NPR interpolation factors better detect different types of AI-generated images (GANs vs. Diffusion models)**.

---

## 📁 New Files Added for This Project

```
NPR-DeepfakeDetection/
├── FINAL_PRESENTATION.md                  ⭐ Your presentation slides (convert to PowerPoint)
├── EXPERIMENTAL_EXTENSION_GUIDE.md        ⭐ Detailed methodology and execution plan
├── README_FINAL_PROJECT.md                ⭐ This file (quick start)
│
├── networks/
│   ├── resnet.py                          📄 Original implementation
│   └── resnet_configurable.py             ⭐ NEW: Configurable NPR factor
│
├── run_factor_experiment.py               ⭐ NEW: Automated experiment script
└── visualize_npr_maps.py                  ⭐ NEW: Create figures for presentation
```

---

## 🚀 Quick Start (Minimum Viable Experiment)

If you're short on time, follow these steps:

### Step 1: Verify Setup (5 minutes)

```bash
# Check if everything works
python -c "from networks.resnet_configurable import resnet50; print('✓ Setup OK')"

# Check if you have the baseline model
ls -lh NPR.pth  # Should be ~17MB
```

### Step 2: Test Different Factors (30 minutes)

```bash
# Quick test using the provided baseline model
python run_factor_experiment.py --run_all --model_path ./NPR.pth
```

**Note:** This won't give you the full experimental results (since the model wasn't trained with different factors), but it will:
- Generate comparison data
- Create result files you can analyze
- Give you something to present!

### Step 3: Generate Visualizations (10 minutes)

```bash
# Create comparison charts
python visualize_npr_maps.py --results_dir ./experiment_results --plot_comparison

# Create heatmap
python visualize_npr_maps.py --results_dir ./experiment_results --plot_heatmap

# All figures will be in ./visualizations/
```

### Step 4: Prepare Presentation (30 minutes)

```bash
# Open the presentation file
cat FINAL_PRESENTATION.md

# Convert to slides (use Pandoc, Google Docs, or copy to PowerPoint)
# Add your experimental results to Section 9
```

---

## 📊 Full Experiment (If You Have Time)

### Training Models with Different Factors

This gives you the complete experimental results:

```bash
# Download datasets first (if not already done)
./download_dataset.sh

# Train three models (2-3 hours each)
python train.py --name factor_025 --npr_factor 0.25 --batch_size 32 --niter 50 \
    --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse

python train.py --name factor_050 --npr_factor 0.50 --batch_size 32 --niter 50 \
    --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse

python train.py --name factor_075 --npr_factor 0.75 --batch_size 32 --niter 50 \
    --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse

# Test each model
python run_factor_experiment.py --factor 0.25 \
    --model_path ./checkpoints/factor_025/model_epoch_last.pth

python run_factor_experiment.py --factor 0.50 \
    --model_path ./checkpoints/factor_050/model_epoch_last.pth

python run_factor_experiment.py --factor 0.75 \
    --model_path ./checkpoints/factor_075/model_epoch_last.pth

# Compare results
python run_factor_experiment.py --compare
```

---

## 🎯 Hypothesis & Expected Results

### Research Question

**"Does the NPR interpolation factor affect detection accuracy differently for GAN-based vs. Diffusion-based generators?"**

### Hypothesis

- **Smaller factors (0.25)** capture finer artifacts → better for Diffusion models
- **Medium factors (0.5)** balanced performance → good baseline (original paper)
- **Larger factors (0.75)** capture coarser artifacts → better for GANs

### Expected Outcome Table

| Generator Type | Best Factor | Reasoning |
|----------------|-------------|-----------|
| GANs (ProGAN, StyleGAN, etc.) | 0.5 - 0.75 | GANs use 2× nearest neighbor upsampling → coarser artifacts |
| Diffusion (Stable Diffusion, DALL-E) | 0.25 - 0.5 | Iterative denoising → finer, multi-scale artifacts |

**Even if your results differ, you have a contribution!**
- Matching hypothesis = Novel finding about scale-dependent artifacts
- Contradicting hypothesis = NPR is robust to scale (also valuable!)

---

## 📈 Presentation Structure (Follow the Provided Slides)

Your `FINAL_PRESENTATION.md` file contains:

1. **Project Topic** - What you're investigating
2. **Background** - How NPR works, why generalization is hard
3. **Hypothesis** - Your experimental prediction
4. **Experimental Setup** - How you tested it
5. **Success Criteria** - What counts as success
6. **Deliverables** - What you're submitting
7. **Expected Results** - What you found (fill in your data!)

**Tips:**
- Spend most time on sections 3-5 (your contribution!)
- Include at least 2-3 figures (NPR maps, comparison chart, heatmap)
- Prepare for Q&A using speaker notes

---

## 🔬 Understanding the NPR Method (For Your Presentation)

### What is NPR?

Neural Perceptual Residual = **Original Image - Reconstructed Image**

```python
# 1. Take image (e.g., 224×224)
original = load_image()

# 2. Downsample (e.g., 224×224 → 112×112)
downsampled = downsample(original, factor=0.5)

# 3. Upsample back (112×112 → 224×224)
reconstructed = upsample(downsampled, factor=2.0)

# 4. Compute difference
NPR = original - reconstructed
```

### Why Does This Detect Fakes?

- **Real photos:** Natural images are information-rich → small residuals
- **AI-generated:** Generated using upsampling → large residuals (same artifacts!)

### Analogy for Non-Technical Audience

> "It's like photocopying a document. Each time you copy, you lose a bit of detail. AI-generated images have 'copy artifacts' from the upsampling process, which NPR reveals."

---

## 📊 Understanding Your Results

### What to Look For

1. **Overall Performance:**
   - Does any factor significantly outperform others?
   - Calculate mean accuracy across all generators

2. **GAN vs. Diffusion Split:**
   - Compare mean accuracy on GAN datasets vs. Diffusion datasets
   - Is there a crossover effect? (Factor X best for GANs, Factor Y best for Diffusion)

3. **Per-Generator Patterns:**
   - Which generators are hardest to detect?
   - Do they share characteristics (e.g., all diffusion models)?

4. **Statistical Significance:**
   - Calculate standard deviation
   - If differences are < 2%, might not be significant
   - If differences are > 5%, strong signal!

### Example Interpretation

**Scenario 1: Clear Difference**
```
GAN Mean Accuracy:        Factor 0.5 = 92%, Factor 0.25 = 88%
Diffusion Mean Accuracy:  Factor 0.5 = 85%, Factor 0.25 = 89%
```
**Interpretation:** ✅ Hypothesis confirmed! Diffusion benefits from smaller factors.

**Scenario 2: No Clear Difference**
```
GAN Mean Accuracy:        Factor 0.5 = 92%, Factor 0.25 = 91%
Diffusion Mean Accuracy:  Factor 0.5 = 86%, Factor 0.25 = 85%
```
**Interpretation:** ✅ Still valuable! NPR is robust to factor choice.

---

## 🎨 Creating Figures for Presentation

### Figure 1: NPR Maps Comparison

```bash
# Pick a fake image (e.g., from test set)
python visualize_npr_maps.py --image ./data/test_images/fake_sample.png --show_maps
```

**Use in presentation:** Show audience what NPR "sees" at different scales

### Figure 2: Accuracy Comparison Bar Chart

```bash
python visualize_npr_maps.py --results_dir ./experiment_results --plot_comparison
```

**Use in presentation:** Main results figure (put in Section 9)

### Figure 3: Performance Heatmap

```bash
python visualize_npr_maps.py --results_dir ./experiment_results --plot_heatmap
```

**Use in presentation:** Shows GAN vs. Diffusion patterns clearly

---

## 🎤 Presentation Day Checklist

### Before Class

- [ ] Test presentation on your laptop (slides + demo)
- [ ] Print backup slides (in case of technical issues)
- [ ] Rehearse timing (15-20 minutes including Q&A)
- [ ] Prepare 2-3 questions to ask other teams
- [ ] Charge laptop fully

### What to Bring

- [ ] Laptop with presentation loaded
- [ ] USB drive with backup files
- [ ] Printed speaker notes
- [ ] Water bottle (you'll be talking a lot!)

### During Presentation

**Dos:**
- ✅ Make eye contact with audience
- ✅ Explain figures clearly ("As you can see in this chart...")
- ✅ Speak slowly and clearly
- ✅ Show enthusiasm for your work!
- ✅ Admit limitations honestly ("We didn't have time to test X, but it would be interesting future work")

**Don'ts:**
- ❌ Read directly from slides
- ❌ Apologize excessively ("Sorry, this slide is bad...")
- ❌ Go over time limit
- ❌ Skip Q&A (it's 5 points!)

---

## 🏆 Grading Rubric Alignment

### Presentation (5 points)

| Criterion | How to Earn Points |
|-----------|-------------------|
| Clarity | Use the provided slides, explain NPR clearly |
| Visuals | Include 3+ figures (NPR maps, charts, heatmap) |
| Hypothesis | Clearly state what you expected and why |
| Discussion | Show in `FINAL_PRESENTATION.md`, invite feedback |

**To maximize score:** Ask a specific question at the end (e.g., "Should we test bilinear interpolation next?")

### Project (20 points)

| Criterion | How to Earn Points |
|-----------|-------------------|
| Exploration (5 pts) | Test multiple factors, analyze differences |
| Implementation (5 pts) | Working code, reproducible results |
| Analysis (5 pts) | Interpret results, statistical tests |
| Documentation (5 pts) | Report, code comments, README |

**To maximize score:** Submit everything in the checklist below!

---

## 📦 Submission Checklist

### On Canvas

**By November 17, 5:00 PM (Presentation):**
- [ ] `FINAL_PRESENTATION.pdf` (convert from .md file)

**By December 9, 10:00 AM (Final Project):**

**Code (as .zip or GitHub link):**
- [ ] `networks/resnet_configurable.py`
- [ ] `run_factor_experiment.py`
- [ ] `visualize_npr_maps.py`
- [ ] `README_FINAL_PROJECT.md` (this file)

**Results:**
- [ ] `experiment_results/*.json`
- [ ] `visualizations/*.png` (all figures)

**Documentation:**
- [ ] `FINAL_REPORT.pdf` (5-10 pages)
  - Introduction
  - Background
  - Methodology
  - Results (with tables/figures)
  - Discussion
  - Conclusion
  - References

---

## 🆘 Troubleshooting

### "I can't download the datasets!"

**Solution 1:** Use smaller test sets
```bash
# Create dummy data for testing code
mkdir -p ./datasets/test_small/fake
mkdir -p ./datasets/test_small/real
# Copy a few images there
```

**Solution 2:** Use only ForenSynths (Table 1 from paper)
- Smallest dataset, still shows generalization

### "Training is too slow!"

**Solution 1:** Reduce epochs
```bash
python train.py --niter 20  # Instead of 50
```

**Solution 2:** Use smaller batch size
```bash
python train.py --batch_size 16  # Instead of 32
```

**Solution 3:** Skip training, use baseline model
```bash
# Test factors on pre-trained model
python run_factor_experiment.py --run_all --model_path ./NPR.pth
```

### "My results don't match the hypothesis!"

**That's okay!** Science is about discovering truth, not confirming what you expected.

**What to do:**
1. Double-check your code for bugs
2. Verify datasets are correct
3. If everything is right, embrace the negative result!
4. In your report, discuss why results differ from expectations
5. Propose alternative hypotheses

**Example statement:**
> "Contrary to our hypothesis, we found no significant difference between factors. This suggests NPR captures artifacts at multiple scales simultaneously, which actually makes it more robust for practical deployment."

### "I'm running out of time!"

**Minimum viable project (can still get B+/A-):**

**Day 1 (2 hours):**
- Run baseline test: `python test.py --model_path ./NPR.pth`
- Test different factors: `python run_factor_experiment.py --run_all --model_path ./NPR.pth`

**Day 2 (3 hours):**
- Generate visualizations: `python visualize_npr_maps.py --plot_comparison --plot_heatmap`
- Fill in results in presentation slides (Section 9)

**Day 3 (4 hours):**
- Write 5-page report (adapt from EXPERIMENTAL_EXTENSION_GUIDE.md)
- Rehearse presentation

**Day 4 (1 hour):**
- Convert slides to PowerPoint/PDF
- Submit!

---

## 🎓 Learning Objectives

By completing this project, you will:

✅ **Understand** how generative models (GANs & Diffusion) create images
✅ **Implement** a modification to a state-of-the-art detection method
✅ **Conduct** a controlled scientific experiment
✅ **Analyze** results statistically and visually
✅ **Communicate** technical findings to an audience
✅ **Contribute** to the field of deepfake detection

Even if results are unexpected, you've gained valuable skills!

---

## 📚 Additional Resources

### Understanding NPR
- Read Section 3.1 of the paper (2312.10461v2.pdf)
- Look at Figure 2 in the paper (shows NPR computation)

### Understanding GANs
- Original GAN paper: Goodfellow et al., 2014
- StyleGAN: Karras et al., 2019
- Focus on the "upsampling" operations in generator

### Understanding Diffusion
- DDPM paper: Ho et al., 2020
- Stable Diffusion: Rombach et al., 2022
- Focus on the "denoising" process

### Presentation Skills
- Search YouTube: "how to give academic presentation"
- Practice with roommates/friends
- Record yourself and watch back

---

## 💡 Pro Tips

### For A+ Grade

1. **Go Beyond Minimum:**
   - Implement multi-scale fusion (see EXPERIMENTAL_EXTENSION_GUIDE.md)
   - Test on additional datasets
   - Create publication-quality figures

2. **Show Deep Understanding:**
   - Explain WHY factors matter (architectural reasons)
   - Connect to signal processing theory (frequency domain)
   - Discuss limitations honestly

3. **Engage Audience:**
   - Start with compelling demo (real vs. fake image)
   - Ask thought-provoking questions
   - Provide feedback to other teams

4. **Polish Everything:**
   - Proofread report multiple times
   - Make figures consistent (same colors, fonts)
   - Practice presentation until smooth

---

## 🎉 Final Words

**You've got this!** This project is:
- ✅ **Feasible:** Minimal code changes, clear methodology
- ✅ **Interesting:** Tests a novel hypothesis about artifact scales
- ✅ **Valuable:** Either outcome contributes to knowledge
- ✅ **Well-supported:** Comprehensive guides and code provided

**Remember:**
- Perfect results are not required
- Clear thinking and communication matter most
- Learning is the primary goal
- Have fun exploring!

---

## 📞 Questions?

If you get stuck:
1. Read `EXPERIMENTAL_EXTENSION_GUIDE.md` (comprehensive troubleshooting)
2. Check GitHub issues: https://github.com/chuangchuangtan/NPR-DeepfakeDetection
3. Ask your instructor/TA
4. Google the error message
5. Ask classmates (collaboration on understanding is okay!)

---

## 🚀 Ready to Start?

```bash
# Verify everything is set up
python -c "
from networks.resnet_configurable import resnet50
import torch
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ Configurable ResNet ready!')
"

# Run your first experiment
python run_factor_experiment.py --factor 0.5 --model_path ./NPR.pth
```

**Good luck! You're going to do great! 🌟**

---

**Last updated:** November 2024
**For questions about this project setup, contact your instructor.**
