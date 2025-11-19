# Attention-NPR Architecture Diagram
## Detailed Visual Specifications for Presentation

---

## MAIN ARCHITECTURE DIAGRAM

### Layout: Left-to-Right Flow

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    ATTENTION-WEIGHTED MULTI-SCALE NPR                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌──────────────┐
│              │
│   Input      │
│   Image      │        STAGE 1: Multi-Scale NPR Extraction
│ (3×224×224)  │        ════════════════════════════════════
│              │
└──────┬───────┘
       │
       │    ┌────────────────────────────────────────────────┐
       │    │                                                │
       ├────┤  Scale 0.25x: Downsample→Upsample→Residual   │──→ NPR_0.25
       │    │  F.interpolate(x, 0.25) → (x - x_recon)      │   (3×224×224)
       │    │                                                │
       ├────┤  Scale 0.50x: Downsample→Upsample→Residual   │──→ NPR_0.50
       │    │  F.interpolate(x, 0.50) → (x - x_recon)      │   (3×224×224)
       │    │  [ORIGINAL NPR SCALE]                         │
       │    │                                                │
       ├────┤  Scale 0.75x: Downsample→Upsample→Residual   │──→ NPR_0.75
       │    │  F.interpolate(x, 0.75) → (x - x_recon)      │   (3×224×224)
       │    │                                                │
       └────┤  Scale 1.00x: Identity (No downsampling)      │──→ NPR_1.00
            │  NPR_1.00 = Original Image                    │   (3×224×224)
            └────────────────────────────────────────────────┘
                                 │
                                 │
                    ┌────────────┴────────────┐
                    │  Concatenate Channel-Wise│
                    │     (4 scales × 3 RGB)   │
                    └────────────┬─────────────┘
                                 │
                        (12×224×224)
                                 │
                                 ▼

       STAGE 2: Channel Attention Module (SENet-Inspired)
       ═════════════════════════════════════════════════════

                    ┌─────────────────────┐
                    │  Global Avg Pool    │
                    │  12×224×224 → 12×1×1│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Squeeze: FC Layer  │
                    │    12 → 6 (÷2)      │
                    │    Activation: ReLU │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Excite: FC Layer   │
                    │    6 → 12 (×2)      │
                    │  Activation: Sigmoid│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Reshape & Tile    │
                    │   12 → (4, 3, 1, 1) │
                    │   4 scales × 3 ch   │
                    └──────────┬──────────┘
                               │
                  ┌────────────▼────────────┐
                  │  Attention Weights      │
                  │  w1, w2, w3, w4         │
                  │  (per RGB channel)      │
                  └─────────────────────────┘
                               │
                               ▼

       STAGE 3: Weighted Fusion
       ═════════════════════════

            ┌───────────────────────────┐
            │  Element-wise Multiply:   │
            │  NPR_0.25 × w1            │
            │  NPR_0.50 × w2            │
            │  NPR_0.75 × w3            │
            │  NPR_1.00 × w4            │
            └───────────┬───────────────┘
                        │
            ┌───────────▼───────────────┐
            │  Sum across scales:       │
            │  NPR_fused = Σ(wi × NPRi) │
            │  Output: (3×224×224)      │
            └───────────┬───────────────┘
                        │
                        ▼

       STAGE 4: ResNet50 Classifier
       ══════════════════════════════

            ┌───────────────────────────┐
            │  Conv1: 3→64, 3×3, s=2    │ 112×112
            │  BatchNorm + ReLU         │
            │  MaxPool: 3×3, s=2        │ 56×56
            ├───────────────────────────┤
            │  Layer1: BasicBlock×3     │ 56×56
            │  64 → 256 channels        │
            ├───────────────────────────┤
            │  Layer2: BasicBlock×4     │ 28×28
            │  256 → 512, stride=2      │
            ├───────────────────────────┤
            │  Layer3: BasicBlock×6     │ 14×14
            │  512 → 1024, stride=2     │
            ├───────────────────────────┤
            │  Layer4: BasicBlock×3     │ 7×7
            │  1024 → 2048, stride=2    │
            ├───────────────────────────┤
            │  Global Avg Pool          │ 1×1
            │  2048 → 2048×1×1          │
            ├───────────────────────────┤
            │  Fully Connected          │
            │  2048 → 1 (logit)         │
            └───────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────────┐
            │  BCEWithLogitsLoss        │
            │  Sigmoid(logit)           │
            │  Output: P(Fake)          │
            └───────────────────────────┘
```

---

## DIAGRAM COMPONENTS FOR POWERPOINT

### Component 1: Multi-Scale NPR Extraction (Parallel Paths)

**Visual Style:** 4 parallel horizontal bars with different colors

```
Scale 0.25x: [BLUE]    ──→  NPR₀.₂₅
             ↓↑ (artifacts at high frequency)

Scale 0.50x: [GREEN]   ──→  NPR₀.₅₀  ⭐ BASELINE
             ↓↑ (original NPR scale)

Scale 0.75x: [ORANGE]  ──→  NPR₀.₇₅
             ↓↑ (artifacts at low frequency)

Scale 1.00x: [RED]     ──→  NPR₁.₀₀
             -- (identity, no downsampling)
```

**PowerPoint Instructions:**
1. Use SmartArt "Process" → "Vertical Process"
2. 4 shapes, each labeled with scale
3. Add small image icons showing downsampling operation
4. Color-code each scale (blue → green → orange → red)

---

### Component 2: Attention Module (Squeeze-and-Excitation)

**Visual Style:** Diamond/funnel shape

```
      [12-dim vector]  ← Concatenated NPR features
            │
            ▼
      ┌─────────┐
      │   GAP   │      Global Average Pooling
      └────┬────┘
           │
      ┌────▼────┐
      │FC: 12→6 │      Squeeze (compression)
      │  ReLU   │
      └────┬────┘
           │
      ┌────▼────┐
      │FC: 6→12 │      Excitation (expansion)
      │ Sigmoid │
      └────┬────┘
           │
      [w1,w2,w3,w4]    Scale weights (0-1)
```

**PowerPoint Instructions:**
1. Use shapes: Rectangle → Trapezoid (narrowing) → Trapezoid (widening)
2. Label dimensions on the side
3. Add small mathematical notation: σ(FC(ReLU(FC(GAP(x)))))

---

### Component 3: Weighted Fusion (Matrix Multiplication)

**Visual Style:** Matrix diagram

```
NPR_0.25  ×  w1  =  w1·NPR_0.25
NPR_0.50  ×  w2  =  w2·NPR_0.50    ┐
NPR_0.75  ×  w3  =  w3·NPR_0.75    ├──→  Σ  →  NPR_fused
NPR_1.00  ×  w4  =  w4·NPR_1.00    ┘
```

**PowerPoint Instructions:**
1. Use table with 4 rows
2. Add × and = symbols
3. Final summation symbol (Σ) in large font
4. Arrows showing flow to final output

---

## COMPARISON DIAGRAM: Baseline vs. Attention-NPR

```
┌──────────────────────────────────────────────────────────────┐
│                 BASELINE NPR-ResNet50                        │
└──────────────────────────────────────────────────────────────┘

Input → NPR_0.5 (fixed) → ResNet50 → Output
        (single scale)

        ❌ Cannot adapt to different generators
        ❌ Misses artifacts at other scales


┌──────────────────────────────────────────────────────────────┐
│            ATTENTION-NPR-ResNet50 (Proposed)                 │
└──────────────────────────────────────────────────────────────┘

Input → Multi-Scale NPR → Attention → Fusion → ResNet50 → Output
        (4 scales)        (learned)    (weighted)

        ✓ Adapts weights to input characteristics
        ✓ Captures artifacts across frequency spectrum
        ✓ Generalizes to unseen generators
```

---

## VISUALIZATION: Attention Weights by Generator Type

**Mock Heatmap for Presentation:**

```
Generator Type    │ w1 (0.25x) │ w2 (0.50x) │ w3 (0.75x) │ w4 (1.00x)
──────────────────┼────────────┼────────────┼────────────┼──────────
ProGAN            │    0.15    │  ⬛ 0.55   │    0.20    │    0.10
StyleGAN2         │    0.20    │  ⬛ 0.50   │    0.25    │    0.05
DALL-E 3          │  ⬛ 0.45   │    0.25    │    0.20    │    0.10
Midjourney v6     │  ⬛ 0.40   │    0.30    │    0.20    │    0.10
FLUX              │  ⬛ 0.50   │    0.20    │    0.15    │    0.15
Stable Diffusion  │    0.35    │    0.30    │  ⬛ 0.25   │    0.10
```

**Observation (Hypothesis):**
- GANs (ProGAN, StyleGAN2): Higher weight on 0.5x (original NPR scale)
- Diffusion (DALL-E, Midjourney, FLUX): Higher weight on 0.25x (finer artifacts)
- Demonstrates learned scale-specific detection strategy

**PowerPoint Instructions:**
1. Create table with conditional formatting
2. Highest weight per row: dark fill
3. Color gradient from white (low) to dark (high)

---

## TIMELINE GANTT CHART

```
Week 1 (Nov 18-24)     [████████████████████░░░░░░░░] Baseline + Implement
│                       ████████ Reproduce baseline
│                               ████████ Code multi-scale NPR
│                                       ████ Attention module
│
Week 2 (Nov 25-Dec 1)  [░░░░████████████████████████░] Training
│                           ████████████ Train Attention-NPR
│                                       ████████ Ablation studies
│                                               ████ Test seen gens
│
Week 3 (Dec 2-8)       [░░░░░░░░████████████████████] Testing & Analysis
│                               ████████ Collect 2025 data
│                                       ████████ Test unseen gens
│                                               ████████ Stats + Viz
│
Week 4 (Dec 9)         [░░░░░░░░░░░░░░░░░░░░░░░░████] Documentation
│                                               ████ Final report
│                                                   📅 DEC 9 DEADLINE
```

**Critical Milestones:**
- ✓ Nov 17, 5:00 PM: Presentation submitted ✅
- ⏳ Nov 24: Implementation complete
- ⏳ Dec 1: Training + ablation done
- ⏳ Dec 8: All experiments complete
- 📅 Dec 9, 10:00 AM: **FINAL SUBMISSION**

---

## HARDWARE REQUIREMENTS DIAGRAM

```
┌────────────────────────────────────────────────────────────┐
│                    COMPUTE RESOURCES                       │
└────────────────────────────────────────────────────────────┘

GPU: NVIDIA RTX 3090 / A100
     ┌──────────────────────────────────┐
     │  ████████████████████░░░░░░       │  24GB VRAM
     │  18GB used (Attention-NPR)       │
     │  Peak: 22GB during training      │
     └──────────────────────────────────┘

CPU: 32-core (Intel Xeon / AMD EPYC)
     ┌──────────────────────────────────┐
     │  ████████░░░░░░░░░░░░░░░░░░░░░░  │  128GB RAM
     │  24GB used (data loading)        │
     └──────────────────────────────────┘

Storage: 500GB SSD
     ┌──────────────────────────────────┐
     │  ████████████░░░░░░░░░░░░░░░░░░  │  500GB
     │  200GB: Datasets                 │
     │  20GB: Checkpoints               │
     │  10GB: Results/Logs              │
     └──────────────────────────────────┘

Training Time Estimate:
  Baseline:        18 hours  ████████████░░░░
  Attention-NPR:   24 hours  ████████████████
  Ablations (3):   72 hours  ████████████████████████████████████
  Testing:          4 hours  ███
  ─────────────────────────────────────────────
  TOTAL:          ~120 hours (5 days continuous)
```

---

## EXPECTED RESULTS VISUALIZATION

### ROC Curve Comparison (Mock)

```
     1.0 │                    ┌─── Attention-NPR (AUC=0.96)
         │                   ╱│
         │                  ╱ │
   TPR   │                 ╱  │
         │                ╱   └─── Baseline (AUC=0.92)
         │               ╱   ╱
         │              ╱   ╱
     0.5 │             ╱   ╱
         │            ╱   ╱
         │           ╱   ╱
         │          ╱   ╱
         │         ╱   ╱
     0.0 └────────┴───┴──────────────
         0.0    0.5                1.0
                    FPR
```

### Generalization Gap (Bar Chart)

```
Accuracy (%)
100 │
    │     ██████       ██████                Baseline: Δ=8%
 95 │     ██████       ██████                Attention: Δ=3%
    │     ██████       ██████  ██████
 90 │     ██████       ██████  ██████
    │     ██████       ██████  ██████
 85 │     ██████               ██████
    └─────┴─────────────┴──────┴──────
         Base-Seen   Attn-Seen  Attn-Unseen

    ⬛ Baseline (Seen):    92%
    ⬛ Attention (Seen):   95%  (+3% vs baseline)
    ⬛ Attention (Unseen): 92%  (only 3% gap)
```

---

## SUCCESS CRITERIA TABLE

| Metric                    | Baseline | Target | Status |
|---------------------------|----------|--------|--------|
| Accuracy (ForenSynths)    | 91.7%    | ≥93.7% | ⏳ TBD |
| Generalization Gap        | ~8%      | <5%    | ⏳ TBD |
| Attention Interpretability| N/A      | ✓      | ⏳ TBD |
| Inference Time Overhead   | 1.0×     | <1.3×  | ⏳ TBD |
| Memory Overhead           | 12GB     | <24GB  | ⏳ TBD |

**Statistical Significance:** Paired t-test, p < 0.05

---

## LIMITATIONS SUMMARY (Slide Format)

```
┌────────────────────────────────────────┐
│         KNOWN LIMITATIONS              │
└────────────────────────────────────────┘

1. Computational
   • Limited to 4 scales (memory constraint)
   • Cannot exhaustively search scale space

2. Data
   • 2025 generators: limited sample availability
   • Midjourney v6: no ground truth parameters

3. Architecture
   • Simple channel attention (SENet-style)
   • Fixed ResNet50 backbone

4. Evaluation
   • Post-processing (JPEG, filtering) not fully tested
   • Real-world deployment not validated
```

---

## SLIDE-BY-SLIDE CONTENT RECOMMENDATIONS

**Slide 1: Title**
- Title: "Attention-Weighted Multi-Scale NPR for Deepfake Detection"
- Subtitle: "Generalizable Detection for 2025 Generators"
- Names, Date, Course

**Slide 2: Motivation**
- Image: Deepfake examples (Midjourney v6, FLUX, DALL-E 3)
- Stat: "91.7% baseline accuracy, but can we do better?"
- Question: "How do we generalize to unseen generators?"

**Slide 3: Background - NPR**
- Diagram: Upsampling operation creating artifacts
- Formula: NPR = x - interpolate(x, scale_factor)
- Current limitation: Fixed 0.5x scale

**Slide 4: Problem Statement**
- "Different generators → Different artifact scales"
- Visual: Frequency analysis showing artifacts at multiple scales
- Gap: "No adaptive scale selection mechanism"

**Slide 5: Our Approach**
- Full architecture diagram (main diagram from above)
- 3 key innovations:
  1. Multi-scale NPR (4 scales)
  2. Learned attention weights
  3. Adaptive fusion

**Slide 6: Hypotheses**
- H2: Scale-adaptive attention improves accuracy (≥2%)
- H3: Better generalization to 2025 generators (gap <5%)
- Visual: Expected attention weight heatmap

**Slide 7: Training Data**
- ForenSynths: 240K images, 4 classes, 8 generators
- Augmentation strategy
- Test sets: Seen (GANGen) + Unseen (FLUX, Midjourney v6, DALL-E 3)

**Slide 8: Timeline**
- Gantt chart (from above)
- Milestones clearly marked
- Dec 9 deadline highlighted

**Slide 9: Hardware & Resources**
- Hardware diagram (from above)
- 120-hour training estimate
- Confirmed availability ✓

**Slide 10: Success Criteria**
- Table: Baseline vs. Target metrics
- Statistical testing approach
- Minimal acceptable outcome defined

**Slide 11: Expected Results**
- Mock ROC curves
- Generalization gap bar chart
- Attention weight interpretation

**Slide 12: Ablation Studies**
- Fixed equal weights (0.25 each)
- Fixed learned weights (non-adaptive)
- Full adaptive attention
- Comparison table

**Slide 13: Limitations & Risks**
- Summary table (from above)
- Mitigation strategies
- Fallback plans

**Slide 14: Deliverables**
- Code repository (with documentation)
- Trained models (checkpoints)
- Final report (8-10 pages)
- Presentation + supplementary materials

**Slide 15: Questions & Discussion**
- "Which scale do you expect to be most important for FLUX?"
- "Should we explore spatial attention as well?"
- Contact info

---

*This architecture diagram guide provides all visual elements needed for a professional, academic presentation. Each diagram can be created in PowerPoint using built-in SmartArt, shapes, and tables.*
