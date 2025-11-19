# Presentation Script & Talking Points
## 15-Minute Presentation Guide

**Target Duration:** 15 minutes (12 min presentation + 3 min Q&A)
**Pace:** ~1 minute per slide for 12 content slides

---

## OPENING (1 minute)

**[Slide 1: Title]**

> "Good evening everyone. Today we're presenting our project on enhancing deepfake detection using attention-weighted multi-scale Natural Pattern Region analysis. I'm [Name], and I'll be walking you through our proposed approach for building a more generalizable deepfake detector that can handle emerging 2025 generators like FLUX and Midjourney v6."

**Key Points:**
- Professional opening
- State topic clearly
- Mention timeliness (2025 generators)

---

## MOTIVATION (1 minute)

**[Slide 2: Motivation]**

> "The problem we're addressing is critical: new image generators are being released constantly. Just in 2024, we saw FLUX, Midjourney v6, and DALL-E 3 reach unprecedented quality. Current deepfake detectors, while effective on training data, often fail to generalize to these unseen generators. The baseline NPR method achieves 91.7% accuracy on known generators, but we believe we can do better—especially for out-of-distribution cases."

**Key Points:**
- **Emphasize real-world relevance**: New generators = new challenges
- **Set up the gap**: Current methods struggle with generalization
- **Tease solution**: "We can do better"

**Optional Detail** (if time permits):
- "For context, these models generate images indistinguishable to human eyes, making automated detection essential."

---

## BACKGROUND: NPR CONCEPT (2 minutes)

**[Slide 3: Background - NPR]**

> "Let me briefly explain the foundation we're building on: Natural Pattern Region detection. The key insight from the CVPR 2024 paper is that generative models—both GANs and diffusion models—leave subtle artifacts during upsampling operations."

**[Point to diagram]**

> "NPR is computed as a simple residual: take the original image, downsample it, upsample it back, and subtract from the original. What remains are the upsampling artifacts—patterns that real cameras don't produce but generators do. Currently, this uses a fixed 0.5x downsampling scale, which works well but has limitations."

**Key Points:**
- **Visual explanation**: Use diagram to show downsampling → upsampling → residual
- **Intuition**: "Artifacts that generators leave behind"
- **Baseline performance**: 91.7% accuracy
- **Limitation setup**: "Fixed scale"

**Mathematical Detail** (if asked):
- NPR = x - F.interpolate(F.interpolate(x, 0.5), 2.0)
- Nearest neighbor interpolation preserves artifacts

---

## PROBLEM STATEMENT (1 minute)

**[Slide 4: Problem]**

> "The core problem is that different generators produce artifacts at different scales. ProGAN uses progressive generation from 4×4 to 1024×1024, leaving artifacts at multiple resolutions. Diffusion models have entirely different noise schedules. A fixed 0.5x scale might capture artifacts from some generators but miss others entirely."

**[Gesture to frequency diagram if present]**

> "What we need is an adaptive mechanism that can identify which scales contain the most discriminative information for a given input."

**Key Points:**
- **Generator diversity**: Different architectures → different artifact patterns
- **Fixed scale limitation**: One size doesn't fit all
- **Solution preview**: "Adaptive mechanism"

---

## OUR APPROACH (2 minutes)

**[Slide 5: Architecture]**

> "This brings us to our proposed solution: Attention-weighted Multi-Scale NPR. The architecture has three key innovations."

**[Point to diagram sections as you explain]**

> "First, instead of one NPR scale, we extract four in parallel: 0.25x, 0.5x—the original—0.75x, and 1.0x identity. This captures artifacts across different frequency bands.

> Second, we add a channel attention module—inspired by Squeeze-and-Excitation Networks—that learns to assign importance weights to each scale. It compresses the concatenated NPR features through global pooling and two fully connected layers, outputting four weights between 0 and 1.

> Third, we fuse the NPR scales using these learned weights before feeding into the ResNet50 classifier. The key is that these weights are *learned* and *input-specific*—the network decides which scales matter most for each image."

**Key Points:**
- **Multi-scale**: 4 NPR representations
- **Attention**: Learned, input-specific weights
- **Fusion**: Weighted combination before classification
- **End-to-end**: Trained jointly

**Technical Detail** (if asked):
- "The attention module has minimal overhead: only ~10% parameter increase"
- "Uses Sigmoid activation to keep weights in [0,1] range"

---

## HYPOTHESES (1.5 minutes)

**[Slide 6: Hypotheses]**

> "We have two testable hypotheses. H2: An attention mechanism can learn to automatically weigh NPR scales based on input characteristics, improving detection accuracy over fixed-scale approaches. Specifically, we expect at least a 2% improvement on the ForenSynths test set, from 91.7% to 93.7% or higher.

> H3: Attention-weighted multi-scale NPR will generalize better to unseen 2025 generators. We expect the generalization gap—the drop from seen to unseen generators—to be less than 5%, compared to ~8% for the baseline."

**[Point to attention heatmap visualization]**

> "We also expect the attention weights to be interpretable. For example, diffusion models might activate the 0.25x scale more strongly, while GANs might favor the original 0.5x scale. This would confirm the network is learning meaningful scale-specific strategies."

**Key Points:**
- **H2**: Accuracy improvement (quantitative)
- **H3**: Generalization gap reduction (quantitative)
- **Interpretability**: Attention weights reveal learned strategies (qualitative)
- **Testable**: Clear metrics, statistical testing

---

## EXPERIMENTAL SETUP: DATA (1.5 minutes)

**[Slide 7: Training Data]**

> "For training, we'll use the ForenSynths dataset: 240,000 images across four object categories, generated by eight different models including ProGAN, StyleGAN2, and various diffusion models. We'll split this 80-20 for training and validation.

> For testing, we have two categories. First, seen generators: additional GAN and diffusion models from GANGen-Detection and DiffusionForensics benchmarks. These test in-distribution generalization.

> Second, and more importantly, unseen 2025 generators: FLUX, Midjourney v6, and DALL-E 3. We'll collect approximately 5,000 images per generator—FLUX from the open-source model, Midjourney v6 from public datasets, and DALL-E 3 from existing benchmarks. This tests true out-of-distribution generalization, which is the core of H3."

**Key Points:**
- **Training**: ForenSynths (240K images, diverse)
- **Test split**: Seen vs. Unseen generators
- **Unseen = 2025**: FLUX, Midjourney v6, DALL-E 3
- **Sample size**: 5K per generator (sufficient for statistics)

**If asked about data collection:**
- FLUX: Official Hugging Face model
- Midjourney: Public Discord archives
- DALL-E 3: UniversalFakeDetect dataset

---

## TIMELINE (1 minute)

**[Slide 8: Timeline]**

> "Here's our four-week timeline. Week 1: We'll reproduce the baseline to confirm we can match published results, then implement the multi-scale NPR extraction and attention module. Week 2: Training the attention model on ForenSynths, plus ablation studies comparing fixed weights to adaptive attention. Week 3: We collect the 2025 generator data and run comprehensive testing on both seen and unseen generators. Week 4: Statistical analysis, writing the final report, and preparing deliverables."

**[Point to milestones]**

> "The critical milestone is completing all experiments by December 8th to leave a full day for documentation before the December 9th submission deadline."

**Key Points:**
- **Week 1**: Implementation (baseline + new modules)
- **Week 2**: Training + ablation
- **Week 3**: Testing (focus on unseen generators)
- **Week 4**: Documentation
- **Deadline**: Dec 9, 10:00 AM

---

## HARDWARE & RESOURCES (1 minute)

**[Slide 9: Hardware]**

> "We have access to an NVIDIA RTX 3090 with 24GB VRAM, which is sufficient for our needs. The baseline model uses about 12GB; our attention model will use approximately 18GB due to the four parallel NPR scales. Peak memory during training should stay under 22GB, well within our capacity.

> Total training time is estimated at 120 hours—that's five days continuous, but spread across two weeks in practice. The majority is ablation studies testing different configurations. We've confirmed resource availability with the lab, so compute is not a risk factor."

**Key Points:**
- **Hardware**: RTX 3090 (24GB) ✓ Available
- **Memory**: 18GB typical, 22GB peak (safe)
- **Time**: 120 hours total (~5 days)
- **Risk**: Low—resources confirmed

**If asked about alternatives:**
- "If we encounter memory issues, we can reduce batch size from 32 to 24 with minimal impact on convergence."

---

## SUCCESS CRITERIA (1 minute)

**[Slide 10: Success Criteria]**

> "Our success criteria are clearly defined. For H2, we need at least 2% accuracy improvement on ForenSynths—from 91.7% baseline to 93.7% or higher. We'll use a paired t-test with p<0.05 to confirm statistical significance.

> For H3, the generalization gap must be under 5%. That means if we achieve 95% on seen generators, we should maintain at least 90% on FLUX, Midjourney v6, and DALL-E 3.

> We also have secondary criteria: the attention weights should be interpretable—showing different patterns for different generator types—and inference overhead should be under 30% compared to baseline."

**Key Points:**
- **H2 metric**: ≥2% accuracy gain (93.7% target)
- **H3 metric**: Generalization gap <5%
- **Statistical rigor**: Paired t-test, p<0.05
- **Secondary**: Interpretability, efficiency

---

## EXPECTED RESULTS (1 minute)

**[Slide 11: Expected Results]**

> "Based on related work in multi-scale detection and attention mechanisms, we anticipate several outcomes. First, the ROC curve comparison should show improved AUC for the attention model, particularly at low false positive rates—critical for real-world deployment.

> Second, the attention weights should cluster by generator type. We expect GANs like ProGAN to activate the 0.5x scale most strongly, since that's where progressive generation artifacts appear. Diffusion models should favor 0.25x or even finer scales where noise residuals are visible.

> Third, the generalization gap should narrow significantly. Even if unseen generators aren't perfect, the multi-scale representation should provide more robust features than any single scale."

**Key Points:**
- **ROC improvement**: Higher AUC, especially at low FPR
- **Attention patterns**: Generator-specific (GANs ≠ Diffusion)
- **Generalization**: Narrower gap on unseen data
- **Hypothesis confirmation**: Quantitative + qualitative evidence

---

## ABLATION STUDIES (30 seconds)

**[Slide 12: Ablations]**

> "To isolate the contribution of attention, we'll run ablations. First, fixed equal weights—just averaging all four scales without learning. Second, fixed learned weights—optimizing one global set of weights for all inputs. Third, our full adaptive attention that computes input-specific weights. This will confirm that adaptivity is the key factor, not just having multiple scales."

**Key Points:**
- **Ablation 1**: Equal weights (baseline multi-scale)
- **Ablation 2**: Fixed learned weights (non-adaptive)
- **Ablation 3**: Full model (adaptive)
- **Purpose**: Isolate attention contribution

---

## LIMITATIONS (1 minute)

**[Slide 13: Limitations]**

> "We want to be transparent about limitations. First, computational constraints limit us to four scales—we can't exhaustively search all possible scale combinations. Second, the 2025 generator datasets will be smaller than ideal, especially Midjourney v6 where we're limited to publicly available images. Third, our attention mechanism is relatively simple channel attention; we're not exploring spatial attention or self-attention due to time constraints.

> Finally, we're not testing post-processing robustness—JPEG compression, filtering, et cetera—though these are important for real-world deployment. We acknowledge these as areas for future work."

**Key Points:**
- **Computational**: Limited to 4 scales
- **Data**: 2025 generators (smaller samples)
- **Architecture**: Simple attention (no spatial/self-attention)
- **Scope**: Not testing post-processing robustness
- **Framing**: "Future work opportunities" (positive spin)

---

## DELIVERABLES (30 seconds)

**[Slide 14: Deliverables]**

> "Our final deliverables will include: fully documented source code on GitHub, trained model checkpoints for reproducibility, a comprehensive results package with metrics and visualizations, and an 8-10 page final report in IEEE format. We'll also provide a one-click reproduction script so anyone can validate our results."

**Key Points:**
- **Code**: GitHub repo + documentation
- **Models**: Checkpoints (best + baseline)
- **Results**: Metrics, visualizations, statistics
- **Report**: 8-10 pages, IEEE format
- **Reproducibility**: Scripts + instructions

---

## CLOSING & QUESTIONS (30 seconds + Q&A)

**[Slide 15: Questions]**

> "To summarize: we're proposing attention-weighted multi-scale NPR to improve deepfake detection accuracy and generalization to 2025 generators. Our hypotheses are testable, our experimental design is rigorous, and we have confirmed computational resources. We're excited to explore whether learned attention can outperform fixed-scale approaches.

> We'd love your feedback, especially on these points: [read slide]—Which scale do you expect to be most important for FLUX? Should we explore spatial attention as well? Any other suggestions?"

**Key Points:**
- **Recap**: Multi-scale + attention for generalization
- **Invite feedback**: Specific questions to spark discussion
- **Professional close**: Thank the audience

---

## Q&A PREPARATION

### Anticipated Questions & Responses

**Q: Why not use more scales, like 8 or 16?**
> "Great question. Memory is the main constraint—each scale adds 3 channels, so 8 scales would be 24 channels, likely exceeding our 24GB VRAM. We chose 4 scales to balance coverage and feasibility. That said, if we see promising results, future work could explore finer-grained scales on larger GPUs."

**Q: Have you considered frequency-domain analysis instead of spatial NPR?**
> "Yes, there's prior work like FreDect that uses frequency analysis. NPR is complementary—it specifically targets upsampling artifacts in the spatial domain. Combining NPR with frequency features could be powerful, but we're focusing on demonstrating attention's value within the NPR framework first."

**Q: How will you ensure fair comparison if you can't perfectly reproduce the baseline?**
> "We'll use the same training data, same hyperparameters, and same ResNet50 architecture. If our reproduced baseline is within 1-2% of published results, we'll consider it valid. We'll also report both our baseline and the attention model trained under identical conditions, so the comparison is apples-to-apples."

**Q: What if the attention weights are uniform (all ~0.25)?**
> "That would actually be an interesting negative result—it would suggest that simple averaging is optimal, and adaptivity doesn't help. We'd still gain interpretability (confirming no scale dominates), and it would inform future research. However, based on generator diversity, we expect non-uniform weights."

**Q: Why not use Vision Transformers (ViT) instead of ResNet?**
> "ViTs have powerful self-attention, but they require much larger datasets and compute. ResNet50 is proven for this task, and we want to isolate the contribution of multi-scale NPR attention. Adding ViT would confound variables. Future work could certainly explore ViT backbones."

**Q: How will you handle class imbalance in test sets?**
> "Good point. We'll report both accuracy and Average Precision (AP), which is robust to imbalance. We'll also ensure test sets have balanced real/fake splits where possible. For generators with limited samples, we'll at least match real image counts."

**Q: What's your plan if you can't collect enough FLUX/Midjourney data?**
> "We have fallbacks. For FLUX, we can generate as many images as needed using the open model. For Midjourney, we'll supplement with Stable Diffusion variants if public data is insufficient. We'll be transparent about sample sizes and their potential impact on statistical power."

**Q: Have you considered adversarial robustness?**
> "Not in this project's scope, but it's critical for deployment. Adversarial attacks that specifically target NPR would be fascinating follow-up work. For now, we're focused on the base generalization problem."

---

## DISCUSSION PROMPTS (To Encourage Engagement)

If the audience is quiet, prompt with:

> "We're particularly interested in your thoughts on scale selection. Intuitively, which scale do you think would be most discriminative for diffusion models versus GANs? 0.25x, 0.5x, or 0.75x?"

> "Another open question: should the attention module have access to image-level metadata like resolution or suspected generator type? Or is that cheating?"

> "We're debating whether to visualize attention weights as heatmaps overlaid on images, or as bar charts per generator. Which would you find more interpretable?"

---

## TONE & DELIVERY TIPS

**Professional Academic Tone:**
- Avoid: "This is super cool!" → Use: "This approach offers several advantages."
- Avoid: "We think it'll work." → Use: "We hypothesize that..."
- Avoid: "Deepfakes are scary." → Use: "Deepfake detection is critical for media authenticity."

**Confidence Calibration:**
- Be confident in methodology: "We designed a rigorous experimental protocol."
- Be humble about results: "We expect improvement, but will report honestly."
- Acknowledge limitations upfront: Shows maturity and scientific rigor.

**Engagement Strategies:**
- Make eye contact (if in-person)
- Pause after key points (let them sink in)
- Use gestures to point at diagrams (direct attention)
- Vary pace (slow down for complex concepts)

**Time Management:**
- Practice to hit 12-13 minutes (leaves buffer)
- Have "cut" slides if running long (ablations, limitations can be condensed)
- Have "expansion" points if running short (more technical detail on attention module)

---

## POST-PRESENTATION FOLLOW-UP

**After Q&A, if someone suggests an idea:**
> "That's an excellent suggestion. We'll definitely consider [idea] in our analysis. Would you be willing to chat more after the session? We'd love to incorporate feedback into the final project."

**If criticism is raised:**
> "Thank you for that critique. You're right that [acknowledge valid point]. We'll address this in our limitations section and discuss potential mitigations. Do you have suggestions on how we might [address issue] within our timeline?"

**Close positively:**
> "Thank you all for the insightful questions and feedback. We're looking forward to sharing results in December. If anyone wants to discuss further, we'll be around after class."

---

*This script provides a complete 15-minute presentation flow with natural transitions, technical depth where appropriate, and preparedness for Q&A. Adjust timing based on practice runs.*
