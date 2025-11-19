# Presentation Materials - README
## Attention-Weighted Multi-Scale NPR for Deepfake Detection

This directory contains comprehensive presentation materials for your graduate-level AI project presentation on November 17, 2025.

---

## CONTENTS OVERVIEW

### 1. **PRESENTATION_OUTLINE.md** (Main Document)
**Purpose:** Complete presentation content organized by instructor's required sections

**Includes:**
- Project Topic (clear problem statement)
- Background and Context (NPR concept, related work)
- Hypotheses H2 & H3 (testable, with rationale)
- Experimental Setup (architecture, data, timeline, hardware)
- Success Criteria (quantitative metrics)
- Deliverables (code, models, reports)
- Timeline (4-week Gantt chart)
- Limitations & Risks (with mitigations)

**Usage:** This is your primary reference document. All content is written at graduate academic level with proper technical depth.

---

### 2. **ARCHITECTURE_DIAGRAM.md** (Visual Guide)
**Purpose:** Detailed specifications for creating presentation diagrams

**Includes:**
- Main architecture diagram (text-based visual)
- Component breakdowns (Multi-scale NPR, Attention module, Fusion)
- Comparison diagram (Baseline vs. Attention-NPR)
- Mock attention weight heatmap (expected results)
- Timeline Gantt chart visualization
- Hardware requirements diagram
- Expected results (ROC curves, bar charts)
- Slide-by-slide content recommendations

**Usage:** Use this to create PowerPoint visuals. Each diagram includes:
- Text representation (easy to understand)
- PowerPoint instructions (SmartArt, shapes, tables)
- Color coding suggestions
- Layout specifications

---

### 3. **PRESENTATION_SCRIPT.md** (Speaker Notes)
**Purpose:** 15-minute presentation script with timing and talking points

**Includes:**
- Slide-by-slide script (~1 min per slide)
- Key talking points (bullet format for quick reference)
- Transition sentences (smooth flow between topics)
- Technical details (expandable if asked)
- Q&A preparation (anticipated questions + answers)
- Discussion prompts (to encourage engagement)
- Tone & delivery tips (professional academic style)

**Usage:** Practice with this script to:
- Hit 15-minute target (12 min + 3 min Q&A)
- Maintain professional tone
- Prepare for audience questions
- Engage discussion (important for presentation grade!)

---

### 4. **PRESENTATION_SUMMARY.md** (Quick Reference)
**Purpose:** One-page executive summary

**Includes:**
- Project at a glance (title, duration, innovation)
- Hypotheses (concise statement)
- Methodology (architecture overview)
- Timeline (table format)
- Success metrics (table format)
- Key contributions
- Deliverables checklist

**Usage:**
- Print as handout (optional)
- Quick reference during preparation
- Backup if you need to condense presentation

---

## HOW TO BUILD YOUR PRESENTATION

### Recommended Workflow:

**Step 1: Read PRESENTATION_OUTLINE.md**
- Understand full content and structure
- Note which sections are most important for your project

**Step 2: Review ARCHITECTURE_DIAGRAM.md**
- Identify which diagrams to create first
- Prioritize: Main architecture, Timeline, Comparison diagram

**Step 3: Create PowerPoint Slides (15 slides suggested)**

**Slide Order:**
1. Title + Team names
2. Motivation (why this problem matters)
3. Background - NPR concept
4. Problem statement (fixed-scale limitation)
5. **Our Approach** - Main architecture diagram ⭐ Key Slide
6. **Hypotheses H2 & H3** ⭐ Key Slide
7. Training data & test sets
8. Timeline (Gantt chart)
9. Hardware & computational resources
10. **Success criteria** ⭐ Key Slide
11. Expected results (ROC, attention heatmap)
12. Ablation studies (optional, can skip if short on time)
13. Limitations & risks
14. Deliverables
15. Questions & Discussion

**Visual Guidelines:**
- Use diagrams over text wherever possible
- Limit text to 3-5 bullets per slide
- Use clear, large fonts (≥24pt body, ≥36pt headers)
- Color-code components (e.g., Blue=0.25x, Green=0.5x, Orange=0.75x, Red=1.0x)
- Include ONE main visual per slide

**Step 4: Practice with PRESENTATION_SCRIPT.md**
- Read through entire script
- Time yourself (should be 12-13 minutes for content, leaving 2-3 min for Q&A)
- Adjust pace as needed
- Identify places to expand/condense based on audience interest

**Step 5: Prepare Q&A**
- Review anticipated questions in script
- Think through your own potential weak points
- Prepare 2-3 discussion questions to ask audience (shows engagement!)

---

## INSTRUCTOR'S REQUIREMENTS CHECKLIST

Your presentation materials address all required sections:

- ✅ **Project Topic:** Clearly explained in Slide 1 + PRESENTATION_OUTLINE.md Section 1
- ✅ **Background and Context:** PRESENTATION_OUTLINE.md Section 2, Slides 3-4
- ✅ **Hypothesis / Explorative Element:** PRESENTATION_OUTLINE.md Section 3, Slide 6
  - ✅ What do you expect to observe? (H2: 2% accuracy gain, H3: <5% gap)
  - ✅ Why does it require an experiment? (Must test on unseen generators)
- ✅ **Experimental Setup:** PRESENTATION_OUTLINE.md Section 4, Slides 5, 7-9
  - ✅ Code (adaptations of resnet.py, new attention module)
  - ✅ Training data (ForenSynths, 240K images)
  - ✅ Computational resources (RTX 3090, 120 hours confirmed)
  - ✅ Planned outputs (metrics, visualizations, attention weights)
- ✅ **Success Criteria:** PRESENTATION_OUTLINE.md Section 5, Slide 10
- ✅ **Deliverables:** PRESENTATION_OUTLINE.md Section 6, Slide 14

**Additional Materials (Requested):**
- ✅ **Architecture Diagram:** ARCHITECTURE_DIAGRAM.md (multiple versions)
- ✅ **Timeline:** PRESENTATION_OUTLINE.md Section 7 + ARCHITECTURE_DIAGRAM.md (Gantt chart)
- ✅ **Hardware Requirements:** PRESENTATION_OUTLINE.md Section 4.5 + diagram
- ✅ **Limitations:** PRESENTATION_OUTLINE.md Section 8, Slide 13

---

## PRESENTATION TIPS FROM INSTRUCTOR

> "Make your slides clear and focused. Prioritize visuals and diagrams over long sentences whenever possible."

**Your materials follow this:**
- ARCHITECTURE_DIAGRAM.md provides 7+ visual diagrams
- PRESENTATION_OUTLINE.md separates content (for speaker notes) from visuals
- Each slide recommendation emphasizes ONE main visual

> "Extended explanations can be placed in your speaker notes instead of on the slides."

**Your materials follow this:**
- PRESENTATION_SCRIPT.md provides full speaker notes
- Includes "Optional Detail" and "Technical Detail" sections for Q&A
- Slides should have bullets, speaker notes have full explanations

> "Presentations that foster discussion or invite feedback on specific aspects will be favorably reflected in the presentation grade."

**Your materials include:**
- Discussion prompts in PRESENTATION_SCRIPT.md (Section: "Discussion Prompts")
- Slide 15: Questions & Discussion with specific questions for audience
- Q&A preparation to engage constructively with feedback

---

## TIME MANAGEMENT

**Presentation Deadline:** November 17, 2025, 5:00 PM
**Duration:** 15-20 minutes (aim for 15: 12 min + 3 min Q&A)

**Preparation Time Estimate:**
- Create PowerPoint slides: 4-6 hours
- Practice presentation: 2-3 hours (3-5 run-throughs)
- Refine based on timing: 1 hour
- **Total:** ~8-10 hours

**Recommended Schedule:**
- **Nov 15 (2 days before):** Create all slides using ARCHITECTURE_DIAGRAM.md
- **Nov 16 (1 day before):** Practice with PRESENTATION_SCRIPT.md, refine timing
- **Nov 17 (day of):** Final review, upload to Canvas by 5:00 PM

---

## STYLE GUIDELINES

**Academic Tone (Graduate Level):**
- ✅ Use technical terminology correctly (e.g., "interpolate", "residual", "excitation")
- ✅ Cite references where appropriate (CVPR 2024, SENet)
- ✅ Frame hypotheses formally ("We hypothesize that..." not "We think...")
- ✅ Acknowledge limitations transparently
- ✅ Use quantitative metrics (2% improvement, <5% gap)

**Professional Presentation:**
- ✅ Clear structure (intro → background → approach → experiments → results → conclusion)
- ✅ Logical flow (each slide builds on previous)
- ✅ Concise language (avoid unnecessary jargon)
- ✅ Visual clarity (diagrams over text)

**Engagement:**
- ✅ Ask questions to audience (Slide 15)
- ✅ Invite feedback on specific design choices
- ✅ Show openness to suggestions ("That's an excellent point...")

---

## KEY TALKING POINTS (MEMORIZE THESE)

**1. Problem:**
"Current deepfake detectors use fixed-scale NPR (0.5x), which misses artifacts at other scales and fails to generalize to 2025 generators."

**2. Solution:**
"We propose attention-weighted multi-scale NPR: extract 4 scales, use learned attention to weigh them, fuse adaptively before classification."

**3. Hypotheses:**
"H2: Attention improves accuracy by ≥2%. H3: Multi-scale reduces generalization gap to <5% on unseen generators."

**4. Why It Matters:**
"New generators like FLUX and Midjourney v6 require adaptive detection. Fixed scales can't keep up."

**5. Expected Outcome:**
"We expect the attention module to learn generator-specific strategies—GANs favor 0.5x, diffusion models favor 0.25x—improving both accuracy and interpretability."

---

## FINAL CHECKLIST BEFORE PRESENTATION

- [ ] All 15 slides created in PowerPoint
- [ ] Main architecture diagram included (Slide 5)
- [ ] Hypotheses clearly stated (Slide 6)
- [ ] Timeline/Gantt chart included (Slide 8)
- [ ] Success criteria table included (Slide 10)
- [ ] Practiced presentation at least 3 times
- [ ] Timing: 12-13 minutes (leaving buffer for Q&A)
- [ ] Prepared answers for 5+ anticipated questions
- [ ] Discussion questions ready (Slide 15)
- [ ] Uploaded to Canvas by Nov 17, 5:00 PM

---

## CONTACT & SUPPORT

If you need clarification on any materials:
1. Review PRESENTATION_OUTLINE.md (most comprehensive)
2. Check PRESENTATION_SCRIPT.md (includes Q&A preparation)
3. Refer to ARCHITECTURE_DIAGRAM.md (visual guidance)

**Good luck with your presentation! These materials provide everything you need for a professional, graduate-level presentation that meets all instructor requirements.**

---

## APPENDIX: FILE STRUCTURE

```
NPR-DeepfakeDetection/
├── PRESENTATION_OUTLINE.md       ← Main content (all sections)
├── ARCHITECTURE_DIAGRAM.md       ← Visual specifications
├── PRESENTATION_SCRIPT.md        ← Speaker notes + timing
├── PRESENTATION_SUMMARY.md       ← One-page quick reference
├── PRESENTATION_README.md        ← This file (usage guide)
├── Generative AI FInal Project Presentation v1.pptx  ← [Your slides here]
└── 2312.10461v2.pdf             ← Original NPR paper (reference)
```

**Primary Workflow:**
PRESENTATION_OUTLINE.md → ARCHITECTURE_DIAGRAM.md → [Create PowerPoint] → PRESENTATION_SCRIPT.md → [Practice!]

---

*All materials are designed for a 15-minute graduate-level academic presentation, emphasizing clarity, rigor, and engagement. Use them as your comprehensive preparation toolkit.*
