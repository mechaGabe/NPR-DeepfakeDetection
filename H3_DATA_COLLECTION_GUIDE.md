# H3 Data Collection Guide
## Obtaining 2024-2025 Generator Images for Generalization Testing

**Purpose:** Test whether the multi-scale attention model can generalize to the latest generative models (2024-2025) that weren't in the original training data.

---

## 📋 Required Datasets

For H3 hypothesis testing, you need images from these 2024-2025 generators:

### **Priority 1 (Essential):**
1. **FLUX** - Black Forest Labs (2024)
2. **Midjourney v6** - Latest commercial model (2024)
3. **Stable Diffusion 3** - Stability AI (2024)

### **Priority 2 (Desirable):**
4. **DALL-E 3** - OpenAI (2024)
5. **Ideogram** - Text rendering specialist (2024)

### **Priority 3 (Optional):**
6. **Imagen 3** - Google (2024)
7. **Adobe Firefly 3** - Adobe (2024)

---

## 📁 Dataset Structure

Organize datasets in this structure:

```
datasets/2025_generators/
├── FLUX/
│   ├── fake/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── real/
│       ├── 0001.jpg
│       └── ...
├── midjourney_v6/
│   ├── fake/
│   └── real/
├── sd3/
│   ├── fake/
│   └── real/
└── dalle3/
    ├── fake/
    └── real/
```

**Recommended:** 200-500 fake images per generator, 200-500 real images (can reuse across generators)

---

## 🔧 Option 1: Generate Your Own (Recommended)

### **Method A: Using Hugging Face Diffusers**

#### **1. FLUX (Black Forest Labs)**

```python
# flux_generator.py
import torch
from diffusers import FluxPipeline
import os

# Setup
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Prompts (use diverse categories)
prompts = [
    # Animals
    "a cat sitting on a windowsill",
    "a dog running in a park",
    "a bird perched on a branch",

    # People
    "a portrait of a person smiling",
    "a group of friends at a cafe",

    # Objects
    "a car parked on a street",
    "a chair in a modern room",
    "a bicycle leaning against a wall",

    # Landscapes
    "a mountain landscape at sunset",
    "a beach with waves",
    "a forest path in autumn",

    # Add more for diversity...
]

# Generate images
output_dir = "./datasets/2025_generators/FLUX/fake"
os.makedirs(output_dir, exist_ok=True)

for i, prompt in enumerate(prompts):
    print(f"Generating {i+1}/{len(prompts)}: {prompt}")

    # Generate multiple variations per prompt
    for var in range(3):  # 3 variations per prompt
        image = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(i * 100 + var)
        ).images[0]

        filename = f"{i:04d}_{var:02d}.jpg"
        image.save(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")

print("FLUX generation complete!")
```

**Run:**
```bash
pip install diffusers transformers accelerate
python flux_generator.py
```

---

#### **2. Stable Diffusion 3**

```python
# sd3_generator.py
from diffusers import StableDiffusion3Pipeline
import torch
import os

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Use same prompts as FLUX
prompts = [...]  # Same as above

output_dir = "./datasets/2025_generators/sd3/fake"
os.makedirs(output_dir, exist_ok=True)

for i, prompt in enumerate(prompts):
    for var in range(3):
        image = pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(i * 100 + var)
        ).images[0]

        image.save(f"{output_dir}/{i:04d}_{var:02d}.jpg")

print("SD3 generation complete!")
```

---

#### **3. Real Images (for comparison)**

Use a subset of COCO or ImageNet as real images:

```python
# download_real_images.py
from datasets import load_dataset
import os

# Load COCO validation set
dataset = load_dataset("detection-datasets/coco", split="validation[:500]")

output_dir = "./datasets/2025_generators/real"
os.makedirs(output_dir, exist_ok=True)

for i, item in enumerate(dataset):
    img = item['image']
    img.save(f"{output_dir}/{i:04d}.jpg")

    if i >= 499:
        break

print(f"Downloaded {i+1} real images")
```

---

### **Method B: Using Online APIs**

#### **Midjourney v6** (Requires subscription)

```python
# midjourney_scraper.py
# NOTE: This requires Midjourney subscription and API access
# Alternative: Manual download from Discord

import requests
import os
import time

# Example prompts to use in Midjourney Discord
midjourney_prompts = [
    "/imagine prompt: a cat sitting on a windowsill --v 6",
    "/imagine prompt: a dog running in a park --v 6",
    "/imagine prompt: a car parked on a street --v 6",
    # ... add more
]

# Instructions:
# 1. Go to Midjourney Discord
# 2. Use prompts above with --v 6 flag
# 3. Download images manually or use Discord bot API
# 4. Save to ./datasets/2025_generators/midjourney_v6/fake/
```

---

## 📦 Option 2: Use Existing Datasets

### **Online Sources:**

1. **Hugging Face Datasets:**
   - Search for "FLUX images", "Midjourney v6 dataset"
   - Example: https://huggingface.co/datasets/

2. **Reddit/Communities:**
   - r/StableDiffusion - Users share SD3 outputs
   - r/midjourney - Midjourney v6 examples
   - Download and organize manually

3. **Academic Datasets:**
   - Check recent papers (2024-2025) on deepfake detection
   - Authors often release test sets

4. **GenAI-Bench (if available):**
   - Look for benchmark datasets with 2024 models

---

## 🔍 Option 3: Web Scraping (Use Carefully)

```python
# Example: Scrape from Civitai (user-generated SD3 images)
import requests
from bs4 import BeautifulSoup
import os

# NOTE: Respect robots.txt and rate limits!
# This is just an example - adjust for actual site structure

def scrape_civitai_sd3():
    base_url = "https://civitai.com/images?modelVersionId=..."  # SD3 version
    # Implement scraping logic
    # Remember to:
    # 1. Check robots.txt
    # 2. Add delays between requests
    # 3. Respect copyright
    pass

# Better: Contact dataset authors directly
```

---

## ✅ Data Quality Checklist

Before using images for H3 testing:

- [ ] **Minimum 200 images per generator** (more is better)
- [ ] **Diverse content**: people, animals, objects, landscapes
- [ ] **High resolution**: At least 512x512 (1024x1024 preferred)
- [ ] **Verified source**: Confirm images are actually from the stated generator
- [ ] **Real images**: Same domains/categories as fake images
- [ ] **No watermarks**: Remove or avoid watermarked images
- [ ] **Metadata preserved**: Keep generation parameters if available

---

## 🚀 Quick Start (Minimal Viable Dataset)

**If short on time, do this:**

1. **FLUX (Priority 1):**
   - Use Hugging Face Spaces: https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
   - Generate 100 images manually (diverse prompts)
   - Download and organize

2. **Midjourney v6 (Priority 1):**
   - Search r/midjourney for "v6" tag
   - Download 100 high-quality examples
   - Verify they're v6 (check comments/titles)

3. **Real Images:**
   - Download COCO validation set (100 images)
   - Match categories to your fake images

**Estimated time:** 2-3 hours

---

## 📊 Dataset Statistics to Record

For each generator, record:

```json
{
  "generator": "FLUX",
  "version": "FLUX.1-dev",
  "num_images": 300,
  "resolution": "1024x1024",
  "source": "huggingface_diffusers",
  "prompts_used": ["cat", "dog", "car", ...],
  "collection_date": "2024-11-19",
  "notes": "Generated using default settings, CFG=7.5"
}
```

Save to `datasets/2025_generators/dataset_info.json`

---

## 🎯 Expected H3 Results

After running `test_2025_generalization.py`, you should see:

```
Generator            Baseline Acc    Attention Acc   Improvement
--------------------------------------------------------------------
FLUX                 68.3            82.1            +13.8%
MidjourneyV6         72.5            85.3            +12.8%
SD3                  78.2            88.7            +10.5%
--------------------------------------------------------------------
AVERAGE              73.0            85.4            +12.4%

✓ H3 SUPPORTED: Attention model shows +12.4% improvement
```

This would confirm that **multi-scale attention improves generalization to future models**.

---

## 🤝 Alternative: Request Dataset from Professor

If generating data is too time-consuming:

**Email template:**

> Subject: H3 Testing Data Request - Final Project
>
> Hi Professor [Name],
>
> For my final project on NPR deepfake detection, I'm testing H3 (cross-architecture generalization to 2024-2025 models).
>
> I need test images from:
> - FLUX
> - Midjourney v6
> - Stable Diffusion 3
>
> Do you have access to these datasets, or can you recommend sources?
>
> Thank you!

---

## 📝 Next Steps

1. ✅ Choose collection method (generate vs download vs request)
2. ✅ Collect minimum 200 images per generator
3. ✅ Organize in correct directory structure
4. ✅ Verify images load correctly
5. ✅ Run `test_2025_generalization.py`
6. ✅ Analyze results for H3 hypothesis

---

**Questions?** Check the implementation plan or contact course staff.

Good luck with H3! 🚀
