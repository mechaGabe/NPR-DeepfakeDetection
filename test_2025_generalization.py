"""
H3: Cross-Architecture Generalization Testing
Test both baseline and multi-scale attention models on 2024-2025 generators

This script tests the hypothesis that multi-scale attention improves
generalization to unseen/future generative models.

2025 Generators to test:
- FLUX (2024)
- Midjourney v6 (2024)
- DALL-E 3 (2024)
- Stable Diffusion 3 (2024)
- Ideogram (2024)
- And potentially newer models
"""

import sys
import time
import os
import json
import torch
import numpy as np
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from networks.resnet_multiscale import resnet50_multiscale
from options.test_options import TestOptions
import random


def seed_torch(seed=1029):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


seed_torch(100)


# H3: 2025 Generator Test Sets
# These are generators that were NOT in the original training data
Generators2025 = {
    'FLUX': {
        'dataroot': './datasets/2025_generators/FLUX',
        'no_resize': False,
        'no_crop': True,
        'description': 'Black Forest Labs FLUX.1 (2024) - Latest diffusion model'
    },
    'MidjourneyV6': {
        'dataroot': './datasets/2025_generators/midjourney_v6',
        'no_resize': False,
        'no_crop': True,
        'description': 'Midjourney v6 (2024) - Latest commercial model'
    },
    'DALLE3': {
        'dataroot': './datasets/2025_generators/dalle3',
        'no_resize': False,
        'no_crop': True,
        'description': 'OpenAI DALL-E 3 (2024) - Latest OpenAI model'
    },
    'SD3': {
        'dataroot': './datasets/2025_generators/sd3',
        'no_resize': False,
        'no_crop': True,
        'description': 'Stable Diffusion 3 (2024) - Latest Stability AI model'
    },
    'Ideogram': {
        'dataroot': './datasets/2025_generators/ideogram',
        'no_resize': False,
        'no_crop': True,
        'description': 'Ideogram (2024) - Text rendering specialist'
    },
}


def test_model_on_generator(model, generator_name, generator_config, opt):
    """
    Test a model on a specific 2025 generator

    Args:
        model: Model to test
        generator_name: Name of generator
        generator_config: Config dict with dataroot, resize, crop settings
        opt: Test options

    Returns:
        results: Dict with accuracy, AP, and per-class results
    """
    dataroot = generator_config['dataroot']

    # Check if dataset exists
    if not os.path.exists(dataroot):
        print(f"WARNING: Dataset not found at {dataroot}")
        print(f"Skipping {generator_name}...")
        return None

    print(f"\nTesting on {generator_name}")
    print(f"Description: {generator_config['description']}")
    print(f"Dataroot: {dataroot}")

    accs = []
    aps = []

    # Test on each class in the generator directory
    classes = [d for d in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, d))]

    if len(classes) == 0:
        # No subdirectories, test on root
        classes = ['']

    for v_id, val in enumerate(classes):
        if val:
            opt.dataroot = os.path.join(dataroot, val)
        else:
            opt.dataroot = dataroot

        opt.classes = ''
        opt.no_resize = generator_config['no_resize']
        opt.no_crop = generator_config['no_crop']

        try:
            acc, ap, _, _, _, _ = validate(model, opt)
            accs.append(acc)
            aps.append(ap)

            class_name = val if val else 'all'
            print(f"  ({v_id}) {class_name:12} acc: {acc*100:.1f}%; ap: {ap*100:.1f}%")
        except Exception as e:
            print(f"  ERROR testing {val}: {e}")
            continue

    if len(accs) > 0:
        mean_acc = np.array(accs).mean()
        mean_ap = np.array(aps).mean()
        print(f"  {'Mean':14} acc: {mean_acc*100:.1f}%; ap: {mean_ap*100:.1f}%")

        return {
            'generator': generator_name,
            'description': generator_config['description'],
            'accuracy': float(mean_acc),
            'ap': float(mean_ap),
            'num_classes': len(accs),
            'per_class_acc': [float(a) for a in accs],
            'per_class_ap': [float(a) for a in aps]
        }
    else:
        return None


def main():
    """
    Main H3 testing script
    Compares baseline vs attention model on 2025 generators
    """
    opt = TestOptions().parse(print_options=False)

    print("=" * 80)
    print("H3: CROSS-ARCHITECTURE GENERALIZATION TESTING")
    print("Testing on 2024-2025 generative models")
    print("=" * 80)

    # Results storage
    all_results = {
        'baseline_single_scale': {},
        'attention_multiscale': {},
        'comparison': {}
    }

    # ========================================================================
    # Test 1: Baseline Single-Scale Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("TESTING BASELINE SINGLE-SCALE MODEL (NPR factor=0.5)")
    print("=" * 80)

    baseline_model_path = opt.model_path  # e.g., 'checkpoints/baseline/model_last.pth'
    print(f"Loading baseline model from: {baseline_model_path}")

    baseline_model = resnet50(num_classes=1)
    baseline_model.load_state_dict(torch.load(baseline_model_path, map_location='cpu'), strict=True)
    baseline_model.cuda()
    baseline_model.eval()

    baseline_results = []
    for gen_name, gen_config in Generators2025.items():
        result = test_model_on_generator(baseline_model, gen_name, gen_config, opt)
        if result:
            baseline_results.append(result)
            all_results['baseline_single_scale'][gen_name] = result

    # ========================================================================
    # Test 2: Multi-Scale Attention Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("TESTING MULTI-SCALE ATTENTION MODEL (NPR scales=[0.25, 0.5, 0.75])")
    print("=" * 80)

    # Check if attention model exists
    attention_model_path = opt.model_path.replace('baseline', 'h2_attention')
    if not os.path.exists(attention_model_path):
        print(f"WARNING: Attention model not found at {attention_model_path}")
        print("Using alternative path...")
        attention_model_path = './checkpoints/h2_attention/model_last.pth'

    print(f"Loading attention model from: {attention_model_path}")

    attention_model = resnet50_multiscale(num_classes=1, npr_scales=[0.25, 0.5, 0.75])

    if os.path.exists(attention_model_path):
        attention_model.load_state_dict(torch.load(attention_model_path, map_location='cpu'), strict=True)
        attention_model.cuda()
        attention_model.eval()

        attention_results = []
        for gen_name, gen_config in Generators2025.items():
            result = test_model_on_generator(attention_model, gen_name, gen_config, opt)
            if result:
                attention_results.append(result)
                all_results['attention_multiscale'][gen_name] = result
    else:
        print(f"ERROR: Could not find attention model at {attention_model_path}")
        print("Skipping attention model testing...")
        attention_results = []

    # ========================================================================
    # Comparison and Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("H3 RESULTS SUMMARY: BASELINE VS ATTENTION")
    print("=" * 80)

    comparison_table = []

    print(f"\n{'Generator':<20} {'Baseline Acc':<15} {'Attention Acc':<15} {'Improvement':<15}")
    print("-" * 70)

    for gen_name in Generators2025.keys():
        baseline_data = all_results['baseline_single_scale'].get(gen_name)
        attention_data = all_results['attention_multiscale'].get(gen_name)

        if baseline_data and attention_data:
            baseline_acc = baseline_data['accuracy'] * 100
            attention_acc = attention_data['accuracy'] * 100
            improvement = attention_acc - baseline_acc

            print(f"{gen_name:<20} {baseline_acc:<15.1f} {attention_acc:<15.1f} {improvement:+.1f}%")

            comparison_table.append({
                'generator': gen_name,
                'baseline_acc': float(baseline_acc),
                'attention_acc': float(attention_acc),
                'improvement': float(improvement)
            })

    all_results['comparison']['per_generator'] = comparison_table

    # Overall statistics
    if len(comparison_table) > 0:
        avg_baseline = np.mean([x['baseline_acc'] for x in comparison_table])
        avg_attention = np.mean([x['attention_acc'] for x in comparison_table])
        avg_improvement = avg_attention - avg_baseline

        print("-" * 70)
        print(f"{'AVERAGE':<20} {avg_baseline:<15.1f} {avg_attention:<15.1f} {avg_improvement:+.1f}%")

        all_results['comparison']['average'] = {
            'baseline_acc': float(avg_baseline),
            'attention_acc': float(avg_attention),
            'improvement': float(avg_improvement)
        }

        print("\n" + "=" * 80)
        print("H3 HYPOTHESIS EVALUATION")
        print("=" * 80)

        if avg_improvement > 0:
            print(f"✓ H3 SUPPORTED: Attention model shows {avg_improvement:.1f}% improvement")
            print("  Multi-scale attention DOES improve generalization to 2025 generators")
        else:
            print(f"✗ H3 NOT SUPPORTED: Attention model shows {avg_improvement:.1f}% change")
            print("  Multi-scale attention does not improve generalization")

    # Save results
    results_path = './h3_results_2025_generalization.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return all_results


if __name__ == '__main__':
    main()
