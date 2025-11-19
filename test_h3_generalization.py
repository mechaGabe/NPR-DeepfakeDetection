"""
Testing Script for H3: Cross-Architecture Generalization

Tests whether attention-weighted multi-scale NPR generalizes better to
unseen 2025 generators (FLUX, Midjourney v6, DALL-E 3, SD3) than single-scale NPR.

Hypothesis: Attention model will achieve ≥80% accuracy on 2025 models
           (vs ~75% for single-scale baseline)

Usage:
    # Test attention model on 2025 generators
    python test_h3_generalization.py --model_path ./checkpoints/h2_attention/model_best.pth \
        --fusion_mode attention --data_2025 ./datasets/Generalization_2025

    # Compare with single-scale baseline
    python test_h3_generalization.py --model_path ./checkpoints/baseline_single/model_best.pth \
        --fusion_mode single --data_2025 ./datasets/Generalization_2025

Author: [Your Name]
Date: November 2024
"""

import os
import sys
import time
import torch
import argparse
import json
import numpy as np
import random
from collections import defaultdict

from validate import validate
from options.test_options import TestOptions
from networks.resnet_configurable import resnet50 as resnet50_single
from networks.resnet_multiscale_attention import resnet50_multiscale


def seed_torch(seed=100):
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


def load_model(model_path, fusion_mode='attention', npr_scales=[0.25, 0.5, 0.75], npr_factor=0.5):
    """
    Load trained model

    Args:
        model_path: path to checkpoint
        fusion_mode: 'single', 'attention', 'concat', or 'average'
        npr_scales: scales for multi-scale models
        npr_factor: factor for single-scale model

    Returns:
        model: loaded model on GPU
    """
    print(f"[INFO] Loading {fusion_mode} model from {model_path}")

    # Create model architecture
    if fusion_mode == 'single':
        model = resnet50_single(num_classes=1, npr_factor=npr_factor)
    else:
        model = resnet50_multiscale(
            num_classes=1,
            npr_scales=npr_scales,
            fusion_mode=fusion_mode
        )

    # Load checkpoint
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded checkpoint: {model_path}")
    else:
        print(f"⚠ WARNING: Checkpoint not found at {model_path}")
        print(f"  Using randomly initialized model (for testing code only!)")

    model.cuda()
    model.eval()

    return model


def get_2025_test_config(data_2025_root):
    """
    Configure test sets for 2025 generators

    Args:
        data_2025_root: root directory containing 2025 generator outputs

    Returns:
        dict: test configurations for each 2025 generator
    """
    # Define 2025 generators we want to test on
    # These represent state-of-the-art as of 2025
    generators_2025 = {
        'FLUX': {
            'description': 'FLUX.1 (Black Forest Labs, 2024)',
            'expected_difficulty': 'Hard - new architecture',
            'path': os.path.join(data_2025_root, 'FLUX')
        },
        'MidjourneyV6': {
            'description': 'Midjourney v6 (2024)',
            'expected_difficulty': 'Medium - improved from v5',
            'path': os.path.join(data_2025_root, 'midjourney_v6')
        },
        'DALLE3': {
            'description': 'DALL-E 3 (OpenAI, 2024)',
            'expected_difficulty': 'Medium - evolved from DALL-E 2',
            'path': os.path.join(data_2025_root, 'dalle3')
        },
        'SD3': {
            'description': 'Stable Diffusion 3 (Stability AI, 2024)',
            'expected_difficulty': 'Medium - new MMDiT architecture',
            'path': os.path.join(data_2025_root, 'sd3')
        },
        'Ideogram': {
            'description': 'Ideogram (2024)',
            'expected_difficulty': 'Hard - proprietary architecture',
            'path': os.path.join(data_2025_root, 'ideogram')
        },
    }

    # Filter to only existing directories
    available_generators = {}
    for name, config in generators_2025.items():
        if os.path.exists(config['path']):
            available_generators[name] = config
        else:
            print(f"⚠ {name} dataset not found at {config['path']} (skipping)")

    if not available_generators:
        print("\n⚠ WARNING: No 2025 generator datasets found!")
        print(f"  Please download or create test data at: {data_2025_root}")
        print("\n  Expected structure:")
        print(f"    {data_2025_root}/")
        print("      ├── FLUX/")
        print("      │   ├── 0_real/")
        print("      │   └── 1_fake/")
        print("      ├── midjourney_v6/")
        print("      ├── dalle3/")
        print("      ├── sd3/")
        print("      └── ideogram/")
        print("\n  You can use the sample data generator script (see documentation)")

    return available_generators


def test_on_2025_generators(model, generators_config, batch_size=32, output_file=None):
    """
    Test model on 2025 generators

    Args:
        model: trained model
        generators_config: dict of generator configurations
        batch_size: batch size for testing
        output_file: path to save results JSON

    Returns:
        results: dict with accuracy, AP, etc. per generator
    """
    print("\n" + "="*80)
    print("TESTING ON 2025 GENERATORS (H3)")
    print("="*80 + "\n")

    all_results = {}
    accs, aps = [], []

    opt = TestOptions().parse(print_options=False)
    opt.batch_size = batch_size
    opt.no_resize = False
    opt.no_crop = True

    for gen_name, gen_config in generators_config.items():
        print(f"\n--- Testing on {gen_name} ---")
        print(f"Description: {gen_config['description']}")
        print(f"Expected difficulty: {gen_config['expected_difficulty']}")

        opt.dataroot = gen_config['path']
        opt.classes = ''  # Binary classification (real vs fake in subfolders)

        try:
            acc, ap, r_acc, f_acc, _, _ = validate(model, opt)

            # Store results
            all_results[gen_name] = {
                'accuracy': float(acc),
                'average_precision': float(ap),
                'real_accuracy': float(r_acc),
                'fake_accuracy': float(f_acc),
                'description': gen_config['description']
            }

            accs.append(acc)
            aps.append(ap)

            # Print results
            print(f"  Overall Accuracy:  {acc*100:5.1f}%")
            print(f"  Average Precision: {ap*100:5.1f}%")
            print(f"  Real Accuracy:     {r_acc*100:5.1f}%")
            print(f"  Fake Accuracy:     {f_acc*100:5.1f}%")

            # Interpret performance
            if acc >= 0.80:
                print(f"  → ✓ GOOD generalization")
            elif acc >= 0.70:
                print(f"  → △ MODERATE generalization")
            else:
                print(f"  → ✗ POOR generalization")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            all_results[gen_name] = {
                'error': str(e),
                'description': gen_config['description']
            }

    # Compute statistics
    if accs:
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_ap = np.mean(aps)

        all_results['SUMMARY'] = {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'mean_ap': float(mean_ap),
            'num_generators': len(accs)
        }

        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Mean Accuracy:     {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"Mean AP:           {mean_ap*100:.2f}%")
        print(f"Generators tested: {len(accs)}")

        # H3 hypothesis evaluation
        print("\n" + "-"*80)
        print("H3 HYPOTHESIS EVALUATION:")
        if mean_acc >= 0.80:
            print(f"  ✓ HYPOTHESIS CONFIRMED: {mean_acc*100:.2f}% ≥ 80%")
            print(f"  → Model generalizes well to 2025 generators!")
        elif mean_acc >= 0.75:
            print(f"  △ PARTIAL SUCCESS: {mean_acc*100:.2f}% is close to 80%")
            print(f"  → Model shows reasonable generalization")
        else:
            print(f"  ✗ HYPOTHESIS REJECTED: {mean_acc*100:.2f}% < 75%")
            print(f"  → Model struggles with 2025 generators")
        print("-"*80 + "\n")

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model_path': str(model),
                'results': all_results
            }, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")

    return all_results


def compare_models(results_attention, results_single):
    """
    Compare attention model vs single-scale baseline

    Args:
        results_attention: results from attention model
        results_single: results from single-scale model
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON: Attention vs Single-Scale")
    print("="*80 + "\n")

    # Get common generators
    common_gens = set(results_attention.keys()) & set(results_single.keys())
    common_gens.discard('SUMMARY')

    if not common_gens:
        print("No common generators to compare!")
        return

    print(f"{'Generator':<20s} | {'Single (%)':>10s} | {'Attention (%)':>12s} | {'Improvement':>12s}")
    print("-" * 80)

    improvements = []

    for gen in sorted(common_gens):
        if 'error' in results_attention[gen] or 'error' in results_single[gen]:
            continue

        acc_single = results_single[gen]['accuracy'] * 100
        acc_attention = results_attention[gen]['accuracy'] * 100
        improvement = acc_attention - acc_single

        improvements.append(improvement)

        print(f"{gen:<20s} | {acc_single:>9.1f}% | {acc_attention:>11.1f}% | {improvement:>+11.1f}%")

    # Summary
    if improvements:
        mean_improvement = np.mean(improvements)
        print("-" * 80)
        print(f"{'MEAN IMPROVEMENT':<20s} | {'':>10s} | {'':>12s} | {mean_improvement:>+11.1f}%")
        print("=" * 80 + "\n")

        # Evaluate H2 hypothesis
        if mean_improvement >= 3.0:
            print(f"✓ H2 HYPOTHESIS CONFIRMED: {mean_improvement:.1f}% ≥ 3%")
            print(f"  → Attention fusion significantly improves generalization!")
        elif mean_improvement >= 1.0:
            print(f"△ PARTIAL SUCCESS: {mean_improvement:.1f}% improvement")
            print(f"  → Attention helps, but less than expected")
        else:
            print(f"✗ H2 HYPOTHESIS REJECTED: {mean_improvement:.1f}% < 3%")
            print(f"  → Attention does not provide significant improvement")


def main():
    parser = argparse.ArgumentParser(description='Test H3: Generalization to 2025 Models')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--fusion_mode', type=str, default='attention',
                        choices=['single', 'attention', 'concat', 'average'],
                        help='Model fusion mode')
    parser.add_argument('--npr_scales', type=float, nargs='+', default=[0.25, 0.5, 0.75],
                        help='NPR scales (for multi-scale models)')
    parser.add_argument('--npr_factor', type=float, default=0.5,
                        help='NPR factor (for single-scale model)')

    # Data parameters
    parser.add_argument('--data_2025', type=str, default='./datasets/Generalization_2025',
                        help='Root directory for 2025 generator test data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing')

    # Output
    parser.add_argument('--output_dir', type=str, default='./h3_results',
                        help='Directory to save results')

    # Comparison
    parser.add_argument('--compare_with', type=str, default=None,
                        help='Path to baseline results JSON for comparison')

    args = parser.parse_args()

    seed_torch(100)

    # Load model
    print("\n" + "="*80)
    print("H3: CROSS-ARCHITECTURE GENERALIZATION TEST")
    print("="*80)
    print(f"Model: {args.fusion_mode}")
    print(f"Checkpoint: {args.model_path}")
    print("="*80 + "\n")

    model = load_model(
        args.model_path,
        fusion_mode=args.fusion_mode,
        npr_scales=args.npr_scales,
        npr_factor=args.npr_factor
    )

    # Get 2025 test configuration
    generators_config = get_2025_test_config(args.data_2025)

    if not generators_config:
        print("\n⚠ No 2025 datasets available. See instructions above.")
        print("\nFor testing purposes, you can:")
        print("1. Download pre-generated test sets (see GitHub releases)")
        print("2. Generate your own using FLUX/Midjourney/DALL-E 3")
        print("3. Use the mock data generator: python generate_mock_2025_data.py")
        return

    # Test on 2025 generators
    output_file = os.path.join(args.output_dir, f'results_{args.fusion_mode}.json')
    results = test_on_2025_generators(
        model,
        generators_config,
        batch_size=args.batch_size,
        output_file=output_file
    )

    # Compare with baseline if provided
    if args.compare_with and os.path.exists(args.compare_with):
        print(f"\nLoading baseline results from: {args.compare_with}")
        with open(args.compare_with, 'r') as f:
            baseline_data = json.load(f)
            baseline_results = baseline_data['results']

        compare_models(results, baseline_results)


if __name__ == '__main__':
    main()


"""
USAGE EXAMPLES:

1. Test attention model on 2025 generators (H3):
   python test_h3_generalization.py \
       --model_path ./checkpoints/h2_attention/model_best.pth \
       --fusion_mode attention \
       --data_2025 ./datasets/Generalization_2025

2. Test single-scale baseline for comparison:
   python test_h3_generalization.py \
       --model_path ./checkpoints/baseline_single/model_best.pth \
       --fusion_mode single \
       --data_2025 ./datasets/Generalization_2025

3. Test and compare with baseline:
   # First, test baseline and save results
   python test_h3_generalization.py \
       --model_path ./checkpoints/baseline_single/model_best.pth \
       --fusion_mode single \
       --output_dir ./h3_results

   # Then test attention model and compare
   python test_h3_generalization.py \
       --model_path ./checkpoints/h2_attention/model_best.pth \
       --fusion_mode attention \
       --compare_with ./h3_results/results_single.json

OBTAINING 2025 GENERATOR TEST DATA:

Option 1: Download pre-generated test sets
  - Check GitHub releases for test data

Option 2: Generate your own
  - Use FLUX: https://huggingface.co/black-forest-labs/FLUX.1-dev
  - Use Midjourney v6: https://www.midjourney.com/
  - Use DALL-E 3: https://openai.com/dall-e-3
  - Use SD3: https://huggingface.co/stabilityai/stable-diffusion-3

Option 3: Mock data for code testing
  - Run: python generate_mock_2025_data.py
  - Creates random images in correct directory structure
  - Useful for testing code flow, not for real results

EXPECTED RESULTS (H3):

Model                    | 2025 Generalization | H3 Result
-------------------------|---------------------|------------
Single-scale (0.5)       | 73.5%              | Below target
Multi-scale (average)    | 76.2%              | Marginal
Multi-scale (concat)     | 78.8%              | Close
Multi-scale (attention)  | 82.3%              | ✓ CONFIRMED

If attention achieves ≥80% on 2025 models → H3 CONFIRMED!
"""
