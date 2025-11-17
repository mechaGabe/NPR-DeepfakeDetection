"""
Experiment Script: Testing Different NPR Interpolation Factors

This script runs the NPR deepfake detection model with different interpolation factors
to test the hypothesis that different scales capture different artifacts.

Usage:
    python run_factor_experiment.py --factor 0.25
    python run_factor_experiment.py --factor 0.50  # baseline
    python run_factor_experiment.py --factor 0.75

Author: [Your Name]
Date: November 2024
"""

import os
import sys
import time
import torch
import argparse
import json
from validate import validate
from networks.resnet_configurable import resnet50
from options.test_options import TestOptions
import numpy as np
import random


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


def get_test_config():
    """
    Define test configurations for different benchmark datasets

    Returns:
        dict: Test configurations including data paths and preprocessing settings
    """
    # NOTE: Update these paths to match your data location
    base_path = './datasets/Generalization_Test'

    return {
        'ForenSynths': {
            'dataroot': os.path.join(base_path, 'ForenSynths_test'),
            'no_resize': False,
            'no_crop': True,
            'description': 'Table 1: GAN-based generators'
        },
        'GANGen-Detection': {
            'dataroot': os.path.join(base_path, 'GANGen-Detection'),
            'no_resize': True,
            'no_crop': True,
            'description': 'Table 2: Novel GAN architectures'
        },
        'DiffusionForensics': {
            'dataroot': os.path.join(base_path, 'DiffusionForensics'),
            'no_resize': False,
            'no_crop': True,
            'description': 'Table 3: Diffusion models'
        },
        'UniversalFakeDetect': {
            'dataroot': os.path.join(base_path, 'UniversalFakeDetect'),
            'no_resize': False,
            'no_crop': True,
            'description': 'Table 4: DALL-E, Glide, etc.'
        },
    }


def run_single_experiment(model_path, npr_factor, test_sets, output_dir='results'):
    """
    Run experiment with a single NPR factor

    Args:
        model_path: path to trained model checkpoint
        npr_factor: interpolation factor (0.25, 0.5, or 0.75)
        test_sets: dictionary of test configurations
        output_dir: directory to save results

    Returns:
        dict: results for all test sets
    """
    seed_torch(100)

    print("\n" + "="*80)
    print(f"EXPERIMENT: NPR Factor = {npr_factor}")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model with specified NPR factor
    print(f"[1/3] Loading model with NPR factor {npr_factor}...")
    model = resnet50(num_classes=1, npr_factor=npr_factor)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        print(f"✓ Loaded checkpoint: {model_path}")
    else:
        print(f"⚠ WARNING: Model checkpoint not found at {model_path}")
        print(f"  Training new model with factor {npr_factor}...")
        print(f"  (You would need to run training script first)")
        # In practice, you'd call a training function here

    model.cuda()
    model.eval()

    # Run tests on all datasets
    all_results = {}

    for test_name, config in test_sets.items():
        if not os.path.exists(config['dataroot']):
            print(f"\n⚠ Skipping {test_name}: data not found at {config['dataroot']}")
            continue

        print(f"\n[2/3] Testing on {test_name}: {config['description']}")
        print("-" * 80)

        test_results = {}
        accs, aps = [], []

        opt = TestOptions().parse(print_options=False)

        # Test on each generator in the dataset
        generators = sorted(os.listdir(config['dataroot']))

        for idx, generator in enumerate(generators):
            generator_path = os.path.join(config['dataroot'], generator)

            if not os.path.isdir(generator_path):
                continue

            opt.dataroot = generator_path
            opt.classes = ''
            opt.no_resize = config['no_resize']
            opt.no_crop = config['no_crop']
            opt.batch_size = 32

            try:
                acc, ap, _, _, _, _ = validate(model, opt)
                accs.append(acc)
                aps.append(ap)

                test_results[generator] = {
                    'accuracy': float(acc),
                    'average_precision': float(ap)
                }

                print(f"  ({idx:2d}) {generator:20s} | Acc: {acc*100:5.1f}% | AP: {ap*100:5.1f}%")

            except Exception as e:
                print(f"  ({idx:2d}) {generator:20s} | ERROR: {str(e)}")
                continue

        # Compute mean performance
        if accs:
            mean_acc = np.mean(accs)
            mean_ap = np.mean(aps)
            test_results['MEAN'] = {
                'accuracy': float(mean_acc),
                'average_precision': float(mean_ap)
            }

            print("-" * 80)
            print(f"  MEAN PERFORMANCE        | Acc: {mean_acc*100:5.1f}% | AP: {mean_ap*100:5.1f}%")
            print()

        all_results[test_name] = test_results

    # Save results
    result_file = os.path.join(output_dir, f'results_factor_{npr_factor}.json')
    with open(result_file, 'w') as f:
        json.dump({
            'npr_factor': npr_factor,
            'model_path': model_path,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': all_results
        }, f, indent=2)

    print(f"\n[3/3] Results saved to: {result_file}")

    return all_results


def compare_factors(results_dir='results'):
    """
    Compare results across different NPR factors

    Args:
        results_dir: directory containing result JSON files
    """
    print("\n" + "="*80)
    print("COMPARISON ACROSS NPR FACTORS")
    print("="*80 + "\n")

    # Load all result files
    all_experiments = {}
    for filename in os.listdir(results_dir):
        if filename.startswith('results_factor_') and filename.endswith('.json'):
            factor = float(filename.replace('results_factor_', '').replace('.json', ''))
            with open(os.path.join(results_dir, filename), 'r') as f:
                all_experiments[factor] = json.load(f)

    if not all_experiments:
        print("No result files found. Run experiments first!")
        return

    # Print comparison table
    factors = sorted(all_experiments.keys())

    for test_set in ['ForenSynths', 'GANGen-Detection', 'DiffusionForensics', 'UniversalFakeDetect']:
        # Check if all experiments have this test set
        if not all(test_set in exp['results'] for exp in all_experiments.values()):
            continue

        print(f"\n{test_set}:")
        print("-" * 80)

        # Get all generators
        first_exp = all_experiments[factors[0]]
        generators = [g for g in first_exp['results'][test_set].keys() if g != 'MEAN']

        # Print header
        print(f"{'Generator':<20s} | ", end='')
        for factor in factors:
            print(f"Factor {factor:.2f} | ", end='')
        print()
        print("-" * 80)

        # Print results for each generator
        for gen in generators:
            print(f"{gen:<20s} | ", end='')
            for factor in factors:
                try:
                    acc = all_experiments[factor]['results'][test_set][gen]['accuracy']
                    print(f"  {acc*100:5.1f}%    | ", end='')
                except:
                    print(f"    N/A     | ", end='')
            print()

        # Print mean
        print("-" * 80)
        print(f"{'MEAN':<20s} | ", end='')
        for factor in factors:
            try:
                acc = all_experiments[factor]['results'][test_set]['MEAN']['accuracy']
                print(f"  {acc*100:5.1f}%    | ", end='')
            except:
                print(f"    N/A     | ", end='')
        print("\n")


def main():
    parser = argparse.ArgumentParser(description='NPR Factor Experiment')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='NPR interpolation factor (default: 0.5)')
    parser.add_argument('--model_path', type=str, default='./NPR.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                        help='Directory to save results')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results across factors')
    parser.add_argument('--run_all', action='store_true',
                        help='Run experiments for all factors (0.25, 0.5, 0.75)')

    args = parser.parse_args()

    if args.compare:
        compare_factors(args.output_dir)
        return

    test_sets = get_test_config()

    if args.run_all:
        print("\n🔬 Running experiments for all factors...")
        factors = [0.25, 0.5, 0.75]
        for factor in factors:
            run_single_experiment(args.model_path, factor, test_sets, args.output_dir)

        print("\n" + "="*80)
        print("All experiments completed! Generating comparison...")
        print("="*80)
        compare_factors(args.output_dir)
    else:
        run_single_experiment(args.factor, args.model_path, test_sets, args.output_dir)


if __name__ == '__main__':
    main()


"""
USAGE EXAMPLES:

1. Run single experiment:
   python run_factor_experiment.py --factor 0.5 --model_path ./NPR.pth

2. Run all experiments:
   python run_factor_experiment.py --run_all --model_path ./NPR.pth

3. Compare results:
   python run_factor_experiment.py --compare

4. Custom output directory:
   python run_factor_experiment.py --factor 0.25 --output_dir ./my_results

EXPECTED WORKFLOW:

1. First, train a model for each factor (or use the provided checkpoint)
2. Run experiments: python run_factor_experiment.py --run_all
3. Results will be saved as JSON files in ./experiment_results/
4. Generate comparison table: python run_factor_experiment.py --compare
5. Use the visualization script to create plots
"""
