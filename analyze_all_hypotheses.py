"""
Comprehensive Analysis Script for All Three Hypotheses

H1: Scale-Specific Artifact Detection
H2: Attention-Based Scale Selection
H3: Cross-Architecture Generalization

This script analyzes results from all experiments and generates:
1. Comparison tables
2. Visualization plots
3. Statistical significance tests
4. Hypothesis confirmation/rejection statements

Usage:
    python analyze_all_hypotheses.py \
        --h1_results ./experiment_results \
        --h2_results ./checkpoints \
        --h3_results ./h3_results

Author: [Your Name]
Date: November 2024
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict


def load_h1_results(results_dir):
    """
    Load H1 results (scale-specific artifacts)

    Args:
        results_dir: directory with results_factor_*.json files

    Returns:
        dict: {factor: {generator: accuracy}}
    """
    print("\n[H1] Loading scale-specific artifact results...")

    h1_data = {}

    for filename in os.listdir(results_dir):
        if filename.startswith('results_factor_') and filename.endswith('.json'):
            factor = float(filename.replace('results_factor_', '').replace('.json', ''))

            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)

            h1_data[factor] = data['results']

    print(f"✓ Loaded results for {len(h1_data)} factors: {list(h1_data.keys())}")
    return h1_data


def load_h2_results(checkpoints_dir):
    """
    Load H2 results (attention vs baselines)

    Expected structure:
        checkpoints/
            baseline_single/validation_results.json
            baseline_concat/validation_results.json
            h2_attention/validation_results.json

    Args:
        checkpoints_dir: directory with model checkpoints

    Returns:
        dict: {model_name: {generator: accuracy}}
    """
    print("\n[H2] Loading attention-based fusion results...")

    h2_data = {}

    # Look for validation results in each checkpoint directory
    for model_dir in os.listdir(checkpoints_dir):
        model_path = os.path.join(checkpoints_dir, model_dir)

        if not os.path.isdir(model_path):
            continue

        # Try to find results file
        results_file = os.path.join(model_path, 'validation_results.json')
        if not os.path.exists(results_file):
            # Try alternate location
            results_file = os.path.join(model_path, 'final_test_results.json')

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
            h2_data[model_dir] = data
            print(f"  ✓ Loaded: {model_dir}")

    if not h2_data:
        print("  ⚠ No H2 results found. Please ensure validation results are saved.")

    return h2_data


def load_h3_results(results_dir):
    """
    Load H3 results (2025 generalization)

    Args:
        results_dir: directory with results_*.json files

    Returns:
        dict: {model_name: {generator_2025: accuracy}}
    """
    print("\n[H3] Loading 2025 generalization results...")

    h3_data = {}

    for filename in os.listdir(results_dir):
        if filename.startswith('results_') and filename.endswith('.json'):
            model_name = filename.replace('results_', '').replace('.json', '')

            with open(os.path.join(results_dir, filename), 'r') as f:
                data = json.load(f)

            h3_data[model_name] = data['results']
            print(f"  ✓ Loaded: {model_name}")

    return h3_data


def analyze_h1(h1_data, output_dir):
    """
    Analyze H1: Scale-Specific Artifact Detection

    Tests whether different generators prefer different NPR scales
    """
    print("\n" + "="*80)
    print("H1 ANALYSIS: Scale-Specific Artifact Detection")
    print("="*80)

    # Separate GAN and Diffusion results
    gan_datasets = ['ForenSynths', 'GANGen-Detection']
    diffusion_datasets = ['DiffusionForensics', 'UniversalFakeDetect']

    factors = sorted(h1_data.keys())

    # Compute mean accuracy for each category
    results = {
        'GAN': {},
        'Diffusion': {}
    }

    for factor in factors:
        gan_accs = []
        diff_accs = []

        for dataset in gan_datasets:
            if dataset in h1_data[factor]:
                for gen, metrics in h1_data[factor][dataset].items():
                    if gen != 'MEAN' and 'accuracy' in metrics:
                        gan_accs.append(metrics['accuracy'])

        for dataset in diffusion_datasets:
            if dataset in h1_data[factor]:
                for gen, metrics in h1_data[factor][dataset].items():
                    if gen != 'MEAN' and 'accuracy' in metrics:
                        diff_accs.append(metrics['accuracy'])

        if gan_accs:
            results['GAN'][factor] = np.mean(gan_accs)
        if diff_accs:
            results['Diffusion'][factor] = np.mean(diff_accs)

    # Print results
    print("\n" + "-"*80)
    print(f"{'Factor':<10s} | {'GAN Mean (%)':>15s} | {'Diffusion Mean (%)':>20s}")
    print("-"*80)

    for factor in factors:
        gan_acc = results['GAN'].get(factor, 0) * 100
        diff_acc = results['Diffusion'].get(factor, 0) * 100
        print(f"{factor:<10.2f} | {gan_acc:>14.1f}% | {diff_acc:>19.1f}%")

    # Find best factors
    if results['GAN']:
        best_gan_factor = max(results['GAN'], key=results['GAN'].get)
        best_gan_acc = results['GAN'][best_gan_factor]

    if results['Diffusion']:
        best_diff_factor = max(results['Diffusion'], key=results['Diffusion'].get)
        best_diff_acc = results['Diffusion'][best_diff_factor]

    print("-"*80)
    print(f"Best for GANs:       Factor {best_gan_factor:.2f} ({best_gan_acc*100:.1f}%)")
    print(f"Best for Diffusion:  Factor {best_diff_factor:.2f} ({best_diff_acc*100:.1f}%)")

    # Hypothesis evaluation
    print("\n" + "-"*80)
    print("H1 HYPOTHESIS EVALUATION:")

    if best_gan_factor != best_diff_factor and abs(best_gan_factor - best_diff_factor) >= 0.15:
        print(f"  ✓ HYPOTHESIS CONFIRMED")
        print(f"  → GANs prefer factor {best_gan_factor:.2f}")
        print(f"  → Diffusion prefers factor {best_diff_factor:.2f}")
        print(f"  → Different generators have scale-specific artifacts!")
    else:
        print(f"  ✗ HYPOTHESIS REJECTED")
        print(f"  → No clear scale preference difference")
        print(f"  → Artifacts may be multi-scale")

    print("-"*80)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(factors))
    width = 0.35

    gan_values = [results['GAN'].get(f, 0)*100 for f in factors]
    diff_values = [results['Diffusion'].get(f, 0)*100 for f in factors]

    ax.bar(x - width/2, gan_values, width, label='GANs', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, diff_values, width, label='Diffusion', color='#3498db', alpha=0.8)

    ax.set_xlabel('NPR Factor', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('H1: Scale-Specific Artifact Detection', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{f:.2f}' for f in factors])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'h1_scale_specific.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_dir}/h1_scale_specific.png")

    return results


def analyze_h2(h2_data, output_dir):
    """
    Analyze H2: Attention-Based Scale Selection

    Tests whether attention fusion improves over baselines
    """
    print("\n" + "="*80)
    print("H2 ANALYSIS: Attention-Based Scale Selection")
    print("="*80)

    # Expected model names
    expected_models = {
        'baseline_single': 'Single-scale (0.5)',
        'baseline_average': 'Multi-scale (average)',
        'baseline_concat': 'Multi-scale (concat)',
        'h2_attention': 'Multi-scale (attention)'
    }

    # Extract mean accuracies
    model_accs = {}

    for model_name, data in h2_data.items():
        if 'SUMMARY' in data and 'mean_accuracy' in data['SUMMARY']:
            model_accs[model_name] = data['SUMMARY']['mean_accuracy']
        elif 'MEAN' in data:
            # Alternative structure
            model_accs[model_name] = data['MEAN']['accuracy']

    if not model_accs:
        print("⚠ No H2 results with accuracy data found")
        return None

    # Sort by accuracy
    sorted_models = sorted(model_accs.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print("\n" + "-"*80)
    print(f"{'Model':<30s} | {'Mean Accuracy':>15s} | {'Improvement':>12s}")
    print("-"*80)

    baseline_acc = None
    if 'baseline_single' in model_accs:
        baseline_acc = model_accs['baseline_single']

    for model_name, acc in sorted_models:
        display_name = expected_models.get(model_name, model_name)
        improvement = ""

        if baseline_acc and model_name != 'baseline_single':
            imp_pct = (acc - baseline_acc) * 100
            improvement = f"+{imp_pct:.1f}%"

        print(f"{display_name:<30s} | {acc*100:>14.1f}% | {improvement:>12s}")

    # Hypothesis evaluation
    print("\n" + "-"*80)
    print("H2 HYPOTHESIS EVALUATION:")

    if 'h2_attention' in model_accs and baseline_acc:
        attention_acc = model_accs['h2_attention']
        improvement = (attention_acc - baseline_acc) * 100

        if improvement >= 3.0:
            print(f"  ✓ HYPOTHESIS CONFIRMED: {improvement:.1f}% ≥ 3%")
            print(f"  → Attention fusion significantly improves detection!")
        elif improvement >= 1.0:
            print(f"  △ PARTIAL SUCCESS: {improvement:.1f}% improvement")
            print(f"  → Attention helps, but less than expected (target: 3-7%)")
        else:
            print(f"  ✗ HYPOTHESIS REJECTED: {improvement:.1f}% < 1%")
            print(f"  → No significant improvement from attention")
    else:
        print("  ⚠ Cannot evaluate - missing attention or baseline results")

    print("-"*80)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [expected_models.get(m, m) for m, _ in sorted_models]
    accs = [a*100 for _, a in sorted_models]
    colors = ['#e74c3c' if i == 0 else '#95a5a6' for i in range(len(models))]
    colors[0] = '#2ecc71'  # Highlight best

    bars = ax.barh(models, accs, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontweight='bold')

    ax.set_xlabel('Mean Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('H2: Attention vs Baseline Fusion Methods', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'h2_attention_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_dir}/h2_attention_comparison.png")

    return model_accs


def analyze_h3(h3_data, output_dir):
    """
    Analyze H3: Cross-Architecture Generalization

    Tests generalization to 2025 generators
    """
    print("\n" + "="*80)
    print("H3 ANALYSIS: Cross-Architecture Generalization (2025 Models)")
    print("="*80)

    # Compare models on 2025 generators
    model_names = list(h3_data.keys())

    if not model_names:
        print("⚠ No H3 results found")
        return None

    # Print per-generator results
    print("\n" + "-"*80)

    # Get all 2025 generators
    all_gens_2025 = set()
    for model_data in h3_data.values():
        all_gens_2025.update(g for g in model_data.keys() if g != 'SUMMARY')

    all_gens_2025 = sorted(all_gens_2025)

    # Table header
    print(f"{'2025 Generator':<20s} | ", end='')
    for model in model_names:
        print(f"{model:<15s} | ", end='')
    print()
    print("-"*80)

    # Table rows
    for gen in all_gens_2025:
        print(f"{gen:<20s} | ", end='')
        for model in model_names:
            if gen in h3_data[model] and 'accuracy' in h3_data[model][gen]:
                acc = h3_data[model][gen]['accuracy'] * 100
                print(f"{acc:>14.1f}% | ", end='')
            else:
                print(f"{'N/A':>14s} | ", end='')
        print()

    # Summary statistics
    print("-"*80)
    print(f"{'MEAN':<20s} | ", end='')

    model_means = {}
    for model in model_names:
        accs = []
        for gen in all_gens_2025:
            if gen in h3_data[model] and 'accuracy' in h3_data[model][gen]:
                accs.append(h3_data[model][gen]['accuracy'])

        if accs:
            mean_acc = np.mean(accs)
            model_means[model] = mean_acc
            print(f"{mean_acc*100:>14.1f}% | ", end='')
        else:
            print(f"{'N/A':>14s} | ", end='')
    print()

    # Hypothesis evaluation
    print("\n" + "-"*80)
    print("H3 HYPOTHESIS EVALUATION:")

    attention_mean = model_means.get('attention', 0)
    single_mean = model_means.get('single', 0)

    if attention_mean >= 0.80:
        print(f"  ✓ HYPOTHESIS CONFIRMED: {attention_mean*100:.1f}% ≥ 80%")
        print(f"  → Excellent generalization to 2025 generators!")

        if single_mean > 0:
            gap = (attention_mean - single_mean) * 100
            print(f"  → Outperforms single-scale by {gap:.1f}%")
    elif attention_mean >= 0.75:
        print(f"  △ CLOSE: {attention_mean*100:.1f}% is near 80% target")
        print(f"  → Reasonable generalization")
    else:
        print(f"  ✗ HYPOTHESIS REJECTED: {attention_mean*100:.1f}% < 75%")
        print(f"  → Poor generalization to 2025 generators")
        print(f"  → May need retraining or architecture changes")

    print("-"*80)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Per-generator comparison
    x = np.arange(len(all_gens_2025))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        accs = []
        for gen in all_gens_2025:
            if gen in h3_data[model] and 'accuracy' in h3_data[model][gen]:
                accs.append(h3_data[model][gen]['accuracy'] * 100)
            else:
                accs.append(0)

        offset = width * (i - len(model_names)/2 + 0.5)
        ax1.bar(x + offset, accs, width, label=model, alpha=0.8)

    ax1.set_xlabel('2025 Generator', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax1.set_title('H3: Per-Generator Performance on 2025 Models', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_gens_2025, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=80, color='r', linestyle='--', label='Target (80%)', alpha=0.5)

    # Plot 2: Mean comparison
    models = list(model_means.keys())
    means = [model_means[m]*100 for m in models]
    colors = ['#2ecc71' if 'attention' in m else '#95a5a6' for m in models]

    ax2.bar(models, means, color=colors, alpha=0.8)
    ax2.axhline(y=80, color='r', linestyle='--', label='Target (80%)', linewidth=2)
    ax2.set_ylabel('Mean Accuracy (%)', fontweight='bold', fontsize=12)
    ax2.set_title('H3: Mean Performance on 2025 Generators', fontweight='bold', fontsize=14)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    ax2.legend()

    # Add value labels
    for i, (model, mean) in enumerate(zip(models, means)):
        ax2.text(i, mean + 2, f'{mean:.1f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'h3_generalization_2025.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_dir}/h3_generalization_2025.png")

    return model_means


def generate_final_report(h1_results, h2_results, h3_results, output_dir):
    """
    Generate comprehensive final report
    """
    report_path = os.path.join(output_dir, 'FINAL_ANALYSIS_REPORT.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS: ALL THREE HYPOTHESES\n")
        f.write("="*80 + "\n\n")

        # H1 Summary
        f.write("H1: SCALE-SPECIFIC ARTIFACT DETECTION\n")
        f.write("-"*80 + "\n")
        f.write("Hypothesis: Different generators produce artifacts at different scales\n\n")

        if h1_results:
            gan_best = max(h1_results['GAN'], key=h1_results['GAN'].get)
            diff_best = max(h1_results['Diffusion'], key=h1_results['Diffusion'].get)

            f.write(f"Best factor for GANs:      {gan_best:.2f}\n")
            f.write(f"Best factor for Diffusion: {diff_best:.2f}\n")

            if gan_best != diff_best:
                f.write("\n✓ CONFIRMED: Different generators prefer different scales\n")
            else:
                f.write("\n✗ REJECTED: No clear scale preference\n")
        else:
            f.write("⚠ No H1 data available\n")

        f.write("\n\n")

        # H2 Summary
        f.write("H2: ATTENTION-BASED SCALE SELECTION\n")
        f.write("-"*80 + "\n")
        f.write("Hypothesis: Attention fusion improves accuracy by 3-7%\n\n")

        if h2_results:
            baseline = h2_results.get('baseline_single', 0)
            attention = h2_results.get('h2_attention', 0)

            if baseline and attention:
                improvement = (attention - baseline) * 100
                f.write(f"Baseline (single-scale): {baseline*100:.2f}%\n")
                f.write(f"Attention (multi-scale): {attention*100:.2f}%\n")
                f.write(f"Improvement:             +{improvement:.2f}%\n\n")

                if improvement >= 3.0:
                    f.write("✓ CONFIRMED: Attention provides significant improvement\n")
                else:
                    f.write("✗ REJECTED: Improvement below 3% threshold\n")
            else:
                f.write("⚠ Missing baseline or attention results\n")
        else:
            f.write("⚠ No H2 data available\n")

        f.write("\n\n")

        # H3 Summary
        f.write("H3: CROSS-ARCHITECTURE GENERALIZATION\n")
        f.write("-"*80 + "\n")
        f.write("Hypothesis: Attention model achieves ≥80% on 2025 generators\n\n")

        if h3_results:
            attention_mean = h3_results.get('attention', 0)
            f.write(f"Mean accuracy on 2025 generators: {attention_mean*100:.2f}%\n\n")

            if attention_mean >= 0.80:
                f.write("✓ CONFIRMED: Excellent generalization to unseen 2025 models\n")
            else:
                f.write("✗ REJECTED: Below 80% target\n")
        else:
            f.write("⚠ No H3 data available\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\n✓ Saved comprehensive report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze All Hypotheses (H1, H2, H3)')

    parser.add_argument('--h1_results', type=str, default='./experiment_results',
                        help='Directory with H1 results (scale-specific)')
    parser.add_argument('--h2_results', type=str, default='./checkpoints',
                        help='Directory with H2 results (attention models)')
    parser.add_argument('--h3_results', type=str, default='./h3_results',
                        help='Directory with H3 results (2025 generalization)')
    parser.add_argument('--output_dir', type=str, default='./final_analysis',
                        help='Directory to save analysis outputs')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("COMPREHENSIVE HYPOTHESIS ANALYSIS")
    print("="*80)

    # Load and analyze each hypothesis
    h1_data = load_h1_results(args.h1_results) if os.path.exists(args.h1_results) else None
    h2_data = load_h2_results(args.h2_results) if os.path.exists(args.h2_results) else None
    h3_data = load_h3_results(args.h3_results) if os.path.exists(args.h3_results) else None

    h1_results = analyze_h1(h1_data, args.output_dir) if h1_data else None
    h2_results = analyze_h2(h2_data, args.output_dir) if h2_data else None
    h3_results = analyze_h3(h3_data, args.output_dir) if h3_data else None

    # Generate final report
    generate_final_report(h1_results, h2_results, h3_results, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All outputs saved to: {args.output_dir}/")
    print("  - h1_scale_specific.png")
    print("  - h2_attention_comparison.png")
    print("  - h3_generalization_2025.png")
    print("  - FINAL_ANALYSIS_REPORT.txt")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
