"""
Visualization Script: NPR Maps and Experimental Results

This script helps visualize:
1. NPR maps at different interpolation factors
2. Comparison plots of accuracy across factors
3. Heatmaps showing per-generator performance

Usage:
    # Visualize NPR maps for a single image
    python visualize_npr_maps.py --image path/to/image.png --show_maps

    # Create comparison plots from experimental results
    python visualize_npr_maps.py --results_dir ./experiment_results --plot_comparison

Author: [Your Name]
Date: November 2024
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms
from networks.resnet_configurable import resnet50


def compute_npr_map(image, factor):
    """
    Compute NPR (Neural Perceptual Residual) map for visualization

    Args:
        image: PIL Image or numpy array
        factor: interpolation factor (0.25, 0.5, 0.75)

    Returns:
        numpy array: NPR map
    """
    import torch.nn.functional as F

    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)
    elif isinstance(image, np.ndarray):
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        img_tensor = image

    # Compute NPR
    x_downsampled = F.interpolate(img_tensor, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
    x_reconstructed = F.interpolate(x_downsampled, scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    npr_map = img_tensor - x_reconstructed

    # Convert back to numpy
    npr_numpy = npr_map.squeeze(0).permute(1, 2, 0).numpy()

    return npr_numpy


def visualize_npr_comparison(image_path, factors=[0.25, 0.5, 0.75], save_path=None):
    """
    Create a visualization comparing NPR maps at different factors

    Args:
        image_path: path to input image
        factors: list of interpolation factors to compare
        save_path: path to save the figure (optional)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Create figure
    fig, axes = plt.subplots(2, len(factors) + 1, figsize=(4 * (len(factors) + 1), 8))

    # Show original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].text(0.5, 0.5, 'NPR Maps →', ha='center', va='center',
                    fontsize=16, fontweight='bold', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

    # Compute and show NPR maps for each factor
    for idx, factor in enumerate(factors, 1):
        # Compute NPR
        npr_map = compute_npr_map(image, factor)

        # Show original with factor label
        axes[0, idx].imshow(image_np)
        axes[0, idx].set_title(f'Factor = {factor}', fontsize=14, fontweight='bold')
        axes[0, idx].axis('off')

        # Show NPR map (amplified for visibility)
        npr_vis = np.clip(npr_map * 5 + 0.5, 0, 1)  # Amplify and center
        axes[1, idx].imshow(npr_vis)
        axes[1, idx].set_title(f'NPR (factor={factor})', fontsize=12)
        axes[1, idx].axis('off')

        # Add statistics
        npr_std = np.std(npr_map)
        axes[1, idx].text(0.5, -0.1, f'Std: {npr_std:.4f}',
                         ha='center', transform=axes[1, idx].transAxes,
                         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'NPR Maps at Different Interpolation Factors\n{os.path.basename(image_path)}',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_chart(results_dir, save_path=None):
    """
    Create comparison charts from experimental results

    Args:
        results_dir: directory containing result JSON files
        save_path: path to save the figure (optional)
    """
    # Load all results
    all_results = {}
    for filename in os.listdir(results_dir):
        if filename.startswith('results_factor_') and filename.endswith('.json'):
            factor = float(filename.replace('results_factor_', '').replace('.json', ''))
            with open(os.path.join(results_dir, filename), 'r') as f:
                all_results[factor] = json.load(f)

    if not all_results:
        print("No result files found!")
        return

    factors = sorted(all_results.keys())
    test_sets = ['ForenSynths', 'GANGen-Detection', 'DiffusionForensics', 'UniversalFakeDetect']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(factors)))

    for idx, test_set in enumerate(test_sets):
        ax = axes[idx]

        # Check if data exists
        if not all(test_set in exp['results'] for exp in all_results.values()):
            ax.text(0.5, 0.5, f'No data for {test_set}', ha='center', va='center')
            ax.set_title(test_set)
            continue

        # Get generators
        first_exp = all_results[factors[0]]
        generators = [g for g in first_exp['results'][test_set].keys() if g != 'MEAN']

        # Prepare data for plotting
        x_pos = np.arange(len(generators))
        width = 0.8 / len(factors)

        for f_idx, factor in enumerate(factors):
            accuracies = []
            for gen in generators:
                try:
                    acc = all_results[factor]['results'][test_set][gen]['accuracy'] * 100
                    accuracies.append(acc)
                except:
                    accuracies.append(0)

            offset = width * (f_idx - len(factors)/2 + 0.5)
            ax.bar(x_pos + offset, accuracies, width, label=f'Factor {factor}',
                   color=colors[f_idx], alpha=0.8)

        # Formatting
        ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(test_set, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(generators, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])

        # Add mean line
        for f_idx, factor in enumerate(factors):
            try:
                mean_acc = all_results[factor]['results'][test_set]['MEAN']['accuracy'] * 100
                ax.axhline(y=mean_acc, color=colors[f_idx], linestyle='--', alpha=0.5, linewidth=2)
            except:
                pass

    plt.suptitle('NPR Factor Comparison: Accuracy Across Different Generators',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison chart to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_heatmap(results_dir, save_path=None):
    """
    Create heatmap showing performance across factors and generators

    Args:
        results_dir: directory containing result JSON files
        save_path: path to save the figure (optional)
    """
    # Load all results
    all_results = {}
    for filename in os.listdir(results_dir):
        if filename.startswith('results_factor_') and filename.endswith('.json'):
            factor = float(filename.replace('results_factor_', '').replace('.json', ''))
            with open(os.path.join(results_dir, filename), 'r') as f:
                all_results[factor] = json.load(f)

    if not all_results:
        print("No result files found!")
        return

    factors = sorted(all_results.keys())

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Separate GAN and Diffusion results
    gan_test_sets = ['ForenSynths', 'GANGen-Detection']
    diffusion_test_sets = ['DiffusionForensics', 'UniversalFakeDetect']

    for ax_idx, (test_sets, title) in enumerate([(gan_test_sets, 'GAN-based Generators'),
                                                   (diffusion_test_sets, 'Diffusion Models')]):
        # Collect all generators from these test sets
        all_generators = []
        for test_set in test_sets:
            if test_set in all_results[factors[0]]['results']:
                gens = [g for g in all_results[factors[0]]['results'][test_set].keys() if g != 'MEAN']
                all_generators.extend([f"{test_set}/{g}" for g in gens])

        # Create data matrix
        data = np.zeros((len(factors), len(all_generators)))

        for f_idx, factor in enumerate(factors):
            gen_idx = 0
            for test_set in test_sets:
                if test_set not in all_results[factor]['results']:
                    continue
                gens = [g for g in all_results[factor]['results'][test_set].keys() if g != 'MEAN']
                for gen in gens:
                    try:
                        acc = all_results[factor]['results'][test_set][gen]['accuracy'] * 100
                        data[f_idx, gen_idx] = acc
                    except:
                        data[f_idx, gen_idx] = 0
                    gen_idx += 1

        # Create heatmap
        sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', vmin=50, vmax=100,
                    xticklabels=[g.split('/')[-1] for g in all_generators],
                    yticklabels=[f'Factor {f}' for f in factors],
                    ax=axes[ax_idx], cbar_kws={'label': 'Accuracy (%)'})

        axes[ax_idx].set_title(title, fontsize=14, fontweight='bold')
        axes[ax_idx].set_xlabel('Generator', fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylabel('NPR Factor', fontsize=12, fontweight='bold')
        plt.setp(axes[ax_idx].get_xticklabels(), rotation=45, ha='right', fontsize=9)

    plt.suptitle('Performance Heatmap: NPR Factor vs Generator Type',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize NPR maps and experimental results')
    parser.add_argument('--image', type=str, help='Path to image for NPR visualization')
    parser.add_argument('--show_maps', action='store_true', help='Show NPR maps for the image')
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                        help='Directory containing experimental results')
    parser.add_argument('--plot_comparison', action='store_true',
                        help='Create comparison bar charts')
    parser.add_argument('--plot_heatmap', action='store_true',
                        help='Create performance heatmap')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Visualize NPR maps
    if args.show_maps and args.image:
        if os.path.exists(args.image):
            save_path = os.path.join(args.save_dir, f'npr_maps_{os.path.basename(args.image)}')
            visualize_npr_comparison(args.image, save_path=save_path)
        else:
            print(f"Error: Image not found at {args.image}")

    # Plot comparison charts
    if args.plot_comparison:
        save_path = os.path.join(args.save_dir, 'comparison_chart.png')
        plot_comparison_chart(args.results_dir, save_path=save_path)

    # Plot heatmap
    if args.plot_heatmap:
        save_path = os.path.join(args.save_dir, 'performance_heatmap.png')
        plot_heatmap(args.results_dir, save_path=save_path)

    # If no specific action, show help
    if not (args.show_maps or args.plot_comparison or args.plot_heatmap):
        parser.print_help()


if __name__ == '__main__':
    main()


"""
USAGE EXAMPLES:

1. Visualize NPR maps for a sample image:
   python visualize_npr_maps.py --image ./data/test_image.png --show_maps

2. Create comparison charts:
   python visualize_npr_maps.py --results_dir ./experiment_results --plot_comparison

3. Create performance heatmap:
   python visualize_npr_maps.py --results_dir ./experiment_results --plot_heatmap

4. Generate all visualizations:
   python visualize_npr_maps.py --image ./sample.png --show_maps --plot_comparison --plot_heatmap

5. Custom save directory:
   python visualize_npr_maps.py --plot_comparison --save_dir ./my_figures
"""
