"""
Attention Weight Visualization for Multi-Scale NPR

This script visualizes:
1. Attention weights across different generators (GANs vs Diffusion models)
2. NPR artifacts at different scales
3. Per-image attention patterns

Usage:
    python visualize_attention.py --model_path checkpoints/multiscale/model.pth \
                                   --dataroot datasets/test/stylegan \
                                   --output_dir visualizations/
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from networks.multiscale_npr import attention_multiscale_npr18
from data import create_dataloader
from options.test_options import TestOptions
from tqdm import tqdm


def setup_args():
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('--model_path', type=str, required=True,
                       help='path to trained model')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='path to test images')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='directory to save visualizations')
    parser.add_argument('--scales', type=str, default='0.25,0.5,0.75',
                       help='NPR scales used in the model')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='number of images to analyze')
    parser.add_argument('--save_npr_maps', action='store_true',
                       help='save NPR artifact visualizations')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    return parser.parse_args()


def visualize_npr_artifacts(image, npr_maps, attention_weights, scales, save_path):
    """
    Visualize NPR artifacts at different scales with their attention weights.

    Args:
        image: Original image tensor [3, H, W]
        npr_maps: Dict of NPR maps at different scales
        attention_weights: Attention weight tensor [num_scales]
        scales: List of scale values
        save_path: Path to save the visualization
    """
    num_scales = len(scales)
    fig, axes = plt.subplots(2, num_scales + 1, figsize=(4 * (num_scales + 1), 8))

    # Convert image to numpy for visualization
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Top row: Original image + NPR maps
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    for i, scale in enumerate(scales):
        npr = npr_maps[f'npr_{scale}'].cpu().squeeze(0)
        # Visualize magnitude of NPR (across RGB channels)
        npr_mag = torch.abs(npr).mean(dim=0).numpy()

        im = axes[0, i + 1].imshow(npr_mag, cmap='hot')
        axes[0, i + 1].set_title(f'NPR @ {scale}x\nWeight: {attention_weights[i]:.3f}')
        axes[0, i + 1].axis('off')
        plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046)

    # Bottom row: Attention weights bar chart
    ax_bar = plt.subplot(2, num_scales + 1, (num_scales + 2, 2 * num_scales + 2))
    bars = ax_bar.bar(range(num_scales), attention_weights.cpu().numpy())

    # Color bars by weight magnitude
    colors = plt.cm.viridis(attention_weights.cpu().numpy() / attention_weights.max().item())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax_bar.set_xlabel('Scale')
    ax_bar.set_ylabel('Attention Weight')
    ax_bar.set_title('Scale Importance')
    ax_bar.set_xticks(range(num_scales))
    ax_bar.set_xticklabels([f'{s}x' for s in scales])
    ax_bar.set_ylim([0, 1])
    ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_attention_by_generator(model, dataloader, scales, device, num_samples=100):
    """
    Analyze average attention weights for a specific generator.

    Returns:
        mean_attention: Mean attention weights [num_scales]
        std_attention: Std of attention weights [num_scales]
        all_attentions: All attention weights [num_samples, num_scales]
    """
    model.eval()
    all_attentions = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Analyzing")):
            if i >= num_samples:
                break

            images = images.to(device)

            # Get attention weights
            _, attention_weights = model(images, return_attention=True)
            all_attentions.append(attention_weights.cpu())

    all_attentions = torch.cat(all_attentions, dim=0)  # [num_samples, num_scales]

    mean_attention = all_attentions.mean(dim=0)
    std_attention = all_attentions.std(dim=0)

    return mean_attention, std_attention, all_attentions


def plot_attention_distribution(all_attentions, scales, save_path):
    """
    Plot distribution of attention weights across scales.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    data_for_box = [all_attentions[:, i].numpy() for i in range(len(scales))]
    bp = axes[0].boxplot(data_for_box, labels=[f'{s}x' for s in scales],
                         patch_artist=True)

    for patch, color in zip(bp['boxes'], plt.cm.Set3(range(len(scales)))):
        patch.set_facecolor(color)

    axes[0].set_ylabel('Attention Weight')
    axes[0].set_xlabel('Scale')
    axes[0].set_title('Attention Weight Distribution')
    axes[0].grid(axis='y', alpha=0.3)

    # Violin plot
    for i, scale in enumerate(scales):
        parts = axes[1].violinplot([all_attentions[:, i].numpy()],
                                   positions=[i],
                                   widths=0.7,
                                   showmeans=True,
                                   showmedians=True)

        for pc in parts['bodies']:
            pc.set_facecolor(plt.cm.Set3(i))
            pc.set_alpha(0.7)

    axes[1].set_ylabel('Attention Weight')
    axes[1].set_xlabel('Scale')
    axes[1].set_title('Attention Weight Density')
    axes[1].set_xticks(range(len(scales)))
    axes[1].set_xticklabels([f'{s}x' for s in scales])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(mean_attentions_dict, scales, save_path):
    """
    Create heatmap comparing attention patterns across different generators.

    Args:
        mean_attentions_dict: Dict mapping generator names to mean attention weights
        scales: List of scale values
        save_path: Path to save the heatmap
    """
    # Prepare data for heatmap
    generator_names = list(mean_attentions_dict.keys())
    attention_matrix = np.array([mean_attentions_dict[gen].numpy()
                                for gen in generator_names])

    # Create heatmap
    plt.figure(figsize=(10, len(generator_names) * 0.5 + 2))
    sns.heatmap(attention_matrix,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=[f'{s}x' for s in scales],
                yticklabels=generator_names,
                cbar_kws={'label': 'Attention Weight'},
                vmin=0,
                vmax=1)

    plt.xlabel('NPR Scale', fontsize=12)
    plt.ylabel('Generator', fontsize=12)
    plt.title('Attention Patterns Across Generators', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = setup_args()

    # Parse scales
    scales = [float(s) for s in args.scales.split(',')]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = attention_multiscale_npr18(num_classes=1, scales=scales)
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")

    # Create data loader
    print(f"Loading data from {args.dataroot}")
    opt = TestOptions().parse(print_options=False)
    opt.dataroot = args.dataroot
    opt.batch_size = 1
    opt.serial_batches = True

    from data import create_dataloader
    dataloader = create_dataloader(opt)

    print(f"Analyzing {args.num_samples} images...")

    # Analyze attention weights
    mean_attn, std_attn, all_attn = analyze_attention_by_generator(
        model, dataloader, scales, device, num_samples=args.num_samples
    )

    # Print statistics
    print("\n" + "="*50)
    print("Attention Weight Statistics")
    print("="*50)
    for i, scale in enumerate(scales):
        print(f"Scale {scale}x: {mean_attn[i]:.4f} ± {std_attn[i]:.4f}")
    print("="*50 + "\n")

    # Save statistics to file
    stats_path = os.path.join(args.output_dir, 'attention_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Attention Weight Statistics\n")
        f.write("="*50 + "\n")
        for i, scale in enumerate(scales):
            f.write(f"Scale {scale}x: {mean_attn[i]:.4f} ± {std_attn[i]:.4f}\n")

    # Plot attention distribution
    dist_path = os.path.join(args.output_dir, 'attention_distribution.png')
    plot_attention_distribution(all_attn, scales, dist_path)
    print(f"Saved attention distribution to {dist_path}")

    # Visualize NPR artifacts for sample images
    if args.save_npr_maps:
        print("\nGenerating NPR visualizations...")
        npr_dir = os.path.join(args.output_dir, 'npr_artifacts')
        os.makedirs(npr_dir, exist_ok=True)

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                if i >= 10:  # Save first 10 samples
                    break

                images = images.to(device)

                # Get NPR maps and attention
                analysis = model.get_scale_contributions(images)

                visualize_npr_artifacts(
                    images[0],
                    analysis['npr_maps'],
                    analysis['attention_weights'][0],
                    scales,
                    os.path.join(npr_dir, f'sample_{i:03d}.png')
                )

        print(f"Saved NPR visualizations to {npr_dir}")

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
