"""
Visualization and Analysis Script for H2 and H3 Results
Based on HW3 plotting experience

Creates publication-quality figures for:
- H2: Multi-scale attention analysis
- H3: Generalization to 2025 models
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import torch
from networks.resnet_multiscale import resnet50_multiscale
from PIL import Image
import torchvision.transforms as transforms


# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_h3_comparison(results_file='h3_results_2025_generalization.json', save_path='h3_comparison.png'):
    """
    Plot H3 results: Baseline vs Attention on 2025 generators

    Creates a grouped bar chart comparing detection accuracy
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    comparison = results['comparison']['per_generator']

    # Extract data
    generators = [c['generator'] for c in comparison]
    baseline_accs = [c['baseline_acc'] for c in comparison]
    attention_accs = [c['attention_acc'] for c in comparison]
    improvements = [c['improvement'] for c in comparison]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Grouped bar chart
    x = np.arange(len(generators))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_accs, width, label='Baseline (Single-Scale)',
                    color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, attention_accs, width, label='Attention (Multi-Scale)',
                    color='#4ECDC4', alpha=0.8)

    ax1.set_xlabel('Generator (2024-2025)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('H3: Detection Accuracy on 2025 Generators')
    ax1.set_xticks(x)
    ax1.set_xticklabels(generators, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    # Plot 2: Improvement bar chart
    colors = ['#51CF66' if imp > 0 else '#FF6B6B' for imp in improvements]
    bars3 = ax2.bar(x, improvements, color=colors, alpha=0.8)

    ax2.set_xlabel('Generator (2024-2025)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('H3: Attention Model Improvement over Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(generators, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars3, improvements)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"H3 comparison plot saved to: {save_path}")

    return fig


def plot_h3_summary(results_file='h3_results_2025_generalization.json', save_path='h3_summary.png'):
    """
    Plot H3 summary statistics
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    avg_stats = results['comparison']['average']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Baseline\n(Single-Scale)', 'Attention\n(Multi-Scale)']
    accs = [avg_stats['baseline_acc'], avg_stats['attention_acc']]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax.bar(models, accs, color=colors, alpha=0.8, width=0.5)

    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('H3: Average Detection Accuracy on 2025 Generators')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement annotation
    improvement = avg_stats['improvement']
    ax.annotate(f'Improvement: {improvement:+.1f}%',
                xy=(1, accs[1]), xytext=(1.3, accs[1]),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=14, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"H3 summary plot saved to: {save_path}")

    return fig


def visualize_attention_weights(model_path, image_paths, save_path='attention_weights.png'):
    """
    Visualize attention weights for sample images

    Shows which NPR scales the model focuses on for different generators
    """
    # Load model
    model = resnet50_multiscale(num_classes=1, npr_scales=[0.25, 0.5, 0.75])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.cuda()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Process images
    images = []
    labels = []
    attention_weights = []

    for img_path, label in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()

        with torch.no_grad():
            weights = model.get_attention_weights(img_tensor)

        images.append(img)
        labels.append(label)
        attention_weights.append(weights.cpu().numpy()[0])

    if len(images) == 0:
        print("No valid images found for attention visualization")
        return None

    # Create visualization
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 2, figsize=(12, 4*n_images))

    if n_images == 1:
        axes = axes.reshape(1, -1)

    scales = ['0.25', '0.5', '0.75']

    for i, (img, label, weights) in enumerate(zip(images, labels, attention_weights)):
        # Show image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'{label}', fontsize=14)
        axes[i, 0].axis('off')

        # Show attention weights
        bars = axes[i, 1].bar(scales, weights, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
        axes[i, 1].set_ylabel('Attention Weight')
        axes[i, 1].set_xlabel('NPR Scale')
        axes[i, 1].set_title(f'Scale Attention Weights', fontsize=14)
        axes[i, 1].set_ylim([0, 1])
        axes[i, 1].grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            axes[i, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{weight:.3f}',
                           ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention weights visualization saved to: {save_path}")

    return fig


def create_h2_analysis_plot(save_path='h2_analysis.png'):
    """
    Create H2 analysis plot showing multi-scale benefit

    This is a conceptual plot showing how different generators
    might benefit from different scales
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulated data (replace with actual results if available)
    scales = [0.25, 0.5, 0.75]
    gan_performance = [72, 85, 78]  # Example: GANs perform best at 0.5
    diffusion_performance = [88, 80, 75]  # Example: Diffusion best at 0.25

    ax.plot(scales, gan_performance, 'o-', linewidth=2, markersize=10,
            label='GAN-generated', color='#FF6B6B')
    ax.plot(scales, diffusion_performance, 's-', linewidth=2, markersize=10,
            label='Diffusion-generated', color='#4ECDC4')

    ax.set_xlabel('NPR Scale Factor')
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_title('H2: Scale-Specific Detection Performance')
    ax.set_xticks(scales)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # Annotate optimal scales
    ax.annotate('Optimal for GANs', xy=(0.5, 85), xytext=(0.5, 90),
                ha='center', fontsize=11, color='#FF6B6B',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B'))
    ax.annotate('Optimal for Diffusion', xy=(0.25, 88), xytext=(0.4, 92),
                ha='center', fontsize=11, color='#4ECDC4',
                arrowprops=dict(arrowstyle='->', color='#4ECDC4'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"H2 analysis plot saved to: {save_path}")

    return fig


def generate_all_visualizations():
    """
    Generate all visualization figures for the presentation
    """
    print("=" * 60)
    print("Generating H2 and H3 Visualizations")
    print("=" * 60)

    output_dir = './figures'
    os.makedirs(output_dir, exist_ok=True)

    # H3 visualizations
    if os.path.exists('h3_results_2025_generalization.json'):
        print("\nGenerating H3 visualizations...")
        plot_h3_comparison(save_path=os.path.join(output_dir, 'h3_comparison.png'))
        plot_h3_summary(save_path=os.path.join(output_dir, 'h3_summary.png'))
    else:
        print("\nWarning: h3_results_2025_generalization.json not found")
        print("Run test_2025_generalization.py first to generate H3 results")

    # H2 visualizations
    print("\nGenerating H2 visualizations...")
    create_h2_analysis_plot(save_path=os.path.join(output_dir, 'h2_analysis.png'))

    # Attention weights visualization (if model and images available)
    model_path = './checkpoints/h2_attention/model_last.pth'
    if os.path.exists(model_path):
        # Example image paths (update with actual paths)
        example_images = [
            ('./datasets/2025_generators/FLUX/fake/0001.jpg', 'FLUX (Fake)'),
            ('./datasets/2025_generators/midjourney_v6/fake/0001.jpg', 'Midjourney v6 (Fake)'),
            ('./datasets/2025_generators/real/0001.jpg', 'Real Image'),
        ]

        print("\nGenerating attention weights visualization...")
        try:
            visualize_attention_weights(
                model_path,
                example_images,
                save_path=os.path.join(output_dir, 'attention_weights.png')
            )
        except Exception as e:
            print(f"Could not generate attention weights visualization: {e}")
    else:
        print(f"\nWarning: Model not found at {model_path}")
        print("Train the attention model first using train_multiscale.py")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_visualizations()
