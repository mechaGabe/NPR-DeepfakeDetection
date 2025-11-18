#!/bin/bash

# Multi-Scale NPR Experimental Setup Script
# This script runs comprehensive experiments comparing:
# 1. Baseline (single-scale NPR @ 0.5x)
# 2. Multi-scale with attention fusion
# 3. Ablation studies with different scale combinations

set -e  # Exit on error

# Configuration
GPU_ID=0
DATAROOT="./datasets/ForenSynths_train_val"
CLASSES="car,cat,chair,horse"
BATCH_SIZE=32
LR=0.0002
EPOCHS=50
DELR_FREQ=10

# Create experiment directory
EXPERIMENT_DIR="./experiments/multiscale_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

echo "========================================"
echo "Multi-Scale NPR Experiments"
echo "========================================"
echo "Experiment directory: $EXPERIMENT_DIR"
echo ""

# ========================================
# Experiment 1: Baseline (Original NPR)
# ========================================
echo "Experiment 1: Training Baseline (Single-Scale NPR @ 0.5x)"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --name baseline_single_scale \
    --model_type single_scale \
    --dataroot $DATAROOT \
    --classes $CLASSES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --niter $EPOCHS \
    --delr_freq $DELR_FREQ \
    --checkpoints_dir $EXPERIMENT_DIR

echo "Baseline training complete!"
echo ""

# ========================================
# Experiment 2: Multi-Scale Attention (3 scales)
# ========================================
echo "Experiment 2: Training Multi-Scale Attention NPR (0.25x, 0.5x, 0.75x)"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --name multiscale_attention_3scales \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5,0.75 \
    --dataroot $DATAROOT \
    --classes $CLASSES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --niter $EPOCHS \
    --delr_freq $DELR_FREQ \
    --checkpoints_dir $EXPERIMENT_DIR

echo "Multi-scale attention training complete!"
echo ""

# ========================================
# Experiment 3: Ablation - 2 Scales (Coarse + Medium)
# ========================================
echo "Experiment 3: Ablation Study - 2 Scales (0.25x, 0.5x)"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --name ablation_2scales_coarse \
    --model_type multiscale_attention \
    --npr_scales 0.25,0.5 \
    --dataroot $DATAROOT \
    --classes $CLASSES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --niter $EPOCHS \
    --delr_freq $DELR_FREQ \
    --checkpoints_dir $EXPERIMENT_DIR

echo "Ablation (2 scales - coarse) complete!"
echo ""

# ========================================
# Experiment 4: Ablation - 2 Scales (Medium + Fine)
# ========================================
echo "Experiment 4: Ablation Study - 2 Scales (0.5x, 0.75x)"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --name ablation_2scales_fine \
    --model_type multiscale_attention \
    --npr_scales 0.5,0.75 \
    --dataroot $DATAROOT \
    --classes $CLASSES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --niter $EPOCHS \
    --delr_freq $DELR_FREQ \
    --checkpoints_dir $EXPERIMENT_DIR

echo "Ablation (2 scales - fine) complete!"
echo ""

# ========================================
# Experiment 5: Ablation - 4 Scales
# ========================================
echo "Experiment 5: Ablation Study - 4 Scales (0.2x, 0.4x, 0.6x, 0.8x)"
echo "----------------------------------------"

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --name ablation_4scales \
    --model_type multiscale_attention \
    --npr_scales 0.2,0.4,0.6,0.8 \
    --dataroot $DATAROOT \
    --classes $CLASSES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --niter $EPOCHS \
    --delr_freq $DELR_FREQ \
    --checkpoints_dir $EXPERIMENT_DIR

echo "Ablation (4 scales) complete!"
echo ""

# ========================================
# Testing Phase
# ========================================
echo "========================================"
echo "Testing all models on benchmark datasets"
echo "========================================"

# Create results directory
RESULTS_DIR="$EXPERIMENT_DIR/results"
mkdir -p $RESULTS_DIR

# Test datasets (modify paths as needed)
TEST_DATASETS=(
    "progan"
    "stylegan"
    "stylegan2"
    "biggan"
    "cyclegan"
    "stargan"
    "gaugan"
    "deepfake"
)

# Models to test
MODELS=(
    "baseline_single_scale"
    "multiscale_attention_3scales"
    "ablation_2scales_coarse"
    "ablation_2scales_fine"
    "ablation_4scales"
)

# Find the actual checkpoint directories (they have timestamps)
for model_name in "${MODELS[@]}"; do
    checkpoint_dir=$(find $EXPERIMENT_DIR -type d -name "${model_name}*" | head -1)

    if [ -z "$checkpoint_dir" ]; then
        echo "Warning: Could not find checkpoint for $model_name, skipping..."
        continue
    fi

    echo "Testing model: $model_name"
    echo "Checkpoint: $checkpoint_dir"

    # Test on each dataset
    for dataset in "${TEST_DATASETS[@]}"; do
        echo "  - Testing on $dataset..."

        # Set model type based on model name
        if [[ $model_name == *"multiscale"* ]] || [[ $model_name == *"ablation"* ]]; then
            model_type="multiscale_attention"

            # Extract scales from model name or use defaults
            if [[ $model_name == *"2scales_coarse"* ]]; then
                scales="0.25,0.5"
            elif [[ $model_name == *"2scales_fine"* ]]; then
                scales="0.5,0.75"
            elif [[ $model_name == *"4scales"* ]]; then
                scales="0.2,0.4,0.6,0.8"
            else
                scales="0.25,0.5,0.75"
            fi
        else
            model_type="single_scale"
            scales="0.5"
        fi

        # Run test (you'll need to create test_multiscale.py or modify test.py)
        CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
            --model_path "$checkpoint_dir/model_epoch_last.pth" \
            --model_type $model_type \
            --npr_scales $scales \
            --dataroot "./datasets/Generalization_Test/ForenSynths_test/$dataset" \
            --batch_size 64 \
            >> "$RESULTS_DIR/${model_name}_${dataset}.txt" 2>&1
    done

    echo ""
done

# ========================================
# Attention Visualization
# ========================================
echo "========================================"
echo "Generating attention visualizations"
echo "========================================"

VIZ_DIR="$EXPERIMENT_DIR/visualizations"
mkdir -p $VIZ_DIR

# Visualize attention for the main multi-scale model on different generators
multiscale_checkpoint=$(find $EXPERIMENT_DIR -type d -name "multiscale_attention_3scales*" | head -1)

if [ -n "$multiscale_checkpoint" ]; then
    for dataset in "${TEST_DATASETS[@]}"; do
        echo "Visualizing attention for $dataset..."

        output_viz_dir="$VIZ_DIR/$dataset"
        mkdir -p $output_viz_dir

        CUDA_VISIBLE_DEVICES=$GPU_ID python visualize_attention.py \
            --model_path "$multiscale_checkpoint/model_epoch_last.pth" \
            --dataroot "./datasets/Generalization_Test/ForenSynths_test/$dataset" \
            --output_dir $output_viz_dir \
            --scales 0.25,0.5,0.75 \
            --num_samples 100 \
            --save_npr_maps \
            --gpu_id $GPU_ID
    done
fi

echo ""
echo "========================================"
echo "All experiments complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo "Visualizations saved to: $VIZ_DIR"
echo ""
echo "Next steps:"
echo "1. Analyze results in $RESULTS_DIR"
echo "2. Check attention visualizations in $VIZ_DIR"
echo "3. Create comparison tables for your presentation"
echo "========================================"
