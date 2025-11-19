"""
Training Script for Multi-Scale NPR with Attention

Tests H2: Attention-Based Scale Selection

Compares:
1. Baseline: Single-scale NPR (factor=0.5)
2. Multi-scale average: NPR at [0.25, 0.5, 0.75], simple average
3. Multi-scale concat: NPR at [0.25, 0.5, 0.75], concatenation
4. Multi-scale attention: NPR at [0.25, 0.5, 0.75], learned weighting (H2)

Expected: Attention fusion improves accuracy by 3-7% over single-scale baseline

Usage:
    # Train attention model (H2)
    python train_multiscale.py --fusion_mode attention --name multiscale_attention

    # Train baseline (single-scale)
    python train_multiscale.py --fusion_mode single --name single_scale_baseline

    # Train concat baseline
    python train_multiscale.py --fusion_mode concat --name multiscale_concat

Author: [Your Name]
Date: November 2024
"""

import os
import sys
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import random
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

# Import both single-scale and multi-scale models
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


class MultiScaleTrainer:
    """
    Trainer for multi-scale NPR models

    Handles both single-scale baseline and multi-scale variants
    """
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model based on fusion mode
        if opt.fusion_mode == 'single':
            # Baseline: single-scale NPR
            print(f"[INFO] Creating single-scale model (factor={opt.npr_factor})")
            self.model = resnet50_single(num_classes=opt.num_classes, npr_factor=opt.npr_factor)
        else:
            # Multi-scale: attention, concat, or average
            print(f"[INFO] Creating multi-scale model (fusion={opt.fusion_mode})")
            self.model = resnet50_multiscale(
                num_classes=opt.num_classes,
                npr_scales=opt.npr_scales,
                fusion_mode=opt.fusion_mode
            )

        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=opt.lr,
            betas=(opt.beta1, 0.999)
        )

        self.total_steps = 0
        self.lr = opt.lr

    def set_input(self, data):
        """Set input data"""
        self.input = data['img'].to(self.device)
        self.label = data['label'].to(self.device).float()

    def forward(self):
        """Forward pass"""
        self.output = self.model(self.input).squeeze(1)

    def backward(self):
        """Backward pass"""
        self.loss = self.criterion(self.output, self.label)
        self.loss.backward()

    def optimize_parameters(self):
        """Optimization step"""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def adjust_learning_rate(self):
        """Decay learning rate by 0.5"""
        self.lr *= 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def save_networks(self, epoch):
        """Save model checkpoint"""
        save_filename = f'model_epoch_{epoch}.pth'
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, save_filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"✓ Saved checkpoint: {save_path}")

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Scale NPR Training')

    # Model architecture
    parser.add_argument('--fusion_mode', type=str, default='attention',
                        choices=['single', 'attention', 'concat', 'average'],
                        help='Fusion mode: single (baseline), attention (H2), concat, average')
    parser.add_argument('--npr_scales', type=float, nargs='+', default=[0.25, 0.5, 0.75],
                        help='NPR scales for multi-scale models')
    parser.add_argument('--npr_factor', type=float, default=0.5,
                        help='NPR factor for single-scale baseline')

    # Training parameters
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--dataroot', type=str, default='./datasets/ForenSynths_train_val',
                        help='Path to training data')
    parser.add_argument('--classes', type=str, default='car,cat,chair,horse',
                        help='Training classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--niter', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--delr_freq', type=int, default=10,
                        help='Frequency of learning rate decay')
    parser.add_argument('--loss_freq', type=int, default=100,
                        help='Frequency of printing training loss')

    # Data parameters
    parser.add_argument('--train_split', type=str, default='train',
                        help='Train split name')
    parser.add_argument('--val_split', type=str, default='val',
                        help='Validation split name')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of output classes')
    parser.add_argument('--cropSize', type=int, default=224,
                        help='Crop size')

    # Checkpoint
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Checkpoints directory')

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    opt = parse_args()
    seed_torch(100)

    # Create checkpoint directory
    checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Setup logging
    log_path = os.path.join(checkpoint_path, 'log.log')
    Logger(log_path)

    print("\n" + "="*80)
    print(f"TRAINING MULTI-SCALE NPR MODEL: {opt.name}")
    print("="*80)
    print(f"Fusion mode: {opt.fusion_mode}")
    if opt.fusion_mode != 'single':
        print(f"NPR scales: {opt.npr_scales}")
    else:
        print(f"NPR factor: {opt.npr_factor}")
    print(f"Training classes: {opt.classes}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Learning rate: {opt.lr}")
    print(f"Epochs: {opt.niter}")
    print("="*80 + "\n")

    # Prepare data paths
    train_dataroot = os.path.join(opt.dataroot, opt.train_split)
    val_dataroot = os.path.join(opt.dataroot, opt.val_split)

    # Update opt for data loader
    opt.dataroot = train_dataroot
    opt.isTrain = True

    # Create data loader
    print("[1/5] Loading training data...")
    data_loader = create_dataloader(opt)
    print(f"✓ Training samples: {len(data_loader.dataset)}")

    # Create validation options
    print("[2/5] Setting up validation...")
    val_opt = TestOptions().parse(print_options=False)
    val_opt.dataroot = val_dataroot
    val_opt.classes = opt.classes.split(',')
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.batch_size = opt.batch_size

    # Create model
    print("[3/5] Creating model...")
    trainer = MultiScaleTrainer(opt)

    # TensorBoard writers
    train_writer = SummaryWriter(os.path.join(checkpoint_path, "train"))
    val_writer = SummaryWriter(os.path.join(checkpoint_path, "val"))

    # Test configuration (for final evaluation)
    test_vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
    test_dataroot = opt.dataroot.replace(opt.train_split, 'test')

    def test_model():
        """Test model on all test sets"""
        print('\n' + '*'*80)
        print(f"TESTING @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print('*'*80)

        accs, aps = [], []
        test_opt = TestOptions().parse(print_options=False)

        for v_id, val in enumerate(test_vals):
            test_opt.dataroot = os.path.join(test_dataroot, val)
            if not os.path.exists(test_opt.dataroot):
                continue

            test_opt.classes = os.listdir(test_opt.dataroot)
            test_opt.no_resize = False
            test_opt.no_crop = True
            test_opt.batch_size = opt.batch_size

            acc, ap, _, _, _, _ = validate(trainer.model, test_opt)
            accs.append(acc)
            aps.append(ap)
            print(f"  ({v_id}) {val:15s} | Acc: {acc*100:5.1f}% | AP: {ap*100:5.1f}%")

        if accs:
            mean_acc = np.array(accs).mean()
            mean_ap = np.array(aps).mean()
            print('-'*80)
            print(f"  MEAN                 | Acc: {mean_acc*100:5.1f}% | AP: {mean_ap*100:5.1f}%")
        print('*'*80 + '\n')

        return mean_acc if accs else 0

    # Initial test
    print("[4/5] Initial evaluation...")
    trainer.eval()
    test_model()

    # Training loop
    print(f"[5/5] Starting training for {opt.niter} epochs...")
    best_val_acc = 0

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        trainer.train()

        for i, data in enumerate(data_loader):
            trainer.total_steps += 1
            trainer.set_input(data)
            trainer.optimize_parameters()

            # Print training loss
            if trainer.total_steps % opt.loss_freq == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{opt.niter} | "
                      f"Step {trainer.total_steps} | Loss: {trainer.loss.item():.4f} | "
                      f"LR: {trainer.lr:.6f}")
                train_writer.add_scalar('loss', trainer.loss.item(), trainer.total_steps)

        # Learning rate decay
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(f"\n[LR DECAY] Epoch {epoch}: {trainer.lr:.6f} → {trainer.lr*0.5:.6f}\n")
            trainer.adjust_learning_rate()

        # Validation
        trainer.eval()
        print(f"\n--- Validation @ Epoch {epoch} ---")
        acc, ap = validate(trainer.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, trainer.total_steps)
        val_writer.add_scalar('ap', ap, trainer.total_steps)
        print(f"Val Accuracy: {acc*100:.2f}% | Val AP: {ap*100:.2f}%")

        # Save best model
        if acc > best_val_acc:
            best_val_acc = acc
            trainer.save_networks('best')
            print(f"✓ New best validation accuracy: {best_val_acc*100:.2f}%")

        # Test on all datasets
        test_acc = test_model()

        # Time info
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time/60:.1f} minutes\n")

    # Final test
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    trainer.eval()
    final_acc = test_model()

    # Save final model
    trainer.save_networks('last')

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Final test accuracy: {final_acc*100:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()


"""
USAGE EXAMPLES:

1. Train attention model (H2 - main experiment):
   python train_multiscale.py --fusion_mode attention --name h2_attention --niter 50

2. Train single-scale baseline (for comparison):
   python train_multiscale.py --fusion_mode single --name baseline_single --niter 50

3. Train concatenation baseline:
   python train_multiscale.py --fusion_mode concat --name baseline_concat --niter 50

4. Train average baseline:
   python train_multiscale.py --fusion_mode average --name baseline_average --niter 50

5. Quick test (fewer epochs):
   python train_multiscale.py --fusion_mode attention --name quick_test --niter 10

6. Custom scales:
   python train_multiscale.py --fusion_mode attention --name custom_scales \
       --npr_scales 0.2 0.5 0.8 --niter 50

EXPECTED RESULTS (H2):

Model                    | Mean Accuracy | Improvement
-------------------------|---------------|-------------
Single-scale (0.5)       | 92.5%        | Baseline
Multi-scale (average)    | 93.1%        | +0.6%
Multi-scale (concat)     | 93.8%        | +1.3%
Multi-scale (attention)  | 95.2%        | +2.7% ✓

If attention achieves 3-7% improvement → H2 CONFIRMED!
"""
