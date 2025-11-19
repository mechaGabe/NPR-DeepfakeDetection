"""
Training Script for Multi-Scale NPR Model
For H2: Attention-Based Scale Selection hypothesis

Based on train.py but uses TrainerMultiScale instead of Trainer
"""

import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer_multiscale import TrainerMultiScale
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

import random


def seed_torch(seed=1029):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# Test config (same as original)
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


def get_val_opt():
    """
    Get validation options
    """
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    # Parse options
    opt = TrainOptions().parse()

    # Add multi-scale specific options
    parser = argparse.ArgumentParser()
    parser.add_argument('--npr_scales', type=str, default='0.25,0.5,0.75',
                        help='Comma-separated list of NPR scales')
    multiscale_args, _ = parser.parse_known_args()

    # Parse NPR scales
    npr_scales = [float(s) for s in multiscale_args.npr_scales.split(',')]
    print(f'Using NPR scales: {npr_scales}')

    # Set random seed for reproducibility
    seed_torch(100)

    # Setup paths
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)

    # Setup logger
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('Training multi-scale NPR model')
    print('Command: ' + '  '.join(list(sys.argv)))
    print(f'NPR scales: {npr_scales}')

    # Get validation and test options
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)

    # Create data loader
    data_loader = create_dataloader(opt)

    # Setup tensorboard writers
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    # Create multi-scale trainer
    print(f'Creating multi-scale trainer with scales: {npr_scales}')
    model = TrainerMultiScale(opt, npr_scales=npr_scales)

    def testmodel():
        """
        Test model on all test datasets
        """
        print('*' * 25)
        accs = []
        aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = True

            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc)
            aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))

        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(
            v_id+1, 'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        return np.array(accs).mean(), np.array(aps).mean()

    # Initial test before training
    print("Initial test (before training):")
    model.eval()
    testmodel()
    model.train()

    print(f'cwd: {os.getcwd()}')
    print(f'Starting training for {opt.niter} epochs')

    # Training loop
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      "Train loss: {} at step: {} lr {}".format(
                          model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        # Learning rate decay
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  'changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.adjust_learning_rate()

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        # Test on all datasets
        testmodel()
        model.train()

        print(f'Epoch {epoch} completed in {time.time() - epoch_start_time:.2f}s')

    # Final test after training
    print("\nFinal test (after training):")
    model.eval()
    final_acc, final_ap = testmodel()

    # Save final model
    model.save_networks('last')
    print(f'\nTraining complete! Final accuracy: {final_acc*100:.1f}%, AP: {final_ap*100:.1f}%')
    print(f'Model saved to: {os.path.join(opt.checkpoints_dir, opt.name)}')
