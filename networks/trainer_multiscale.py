"""
Trainer for Multi-Scale NPR Model
For H2: Attention-Based Scale Selection hypothesis

Based on networks/trainer.py but uses resnet_multiscale instead of resnet
"""

import torch
import torch.nn as nn
from networks.resnet_multiscale import resnet50_multiscale
from networks.base_model import BaseModel


class TrainerMultiScale(BaseModel):
    """
    Trainer for multi-scale attention model

    Same structure as original Trainer, but uses:
    - resnet50_multiscale instead of resnet50
    - Supports returning attention weights for analysis
    """

    def name(self):
        return 'TrainerMultiScale'

    def __init__(self, opt, npr_scales=[0.25, 0.5, 0.75]):
        """
        Args:
            opt: Training options
            npr_scales: List of NPR interpolation factors
        """
        super(TrainerMultiScale, self).__init__(opt)

        self.npr_scales = npr_scales

        # Create multi-scale model
        if self.isTrain and not opt.continue_train:
            self.model = resnet50_multiscale(
                pretrained=False,
                num_classes=1,
                npr_scales=npr_scales
            )

        if not self.isTrain or opt.continue_train:
            self.model = resnet50_multiscale(
                num_classes=1,
                npr_scales=npr_scales
            )

        # Setup training components
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Initialize optimizer
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999)
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=opt.lr,
                    momentum=0.0,
                    weight_decay=0
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load pretrained weights if needed
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        # Move to GPU
        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        """
        Decay learning rate by 0.9
        Same as original trainer
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*' * 25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*' * 25)
        return True

    def set_input(self, input):
        """
        Set input data and labels
        """
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self, return_attention=False):
        """
        Forward pass

        Args:
            return_attention: If True, also return attention weights

        Returns:
            output: Model predictions
            (optional) attention_weights: Scale attention weights
        """
        if return_attention:
            self.output, self.attention_weights = self.model(
                self.input,
                return_attention=True
            )
            return self.output, self.attention_weights
        else:
            self.output = self.model(self.input)
            return self.output

    def get_loss(self):
        """
        Compute loss
        """
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        """
        Forward pass + backward pass + optimization step
        """
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_attention_weights(self):
        """
        Get attention weights for current batch
        Useful for analysis and visualization

        Returns:
            weights: [B, num_scales] attention weights
        """
        with torch.no_grad():
            return self.model.get_attention_weights(self.input)
