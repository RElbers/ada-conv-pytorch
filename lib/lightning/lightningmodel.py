from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torchvision.utils import make_grid

from lib.nn.adaconv.adaconv_model import AdaConvModel
from lib.nn.adain.adain_model import AdaINModel
from lib.nn.loss import MomentMatchingStyleLoss, GramStyleLoss, CMDStyleLoss, MSEContentLoss


class LightningModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model-type', type=str, default='adaconv', choices=['adain', 'adaconv'])

        # Losses
        parser.add_argument('--style-loss', type=str, default='mm', choices=['mm', 'gram', 'cmd'])
        parser.add_argument('--style-loss-weight', type=float, default=10.0)
        parser.add_argument('--content-loss', type=str, default='mse', choices=['mse'])
        parser.add_argument('--content-loss-weight', type=float, default=1.0)

        # Optimizer
        parser.add_argument('--lr', type=float, default=0.00003)
        # Decays to from 0.0003 to 0.00001 in 160_000 iterations
        parser.add_argument('--lr-decay', type=float, default=0.99998)

        # Add params of other models
        parser = AdaINModel.add_argparse_args(parser)
        parser = AdaConvModel.add_argparse_args(parser)
        return parser

    def __init__(self,
                 model_type,
                 alpha,
                 style_img_size, style_descriptor_depth, predicted_kernel_size,
                 style_loss, style_loss_weight,
                 content_loss, content_loss_weight,
                 lr, lr_decay,
                 **_):
        super().__init__()
        self.save_hyperparameters()

        self.lr_decay = lr_decay
        self.lr = lr

        # Style loss
        loss_terms = {}
        if style_loss == 'mm':
            loss_terms['style'] = (MomentMatchingStyleLoss(), style_loss_weight)
        elif style_loss == 'gram':
            loss_terms['style'] = (GramStyleLoss(), style_loss_weight)
        elif style_loss == 'cmd':
            loss_terms['style'] = (CMDStyleLoss(), style_loss_weight)
        else:
            raise ValueError('style_loss')

        # Content loss
        if content_loss == 'mse':
            loss_terms['content'] = (MSEContentLoss(), content_loss_weight)
        else:
            raise ValueError('content_loss')

        # Model type
        if model_type == 'adain':
            self.model = AdaINModel(loss_terms, alpha)
        elif model_type == 'adaconv':
            self.model = AdaConvModel(loss_terms,
                                      style_img_size,
                                      style_descriptor_depth,
                                      predicted_kernel_size)
        else:
            raise ValueError('model_type')

    def forward(self, content, style,with_embeddings=False):
        return self.model(content, style,with_embeddings=False)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, 'val')

    def shared_step(self, batch, step):
        content, style = batch['content'], batch['style']
        loss_terms, output = self.model.loss(content, style)

        # Log metrics
        for name, loss in loss_terms.items():
            self.log(rf'{step}/loss_{name}', loss.item(), prog_bar=step == 'train')

        # Log images
        if step == 'val':
            imgs = zip(content, style, output)
            imgs = [img for triple in imgs for img in triple]
            grid = make_grid(imgs, nrow=3, padding=0)
            logger = self.logger.experiment
            logger.add_image(rf'{step}_img', grid, global_step=self.global_step + 1)

        return sum(loss_terms.values())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
