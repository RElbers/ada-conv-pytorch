from argparse import ArgumentParser
from math import sqrt
from statistics import mean

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from lib.adaconv.adaconv_model import AdaConvModel
from lib.adain.adain_model import AdaINModel
from lib.loss import MomentMatchingStyleLoss, GramStyleLoss, CMDStyleLoss, MSEContentLoss


class LightningModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Add params of other models
        parser = AdaConvModel.add_argparse_args(parser)
        parser = AdaINModel.add_argparse_args(parser)
        parser.add_argument('--model-type', type=str, default='adaconv', choices=['adain', 'adaconv'])

        # Losses
        # mm = Moment Matching, gram = Gram matrix based, cmd = Central Moment Discrepancy
        parser.add_argument('--style-loss', type=str, default='mm', choices=['mm', 'gram', 'cmd'])
        parser.add_argument('--style-weight', type=float, default=10.0)
        parser.add_argument('--content-loss', type=str, default='mse', choices=['mse'])
        parser.add_argument('--content-weight', type=float, default=1.0)

        # Optimizer
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--lr-decay', type=float, default=0.00005)
        return parser

    def __init__(self,
                 model_type,
                 alpha,
                 style_size, style_channels, kernel_size,
                 style_loss, style_weight,
                 content_loss, content_weight,
                 lr, lr_decay,
                 **_):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_decay = lr_decay
        self.style_weight = style_weight
        self.content_weight = content_weight

        # Style loss
        if style_loss == 'mm':
            self.style_loss = MomentMatchingStyleLoss()
        elif style_loss == 'gram':
            self.style_loss = GramStyleLoss()
        elif style_loss == 'cmd':
            self.style_loss = CMDStyleLoss()
        else:
            raise ValueError('style_loss')

        # Content loss
        if content_loss == 'mse':
            self.content_loss = MSEContentLoss()
        else:
            raise ValueError('content_loss')

        # Model type
        if model_type == 'adain':
            self.model = AdaINModel(alpha)
        elif model_type == 'adaconv':
            self.model = AdaConvModel(style_size, style_channels, kernel_size)
        else:
            raise ValueError('model_type')

    def forward(self, content, style, return_embeddings=False):
        return self.model(content, style, return_embeddings)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def shared_step(self, batch, step):
        content, style = batch['content'], batch['style']
        output, embeddings = self.model(content, style, return_embeddings=True)
        content_loss, style_loss = self.loss(embeddings)

        # Log metrics
        self.log(rf'{step}/loss_style', style_loss.item(), prog_bar=step == 'train')
        self.log(rf'{step}/loss_content', content_loss.item(), prog_bar=step == 'train')

        # Return output only for validation step
        if step == 'val':
            return {
                'loss': content_loss + style_loss,
                'output': output,
            }
        return content_loss + style_loss

    def validation_epoch_end(self, outputs):
        if self.global_step == 0:
            return

        with torch.no_grad():
            imgs = [x['output'] for x in outputs]
            imgs = [img for triple in imgs for img in triple]
            nrow = int(sqrt(len(imgs)))
            grid = make_grid(imgs, nrow=nrow, padding=0)
            logger = self.logger.experiment
            logger.add_image(rf'val_img', grid, global_step=self.global_step + 1)

    def loss(self, embeddings):
        # Content
        content_loss = self.content_loss(embeddings['content'][-1], embeddings['output'][-1])

        # Style
        style_loss = []
        for (style_features, output_features) in zip(embeddings['style'], embeddings['output']):
            style_loss.append(self.style_loss(style_features, output_features))
        style_loss = sum(style_loss)

        return self.content_weight * content_loss, self.style_weight * style_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        def lr_lambda(iter):
            return 1 / (1 + 0.0002 * iter)

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
