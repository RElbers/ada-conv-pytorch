import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from lib import dataset
from lib.lightning.datamodule import DataModule
from lib.lightning.lightningmodel import LightningModel


class TensorBoardImageLogger(TensorBoardLogger):
    """
    Wrapper for TensorBoardLogger which logs images to disk,
        instead of the TensorBoard log file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp = self.experiment

        # if not hasattr(exp, 'add_image'):
        exp.add_image = self.add_image

    def add_image(self, tag, img_tensor, global_step):
        dir = Path(self.log_dir, 'images')
        dir.mkdir(parents=True, exist_ok=True)

        file = dir.joinpath(f'{tag}_{global_step:09}.jpg')
        dataset.save(img_tensor, file)


def parse_args():
    # Init parser
    parser = ArgumentParser()
    parser.add_argument('--iterations', type=int, default=160_000,
                        help='The number of training iterations.')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='The directory where the logs are saved to.')
    parser.add_argument('--checkpoint', type=str,
                        help='Resume training from a checkpoint file.')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='How often a validation step is performed. '
                             'Applies the model to several fixed images and calculate the loss.')

    parser = DataModule.add_argparse_args(parser)
    parser = LightningModel.add_argparse_args(parser)

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    if args['checkpoint'] is None:
        max_epochs = 1
        model = LightningModel(**args)
    else:
        # We need to increment the max_epoch variable, because PyTorch Lightning will
        #   resume training from the beginning of the next epoch if resuming from a mid-epoch checkpoint.
        max_epochs = torch.load(args['checkpoint'])['epoch'] + 1
        model = LightningModel.load_from_checkpoint(checkpoint_path=args['checkpoint'])
    datamodule = DataModule(**args)

    logger = TensorBoardImageLogger(args['log_dir'], name='logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(gpus=1,
                      resume_from_checkpoint=args['checkpoint'],
                      max_epochs=max_epochs,
                      max_steps=args['iterations'],
                      checkpoint_callback=True,
                      val_check_interval=args['val_interval'],
                      logger=logger,
                      callbacks=[lr_monitor])

    trainer.fit(model, datamodule=datamodule)
    trainer.save_checkpoint("./model.ckpt")
