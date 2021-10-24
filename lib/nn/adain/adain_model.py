from argparse import ArgumentParser

from lib.nn.adain.adain import AdaInstanceNorm2d
from lib.style_transfer_model import StyleTransferModel
from lib.nn.vgg import VGGDecoder, VGGEncoder


class AdaINModel(StyleTransferModel):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alpha', type=float, default=1.0)
        return parser

    def __init__(self, loss_terms, alpha):
        super().__init__(VGGEncoder(),
                         VGGDecoder(),
                         loss_terms)

        self.adain = AdaInstanceNorm2d()
        self.alpha = alpha

    def _encode(self, content, style):
        content_embeddings, style_embeddings = super()._encode(content, style)

        t = self.adain(content_embeddings[-1], style_embeddings[-1])
        t = self.alpha * t + (1 - self.alpha) * content_embeddings[-1]

        content_embeddings[-1] = t
        return content_embeddings, style_embeddings
