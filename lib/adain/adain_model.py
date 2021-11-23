from argparse import ArgumentParser

from torch import nn

from lib.adain.adain import AdaInstanceNorm2d
from lib.vgg import VGGDecoder, VGGEncoder


class AdaINModel(nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alpha', type=float, default=1.0)
        return parser

    def __init__(self, alpha):
        super().__init__()

        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder()
        self.adain = AdaInstanceNorm2d()
        self.alpha = alpha

    def forward(self, content, style, return_embeddings=False):
        self.encoder.freeze()

        # Encode -> Decode
        content_embeddings, style_embeddings = self._encode(content, style)
        output = self.decoder(content_embeddings[-1])

        # Return embeddings if training
        if return_embeddings:
            output_embeddings = self.encoder(output)
            embeddings = {
                'content': content_embeddings,
                'style': style_embeddings,
                'output': output_embeddings
            }
            return output, embeddings
        else:
            return output

    def _encode(self, content, style):
        content_embeddings = self.encoder(content)
        style_embeddings = self.encoder(style)

        t = self.adain(content_embeddings[-1], style_embeddings[-1])
        t = self.alpha * t + (1 - self.alpha) * content_embeddings[-1]

        content_embeddings[-1] = t
        return content_embeddings, style_embeddings
