from argparse import ArgumentParser

from torch import nn

from lib.nn.adaconv.kernel_predictor import KernelPredictor
from lib.nn.adaconv.adaconv import AdaConv2d
from lib.style_transfer_model import StyleTransferModel
from lib.nn.vgg import VGGEncoder


class AdaConvModel(StyleTransferModel):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--style-img-size', type=int, default=256)
        parser.add_argument('--style-descriptor-depth', type=int, default=512)
        parser.add_argument('--predicted-kernel-size', type=int, default=3)
        return parser

    def __init__(self, loss_terms, style_img_size, style_descriptor_depth, predicted_kernel_size):
        encoder = VGGEncoder()

        style_in_shape = (encoder.out_channels,
                          style_img_size // encoder.scale_factor,
                          style_img_size // encoder.scale_factor)
        style_out_shape = (style_descriptor_depth,
                           predicted_kernel_size,
                           predicted_kernel_size)
        decoder = AdaConvDecoder(style_in_shape=style_in_shape, style_out_shape=style_out_shape)
        super().__init__(encoder, decoder, loss_terms)


class AdaConvDecoder(nn.Module):
    def __init__(self, style_in_shape, style_out_shape):
        super().__init__()
        self.style_in_shape = style_in_shape
        self.style_out_shape = style_out_shape
        assert self.style_out_shape[1] == self.style_out_shape[2]

        self.style_encoder = GlobalStyleEncoder(in_shape=self.style_in_shape,
                                                out_shape=self.style_out_shape)

        # Inverted VGG with first conv in each scale replaced with AdaConv
        layers = [
            self.kernel_predictor(512, 512, n_groups=512),
            self.ada_conv(512, 256, n_groups=512),
            nn.ReLU(),
            _upsample(),

            self.kernel_predictor(256, 256, n_groups=128),
            self.ada_conv(256, 256, n_groups=128),
            nn.ReLU(),
            _conv(256, 256),
            nn.ReLU(),
            _conv(256, 256),
            nn.ReLU(),
            _conv(256, 128),
            nn.ReLU(),
            _upsample(),

            self.kernel_predictor(128, 128, n_groups=32),
            self.ada_conv(128, 128, n_groups=32),
            nn.ReLU(),
            _conv(128, 64),
            nn.ReLU(),
            _upsample(),

            self.kernel_predictor(64, 64, n_groups=8),
            self.ada_conv(64, 64, n_groups=8),
            nn.ReLU(),
            _conv(64, 3),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, content, style):
        style = self.style_encoder(style)

        for module in self.layers:
            if isinstance(module, KernelPredictor):
                w_spatial, w_pointwise, bias = module(style)
            elif isinstance(module, AdaConv2d):
                content = module(content,
                                 w_spatial,
                                 w_pointwise,
                                 bias)
            else:
                content = module(content)

        output = content
        return output

    def kernel_predictor(self, in_channels, out_channels, n_groups):
        style_channels = self.style_out_shape[0]
        kernel_size = self.style_out_shape[1]
        return KernelPredictor(in_channels=in_channels,
                               out_channels=out_channels,
                               style_channels=style_channels,
                               n_groups=n_groups,
                               kernel_size=kernel_size)

    def ada_conv(self, in_channels, out_channels, n_groups):
        return AdaConv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         n_groups=n_groups)


class GlobalStyleEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        channels = in_shape[0]

        self.downscale = _downsample(scale=8)
        self.downscale = nn.Sequential(
            _conv(channels, channels),
            nn.LeakyReLU(),
            _downsample(),
            #
            _conv(channels, channels),
            nn.LeakyReLU(),
            _downsample(),
            #
            _conv(channels, channels),
            nn.LeakyReLU(),
            _downsample(),
        )

        in_features = self.in_shape[0] * (self.in_shape[1] // 8) * self.in_shape[2] // 8
        out_features = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, xs):
        ys = self.downscale(xs)
        ys = ys.reshape(len(xs), -1)

        W = self.fc(ys)
        W = W.reshape(len(xs), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return W


def _conv(in_channels, out_channels, kernel_size=3, padding_mode='reflect'):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     padding_mode=padding_mode)


def _upsample(scale=2, mode='nearest'):
    return nn.Upsample(scale_factor=scale, mode=mode)


def _downsample(scale=2):
    return nn.AvgPool2d(scale, scale)
