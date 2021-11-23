import warnings

from torch import nn
from torchvision import models
from torchvision.transforms import transforms


class VGGEncoder(nn.Module):
    def __init__(self, normalize=True, post_activation=True):
        super().__init__()

        if normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.normalize = transforms.Normalize(mean=mean, std=std)
        else:
            self.normalize = nn.Identity()

        if post_activation:
            layer_names = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'}
        else:
            layer_names = {'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'}
        blocks, block_names, scale_factor, out_channels = extract_vgg_blocks(models.vgg19(pretrained=True).features,
                                                                             layer_names)

        self.blocks = nn.ModuleList(blocks)
        self.block_names = block_names
        self.scale_factor = scale_factor
        self.out_channels = out_channels

    def forward(self, xs):
        xs = self.normalize(xs)

        features = []
        for block in self.blocks:
            xs = block(xs)
            features.append(xs)

        return features

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False


# For AdaIn, not used in AdaConv.
class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            self._conv(512, 256),
            nn.ReLU(),
            self._upsample(),

            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 256),
            nn.ReLU(),
            self._conv(256, 128),
            nn.ReLU(),
            self._upsample(),

            self._conv(128, 128),
            nn.ReLU(),
            self._conv(128, 64),
            nn.ReLU(),
            self._upsample(),

            self._conv(64, 64),
            nn.ReLU(),
            self._conv(64, 3),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, content):
        ys = self.layers(content)
        return ys

    @staticmethod
    def _conv(in_channels, out_channels, kernel_size=3, padding_mode='reflect'):
        padding = (kernel_size - 1) // 2
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=padding,
                         padding_mode=padding_mode)

    @staticmethod
    def _upsample(scale_factor=2, mode='nearest'):
        return nn.Upsample(scale_factor=scale_factor, mode=mode)


def extract_vgg_blocks(layers, layer_names):
    blocks, current_block, block_names = [], [], []
    scale_factor, out_channels = -1, -1
    depth_idx, relu_idx, conv_idx = 1, 1, 1
    for layer in layers:
        name = ''
        if isinstance(layer, nn.Conv2d):
            name = f'conv{depth_idx}_{conv_idx}'
            current_out_channels = layer.out_channels
            layer.padding_mode = 'reflect'
            conv_idx += 1
        elif isinstance(layer, nn.ReLU):
            name = f'relu{depth_idx}_{relu_idx}'
            layer = nn.ReLU(inplace=False)
            relu_idx += 1
        elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
            name = f'pool{depth_idx}'
            depth_idx += 1
            conv_idx = 1
            relu_idx = 1
        else:
            warnings.warn(f' Unexpected layer type: {type(layer)}')

        current_block.append(layer)
        if name in layer_names:
            blocks.append(nn.Sequential(*current_block))
            block_names.append(name)
            scale_factor = 1 * 2 ** (depth_idx - 1)
            out_channels = current_out_channels
            current_block = []

    return blocks, block_names, scale_factor, out_channels
