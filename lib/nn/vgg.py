from torch import nn
from torchvision import models
from torchvision.transforms import transforms


class VGGEncoder(nn.Module):
    def __init__(self, normalize=True, include_activation=False):
        super().__init__()
        self.include_activation = include_activation
        self.vgg = models.vgg19(pretrained=True).features
        self.out_channels = 512
        self.scale_factor = 8

        if normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.normalize = transforms.Normalize(mean=mean, std=std)
        else:
            self.normalize = nn.Identity()

    def forward(self, xs):
        xs = self.normalize(xs)

        if self.include_activation:
            f1 = self.vgg[:2](xs)
            f2 = self.vgg[2:7](f1)
            f3 = self.vgg[7:12](f2)
            f4 = self.vgg[12:21](f3)
        else:
            f1 = self.vgg[:1](xs)
            f2 = self.vgg[1:6](f1)
            f3 = self.vgg[6:11](f2)
            f4 = self.vgg[11:20](f3)

        return [f1, f2, f3, f4]


class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            _conv(512, 256),
            nn.ReLU(),
            _upsample(),

            _conv(256, 256),
            nn.ReLU(),
            _conv(256, 256),
            nn.ReLU(),
            _conv(256, 256),
            nn.ReLU(),
            _conv(256, 128),
            nn.ReLU(),
            _upsample(),

            _conv(128, 128),
            nn.ReLU(),
            _conv(128, 64),
            nn.ReLU(),
            _upsample(),

            _conv(64, 64),
            nn.ReLU(),
            _conv(64, 3),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, content, style):
        ys = self.layers(content)
        return ys


def _conv(in_channels, out_channels, kernel_size=3, padding_mode='reflect'):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding=padding,
                     padding_mode=padding_mode)


def _upsample(scale_factor=2, mode='bilinear'):
    return nn.Upsample(scale_factor=scale_factor, mode=mode)
