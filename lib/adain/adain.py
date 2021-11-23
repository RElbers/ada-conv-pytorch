import torch
from torch import nn
from torch.nn import functional as F


class AdaInstanceNorm2d(nn.Module):
    def __init__(self, mlp_features=None):
        super().__init__()

        # If mlp_features is specified, the bias and scale are estimated by transforming a code vector,
        #   as in MUNIT (https://arxiv.org/pdf/1804.04732.pdf).
        if mlp_features is not None:
            in_features = mlp_features[0]
            out_features = mlp_features[1]

            self._scale = nn.Linear(in_features, out_features)
            self._bias = nn.Linear(in_features, out_features)
        # If mlp_features is not specified, the bias and scale are the mean and std of 2d feature maps,
        #   as in standard AdaIN (https://arxiv.org/pdf/1703.06868.pdf).
        else:
            self._scale = self._std
            self._bias = self._mean

    def forward(self, x, y):
        y_scale = self._scale(y).unsqueeze(-1).unsqueeze(-1)
        y_bias = self._bias(y).unsqueeze(-1).unsqueeze(-1)

        x = F.instance_norm(x)
        x = (x * y_scale) + y_bias
        return x

    def _std(self, x):
        return torch.std(x, dim=[2, 3])

    def _mean(self, x):
        return torch.mean(x, dim=[2, 3])
