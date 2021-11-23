import torch

from torch import nn
import torch.nn.functional as F


class MSEContentLoss(nn.Module):
    # https://arxiv.org/pdf/1508.06576.pdf

    def forward(self, x, y):
        return F.mse_loss(x, y)


class GramStyleLoss(nn.Module):
    # https://arxiv.org/pdf/1508.06576.pdf

    def forward(self, x, y):
        gram_diff = self.gram_matrix(x) - self.gram_matrix(y)
        return torch.mean(torch.sum(gram_diff ** 2, dim=[1, 2]))

    def gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h * w)
        return x @ x.transpose(-2, -1) / (c * h * w)


class MomentMatchingStyleLoss(nn.Module):
    # https://arxiv.org/pdf/1703.06868.pdf

    def forward(self, x, y):
        x_mean = torch.mean(x, dim=[2, 3])
        y_mean = torch.mean(y, dim=[2, 3])
        mean_loss = F.mse_loss(x_mean, y_mean)

        x_std = torch.std(x, dim=[2, 3])
        y_std = torch.std(y, dim=[2, 3])
        std_loss = F.mse_loss(x_std, y_std)

        return mean_loss + std_loss


class CMDStyleLoss(nn.Module):
    # https://arxiv.org/pdf/2103.07208.pdf
    # CMDStyleLoss works with pre-activation outputs of VGG19 (without ReLU)

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, x, y):
        x, y = torch.sigmoid(x), torch.sigmoid(y)

        loss = 0
        for x_k, y_k in zip(self.moments(x), self.moments(y)):
            loss += self.l2_dist(x_k, y_k).mean()
        return loss

    def moments(self, x):
        # First vectorize feature maps
        n, c, h, w = x.size()
        x = x.view(n, c, h * w)

        x_mean = torch.mean(x, dim=2, keepdim=True)
        x_centered = x - x_mean

        moments = [x_mean.squeeze(-1)]
        for n in range(2, self.k + 1):
            moments.append(torch.mean(x_centered ** n, dim=2))
        return moments

    def l2_dist(self, x, y):
        return torch.norm(x - y, dim=1)
