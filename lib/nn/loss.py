import torch

from torch import nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def forward(self, content_embeddings, _, output_embeddings):
        return self.content_loss(content_embeddings[-1], output_embeddings[-1])

    def content_loss(self, content, output):
        raise NotImplementedError()


class StyleLoss(nn.Module):
    def forward(self, _, style_embeddings, output_embeddings):
        style_loss = 0
        for (s, o) in zip(style_embeddings, output_embeddings):
            style_loss += self.style_loss(s, o)

        return style_loss

    def style_loss(self, style, output):
        raise NotImplementedError()


class MSEContentLoss(ContentLoss):
    # https://arxiv.org/pdf/1508.06576.pdf

    def content_loss(self, content, output):
        return F.mse_loss(content, output)


class GramStyleLoss(StyleLoss):
    # https://arxiv.org/pdf/1508.06576.pdf

    def style_loss(self, x, y):
        gram_diff = self.gram_matrix(x) - self.gram_matrix(y)
        return torch.mean(torch.sum(gram_diff ** 2, dim=[1, 2]))

    def gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h * w)
        return x @ x.transpose(-2, -1) / (c * h * w)


class MomentMatchingStyleLoss(StyleLoss):
    # https://arxiv.org/pdf/1703.06868.pdf

    def style_loss(self, x, y):
        x_mean = torch.mean(x, dim=[2, 3])
        y_mean = torch.mean(y, dim=[2, 3])
        mean_loss = F.mse_loss(x_mean, y_mean)

        x_std = torch.std(x, dim=[2, 3])
        y_std = torch.std(y, dim=[2, 3])
        std_loss = F.mse_loss(x_std, y_std)

        return mean_loss + std_loss


class CMDStyleLoss(StyleLoss):
    # https://arxiv.org/pdf/2103.07208.pdf

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def style_loss(self, x, y):
        n, c, h, w = x.size()
        x = x.view(n, c, h * w)
        y = y.view(n, c, h * w)

        x = torch.sigmoid(x)
        y = torch.sigmoid(y)

        E_x = torch.mean(x, dim=2)
        E_y = torch.mean(y, dim=2)

        x_centered = x - E_x.unsqueeze(-1)
        y_centered = y - E_y.unsqueeze(-1)

        loss = self.l2_dist(E_x, E_y)
        for k in range(2, self.k + 1):
            x_k = torch.mean(x_centered ** k, dim=2)
            y_k = torch.mean(y_centered ** k, dim=2)

            loss += self.l2_dist(x_k, y_k)

        return torch.mean(loss)

    def l2_dist(self, x, y):
        return torch.norm(x - y, dim=1)
