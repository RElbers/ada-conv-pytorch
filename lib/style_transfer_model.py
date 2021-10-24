import torch
from torch import nn


class StyleTransferModel(nn.Module):
    def __init__(self, encoder, decoder, loss_terms):
        super().__init__()
        if not isinstance(loss_terms, dict):
            raise ValueError("<loss_terms> should be a dict of (loss, weight) pairs")

        self.encoder = encoder
        self.decoder = decoder
        self.loss_terms = loss_terms

    def forward(self, content, style, with_embeddings=False):
        self._freeze_encoder()

        # Encode -> Decode
        content_embeddings, style_embeddings = self._encode(content, style)
        output = self._decode(content_embeddings, style_embeddings)

        # Return embeddings if training
        if with_embeddings:
            output_embeddings = self.encoder(output)
            return output, content_embeddings, style_embeddings, output_embeddings
        else:
            return output

    def _decode(self, content_embeddings, style_embeddings):
        output = self.decoder(content_embeddings[-1], style_embeddings[-1])
        return output

    def _encode(self, content, style):
        content_embeddings = self.encoder(content)
        style_embeddings = self.encoder(style)
        return content_embeddings, style_embeddings

    def loss(self, content, style):
        output, content_embeddings, style_embeddings, output_embeddings = self(content, style, with_embeddings=True)

        loss = {}
        for name, (loss_term, weight) in self.loss_terms.items():
            if weight != 0:
                loss[name] = weight * loss_term(content_embeddings, style_embeddings, output_embeddings)

        return loss, output

    def _freeze_encoder(self):
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def post_process(self, img_tensor):
        # [0, 1] -> [0, 255]
        img_tensor = torch.clamp(img_tensor * 255, min=0, max=255)
        img_tensor = torch.einsum('chw->hwc', img_tensor)
        return img_tensor.detach().cpu().numpy()
