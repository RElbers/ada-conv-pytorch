import argparse
from argparse import ArgumentParser

import torch

from lib import dataset
from lib.lightning.lightningmodel import LightningModel


def stylize_image(model, content_file, style_file, content_size=None):
    device = next(model.parameters()).device

    content = dataset.load(content_file)
    style = dataset.load(style_file)

    content = dataset.content_transforms(content_size)(content)
    style = dataset.style_transforms()(style)

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    output = model(content, style)
    return output[0].detach().cpu()


def parse_args():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content', type=str, default='./content.png')
    parser.add_argument('--style', type=str, default='./style.png')
    parser.add_argument('--output', type=str, default='./output.png')
    parser.add_argument('--model', type=str, default='./model.ckpt')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    model = LightningModel.load_from_checkpoint(checkpoint_path=args['model'])
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    with torch.no_grad():
        output = stylize_image(model, args['content'], args['style'])
    dataset.save(output, args['output'])
