import argparse
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from tqdm import tqdm

from lib import dataset
from lib.lightning.lightningmodel import LightningModel
from stylize import stylize_image


def resize(img, size):
    c, h, w = img.size()
    if h < w:
        small_size = size[0]
    else:
        small_size = size[1]

    img = TF.resize(img, small_size)
    img = TF.center_crop(img, size)
    return img


def parse_args():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content-dir', type=str, default='./test_images/content')
    parser.add_argument('--style-dir', type=str, default='./test_images/style')
    parser.add_argument('--output-dir', type=str, default='./test_images/output')
    parser.add_argument('--model', type=str, default='./model.ckpt')
    parser.add_argument('--save-as', type=str, default='png')
    parser.add_argument('--content-size', type=int, default=512,
                        help='Content images are resized such that the smaller edge has this size.')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    ext = args['save_as']
    content_transform = dataset.content_transforms(args['content_size'])
    style_transform = dataset.style_transforms()

    content_files = dataset.files_in(args['content_dir'])
    style_files = dataset.files_in(args['style_dir'])
    output_dir = Path(args['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    model = LightningModel.load_from_checkpoint(checkpoint_path=args['model'])
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    pbar = tqdm(total=len(content_files) * len(style_files))
    with torch.no_grad():
        # Add style images at top row
        imgs = [style_transform(dataset.load(f)) for f in style_files]

        for i, content in enumerate(content_files):
            # Add content images at left column
            imgs.append(content_transform(dataset.load(content)))

            for j, style in enumerate(style_files):
                # Stylize content-style pair
                output = stylize_image(model, content, style, content_size=args['content_size'])

                dataset.save(output, output_dir.joinpath(f'{i:02}--{j:02}.{ext}'))
                imgs.append(output)
                pbar.update(1)

        # Make all same size for table
        avg_h = int(sum([img.size(1) for img in imgs]) / len(imgs))
        avg_w = int(sum([img.size(2) for img in imgs]) / len(imgs))
        imgs = [resize(img, [avg_h, avg_w]) for img in imgs]
        imgs = [torch.ones((3, avg_h, avg_w)), *imgs]  # Add empty top left square.
        grid = make_grid(imgs, nrow=len(style_files) + 1, padding=16, pad_value=1)
        dataset.save(grid, output_dir.joinpath(f'table.{ext}'))
