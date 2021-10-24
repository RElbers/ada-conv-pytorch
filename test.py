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
    parser = ArgumentParser()
    parser.add_argument('--content-dir', type=str, default='./test_images/content')
    parser.add_argument('--style-dir', type=str, default='./test_images/style')
    parser.add_argument('--output-dir', type=str, default='./test_images/output')
    parser.add_argument('--model', type=str, default='./model.ckpt')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    content_files = dataset.files_in(args['content_dir'])
    style_files = dataset.files_in(args['style_dir'])
    output_dir = Path(args['output_dir'])
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    model = LightningModel.load_from_checkpoint(checkpoint_path=args['model'])
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(content_files) * len(style_files))

        imgs = [dataset.style_transforms()(dataset.load(f)) for f in style_files]
        for i, content in enumerate(content_files):
            imgs.append(dataset.content_transforms()(dataset.load(content)))

            for j, style in enumerate(style_files):
                output = stylize_image(model, content, style)
                dataset.save(output, output_dir.joinpath(rf'{i:02}--{j:02}.png'))
                imgs.append(output)
                pbar.update(1)

        # Make all same size for table
        avg_h = int(sum([img.size(1) for img in imgs]) / len(imgs))
        avg_w = int(sum([img.size(2) for img in imgs]) / len(imgs))
        imgs = [resize(img, [avg_h, avg_w]) for img in imgs]
        imgs = [torch.ones((3, avg_h, avg_w)), *imgs]
        grid = make_grid(imgs, nrow=len(style_files) + 1, padding=16, pad_value=1)
        dataset.save(grid, output_dir.joinpath('table.png'))
