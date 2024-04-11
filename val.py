import argparse
import re
import sys

import cv2
import numpy as np
import torch
import ttach as tta
from preprocessing import SpaceShiftNdviDataset, do_nothing, transform_vvvh

if True:
    sys.path.append("pytorch-CycleGAN-and-pix2pix")
    from models import networks

parser = argparse.ArgumentParser()
parser.add_argument("--pt", type=str, required=True)
parser.add_argument("--no_tta", action='store_true')
args, remaining = parser.parse_known_args()

pt_filepath = args.pt
print(pt_filepath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_types = re.search("-it([0-9a-z_]+)", pt_filepath)[1]
output_channel = int(re.search("-oc([0-9]+)", pt_filepath)[1])
gen_name = re.search("-gn([a-z0-9_]+)", pt_filepath)[1]
image_size = int(re.search("-is([0-9]+)", pt_filepath)[1])
seed = int(re.search("-sd(\\d+)", pt_filepath)[1])

gen = networks.define_G(input_nc=len(input_types.split("_")),
                        output_nc=output_channel,
                        ngf=64, netG=gen_name,
                        use_dropout=True, gpu_ids=[0])
gen.load_state_dict(torch.load(pt_filepath))

if args.no_tta:
    print("NO TTA")


def apply_tta(model, x):
    x_torch = torch.from_numpy(x).unsqueeze(0).to(device)

    if args.no_tta:
        transforms = tta.Compose([])
    else:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )

    ys = None
    for transformer in transforms:
        augmented_image = transformer.augment_image(x_torch)
        with torch.no_grad():
            y = model(augmented_image)
        deaug_y = transformer.deaugment_mask(y)
        y_numpy = deaug_y.squeeze(0).cpu().detach().numpy().copy()
        ys = y_numpy if ys is None else \
            np.concatenate([ys, y_numpy], axis=0)

    return np.mean(ys, axis=0)


val_dataset = SpaceShiftNdviDataset(
    split="003",
    vvvh_original=True,
    vvvh_crop=False,
    ndvi_original=False,
    ndvi_crop=True,
    transform_x=lambda x, dems:
        transform_vvvh(x, dems=dems, img_size=image_size,
                       input_types=input_types),
    transform_y=do_nothing,
    seed=seed,
)

sum_pixel_wise_mse = 0
for i, (vvvh, ndvi) in enumerate(val_dataset):
    # print(vvvh.shape)

    fake_ndvi = apply_tta(gen, vvvh)
    p = fake_ndvi * 2 - 1
    gt = ndvi.squeeze()

    height, width = gt.shape
    p = cv2.resize(p, dsize=(width, height))

    p[gt == -100] = np.nan
    gt[gt == -100] = np.nan
    # print("gt min: %.3f, max: %.3f" % (np.nanmin(gt), np.nanmax(gt)))
    # print("p  min: %.3f, max: %.3f" % (np.nanmin(p), np.nanmax(p)))
    gt[np.isnan(gt)] = -100
    p[gt == -100] = -100

    pixel_wise_mse = np.sum((p - gt)**2) / (gt.shape[0] * gt.shape[1])
    sum_pixel_wise_mse += pixel_wise_mse
    # print("pixel_wise_mse:", pixel_wise_mse)

# print()
print("p mean pixel_wise_mse:", sum_pixel_wise_mse / len(val_dataset))
