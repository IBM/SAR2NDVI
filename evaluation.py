import argparse
import os
import re
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import ttach as tta

from preprocessing import SpaceShiftNdviDataset, do_nothing, transform_vvvh

if True:
    sys.path.append("pytorch-CycleGAN-and-pix2pix")
    from models import networks


def apply_tta(model, x, transforms, device):
    x_torch = torch.from_numpy(x).unsqueeze(0).to(device)

    ys = None
    for transformer in transforms:
        augmented_image = transformer.augment_image(x_torch)
        with torch.no_grad():
            y = model(augmented_image)
        deaug_y = transformer.deaugment_mask(y)
        if torch.cuda.is_available():
            y_numpy = deaug_y.squeeze(0).cpu().detach().numpy().copy()
        else:
            y_numpy = deaug_y.squeeze(0).detach().numpy().copy()
        ys = y_numpy if ys is None else \
            np.concatenate([ys, y_numpy], axis=0)

    return np.mean(ys, axis=0)


def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/training-sd0-itvv01_vv995_vh01_vh_prvi_rfdi_rvi4s1_rvi_pvhvv_mvhvv-ataspect_rot20_flip-is128-oc1-gnunet_128-dnbasic-ne5000-ed5000-bs1-la50000-es1000-imbicubic/gen_best-ep05302-vl00906.pt")
    parser.add_argument("--output_folder", type=str,
                        default="output/test_tif/")
    args, remaining = parser.parse_known_args()

    model_path = args.model_path
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_types = re.search("-it([0-9a-z_]+)", model_path)[1]
    output_channel = int(re.search("-oc([0-9]+)", model_path)[1])
    gen_name = re.search("-gn([a-z0-9_]+)", model_path)[1]
    image_size = int(re.search("-is([0-9]+)", model_path)[1])
    seed = int(re.search("-sd(\\d+)", model_path)[1])

    gen = networks.define_G(input_nc=len(input_types.split("_")),
                            output_nc=output_channel,
                            ngf=64, netG=gen_name,
                            use_dropout=True,
                            norm="batch",
                            gpu_ids=([0] if torch.cuda.is_available() else []))

    if torch.cuda.is_available():
        gen.load_state_dict(torch.load(model_path, map_location=device))
    else:
        gen.load_state_dict(fix_key(
            torch.load(model_path, map_location=device)))

    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Multiply(factors=[0.9, 1, 1.1]),
    ])

    test_dataset = SpaceShiftNdviDataset(
        split="test",
        vvvh_original=True,
        vvvh_crop=False,
        transform_x=lambda x, **kwargs:
            transform_vvvh(x, img_size=image_size,
                           input_types=input_types, **kwargs),
        transform_y=do_nothing,
        seed=seed,
    )

    test_crop_dataset = SpaceShiftNdviDataset(
        split="test",
        vvvh_original=False,
        vvvh_crop=True,
        ndvi_original=False,
        ndvi_crop=True,
        transform_x=do_nothing,
        transform_y=do_nothing,
        seed=seed,
    )

    s2_001_dataset = SpaceShiftNdviDataset(
        split="sentinel2_adjusted_001",
        vvvh_original=True,
        vvvh_crop=False,
        ndvi_original=False,
        ndvi_crop=True,
        seed=seed,
    )

    for ((vvvh_resized, _), (vvvh_crop, _), vvvh_filepaths) \
            in zip(test_dataset, test_crop_dataset,
                   test_dataset.selected_paths_list_dict["x"][0]):

        date = vvvh_filepaths[0].split("/")[-1].replace("_VV.tif", "")
        print(date)

        _, s2_path = \
            s2_001_dataset.find_nearest_ndvi_path(vvvh_filepaths[0], "001")
        s2 = None if s2_path is None else s2_001_dataset.load_tif_image(
            s2_path)

        fake_ndvi = apply_tta(gen, vvvh_resized, transforms, device)
        p = fake_ndvi * 2 - 1
        vvvh = vvvh_crop

        _, height, width = vvvh.shape
        p = cv2.resize(p, dsize=(width, height))

        p[(vvvh[0] + vvvh[1]) == 0.0] = np.nan
        print("min: %.3f, max: %.3f" % (np.nanmin(p), np.nanmax(p)))
        p[(vvvh[0] + vvvh[1]) == 0.0] = -100

        # plt.subplot(321).imshow(vvvh[0], vmin=0, vmax=1)
        # plt.subplot(322).imshow(vvvh[1], vmin=0, vmax=1)
        # plt.subplot(323).imshow(p, vmin=-1, vmax=1)
        # if s2 is not None:
        #     plt.subplot(324).imshow(s2, vmin=-1, vmax=1)
        # plt.subplot(325).imshow(p, vmin=0, vmax=1)
        # if s2 is not None:
        #     plt.subplot(326).imshow(s2, vmin=0, vmax=1)
        # plt.show()

        cv2.imwrite("%s/daiki_%s_ndvi.tif" % (output_folder, date), p)
