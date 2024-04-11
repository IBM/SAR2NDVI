import argparse
import datetime
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchsummary
from preprocessing \
    import (SpaceShiftNdviDataset, do_nothing, transform_ndvi, transform_vvvh,
            augment_vvvh_ndvi)
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

if True:
    sys.path.append("pytorch-CycleGAN-and-pix2pix")
    from models import networks


def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find("Conv") != -1 or
                                 classname.find("Linear") != -1):
        init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def lambda_rule(epoch, args):
    lr_l = 1.0 - max(0, epoch - args.n_epoch) / float(args.n_epoch_decay + 1)
    return lr_l


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def validate(val_dataloader, model_G, args):
    mse_loss = nn.MSELoss()
    log_loss_mse = []

    for (vvvh, ndvi) in val_dataloader:
        vvvh = vvvh.to(device)
        fake_ndvi = model_G(vvvh)
        fake_ndvi_original_size = \
            (torch.nn.functional.interpolate(
                fake_ndvi.mean(axis=1).unsqueeze(0),
                ndvi.shape[2:], mode=args.interpolate_mode) * 2 - 1).cpu()
        fake_ndvi_original_size[ndvi == -100] = 0
        ndvi[ndvi == -100] = 0
        loss_mse = mse_loss(fake_ndvi_original_size, ndvi)
        log_loss_mse.append(loss_mse.item())

    return statistics.mean(log_loss_mse)


class EarlyStop:
    def __init__(self, save_model_paths,
                 patience=500, max_best_num=3, no_output=False):
        self.patience = patience
        self.counter = 0
        self.min_val_loss = None
        self.save_model_paths = save_model_paths
        self.saved_pathss = list()
        self.max_best_num = max_best_num
        self.no_output = no_output

    def __call__(self, i, val_loss, models):
        if self.min_val_loss is None or val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            if len(self.saved_pathss) == self.max_best_num:
                paths = self.saved_pathss.pop(0)
                for path in paths:
                    os.remove(path)
            saved_paths = \
                [path % (i + 1, ("%.05f" % val_loss).replace("0.", ""))
                 for path in self.save_model_paths]
            if not self.no_output:
                for model, path in zip(models, saved_paths):
                    torch.save(model.state_dict(), path)
            self.saved_pathss.append(saved_paths)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def remove_continuous_first_end(s, mark="_"):
    old_s = ""
    while old_s != s:
        old_s = s
        s = s.replace(mark + mark, mark)
    if len(s) > 0 and s[0] == mark:
        s = s[1:]
    if len(s) > 0 and s[-1] == mark:
        s = s[:-1]
    return s


if __name__ == "__main__":
    print(datetime.datetime.today())

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input_types", type=str,
        default="vv01_vv995_vh01_vh_prvi_rfdi_rvi4s1_rvi_pvhvv_mvhvv")
    parser.add_argument("--aug_types", type=str,
                        default="aspect_rot20_flip")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--output_channel", type=int, default=1)
    parser.add_argument("--gen_name", type=str, default="unet_128")
    parser.add_argument("--dis_name", type=str, default="basic")
    parser.add_argument("--n_epoch", type=int, default=5000)
    parser.add_argument("--n_epoch_decay", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lambda_l1", type=float, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=1000)
    parser.add_argument("--interpolate_mode", type=str, default="bicubic")
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--no_output", action="store_true")
    parser.add_argument("--no_gan_loss", action="store_true")
    args, remaining = parser.parse_known_args()

    args.input_types = remove_continuous_first_end(args.input_types, "_")
    if args.input_types == "":
        print("No input_types")
        exit(1)

    options_str = \
        "-sd%d-it%s-at%s-is%d-oc%d-gn%s-dn%s-ne%d-ed%d-bs%d-la%s-es%d-im%s%s" \
        % (
            args.seed,
            args.input_types,
            args.aug_types,
            args.image_size,
            args.output_channel,
            args.gen_name,
            args.dis_name,
            args.n_epoch,
            args.n_epoch_decay,
            args.batch_size,
            ("%.2f" % args.lambda_l1).replace(".", ""),
            args.early_stop_patience,
            args.interpolate_mode,
            ("_ngl" if args.no_gan_loss else "")
        )

    executing_filename = \
        os.path.abspath(__file__).split("/")[-1].replace(".py", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = "output/" + executing_filename + options_str + "/"
    output_csv = "output/output.csv"

    if not args.no_output:
        os.makedirs(output_dir, exist_ok=True)
    print("output_dir: " + output_dir)

    if torch.cuda.device_count() > 0:
        print("GPU mode (%d GPUs)" % torch.cuda.device_count())
    else:
        print("CPU mode")
    gpu_ids = [i for i in range(torch.cuda.device_count())]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train1_dataset = SpaceShiftNdviDataset(
        split="sentinel2_adjusted",
        vvvh_original=True,
        vvvh_crop=False,
        transform_x=lambda x, **kwargs:
            transform_vvvh(x, input_types=args.input_types, **kwargs),
        transform_y=lambda x: transform_ndvi(x, n_channel=1),
        seed=args.seed,
        augmentation=lambda x, y:
            augment_vvvh_ndvi(x, y, img_size=args.image_size,
                              aug_types=args.aug_types)
    )

    train2_dataset = SpaceShiftNdviDataset(
        split="002",
        vvvh_original=True,
        vvvh_crop=False,
        transform_x=lambda x, **kwargs:
            transform_vvvh(x, input_types=args.input_types, **kwargs),
        transform_y=lambda x: transform_ndvi(x, n_channel=1),
        seed=args.seed,
        augmentation=lambda x, y:
            augment_vvvh_ndvi(x, y, img_size=args.image_size,
                              aug_types=args.aug_types)
    )

    val_dataset = SpaceShiftNdviDataset(
        split="003",
        vvvh_original=True,
        vvvh_crop=False,
        ndvi_original=False,
        ndvi_crop=True,
        transform_x=lambda x, **kwargs:
            transform_vvvh(x, img_size=args.image_size,
                           input_types=args.input_types, **kwargs),
        transform_y=do_nothing,
        seed=args.seed,
    )

    torch.backends.cudnn.benchmark = True

    train1_dataloader = DataLoader(
        train1_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device == torch.device("cpu") else 1,
        pin_memory=True
    )

    train2_dataloader = DataLoader(
        train2_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if device == torch.device("cpu") else 1,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device == torch.device("cpu") else 1,
        pin_memory=True
    )

    input_channel = len(args.input_types.split("_"))

    gen = networks.define_G(input_nc=input_channel,
                            output_nc=args.output_channel,
                            ngf=64, netG=args.gen_name,
                            use_dropout=True, gpu_ids=gpu_ids)
    dis = networks.define_D(input_nc=input_channel + 1, ndf=64,
                            netD=args.dis_name, n_layers_D=3,
                            gpu_ids=gpu_ids)

    torchsummary.summary(
        gen, input_size=(input_channel,
                         args.image_size, args.image_size), device="cpu")
    torchsummary.summary(
        dis, input_size=(input_channel + 1,
                         args.image_size, args.image_size), device="cpu")

    optimizer_gen = torch.optim.Adam(gen.parameters(),
                                     lr=0.0002, betas=(0.5, 0.999))
    optimizer_dis = torch.optim.Adam(dis.parameters(),
                                     lr=0.0002, betas=(0.5, 0.999))

    scheduler_gen = lr_scheduler.LambdaLR(
        optimizer_gen, lr_lambda=lambda x: lambda_rule(x, args))
    scheduler_dis = lr_scheduler.LambdaLR(
        optimizer_dis, lr_lambda=lambda x: lambda_rule(x, args))

    ones_for_dis_output = torch.ones(args.batch_size, 1, 14, 14).to(device)
    zeros_for_dis_output = torch.zeros(args.batch_size, 1, 14, 14).to(device)

    gan_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    result = dict()
    result["log_loss_gen_sum"] = list()
    result["log_loss_gen_gan"] = list()
    result["log_loss_gen_mae"] = list()
    result["log_loss_dis"] = list()
    result["val_loss"] = list()

    early_stop = EarlyStop(
        save_model_paths=[output_dir + "gen_best-ep%05d-vl%s.pt",
                          output_dir + "dis_best-ep%05d-vl%s.pt"],
        patience=args.early_stop_patience,
        no_output=args.no_output)

    start_time = time.time()

    for i in range(args.n_epoch + args.n_epoch_decay):
        log_loss_gen_sum, log_loss_gen_gan, \
            log_loss_gen_mae, log_loss_dis = [], [], [], []

        for (vvvh, ndvi) in (train1_dataloader
                             if i < args.n_epoch else train2_dataloader):
            ndvi, vvvh = ndvi.to(device), vvvh.to(device)

            fake_ndvi = gen(vvvh)

            set_requires_grad(dis, True)
            optimizer_dis.zero_grad()

            dis_fake = dis(torch.cat([fake_ndvi, vvvh], dim=1).detach())
            loss_dis_fake = gan_loss(dis_fake, zeros_for_dis_output)

            dis_real = dis(torch.cat([ndvi, vvvh], dim=1))
            loss_dis_real = gan_loss(dis_real, ones_for_dis_output)

            loss_dis = (loss_dis_fake + loss_dis_real) * 0.5
            log_loss_dis.append(loss_dis.item())
            loss_dis.backward()

            optimizer_dis.step()
            set_requires_grad(dis, False)

            optimizer_gen.zero_grad()
            dis_fake = dis(torch.cat([fake_ndvi, vvvh], dim=1))
            loss_gen_gan = gan_loss(dis_fake, ones_for_dis_output)
            loss_gen_mae = mae_loss(fake_ndvi, ndvi)
            if args.no_gan_loss:
                loss_gen_sum = args.lambda_l1 * loss_gen_mae
            else:
                loss_gen_sum = loss_gen_gan + args.lambda_l1 * loss_gen_mae
            log_loss_gen_gan.append(loss_gen_gan.item())
            log_loss_gen_mae.append(loss_gen_mae.item())
            log_loss_gen_sum.append(loss_gen_sum.item())
            loss_gen_sum.backward()

            optimizer_gen.step()

        scheduler_dis.step()
        scheduler_gen.step()

        # save and print loss
        result["log_loss_gen_sum"].append(statistics.mean(log_loss_gen_sum))
        result["log_loss_gen_gan"].append(statistics.mean(log_loss_gen_gan))
        result["log_loss_gen_mae"].append(statistics.mean(log_loss_gen_mae))
        result["log_loss_dis"].append(statistics.mean(log_loss_dis))

        val_loss = validate(val_dataloader, gen, args)
        result["val_loss"].append(val_loss)

        if i == args.n_epoch:
            early_stop.counter = 0
        will_early_stop = early_stop(i, val_loss, [gen, dis])

        elapsed_time = time.time() - start_time

        print(
            "Ep %05d: Loss G %.4f (GAN %.3f, L1 %.3f), " %
            (i + 1,
             result["log_loss_gen_sum"][-1],
             result["log_loss_gen_gan"][-1],
             result["log_loss_gen_mae"][-1]) +
            "Loss D %.4f, Val MSE %.4f (ES %03d), LR: %.6f, " %
            (result["log_loss_dis"][-1],
             result["val_loss"][-1],
             early_stop.counter,
             scheduler_gen.get_last_lr()[0]) +
            "Elapsed %.2fh, ETA %.2fh" %
            (elapsed_time / 3600.0,
             ((args.n_epoch + args.n_epoch_decay) * elapsed_time /
              (i + 1) - elapsed_time) / 3600.0)
        )

        if not args.no_output:
            if (i + 1) % args.save_freq == 0:
                for model, path in zip(
                    [gen, dis], [output_dir + "gen-ep%05d.pt" % (i + 1),
                                 output_dir + "dis-ep%05d.pt" % (i + 1)]):
                    torch.save(model.state_dict(), path)

            with open(os.path.join(output_dir, "result.json"), "w") as fp:
                json.dump(result, fp, indent=2)

        if i >= args.n_epoch and will_early_stop:
            print("Early Stopped!")
            break

    if not args.no_output:
        with open(output_csv, "a") as fp:
            fp.writelines(executing_filename + "," +
                          options_str + "," +
                          str(early_stop.min_val_loss) + "," +
                          str(i + 1) + "\n")

    print(datetime.datetime.today())
