#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted form MONAI Tutorial: https://github.com/Project-MONAI/tutorials/tree/main/2d_segmentation/torch
"""

import argparse
import os

join = os.path.join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

from baseline.models.unetr2d import UNETR2D

print("Successfully imported all requirements!")


def main():
    parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
    # Dataset parameters
    parser.add_argument(
        "--dataset",
        default="lucchi",
        choices=["lucchi","livecella","livecell", "mitoem", "kavsir", "nips", "cellpose"],
        type=str,
        help="select dataset: lucchi, mitoem, kavsir, nips, cellpose"
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        type=str,
        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"],
        help="select device: cuda:0, cuda:1, cuda:2, cuda:3, cpu",
    )
    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "--work_dir", default="./baseline/work_dir", help="path where to save models and logs"
    )
    parser.add_argument("--seed", default=43, type=int)
    # parser.add_argument("--resume", default=False, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=1, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name", 
        default="dymamba",
        choices=["dymamba"], 
    )
    parser.add_argument("--num_class", default=2, type=int, help="segmentation classes")
    parser.add_argument(
        "--input_size", default=512, type=int, help="segmentation classes"
    )
    # Training parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--max_epochs", default=300, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--epoch_tolerance", default=100, type=int)
    parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")

    parser.add_argument("--patch_num", type=int, default=4, help="patch number for swinmamba wa")

    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset.lower() == "lucchi":
        args.data_path = "/data/Lucchi"
    if args.dataset.lower() == "mitoem":
        args.data_path = "/data/MitoEMR"
    if args.dataset.lower() == "kavsir":
        args.data_path = "/data/Kvasir-SEG"
    if args.dataset.lower() == "nips":
        args.data_path = "/data/nips"
    if args.dataset.lower() == "cellpose":
        args.data_path = "/data/cellpose"
    if args.dataset.lower() == "livecell":
        args.data_path = "/data/livecell"
    
    print(f"Using dataset: {args.dataset}, data path: {args.data_path}")

    monai.config.print_config()

    #%% set training/validation split
    np.random.seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_path = join(args.work_dir, args.model_name + "_" + args.dataset + "_" + run_id)
    os.makedirs(model_path, exist_ok=True)
    
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    img_path = join(args.data_path, "images")
    gt_path = join(args.data_path, "labels")

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split(".")[0] + "_label.png" for img_name in img_names]
    img_num = len(img_names)
    val_frac = 0.1
    indices = np.arange(img_num)
    np.random.shuffle(indices)
    val_split = int(img_num * val_frac)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in train_indices
    ]
    val_files = [
        {"img": join(img_path, img_names[i]), "label": join(gt_path, gt_names[i])}
        for i in val_indices
    ]
    print(
        f"training image num: {len(train_files)}, validation image num: {len(val_files)}"
    )
    #%% define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["img", "label"], reader=PILReader, dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
            RandAxisFlipd(keys=["img", "label"], prob=0.5),
            RandRotate90d(keys=["img", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["img"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["img", "label"],
                prob=0.15,
                min_zoom=0.8,
                max_zoom=1.5,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"], reader=PILReader, dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    #% define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4)
    check_data = monai.utils.misc.first(check_loader)
    print(
        "sanity check:",
        check_data["img"].shape,
        torch.max(check_data["img"]),
        check_data["label"].shape,
        torch.max(check_data["label"]),
    )

    #%% create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    post_pred = Compose(
        [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
    )
    post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
    # create UNet, DiceLoss and Adam optimizer
    
    if args.model_name.lower() == "dymamba":
        print('prepare to load swinmamba success')
        from vmamba.DynamicM import DyMamba
        vss_args = dict(
            in_chans=3, 
            patch_size=4, 
            depths=[2,2,4,2], 
            dims=96, # keep with first of features_per_stage
            drop_path_rate=0.2)
        decoder_args = dict(
            num_classes=args.num_class,
            deep_supervision=False, 
            features_per_stage=[96, 192, 384, 768],  
            #features_per_stage=[64, 128, 256, 512],  
            drop_path_rate=0.2,
            d_state=16)
        model = DyMamba(vss_args, decoder_args).to(device)
        print('loading dymamba success')


    loss_function = monai.losses.DiceCELoss(softmax=True)
    initial_lr = args.initial_lr
    optimizer = torch.optim.AdamW(model.parameters(), initial_lr)

    # start a typical PyTorch training
    max_epochs = args.max_epochs
    epoch_tolerance = args.epoch_tolerance
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(model_path)
    for epoch in range(1, max_epochs):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader, 1):
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(
                device
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            labels_onehot = monai.networks.one_hot(
                labels, args.num_class
            )  # (b,cls,256,256)
            loss = loss_function(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss_values,
        }

        if epoch > 20 and epoch % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data[
                        "label"
                    ].to(device)
                    val_labels_onehot = monai.networks.one_hot(
                        val_labels, args.num_class
                    )
                    roi_size = (256, 256)
                    if args.model_name=="swinunet" or args.model_name=="unetr":
                        roi_size = (args.input_size, args.input_size)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model
                    )
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_onehot = [
                        post_gt(i) for i in decollate_batch(val_labels_onehot)
                    ]
                    # compute metric for current iteration
                    print(
                        os.path.basename(
                            val_data["img_meta_dict"]["filename_or_obj"][0]
                        ),
                        dice_metric(y_pred=val_outputs, y=val_labels_onehot),
                    )

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")
            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(
                    f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch=}, {best_metric_epoch=}"
                )
                break

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()
    torch.save(checkpoint, join(model_path, "final_model.pth"))
    np.savez_compressed(
        join(model_path, "train_log.npz"),
        val_dice=metric_values,
        epoch_loss=epoch_loss_values,
    )


if __name__ == "__main__":
    main()
