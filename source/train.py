
import argparse
import yaml
import json
import torch
import albumentations as A, albumentations.pytorch as AT
from data import ObjectDetectionDataset
from models import create_model_from_effnet_backbone, save_basic_config
from engines import train_fn
from utils.training import collate_fn

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--hyps_file", type=str)
args = parser.parse_args()

data_file = yaml.load(open(args.data_file), Loader=yaml.FullLoader)
hyps_file = yaml.load(open(args.hyps_file), Loader=yaml.FullLoader)

model = create_model_from_effnet_backbone(
    backbone_name=hyps_file["backbone_name"], num_classes=data_file["num_classes"]
    , image_size=[hyps_file["image_size"], hyps_file["image_size"]]
)
save_basic_config(model.config, "{}/{}.json".format(hyps_file["ckps_path"], model.config.name))

with open(data_file["train_annotations_path"]) as f:
    train_annotations = json.load(f)
with open(data_file["val_annotations_path"]) as f:
    val_annotations = json.load(f)

train_transform = A.Compose(
    [
        A.Resize(*tuple(model.config.image_size)), 
        A.HorizontalFlip(), 
        A.RandomBrightnessContrast(), 
        A.Normalize(), 
        AT.transforms.ToTensorV2(), 
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["clses"])
)
val_transform = A.Compose(
    [
        A.Resize(*tuple(model.config.image_size)), 
        A.Normalize(), 
        AT.transforms.ToTensorV2(), 
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["clses"])
)

train_loader = torch.utils.data.DataLoader(
    dataset=ObjectDetectionDataset(
        annotations=train_annotations, 
        class_names=data_file["class_names"], 
        image_folder=data_file["train_images_path"], image_transform=train_transform
    ), 
    collate_fn=collate_fn, 
    num_workers=hyps_file["num_workers"], 
    batch_size=hyps_file["batch_size"], 
    shuffle=True, 
)
val_loader = torch.utils.data.DataLoader(
    dataset=ObjectDetectionDataset(
        annotations=val_annotations, 
        class_names=data_file["class_names"], 
        image_folder=data_file["val_images_path"], image_transform=val_transform
    ), 
    collate_fn=collate_fn, 
    num_workers=hyps_file["num_workers"], 
    batch_size=hyps_file["batch_size"]*2, 
)

loaders = {
    "train": train_loader, 
    "val": val_loader, 
}

optimizer = torch.optim.AdamW(model.parameters(), lr=float(hyps_file["lr"]))
train_fn(
    loaders, model, torch.device(hyps_file["device"]), 
    optimizer, 
    epochs=hyps_file["epochs"], 
    conf_threshold=float(hyps_file["conf_threshold"]), 
    iou_threshold=float(hyps_file["iou_threshold"]), 
    monitor="val_ap", 
    ckps_path=hyps_file["ckps_path"], 
)