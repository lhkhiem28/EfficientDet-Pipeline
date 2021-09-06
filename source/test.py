
import argparse
import yaml
import json
import torch
import albumentations as A, albumentations.pytorch as AT
from data import ObjectDetectionDataset
from models import create_model_from_effnet_backbone
from engines import test_fn
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
model.load_state_dict(torch.load("{}/{}.pt".format(hyps_file["ckps_path"], model.config.name), map_location=torch.device(hyps_file["device"])))

with open(data_file["test_annotations_path"]) as f:
    test_annotations = json.load(f)

test_transform = A.Compose(
    [
        A.Resize(*tuple(model.config.image_size)), 
        A.Normalize(), 
        AT.transforms.ToTensorV2(), 
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["clses"])
)

test_loader = torch.utils.data.DataLoader(
    dataset=ObjectDetectionDataset(
        annotations=test_annotations, 
        class_names=data_file["class_names"], 
        image_folder=data_file["test_images_path"], image_transform=test_transform
    ), 
    collate_fn=collate_fn, 
    num_workers=hyps_file["num_workers"], 
    batch_size=hyps_file["batch_size"]*2, 
)

test_fn(
    test_loader, model, torch.device(hyps_file["device"]), 
    conf_threshold=float(hyps_file["conf_threshold"]), 
    iou_threshold=float(hyps_file["iou_threshold"]), 
)