# **EfficientDet-Pipeline**

## 👋 **Introduction**
A Pytorch full pipeline (training and inference) for [EfficientDet](https://arxiv.org/abs/1911.09070). Easily adapt to custom data.\
This pipeline is based on the [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch) implementation of rwightman.

## 🚀 **Quick Start**
How to train on your custom data? For example, I will use the [VinDrCXR](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) dataset.

### **1. Create dataset.yaml**
[data/VinDrCXR/vindrcxr.yaml](data/VinDrCXR/vindrcxr.yaml), shown below, is the dataset config file that defines:
- absolute paths to `train` / `val` / `test` image directories and annotation files.
- the number of classes `num_classes`.
- a list of class names.
```bash

train_annotations_path: ../data/VinDrCXR/annotations/full_union_train.json
val_annotations_path: ../data/VinDrCXR/annotations/full_union_val.json
train_images_path: ../data/VinDrCXR/images/train
val_images_path: ../data/VinDrCXR/images/val
test_images_path: ../data/VinDrCXR/images/test

num_classes: 14
class_names: [
  "Aortic enlargement", 
  "Atelectasis", 
  "Calcification", 
  "Cardiomegaly", 
  "Consolidation", 
  "ILD", 
  "Infiltration", 
  "Lung Opacity", 
  "Nodule/Mass", 
  "Other lesion", 
  "Pleural effusion", 
  "Pleural thickening", 
  "Pneumothorax", 
  "Pulmonary fibrosis", 
]
```