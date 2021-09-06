
import json
import cv2
import numpy as np
import torch
import albumentations as A, albumentations.pytorch as AT
from models import create_model_from_effnet_backbone
from effdet.utils import create_dummy_target
from utils.training import forward_device
from utils.postprocessing import filter_image_detections
from matplotlib import pyplot as plt

class ObjectDetector():
    def __init__(self, basic_config_path):
        with open(basic_config_path) as f:
            self.basic_config = json.load(f)

        self.model = create_model_from_effnet_backbone(
            backbone_name=self.basic_config["backbone_name"], num_classes=self.basic_config["num_classes"]
            , image_size=self.basic_config["image_size"]
            , verbose=False
        )
        self.image_width, self.image_height = tuple(self.model.config.image_size)
        self.image_transform = A.Compose([
            A.Resize(*tuple(self.model.config.image_size)), 
            A.Normalize(), 
            AT.transforms.ToTensorV2(), 
        ])

        self.class_colors = [tuple([np.random.randint(0, 255) for _ in range(3)]) for _ in range(self.basic_config["num_classes"])]

    def from_pretrained(self, ckp_path, device=torch.device("cpu")):
        self.device = device
        self.model.load_state_dict(torch.load(ckp_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Loaded pretrained model.")

        return self

    def predict_image(
        self, 
        image_path
        , conf_threshold
        , iou_threshold
        , show_image=False
    ):
        image = cv2.cvtColor(cv2.imread(image_path), code=cv2.COLOR_BGR2RGB)
        image_width, image_height = image.shape[:2][::-1]

        image, dummy_target = self.image_transform(image=image)["image"].unsqueeze(0), create_dummy_target([self.image_width, self.image_height])
        image, dummy_target = forward_device([image, dummy_target], self.device)
        with torch.no_grad():
            image_output = self.model(image, dummy_target)
            image_detections = image_output["detections"].detach()[0]

        image_detections = filter_image_detections(
            image_detections
            , conf_threshold
            , iou_threshold
        )
        image_detections_bboxes, image_detections_scores = image_detections[:2]
        image_detections_clses = image_detections[2]

        image_detections_bboxes = image_detections_bboxes*[image_width/self.image_width, image_height/self.image_height, image_width/self.image_width, image_height/self.image_height]
        if show_image:
            image = cv2.cvtColor(cv2.imread(image_path), code=cv2.COLOR_BGR2RGB)
            for (i, image_detections_bbox) in enumerate(image_detections_bboxes):
                image = cv2.rectangle(
                    image, 
                    (int(image_detections_bbox[0]), int(image_detections_bbox[1])), (int(image_detections_bbox[2]), int(image_detections_bbox[3])), 
                    color=self.class_colors[int(image_detections_clses[i]-1)], 
                    thickness=3
                )
            plt.imshow(image)
            plt.axis("off"), plt.show()

        return_list = []
        return_list.extend([image_detections_bboxes, image_detections_scores])
        return_list.extend([image_detections_clses])

        return return_list