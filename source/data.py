
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        annotations, 
        class_names, 
        image_folder, image_transform=None
    ):
        self.annotations = annotations
        self.image_names = list(annotations.keys())
        self.image_folder, self.image_transform = image_folder, image_transform

        self.class_names = class_names
        self.class_colors = [tuple([np.random.randint(0, 255) for _ in range(3)]) for _ in range(len(self.class_names))]

    def show_image(self, idx):
        item = self.__getitem__(idx)
        image = item["image"]
        bboxes = item["bboxes"]
        clses = item["clses"]

        for (i, bbox) in enumerate(bboxes):
            image = cv2.rectangle(
                image, 
                (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                color=self.class_colors[int(clses[i]-1)], 
                thickness=3
            )
        plt.imshow(image)
        plt.axis("off"), plt.show()

    def __len__(self):
        dataset_len = len(self.image_names)

        return dataset_len

    def __getitem__(self, idx):
        image_annotations = self.annotations[self.image_names[idx]]

        item = {
            "image": cv2.cvtColor(cv2.imread("{}/{}.png".format(self.image_folder, self.image_names[idx])), code=cv2.COLOR_BGR2RGB), 
            "bboxes": np.array([anno["bbox"] for anno in image_annotations]), 
            "clses": np.array([anno["cls"] for anno in image_annotations]), 
        }
        if self.image_transform is not None:
            item = self.image_transform(**item)
            item["bboxes"] = torch.as_tensor(item["bboxes"], dtype=torch.float32)
            item["clses"] = torch.as_tensor(item["clses"], dtype=torch.float32)

        return item