
import torch

def forward_device(tensors, device):
    tensor = tensors[0].to(device)

    targets = tensors[1]
    for k in [k for k in list(targets.keys()) if "img" in k]:
        targets[k] = targets[k].to(device)
    targets["bbox"] = [bbox.to(device) for bbox in targets["bbox"]]
    targets["cls"] = [cls.to(device) for cls in targets["cls"]]

    return tensor, targets

def collate_fn(batch):
    images = torch.stack(tuple([item["image"] for item in batch]))
    batch_size, _, img_width, img_height = images.shape

    bboxes = [item["bboxes"] for item in batch]
    clses = [item["clses"] for item in batch]
    targets = {
        "img_size": torch.stack([torch.tensor([img_width, img_height])]*batch_size), 
        "img_scale": torch.ones(batch_size), 
        "bbox": bboxes, 
        "cls": clses, 
    }

    return images, targets