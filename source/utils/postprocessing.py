
import torch
import torchvision

def filter_image_detections(
    image_detections
    , conf_threshold
    , iou_threshold
):
    image_bboxes, image_scores = image_detections[:, :4], image_detections[:, 4]
    image_clses = image_detections[:, 5]

    indexes = torch.where(image_scores >= conf_threshold)[0].cpu().numpy()
    image_bboxes, image_scores = image_bboxes[indexes], image_scores[indexes]
    image_clses = image_clses[indexes]

    indexes = torchvision.ops.nms(
        image_bboxes, image_scores
        , iou_threshold
    )
    image_bboxes, image_scores = image_bboxes[indexes].cpu().numpy(), image_scores[indexes].cpu().numpy()
    image_clses = image_clses[indexes].cpu().numpy()

    image_detections = []
    image_detections.extend([image_bboxes, image_scores])
    image_detections.extend([image_clses])

    return image_detections