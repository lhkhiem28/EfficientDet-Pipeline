
import itertools
import numpy as np
from utils.postprocessing import filter_image_detections
from objdetecteval.metrics import coco_metrics

def get_coco_stats(
    running_detections, 
    running_targets, 
    conf_threshold, 
    iou_threshold, 
):
    running_detections = [
        [
            filter_image_detections(
                image_detections
                , conf_threshold
                , iou_threshold
            ) for image_detections in batch_detections
        ] for batch_detections in running_detections
    ]
    running_detections = list(itertools.chain(*running_detections))
    running_detections_bboxes, running_detections_scores = [[batch_detections.tolist() for batch_detections in np.array(running_detections)[:, i].tolist()] for i in range(2)]
    running_detections_clses = [batch_detections.tolist() for batch_detections in np.array(running_detections)[:, 2].tolist()]
    running_detections = []
    running_detections.extend([list(range(len(running_detections_clses)))])
    running_detections.extend([running_detections_bboxes, running_detections_scores])
    running_detections.extend([running_detections_clses])
    running_targets_bboxes = [[batch_targets_bboxes.numpy().tolist() for batch_targets_bboxes in batch_targets["bbox"]] for batch_targets in running_targets]
    running_targets_clses = [[batch_targets_clses.numpy().tolist() for batch_targets_clses in batch_targets["cls"]] for batch_targets in running_targets]
    running_targets_bboxes = list(itertools.chain(*running_targets_bboxes))
    running_targets_clses = list(itertools.chain(*running_targets_clses))
    running_targets = []
    running_targets.extend([list(range(len(running_targets_clses)))])
    running_targets.extend([running_targets_bboxes])
    running_targets.extend([running_targets_clses])

    coco_stats = coco_metrics.get_coco_stats(
        running_detections, 
        running_targets, 
    )["All"]

    return coco_stats