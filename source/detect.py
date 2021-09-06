
import argparse
import yaml
import glob
import tqdm
import torch
from api import ObjectDetector

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str)
parser.add_argument("--basic_config_path", type=str)
parser.add_argument("--ckp_path", type=str), parser.add_argument("--device", type=str)
parser.add_argument("--conf_threshold", type=float)
parser.add_argument("--iou_threshold", type=float)
parser.add_argument("--detect_path", type=str)
args = parser.parse_args()

detector = ObjectDetector(args.basic_config_path)
detector.from_pretrained(args.ckp_path, torch.device(args.device))

data_file = yaml.load(open(args.data_file), Loader=yaml.FullLoader)
image_paths = glob.glob("{}/*.png".format(data_file["test_images_path"]))
for image_path in tqdm.tqdm(image_paths):
    image_detections = detector.predict_image(
        image_path
        , args.conf_threshold
        , args.iou_threshold
    )
    image_detections_bboxes, image_detections_scores = image_detections[:2]
    image_detections_clses = image_detections[2]

    with open("{}/{}".format(args.detect_path, image_path.split("/")[-1].replace("png", "txt")), "w") as f:
        if len(image_detections_clses) > 0:
            for i in range(len(image_detections_clses)):
                f.write("{} {} {} {} {} ".format(*image_detections_bboxes[i], image_detections_scores[i]))
                f.write("{}".format(int(image_detections_clses[i])))
                f.write("\n")
        else:
            f.write("{} {} {} {} {} ".format(*[0.0, 0.0, 1.0, 1.0], 1.0))
            f.write("{}".format(int(-1.0)))
            f.write("\n")