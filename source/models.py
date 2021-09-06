
import json
import effdet
from effdet.config.model_config import efficientdet_model_param_dict

def create_model_from_effnet_backbone(
    backbone_name, num_classes
    , image_size
    , verbose=True
):
    """
        The supported effnet backbones include:
            + the official tf_efficientnet_b* (0, 1, 2, 3, 4, 5, 6, 7)
            + the official tf_efficientnetv2_* (s, m, l)
    """
    name = backbone_name.replace("net", "det").replace("_b", "_d")
    pretrained = True
    if name not in list(efficientdet_model_param_dict.keys()):
        efficientdet_model_param_dict[name] = dict(
            name=name, 
            backbone_name=backbone_name, backbone_args=dict(drop_path_rate=0.2)
        )
        pretrained = False

    config = effdet.get_efficientdet_config(name)
    config["image_size"] = image_size
    efficientdet_model_param_dict.pop(name)

    if verbose:
        if pretrained:
            print("Loaded pretrained model.")
        else:
            print("Initialized model and loaded pretrained backbone.")
    model = effdet.create_model_from_config(
        config=config, num_classes=num_classes, 
        pretrained=pretrained
    )
    bench = effdet.DetBenchTrain(model, config)

    return bench

def save_basic_config(config, config_path):
    basic_config = {}
    basic_config["backbone_name"], basic_config["num_classes"] = config["backbone_name"], config["num_classes"]
    basic_config["image_size"] = list(config["image_size"])

    with open(config_path, "w") as f:
        json.dump(basic_config, f)