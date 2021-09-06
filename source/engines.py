
import tqdm
import torch
from utils.training import forward_device
from metrics import get_coco_stats

def train_fn(
    loaders, model, device, 
    optimizer, 
    epochs, 
    conf_threshold, 
    iou_threshold, 
    monitor, 
    ckps_path, 
):
    print("Number of Epochs: {}\n".format(epochs))
    best_loss = 1e+8
    best_ap = 1e-8

    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, epochs + 1):
        print("epoch {:2}/{:2}".format(epoch, epochs) + "\n" + "-"*16)

        model.train()
        running_loss = 0.0
        for batch_images, batch_targets in tqdm.tqdm(loaders["train"]):
            batch_targets["bbox"] = [batch_targets_bboxes[:, [1, 0, 3, 2]] for batch_targets_bboxes in batch_targets["bbox"]]
            batch_images, batch_targets = forward_device([batch_images, batch_targets], device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                batch_outputs = model(batch_images, batch_targets)
                batch_loss = batch_outputs["loss"]

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss = running_loss + batch_loss.item()*batch_images.size(0)

        epoch_loss = running_loss/len(loaders["train"].dataset)
        print("{}-loss: {:.4f}".format("train", epoch_loss))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_detections, running_targets = [], []
            for batch_images, batch_targets in tqdm.tqdm(loaders["val"]):
                batch_targets["bbox"] = [batch_targets_bboxes[:, [1, 0, 3, 2]] for batch_targets_bboxes in batch_targets["bbox"]]
                batch_images, batch_targets = forward_device([batch_images, batch_targets], device)

                batch_outputs = model(batch_images, batch_targets)
                batch_loss, batch_detections = batch_outputs["loss"], batch_outputs["detections"].detach()

                batch_targets["bbox"] = [batch_targets_bboxes[:, [1, 0, 3, 2]] for batch_targets_bboxes in batch_targets["bbox"]]
                batch_detections, batch_targets = forward_device([batch_detections, batch_targets], torch.device("cpu"))

                running_loss = running_loss + batch_loss.item()*batch_images.size(0)
                running_detections.append(batch_detections), running_targets.append(batch_targets)

        epoch_loss = running_loss/len(loaders["val"].dataset)
        epoch_ap = get_coco_stats(
            running_detections, running_targets
            , conf_threshold
            , iou_threshold
        )["AP_all_IOU_0_50"]
        print("{}-loss: {:.4f}".format("val", epoch_loss))
        print("{}-ap: {:.4f}".format("val", epoch_ap))

        if monitor == "val_loss":
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), "{}/{}.pt".format(ckps_path, model.config.name))
        if monitor == "val_ap":
            if epoch_ap > best_ap:
                best_ap = epoch_ap
                torch.save(model.state_dict(), "{}/{}.pt".format(ckps_path, model.config.name))

    print("Finish-Best val-ap: {:.4f}".format(best_ap))