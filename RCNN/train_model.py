# Faster R-CNN Training on Colab
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import datetime
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T

if not hasattr(torchvision, "_is_tracing"):
    torchvision._is_tracing = lambda: False

import numpy as np
from torchvision.ops import box_iou

def evaluate_map(model, data_loader, device, iou_threshold=0.5, num_classes=13):
    model.eval()

    
    tp = {c: 0 for c in range(1, num_classes)} # Statistics storage
    fp = {c: 0 for c in range(1, num_classes)}
    fn = {c: 0 for c in range(1, num_classes)}

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue

            images, targets = batch
            images = [img.to(device) for img in images]

            preds = model(images)

            for pred, target in zip(preds, targets):
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)
                pred_boxes = pred["boxes"]
                pred_labels = pred["labels"]

                matched = set()

                for pb, pl in zip(pred_boxes, pred_labels):
                    if pl.item() == 0:
                        continue  # skip background

                    best_iou = 0
                    best_gt = -1

                    for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                        if gl.item() != pl.item():
                            continue

                        iou = box_iou(pb.unsqueeze(0), gb.unsqueeze(0)).item()
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = i

                    if best_iou >= iou_threshold and best_gt not in matched:
                        tp[pl.item()] += 1
                        matched.add(best_gt)
                    else:
                        fp[pl.item()] += 1

                for i, gl in enumerate(gt_labels):
                    if gl.item() == 0:
                        continue
                    if i not in matched:
                        fn[gl.item()] += 1

    
    APs = {} # Compute precision, recall, AP
    for c in range(1, num_classes):
        tp_c = tp[c]
        fp_c = fp[c]
        fn_c = fn[c]

        precision = tp_c / (tp_c + fp_c + 1e-6)
        recall = tp_c / (tp_c + fn_c + 1e-6)

        APs[c] = precision

    mAP = sum(APs.values()) / (num_classes - 1)
    mean_recall = sum([tp[c] / (tp[c] + fn[c] + 1e-6) for c in range(1, num_classes)]) / (num_classes - 1)

    return mAP, mean_recall, APs

class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(img_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.imgs[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                    xmin = (x_c - bw / 2) * w
                    xmax = (x_c + bw / 2) * w
                    ymin = (y_c - bh / 2) * h
                    ymax = (y_c + bh / 2) * h
                    if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(cls) + 1)  # background = 0

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch): # Collate function
    batch = [item for item in batch if len(item[1]["boxes"]) > 0]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))

def get_model_mobilenet(num_classes): # Model definition (MobileNet backbone)
    backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features
    backbone.out_channels = 960

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model
starttime = datetime.datetime.now()

train_img_dir = '/content/train/images' # Paths (adjust for Colab + Drive)
train_label_dir = '/content/train/labels'
valid_img_dir = '/content/valid/images'
valid_label_dir = '/content/valid/labels'


transform = T.Compose([ # Transforms & datasets
    T.Resize((480, 480)),
    T.ToTensor()
])

train_dataset = YoloToFasterRCNNDataset(train_img_dir, train_label_dir, transforms=transform)
valid_dataset = YoloToFasterRCNNDataset(valid_img_dir, valid_label_dir, transforms=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

num_classes = 13 # Model setup
class_names = ["background", "Ants", "Bees", "Beetles", "Caterpillars", "Earthworms",
               "Earwigs", "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils"]

print(f"Training Faster R-CNN (MobileNetV3 backbone) with {num_classes} classes")
print(f"Classes: {class_names}")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_mobilenet(num_classes).to(device)
print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


params = [p for p in model.parameters() if p.requires_grad] # Optimizer & scheduler (DEFAULT)
optimizer = torch.optim.AdamW(params, lr=0.00017173834817695884, weight_decay=0.000018317211670360036, betas=(0.9465235511931575, 0.99))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


num_epochs = 30 # Training loop (baseline)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for batch in train_loader:
        if batch is None:
            continue
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    print(f"\nEpoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    lr_scheduler.step()

    print("Evaluating mAP and recall...")
    mAP, mean_recall, APs = evaluate_map(model, valid_loader, device, num_classes=num_classes)

    print(f"mAP@0.5: {mAP:.4f}")
    print(f"Mean Recall@0.5: {mean_recall:.4f}")
    print(f"AP per class: {APs}")


endtime = datetime.datetime.now()
save_path = '/content/drive/MyDrive/faster_rcnn_mobilenet_AdamW30_optimised.pth'
torch.save(model.state_dict(), save_path) # Save model
print(f"\nTraining complete! Model saved to {save_path}")