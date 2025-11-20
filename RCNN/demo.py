import os
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import numpy as np
from collections import defaultdict
import random


def get_model_mobilenet(num_classes): #Define the model
    backbone = torchvision.models.mobilenet_v3_large(weights='DEFAULT').features
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



num_classes = 13 # Load trained weights
device = torch.device('cpu')

model = get_model_mobilenet(num_classes)
model.load_state_dict(torch.load("faster_rcnn_mobilenet_AdamW30_dropout.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")


class_names = ["background", "Ants", "Bees", "Beetle", "Catterpillar", "Earthworms", # Class names
               "Earwig", "Grasshopper", "Moth", "Slug", "Snail", "Wasp", "Weevil"]


class_to_idx = {name.lower(): idx for idx, name in enumerate(class_names)} # class name → id (case-insensitive)


transform = T.Compose([ # Transform 
    T.ToTensor()
])


test_dir = "test/images" # Directories
test_labels_dir = "test/labels"

img_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])


def load_yolo_labels(label_path, img_width, img_height): #Parse YOLO Label Files
    boxes = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            idx, xc, yc, w, h = map(float, line.strip().split())
            idx = int(idx) + 1

            x1 = int((xc - w/2) * img_width)
            y1 = int((yc - h/2) * img_height)
            x2 = int((xc + w/2) * img_width)
            y2 = int((yc + h/2) * img_height)

            boxes.append((idx, x1, y1, x2, y2))

    return boxes



class_images = defaultdict(list) # GROUP IMAGES BY CLASS

for img in img_files:
    base = img.rsplit(".", 1)[0]
    cls_name = base.split("-")[0].lower()
    cls_id = class_to_idx[cls_name]
    class_images[cls_id].append(img)




selected_images = [] #SELECT 5 RANDOM IMAGES PER CLASS
for cls_id, imgs in class_images.items():
    if len(imgs) >= 5:
        selected_images.extend(random.sample(imgs, 5))
    else:
        selected_images.extend(imgs)

print(f"Total selected images: {len(selected_images)}")



for img_name in selected_images: #RUN INFERENCE + DRAW BOUNDING BOXES
    img_path = os.path.join(test_dir, img_name)
    label_path = os.path.join(test_labels_dir, img_name.rsplit(".", 1)[0] + ".txt")

    
    img = cv2.imread(img_path) # Load image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    gt_boxes = load_yolo_labels(label_path, w, h) # Load GT labels (BLUE)

    input_tensor = transform(Image.fromarray(img_rgb)).to(device) # Run model
    with torch.no_grad():
        outputs = model([input_tensor])[0]

    pred_boxes = outputs["boxes"].cpu().numpy()
    pred_scores = outputs["scores"].cpu().numpy()
    pred_labels = outputs["labels"].cpu().numpy()

    
    for (idx, x1, y1, x2, y2) in gt_boxes: # Draw GT (blue)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"GT: {class_names[idx]}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    
    for box, score, idx in zip(pred_boxes, pred_scores, pred_labels): # Draw predictions (red)
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        conf = int(score * 100)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{class_names[idx]} {conf}%", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("GT (blue) vs Prediction (red)", img)# Show result
    cv2.waitKey(0)

cv2.destroyAllWindows()
