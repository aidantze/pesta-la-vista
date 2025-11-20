# Faster R-CNN Insect Detection - Training

Training script for Faster R-CNN with MobileNetV3 backbone and dropout regularization on Google Colab.

## Prerequisites

- Google Colab account
- Dataset files
- Google Drive account for storage

## Setup Instructions for Google Colab

### Step 1: Prepare Your Dataset

1. Create a zip file containing your dataset with the following structure:
```
dataset.zip
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

2. Upload `dataset.zip` to your Google Drive

### Step 2: Create Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy and run the following cells:

### Step 3: Mount Google Drive

From google collab, mount a google drive to the runtime.

### Step 4: Extract Dataset

In the terminal, run:
unzip /content/drive/MyDrive/archive.zip -d /content/

### Step 5: Upload and Run Training Script

1. Upload `train_dropout.py` or `train_model.py` to your Colab environment:

## Directory Structure After Setup

After completing the setup, your `/content` directory should look like:

```
/content/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── train_dropout.py
```

## Output

After training completes:

- Model checkpoint saved to: e.g. `/content/drive/MyDrive/faster_rcnn_mobilenet_AdamW30_dropout.pth`
- Training logs printed to Colab console showing:
  - Epoch loss
  - Learning rate schedule
  - mAP@0.5 validation metrics
  - Per-class average precision

---

# Visualization - demo.py

Run inference on test images and visualize predictions vs ground truth annotations.

## Local Setup Requirements

### Step 1: Prepare Your Directory

Create a directory structure with the demo script, dataset, and trained weights:

```
project_directory/
├── demo.py
├── faster_rcnn_mobilenet_AdamW30_dropout.pth
├── test/
│   ├── images/
│   │   ├── ants-112-_jpg.rf.xxx.jpg
│   │   ├── bees-10-_jpg.rf.xxx.jpg
│   │   └── ...
│   └── labels/
│       ├── ants-112-_jpg.rf.xxx.txt
│       ├── bees-10-_jpg.rf.xxx.txt
│       └── ...
```

### Step 2: Download Required Files
```bash
pip install torch torchvision pillow opencv-python numpy
```

### Step 4: Run the Demo

```bash
python demo.py
```
