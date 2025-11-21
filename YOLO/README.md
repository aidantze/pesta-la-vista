# YOLOv8n testing, final evaluation and demo
This repository contains scripts and notebooks for benchmarking YOLOv8n on the CropPests dataset, including validation-set testing, noise robustness evaluation, mosaic and resolution experiments,final model evaluation and demo.

---

## Overview  
The project evaluates:

- Baseline YOLOv8n performance  
- Effect of **noise corruption** (Gaussian, Salt-and-Pepper, Blur), **mosaic augmentation**, **different input resolutions**  
- Final evaluation of three model with epoch Optimising
- Demo for the best model
  
Each script is independent and focuses on one experimental component.

---

## Prerequisites

### **Google Colab Setup**
- A Google Colab account  
- (Optional) Google Drive for saving outputs  

### **Required Packages**
Install the dependencies in Colab:

```bash
!pip install -q ultralytics kagglehub pyyaml opencv-python efficientnet_pytorch torch torchvision

# Imports:
import torch
import numpy as np
import kagglehub
import pathlib
import yaml
import shutil
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
import random
```

# **Setup Instructions for Google Colab**

## Step 1 — Create a New Colab Notebook
1. Go to Google Colab￼
2. Create a new notebook

## Step 2 — Mount Google Drive (Optional but Recommended)

  In a Colab cell, run:
  ```bash
from google.colab import drive
drive.mount('/content/drive')
```
## Step 3 — Upload and Run YOLOv8n Scripts

1. Run the following notebooks/scripts in this order:
- yolov8n_benchmark_EDA.ipynb
- yolov8n_hyperparameter_testing.ipynb
- yolov8n_models_optimised_eval.ipynb
- yolov8n_demo.ipynb
2. Each file is independent and focuses on what its name describes:
- Benchmark & EDA – baseline performance and exploratory analysis
- Hyperparameter Testing – tests mosaic, noise, resolution
- Optimised Model Evaluation – evaluates best configurations and epochs
- Demo – visualises predictions and plots bounding boxes
  
# Expected Directory Structure

After dataset setup, your /content directory in Colab should look like:
```bash
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
```
# Output
After training and evaluation, you will obtain:
Model checkpoints (e.g. best.pt) saved either in /content or in your Google Drive (depending on paths used). Training logs shown in the Colab output cells. Evaluation metrics, including:
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1-score
- Training time
- Testing time
  
This YOLOv8n benchmark and hyperparameter testing workflow was developed as part of a computer vision project on pest detection.
