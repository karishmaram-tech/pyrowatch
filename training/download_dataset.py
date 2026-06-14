"""
training/download_dataset.py
Downloads the fire and smoke detection dataset from Roboflow.
Run with: python training\download_dataset.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read API key from .env
api_key = None
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("ROBOFLOW_API_KEY"):
                api_key = line.split("=", 1)[1].strip()
                break

if not api_key or api_key == "YOUR_KEY_HERE":
    print("ERROR: ROBOFLOW_API_KEY not set in .env file")
    print("       1. Go to https://app.roboflow.com/settings/api")
    print("       2. Copy your Private API Key")
    print("       3. Add it to .env:  ROBOFLOW_API_KEY=your_key_here")
    sys.exit(1)

print("Roboflow API key found.")
print("Downloading fire and smoke detection dataset...")
print("Dataset: YOLOv9 fire and smoke -- 2,301 images")
print("This may take 2-5 minutes depending on your connection.")
print()

from roboflow import Roboflow

rf      = Roboflow(api_key=api_key)
project = rf.workspace("yolov9-9uvwb").project("yolov9-fire-and-smoke-dataset")
dataset = project.version(1).download("yolov8", location="training/dataset")

print()
print("Dataset downloaded successfully!")
print(f"Location: training/dataset/")
print()

# Count images
import glob
train_imgs = glob.glob("training/dataset/train/images/*.jpg")
val_imgs   = glob.glob("training/dataset/valid/images/*.jpg")
test_imgs  = glob.glob("training/dataset/test/images/*.jpg")

print(f"Train images : {len(train_imgs)}")
print(f"Val images   : {len(val_imgs)}")
print(f"Test images  : {len(test_imgs)}")
print(f"Total        : {len(train_imgs)+len(val_imgs)+len(test_imgs)}")
print()
print("Next step: python training\\train.py")



