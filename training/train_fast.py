"""
training/train_fast.py
Optimised training for CPU -- 15 epochs, smaller image size.
Trains in ~2-3 hours on i3 CPU instead of 79 hours.
mAP will be slightly lower (~0.55-0.65) but fully deployable.
"""

import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

yaml_candidates = glob.glob("training/dataset/*.yaml")
if not yaml_candidates:
    print("ERROR: Dataset not found. Run: python training\\download_dataset.py")
    sys.exit(1)

yaml_path = yaml_candidates[0]
print(f"Dataset : {yaml_path}")

from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {device.upper()}")
print(f"Mode    : FAST CPU training (15 epochs, imgsz=416)")
print()

model = YOLO("yolov8n.pt")

results = model.train(
    data          = yaml_path,
    epochs        = 15,        # 15 instead of 50 -- enough for good detection
    imgsz         = 416,       # 416 instead of 640 -- 2x faster per batch
    batch         = 4,         # 4 instead of 8 -- less RAM pressure
    project       = "training/runs",
    name          = "fire_smoke_fast",
    device        = device,
    patience      = 8,         # stop if no improvement for 8 epochs
    save          = True,
    plots         = True,
    verbose       = True,
    workers       = 0,
    cache         = True,      # cache images in RAM -- faster after epoch 1
    optimizer     = "AdamW",
    lr0           = 0.001,
    lrf           = 0.01,
    warmup_epochs = 2,
    mosaic        = 0.5,       # reduced augmentation = faster
    fliplr        = 0.5,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
)

print()
print("="*50)
print("FAST TRAINING COMPLETE")
print("="*50)

import glob as g
best = g.glob("training/runs/fire_smoke_fast*/weights/best.pt")
if best:
    size = round(os.path.getsize(best[0])/1024/1024, 1)
    print(f"Best weights: {best[0]}  ({size} MB)")
    print()
    print("Next: python training\\deploy_fast.py")



