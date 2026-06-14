"""
training/train.py
Trains YOLOv8n on the fire and smoke dataset.
Run with: python training\train.py

EXPECTED TRAINING TIME:
  CPU only, 50 epochs: 3-6 hours
  GPU (CUDA):          20-40 minutes

WHAT GETS SAVED:
  training/runs/fire_smoke_v1/weights/best.pt   <- use this in production
  training/runs/fire_smoke_v1/weights/last.pt   <- last epoch checkpoint
  training/runs/fire_smoke_v1/results.png       <- training curves
  training/runs/fire_smoke_v1/confusion_matrix.png
"""

import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Verify dataset exists ──────────────────────────────────────────────────
yaml_candidates = glob.glob("training/dataset/*.yaml")
if not yaml_candidates:
    print("ERROR: Dataset not found.")
    print("       Run first: python training\\download_dataset.py")
    sys.exit(1)

yaml_path = yaml_candidates[0]
print(f"Dataset config : {yaml_path}")

# ── Read and display class names ───────────────────────────────────────────
import yaml
with open(yaml_path) as f:
    dataset_cfg = yaml.safe_load(f)

print(f"Classes        : {dataset_cfg.get('names', 'unknown')}")
print(f"Num classes    : {dataset_cfg.get('nc', 'unknown')}")
print()

# ── Training configuration ─────────────────────────────────────────────────
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device.upper()}")

if device == "cpu":
    print("WARNING: Training on CPU is slow (3-6 hours for 50 epochs).")
    print("         The model will still work correctly -- just takes time.")
    print("         You can safely leave this running overnight.")
    print()

# Load the base YOLOv8 nano model (pretrained on COCO)
# We fine-tune it on fire/smoke images -- much faster than training from scratch
# because the model already knows how to detect shapes and textures
model = YOLO("yolov8n.pt")

print("Starting training...")
print("Progress is printed every epoch.")
print("Press Ctrl+C to stop early -- best.pt is saved automatically.")
print()

results = model.train(
    data       = yaml_path,
    epochs     = 50,           # 50 passes through the full dataset
    imgsz      = 640,          # input resolution (matches our pipeline)
    batch      = 8,            # images per batch (safe for 8GB RAM on CPU)
    project    = "training/runs",
    name       = "fire_smoke_v1",
    device     = device,
    patience   = 15,           # stop early if no improvement for 15 epochs
    save       = True,         # save best.pt and last.pt
    plots      = True,         # save training curve graphs
    verbose    = True,
    workers    = 0,            # 0 = use main process (safer on Windows)
    cache      = False,        # don't cache dataset in RAM (saves memory)
    optimizer  = "AdamW",      # good for fine-tuning
    lr0        = 0.001,        # initial learning rate
    lrf        = 0.01,         # final learning rate multiplier
    warmup_epochs = 3,         # gentle warmup for first 3 epochs
    mosaic     = 1.0,          # data augmentation: mosaic (combines 4 images)
    flipud     = 0.1,          # random vertical flip 10% of the time
    fliplr     = 0.5,          # random horizontal flip 50% of the time
    hsv_h      = 0.015,        # hue augmentation (helps with lighting variation)
    hsv_s      = 0.7,          # saturation augmentation
    hsv_v      = 0.4,          # brightness augmentation
)

print()
print("="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best mAP50     : {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
print(f"Best mAP50-95  : {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
print()

best_weights = "training/runs/fire_smoke_v1/weights/best.pt"
if os.path.exists(best_weights):
    size_mb = round(os.path.getsize(best_weights) / 1024 / 1024, 1)
    print(f"Best weights   : {best_weights}  ({size_mb} MB)")
    print()
    print("Next step: python training\\evaluate.py")
else:
    print("WARNING: best.pt not found -- training may have been interrupted")
    print("         Run again to resume from last checkpoint")



