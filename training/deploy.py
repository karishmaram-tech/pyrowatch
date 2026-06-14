"""
training/deploy.py
Copies the trained model into the PyroWatch package and updates
the fire detector to use YOLO instead of HSV colour filters.
Run with: python training\deploy.py
"""

import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_SRC = "training/runs/fire_smoke_v1/weights/best.pt"
WEIGHTS_DST = "PyroWatch/detectors/yolo_fire/fire_smoke_best.pt"

if not os.path.exists(WEIGHTS_SRC):
    print(f"ERROR: {WEIGHTS_SRC} not found")
    print("       Run first: python training\\train.py")
    sys.exit(1)

# Copy weights into the package
os.makedirs("PyroWatch/detectors/yolo_fire", exist_ok=True)
shutil.copy2(WEIGHTS_SRC, WEIGHTS_DST)
size_mb = round(os.path.getsize(WEIGHTS_DST)/1024/1024, 1)
print(f"Weights copied: {WEIGHTS_DST}  ({size_mb} MB)")

# Write the model info file
with open("PyroWatch/detectors/yolo_fire/model_info.txt", "w") as f:
    f.write(f"Model      : YOLOv8n fine-tuned on fire/smoke dataset\n")
    f.write(f"Dataset    : YOLOv9 fire and smoke (Roboflow, 2301 images)\n")
    f.write(f"Classes    : fire, smoke\n")
    f.write(f"Weights    : {WEIGHTS_DST}\n")
    f.write(f"Source     : {WEIGHTS_SRC}\n")

print("Model info saved: PyroWatch/detectors/yolo_fire/model_info.txt")
print()
print("="*60)
print("DEPLOYMENT COMPLETE")
print("="*60)
print()
print("Your trained model is now part of the PyroWatch package.")
print()
print("To use it in run.py, replace FireDetector with YOLOFireDetector:")
print()
print("  from ifsd.detectors.yolo_fire.detector import YOLOFireDetector")
print("  fire_det = YOLOFireDetector()")
print()
print("  The rest of run.py stays exactly the same.")



