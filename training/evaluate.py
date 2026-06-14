"""
training/evaluate.py
Evaluates the trained model and shows detection results on your video.
Run with: python training\evaluate.py
"""

import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS  = "training/runs/fire_smoke_v1/weights/best.pt"
VIDEO    = "real_fire.mp4"

if not os.path.exists(WEIGHTS):
    print(f"ERROR: {WEIGHTS} not found")
    print("       Run first: python training\\train.py")
    sys.exit(1)

from ultralytics import YOLO
import cv2, torch
import numpy as np

print(f"Loading trained model: {WEIGHTS}")
model  = YOLO(WEIGHTS)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Run validation on the test set ────────────────────────────────────────
print()
print("Running validation on test set...")
yaml_candidates = glob.glob("training/dataset/*.yaml")
if yaml_candidates:
    metrics = model.val(data=yaml_candidates[0], device=device)
    print()
    print("="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"mAP50          : {metrics.box.map50:.3f}")
    print(f"mAP50-95       : {metrics.box.map:.3f}")
    print(f"Precision      : {metrics.box.mp:.3f}")
    print(f"Recall         : {metrics.box.mr:.3f}")
    print()
    print("SCORE GUIDE:")
    print("  mAP50 > 0.80  = Excellent -- deploy with confidence")
    print("  mAP50 > 0.60  = Good      -- acceptable for production")
    print("  mAP50 > 0.40  = Fair      -- train more epochs")
    print("  mAP50 < 0.40  = Poor      -- check dataset or hyperparameters")
    print()

# ── Run detection on real fire video ─────────────────────────────────────
if not os.path.exists(VIDEO):
    print(f"No video file found ({VIDEO}) -- skipping live demo")
    sys.exit(0)

print(f"Running detection on: {VIDEO}")
print("Press Q to quit the preview window")
print()

cap    = cv2.VideoCapture(VIDEO)
font   = cv2.FONT_HERSHEY_SIMPLEX
colors = {"fire": (0,100,255), "smoke": (200,200,200)}
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame     = cv2.resize(frame, (1280, 720))
    frame_num += 1

    results = model.predict(
        source  = frame,
        conf    = 0.35,
        device  = device,
        verbose = False,
    )

    # Draw detections
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names[cls_id].lower()
        col    = colors.get(label, (255,255,255))

        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]

        # Glow effect
        for expand in [4,2,0]:
            alpha = [0.2, 0.5, 1.0][expand//2 if expand > 0 else 2]
            cv2.rectangle(frame,
                          (x1-expand, y1-expand),
                          (x2+expand, y2+expand),
                          col, 1)

        # Label badge
        badge = f"{label.upper()} {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(badge, font, 0.55, 1)
        cv2.rectangle(frame, (x1,y1-th-10), (x1+tw+8,y1), (10,10,20), -1)
        cv2.putText(frame, badge, (x1+4,y1-6),
                    font, 0.55, col, 1, cv2.LINE_AA)

    # Stats overlay
    n_fire  = sum(1 for b in results[0].boxes
                  if model.names[int(b.cls[0])].lower() == "fire")
    n_smoke = sum(1 for b in results[0].boxes
                  if model.names[int(b.cls[0])].lower() == "smoke")

    cv2.rectangle(frame, (0,0), (340,30), (10,10,20), -1)
    cv2.putText(frame,
                f"YOLO FIRE DETECTOR  |  fire:{n_fire}  smoke:{n_smoke}  "
                f"frame:{frame_num}",
                (6,20), font, 0.45, (100,200,255), 1, cv2.LINE_AA)

    cv2.imshow("PyroWatch -- Trained YOLO Fire Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Evaluation complete.")
print()
print("Next step: python training\\deploy.py")



