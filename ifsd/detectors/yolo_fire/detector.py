"""
PyroWatch/detectors/yolo_fire/detector.py
YOLOFireDetector -- drop-in replacement for FireDetector
Uses the trained YOLOv8 model instead of HSV colour filters.

Same output format as FireDetector -- fully compatible with run.py.
"""

import cv2
import numpy as np
import torch
import os

from ultralytics import YOLO
from ifsd.utils import ExpSmooth
from ifsd.config import CFG

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fire_smoke_best.pt"
)


class YOLOFireDetector:
    """
    Drop-in replacement for FireDetector using trained YOLOv8.

    Output format is identical to FireDetector.detect() so run.py
    needs zero changes except the import line.

    HOW TO USE:
        from ifsd.detectors.yolo_fire.detector import YOLOFireDetector
        fire_det = YOLOFireDetector()

        # inside your frame loop -- exactly the same as before:
        result = fire_det.detect(frame)
        # result["boxes"], result["confidence"], result["mask"] -- all identical
    """

    def __init__(self, weights: str = WEIGHTS_PATH) -> None:
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"Trained weights not found: {weights}\n"
                f"Run first: python training\\deploy.py"
            )

        self._device   = "cuda" if torch.cuda.is_available() else "cpu"
        self._model    = YOLO(weights)
        self._model.to(self._device)
        self._smoother = ExpSmooth(alpha=CFG["FIRE_EMA_ALPHA"])
        self._frame_area = 0.0

        # Get class names from the model
        self._names = self._model.names   # {0: "fire", 1: "smoke"}
        print(f"[PyroWatch YOLO] Loaded: {weights}")
        print(f"[PyroWatch YOLO] Classes: {self._names}")
        print(f"[PyroWatch YOLO] Device : {self._device.upper()}")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run YOLO fire+smoke detection. Returns same format as FireDetector.
        """
        h, w = frame.shape[:2]
        if self._frame_area == 0.0:
            self._frame_area = float(h * w)

        results = self._model.predict(
            source  = frame,
            conf    = CFG["YOLO_CONF"],
            iou     = CFG["YOLO_IOU"],
            device  = self._device,
            verbose = False,
        )

        boxes         = []
        contours      = []
        total_fire_px = 0

        for box in results[0].boxes:
            cls_name = self._names[int(box.cls[0])].lower()
            if cls_name not in ("fire", "smoke"):
                continue

            conf_score = float(box.conf[0])
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
            bw = x2 - x1
            bh = y2 - y1

            boxes.append((x1, y1, bw, bh))
            total_fire_px += bw * bh

            # Synthetic contour from bounding box corners
            cnt = np.array([
                [[x1, y1]], [[x2, y1]],
                [[x2, y2]], [[x1, y2]]
            ], dtype=np.int32)
            contours.append(cnt)

        # Build a blank mask (kept for API compatibility)
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y, bw, bh) in boxes:
            mask[y:y+bh, x:x+bw] = 255

        raw_conf    = min(total_fire_px / self._frame_area, 1.0)
        smooth_conf = self._smoother.update(raw_conf)

        return {
            "boxes"      : boxes,
            "contours"   : contours,
            "confidence" : smooth_conf,
            "mask"       : mask,
        }

    def reset(self) -> None:
        self._smoother.reset()
        self._frame_area = 0.0



