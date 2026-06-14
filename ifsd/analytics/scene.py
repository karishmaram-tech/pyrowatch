import cv2
import numpy as np
from ifsd.config import CFG

class SceneDetector:
    """
    Detects people and industrial vehicles using YOLOv8 with frame-skipping optimizations.
    """
    def __init__(self) -> None:
        from ultralytics import YOLO
        import torch

        # Device selection: prefer CUDA GPU, fall back to CPU
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the YOLOv8 nano model
        self._model = YOLO(CFG["YOLO_MODEL"])
        self._model.to(self._device)

        # Target class IDs and human-readable tokens
        self._target_ids = CFG["YOLO_TARGET_IDS"]
        self._conf_thresh = CFG["YOLO_CONF"]
        self._iou_thresh  = CFG["YOLO_IOU"]

        # Frame-skip optimizations tracking metrics
        self._skip_counter = 0
        self._skip_every    = CFG["YOLO_SKIP_FRAMES"]
        self._last_result   = []

        print(f"  [PyroWatch Scene] Running on operational back-end: {self._device.upper()}")
    def detect(self, frame: np.ndarray) -> list[dict]:
        # Frame-skip check pipeline constraint logic
        self._skip_counter += 1
        if self._skip_counter <= self._skip_every:
            return self._last_result
        self._skip_counter = 0

        # Run model prediction vector matrix array math
        results = self._model.predict(
            source=frame,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            classes=list(self._target_ids.keys()),
            device=self._device,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self._target_ids:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x  = int(x1)
            y  = int(y1)
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            cx = int(x + bw / 2)
            cy = int(y + bh / 2)

            detections.append({
                "box": (x, y, bw, bh),
                "label": self._target_ids[cls_id],
                "conf": conf,
                "center": (cx, cy),
            })

        self._last_result = detections
        return detections



