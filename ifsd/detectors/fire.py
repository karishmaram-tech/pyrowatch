"""
================================================================================
  PyroWatch/detectors/fire.py
  FireDetector — HSV Colour Segmentation + Morphological Cleanup Pipeline
================================================================================
  PIPELINE OVERVIEW (what happens to each frame, in order):
    1. Convert BGR frame → HSV colour space
    2. Build three HSV masks (red-orange body A, red wrap-around B, yellow core C)
    3. Combine all three masks with bitwise OR
    4. Morphological CLOSE  → fills small holes inside flame blobs
    5. Morphological OPEN   → removes isolated speckle noise pixels
    6. Find external contours on the cleaned mask
    7. Filter: drop contours below minimum area threshold
    8. Filter: check mean V-channel brightness inside bounding box ROI
    9. Collect surviving detections, compute normalised confidence score
   10. Feed raw confidence through ExpSmooth, return results
================================================================================
"""

import cv2
import numpy as np

from ifsd.config import CFG
from ifsd.utils  import ExpSmooth


class FireDetector:
    """
    Detects fire in a single BGR video frame using HSV colour segmentation.

    HOW TO USE:
        detector = FireDetector()

        # inside your frame loop:
        result = detector.detect(frame)

        # result is a dict with keys:
        #   "boxes"      → list of (x, y, w, h) tuples  — bounding rectangles
        #   "contours"   → list of raw contour point arrays (for precise outlines)
        #   "confidence" → float 0.0–1.0, smoothed fire confidence this frame
        #   "mask"       → the final cleaned binary mask (useful for HUD overlay)
    """

    def __init__(self) -> None:
        # ── Pre-build morphological kernels once at construction time ──────
        # Creating kernels inside detect() would allocate new numpy arrays on
        # every single frame (30x per second). Building them once here and
        # reusing them is a free performance win.
        #
        # CLOSE kernel (larger): joins nearby blobs, fills internal holes.
        # Think of it as "expanding then shrinking" — gaps smaller than the
        # kernel disappear.
        close_k = CFG["FIRE_MORPH_CLOSE_K"]
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k)
        )

        # OPEN kernel (smaller): removes isolated noise specks.
        # Think of it as "shrinking then expanding" — blobs smaller than the
        # kernel disappear while large blobs stay roughly the same size.
        open_k = CFG["FIRE_MORPH_OPEN_K"]
        self._kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_k, open_k)
        )

        # ── Pre-read thresholds from CFG into instance variables ──────────
        # Attribute lookups on self are faster than repeated CFG dict lookups
        # inside the hot path of detect() which runs 30 times per second.
        self._min_area   = CFG["FIRE_MIN_AREA"]
        self._min_v_mean = CFG["FIRE_MIN_V_MEAN"]

        # ── HSV range arrays (already np.uint8 from config.py) ───────────
        self._lower_a = CFG["FIRE_HSV_LOWER_A"]
        self._upper_a = CFG["FIRE_HSV_UPPER_A"]
        self._lower_b = CFG["FIRE_HSV_LOWER_B"]
        self._upper_b = CFG["FIRE_HSV_UPPER_B"]
        self._lower_c = CFG["FIRE_HSV_LOWER_C"]
        self._upper_c = CFG["FIRE_HSV_UPPER_C"]

        # ── Confidence smoother ───────────────────────────────────────────
        self._smoother = ExpSmooth(alpha=CFG["FIRE_EMA_ALPHA"])

        # ── Frame area cached after first detection call ──────────────────
        # Used to normalise raw pixel count into a 0.0–1.0 confidence score.
        self._frame_area: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD — call this every frame
    # ─────────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> dict:
        """
        Run the full fire detection pipeline on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image as returned by cv2.VideoCapture.read() or cv2.imread().
            Shape must be (height, width, 3).

        Returns
        -------
        dict with keys:
            boxes      : list[tuple[int,int,int,int]]  — (x, y, w, h) per detection
            contours   : list[np.ndarray]              — raw contour point arrays
            confidence : float                         — smoothed score 0.0–1.0
            mask       : np.ndarray                    — cleaned binary mask (uint8)
        """
        h, w = frame.shape[:2]

        # Cache frame area on first call (avoids multiply every frame after)
        if self._frame_area == 0.0:
            self._frame_area = float(h * w)

        # ── STEP 1: Convert BGR → HSV ─────────────────────────────────────
        #
        # WHY HSV instead of staying in BGR?
        #
        # In BGR, a pixel's "colour" and "brightness" are tangled together.
        # A dim red (10, 0, 30) and a bright red (50, 0, 200) look completely
        # different numerically even though both are "red".
        #
        # In HSV:
        #   H (Hue)        = the pure colour, 0-179 in OpenCV
        #   S (Saturation) = how vivid vs washed-out, 0-255
        #   V (Value)      = how bright vs dark, 0-255
        #
        # This separation lets us write a simple range check like
        # "H between 0-18 AND S > 120 AND V > 120" to reliably catch ANY
        # shade of orange-red flame regardless of lighting conditions.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── STEP 2: Build the three HSV masks ─────────────────────────────
        #
        # cv2.inRange(image, lower, upper) returns a binary mask where:
        #   255 = pixel is INSIDE the colour range (fire candidate)
        #     0 = pixel is OUTSIDE the range (not fire)
        #
        # WHY THREE RANGES?
        #   OpenCV's Hue axis goes 0 → 179. Red sits at BOTH ends:
        #   near Hue=0 (deep red) and near Hue=179 (also deep red, wrap-around).
        #   A single range [160, 180] misses the red near 0, and vice versa.
        #   Range A catches orange-red (H: 0-18).
        #   Range B catches the wrap-around red (H: 160-179).
        #   Range C catches yellow-white hot cores (H: 20-35, low-sat, bright).
        mask_a = cv2.inRange(hsv, self._lower_a, self._upper_a)
        mask_b = cv2.inRange(hsv, self._lower_b, self._upper_b)
        mask_c = cv2.inRange(hsv, self._lower_c, self._upper_c)

        # ── STEP 3: Combine all three masks with bitwise OR ───────────────
        # A pixel is "fire" if it matches ANY of the three ranges.
        # cv2.bitwise_or: output pixel = 255 if EITHER input pixel = 255
        combined = cv2.bitwise_or(mask_a, mask_b)
        combined = cv2.bitwise_or(combined, mask_c)

        # ── STEP 4: Morphological CLOSING ─────────────────────────────────
        #
        # Operation: dilate THEN erode (with the same kernel)
        #
        # Effect on the mask:
        #   BEFORE closing:  [255, 255, 0, 0, 255, 255]  ← gap in the middle
        #   AFTER  closing:  [255, 255, 255, 255, 255, 255]  ← gap filled
        #
        # Real-world benefit: flames are not solid blobs — they have dark
        # "holes" where smoke passes through. Closing connects these fragments
        # into a single solid region, giving cleaner bounding boxes.
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._kernel_close)

        # ── STEP 5: Morphological OPENING ─────────────────────────────────
        #
        # Operation: erode THEN dilate (reverse order of closing)
        #
        # Effect on the mask:
        #   BEFORE opening: main blob intact + tiny isolated speckle
        #   AFTER  opening: main blob intact, tiny speckle REMOVED
        #
        # Real-world benefit: a red traffic light, a red LED, a reflective
        # red safety sign — all produce tiny isolated red blobs in the mask.
        # Opening erases anything smaller than the kernel (3×3 px), while
        # leaving the larger genuine flame regions untouched.
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self._kernel_open)

        # ── STEP 6: Find contours ─────────────────────────────────────────
        #
        # cv2.findContours returns a list of contours, where each contour is
        # an array of (x, y) points tracing the boundary of one white blob.
        #
        # RETR_EXTERNAL: only find outermost contours (ignore holes inside blobs)
        # CHAIN_APPROX_SIMPLE: compress straight edges to just their endpoints,
        #                       saving memory vs storing every pixel on the edge
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── STEPS 7 + 8: Filter contours ─────────────────────────────────
        boxes        = []
        kept_contours = []
        total_fire_px = 0

        for cnt in contours:

            # FILTER 7: Area gate
            # cv2.contourArea() counts pixels enclosed by the contour boundary.
            # Drop anything smaller than FIRE_MIN_AREA (800 px²).
            # At 1280×720 resolution, 800 px² is a ~28×28 pixel square — small
            # enough to catch real flames but large enough to ignore LED pinpoints.
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue

            # Get the axis-aligned bounding box around this contour
            x, y, bw, bh = cv2.boundingRect(cnt)

            # FILTER 8: V-channel brightness validation
            #
            # Problem: A red car, red sign, or orange machinery could survive
            # the colour filter AND the area filter. The key difference between
            # those and a real flame is BRIGHTNESS. Flames are extremely bright
            # (high V in HSV). Painted surfaces and fabrics are usually duller.
            #
            # We extract the Region of Interest (ROI) from the HSV image,
            # take only the V channel (index 2), and compute its mean value.
            # If the mean V is below FIRE_MIN_V_MEAN (160 out of 255), reject it.
            #
            # ROI extraction: hsv[y:y+h, x:x+w] is numpy slice notation —
            # it crops a rectangle from row y to y+bh, column x to x+bw.
            roi_v = hsv[y : y + bh, x : x + bw, 2]   # channel 2 = Value
            mean_v = float(np.mean(roi_v))

            if mean_v < self._min_v_mean:
                continue   # not bright enough to be a real flame

            # This detection survived all filters — keep it
            boxes.append((x, y, bw, bh))
            kept_contours.append(cnt)
            total_fire_px += int(area)

        # ── STEP 9: Compute normalised confidence ─────────────────────────
        #
        # Raw confidence = fraction of the total frame covered by fire pixels.
        #
        #   raw_conf = total_fire_pixels / total_frame_pixels
        #
        # Example: if 15,000 pixels out of 921,600 (1280×720) are fire:
        #   raw_conf = 15000 / 921600 = 0.0163  (about 1.6% of the frame)
        #
        # We then clamp to [0.0, 1.0] as a safety measure (should never
        # exceed 1.0 in practice, but floating point can surprise you).
        raw_conf = min(total_fire_px / self._frame_area, 1.0) if self._frame_area > 0 else 0.0

        # ── STEP 10: Smooth the confidence ───────────────────────────────
        # Feed the raw score through EMA to eliminate single-frame spikes.
        # A static red object appearing suddenly for one frame won't spike
        # the alarm — it has to persist across several frames to register.
        smooth_conf = self._smoother.update(raw_conf)

        return {
            "boxes"      : boxes,
            "contours"   : kept_contours,
            "confidence" : smooth_conf,
            "mask"       : cleaned,
        }

    def reset(self) -> None:
        """Reset the EMA smoother. Call this when switching to a new video source."""
        self._smoother.reset()
        self._frame_area = 0.0



