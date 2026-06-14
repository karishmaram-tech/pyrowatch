"""
================================================================================
  PyroWatch/detectors/smoke.py
  SmokeDetector -- MOG2 Background Subtraction + HSV Grey-Tone Mask Pipeline
================================================================================
  PIPELINE OVERVIEW (what happens to each frame, in order):
    1. Resize/blur the frame with a large Gaussian kernel (softens smoke edges)
    2. Feed blurred frame into MOG2 background subtractor -> motion mask
    3. Convert original frame to HSV -> build grey-tone colour mask
    4. Bitwise AND: motion mask AND colour mask -> only moving grey regions survive
    5. Find contours on the combined mask
    6. Filter contours below minimum area threshold
    7. Compute raw confidence from surviving pixel area
    8. Smooth confidence through ExpSmooth, return results

  WHY TWO MASKS COMBINED?
    Motion mask alone catches everything that moves -- people, vehicles,
    waving flags, flickering lights. False positives everywhere.
    Colour mask alone catches all grey/white regions -- walls, floors,
    white machinery, steam pipes. Also too many false positives.
    COMBINED: only regions that are BOTH moving AND grey/white survive.
    That intersection is a very strong smoke signature.
================================================================================
"""

import cv2
import numpy as np

from ifsd.config import CFG
from ifsd.utils  import ExpSmooth


class SmokeDetector:
    """
    Detects smoke in a single BGR video frame using temporal motion analysis
    combined with HSV grey-tone colour segmentation.

    HOW TO USE:
        detector = SmokeDetector()

        # inside your frame loop -- ORDER MATTERS, must be called every frame:
        result = detector.detect(frame)

        # result is a dict with keys:
        #   "boxes"      -> list of (x, y, w, h) tuples
        #   "contours"   -> list of raw contour point arrays
        #   "confidence" -> float 0.0-1.0, smoothed smoke confidence
        #   "mask"       -> final combined binary mask

    IMPORTANT: The MOG2 background model needs roughly 100-200 frames of
    "normal" footage before it becomes accurate. During this warm-up period
    expect slightly elevated false positives. This is normal behaviour.
    """

    def __init__(self) -> None:
        # -- MOG2 Background Subtractor ------------------------------------
        #
        # MOG2 = Mixture of Gaussians version 2
        #
        # HOW IT WORKS (plain English):
        #   For every pixel position in the frame, MOG2 maintains a statistical
        #   model of what that pixel "normally" looks like, built from the last
        #   SMOKE_MOG2_HISTORY frames (400 by default).
        #
        #   Think of it as: "I have watched this pixel for 400 frames. Its
        #   average colour is (180, 170, 165) with a small spread. If today's
        #   pixel value is (220, 215, 210) -- close to normal -- it is
        #   BACKGROUND. If today's value is (240, 240, 240) -- very different
        #   -- it is FOREGROUND (something has changed, likely smoke or motion)."
        #
        #   The threshold parameter controls how different a pixel must be
        #   to count as foreground. Higher = only catches dramatic changes.
        #   Lower = catches subtle changes but more noise.
        #
        # detectShadows=False: shadows appear as dark grey blobs on the motion
        #   mask by default. For smoke detection we don't want shadow pixels
        #   competing with smoke pixels, so we disable shadow detection.
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=CFG["SMOKE_MOG2_HISTORY"],
            varThreshold=CFG["SMOKE_MOG2_THRESH"],
            detectShadows=False,
        )

        # -- Gaussian blur kernel size -------------------------------------
        #
        # WHY SUCH A LARGE BLUR KERNEL (21x21)?
        #
        # Smoke does not have sharp edges. It is a diffuse, translucent cloud
        # that gradually fades into the background over 20-50 pixels.
        # If we feed the raw crisp frame into MOG2 and the colour filter,
        # the detector sees fragmented, patchy blobs instead of one solid region.
        #
        # A 21x21 Gaussian blur spreads each pixel's colour information across
        # its 10-pixel neighbourhood BEFORE analysis. This means:
        #   - Smoke edges become wide gradients that merge into solid blobs
        #   - Sharp noise (single bright pixels, camera grain) gets averaged away
        #   - The resulting mask has smoother, more connected smoke regions
        #
        # The kernel size MUST be odd (cv2 requirement). 21 is a good default
        # for 720p footage. For lower resolution, try 11 or 15.
        k = CFG["SMOKE_BLUR_K"]
        self._blur_k = (k, k)

        # -- HSV grey-tone range -------------------------------------------
        self._lower_grey = CFG["SMOKE_HSV_LOWER"]
        self._upper_grey = CFG["SMOKE_HSV_UPPER"]

        # -- Minimum contour area ------------------------------------------
        # Smoke blobs must be larger than fire blobs because smoke spreads
        # across a wider area. Small isolated grey patches (dust, steam vents)
        # are filtered out by this threshold.
        self._min_area = CFG["SMOKE_MIN_AREA"]

        # -- Morphological kernel for cleaning the combined mask -----------
        # A modest 5x5 ellipse kernel to connect nearby smoke fragments
        # after the AND operation, which can break up smoke into pieces.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # -- EMA confidence smoother ---------------------------------------
        # Lower alpha (0.25) than fire because smoke takes longer to appear
        # and longer to clear. The slow decay keeps the alert active during
        # brief gaps in smoke visibility (e.g. a fan gust dispersing it).
        self._smoother = ExpSmooth(alpha=CFG["SMOKE_EMA_ALPHA"])

        # -- Cache frame area after first call ----------------------------
        self._frame_area: float = 0.0

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run the full smoke detection pipeline on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image, shape (height, width, 3), dtype uint8.

        Returns
        -------
        dict with keys:
            boxes      : list of (x, y, w, h) tuples
            contours   : list of np.ndarray contour point arrays
            confidence : float 0.0-1.0 smoothed smoke confidence
            mask       : np.ndarray 2D binary mask (uint8, values 0 or 255)
        """
        h, w = frame.shape[:2]

        if self._frame_area == 0.0:
            self._frame_area = float(h * w)

        # -- STEP 1: Gaussian blur -----------------------------------------
        #
        # We blur BEFORE everything else. This is deliberate.
        # The blurred copy is what MOG2 sees -- so it learns a blurred
        # background model. When smoke arrives, the blurred smoke also
        # appears as a large soft foreground region, which is exactly what
        # we want to detect.
        #
        # cv2.GaussianBlur applies a weighted average where pixels closer
        # to the centre contribute more than distant pixels (bell curve shape).
        # sigmaX=0 tells OpenCV to auto-calculate the sigma from kernel size.
        blurred = cv2.GaussianBlur(frame, self._blur_k, sigmaX=0)

        # -- STEP 2: MOG2 background subtraction -> motion mask -----------
        #
        # apply() updates the background model with this frame AND returns
        # a binary mask where:
        #   255 = this pixel is DIFFERENT from the learned background (moving)
        #     0 = this pixel matches the background (static)
        #
        # learningRate=-1 means MOG2 auto-adjusts its learning speed based
        # on the history parameter set at construction time. You can pass
        # a value like 0.001 to slow learning (useful for stable cameras)
        # or 0.01 to speed it up (useful for cameras that shift/vibrate).
        motion_mask = self._mog2.apply(blurred, learningRate=-1)

        # -- STEP 3: HSV grey-tone colour mask ----------------------------
        #
        # Convert the ORIGINAL (non-blurred) frame to HSV for colour analysis.
        # We use the original here because we want accurate colour reading --
        # blurring can shift hue/saturation values near edges.
        #
        # SMOKE COLOUR PROFILE in HSV:
        #   H (Hue):        0-180  -- doesn't matter, smoke has no dominant hue
        #   S (Saturation): 0-55   -- LOW saturation = grey/white, not colourful
        #   V (Value):      140-255 -- HIGH brightness = light grey, not dark soot
        #
        # This range captures:
        #   - Light grey smoke    (S~20,  V~200) ✓
        #   - White/cream smoke   (S~5,   V~240) ✓
        #   - Yellowish haze      (S~45,  V~180) ✓
        # But rejects:
        #   - Dark grey/black soot (V<140)        ✗
        #   - Colourful objects    (S>55)          ✗
        #   - Dark shadows         (V<140)         ✗
        hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grey_mask  = cv2.inRange(hsv, self._lower_grey, self._upper_grey)

        # -- STEP 4: Combine motion mask AND colour mask ------------------
        #
        # cv2.bitwise_and: output pixel = 255 ONLY if BOTH inputs are 255
        #
        # Visualise it as a Venn diagram:
        #
        #   Motion mask circle:  everything that moved this frame
        #   Colour mask circle:  everything that is grey/white this frame
        #   INTERSECTION:        things that are BOTH moving AND grey/white
        #                        = smoke (or steam, which we also want to catch)
        #
        # This single operation eliminates:
        #   - Moving people/vehicles (coloured, high saturation -> not in grey mask)
        #   - Static white walls (not moving -> not in motion mask)
        #   - Static grey machinery (not moving -> not in motion mask)
        combined_mask = cv2.bitwise_and(motion_mask, grey_mask)

        # -- STEP 4b: Morphological closing --------------------------------
        # The AND operation can fragment a smoke cloud into many small pieces
        # (wherever the motion and colour masks don't perfectly overlap).
        # Closing reconnects these fragments into coherent smoke regions.
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self._kernel)

        # -- STEP 5: Find contours ----------------------------------------
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # -- STEP 6: Filter by minimum area ------------------------------
        boxes         = []
        kept_contours = []
        total_smoke_px = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((x, y, bw, bh))
            kept_contours.append(cnt)
            total_smoke_px += int(area)

        # -- STEP 7: Normalise to confidence score 0.0-1.0 ---------------
        raw_conf = (
            min(total_smoke_px / self._frame_area, 1.0)
            if self._frame_area > 0 else 0.0
        )

        # -- STEP 8: EMA smoothing ----------------------------------------
        smooth_conf = self._smoother.update(raw_conf)

        return {
            "boxes"      : boxes,
            "contours"   : kept_contours,
            "confidence" : smooth_conf,
            "mask"       : cleaned,
        }

    def reset(self) -> None:
        """
        Reset the detector to a clean state.

        This recreates the MOG2 model from scratch (erasing all learned
        background knowledge) and resets the EMA smoother.
        Call this whenever you switch to a new camera or video file.
        """
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=CFG["SMOKE_MOG2_HISTORY"],
            varThreshold=CFG["SMOKE_MOG2_THRESH"],
            detectShadows=False,
        )
        self._smoother.reset()
        self._frame_area = 0.0



