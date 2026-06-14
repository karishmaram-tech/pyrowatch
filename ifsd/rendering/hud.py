"""
================================================================================
  PyroWatch/rendering/hud.py
  HUDRenderer -- Cyberpunk Heads-Up Display for PYROWATCH HAZARD MONITORing
================================================================================
  All drawing operations use OpenCV on numpy arrays.
  No external GUI framework required.

  COORDINATE SYSTEM REMINDER:
    OpenCV uses (x, y) where x increases RIGHT, y increases DOWN.
    Array indexing uses [row, col] = [y, x] -- the opposite order.
    rectangle/putText functions take (x, y) tuples.
    Array slicing uses [y1:y2, x1:x2] notation.

  ALPHA BLENDING FORMULA:
    output = alpha * overlay + (1 - alpha) * background
    alpha=0.0 -> fully transparent (background shows through)
    alpha=1.0 -> fully opaque (overlay completely replaces background)
================================================================================
"""

import cv2
import math
import time
import numpy as np

from ifsd.config import CFG
from ifsd.analytics.risk import (
    RISK_CLEAR, RISK_CAUTION, RISK_WARNING, RISK_CRITICAL,
    risk_colour, risk_index
)


class HUDRenderer:
    """
    Renders all HUD elements onto a BGR video frame in-place.

    HOW TO USE:
        hud = HUDRenderer()

        # inside your frame loop, after detections are ready:
        canvas = hud.render(
            frame        = frame,
            fire_result  = fire_det.detect(frame),
            smoke_result = smoke_det.detect(frame),
            scene_result = scene_det.detect(frame),
            risk_tier    = tier,
            risk_score   = score,
            fps          = fps_counter.fps,
            latency      = fps_counter.latency,
        )
        cv2.imshow("PyroWatch", canvas)
    """

    def __init__(self) -> None:
        self._hud_alpha    = CFG["HUD_ALPHA"]
        self._scan_alpha   = CFG["HUD_SCAN_ALPHA"]
        self._scan_spacing = CFG["HUD_SCAN_SPACING"]
        self._glow_layers  = CFG["HUD_GLOW_LAYERS"]
        self._blink_hz     = CFG["HUD_BANNER_BLINK_HZ"]
        self._bg_col       = CFG["COL_HUD_BG"]
        self._text_col     = CFG["COL_TEXT"]

        self._font      = cv2.FONT_HERSHEY_SIMPLEX
        self._font_mono = cv2.FONT_HERSHEY_DUPLEX

        # Scanline overlay is built lazily on first render call
        self._scanline_overlay = None
        self._last_frame_shape = None

    # =========================================================================
    # LOW-LEVEL DRAWING HELPERS
    # =========================================================================

    def _alpha_rect(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        colour: tuple,
        alpha: float,
    ) -> None:
        """
        Draw a filled rectangle with alpha transparency onto canvas in-place.

        Technique:
          1. Extract the Region of Interest (ROI) from the canvas
          2. Fill a copy of the ROI with the solid colour
          3. Blend back:  canvas_roi = alpha*solid + (1-alpha)*original_roi
        """
        img_h, img_w = canvas.shape[:2]
        x1 = max(0, x);       y1 = max(0, y)
        x2 = min(img_w, x+w); y2 = min(img_h, y+h)
        if x2 <= x1 or y2 <= y1:
            return

        roi   = canvas[y1:y2, x1:x2]
        solid = roi.copy()
        solid[:] = colour
        cv2.addWeighted(solid, alpha, roi, 1.0 - alpha, 0, roi)
        canvas[y1:y2, x1:x2] = roi

    def _text(
        self,
        canvas: np.ndarray,
        text: str,
        x: int, y: int,
        colour: tuple  = None,
        scale: float   = 0.55,
        thickness: int = 1,
        shadow: bool   = True,
    ) -> None:
        """Draw text with an optional dark drop-shadow for readability."""
        if colour is None:
            colour = self._text_col
        if shadow:
            cv2.putText(canvas, text, (x+1, y+1),
                        self._font, scale, (10, 10, 10), thickness + 1,
                        cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y),
                    self._font, scale, colour, thickness, cv2.LINE_AA)

    def _glow_rect(
        self,
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        colour: tuple,
        thickness: int = 2,
    ) -> None:
        """
        Draw a multi-layered rectangle to simulate a neon glow effect.

        Draws HUD_GLOW_LAYERS concentric rectangles, each one pixel larger
        than the last, with halving opacity per layer:
          Layer 0 (inner):  alpha=1.00, thickness=base
          Layer 1:          alpha=0.50, thickness=base+1
          Layer 2 (outer):  alpha=0.25, thickness=base+2
        """
        img_h, img_w = canvas.shape[:2]

        for i in range(self._glow_layers):
            expand      = i
            lx          = max(0, x - expand)
            ly          = max(0, y - expand)
            lx2         = min(img_w - 1, x + w + expand)
            ly2         = min(img_h - 1, y + h + expand)
            layer_alpha = 1.0 / (2 ** i)
            layer_thick = thickness + i

            overlay = canvas.copy()
            cv2.rectangle(overlay, (lx, ly), (lx2, ly2), colour, layer_thick)
            cv2.addWeighted(overlay, layer_alpha, canvas,
                            1.0 - layer_alpha, 0, canvas)

    def _build_scanline_overlay(self, h: int, w: int) -> np.ndarray:
        """
        Build a static horizontal stripe pattern for the scanline effect.
        Every other band of SCAN_SPACING pixels is set to a dark value.
        Built once and reused every frame.
        """
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, self._scan_spacing * 2):
            overlay[y : y + self._scan_spacing] = (15, 15, 15)
        return overlay

    # =========================================================================
    # MID-LEVEL ELEMENT RENDERERS
    # =========================================================================

    def draw_fire_blobs(self, canvas: np.ndarray, fire_result: dict) -> None:
        """Draw semi-transparent fire hazard zones and contour outlines."""
        if not fire_result["boxes"]:
            return

        col_fire = CFG["COL_FIRE"]

        for x, y, w, h in fire_result["boxes"]:
            self._alpha_rect(canvas, x, y, w, h, col_fire, alpha=0.20)
            self._glow_rect(canvas, x, y, w, h, col_fire, thickness=2)
            label = f"FIRE {fire_result['confidence']:.0%}"
            self._text(canvas, label, x + 4, max(y - 8, 14),
                       colour=col_fire, scale=0.5)

        if fire_result["contours"]:
            cv2.drawContours(canvas, fire_result["contours"], -1,
                             (0, 220, 255), 1, cv2.LINE_AA)

    def draw_smoke_blobs(self, canvas: np.ndarray, smoke_result: dict) -> None:
        """Draw semi-transparent smoke hazard zones."""
        if not smoke_result["boxes"]:
            return

        col_smoke = CFG["COL_SMOKE"]

        for x, y, w, h in smoke_result["boxes"]:
            self._alpha_rect(canvas, x, y, w, h, col_smoke, alpha=0.15)
            self._glow_rect(canvas, x, y, w, h, col_smoke, thickness=1)
            label = f"SMOKE {smoke_result['confidence']:.0%}"
            self._text(canvas, label, x + 4, max(y - 8, 14),
                       colour=col_smoke, scale=0.5)

    def draw_scene_objects(
        self, canvas: np.ndarray, scene_result: list
    ) -> None:
        """Draw YOLO detection boxes for people and vehicles."""
        for det in scene_result:
            x, y, w, h = det["box"]
            label       = det["label"]
            conf        = det["conf"]
            col         = (CFG["COL_PERSON"] if label == "PERSON"
                           else CFG["COL_VEHICLE"])

            cv2.rectangle(canvas, (x, y), (x+w, y+h), col, 2)

            badge_text    = f"{label} {conf:.0%}"
            (tw, th), _   = cv2.getTextSize(badge_text, self._font, 0.45, 1)
            self._alpha_rect(canvas, x, y - th - 8, tw + 8, th + 6,
                             self._bg_col, alpha=0.75)
            self._text(canvas, badge_text, x + 4, y - 6,
                       colour=col, scale=0.45)

    def draw_top_bar(
        self,
        canvas: np.ndarray,
        fps: float,
        latency: float,
        risk_tier: str,
        risk_score: float,
    ) -> None:
        """
        Full-width telemetry bar at the top of the frame.
        Left: system title  |  Centre: FPS + latency  |  Right: risk tier
        """
        h, w  = canvas.shape[:2]
        bar_h = 32

        self._alpha_rect(canvas, 0, 0, w, bar_h, self._bg_col, alpha=0.80)

        self._text(canvas, "PyroWatch // PYROWATCH HAZARD MONITOR",
                   10, 20, colour=(100, 200, 255), scale=0.50)

        tele = f"FPS {fps:5.1f}   LAT {latency:5.1f}ms"
        (tw, _), _ = cv2.getTextSize(tele, self._font, 0.48, 1)
        self._text(canvas, tele, (w - tw) // 2, 20,
                   colour=self._text_col, scale=0.48)

        risk_col   = risk_colour(risk_tier)
        tier_label = f"RISK: {risk_tier}  W={risk_score:.3f}"
        (rw, _), _ = cv2.getTextSize(tier_label, self._font, 0.50, 1)
        self._text(canvas, tier_label, w - rw - 10, 20,
                   colour=risk_col, scale=0.50)

        cv2.line(canvas, (0, bar_h), (w, bar_h), (60, 60, 80), 1)

    def draw_side_panel(
        self,
        canvas: np.ndarray,
        fire_conf: float,
        smoke_conf: float,
        risk_tier: str,
        risk_score: float,
        scene_result: list,
    ) -> None:
        """
        Status panel on the right side with confidence bars and scene counts.
        """
        h, w      = canvas.shape[:2]
        panel_w   = 200
        panel_x   = w - panel_w - 8
        panel_y   = 42
        panel_h   = 220
        padding   = 10
        bar_h     = 12
        bar_max_w = panel_w - 2 * padding - 55

        self._alpha_rect(canvas, panel_x, panel_y, panel_w, panel_h,
                         self._bg_col, alpha=0.75)
        cv2.rectangle(canvas,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (60, 60, 90), 1)

        self._text(canvas, "HAZARD STATUS",
                   panel_x + padding, panel_y + 18,
                   colour=(150, 180, 255), scale=0.45)
        cv2.line(canvas,
                 (panel_x + padding, panel_y + 24),
                 (panel_x + panel_w - padding, panel_y + 24),
                 (60, 60, 90), 1)

        def _bar(label, value, colour, y_offset):
            y_pos  = panel_y + y_offset
            fill_w = int(bar_max_w * min(value, 1.0))
            self._text(canvas, label, panel_x + padding, y_pos + bar_h - 2,
                       colour=self._text_col, scale=0.38)
            track_x = panel_x + padding + 48
            cv2.rectangle(canvas,
                          (track_x, y_pos),
                          (track_x + bar_max_w, y_pos + bar_h),
                          (40, 40, 50), cv2.FILLED)
            if fill_w > 0:
                cv2.rectangle(canvas,
                              (track_x, y_pos),
                              (track_x + fill_w, y_pos + bar_h),
                              colour, cv2.FILLED)
            self._text(canvas, f"{value:.0%}",
                       track_x + bar_max_w + 4, y_pos + bar_h - 2,
                       colour=colour, scale=0.38)

        _bar("FIRE",  fire_conf,  CFG["COL_FIRE"],         34)
        _bar("SMOKE", smoke_conf, CFG["COL_SMOKE"],         58)
        _bar("RISK",  risk_score, risk_colour(risk_tier),   82)

        cv2.line(canvas,
                 (panel_x + padding, panel_y + 106),
                 (panel_x + panel_w - padding, panel_y + 106),
                 (60, 60, 90), 1)
        self._text(canvas, f"TIER: {risk_tier}",
                   panel_x + padding, panel_y + 124,
                   colour=risk_colour(risk_tier), scale=0.50)

        counts = {}
        for det in scene_result:
            counts[det["label"]] = counts.get(det["label"], 0) + 1

        cv2.line(canvas,
                 (panel_x + padding, panel_y + 132),
                 (panel_x + panel_w - padding, panel_y + 132),
                 (60, 60, 90), 1)
        self._text(canvas, "SCENE OBJECTS",
                   panel_x + padding, panel_y + 148,
                   colour=(150, 180, 255), scale=0.38)

        y_obj = panel_y + 164
        if counts:
            for label, count in counts.items():
                col = (CFG["COL_PERSON"] if label == "PERSON"
                       else CFG["COL_VEHICLE"])
                self._text(canvas, f"  {label}: {count}",
                           panel_x + padding, y_obj, colour=col, scale=0.40)
                y_obj += 16
        else:
            self._text(canvas, "  none detected",
                       panel_x + padding, y_obj,
                       colour=(80, 80, 100), scale=0.38)

    def draw_danger_banner(
        self,
        canvas: np.ndarray,
        risk_tier: str,
    ) -> None:
        """
        Animated flashing banner at the bottom for WARNING and CRITICAL tiers.

        Alpha is driven by a sine wave:
          alpha = 0.45 + 0.35 * sin(2*pi*hz*t)
          Oscillates smoothly between 0.10 and 0.80 at hz cycles per second.
        """
        if risk_tier not in (RISK_WARNING, RISK_CRITICAL):
            return

        h, w     = canvas.shape[:2]
        banner_h = 36
        banner_y = h - banner_h

        t     = time.time()
        phase = math.sin(2 * math.pi * self._blink_hz * t)
        alpha = 0.45 + 0.35 * phase

        self._alpha_rect(canvas, 0, banner_y, w, banner_h,
                         risk_colour(risk_tier), alpha=alpha)

        if risk_tier == RISK_CRITICAL:
            msg = "!! CRITICAL HAZARD DETECTED -- EVACUATE IMMEDIATELY !!"
        else:
            msg = "WARNING -- HAZARD DETECTED -- INITIATE RESPONSE PROTOCOL"

        (tw, th), _ = cv2.getTextSize(msg, self._font, 0.55, 2)
        tx = (w - tw) // 2
        ty = banner_y + (banner_h + th) // 2

        cv2.putText(canvas, msg, (tx+2, ty+2),
                    self._font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, msg, (tx, ty),
                    self._font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_scanlines(self, canvas: np.ndarray) -> None:
        """
        Apply vintage CRT scanline overlay.
        Pattern is built once and cached; rebuilt only if frame size changes.
        """
        h, w = canvas.shape[:2]

        if (self._scanline_overlay is None or
                self._last_frame_shape != (h, w)):
            self._scanline_overlay = self._build_scanline_overlay(h, w)
            self._last_frame_shape = (h, w)

        cv2.addWeighted(
            self._scanline_overlay, self._scan_alpha,
            canvas, 1.0 - self._scan_alpha,
            0, canvas
        )

    # =========================================================================
    # TOP-LEVEL RENDER METHOD
    # =========================================================================

    def render(
        self,
        frame        : np.ndarray,
        fire_result  : dict,
        smoke_result : dict,
        scene_result : list,
        risk_tier    : str,
        risk_score   : float,
        fps          : float = 0.0,
        latency      : float = 0.0,
    ) -> np.ndarray:
        """
        Compose all HUD layers onto a copy of the frame and return the canvas.

        Drawing order (bottom to top):
          1. Fire blobs
          2. Smoke blobs
          3. Scene object boxes
          4. Top telemetry bar
          5. Side status panel
          6. Danger banner
          7. Scanlines (topmost)

        The original frame is never modified.
        """
        canvas     = frame.copy()
        fire_conf  = fire_result["confidence"]
        smoke_conf = smoke_result["confidence"]

        self.draw_fire_blobs(canvas, fire_result)
        self.draw_smoke_blobs(canvas, smoke_result)
        self.draw_scene_objects(canvas, scene_result)
        self.draw_top_bar(canvas, fps, latency, risk_tier, risk_score)
        self.draw_side_panel(canvas, fire_conf, smoke_conf,
                             risk_tier, risk_score, scene_result)
        self.draw_danger_banner(canvas, risk_tier)
        self.draw_scanlines(canvas)

        return canvas



