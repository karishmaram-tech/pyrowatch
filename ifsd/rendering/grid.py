"""
================================================================================
  PyroWatch/rendering/grid.py
  GridRenderer -- 2x2 Multi-Camera HUD Grid Compositor
================================================================================
  Composites 4 individual camera tiles into a single 1280x720 canvas.

  LAYOUT:
    +------------------+------------------+
    |   CAM 1 (0,0)   |   CAM 2 (1,0)   |  <- row 0
    |   640x360        |   640x360        |
    +------------------+------------------+
    |   CAM 3 (0,1)   |   CAM 4 (1,1)   |  <- row 1
    |   640x360        |   640x360        |
    +------------------+------------------+

  Each tile has its own:
    - Fire/smoke detection overlays
    - Camera label and individual risk tier
    - Confidence bars
    - Danger border glow when WARNING/CRITICAL

  The master bar at the top shows:
    - Overall system FPS
    - Worst-case risk tier across all cameras
    - Total people and vehicles detected
================================================================================
"""

import cv2
import numpy as np
import time
import math

from ifsd.config       import CFG
from ifsd.analytics.risk import risk_colour, RISK_CLEAR, RISK_WARNING, RISK_CRITICAL


class GridRenderer:
    """
    Renders a 2x2 grid of camera feeds with per-tile HUD overlays.

    HOW TO USE:
        grid = GridRenderer(cam_labels=["CAM 1","CAM 2","CAM 3","CAM 4"])

        # inside your frame loop:
        canvas = grid.render(
            tiles = [
                {
                  "frame"       : frame1,
                  "fire_result" : fr1,
                  "smoke_result": sr1,
                  "scene_result": sc1,
                  "risk_tier"   : tier1,
                  "risk_score"  : score1,
                },
                ... (4 total)
            ],
            fps = fps_counter.fps,
        )
        cv2.imshow("PyroWatch Multi-Cam", canvas)
    """

    TILE_W = 640
    TILE_H = 360
    GRID_W = 1280   # 2 * TILE_W
    GRID_H = 720    # 2 * TILE_H

    def __init__(self, cam_labels: list[str] = None) -> None:
        self._labels    = cam_labels or [f"CAM {i+1}" for i in range(4)]
        self._font      = cv2.FONT_HERSHEY_SIMPLEX
        self._bg_col    = CFG["COL_HUD_BG"]
        self._text_col  = CFG["COL_TEXT"]
        self._blink_hz  = CFG["HUD_BANNER_BLINK_HZ"]

        # Tile positions: (col, row) -> (x_offset, y_offset) in pixels
        self._tile_origins = [
            (0,            0),            # CAM 1 top-left
            (self.TILE_W,  0),            # CAM 2 top-right
            (0,            self.TILE_H),  # CAM 3 bottom-left
            (self.TILE_W,  self.TILE_H),  # CAM 4 bottom-right
        ]

    # =========================================================================
    # TOP-LEVEL RENDER
    # =========================================================================

    def render(
        self,
        tiles : list[dict],
        fps   : float = 0.0,
    ) -> np.ndarray:
        """
        Composite all 4 camera tiles into a single 1280x720 canvas.

        Parameters
        ----------
        tiles : list of 4 dicts, each containing:
                  frame, fire_result, smoke_result,
                  scene_result, risk_tier, risk_score
        fps   : overall system FPS from FPSCounter

        Returns
        -------
        np.ndarray : 1280x720 BGR composite canvas
        """
        canvas = np.zeros((self.GRID_H, self.GRID_W, 3), dtype=np.uint8)

        # Find worst-case risk tier across all cameras
        tier_rank    = {"CLEAR":0,"CAUTION":1,"WARNING":2,"CRITICAL":3}
        worst_tier   = RISK_CLEAR
        worst_score  = 0.0
        total_people = 0
        total_veh    = 0

        for i, tile_data in enumerate(tiles[:4]):
            t = tile_data.get("risk_tier",   RISK_CLEAR)
            s = tile_data.get("risk_score",  0.0)
            sc= tile_data.get("scene_result",[])

            if tier_rank.get(t, 0) > tier_rank.get(worst_tier, 0):
                worst_tier  = t
                worst_score = s

            total_people += sum(1 for d in sc if d.get("label") == "PERSON")
            total_veh    += sum(1 for d in sc if d.get("label") != "PERSON"
                               and d.get("label") is not None)

            # Render individual tile and paste into canvas
            tile_canvas = self._render_tile(i, tile_data)
            ox, oy      = self._tile_origins[i]
            canvas[oy:oy+self.TILE_H, ox:ox+self.TILE_W] = tile_canvas

        # Draw grid divider lines
        cv2.line(canvas, (self.TILE_W, 0),
                 (self.TILE_W, self.GRID_H), (40, 40, 60), 2)
        cv2.line(canvas, (0, self.TILE_H),
                 (self.GRID_W, self.TILE_H), (40, 40, 60), 2)

        # Draw master top bar over everything
        self._draw_master_bar(
            canvas, fps, worst_tier, worst_score,
            total_people, total_veh
        )

        # Draw full-border glow on worst tile if CRITICAL
        if worst_tier == RISK_CRITICAL:
            self._draw_critical_border(canvas)

        return canvas

    # =========================================================================
    # TILE RENDERER
    # =========================================================================

    def _render_tile(self, idx: int, tile_data: dict) -> np.ndarray:
        """
        Render one 640x360 camera tile with its own HUD overlay.

        Steps:
          1. Resize the frame to tile resolution
          2. Draw fire blobs
          3. Draw smoke blobs
          4. Draw scene objects
          5. Draw tile header bar (cam label + risk tier)
          6. Draw mini confidence bars
          7. Draw danger border if WARNING/CRITICAL
        """
        frame        = tile_data.get("frame")
        fire_result  = tile_data.get("fire_result",  {"boxes":[],"contours":[],"confidence":0.0,"mask":None})
        smoke_result = tile_data.get("smoke_result", {"boxes":[],"contours":[],"confidence":0.0,"mask":None})
        scene_result = tile_data.get("scene_result", [])
        risk_tier    = tile_data.get("risk_tier",    RISK_CLEAR)
        risk_score   = tile_data.get("risk_score",   0.0)
        label        = self._labels[idx]

        # Handle missing/black frame
        if frame is None:
            tile = np.zeros((self.TILE_H, self.TILE_W, 3), dtype=np.uint8)
            self._text(tile, "NO SIGNAL", self.TILE_W//2 - 50,
                       self.TILE_H//2, colour=(60,60,80))
        else:
            tile = cv2.resize(frame, (self.TILE_W, self.TILE_H))

        # Scale factor from original (1280x720) to tile (640x360)
        sx = self.TILE_W / 1280.0
        sy = self.TILE_H / 720.0

        # ── Fire blobs ───────────────────────────────────────────────────
        col_fire = CFG["COL_FIRE"]
        for (x, y, w, h) in fire_result["boxes"]:
            tx = int(x*sx); ty = int(y*sy)
            tw = int(w*sx); th = int(h*sy)
            self._alpha_rect(tile, tx, ty, tw, th, col_fire, 0.18)
            cv2.rectangle(tile, (tx,ty), (tx+tw,ty+th), col_fire, 1)

        # Draw scaled contours
        if fire_result.get("contours"):
            scaled = []
            for cnt in fire_result["contours"]:
                sc_cnt = (cnt * [sx, sy]).astype(np.int32)
                scaled.append(sc_cnt)
            cv2.drawContours(tile, scaled, -1, (0,220,255), 1, cv2.LINE_AA)

        # ── Smoke blobs ──────────────────────────────────────────────────
        col_smoke = CFG["COL_SMOKE"]
        for (x, y, w, h) in smoke_result["boxes"]:
            tx = int(x*sx); ty = int(y*sy)
            tw = int(w*sx); th = int(h*sy)
            self._alpha_rect(tile, tx, ty, tw, th, col_smoke, 0.13)
            cv2.rectangle(tile, (tx,ty), (tx+tw,ty+th), col_smoke, 1)

        # ── Scene objects ────────────────────────────────────────────────
        for det in scene_result:
            x, y, w, h = det["box"]
            tx = int(x*sx); ty = int(y*sy)
            tw = int(w*sx); th = int(h*sy)
            col = (CFG["COL_PERSON"] if det["label"] == "PERSON"
                   else CFG["COL_VEHICLE"])
            cv2.rectangle(tile, (tx,ty), (tx+tw,ty+th), col, 1)

        # ── Tile header bar ──────────────────────────────────────────────
        self._alpha_rect(tile, 0, 0, self.TILE_W, 26,
                         self._bg_col, alpha=0.80)

        tier_col = risk_colour(risk_tier)
        self._text(tile, label, 6, 17,
                   colour=(100,200,255), scale=0.45)

        tier_text = f"{risk_tier}  W={risk_score:.3f}"
        (tw2,_),_ = cv2.getTextSize(tier_text, self._font, 0.40, 1)
        self._text(tile, tier_text,
                   self.TILE_W - tw2 - 6, 17,
                   colour=tier_col, scale=0.40)

        # ── Mini confidence bars ─────────────────────────────────────────
        self._draw_mini_bars(
            tile,
            fire_result["confidence"],
            smoke_result["confidence"],
        )

        # ── Danger border glow ───────────────────────────────────────────
        if risk_tier in (RISK_WARNING, RISK_CRITICAL):
            t_now   = time.time()
            phase   = math.sin(2 * math.pi * self._blink_hz * t_now)
            alpha   = 0.5 + 0.4 * phase
            thick   = 3 if risk_tier == RISK_CRITICAL else 2
            overlay = tile.copy()
            cv2.rectangle(overlay, (0,0),
                          (self.TILE_W-1, self.TILE_H-1),
                          tier_col, thick*3)
            cv2.addWeighted(overlay, alpha, tile,
                            1.0-alpha, 0, tile)
            cv2.rectangle(tile, (0,0),
                          (self.TILE_W-1, self.TILE_H-1),
                          tier_col, thick)

        return tile

    # =========================================================================
    # MASTER BAR
    # =========================================================================

    def _draw_master_bar(
        self,
        canvas      : np.ndarray,
        fps         : float,
        worst_tier  : str,
        worst_score : float,
        people      : int,
        vehicles    : int,
    ) -> None:
        """
        Draw the master telemetry bar at the very top of the 1280x720 canvas.
        Spans both columns. Sits on top of both top tiles.
        """
        bar_h = 28
        self._alpha_rect(canvas, 0, 0, self.GRID_W, bar_h,
                         (5, 5, 15), alpha=0.90)

        # Left: system title
        self._text(canvas,
                   "PyroWatch // MULTI-CAMERA PYROWATCH HAZARD MONITOR",
                   8, 18, colour=(100,200,255), scale=0.45)

        # Centre: FPS + object counts
        mid_text = (f"FPS {fps:4.1f}   "
                    f"PEOPLE {people}   VEHICLES {vehicles}")
        (tw,_),_ = cv2.getTextSize(mid_text, self._font, 0.40, 1)
        self._text(canvas, mid_text,
                   (self.GRID_W - tw)//2, 18,
                   colour=self._text_col, scale=0.40)

        # Right: worst-case risk
        risk_col  = risk_colour(worst_tier)
        risk_text = f"SYSTEM RISK: {worst_tier}  W={worst_score:.3f}"
        (rw,_),_ = cv2.getTextSize(risk_text, self._font, 0.45, 1)
        self._text(canvas, risk_text,
                   self.GRID_W - rw - 8, 18,
                   colour=risk_col, scale=0.45)

        cv2.line(canvas, (0, bar_h),
                 (self.GRID_W, bar_h), (40,40,60), 1)

    # =========================================================================
    # MINI CONFIDENCE BARS
    # =========================================================================

    def _draw_mini_bars(
        self,
        tile      : np.ndarray,
        fire_conf : float,
        smoke_conf: float,
    ) -> None:
        """
        Draw two tiny progress bars in the bottom-left of the tile.
        FIRE bar: orange.   SMOKE bar: grey.
        """
        bar_w   = 100
        bar_h   = 5
        x_start = 6
        y_fire  = self.TILE_H - 18
        y_smoke = self.TILE_H - 10

        for label, val, col, y in [
            ("F", fire_conf,  CFG["COL_FIRE"],  y_fire),
            ("S", smoke_conf, CFG["COL_SMOKE"], y_smoke),
        ]:
            # Track
            cv2.rectangle(tile,
                          (x_start+10, y),
                          (x_start+10+bar_w, y+bar_h),
                          (30,30,40), cv2.FILLED)
            # Fill
            fill = int(bar_w * min(val, 1.0))
            if fill > 0:
                cv2.rectangle(tile,
                              (x_start+10, y),
                              (x_start+10+fill, y+bar_h),
                              col, cv2.FILLED)
            # Label
            self._text(tile, label, x_start, y+bar_h,
                       colour=col, scale=0.28)
            # Percentage
            self._text(tile, f"{val:.0%}",
                       x_start+10+bar_w+3, y+bar_h,
                       colour=col, scale=0.28)

    # =========================================================================
    # CRITICAL BORDER
    # =========================================================================

    def _draw_critical_border(self, canvas: np.ndarray) -> None:
        """Draw a pulsing red border around the entire 1280x720 canvas."""
        t     = time.time()
        phase = math.sin(2 * math.pi * self._blink_hz * t)
        alpha = 0.4 + 0.4 * phase
        col   = CFG["COL_CRITICAL"]

        overlay = canvas.copy()
        for thickness in [6, 4, 2]:
            cv2.rectangle(overlay, (0,0),
                          (self.GRID_W-1, self.GRID_H-1),
                          col, thickness)
        cv2.addWeighted(overlay, alpha, canvas, 1.0-alpha, 0, canvas)

    # =========================================================================
    # LOW-LEVEL HELPERS
    # =========================================================================

    def _alpha_rect(self, canvas, x, y, w, h, colour, alpha):
        img_h, img_w = canvas.shape[:2]
        x1 = max(0,x);       y1 = max(0,y)
        x2 = min(img_w,x+w); y2 = min(img_h,y+h)
        if x2<=x1 or y2<=y1: return
        roi   = canvas[y1:y2, x1:x2]
        solid = roi.copy()
        solid[:] = colour
        cv2.addWeighted(solid, alpha, roi, 1.0-alpha, 0, roi)
        canvas[y1:y2, x1:x2] = roi

    def _text(self, canvas, text, x, y,
              colour=None, scale=0.45, thickness=1):
        if colour is None:
            colour = self._text_col
        cv2.putText(canvas, text, (x+1,y+1),
                    self._font, scale, (10,10,10),
                    thickness+1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x,y),
                    self._font, scale, colour,
                    thickness, cv2.LINE_AA)



