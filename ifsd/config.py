"""
================================================================================
  PyroWatch/config.py  -- TUNED v2 for real aerial industrial fire footage
  Changes from v1:
    - YOLO_SKIP_FRAMES 4->9  (run YOLO once per 10 frames -> big FPS boost)
    - SMOKE_MOG2_THRESH 20->40  (less sensitive, stops rooftop false positives)
    - SMOKE_MIN_AREA 800->2500  (small grey patches on roofs filtered out)
    - SMOKE_HSV_UPPER V 220->180 (tighter brightness band, rejects bright roofs)
================================================================================
"""

import numpy as np

CFG: dict = {

    # ── Input / Output ────────────────────────────────────────────────────────
    "RESOLUTION"      : (1280, 720),
    "FPS_CAP"         : 30,
    "OUTPUT_FOURCC"   : "mp4v",
    "OUTPUT_EXT"      : ".mp4",

    # ── YOLOv8 ───────────────────────────────────────────────────────────────
    "YOLO_MODEL"      : "yolov8n.pt",
    "YOLO_CONF"       : 0.40,
    "YOLO_IOU"        : 0.45,
    # Run YOLO once every 10 frames -- fire/smoke still run every frame
    # This should push FPS from 0.4 up to 8-15 on CPU
    "YOLO_SKIP_FRAMES": 9,
    "YOLO_TARGET_IDS" : {0: "PERSON", 2: "CAR", 5: "BUS", 7: "TRUCK"},

    # ── Fire Detection ────────────────────────────────────────────────────────
    "FIRE_HSV_LOWER_A": np.array([0,   100, 100], dtype=np.uint8),
    "FIRE_HSV_UPPER_A": np.array([18,  255, 255], dtype=np.uint8),
    "FIRE_HSV_LOWER_B": np.array([160, 100, 100], dtype=np.uint8),
    "FIRE_HSV_UPPER_B": np.array([179, 255, 255], dtype=np.uint8),
    "FIRE_HSV_LOWER_C": np.array([19,   80, 180], dtype=np.uint8),
    "FIRE_HSV_UPPER_C": np.array([35,  255, 255], dtype=np.uint8),

    "FIRE_MIN_AREA"     : 300,
    "FIRE_MIN_V_MEAN"   : 130,
    "FIRE_MORPH_CLOSE_K": 7,
    "FIRE_MORPH_OPEN_K" : 3,
    "FIRE_EMA_ALPHA"    : 0.35,

    # ── Smoke Detection ───────────────────────────────────────────────────────
    # Key fixes:
    #   MOG2_THRESH 20->40  : only flag pixels that changed significantly
    #                         stops static grey rooftops triggering
    #   SMOKE_MIN_AREA 800->2500 : rooftop patches are small, real smoke is large
    #   V upper 220->175    : rooftops are bright (V>180), smoke is darker
    "SMOKE_HSV_LOWER"   : np.array([0,   0,  35], dtype=np.uint8),
    "SMOKE_HSV_UPPER"   : np.array([180, 55, 175], dtype=np.uint8),

    "SMOKE_BLUR_K"      : 21,
    "SMOKE_MIN_AREA"    : 2500,
    "SMOKE_MOG2_HISTORY": 400,
    "SMOKE_MOG2_THRESH" : 40,
    "SMOKE_EMA_ALPHA"   : 0.25,

    # ── Risk Engine ───────────────────────────────────────────────────────────
    "RISK_CAUTION_THRESH" : 0.005,
    "RISK_WARNING_THRESH" : 0.020,
    "RISK_CRITICAL_THRESH": 0.060,
    "RISK_FIRE_WEIGHT"    : 0.60,
    "RISK_SMOKE_WEIGHT"   : 0.40,

    # ── HUD ───────────────────────────────────────────────────────────────────
    "HUD_ALPHA"           : 0.55,
    "HUD_SCAN_ALPHA"      : 0.08,
    "HUD_SCAN_SPACING"    : 4,
    "HUD_GLOW_LAYERS"     : 3,
    "HUD_BANNER_BLINK_HZ" : 2.0,

    "COL_FIRE"    : (0,   100, 255),
    "COL_SMOKE"   : (200, 200, 200),
    "COL_PERSON"  : (0,   255, 140),
    "COL_VEHICLE" : (255, 200,   0),
    "COL_CLEAR"   : (0,   220,  80),
    "COL_CAUTION" : (0,   200, 255),
    "COL_WARNING" : (0,   140, 255),
    "COL_CRITICAL": (0,    40, 255),
    "COL_HUD_BG"  : (10,   10,  20),
    "COL_TEXT"    : (220, 220, 220),

    # ── Telemetry ─────────────────────────────────────────────────────────────
    "FPS_WINDOW"  : 30,
}



