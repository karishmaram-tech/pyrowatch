"""
================================================================================
  tests/test_step2.py
  Verification tests for FireDetector
  Run with: python tests\test_step2.py
================================================================================
"""

import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; RESET = "\033[0m"; BOLD = "\033[1m"

def passed(msg): print(f"  {GREEN}✓ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}✗ FAILED{RESET}  {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")


# ── Synthetic frame factory ──────────────────────────────────────────────────
def make_frame(width=640, height=480, bgr_fill=(0, 0, 0)):
    """Create a solid-colour BGR frame for controlled testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr_fill
    return frame

def paint_rect(frame, x, y, w, h, bgr_colour):
    """Paint a solid rectangle onto a frame in-place."""
    frame[y:y+h, x:x+w] = bgr_colour
    return frame


# ════════════════════════════════════════════════════════════════════════════
header("TEST 1 — FireDetector imports and initialises cleanly")
# ════════════════════════════════════════════════════════════════════════════
try:
    from ifsd.detectors.fire import FireDetector
    passed("FireDetector imported from ifsd.detectors.fire")
except ImportError as e:
    failed(f"Import failed: {e}")

try:
    det = FireDetector()
    passed("FireDetector() constructed without error")
except Exception as e:
    failed(f"Constructor raised: {e}")


# ════════════════════════════════════════════════════════════════════════════
header("TEST 2 — Completely black frame returns zero detections")
# ════════════════════════════════════════════════════════════════════════════
# A black frame has no colour information at all. The detector must return
# empty lists and a confidence of 0.0.
frame_black = make_frame(bgr_fill=(0, 0, 0))
result = det.detect(frame_black)

print(f"  boxes      : {result['boxes']}")
print(f"  confidence : {result['confidence']:.4f}")
print(f"  mask pixels: {int(result['mask'].sum())}")

if len(result["boxes"]) == 0:
    passed("No detections on pure black frame")
else:
    failed(f"Got {len(result['boxes'])} detections on a black frame — should be 0")

if result["confidence"] < 0.001:
    passed("Confidence is near-zero on black frame")
else:
    failed(f"Confidence should be ~0 on black frame, got {result['confidence']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
header("TEST 3 — Bright orange-red region triggers a detection")
# ════════════════════════════════════════════════════════════════════════════
# We paint a 120x120 pixel bright orange rectangle on a black frame.
# BGR (0, 80, 220) = orange in BGR = a typical flame colour.
# This is large enough (14400 px²) to clear the FIRE_MIN_AREA=800 filter.
det2 = FireDetector()
frame_fire = make_frame(640, 480)
paint_rect(frame_fire, x=200, y=150, w=120, h=120, bgr_colour=(0, 80, 220))

result_fire = det2.detect(frame_fire)

print(f"  boxes found : {len(result_fire['boxes'])}")
print(f"  confidence  : {result_fire['confidence']:.4f}")
if result_fire["boxes"]:
    x, y, w, h = result_fire["boxes"][0]
    print(f"  first box   : x={x}, y={y}, w={w}, h={h}")

if len(result_fire["boxes"]) >= 1:
    passed("At least one fire blob detected in orange frame")
else:
    failed("Expected at least 1 detection for bright orange region, got 0")

if result_fire["confidence"] > 0.005:
    passed(f"Confidence is positive ({result_fire['confidence']:.4f})")
else:
    failed(f"Confidence too low for a visible flame region: {result_fire['confidence']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
header("TEST 4 — Dark red region is rejected by the V-channel filter")
# ════════════════════════════════════════════════════════════════════════════
# A dark, dull red (e.g. a painted wall, a stop sign in shadow) has
# the right hue but fails the brightness (Value) check.
# BGR (0, 0, 80) is dark red — low V in HSV space.
det3 = FireDetector()
frame_dark = make_frame(640, 480)
paint_rect(frame_dark, x=200, y=150, w=120, h=120, bgr_colour=(0, 0, 80))

result_dark = det3.detect(frame_dark)

print(f"  boxes found : {len(result_dark['boxes'])}")
print(f"  confidence  : {result_dark['confidence']:.4f}")

# Convert one corner pixel to HSV so we can print its actual V value
pixel_bgr  = np.uint8([[[0, 0, 80]]])
pixel_hsv  = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
print(f"  pixel HSV   : H={pixel_hsv[0,0,0]}, S={pixel_hsv[0,0,1]}, V={pixel_hsv[0,0,2]}")
print(f"  (V must be >= {80} from CFG FIRE_MIN_V_MEAN=160 to pass brightness filter)")

if len(result_dark["boxes"]) == 0:
    passed("Dark red region correctly REJECTED by brightness filter")
else:
    failed(f"Dark red region should be rejected, but {len(result_dark['boxes'])} box(es) detected")


# ════════════════════════════════════════════════════════════════════════════
header("TEST 5 — Return dict has the correct structure and types")
# ════════════════════════════════════════════════════════════════════════════
det4   = FireDetector()
result = det4.detect(make_frame())   # black frame — safe baseline

checks = [
    ("boxes is a list",      isinstance(result["boxes"],      list)),
    ("contours is a list",   isinstance(result["contours"],   list)),
    ("confidence is float",  isinstance(result["confidence"], float)),
    ("mask is np.ndarray",   isinstance(result["mask"],       np.ndarray)),
    ("mask is 2D",           result["mask"].ndim == 2),
    ("confidence in [0,1]",  0.0 <= result["confidence"] <= 1.0),
]

for label, ok in checks:
    if ok:
        passed(label)
    else:
        failed(label)


# ════════════════════════════════════════════════════════════════════════════
header("TEST 6 — EMA smoothing dampens a sudden spike")
# ════════════════════════════════════════════════════════════════════════════
# Feed 20 black frames (confidence ≈ 0), then one fire frame.
# The EMA-smoothed confidence after the fire frame must be LESS than
# the raw confidence would be, proving the smoother is dampening spikes.
det5       = FireDetector()
blank      = make_frame(640, 480)

for _ in range(20):
    det5.detect(blank)   # warm up the smoother near 0

# Now inject one bright fire frame
fire_frame = make_frame(640, 480)
paint_rect(fire_frame, 100, 100, 200, 200, bgr_colour=(0, 60, 240))
result_spike = det5.detect(fire_frame)

# Raw confidence would be 200*200 / (640*480) ≈ 0.130
raw_approx = (200 * 200) / (640 * 480)
smoothed   = result_spike["confidence"]

print(f"  raw approx confidence  : {raw_approx:.4f}")
print(f"  EMA smoothed confidence: {smoothed:.4f}")
print(f"  (smoothed must be < raw — EMA is dampening the spike)")

if smoothed < raw_approx:
    passed(f"EMA correctly dampens spike: {smoothed:.4f} < {raw_approx:.4f}")
else:
    failed(f"EMA not dampening: smoothed {smoothed:.4f} >= raw {raw_approx:.4f}")


# ════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{GREEN}{'═'*60}")
print(f"  ALL TESTS PASSED — Step 2 FireDetector is verified!")
print(f"  You are ready to move on to Step 3: SmokeDetector")
print(f"{'═'*60}{RESET}\n")



