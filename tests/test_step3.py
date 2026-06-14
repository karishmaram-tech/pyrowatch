"""
================================================================================
  tests/test_step3.py
  Verification tests for SmokeDetector
  Run with: python tests\test_step3.py
================================================================================
"""

import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def passed(msg): print(f"  {GREEN}+ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}x FAILED{RESET}  {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


def make_frame(width=640, height=480, bgr_fill=(0, 0, 0)):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bgr_fill
    return frame

def paint_rect(frame, x, y, w, h, colour):
    frame[y:y+h, x:x+w] = colour
    return frame


# ============================================================================
header("TEST 1 -- SmokeDetector imports and constructs cleanly")
# ============================================================================

try:
    from ifsd.detectors.smoke import SmokeDetector
    passed("SmokeDetector imported from ifsd.detectors.smoke")
except ImportError as e:
    failed(f"Import failed: {e}")

try:
    det = SmokeDetector()
    passed("SmokeDetector() constructed without error")
except Exception as e:
    failed(f"Constructor raised: {e}")


# ============================================================================
header("TEST 2 -- Return dict has correct structure and types")
# ============================================================================

blank = make_frame()
result = det.detect(blank)

checks = [
    ("boxes is a list",      isinstance(result["boxes"],      list)),
    ("contours is a list",   isinstance(result["contours"],   list)),
    ("confidence is float",  isinstance(result["confidence"], float)),
    ("mask is np.ndarray",   isinstance(result["mask"],       np.ndarray)),
    ("mask is 2D",           result["mask"].ndim == 2),
    ("confidence in [0,1]",  0.0 <= result["confidence"] <= 1.0),
]
for label, ok in checks:
    if ok: passed(label)
    else:  failed(label)


# ============================================================================
header("TEST 3 -- Static scene returns near-zero confidence after warm-up")
# ============================================================================
#
# When nothing moves, MOG2 learns the background and the motion mask
# becomes empty. After enough frames the confidence should drop to ~0.
# We feed 60 identical grey frames -- enough for MOG2 to stabilise.

det2        = SmokeDetector()
static_grey = make_frame(bgr_fill=(180, 180, 180))   # static grey wall

for i in range(60):
    r = det2.detect(static_grey)

final_conf = det2.detect(static_grey)["confidence"]
print(f"  confidence after 61 identical frames: {final_conf:.4f}")
print(f"  (should be near 0.0 -- nothing is moving)")

if final_conf < 0.05:
    passed(f"Static scene confidence near zero ({final_conf:.4f})")
else:
    failed(f"Static scene confidence too high: {final_conf:.4f} (expected < 0.05)")


# ============================================================================
header("TEST 4 -- Moving bright-grey region triggers smoke detection")
# ============================================================================
#
# Strategy:
#   Phase 1 (50 frames): feed a dark background so MOG2 learns "dark = normal"
#   Phase 2 (1 frame):   inject a large bright-grey patch that is BOTH
#                        moving (different from dark background) AND
#                        grey-coloured (passes HSV colour filter)
#
# The patch colour BGR=(200,200,200) is neutral grey.
# In HSV: H~0, S~0, V~200 -- perfectly inside our smoke colour range.

det3       = SmokeDetector()
dark_bg    = make_frame(640, 480, bgr_fill=(30, 30, 30))   # dark background
smoke_frame = make_frame(640, 480, bgr_fill=(30, 30, 30))
paint_rect(smoke_frame, x=150, y=100, w=300, h=200, colour=(200, 200, 200))

# Verify the patch colour is actually inside our HSV smoke range
pixel_bgr = np.uint8([[[200, 200, 200]]])
pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
print(f"  smoke patch HSV: H={pixel_hsv[0,0,0]}, S={pixel_hsv[0,0,1]}, V={pixel_hsv[0,0,2]}")
print(f"  target range   : H 0-180, S 0-55, V 140-255")

for _ in range(50):
    det3.detect(dark_bg)   # teach MOG2 that "dark = background"

result_smoke = det3.detect(smoke_frame)
print(f"  boxes found    : {len(result_smoke['boxes'])}")
print(f"  confidence     : {result_smoke['confidence']:.4f}")

if result_smoke["confidence"] > 0.001:
    passed(f"Moving grey region detected as smoke (conf={result_smoke['confidence']:.4f})")
else:
    failed(f"Smoke region not detected -- confidence={result_smoke['confidence']:.4f}")


# ============================================================================
header("TEST 5 -- Colourful moving object does NOT trigger smoke")
# ============================================================================
#
# A moving red object passes the motion filter but FAILS the grey-tone
# colour filter (red has high saturation -- S is large, not near 0).
# The AND operation should produce an empty combined mask.

det4      = SmokeDetector()
dark_bg2  = make_frame(640, 480, bgr_fill=(30, 30, 30))
red_frame = make_frame(640, 480, bgr_fill=(30, 30, 30))
paint_rect(red_frame, x=150, y=100, w=300, h=200, colour=(0, 0, 220))  # bright red

pixel_red = np.uint8([[[0, 0, 220]]])
pixel_red_hsv = cv2.cvtColor(pixel_red, cv2.COLOR_BGR2HSV)
print(f"  red patch HSV  : H={pixel_red_hsv[0,0,0]}, S={pixel_red_hsv[0,0,1]}, V={pixel_red_hsv[0,0,2]}")
print(f"  S={pixel_red_hsv[0,0,1]} >> 55 threshold -- should FAIL colour filter")

for _ in range(50):
    det4.detect(dark_bg2)

result_red = det4.detect(red_frame)
print(f"  boxes found    : {len(result_red['boxes'])}")
print(f"  confidence     : {result_red['confidence']:.4f}")

if result_red["confidence"] < 0.02:
    passed(f"Colourful moving object correctly ignored (conf={result_red['confidence']:.4f})")
else:
    failed(f"Red object should not trigger smoke, got conf={result_red['confidence']:.4f}")


# ============================================================================
header("TEST 6 -- EMA slow decay: smoke lingers after source disappears")
# ============================================================================
#
# After smoke is detected, replacing it with a clean dark frame should NOT
# instantly drop the confidence to zero. The EMA alpha=0.25 means the
# smoother "remembers" 75% of the previous value each frame.
# After 1 clean frame, conf should still be meaningfully above 0.

det5      = SmokeDetector()
dark_bg3  = make_frame(640, 480, bgr_fill=(20, 20, 20))
smk_frame = make_frame(640, 480, bgr_fill=(20, 20, 20))
paint_rect(smk_frame, x=100, y=100, w=350, h=250, colour=(210, 210, 210))

# Warm up MOG2
for _ in range(50):
    det5.detect(dark_bg3)

# Inject smoke for several frames to build up the smoothed confidence
for _ in range(8):
    det5.detect(smk_frame)

smoke_conf = det5.detect(smk_frame)["confidence"]
print(f"  confidence WITH smoke   : {smoke_conf:.4f}")

# Now remove smoke -- replace with clean dark frame
after_conf = det5.detect(dark_bg3)["confidence"]
print(f"  confidence 1 frame AFTER: {after_conf:.4f}")
print(f"  (after_conf should still be > 0 due to EMA slow decay)")

if after_conf > 0.0:
    passed(f"EMA slow decay working: conf dropped from {smoke_conf:.4f} to {after_conf:.4f} (not zero)")
else:
    failed(f"EMA decay too fast: confidence hit zero instantly ({after_conf:.4f})")


# ============================================================================
header("TEST 7 -- reset() wipes MOG2 model and EMA smoother")
# ============================================================================

det6 = SmokeDetector()
test_frame = make_frame(640, 480, bgr_fill=(100, 100, 100))

for _ in range(30):
    det6.detect(test_frame)

before_reset = det6._smoother.value
det6.reset()
after_reset  = det6._smoother.value

print(f"  smoother value before reset: {before_reset:.4f}")
print(f"  smoother value after  reset: {after_reset:.4f}")

if after_reset == 0.0:
    passed("reset() correctly zeroed the EMA smoother")
else:
    failed(f"reset() did not zero smoother: got {after_reset:.4f}")

result_after = det6.detect(make_frame())
if isinstance(result_after["boxes"], list):
    passed("detector works normally after reset()")
else:
    failed("detector broken after reset()")


# ============================================================================
print(f"\n{BOLD}{GREEN}{'='*60}")
print(f"  ALL TESTS PASSED -- Step 3 SmokeDetector is verified!")
print(f"  You are ready to move on to Step 4: SceneDetector + RiskEngine")
print(f"{'='*60}{RESET}\n")



