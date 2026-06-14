"""
================================================================================
  tests/test_step4.py
  Verification tests for SceneDetector and Risk Classification Engine
  Run with: python tests\test_step4.py
================================================================================
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def passed(msg): print(f"  {GREEN}+ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}x FAILED{RESET}  {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


# ============================================================================
header("TEST 1 -- Risk module imports cleanly")
# ============================================================================

try:
    from ifsd.analytics.risk import (
        classify_risk, risk_colour, risk_index,
        RISK_CLEAR, RISK_CAUTION, RISK_WARNING, RISK_CRITICAL, RISK_LEVELS
    )
    passed("All risk symbols imported from ifsd.analytics.risk")
except ImportError as e:
    failed(f"Import failed: {e}")


# ============================================================================
header("TEST 2 -- classify_risk tier mapping (verified against formula)")
# ============================================================================
#
# Formula:  W = 0.60*fire + 0.40*smoke
# Thresholds (from CFG):
#   CLEAR    W <  0.15
#   CAUTION  W >= 0.15  and  W < 0.35
#   WARNING  W >= 0.35  and  W < 0.60
#   CRITICAL W >= 0.60
#
# Every expected value below was computed manually first:
#   fire=0.40 smoke=0.20 -> W = 0.60*0.40 + 0.40*0.20 = 0.240 + 0.080 = 0.320 -> CAUTION
#   fire=0.00 smoke=1.00 -> W = 0.60*0.00 + 0.40*1.00 = 0.000 + 0.400 = 0.400 -> WARNING
#   fire=0.60 smoke=0.00 -> W = 0.60*0.60 + 0.40*0.00 = 0.360 + 0.000 = 0.360 -> WARNING

test_cases = [
    # (fire,  smoke,  expected_tier,   W_manual,  description)
    (0.00,  0.00,  RISK_CLEAR,     0.000, "W=0.000  -> CLEAR"),
    (0.10,  0.05,  RISK_CLEAR,     0.080, "W=0.080  -> CLEAR    (below 0.15)"),
    (0.20,  0.10,  RISK_CAUTION,   0.160, "W=0.160  -> CAUTION  (0.15 <= W < 0.35)"),
    (0.40,  0.20,  RISK_CAUTION,   0.320, "W=0.320  -> CAUTION  (0.15 <= W < 0.35)"),
    (0.50,  0.25,  RISK_WARNING,   0.400, "W=0.400  -> WARNING  (0.35 <= W < 0.60)"),
    (0.50,  0.40,  RISK_WARNING,   0.460, "W=0.460  -> WARNING  (0.35 <= W < 0.60)"),
    (0.80,  0.60,  RISK_CRITICAL,  0.720, "W=0.720  -> CRITICAL (W >= 0.60)"),
    (1.00,  1.00,  RISK_CRITICAL,  1.000, "W=1.000  -> CRITICAL (maximum)"),
    (0.00,  1.00,  RISK_WARNING,   0.400, "W=0.400  -> WARNING  (smoke-only, W >= 0.35)"),
    (0.60,  0.00,  RISK_WARNING,   0.360, "W=0.360  -> WARNING  (fire-only,  W >= 0.35)"),
    (0.90,  0.00,  RISK_CRITICAL,  0.540, "W=0.540  -> WARNING  (fire=0.9 alone not critical)"),
]

# Recompute expected tier for last row -- 0.60*0.90=0.54 < 0.60 -> WARNING not CRITICAL
# Fix that row:
test_cases[-1] = (0.90, 0.00, RISK_WARNING, 0.540, "W=0.540  -> WARNING  (fire=0.9 alone not critical)")

all_ok = True
for fire, smoke, expected, w_manual, desc in test_cases:
    tier, score = classify_risk(fire, smoke)
    match = (tier == expected)
    sym   = "+" if match else "x"
    col   = GREEN if match else RED
    print(f"  {col}{sym}{RESET}  fire={fire:.2f} smoke={smoke:.2f}  "
          f"W={w_manual:.3f}  got={tier:<10} expected={expected:<10}  [{desc}]")
    if not match:
        all_ok = False

if all_ok:
    passed("All tier mapping cases correct")
else:
    failed("One or more tier mappings were wrong (see above)")


# ============================================================================
header("TEST 3 -- classify_risk returns correct weighted score")
# ============================================================================

tier, score = classify_risk(0.5, 0.25)
expected_w  = round(0.60 * 0.5 + 0.40 * 0.25, 6)
print(f"  fire=0.50, smoke=0.25")
print(f"  expected W = 0.60*0.5 + 0.40*0.25 = {expected_w}")
print(f"  returned W = {score}")

if abs(score - expected_w) < 1e-5:
    passed(f"Weighted score mathematically correct ({score})")
else:
    failed(f"Weighted score wrong: expected {expected_w}, got {score}")


# ============================================================================
header("TEST 4 -- Boundary values sit on correct side of each threshold")
# ============================================================================
#
# Testing the exact threshold values themselves is the most rigorous check.
# A value of exactly 0.35 must land in WARNING (not CAUTION).
# A value just below 0.35 must land in CAUTION.
#
# We reverse-engineer fire inputs that produce exact threshold W values.
# With smoke=0, W = 0.60*fire, so fire = W/0.60

boundary_cases = [
    # fire input such that W is exactly at or just around each threshold
    # (fire,            smoke, expected,      note)
    (0.15/0.60,        0.0,   RISK_CAUTION,  "W=exactly 0.15 -> CAUTION boundary"),
    (0.15/0.60 - 0.01, 0.0,   RISK_CLEAR,   "W just below 0.15 -> CLEAR"),
    (0.35/0.60,        0.0,   RISK_WARNING,  "W=exactly 0.35 -> WARNING boundary"),
    (0.35/0.60 - 0.01, 0.0,   RISK_CAUTION,  "W just below 0.35 -> CAUTION"),
    (0.60/0.60,        0.0,   RISK_CRITICAL, "W=exactly 0.60 -> CRITICAL boundary"),
    (0.60/0.60 - 0.01, 0.0,   RISK_WARNING,  "W just below 0.60 -> WARNING"),
]

all_ok = True
for fire, smoke, expected, note in boundary_cases:
    tier, score = classify_risk(fire, smoke)
    match = (tier == expected)
    sym   = "+" if match else "x"
    col   = GREEN if match else RED
    print(f"  {col}{sym}{RESET}  fire={fire:.4f} smoke={smoke:.2f}  "
          f"W={score:.4f}  got={tier:<10} [{note}]")
    if not match:
        all_ok = False

if all_ok:
    passed("All threshold boundary cases land on the correct side")
else:
    failed("Threshold boundary case(s) failed (see above)")


# ============================================================================
header("TEST 5 -- classify_risk clamps out-of-range inputs safely")
# ============================================================================

tier_over, score_over = classify_risk(2.0, -0.5)
print(f"  Input fire=2.0, smoke=-0.5 (invalid range)")
print(f"  Clamped result: tier={tier_over}, score={score_over}")

if 0.0 <= score_over <= 1.0:
    passed("Out-of-range inputs safely clamped to valid score")
else:
    failed(f"Clamping failed: score={score_over} outside [0, 1]")


# ============================================================================
header("TEST 6 -- risk_colour returns valid BGR tuples")
# ============================================================================

for tier in [RISK_CLEAR, RISK_CAUTION, RISK_WARNING, RISK_CRITICAL]:
    col = risk_colour(tier)
    ok  = (isinstance(col, tuple) and len(col) == 3
           and all(isinstance(c, int) and 0 <= c <= 255 for c in col))
    print(f"  {tier:<10} -> BGR{col}")
    if ok:
        passed(f"risk_colour({tier}) is a valid BGR tuple")
    else:
        failed(f"risk_colour({tier}) returned invalid value: {col}")


# ============================================================================
header("TEST 7 -- risk_index returns correct ordering")
# ============================================================================

indices = {t: risk_index(t) for t in [RISK_CLEAR, RISK_CAUTION, RISK_WARNING, RISK_CRITICAL]}
print(f"  Indices: {indices}")

order_ok = (indices[RISK_CLEAR]    == 0 and
            indices[RISK_CAUTION]  == 1 and
            indices[RISK_WARNING]  == 2 and
            indices[RISK_CRITICAL] == 3)

if order_ok:
    passed("Risk levels correctly ordered 0=CLEAR -> 3=CRITICAL")
else:
    failed(f"Risk ordering wrong: {indices}")

if risk_index(RISK_CRITICAL) > risk_index(RISK_WARNING):
    passed("CRITICAL > WARNING numeric comparison works")
else:
    failed("CRITICAL should have higher index than WARNING")


# ============================================================================
header("TEST 8 -- SceneDetector imports and constructs")
# ============================================================================

print("  Loading SceneDetector (may download yolov8n.pt on first run ~6MB)...")
try:
    from ifsd.analytics.scene import SceneDetector
    passed("SceneDetector imported from ifsd.analytics.scene")
except ImportError as e:
    failed(f"Import failed: {e}")

try:
    scene_det = SceneDetector()
    passed(f"SceneDetector() constructed (device: {scene_det._device.upper()})")
except Exception as e:
    failed(f"SceneDetector constructor raised: {e}")


# ============================================================================
header("TEST 9 -- SceneDetector returns a list on a blank frame")
# ============================================================================

blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
try:
    scene_result = scene_det.detect(blank_frame)
    if isinstance(scene_result, list):
        passed(f"detect() returned a list ({len(scene_result)} detections on blank frame)")
    else:
        failed(f"detect() should return a list, got {type(scene_result)}")
except Exception as e:
    failed(f"detect() raised an exception: {e}")


# ============================================================================
header("TEST 10 -- SceneDetector frame-skip caching works")
# ============================================================================

det_skip = SceneDetector()
det_skip._last_result = [
    {"box": (0,0,10,10), "label": "PERSON", "conf": 0.9, "center": (5,5)}
]

frame_a  = np.zeros((480, 640, 3), dtype=np.uint8)
result_a = det_skip.detect(frame_a)   # should return cache (skip frame)

print(f"  Cached result length (skipped frame): {len(result_a)}")
print(f"  Skip counter after skipped frame    : {det_skip._skip_counter}")

if len(result_a) == 1:
    passed("Frame-skip returned cached result on skipped frame")
else:
    failed(f"Expected cached length 1, got {len(result_a)}")


# ============================================================================
print(f"\n{BOLD}{GREEN}{'='*60}")
print(f"  ALL TESTS PASSED -- Step 4 is verified!")
print(f"  SceneDetector + RiskEngine are ready.")
print(f"  You are ready to move on to Step 5: HUDRenderer")
print(f"{'='*60}{RESET}\n")



