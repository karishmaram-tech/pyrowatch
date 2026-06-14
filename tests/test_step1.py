"""
================================================================================
  tests/test_step1.py
  Verification test for Step 1: config.py and utils.py
  Run with:  python tests/test_step1.py
================================================================================
"""

import sys
import os
import time

# ── Make sure Python can find the PyroWatch package ──────────────────────────────
# When you run this script from the project root folder, Python needs to know
# where to look for 'PyroWatch'. Adding the current directory to the search path
# ensures  "from ifsd.config import CFG"  works correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Colour helpers for readable terminal output ──────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def passed(msg): print(f"  {GREEN}✓ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}✗ FAILED{RESET}  {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'─'*60}{RESET}")


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — Can Python import our modules without errors?
# ════════════════════════════════════════════════════════════════════════════
header("TEST 1 — Module Imports")

try:
    from ifsd.config import CFG
    passed("PyroWatch.config imported successfully")
except ImportError as e:
    failed(f"Could not import ifsd.config: {e}")

try:
    from ifsd.utils import FPSCounter, ExpSmooth
    passed("PyroWatch.utils imported successfully (FPSCounter, ExpSmooth)")
except ImportError as e:
    failed(f"Could not import ifsd.utils: {e}")

try:
    import numpy as np
    passed(f"numpy available (version {np.__version__})")
except ImportError:
    failed("numpy not found — run: pip install numpy")

try:
    import cv2
    passed(f"opencv-python available (version {cv2.__version__})")
except ImportError:
    print(f"  {YELLOW}⚠ WARNING{RESET}  opencv-python not yet installed (needed for Steps 2+)")
    print(f"           Run: pip install opencv-python")


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — Does CFG contain every required key?
# ════════════════════════════════════════════════════════════════════════════
header("TEST 2 — CFG Required Keys")

REQUIRED_KEYS = [
    "RESOLUTION", "FPS_CAP",
    "FIRE_HSV_LOWER_A", "FIRE_HSV_UPPER_A",
    "FIRE_HSV_LOWER_B", "FIRE_HSV_UPPER_B",
    "FIRE_HSV_LOWER_C", "FIRE_HSV_UPPER_C",
    "FIRE_MIN_AREA", "FIRE_MIN_V_MEAN",
    "FIRE_EMA_ALPHA",
    "SMOKE_HSV_LOWER", "SMOKE_HSV_UPPER",
    "SMOKE_EMA_ALPHA",
    "RISK_FIRE_WEIGHT", "RISK_SMOKE_WEIGHT",
    "RISK_CAUTION_THRESH", "RISK_WARNING_THRESH", "RISK_CRITICAL_THRESH",
    "HUD_ALPHA", "FPS_WINDOW",
]

all_present = True
for key in REQUIRED_KEYS:
    if key not in CFG:
        failed(f"CFG is missing required key: '{key}'")
        all_present = False

if all_present:
    passed(f"All {len(REQUIRED_KEYS)} required CFG keys are present")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — Do the risk weights add up to exactly 1.0?
# ════════════════════════════════════════════════════════════════════════════
header("TEST 3 — CFG Risk Weight Integrity")

fire_w  = CFG["RISK_FIRE_WEIGHT"]
smoke_w = CFG["RISK_SMOKE_WEIGHT"]
total   = fire_w + smoke_w

print(f"  RISK_FIRE_WEIGHT  = {fire_w}")
print(f"  RISK_SMOKE_WEIGHT = {smoke_w}")
print(f"  Sum               = {total}")

if abs(total - 1.0) < 1e-9:
    passed(f"Risk weights sum to exactly 1.0")
else:
    failed(f"Risk weights sum to {total}, must be exactly 1.0")


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — ExpSmooth mathematical correctness
# ════════════════════════════════════════════════════════════════════════════
header("TEST 4 — ExpSmooth EMA Mathematics")

# ── Sub-test 4a: Warm-start ──────────────────────────────────────────────────
# The very first call should return the input directly (no blending)
smoother = ExpSmooth(alpha=0.5)
result = smoother.update(0.80)
print(f"  Warm-start test: update(0.80) → {result}")
if abs(result - 0.80) < 1e-9:
    passed("Warm-start correct: first output equals first input")
else:
    failed(f"Warm-start broken: expected 0.80, got {result}")

# ── Sub-test 4b: Steady-state convergence ───────────────────────────────────
# After feeding the same value many times, the smoother should
# converge to that value (no matter what alpha is).
smoother2 = ExpSmooth(alpha=0.3)
for _ in range(200):
    smoother2.update(0.50)
print(f"  Steady-state test: 200x update(0.50) → {smoother2.value:.8f}")
if abs(smoother2.value - 0.50) < 0.0001:
    passed("Steady-state convergence: value converged to 0.50 after 200 iterations")
else:
    failed(f"Steady-state broken: expected ~0.50, got {smoother2.value:.8f}")

# ── Sub-test 4c: Step response formula verification ─────────────────────────
# Closed-form solution for a step from 0 to 1.0:
#   After k steps: v_k = 1 - (1-alpha)^k
# We can verify the implementation is mathematically exact.
alpha  = 0.8
steps  = 10
expected = 1.0 - (1.0 - alpha) ** steps
smoother3 = ExpSmooth(alpha=alpha)
smoother3.update(0.0)   # warm-start at 0
# Override internal state to test pure formula (bypass warm-start)
smoother3._initialised = True
smoother3._value = 0.0
for _ in range(steps):
    smoother3.update(1.0)
print(f"  Step response test: alpha={alpha}, {steps} steps")
print(f"    Formula expected:  {expected:.8f}")
print(f"    Implementation got:{smoother3.value:.8f}")
if abs(smoother3.value - expected) < 1e-6:
    passed("Step response matches closed-form formula exactly")
else:
    failed(f"Step response mismatch: expected {expected:.8f}, got {smoother3.value:.8f}")

# ── Sub-test 4d: alpha validation ───────────────────────────────────────────
try:
    bad = ExpSmooth(alpha=1.5)   # must raise ValueError
    failed("Should have raised ValueError for alpha=1.5")
except ValueError:
    passed("Correctly raises ValueError for invalid alpha=1.5")


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — FPSCounter sliding window accuracy
# ════════════════════════════════════════════════════════════════════════════
header("TEST 5 — FPSCounter Sliding Window")

TARGET_FPS = 20.0
INTERVAL   = 1.0 / TARGET_FPS    # 0.05 seconds between frames
TICKS      = 25                   # simulate 25 frames

counter = FPSCounter(window=20)

print(f"  Simulating {TICKS} frames at {TARGET_FPS} fps target...")
for i in range(TICKS):
    counter.tick()
    if i < TICKS - 1:             # no sleep after the last tick
        time.sleep(INTERVAL)

measured_fps     = counter.fps
measured_latency = counter.latency

print(f"  Target FPS  : {TARGET_FPS}")
print(f"  Measured FPS: {measured_fps:.2f}")
print(f"  Latency     : {measured_latency:.2f} ms")

# Allow ±15% tolerance for Windows timer jitter
tolerance = TARGET_FPS * 0.15
if abs(measured_fps - TARGET_FPS) < tolerance:
    passed(f"FPS measurement accurate within ±15% tolerance ({measured_fps:.2f} fps)")
else:
    failed(f"FPS too far off: expected ~{TARGET_FPS}, got {measured_fps:.2f}")

# Test reset()
counter.reset()
if counter.fps == 0.0:
    passed("reset() correctly clears all timestamps (fps returns 0.0)")
else:
    failed(f"reset() failed: fps should be 0.0 after reset, got {counter.fps}")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{GREEN}{'═'*60}")
print(f"  ALL TESTS PASSED — Step 1 is complete and verified!")
print(f"  You are ready to move on to Step 2: FireDetector")
print(f"{'═'*60}{RESET}\n")



