"""
================================================================================
  tests/test_step5.py
  Verification tests for HUDRenderer
  Run with: python tests\test_step5.py
================================================================================
"""

import sys, os, math, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def passed(msg): print(f"  {GREEN}+ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}x FAILED{RESET}  {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")

def blank_fire():
    return {"boxes": [], "contours": [], "confidence": 0.0,
            "mask": np.zeros((480, 640), dtype=np.uint8)}

def blank_smoke():
    return {"boxes": [], "contours": [], "confidence": 0.0,
            "mask": np.zeros((480, 640), dtype=np.uint8)}

def make_frame(w=640, h=480, fill=(20, 20, 30)):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:] = fill
    return f


header("TEST 1 -- HUDRenderer imports and constructs")

try:
    from ifsd.rendering.hud import HUDRenderer
    passed("HUDRenderer imported from ifsd.rendering.hud")
except ImportError as e:
    failed(f"Import failed: {e}")

try:
    hud = HUDRenderer()
    passed("HUDRenderer() constructed without error")
except Exception as e:
    failed(f"Constructor raised: {e}")


header("TEST 2 -- render() returns correct shape and dtype")

from ifsd.analytics.risk import (RISK_CLEAR, RISK_CAUTION,
                                  RISK_WARNING, RISK_CRITICAL)

frame  = make_frame()
canvas = hud.render(
    frame=frame, fire_result=blank_fire(), smoke_result=blank_smoke(),
    scene_result=[], risk_tier=RISK_CLEAR, risk_score=0.0,
    fps=30.0, latency=33.3,
)

print(f"  input  shape : {frame.shape}")
print(f"  output shape : {canvas.shape}")
print(f"  output dtype : {canvas.dtype}")

if canvas.shape == frame.shape:
    passed("Output canvas has same shape as input frame")
else:
    failed(f"Shape mismatch: {frame.shape} vs {canvas.shape}")

if canvas.dtype == np.uint8:
    passed("Output dtype is uint8")
else:
    failed(f"dtype should be uint8, got {canvas.dtype}")


header("TEST 3 -- render() does not modify the original frame")

original      = make_frame(fill=(50, 100, 150))
original_copy = original.copy()

hud.render(frame=original, fire_result=blank_fire(),
           smoke_result=blank_smoke(), scene_result=[],
           risk_tier=RISK_CLEAR, risk_score=0.0)

if np.array_equal(original, original_copy):
    passed("Original frame unmodified after render()")
else:
    failed("render() modified the original frame -- must work on a copy")


header("TEST 4 -- render() modifies the canvas (HUD elements are drawn)")

plain  = make_frame(fill=(20, 20, 20))
canvas = hud.render(
    frame=plain, fire_result=blank_fire(), smoke_result=blank_smoke(),
    scene_result=[], risk_tier=RISK_CLEAR, risk_score=0.0,
    fps=25.0, latency=40.0,
)

pixels_changed = int(np.sum(canvas != plain))
print(f"  pixels changed by HUD drawing: {pixels_changed}")

if pixels_changed > 500:
    passed(f"Canvas modified ({pixels_changed} pixels) -- HUD is drawing")
else:
    failed(f"Too few pixels changed ({pixels_changed}) -- HUD may not be drawing")


header("TEST 5 -- _alpha_rect transparency check")

hud2          = HUDRenderer()
canvas_opaque = make_frame(fill=(0, 0, 0))
hud2._alpha_rect(canvas_opaque, 100, 100, 50, 50, (0, 255, 0), alpha=1.0)
roi_opaque    = canvas_opaque[100:150, 100:150]

canvas_transp = make_frame(fill=(80, 80, 80))
hud2._alpha_rect(canvas_transp, 100, 100, 50, 50, (0, 255, 0), alpha=0.0)
roi_transp    = canvas_transp[100:150, 100:150]

print(f"  alpha=1.0 ROI green channel centre: {roi_opaque[25,25,1]}")
print(f"  alpha=0.0 ROI mean value          : {roi_transp.mean():.1f}")

if roi_opaque[25, 25, 1] == 255:
    passed("alpha=1.0 produces fully opaque fill (green=255)")
else:
    failed(f"alpha=1.0 failed: green={roi_opaque[25,25,1]}")

if np.all(roi_transp == 80):
    passed("alpha=0.0 leaves ROI completely unchanged")
else:
    failed("alpha=0.0 should not change the canvas")


header("TEST 6 -- danger banner only appears for WARNING and CRITICAL")

def render_tier(tier):
    f = make_frame(fill=(20, 20, 20))
    return hud.render(
        frame=f, fire_result=blank_fire(), smoke_result=blank_smoke(),
        scene_result=[], risk_tier=tier, risk_score=0.5,
        fps=30.0, latency=33.0,
    )

plain        = make_frame(fill=(20, 20, 20))
c_clear      = render_tier(RISK_CLEAR)
c_caution    = render_tier(RISK_CAUTION)
c_warning    = render_tier(RISK_WARNING)
c_critical   = render_tier(RISK_CRITICAL)

def bottom_diff(c):
    bh = c.shape[0]
    return int(np.sum(c[bh-36:bh] != plain[bh-36:bh]))

d_clear    = bottom_diff(c_clear)
d_caution  = bottom_diff(c_caution)
d_warning  = bottom_diff(c_warning)
d_critical = bottom_diff(c_critical)

print(f"  Bottom banner pixel changes -- "
      f"CLEAR:{d_clear}  CAUTION:{d_caution}  "
      f"WARNING:{d_warning}  CRITICAL:{d_critical}")

if d_warning > d_clear:
    passed("WARNING banner active (more changes than CLEAR)")
else:
    failed("WARNING should have more bottom-row changes than CLEAR")

if d_critical > d_clear:
    passed("CRITICAL banner active (more changes than CLEAR)")
else:
    failed("CRITICAL should have more bottom-row changes than CLEAR")


header("TEST 7 -- scanline overlay built and cached correctly")

hud3 = HUDRenderer()
f1   = make_frame(640, 480)

hud3.render(frame=f1, fire_result=blank_fire(), smoke_result=blank_smoke(),
            scene_result=[], risk_tier=RISK_CLEAR, risk_score=0.0)

if hud3._scanline_overlay is not None:
    passed("Scanline overlay built on first render")
else:
    failed("Scanline overlay not built after first render")

cached_id = id(hud3._scanline_overlay)

hud3.render(frame=f1, fire_result=blank_fire(), smoke_result=blank_smoke(),
            scene_result=[], risk_tier=RISK_CLEAR, risk_score=0.0)

if id(hud3._scanline_overlay) == cached_id:
    passed("Scanline overlay reused on second render (cached)")
else:
    failed("Scanline overlay rebuilt unnecessarily")

f2         = make_frame(320, 240)
smoke_sm   = {"boxes":[],"contours":[],"confidence":0.0,
              "mask":np.zeros((240,320),dtype=np.uint8)}
fire_sm    = {"boxes":[],"contours":[],"confidence":0.0,
              "mask":np.zeros((240,320),dtype=np.uint8)}
hud3.render(frame=f2, fire_result=fire_sm, smoke_result=smoke_sm,
            scene_result=[], risk_tier=RISK_CLEAR, risk_score=0.0)

if id(hud3._scanline_overlay) != cached_id:
    passed("Scanline overlay rebuilt when frame size changed")
else:
    failed("Scanline overlay should rebuild on resolution change")


header("TEST 8 -- render() works with active fire detections")

fire_active = {
    "boxes"     : [(100, 100, 150, 120)],
    "contours"  : [np.array([[[100,100]],[[250,100]],[[250,220]],[[100,220]]])],
    "confidence": 0.45,
    "mask"      : np.zeros((480, 640), dtype=np.uint8),
}

try:
    canvas_fire = hud.render(
        frame=make_frame(), fire_result=fire_active,
        smoke_result=blank_smoke(), scene_result=[],
        risk_tier=RISK_WARNING, risk_score=0.27,
        fps=28.5, latency=35.1,
    )
    passed("render() completed with active fire detection")
    if canvas_fire.shape == (480, 640, 3):
        passed("Output shape correct with fire detections")
    else:
        failed(f"Wrong shape: {canvas_fire.shape}")
except Exception as e:
    failed(f"render() raised exception: {e}")


header("TEST 9 -- render() works with scene objects present")

scene_active = [
    {"box": (50,  50,  80, 160), "label": "PERSON",  "conf": 0.88, "center": (90,  130)},
    {"box": (300, 200, 180, 100),"label": "VEHICLE", "conf": 0.72, "center": (390, 250)},
]

try:
    canvas_scene = hud.render(
        frame=make_frame(), fire_result=blank_fire(),
        smoke_result=blank_smoke(), scene_result=scene_active,
        risk_tier=RISK_CAUTION, risk_score=0.18,
        fps=29.1, latency=34.4,
    )
    passed("render() completed with scene objects present")
except Exception as e:
    failed(f"render() raised exception: {e}")


print(f"\n{BOLD}{GREEN}{'='*60}")
print(f"  ALL TESTS PASSED -- Step 5 HUDRenderer is verified!")
print(f"  You are ready to move on to Step 6: Main Pipeline")
print(f"{'='*60}{RESET}\n")



