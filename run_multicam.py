"""
================================================================================
  run_multicam.py
  Multi-Camera Pipeline -- 4 feeds in a 2x2 grid from one video file
  Run with: python run_multicam.py
================================================================================
  Each camera reads the same video file but starts at a different offset
  so all 4 feeds look like independent camera angles:
    CAM 1: starts at   0% through the video
    CAM 2: starts at  25% through the video
    CAM 3: starts at  50% through the video
    CAM 4: starts at  75% through the video
================================================================================
"""

import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np

from ifsd.utils              import FPSCounter
from ifsd.detectors.fire     import FireDetector
from ifsd.detectors.smoke    import SmokeDetector
from ifsd.analytics.risk     import classify_risk, RISK_CRITICAL
from ifsd.analytics.logger   import AlertLogger
from ifsd.analytics.alerter  import AlertMailer
from ifsd.rendering.grid     import GridRenderer

VIDEO        = "real_fire.mp4"
NUM_CAMS     = 4
WARMUP       = 30
CAM_LABELS   = ["CAM 1 -- SECTOR A", "CAM 2 -- SECTOR B",
                "CAM 3 -- SECTOR C", "CAM 4 -- SECTOR D"]

# ── Open video and get properties ────────────────────────────────────────────
cap_probe = cv2.VideoCapture(VIDEO)
if not cap_probe.isOpened():
    print(f"ERROR: Cannot open {VIDEO}")
    sys.exit(1)

total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
fps_src      = cap_probe.get(cv2.CAP_PROP_FPS)
duration     = total_frames / fps_src if fps_src > 0 else 0
cap_probe.release()

print(f"Video      : {VIDEO}")
print(f"Duration   : {duration:.1f}s  ({total_frames} frames)")
print(f"Cameras    : {NUM_CAMS} independent feeds")
print()

# ── Open 4 independent VideoCapture handles with offset start frames ─────────
# Each handle opens the same file but seeks to a different position.
# This is the key trick -- 4 caps = 4 independent read heads on one file.
offsets = [
    0,
    total_frames // 4,
    total_frames // 2,
    (total_frames * 3) // 4,
]

caps = []
for i, offset in enumerate(offsets):
    c = cv2.VideoCapture(VIDEO)
    if not c.isOpened():
        print(f"ERROR: Cannot open camera {i+1}")
        sys.exit(1)
    c.set(cv2.CAP_PROP_POS_FRAMES, offset)
    caps.append(c)
    print(f"  CAM {i+1}: starting at frame {offset:5d} "
          f"(t={offset/fps_src:.1f}s)")

print()

# ── Initialise one detector pair per camera ───────────────────────────────────
# Each camera has its own FireDetector and SmokeDetector so their
# MOG2 background models are completely independent.
fire_dets  = [FireDetector()  for _ in range(NUM_CAMS)]
smoke_dets = [SmokeDetector() for _ in range(NUM_CAMS)]

# Shared components
grid     = GridRenderer(cam_labels=CAM_LABELS)
fps_ctr  = FPSCounter()
logger   = AlertLogger()
mailer   = AlertMailer(cooldown_seconds=60)
frame_num = 0
paused    = False

print("PyroWatch Multi-Camera Monitor running")
print("Controls: Q=quit  P=pause/resume  S=screenshot")
print()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break

    if key == ord('p') or key == ord('P'):
        paused = not paused
        print("PAUSED" if paused else "RESUMED")

    if paused:
        continue

    frame_num += 1
    fps_ctr.tick()

    # ── Read one frame from each camera ──────────────────────────────────────
    tiles        = []
    worst_tier   = "CLEAR"
    worst_score  = 0.0
    tier_rank    = {"CLEAR":0,"CAUTION":1,"WARNING":2,"CRITICAL":3}

    for i in range(NUM_CAMS):
        ret, frame = caps[i].read()

        # Loop this camera back to start if it reaches end of file
        if not ret:
            caps[i].set(cv2.CAP_PROP_POS_FRAMES, offsets[i])
            fire_dets[i].reset()
            smoke_dets[i].reset()
            ret, frame = caps[i].read()

        if not ret or frame is None:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Resize to standard processing resolution
        frame = cv2.resize(frame, (1280, 720))

        # Run detectors
        fr          = fire_dets[i].detect(frame)
        sr          = smoke_dets[i].detect(frame)
        tier, score = classify_risk(fr["confidence"], sr["confidence"])

        # Track worst-case across all cameras
        if tier_rank.get(tier,0) > tier_rank.get(worst_tier,0):
            worst_tier  = tier
            worst_score = score

        tiles.append({
            "frame"        : frame,
            "fire_result"  : fr,
            "smoke_result" : sr,
            "scene_result" : [],
            "risk_tier"    : tier,
            "risk_score"   : score,
            "cam_id"       : i + 1,
        })

    # ── Compose the 2x2 grid canvas ──────────────────────────────────────────
    canvas = grid.render(tiles=tiles, fps=fps_ctr.fps)

    # ── Logging and alerting (after warmup) ───────────────────────────────────
    if frame_num > WARMUP:
        wrote = logger.log(
            frame_num    = frame_num,
            tier         = worst_tier,
            fire_conf    = max(t["fire_result"]["confidence"]  for t in tiles),
            smoke_conf   = max(t["smoke_result"]["confidence"] for t in tiles),
            risk_score   = worst_score,
            scene_result = [],
        )
        if wrote:
            print(f"  [LOG] Frame {frame_num:5d} | SYSTEM {worst_tier:<10} | "
                  f"W={worst_score:.4f}")

        if worst_tier == RISK_CRITICAL:
            mailer.send_alert(
                tier       = worst_tier,
                fire_conf  = max(t["fire_result"]["confidence"]  for t in tiles),
                smoke_conf = max(t["smoke_result"]["confidence"] for t in tiles),
                risk_score = worst_score,
                frame_num  = frame_num,
                canvas     = canvas,
            )
    else:
        if frame_num % 10 == 0:
            print(f"  [WARMUP] {frame_num}/{WARMUP}")

    # ── Screenshot ────────────────────────────────────────────────────────────
    if key == ord('s') or key == ord('S'):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"multicam_screenshot_{ts}.png"
        cv2.imwrite(name, canvas)
        print(f"Screenshot saved: {name}")

    cv2.imshow("PyroWatch Multi-Camera Monitor", canvas)

    # Terminal status every 60 frames
    if frame_num % 60 == 0:
        cam_status = "  ".join(
            f"C{i+1}:{t['risk_tier'][:3]}" for i,t in enumerate(tiles)
        )
        print(f"  Frame {frame_num:5d} | FPS {fps_ctr.fps:4.1f} | "
              f"System:{worst_tier:<10} | {cam_status}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
for c in caps:
    c.release()
cv2.destroyAllWindows()
logger.close(total_frames=frame_num)
print(f"Emails sent: {mailer.send_count}")



