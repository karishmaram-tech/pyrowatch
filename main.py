"""
================================================================================
  main.py  --  PyroWatch -- Industrial Hazard Detection System
  Unified Real-Time Inference Pipeline & Entry Point
================================================================================
  USAGE EXAMPLES:
    Webcam (default device 0):
      python main.py

    Specific webcam index:
      python main.py --source 1

    Video file:
      python main.py --source "C:\Videos\factory_floor.mp4"

    RTSP stream:
      python main.py --source "rtsp://192.168.1.100:554/stream"

    Save output video:
      python main.py --source 0 --output "C:\Videos\output.mp4"

    Custom resolution:
      python main.py --width 1280 --height 720

  KEYBOARD CONTROLS (while the display window is focused):
    Q  --  Quit
    P  --  Pause / Resume  (also prints timestamp to terminal)
    S  --  Save screenshot  (saved to working directory)
================================================================================
"""

import argparse
import sys
import os
import time
import datetime
import cv2
import numpy as np

# Make sure the project root is on the Python path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifsd.config             import CFG
from ifsd.utils              import FPSCounter, ExpSmooth
from ifsd.detectors.fire     import FireDetector
from ifsd.detectors.smoke    import SmokeDetector
from ifsd.analytics.scene    import SceneDetector
from ifsd.analytics.risk     import classify_risk
from ifsd.rendering.hud      import HUDRenderer


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PyroWatch -- Industrial Hazard Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--source", default="0",
        help=(
            "Input source. Options:\n"
            "  0, 1, 2 ...      webcam device index\n"
            "  path\\\\to\\\\file.mp4 video file\n"
            "  rtsp://...       RTSP network stream"
        ),
    )
    p.add_argument(
        "--output", default=None,
        help="Optional path to save the output video, e.g. output.mp4"
    )
    p.add_argument("--width",  type=int, default=CFG["RESOLUTION"][0],
                   help="Frame width  (default: %(default)s)")
    p.add_argument("--height", type=int, default=CFG["RESOLUTION"][1],
                   help="Frame height (default: %(default)s)")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no cv2.imshow window). Useful for servers.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def open_source(source_str: str) -> cv2.VideoCapture:
    """
    Open a VideoCapture from a webcam index, file path, or RTSP URL.

    Webcam indices are passed as integers ("0", "1").
    Everything else is treated as a string path or URL.
    """
    # Try parsing as an integer first (webcam index)
    try:
        src = int(source_str)
    except ValueError:
        src = source_str   # file path or RTSP URL

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source_str}")
        print("        Check the device index, file path, or stream URL.")
        sys.exit(1)

    return cap


def setup_writer(
    path: str,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """
    Create a VideoWriter that saves the composited HUD output to a file.

    Uses the FOURCC codec defined in CFG["OUTPUT_FOURCC"] (mp4v).
    FPS is fixed at CFG["FPS_CAP"] (30) for smooth playback.
    """
    fourcc = cv2.VideoWriter_fourcc(*CFG["OUTPUT_FOURCC"])
    writer = cv2.VideoWriter(path, fourcc, CFG["FPS_CAP"], (width, height))

    if not writer.isOpened():
        print(f"[WARNING] Could not create output writer at: {path}")
        print("          Output will not be saved.")
        return None

    print(f"[INFO] Saving output to: {path}")
    return writer


# ─────────────────────────────────────────────────────────────────────────────
# SCREENSHOT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def save_screenshot(canvas: np.ndarray) -> None:
    """
    Save the current composited canvas as a timestamped PNG file.
    Files are saved in the working directory with format:
      PyroWatch_screenshot_YYYYMMDD_HHMMSS.png
    """
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PyroWatch_screenshot_{ts}.png"
    cv2.imwrite(filename, canvas)
    print(f"[SCREENSHOT] Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    """
    Main detection loop.

    FRAME PROCESSING STRATEGY:
      Every frame:
        - FireDetector.detect()    (~1-3ms,  colour + morphology)
        - SmokeDetector.detect()   (~2-4ms,  MOG2 + colour)
        - classify_risk()          (<0.1ms,  pure arithmetic)
        - HUDRenderer.render()     (~3-8ms,  OpenCV drawing)

      Every other frame (YOLO_SKIP_FRAMES=1):
        - SceneDetector.detect()   (~15-40ms on CPU, ~3-8ms on GPU)

      This keeps the loop near 25-30fps on a modern CPU without a GPU.

    KEYBOARD HANDLING:
      cv2.waitKey(1) checks for a key press and waits 1ms.
      Returns -1 if no key pressed, or the ASCII code of the key.
      ord('q') converts the character 'q' to its ASCII value (113).
    """
    width  = args.width
    height = args.height

    # ── Open video source ────────────────────────────────────────────────────
    print(f"[INFO] Opening source: {args.source}")
    cap = open_source(args.source)

    # Request the desired resolution from the capture device.
    # Note: cameras may not honour this exactly -- we resize frames ourselves.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ── Initialise all modules ───────────────────────────────────────────────
    print("[INFO] Initialising detectors...")
    fire_det  = FireDetector()
    smoke_det = SmokeDetector()
    scene_det = SceneDetector()   # prints device (CPU/GPU)
    hud       = HUDRenderer()
    fps_ctr   = FPSCounter(window=CFG["FPS_WINDOW"])

    # ── Optional output writer ───────────────────────────────────────────────
    writer = None
    if args.output:
        writer = setup_writer(args.output, width, height)

    # ── State variables ──────────────────────────────────────────────────────
    paused       = False
    frame_count  = 0
    # Cache last scene result so YOLO skip frames still display detections
    last_scene   = []
    # Cache last risk result for display continuity on skipped frames
    last_tier    = "CLEAR"
    last_score   = 0.0

    print("[INFO] Pipeline running. Controls: Q=quit  P=pause  S=screenshot")
    print(f"[INFO] Resolution: {width}x{height}")
    if not args.no_display:
        print("[INFO] Display window: 'PyroWatch Monitor'")

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:

        # ── Keyboard handling ────────────────────────────────────────────────
        # waitKey(1) is required to actually render the imshow window.
        # Without it the window freezes. The 1ms wait is negligible.
        key = cv2.waitKey(1) & 0xFF   # mask to 8 bits (handles 64-bit systems)

        if key == ord('q') or key == ord('Q'):
            print("[INFO] Q pressed -- quitting.")
            break

        if key == ord('p') or key == ord('P'):
            paused = not paused
            state  = "PAUSED" if paused else "RESUMED"
            ts     = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {state}")

        # ── Pause: skip frame capture but keep the window responsive ─────────
        if paused:
            continue

        # ── Capture frame ────────────────────────────────────────────────────
        ret, frame = cap.read()

        if not ret:
            # End of file for video inputs -- loop back to start
            if isinstance(args.source, str) and not args.source.isdigit():
                print("[INFO] End of video -- looping.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fire_det.reset()
                smoke_det.reset()
                fps_ctr.reset()
                continue
            else:
                print("[ERROR] Failed to read frame from source.")
                break

        # ── Resize to target resolution ──────────────────────────────────────
        # cv2.resize(src, (width, height)) -- note (width, height) not (h, w)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        frame_count += 1
        fps_ctr.tick()

        # ── Screenshot key (needs a valid canvas -- handled after render) ────
        # We store a flag here and act on it after the canvas is ready
        take_screenshot = (key == ord('s') or key == ord('S'))

        # ── Fire detection  (runs EVERY frame) ───────────────────────────────
        fire_result  = fire_det.detect(frame)

        # ── Smoke detection  (runs EVERY frame) ──────────────────────────────
        smoke_result = smoke_det.detect(frame)

        # ── Scene detection  (runs every OTHER frame via internal skip) ───────
        # SceneDetector handles the skip logic internally.
        # On skipped frames it returns the cached result from last run.
        scene_result = scene_det.detect(frame)

        # ── Risk classification ───────────────────────────────────────────────
        tier, score  = classify_risk(
            fire_result["confidence"],
            smoke_result["confidence"],
        )
        last_tier  = tier
        last_score = score

        # ── Compose HUD ───────────────────────────────────────────────────────
        canvas = hud.render(
            frame        = frame,
            fire_result  = fire_result,
            smoke_result = smoke_result,
            scene_result = scene_result,
            risk_tier    = tier,
            risk_score   = score,
            fps          = fps_ctr.fps,
            latency      = fps_ctr.latency,
        )

        # ── Screenshot ────────────────────────────────────────────────────────
        if take_screenshot:
            save_screenshot(canvas)

        # ── Write to output file ──────────────────────────────────────────────
        if writer is not None:
            writer.write(canvas)

        # ── Display ───────────────────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow("PyroWatch Monitor", canvas)

        # ── Terminal status every 30 frames ──────────────────────────────────
        if frame_count % 30 == 0:
            people  = sum(1 for d in scene_result if d["label"] == "PERSON")
            vehicles= sum(1 for d in scene_result if d["label"] != "PERSON")
            print(
                f"  Frame {frame_count:6d} | "
                f"FPS {fps_ctr.fps:5.1f} | "
                f"Fire {fire_result['confidence']:.3f} | "
                f"Smoke {smoke_result['confidence']:.3f} | "
                f"Risk {tier:<8} | "
                f"People {people} Vehicles {vehicles}"
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[INFO] Releasing resources...")
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_count} frames.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    # Convert numeric string sources to int for webcam access
    # e.g. "--source 0" becomes int(0) for cv2.VideoCapture(0)
    if args.source.isdigit():
        args.source = int(args.source)

    run(args)




