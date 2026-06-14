import sys, os, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from ifsd.utils              import FPSCounter
from ifsd.detectors.fire     import FireDetector
from ifsd.detectors.smoke    import SmokeDetector
from ifsd.analytics.scene    import SceneDetector
from ifsd.analytics.risk     import classify_risk, RISK_CRITICAL
from ifsd.analytics.logger   import AlertLogger
from ifsd.analytics.alerter  import AlertMailer
from ifsd.rendering.hud      import HUDRenderer

VIDEO         = "real_fire.mp4"
WARMUP_FRAMES = 30

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO}")
    sys.exit(1)

fps_v  = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur    = frames / fps_v if fps_v > 0 else 0
print(f"Video      : {VIDEO}")
print(f"Resolution : {w}x{h}  |  FPS: {fps_v}  |  Duration: {dur:.1f}s")
print()

fire_det  = FireDetector()
smoke_det = SmokeDetector()
scene_det = SceneDetector()
hud       = HUDRenderer()
fps_ctr   = FPSCounter()
logger    = AlertLogger()
mailer    = AlertMailer(cooldown_seconds=60)
frame_num = 0
paused    = False

print("PyroWatch Monitor running")
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

    ret, frame = cap.read()
    if not ret:
        print("End of video -- looping...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fire_det.reset()
        smoke_det.reset()
        fps_ctr.reset()
        frame_num = 0
        continue

    frame     = cv2.resize(frame, (1280, 720))
    frame_num += 1
    fps_ctr.tick()

    fr          = fire_det.detect(frame)
    sr          = smoke_det.detect(frame)
    sc          = []
    tier, score = classify_risk(fr["confidence"], sr["confidence"])

    canvas = hud.render(
        frame        = frame,
        fire_result  = fr,
        smoke_result = sr,
        scene_result = sc,
        risk_tier    = tier,
        risk_score   = score,
        fps          = fps_ctr.fps,
        latency      = fps_ctr.latency,
    )

    # ── Log tier changes ──────────────────────────────────────────────────
    if frame_num > WARMUP_FRAMES:
        wrote = logger.log(
            frame_num    = frame_num,
            tier         = tier,
            fire_conf    = fr["confidence"],
            smoke_conf   = sr["confidence"],
            risk_score   = score,
            scene_result = sc,
        )
        if wrote:
            print(f"  [LOG] Frame {frame_num:5d} | {tier:<10} | "
                  f"Fire {fr['confidence']:.3f} | "
                  f"Smoke {sr['confidence']:.3f} | "
                  f"W={score:.4f}")

        # ── Send email alert on CRITICAL ──────────────────────────────────
        if tier == RISK_CRITICAL:
            mailer.send_alert(
                tier       = tier,
                fire_conf  = fr["confidence"],
                smoke_conf = sr["confidence"],
                risk_score = score,
                frame_num  = frame_num,
                canvas     = canvas,
            )
    else:
        print(f"  [WARMUP] Frame {frame_num:2d}/{WARMUP_FRAMES}")

    if key == ord('s') or key == ord('S'):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"screenshot_{ts}.png"
        cv2.imwrite(name, canvas)
        print(f"Screenshot saved: {name}")

    cv2.imshow("PyroWatch Monitor", canvas)

    if frame_num % 60 == 0:
        print(
            f"  Frame {frame_num:5d} | "
            f"FPS {fps_ctr.fps:5.1f} | "
            f"Fire {fr['confidence']:.3f} | "
            f"Smoke {sr['confidence']:.3f} | "
            f"Risk {tier}"
        )

cap.release()
cv2.destroyAllWindows()
logger.close(total_frames=frame_num)
print(f"Emails sent this session: {mailer.send_count}")



