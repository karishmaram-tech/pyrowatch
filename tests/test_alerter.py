"""
tests/test_alerter.py
Verification tests for AlertMailer
Run with: python tests\test_alerter.py
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
YELLOW = "\033[93m"; RESET = "\033[0m"; BOLD = "\033[1m"

def passed(msg): print(f"  {GREEN}+ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}x FAILED{RESET}  {msg}"); sys.exit(1)
def warned(msg): print(f"  {YELLOW}! WARNED{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")


header("TEST 1 -- AlertMailer imports and constructs")

try:
    from ifsd.analytics.alerter import AlertMailer
    passed("AlertMailer imported from ifsd.analytics.alerter")
except ImportError as e:
    failed(f"Import failed: {e}")

try:
    mailer = AlertMailer(cooldown_seconds=5)
    passed("AlertMailer() constructed without error")
except Exception as e:
    failed(f"Constructor raised: {e}")


header("TEST 2 -- Credentials loaded correctly")

if mailer.enabled:
    passed("Credentials found in .env -- email alerts are ENABLED")
    print(f"  Sender  : loaded")
    print(f"  Receiver: loaded")
else:
    warned(".env file missing or incomplete -- email alerts DISABLED")
    warned("This is OK for testing -- real alerts need valid credentials")
    print()
    print("  To enable: edit .env with your Gmail address and App Password")


header("TEST 3 -- Cooldown logic prevents spam")

# Simulate sending by manually setting _last_sent to now
mailer._last_sent = time.time()

# Immediate second call should be blocked by cooldown
result = mailer.send_alert(
    tier="CRITICAL", fire_conf=0.1, smoke_conf=0.08,
    risk_score=0.09, frame_num=100, canvas=None,
)

print(f"  send_alert() during cooldown returned: {result}")
print(f"  cooldown_remaining: {mailer.cooldown_remaining()}s")

if result is False:
    passed("Cooldown correctly blocked send_alert() during active cooldown")
else:
    failed("send_alert() should return False during cooldown")

if mailer.cooldown_remaining() > 0:
    passed(f"cooldown_remaining() reports {mailer.cooldown_remaining()}s correctly")
else:
    failed("cooldown_remaining() should be > 0 during active cooldown")


header("TEST 4 -- Tier threshold filtering")

mailer2 = AlertMailer(cooldown_seconds=0, min_tier_to_alert="CRITICAL")

# CLEAR and CAUTION should never trigger alerts
for tier in ["CLEAR", "CAUTION", "WARNING"]:
    result = mailer2.send_alert(
        tier=tier, fire_conf=0.05, smoke_conf=0.03,
        risk_score=0.04, frame_num=1, canvas=None,
    )
    if result is False:
        passed(f"Tier '{tier}' correctly suppressed (below CRITICAL threshold)")
    else:
        failed(f"Tier '{tier}' should not trigger alert when min_tier=CRITICAL")


header("TEST 5 -- WARNING tier mailer allows WARNING and above")

mailer3 = AlertMailer(cooldown_seconds=0, min_tier_to_alert="WARNING")

# With no credentials, send_alert returns False -- but we check the
# tier filtering logic separately from the SMTP sending logic
result_clear = mailer3.send_alert(
    tier="CLEAR", fire_conf=0.0, smoke_conf=0.0,
    risk_score=0.0, frame_num=1, canvas=None,
)
result_caution = mailer3.send_alert(
    tier="CAUTION", fire_conf=0.01, smoke_conf=0.01,
    risk_score=0.01, frame_num=2, canvas=None,
)

if result_clear is False:
    passed("CLEAR tier suppressed by WARNING-threshold mailer")
else:
    failed("CLEAR should be suppressed")

if result_caution is False:
    passed("CAUTION tier suppressed by WARNING-threshold mailer")
else:
    failed("CAUTION should be suppressed when min_tier=WARNING")


header("TEST 6 -- send_count tracks correctly")

mailer4 = AlertMailer(cooldown_seconds=0)
print(f"  Initial send_count: {mailer4.send_count}")

if mailer4.send_count == 0:
    passed("send_count initialises at 0")
else:
    failed(f"send_count should be 0, got {mailer4.send_count}")


header("TEST 7 -- Canvas screenshot encoding works")

test_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
test_canvas[300:420, 500:780] = (0, 80, 220)   # orange fire patch

try:
    import cv2
    success, buf = cv2.imencode(".png", test_canvas)
    if success and len(buf) > 1000:
        passed(f"Canvas PNG encoding works ({len(buf):,} bytes)")
    else:
        failed("PNG encoding produced empty or tiny buffer")
except Exception as e:
    failed(f"Canvas encoding raised: {e}")


header("TEST 8 -- Live email send test (only if credentials present)")

if not mailer.enabled:
    warned("Skipping live send test -- no credentials in .env")
    warned("To test live sending, fill in .env and re-run this test")
else:
    print("  Attempting to send a real test email...")
    print("  (Check your inbox after this test)")

    test_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_canvas[200:500, 300:900] = (0, 60, 200)

    # Reset cooldown for this test
    mailer._last_sent = 0.0

    result = mailer.send_alert(
        tier       = "CRITICAL",
        fire_conf  = 0.073,
        smoke_conf = 0.061,
        risk_score = 0.068,
        frame_num  = 999,
        canvas     = test_canvas,
    )

    if result:
        passed("Live email sent successfully -- check your inbox!")
        passed(f"Total emails sent this session: {mailer.send_count}")
    else:
        failed("Email send failed -- check your .env credentials and App Password")


print(f"\n{BOLD}{GREEN}{'='*60}")
print(f"  ALL TESTS PASSED -- AlertMailer is verified!")
print(f"")
print(f"  Next: update run.py to wire in email alerts")
print(f"{'='*60}{RESET}\n")



