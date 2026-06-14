"""
tests/test_logger.py
Verification tests for AlertLogger
Run with: python tests\test_logger.py
"""

import sys, os, csv, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN = "\033[92m"; RED = "\033[91m"; CYAN = "\033[96m"
RESET = "\033[0m";  BOLD = "\033[1m"

def passed(msg): print(f"  {GREEN}+ PASSED{RESET}  {msg}")
def failed(msg): print(f"  {RED}x FAILED{RESET}  {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{CYAN}{'-'*60}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'-'*60}{RESET}")

TEST_LOG_DIR = "logs_test_temp"


header("TEST 1 -- AlertLogger imports and constructs")

try:
    from ifsd.analytics.logger import AlertLogger
    passed("AlertLogger imported")
except ImportError as e:
    failed(f"Import failed: {e}")

try:
    logger = AlertLogger(log_dir=TEST_LOG_DIR)
    passed(f"AlertLogger constructed -- log dir: {TEST_LOG_DIR}")
except Exception as e:
    failed(f"Constructor raised: {e}")


header("TEST 2 -- Log file was created with correct headers")

import glob
files = glob.glob(os.path.join(TEST_LOG_DIR, "PyroWatch_session_*.csv"))
if not files:
    failed("No CSV file created in log directory")

log_path = files[0]
print(f"  Log file: {log_path}")
passed("CSV file created")

with open(log_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames

expected_headers = [
    "timestamp", "frame", "elapsed_s", "event",
    "tier_from", "tier_to", "fire_conf", "smoke_conf",
    "risk_score", "people_count", "vehicle_count", "notes"
]

print(f"  Headers found: {headers}")
if list(headers) == expected_headers:
    passed("All 12 CSV columns present in correct order")
else:
    failed(f"Header mismatch.\nExpected: {expected_headers}\nGot: {list(headers)}")


header("TEST 3 -- SESSION_START row was written automatically")

with open(log_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

start_rows = [r for r in rows if r["event"] == "SESSION_START"]
print(f"  SESSION_START rows found: {len(start_rows)}")
if len(start_rows) == 1:
    passed("Exactly one SESSION_START row written on construction")
else:
    failed(f"Expected 1 SESSION_START row, got {len(start_rows)}")


header("TEST 4 -- log() only writes on tier change, not every frame")

from ifsd.analytics.risk import RISK_CLEAR, RISK_WARNING, RISK_CRITICAL

# Feed 10 frames of CLEAR -- should produce exactly 1 write (first call)
for i in range(10):
    logger.log(i, RISK_CLEAR, 0.01, 0.01, 0.01, [])

# Switch to WARNING -- should produce 1 write
logger.log(11, RISK_WARNING, 0.05, 0.03, 0.04, [])

# 5 more WARNING frames -- no new writes
for i in range(12, 17):
    logger.log(i, RISK_WARNING, 0.05, 0.03, 0.04, [])

# Switch to CRITICAL -- 1 write
logger.log(17, RISK_CRITICAL, 0.10, 0.08, 0.09, [])

# Back to CLEAR -- 1 write
logger.log(18, RISK_CLEAR, 0.00, 0.00, 0.00, [])

with open(log_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

tier_change_rows = [r for r in rows if r["event"] == "TIER_CHANGE"]
print(f"  TIER_CHANGE rows written: {len(tier_change_rows)}")
print(f"  (fed 18 frames with 4 actual tier changes)")

# Expected: CLEAR(first), WARNING, CRITICAL, CLEAR = 4 tier changes
if len(tier_change_rows) == 4:
    passed("Exactly 4 TIER_CHANGE rows written for 4 actual tier changes")
else:
    failed(f"Expected 4 TIER_CHANGE rows, got {len(tier_change_rows)}")


header("TEST 5 -- Escalation and de-escalation notes are correct")

notes = [r["notes"] for r in tier_change_rows]
print(f"  Tier change notes:")
for n in notes:
    print(f"    {n}")

escalations   = [n for n in notes if "ESCALATION" in n and "DE-" not in n]
deescalations = [n for n in notes if "DE-ESCALATION" in n]

if len(escalations) >= 2:
    passed(f"Escalation notes present ({len(escalations)} found)")
else:
    failed(f"Expected escalation notes, got: {notes}")

if len(deescalations) >= 1:
    passed(f"De-escalation note present ({len(deescalations)} found)")
else:
    failed(f"Expected de-escalation note, got: {notes}")


header("TEST 6 -- Values written to CSV are numerically correct")

warning_row = next(r for r in tier_change_rows if r["tier_to"] == "WARNING")
print(f"  WARNING row: fire={warning_row['fire_conf']}  "
      f"smoke={warning_row['smoke_conf']}  score={warning_row['risk_score']}")

if abs(float(warning_row["fire_conf"]) - 0.05) < 0.001:
    passed("fire_conf written correctly to CSV")
else:
    failed(f"fire_conf wrong: expected 0.05, got {warning_row['fire_conf']}")

if warning_row["tier_from"] == "CLEAR" and warning_row["tier_to"] == "WARNING":
    passed("tier_from and tier_to correctly record the transition")
else:
    failed(f"Tier transition wrong: {warning_row['tier_from']} -> {warning_row['tier_to']}")


header("TEST 7 -- close() writes SESSION_END with summary")

logger.close(total_frames=18)

with open(log_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

end_rows = [r for r in rows if r["event"] == "SESSION_END"]
print(f"  SESSION_END rows: {len(end_rows)}")

if len(end_rows) == 1:
    passed("Exactly one SESSION_END row written by close()")
else:
    failed(f"Expected 1 SESSION_END row, got {len(end_rows)}")

end_notes = end_rows[0]["notes"]
print(f"  Summary: {end_notes}")

if "Total frames: 18" in end_notes and "Peak fire" in end_notes:
    passed("SESSION_END notes contain frame count and peak metrics")
else:
    failed(f"SESSION_END notes incomplete: {end_notes}")


header("TEST 8 -- Full CSV is readable and well-formed")

with open(log_path, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

print(f"  Total rows in CSV (including header): {len(all_rows)+1}")
print(f"  Row breakdown:")

for event_type in ["SESSION_START", "TIER_CHANGE", "SESSION_END"]:
    count = sum(1 for r in all_rows if r["event"] == event_type)
    print(f"    {event_type:<15}: {count}")

if len(all_rows) >= 6:
    passed(f"CSV contains {len(all_rows)} data rows -- well formed")
else:
    failed(f"CSV too short: only {len(all_rows)} rows")


# Cleanup temp log directory
shutil.rmtree(TEST_LOG_DIR, ignore_errors=True)
passed("Temp log directory cleaned up")


print(f"\n{BOLD}{GREEN}{'='*60}")
print(f"  ALL TESTS PASSED -- AlertLogger is verified!")
print(f"")
print(f"  Run the full system with logging:")
print(f"    python run.py")
print(f"")
print(f"  Logs are saved to: logs\\PyroWatch_session_YYYYMMDD_HHMMSS.csv")
print(f"  Open with Excel or Notepad to view the alert history.")
print(f"{'='*60}{RESET}\n")



