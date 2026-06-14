"""
================================================================================
  PyroWatch/analytics/logger.py
  AlertLogger -- CSV-based detection event logger
================================================================================
  Logs risk tier CHANGES to a CSV file. Only writes a new row when the
  risk tier changes (e.g. CLEAR->WARNING) -- not every frame.
  This keeps the CSV readable and meaningful rather than millions of rows.

  CSV COLUMNS:
    timestamp       -- wall-clock time of the event  (YYYY-MM-DD HH:MM:SS.mmm)
    frame           -- frame number in the video
    elapsed_s       -- seconds since session started
    event           -- what changed (TIER_CHANGE, SESSION_START, SESSION_END)
    tier_from       -- previous risk tier
    tier_to         -- new risk tier
    fire_conf       -- smoothed fire confidence 0.0-1.0
    smoke_conf      -- smoothed smoke confidence 0.0-1.0
    risk_score      -- weighted score W = 0.6*fire + 0.4*smoke
    people_count    -- number of people detected by YOLO
    vehicle_count   -- number of vehicles detected by YOLO
    notes           -- any extra context

  OUTPUT FILE:
    logs/PyroWatch_session_YYYYMMDD_HHMMSS.csv
    A new file is created for every run so logs never overwrite each other.
================================================================================
"""

import csv
import os
import datetime
import time

from ifsd.analytics.risk import RISK_CLEAR


class AlertLogger:
    """
    Logs risk tier change events to a CSV file.

    HOW TO USE:
        logger = AlertLogger()

        # inside your frame loop:
        logger.log(
            frame_num    = frame_num,
            tier         = tier,
            fire_conf    = fr["confidence"],
            smoke_conf   = sr["confidence"],
            risk_score   = score,
            scene_result = sc,
        )

        # when the loop ends:
        logger.close(total_frames=frame_num)
    """

    def __init__(self, log_dir: str = "logs") -> None:
        # ── Create logs directory if it doesn't exist ─────────────────────
        os.makedirs(log_dir, exist_ok=True)

        # ── Build a unique filename for this session ───────────────────────
        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"PyroWatch_session_{ts}.csv")

        # ── Session tracking ───────────────────────────────────────────────
        self._start_time   = time.time()
        self._prev_tier    = None          # tracks last written tier
        self._event_count  = 0            # total rows written
        self._tier_counts  = {}           # how many frames in each tier
        self._max_fire     = 0.0
        self._max_smoke    = 0.0
        self._max_score    = 0.0

        # ── Open CSV and write header ──────────────────────────────────────
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=[
            "timestamp",
            "frame",
            "elapsed_s",
            "event",
            "tier_from",
            "tier_to",
            "fire_conf",
            "smoke_conf",
            "risk_score",
            "people_count",
            "vehicle_count",
            "notes",
        ])
        self._writer.writeheader()
        self._file.flush()

        # ── Write session start marker ─────────────────────────────────────
        self._write_row(
            frame      = 0,
            event      = "SESSION_START",
            tier_from  = "-",
            tier_to    = RISK_CLEAR,
            fire_conf  = 0.0,
            smoke_conf = 0.0,
            risk_score = 0.0,
            people     = 0,
            vehicles   = 0,
            notes      = f"Log file: {self._path}",
        )

        print(f"[PyroWatch Logger] Logging to: {self._path}")

    # ─────────────────────────────────────────────────────────────────────────
    def log(
        self,
        frame_num    : int,
        tier         : str,
        fire_conf    : float,
        smoke_conf   : float,
        risk_score   : float,
        scene_result : list,
    ) -> bool:
        """
        Call every frame. Only writes to CSV when the risk tier changes.

        Parameters
        ----------
        frame_num    : current frame number
        tier         : current risk tier string (CLEAR/CAUTION/WARNING/CRITICAL)
        fire_conf    : smoothed fire confidence 0.0-1.0
        smoke_conf   : smoothed smoke confidence 0.0-1.0
        risk_score   : weighted score W
        scene_result : list of YOLO detections

        Returns
        -------
        bool : True if a new row was written (tier changed), False otherwise
        """
        # Track running maximums
        self._max_fire  = max(self._max_fire,  fire_conf)
        self._max_smoke = max(self._max_smoke, smoke_conf)
        self._max_score = max(self._max_score, risk_score)

        # Track time in each tier
        self._tier_counts[tier] = self._tier_counts.get(tier, 0) + 1

        # Count scene objects
        people   = sum(1 for d in scene_result if d.get("label") == "PERSON")
        vehicles = sum(1 for d in scene_result if d.get("label") != "PERSON"
                       and d.get("label") is not None)

        # Only write a row when the tier actually changes
        if tier == self._prev_tier:
            return False

        prev = self._prev_tier if self._prev_tier is not None else "-"

        # Determine note based on direction of change
        if self._prev_tier is None:
            note = "System initialised"
        elif self._tier_index(tier) > self._tier_index(prev):
            note = f"ESCALATION: {prev} -> {tier}"
        else:
            note = f"DE-ESCALATION: {prev} -> {tier}"

        self._write_row(
            frame      = frame_num,
            event      = "TIER_CHANGE",
            tier_from  = prev,
            tier_to    = tier,
            fire_conf  = fire_conf,
            smoke_conf = smoke_conf,
            risk_score = risk_score,
            people     = people,
            vehicles   = vehicles,
            notes      = note,
        )

        self._prev_tier = tier
        return True

    # ─────────────────────────────────────────────────────────────────────────
    def close(self, total_frames: int = 0) -> None:
        """
        Write session summary and close the CSV file cleanly.
        Always call this when the detection loop ends.
        """
        elapsed = time.time() - self._start_time

        # Build summary note
        tier_summary = "  |  ".join(
            f"{t}:{c}f" for t, c in sorted(self._tier_counts.items())
        )
        notes = (
            f"Total frames: {total_frames}  |  "
            f"Duration: {elapsed:.1f}s  |  "
            f"Peak fire: {self._max_fire:.3f}  |  "
            f"Peak smoke: {self._max_smoke:.3f}  |  "
            f"Peak score: {self._max_score:.3f}  |  "
            f"Tier distribution: {tier_summary}"
        )

        self._write_row(
            frame      = total_frames,
            event      = "SESSION_END",
            tier_from  = self._prev_tier or "-",
            tier_to    = "-",
            fire_conf  = self._max_fire,
            smoke_conf = self._max_smoke,
            risk_score = self._max_score,
            people     = 0,
            vehicles   = 0,
            notes      = notes,
        )

        self._file.close()

        print(f"\n[PyroWatch Logger] Session complete.")
        print(f"  Log file    : {self._path}")
        print(f"  Events logged: {self._event_count}")
        print(f"  Duration    : {elapsed:.1f}s")
        print(f"  Peak fire   : {self._max_fire:.3f}")
        print(f"  Peak smoke  : {self._max_smoke:.3f}")
        print(f"  Peak score  : {self._max_score:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _write_row(
        self,
        frame      : int,
        event      : str,
        tier_from  : str,
        tier_to    : str,
        fire_conf  : float,
        smoke_conf : float,
        risk_score : float,
        people     : int,
        vehicles   : int,
        notes      : str,
    ) -> None:
        """Write one row to the CSV file."""
        elapsed = time.time() - self._start_time
        ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self._writer.writerow({
            "timestamp"    : ts,
            "frame"        : frame,
            "elapsed_s"    : round(elapsed, 3),
            "event"        : event,
            "tier_from"    : tier_from,
            "tier_to"      : tier_to,
            "fire_conf"    : round(fire_conf,  4),
            "smoke_conf"   : round(smoke_conf, 4),
            "risk_score"   : round(risk_score, 4),
            "people_count" : people,
            "vehicle_count": vehicles,
            "notes"        : notes,
        })
        self._file.flush()   # write to disk immediately -- no data loss on crash
        self._event_count += 1

    @staticmethod
    def _tier_index(tier: str) -> int:
        """Return numeric rank of a tier for escalation/de-escalation comparison."""
        return {"CLEAR": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}.get(tier, 0)



