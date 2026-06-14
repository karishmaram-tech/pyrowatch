"""
================================================================================
  PyroWatch/utils.py
  Telemetry Utilities: FPSCounter and ExpSmooth
  Import these in any module that needs performance tracking or signal smoothing.

  Usage example:
      from ifsd.utils import FPSCounter, ExpSmooth
      fps_counter = FPSCounter()
      smoother    = ExpSmooth(alpha=0.35)

      # inside your frame loop:
      fps_counter.tick()
      smooth_val = smoother.update(raw_confidence)
================================================================================
"""

import time
import collections
from typing import Deque


# ─────────────────────────────────────────────────────────────────────────────
# FPSCounter — Sliding Window Frame Rate Tracker
#
# HOW IT WORKS (the math in plain English):
#
#   Every time a new frame is processed, we call .tick() which records the
#   current time into a queue. The queue has a maximum length of W frames.
#   When the queue is full, adding a new time automatically removes the oldest.
#
#   To calculate FPS, we look at how long it took to process the last W frames:
#
#       fps = (number of gaps between timestamps) / (newest time - oldest time)
#           = (W - 1) / (t_last - t_first)
#
#   Example with W=5 and times [1.0, 1.04, 1.08, 1.12, 1.16] seconds:
#       fps = (5-1) / (1.16 - 1.00) = 4 / 0.16 = 25.0 fps  ✓
#
#   MEMORY SAFETY: collections.deque(maxlen=W) automatically discards the oldest
#   entry when a new one is added. In a 24/7 industrial stream running at 30fps
#   for 1 year, a plain list would grow to ~946 MILLION entries. The deque stays
#   permanently at W=30 entries, using the same tiny amount of memory forever.
# ─────────────────────────────────────────────────────────────────────────────
class FPSCounter:
    """
    Sliding-window frames-per-second and latency tracker.

    Parameters
    ----------
    window : int, optional
        How many frames to average over. Defaults to CFG["FPS_WINDOW"] (30).
        A larger window gives a smoother reading; smaller reacts faster.
    """

    def __init__(self, window: int = 30) -> None:
        # deque with maxlen is the key data structure — it is a circular buffer.
        # Adding a new item when full automatically removes the oldest. O(1) speed.
        self._timestamps: Deque[float] = collections.deque(maxlen=window)
        self._window = window

    def tick(self) -> None:
        """
        Call this ONCE per frame, at the top of your processing loop.
        Records the current wall-clock time as a frame boundary.

        Example:
            while True:
                fps_counter.tick()      # <-- call here, before any processing
                frame = camera.read()
                ... process frame ...
        """
        # time.perf_counter() is the highest-resolution clock available on
        # Windows. It measures elapsed time in seconds as a float.
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """
        Current rolling-average FPS.

        Returns 0.0 if fewer than 2 frames have been recorded yet
        (can't calculate a rate from a single point in time).
        """
        n = len(self._timestamps)
        if n < 2:
            return 0.0
        # Time elapsed across the whole window
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        # There are (n-1) inter-frame intervals within n timestamps
        return (n - 1) / elapsed

    @property
    def latency(self) -> float:
        """
        Average time per frame in milliseconds.
        This is simply the reciprocal of FPS, scaled to ms.

        Example: 25 fps → 1000/25 = 40.0 ms per frame
        """
        f = self.fps
        return (1000.0 / f) if f > 0 else 0.0

    def reset(self) -> None:
        """Clear all stored timestamps. Useful when resuming after a pause."""
        self._timestamps.clear()

    def __repr__(self) -> str:
        return f"FPSCounter(fps={self.fps:.1f}, latency={self.latency:.1f}ms, window={self._window})"


# ─────────────────────────────────────────────────────────────────────────────
# ExpSmooth — Exponential Moving Average Filter
#
# THE FORMULA:  v_t = α · x_t + (1 - α) · v_(t-1)
#
# In plain English, the new smoothed value is a WEIGHTED BLEND of:
#   - the brand-new raw measurement  (weighted by α)
#   - the previous smoothed value    (weighted by 1-α, i.e. everything before)
#
# PHYSICAL MEANING OF α (alpha):
#   α = 0.9  →  "I trust new data a LOT. I update quickly. I'm reactive and jumpy."
#   α = 0.1  →  "I trust new data a LITTLE. I change slowly. I'm smooth and sluggish."
#
# WHY DIFFERENT α FOR FIRE vs SMOKE?
#
#   FIRE   (α = 0.35, medium-high):
#     A new flame can appear in a single frame. We need to react in ~3-5 frames.
#     But we also don't want a single bright-red frame to spike the alarm.
#     0.35 is a balance: responsive but not hair-trigger.
#
#   SMOKE  (α = 0.25, medium-low):
#     Real smoke takes 10-30 seconds to build up and equally long to clear.
#     A lower α mirrors this physical reality — the confidence score rises and
#     falls slowly, matching how smoke actually behaves in the real world.
#     This also prevents a fan blowing smoke away briefly from clearing the alert.
#
# WARM-START (why we set v_0 = x_0 on the first call):
#   If we initialise v at 0.0 and the first real measurement is 0.8,
#   the filter would report ~0.28 on frame 1, ~0.47 on frame 2, slowly
#   "climbing" toward reality for dozens of frames. This "cold-start lag"
#   can cause missed detections at the start of a video. Instead, we
#   set v_0 = x_0 directly — the filter starts at the right level immediately.
# ─────────────────────────────────────────────────────────────────────────────
class ExpSmooth:
    """
    Single-value Exponential Moving Average (EMA) filter.

    Parameters
    ----------
    alpha : float
        Smoothing coefficient, must be in range (0.0, 1.0].
        Higher = more reactive to new data.
        Lower  = smoother output, slower to react.
    init : float, optional
        Starting value before any data arrives. Default 0.0.

    Example
    -------
        smoother = ExpSmooth(alpha=0.35)
        for raw_score in detector_output:
            smooth_score = smoother.update(raw_score)
            display(smooth_score)   # this value won't flicker
    """

    def __init__(self, alpha: float, init: float = 0.0) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(
                f"alpha must be between 0.0 (exclusive) and 1.0 (inclusive). "
                f"You passed: {alpha}"
            )
        self._alpha = float(alpha)
        self._value = float(init)
        self._initialised = False   # tracks whether warm-start has happened yet

    def update(self, x: float) -> float:
        """
        Feed in a new raw measurement. Returns the updated smoothed value.

        On the very first call: sets internal state to x directly (warm-start).
        On all subsequent calls: applies the EMA formula.

        Parameters
        ----------
        x : float
            The new raw measurement (e.g. raw fire confidence 0.0 to 1.0).

        Returns
        -------
        float
            The smoothed value after applying the EMA formula.
        """
        if not self._initialised:
            # Warm-start: skip the blend on the first observation.
            # v_0 = x_0  (no blending needed — nothing to blend with yet)
            self._value = float(x)
            self._initialised = True
        else:
            # Standard EMA: v_t = α * x_t + (1 - α) * v_(t-1)
            self._value = self._alpha * float(x) + (1.0 - self._alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        """Read the current smoothed value without feeding a new measurement."""
        return self._value

    def reset(self, init: float = 0.0) -> None:
        """
        Hard-reset the filter back to a starting value.
        Call this when switching to a new video source or after a long pause.
        """
        self._value = float(init)
        self._initialised = False

    def __repr__(self) -> str:
        return (
            f"ExpSmooth("
            f"alpha={self._alpha}, "
            f"current_value={self._value:.4f}, "
            f"warmed_up={self._initialised})"
        )



