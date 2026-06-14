"""
================================================================================
  PyroWatch/analytics/risk.py
  Risk Classification Engine
================================================================================
  Maps fire and smoke confidence scores to a discrete risk tier.

  RISK TIERS (in ascending severity):
    CLEAR    -- normal operations, no hazard detected
    CAUTION  -- low-level signal, monitor closely
    WARNING  -- significant hazard, prepare response
    CRITICAL -- immediate danger, evacuate / respond now

  WEIGHTED SCORE FORMULA:
    W = (RISK_FIRE_WEIGHT * fire_conf) + (RISK_SMOKE_WEIGHT * smoke_conf)
    W = (0.60 * fire_conf) + (0.40 * smoke_conf)

  WHY WEIGHTED?
    Fire is weighted more heavily (0.60) than smoke (0.40) because:
    - Visible flame is a more certain indicator of active fire
    - Smoke can come from non-fire sources (steam, dust, machinery exhaust)
    - False smoke positives are more common than false fire positives
    A pure smoke signal needs to be stronger to escalate the risk tier.

  TIER THRESHOLDS (from CFG):
    W >= 0.60 -> CRITICAL   (more than 60% weighted hazard coverage)
    W >= 0.35 -> WARNING
    W >= 0.15 -> CAUTION
    W <  0.15 -> CLEAR
================================================================================
"""

from ifsd.config import CFG

# Risk tier constants -- use these strings throughout the codebase
# so a typo like "CRITCAL" causes a NameError rather than silent wrong behaviour
RISK_CLEAR    = "CLEAR"
RISK_CAUTION  = "CAUTION"
RISK_WARNING  = "WARNING"
RISK_CRITICAL = "CRITICAL"

# Ordered list from lowest to highest -- useful for comparisons
RISK_LEVELS = [RISK_CLEAR, RISK_CAUTION, RISK_WARNING, RISK_CRITICAL]


def classify_risk(fire_conf: float, smoke_conf: float) -> tuple[str, float]:
    """
    Compute the weighted hazard score and map it to a risk tier.

    Parameters
    ----------
    fire_conf  : float
        Smoothed fire confidence from FireDetector, range 0.0-1.0.
    smoke_conf : float
        Smoothed smoke confidence from SmokeDetector, range 0.0-1.0.

    Returns
    -------
    tuple[str, float]
        (risk_tier, weighted_score)
        risk_tier     : one of RISK_CLEAR / CAUTION / WARNING / CRITICAL
        weighted_score: the raw W value before threshold mapping, 0.0-1.0

    Examples
    --------
    >>> classify_risk(0.0,  0.0)   -> ("CLEAR",    0.0)
    >>> classify_risk(0.2,  0.1)   -> ("CAUTION",  0.16)
    >>> classify_risk(0.5,  0.3)   -> ("WARNING",  0.42)
    >>> classify_risk(0.8,  0.7)   -> ("CRITICAL", 0.76)
    """
    # Clamp inputs to valid range as defensive measure
    fire_conf  = max(0.0, min(1.0, float(fire_conf)))
    smoke_conf = max(0.0, min(1.0, float(smoke_conf)))

    # Weighted combination
    w = (CFG["RISK_FIRE_WEIGHT"]  * fire_conf +
         CFG["RISK_SMOKE_WEIGHT"] * smoke_conf)

    # Map to tier using cascading threshold checks (highest first)
    if w >= CFG["RISK_CRITICAL_THRESH"]:
        tier = RISK_CRITICAL
    elif w >= CFG["RISK_WARNING_THRESH"]:
        tier = RISK_WARNING
    elif w >= CFG["RISK_CAUTION_THRESH"]:
        tier = RISK_CAUTION
    else:
        tier = RISK_CLEAR

    return tier, round(w, 6)


def risk_colour(tier: str) -> tuple[int, int, int]:
    """
    Return the BGR display colour associated with a risk tier.
    Used by the HUD renderer to colour-code panels and banners.

    Parameters
    ----------
    tier : str
        One of the RISK_* constants.

    Returns
    -------
    tuple[int, int, int]  BGR colour tuple for OpenCV drawing functions.
    """
    return {
        RISK_CLEAR    : CFG["COL_CLEAR"],
        RISK_CAUTION  : CFG["COL_CAUTION"],
        RISK_WARNING  : CFG["COL_WARNING"],
        RISK_CRITICAL : CFG["COL_CRITICAL"],
    }.get(tier, CFG["COL_TEXT"])


def risk_index(tier: str) -> int:
    """
    Return 0-3 integer index for a tier (CLEAR=0 ... CRITICAL=3).
    Useful for numeric comparisons: risk_index(a) > risk_index(b).
    """
    try:
        return RISK_LEVELS.index(tier)
    except ValueError:
        return 0



