from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union


def check_sbp_trend(
    csv_path: Union[Path, str],
    threshold: float = 10.0,
    window_minutes: int = 10,
) -> Optional[Dict[str, Any]]:
    """Check SBP change over a time window and provide instructions.

    Parameters
    ----------
    csv_path : Path or str
        Path to vital history CSV that contains ``timestamp`` and ``SBP`` columns.
    threshold : float, default 10.0
        Absolute change in SBP required to trigger an alarm.
    window_minutes : int, default 10
        Time window (minutes) to compare against the current SBP.

    Returns
    -------
    dict or None
        ``{"alarm": True, "change": diff, "instruction": str}`` if triggered,
        otherwise ``None``.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None

    records = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r.get("timestamp", ""))
            sbp = float(r.get("SBP", ""))
        except Exception:
            continue
        records.append({"timestamp": ts, "SBP": sbp})
    if not records:
        return None

    latest = records[-1]
    cutoff = latest["timestamp"] - timedelta(minutes=window_minutes)
    past_candidates = [r for r in records if r["timestamp"] <= cutoff]
    if not past_candidates:
        return None
    past = past_candidates[-1]

    diff = latest["SBP"] - past["SBP"]
    if diff >= threshold:
        return {
            "alarm": True,
            "change": diff,
            "instruction": "血圧上昇: 血管拡張薬を増量してください",
        }
    if diff <= -threshold:
        return {
            "alarm": True,
            "change": diff,
            "instruction": "血圧低下: 昇圧剤を増量してください",
        }
    return None
