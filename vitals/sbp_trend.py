from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union


BASE_INSTRUCTION = "急激な血圧の増減を確認！直前に行った介入を確認し、バイタルを注意深く観察する"


def _parse_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


def _format_change(value: float) -> str:
    text = f"{value:.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


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
        vasopressin = _parse_float(r.get("vasopressin"), r.get("pitressin"))
        hanp = _parse_float(r.get("hanp"))
        records.append({
            "timestamp": ts,
            "SBP": sbp,
            "vasopressin": vasopressin,
            "hanp": hanp,
        })
    if not records:
        return None

    latest = records[-1]
    cutoff = latest["timestamp"] - timedelta(minutes=window_minutes)
    past_candidates = [r for r in records if r["timestamp"] <= cutoff]
    if not past_candidates:
        return None
    past = past_candidates[-1]

    diff = latest["SBP"] - past["SBP"]
    vasopressin = latest.get("vasopressin")
    hanp = latest.get("hanp")
    if diff >= threshold:
        change_text = _format_change(diff)
        instruction_lines = [
            BASE_INSTRUCTION,
            f"SBPが10分で{change_text} mmHg上昇しています。",
        ]
        if vasopressin is not None and vasopressin > 0.03:
            instruction_lines.append(
                f"vasopressinが0.03U/kg/hを超えています（現在: {vasopressin:.3f}）。ピトレシンを減量することを検討してください。"
            )
        if hanp is not None and hanp < 0.2:
            instruction_lines.append(
                f"HANPが0.2μg/kg/min未満です（現在: {hanp:.3f}）。ハンプの増量を検討してください。"
            )
        instruction = "\n".join(instruction_lines)
        return {"alarm": True, "change": diff, "instruction": instruction}
    if diff <= -threshold:
        change_text = _format_change(abs(diff))
        instruction_lines = [
            BASE_INSTRUCTION,
            f"SBPが10分で{change_text} mmHg低下しています。",
            "出血パネルを入力してください。",
        ]
        if vasopressin is not None and vasopressin < 0.1:
            instruction_lines.append(
                f"vasopressinが0.1U/kg/h未満です（現在: {vasopressin:.3f}）。ピトレシンの増量を検討してください。"
            )
        if hanp is not None and hanp < 0.05:
            instruction_lines.append(
                f"HANPが0.05μg/kg/min未満です（現在: {hanp:.3f}）。ハンプを0.05μg/kg/minまで増量することを検討してください。"
            )
        instruction = "\n".join(instruction_lines)
        return {"alarm": True, "change": diff, "instruction": instruction}
    return None
