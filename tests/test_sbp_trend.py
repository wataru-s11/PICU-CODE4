import csv
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vitals.sbp_trend import check_sbp_trend


def create_csv(rows):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    fieldnames = ["timestamp", "SBP"]
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    tmp.close()
    return tmp.name


ALERT_MESSAGE = "急激な血圧の増減を確認！直前に行った介入を確認し、バイタルを注意深く観察する"


def test_sbp_increase_high_vasopressin_suggests_pitressin_reduction():
    path = create_csv([
        {"timestamp": "2024-01-01 00:00:00", "SBP": 80, "vasopressin": 0.02, "hanp": 0.25},
        {"timestamp": "2024-01-01 00:10:00", "SBP": 95, "vasopressin": 0.04, "hanp": 0.25},
    ])
    try:
        result = check_sbp_trend(path)
        assert result and result["alarm"]
        assert result["change"] == 15
        instruction = result["instruction"]
        assert ALERT_MESSAGE in instruction
        assert "上昇" in instruction
        assert "ピトレシンを減量" in instruction
        assert "ハンプの増量" not in instruction
    finally:
        os.unlink(path)


def test_sbp_increase_low_hanp_suggests_hanp_increase():
    path = create_csv([
        {"timestamp": "2024-01-01 00:00:00", "SBP": 85, "vasopressin": 0.02, "hanp": 0.18},
        {"timestamp": "2024-01-01 00:10:00", "SBP": 100, "vasopressin": 0.02, "hanp": 0.18},
    ])
    try:
        result = check_sbp_trend(path)
        assert result and result["alarm"]
        assert result["change"] == 15
        instruction = result["instruction"]
        assert ALERT_MESSAGE in instruction
        assert "上昇" in instruction
        assert "ハンプの増量を検討" in instruction
    finally:
        os.unlink(path)


def test_sbp_decrease_prompts_bleeding_panel_and_drug_adjustments():
    path = create_csv([
        {"timestamp": "2024-01-01 00:00:00", "SBP": 110, "vasopressin": 0.12, "hanp": 0.07},
        {"timestamp": "2024-01-01 00:10:00", "SBP": 95, "vasopressin": 0.08, "hanp": 0.03},
    ])
    try:
        result = check_sbp_trend(path)
        assert result and result["alarm"]
        assert result["change"] == -15
        instruction = result["instruction"]
        assert ALERT_MESSAGE in instruction
        assert "低下" in instruction
        assert "出血パネル" in instruction
        assert "ピトレシンの増量を検討" in instruction
        assert "ハンプを0.05" in instruction
    finally:
        os.unlink(path)
