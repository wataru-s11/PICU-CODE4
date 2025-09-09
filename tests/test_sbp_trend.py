import csv
import os
import tempfile

from vitals.sbp_trend import check_sbp_trend


def create_csv(rows):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    writer = csv.DictWriter(tmp, fieldnames=["timestamp", "SBP"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    tmp.close()
    return tmp.name


def test_sbp_increase_triggers_vasodilator_instruction():
    path = create_csv([
        {"timestamp": "2024-01-01 00:00:00", "SBP": 80},
        {"timestamp": "2024-01-01 00:10:00", "SBP": 95},
    ])
    try:
        result = check_sbp_trend(path)
        assert result and result["alarm"]
        assert result["change"] == 15
        assert "血管拡張薬" in result["instruction"]
    finally:
        os.unlink(path)


def test_sbp_decrease_triggers_vasopressor_instruction():
    path = create_csv([
        {"timestamp": "2024-01-01 00:00:00", "SBP": 100},
        {"timestamp": "2024-01-01 00:10:00", "SBP": 85},
    ])
    try:
        result = check_sbp_trend(path)
        assert result and result["alarm"]
        assert result["change"] == -15
        assert "昇圧剤" in result["instruction"]
    finally:
        os.unlink(path)
