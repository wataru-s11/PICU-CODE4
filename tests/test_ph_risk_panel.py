import csv
import tkinter as tk
from tkinter import messagebox
import pytest

from ph_risk_panel import score_pre_ph, score_current_ph, PHRiskPanel


def test_score_pre_ph():
    total, pre_point = score_pre_ph(
        days=2,
        xray_peripheral=True,
        lung_opacity=False,
        down_syndrome=False,
        vein_stenosis=False,
    )
    assert total == 4
    assert pre_point == 1


def test_score_current_ph():
    # No preoperative risk
    _, pre_point = score_pre_ph(40, False, False, False, False)
    assert pre_point == 0
    total = score_current_ph(
        pre_point,
        po2_decreasing=True,
        pco2_increasing=True,
        cvp_increasing=False,
        urine_decreasing=False,
    )
    assert total == 2


def test_ph_risk_panel_record(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")

    # Suppress message boxes during the test
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: None)

    csv_path = tmp_path / "vitals.csv"
    panel = PHRiskPanel(root, csv_path=str(csv_path))
    panel.days_var.set(5)
    panel.xray_var.set(True)
    panel.po2_var.set(True)
    panel.time_var.set("2025-01-02 03:04")
    panel._on_check()
    root.destroy()

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["PH_pre"] == "3"
    assert rows[0]["PH_current"] == "2"
    assert rows[0]["timestamp"] == "2025-01-02 03:04:00"
