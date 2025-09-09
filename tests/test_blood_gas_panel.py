import os
import sys
import csv
import tkinter as tk
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import blood_gas_panel


def test_blood_gas_panel_get_values():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    panel = blood_gas_panel.BloodGasPanel(root)
    panel.pack()
    values = panel.get_values()
    expected = {"ph", "pco2", "po2", "hct", "k", "na", "cl", "ca", "glu", "lac", "tbil", "hco3", "abe", "alb"}
    assert set(values.keys()) == expected
    root.destroy()


def test_blood_gas_panel_record(tmp_path):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    os.chdir(tmp_path)
    csv_path = tmp_path / "vitals.csv"
    panel = blood_gas_panel.BloodGasPanel(root, csv_path=str(csv_path))
    panel.vars["ph"].set(7.30)
    panel.time_var.set("2025-01-02 03:04")
    panel._commit_current_time()
    assert "2025-01-02 03:04" in panel.history
    if blood_gas_panel.Workbook is not None:
        assert os.path.exists("blood_gas_panel.xlsx")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["timestamp"] == "2025-01-02 03:04:00"
    assert float(rows[-1]["pH"]) == pytest.approx(7.30)
    root.destroy()


def test_blood_gas_panel_evaluate(monkeypatch):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    panel = blood_gas_panel.BloodGasPanel(root)
    # set sample values
    panel.vars["ph"].set(7.20)
    panel.vars["pco2"].set(60)
    panel.vars["po2"].set(80)
    panel.vars["abe"].set(-5)
    panel.vars["hco3"].set(20)
    panel.vars["k"].set(3.1)
    panel.vars["ca"].set(1.0)
    panel.vars["hct"].set(30)
    panel.vars["na"].set(140)
    panel.vars["cl"].set(100)
    panel.disease_var.set("根治術")

    msgs = []
    monkeypatch.setattr(blood_gas_panel.messagebox, "showinfo", lambda title, msg: msgs.append(msg))
    result = panel.evaluate_current_data()
    assert result is not None
    assert "一次性呼吸性アシドーシス" in "\n".join(result["messages"])
    root.destroy()
