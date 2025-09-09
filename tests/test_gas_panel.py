import os
import sys
import csv
import tkinter as tk
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gas_panel


def test_gas_panel_get_values():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    panel = gas_panel.GasPanel(root)
    panel.pack()
    vals = panel.get_values()
    assert set(vals.keys()) == {"FiO2", "NO", "N2"}
    root.destroy()


def test_gas_panel_record(tmp_path):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    csv_path = tmp_path / "vitals.csv"
    panel = gas_panel.GasPanel(root, csv_path=str(csv_path))
    panel.vars["FiO2"].set(40.0)
    panel.vars["NO"].set(20.0)
    panel.vars["N2"].set(30.0)
    panel.time_var.set("2025-01-02 03:04")
    panel._commit_current_time()
    assert "2025-01-02 03:04" in panel.history
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows[0]["FiO2"] == "40.0"
    assert rows[0]["NO"] == "20.0"
    assert rows[0]["N2"] == "30.0"
    assert rows[0]["timestamp"] == "2025-01-02 03:04:00"
    root.destroy()
