import os
import sys
import tkinter as tk

import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import blood_gas_panel


def _set_safe_values(panel: blood_gas_panel.BloodGasPanel) -> None:
    base_values = {
        "ph": 7.2,
        "pco2": 60,
        "po2": 80,
        "abe": -5,
        "hco3": 20,
        "k": 3.1,
        "ca": 1.0,
        "na": 140,
        "cl": 100,
    }
    for key, value in base_values.items():
        panel.vars[key].set(value)


def test_transfusion_message_when_rbc_sufficient(monkeypatch):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")

    panel = blood_gas_panel.BloodGasPanel(root)
    panel.disease_var.set("根治術")
    _set_safe_values(panel)
    panel.vars["hct"].set(30)
    panel.weight_var.set(20)
    panel.rbc_rate_var.set(20)
    panel.time_var.set("2025-01-01 00:00")
    panel._commit_current_time()
    assert panel.history["2025-01-01 00:00"]["weight_kg"] == 20
    assert panel.history["2025-01-01 00:00"]["rbc_ml_per_h"] == 20

    panel.vars["hct"].set(34)
    panel.weight_var.set(20)
    panel.rbc_rate_var.set(25)
    panel.time_var.set("2025-01-01 01:00")

    messages = []
    monkeypatch.setattr(blood_gas_panel.messagebox, "showinfo", lambda title, msg: messages.append(msg))

    result = panel.evaluate_current_data()

    assert result is not None
    joined = "\n".join(result["messages"])
    assert "十分な輸血がされています" in joined
    assert messages
    assert "十分な輸血がされています" in messages[0]
    root.destroy()


def test_transfusion_message_when_rbc_insufficient(monkeypatch):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")

    panel = blood_gas_panel.BloodGasPanel(root)
    panel.disease_var.set("根治術")
    _set_safe_values(panel)
    panel.vars["hct"].set(30)
    panel.weight_var.set(20)
    panel.rbc_rate_var.set(20)
    panel.time_var.set("2025-01-01 00:00")
    panel._commit_current_time()
    assert panel.history["2025-01-01 00:00"]["weight_kg"] == 20
    assert panel.history["2025-01-01 00:00"]["rbc_ml_per_h"] == 20

    panel.vars["hct"].set(34)
    panel.weight_var.set(20)
    panel.rbc_rate_var.set(15)
    panel.time_var.set("2025-01-01 01:00")

    messages = []
    monkeypatch.setattr(blood_gas_panel.messagebox, "showinfo", lambda title, msg: messages.append(msg))

    result = panel.evaluate_current_data()

    assert result is not None
    joined = "\n".join(result["messages"])
    assert "血管漏出が増えている可能性…ピトレシン0.02–0.05増量" in joined
    assert messages
    assert "血管漏出が増えている可能性…ピトレシン0.02–0.05増量" in messages[0]
    root.destroy()
