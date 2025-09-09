import os
import sys
import tkinter as tk
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import extubation_panel


def test_evaluate_extubation_basic():
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    assert extubation_panel.evaluate_extubation(10, 5, 70, 90, thresholds)

def test_extubation_panel_construct():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    panel = extubation_panel.ExtubationPanel(root)
    panel.pack()
    root.destroy()
