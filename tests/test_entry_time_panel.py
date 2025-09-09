import datetime as dt
import tkinter as tk
import pytest

from entry_time_panel import EntryTimePanel, determine_phase


def test_determine_phase():
    assert determine_phase(dt.timedelta(hours=2)) == "術後3時間以内"
    assert determine_phase(dt.timedelta(hours=5)) == "術後3-24時間"
    assert determine_phase(dt.timedelta(hours=30)) == "術後24時間以降"


def test_entry_time_panel_construct():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    panel = EntryTimePanel(root)
    panel.pack()
    root.destroy()
