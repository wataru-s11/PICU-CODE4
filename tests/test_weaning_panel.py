import os
import sys
import tkinter as tk
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import weaning_panel


def test_manage_weaning_basic(capsys):
    responses = iter(["1", "いいえ", ""])  # TOF<2 then exit
    weaning_panel.manage_weaning(input_func=lambda _: next(responses))
    out = capsys.readouterr().out
    assert "TOFが 2 回未満です" in out


def test_panel_ps_guidance(monkeypatch):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    panel = weaning_panel.WeaningPanel(root)
    panel.tof_var.set(2)
    panel.pco2_var.set(True)
    panel.pause_var.set(True)
    msgs = []
    monkeypatch.setattr(weaning_panel.messagebox, "showinfo", lambda t, m: msgs.append(m))
    panel._on_check()
    assert msgs[-1] == "TV≧7ml/kgを維持してPSを10まで段階的に下げてください。"
    root.destroy()
