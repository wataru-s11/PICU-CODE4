import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import panel_tabs

created_notebooks = []

class DummyNotebook:
    def __init__(self, master):
        self.tabs = []
        created_notebooks.append(self)
    def pack(self, *args, **kwargs):
        pass
    def add(self, widget, text=""):
        self.tabs.append(text)

class DummyTk:
    def title(self, _):
        pass
    def mainloop(self):
        pass
    def attributes(self, *args, **kwargs):
        pass

def test_run_assessment_tabs_adds_panels(monkeypatch):
    def simple_module(attr_name):
        mod = types.ModuleType("m")
        setattr(mod, attr_name, lambda *args, **kwargs: None)
        return mod

    monkeypatch.setitem(sys.modules, "extubation_panel", simple_module("ExtubationPanel"))
    monkeypatch.setitem(sys.modules, "airway_obstruction_panel", simple_module("AirwayPanel"))
    monkeypatch.setitem(sys.modules, "bleeding_panel", simple_module("BleedingPanel"))
    monkeypatch.setitem(sys.modules, "weaning_panel", simple_module("WeaningPanel"))
    monkeypatch.setitem(sys.modules, "threshold_panel", simple_module("ThresholdPanel"))

    monkeypatch.setattr(panel_tabs.tk, "Tk", DummyTk)
    monkeypatch.setattr(panel_tabs.ttk, "Notebook", DummyNotebook)

    panel_tabs.run_assessment_tabs()
    notebook = created_notebooks[0]
    assert set(["抜管基準", "気道", "出血", "ウィーニング"]).issubset(set(notebook.tabs))
