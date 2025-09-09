import importlib
import builtins
import sys


def test_drug_panel_import_independent(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fluid_panel":
            raise ImportError("missing fluid panel")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "main_surgery", raising=False)
    monkeypatch.delitem(sys.modules, "fluid_panel", raising=False)

    ms = importlib.import_module("main_surgery")
    assert ms.launch_fluid_panel is None
    assert ms.launch_drug_panel is not None
