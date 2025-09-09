import importlib


def test_panel_tabs_module_available():
    mod = importlib.import_module("panel_tabs")
    assert hasattr(mod, "launch_drug_fluid_tabs")
    assert hasattr(mod, "launch_assessment_tabs")
