import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vital_reader import (
    apply_pressure_support_split,
    is_pressure_support_mode,
    normalize_vent_mode_label,
)


def test_normalize_vent_mode_translates_katakana_label():
    assert normalize_vent_mode_label("オートモード") == "AUTO MODE"


def test_normalize_vent_mode_normalizes_ascii_case_and_spacing():
    assert normalize_vent_mode_label("  auto mode  ") == "AUTO MODE"


def test_normalize_vent_mode_handles_full_width_space():
    assert normalize_vent_mode_label("オート\u3000モード") == "AUTO MODE"


def test_normalize_vent_mode_handles_mixed_ascii_and_katakana():
    assert normalize_vent_mode_label("AUTOモード") == "AUTO MODE"


def test_is_pressure_support_mode_detects_psv_token():
    assert is_pressure_support_mode("SIMV/PSV")


def test_apply_pressure_support_split_assigns_ps_when_mode_matches():
    results = {"VentMode": "SPONT", "VTset": "12"}
    apply_pressure_support_split(results)
    assert results["PS"] == "12"
    assert results["VTset"] == ""


def test_apply_pressure_support_split_keeps_vtset_when_not_ps_mode():
    results = {"VentMode": "SIMV", "VTset": "350"}
    apply_pressure_support_split(results)
    assert results["PS"] == ""
    assert results["VTset"] == "350"
