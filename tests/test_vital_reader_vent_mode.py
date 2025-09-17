import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vital_reader import normalize_vent_mode_label


def test_normalize_vent_mode_translates_katakana_label():
    assert normalize_vent_mode_label("オートモード") == "AUTO MODE"


def test_normalize_vent_mode_normalizes_ascii_case_and_spacing():
    assert normalize_vent_mode_label("  auto mode  ") == "AUTO MODE"


def test_normalize_vent_mode_handles_full_width_space():
    assert normalize_vent_mode_label("オート\u3000モード") == "AUTO MODE"


def test_normalize_vent_mode_handles_mixed_ascii_and_katakana():
    assert normalize_vent_mode_label("AUTOモード") == "AUTO MODE"
