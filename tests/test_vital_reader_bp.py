import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vital_reader


def test_parse_bp_text_strips_punctuation():
    raw = "11.0/97(57)"
    text, sbp, dbp, map_val = vital_reader.parse_bp_text(raw)
    assert text == "110/97(57)"
    assert sbp == "110"
    assert dbp == "97"
    assert map_val == "57"


def test_parse_bp_text_handles_none():
    text, sbp, dbp, map_val = vital_reader.parse_bp_text(None)  # type: ignore[arg-type]
    assert text == ""
    assert sbp == ""
    assert dbp == ""
    assert map_val == ""


def test_parse_bp_text_fallback_four_digits():
    raw = "11971(89)"
    text, sbp, dbp, map_val = vital_reader.parse_bp_text(raw)
    assert text == "119/71(89)"
    assert sbp == "119"
    assert dbp == "71"
    assert map_val == "89"


def test_parse_bp_text_handles_space_separated_decimals():
    raw = "11.0 97.0 57.0"
    text, sbp, dbp, map_val = vital_reader.parse_bp_text(raw)
    assert text == "110/97(57)"
    assert sbp == "110"
    assert dbp == "97"
    assert map_val == "57"


def test_parse_bp_text_handles_missing_slash():
    raw = "11097(57"
    text, sbp, dbp, map_val = vital_reader.parse_bp_text(raw)
    assert text == "110/97(57)"
    assert sbp == "110"
    assert dbp == "97"
    assert map_val == "57"


def test_parse_bp_text_trims_out_of_range_values():
    raw = "102.0 461.0 477.0"
    text, sbp, dbp, map_val = vital_reader.parse_bp_text(raw)
    assert text == "102/61(77)"
    assert sbp == "102"
    assert dbp == "61"
    assert map_val == "77"


@pytest.mark.skipif(vital_reader.cv2 is None, reason="OpenCV not available")
def test_read_bp_roi_uses_sanitized_text(monkeypatch):
    np = pytest.importorskip("numpy")
    responses = iter([
        ("11.0/97(57)", 0.9),
        ("", 0.0),
        ("", 0.0),
        ("", 0.0),
    ])

    def fake_read_easy(img, allow):
        return next(responses)

    monkeypatch.setattr(vital_reader, "read_easy", fake_read_easy)

    roi = np.zeros((10, 10, 3), dtype=np.uint8)
    text, sbp, dbp, map_val = vital_reader.read_bp_roi(roi)

    assert text == "110/97(57)"
    assert sbp == "110"
    assert dbp == "97"
    assert map_val == "57"
