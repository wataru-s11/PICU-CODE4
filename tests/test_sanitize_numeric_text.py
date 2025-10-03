import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vital_reader import sanitize_numeric_text


@pytest.mark.parametrize(
    "input_text, allow_dot, expected",
    [
        ("5・0", True, "5.0"),
        ("5・0", False, "50"),
        ("7･2", True, "7.2"),
        ("8·3", True, "8.3"),
        ("9点4", True, "9.4"),
    ],
)
def test_sanitize_numeric_text_decimal_variants(input_text, allow_dot, expected):
    assert sanitize_numeric_text(input_text, allow_dot=allow_dot) == expected


def test_sanitize_numeric_text_signs_preserved():
    assert sanitize_numeric_text("-1・5", allow_dot=True, allow_sign=True) == "-1.5"
