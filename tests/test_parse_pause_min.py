import os
import sys
import types

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

import pytest
from main_surgery import parse_pause_min, DEFAULT_PAUSE_MIN

@pytest.mark.parametrize("value, expected", [
    ("60", 60),
    ("60m", 60),
    ("600s", 10),
    ("00:10:00", 10),
    ("", DEFAULT_PAUSE_MIN),
    (None, DEFAULT_PAUSE_MIN),
])
def test_parse_pause_min(value, expected):
    assert parse_pause_min(value) == expected
