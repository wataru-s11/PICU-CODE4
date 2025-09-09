import os
import sys
import types

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Minimal pandas stub so tree_parser can be imported without the real dependency
pd_stub = types.SimpleNamespace()
sys.modules.setdefault("pandas", pd_stub)

from common.tree_parser import _parse_condition


def test_parse_condition_uses_get_for_vitals():
    cond = "vitals['CVP'] > {{CVP_u}} and vitals['SBP'] > {{SBP_u}}"
    parsed = _parse_condition(cond, "")
    assert parsed == "vitals.get('CVP') > CVP_u and vitals.get('SBP') > SBP_u"
