import os
import os
import sys
import types

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

from common.rule_engine import evaluate_rules


class DummyDF:
    def __init__(self, rows):
        self._rows = rows
        self.empty = len(rows) == 0

    def iterrows(self):
        for idx, row in enumerate(self._rows):
            yield idx, row


def test_basic_condition_evaluation():
    vitals = {"SBP": 120}
    row = {
        "id": "SBP_HIGH",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "SBP > 100",
        "介入": "ok",
        "ポーズ(min)": "",
        "再評価用NextID": None,
        "備考": "",
    }
    df = DummyDF([row])
    result = evaluate_rules(vitals, df, ["SBP"])
    assert result == [{
        "id": "SBP_HIGH",
        "instruction": "ok",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_threshold_change_triggers_reevaluation():
    vitals = {"SpO2": 97}
    row = {
        "id": "CRIT_SpO2_ACUTE_UPPER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals['SpO2'] > Critical_SpO2_u",
        "介入": "lower FiO2",
        "ポーズ(min)": "",
        "再評価用NextID": None,
        "備考": "",
    }
    df = DummyDF([row])

    first = evaluate_rules(vitals, df, ["CRIT"], {"Critical_SpO2_u": 95})
    assert first, "Rule should trigger with lower threshold"

    second = evaluate_rules(vitals, df, ["CRIT"], {"Critical_SpO2_u": 99})
    assert second == [], "Rule should not trigger after threshold increase"
