import os
import sys
import types

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Minimal pandas stub
pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

from vitals.cvp_logic import evaluate_cvp


class DummyDF:
    def __init__(self, rows):
        self._rows = rows
        self.empty = len(rows) == 0

    def iterrows(self):
        for idx, row in enumerate(self._rows):
            yield idx, row


def test_additional_condition_with_value_placeholder():
    vitals = {"SBP": 120}
    thresholds = {"SBP_u": 90}
    row = {
        "id": 1,
        "phase(acute=a, reevaluate=r)": "a",
        "項目": "SBP",
        "条件": "value > 0",
        "追加項目1": "SBP",
        "追加条件1": "value > {{SBP_u}}",
        "介入": "ok",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds)
    assert result == [
        {
            "id": 1,
            "instruction": "ok",
            "comment": "",
            "pause_min": "",
        }
    ]


def test_additional_condition_ignored_when_value_missing():
    vitals = {"SBP": 120}
    thresholds = {"SBP_u": 90}
    row = {
        "id": 1,
        "phase(acute=a, reevaluate=r)": "a",
        "項目": "SBP",
        "条件": "value > 0",
        "追加項目1": "MISSING",
        "追加条件1": "value > {{SBP_u}}",
        "介入": "ok",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    # Should not raise and should still evaluate main condition
    result = evaluate_cvp(vitals, tree_df, thresholds)
    assert result == [
        {
            "id": 1,
            "instruction": "ok",
            "comment": "",
            "pause_min": "",
        }
    ]


def test_cvp_upper_check_triggered_when_exceeding_threshold():
    vitals = {"CVP": 10}
    thresholds = {"CVP_u": 8}
    row = {
        "id": "CVP_UPPER_CHECK",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('CVP') > CVP_u",
        "介入": "check",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds)
    assert result == [
        {
            "id": "CVP_UPPER_CHECK",
            "instruction": "check",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        }
    ]


def test_cvp_upper_a_sbp_upper_requires_high_cvp():
    vitals = {"SBP": 100, "CVP": 4}
    thresholds = {"SBP_u": 90, "CVP_u": 5}
    row = {
        "id": "CVP_UPPER_A_SBP_UPPER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SBP') > SBP_u and vitals.get('CVP') > CVP_u",
        "介入": "furosemide",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds)
    assert result == []


def test_cvp_upper_a_sbp_upper_triggers_when_cvp_high():
    vitals = {"SBP": 100, "CVP": 6}
    thresholds = {"SBP_u": 90, "CVP_u": 5}
    row = {
        "id": "CVP_UPPER_A_SBP_UPPER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SBP') > SBP_u and vitals.get('CVP') > CVP_u",
        "介入": "furosemide",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds)
    assert result == [
        {
            "id": "CVP_UPPER_A_SBP_UPPER",
            "instruction": "furosemide",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        }
    ]


def test_cvp_upper_check_y_requires_high_cvp():
    vitals = {"CVP": 3, "CVP_LINE_CHECK": "Y"}
    thresholds = {"CVP_u": 4}
    row = {
        "id": "CVP_UPPER_CHECK_Y",
        "phase(acute=a, reevaluate=r)": "r",
        "condition": "vitals.get('CVP_LINE_CHECK') == 'Y' and vitals.get('CVP') > CVP_u",
        "介入": "echo",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds, phase='r')
    assert result == []


def test_cvp_upper_check_y_triggers_with_high_cvp():
    vitals = {"CVP": 5, "CVP_LINE_CHECK": "Y"}
    thresholds = {"CVP_u": 4}
    row = {
        "id": "CVP_UPPER_CHECK_Y",
        "phase(acute=a, reevaluate=r)": "r",
        "condition": "vitals.get('CVP_LINE_CHECK') == 'Y' and vitals.get('CVP') > CVP_u",
        "介入": "echo",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds, phase='r')
    assert result == [
        {
            "id": "CVP_UPPER_CHECK_Y",
            "instruction": "echo",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        }
    ]


def test_cvp_upper_a_sbp_upper_triggers_after_line_check():
    vitals = {"SBP": 100, "CVP": 6}
    thresholds = {"SBP_u": 90, "CVP_u": 5}
    row = {
        "id": "CVP_UPPER_A_SBP_UPPER",
        "phase(acute=a, reevaluate=r)": "r",
        "condition": "vitals.get('SBP') > SBP_u and vitals.get('CVP') > CVP_u",
        "介入": "furosemide",
        "備考": "",
        "ポーズ(min)": "",
    }
    tree_df = DummyDF([row])
    result = evaluate_cvp(vitals, tree_df, thresholds, phase='r')
    assert result == [
        {
            "id": "CVP_UPPER_A_SBP_UPPER",
            "instruction": "furosemide",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        }
    ]

