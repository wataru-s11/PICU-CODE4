from types import SimpleNamespace
import sys, types

pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

import main_surgery as ms
from vitals.spo2_logic import evaluate_spo2
import pytest


def _df(rows):
    def iterrows():
        for i, r in enumerate(rows):
            yield i, r
    return SimpleNamespace(iterrows=iterrows)


def test_spo2_check_triggered_when_out_of_range():
    vitals = {"SpO2": 70}
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    row = {
        "id": "SPO2_CHECK",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SpO2') < SpO2_l or vitals.get('SpO2') > SpO2_u",
        "介入": "check",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    result = evaluate_spo2(vitals, tree_df, thresholds)
    assert result == [{
        "id": "SPO2_CHECK",
        "instruction": "check",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_spo2_high_reduces_fio2():
    vitals = {"SpO2": 105, "FiO2": 40}
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_CHECK",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l or vitals.get('SpO2') > SpO2_u",
            "介入": "check",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
        {
            "id": "SPO2_UPPER_FIO2_upper",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') > SpO2_u and vitals.get('FiO2') > 21",
            "介入": "FiO2を10％下げる",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
    ]
    tree_df = _df(rows)
    result = [r for r in evaluate_spo2(vitals, tree_df, thresholds) if r["id"] != "SPO2_CHECK"]
    assert result == [{
        "id": "SPO2_UPPER_FIO2_upper",
        "instruction": "FiO2を10％下げる",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_spo2_high_reduces_no_when_fio2_low():
    vitals = {"SpO2": 105, "FiO2": 21, "NO": 20}
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_CHECK",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l or vitals.get('SpO2') > SpO2_u",
            "介入": "check",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
        {
            "id": "SPO2_UPPER_NO",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') > SpO2_u and vitals.get('FiO2', 21) <= 21 and vitals.get('NO', 0) > 0",
            "介入": "NOを減量する",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
    ]
    tree_df = _df(rows)
    result = [r for r in evaluate_spo2(vitals, tree_df, thresholds) if r["id"] != "SPO2_CHECK"]
    assert result == [{
        "id": "SPO2_UPPER_NO",
        "instruction": "NOを減量する",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_spo2_low_increases_fio2():
    vitals = {"SpO2": 70, "FiO2": 40}
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_CHECK",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l or vitals.get('SpO2') > SpO2_u",
            "介入": "check",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
        {
            "id": "SPO2_LOWER",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l and vitals.get('FiO2') < 100",
            "介入": "FiO2を10％上げる",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
    ]
    tree_df = _df(rows)
    result = [r for r in evaluate_spo2(vitals, tree_df, thresholds) if r["id"] != "SPO2_CHECK"]
    assert result == [{
        "id": "SPO2_LOWER",
        "instruction": "FiO2を10％上げる",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_spo2_low_increases_no_when_fio2_max():
    vitals = {"SpO2": 70, "FiO2": 100}
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_CHECK",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l or vitals.get('SpO2') > SpO2_u",
            "介入": "check",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
        {
            "id": "SPO2_LOWER_FIO2_100",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l and vitals.get('FiO2') >= 100",
            "介入": "NO20ppmで使用してください",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
    ]
    tree_df = _df(rows)
    result = [r for r in evaluate_spo2(vitals, tree_df, thresholds) if r["id"] != "SPO2_CHECK"]
    assert result == [{
        "id": "SPO2_LOWER_FIO2_100",
        "instruction": "NO20ppmで使用してください",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_handle_spo2_check_n_sets_pause(monkeypatch):
    vitals_memory = {}
    monkeypatch.setattr(ms.time, "time", lambda: 100)
    ms.handle_spo2_check_n(vitals_memory)
    assert vitals_memory["SPO2_CHECK_PAUSE_UNTIL"] == 100 + 60 * 60


def test_tree_spo2_lower_condition():
    """Ensure tree.yaml only triggers SPO2_LOWER when SpO2 is below threshold."""
    pytest.importorskip("yaml")
    thresholds = {"SpO2_l": 80, "SpO2_u": 92}
    tree_df = ms.load_tree("tree.yaml")

    # SpO2 above lower threshold -> no instruction
    vitals = {"SpO2": 91, "FiO2": 21}
    ids = [r["id"] for r in evaluate_spo2(vitals, tree_df, thresholds)]
    assert "SPO2_LOWER" not in ids

    # SpO2 below threshold -> instruction appears
    vitals = {"SpO2": 70, "FiO2": 21}
    ids = [r["id"] for r in evaluate_spo2(vitals, tree_df, thresholds)]
    assert "SPO2_LOWER" in ids


def test_evaluate_all_requires_spo2_check_before_actions(monkeypatch):
    vitals = {"SpO2": 70}
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_CHECK",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l",
            "介入": "check",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
        {
            "id": "SPO2_LOWER",
            "phase(acute=a, reevaluate=r)": "a",
            "condition": "vitals.get('SpO2') < SpO2_l",
            "介入": "FiO2を10％上げる",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
    ]
    tree_df = _df(rows)

    for fname in [
        "evaluate_cvp",
        "evaluate_sbp",
        "evaluate_critical_spo2",
        "evaluate_adrenaline",
        "evaluate_dobutamine",
        "evaluate_bpup",
        "evaluate_bpdown",
        "evaluate_bleed",
        "evaluate_transfusion",
    ]:
        monkeypatch.setattr(ms, fname, lambda *a, **k: [])

    res = ms.evaluate_all(vitals, tree_df, thresholds)
    assert [r["id"] for r in res] == ["SPO2_CHECK"]

    vitals["SPO2_CHECK_DONE"] = "Y"
    res2 = ms.evaluate_all(vitals, tree_df, thresholds)
    ids2 = [r["id"] for r in res2 if r["id"] != "SPO2_CHECK"]
    assert ids2 == ["SPO2_LOWER"]


def test_spo2_resolve_requires_normal_value():
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_UPPER_resolve",
            "phase(acute=a, reevaluate=r)": "r",
            "condition": "vitals.get('SpO2') >= SpO2_l and vitals.get('SpO2') <= SpO2_u",
            "介入": "resolved",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
        {
            "id": "SPO2_LOWER_resolve",
            "phase(acute=a, reevaluate=r)": "r",
            "condition": "vitals.get('SpO2') >= SpO2_l and vitals.get('SpO2') <= SpO2_u",
            "介入": "resolved",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        },
    ]
    tree_df = _df(rows)

    vitals_ok = {"SpO2": 95}
    res_ok = evaluate_spo2(vitals_ok, tree_df, thresholds, phase="r")
    assert res_ok == [
        {
            "id": "SPO2_UPPER_resolve",
            "instruction": "resolved",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        },
        {
            "id": "SPO2_LOWER_resolve",
            "instruction": "resolved",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        },
    ]

    vitals_bad = {"SpO2": 105}
    res_bad = evaluate_spo2(vitals_bad, tree_df, thresholds, phase="r")
    assert res_bad == []


def test_spo2_lower_no20_triggers_only_when_low():
    thresholds = {"SpO2_l": 80, "SpO2_u": 100}
    rows = [
        {
            "id": "SPO2_LOWER_NO_20",
            "phase(acute=a, reevaluate=r)": "r",
            "condition": "vitals.get('SpO2') < SpO2_l and vitals.get('FiO2') >= 100 and vitals.get('NO', 0) >= 20",
            "介入": "escalate",
            "備考": "",
            "ポーズ(min)": "",
            "再評価用NextID": None,
        }
    ]
    tree_df = _df(rows)

    vitals_trigger = {"SpO2": 70, "FiO2": 100, "NO": 20}
    res_trigger = evaluate_spo2(vitals_trigger, tree_df, thresholds, phase="r")
    assert res_trigger == [
        {
            "id": "SPO2_LOWER_NO_20",
            "instruction": "escalate",
            "pause_min": "",
            "next_id": None,
            "comment": "",
        }
    ]

    vitals_no_trigger = {"SpO2": 85, "FiO2": 100, "NO": 20}
    res_no_trigger = evaluate_spo2(vitals_no_trigger, tree_df, thresholds, phase="r")
    assert res_no_trigger == []

