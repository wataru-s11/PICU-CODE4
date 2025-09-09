import os
import sys
import types
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

from vitals.critical_spo2_logic import evaluate_critical_spo2
import main_surgery as ms
import pytest


def _df(rows):
    def iterrows():
        for i, r in enumerate(rows):
            yield i, r
    return SimpleNamespace(iterrows=iterrows)


def test_critical_spo2_high_triggers_reduction():
    vitals = {"SpO2": 105}
    thresholds = {"Critical_SpO2_u": 100}
    row = {
        "id": "CRIT_SpO2_ACUTE_UPPER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SpO2') > Critical_SpO2_u",
        "介入": "FiO2を下げ、NOを減量または中止",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    assert evaluate_critical_spo2(vitals, tree_df, thresholds) == [{
        "id": "CRIT_SpO2_ACUTE_UPPER",
        "instruction": "FiO2を下げ、NOを減量または中止",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_persistent_critical_spo2_recommends_nitrogen_or_threshold_change():
    """If high SpO2 persists after initial alert, advise nitrogen use or threshold change."""
    vitals = {"SpO2": 105}
    thresholds = {"Critical_SpO2_u": 100}
    row = {
        "id": "CRIT_SpO2_REEVAL_UPPER",
        "phase(acute=a, reevaluate=r)": "r",
        "condition": "vitals.get('SpO2') > Critical_SpO2_u",
        "介入": "窒素使用またはCritical_SpO2_uの基準値を変更することを検討してください",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    assert evaluate_critical_spo2(vitals, tree_df, thresholds, phase="r") == [{
        "id": "CRIT_SpO2_REEVAL_UPPER",
        "instruction": "窒素使用またはCritical_SpO2_uの基準値を変更することを検討してください",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_spo2_between_bounds_no_critical():
    """SpO2 between upper and critical upper should not trigger critical rule."""
    vitals = {"SpO2": 94}
    thresholds = {"SpO2_u": 92, "Critical_SpO2_u": 96}
    row = {
        "id": "CRIT_SpO2_ACUTE_UPPER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SpO2') > Critical_SpO2_u",
        "介入": "FiO2を下げ、NOを減量または中止",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    assert evaluate_critical_spo2(vitals, tree_df, thresholds) == []


def test_critical_spo2_low_triggers_airway_panel():
    vitals = {"SpO2": 70}
    thresholds = {"Critical_SpO2_l": 75}
    row = {
        "id": "CRIT_SpO2_ACUTE_LOWER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SpO2') < Critical_SpO2_l",
        "介入": "FiO2を上げ、NOを増量または再開、気道閉塞パネルを起動",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    assert evaluate_critical_spo2(vitals, tree_df, thresholds) == [{
        "id": "CRIT_SpO2_ACUTE_LOWER",
        "instruction": "FiO2を上げ、NOを増量または再開、気道閉塞パネルを起動",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_persistent_critical_spo2_low_recommends_no_or_threshold_change():
    """If low SpO2 persists after initial alert, advise NO use or threshold change."""
    vitals = {"SpO2": 70}
    thresholds = {"Critical_SpO2_l": 75}
    row = {
        "id": "CRIT_SpO2_REEVAL_LOWER",
        "phase(acute=a, reevaluate=r)": "r",
        "condition": "vitals.get('SpO2') < Critical_SpO2_l",
        "介入": "NO使用またはCritical_SpO2_lの基準値を変更することを検討してください",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    assert evaluate_critical_spo2(vitals, tree_df, thresholds, phase="r") == [{
        "id": "CRIT_SpO2_REEVAL_LOWER",
        "instruction": "NO使用またはCritical_SpO2_lの基準値を変更することを検討してください",
        "pause_min": "",
        "next_id": None,
        "comment": "",
    }]


def test_critical_spo2_string_values_do_not_trigger():
    """String vitals and thresholds should compare numerically."""
    vitals = {"SpO2": "91"}
    thresholds = {"Critical_SpO2_u": "100"}
    row = {
        "id": "CRIT_SpO2_ACUTE_UPPER",
        "phase(acute=a, reevaluate=r)": "a",
        "condition": "vitals.get('SpO2') > Critical_SpO2_u",
        "介入": "FiO2を下げ、NOを減量または中止",
        "備考": "",
        "ポーズ(min)": "",
        "再評価用NextID": None,
    }
    tree_df = _df([row])
    assert evaluate_critical_spo2(vitals, tree_df, thresholds) == []


def test_tree_critical_spo2_upper_condition():
    """Ensure tree.yaml triggers upper critical rule only above threshold."""
    pytest.importorskip("yaml")
    thresholds = {"Critical_SpO2_l": 75, "Critical_SpO2_u": 95}
    tree_df = ms.load_tree("tree.yaml")

    # Above upper threshold -> instruction present
    vitals = {"SpO2": 96}



def test_tree_critical_spo2_lower_condition():
    """Ensure tree.yaml triggers lower critical rule only below threshold."""
    pytest.importorskip("yaml")
    thresholds = {"Critical_SpO2_l": 75, "Critical_SpO2_u": 95}
    tree_df = ms.load_tree("tree.yaml")

    # Below lower threshold -> instruction present
    vitals = {"SpO2": 74}

