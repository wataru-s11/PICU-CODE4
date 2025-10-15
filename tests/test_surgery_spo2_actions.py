import sys
import types


if "pandas" not in sys.modules:
    def _pandas_stub_raise(*args, **kwargs):  # pragma: no cover - helper for tests without pandas
        raise ModuleNotFoundError("pandas is required for this operation")


    sys.modules["pandas"] = types.SimpleNamespace(
        DataFrame=_pandas_stub_raise,
        read_excel=_pandas_stub_raise,
        isna=lambda value: False,
    )


import main_surgery as ms


def test_adjust_spo2_actions_changes_instruction():
    instructions = [
        {"id": "SPO2_UPPER_FIO2_upper", "instruction": "orig"},
        {"id": "SPO2_LOWER", "instruction": "orig"},
        {"id": "OTHER", "instruction": "keep"},
    ]
    res = ms.adjust_spo2_actions(instructions, "Glenn")
    assert res[0]["instruction"] == ms.SPO2_ACTIONS["Glenn"]["upper"]
    assert res[1]["instruction"] == ms.SPO2_ACTIONS["Glenn"]["lower"]
    assert res[2]["instruction"] == "keep"


def test_adjust_spo2_actions_preserves_resolve():
    instructions = [
        {"id": "SPO2_UPPER_resolve", "instruction": "resolved"},
        {"id": "SPO2_LOWER_resolve", "instruction": "resolved"},
    ]
    res = ms.adjust_spo2_actions(instructions, "根治術")
    assert res == instructions


def test_adjust_spo2_actions_for_palliative_variants():
    instructions = [
        {"id": "SPO2_UPPER", "instruction": "orig"},
        {"id": "SPO2_LOWER", "instruction": "orig"},
    ]
    for surgery in ("姑息術（シャント）", "姑息術（バンド）"):
        res = ms.adjust_spo2_actions(instructions, surgery)
        assert res[0]["instruction"] == ms.SPO2_ACTIONS[surgery]["upper"]
        assert res[1]["instruction"] == ms.SPO2_ACTIONS[surgery]["lower"]


def test_shunt_low_flow_guidance_added():
    instructions = []
    vitals = {
        "SBP": 35,
        "SpO2": 85,
        "pitressin": 0.04,
        "adrenaline": 0.05,
    }
    thresholds = {"SBP_l": 40, "SpO2_u": 90}

    res = ms.adjust_spo2_actions(
        instructions,
        "姑息術（シャント）",
        vitals=vitals,
        thresholds=thresholds,
    )

    ids = {inst["id"] for inst in res}
    assert "SHUNT_LOW_FLOW_ALERT" in ids
    assert "SHUNT_LOW_FLOW_PITRESSIN" in ids
    assert "SHUNT_LOW_FLOW_ADRENALINE" in ids


def test_shunt_low_flow_not_added_when_thresholds_not_met():
    vitals = {"SBP": 50, "SpO2": 92, "pitressin": 0.03, "adrenaline": 0.05}
    thresholds = {"SBP_l": 40, "SpO2_u": 90}

    res = ms.adjust_spo2_actions(
        [],
        "姑息術（シャント）",
        vitals=vitals,
        thresholds=thresholds,
    )

    ids = {inst["id"] for inst in res}
    assert "SHUNT_LOW_FLOW_ALERT" not in ids
    assert "SHUNT_LOW_FLOW_PITRESSIN" not in ids
    assert "SHUNT_LOW_FLOW_ADRENALINE" not in ids
