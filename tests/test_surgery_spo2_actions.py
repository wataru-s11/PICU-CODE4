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
