import pytest

import main_surgery as ms


def test_evaluate_all_returns_observation_when_no_rules(monkeypatch):
    def empty(*args, **kwargs):
        return []

    for fname in [
        "evaluate_spo2",
        "evaluate_critical_spo2",
        "evaluate_sbp",
        "evaluate_cvp",
        "evaluate_adrenaline",
        "evaluate_dobutamine",
        "evaluate_bpup",
        "evaluate_bpdown",
        "evaluate_bleed",
        "evaluate_transfusion",
    ]:
        monkeypatch.setattr(ms, fname, empty)

    result = ms.evaluate_all({}, None, {})
    assert result == [{
        "id": "OBSERVATION",
        "instruction": "経過観察",
        "pause_min": 0,
        "next_id": None,
        "comment": "",
    }]


def test_cvp_no_priority_includes_spo2(monkeypatch):
    """CVP 指示があっても SpO2 指示が同時に評価される"""

    def mock_cvp(*args, **kwargs):
        return [{
            "id": "CVP_R",
            "instruction": "cvp",
            "pause_min": 1,
            "next_id": None,
            "comment": "",
        }]

    def mock_spo2(*args, **kwargs):
        return [{
            "id": "SPO2_R",
            "instruction": "spo2",
            "pause_min": 1,
            "next_id": None,
            "comment": "",
        }]

    monkeypatch.setattr(ms, "evaluate_cvp", mock_cvp)
    monkeypatch.setattr(ms, "evaluate_spo2", mock_spo2)

    for fname in [
        "evaluate_critical_spo2",
        "evaluate_sbp",
        "evaluate_adrenaline",
        "evaluate_dobutamine",
        "evaluate_bpup",
        "evaluate_bpdown",
        "evaluate_bleed",
        "evaluate_transfusion",
    ]:
        monkeypatch.setattr(ms, fname, lambda *a, **k: [])

    result = ms.evaluate_all({}, None, {})

    assert result == [{
        "id": "SPO2_R",
        "instruction": "spo2",
        "pause_min": 1,
        "next_id": None,
        "comment": "",
    }, {
        "id": "CVP_R",
        "instruction": "cvp",
        "pause_min": 1,
        "next_id": None,
        "comment": "",
    }]


def test_dedup_by_id_handles_observation():
    inst = {
        "id": "OBSERVATION",
        "instruction": "経過観察",
        "pause_min": 0,
        "next_id": None,
        "comment": "",
    }
    deduped = ms.dedup_by_id([inst, inst.copy()])
    assert deduped == [inst]


def test_no_pause_for_observation():
    inst = {
        "id": "OBSERVATION",
        "instruction": "経過観察",
        "pause_min": 0,
    }
    last_instruction_time = {"OBSERVATION": 1000.0}
    now = 1001.0
    pause_min = ms.parse_pause_min(inst.get("pause_min", ms.DEFAULT_PAUSE_MIN))
    prev = last_instruction_time.get(inst["id"], 0)
    assert (now - prev) > pause_min * 60


def test_evaluate_all_bpup_only_when_sbp_high(monkeypatch):
    called = {}

    def mock_bpup(*args, **kwargs):
        called["bpup"] = True
        return [{
            "id": "BPUP_R",
            "instruction": "up",
            "pause_min": 1,
            "next_id": None,
            "comment": "",
        }]

    def mock_bpdown(*args, **kwargs):
        called["bpdown"] = True
        return [{
            "id": "BPDOWN_R",
            "instruction": "down",
            "pause_min": 1,
            "next_id": None,
            "comment": "",
        }]

    monkeypatch.setattr(ms, "evaluate_bpup", mock_bpup)
    monkeypatch.setattr(ms, "evaluate_bpdown", mock_bpdown)
    for fname in [
        "evaluate_spo2",
        "evaluate_critical_spo2",
        "evaluate_sbp",
        "evaluate_cvp",
        "evaluate_adrenaline",
        "evaluate_dobutamine",
        "evaluate_bleed",
        "evaluate_transfusion",
    ]:
        monkeypatch.setattr(ms, fname, lambda *a, **k: [])

    vitals = {"SBP": 120}
    thresholds = {"SBP_u": 90, "SBP_l": 70}
    result = ms.evaluate_all(vitals, None, thresholds)

    assert called.get("bpup") is True
    assert called.get("bpdown") is None
    assert result == [{
        "id": "BPUP_R",
        "instruction": "up",
        "pause_min": 1,
        "next_id": None,
        "comment": "",
    }]


def test_evaluate_all_bpdown_only_when_sbp_low(monkeypatch):
    called = {}

    def mock_bpup(*args, **kwargs):
        called["bpup"] = True
        return [{
            "id": "BPUP_R",
            "instruction": "up",
            "pause_min": 1,
            "next_id": None,
            "comment": "",
        }]

    def mock_bpdown(*args, **kwargs):
        called["bpdown"] = True
        return [{
            "id": "BPDOWN_R",
            "instruction": "down",
            "pause_min": 1,
            "next_id": None,
            "comment": "",
        }]

    monkeypatch.setattr(ms, "evaluate_bpup", mock_bpup)
    monkeypatch.setattr(ms, "evaluate_bpdown", mock_bpdown)
    for fname in [
        "evaluate_spo2",
        "evaluate_critical_spo2",
        "evaluate_sbp",
        "evaluate_cvp",
        "evaluate_adrenaline",
        "evaluate_dobutamine",
        "evaluate_bleed",
        "evaluate_transfusion",
    ]:
        monkeypatch.setattr(ms, fname, lambda *a, **k: [])

    vitals = {"SBP": 60}
    thresholds = {"SBP_u": 90, "SBP_l": 70}
    result = ms.evaluate_all(vitals, None, thresholds)

    assert called.get("bpup") is None
    assert called.get("bpdown") is True
    assert result == [{
        "id": "BPDOWN_R",
        "instruction": "down",
        "pause_min": 1,
        "next_id": None,
        "comment": "",
    }]


def test_evaluate_all_forwards_previous_vitals(monkeypatch):
    captured = {}

    def mock_bpup(*args, **kwargs):
        captured["previous"] = kwargs.get("previous_vitals")
        return []

    monkeypatch.setattr(ms, "evaluate_bpup", mock_bpup)
    for fname in [
        "evaluate_spo2",
        "evaluate_critical_spo2",
        "evaluate_sbp",
        "evaluate_cvp",
        "evaluate_adrenaline",
        "evaluate_dobutamine",
        "evaluate_bpdown",
        "evaluate_bleed",
        "evaluate_transfusion",
    ]:
        monkeypatch.setattr(ms, fname, lambda *a, **k: [])

    vitals = {"SBP": 120}
    thresholds = {"SBP_u": 90, "SBP_l": 70}
    previous = {"pitressin": 0.05}
    ms.evaluate_all(vitals, None, thresholds, previous_vitals=previous)

    assert captured["previous"] is previous


def test_merge_latest_adrenaline_reuses_previous_value():
    vitals_memory = {"LATEST_DRUG_VALUES": {}}

    vitals = {"Adrenaline": "0.2"}
    ms.merge_latest_adrenaline(vitals, vitals_memory)
    assert vitals["adrenaline"] == 0.2
    assert vitals_memory["LATEST_DRUG_VALUES"]["adrenaline"] == 0.2

    next_vitals = {}
    ms.merge_latest_adrenaline(next_vitals, vitals_memory)
    assert next_vitals["adrenaline"] == 0.2


def test_tree_adrenaline_reduction_rule_triggers():
    pytest.importorskip("yaml")
    thresholds = {"CVP_u": 5, "SBP_u": 90, "Ad_u": 0.1}
    tree_df = ms.load_tree("tree.yaml")

    vitals = {"CVP": 4.0, "SBP": 100.0, "adrenaline": 0.2}
    results = ms.evaluate_adrenaline(vitals, tree_df, thresholds)
    ids = {r["id"] for r in results}
    assert "AD_REDUCE_WHEN_HIGH_BP" in ids
    message = [r["instruction"] for r in results if r["id"] == "AD_REDUCE_WHEN_HIGH_BP"]
    assert message == ["アドレナリンを減量してもよいです。"]

    vitals["adrenaline"] = 0.05
    results = ms.evaluate_adrenaline(vitals, tree_df, thresholds)
    assert "AD_REDUCE_WHEN_HIGH_BP" not in {r["id"] for r in results}
