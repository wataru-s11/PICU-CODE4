import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import bga_protocol


def test_evaluate_bga_triggers_messages():
    values = {
        "pH": 7.20,
        "PaCO2": 60,
        "pO2": 80,
        "BE": -5,
        "HCO3": 20,
        "K": 3.1,
        "Ca": 1.0,
        "Hct": 30,
        "Na": 140,
        "Cl": 100,
    }
    result = bga_protocol.evaluate_bga(values, "根治術")
    assert result["estimated_pCO2"] == pytest.approx(38.0)
    assert result["anion_gap"] == pytest.approx(20.0)
    msgs = "\n".join(result["messages"])
    assert "一次性呼吸性アシドーシス" in msgs
    assert "アニオンギャップが高値です" in msgs
    assert "BEが -2 未満です" in msgs
    assert "Kが 3.2 未満です" in msgs
    assert "Caが 1.1 未満です" in msgs
    assert "pO2が100未満です" in msgs
    assert "換気量が足りません" in msgs


def test_evaluate_bga_hyperventilation_message():
    values = {
        "pH": 7.40,
        "PaCO2": 30,
        "pO2": 80,
        "BE": 0,
        "HCO3": 20,
        "K": 3.4,
        "Ca": 1.2,
        "Hct": 40,
        "Na": 140,
        "Cl": 100,
    }
    result = bga_protocol.evaluate_bga(values, "根治術")
    msgs = "\n".join(result["messages"])
    assert "過換気です" in msgs
