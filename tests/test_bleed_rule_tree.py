import os
import sys
import types
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

from vitals.bleed_logic import evaluate_bleed
import main_surgery as ms


def test_tree_bleed_rule_triggers_only_with_drain():
    pytest.importorskip("yaml")
    tree_df = ms.load_tree("tree.yaml")

    vitals = {"drain_ml": 20}
    ids = [r["id"] for r in evaluate_bleed(vitals, tree_df, phase="r")]
    assert "BLEED" in ids

    # Even if the furosemide check is answered "Y", BLEED should not fire
    # without sufficient drainage volume.
    vitals = {"drain_ml": 5, "Y": True}
    ids = [r["id"] for r in evaluate_bleed(vitals, tree_df, phase="r")]
    assert "BLEED" not in ids
