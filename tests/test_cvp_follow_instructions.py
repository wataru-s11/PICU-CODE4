import os
import sys
import types

pd_stub = types.SimpleNamespace(isna=lambda x: x != x)
sys.modules.setdefault("pandas", pd_stub)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import main_surgery as ms


def test_compute_cvp_follow_instructions_preserves_existing(monkeypatch):
    base = [
        {"id": "CVP_UPPER_CHECK", "instruction": "check"},
        {"id": "CVP_UPPER_A_SBP_UPPER", "instruction": "furo"},
    ]
    # Stub evaluate_all to return only CVP_UPPER_CHECK (no follow-up)
    monkeypatch.setattr(ms, "evaluate_all", lambda *args, **kwargs: [{"id": "CVP_UPPER_CHECK"}])
    monkeypatch.setattr(ms, "adjust_spo2_actions", lambda inst, surgery: inst)
    monkeypatch.setattr(ms, "dedup_by_id", lambda inst: inst)
    follow = ms.compute_cvp_follow_instructions(base, {}, None, {}, "dummy")
    assert any(r.get("id") == "CVP_UPPER_A_SBP_UPPER" for r in follow)
