import pytest

pandas = pytest.importorskip("pandas")
if not hasattr(pandas, "DataFrame"):
    pytest.skip("pandas DataFrame not available", allow_module_level=True)
pd = pandas

from main_surgery import get_latest_vitals


def test_get_latest_vitals_forward_fill(tmp_path):
    df = pd.DataFrame([
        {"timestamp": "2025-08-23 15:50:00", "SBP": 120},
        {"timestamp": "2025-08-23 15:51:00", "SBP": pd.NA},
    ])
    path = tmp_path / "vitals.csv"
    df.to_csv(path, index=False)
    latest = get_latest_vitals(path)
    assert latest["SBP"] == 120
