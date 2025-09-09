import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from main_surgery import get_latest_vitals


def test_get_latest_vitals_excel(tmp_path):
    df = pd.DataFrame([
        {"timestamp": "2025-08-22 18:35:00", "SBP": 120, "DBP": 80}
    ])
    path = tmp_path / "vitals.xlsx"
    df.to_excel(path, index=False)
    latest = get_latest_vitals(path)
    assert latest["SBP"] == 120
    assert latest["DBP"] == 80
