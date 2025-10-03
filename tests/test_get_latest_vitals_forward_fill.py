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


def test_get_latest_vitals_skips_non_persistent_columns(tmp_path):
    df = pd.DataFrame(
        [
            {
                "timestamp": "2025-08-23 15:50:00",
                "SBP": 120,
                "furosemide_mg": 3,
            },
            {
                "timestamp": "2025-08-23 15:51:00",
                "SBP": pd.NA,
                "furosemide_mg": pd.NA,
            },
        ]
    )
    path = tmp_path / "vitals.csv"
    df.to_csv(path, index=False)

    latest = get_latest_vitals(path)

    assert latest["SBP"] == 120
    assert latest["furosemide_mg"] is None


def test_get_latest_vitals_handles_utf8_bom(tmp_path):
    df = pd.DataFrame(
        [
            {"timestamp": "2025-08-23 15:50:00", "SBP": 120},
            {"timestamp": "2025-08-23 15:55:00", "SBP": 125},
        ]
    )
    path = tmp_path / "vitals.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    latest = get_latest_vitals(path)

    assert latest["timestamp"] == "2025-08-23 15:55:00"
    assert latest["SBP"] == 125


def test_get_latest_vitals_returns_python_scalars(tmp_path):
    df = pd.DataFrame(
        [
            {"timestamp": "2025-08-23 15:55:00", "SBP": 125.0, "HR": 88},
        ]
    )
    path = tmp_path / "vitals.csv"
    df.to_csv(path, index=False)

    latest = get_latest_vitals(path)

    assert type(latest["SBP"]) is float
    assert type(latest["HR"]) is int
    assert "np.float" not in repr(latest)
