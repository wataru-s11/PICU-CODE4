import csv
from vital_reader import save_vitals_to_csv


def test_save_vitals_creates_missing_columns(tmp_path):
    csv_path = tmp_path / "vitals.csv"

    save_vitals_to_csv({"SBP": 120, "SpontaneousBreath": "1"}, csv_path)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows[-1]["SpontaneousBreath"] == "1"
    assert rows[-1]["SBP"] == "120"
