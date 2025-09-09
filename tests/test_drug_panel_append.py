import csv
from datetime import datetime
import drug_panel


def test_append_to_csv_appends_rows(tmp_path):
    panel = drug_panel.DrugPanel.__new__(drug_panel.DrugPanel)
    panel.csv_path = tmp_path / "vitals.csv"

    ts1 = datetime(2023, 1, 1, 1, 0, 0)
    panel._append_to_csv(ts1, {"adrenaline": 0.1})
    ts2 = datetime(2023, 1, 1, 1, 5, 0)
    panel._append_to_csv(ts2, {"noradrenaline": 0.2})

    with open(panel.csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["adrenaline"] == "0.1"
    assert rows[0].get("noradrenaline", "") == ""
    assert rows[1]["noradrenaline"] == "0.2"
