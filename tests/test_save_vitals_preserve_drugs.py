import csv
from datetime import datetime

import drug_panel
from vital_reader import save_vitals_to_csv


def test_save_vitals_preserves_drug_columns(tmp_path):
    csv_path = tmp_path / "vitals.csv"
    # write initial drug entry
    panel = drug_panel.DrugPanel.__new__(drug_panel.DrugPanel)
    panel.csv_path = csv_path
    ts = datetime(2023, 1, 1, 0, 0)
    panel._append_to_csv(ts, {"adrenaline": 0.1})

    # append vitals without drug info
    save_vitals_to_csv({"SBP": 120}, csv_path)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert rows[-1]["adrenaline"] == "0.1"
