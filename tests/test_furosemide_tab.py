import tkinter as tk
import csv
from datetime import datetime
import pytest
import drug_panel


def test_furosemide_record_uses_selected_time(tmp_path):
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    csv_path = tmp_path / "record.csv"
    panel = drug_panel.DrugPanel(root, csv_path=str(csv_path))
    # set specific time
    panel.time_var.set("1234")
    panel.furo_var.set("5")
    panel._record_furosemide()
    root.destroy()
    # read csv and verify
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    ts = datetime.strptime(rows[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
    assert ts.strftime("%H%M") == "1234"
    assert rows[0]["furosemide_mg"] == "5.0"
