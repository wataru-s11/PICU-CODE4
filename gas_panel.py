# -*- coding: utf-8 -*-
from __future__ import annotations

"""Simple panel to input FiO2, NO, and nitrogen levels."""

from datetime import datetime
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from typing import Dict, Optional

from vital_reader import save_vitals_to_csv


class GasPanel(tk.Frame):
    """Panel with entry boxes for FiO2, NO, and nitrogen."""

    def __init__(self, master: tk.Misc, csv_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.vars: Dict[str, tk.DoubleVar] = {}
        self.history: Dict[str, Dict[str, float]] = {}
        # Use ``VITALS_PATH`` if explicit ``csv_path`` is not provided
        path = csv_path or os.getenv("VITALS_PATH")
        self.csv_path = Path(path) if path else None

        grid = ttk.Frame(self)
        grid.pack(padx=8, pady=8)

        fields = [
            ("FiO2", "FiO2 (%)"),
            ("NO", "NO (ppm)"),
            ("N2", "窒素 (%)"),
        ]
        for i, (key, label) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=3)
            var = tk.DoubleVar(value=0.0)
            ttk.Entry(grid, textvariable=var, width=10).grid(row=i, column=1, padx=4, pady=3)
            self.vars[key] = var

        timef = ttk.Frame(self)
        timef.pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(timef, text="記録時刻：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.time_var = tk.StringVar()
        ttk.Entry(timef, textvariable=self.time_var, width=20).pack(side="left", padx=(6, 0))
        ttk.Button(timef, text="現在時刻", command=self._select_now).pack(side="left", padx=(6, 0))

        btnf = ttk.Frame(self)
        btnf.pack(anchor="w", padx=8, pady=(0, 8))
        ttk.Button(btnf, text="この時間を記録", command=self._commit_current_time).pack(side="left")

        self._select_now()

    def get_values(self) -> Dict[str, float]:
        """Return current gas values."""
        return {k: v.get() for k, v in self.vars.items()}

    def _select_now(self) -> None:
        now = datetime.now().replace(second=0, microsecond=0)
        self.time_var.set(now.strftime("%Y-%m-%d %H:%M"))

    def _commit_current_time(self) -> None:
        s = self.time_var.get().strip()
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror("エラー", "時刻の形式が不正です。YYYY-mm-dd HH:MM 形式で入力してください。")
            return
        key = dt.strftime("%Y-%m-%d %H:%M")
        vals = self.get_values()
        self.history[key] = vals
        self._append_to_csv(dt, vals)

    def _append_to_csv(self, ts: datetime, vals: Dict[str, float]) -> None:
        if not self.csv_path:
            return
        row = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")}
        row.update(vals)
        try:
            save_vitals_to_csv(row, str(self.csv_path))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] CSV書き込み失敗: {e}")


def run_gas_panel(topmost: bool = False, csv_path: Optional[str] = None) -> None:  # pragma: no cover - UI helper
    """Run ``GasPanel`` in a blocking window."""
    root = tk.Tk()
    root.title("GasPanel")
    if topmost:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
    panel = GasPanel(root, csv_path=csv_path)
    panel.pack(fill="both", expand=True)
    root.mainloop()
