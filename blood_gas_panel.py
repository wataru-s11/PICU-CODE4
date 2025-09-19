# -*- coding: utf-8 -*-
# ファイル名: blood_gas_panel.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

import bga_protocol
from vital_reader import save_vitals_to_csv

try:  # pragma: no cover - optional dependency
    from openpyxl import Workbook
except Exception:  # pragma: no cover - gracefully degrade
    Workbook = None

COLUMNS = [
    ("ph", "pH"),
    ("pco2", "pCO2"),
    ("po2", "pO2"),
    ("hct", "Hct"),
    ("k", "K"),
    ("na", "Na"),
    ("cl", "Cl"),
    ("ca", "Ca"),
    ("glu", "Glu"),
    ("lac", "Lac"),
    ("tbil", "tBil"),
    ("hco3", "HCO3-"),
    ("abe", "ABE"),
    ("alb", "Alb"),
]


class BloodGasPanel(tk.Frame):
    """血液ガス分析値の入力パネル"""

    def __init__(self, master: tk.Misc, csv_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.vars: Dict[str, tk.DoubleVar] = {}
        self.weight_var = tk.DoubleVar(value=0.0)
        self.rbc_rate_var = tk.DoubleVar(value=0.0)
        # 採血記録: key="YYYY-mm-dd HH:MM"
        self.history: Dict[str, Dict[str, float]] = {}
        self.csv_path = Path(csv_path) if csv_path else None
        self._build_ui()
        self._select_now()

    def _build_ui(self) -> None:
        title = ttk.Label(self, text="血液ガス分析入力", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))

        grid = ttk.Frame(self)
        grid.pack(padx=8, pady=8)

        for i, (key, label) in enumerate(COLUMNS):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=3)
            var = tk.DoubleVar(value=0.0)
            self.vars[key] = var
            ttk.Entry(grid, textvariable=var, width=10).grid(row=i, column=1, padx=4, pady=3)

        weight_row = len(COLUMNS)
        ttk.Label(grid, text="体重 (kg)").grid(row=weight_row, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(grid, textvariable=self.weight_var, width=10).grid(row=weight_row, column=1, padx=4, pady=3)

        rbc_row = weight_row + 1
        ttk.Label(grid, text="RBC輸血速度 (ml/h)").grid(row=rbc_row, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(grid, textvariable=self.rbc_rate_var, width=10).grid(row=rbc_row, column=1, padx=4, pady=3)

        # 疾患選択
        disf = ttk.Frame(self)
        disf.pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(disf, text="疾患：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.disease_var = tk.StringVar(value=bga_protocol.BGA_DISEASES[0])
        ttk.Combobox(
            disf,
            textvariable=self.disease_var,
            values=bga_protocol.BGA_DISEASES,
            state="readonly",
            width=25,
        ).pack(side="left", padx=(6, 0))

        # 採血時刻入力
        timef = ttk.Frame(self)
        timef.pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(timef, text="採血時刻：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.time_var = tk.StringVar()
        ttk.Entry(timef, textvariable=self.time_var, width=20).pack(side="left", padx=(6, 0))
        ttk.Button(timef, text="現在時刻", command=self._select_now).pack(side="left", padx=(6, 0))

        # 記録ボタン
        btnf = ttk.Frame(self)
        btnf.pack(anchor="w", padx=8, pady=(0, 8))
        ttk.Button(btnf, text="この時間を記録", command=self._commit_current_time).pack(side="left")
        ttk.Button(btnf, text="評価", command=self.evaluate_current_data).pack(side="left", padx=(6, 0))

    def get_values(self) -> Dict[str, float]:
        """現在の入力値を辞書で取得"""
        return {k: v.get() for k, v in self.vars.items()}

    # ===== 時刻操作 =====
    def _select_now(self) -> None:
        now = datetime.now().replace(second=0, microsecond=0)
        self.time_var.set(now.strftime("%Y-%m-%d %H:%M"))

    # ===== データ記録 =====
    def _commit_current_time(self) -> None:
        s = self.time_var.get().strip()
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror("エラー", "時刻の形式が不正です。YYYY-mm-dd HH:MM 形式で入力してください。")
            return
        key = dt.strftime("%Y-%m-%d %H:%M")
        current = self.get_values()
        current["weight_kg"] = self.weight_var.get()
        current["rbc_ml_per_h"] = self.rbc_rate_var.get()
        self.history[key] = current
        self._export_excel_auto()
        if self.csv_path:
            csv_vals = {
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "pH": current.get("ph"),
                "PaCO2": current.get("pco2"),
                "pO2": current.get("po2"),
                "Hct": current.get("hct"),
                "K": current.get("k"),
                "Na": current.get("na"),
                "Cl": current.get("cl"),
                "Ca": current.get("ca"),
                "Glu": current.get("glu"),
                "Lac": current.get("lac"),
                "tBil": current.get("tbil"),
                "HCO3": current.get("hco3"),
                "BE": current.get("abe"),
                "Alb": current.get("alb"),
                "weight_kg": current.get("weight_kg"),
                "rbc_ml_per_h": current.get("rbc_ml_per_h"),
            }
            save_vitals_to_csv(csv_vals, str(self.csv_path))

    def _export_excel_auto(self) -> None:
        if Workbook is None:
            return
        wb = Workbook()
        ws = wb.active
        ws.append(["timestamp"] + [k for k, _ in COLUMNS])
        for ts in sorted(self.history.keys()):
            rec = self.history[ts]
            ws.append([ts] + [rec.get(k, 0.0) for k, _ in COLUMNS])
        path = Path("blood_gas_panel.xlsx")
        wb.save(path)

    def _get_previous_hct_record(self, current_dt: Optional[datetime] = None) -> Optional[Dict[str, float]]:
        """Return the most recent history record prior to ``current_dt``."""

        if not self.history:
            return None

        candidates = []
        for ts_str, record in self.history.items():
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if current_dt is None or ts < current_dt:
                candidates.append((ts, record))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    def evaluate_current_data(self) -> Optional[Dict[str, object]]:
        """現在の入力値を ``bga_protocol`` で評価する."""
        vals = self.get_values()
        values = {
            "pH": vals.get("ph", 0.0),
            "PaCO2": vals.get("pco2", 0.0),
            "pO2": vals.get("po2", 0.0),
            "BE": vals.get("abe", 0.0),
            "HCO3": vals.get("hco3", 0.0),
            "K": vals.get("k", 0.0),
            "Ca": vals.get("ca", 0.0),
            "Hct": vals.get("hct", 0.0),
            "Na": vals.get("na", 0.0),
            "Cl": vals.get("cl", 0.0),
        }
        albumin = vals.get("alb", 0.0) or None
        try:
            result = bga_protocol.evaluate_bga(values, self.disease_var.get(), albumin)
        except Exception as e:  # pragma: no cover - defensive
            messagebox.showerror("エラー", f"BGA評価に失敗しました: {e}")
            return None
        messages = result.get("messages")
        if not isinstance(messages, list):
            messages = list(messages or [])
            result["messages"] = messages

        current_time = self.time_var.get().strip()
        current_dt: Optional[datetime]
        try:
            current_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M") if current_time else None
        except ValueError:
            current_dt = None

        previous_record = self._get_previous_hct_record(current_dt)
        if previous_record is not None:
            prev_hct = previous_record.get("hct")
            current_hct = vals.get("hct", 0.0)
            if prev_hct is not None and current_hct > prev_hct:
                weight_kg = self.weight_var.get()
                if weight_kg <= 0:
                    weight_kg = float(previous_record.get("weight_kg") or 0.0)
                rbc_rate = self.rbc_rate_var.get()
                if rbc_rate <= 0:
                    rbc_rate = float(previous_record.get("rbc_ml_per_h") or 0.0)
                if weight_kg > 0:
                    if rbc_rate >= weight_kg:
                        messages.append("十分な輸血がされています")
                    else:
                        messages.append("血管漏出が増えている可能性…ピトレシン0.02–0.05増量")

        messagebox.showinfo("BGA診断結果", "\n".join(messages))
        return result


def run_blood_gas_panel(topmost: bool = False, csv_path: Optional[str] = None) -> None:
    """Run ``BloodGasPanel`` in the current thread (blocking)."""
    root = tk.Tk()
    root.title("BloodGasPanel")
    if topmost:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
    panel = BloodGasPanel(root, csv_path=csv_path)
    panel.pack(fill="both", expand=True)
    root.mainloop()


def launch_blood_gas_panel(topmost: bool = False, csv_path: Optional[str] = None):
    """Launch ``BloodGasPanel`` without blocking the caller."""
    import sys

    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process

        proc = Process(target=run_blood_gas_panel, args=(topmost, csv_path), daemon=True)
        proc.start()
        return proc
    else:
        import threading

        th = threading.Thread(target=run_blood_gas_panel, args=(topmost, csv_path), daemon=True)
        th.start()
        return th


if __name__ == "__main__":
    run_blood_gas_panel()
