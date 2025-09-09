# -*- coding: utf-8 -*-
"""Pulmonary hypertension risk evaluation panel."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Tuple, Optional

from vital_reader import save_vitals_to_csv


def score_pre_ph(
    days: int,
    xray_peripheral: bool,
    lung_opacity: bool,
    down_syndrome: bool,
    vein_stenosis: bool,
) -> Tuple[int, int]:
    """Return total preoperative score and whether it adds one point to current risk."""
    if days <= 3:
        score_days = 3
    elif days <= 14:
        score_days = 2
    elif days <= 28:
        score_days = 1
    else:
        score_days = 0

    score_xray = 1 if xray_peripheral else 0
    score_opacity = 1 if lung_opacity else 0
    score_down = 1 if down_syndrome else 0
    score_vein = 3 if vein_stenosis else 0

    total = score_days + score_xray + score_opacity + score_down + score_vein
    pre_risk_point = 1 if total >= 3 else 0
    return total, pre_risk_point


def score_current_ph(
    pre_risk_point: int,
    po2_decreasing: bool,
    pco2_increasing: bool,
    cvp_increasing: bool,
    urine_decreasing: bool,
) -> int:
    """Return current PH risk score including preoperative contribution."""
    total = pre_risk_point
    total += 1 if po2_decreasing else 0
    total += 1 if pco2_increasing else 0
    total += 1 if cvp_increasing else 0
    total += 1 if urine_decreasing else 0
    return total


class PHRiskPanel(tk.Frame):
    """Simple UI for pulmonary hypertension risk assessment."""

    def __init__(self, master: tk.Misc, csv_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        path = csv_path or os.getenv("VITALS_PATH")
        self.csv_path = Path(path) if path else None
        self.history: Dict[str, Dict[str, int]] = {}
        self.time_var = tk.StringVar()
        self._build_ui()
        self._select_now()

    def _build_ui(self) -> None:
        title = ttk.Label(self, text="PHリスク判定", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))

        form = ttk.Frame(self)
        form.pack(padx=8, pady=8)

        # Preoperative factors
        ttk.Label(form, text="生後日数").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.days_var = tk.IntVar(value=0)
        tk.Spinbox(form, from_=0, to=365, textvariable=self.days_var, width=10).grid(row=0, column=1, padx=4, pady=3)

        self.xray_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="胸部レントゲンで肺末梢陰影あり", variable=self.xray_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.opacity_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="肺野全体の陰影増強", variable=self.opacity_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.down_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="ダウン症", variable=self.down_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.vein_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="肺静脈狭窄/左房圧上昇", variable=self.vein_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        ttk.Separator(form, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 6))

        # Current factors
        self.po2_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="pO2低下", variable=self.po2_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.pco2_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="pCO2上昇", variable=self.pco2_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.cvp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="CVP上昇", variable=self.cvp_var).grid(row=8, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.urine_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="尿量低下", variable=self.urine_var).grid(row=9, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        timef = ttk.Frame(self)
        timef.pack(anchor="w", padx=8, pady=(0, 4))
        ttk.Label(timef, text="記録時刻：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        ttk.Entry(timef, textvariable=self.time_var, width=20).pack(side="left", padx=(6, 0))
        ttk.Button(timef, text="現在時刻", command=self._select_now).pack(side="left", padx=(6, 0))

        ttk.Button(self, text="判定", command=self._on_check).pack(pady=(0, 8))

    def _select_now(self) -> None:
        now = datetime.now().replace(second=0, microsecond=0)
        self.time_var.set(now.strftime("%Y-%m-%d %H:%M"))

    def _append_to_csv(self, ts: datetime, pre: int, current: int) -> None:
        if not self.csv_path:
            return
        row = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "PH_pre": pre,
            "PH_current": current,
        }
        try:
            save_vitals_to_csv(row, str(self.csv_path))
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] CSV書き込み失敗: {e}")

    def _on_check(self) -> None:
        try:
            days = int(self.days_var.get())
        except Exception:
            messagebox.showerror("エラー", "生後日数を正しく入力してください")
            return

        pre_total, pre_point = score_pre_ph(
            days,
            self.xray_var.get(),
            self.opacity_var.get(),
            self.down_var.get(),
            self.vein_var.get(),
        )
        current_total = score_current_ph(
            pre_point,
            self.po2_var.get(),
            self.pco2_var.get(),
            self.cvp_var.get(),
            self.urine_var.get(),
        )

        s = self.time_var.get().strip()
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror("エラー", "時刻の形式が不正です。YYYY-mm-dd HH:MM 形式で入力してください。")
            return
        key = dt.strftime("%Y-%m-%d %H:%M")
        self.history[key] = {"PH_pre": pre_total, "PH_current": current_total}
        self._append_to_csv(dt, pre_total, current_total)

        msg = f"術前PHリスクスコア: {pre_total}\n現在のPHリスクスコア: {current_total}"
        if current_total >= 2:
            msg += "\n現在PHの可能性が高いです。管理に注意してください。"
        else:
            msg += "\n現在PHのリスクは低いと考えられます。"
        messagebox.showinfo("結果", msg)


def run_ph_risk_panel(topmost: bool = False, csv_path: Optional[str] = None) -> None:
    """Run ``PHRiskPanel`` in the current thread (blocking)."""
    root = tk.Tk()
    root.title("PHRiskPanel")
    if topmost:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
    panel = PHRiskPanel(root, csv_path=csv_path)
    panel.pack(fill="both", expand=True)
    root.mainloop()


def launch_ph_risk_panel(topmost: bool = False, csv_path: Optional[str] = None):
    """Launch ``PHRiskPanel`` without blocking the caller."""
    import sys
    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process
        proc = Process(target=run_ph_risk_panel, args=(topmost, csv_path), daemon=True)
        proc.start()
        return proc
    else:
        import threading
        th = threading.Thread(target=run_ph_risk_panel, args=(topmost, csv_path), daemon=True)
        th.start()
        return th


if __name__ == "__main__":
    run_ph_risk_panel()
