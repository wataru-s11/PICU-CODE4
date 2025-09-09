# -*- coding: utf-8 -*-
# ファイル名: fluid_panel.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import csv
import pandas as pd
try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover - gracefully degrade if not available
    Workbook = None

COLUMNS = [
    ("urine_ml",  "尿量(ml)"),
    ("drain_ml",  "ドレーン(ml)"),
    ("gr_ml",     "胃残量(ml)"),
    ("stool_ml",  "便(ml)"),
    ("inf_ml",    "輸液量(ml)"),
    ("rbc_ml",    "RBC(ml)"),
    ("ffp_ml",    "FFP(ml)"),
    ("pc_ml",     "PC(ml)"),
]

def _hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

class FluidPanel(tk.Frame):
    """
    1時間ごとの水分管理パネル
    - 時刻は「時単位」で管理（YYYY-mm-dd HH:00）
    - 尿量/ドレーン量/胃残量/便/輸液量/RBC/FFP/PC を入力して追加・更新
    - 上部にサマリー（現在時刻の値＋直近24h合計・バランス）を常時表示
    - CSV 読み書き対応
    """
    def __init__(self, master: tk.Misc, topmost: bool = False, csv_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.hourly: Dict[str, Dict[str, float]] = {}  # key: "YYYY-mm-dd HH:00"
        self.csv_path = Path(csv_path) if csv_path else None
        self._build_ui()
        if topmost:
            try:
                self.winfo_toplevel().attributes("-topmost", True)
            except Exception:
                pass
        self._select_now()

    # ===== UI =====
    def _build_ui(self) -> None:
        title = ttk.Label(self, text="水分管理パネル（1時間ごと）", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8,4))

        # サマリー
        sumf = ttk.Frame(self)
        sumf.pack(fill="x", padx=8)
        ttk.Label(sumf, text="サマリー：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.summary_var = tk.StringVar(value="-")
        ttk.Label(sumf, textvariable=self.summary_var, font=("Meiryo UI", 10)).pack(side="left")

        # 操作バー
        ops = ttk.Frame(self)
        ops.pack(fill="x", padx=8, pady=(6,4))
        ttk.Button(ops, text="CSV読込…", command=self._import_csv).pack(side="left")
        ttk.Button(ops, text="CSV保存…", command=self._export_csv).pack(side="left", padx=(6,0))
        ttk.Button(ops, text="直近24hをゼロで初期化", command=self._init_last_24h_zero).pack(side="left", padx=(12,0))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=(6,6))

        # 入力ブロック
        inf = ttk.Frame(self)
        inf.pack(fill="x", padx=8)

        # 時刻選択
        whenf = ttk.Frame(inf); whenf.pack(fill="x")
        ttk.Label(whenf, text="対象時刻：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.hour_var = tk.StringVar()
        self.hour_cb = ttk.Combobox(whenf, textvariable=self.hour_var, width=20, state="normal")
        self.hour_cb.pack(side="left", padx=(6,0))
        ttk.Button(whenf, text="現在時刻", command=self._select_now).pack(side="left", padx=(6,0))
        ttk.Button(whenf, text="←1h", command=lambda: self._shift_hour(-1)).pack(side="left", padx=(6,0))
        ttk.Button(whenf, text="1h→", command=lambda: self._shift_hour(+1)).pack(side="left", padx=(3,0))

        # 数値入力
        grid = ttk.Frame(inf); grid.pack(fill="x", pady=(6,6))
        self.vars: Dict[str, tk.DoubleVar] = {}
        for i, (key, label) in enumerate(COLUMNS):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=3)
            v = tk.DoubleVar(value=0.0)
            self.vars[key] = v
            tk.Spinbox(grid, from_=0, to=100000, increment=10, width=10, textvariable=v, justify="right").grid(
                row=i, column=1, padx=4, pady=3, sticky="w"
            )

        # ボタン
        btnf = ttk.Frame(inf); btnf.pack(fill="x")
        ttk.Button(btnf, text="この時間を登録/更新", command=self._commit_current_hour).pack(side="left")
        ttk.Button(btnf, text="この時間を0でクリア", command=self._clear_current_hour).pack(side="left", padx=(6,0))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=(6,6))

        # 一覧
        listf = ttk.Frame(self); listf.pack(fill="both", expand=True, padx=8, pady=(0,8))
        columns = ["hour"] + [k for k, _ in COLUMNS] + ["balance"]
        self.tree = ttk.Treeview(listf, columns=columns, show="headings", height=12)
        self.tree.pack(side="left", fill="both", expand=True)

        # ヘッダ
        self.tree.heading("hour", text="時刻（時単位）")
        self.tree.column("hour", width=150, anchor="center")
        for key, label in COLUMNS:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=100, anchor="e")
        self.tree.heading("balance", text="IN-OUTバランス(ml)")
        self.tree.column("balance", width=100, anchor="e")

        sb = ttk.Scrollbar(listf, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

    # ===== 時刻操作 =====
    def _generate_hour_options(self, center: datetime, span_hours: int = 36) -> None:
        start = _hour_floor(center - timedelta(hours=span_hours//2))
        options = [(start + timedelta(hours=i)).strftime("%Y-%m-%d %H:00") for i in range(span_hours + 1)]
        self.hour_cb["values"] = options

    def _select_now(self) -> None:
        nowh = _hour_floor(datetime.now())
        self._generate_hour_options(nowh, span_hours=48)
        s = nowh.strftime("%Y-%m-%d %H:00")
        self.hour_var.set(s)
        if s not in self.hourly:
            self.hourly[s] = {k: 0.0 for k, _ in COLUMNS}
        self._load_hour_to_form(s)
        self._refresh_tree()
        self._refresh_summary()

    def _shift_hour(self, delta: int) -> None:
        if not self.hour_var.get():
            return
        dt = datetime.strptime(self.hour_var.get(), "%Y-%m-%d %H:00")
        dt += timedelta(hours=delta)
        s = dt.strftime("%Y-%m-%d %H:00")
        if s not in self.hour_cb["values"]:
            self._generate_hour_options(dt, span_hours=48)
        self.hour_var.set(s)
        self._load_hour_to_form(s)
        self._refresh_summary()

    # ===== フォーム <-> データ =====
    def _load_hour_to_form(self, hour_key: str) -> None:
        rec = self.hourly.get(hour_key, {})
        for k, _ in COLUMNS:
            val = float(rec.get(k, 0.0))
            self.vars[k].set(val)

    def _form_to_record(self) -> Dict[str, float]:
        return {k: float(v.get()) for k, v in self.vars.items()}

    def _append_to_csv(self, hour_key: str, rec: Dict[str, float]) -> None:
        if not self.csv_path:
            return
        row = {"timestamp": f"{hour_key}:00"}
        row.update(rec)
        try:
            if self.csv_path.exists():
                df = pd.read_csv(self.csv_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False)
            else:
                df = pd.DataFrame([row])
            df.to_csv(self.csv_path, index=False)
        except Exception:
            pass

    def _commit_current_hour(self) -> None:
        hour_key_raw = self.hour_var.get()
        if not hour_key_raw:
            messagebox.showerror("エラー", "対象時刻が選択されていません。")
            return
        try:
            dt = datetime.strptime(hour_key_raw, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror("エラー", "時刻の形式が不正です。YYYY-mm-dd HH:MM 形式で入力してください。")
            return
        hour_key = _hour_floor(dt).strftime("%Y-%m-%d %H:00")
        if hour_key not in self.hour_cb["values"]:
            self._generate_hour_options(dt, span_hours=48)
        self.hour_var.set(hour_key)
        rec = self._form_to_record()
        self.hourly[hour_key] = rec
        self._append_to_csv(hour_key, rec)
        self._refresh_tree()
        self._refresh_summary()
        self._export_excel_auto()

    def _clear_current_hour(self) -> None:
        for v in self.vars.values():
            v.set(0.0)
        self._commit_current_hour()

    # ===== 表示系 =====
    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        # 時系列順に表示
        for hour_key in sorted(self.hourly.keys()):
            rec = self.hourly[hour_key]
            intake = rec.get("inf_ml",0)+rec.get("rbc_ml",0)+rec.get("ffp_ml",0)+rec.get("pc_ml",0)
            output = rec.get("urine_ml",0)+rec.get("drain_ml",0)+rec.get("gr_ml",0)+rec.get("stool_ml",0)
            balance = intake - output
            row = [hour_key] + [self._fmt(rec.get(k, 0.0)) for k, _ in COLUMNS] + [self._fmt(balance)]
            self.tree.insert("", "end", iid=hour_key, values=row)

    def _on_select_row(self, _evt=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        hour_key = sel[0]
        self.hour_var.set(hour_key)
        self._load_hour_to_form(hour_key)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        # 現在選択時刻の値
        hour_key = self.hour_var.get()
        cur = self.hourly.get(hour_key, {k:0.0 for k,_ in COLUMNS})
        cur_intake = cur.get("inf_ml",0)+cur.get("rbc_ml",0)+cur.get("ffp_ml",0)+cur.get("pc_ml",0)
        cur_output = cur.get("urine_ml",0)+cur.get("drain_ml",0)+cur.get("gr_ml",0)+cur.get("stool_ml",0)
        cur_balance = cur_intake - cur_output
        totals = self.get_totals(hours=24, ref_hour=hour_key)
        intake = totals["inf_ml"] + totals["rbc_ml"] + totals["ffp_ml"] + totals["pc_ml"]
        output = totals["urine_ml"] + totals["drain_ml"] + totals.get("gr_ml",0) + totals.get("stool_ml",0)
        balance = intake - output
        cur_txt = " / ".join([f"{label}:{self._fmt(cur.get(k,0))}" for k, label in COLUMNS])
        cur_txt += f" / IN-OUTバランス:{self._fmt(cur_balance)} / 24h水分量:{self._fmt(24*cur_intake)}"
        total_txt = f"直近24h 合計  収入:{self._fmt(intake)}  排出:{self._fmt(output)}  IN-OUTバランス:{self._fmt(balance)}"
        self.summary_var.set(f"[{hour_key}] {cur_txt} | {total_txt}")

    @staticmethod
    def _fmt(x: float) -> str:
        return f"{x:.0f}"

    # ===== CSV I/O =====
    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            title="CSVに保存"
        )
        if not path:
            return
        try:
            self._write_csv(path)
            messagebox.showinfo("完了", "CSVを書き出しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"CSV書き出しに失敗しました:\n{e}")

    def _import_csv(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV","*.csv")],
            title="CSVを読み込み"
        )
        if not path:
            return
        try:
            self._read_csv(path)
            self._refresh_tree()
            self._refresh_summary()
            messagebox.showinfo("完了", "CSVを読み込みました。")
        except Exception as e:
            messagebox.showerror("エラー", f"CSV読み込みに失敗しました:\n{e}")

    def _write_csv(self, path: str) -> None:
        fieldnames = ["hour"] + [k for k,_ in COLUMNS]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for hour_key in sorted(self.hourly.keys()):
                row = {"hour": hour_key}
                row.update({k: self.hourly[hour_key].get(k, 0.0) for k,_ in COLUMNS})
                w.writerow(row)

    def _read_csv(self, path: str) -> None:
        self.hourly.clear()
        with open(path, "r", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            for row in r:
                hour_key = row.get("hour")
                if not hour_key:
                    continue
                rec = {}
                for k, _ in COLUMNS:
                    try:
                        rec[k] = float(row.get(k, "0") or 0)
                    except ValueError:
                        rec[k] = 0.0
                self.hourly[hour_key] = rec

    def _export_excel_auto(self) -> None:
        if Workbook is None:
            return
        path = "fluid_panel.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.append(["hour"] + [k for k,_ in COLUMNS] + ["balance"])
        for hour_key in sorted(self.hourly.keys()):
            rec = self.hourly[hour_key]
            intake = rec.get("inf_ml",0)+rec.get("rbc_ml",0)+rec.get("ffp_ml",0)+rec.get("pc_ml",0)
            output = rec.get("urine_ml",0)+rec.get("drain_ml",0)+rec.get("gr_ml",0)+rec.get("stool_ml",0)
            balance = intake - output
            ws.append([hour_key] + [rec.get(k,0.0) for k,_ in COLUMNS] + [balance])
        wb.save(path)

    # ===== 公開API =====
    def get_hourly_data(self) -> Dict[str, Dict[str, float]]:
        """全時間のデータ（deep copy不要ならそのまま返す）"""
        return self.hourly

    def get_totals(self, hours: int = 24, ref_hour: Optional[str] = None) -> Dict[str, float]:
        """ref_hour を基準に過去 hours 時間の合計を返す"""
        if ref_hour is None:
            ref_dt = _hour_floor(datetime.now())
        else:
            ref_dt = datetime.strptime(ref_hour, "%Y-%m-%d %H:00")
        start_dt = ref_dt - timedelta(hours=hours-1)  # refを含め hours 本
        totals = {k: 0.0 for k,_ in COLUMNS}
        for hour_key, rec in self.hourly.items():
            dt = datetime.strptime(hour_key, "%Y-%m-%d %H:00")
            if start_dt <= dt <= ref_dt:
                for k,_ in COLUMNS:
                    totals[k] += float(rec.get(k, 0.0))
        return totals

    def _init_last_24h_zero(self) -> None:
        ref_dt = _hour_floor(datetime.now())
        for i in range(24):
            dt = ref_dt - timedelta(hours=i)
            key = dt.strftime("%Y-%m-%d %H:00")
            if key not in self.hourly:
                self.hourly[key] = {k:0.0 for k,_ in COLUMNS}
        self._refresh_tree()
        self._refresh_summary()

def run_fluid_panel(topmost: bool = False, csv_path: Optional[str] = None) -> None:
    """Run ``FluidPanel`` in the current thread (blocking)."""
    root = tk.Tk()
    root.title("FluidPanel")
    root.geometry("860x560")
    panel = FluidPanel(root, topmost=topmost, csv_path=csv_path)
    panel.pack(fill="both", expand=True)
    root.mainloop()

def launch_fluid_panel(topmost: bool = False, csv_path: Optional[str] = None):
    """Launch ``FluidPanel`` without blocking the caller.

    On Windows and macOS the panel runs in a separate process to satisfy
    Tkinter's requirement of using the main thread.  Other platforms keep the
    previous background-thread behaviour.
    """
    import sys
    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process
        proc = Process(target=run_fluid_panel, args=(topmost, csv_path), daemon=True)
        proc.start()
        return proc
    else:
        import threading
        th = threading.Thread(target=run_fluid_panel, args=(topmost, csv_path), daemon=True)
        th.start()
        return th

# 単体デモ
if __name__ == "__main__":
    run_fluid_panel()
