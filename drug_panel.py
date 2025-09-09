# -*- coding: utf-8 -*-
# ファイル名: drug_panel.py
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import csv
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class DrugConfig:
    key: str                 # 内部キー（英字）
    label: str               # 画面表示名（日本語）
    unit: str                # 単位表示
    min_val: float           # スライダー最小
    max_val: float           # スライダー最大
    step: float = 0.005      # スピンボックス刻み
    default: float = 0.0     # 初期値

# 既定薬剤（必要なら値域は自由に調整してください）
DEFAULT_DRUGS: List[DrugConfig] = [
    DrugConfig("adrenaline",     "アドレナリン",     "μg/kg/min", 0.0, 0.3, 0.005, 0.0),
    DrugConfig("noradrenaline",  "ノルアドレナリン", "μg/kg/min", 0.0, 0.3, 0.005, 0.0),
    DrugConfig("dobutamine",     "ドブタミン",       "μg/kg/min", 0.0, 10.0, 0.1,   0.0),
    DrugConfig("pitressin",      "ピトレシン",       "U/kg/h",    0.0, 0.2, 0.005, 0.0),
    DrugConfig("milrinone",      "ミルリノン",       "μg/kg/min", 0.0, 0.5, 0.005, 0.0),
    DrugConfig("hanp",           "ハンプ",           "μg/kg/min", 0.0, 0.3, 0.005, 0.0),
    DrugConfig("contomin",       "コントミン",       "mg/kg/h",   0.0, 0.3, 0.005, 0.0),
    DrugConfig("perdipine",      "ペルジピン",       "μg/kg/min",  0.0, 1.0, 0.005, 0.0),
    DrugConfig("furosemide_civ", "フロセミドCIV",   "mg/kg/h",   0.0, 1.0, 0.005, 0.0),
]

class DrugPanel(tk.Frame):
    """
    薬剤入力用の右サイドパネル。
    - 上部に現在値サマリーを常時表示
    - タブ（Notebook）で各薬剤の入力UI
    - get_values() で dict 取得
    - on_change で変更イベントをフック可能
    - [この時刻を記録] ボタンで明示的に履歴追加
    - [履歴CSV書き出し] ボタンで手動エクスポート
    """
    def __init__(
        self,
        master: tk.Misc,
        drugs: List[DrugConfig] = DEFAULT_DRUGS,
        on_change: Optional[Callable[[str, float], None]] = None,
        topmost: bool = False,
        csv_path: Optional[str] = None,
        auto_log_interval: Optional[int] = 60,
        **kwargs
    ) -> None:
        super().__init__(master, **kwargs)
        self.drugs = drugs
        self.on_change = on_change
        self.vars: Dict[str, tk.DoubleVar] = {}
        # 各薬剤の Spinbox ウィジェットを保持し、未確定入力を直接取得できるようにする
        self.spinboxes: Dict[str, tk.Spinbox] = {}
        self.csv_path = Path(csv_path) if csv_path else None
        # 投与量の変更履歴（時刻ごと）
        self.history: List[tuple[datetime, Dict[str, float]]] = []
        # フロセミド投与記録
        self.furosemide_log: List[tuple[datetime, float]] = []
        self.auto_log_interval = auto_log_interval

        self._build_ui()
        self._select_now()
        if topmost:
            # 画面を常時前面にしたい場合（親ごと前面になります）
            try:
                self.winfo_toplevel().attributes("-topmost", True)
            except Exception:
                pass  # 環境により未対応な場合もある

        if self.auto_log_interval and self.auto_log_interval > 0:
            try:
                self.after(int(self.auto_log_interval * 1000), self._auto_log_tick)
            except Exception:
                pass

    # ---------- UI 構築 ----------
    def _build_ui(self) -> None:
        # タイトル
        title = ttk.Label(self, text="薬剤入力パネル", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))

        # サマリー（常時表示）
        summary_frame = ttk.Frame(self)
        summary_frame.pack(fill="x", padx=8)
        ttk.Label(summary_frame, text="現在の投与：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.summary_var = tk.StringVar(value="-")
        self.summary_label = ttk.Label(summary_frame, textvariable=self.summary_var, font=("Meiryo UI", 10))
        self.summary_label.pack(side="left")

        # 更新時刻選択
        time_frame = ttk.Frame(self)
        time_frame.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(time_frame, text="更新時刻（例:0930）：", font=("Meiryo UI", 10, "bold")).pack(side="left")
        self.time_var = tk.StringVar()
        ttk.Entry(time_frame, textvariable=self.time_var, width=8).pack(side="left", padx=(6, 0))
        # 現在時刻ボタンを押しても即ログには記録しない（水分パネルと同様の動作）
        ttk.Button(time_frame, text="現在時刻", command=self._select_now).pack(side="left", padx=(6, 0))
        ttk.Button(time_frame, text="←1m", command=lambda: self._shift_min(-1)).pack(side="left", padx=(6, 0))
        ttk.Button(time_frame, text="1m→", command=lambda: self._shift_min(1)).pack(side="left", padx=(3, 0))

        # 操作行（保存ボタン等）
        ops = ttk.Frame(self)
        ops.pack(fill="x", padx=8, pady=(6, 6))
        ttk.Button(ops, text="すべて 0 に", command=self._all_zero).pack(side="left")
        ttk.Button(ops, text="この時間を登録/更新", command=self._record_current).pack(side="left", padx=(6, 0))
        ttk.Button(ops, text="履歴CSV書き出し…", command=self._export_csv_dialog).pack(side="left", padx=(6, 0))

        # 区切り
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=(4, 6))

        # タブ
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 各薬剤タブは遅延生成することで初期表示を軽量化する
        self._tab_frames: Dict[str, tk.Widget] = {}
        self._tab_initialized: Dict[str, bool] = {}
        self.config_map: Dict[str, DrugConfig] = {cfg.key: cfg for cfg in self.drugs}
        for cfg in self.drugs:
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=cfg.label)
            self._tab_frames[cfg.key] = frame
            self._tab_initialized[cfg.key] = False
            # 変数だけ先に用意しておく
            self.vars[cfg.key] = tk.DoubleVar(value=cfg.default)

        # フロセミドIV入力タブも必要になるまで遅延生成
        self._furo_frame = ttk.Frame(self.nb)
        self.nb.add(self._furo_frame, text="フロセミドIV")
        self._furo_initialized = False

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # 最初のタブのみ即時生成
        if self.drugs:
            self._build_drug_tab(self.drugs[0].key)

        self._refresh_summary()

    # ----- 時刻操作 -----
    def _select_now(self) -> None:
        """現在時刻をフォームに設定"""
        now = datetime.now().replace(second=0, microsecond=0)
        self.time_var.set(now.strftime("%H%M"))
        self._refresh_summary()

    def _shift_min(self, delta: int) -> None:
        """選択時刻を delta 分シフト"""
        dt = self._get_log_time()
        dt += timedelta(minutes=delta)
        self.time_var.set(dt.strftime("%H%M"))

    def _get_log_time(self) -> datetime:
        s = self.time_var.get().strip()
        # datetime.now() は比較的高コストなので、1度だけ呼び出して使い回す
        now = datetime.now().replace(second=0, microsecond=0)
        if not s:
            return now
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M")
        except ValueError:
            pass
        s2 = s.replace(":", "")
        if len(s2) == 4 and s2.isdigit():
            h = int(s2[:2])
            m = int(s2[2:])
            return now.replace(hour=h, minute=m)
        return now

    def _build_drug_tab(self, key: str) -> None:
        """指定された薬剤タブのウィジェットを生成"""
        if self._tab_initialized.get(key):
            return
        frame = self._tab_frames[key]
        cfg = self.config_map[key]
        var = self.vars[key]

        # ラベル
        head = ttk.Label(frame, text=f"{cfg.label}（{cfg.unit}）", font=("Meiryo UI", 11, "bold"))
        head.pack(anchor="w", padx=10, pady=(10, 6))

        # スライダー
        scale_frame = ttk.Frame(frame)
        scale_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(scale_frame, text=f"{cfg.min_val:.3f}").pack(side="left")
        scale = ttk.Scale(
            scale_frame,
            from_=cfg.min_val,
            to=cfg.max_val,
            orient="horizontal",
            command=lambda _=None, k=cfg.key: self._on_scale_changed(k),
            variable=var,
        )
        scale.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(scale_frame, text=f"{cfg.max_val:.3f}").pack(side="left")

        # 数値入力（Spinbox）
        entry_frame = ttk.Frame(frame)
        entry_frame.pack(fill="x", padx=10, pady=(2, 10))
        ttk.Label(entry_frame, text="投与量:").pack(side="left")
        sb = tk.Spinbox(
            entry_frame,
            from_=cfg.min_val,
            to=cfg.max_val,
            increment=cfg.step,
            textvariable=var,
            width=10,
            justify="right",
        )
        sb.pack(side="left", padx=(6, 6))
        ttk.Label(entry_frame, text=cfg.unit).pack(side="left")
        self.spinboxes[cfg.key] = sb

        # クイックボタン
        quick = ttk.Frame(frame)
        quick.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(quick, text="0", width=4, command=lambda k=cfg.key: self._set_value(k, 0.0)).pack(side="left")
        ttk.Button(quick, text="既定", width=4, command=lambda k=cfg.key, v=cfg.default: self._set_value(k, v)).pack(side="left", padx=(6, 0))

        # 変更監視
        var.trace_add("write", lambda *_args, k=cfg.key: self._on_var_changed(k))

        self._tab_initialized[key] = True

    def _build_furosemide_tab(self) -> None:
        """フロセミドIV専用タブを生成"""
        if self._furo_initialized:
            return
        frame = self._furo_frame

        ttk.Label(frame, text="フロセミドIV（mg）", font=("Meiryo UI", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 6))

        # 投与量候補（mg）
        opts = ["1", "2", "3", "5", "10"]
        self.furo_var = tk.StringVar()
        self.furo_cb = ttk.Combobox(
            frame,
            values=opts,
            textvariable=self.furo_var,
            state="readonly",
            width=10,
        )
        self.furo_cb.pack(anchor="w", padx=10)

        ttk.Button(frame, text="記録", command=self._record_furosemide).pack(anchor="w", padx=10, pady=(6, 10))
        self._furo_initialized = True

    def _on_tab_changed(self, _event: tk.Event) -> None:
        """タブ切り替え時に必要なコンテンツを遅延生成"""
        current = self.nb.select()
        frame = self.nb.nametowidget(current)
        for key, frm in self._tab_frames.items():
            if frm is frame:
                self._build_drug_tab(key)
                return
        if frame is self._furo_frame:
            self._build_furosemide_tab()

    def _record_furosemide(self) -> None:
        # StringVar が更新されていない環境があるため Combobox から直接取得する
        s = self.furo_cb.get().strip() if hasattr(self, "furo_cb") else self.furo_var.get().strip()
        try:
            dose = float(s)
        except ValueError:
            messagebox.showerror("エラー", "投与量を選択してください。")
            return
        ts = self._get_log_time()
        self.furosemide_log.append((ts, dose))
        self._append_to_csv(ts, {"furosemide_mg": dose})
        messagebox.showinfo("記録", f"{dose} mg を記録しました。")
        self.furo_var.set("")

    # ---------- イベント/内部処理 ----------
    def _on_scale_changed(self, key: str) -> None:
        # ttk.Scale は resolution 未対応なので Spinbox と変数同期のみ
        #（var の trace でサマリー更新）
        pass

    def _append_to_csv(self, ts: datetime, vals: Dict[str, float]) -> None:
        """指定時刻と値を vitals_history CSV へ追記する。

        既存 CSV に新しい列が必要な場合はヘッダーを更新して全体を書き戻す。
        それ以外は追記モードで 1 行だけ追加する。
        """
        if not self.csv_path:
            return
        row = {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")}
        row.update(vals)
        try:
            if self.csv_path.exists():
                with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
                missing = [k for k in row.keys() if k not in fieldnames]
                if missing:
                    fieldnames.extend(missing)
                    rows.append(row)
                    with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(row)
            else:
                fieldnames = list(row.keys())
                with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
        except Exception as e:
            print(f"[WARN] CSV書き込み失敗: {e}")

    def _log_values(self) -> None:
        """現在時刻の値を履歴に追加"""
        self._log_values_at(self._get_log_time())

    def _log_values_at(self, ts: datetime) -> None:
        """指定時刻の値を履歴に追加"""
        vals = self.get_values()
        self.history.append((ts, vals))
        self._append_to_csv(ts, vals)

    def _auto_log_tick(self) -> None:
        """定期的に現在値を記録する内部タイマー"""
        try:
            self._log_values_at(datetime.now().replace(second=0, microsecond=0))
        except Exception:
            pass
        if self.auto_log_interval and self.auto_log_interval > 0:
            try:
                self.after(int(self.auto_log_interval * 1000), self._auto_log_tick)
            except Exception:
                pass

    def _on_var_changed(self, key: str) -> None:
        self._refresh_summary()
        if self.on_change:
            try:
                self.on_change(key, float(self.vars[key].get()))
            except Exception:
                pass

    def _record_current(self) -> None:
        """ボタン操作でこの時間の値を登録/更新"""
        self._log_values()
        messagebox.showinfo("記録", "この時間の値を登録/更新しました。")

    def _set_value(self, key: str, value: float) -> None:
        v = self.vars.get(key)
        if v is not None:
            v.set(value)

    def _all_zero(self) -> None:
        for k in self.vars:
            self.vars[k].set(0.0)

    def _refresh_summary(self) -> None:
        vals = self.get_values(sync_vars=False)
        parts = []
        for cfg in self.drugs:
            val = vals.get(cfg.key, 0.0)
            # 表示は 3桁精度（整数/小数を見やすく）
            if abs(val - round(val)) < 1e-6:
                s = f"{int(round(val))}"
            else:
                s = f"{val:.3f}".rstrip("0").rstrip(".")
            parts.append(f"{cfg.label}:{s} {cfg.unit}")
        self.summary_var.set(" | ".join(parts))

    # ---------- 公開API ----------
    def get_values(self, sync_vars: bool = True) -> Dict[str, float]:
        """現在の薬剤値（key: value）を返す"""
        values: Dict[str, float] = {}
        spinboxes = getattr(self, "spinboxes", {})
        for k, var in self.vars.items():
            val: Optional[float] = None
            if k in spinboxes:
                s = spinboxes[k].get().strip()
                try:
                    val = float(s)
                    if sync_vars and abs(var.get() - val) > 1e-9:
                        var.set(val)
                except ValueError:
                    pass
            if val is None:
                try:
                    val = float(var.get())
                except Exception:
                    # 入力途中などで数値に変換できない場合は 0 とみなす
                    val = 0.0
                    if sync_vars:
                        try:
                            var.set(val)
                        except Exception:
                            pass
            values[k] = val
        return values

    def set_values(self, values: Dict[str, float]) -> None:
        """外部から値を一括設定"""
        for k, val in values.items():
            if k in self.vars:
                self.vars[k].set(float(val))

    # ---------- CSV エクスポート ----------
    def _export_csv_dialog(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="CSVに記録",
        )
        if not path:
            return
        try:
            self._export_csv(path)
            furo_path = str(Path(path).with_name(Path(path).stem + "_furosemide.csv"))
            self._export_furosemide_csv(furo_path)
            messagebox.showinfo("完了", f"CSVを保存しました。\nフロセミド記録: {furo_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"CSV書き込みに失敗しました:\n{e}")

    def _export_csv(self, path: str) -> None:
        """履歴を欠損なくCSV出力"""
        if not self.history:
            return
        fieldnames = ["timestamp"] + [cfg.key for cfg in self.drugs]
        hist = sorted(self.history, key=lambda h: h[0])
        start = hist[0][0].replace(second=0, microsecond=0)
        end = hist[-1][0].replace(second=0, microsecond=0)
        current = {k: 0.0 for k in self.vars.keys()}
        rows: List[Dict[str, float]] = []
        idx = 0
        t = start
        while t <= end:
            while idx < len(hist) and hist[idx][0] <= t:
                current.update(hist[idx][1])
                idx += 1
            row = {"timestamp": t.strftime("%Y-%m-%d %H:%M:%S")}
            row.update(current)
            rows.append(row)
            t += timedelta(minutes=1)

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _export_furosemide_csv(self, path: str) -> None:
        fieldnames = ["timestamp", "dose_mg"]
        rows = [
            {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "dose_mg": dose}
            for ts, dose in self.furosemide_log
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

# ---- 外部利用向けユーティリティ ----

def run_drug_panel(
    topmost: bool = False,
    csv_path: Optional[str] = None,
    auto_log_interval: Optional[int] = 60,
) -> None:
    """Run ``DrugPanel`` in the current thread (blocking)."""
    root = tk.Tk()
    root.title("DrugPanel")
    root.geometry("560x460")
    panel = DrugPanel(
        root,
        drugs=DEFAULT_DRUGS,
        topmost=topmost,
        csv_path=csv_path,
        auto_log_interval=auto_log_interval,
    )
    panel.pack(side="right", fill="both", expand=True)
    root.mainloop()


def launch_drug_panel(
    topmost: bool = False,
    csv_path: Optional[str] = None,
    auto_log_interval: Optional[int] = 60,
):
    """Launch ``DrugPanel`` without blocking the caller.

    On platforms such as Windows and macOS where Tkinter requires the main
    thread, the panel is started in a separate process.  For other platforms a
    background thread is used (previous behaviour).
    """
    import sys
    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process
        proc = Process(
            target=run_drug_panel,
            args=(topmost, csv_path, auto_log_interval),
            daemon=True,
        )
        proc.start()
        return proc
    else:
        import threading
        th = threading.Thread(
            target=run_drug_panel,
            args=(topmost, csv_path, auto_log_interval),
            daemon=True,
        )
        th.start()
        return th

# 単体テスト起動用（直接実行したときだけ）
if __name__ == "__main__":
    run_drug_panel()
