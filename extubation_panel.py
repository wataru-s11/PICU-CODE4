# -*- coding: utf-8 -*-
from __future__ import annotations

"""Extubation criteria checking panel."""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Mapping, MutableMapping

from threshold_panel import DEFAULT_THRESHOLDS


def evaluate_extubation(
    weight: float,
    peep_set: float,
    vte: float,
    spo2: float,
    thresholds: Mapping[str, float] | None = None,
) -> bool:
    """Return ``True`` if basic extubation criteria are satisfied.


    Parameters
    ----------
    weight:
        Patient body weight in kilograms.
    peep_set:
        Set PEEP value.
    vte:
        Exhaled tidal volume in millilitres.
    spo2:
        Oxygen saturation value.
    thresholds:
        Mapping containing ``SpO2_l`` and ``SpO2_u`` values. If omitted,
        :data:`threshold_panel.DEFAULT_THRESHOLDS` is used.
    """

    th = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    return (
        peep_set <= 5
        and vte >= weight * 7

    )


class ExtubationPanel(tk.Frame):
    """Simple UI to judge extubation readiness."""

    def __init__(
        self,
        master: tk.Misc,
        thresholds: MutableMapping[str, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True)

        # --- First tab: numeric criteria ---
        criteria = ttk.Frame(self._notebook)
        self._notebook.add(criteria, text="基準1")

        grid = ttk.Frame(criteria)
        grid.pack(padx=8, pady=8)

        ttk.Label(grid, text="体重 (kg)").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.weight_var = tk.DoubleVar(value=0.0)
        ttk.Entry(grid, textvariable=self.weight_var, width=10).grid(
            row=0, column=1, padx=4, pady=3
        )

        ttk.Label(grid, text="PEEPset").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.peep_var = tk.DoubleVar(value=0.0)
        ttk.Entry(grid, textvariable=self.peep_var, width=10).grid(
            row=1, column=1, padx=4, pady=3
        )

        ttk.Label(grid, text="VTe (ml)").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        self.vte_var = tk.DoubleVar(value=0.0)
        ttk.Entry(grid, textvariable=self.vte_var, width=10).grid(
            row=2, column=1, padx=4, pady=3
        )

        ttk.Label(grid, text="SpO2").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        self.spo2_var = tk.DoubleVar(value=0.0)
        ttk.Entry(grid, textvariable=self.spo2_var, width=10).grid(
            row=3, column=1, padx=4, pady=3
        )

        ttk.Button(criteria, text="判定", command=self._on_check).pack(pady=(0, 8))

        # --- Second tab: manual checks ---
        confirm = ttk.Frame(self._notebook)
        self._confirm_tab = confirm
        self._notebook.add(confirm, text="基準2", state="disabled")

        checks = ttk.Frame(confirm)
        checks.pack(padx=8, pady=8)

        self.cough_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(checks, text="患者は咳をできる", variable=self.cough_var).grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )

        self.sputum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            checks,
            text="痰は一回の吸引でほとんど引けきれる",
            variable=self.sputum_var,
        ).grid(row=1, column=0, sticky="w", padx=4, pady=3)

        self.cvp_spo2_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            checks,
            text=(
                f"バッグ換気を戻してもCVP< {self._thresholds['CVP_u']}"
                f"かつSpO2> {self._thresholds['SpO2_l']}"
            ),
            variable=self.cvp_spo2_var,
        ).grid(row=2, column=0, sticky="w", padx=4, pady=3)

        ttk.Button(confirm, text="判定", command=self._on_finalize).pack(pady=(0, 8))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_check(self) -> None:  # pragma: no cover - UI callback
        try:
            weight = float(self.weight_var.get())
            peep = float(self.peep_var.get())
            vte = float(self.vte_var.get())
            spo2 = float(self.spo2_var.get())
        except Exception:
            messagebox.showerror("エラー", "数値を正しく入力してください")
            return

        if evaluate_extubation(weight, peep, vte, spo2, self._thresholds):
            messagebox.showinfo("結果", "抜管に進める可能性があります")
            self._notebook.tab(self._confirm_tab, state="normal")
            self._notebook.select(self._confirm_tab)
        else:
            messagebox.showinfo("結果", "抜管の基準を満たしていません。")

    def _on_finalize(self) -> None:  # pragma: no cover - UI callback
        if self.cough_var.get() and self.sputum_var.get() and self.cvp_spo2_var.get():
            messagebox.showinfo("結果", "抜管可能、準備を進めてください。")
        else:
            messagebox.showinfo("結果", "抜管の基準を満たしていません。")


def run_extubation_panel(topmost: bool = False) -> None:  # pragma: no cover - UI helper
    """Run ``ExtubationPanel`` in the current thread (blocking)."""
    root = tk.Tk()
    root.title("ExtubationPanel")
    if topmost:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
    panel = ExtubationPanel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()


def launch_extubation_panel(topmost: bool = False):  # pragma: no cover - UI helper
    """Launch ``ExtubationPanel`` without blocking the caller."""
    import sys

    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process

        proc = Process(target=run_extubation_panel, args=(topmost,), daemon=True)
        proc.start()
        return proc
    else:
        import threading

        th = threading.Thread(target=run_extubation_panel, args=(topmost,), daemon=True)
        th.start()
        return th


if __name__ == "__main__":  # pragma: no cover - script entry
    run_extubation_panel()

