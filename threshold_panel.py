# -*- coding: utf-8 -*-
"""Simple panel for editing evaluation thresholds.

Includes an update button so values changed programmatically can be
reflected in the UI.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Callable, MutableMapping

# Default threshold values used by ``main_surgery.prompt_thresholds``.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "SpO2_l": 80.0,
    "SpO2_u": 100.0,
    "Critical_SpO2_l": 75.0,
    "Critical_SpO2_u": 100.0,
    "SBP_l": 70.0,
    "SBP_u": 90.0,
    "CVP_u": 5.0,
    "CVP_c": 8.0,
}


class ThresholdPanel(tk.Frame):
    """Editable thresholds displayed in a tab.

    Parameters
    ----------
    master:
        Parent widget.
    thresholds:
        Initial threshold values. If omitted, ``DEFAULT_THRESHOLDS`` is used.
    on_change:
        Optional callback executed when the user presses the "apply" button.
        Receives the updated thresholds dictionary.
    """

    def __init__(
        self,
        master: tk.Misc,
        thresholds: Optional[MutableMapping[str, float]] = None,
        on_change: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_change = on_change
        self._vars: Dict[str, tk.DoubleVar] = {}
        # Keep reference to external thresholds so they can be reloaded later.
        self._thresholds_ref = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
        values = self._thresholds_ref.copy()

        grid = ttk.Frame(self)
        grid.pack(padx=8, pady=8)

        for i, (key, val) in enumerate(values.items()):
            ttk.Label(grid, text=key).grid(row=i, column=0, sticky="w", padx=4, pady=3)
            var = tk.DoubleVar(value=val)
            entry = ttk.Entry(grid, textvariable=var, width=10)
            entry.grid(row=i, column=1, padx=4, pady=3)
            entry.bind("<Return>", self._apply)
            self._vars[key] = var

        # Buttons for applying changes and reloading external values.
        btns = ttk.Frame(self)
        btns.pack(pady=(0, 8))
        ttk.Button(btns, text="更新", command=self._reload).pack(side="left", padx=4)
        ttk.Button(btns, text="適用", command=self._apply).pack(side="left", padx=4)

    def get_thresholds(self) -> Dict[str, float]:
        """Return current threshold values."""
        return {k: v.get() for k, v in self._vars.items()}

    def _apply(self, event=None) -> None:  # pragma: no cover - UI callback
        if self._on_change:
            self._on_change(self.get_thresholds())

    def _reload(self) -> None:  # pragma: no cover - UI callback
        """Reload widget values from the underlying thresholds mapping."""
        for key, var in self._vars.items():
            if key in self._thresholds_ref:
                var.set(self._thresholds_ref[key])
