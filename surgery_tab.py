# -*- coding: utf-8 -*-
"""Tkinter tab for selecting surgery type."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import MutableMapping

SURGERY_OPTIONS = [
    "根治術",
    "姑息術",
    "Glenn",
    "Fontan(フェネストレーションあり)",
]


class SurgeryTab(tk.Frame):
    """Simple tab containing radio buttons for surgery selection."""

    def __init__(self, master: tk.Misc, state: MutableMapping[str, str]):
        super().__init__(master)
        self._state = state
        current = state.get("type", SURGERY_OPTIONS[0])
        self._var = tk.StringVar(value=current)

        ttk.Label(self, text="術式を選択してください").pack(padx=8, pady=(8, 4))
        for opt in SURGERY_OPTIONS:
            ttk.Radiobutton(
                self,
                text=opt,
                value=opt,
                variable=self._var,
                command=self._update,
            ).pack(anchor="w", padx=8, pady=2)

        self._update()

    def _update(self) -> None:
        self._state["type"] = self._var.get()
