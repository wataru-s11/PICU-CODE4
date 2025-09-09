# -*- coding: utf-8 -*-
"""Panel to record room entry time and show elapsed duration.

The panel computes the time since the patient entered the ICU and displays
both the elapsed time and the current postoperative phase.  Phases are
currently split into the following ranges (in hours)::

    0-3, 3-24, 24-

The ``PHASES`` constant and associated handlers make it easy to tweak the
algorithm for future requirements.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional, Tuple

# Last recorded entry datetime shared across modules
_ENTRY_DT: Optional[datetime] = None


def set_entry_time(dt: datetime) -> None:
    """Store the entry datetime for global access."""
    global _ENTRY_DT
    _ENTRY_DT = dt


def get_elapsed_minutes() -> Optional[float]:
    """Return elapsed minutes since entry time, or ``None`` if unknown."""
    if _ENTRY_DT is None:
        return None
    return (datetime.now() - _ENTRY_DT).total_seconds() / 60.0

# (upper bound in hours, label)
PHASES: Tuple[Tuple[float, str], ...] = (
    (3.0, "術後3時間以内"),
    (24.0, "術後3-24時間"),
    (float("inf"), "術後24時間以降"),
)

# Optional callbacks executed when entering a given phase.
# Functions receive the elapsed ``timedelta`` and can return any message.
PHASE_HANDLERS: Dict[str, Callable[[timedelta], str]] = {}


def determine_phase(delta: timedelta) -> str:
    """Return the phase label for an elapsed ``timedelta``."""
    hours = delta.total_seconds() / 3600.0
    for limit, label in PHASES:
        if hours < limit:
            return label
    # Fallback, though loop should always return
    return PHASES[-1][1]


class EntryTimePanel(tk.Frame):
    """Panel displaying elapsed time from patient room entry."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._entry_var = tk.StringVar()
        self._elapsed_var = tk.StringVar(value="--:--")
        self._phase_var = tk.StringVar(value="---")

        row = ttk.Frame(self)
        row.pack(padx=8, pady=8, anchor="w")
        ttk.Label(row, text="入室時間 (YYYY-mm-dd HH:MM)").pack(side="left")
        ttk.Entry(row, textvariable=self._entry_var, width=20).pack(side="left", padx=(4, 0))
        ttk.Button(row, text="現在時刻", command=self._set_now).pack(side="left", padx=(4, 0))

        info = ttk.Frame(self)
        info.pack(padx=8, pady=(0, 8), anchor="w")
        ttk.Label(info, text="経過時間:").pack(side="left")
        ttk.Label(info, textvariable=self._elapsed_var, width=8).pack(side="left", padx=(4, 0))
        ttk.Label(info, textvariable=self._phase_var).pack(side="left", padx=(4, 0))

        self._update()  # start periodic updates

    def _set_now(self) -> None:
        now = datetime.now().replace(second=0, microsecond=0)
        self._entry_var.set(now.strftime("%Y-%m-%d %H:%M"))
        set_entry_time(now)

    def _update(self) -> None:
        text = self._entry_var.get().strip()
        try:
            entry = datetime.strptime(text, "%Y-%m-%d %H:%M")
        except ValueError:
            self._elapsed_var.set("--:--")
            self._phase_var.set("---")
        else:
            delta = datetime.now() - entry
            set_entry_time(entry)
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            self._elapsed_var.set(f"{hours:02d}:{minutes:02d}")
            label = determine_phase(delta)
            handler = PHASE_HANDLERS.get(label)
            self._phase_var.set(handler(delta) if handler else label)

        # schedule next update in 1 minute
        self.after(60000, self._update)
