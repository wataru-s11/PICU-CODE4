import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable


class ConsolePanel(tk.Frame):
    """A simple text-based interactive panel running in a thread."""

    def __init__(self, master: tk.Misc, runner: Callable[[Callable[[str], str], Callable[[str], None]], None]) -> None:
        super().__init__(master)
        self._text = tk.Text(self, state="disabled", wrap="word", height=20)
        self._text.pack(fill="both", expand=True)
        self._entry = ttk.Entry(self)
        self._entry.pack(fill="x")
        self._entry.bind("<Return>", self._on_enter)
        self._queue: "queue.Queue[str]" = queue.Queue()

        def input_func(prompt: str = "") -> str:
            self._append(prompt)
            return self._queue.get()

        def print_func(message: str = "") -> None:
            self._append(message + "\n")

        threading.Thread(
            target=lambda: runner(input_func=input_func, print_func=print_func),
            daemon=True,
        ).start()

    def _append(self, msg: str) -> None:
        self._text.configure(state="normal")
        self._text.insert("end", msg)
        self._text.see("end")
        self._text.configure(state="disabled")

    def _on_enter(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        value = self._entry.get()
        self._entry.delete(0, "end")
        self._queue.put(value)
