# -*- coding: utf-8 -*-
"""Combined tabbed UI for clinical panels."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, MutableMapping


class ConsoleLauncher(tk.Frame):
    """Simple frame with a button to launch console-based panels."""

    def __init__(
        self, master: tk.Misc, label: str, callback: Callable[[], None]
    ) -> None:
        super().__init__(master)
        ttk.Label(self, text=label).pack(padx=8, pady=(8, 4))
        ttk.Button(
            self,
            text="開く",
            command=lambda: threading.Thread(target=callback, daemon=True).start(),
        ).pack(pady=8)


def _error_tab(parent: tk.Misc, message: str) -> tk.Frame:
    """Return a placeholder tab with an error message."""
    frame = tk.Frame(parent)
    ttk.Label(frame, text=message, foreground="red").pack(padx=8, pady=8)
    return frame


def run_drug_fluid_tabs(
    topmost: bool = False,
    drug_csv_path: Optional[str] = None,
    fluid_csv_path: Optional[str] = None,
    auto_log_interval: Optional[int] = 60,
    thresholds: Optional[MutableMapping[str, float]] = None,
    surgery_state: Optional[MutableMapping[str, str]] = None,
) -> None:
    """Run multiple panels within a single tabbed window (blocking).

    Each panel is imported lazily so that the absence of optional
    dependencies (e.g. ``pandas`` for the fluid panel) does not prevent
    the rest of the interface from appearing.
    """

    root = tk.Tk()
    root.title("Drug & Fluid Panels")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    if surgery_state is not None:
        try:
            from surgery_tab import SurgeryTab

            notebook.add(SurgeryTab(notebook, surgery_state), text="術式")
        except Exception as e:  # pragma: no cover - defensive
            notebook.add(
                _error_tab(notebook, f"術式タブを読み込めませんでした: {e}"),
                text="術式",
            )

    # --- Drug panel ---
    try:
        from drug_panel import DrugPanel

        drug = DrugPanel(
            notebook,
            topmost=topmost,
            csv_path=drug_csv_path,
            auto_log_interval=auto_log_interval,
        )
        notebook.add(drug, text="薬剤")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"薬剤パネルを読み込めませんでした: {e}"),
            text="薬剤",
        )

    # --- Fluid panel ---
    try:
        from fluid_panel import FluidPanel

        fluid = FluidPanel(
            notebook,
            topmost=topmost,
            csv_path=fluid_csv_path,
        )
        notebook.add(fluid, text="水分")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"水分パネルを読み込めませんでした: {e}"),
            text="水分",
        )

    # --- Gas panel ---
    try:
        from gas_panel import GasPanel

        notebook.add(GasPanel(notebook), text="ガス")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"ガスパネルを読み込めませんでした: {e}"),
            text="ガス",
        )

    # --- Blood gas panel ---
    try:
        from blood_gas_panel import BloodGasPanel

        notebook.add(BloodGasPanel(notebook, csv_path=drug_csv_path), text="BGA")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"BGAパネルを読み込めませんでした: {e}"),
            text="BGA",
        )

    # --- Pulmonary hypertension risk ---
    try:
        from ph_risk_panel import PHRiskPanel

        notebook.add(PHRiskPanel(notebook), text="PHリスク")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"PHリスクパネルを読み込めませんでした: {e}"),
            text="PHリスク",
        )

    # --- Extubation panel ---
    try:
        from extubation_panel import ExtubationPanel

        notebook.add(ExtubationPanel(notebook), text="抜管基準")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"抜管パネルを読み込めませんでした: {e}"),
            text="抜管基準",
        )

    # --- Threshold editor ---
    try:
        from threshold_panel import ThresholdPanel

        if thresholds is not None:
            notebook.add(
                ThresholdPanel(
                    notebook,
                    thresholds=thresholds,
                    on_change=lambda vals: thresholds.update(vals),
                ),
                text="閾値",
            )
        else:
            notebook.add(ThresholdPanel(notebook), text="閾値")
    except Exception:  # pragma: no cover - defensive
        # 無視して閾値タブを追加しない（GUI未使用環境向け）
        pass

    # --- Additional panels ---
    try:
        from airway_obstruction_panel import AirwayPanel
        notebook.add(AirwayPanel(notebook), text="気道")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"気道評価を読み込めませんでした: {e}"),
            text="気道",
        )

    try:
        from bleeding_panel import BleedingPanel
        notebook.add(BleedingPanel(notebook), text="出血")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"出血評価を読み込めませんでした: {e}"),
            text="出血",
        )

    try:
        from weaning_panel import WeaningPanel
        notebook.add(WeaningPanel(notebook), text="ウィーニング")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"ウィーニングパネルを読み込めませんでした: {e}"),
            text="ウィーニング",
        )

    try:
        from entry_time_panel import EntryTimePanel
        notebook.add(EntryTimePanel(notebook), text="入室時間")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"入室時間パネルを読み込めませんでした: {e}"),
            text="入室時間",
        )

    if topmost:
        try:  # pragma: no cover - system dependent
            root.attributes("-topmost", True)
        except Exception:
            pass

    root.mainloop()


def launch_drug_fluid_tabs(
    topmost: bool = False,
    drug_csv_path: Optional[str] = None,
    fluid_csv_path: Optional[str] = None,
    auto_log_interval: Optional[int] = 60,
    thresholds: Optional[MutableMapping[str, float]] = None,
    surgery_state: Optional[MutableMapping[str, str]] = None,
):
    """Launch tabbed panels without blocking the caller."""
    import sys

    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process

        proc = Process(
            target=run_drug_fluid_tabs,
            args=(topmost, drug_csv_path, fluid_csv_path, auto_log_interval, thresholds, surgery_state),
            daemon=True,
        )
        proc.start()
        return proc
    else:
        import threading

        th = threading.Thread(
            target=run_drug_fluid_tabs,
            args=(topmost, drug_csv_path, fluid_csv_path, auto_log_interval, thresholds, surgery_state),
            daemon=True,
        )
        th.start()
        return th


def run_assessment_tabs(
    topmost: bool = False,
    thresholds: Optional[MutableMapping[str, float]] = None,
) -> None:
    """Run extubation and console-based assessments in tabs."""
    root = tk.Tk()
    root.title("Clinical Assessments")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # --- Extubation panel ---
    try:
        from extubation_panel import ExtubationPanel

        notebook.add(ExtubationPanel(notebook), text="抜管基準")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"抜管パネルを読み込めませんでした: {e}"),
            text="抜管基準",
        )

    # --- Airway obstruction ---
    try:
        from airway_obstruction_panel import AirwayPanel
        notebook.add(AirwayPanel(notebook), text="気道")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"気道評価を読み込めませんでした: {e}"),
            text="気道",
        )

    # --- Bleeding management ---
    try:
        from bleeding_panel import BleedingPanel
        notebook.add(BleedingPanel(notebook), text="出血")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"出血評価を読み込めませんでした: {e}"),
            text="出血",
        )

    # --- Weaning management ---
    try:
        from weaning_panel import WeaningPanel
        notebook.add(WeaningPanel(notebook), text="ウィーニング")
    except Exception as e:  # pragma: no cover - defensive
        notebook.add(
            _error_tab(notebook, f"ウィーニングパネルを読み込めませんでした: {e}"),
            text="ウィーニング",
        )

    # --- Threshold editor ---
    try:
        from threshold_panel import ThresholdPanel

        if thresholds is not None:
            notebook.add(
                ThresholdPanel(
                    notebook,
                    thresholds=thresholds,
                    on_change=lambda vals: thresholds.update(vals),
                ),
                text="閾値",
            )
        else:
            notebook.add(ThresholdPanel(notebook), text="閾値")
    except Exception:  # pragma: no cover - defensive
        # タブ追加失敗時は黙ってスキップ
        pass

    if topmost:
        try:  # pragma: no cover - system dependent
            root.attributes("-topmost", True)
        except Exception:
            pass

    root.mainloop()


def launch_assessment_tabs(topmost: bool = False, thresholds: Optional[MutableMapping[str, float]] = None):
    """Launch assessment tabs without blocking the caller."""
    import sys

    if sys.platform.startswith("win") or sys.platform == "darwin":
        from multiprocessing import Process

        proc = Process(target=run_assessment_tabs, args=(topmost, thresholds), daemon=True)
        proc.start()
        return proc
    else:
        import threading

        th = threading.Thread(target=run_assessment_tabs, args=(topmost, thresholds), daemon=True)
        th.start()
        return th


if __name__ == "__main__":
    run_drug_fluid_tabs()
