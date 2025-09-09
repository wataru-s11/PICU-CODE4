# -*- coding: utf-8 -*-
"""Airway trouble checking panel."""
from __future__ import annotations

import threading
import sys
from typing import Callable

import tkinter as tk
from tkinter import ttk, messagebox


def manage_airway_trouble(
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> None:
    """Interactive DOPE-based airway trouble check."""
    original_stderr = sys.stderr
    sys.stderr = open("error.log", "w")
    try:
        print_func("DEBUG: airway_check スクリプトが開始されました。")

        print_func("現在の状態を選択してください:")
        print_func("1. 自発呼吸下")
        print_func("2. 挿管下")
        state_choice = input_func("番号を入力してください (1 または 2): ")

        if state_choice not in {"1", "2"}:
            print_func("無効な入力です。終了します。")
            return

        if state_choice == "1":
            tracheal_tag = input_func("tracheal tag が見られますか？ (はい/いいえ): ")
            if tracheal_tag.lower() == "はい":
                print_func("首の角度や肩枕、側臥位、伏臥位、ネーザルハイフローの流量増量を検討してください。")
                print_func("レントゲン写真を撮影し、肺炎・無気肺・気胸・胸水貯留を確認してください。")
                improved = input_func("呼吸状態が改善しましたか？ (はい/いいえ): ")
                if improved.lower() == "いいえ":
                    print_func("原因を取り除いても改善しない場合、再度当プロトコルを起動してください。")
                    print_func("気管挿管の可能性を検討し、上級医をコールしてください。")
                    input_func("Enterキーを押して終了...")
                    return
                else:
                    print_func("気道開通が確認されました。経過観察を続けてください。")
            else:
                print_func("tracheal tag は見られません。次に進みます。")

        print_func("\nDOPEチェックを開始します。")

        if state_choice == "2":
            displacement = input_func("チューブの深さや位置に異常はありますか？ (はい/いいえ): ")
            if displacement.lower() == "はい":
                print_func("チューブの位置を確認し、必要に応じて再挿管を検討してください。")
                print_func("原因を取り除いても改善しない場合、再度当プロトコルを起動してください。")
                input_func("Enterキーを押して終了...")
                return

        obstruction = input_func("痰や異物による閉塞の疑いがありますか？ (はい/いいえ): ")
        if obstruction.lower() == "はい":
            print_func("気管サクションを行い、再評価してください。")
            print_func("原因を取り除いても改善しない場合、再度当プロトコルを起動してください。")
            input_func("Enterキーを押して終了...")
            return

        pneumothorax = input_func("片側の呼吸音が減弱していますか？ (はい/いいえ): ")
        if pneumothorax.lower() == "はい":
            print_func("胸部エコーまたはレントゲンで気胸の確認を行ってください。")
            print_func("原因を取り除いても改善しない場合、再度当プロトコルを起動してください。")
            input_func("Enterキーを押して終了...")
            return

        if state_choice == "2":
            equipment = input_func("換気回路や酸素供給に問題はありますか？ (はい/いいえ): ")
            if equipment.lower() == "はい":
                print_func("機器の接続や設定を確認し、必要に応じて交換してください。")
                print_func("原因を取り除いても改善しない場合、再度当プロトコルを起動してください。")
                input_func("Enterキーを押して終了...")
                return

        print_func("特に問題が見つかりませんでした。")
        print_func("肺高血圧症発作やシャント閉塞、グレン不全など肺血流低下の病態がある可能性を考慮してください。")
        print_func("DEBUG: airway_check スクリプト終了")
        input_func("Enterキーを押して終了...")
    except Exception as e:  # pragma: no cover - runtime safeguard
        with open("error.log", "a") as log_file:
            log_file.write(f"エラーが発生しました: {e}\n")
        print_func("エラーが発生しました。詳細は error.log を確認してください。")
        input_func("Enterキーを押して終了...")
    finally:
        try:
            sys.stderr.close()
        finally:
            sys.stderr = original_stderr


class AirwayPanel(tk.Frame):
    """Simple panel to launch airway trouble check."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ttk.Label(self, text="気道トラブルチェック", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))
        ttk.Button(self, text="コンソールで開始", command=self._on_start).pack(pady=(0, 8))

    def _on_start(self) -> None:
        """Run the console-based airway check in a background thread."""
        threading.Thread(target=manage_airway_trouble, daemon=True).start()
        try:
            messagebox.showinfo("情報", "コンソールで気道トラブルチェックを開始しました。")
        except Exception:
            pass

