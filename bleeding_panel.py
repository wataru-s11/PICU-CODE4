# -*- coding: utf-8 -*-
"""Console-based bleeding management panel."""
from __future__ import annotations

import sys
from typing import Callable

import tkinter as tk
from tkinter import ttk, messagebox


class BleedingPanel(tk.Frame):
    """Simple UI for postoperative bleeding management."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ttk.Label(self, text="出血管理", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))

        form = ttk.Frame(self)
        form.pack(padx=8, pady=8)

        ttk.Label(form, text="色調").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.color_var = tk.StringVar(value="venous")
        ttk.Radiobutton(form, text="動脈血様", variable=self.color_var, value="arterial").grid(
            row=0, column=1, padx=4, pady=3
        )
        ttk.Radiobutton(form, text="静脈血様", variable=self.color_var, value="venous").grid(
            row=0, column=2, padx=4, pady=3
        )

        ttk.Label(form, text="量").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.amount_var = tk.StringVar(value="small")
        ttk.Radiobutton(form, text="大量", variable=self.amount_var, value="large").grid(
            row=1, column=1, padx=4, pady=3
        )
        ttk.Radiobutton(form, text="中等量", variable=self.amount_var, value="moderate").grid(
            row=1, column=2, padx=4, pady=3
        )
        ttk.Radiobutton(form, text="少量", variable=self.amount_var, value="small").grid(
            row=1, column=3, padx=4, pady=3
        )

        self.sticky_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="粘性あり", variable=self.sticky_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=3
        )

        ttk.Label(form, text="CVP").grid(row=2, column=2, sticky="e", padx=4, pady=3)
        self.cvp_var = tk.DoubleVar(value=0.0)
        ttk.Entry(form, textvariable=self.cvp_var, width=8).grid(row=2, column=3, padx=4, pady=3)

        ttk.Label(form, text="部位").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        self.site_var = tk.StringVar(value="none")
        ttk.Combobox(
            form,
            textvariable=self.site_var,
            values=["left", "right", "mediastinum", "none"],
            state="readonly",
            width=15,
        ).grid(row=3, column=1, columnspan=3, padx=4, pady=3)

        ttk.Button(self, text="判定", command=self._on_check).pack(pady=(0, 8))

    def _on_check(self) -> None:
        color = self.color_var.get()
        amount = self.amount_var.get()
        sticky = self.sticky_var.get()
        cvp = self.cvp_var.get()
        site = self.site_var.get()

        if color == "arterial":
            msg = "外科医をコールしてください。"
        else:
            if amount == "large":
                msg = "外科医をコールしてください。"
            elif amount == "moderate":
                if not sticky:
                    if cvp >= 8:
                        msg = "[フロセミド5mg IV]を提案します。"
                    else:
                        msg = "洗浄水の混入またはプロタミンの投与を検討してください。"
                else:
                    msg = "経過観察を行います。"
            else:
                msg = "経過観察を行います。"

            if amount in {"large", "moderate"}:
                msg += "\nRBC:FFP:PC = 2:1:1 で投与することを検討してください。"
                msg += "\n目標値は SBP が下限より上で、CVP が上限値を超えないように注意してください。"

            if site == "mediastinum":
                msg += "\n縦隔出血です。タンポナーデにならないように定期的なミルキングを行ってください。"
            elif site == "none":
                msg += "\n特になし。経過観察を行い、ヘパリン持続投与を開始するか検討してください。"
            else:
                msg += "\n部位ごとの対策を検討してください。"

        messagebox.showinfo("結果", msg)


def manage_bleeding(
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> None:
    """Entry point for postoperative bleeding management."""
    sys.stderr = open("error.log", "w")
    try:
        print_func("出血の色調を選択してください:")
        print_func("1. 動脈血様")
        print_func("2. 静脈血様")
        color_choice = input_func("番号を入力してください (1 または 2): ")

        if color_choice == "1":
            print_func("外科医をコールしてください。")
            return  # 終了して再度メニューに戻る

        print_func("出血の量を選択してください:")
        print_func("1. 大量")
        print_func("2. 中等量")
        print_func("3. 少量")
        amount_choice = input_func("番号を入力してください (1–3): ")

        if amount_choice == "1":
            print_func("外科医をコールしてください。")
            return  # 終了して再度メニューに戻る
        elif amount_choice == "2":
            print_func("出血の性状を選択してください:")
            print_func("1. 粘性がある")
            print_func("2. 粘性がない")
            consistency_choice = input_func("番号を入力してください (1 または 2): ")

            if consistency_choice == "2":
                cvp = float(input_func("CVPの値を入力してください: "))
                if cvp >= 8:
                    print_func("[フロセミド5mg IV]を提案します。")
                else:
                    print_func("洗浄水の混入またはプロタミンの投与を検討してください。")
            else:
                print_func("経過観察を行います。")
        elif amount_choice == "3":
            print_func("経過観察を行います。")

        # 量が1または2のときの共通指示
        if amount_choice in ["1", "2"]:
            print_func("RBC:FFP:PC = 2:1:1 で投与することを検討してください。")
            print_func("目標値は SBP が下限より上で、CVP が上限値を超えないように注意してください。")

        # 部位の確認
        print_func("最も出血が多い部位を選択してください:")
        print_func("1. 左胸腔")
        print_func("2. 右胸腔")
        print_func("3. 縦隔")
        print_func("4. 特になし")  # 新しい選択肢を追加
        site_choice = input_func("番号を入力してください (1–4): ")

        if site_choice == "3":
            print_func("縦隔出血です。タンポナーデにならないように定期的なミルキングを行ってください。")
        elif site_choice == "4":
            print_func("特になし。経過観察を行い、ヘパリン持続投与を開始するか検討してください。")
        else:
            print_func("部位ごとの対策を検討してください。")

    except Exception as e:  # pragma: no cover - runtime safeguard
        with open("error.log", "a") as log_file:
            log_file.write(f"エラーが発生しました: {e}\n")
        print_func("エラーが発生しました。詳細は error.log を確認してください。")
    finally:
        try:
            sys.stderr.close()
        finally:
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    print("DEBUG: bleeding.py スクリプトが開始されました。")
    manage_bleeding()
