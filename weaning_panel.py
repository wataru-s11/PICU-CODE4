# -*- coding: utf-8 -*-
"""Console-based ventilator weaning guidance panel."""
from __future__ import annotations

import sys
from typing import Callable

import tkinter as tk
from tkinter import ttk, messagebox


class WeaningPanel(tk.Frame):
    """Simple UI for ventilator weaning guidance."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        title = ttk.Label(self, text="ウィーニング", font=("Meiryo UI", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(8, 4))

        form = ttk.Frame(self)
        form.pack(padx=8, pady=8)

        ttk.Label(form, text="TOF").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.tof_var = tk.IntVar(value=0)
        tk.Spinbox(form, from_=0, to=4, textvariable=self.tof_var, width=5).grid(row=0, column=1, padx=4, pady=3)

        self.pco2_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="pCO2 45–55", variable=self.pco2_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=3
        )

        self.spont_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="自発呼吸あり", variable=self.spont_var).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=3
        )

        # --- Individual requirements ---
        self.pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text="ポーズ≧60秒/RR",
            variable=self.pause_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.tv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text="TV≧7ml/kg・PS10で検討",
            variable=self.tv_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.cvp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text="CVP上昇なし",
            variable=self.cvp_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.irregular_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text="自発呼吸規則的",
            variable=self.irregular_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.agitation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            form,
            text="アジテーションなし/鎮静指示使用",
            variable=self.agitation_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.weak_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="抜管後呼吸弱い", variable=self.weak_var).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=4, pady=3
        )

        ttk.Button(self, text="判定", command=self._on_check).pack(pady=(0, 8))

    def _on_check(self) -> None:
        tof = self.tof_var.get()
        pco2_ok = self.pco2_var.get()
        spont = self.spont_var.get()
        weak = self.weak_var.get()
        # Early guidance: TOF ≥2, pCO2 in range, and pause requirement met
        if tof >= 2 and pco2_ok and self.pause_var.get():
            messagebox.showinfo(
                "結果", "TV≧7ml/kgを維持してPSを10まで段階的に下げてください。"
            )
            return

        conditions = all(
            [
                self.pause_var.get(),
                self.tv_var.get(),
                self.cvp_var.get(),
                self.irregular_var.get(),
                self.agitation_var.get(),
            ]
        )

        if tof < 2:
            msg = "TOFが 2 回未満です。筋弛緩薬の効果が残っています。待機してください。"
        else:
            msg = "TOFが 2 回以上です。次の段階に進みます。\nフロートリガーを 0.4 に設定しました。"
            if not pco2_ok:
                msg += "\npCO2 が基準範囲外です。呼吸回数を調整してください。"
            if spont:
                msg += "\nウィーニングプロセスを開始します。\n以下の要件を満たしてください:\n① ポーズ時間を60秒/RR以上にしてください\n② TV ≧ 7ml/kg となれば PS を落とし、PS が 10 となったら抜管を検討します。\n③ CVP 上昇があったときは PS を落とさないでください。\n④ 自発呼吸が不規則な場合はフェンタニルの増量または「気道トラブルチェック」を起動してください。\n⑤ 覚醒下に抜管する必要はありません。アジテーション時には鎮静約束指示を使用してください。"
                if conditions:
                    msg += "\nすべての要件を満たしました。抜管に進みます。\n抜管が行われました。"
                    if weak:
                        msg += "\n筋弛緩薬拮抗 (スガマデクス 2mg/kg IV) を検討してください。\n気道トラブルチェックに進んでください。"
                    else:
                        msg += "\n呼吸は安定しています。経過観察を続けてください。"
                else:
                    msg += "\n要件を満たしていません。再確認してください。"
            else:
                msg += "\n自発呼吸が見られません。経過観察を続けてください。"

        messagebox.showinfo("結果", msg)


if __name__ == "__main__":
    print("DEBUG: weaning.py スクリプトが開始されました。")

def manage_weaning(
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> None:
    """Interactive ventilator weaning management flow.

    Parameters
    ----------
    input_func:
        Function used to obtain user input. Defaults to :func:`input`.
    print_func:
        Function used to output messages. Defaults to :func:`print`.
    """
    original_stderr = sys.stderr
    sys.stderr = open("error.log", "w")

    try:
        print_func("TOFウォッチで筋弛緩薬の効果を確認します。")
        while True:
            tof_input = input_func("TOFの回数を入力してください (0–4): ")
            if not tof_input.isdigit():
                print_func("⚠️ 入力エラー: 数字で入力してください。")
                continue
            tof = int(tof_input)
            if tof >= 2:
                print_func("TOFが 2 回以上です。次の段階に進みます。")
                print_func("フロートリガーを 0.4 に設定しました。")
                pco2_check = input_func("pCO2が 45–55 の範囲内ですか？ (はい/いいえ): ")
                if pco2_check.lower() == "いいえ":
                    print_func("pCO2 が基準範囲外です。呼吸回数を調整してください。")
                spontaneous_breath = input_func("自発呼吸が見られますか？ (はい/いいえ): ")
                if spontaneous_breath.lower() == "はい":
                    print_func("ウィーニングプロセスを開始します。")
                    print_func("以下の要件を満たしてください:")
                    print_func("① ポーズ時間を60秒/RR以上にしてください")
                    print_func("② TV ≧ 7ml/kg となれば PS を落とし、PS が 10 となったら抜管を検討します。")
                    print_func("③ CVP 上昇があったときは PS を落とさないでください。")
                    print_func("④ 自発呼吸が不規則な場合はフェンタニルの増量または「気道トラブルチェック」を起動してください。")
                    print_func("⑤ 覚醒下に抜管する必要はありません。アジテーション時には鎮静約束指示を使用してください。")
                    all_conditions_met = input_func("すべての要件を満たしましたか？ (はい/いいえ): ")
                    if all_conditions_met.lower() == "はい":
                        print_func("すべての要件を満たしました。抜管に進みます。")
                        print_func("抜管が行われました。")
                        weak_breath = input_func("呼吸が弱いですか？ (はい/いいえ): ")
                        if weak_breath.lower() == "はい":
                            print_func("筋弛緩薬拮抗 (スガマデクス 2mg/kg IV) を検討してください。")
                            print_func("気道トラブルチェックに進んでください。")
                        else:
                            print_func("呼吸は安定しています。経過観察を続けてください。")
                    else:
                        print_func("要件を満たしていません。再確認してください。")
                else:
                    print_func("自発呼吸が見られません。経過観察を続けてください。")
            else:
                print_func("TOFが 2 回未満です。筋弛緩薬の効果が残っています。待機してください。")
            cont = input_func("🔄 ウィーニングを継続しますか？ (はい/いいえ): ")
            if cont.lower() == "いいえ":
                print_func("DEBUG: weaning.py スクリプト終了")
                input_func("Enterキーを押してメニューに戻る...")
                return
    except Exception as e:  # pragma: no cover - runtime safeguard
        with open("error.log", "a") as log_file:
            log_file.write(f"エラーが発生しました: {e}\n")
        print_func("⚠️ エラーが発生しました。詳細は error.log を確認してください。")
        input_func("Enterキーを押して終了...")
    finally:
        try:
            sys.stderr.close()
        finally:  # ensure stderr restored even if close fails
            sys.stderr = original_stderr


if __name__ == "__main__":
    manage_weaning()
