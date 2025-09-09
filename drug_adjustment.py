# -*- coding: utf-8 -*-
"""Medication adjustment algorithm for select vasoactive drugs.

This module provides a simple helper function to generate suggestions for
adjusting four common drugs based on their current infusion rates. The logic
is derived from clinician-supplied rules and is intended for educational
and decision-support purposes only. Clinical judgment by qualified medical
professionals should always take precedence.
"""
from __future__ import annotations
from typing import List


def adjust_medication(
    noradrenaline: float,
    pitresin: float,
    hamp: float,
    kontomin: float,
) -> List[str]:
    """Return a list of suggested adjustment actions.

    Parameters
    ----------
    noradrenaline: float
        Current infusion rate of noradrenaline.
    pitresin: float
        Current infusion rate of pitresin.
    hamp: float
        Current infusion rate of hamp.
    kontomin: float
        Current infusion rate of kontomin.

    Returns
    -------
    List[str]
        A list of human-readable recommendations.
    """
    actions: List[str] = []

    # ① ノルアドレナリン > 0 のとき
    if noradrenaline > 0:
        actions.append("ノルアドレナリンの減量を検討")

    # ② ピトレシン ≥ 0.03 のとき
    if pitresin >= 0.03:
        actions.append("ピトレシンの減量を検討")
    # ③ ピトレシン 0～0.02 の範囲
    elif 0 <= pitresin <= 0.02:
        # ハンプ 0～0.19 の範囲
        if 0 <= hamp <= 0.19:
            actions.append("ハンプ増量を検討")
        # ハンプ 0.2～0.3 の範囲
        elif 0.2 <= hamp <= 0.3:
            # ④ コントミンの量をチェック
            if kontomin == 0:
                actions.append("コントミンを0.1で開始を検討")
            elif 0.11 <= kontomin <= 0.3:
                actions.append("昇圧/降圧薬調整のみでの降圧は難しい可能性あり")
            else:
                actions.append("コントミン量に関する特別な指示なし")
        else:
            actions.append("ハンプの現状維持を検討")
    else:
        actions.append("ピトレシンの投与量に関する特別な指示なし")

    return actions


__all__ = ["adjust_medication"]
