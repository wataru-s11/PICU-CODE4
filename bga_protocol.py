# -*- coding: utf-8 -*-
"""簡易血液ガス解析プロトコル.

与えられた血液ガス分析値から主要な異常を検出し、
必要な介入をメッセージとして返す。
"""
from __future__ import annotations

from typing import Dict, List, Optional

# 基準値の設定
BGA_CRITERIA = {
    "Common": {
        "pH": (7.35, 7.45),
        "PaCO2": (35, 45),
        "HCO3": 20,
        "BE": -2,
        "AnionGap": (8, 12),
        "K": (3.3, 3.6),
        "Ca": 1.1,
    },
}

# 対応疾患一覧
BGA_DISEASES = [
    "根治術",
    "単心室姑息術",
    "両心室姑息術",
    "グレン手術",
    "フォンタン手術（フェネストレーションなし）",
    "フォンタン手術（フェネストレーションあり）",
]

def calculate_anion_gap(na: float, cl: float, hco3: float, albumin: Optional[float] = None) -> float:
    """アニオンギャップを計算する.

    補正アニオンギャップにも対応するため、アルブミン値を任意で受け取る。
    """
    ag = na - (cl + hco3)
    if albumin is not None:
        ag += (4 - albumin) * 2.5
    return ag

def evaluate_bga(values: Dict[str, float], disease: str, albumin: Optional[float] = None) -> Dict[str, object]:
    """血液ガス分析値を評価する.

    Parameters
    ----------
    values: dict
        ``pH``, ``PaCO2``, ``pO2``, ``BE``, ``HCO3``, ``K``, ``Ca``, ``Hct``, ``Na``, ``Cl`` を含む辞書。
    disease: str
        ``BGA_DISEASES`` に含まれる疾患名。
    albumin: float, optional
        アルブミン値。補正アニオンギャップ計算に使用。

    Returns
    -------
    dict
        ``estimated_pCO2``、``anion_gap``、``messages`` を含む辞書。
    """
    common = BGA_CRITERIA["Common"]

    pH = values.get("pH", 0.0)
    PaCO2 = values.get("PaCO2", 0.0)
    pO2 = values.get("pO2", 0.0)
    BE = values.get("BE", 0.0)
    HCO3 = values.get("HCO3", 0.0)
    K = values.get("K", 0.0)
    Ca = values.get("Ca", 0.0)
    Hct = values.get("Hct", 0.0)
    Na = values.get("Na", 0.0)
    Cl = values.get("Cl", 0.0)

    messages: List[str] = []

    estimated_pco2 = 1.5 * HCO3 + 8
    if PaCO2 > estimated_pco2:
        messages.append("一次性呼吸性アシドーシス (CO2 貯留によるアシドーシス) の可能性があります。")
    else:
        messages.append("pCO2 と HCO3 のバランスは正常範囲です。")

    if PaCO2 < common["PaCO2"][0]:
        messages.append(
            "pCO2が35未満です。過換気です。呼吸数または一回換気量を減量してください。"
        )
    elif PaCO2 > common["PaCO2"][1]:
        messages.append(
            "pCO2が45より高値です。換気量が足りません。呼吸数または一回換気量を増量してください。"
        )
    else:
        messages.append("pCO2は基準範囲内です。")

    anion_gap = calculate_anion_gap(Na, Cl, HCO3, albumin)
    if anion_gap > common["AnionGap"][1]:
        messages.append("アニオンギャップが高値です。代謝性アシドーシスの可能性があります。")
    elif anion_gap < common["AnionGap"][0]:
        messages.append("アニオンギャップが低値です。異常な電解質バランスの可能性があります。")
    else:
        messages.append("アニオンギャップは正常範囲です。")

    if BE < common["BE"] and anion_gap > common["AnionGap"][1]:
        messages.append("BEが -2 未満です。メイロン 0.5ml/kg を投与してください。")

    if K < 3.2:
        messages.append("Kが 3.2 未満です。カリウムの大量補正を行ってください。")
    elif 3.3 <= K <= 3.6:
        messages.append("Kが 3.3–3.6 です。カリウムの少量補正を行ってください。")
    else:
        messages.append("Kは基準値内です。")

    if Ca < common["Ca"]:
        messages.append("Caが 1.1 未満です。カルチコール 0.5ml/kg を投与してください。")
    else:
        messages.append("Caは基準値内です。")

    # pO2 評価 (酸素/NO 調整)
    if disease in ("根治術", "フォンタン手術（フェネストレーションなし）"):
        if pO2 >= 100:
            messages.append("pO2が100以上です。酸素またはNOの減量を検討してください。")
        else:
            messages.append("pO2が100未満です。酸素またはNOの増量を検討してください。")
    elif disease == "単心室姑息術":
        if pO2 >= 50:
            messages.append("pO2が50以上です。酸素またはNOの減量を検討してください。")
        elif pO2 < 38:
            messages.append("pO2が38未満です。酸素またはNOの増量を検討してください。")
        else:
            messages.append("pO2が38-49です。酸素は適正で経過観察。")
    elif disease == "両心室姑息術":
        if pO2 >= 65:
            messages.append("pO2が65以上です。酸素の減量を検討してください。")
        elif pO2 < 50:
            messages.append("pO2が50未満です。酸素を増量してください。")
        else:
            messages.append("pO2が50-64です。酸素は適正で経過観察。")
    elif disease == "グレン手術":
        if pO2 >= 55:
            messages.append("pO2が55以上です。酸素またはNOの減量を検討してください。")
        elif pO2 < 45:
            messages.append("pO2が45未満です。酸素またはNOの増量を検討してください。")
        else:
            messages.append("pO2が45-54です。酸素とNOは適正で経過観察。")
    elif disease == "フォンタン手術（フェネストレーションあり）":
        if pO2 >= 60:
            messages.append("pO2が60以上です。酸素またはNOの減量を検討してください。")
        elif pO2 < 45:
            messages.append("pO2が45未満です。酸素またはNOの増量を検討してください。")
        else:
            messages.append("pO2が45-59です。酸素とNOは適正で経過観察。")

    return {
        "estimated_pCO2": estimated_pco2,
        "anion_gap": anion_gap,
        "messages": messages,
    }
