try:  # pandas は必須ではなく、利用可能な場合のみ読み込む
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas が無い環境でも動作させる
    pd = None  # type: ignore
import time
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import re
import os
from pathlib import Path
import json
from typing import List, Optional, Union, Dict, MutableMapping

# ==== 各評価ロジック ====
from vitals.spo2_logic import evaluate_spo2
from vitals.critical_spo2_logic import evaluate_critical_spo2
from vitals.sbp_logic import evaluate_sbp
from vitals.cvp_logic import evaluate_cvp
from vitals.adrenaline_logic import evaluate_adrenaline
from vitals.dobutamine_logic import evaluate_dobutamine
from vitals.bpup_logic import evaluate_bpup
from vitals.bpdown_logic import evaluate_bpdown
from vitals.bleed_logic import evaluate_bleed
from vitals.transfusion_logic import evaluate_transfusion
from vitals.sbp_trend import check_sbp_trend
from common.tree_parser import load_tree

# パネルUI
try:
    from fluid_panel import launch_fluid_panel
except Exception:
    launch_fluid_panel = None

try:
    from drug_panel import launch_drug_panel
except Exception:
    launch_drug_panel = None

try:
    from panel_tabs import launch_drug_fluid_tabs
except Exception:
    launch_drug_fluid_tabs = None

# =========================
# 設定ロード
# =========================

def load_config():
    cfg = Path(__file__).with_name("config.json")
    if cfg.is_file():
        with open(cfg, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

CONFIG = load_config()

# Shared thresholds editable via GUI tabs
THRESHOLDS: MutableMapping[str, float] = {}

# Shared surgery selection
SURGERY_STATE: MutableMapping[str, str] = {"type": "根治術"}

# Mapping of surgery-specific SpO2 actions
SPO2_ACTIONS: Dict[str, Dict[str, str]] = {
    "根治術": {
        "upper": "SpO2_uより大きいSpO2の場合FiO2を下げるまたNOを減量してもよいです",
        "lower": "SpO2_lより小さいSpO2の場合FiO2をあげるまたNOを増量してもよいです",
    },
    "姑息術": {
        "upper": "SpO2_uより大きいSpO2の場合FiO2を下げるまたNOを減量してください",
        "lower": "SpO2_lより小さいSpO2の場合FiO2をあげるまたNOを増量してください",
    },
    "Glenn": {
        "upper": "SpO2_uより大きいSpO2の場合FiO2を下げるまたNOを減量してもよいです",
        "lower": "SpO2_lより小さいSpO2の場合FiO2をあげるまたNOを増量してもよいです",
    },
    "Fontan(フェネストレーションあり)": {
        "upper": "SpO2_uより大きいSpO2の場合FiO2を下げるまたNOを減量してもよいです",
        "lower": "SpO2_lより小さいSpO2の場合FiO2をあげるまたNOを増量してもよいです",
    },
}

# 既定の親フォルダ候補（vital_reader と揃える）
DEFAULT_VITALS_BASE_CANDIDATES = [
    str(Path(__file__).with_name("vitals")),
    r"C:\\Users\\sakai\\OneDrive\\Desktop\\BOT\\algorithm\\vitals",
]

# 環境変数 > config.json > 既定候補 の順

def resolve_path(
    env_key: str,
    cfg_key: str,
    candidates: Optional[List[str]] = None,
    must_exist: bool = False,
) -> Path:
    # 1) env
    p = os.getenv(env_key)
    if p:
        pth = Path(p).expanduser()
        if (not must_exist) or pth.exists():
            return pth
    # 2) config
    p = CONFIG.get(cfg_key)
    if p:
        pth = Path(p).expanduser()
        if (not must_exist) or pth.exists():
            return pth
    # 3) candidates
    for c in candidates or []:
        pth = Path(c).expanduser()
        if (not must_exist) or pth.exists():
            return pth
    # 4) 最後にスクリプト隣を返す
    return Path(candidates[0]).expanduser() if candidates else Path.cwd()

VITALS_BASE_DIR = resolve_path("VITALS_BASE_DIR", "VITALS_BASE_DIR", DEFAULT_VITALS_BASE_CANDIDATES, must_exist=False)

DEFAULT_PAUSE_MIN = 10  # 予備のデフォルト（Treeに数値が無い等のとき）

# ---------------- ユーティリティ ----------------

def parse_pause_min(val):
    """Treeの『pause_min』文字列を float[分] に正規化。
    - 全角→半角、空白除去
    - "60" / "60m" / "60min" → 60
    - "600s" → 10
    - "00:10:00" → 10
    変換できない/NaN/空 → DEFAULT_PAUSE_MIN
    """
    if val is None:
        return DEFAULT_PAUSE_MIN
    try:
        if isinstance(val, float):
            try:
                if pd is not None and pd.isna(val):
                    return DEFAULT_PAUSE_MIN
            except Exception:
                pass
            # pandas が無い場合は NaN 判定を簡易的に行う
            if pd is None and val != val:
                return DEFAULT_PAUSE_MIN
        s = str(val).strip()
        if not s:
            return DEFAULT_PAUSE_MIN
        # 全角→半角
        s = s.translate(str.maketrans({
            '０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9','．':'.','：':':'
        }))
        s = re.sub(r"\s+", "", s)
        if ':' in s:  # HH:MM:SS or MM:SS
            parts = [float(p) for p in s.split(':')]
            if len(parts) == 3:
                h, m, sec = parts
            elif len(parts) == 2:
                h, m, sec = 0.0, parts[0], parts[1]
            else:
                return DEFAULT_PAUSE_MIN
            return h*60 + m + sec/60
        m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-zA-Z]*)", s)
        if m:
            num = float(m.group(1)); unit = m.group(2).lower()
            if unit in ("", "m", "min", "mins", "minute", "minutes"):
                return num
            if unit in ("s", "sec", "secs", "second", "seconds"):
                return num/60
        return float(s)
    except Exception:
        return DEFAULT_PAUSE_MIN


def dedup_by_id(instructions):
    seen = set(); out = []
    for inst in instructions:
        _id = inst.get('id')
        if _id in seen:
            continue
        seen.add(_id)
        out.append(inst)
    return out


def fmt_comment(cmt):
    if cmt is None:
        return ""
    try:
        if pd is not None and pd.isna(cmt):
            return ""
    except Exception:
        pass
    c = str(cmt).strip()
    return c if c else ""


def adjust_spo2_actions(instructions, surgery_type: str):
    """Modify SpO2 related instructions based on selected surgery type."""
    mapping = SPO2_ACTIONS.get(surgery_type)
    if not mapping:
        return instructions
    adjusted = []
    for inst in instructions:
        _id = inst.get("id", "")
        if _id.startswith("SPO2_UPPER") and "resolve" not in _id:
            new_inst = dict(inst)
            new_inst["instruction"] = mapping["upper"]
            adjusted.append(new_inst)
        elif _id.startswith("SPO2_LOWER") and "resolve" not in _id:
            new_inst = dict(inst)
            new_inst["instruction"] = mapping["lower"]
            adjusted.append(new_inst)
        else:
            adjusted.append(inst)
    return adjusted

# ---------------- 共通評価 ----------------

def evaluate_all(vitals: dict, tree_df, thresholds, phase='a', bpup_tree_df=None):
    """Evaluate all vitals and return intervention instructions.
    """
    instructions = []
    spo2_instructions = evaluate_spo2(vitals, tree_df, thresholds, phase)
    if any(i["id"] == "SPO2_CHECK" for i in spo2_instructions):
        if vitals.get("SPO2_CHECK_DONE") == "Y":
            spo2_instructions = [i for i in spo2_instructions if i["id"] != "SPO2_CHECK"]
        else:
            spo2_instructions = [i for i in spo2_instructions if i["id"] == "SPO2_CHECK"]
    instructions += spo2_instructions

    instructions += evaluate_critical_spo2(vitals, tree_df, thresholds, phase)
    instructions += evaluate_cvp(vitals, tree_df, thresholds, phase)
    instructions += evaluate_sbp(vitals, tree_df, thresholds, phase)
    instructions += evaluate_adrenaline(vitals, tree_df, thresholds, phase)
    instructions += evaluate_dobutamine(vitals, tree_df, thresholds, phase)
    sbp = vitals.get("SBP")
    sbp_u = thresholds.get("SBP_u")
    sbp_l = thresholds.get("SBP_l")
    if sbp is not None:
        if sbp_u is not None and sbp > sbp_u:
            # ``bpup_tree_df`` may be a DataFrame, which cannot be used directly in
            # boolean expressions like ``bpup_tree_df or tree_df``.  Doing so raises
            # ``ValueError: The truth value of a DataFrame is ambiguous``.  To avoid
            # this we explicitly check for ``None`` and fall back to ``tree_df``.
            chosen_tree = bpup_tree_df if bpup_tree_df is not None else tree_df
            try:
                from entry_time_panel import get_elapsed_minutes

                elapsed = get_elapsed_minutes()
            except Exception:
                elapsed = None
            instructions += evaluate_bpup(
                vitals, chosen_tree, thresholds, phase, elapsed_minutes=elapsed
            )
        elif sbp_l is not None and sbp < sbp_l:
            instructions += evaluate_bpdown(vitals, tree_df, thresholds, phase)
    instructions += evaluate_bleed(vitals, tree_df, thresholds, phase)
    instructions += evaluate_transfusion(vitals, tree_df, thresholds, phase)
    if not instructions:
        instructions.append({
            "id": "OBSERVATION",
            "instruction": "経過観察",
            "pause_min": 0,
            "next_id": None,
            "comment": "",
        })
    return instructions

# ---------------- CVP follow-up ----------------

def compute_cvp_follow_instructions(base_results, vitals, tree_df, thresholds, surgery_type, bpup_tree_df=None):
    """Return follow-up instructions after CVP check.

    Any instructions that were already present alongside ``CVP_UPPER_CHECK`` in
    ``base_results`` (e.g. ``CVP_UPPER_A_SBP_UPPER``) are preserved and merged
    with a fresh evaluation of all rules.  This ensures that subsequent actions
    tied to elevated SBP are not lost when the CVP check is triggered.
    """
    other = [r for r in base_results if r.get("id") != "CVP_UPPER_CHECK"]
    follow_raw = other + evaluate_all(
        vitals, tree_df, thresholds, phase="a", bpup_tree_df=bpup_tree_df
    )
    return [
        r
        for r in adjust_spo2_actions(dedup_by_id(follow_raw), surgery_type)
        if r["id"] not in ("CVP_UPPER_CHECK", "CVP_UPPER_CHECK_Y", "CVP_UPPER_CHECK_N")
    ]

# ---------------- データ取得 ----------------

def get_latest_vitals(path: Union[Path, str]):
    if pd is None:
        print("[!] pandas が利用できないため CSV を読み込めません")
        return None
    try:
        path = Path(path)
        if path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        if df.empty:
            return None
        # Forward-fill missing values so that failed OCR or partial updates
        # do not erase previously captured vitals.  Any remaining NaNs (for
        # columns that have never been populated) are converted to ``None`` to
        # avoid propagating ``nan`` values to downstream logic.
        df = df.ffill()
        last_row = df.iloc[-1]
        return {k: (None if pd.isna(v) else v) for k, v in last_row.items()}
    except Exception as e:
        print(f"[!] 読み込みエラー: {e}")
        return None

# ---------------- UI ----------------

def prompt_thresholds():
    print("しきい値を設定してください（Enterでデフォルト値使用）")
    def get_val(label, default):
        val = input(f"{label} [{default}]: ")
        return float(val) if val else default
    return {
        "SpO2_l": get_val("SpO2 下限 (SpO2_l)", 80),
        "SpO2_u": get_val("SpO2 上限 (SpO2_u)", 100.0),
        "Critical_SpO2_l": get_val("Critical SpO2 下限 (Critical_SpO2_l)", 75),
        "Critical_SpO2_u": get_val("Critical SpO2 上限 (Critical_SpO2_u)", 100.0),
        "SBP_l": get_val("SBP 下限 (SBP_l)", 70),
        "SBP_u": get_val("SBP 上限 (SBP_u)", 90),
        "CVP_u": get_val("CVP 上限 (CVP_u)", 5),
        "CVP_c": get_val("Critical CVP 上限 (CVP_c)", 8),
    }


def yn_dialog(title, prompt):
    root = tk.Tk(); root.withdraw()
    result = None
    while result not in ("Y", "N"):
        result = simpledialog.askstring(title, f"{prompt} [Y/N]", parent=root)
        if result is not None:
            result = result.upper()
        if result not in ("Y", "N"):
            messagebox.showerror("入力エラー", "YかNを入力してください", parent=root)
    root.destroy()
    return result


def update_threshold(thresholds, key, new_value):
    thresholds[key] = new_value
    print(f"{key} を {new_value} に更新しました。")


def handle_spo2_check_n(vitals_memory):
    """Handle state updates when SPO2_CHECK receives an 'N' answer."""
    vitals_memory["SPO2_CHECK_PAUSE_UNTIL"] = time.time() + 60 * 60

def handle_cvp_check_n(vitals_memory, vitals):
    """Handle state updates when CVP_UPPER_CHECK receives an 'N' answer."""
    vitals_memory["CVP_LINE_CHECK_count"] = 0
    vitals_memory["CVP_CHECK_PAUSE_UNTIL"] = None
    vitals_memory["CVP_NEXT_R_TS"] = None
    vitals["CVP_NEXT_R_TS"] = None


def handle_cvp_observation_comment(vitals_memory):
    """Increment counter and return comment after three consecutive observations."""
    vitals_memory["CVP_OBS_COUNT"] = vitals_memory.get("CVP_OBS_COUNT", 0) + 1
    if vitals_memory["CVP_OBS_COUNT"] >= 3:
        vitals_memory["CVP_OBS_COUNT"] = 0
        return "僧帽弁逆流・三尖弁逆流と両心室の動きをみてCVPの基準値を変えてください"
    return ""

# ---------------- ベッド選択＆CSVパス解決 ----------------

def select_bed_and_csv(vitals_base_dir: Path) -> Path:
    """vitals_base_dir/YYYYMMDD/vitals_history_{bed}.csv を返す（存在しなければ作成）。"""
    root = tk.Tk(); root.withdraw()
    valid_beds = ["2", "3", "4", "5"]
    while True:
        bed_choice = simpledialog.askstring("ベッド選択", "ベッド番号を入力してください（2～5）：", parent=root)
        if bed_choice in valid_beds:
            today = datetime.now().strftime("%Y%m%d")
            day_dir = vitals_base_dir / today
            day_dir.mkdir(parents=True, exist_ok=True)
            csv_path = day_dir / f"vitals_history_{bed_choice}.csv"
            if not csv_path.exists():
                if pd is not None:
                    pd.DataFrame(columns=["timestamp"]).to_csv(csv_path, index=False)
                else:
                    csv_path.write_text("timestamp\n", encoding="utf-8")
            root.destroy();
            print(f"選択されたベッド: {bed_choice}")
            print(f"保存先CSV: {csv_path}")
            return csv_path
        else:
            messagebox.showerror("エラー", "2～5 の数字を入力してください。", parent=root)

# ---------------- メインループ ----------------

def main_loop(
    vitals_path: Path,
    thresholds: Optional[MutableMapping[str, float]] = None,
    surgery_state: Optional[MutableMapping[str, str]] = None,
):
    global THRESHOLDS, SURGERY_STATE
    THRESHOLDS = thresholds if thresholds is not None else prompt_thresholds()
    thresholds = THRESHOLDS
    SURGERY_STATE = surgery_state if surgery_state is not None else SURGERY_STATE
    print("\n【現在のしきい値】")
    for k, v in thresholds.items():
        print(f"{k}: {v}")

    # Load rule trees relative to this script so execution works regardless of
    # the current working directory.
    tree_df = load_tree(Path(__file__).with_name("tree.yaml"))
    bpup_tree_df = load_tree(Path(__file__).with_name("bpup_tree.yaml"))
    last_timestamp = None
    last_instruction_time: dict[str, float] = {}
    vitals_memory = {
        "CVP_LINE_CHECK_count": 0,
        "CVP_NEXT_R_TS": None,
        "EPISODE_LATCH": set(),
        "FRO_CHECK_ASKED": False,
        "FRO_CHECK": None,
        "CVP_CHECK_PAUSE_UNTIL": None,
        "SPO2_CHECK_PAUSE_UNTIL": None,
        "CVP_OBS_COUNT": 0,
        "FRO_CVP_BASE": None,
    }
    print("\n==== 自動判定を開始（Ctrl+Cで終了）====")
    while True:
        vitals = get_latest_vitals(vitals_path)
        if vitals is None or 'timestamp' not in vitals:
            print("[!] バイタル情報が不完全、再試行します")
            time.sleep(10); continue

        print("【判定直前しきい値】", thresholds)
        print("【判定直前バイタル】", vitals)

        surgery_type = SURGERY_STATE.get("type", "根治術")

        # 状態注入
        for key in vitals_memory:
            vitals[key] = vitals_memory[key]

        # 新しいデータ行でのみ評価
        if last_timestamp is None or vitals['timestamp'] != last_timestamp:
            print(f"\n--- {vitals['timestamp']} の判定 ---")

            trend = check_sbp_trend(vitals_path)
            if trend:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ALARM ΔSBP={trend['change']:+.0f}: {trend['instruction']}"
                )

            # A相
            a_results_raw = evaluate_all(vitals, tree_df, thresholds, phase='a', bpup_tree_df=bpup_tree_df)
            a_results = adjust_spo2_actions(dedup_by_id(a_results_raw), surgery_type)

            ids = {r['id'] for r in a_results}
            if 'CVP_UPPER_CHECK' in ids:
                check_list = [r for r in a_results if r['id'] == 'CVP_UPPER_CHECK']
                for inst in check_list:
                    _id = inst['id']
                    now = time.time()

                    # N応答でのポーズ
                    pause_until = vitals_memory.get("CVP_CHECK_PAUSE_UNTIL")
                    if pause_until and now < pause_until:
                        rem = int(pause_until - now); rem = max(rem, 0)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                        continue
                    else:
                        vitals_memory["CVP_CHECK_PAUSE_UNTIL"] = None

                    pause_min = parse_pause_min(inst.get('pause_min', DEFAULT_PAUSE_MIN) if 'pause_min' in inst else inst.get('ポーズ(min)', DEFAULT_PAUSE_MIN))
                    prev = last_instruction_time.get(_id, 0)
                    if (now - prev) <= (pause_min * 60):
                        rem = int(pause_min*60 - (now - prev)); rem = max(rem, 0)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                        continue

                    # Y/N ダイアログ
                    answer = yn_dialog(
                        "CVPの値確認",
                        "CVPの値が正しいかチェックしてください：ライン閉塞・空気混入・トランスデューサの高さ調整\nCVPの値は正しいですか？",
                    )
                    vitals_memory["CVP_LINE_CHECK"] = answer
                    vitals["CVP_LINE_CHECK"] = answer
                    last_instruction_time[_id] = now

                    if answer == "N":
                        handle_cvp_check_n(vitals_memory, vitals)
                        continue

                    # 3回連続Yで心エコー評価
                    skip_follow = False
                    vitals_memory["CVP_LINE_CHECK_count"] = vitals_memory.get("CVP_LINE_CHECK_count", 0) + 1
                    if vitals_memory["CVP_LINE_CHECK_count"] >= 3:
                        echo_ans = yn_dialog(
                            "心エコー確認",
                            "僧帽弁逆流・三尖弁逆流・心室の動きは許容範囲内でしたか？",
                        )
                        vitals_memory["CVP_LINE_CHECK_count"] = 0
                        if echo_ans == "Y":
                            root = tk.Tk(); root.withdraw()
                            new_val = simpledialog.askfloat(
                                "CVP基準値変更",
                                f"CVP_u基準値を変更してください（現在値: {thresholds['CVP_u']:.1f}）",
                                parent=root,
                            )
                            root.destroy()
                            if new_val is not None:
                                update_threshold(thresholds, "CVP_u", new_val)
                                print(f"CVP_uを{new_val}に変更しました。")
                                last_timestamp = None
                            skip_follow = True
                        else:
                            # 許容できなければ後続指示へ
                            skip_follow = False

                    # R相は10分後
                    vitals_memory["CVP_NEXT_R_TS"] = time.time() + 10*60
                    vitals["CVP_NEXT_R_TS"] = vitals_memory["CVP_NEXT_R_TS"]

                    # CHECK直後に、A相のうちCHECK以外を再評価して表示
                    if not skip_follow:
                        follow = compute_cvp_follow_instructions(
                            a_results, vitals, tree_df, thresholds, surgery_type, bpup_tree_df
                        )
                        if not follow:
                            comment = handle_cvp_observation_comment(vitals_memory)
                            follow = [{
                                'id': 'OBSERVATION',
                                'instruction': '経過観察',
                                'comment': comment,
                                'pause_min': 0,
                            }]
                        else:
                            vitals_memory['CVP_OBS_COUNT'] = 0
                        now = time.time()
                        for nxt in follow:
                            _nid = nxt['id']
                            pause_min = parse_pause_min(nxt.get('pause_min', DEFAULT_PAUSE_MIN))
                            prev = last_instruction_time.get(_nid, 0)
                            if (now - prev) <= (pause_min * 60):
                                rem = int(pause_min*60 - (now - prev)); rem = max(rem, 0)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_nid}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                                continue
                            if '終了' in str(nxt['instruction']):
                                vitals_memory['EPISODE_LATCH'].add(_nid)
                            cmt = fmt_comment(nxt.get('comment'))
                            cmt_str = f"（{cmt})" if cmt else ""
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_nid} → {nxt['instruction']}{cmt_str}")
                            last_instruction_time[_nid] = now
                            if _nid == 'CVP_UPPER_A_SBP_UPPER':
                                try:
                                    vitals_memory['FRO_CVP_BASE'] = float(vitals.get('CVP'))
                                except (TypeError, ValueError):
                                    vitals_memory['FRO_CVP_BASE'] = None
            elif 'SPO2_CHECK' in ids:
                check_list = [r for r in a_results if r['id'] == 'SPO2_CHECK']
                for inst in check_list:
                    _id = inst['id']
                    now = time.time()

                    pause_until = vitals_memory.get('SPO2_CHECK_PAUSE_UNTIL')
                    if pause_until and now < pause_until:
                        rem = int(pause_until - now); rem = max(rem, 0)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                        continue
                    else:
                        vitals_memory['SPO2_CHECK_PAUSE_UNTIL'] = None
                        vitals_memory['SPO2_CHECK_DONE'] = None
                        vitals['SPO2_CHECK_DONE'] = None

                    pause_min = parse_pause_min(inst.get('pause_min', DEFAULT_PAUSE_MIN) if 'pause_min' in inst else inst.get('ポーズ(min)', DEFAULT_PAUSE_MIN))
                    prev = last_instruction_time.get(_id, 0)
                    if (now - prev) <= (pause_min * 60):
                        rem = int(pause_min*60 - (now - prev)); rem = max(rem, 0)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                        continue

                    answer = yn_dialog('SpO2の値確認', 'SpO2の値は正しいですか？')
                    last_instruction_time[_id] = now
                    if answer == 'N':
                        handle_spo2_check_n(vitals_memory)
                        vitals['SPO2_CHECK_DONE'] = None
                        continue

                    vitals_memory['SPO2_CHECK_DONE'] = 'Y'
                    vitals['SPO2_CHECK_DONE'] = 'Y'
                    follow_raw = evaluate_all(vitals, tree_df, thresholds, phase='a', bpup_tree_df=bpup_tree_df)
                    follow = [
                        r for r in adjust_spo2_actions(dedup_by_id(follow_raw), surgery_type)
                        if r['id'] != 'SPO2_CHECK'
                    ]
                    if not follow:
                        follow = [{
                            'id': 'OBSERVATION',
                            'instruction': '経過観察',
                            'comment': '',
                            'pause_min': 0,
                        }]
                    now2 = time.time()
                    for nxt in follow:
                        _nid = nxt['id']
                        pause_min = parse_pause_min(nxt.get('pause_min', DEFAULT_PAUSE_MIN))
                        prev = last_instruction_time.get(_nid, 0)
                        if (now2 - prev) <= (pause_min * 60):
                            rem = int(pause_min*60 - (now2 - prev)); rem = max(rem, 0)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_nid}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                            continue
                        if '終了' in str(nxt['instruction']):
                            vitals_memory['EPISODE_LATCH'].add(_nid)
                        cmt = fmt_comment(nxt.get('comment'))
                        cmt_str = f"（{cmt})" if cmt else ""
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_nid} → {nxt['instruction']}{cmt_str}")
                        last_instruction_time[_nid] = now2
                    vitals_memory['SPO2_CHECK_PAUSE_UNTIL'] = time.time() + 60*60
            else:
                # 通常A相
                for inst in a_results:
                    _id = inst['id']
                    if _id in vitals_memory['EPISODE_LATCH']:
                        continue
                    pause_min = parse_pause_min(inst.get('pause_min', inst.get('ポーズ(min)', DEFAULT_PAUSE_MIN)))
                    now = time.time(); prev = last_instruction_time.get(_id, 0)
                    if (now - prev) <= (pause_min * 60):
                        rem = int(pause_min*60 - (now - prev)); rem = max(rem, 0)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                        continue
                    cmt = fmt_comment(inst.get('comment'))
                    cmt_str = f"（{cmt})" if cmt else ""
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id} → {inst['instruction']}{cmt_str}")
                    last_instruction_time[_id] = now
                    if _id == 'CVP_UPPER_A_SBP_UPPER':
                        try:
                            vitals_memory['FRO_CVP_BASE'] = float(vitals.get('CVP'))
                        except (TypeError, ValueError):
                            vitals_memory['FRO_CVP_BASE'] = None
                    if '終了' in str(inst['instruction']):
                        vitals_memory['EPISODE_LATCH'].add(_id)

            last_timestamp = vitals['timestamp']

        # R相（スケジュールで10分後）
        now_ts = time.time()
        if vitals_memory.get("CVP_NEXT_R_TS") and now_ts >= vitals_memory["CVP_NEXT_R_TS"]:
            for key in vitals_memory:
                vitals[key] = vitals_memory[key]

            r_results = adjust_spo2_actions(
                dedup_by_id(evaluate_all(vitals, tree_df, thresholds, phase='r', bpup_tree_df=bpup_tree_df)),
                surgery_type,
            )
            for inst in r_results:
                _id = inst['id']
                if _id in vitals_memory['EPISODE_LATCH']:
                    continue

                # フロセミド効果チェック（1回だけY/N取得）
                if _id == 'CVP_FRO_CHECK' and not vitals_memory.get('FRO_CHECK_ASKED'):
                    ans = yn_dialog("フロセミド効果チェック", inst['instruction'])
                    vitals_memory['FRO_CHECK'] = ans
                    vitals_memory['FRO_CHECK_ASKED'] = True
                    vitals['フロセミドチェック'] = ans
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id} → {inst['instruction']}（Y/N入力済）")
                    try:
                        cvp_now = float(vitals.get('CVP')) if vitals.get('CVP') not in (None, "") else None
                    except (TypeError, ValueError):
                        cvp_now = None
                    cvp_base = vitals_memory.get('FRO_CVP_BASE')
                    if ans == 'Y':
                        if cvp_base is not None and cvp_now is not None and cvp_now < cvp_base:
                            msg = "CVP下降傾向。経過観察してください。"
                        else:
                            msg = "CVP下降なし。追加対応を検討してください。"
                        vitals_memory['EPISODE_LATCH'].add('CVP_FRO_YES')
                    else:
                        msg = (
                            "輸血量を減らすことを検討してください。ＣＶＰの基準値を僧帽弁逆流・三尖弁逆流・心室の動きをエコーで見て変更することを考慮してください。"
                        )
                        vitals_memory['EPISODE_LATCH'].add('CVP_FRO_NO')
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                    vitals_memory['FRO_CVP_BASE'] = None
                    last_instruction_time[_id] = now_ts
                    continue

                pause_min = parse_pause_min(inst.get('pause_min', inst.get('ポーズ(min)', DEFAULT_PAUSE_MIN)))
                prev = last_instruction_time.get(_id, 0)
                if (now_ts - prev) <= (pause_min * 60):
                    rem = int(pause_min*60 - (now_ts - prev)); rem = max(rem, 0)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id}（指示は{rem//60}分{rem%60}秒後までポーズ中）")
                    continue
                cmt = fmt_comment(inst.get('comment'))
                cmt_str = f"（{cmt})" if cmt else ""
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ID={_id} → {inst['instruction']}{cmt_str}")
                last_instruction_time[_id] = now_ts
                if '終了' in str(inst['instruction']):
                    vitals_memory['EPISODE_LATCH'].add(_id)

            vitals_memory["CVP_NEXT_R_TS"] = None

        # ラッチ解除（CVPが閾値内に戻ったら解除＆FROフラグもリセット）
        try:
            cvp = float(vitals.get('CVP')) if vitals.get('CVP') not in (None, "") else None
        except Exception:
            cvp = None
        if cvp is not None and cvp <= thresholds['CVP_u']:
            vitals_memory['EPISODE_LATCH'].clear()
            vitals_memory['FRO_CHECK_ASKED'] = False
            vitals_memory['FRO_CHECK'] = None

        time.sleep(60)

if __name__ == '__main__':
    vitals_path = select_bed_and_csv(VITALS_BASE_DIR)
    thresholds = prompt_thresholds()
    manager = None
    surgery_state = {"type": "根治術"}
    if launch_drug_fluid_tabs:
        import multiprocessing as mp

        manager = mp.Manager()
        thresholds = manager.dict(thresholds)
        surgery_state = manager.dict(surgery_state)
        launch_drug_fluid_tabs(
            drug_csv_path=str(vitals_path),
            fluid_csv_path=str(vitals_path),
            thresholds=thresholds,
            surgery_state=surgery_state,
        )
    else:
        if launch_fluid_panel:
            launch_fluid_panel(csv_path=str(vitals_path))
        if launch_drug_panel:
            launch_drug_panel(csv_path=str(vitals_path))
    main_loop(vitals_path, thresholds, surgery_state)
