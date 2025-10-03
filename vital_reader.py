import time
import os
import csv
from datetime import datetime
from pathlib import Path
import argparse
import json
import sys
from typing import Optional
from decimal import Decimal, InvalidOperation
try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None
import re
import unicodedata

# Optional heavy dependencies -------------------------------------------------
try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

try:  # pragma: no cover - optional dependency
    import easyocr  # type: ignore
except Exception:  # pragma: no cover
    easyocr = None

try:  # pragma: no cover - optional dependency
    import tkinter as tk  # type: ignore
    from tkinter import messagebox, simpledialog  # type: ignore
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    messagebox = None  # type: ignore
    simpledialog = None  # type: ignore

if easyocr:
    # ``easyocr.Reader`` enables GPU acceleration when ``gpu=True``.  Some
    # environments install ``easyocr`` without installing PyTorch, which caused
    # the previous code to raise an :class:`AttributeError` when evaluating
    # ``torch.cuda.is_available()`` because ``torch`` was ``None``.  Falling back
    # to CPU OCR keeps the feature working even when PyTorch is unavailable.
    use_gpu = torch.cuda.is_available() if torch is not None else False
    easyocr_reader = easyocr.Reader(['en', 'ja'], gpu=use_gpu, verbose=False)
else:  # pragma: no cover - easyocr not available
    easyocr_reader = None

if easyocr_reader is None:  # pragma: no cover - optional dependency warning
    print(
        "[WARN] EasyOCR (easyocr) が利用できません。'pip install easyocr' でインストール"
        " するまで OCR 機能は動作しません。"
    )

from bed_coords import BED_COORDS_8
from bed_coords_4 import BED_COORDS_4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_GUARD_PARAMS = dict(
    band_ratio=0.55,
    blur_kx_ratio=0.30,
    ratio_thr=1.10,
    height_ratio=0.45,
    outside_max=2,
    edge_margin_ratio=0.03,
)

def build_spont_breath_model(backbone: str, img_h: int, img_w: int):
    from torchvision import models, transforms
    import torch.nn as nn
    tfm = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    b = backbone.lower()
    if b in ["mobilenetv3", "mobilenet_v3_small"]:
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
        return m, tfm
    if b in ["efficientnet_v2_s", "efficientnet"]:
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return m, tfm
    if b in ["convnext_tiny", "convnext"]:
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
        return m, tfm
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m, tfm

def guard_ok(crop_bgr, p=DEFAULT_GUARD_PARAMS):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    H, W = g.shape
    band_h = max(8, int(H * p["band_ratio"]))
    y0 = (H - band_h) // 2
    y1 = y0 + band_h
    kx = max(9, int(W * p["blur_kx_ratio"]) | 1)
    base = cv2.boxFilter(g, -1, (kx, 1), normalize=True)
    diff = cv2.subtract(g, base)
    diff[diff < 0] = 0
    band = diff[y0:y1, :]
    col = cv2.GaussianBlur(
        band.sum(axis=0).astype(np.float32).reshape(1, -1), (9, 1), 0
    ).ravel()
    med = float(np.median(col))
    mx = float(col.max())
    ratio = mx / (med + 1e-6) if med > 0 else 0
    x_hit = int(np.argmax(col))
    margin = max(4, int(W * p["edge_margin_ratio"]))
    if not (margin < x_hit < W - margin):
        return False, dict(ratio=ratio, height=0, leak=999)
    mu, sd = float(band.mean()), float(band.std() + 1e-6)
    m = (band >= (mu + 0.8 * sd)).astype(np.uint8)
    height = int(m[:, x_hit].sum())
    height_ok = height >= int(band_h * p["height_ratio"])
    mu2, sd2 = float(diff.mean()), float(diff.std() + 1e-6)
    mask_all = (diff >= (mu2 + 0.8 * sd2)).astype(np.uint8)
    leak = int(mask_all[:y0, x_hit].sum() + mask_all[y1:, x_hit].sum())
    leak_ok = leak <= p["outside_max"]
    return (ratio >= p["ratio_thr"]) and height_ok and leak_ok, dict(
        ratio=ratio, height=height, leak=leak
    )

# =========================
# OCR / 画像ユーティリティ
# =========================

# These globals are populated at runtime once the user selects the screen
# layout.  Providing safe defaults keeps the module importable for unit tests
# and makes helper functions usable without running the interactive workflow.
BP_COMBINED_COORD = (0, 0, 0, 0)
CVP_COORDS = (0, 0, 0, 0)
vital_crop: dict[str, tuple[int, int, int, int]] = {}
SPONT_BREATH_COORDS: list[tuple[int, int, int, int]] = []

# Lazy-loaded resources for spontaneous breath detection
spont_breath_model = None
spont_breath_meta = None
spont_breath_transform = None


def crop_image(img, coord):
    """Safely crop ``img`` according to ``coord``.

    ``coord`` is expected to be a four element tuple ``(x, y, w, h)``.  The
    function clamps the region so that it always lies inside ``img`` and
    supports both :class:`numpy.ndarray` inputs (the typical OpenCV image) and
    nested Python sequences which are convenient for lightweight tests.
    """

    if img is None:
        raise ValueError("img must not be None")

    if not coord:
        return img

    x, y, w, h = coord
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    w = max(0, int(w))
    h = max(0, int(h))
    x1 = x0 + w
    y1 = y0 + h

    if np is not None and isinstance(img, np.ndarray):
        height, width = img.shape[:2]
        x1 = min(width, x1)
        y1 = min(height, y1)
        x0 = min(x0, width)
        y0 = min(y0, height)
        return img[y0:y1, x0:x1].copy()

    # Fallback for simple list-of-lists images used in tests
    rows = img[y0:y1]
    return [row[x0:x1] for row in rows]


def sanitize_ocr_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.strip()
    return re.sub(r"\s+", "", normalized)


def normalize_ie_text(text: str) -> str:
    normalized = sanitize_ocr_text(text)
    if not normalized:
        return ""

    normalized = normalized.replace("\uFF1A", ":")
    normalized = normalized.replace("/", ":")
    normalized = normalized.replace("．", ".")
    normalized = normalized.replace("。", ".")
    normalized = normalized.replace(",", ".")
    normalized = re.sub(r"[^0-9:.]", "", normalized)
    normalized = re.sub(r":+", ":", normalized)

    if ":" not in normalized:
        return normalized.strip(".")

    left_part, right_part = normalized.split(":", 1)
    left_digits = re.sub(r"[^0-9]", "", left_part)
    right_clean = re.sub(r"[^0-9.]", "", right_part)

    if right_clean.count(".") > 1:
        first, *rest = right_clean.split(".")
        right_clean = first + "." + "".join(rest)

    right_clean = right_clean.strip(".")
    left_digits = left_digits.lstrip("0") or ("0" if left_digits else "")

    if left_digits:
        try:
            left_digits = str(int(left_digits))
        except ValueError:
            pass

    formatted_right = right_clean
    if formatted_right:
        try:
            dec = Decimal(formatted_right)
        except InvalidOperation:
            pass
        else:
            if dec == dec.to_integral():
                formatted_right = format(dec.to_integral(), "f")
            else:
                formatted_right = format(dec.normalize(), "f")

    if left_digits and formatted_right:
        return f"{left_digits}:{formatted_right}"
    if left_digits:
        return left_digits
    if formatted_right:
        return formatted_right
    return ""


def read_easy(img, allow_dot: bool = False):
    """Run EasyOCR on ``img`` and return the concatenated text.

    The function gracefully handles environments where EasyOCR (and its heavy
    dependencies) are not available by returning an empty result.
    """

    if easyocr_reader is None or img is None:
        return "", 0.0

    np_img = None
    if np is not None and isinstance(img, np.ndarray):
        np_img = img
    elif np is not None:
        try:
            np_img = np.array(img)
        except Exception:
            np_img = None

    if np_img is None or np_img.size == 0:
        return "", 0.0

    results = easyocr_reader.readtext(np_img, detail=1, paragraph=False)
    texts = []
    confidences = []
    for _, text, conf in results:
        cleaned = sanitize_ocr_text(text)
        if not cleaned:
            continue
        texts.append(cleaned)
        confidences.append(float(conf))

    joined = "".join(texts)
    if not allow_dot:
        joined = joined.replace(".", "")
    joined = joined.strip()
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return joined, confidence


def _bp_penalty(value: int, low: int, high: int) -> int:
    if low <= value <= high:
        return 0
    if value < low:
        return low - value
    return value - high


def _split_bp_digits(digits: str) -> tuple[str, str]:
    if not digits:
        return "", ""
    best = (digits, "")
    best_score = float("inf")
    for idx in range(1, len(digits)):
        left = digits[:idx]
        right = digits[idx:]
        try:
            sbp = int(left)
            dbp = int(right)
        except ValueError:
            continue
        score = _bp_penalty(sbp, 60, 260) + _bp_penalty(dbp, 20, 200)
        if score < best_score:
            best_score = score
            best = (str(sbp), str(dbp))
    return best


def _refine_bp_values(sbp: str, dbp: str, map_val: str) -> tuple[str, str]:
    """Correct implausible blood pressure splits when possible."""

    if not sbp or not dbp:
        return sbp, dbp

    try:
        sbp_int = int(sbp)
        dbp_int = int(dbp)
    except ValueError:
        return sbp, dbp

    if sbp_int >= dbp_int:
        return sbp, dbp

    map_int: Optional[int] = None
    if map_val:
        try:
            map_int = int(map_val)
        except ValueError:
            map_int = None

    def score(value: int) -> float:
        if map_int is not None:
            approx_map = round((sbp_int + 2 * value) / 3)
            return abs(approx_map - map_int)
        return abs(sbp_int - value)

    best_val: Optional[int] = None
    best_score = score(dbp_int)

    for start in range(1, len(dbp)):
        candidate_text = dbp[start:].lstrip("0")
        if not candidate_text:
            continue
        try:
            candidate_val = int(candidate_text)
        except ValueError:
            continue
        if candidate_val < 20 or candidate_val > sbp_int:
            continue
        candidate_score = score(candidate_val)
        if candidate_score < best_score:
            best_score = candidate_score
            best_val = candidate_val

    if best_val is not None:
        return str(sbp_int), str(best_val)

    return sbp, dbp


def parse_bp_text(raw_text: str) -> tuple[str, str, str, str]:
    """Parse blood pressure text into components.

    The monitor sometimes emits strings such as ``"11.0/97(57)"`` or even
    ``"11097(57"``.  The function normalises these variations, attempting to
    recover the systolic, diastolic and mean arterial pressure as strings.
    """

    normalized = sanitize_ocr_text(raw_text)
    if not normalized:
        return "", "", "", ""

    normalized = normalized.replace(",", ".")
    normalized = normalized.replace(".", "")

    map_val = ""
    map_match = re.search(r"\((\d{1,3})\)", normalized)
    if not map_match:
        map_match = re.search(r"\((\d{1,3})", normalized)
    if map_match:
        map_val = map_match.group(1)
        normalized = normalized[: map_match.start()]

    main = normalized
    sbp = ""
    dbp = ""
    if "/" in main:
        left, right = main.split("/", 1)
        sbp = re.sub(r"\D", "", left)
        dbp = re.sub(r"\D", "", right)
    else:
        digits = re.sub(r"\D", "", main)
        sbp, dbp = _split_bp_digits(digits)

    sbp = sbp.lstrip("0") or sbp
    dbp = dbp.lstrip("0") or dbp

    sbp, dbp = _refine_bp_values(sbp, dbp, map_val)

    sanitized = ""
    if sbp and dbp:
        sanitized = f"{sbp}/{dbp}"
    elif sbp:
        sanitized = sbp
    elif dbp:
        sanitized = dbp
    if map_val:
        sanitized = f"{sanitized}({map_val})" if sanitized else f"({map_val})"

    return sanitized, sbp, dbp, map_val


def read_bp_roi(roi):
    text, _ = read_easy(roi, allow_dot=False)
    return parse_bp_text(text)


def sanitize_numeric_text(text: str, allow_dot: bool = False, allow_sign: bool = False) -> str:
    normalized = sanitize_ocr_text(text)
    if not normalized:
        return ""
    allowed = "0123456789"
    if allow_dot:
        allowed += "."
    if allow_sign:
        allowed += "+-"
    cleaned = "".join(ch for ch in normalized if ch in allowed)
    if allow_dot:
        cleaned = cleaned.strip(".")
    return cleaned


def read_num_roi(roi, allow_dot: bool = False, allow_sign: bool = False):
    text, _ = read_easy(roi, allow_dot=allow_dot)
    return sanitize_numeric_text(text, allow_dot=allow_dot, allow_sign=allow_sign)


def read_temp_roi(roi):
    return read_num_roi(roi, allow_dot=True, allow_sign=True)


def read_ie_roi(roi):
    text, _ = read_easy(roi, allow_dot=True)
    return normalize_ie_text(text)


def normalize_vent_mode_label(label: str) -> str:
    if not label:
        return ""
    text = unicodedata.normalize("NFKC", str(label))
    replacements = [
        ("オートモード", " AUTO MODE "),
        ("オート", " AUTO "),
        ("モード", " MODE "),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\s_]+", " ", text)
    text = text.strip()
    text = text.upper()
    return re.sub(r"\s+", " ", text)


def is_pressure_support_mode(label: str) -> bool:
    normalized = normalize_vent_mode_label(label)
    if not normalized:
        return False
    tokens = re.split(r"[\s/+\-]+", normalized)
    token_set = {t for t in tokens if t}
    if "PRESSURE" in token_set and "SUPPORT" in token_set:
        return True
    ps_tokens = {"PS", "PSV", "P.S", "PRESSURESUPPORT", "SPONT", "SPONTANEOUS"}
    if any(token in ps_tokens for token in token_set):
        return True
    if "PRESSURE SUPPORT" in normalized:
        return True
    return False


def apply_pressure_support_split(results):
    if results is None:
        return

    mode = results.get("VentMode", "")
    vtset_val = results.get("VTset", "")
    vtset_str = "" if vtset_val is None else str(vtset_val)

    if is_pressure_support_mode(mode):
        results["PS"] = vtset_str
        results["VTset"] = ""
    else:
        results.setdefault("PS", "")


def read_mode_roi(roi):
    text, _ = read_easy(roi, allow_dot=False)
    return normalize_vent_mode_label(text)


def read_cvp_roi(roi):

    text, _ = read_easy(roi, allow_dot=False)
    return sanitize_numeric_text(text)

# =========================
# 設定ロード
# =========================

def load_config(path=None):
    cfg_path = Path(path) if path else Path(__file__).with_name("config.json")
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# =========================
# 経路解決（堅牢版）
# =========================

def resolve_existing_path(candidates):
    for p in candidates or []:
        if p and Path(p).expanduser().exists():
            return Path(p).expanduser()
    return None

def resolve_path(arg_val, env_name, config, key, candidates=None, must_exist=True):
    # 1) CLI 引数
    if arg_val:
        p = Path(arg_val).expanduser()
        if (not must_exist) or p.exists():
            return p
    # 2) 環境変数
    env_val = os.getenv(env_name)
    if env_val:
        p = Path(env_val).expanduser()
        if (not must_exist) or p.exists():
            return p
    # 3) config.json
    cfg_val = config.get(key)
    if cfg_val:
        p = Path(cfg_val).expanduser()
        if (not must_exist) or p.exists():
            return p
    # 4) 既定候補群
    p = resolve_existing_path(candidates)
    if p is not None:
        return p
    # 見つからない
    raise ValueError(f"{key} is not specified or not found. Set via argument, env {env_name}, config, or place the file at one of default locations.")

# 既定候補（あなたの環境向け）
DEFAULT_IMAGE_BASE_CANDIDATES = [
    r"Z:\\image",
]
DEFAULT_VITALS_BASE_CANDIDATES = [
    str(Path(__file__).with_name("vitals")),
    r"C:\\Users\\sakai\\OneDrive\\Desktop\\BOT\\vitals",
]
DEFAULT_SPONT_BREATH_MODEL_CANDIDATES = [
    str(Path(__file__).with_name("spont_breath_model.keras")),
    r"C:\\Users\\sakai\\OneDrive\\Desktop\\BOT\\spon\\models\\white_line_cls.pt",
]
DEFAULT_SPONT_BREATH_META_CANDIDATES = [
    r"C:\\Users\\sakai\\OneDrive\\Desktop\\BOT\\spon\\models\\white_line_cls.meta.json",
]

# =========================
# リソース初期化
# =========================

def init_resources(
    spont_breath_model_path: Optional[Path] = None,
    spont_breath_meta_path: Optional[Path] = None,
):
    """Load optional resources such as the spontaneous-breathing model."""

    global spont_breath_model, spont_breath_meta, spont_breath_transform

    spont_breath_model = None
    spont_breath_meta = None
    spont_breath_transform = None

    if spont_breath_model_path is None:
        return

    if torch is None:
        print("[WARN] PyTorchが利用できないため自発呼吸モデルを読み込みません。")
        return

    if Image is None:
        print("[WARN] Pillowが利用できないため自発呼吸モデルを読み込みません。")
        return

    try:
        meta: dict[str, object] = {}
        if spont_breath_meta_path and Path(spont_breath_meta_path).is_file():
            with open(spont_breath_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        backbone = str(meta.get("backbone", "mobilenet_v3_small")) if meta else "mobilenet_v3_small"
        img_h = int(meta.get("img_h", 128)) if meta else 128
        img_w = int(meta.get("img_w", 512)) if meta else 512

        model, transform = build_spont_breath_model(backbone, img_h, img_w)
        state = torch.load(str(spont_breath_model_path), map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()

        spont_breath_model = model
        spont_breath_transform = transform
        spont_breath_meta = meta
        print(f"自発呼吸モデルを読み込みました: {spont_breath_model_path}")
        if spont_breath_meta_path:
            print(f"自発呼吸メタデータ: {spont_breath_meta_path}")
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[WARN] 自発呼吸モデル読み込み失敗: {exc}")
        spont_breath_model = None
        spont_breath_transform = None
        spont_breath_meta = None


# =========================
# 引数
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spont-breath-model",
        help="Path to spontaneous-breathing model (.keras)",
    )
    parser.add_argument(
        "--spont-breath-meta",
        help="Path to spontaneous-breathing model metadata (JSON)",
    )
    parser.add_argument("--image-folder", help="Folder containing monitor images (親Z:\\imageでもOK)")
    parser.add_argument("--vitals-base", help="Folder to store CSVs (親フォルダ)。未指定なら自動推定")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument(
        "--beds",
        help="Comma-separated bed numbers to allow. Overrides host-based default.",
    )
    return parser.parse_args()

# =========================
# CSV作成・保存など
# =========================

VITAL_COLUMNS = [
    "timestamp", "SBP", "DBP", "MAP", "HR", "SpO2", "BSR1", "BSR2",
    "Tskin", "Trect", "etCO2", "RR", "Ppeak", "Pmean", "PEEPact", "RRact",
    "I_E", "FiO2", "VTe", "VTi", "PEEPset", "VTset", "CVP",
    "pH", "PaCO2", "pO2", "Hct", "K", "Na", "Cl", "Ca", "Glu", "Lac",
    "tBil", "HCO3", "BE", "Alb"
]

# ``save_vitals_to_csv`` only iterates through ``ALL_COLUMNS`` when constructing
# a row, therefore the list excludes ``timestamp`` which is handled
# separately.  Extra keys (e.g. drug doses, ventilator mode) are added later so
# they do not need to appear here.
ALL_COLUMNS = [c for c in VITAL_COLUMNS if c != "timestamp"]

# Columns such as bolus drug doses should not be forward-filled when appending
# new vital rows.  The set can be extended by other modules (see
# :mod:`main_surgery`).
NON_PERSISTENT_COLUMNS = {"furosemide_mg"}

def create_empty_vitals_csv(path):
    if not os.path.exists(path):
        import pandas as pd
        df = pd.DataFrame(columns=VITAL_COLUMNS)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 空のバイタルCSVを作成: {path}")

def select_display_and_bed(vitals_base_dir: Path, beds_override: Optional[list[str]] = None):
    """画面分割(4 または 8)とベッド番号を聞き取り、対象CSVパスを返す。"""

    display_map = {"4": BED_COORDS_4, "8": BED_COORDS_8}
    vitals_base_dir = Path(vitals_base_dir)
    vitals_base_dir.mkdir(parents=True, exist_ok=True)

    today_dir = vitals_base_dir / datetime.now().strftime("%Y%m%d")
    today_dir.mkdir(parents=True, exist_ok=True)

    use_gui = tk is not None and simpledialog is not None and messagebox is not None
    root = None
    if use_gui:
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:  # pragma: no cover - Tk が使えない環境では対話シェルへフォールバック
            root = None
            use_gui = False

    can_use_cli = not use_gui and sys.stdin is not None and sys.stdin.isatty()
    if not use_gui and not can_use_cli:
        raise RuntimeError(
            "対話ダイアログを表示できず標準入力も対話的ではありません。Tkinter を有効化するか、"
            "対話的なシェルから実行してください。"
        )

    if not use_gui and can_use_cli:
        print("[INFO] Tkinter が利用できないためコンソール入力にフォールバックします。")

    def ask_input(title: str, prompt: str) -> Optional[str]:
        if use_gui and root is not None:
            response = simpledialog.askstring(title, prompt, parent=root)
            return None if response is None else response.strip()
        if can_use_cli:
            try:
                return input(f"{prompt} ").strip()
            except EOFError:  # pragma: no cover - 非対話シェル
                return None
        return None

    def show_error(title: str, message: str) -> None:
        if use_gui and root is not None and messagebox is not None:
            messagebox.showerror(title, message, parent=root)
        else:
            print(f"{title}: {message}")

    display = None
    display_prompt = "画面分割数を入力してください（4 または 8）："
    while True:
        answer = ask_input("画面分割選択", display_prompt)
        if not answer:
            show_error("エラー", "4 または 8 を入力してください。")
            continue
        if answer in display_map:
            display = answer
            break
        show_error("エラー", "4 または 8 を入力してください。")

    available_beds = sorted({str(b) for b in display_map[display].keys()}, key=int)
    if beds_override:
        overrides = [b for b in beds_override if b in available_beds]
        if overrides:
            valid_beds = sorted(set(overrides), key=int)
        else:
            print(
                f"[WARN] ベッド指定 {beds_override} は利用可能なベッド {available_beds} に一致しません。既定値を使用します。"
            )
            valid_beds = available_beds
    else:
        valid_beds = available_beds

    bed_options_text = "、".join(valid_beds)
    bed_prompt = f"ベッド番号を入力してください（{bed_options_text}）："

    while True:
        bed_choice = ask_input("ベッド選択", bed_prompt)
        if bed_choice in valid_beds:
            selected_path = today_dir / f"vitals_history_{bed_choice}.csv"
            create_empty_vitals_csv(str(selected_path))
            if root is not None:
                root.destroy()
            return display, str(selected_path), int(bed_choice)
        show_error(
            "エラー",
            f"{bed_options_text}のいずれかの数字を入力してください。",
        )

def detect_spontaneous_breath(img, coords_list):
    """Detect spontaneous breathing by scanning ``coords_list``.

    If a trained CNN and its metadata were loaded via :func:`init_resources`,
    the model and a guard routine are used to evaluate each region of
    interest.  When those heavy dependencies are unavailable, the function
    falls back to a lightweight heuristic that simply checks for a bright
    horizontal line across the region.
    """
    if not coords_list:
        return False

    use_cnn = (
        spont_breath_model is not None
        and spont_breath_transform is not None
        and torch is not None
        and cv2 is not None
        and np is not None
        and Image is not None
    )

    if use_cnn:
        thr = float(spont_breath_meta.get("threshold", 0.5)) if spont_breath_meta else 0.5
        for x, y, w, h in coords_list:
            crop = img[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            xt = spont_breath_transform(pil).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(spont_breath_model(xt)).item()
            ok, _ = guard_ok(crop)
            if prob >= thr and ok:
                return True
        return False

    # Fallback heuristic
    is_numpy = np is not None and isinstance(img, np.ndarray)

    for x, y, w, h in coords_list:
        if is_numpy:
            crop = img[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            row = crop[h // 2]
        else:
            crop_rows = img[y:y + h]
            if not crop_rows:
                continue
            crop = [row[x:x + w] for row in crop_rows]
            row = crop[len(crop) // 2]

        bright = 0
        for pixel in row:
            if isinstance(pixel, (list, tuple)) or (np is not None and isinstance(pixel, np.ndarray)):
                gray = 0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2]
            else:
                gray = pixel
            if gray >= 200:
                bright += 1
        if bright >= 0.4 * w:
            return True
    return False


_ROI_READER_OVERRIDES = {
    "Tskin": read_temp_roi,
    "Trect": read_temp_roi,
    "I_E": read_ie_roi,
    "VentMode": read_mode_roi,
}


def _ensure_image(image_or_path):
    if np is not None and isinstance(image_or_path, np.ndarray):
        return image_or_path

    if isinstance(image_or_path, (list, tuple)):
        return image_or_path

    path = Path(image_or_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_or_path}")

    if cv2 is None:
        message = (
            "[ERROR] OpenCV (cv2) が利用できないため画像を読み込めません。"
            " 'pip install opencv-python' を実行してから再度お試しください。"
        )
        print(message)
        raise RuntimeError(message)

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_or_path}")
    return img


def ocr_vitals_from_image(image_or_path):
    """Extract vital signs from ``image_or_path`` using OCR.

    The function expects the global ROI definitions (``vital_crop``,
    ``BP_COMBINED_COORD`` etc.) to be populated beforehand via
    :func:`select_display_and_bed`.  When EasyOCR or other heavy dependencies
    are unavailable the function still returns a dictionary with the expected
    keys, defaulting to empty strings.
    """

    if cv2 is None:
        message = (
            "[ERROR] OpenCV (cv2) が利用できないため画像のOCRを実行できません。"
            " 'pip install opencv-python' を実行してから再度お試しください。"
        )
        print(message)
        raise RuntimeError(message)

    img = _ensure_image(image_or_path)

    results: dict[str, str] = {}

    if BP_COMBINED_COORD and BP_COMBINED_COORD[2] > 0 and BP_COMBINED_COORD[3] > 0:
        bp_roi = crop_image(img, BP_COMBINED_COORD)
        bp_text, sbp, dbp, map_val = read_bp_roi(bp_roi)
        if bp_text:
            results["BP"] = bp_text
        results["SBP"] = sbp
        results["DBP"] = dbp
        results["MAP"] = map_val

    for name, coord in vital_crop.items():
        if not coord or coord[2] <= 0 or coord[3] <= 0:
            continue
        roi = crop_image(img, coord)
        reader = _ROI_READER_OVERRIDES.get(name)
        if reader is not None:
            value = reader(roi)
        else:
            value = read_num_roi(roi, allow_dot=True)
        results[name] = value

    if CVP_COORDS and CVP_COORDS[2] > 0 and CVP_COORDS[3] > 0:
        cvp_roi = crop_image(img, CVP_COORDS)
        results["CVP"] = read_cvp_roi(cvp_roi)

    apply_pressure_support_split(results)

    if SPONT_BREATH_COORDS:
        spont = detect_spontaneous_breath(img, SPONT_BREATH_COORDS)
        results["SpontaneousBreath"] = "1" if spont else "0"

    for key in ALL_COLUMNS:
        results.setdefault(key, "")

    return results


def save_vitals_to_csv(vitals_dict, csv_path):
    """Append ``vitals_dict`` to ``csv_path`` while preserving extra columns.

    Existing columns in the CSV that are not part of ``ALL_COLUMNS`` (for
    example drug doses logged via :mod:`drug_panel`) are carried forward using
    the most recent values so that the latest row always reflects the current
    state.
    """

    # Start with the standard vital signs
    row = {k: vitals_dict.get(k, '') for k in ALL_COLUMNS}
    ts = vitals_dict.get("timestamp")
    row["timestamp"] = ts if ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Include any additional columns provided in ``vitals_dict`` such as
    # drug doses or gas measurements. They will be added to the CSV header
    # if not already present.
    for k, v in vitals_dict.items():
        if k not in row:
            row[k] = v

    try:
        tmp_path = f"{csv_path}.tmp"
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)

            # Identify extra columns (e.g., drug doses) that are already in the
            # CSV but not included in the current ``row``. For these columns we
            # carry forward the most recent values so that the latest row always
            # represents the current state.
            extra_cols = [
                c
                for c in fieldnames
                if c not in ["timestamp"] + ALL_COLUMNS
                and c not in row
                and c not in NON_PERSISTENT_COLUMNS
            ]
            if rows and extra_cols:
                last = rows[-1]
                for c in extra_cols:
                    row[c] = last.get(c, '')
            fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))

            with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                writer.writerow(row)
            os.replace(tmp_path, csv_path)
        else:
            # ``row`` may contain dynamically discovered keys (for example
            # SpontaneousBreath or drug doses).  When the CSV does not yet
            # exist we must include those columns up front; otherwise
            # ``DictWriter`` would raise ``ValueError: dict contains fields not
            # in fieldnames`` and the vitals would be dropped silently.  The
            # ``dict.fromkeys`` trick preserves order while removing duplicates.
            fieldnames = list(
                dict.fromkeys(["timestamp"] + ALL_COLUMNS + list(row.keys()))
            )
            with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            os.replace(tmp_path, csv_path)
    except Exception as e:  # pragma: no cover - best effort logging
        print(f"[WARN] CSV書き込み失敗: {e}")

# =========================
# 親Z:\image → 今日 or 最新日付フォルダ 追従
# =========================

def pick_today_or_latest(base: Path) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    today_dir = base / today
    if today_dir.is_dir():
        return today_dir
    dated = [p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)]
    return max(dated) if dated else base

# =========================
# メイン
# =========================

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Display available 8-screen bed coordinate keys
    print(BED_COORDS_8.keys())

    try:
        spont_breath_model_path = resolve_path(
            args.spont_breath_model,
            "SPONT_BREATH_MODEL_PATH",
            config,
            "SPONT_BREATH_MODEL_PATH",
            candidates=DEFAULT_SPONT_BREATH_MODEL_CANDIDATES,
            must_exist=True,
        )
    except ValueError:
        spont_breath_model_path = None
    try:
        spont_breath_meta_path = resolve_path(
            args.spont_breath_meta,
            "SPONT_BREATH_META_PATH",
            config,
            "SPONT_BREATH_META_PATH",
            candidates=DEFAULT_SPONT_BREATH_META_CANDIDATES,
            must_exist=True,
        )
    except ValueError:
        spont_breath_meta_path = None
    image_folder = resolve_path(
        args.image_folder,
        "IMAGE_FOLDER",
        config,
        "IMAGE_FOLDER",
        candidates=DEFAULT_IMAGE_BASE_CANDIDATES,
        must_exist=False,
    )
    vitals_base_dir = resolve_path(
        args.vitals_base,
        "VITALS_BASE_DIR",
        config,
        "VITALS_BASE_DIR",
        candidates=DEFAULT_VITALS_BASE_CANDIDATES,
        must_exist=False,
    )

    print(f"[PATH] SPONT_BREATH_MODEL_PATH = {spont_breath_model_path}")
    print(f"[PATH] SPONT_BREATH_META_PATH = {spont_breath_meta_path}")
    print(f"[PATH] IMAGE_FOLDER(base) = {image_folder}")
    print(f"[PATH] VITALS_BASE_DIR = {vitals_base_dir}")

    init_resources(spont_breath_model_path, spont_breath_meta_path)

    beds_override = None
    if args.beds:
        beds_override = [b.strip() for b in args.beds.split(",") if b.strip()]
    else:
        env_beds = os.getenv("VALID_BEDS")
        if env_beds:
            beds_override = [b.strip() for b in env_beds.split(",") if b.strip()]

    # ==== 表示モード & ベッド選択 ====
    display_mode, VITALS_PATH, bed_num = select_display_and_bed(
        vitals_base_dir, beds_override
    )
    print(f"選択された画面分割: {display_mode}")
    print(f"選択されたベッド番号: {bed_num}")
    print(f"保存先CSV: {VITALS_PATH}")

    coords = (BED_COORDS_4 if display_mode == "4" else BED_COORDS_8)[bed_num]
    BP_COMBINED_COORD = coords["BP_COMBINED_COORD"]
    CVP_COORDS = coords["CVP_COORDS"]
    vital_crop = coords["vital_crop"]
    SPONT_BREATH_COORDS = coords.get("SPONT_BREATH_COORDS", [])

    # ==== 画像フォルダの実体化 ====
    image_folder = Path(image_folder)
    image_folder.mkdir(parents=True, exist_ok=True)

    # 親(Z:\\image)を渡された場合は今日 or 最新日付フォルダに自動で降りる
    if image_folder.is_dir():
        has_dated_subdir = any(p.is_dir() and re.fullmatch(r"\d{8}", p.name) for p in image_folder.iterdir())
        if has_dated_subdir:
            chosen = pick_today_or_latest(image_folder)
            print(f"画像フォルダ自動選択: {chosen}")
            image_folder = chosen

    if cv2 is None:
        print(
            "[ERROR] OpenCV (cv2) がインストールされていないため自動OCRを開始できません。"
            " 'pip install opencv-python' を実行してください。"
        )
        sys.exit(1)

    if easyocr_reader is None:
        print(
            "[ERROR] EasyOCR がインストールされていないため自動OCRを開始できません。"
            " 'pip install easyocr' を実行してから再度お試しください。"
        )
        sys.exit(1)

    print("自動OCR＆CSV保存ループを開始します（Ctrl+Cで停止）")
    try:
        while True:
            files = list(image_folder.glob("*.png"))
            if not files:
                print("画像が見つかりませんでした。")
            else:
                latest_image = max(files, key=lambda p: p.stat().st_mtime)
                vitals = ocr_vitals_from_image(str(latest_image))
                save_vitals_to_csv(vitals, VITALS_PATH)
                print(f"{datetime.now()} 画像:{latest_image.name} のバイタルを保存しました")
            time.sleep(60)
    except KeyboardInterrupt:
        print("中断されました。")
