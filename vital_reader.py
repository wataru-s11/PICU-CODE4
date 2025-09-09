import time
import os
import csv
from datetime import datetime
from pathlib import Path
import argparse
import json
from typing import Optional
try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None
import re

# Optional heavy dependencies -------------------------------------------------
try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional dependency
    from google.cloud import vision  # type: ignore
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover
    vision = None
    service_account = None

try:  # pragma: no cover - optional dependency
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

from bed_coords import BED_COORDS_8
from bed_coords_4 import BED_COORDS_4

cvp_model = None
client = None
# Optional model and metadata for spontaneous-breathing detection (may be None)
spont_breath_model = None
spont_breath_meta = None
spont_breath_transform = None

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
DEFAULT_CVP_MODEL_CANDIDATES = [
    r"C:\\Users\\sakai\\OneDrive\\Desktop\\BOT\\CVP2\\cvp_model.keras",
]
DEFAULT_SA_JSON_CANDIDATES = [
    "ocr-project-1-458704-e25577dc4ea2.json",
    str(Path(__file__).with_name("ocr-project-1-458704-e25577dc4ea2.json")),
    r"C:\\Users\\sakai\\OneDrive\\Desktop\\BOT\\ocr-project-1-458704-e25577dc4ea2.json",
]
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
    model_path: Path,
    service_account_file: Path,
    spont_breath_model_path: Optional[Path] = None,
    spont_breath_meta_path: Optional[Path] = None,
):
    """Load optional heavy resources such as ML models and the Vision client."""

    global cvp_model, client, spont_breath_model, spont_breath_meta, spont_breath_transform
    try:
        print(f"Loading CVP model from: {model_path}")
        cvp_model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        raise RuntimeError(f"CVPモデル読み込み失敗: {model_path} -> {e}")
    try:
        credentials = service_account.Credentials.from_service_account_file(str(service_account_file))
        client = vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        raise RuntimeError(f"Google Vision初期化失敗: {service_account_file} -> {e}")



# =========================
# 引数
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvp-model", help="Path to CVP model (.keras)")
    parser.add_argument(
        "--spont-breath-model",
        help="Path to spontaneous-breathing model (.keras)",
    )
    parser.add_argument(
        "--spont-breath-meta",
        help="Path to spontaneous-breathing model metadata (JSON)",
    )
    parser.add_argument("--service-account-file", help="Path to Google Cloud service account JSON")
    parser.add_argument("--image-folder", help="Folder containing monitor images (親Z:\\imageでもOK)")
    parser.add_argument("--vitals-base", help="Folder to store CSVs (親フォルダ)。未指定なら自動推定")
    parser.add_argument("--config", help="Path to config JSON file")
    return parser.parse_args()

# =========================
# CVP 予測
# =========================

try:
    with open(Path(__file__).with_name("class_indices.json"), "r", encoding="utf-8") as f:
        class_indices = json.load(f)
except FileNotFoundError:  # pragma: no cover - fallback for tests
    class_indices = {str(i): i for i in range(16)}
index_to_label = {v: k for k, v in class_indices.items()}

def predict_cvp_from_image(img):
    """Return the predicted CVP value from a cropped image.

    The function resizes the image to the model's expected size and applies
    light-weight preprocessing to improve digit recognition under various
    lighting conditions.  When the model has only a single channel, the image
    is converted to grayscale.  For three-channel models, the grayscale output
    is duplicated across RGB channels so that the preprocessing remains
    effective regardless of the original model configuration.
    """

    input_shape = getattr(cvp_model, "input_shape", None)
    if not input_shape or len(input_shape) < 4:
        raise RuntimeError("cvp_model must have 4D input shape")

    target_h, target_w, channels = input_shape[1], input_shape[2], input_shape[3]
    img_resized = cv2.resize(img, (target_w, target_h))

    # ---- Preprocessing ----------------------------------------------------
    # Convert to grayscale for consistent preprocessing
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if img_resized.ndim == 3 else img_resized
    # Reduce noise and enhance contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Binarize to highlight digits
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if channels == 1:
        img_processed = bin_img[..., np.newaxis]
    else:
        img_processed = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    # ----------------------------------------------------------------------
    img_norm = img_processed / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    pred = cvp_model.predict(img_input)

    pred_index = np.argmax(pred)
    confidence = pred[0][pred_index]
    print(f"予測インデックス: {pred_index}, 信頼度: {confidence:.2f}")
    print(f"index_to_label[{pred_index}] = {index_to_label.get(pred_index, '未定義')}")
    label = index_to_label.get(pred_index, "")
    if confidence < 0.8:
        return "na"
    return label

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

def create_empty_vitals_csv(path):
    if not os.path.exists(path):
        import pandas as pd
        df = pd.DataFrame(columns=VITAL_COLUMNS)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 空のバイタルCSVを作成: {path}")

def select_display_and_bed(vitals_base_dir: Path):
    """画面分割(4 or 8)とベッド番号を聞いてCSVパスとともに返す"""
    import tkinter as tk
    from tkinter import simpledialog, messagebox

    today_dir = vitals_base_dir / datetime.now().strftime("%Y%m%d")
    today_dir.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.withdraw()

    while True:
        display = simpledialog.askstring(
            "表示選択", "画面分割を入力してください（4 or 8）:", parent=root
        )
        if display in ("4", "8"):
            break
        messagebox.showerror("エラー", "4または8を入力してください。", parent=root)

    valid_beds = ["1", "2", "3", "4"] if display == "4" else ["2", "3", "4", "5"]
    while True:
        bed_choice = simpledialog.askstring(
            "ベッド選択",
            f"ベッド番号を入力してください（{min(valid_beds)}～{max(valid_beds)}):",
            parent=root,
        )
        if bed_choice in valid_beds:
            selected_path = today_dir / f"vitals_history_{bed_choice}.csv"
            create_empty_vitals_csv(str(selected_path))
            root.destroy()
            return display, str(selected_path), int(bed_choice)
        messagebox.showerror(
            "エラー",
            f"{min(valid_beds)}～{max(valid_beds)}のいずれかの数字を入力してください。",
            parent=root,
        )

# =========================
# 画像処理・OCR
# =========================

def contrast_brightness(image, contrast=2.0, brightness=30):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def crop_image(img, coords):
    x, y, w, h = coords
    return img[y:y+h, x:x+w]

def enhance_red_text(img):
    b, g, r = cv2.split(img)
    red_enhanced = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
    return cv2.convertScaleAbs(red_enhanced, alpha=2, beta=0)

def ocr_google_vision(img):
    _, encoded_image = cv2.imencode(".png", img)
    image = vision.Image(content=encoded_image.tobytes())
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.strip().replace("\n", "") if texts else ""

def parse_bp_map(text):
    text = text.replace(" ", "").replace("O", "0")
    match = re.search(r"(\d{2,3})[\/](\d{2,3})[\(（](\d{2,3})[\)）]?", text)
    if match:
        return match.group(1), match.group(2), match.group(3)
    match = re.search(r"(\d{4})[\(（](\d{2,3})[\)）]?", text)
    if match:
        return match.group(1)[:2], match.group(1)[2:], match.group(2)
    return None, None, None

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


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


def ocr_vitals_from_image(image_path):
    img = cv2.imread(image_path)
    results = {}
    bp_crop = crop_image(img, BP_COMBINED_COORD)
    bp_crop = cv2.resize(bp_crop, (bp_crop.shape[1]*2, bp_crop.shape[0]*2))
    bp_crop = sharpen(bp_crop)
    bp_enhanced = enhance_red_text(bp_crop)
    bp_text = ocr_google_vision(bp_enhanced)
    sbp, dbp, map_val = parse_bp_map(bp_text)
    results['SBP'] = sbp or ''
    results['DBP'] = dbp or ''
    results['MAP'] = map_val or ''
    for key, coords in vital_crop.items():
        crop = crop_image(img, coords)
        if key in ["VTi", "VTe"]:
            crop = contrast_brightness(crop)
        result = ocr_google_vision(crop)
        if key == "I_E" and result:
            result = re.sub(r"^0*([1-9]):0*([0-9]+(?:\.[0-9]+)?)$", r"\1:\2", result)
        results[key] = result
    cvp_crop = crop_image(img, CVP_COORDS)
    results['CVP'] = predict_cvp_from_image(cvp_crop)

    if detect_spontaneous_breath(img, SPONT_BREATH_COORDS):
        print("自発呼吸検出")
        results['SpontaneousBreath'] = 'detected'
    else:
        print("自発呼吸を検出しません")
        results['SpontaneousBreath'] = ''
    return results

ALL_COLUMNS = [
    "SBP", "DBP", "MAP", "HR", "SpO2", "BSR1", "BSR2", "Tskin", "Trect", "etCO2",
    "RR", "Ppeak", "Pmean", "PEEPact", "RRact", "I_E", "FiO2", "VTe", "VTi",
    "PEEPset", "VTset", "CVP", "pH", "PaCO2", "pO2", "Hct", "K", "Na", "Cl",
    "Ca", "Glu", "Lac", "tBil", "HCO3", "BE", "Alb"
]

# Columns that represent one-time events and should not be carried forward when
# appending new vital rows. Currently only the IV bolus dose of furosemide is
# treated as non-persistent so that it is logged only at the time of entry.
NON_PERSISTENT_COLUMNS = {"furosemide_mg"}

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
            fieldnames = ["timestamp"] + ALL_COLUMNS
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

    cvp_model_path = resolve_path(
        args.cvp_model,
        "CVP_MODEL_PATH",
        config,
        "CVP_MODEL_PATH",
        candidates=DEFAULT_CVP_MODEL_CANDIDATES,
        must_exist=True,
    )
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
    service_account_file = resolve_path(
        args.service_account_file,
        "SERVICE_ACCOUNT_FILE",
        config,
        "SERVICE_ACCOUNT_FILE",
        candidates=DEFAULT_SA_JSON_CANDIDATES,
        must_exist=True,
    )
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

    print(f"[PATH] CVP_MODEL_PATH = {cvp_model_path}")
    print(f"[PATH] SPONT_BREATH_MODEL_PATH = {spont_breath_model_path}")
    print(f"[PATH] SPONT_BREATH_META_PATH = {spont_breath_meta_path}")
    print(f"[PATH] SERVICE_ACCOUNT_FILE = {service_account_file}")
    print(f"[PATH] IMAGE_FOLDER(base) = {image_folder}")
    print(f"[PATH] VITALS_BASE_DIR = {vitals_base_dir}")

    init_resources(cvp_model_path, service_account_file, spont_breath_model_path, spont_breath_meta_path)

    # ==== 表示モード & ベッド選択 ====
    display_mode, VITALS_PATH, bed_num = select_display_and_bed(vitals_base_dir)
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
