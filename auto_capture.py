# NOTE:
#   The script is frequently executed on machines that may not have all
#   dependencies pre-installed (e.g. when run manually via ``py
#   auto_capture.py``).  Previously this resulted in a rather cryptic
#   ``ModuleNotFoundError`` for ``mss``.  To improve the UX we eagerly
#   validate the imports and provide actionable installation guidance.
try:
    from mss import mss
except ImportError as exc:  # pragma: no cover - defensive branch
    raise SystemExit(
        "Missing optional dependency 'mss'. Install it with 'pip install mss' "
        "(or 'py -m pip install mss' on Windows)."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - defensive branch
    raise SystemExit(
        "Missing optional dependency 'Pillow'. Install it with 'pip install pillow' "
        "(or 'py -m pip install pillow' on Windows)."
    ) from exc
import time
import os
from datetime import datetime
import argparse
from pathlib import Path
import json
from typing import List, Optional
import socket


# =========================
# 設定ロード
# =========================


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else Path(__file__).with_name("config.json")
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_path(
    arg_val,
    env_name: str,
    config: dict,
    key: str,
    candidates: Optional[List[str]] = None,
) -> Path:
    if arg_val:
        return Path(arg_val).expanduser()
    env_val = os.getenv(env_name)
    if env_val:
        return Path(env_val).expanduser()
    cfg_val = config.get(key)
    if cfg_val:
        return Path(cfg_val).expanduser()
    for c in candidates or []:
        if c:
            return Path(c).expanduser()
    raise ValueError(
        f"{key} is not specified. Provide via argument, env {env_name}, or config."
    )


DEFAULT_IMAGE_BASE_CANDIDATES = [r"Z:\\image"]


parser = argparse.ArgumentParser(description="Capture screenshots from a specific monitor.")
parser.add_argument("--config", help="Path to config.json")
parser.add_argument("--image-folder", help="Base directory to save images")
parser.add_argument(
    "--monitor",
    type=int,
    help="Monitor number to capture. If omitted, you will be prompted to choose.",
)
parser.add_argument("--left", type=int, help="Left coordinate for manual capture")
parser.add_argument("--top", type=int, help="Top coordinate for manual capture")
parser.add_argument("--width", type=int, help="Width of capture region")
parser.add_argument("--height", type=int, help="Height of capture region")
parser.add_argument("--interval", type=int, default=60, help="Capture interval in seconds")
parser.add_argument(
    "--name-prefix",
    help="Prefix for saved image files. Defaults to host name when omitted.",
)
args = parser.parse_args()

config = load_config(args.config)
base_dir = resolve_path(
    args.image_folder,
    "IMAGE_FOLDER",
    config,
    "IMAGE_FOLDER",
    DEFAULT_IMAGE_BASE_CANDIDATES,
)
base_dir.mkdir(parents=True, exist_ok=True)

name_prefix = (
    args.name_prefix
    or os.getenv("IMAGE_NAME_PREFIX")
    or config.get("IMAGE_NAME_PREFIX")
    or socket.gethostname()
)
name_prefix = name_prefix.replace(" ", "_") if name_prefix else "capture"

try:
    with mss() as sct:
        if None not in (args.left, args.top, args.width, args.height):
            monitor = {
                "left": args.left,
                "top": args.top,
                "width": args.width,
                "height": args.height,
            }
            print(
                f"📸 座標指定でスクリーンショット開始（{args.interval}秒おき）: {monitor}"
            )
        else:
            available_monitors = sct.monitors[1:]
            if not available_monitors:
                raise RuntimeError("利用可能なモニターが見つかりません。")

            if args.monitor is None:
                print("利用可能なモニター:")
                for i, mon in enumerate(available_monitors, start=1):
                    print(
                        f"  {i}: {mon['width']}x{mon['height']} ({mon['left']},{mon['top']})"
                    )
                max_monitor = len(available_monitors)
                while True:
                    try:
                        prompt = (
                            f"キャプチャするモニター番号を入力してください (1-{max_monitor}): "
                        )
                        monitor_number = int(input(prompt))
                        if 1 <= monitor_number <= max_monitor:
                            break
                        print("⚠️ 無効なモニター番号です。")
                    except ValueError:
                        print("⚠️ 数値を入力してください。")
            else:
                monitor_number = args.monitor
                max_monitor = len(available_monitors)
                if not 1 <= monitor_number <= max_monitor:
                    raise ValueError(
                        f"モニター番号は1から{max_monitor}の範囲で指定してください。"
                    )

            monitor = available_monitors[monitor_number - 1]
            print(
                f"📸 モニター{monitor_number}のスクリーンショット開始（{args.interval}秒おき）"
            )

        while True:
            date_dir = base_dir / datetime.now().strftime("%Y%m%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S_%f")
            filepath = date_dir / f"{name_prefix}_{timestamp}.png"

            # スクリーンショット取得・保存
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            img.save(filepath)

            print(f"✅ 保存完了: {filepath}")

            time.sleep(args.interval)

except KeyboardInterrupt:
    print("\n🛑 中断されました。終了します。")
