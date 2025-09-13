from mss import mss
from PIL import Image
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
    raise ValueError(f"{key} is not specified. Provide via argument, env {env_name}, or config.")


DEFAULT_IMAGE_BASE_CANDIDATES = [r"Z:\\image"]

# Host-specific default monitor and capture region
# (region uses left/top/width/height; full HD as placeholder)
HOST_DEFAULTS = {
    "ws-PC1": {"monitor": 1, "region": {"left": 0, "top": 0, "width": 1920, "height": 1080}},
    "Super-WS": {"monitor": 1, "region": {"left": 0, "top": 0, "width": 1920, "height": 1080}},
}

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

host_defaults = HOST_DEFAULTS.get(socket.gethostname(), {})

monitor_number = args.monitor
left, top, width, height = args.left, args.top, args.width, args.height

if monitor_number is None:
    env_mon = os.getenv("CAPTURE_MONITOR")
    if env_mon is not None:
        monitor_number = int(env_mon)
    elif "monitor" in host_defaults:
        monitor_number = host_defaults["monitor"]

if None in (left, top, width, height):
    env_vals = [
        os.getenv("CAPTURE_LEFT"),
        os.getenv("CAPTURE_TOP"),
        os.getenv("CAPTURE_WIDTH"),
        os.getenv("CAPTURE_HEIGHT"),
    ]
    if all(env_vals):
        left, top, width, height = map(int, env_vals)
    elif "region" in host_defaults:
        region = host_defaults["region"]
        left = region["left"]
        top = region["top"]
        width = region["width"]
        height = region["height"]

try:
    with mss() as sct:
        if None not in (left, top, width, height):
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
            print(
                f"📸 座標指定でスクリーンショット開始（{args.interval}秒おき）: {monitor}"
            )
        else:
            if monitor_number is None:
                print("利用可能なモニター:")
                for i, mon in enumerate(sct.monitors[1:], start=1):
                    print(
                        f"  {i}: {mon['width']}x{mon['height']} ({mon['left']},{mon['top']})"
                    )
                while True:
                    try:
                        monitor_number = int(
                            input("キャプチャするモニター番号を入力してください: ")
                        )
                        if 1 <= monitor_number < len(sct.monitors):
                            break
                        print("⚠️ 無効なモニター番号です。")
                    except ValueError:
                        print("⚠️ 数値を入力してください。")
            monitor = sct.monitors[monitor_number]
            print(
                f"📸 モニター{monitor_number}のスクリーンショット開始（{args.interval}秒おき）"
            )

        while True:
            date_dir = base_dir / datetime.now().strftime("%Y%m%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S")
            filepath = date_dir / f"{timestamp}.png"

            # スクリーンショット取得・保存
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            img.save(filepath)

            print(f"✅ 保存完了: {filepath}")

            time.sleep(args.interval)

except KeyboardInterrupt:
    print("\n🛑 中断されました。終了します。")
