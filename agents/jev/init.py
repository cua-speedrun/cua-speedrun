"""Untimed setup for the Jev agent: the screen parser and a key check.

Jev reads text only, so agent.py turns every screenshot into text and icon
elements first. The parser is Cua's cua-som: OmniParser's icon detector and
EasyOCR. This script installs it, starts it as a local server on port 8765
with its models loaded and warmed up, and checks the OpenRouter key with one
small request, all before the task clock starts.

`python init.py serve` runs the server itself.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import requests


PORT = int(os.environ.get("JEV_PARSER_PORT", "8765"))
JEV_URL = os.environ.get("JEV_URL", "https://openrouter.ai/api/v1/systemone")
MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
STARTUP_TIMEOUT_SEC = 1800
LOG_PATH = os.path.join(tempfile.gettempdir(), "cua-speedrun-jev-parser.log")
# cua-som 0.1.3 is the newest release for Python 3.11; 0.1.4 changes only formatting.
PACKAGES = {
    "cua-som": "0.1.3",
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "ultralytics": "8.4.171",
    "easyocr": "1.7.2",
    "huggingface-hub": "2.1.1",
    "numpy": "2.4.6",
    "Pillow": "12.3.0",
}
OPENCV = "opencv-python-headless==5.0.0.93"


def read_text(parser, image) -> list[dict]:
    """cua-som's OCR with its own reader and thresholds, reading every line in one GPU batch."""
    import numpy as np

    parser.ocr._ensure_reader()
    width, height = image.size
    texts = []
    for box, text, confidence in parser.ocr.reader.readtext(
            np.array(image), paragraph=False, text_threshold=0.5, batch_size=32):
        if float(confidence) < 0.5:
            continue
        xs, ys = [float(point[0]) for point in box], [float(point[1]) for point in box]
        texts.append({"text": text, "box": [min(xs) / width, min(ys) / height,
                                            max(xs) / width, max(ys) / height]})
    return texts


def parse(parser, image) -> dict:
    """Text lines and icons with boxes normalized to 0..1, as agent.py reads them."""
    from concurrent.futures import ThreadPoolExecutor

    def detect() -> tuple[list[dict], float]:
        started = time.monotonic()
        icons = parser.detector.detect_icons(image=image, box_threshold=0.3, iou_threshold=0.1)
        return icons, time.monotonic() - started

    # The detector and OCR are independent, so the detector runs while OCR reads.
    with ThreadPoolExecutor(max_workers=1) as pool:
        detection = pool.submit(detect)
        started = time.monotonic()
        texts = read_text(parser, image)
        ocr_sec = time.monotonic() - started
        icons, detect_sec = detection.result()

    def covers(box, text) -> bool:
        x = (text["box"][0] + text["box"][2]) / 2
        y = (text["box"][1] + text["box"][3]) / 2
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]

    # As cua-som does, an icon box around a text line is that text's control.
    return {
        "texts": texts,
        "icons": [{"box": i["bbox"]} for i in icons if not any(covers(i["bbox"], t) for t in texts)],
        "ocr_sec": ocr_sec, "detect_sec": detect_sec,
    }


def serve() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from PIL import Image
    from som import OmniParser

    parser = OmniParser()
    # Load and run every model once so the first task step pays no setup cost.
    parse(parser, Image.new("RGB", (1920, 1080), "white"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            started = time.monotonic()
            result = parse(parser, Image.open(io.BytesIO(body)).convert("RGB"))
            result["seconds"] = time.monotonic() - started
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    # One request at a time on the main thread, so the models never run
    # concurrently and the OCR timeout, which uses a signal, applies.
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


def healthy() -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def install() -> None:
    for name, version in PACKAGES.items():
        try:
            if importlib.metadata.version(name) != version:
                break
        except importlib.metadata.PackageNotFoundError:
            break
    else:
        return
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    subprocess.run([*pip, *[f"{n}=={v}" for n, v in PACKAGES.items()]], check=True)
    # Ultralytics pulls in the desktop OpenCV build, which needs libGL. The
    # agent image has none, so keep only the headless build.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"], check=True)
    subprocess.run([*pip, "--force-reinstall", "--no-deps", OPENCV], check=True)


def check_key() -> None:
    response = requests.post(JEV_URL, timeout=60, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
    }, json={"model": MODEL, "state": "The setup check is running.",
             "questions": {"ready": {"type": "noul", "instructions": "Is a setup check running?"}}})
    if response.status_code != 200:
        raise SystemExit(f"OpenRouter key check failed: {response.status_code} {response.text[:200]}")


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is required")
    check_key()
    install()
    if not healthy():
        with open(LOG_PATH, "ab") as log:
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve"],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + STARTUP_TIMEOUT_SEC
        while not healthy():
            if time.monotonic() > deadline:
                raise SystemExit(f"screen parser did not start; see {LOG_PATH}")
            time.sleep(2)
    print(f"Jev agent ready: {MODEL}, screen parser on port {PORT}", flush=True)


if __name__ == "__main__":
    serve() if sys.argv[1:] == ["serve"] else main()
