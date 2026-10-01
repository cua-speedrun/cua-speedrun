"""Install the pinned Yutori SDK before the task clock starts."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys


PACKAGES = {"yutori": "0.9.29", "openai": "3.16.2", "httpx": "0.28.1", "Pillow": "12.3.0"}


def main() -> None:
    if not os.environ.get("YUTORI_API_KEY", "").strip():
        raise SystemExit("YUTORI_API_KEY is required")
    for name, version in PACKAGES.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            break
    else:
        print("Yutori n2 dependencies ready", flush=True)
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         *[f"{name}=={version}" for name, version in PACKAGES.items()]],
        check=True,
    )
    print("Yutori n2 dependencies ready", flush=True)


if __name__ == "__main__":
    main()
