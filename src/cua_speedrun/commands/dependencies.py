"""Import probes shared by install and doctor."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from cua_speedrun import __version__
from cua_speedrun.local_runtime import LOCAL_AGENT_PYTHON_VERSION

from .paths import InstallationPaths


UV_VERSION = "0.10.2"


CORE_RUNTIME_IMPORTS = (
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "jinja2",
    "multipart",
    "itsdangerous",
    "gym_anything",
    "modal",
)

OSWORLD_RUNTIME_IMPORTS = (
    "bs4",
    "borb",
    "chardet",
    "cssselect",
    "easyocr",
    "fastdtw",
    "formulas",
    "gdown",
    "imagehash",
    "librosa",
    "lxml",
    "mutagen",
    "numpy",
    "odf",
    "cv2",
    "openpyxl",
    "pandas",
    "paramiko",
    "pdfplumber",
    "PIL",
    "playwright",
    "acoustid",
    "pydrive",
    "fitz",
    "PyPDF2",
    "pygame",
    "docx",
    "pptx",
    "pytz",
    "rapidfuzz",
    "skimage",
    "scipy",
    "tldextract",
    "xmltodict",
)


def missing_imports(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _runtime_install_is_current(paths: InstallationPaths) -> bool:
    if missing_imports(CORE_RUNTIME_IMPORTS + OSWORLD_RUNTIME_IMPORTS):
        return False
    try:
        if importlib.metadata.version("uv") != UV_VERSION:
            return False
    except importlib.metadata.PackageNotFoundError:
        return False
    try:
        record = json.loads(paths.install_record.read_text())
    except (OSError, ValueError):
        return False
    return record.get("cua_speedrun_version") == __version__


def install_runtime_dependencies(paths: InstallationPaths) -> None:
    if _runtime_install_is_current(paths):
        print("[ok] Python runtime dependencies")
        return
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"], check=True
        )

    extras = ("gym", "modal", "osworld", "platform")
    try:
        declared = importlib.metadata.requires("cua-speedrun") or []
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "install cua-speedrun with pip before running cua-speedrun install"
        ) from exc
    requirements: set[str] = set()
    for value in declared:
        requirement, separator, marker = value.partition(";")
        if not separator:
            continue
        marker = marker.strip().lower()
        if any(
            f"extra == '{extra}'" in marker or f'extra == "{extra}"' in marker
            for extra in extras
        ):
            requirements.add(requirement.strip())
    if not requirements:
        raise RuntimeError("the installed package does not declare runtime extras")
    print("[install] Python runtime dependencies")
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(paths.cache / "pip")
    env["UV_CACHE_DIR"] = str(paths.cache / "uv")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *sorted(requirements, key=str.lower),
        ],
        check=True,
        env=env,
    )


def _python_version(python: Path) -> str | None:
    if not python.is_file():
        return None
    result = subprocess.run(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _uv_command() -> Path:
    name = "uv.exe" if os.name == "nt" else "uv"
    beside_python = Path(sys.executable).with_name(name)
    if beside_python.is_file():
        return beside_python
    discovered = shutil.which("uv")
    if discovered:
        return Path(discovered)
    raise RuntimeError(
        "the installed runtime is missing uv; rerun cua-speedrun install"
    )


def install_local_agent_python(paths: InstallationPaths) -> None:
    """Provision the fixed Python used by local submission sandboxes.

    The dashboard may itself run under any supported Python. Submission code
    must not silently inherit that choice: it is a versioned execution
    contract and has to be identical across installations.
    """
    observed = _python_version(paths.agent_python)
    if observed and observed.startswith(f"{LOCAL_AGENT_PYTHON_VERSION}."):
        print(f"[ok] Local agent Python: {paths.agent_python} ({observed})")
        return

    print(f"[install] Local agent Python {LOCAL_AGENT_PYTHON_VERSION}")
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(paths.cache / "uv")
    env["UV_PYTHON_INSTALL_DIR"] = str(paths.runtime / "python")
    subprocess.run(
        [
            str(_uv_command()),
            "venv",
            "--clear",
            "--seed",
            "--managed-python",
            "--python",
            LOCAL_AGENT_PYTHON_VERSION,
            str(paths.agent_runtime),
        ],
        check=True,
        env=env,
    )
    observed = _python_version(paths.agent_python)
    if not observed or not observed.startswith(f"{LOCAL_AGENT_PYTHON_VERSION}."):
        raise RuntimeError(
            "local agent Python provisioning produced an unexpected runtime: "
            f"{observed or 'unavailable'} at {paths.agent_python}"
        )
    print(f"[ok] Local agent Python: {paths.agent_python} ({observed})")


__all__ = [
    "CORE_RUNTIME_IMPORTS",
    "LOCAL_AGENT_PYTHON_VERSION",
    "OSWORLD_RUNTIME_IMPORTS",
    "install_local_agent_python",
    "install_runtime_dependencies",
    "missing_imports",
]
