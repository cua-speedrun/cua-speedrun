"""init.py for the Qwen3-VL submission. Untimed, exits 0 when ready.

Installs vLLM and Pillow, downloads the model, and starts the server before
task timing begins. When restarted from a filesystem snapshot, it reuses the
cached model files.

The starter pins its complete direct model-runtime stack. Set VLLM_MODEL to a
vision-capable checkpoint supported by that stack. Changing the runtime is a
submission change and should be made by editing these exact version pins,
never by resolving a moving nightly or latest release during an evaluation.

agent.py drives the desktop with the osworld-aligned computer_use tool-call
design ported from gym-anything.
"""

import os
import importlib.metadata
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
STARTUP_TIMEOUT_SEC = 2400
DEFAULT_LOG = (
    "/root/vllm.log"
    if os.geteuid() == 0
    else os.path.join(tempfile.gettempdir(), "cua-speedrun-vllm.log")
)
VLLM_LOG_PATH = os.environ.get("VLLM_LOG_PATH", DEFAULT_LOG)

PINNED_RUNTIME = {
    "Pillow": "12.3.0",
    "fastapi-cloud-cli": "0.22.1",
    "vllm": "0.25.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "torchaudio": "2.11.0",
    "transformers": "5.13.1",
}
LOCAL_CU129_RUNTIME = {
    **PINNED_RUNTIME,
    "vllm": "0.25.0+cu129",
    "torch": "2.11.0+cu129",
    "torchvision": "0.26.0+cu129",
    "torchaudio": "2.11.0+cu129",
    # vLLM imports TorchCodec even though this screenshot-only agent does not
    # decode video. Use its CPU wheel so startup does not acquire a second,
    # unrelated CUDA/FFmpeg ABI requirement.
    "torchcodec": "0.11.1+cpu",
}
LOCAL_CU129_KEY = "linux-x86_64-py3.10"
UV_VERSION = "0.10.2"
VLLM_CU129_INDEX = "https://wheels.vllm.ai/0.25.0/cu129"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def server_up() -> bool:
    try:
        response = requests.get(
            f"http://127.0.0.1:{PORT}/v1/models", timeout=2
        )
        response.raise_for_status()
        return bool(response.json().get("data"))
    except (requests.RequestException, ValueError, AttributeError):
        return False


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def _runtime_key() -> str:
    return (
        f"{sys.platform}-{platform.machine().lower()}-"
        f"py{sys.version_info.major}.{sys.version_info.minor}"
    )


def _install_runtime(runtime_key: str, requirements: list[str]) -> None:
    if runtime_key != LOCAL_CU129_KEY:
        _pip(*requirements)
        return

    # vLLM's generic PyPI wheel now targets CUDA 13. uv selects the matching
    # PyTorch cu129 index, while the explicit vLLM index selects its compiled
    # cu129 extension wheel. Pin uv so the submitted installer is explicit.
    _pip(f"uv=={UV_VERSION}")
    uv = Path(sys.executable).with_name("uv")
    cpu_only = [item for item in requirements if item.startswith("torchcodec==")]
    cuda_runtime = [item for item in requirements if item not in cpu_only]
    subprocess.check_call([
        str(uv), "pip", "install",
        "--python", sys.executable,
        "--torch-backend=cu129",
        "--extra-index-url", VLLM_CU129_INDEX,
        "--index-strategy", "unsafe-best-match",
        *cuda_runtime,
    ])
    subprocess.check_call([
        str(uv), "pip", "install",
        "--python", sys.executable,
        "--no-deps",
        "--extra-index-url", PYTORCH_CPU_INDEX,
        "--index-strategy", "unsafe-best-match",
        *cpu_only,
    ])


def ensure_deps() -> None:
    runtime_key = _runtime_key()
    expected_runtime = (
        LOCAL_CU129_RUNTIME
        if runtime_key == LOCAL_CU129_KEY
        else PINNED_RUNTIME
    )
    installed = {}
    for package in expected_runtime:
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
    if installed != expected_runtime:
        requirements = [
            f"{name}=={version}" for name, version in expected_runtime.items()
        ]
        print(f"installing pinned runtime: {requirements}", flush=True)
        _install_runtime(runtime_key, requirements)

    resolved = {
        package: importlib.metadata.version(package)
        for package in expected_runtime
    }
    if resolved != expected_runtime:
        raise RuntimeError(
            f"model runtime did not install requested versions: {resolved!r}"
        )
    # vLLM imports torchcodec while loading its media utilities. The local
    # runtime deliberately resolves the CPU build above.
    import vllm  # noqa: F401
    print(f"pinned runtime ready: {resolved}", flush=True)


def model_cached() -> bool:
    # Must resolve the cache the same way huggingface_hub does, or this check
    # diverges from where vllm actually looks: a model in the default cache
    # with HF_HOME pointing elsewhere would flip on HF_HUB_OFFLINE and strand
    # vllm offline against an empty cache.
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    return os.path.isdir(os.path.join(hub, "models--" + MODEL.replace("/", "--")))


def runtime_cache_env() -> dict[str, str]:
    """Keep generated compiler/runtime caches off quota-limited home dirs."""
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.environ.get("TMPDIR", tempfile.gettempdir()), "cua-speedrun-cache"
    )
    paths = {
        "XDG_CACHE_HOME": cache_root,
        "UV_CACHE_DIR": os.environ.get(
            "UV_CACHE_DIR", os.path.join(cache_root, "uv")
        ),
        "VLLM_CACHE_ROOT": os.path.join(cache_root, "vllm"),
        # FlashInfer derives its cache from this base; unlike vLLM it does not
        # currently honor XDG_CACHE_HOME.
        "FLASHINFER_WORKSPACE_BASE": cache_root,
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_root, "torchinductor"),
        "TORCH_EXTENSIONS_DIR": os.path.join(cache_root, "torch-extensions"),
        "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
        "CUDA_CACHE_PATH": os.path.join(cache_root, "cuda"),
    }
    if not os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
        # Changing XDG_CACHE_HOME also changes huggingface_hub's implicit
        # cache. Keep an existing model in the conventional cache visible to
        # vLLM; on a fresh evaluator, put the first download in scratch.
        default_hf_home = os.path.expanduser("~/.cache/huggingface")
        default_model = os.path.join(
            default_hf_home, "hub", "models--" + MODEL.replace("/", "--")
        )
        paths["HF_HOME"] = (
            default_hf_home
            if os.path.isdir(default_model)
            else os.path.join(cache_root, "huggingface")
        )
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def main() -> None:
    remote = os.environ.get("VLLM_URL")
    if remote:
        # Validate the external endpoint before task timing begins.
        resp = requests.get(f"{remote.rstrip('/')}/v1/models", timeout=15)
        resp.raise_for_status()
        print(f"remote vllm ready at {remote}: {resp.json()['data'][0]['id']}")
        return
    if server_up():
        print("vllm already serving")
        return

    # Dependency installation happens before the model server environment is
    # constructed, so apply the cache paths now. In particular, uv must not
    # fall back to a quota-limited home directory on local evaluators.
    os.environ.update(runtime_cache_env())
    ensure_deps()

    env = dict(
        os.environ,
        **runtime_cache_env(),
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
    )
    env.pop("VLLM_LOG_PATH", None)
    if model_cached():
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        print("model found in cache, starting vllm offline")

    print(f"starting vllm server for {MODEL}...", flush=True)
    server = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--port", str(PORT),
            "--max-model-len", "16384",
        ],
        stdout=open(VLLM_LOG_PATH, "w"),
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    pid_file = os.environ.get("CS_SERVER_PID_FILE")
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as stream:
            stream.write(str(server.pid))
    tail = None
    if os.environ.get("CS_LOCAL_EXECUTION") != "1":
        tail = subprocess.Popen(["tail", "-n", "+1", "-F", VLLM_LOG_PATH],
                                start_new_session=True)

    deadline = time.time() + STARTUP_TIMEOUT_SEC
    while time.time() < deadline:
        if server.poll() is not None:
            time.sleep(1)
            if tail is not None:
                tail.terminate()
            sys.exit(f"vllm exited early with code {server.returncode}")
        if server_up():
            time.sleep(1)
            if tail is not None:
                tail.terminate()
            if "HF_HUB_OFFLINE" not in env:
                # Complete the hub cache so offline snapshot restarts can
                # resolve every file required by huggingface_hub.
                from huggingface_hub import snapshot_download
                snapshot_download(MODEL)
                print("hub cache completed for offline restarts", flush=True)
            print("vllm is up and ready", flush=True)
            return
        time.sleep(5)
    if tail is not None:
        tail.terminate()
    sys.exit("vllm did not become ready in time")


if __name__ == "__main__":
    main()
