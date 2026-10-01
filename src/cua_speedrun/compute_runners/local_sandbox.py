"""Apptainer filesystem caching owned by the local compute provider."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import secrets
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import cua_speedrun


LOCAL_SANDBOX_RECIPE = "apptainer-overlay-snapshot@1"
LOCAL_SANDBOX_BASE_IMAGE = (
    "docker://jrottenberg/ffmpeg@sha256:"
    "7c9ffdf589c75bcadab209fd1b93e3994ad24894f4f16f0c66843c80700c0283"
)
_TOOLCHAIN = "ziglang==0.15.2"
_ROOT = "/opt/cua"
_PYTHON = f"{_ROOT}/runtime/bin/python"
_SUBMISSION = f"{_ROOT}/submission"
_CONTROL = f"{_ROOT}/control"
_LOCK_HEARTBEAT_SEC = 5.0
_LOCK_STALE_SEC = 120.0

_IGNORED_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "CURL_CA_BUNDLE",
    "HOME",
    "HOST",
    "HOSTNAME",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "NVIDIA_VISIBLE_DEVICES",
    "OLDPWD",
    "PATH",
    "PIP_CACHE_DIR",
    "PWD",
    "PYTHONPATH",
    "REQUESTS_CA_BUNDLE",
    "ROCR_VISIBLE_DEVICES",
    "SHLVL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "UV_CACHE_DIR",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
    "_",
}
_IGNORED_PREFIXES = (
    "APPTAINER_",
    "BASH_FUNC_",
    "CONDA_",
    "CS_",
    "CUDA_",
    "GYM_ANYTHING_",
    "LMOD_",
    "MODULE",
    "NVIDIA_",
    "OSWORLD_",
    "PMI_",
    "PMIX_",
    "ROCR_",
    "SINGULARITY_",
    "SLURM_",
    "SSH_",
)

# Triton needs a C compiler. The public FFmpeg image supplies the media
# libraries and this pinned Python wheel supplies the compiler binary.
_CC = f"""#!{_PYTHON}
import os, sys
from pathlib import Path
import ziglang

args = sys.argv[1:]
dirs = [Path(args[i + 1]) for i, arg in enumerate(args[:-1]) if arg == "-L"]
dirs += [Path(arg[2:]) for arg in args if arg.startswith("-L") and len(arg) > 2]
args = [str(next((d / arg[3:] for d in dirs if (d / arg[3:]).is_file()), arg))
        if arg.startswith("-l:") else arg for arg in args]
zig = str(Path(ziglang.__file__).with_name("zig"))
os.execv(zig, [zig, "cc", *args])
"""


def local_sandbox_available() -> bool:
    return (
        platform.system() == "Linux"
        and platform.machine().lower() in {"amd64", "x86_64"}
        and shutil.which("apptainer") is not None
    )


def _stable_env(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        str(name): str(value)
        for name, value in environment.items()
        if name not in _IGNORED_ENV and not name.startswith(_IGNORED_PREFIXES)
    }


def _size(name: str, default: int) -> str:
    value = int(os.environ.get(name, default))
    if value < 1024:
        raise ValueError(f"{name} must be at least 1024 MiB")
    return str(value)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _shared_directory_lock(path: Path) -> Iterator[None]:
    """Cross-host lock for shared filesystems where flock is node-local."""
    owner_name = f"owner-{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(4)}"
    owner_path = path / owner_name
    owner_fd = -1
    stop = threading.Event()

    while True:
        try:
            path.mkdir()
        except FileExistsError:
            try:
                leases = list(path.glob("owner-*"))
                modified = max(
                    [path.stat().st_mtime, *(lease.stat().st_mtime for lease in leases)]
                )
            except (FileNotFoundError, OSError):
                continue
            if time.time() - modified > _LOCK_STALE_SEC:
                stale = path.with_name(f".{path.name}.stale-{secrets.token_hex(4)}")
                try:
                    path.rename(stale)
                except (FileNotFoundError, OSError):
                    continue
                shutil.rmtree(stale, ignore_errors=True)
                continue
            time.sleep(1.0)
            continue
        owner_fd = os.open(owner_path, os.O_CREAT | os.O_WRONLY, 0o600)
        break

    def heartbeat() -> None:
        while not stop.wait(_LOCK_HEARTBEAT_SEC):
            try:
                os.utime(owner_fd)
            except OSError:
                return

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=_LOCK_HEARTBEAT_SEC)
        os.close(owner_fd)
        owner_path.unlink(missing_ok=True)
        try:
            path.rmdir()
        except OSError:
            pass


@dataclass
class LocalSandbox:
    apptainer: str
    name: str
    scratch: Path
    runtime_env: dict[str, str]
    agent_host_env: dict[str, str]
    runtime_overrides: dict[str, str]
    agent_python: Path
    cache_key: str
    cache_hit: bool

    def running(self) -> bool:
        result = subprocess.run(
            [self.apptainer, "instance", "list", "--json", self.name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        try:
            rows = json.loads(result.stdout).get("instances", [])
        except json.JSONDecodeError:
            return False
        return result.returncode == 0 and any(
            row.get("instance") == self.name for row in rows
        )

    def stop(self) -> None:
        subprocess.run(
            [self.apptainer, "instance", "stop", self.name],
            capture_output=True,
            timeout=30,
            check=False,
        )
        shutil.rmtree(self.scratch, ignore_errors=True)


class LocalSandboxCache:
    """Run init once into a cached overlay, then restart it per replica."""

    def __init__(
        self,
        *,
        submission_dir: Path,
        submission_fingerprint: str,
        base_python: Path,
        python_packages: Sequence[str],
        runtime_env: Mapping[str, str],
        gpu: str | None,
        run_id: str,
        init_cache_key: str | None = None,
    ) -> None:
        if not local_sandbox_available():
            raise RuntimeError(
                "isolated local compute requires Apptainer on Linux x86-64"
            )
        self.apptainer = str(Path(shutil.which("apptainer") or "").resolve())
        self.submission = Path(submission_dir).resolve()
        self.submission_fingerprint = submission_fingerprint
        self.base_python = Path(base_python).resolve()
        self.base_python_root = self.base_python.parent.parent
        self.packages = tuple(str(package) for package in python_packages)
        self.environment = _stable_env(runtime_env)
        self.gpu = gpu
        self.run_id = run_id
        self.init_cache_key = init_cache_key

        configured_home = os.environ.get("CUA_SPEEDRUN_HOME", "").strip()
        home = (
            Path(configured_home).expanduser().resolve()
            if configured_home
            else (Path.home() / ".cache/cua-speedrun").resolve()
        )
        configured_scratch = os.environ.get("CS_LOCAL_COMPUTE_SCRATCH", "").strip()
        scratch = (
            Path(configured_scratch).expanduser().resolve()
            if configured_scratch
            else Path(tempfile.gettempdir()).resolve() / getpass.getuser()
        )
        self.cache = home / "cache/compute/apptainer"
        self.apptainer_cache = home / "cache/apptainer"
        self.scratch = scratch / "cua-speedrun-compute"
        # Apptainer keeps its instance registry under the config directory,
        # which defaults to the NFS home. A wide evaluation polls instance
        # state from every node once a second, and one home export cannot
        # serve that: commands time out and get misread as lost sandboxes.
        # Instances are per-node state and belong on node-local scratch.
        self.apptainer_config = self.scratch / "apptainer-config"
        for path in (
            self.cache,
            self.apptainer_cache,
            self.scratch,
            self.apptainer_config,
        ):
            path.mkdir(parents=True, exist_ok=True)
        os.environ["APPTAINER_CONFIGDIR"] = str(self.apptainer_config)
        self.prepared: tuple[str, Path, bool] | None = None

    def _host_env(
        self, container_env: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("APPTAINERENV_")
        }
        environment["APPTAINER_CACHEDIR"] = str(self.apptainer_cache)
        environment.update({
            f"APPTAINERENV_{k}": v for k, v in (container_env or {}).items()
        })
        return environment

    def _key(self) -> str:
        if self.init_cache_key is not None:
            return hashlib.sha256(self.init_cache_key.encode()).hexdigest()
        package = Path(cua_speedrun.__file__).resolve().parent
        identity = {
            "recipe": LOCAL_SANDBOX_RECIPE,
            "base": LOCAL_SANDBOX_BASE_IMAGE,
            "apptainer": subprocess.check_output(
                [self.apptainer, "--version"], text=True, timeout=10
            ).strip(),
            "submission": self.submission_fingerprint,
            "client": hashlib.sha256(
                b"".join(
                    (package / name).read_bytes()
                    for name in ("__init__.py", "client.py")
                )
            ).hexdigest(),
            "python": hashlib.sha256(self.base_python.read_bytes()).hexdigest(),
            "packages": self.packages,
            "toolchain": _TOOLCHAIN,
            "gpu": self.gpu,
            "environment": hashlib.sha256(
                json.dumps(self.environment, sort_keys=True).encode()
            ).hexdigest(),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _harness(self, control: Path) -> Path:
        package = control / "harness/cua_speedrun"
        package.mkdir(parents=True, exist_ok=True)
        source = Path(cua_speedrun.__file__).resolve().parent
        for name in ("__init__.py", "client.py"):
            shutil.copy2(source / name, package / name)
        compiler = package.parent / "cc"
        compiler.write_text(_CC)
        compiler.chmod(0o755)
        return package.parent

    def _start(self, overlay: Path, control: Path, upper: Path | None = None) -> str:
        name = f"cs_{hashlib.sha256(self.run_id.encode()).hexdigest()[:8]}_{secrets.token_hex(4)}"
        command = [self.apptainer, "instance", "start", "--containall", "--cleanenv"]
        if self.gpu:
            command.append("--nv")
        binds = (
            f"{self.base_python_root}:{_ROOT}/base-python:ro",
            f"{self.submission}:{_ROOT}/source:ro",
            f"{self._harness(control)}:{_ROOT}/harness:ro",
            f"{control.resolve()}:{_CONTROL}",
        )
        for bind in binds:
            command.extend(("--bind", bind))
        command.extend(("--overlay", f"{overlay}:ro" if upper else str(overlay)))
        if upper:
            command.extend(("--overlay", str(upper)))
        subprocess.run(
            [*command, LOCAL_SANDBOX_BASE_IMAGE, name],
            env=self._host_env(),
            check=True,
        )
        return name

    def _exec(
        self, name: str, command: Sequence[str], environment: Mapping[str, str], stream
    ) -> None:
        result = subprocess.run(
            [
                self.apptainer,
                "exec",
                "--cleanenv",
                "--pwd",
                _SUBMISSION,
                f"instance://{name}",
                *command,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=self._host_env(environment),
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"local sandbox command exited with code {result.returncode}"
            )

    def _runtime(
        self, control: Path, device: str | None
    ) -> tuple[dict[str, str], dict[str, str]]:
        overrides = {
            "CC": f"{_ROOT}/harness/cc",
            "CS_AGENT_PYTHON": _PYTHON,
            "CS_LOCAL_EXECUTION": "1",
            "CS_SERVER_PID_FILE": f"{_CONTROL}/server.pid",
            "HF_HOME": f"{_ROOT}/cache/huggingface",
            "HOME": f"{_ROOT}/home",
            "LANG": self.environment.get("LANG", "C.UTF-8"),
            "PATH": f"{_ROOT}/harness:{_ROOT}/runtime/bin:{_ROOT}/base-python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PIP_CACHE_DIR": f"{_ROOT}/cache/pip",
            "PYTHONPATH": f"{_ROOT}/harness",
            "TMPDIR": f"{_ROOT}/tmp",
            "UV_CACHE_DIR": f"{_ROOT}/cache/uv",
            "VIRTUAL_ENV": f"{_ROOT}/runtime",
            "VLLM_LOG_PATH": f"{_CONTROL}/model-server.log",
            "VLLM_PORT": str(_port()),
            "XDG_CACHE_HOME": f"{_ROOT}/cache",
            "CUDA_VISIBLE_DEVICES": device or "",
            "NVIDIA_VISIBLE_DEVICES": device or "void",
        }
        return {**self.environment, **overrides}, overrides

    def _prepare(
        self, control: Path, log: Path, device: str | None
    ) -> tuple[str, Path, bool]:
        if self.prepared:
            return self.prepared
        key = self._key()
        directory = self.cache / "prepared" / key[:2]
        overlay = directory / f"{key}.img"
        lock = directory / f"{key}.lockdir"
        directory.mkdir(parents=True, exist_ok=True)
        with _shared_directory_lock(lock):
            if overlay.exists():
                self.prepared = (key, overlay, True)
                return self.prepared
            temporary = overlay.with_name(f".{overlay.name}.{os.getpid()}.tmp")
            name = ""
            try:
                subprocess.run(
                    [
                        self.apptainer,
                        "overlay",
                        "create",
                        "--sparse",
                        "--size",
                        _size("CS_LOCAL_PREPARED_OVERLAY_MB", 65_536),
                        "--create-dir",
                        _SUBMISSION,
                        str(temporary),
                    ],
                    check=True,
                )
                name = self._start(temporary, control)
                environment, _ = self._runtime(control, device)
                packages = shlex.join([*self.packages, _TOOLCHAIN])
                script = f"""set -eu
mkdir -p {_ROOT}/home {_ROOT}/cache {_ROOT}/tmp
cp -a {_ROOT}/source/init.py {_ROOT}/source/agent.py {_SUBMISSION}/
{_ROOT}/base-python/bin/{self.base_python.name} -m venv --clear {_ROOT}/runtime
{_PYTHON} -m pip install --disable-pip-version-check {packages}
exec {_PYTHON} {_SUBMISSION}/init.py
"""
                with log.open("a") as stream:
                    stream.write(f"local filesystem cache miss: {key}\n")
                    stream.flush()
                    self._exec(name, ["/bin/sh", "-c", script], environment, stream)
                subprocess.run([self.apptainer, "instance", "stop", name], check=True)
                name = ""
                (control / "server.pid").unlink(missing_ok=True)
                temporary.chmod(0o600)
                temporary.replace(overlay)
            finally:
                if name:
                    subprocess.run(
                        [self.apptainer, "instance", "stop", name],
                        capture_output=True,
                        check=False,
                    )
                temporary.unlink(missing_ok=True)
        self.prepared = (key, overlay, False)
        return self.prepared

    def start_replica(
        self, *, index: int, control: Path, log: Path, device: str | None
    ) -> LocalSandbox:
        control.mkdir(parents=True, exist_ok=True)
        key, overlay, hit = self._prepare(control, log, device)
        scratch = Path(
            tempfile.mkdtemp(prefix=f"{self.run_id}-{index}-", dir=self.scratch)
        )
        upper, name = scratch / "upper.img", ""
        try:
            subprocess.run(
                [
                    self.apptainer,
                    "overlay",
                    "create",
                    "--sparse",
                    "--size",
                    _size("CS_LOCAL_REPLICA_OVERLAY_MB", 8_192),
                    str(upper),
                ],
                check=True,
            )
            name = self._start(overlay, control, upper)
            environment, overrides = self._runtime(control, device)
            (control / "server.pid").unlink(missing_ok=True)
            with log.open("a") as stream:
                stream.write(
                    f"local filesystem cache {'hit' if hit else 'stored'}: {key}\n"
                )
                stream.flush()
                self._exec(
                    name, [_PYTHON, f"{_SUBMISSION}/init.py"], environment, stream
                )
            if (control / "server.pid").exists():
                overrides["VLLM_URL"] = f"http://127.0.0.1:{overrides['VLLM_PORT']}"
                environment.update(VLLM_URL=overrides["VLLM_URL"])
            wrapper = control / "agent-python"
            source = f"""#!{self.base_python}
import os, sys
env = os.environ.copy()
env["APPTAINER_CONFIGDIR"] = {str(self.apptainer_config)!r}
cmd = [{self.apptainer!r}, "exec", "--cleanenv", "--pwd", {_SUBMISSION!r},
       "instance://{name}", {_PYTHON!r}, {_SUBMISSION + "/agent.py"!r}, *sys.argv[2:]]
os.execvpe(cmd[0], cmd, env)
"""
            wrapper.write_text(source)
            wrapper.chmod(0o700)
            return LocalSandbox(
                self.apptainer,
                name,
                scratch,
                environment,
                self._host_env(environment),
                overrides,
                wrapper,
                key,
                hit,
            )
        except BaseException:
            if name:
                subprocess.run(
                    [self.apptainer, "instance", "stop", name],
                    capture_output=True,
                    check=False,
                )
            shutil.rmtree(scratch, ignore_errors=True)
            raise


__all__ = [
    "LOCAL_SANDBOX_BASE_IMAGE",
    "LOCAL_SANDBOX_RECIPE",
    "LocalSandbox",
    "LocalSandboxCache",
    "local_sandbox_available",
]
