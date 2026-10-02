"""Launcher for the env+gateway Modal sandbox (runs locally, on the executor).

Creates a Modal sandbox that runs the environment natively (KVM via
vm_runtime) plus cua-speedrun's gateway, exposed as an HTTPS tunnel. The
image, volume, and AVD-stack setup are lifted from gym-anything's
ModalRunner so the environment boots identically to the proven path.

Returns a handle with the sandbox, the gateway base URL (tunnel), and the
run/control tokens the executor and agent will use.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# gym-anything's proven sandbox package set.
from gym_anything.runtime.runners.modal_runner import (  # type: ignore
    _SANDBOX_APT,
    _SANDBOX_PIP,
)


def _ga_package_dir() -> Path:
    """The installed gym_anything package directory, to ship into the sandbox.
    Derived from the package itself rather than a repo-root helper, since
    upstream removed the private _repo_root symbol we used to import."""
    import gym_anything

    return Path(gym_anything.__file__).resolve().parent


_CS_PACKAGE = Path(__file__).resolve().parents[1]

# Pin the env-plane Python version independently of the launcher so evaluator
# dependencies and image cache keys stay consistent across client versions.
ENV_PLANE_PYTHON_VERSION = "3.11"


def _host_runtime(env_local_dir: Path) -> dict[str, Any]:
    """Read optional environment-owned host runtime metadata.

    Most environments use the shared Python image. Environments whose
    canonical host controller has a different Python/package contract can
    declare it beside env.json without adding a benchmark-specific launcher.
    """
    path = Path(env_local_dir) / "host-runtime.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    unknown = set(data) - {
        "python_version",
        "pyproject",
        "apt_packages",
        "forward_env",
        "native_runner",
    }
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    if data.get("native_runner") not in (None, "gym-anything"):
        raise ValueError(f"{path}: unsupported native_runner")

    python_version = data.get("python_version")
    if python_version is not None and not re.fullmatch(r"[0-9]+\.[0-9]+", str(python_version)):
        raise ValueError(f"{path}: python_version must look like '3.12'")
    pyproject = data.get("pyproject")
    if pyproject is not None:
        project_path = (Path(env_local_dir).resolve() / str(pyproject)).resolve()
        if not project_path.is_relative_to(Path(env_local_dir).resolve()):
            raise ValueError(f"{path}: pyproject escapes the environment directory")
        if not project_path.is_file():
            raise FileNotFoundError(f"{path}: pyproject does not exist: {project_path}")
    apt_packages = data.get("apt_packages") or []
    if not isinstance(apt_packages, list) or not all(
        isinstance(name, str)
        and re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", name)
        for name in apt_packages
    ):
        raise ValueError(
            f"{path}: apt_packages must be a list of Debian package names"
        )
    if len(apt_packages) != len(set(apt_packages)):
        raise ValueError(f"{path}: apt_packages must not contain duplicates")
    forward_env = data.get("forward_env") or []
    if not isinstance(forward_env, list) or not all(
        isinstance(name, str) and name for name in forward_env
    ):
        raise ValueError(f"{path}: forward_env must be a list of variable names")
    return data


# The AVD-stack extraction + ini path rewrite that ModalRunner runs before
# starting its shim; the AVD emulator needs it in place before boot.
_AVD_CACHE_SETUP = (
    "mkdir -p /root/.cache/gym-anything && "
    "tar -I zstd -xf /cache/qemu/avd_stack.tar.zst -C /root/.cache/gym-anything && "
    "AVDINI=$(ls /root/.cache/gym-anything/avd/*.ini 2>/dev/null | head -1) && "
    "OLDBASE=$(sed -n 's|^path=\\(.*\\)/avd/[^/]*\\.avd$|\\1|p' \"$AVDINI\" | head -1) && "
    "if [ -n \"$OLDBASE\" ] && [ \"$OLDBASE\" != /root/.cache/gym-anything ]; then "
    "grep -rl \"$OLDBASE\" /root/.cache/gym-anything/avd --include='*.ini' | "
    "xargs -r sed -i \"s|$OLDBASE|/root/.cache/gym-anything|g\"; "
    "mkdir -p \"$(dirname \"$OLDBASE\")\" && ln -sfn /root/.cache/gym-anything \"$OLDBASE\"; fi && "
)


@dataclass
class EnvSandbox:
    sandbox: Any
    base_url: str          # https tunnel to the gateway, no trailing path
    run_token: str
    control_token: str
    gateway_port: int
    # Boot budget depends on the env kind: a cached AVD restores in minutes;
    # the first Linux boot provisions a base qcow2 in-sandbox (up to an hour,
    # then cached on the shared volume for every later sandbox).
    boot_budget_sec: int = 900


class _SandboxGroup:
    """Terminate a controller and every desktop it created."""

    def __init__(self, sandbox, owner: str):
        self._sandbox = sandbox
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._sandbox, name)

    def terminate(self):
        import modal

        try:
            self._sandbox.terminate()
        finally:
            for child in modal.Sandbox.list(tags={"cs-parent": self._owner}):
                child.terminate()


def _detect_env_kind(env_local_dir: Path) -> str:
    """'android' or 'linux', from the env definition's base preset. Decides
    the in-sandbox runner and whether the AVD stack setup applies."""
    import json as _json

    for name in ("env.json", "env.yaml", "env.yml"):
        path = Path(env_local_dir) / name
        if not path.exists():
            continue
        if name.endswith(".json"):
            base = str(_json.loads(path.read_text()).get("base", ""))
        else:
            import yaml as _yaml
            base = str((_yaml.safe_load(path.read_text()) or {}).get("base", ""))
        return "android" if "android" in base.lower() else "linux"
    return "android"


def _environment_resources(
    env_local_dir: Path,
    env_spec: "Mapping[str, Any]",
) -> tuple[float, int]:
    """Size the sandbox around the guest resources declared by the env."""
    data: dict[str, Any] = {}
    for name in ("env.json", "env.yaml", "env.yml"):
        path = Path(env_local_dir) / name
        if not path.exists():
            continue
        if name.endswith(".json"):
            data = json.loads(path.read_text())
        else:
            import yaml

            data = yaml.safe_load(path.read_text()) or {}
        break
    resources = dict(data.get("resources") or {})
    overrides = env_spec.get("config_overrides") or {}
    if isinstance(overrides, dict) and isinstance(overrides.get("resources"), dict):
        resources.update(overrides["resources"])
    guest_cpu = float(resources.get("cpu") or 4)
    guest_mem_gb = int(resources.get("mem_gb") or 8)
    if guest_cpu <= 0 or guest_mem_gb <= 0:
        raise ValueError("environment CPU and memory resources must be positive")
    return max(guest_cpu + 1, 5), max(guest_mem_gb + 3, 12) * 1024


def _base_image(env_local_dir: Path, runtime: dict, *, native_runner: bool = False):
    import modal

    from cua_speedrun.remote import osworld_modal_base as _osworld_base

    image = (
        modal.Image.debian_slim(
            python_version=str(
                runtime.get("python_version") or ENV_PLANE_PYTHON_VERSION
            )
        )
        .apt_install(
            *_SANDBOX_APT,
            *_osworld_base.ENV_PLANE_APT,
            *(runtime.get("apt_packages") or []),
        )
        .pip_install(*_SANDBOX_PIP, "litellm")
    )
    if native_runner:
        image = image.pip_install("modal==1.5.5", "vncdotool", "google-genai", "google-generativeai")
    if runtime.get("pyproject"):
        image = image.pip_install_from_pyproject(
            str(Path(env_local_dir) / str(runtime["pyproject"]))
        )
    elif not native_runner:
        # Legacy OSWorld verifiers import desktop_env.evaluators from the
        # maintained environment plane. A declared host runtime is instead
        # authoritative for its own dependencies, so two benchmark stacks
        # cannot silently override one another.
        image = image.pip_install(
            *_osworld_base.ENV_PLANE_TORCH_PIP,
            index_url=_osworld_base.ENV_PLANE_TORCH_INDEX,
        ).pip_install(*_osworld_base.ENV_PLANE_PIP)
    return image


def prepare_native_controller(runtime: dict) -> None:
    """Build shared controller dependencies while the desktop image imports."""
    if runtime.get("pyproject"):
        return  # This dependency needs the materialized environment folder.
    import modal

    app = modal.App.lookup("cua-speedrun-env", create_if_missing=True)
    with modal.enable_output():
        _base_image(Path.cwd(), runtime, native_runner=True).build(app)


def _build_image(env_local_dir: Path, extra_dirs: dict[str, Path], *, native_runner: bool = False):
    image = _base_image(env_local_dir, _host_runtime(env_local_dir), native_runner=native_runner)
    # Modal requires build steps before local-file mounts. The packages and
    # environment definition are added last and remain evaluator-only.
    image = (
        image
        .add_local_dir(
            str(_ga_package_dir()), remote_path="/opt/ga/gym_anything"
        )
        .add_local_dir(str(_CS_PACKAGE), remote_path="/opt/cs/cua_speedrun")
        .add_local_dir(str(env_local_dir), remote_path="/envs/env")
    )
    for remote, local in extra_dirs.items():
        image = image.add_local_dir(str(local), remote_path=remote)
    return image


def _env_spec_payload(env_spec: "Mapping[str, Any]") -> str:
    """The env block the plane's backend receives: byte-for-byte the task's
    own env block, except env_dir, which points at the in-sandbox mount. The
    backend applies its own defaults for anything absent, so the remote path
    cannot diverge from the local one on defaults again."""
    payload = dict(env_spec)
    payload["env_dir"] = "/envs/env"
    return json.dumps(payload, sort_keys=True)


def create_env_sandbox(
    env_local_dir: Path,
    env_spec: "Mapping[str, Any]",
    seed: int,
    timeout_sec: float,
    grace_sec: float,
    *,
    generator_local: Path | None = None,
    task_label: str | None = None,
    gateway_port: int = 8390,
    region: str | None = None,
    sandbox_timeout_sec: int = 3600,
    runtime_env: Mapping[str, str] | None = None,
    native_runner: bool = False,
) -> EnvSandbox:
    """Create the env+gateway sandbox and return immediately with its tunnel.
    Callers overlap other work (agent warmup) with the boot, then call
    wait_env_healthy() before arming. ``env_spec`` is the task's env block,
    forwarded verbatim to the in-sandbox Backend."""
    import modal

    run_token = secrets.token_urlsafe(16)
    control_token = secrets.token_urlsafe(16)

    extra: dict[str, Path] = {}
    gen_remote = None
    if generator_local is not None:
        extra["/gen"] = generator_local.parent
        gen_remote = f"/gen/{generator_local.name}"

    runtime = _host_runtime(env_local_dir)
    app = modal.App.lookup("cua-speedrun-env", create_if_missing=True)
    volume = modal.Volume.from_name("gym-anything-qemu-cache", create_if_missing=True)
    image = _build_image(env_local_dir, extra, native_runner=native_runner)

    # The runner and pre-boot setup depend on the environment kind. Android
    # envs restore the AVD stack from the volume and run avd_native; Linux
    # envs run qemu_native (Modal workers are x86_64 with KVM via
    # vm_runtime), provisioning the base qcow2 on first boot and caching it
    # on the same volume for every later sandbox.
    kind = _detect_env_kind(env_local_dir)
    if native_runner:
        if kind != "linux":
            raise ValueError("native Gym-Anything desktops require Linux")
        env_backend = "gym-anything-modal-native"
        pre_setup = ""
        boot_budget = 7200
        sandbox_timeout_sec = max(sandbox_timeout_sec, 10800)
    elif kind == "android":
        env_backend = "gym-anything-avd-native"
        pre_setup = _AVD_CACHE_SETUP
        boot_budget = 900
    else:
        env_backend = "gym-anything-qemu-native"
        pre_setup = ""
        boot_budget = 7200
        sandbox_timeout_sec = max(sandbox_timeout_sec, 10800)

    env = {
        "PYTHONPATH": "/opt/ga:/opt/cs",
        # The in-sandbox Backend sets GYM_ANYTHING_RUNNER itself; the
        # launcher only names which registered backend the plane constructs.
        "GYM_ANYTHING_QEMU_CACHE": "/cache/qemu",
        # OSWorld-image benchmarks reference ${OSWORLD_QEMU_BASE_IMAGE} in
        # env.json. The prepared qcow2 (plus its .provenance.json sidecar)
        # lives on the shared qemu-cache volume, built once by
        # scripts/build_osworld_modal_qcow2.py; benchmarks that do not
        # reference the variable ignore it.
        "OSWORLD_QEMU_BASE_IMAGE": "/cache/qemu/osworld_ubuntu.qcow2",
        "CS_ENV_BACKEND": env_backend,
        "CS_ENV_SPEC": _env_spec_payload(env_spec),
        "CS_SEED": str(seed),
        "CS_TIMEOUT_SEC": str(timeout_sec),
        "CS_GRACE_SEC": str(grace_sec),
        "CS_GATEWAY_PORT": str(gateway_port),
        "CS_RUN_TOKEN": run_token,
        "CS_CONTROL_TOKEN": control_token,
    }
    if gen_remote:
        env["CS_GENERATOR"] = gen_remote
    if task_label:
        env["CS_TASK_LABEL"] = task_label

    available_runtime_env = {**os.environ, **dict(runtime_env or {})}
    forwarded = {
        name: available_runtime_env[name]
        for name in runtime.get("forward_env") or []
        if available_runtime_env.get(name) is not None
    }
    for selector, api_key_name in list(forwarded.items()):
        if not selector.endswith("_API_KEY_ENV"):
            continue
        if api_key_name and available_runtime_env.get(api_key_name) is not None:
            forwarded[api_key_name] = available_runtime_env[api_key_name]
    if native_runner:
        from modal.config import config

        for name, setting in (("MODAL_TOKEN_ID", "token_id"), ("MODAL_TOKEN_SECRET", "token_secret")):
            value = os.environ.get(name) or config.get(setting)
            if not value:
                raise ValueError("Modal credentials are required to create native desktops")
            forwarded[name] = value
        if config.get("environment"):
            env["MODAL_ENVIRONMENT"] = config.get("environment")
        env["CS_NATIVE_OWNER"] = secrets.token_hex(16)
    runtime_secrets = [modal.Secret.from_dict(forwarded)] if forwarded else []

    sandbox_cpu, sandbox_memory_mb = _environment_resources(
        env_local_dir, env_spec
    )
    if native_runner:
        sandbox_cpu, sandbox_memory_mb = 1, 4096
    sb = modal.Sandbox.create(
        "bash", "-lc",
        # Preserve full environment logs and propagate the env-plane exit
        # status through tee.
        "set -o pipefail; " + pre_setup + "mkdir -p /tmp/cs_run && python3 -u -m "
        "cua_speedrun.remote.env_plane 2>&1 | tee /tmp/cs_run/env_plane.log",
        app=app,
        image=image,
        cpu=sandbox_cpu,
        memory=sandbox_memory_mb,
        timeout=sandbox_timeout_sec,
        volumes={"/cache/qemu": volume},
        encrypted_ports=[gateway_port],
        experimental_options={"vm_runtime": True},
        region=region,
        env=env,
        secrets=runtime_secrets,
        tags={"cs-native-owner": env["CS_NATIVE_OWNER"]} if native_runner else {},
    )
    if native_runner:
        sb = _SandboxGroup(sb, env["CS_NATIVE_OWNER"])

    try:
        tunnel = sb.tunnels(timeout=300)[gateway_port]
    except BaseException:
        # Terminate queued sandboxes when tunnel setup fails so a placement
        # retry does not leave an unused sandbox running.
        try:
            sb.terminate()
        except Exception:
            pass
        raise
    base_url = tunnel.url.rstrip("/")
    return EnvSandbox(sb, base_url, run_token, control_token, gateway_port,
                      boot_budget_sec=boot_budget)


def tail_sandbox_file(sb: Any, path: str, on_line: Callable[[str], None],
                      stop: threading.Event, poll_sec: float = 5.0) -> threading.Thread:
    """Tail a file inside a sandbox by polling short exec calls, forwarding
    complete new lines to on_line. Makes the env boot observable live; the
    env sandbox's output used to be pulled only after the task ended, so a
    hung boot was indistinguishable from a slow one.

    Polling execs instead of holding sb.stdout is deliberate: a blocking
    stdout iterator cannot be cancelled from another thread and an abandoned
    one spews ClientClosed tracebacks at interpreter shutdown. Each poll here
    is a short-lived RPC and the loop stops cleanly via the stop event."""

    def _reader() -> None:
        offset = 0
        buf = ""
        while True:
            try:
                proc = sb.exec(
                    "bash", "-lc",
                    f"tail -c +{offset + 1} '{path}' 2>/dev/null",
                    timeout=30,
                )
                chunk = proc.stdout.read()
                proc.wait()
                if chunk:
                    offset += len(chunk.encode("utf-8", "replace"))
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        on_line(line + "\n")
            except Exception:
                pass  # sandbox still booting or already gone; try again
            if stop.wait(poll_sec):
                return

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return thread


def wait_env_healthy(es: EnvSandbox, boot_budget_sec: int | None = None) -> None:
    """Block until the gateway answers health over the tunnel. The default
    budget comes from the sandbox's env kind (first Linux boots provision a
    base image in-sandbox and need far longer than a cached AVD restore)."""
    import requests

    if boot_budget_sec is None:
        boot_budget_sec = getattr(es, "boot_budget_sec", 900)
    deadline = time.time() + boot_budget_sec
    while time.time() < deadline:
        if es.sandbox.poll() is not None:
            raise RuntimeError(
                f"env sandbox exited early (code {es.sandbox.returncode})"
            )
        try:
            r = requests.get(
                f"{es.base_url}/_ctl/{es.control_token}/health", timeout=10
            )
            if r.status_code == 200 and r.json().get("ready"):
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError("env gateway did not become healthy in time")


def _exec_text(es: EnvSandbox, cmd: str, timeout: int) -> tuple[int, str]:
    proc = es.sandbox.exec("bash", "-lc", cmd, timeout=timeout)
    out = proc.stdout.read()
    proc.wait()
    return (proc.returncode if proc.returncode is not None else -1), out or ""


# One exec's worth of artifact download. 8 MiB of tar becomes ~11 MiB of
# base64 text per chunk, small enough that a single flaky exec retries
# cheaply while a 100-frame episode still needs only a couple dozen chunks.
_ARTIFACT_CHUNK_BYTES = 8 * 1024 * 1024


def pull_env_artifacts(
    es: EnvSandbox, task_dir: Path
) -> tuple[list[str], str | None]:
    """Pull the trajectory (gateway frames + gym-anything episode) and the
    environment-side logs out of the env sandbox into task_dir, before it is
    torn down. Returns (names_pulled, error). A failure must not fail the
    run, but it must be visible: the streaming single-exec version of this
    silently lost every artifact of three 100-frame episodes in one run
    (~200MB of base64 through one exec), so the tar now lands in a sandbox
    file first and downloads in verified, individually-retried chunks."""
    import base64
    import io
    import tarfile

    remote_tar = "/tmp/cs_run/_artifacts.tgz"
    try:
        rc, _ = _exec_text(
            es,
            f"cd /tmp/cs_run 2>/dev/null && rm -f {remote_tar} && "
            f"tar czf {remote_tar} env_plane.log frame_*.png frame_*.a11y.json episode "
            f"2>/dev/null; test -s {remote_tar}",
            timeout=300,
        )
        if rc != 0:
            return [], "env sandbox produced no artifact archive"
        rc, size_text = _exec_text(es, f"wc -c < {remote_tar}", timeout=60)
        if rc != 0 or not size_text.strip().isdigit():
            return [], f"could not stat the artifact archive: {size_text.strip()[:200]}"
        total = int(size_text.strip())

        chunks: list[bytes] = []
        offset = 0
        while offset < total:
            count = min(_ARTIFACT_CHUNK_BYTES, total - offset)
            chunk = None
            error = None
            for _attempt in range(3):
                rc, b64 = _exec_text(
                    es,
                    f"tail -c +{offset + 1} {remote_tar} | head -c {count} "
                    f"| base64 | tr -d '\\n'",
                    timeout=180,
                )
                try:
                    decoded = base64.b64decode(b64.strip()) if rc == 0 else b""
                except Exception as exc:
                    decoded, error = b"", f"chunk decode failed: {exc}"
                    continue
                if len(decoded) == count:
                    chunk = decoded
                    break
                error = (
                    f"chunk at {offset} returned {len(decoded)} of {count} bytes"
                )
            if chunk is None:
                return [], error or f"chunk at {offset} could not be read"
            chunks.append(chunk)
            offset += count

        raw = b"".join(chunks)
        pulled = []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            for member in tf.getmembers():
                # our own sandbox, but stay defensive about extraction paths
                name = member.name.lstrip("./")
                if name.startswith("/") or ".." in name.split("/"):
                    continue
                pulled.append(name)
            tf.extractall(task_dir)
        _exec_text(es, f"rm -f {remote_tar}", timeout=30)
        return pulled, None
    except Exception as exc:
        return [], f"artifact pull failed: {type(exc).__name__}: {exc}"


def read_region(es: EnvSandbox, budget_sec: int = 120) -> str | None:
    """The region Modal actually placed the env sandbox in, read from the
    MODAL_REGION variable inside it. Used to pin the agent sandbox to the
    same region so the environment-control hop on the timed path stays local."""
    deadline = time.time() + budget_sec
    while time.time() < deadline:
        if es.sandbox.poll() is not None:
            return None
        try:
            p = es.sandbox.exec("bash", "-lc", "printenv MODAL_REGION")
            out = p.stdout.read()
            p.wait()
            if p.returncode == 0 and out.strip():
                return out.strip()
        except Exception:
            pass
        time.sleep(3)
    return None


def launch_env_sandbox(
    env_local_dir: Path,
    env_spec: Mapping[str, Any],
    seed: int,
    timeout_sec: float,
    grace_sec: float,
    *,
    generator_local: Path | None = None,
    task_label: str | None = None,
    gateway_port: int = 8390,
    region: str | None = None,
    boot_budget_sec: int = 900,
    sandbox_timeout_sec: int = 3600,
) -> EnvSandbox:
    """Create the env sandbox and block until its gateway is healthy."""
    es = create_env_sandbox(
        env_local_dir, env_spec, seed, timeout_sec, grace_sec,
        generator_local=generator_local,
        task_label=task_label, gateway_port=gateway_port, region=region,
        sandbox_timeout_sec=sandbox_timeout_sec,
    )
    wait_env_healthy(es, boot_budget_sec)
    return es
