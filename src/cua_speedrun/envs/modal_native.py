"""EnvAdapter over a desktop booted directly on Modal (no KVM).

Runs INSIDE the env sandbox, in the modal-native env-plane process. The desktop
rootfs has been booted with its own systemd in a PID+mount namespace via
pivot_root; this adapter reaches the desktop by nsenter-ing into that namespace
for screenshots (scrot), input (xdotool), and shell (exec_read).

It implements the Gateway's observation/action surface and exposes the
canonical OSWorld server on its normal localhost ports. The Gateway is
otherwise decoupled from gym-anything, so no gym-anything object is involved.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from cua_speedrun.envs.base import (
    Backend,
    EnvAdapter,
    Observation,
    PreparedEnv,
    Verdict,
)
from cua_speedrun.envs.keyboard import keyboard_script as _keyboard_script

# The boot namespace's PID, written by the env-plane's boot step.
SYSTEMD_PID_FILE = "/run/osworld-systemd.pid"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_OSWORLD_SERVICE = "osworld.service"
_OSWORLD_READY_COMMAND = (
    "/usr/bin/python",
    "-c",
    "import urllib.request; "
    "urllib.request.urlopen('http://127.0.0.1:5000/platform', timeout=2).read()",
)


def _guest_process_environment() -> dict[str, str]:
    # Evaluator credentials must not enter guest processes.
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
    }


def _nsenter_prefix() -> list[str]:
    with open(SYSTEMD_PID_FILE) as handle:
        pid = handle.read().strip()
    return ["nsenter", "--target", pid, "--mount", "--pid", "--root", "--wd", "--"]


class ModalNativeAdapter(EnvAdapter):
    """One live modal-native OSWorld desktop."""

    def __init__(
        self,
        desktop_user: str = "user",
        desktop_uid: int = 1000,
        display: str = ":0",
        settle_sec: float = 0.3,
    ) -> None:
        self.user = desktop_user
        self.uid = desktop_uid
        self.display = display
        self.settle_sec = settle_sec
        self.trajectory: dict[str, list[dict[str, Any]]] = {"steps": []}
        self._obs_path_guest = f"/home/{desktop_user}/_cs_obs.png"
        # /osworld is the booted rootfs as seen from the outer namespace where
        # this process runs, so it can read files the guest writes.
        self._obs_path_outer = f"/osworld/home/{desktop_user}/_cs_obs.png"

    # -- guest execution helpers -------------------------------------------
    def _run_root(
        self, command: str, timeout: int = 60, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            _nsenter_prefix() + ["bash", "-lc", command],
            capture_output=True,
            timeout=timeout,
            input=stdin,
            env=_guest_process_environment(),
        )

    def _user_wrap(self, command: str) -> str:
        # The GDM session cookie; the env-plane also opens xhost +local: at
        # boot, so either path grants these exec'd commands the display.
        return (
            f"sudo -u {self.user} env "
            f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{self.uid}/bus "
            f"HOME=/home/{self.user} USER={self.user} DISPLAY={self.display} "
            f"XAUTHORITY=/run/user/{self.uid}/gdm/Xauthority "
            f"bash -lc {shlex.quote(command)}"
        )

    def _run_user(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        result = self._run_root(self._user_wrap(command), timeout=timeout)
        result.check_returncode()
        return result

    # -- EnvAdapter surface ------------------------------------------------
    def observe(self) -> Observation:
        # -p draws the mouse pointer into the capture. OSWorld's own server
        # composites the cursor into every Linux screenshot (its handler is
        # literally capture_screen_with_cursor, pasting Xcursor's image at the
        # pointer position), and the QEMU path gets it for free because the
        # VNC framebuffer already contains it. Without this the agent cannot
        # see where the pointer is on this backend alone, which changes every
        # observation of every task.
        self._run_user(f"scrot -o -p {shlex.quote(self._obs_path_guest)}", timeout=30)
        with open(self._obs_path_outer, "rb") as handle:
            png = handle.read()
        if png[:8] != _PNG_MAGIC:
            raise RuntimeError("scrot did not produce a PNG")
        return Observation(png=png, meta={"resolution": self._geometry()})

    def _geometry(self) -> list[int]:
        result = self._run_user("xdotool getdisplaygeometry", timeout=15)
        parts = result.stdout.decode(errors="replace").split()
        try:
            return [int(parts[0]), int(parts[1])]
        except (IndexError, ValueError) as exc:
            raise RuntimeError("could not read the desktop resolution") from exc

    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        for action in actions:
            self._inject(action)
        self.trajectory["steps"].append({"event": "step", "action": actions, "t_mono": started})
        if self.settle_sec:
            time.sleep(self.settle_sec)
        return {"done": False}

    def _inject(self, action: dict[str, Any]) -> None:
        if not isinstance(action, dict):
            return
        if action.get("action") == "wait":
            time.sleep(float(action.get("time", 0.5)))
            return

        mouse = action.get("mouse") or {}
        commands: list[str] = []
        for key, button in (("left_click", 1), ("right_click", 3), ("middle_click", 2)):
            if key in mouse:
                x, y = mouse[key]
                commands.append(f"mousemove {int(x)} {int(y)} click {button}")
        if "double_click" in mouse:
            x, y = mouse["double_click"]
            commands.append(f"mousemove {int(x)} {int(y)} click --repeat 2 1")
        if "triple_click" in mouse:
            x, y = mouse["triple_click"]
            commands.append(f"mousemove {int(x)} {int(y)} click --repeat 3 1")
        if "move" in mouse:
            x, y = mouse["move"]
            commands.append(f"mousemove {int(x)} {int(y)}")
        for drag_key, button in (
            ("left_click_drag", 1),
            ("right_click_drag", 3),
        ):
            if drag_key not in mouse:
                continue
            points = mouse[drag_key]
            if len(points) == 2:
                # [start, end]: press at start, drag to end.
                (x1, y1), (x2, y2) = points
                commands.append(
                    f"mousemove {int(x1)} {int(y1)} mousedown {button} "
                    f"mousemove {int(x2)} {int(y2)} mouseup {button}"
                )
            else:
                # [dest]: drag from the current cursor to dest (the Qwen /
                # Anthropic computer_use single-coordinate convention).
                (x2, y2) = points[0]
                commands.append(
                    f"mousedown {button} mousemove {int(x2)} {int(y2)} "
                    f"mouseup {button}"
                )
        buttons = mouse.get("buttons") or {}
        for field, subcommand, button in (
            ("left_down", "mousedown", 1),
            ("right_down", "mousedown", 3),
            ("middle_down", "mousedown", 2),
            ("left_up", "mouseup", 1),
            ("right_up", "mouseup", 3),
            ("middle_up", "mouseup", 2),
        ):
            if buttons.get(field):
                commands.append(f"{subcommand} {button}")
        if "scroll" in mouse:
            amount = int(mouse["scroll"])
            button = 5 if amount > 0 else 4
            commands.append(("click " + str(button) + " ") * abs(amount))
        for command in commands:
            self._run_user(f"xdotool {command}", timeout=30)

        keyboard = action.get("keyboard") or {}
        if keyboard:
            result = self._run_user(
                f"python3 -c {shlex.quote(_keyboard_script(keyboard))}", timeout=120
            )
            if getattr(result, "returncode", 0) != 0:
                stderr = getattr(result, "stderr", b"") or b""
                raise RuntimeError(
                    "modal-native keyboard injection failed: "
                    + stderr.decode(errors="replace")[-500:]
                )

    def exec_read(self, command: str) -> str:
        """Root guest shell, matching the OSWorld verifier env_info contract
        (the verifier drops to the desktop user itself where it needs to)."""
        result = self._run_root(command, timeout=120)
        return result.stdout.decode(errors="replace")

    def copy_from_env(self, vm_path: str, host_path: str) -> None:
        result = self._run_root(f"base64 {shlex.quote(vm_path)}", timeout=120)
        result.check_returncode()
        with open(host_path, "wb") as handle:
            handle.write(base64.b64decode(result.stdout))

    def copy_to_env(self, host_path: str, vm_path: str) -> None:
        """Write a host file into the guest, the inverse of copy_from_env."""
        with open(host_path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode()
        result = self._run_root(
            f"base64 -d > {shlex.quote(vm_path)}",
            timeout=120,
            stdin=encoded.encode(),
        )
        result.check_returncode()

    def finalize(self) -> Verdict:
        # Only reached if no checker is supplied. Modal-native tasks always
        # supply an OSWorld checker, so this is a safe default.
        return Verdict(passed=False, score=0.0, detail="no checker supplied")

    def close(self) -> None:
        # The Gateway never calls this; the sandbox is torn down by the executor.
        pass


# ---------------------------------------------------------------------------
# Preparation. Everything below boots and stages the desktop; it lives here,
# not in the env plane, so every environment is constructed by a Backend and
# the plane stays a transport shim.

# Boot the extracted rootfs's own systemd in a fresh PID+mount namespace and
# make /osworld the namespace root via pivot_root, so systemd (and logind,
# which needs unit sandboxing) run as PID 1 of a real root rather than under
# a chroot.
_BOOT_SCRIPT = r"""
cat > /osworld-boot.sh <<'EOF'
#!/bin/bash
set -e
mount --bind /osworld /osworld
cd /osworld
mount --rbind /dev dev
mount --rbind /sys sys
mount -t proc proc proc
mount -t tmpfs -o mode=0755,nosuid,nodev tmpfs run
mount -t tmpfs tmpfs tmp
ln -sfn /dev/null etc/systemd/system/acpid.path
mkdir -p old_root
pivot_root . old_root
umount -l /old_root
rmdir /old_root 2>/dev/null || true
exec env -i container=modal TERM=linux PATH=/usr/sbin:/usr/bin:/sbin:/bin /sbin/init
EOF
chmod +x /osworld-boot.sh
rm -f /run/osworld-systemd.pid
nohup unshare --fork --pid --mount --propagation private /osworld-boot.sh > /osworld-boot.log 2>&1 &
launcher=$!
for _ in $(seq 1 300); do
    child=$(cat /proc/"$launcher"/task/"$launcher"/children 2>/dev/null | awk '{print $1}' || true)
    if [ -n "$child" ] && [ -d "/proc/$child" ]; then
        echo "$child" > /run/osworld-systemd.pid
        exit 0
    fi
    kill -0 "$launcher" 2>/dev/null || { cat /osworld-boot.log; exit 1; }
    sleep 0.1
done
echo "systemd never appeared"; cat /osworld-boot.log; exit 1
"""


def _ns(pid: str) -> list[str]:
    return ["nsenter", "--target", pid, "--mount", "--pid", "--root", "--wd", "--"]


def _boot_user_wrap(user: str, command: str) -> str:
    # DISPLAY :0 with the GDM session cookie: the desktop is the image's own
    # GDM autologin session, not a VNC server.
    return (
        f"sudo -u {user} env DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus "
        f"HOME=/home/{user} USER={user} DISPLAY=:0 "
        f"XAUTHORITY=/run/user/1000/gdm/Xauthority "
        f"bash -lc {shlex.quote(command)}"
    )


def _guest_command_succeeds(pid: str, command: tuple[str, ...]) -> bool:
    try:
        result = subprocess.run(
            _ns(pid) + list(command), capture_output=True, timeout=5,
            env=_guest_process_environment(),
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _service_diagnostics(pid: str, service: str) -> str:
    output = []
    for command in (
        ("systemctl", "status", service, "--no-pager", "-l"),
        ("journalctl", "-u", service, "-b", "--no-pager", "-n", "80"),
    ):
        try:
            result = subprocess.run(
                _ns(pid) + list(command), capture_output=True, timeout=30,
                env=_guest_process_environment(),
            )
            output.append((result.stdout + result.stderr).decode(errors="replace"))
        except subprocess.TimeoutExpired:
            output.append(f"{' '.join(command)} timed out")
    return "\n".join(output)[-8000:]


def _ensure_guest_service(
    pid: str,
    service: str,
    ready_command: tuple[str, ...],
    wait_sec: float,
) -> None:
    """Recover an image-owned service after desktop startup, then gate on it.

    Image services may start before their display dependency is usable and
    exhaust systemd's start limit. Once the desktop is ready, reset that limit,
    restart the service once, and require its real interface before exposing
    the environment to an agent.
    """
    if _guest_command_succeeds(pid, ready_command):
        return

    for action in ("reset-failed", "restart"):
        result = subprocess.run(
            _ns(pid) + ["systemctl", action, service],
            capture_output=True,
            timeout=30,
            env=_guest_process_environment(),
        )
        if result.returncode != 0:
            detail = (result.stdout + result.stderr).decode(errors="replace")
            raise RuntimeError(
                f"could not {action} required guest service {service}: {detail[-2000:]}"
            )

    deadline = time.monotonic() + wait_sec
    while True:
        if _guest_command_succeeds(pid, ready_command):
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(1)
    raise RuntimeError(
        f"required guest service {service} did not become ready after restart:\n"
        f"{_service_diagnostics(pid, service)}"
    )


def boot_desktop(
    user: str,
    session_wait_sec: float = 300.0,
    settle_sec: int = 20,
    service_wait_sec: float = 120.0,
    services: list[dict[str, Any]] | None = None,
) -> str:
    from cua_speedrun.envs.packed_rootfs import mount_packed_rootfs

    mount_packed_rootfs()
    subprocess.run(
        ["bash", "-lc", _BOOT_SCRIPT], check=True, capture_output=True, timeout=120,
        env=_guest_process_environment(),
    )
    pid = Path("/run/osworld-systemd.pid").read_text().strip()
    # Ready means the image's own GDM autologin session is up: gnome-shell
    # running for the desktop user on the dummy Xorg display.
    deadline = time.time() + session_wait_sec
    shell_up = False
    while time.time() < deadline:
        probe = subprocess.run(
            _ns(pid) + ["pgrep", "-u", user, "-x", "gnome-shell"],
            capture_output=True,
            env=_guest_process_environment(),
        )
        if probe.stdout.decode().strip():
            shell_up = True
            break
        time.sleep(3)
    if not shell_up:
        raise RuntimeError("the GDM autologin session never started gnome-shell")
    # Exec'd setup hooks and verifier commands run outside the session and
    # carry no X cookie; open local X access once, from inside the session.
    x_access = subprocess.run(
        _ns(pid) + ["bash", "-lc", _boot_user_wrap(user, "xhost +local:")],
        capture_output=True, timeout=30,
        env=_guest_process_environment(),
    )
    if x_access.returncode != 0:
        detail = (x_access.stdout + x_access.stderr).decode(errors="replace")
        raise RuntimeError(f"could not open local X access: {detail[-2000:]}")
    time.sleep(settle_sec)  # let the session finish rendering
    if services is None:
        services = [{"unit": _OSWORLD_SERVICE, "ready_command": _OSWORLD_READY_COMMAND}]
    for service in services:
        _ensure_guest_service(
            pid, service["unit"], tuple(service["ready_command"]), service_wait_sec,
        )
    return pid


def _log_pre_task_output(stdout: bytes | None, stderr: bytes | None) -> None:
    for name, output in (("stdout", stdout), ("stderr", stderr)):
        if output:
            print(f"[modal-native] pre_task {name}:", flush=True)
            text = output.decode(errors="replace")
            print(text, end="" if text.endswith("\n") else "\n", flush=True)


def _run_pre_task_command(command: list[str], *, timeout: int) -> None:
    try:
        result = subprocess.run(
            command, capture_output=True, timeout=timeout,
            env=_guest_process_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        _log_pre_task_output(exc.stdout, exc.stderr)
        raise
    print(f"[modal-native] pre_task rc={result.returncode}", flush=True)
    _log_pre_task_output(result.stdout, result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"pre_task failed with rc={result.returncode}: "
            f"{result.stderr.decode(errors='replace')[-500:]}"
        )


def run_pre_task_setup(pid: str, env_dir: str, gym_task_id: str) -> None:
    """Run the OSWorld task's pre_task hook untimed, as the QEMU backend does
    inside env.reset. The task tree is copied into the guest at /workspace/tasks,
    the same path the QEMU path's env.json mounts it at (there it is a read-only
    mount; here it is a plain copy, deleted by scrub_privileged before arm)."""
    task_json = Path(env_dir) / "tasks" / gym_task_id / "task.json"
    spec = json.loads(task_json.read_text())
    pre_task = (spec.get("hooks") or {}).get("pre_task")
    if not pre_task:
        return
    tasks_src = Path(env_dir) / "tasks"
    subprocess.run(
        _ns(pid) + ["mkdir", "-p", "/workspace"], check=True, capture_output=True,
        env=_guest_process_environment(),
    )
    subprocess.run(
        ["bash", "-lc", f"cp -a {shlex.quote(str(tasks_src))} /osworld/workspace/tasks"],
        check=True, capture_output=True, timeout=120,
        env=_guest_process_environment(),
    )
    print(f"[modal-native] running pre_task: {pre_task}", flush=True)
    _run_pre_task_command(
        _ns(pid) + ["bash", "-lc", pre_task],
        timeout=int((spec.get("hooks") or {}).get("pre_task_timeout", 600)),
    )


def scrub_privileged(pid: str) -> None:
    """Delete verifier/task material from the guest-visible task tree before the
    clock arms, the modal-native analog of scrub_guest_privileged_material."""
    command = (
        "find /workspace/tasks -type f "
        r"\( -name 'task.json' -o -name 'source.json' -o -name 'verifier.py' "
        r"-o -name '*_verifier.py' -o -name '*_judge.py' -o -name 'setup_task.sh' \) -delete"
    )
    subprocess.run(
        _ns(pid) + ["bash", "-lc", command], capture_output=True, timeout=60,
        check=True, env=_guest_process_environment(),
    )


def warm_osworld_evaluators(env_dir: str) -> None:
    """Import the pinned OSWorld evaluator runtime once at boot, untimed.

    Doing it here turns a missing dependency, source mismatch, or failed fetch
    into an env boot failure rather than a task score."""
    import importlib.util

    shared = Path(env_dir) / "tasks" / "_shared" / "osworld_verifier.py"
    if not shared.is_file():
        return  # benchmark ships self-contained verifiers; nothing to warm
    spec = importlib.util.spec_from_file_location("cs_osworld_verifier_warm", shared)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ensure_osworld_evaluators()
    from desktop_env.desktop_env import DesktopEnv  # noqa: F401

    print("[modal-native] pinned OSWorld evaluator importable (warm)", flush=True)


def load_osworld_checker(
    adapter: ModalNativeAdapter, env_dir: str, gym_task_id: str, workdir: Path | None = None,
):
    """Return a Gateway ``checker()`` that runs the task's OSWorld
    ``verifier.py::check``. The verifier is loaded from the env-plane's own
    task tree, which the agent can never reach; the guest copy is deleted by
    ``scrub_privileged`` before the clock arms."""
    import importlib.util

    verifier_path = Path(env_dir) / "tasks" / gym_task_id / "verifier.py"
    spec = importlib.util.spec_from_file_location("cs_osworld_verifier", verifier_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    env_info = {
        "exec_capture": adapter.exec_read,
        "copy_from_env": adapter.copy_from_env,
        "copy_to_env": adapter.copy_to_env,
        "ssh_user": adapter.user,
        "x11_display": adapter.display,
        "artifact_dir": str(workdir) if workdir is not None else None,
        "observe": adapter.observe,
    }

    def checker() -> Verdict:
        result = module.check(traj=adapter.trajectory, env_info=env_info, task_info={})
        score = result.get("score")
        return Verdict(
            passed=bool(result.get("passed")),
            score=float(score) if score is not None else None,
            detail=str(result.get("feedback", "")),
        )

    return checker


class ModalNativeBackend(Backend):
    """Prepares one modal-native desktop.

    Only functions inside the env sandbox: it boots the extracted rootfs with
    pivot_root and expects Modal's VM-runtime kernel. The env plane is its
    only production caller; the registry entry exists so environment
    construction goes through the one Backend factory everywhere.
    """

    name = "modal-native"

    def __init__(
        self,
        desktop_user: str | None = None,
        desktop_settle_sec: int | None = None,
    ) -> None:
        # Explicit arguments override the launcher's environment variables.
        self.desktop_user = desktop_user or os.environ.get("CS_DESKTOP_USER", "user")
        self.desktop_settle_sec = (
            desktop_settle_sec
            if desktop_settle_sec is not None
            else int(os.environ.get("CS_DESKTOP_SETTLE_SEC", "20"))
        )

    def prepare(self, env_spec: dict[str, Any], seed: int, workdir: Path) -> PreparedEnv:
        for key in ("env_dir", "task_id"):
            if key not in env_spec:
                raise ValueError(f"modal-native env block needs '{key}'")
        env_dir = os.path.expanduser(os.path.expandvars(str(env_spec["env_dir"])))
        task_id = str(env_spec["task_id"])

        t0 = time.monotonic()
        print("[modal-native] booting desktop (no KVM) ...", flush=True)
        services = json.loads(os.environ["CS_NATIVE_SERVICES"]) if "CS_NATIVE_SERVICES" in os.environ else None
        pid = boot_desktop(self.desktop_user, settle_sec=self.desktop_settle_sec, services=services)
        print(f"[modal-native] desktop up (systemd pid {pid})", flush=True)

        run_pre_task_setup(pid, env_dir, task_id)

        settle_sec = float(env_spec.get("action_settle_ms", 5000)) / 1000.0
        adapter = ModalNativeAdapter(
            desktop_user=self.desktop_user, settle_sec=settle_sec
        )
        entrypoint = env_spec.get("native_adapter_entrypoint")
        if entrypoint:
            relative, separator, function_name = str(entrypoint).partition(":")
            root = Path(env_dir).resolve()
            path = (root / relative).resolve()
            if not separator or not function_name.isidentifier() or not path.is_relative_to(root):
                raise ValueError("native_adapter_entrypoint must be a file inside env_dir and a function")
            spec = importlib.util.spec_from_file_location("cs_native_adapter", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            prepared = getattr(module, function_name)(
                env=adapter, env_spec=env_spec, env_dir=root,
                workdir=workdir, settle_sec=settle_sec,
            )
            if not isinstance(prepared, PreparedEnv):
                raise TypeError("native adapter must return PreparedEnv")
            prepared.adapter.observe()
            prepared.prepare_time_sec = time.monotonic() - t0
            prepared.info.update(runner="modal-native", systemd_pid=pid)
            return prepared
        warm_osworld_evaluators(env_dir)
        # Build the checker (loads verifier.py plane-side) BEFORE scrubbing
        # deletes the guest's copy of the task tree.
        checker = load_osworld_checker(adapter, env_dir, task_id, workdir)
        scrub_privileged(pid)
        print("[modal-native] scrubbed guest privileged material", flush=True)

        task_json = json.loads(
            (Path(env_dir) / "tasks" / task_id / "task.json").read_text()
        )
        description = (task_json.get("natural_language") or {}).get("prompt", task_id)
        # Same fail-closed contract as the QEMU backend: a dead or unrendered
        # display fails preparation (retried as infrastructure) instead of
        # becoming the agent's first observation. observe() validates the PNG.
        adapter.observe()
        return PreparedEnv(
            adapter=adapter,
            description=description,
            prepare_time_sec=time.monotonic() - t0,
            info={
                "task_id": task_id,
                "runner": "modal-native",
                "systemd_pid": pid,
                "desktop_user": self.desktop_user,
            },
            checker=checker,
        )
