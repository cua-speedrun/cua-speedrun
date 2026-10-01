"""The gym-anything backend.

Every gym-anything environment is constructed by this class, wherever it
runs: the local executor and the Slurm worker call ``prepare`` directly, and
the modal-remote env plane constructs the same registered backend inside the
env sandbox. There is deliberately no second implementation of preparation,
so runner guards, image provenance, SSH overrides, cache defaults, scrubbing,
and the initial screenshot check cannot drift between placements. The local
default backend requests gym-anything's QEMU runner family and records the
concrete runner gym-anything selects; explicit backends require their named
runner and fail closed.

The task's env block looks like:

    env:
      kind: gym-anything
      env_dir: path/to/gym-anything/environment   # a gym-anything env config dir
      task_id: some_task                          # a task under that env's tasks/
      use_cache: true
      cache_level: post_task

Requires gym-anything installed (pip install -e path/to/gym-anything-primerl).
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
import os
import shlex
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any

from cua_speedrun.envs.base import Backend, EnvAdapter, Observation, PreparedEnv, Verdict
from cua_speedrun.envs.keyboard import keyboard_script, patch_runner_keyboard


# Files used by host-side grading or imported benchmark rubrics must not
# remain in the guest after setup. Mounts in VM runners are copies, so the
# host originals remain available to gym-anything's verifier.
_PRIVILEGED_GUEST_NAMES = (
    "task.json",
    "task.yaml",
    "task.yml",
    "source.json",
    "validated_pi.json",
    "vlm_checklist.json",
    "hard_tasks_done.txt",
)


def _posix_privileged_scrub_command(
    targets: list[str], extra_names: set[str] | None = None
) -> str:
    names = [
        *_PRIVILEGED_GUEST_NAMES,
        *sorted(extra_names or ()),
        "verifier.py",
        "*_verifier.py",
        "verifier_*.py",
        "*verifier*.pyc",
    ]
    selector = "\\( " + " -o ".join(
        f"-name {shlex.quote(name)}" for name in names
    ) + " \\)"
    commands = []
    for target in targets:
        quoted = shlex.quote(target)
        commands.append(
            f"find {quoted} -type f {selector} -exec rm -f {{}} \\; "
            f"2>/dev/null; remaining=$(find {quoted} -type f {selector} "
            f"-print -quit 2>/dev/null); test -z \"$remaining\""
        )
    return " && ".join(commands)


def _windows_privileged_scrub_command(
    targets: list[str], extra_names: set[str] | None = None
) -> str:
    privileged_names = (*_PRIVILEGED_GUEST_NAMES, *sorted(extra_names or ()))
    names = ",".join(f"'{name}'" for name in privileged_names)
    commands = [f"$names=@({names})"]
    for raw_target in targets:
        target = raw_target.replace("/", "\\")
        if target.startswith("\\"):
            target = "C:" + target
        target = target.replace("'", "''")
        commands.append(
            f"$bad=Get-ChildItem -LiteralPath '{target}' -Recurse -File "
            f"-ErrorAction SilentlyContinue | Where-Object {{ "
            f"$names -contains $_.Name -or $_.Name -like '*_verifier.py' "
            f"-or $_.Name -like 'verifier_*.py' -or $_.Name -like '*verifier*.pyc' }}; "
            f"$bad | Remove-Item -Force; "
            f"$left=Get-ChildItem -LiteralPath '{target}' -Recurse -File "
            f"-ErrorAction SilentlyContinue | Where-Object {{ "
            f"$names -contains $_.Name -or $_.Name -like '*_verifier.py' "
            f"-or $_.Name -like 'verifier_*.py' -or $_.Name -like '*verifier*.pyc' }}; "
            f"if ($left) {{ throw 'privileged evaluator material remains in guest' }}"
        )
    return "; ".join(commands)


def _success_material_names(env: Any) -> set[str]:
    """Find file references in the selected task's verifier specification."""
    success = getattr(getattr(env, "task_spec", None), "success", None)
    spec = getattr(success, "spec", None)
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            reference = value.split("::", 1)[0]
            suffix = Path(reference).suffix.lower()
            looks_like_file = (
                "::" in value
                or "/" in reference
                or "\\" in reference
                or suffix in {
                    ".py", ".json", ".yaml", ".yml", ".png", ".jpg", ".jpeg"
                }
            )
            if looks_like_file and ":" not in reference:
                name = PureWindowsPath(reference).name
                if name:
                    names.add(name)

    visit(spec)
    return names


def scrub_guest_privileged_material(env: Any) -> list[str]:
    """Remove evaluator/rubric material copied into the interactive guest.

    This runs immediately after ``env.reset`` (therefore after pre-task
    setup) and before an instruction is exposed or the clock can arm. A
    failed deletion aborts the run rather than evaluating a guest that can
    inspect its own checker.
    """
    mounts = getattr(getattr(env, "env_spec", None), "mounts", None) or []
    targets = sorted({
        str(getattr(mount, "target", "") or "")
        for mount in mounts
        if getattr(mount, "target", None)
    })
    if not targets:
        return []
    runner = getattr(env, "_runner", None)
    execute = getattr(runner, "exec", None)
    if not callable(execute):
        raise RuntimeError(
            "cannot scrub guest evaluator material: runner has no exec channel"
        )
    env_os = str(getattr(env.env_spec, "os_type", "") or "").lower()
    is_windows = bool(getattr(runner, "is_windows", False)) or env_os == "windows"
    success_names = _success_material_names(env)
    command = (
        _windows_privileged_scrub_command(targets, success_names)
        if is_windows
        else _posix_privileged_scrub_command(targets, success_names)
    )
    try:
        rc = execute(command, use_pty=False, timeout=120)
    except TypeError:
        rc = execute(command)
    if rc not in (None, 0):
        raise RuntimeError(
            "failed closed: privileged evaluator material remains in guest "
            f"mounts (runner exit {rc})"
        )
    return targets


class GymAnythingAdapter(EnvAdapter):
    def __init__(self, env: Any, settle_sec: float):
        self._env = env
        self._settle_sec = settle_sec
        self._env_step_accepts_settle_sec = True
        self._verifier_report: dict[str, Any] | None = None
        self._patch_keyboard_hold_actions()

    def observe(self) -> Observation:
        obs = self._env.capture_observation()
        screen = obs.get("screen", {})
        png = b""
        if screen.get("png_b64"):
            png = base64.b64decode(screen["png_b64"])
        elif screen.get("path"):
            png = Path(screen["path"]).read_bytes()
        meta = {k: v for k, v in screen.items() if k not in ("png_b64", "path")}
        if "ui_tree" in obs:
            meta["ui_tree"] = obs["ui_tree"]
        return Observation(png=png, meta=meta)

    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        # wait_between_actions=0 and an explicit settle keep environment step
        # time honest: the only pauses are the ones the task actually needs,
        # not gym-anything's default 2-second-per-step tax. Agents insert
        # their own wait actions when the UI needs longer to settle.
        _obs, _reward, done, info = self._step_env(actions)
        report = info.get("verifier") if done and isinstance(info, dict) else None
        if isinstance(report, dict):
            # A redundant step after gym-anything finalizes returns no report.
            # Never erase the real verdict captured by the finalizing step.
            self._verifier_report = report
        return {"done": done}

    def _step_env(self, actions: list[dict[str, Any]]) -> tuple[Any, float, bool, dict[str, Any]]:
        if self._env_step_accepts_settle_sec:
            try:
                return self._env.step(
                    actions, wait_between_actions=0.0, settle_sec=self._settle_sec
                )
            except TypeError as exc:
                if "settle_sec" not in str(exc):
                    raise
                self._env_step_accepts_settle_sec = False
        return self._env.step(actions, wait_between_actions=0.0)

    def _patch_keyboard_hold_actions(self) -> None:
        runner = getattr(self._env, "_runner", None)
        inject_action = getattr(runner, "inject_action", None)
        if runner is None or not callable(inject_action):
            return
        if getattr(runner, "_cs_keyboard_hold_patched", False):
            return

        def inject_action_with_keyboard_hold(action: dict[str, Any]) -> None:
            if not isinstance(action, dict):
                inject_action(action)
                return
            keyboard = action.get("keyboard")
            if not isinstance(keyboard, dict) or not (
                "key_down" in keyboard or "key_up" in keyboard
            ):
                inject_action(action)
                return

            remaining = dict(action)
            remaining_keyboard = dict(keyboard)
            key_down = remaining_keyboard.pop("key_down", None)
            key_up = remaining_keyboard.pop("key_up", None)
            if remaining_keyboard:
                remaining["keyboard"] = remaining_keyboard
            else:
                remaining.pop("keyboard", None)

            if key_down is not None:
                self._inject_keyboard_hold(runner, key_down, down=True)
            if self._action_has_payload(remaining):
                inject_action(remaining)
            if key_up is not None:
                self._inject_keyboard_hold(runner, key_up, down=False)

        runner.inject_action = inject_action_with_keyboard_hold
        runner._cs_keyboard_hold_patched = True

    @staticmethod
    def _action_has_payload(action: dict[str, Any]) -> bool:
        return any(key != "_role" for key in action)

    @staticmethod
    def _normalize_hold_key(runner: Any, key: Any) -> str:
        text = str(key).strip()
        if not text:
            return ""
        normalize = getattr(runner, "_normalize_key_name", None)
        if callable(normalize):
            return str(normalize(text))
        return text

    def _inject_keyboard_hold(self, runner: Any, key: Any, *, down: bool) -> None:
        if getattr(runner, "_cs_pyautogui_keyboard_patched", False):
            runner._run_guest_python(keyboard_script({"keys_down" if down else "keys_up": [key]}))
            return
        key_name = self._normalize_hold_key(runner, key)
        if not key_name:
            return

        pyautogui_client = getattr(runner, "_pyautogui_client", None)
        if pyautogui_client is not None:
            method = getattr(pyautogui_client, "key_down" if down else "key_up", None)
            if callable(method):
                method(key_name)
                return

        run_pyautogui = getattr(runner, "_run_pyautogui", None)
        if callable(run_pyautogui):
            function = "keyDown" if down else "keyUp"
            run_pyautogui([f"pyautogui.{function}({json.dumps(key_name)})"])
            return

        exec_command = getattr(runner, "exec", None)
        if callable(exec_command):
            operation = "keydown" if down else "keyup"
            exec_command(f"xdotool {operation} {shlex.quote(key_name)}")
            return

        raise RuntimeError(
            "keyboard key_down/key_up actions need pyautogui or xdotool runner support"
        )

    def read_state(self) -> str:
        """Dump the current UI hierarchy as XML, host-side, for a seeded
        checker. Uses the runner's device shell; the dump is pulled to the
        host and never leaves an answer on the device."""
        runner = getattr(self._env, "_runner", None)
        adb = getattr(runner, "_adb_command", None)
        if adb is None:
            raise NotImplementedError("this runner does not expose adb for read_state")
        adb(["shell", "uiautomator dump /sdcard/_cs_ui.xml"], timeout=30)
        res = adb(["shell", "cat /sdcard/_cs_ui.xml"], timeout=30)
        out = res.stdout or b""
        return out.decode(errors="replace") if isinstance(out, bytes) else out

    def exec_read(self, command: str) -> str:
        """Run a shell command in the environment and return its stdout, for a
        seeded checker that inspects filesystem or command state (e.g. Linux)."""
        runner = getattr(self._env, "_runner", None)
        exec_capture = getattr(runner, "exec_capture", None)
        if exec_capture is None:
            raise NotImplementedError("this runner does not expose exec_capture")
        out = exec_capture(command)
        return out.decode(errors="replace") if isinstance(out, bytes) else str(out)

    def finalize(self) -> Verdict:
        if self._verifier_report is None:
            # The agent never marked done through gym-anything's own
            # mechanism; finalize the episode now to run the verifier.
            _obs, _reward, _done, info = self._env.step([], mark_done=True)
            if isinstance(info, dict):
                self._verifier_report = info.get("verifier")
        report = self._verifier_report or {}
        if report.get("error"):
            raise RuntimeError(str(report["error"]))
        return Verdict(
            passed=bool(report.get("passed", False)),
            score=report.get("score"),
            detail=str({k: v for k, v in report.items() if k != "passed"}),
        )

    def close(self) -> None:
        # CUA Speedrun owns finalization through ``finalize`` above.  Gym
        # Anything's close fallback would otherwise run the verifier again
        # when that call timed out, delaying the fresh-instance retry.
        self._env._finalized = True
        self._env.close()


class GymAnythingBackend(Backend):
    def __init__(
        self,
        runner: str | None = None,
        *,
        name: str | None = None,
        require_runner: str | None = None,
    ):
        os.environ.setdefault(
            "GYM_ANYTHING_QEMU_CACHE",
            str(Path.home() / ".cache/gym-anything/qemu"),
        )
        self.runner = runner
        self.name = name or ("gym-anything-modal" if runner == "modal" else "gym-anything")
        self.require_runner = require_runner
        self.observed_runner_name: str | None = None
        self._validated_qemu_images: set[tuple[Any, ...]] = set()

    def _record_runner(self, actual_runner: str) -> None:
        if self.runner == "qemu" and actual_runner not in {
            "QemuApptainerRunner",
            "QemuNativeRunner",
        }:
            raise RuntimeError(
                "gym-anything's qemu selector returned an unexpected runner: "
                f"{actual_runner}"
            )
        if (
            self.observed_runner_name is not None
            and self.observed_runner_name != actual_runner
        ):
            raise RuntimeError(
                "gym-anything changed local runner within one evaluation: "
                f"{self.observed_runner_name} -> {actual_runner}"
            )
        self.observed_runner_name = actual_runner

    def _load_adapter_extension(
        self,
        *,
        env: Any,
        env_spec: dict[str, Any],
        env_dir: str,
        workdir: Path,
        settle_sec: float,
    ) -> PreparedEnv | None:
        """Load an optional environment-owned adapter after Gym reset.

        The entrypoint is a Python file inside ``env_dir`` plus a callable,
        written as ``relative/path.py:function``. The callable receives the
        live Gym environment and returns ``PreparedEnv``. This keeps a
        benchmark's canonical setup/observation/evaluation lifecycle beside
        that environment instead of adding benchmark branches here.
        """
        entrypoint = env_spec.get("adapter_entrypoint")
        if not entrypoint:
            return None
        if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
            raise ValueError(
                "adapter_entrypoint must be 'relative/path.py:function'"
            )
        path_text, function_name = entrypoint.split(":", 1)
        root = Path(env_dir).resolve()
        path = (root / path_text).resolve()
        if not path.is_relative_to(root):
            raise ValueError("adapter_entrypoint escapes env_dir")
        if not path.is_file():
            raise FileNotFoundError(f"adapter entrypoint does not exist: {path}")
        if not function_name.isidentifier():
            raise ValueError(
                f"adapter entrypoint has an invalid function name: {function_name!r}"
            )

        module_name = (
            "cua_speedrun_env_adapter_"
            + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load environment adapter: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise ValueError(
                f"environment adapter function {function_name!r} is missing: {path}"
            )
        prepared = function(
            env=env,
            env_spec=env_spec,
            env_dir=root,
            workdir=workdir,
            settle_sec=settle_sec,
        )
        if not isinstance(prepared, PreparedEnv):
            raise TypeError(
                f"environment adapter {entrypoint} must return PreparedEnv"
            )
        if not isinstance(prepared.adapter, EnvAdapter):
            raise TypeError(
                f"environment adapter {entrypoint} returned an invalid adapter"
            )
        return prepared

    def preflight(self, benchmark: Any) -> None:
        """Fail before expensive submission init when the forced runner cannot
        possibly start on this host."""
        if self.runner not in ("qemu", "qemu_native"):
            return

        self._validate_local_kvm()
        self._validate_benchmark_qemu_image_config(benchmark)
        self._ensure_qemu_base_image(benchmark)

    def _validate_local_kvm(self) -> None:
        if sys.platform != "linux" or os.environ.get("CS_ALLOW_QEMU_TCG"):
            return
        if not os.path.exists("/dev/kvm"):
            raise RuntimeError(
                "gym-anything-local needs /dev/kvm for accelerated local VMs. "
                "Install/enable KVM, or set CS_ALLOW_QEMU_TCG=1 to allow "
                "very slow software emulation."
            )
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise RuntimeError(
                "gym-anything-local found /dev/kvm but this user cannot open "
                "it. Add the user to the kvm group or grant an ACL, e.g. "
                "'sudo setfacl -m u:$USER:rw /dev/kvm'. Set "
                "CS_ALLOW_QEMU_TCG=1 only if slow software emulation is OK."
            )

    def _validate_benchmark_qemu_image_config(self, benchmark: Any) -> None:
        seen: set[str] = set()
        for task in getattr(benchmark, "tasks", []):
            env_spec = getattr(task, "env", None)
            if not isinstance(env_spec, dict):
                continue
            if "env_dir" not in env_spec or "task_id" not in env_spec:
                continue
            env_dir = os.path.expanduser(os.path.expandvars(str(env_spec["env_dir"])))
            if env_dir in seen:
                continue
            seen.add(env_dir)
            env_config = self._load_env_config(env_dir)
            self._configured_qemu_base_image(env_dir, env_config)

    def _ensure_qemu_base_image(self, benchmark: Any) -> None:
        """Create the shared native-QEMU base image once before workers fan out.

        gym-anything's checkpoint paths are concurrency-safe, but the native
        base-image creation path is not. A first local run with multiple agents
        per evaluation can otherwise race on the same base_*.qcow2 path.
        """
        env_spec = self._first_gym_anything_env(benchmark)
        if env_spec is None:
            return

        try:
            import fcntl
            from gym_anything import from_config
        except ImportError as exc:
            raise RuntimeError(
                "the gym-anything backend needs gym-anything installed: "
                "pip install -e /path/to/gym-anything-primerl"
            ) from exc

        env_dir = os.path.expanduser(os.path.expandvars(str(env_spec["env_dir"])))
        runner = self.runner or env_spec.get("runner")
        if not runner:
            raise RuntimeError(
                f"{self.name} needs an explicit gym-anything runner; use "
                "'gym-anything-local' for the QEMU family or "
                "'gym-anything-qemu-native' to force native QEMU"
            )

        previous_runner = os.environ.get("GYM_ANYTHING_RUNNER")
        os.environ["GYM_ANYTHING_RUNNER"] = runner
        env = None
        try:
            env = from_config(env_dir, task_id=env_spec["task_id"])
            actual_runner = env.runner_name
            self._record_runner(actual_runner)
            if self.require_runner and actual_runner != self.require_runner:
                raise RuntimeError(
                    f"{self.name} expected gym-anything runner {self.require_runner}, "
                    f"but selected {actual_runner}"
                )
            self._apply_qemu_env_overrides(env, env_dir)
            runner_obj = getattr(env, "_runner", None)
            base_qcow2 = getattr(runner_obj, "base_qcow2", None)
            create_base_qcow2 = getattr(runner_obj, "_create_base_qcow2", None)
            if base_qcow2 is None or create_base_qcow2 is None:
                return

            base_path = Path(base_qcow2)
            lock_path = base_path.parent / f"{base_path.name}.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if base_path.exists() and base_path.stat().st_size > 0:
                    return
                print(
                    f"[gym-anything-local] preparing QEMU base image at {base_path} "
                    "before starting concurrent tasks",
                    flush=True,
                )
                create_base_qcow2()
                if not (base_path.exists() and base_path.stat().st_size > 0):
                    raise RuntimeError(
                        f"gym-anything-local did not create QEMU base image: {base_path}"
                    )
                print(
                    f"[gym-anything-local] QEMU base image ready: {base_path}",
                    flush=True,
                )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if previous_runner is None:
                os.environ.pop("GYM_ANYTHING_RUNNER", None)
            else:
                os.environ["GYM_ANYTHING_RUNNER"] = previous_runner

    def _first_gym_anything_env(self, benchmark: Any) -> dict[str, Any] | None:
        for task in getattr(benchmark, "tasks", []):
            env_spec = getattr(task, "env", None)
            if not isinstance(env_spec, dict):
                continue
            if "env_dir" in env_spec and "task_id" in env_spec:
                return env_spec
        return None

    def _load_env_config(self, env_dir: str) -> dict[str, Any]:
        config_path = Path(env_dir) / "env.json"
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _resolve_env_path(self, env_dir: str, raw_path: Any) -> Path:
        path = Path(os.path.expanduser(os.path.expandvars(str(raw_path))))
        if not path.is_absolute():
            path = Path(env_dir) / path
        return path.resolve()

    def _qemu_base_image_required(self, env_config: dict[str, Any]) -> bool:
        return bool(
            env_config.get("qemu_require_base_image")
            or env_config.get("qemu_required_base_image")
        )

    def _configured_qemu_base_image(
        self, env_dir: str, env_config: dict[str, Any]
    ) -> Path | None:
        base_image = env_config.get("qemu_base_image")
        if not base_image:
            if self._qemu_base_image_required(env_config):
                env_id = env_config.get("id") or env_dir
                raise RuntimeError(
                    f"{env_id} requires qemu_base_image, refusing to run on "
                    "gym-anything's stock Ubuntu image"
                )
            return None

        image_format = str(env_config.get("qemu_base_format", "qcow2")).lower()
        if image_format != "qcow2":
            raise RuntimeError(
                "gym-anything-local currently supports qemu_base_image only "
                f"for qcow2 images, got {image_format!r}"
            )
        image_path = self._resolve_env_path(env_dir, base_image)
        if not image_path.exists():
            raise RuntimeError(f"configured qemu_base_image does not exist: {image_path}")
        self._validate_qemu_base_image_provenance(image_path, env_config)
        return image_path

    def _validate_qemu_base_image_provenance(
        self, image_path: Path, env_config: dict[str, Any]
    ) -> None:
        """Verify a benchmark-declared image recipe before any VM boots.

        The image path is an operator choice, while the required recipe is
        benchmark data.  A sidecar binds the chosen file to that recipe and to
        its full content hash, preventing a similarly named image from silently
        changing the desktop contract.
        """
        expected = env_config.get("qemu_base_image_provenance")
        if expected is None:
            return
        if not isinstance(expected, dict) or not expected:
            raise RuntimeError(
                "qemu_base_image_provenance must be a non-empty object"
            )

        provenance_path = Path(f"{image_path}.provenance.json")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"configured QEMU image is missing its provenance sidecar: "
                f"{provenance_path}. Prepare the image with the benchmark's "
                "documented image builder and copy both files together."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"configured QEMU image has invalid provenance JSON: "
                f"{provenance_path}"
            ) from exc
        if not isinstance(provenance, dict):
            raise RuntimeError(
                f"configured QEMU image provenance must be an object: "
                f"{provenance_path}"
            )

        mismatches = {
            key: {"expected": value, "actual": provenance.get(key)}
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "configured QEMU image was built with the wrong contract: "
                f"{mismatches}"
            )

        recorded_sha = provenance.get("final_image_sha256")
        if not (
            isinstance(recorded_sha, str)
            and len(recorded_sha) == 64
            and all(character in "0123456789abcdef" for character in recorded_sha)
        ):
            raise RuntimeError(
                f"configured QEMU image provenance has no valid "
                f"final_image_sha256: {provenance_path}"
            )

        image_stat = image_path.stat()
        provenance_stat = provenance_path.stat()
        cache_key = (
            str(image_path),
            image_stat.st_size,
            image_stat.st_mtime_ns,
            provenance_stat.st_size,
            provenance_stat.st_mtime_ns,
            json.dumps(expected, sort_keys=True, separators=(",", ":")),
        )
        if cache_key in self._validated_qemu_images:
            return

        digest = hashlib.sha256()
        with image_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != recorded_sha:
            raise RuntimeError(
                "configured QEMU image does not match its provenance: "
                f"expected sha256 {recorded_sha}, got {actual_sha}"
            )
        self._validated_qemu_images.add(cache_key)

    def _verify_qemu_base_image(self, runner: Any, expected_path: Path) -> None:
        actual_raw = getattr(runner, "base_qcow2", None)
        if actual_raw is None:
            raise RuntimeError(
                "configured qemu_base_image cannot be enforced because this "
                "runner does not expose base_qcow2"
            )
        actual_path = Path(actual_raw).expanduser().resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                "QEMU runner is configured with the wrong base image: "
                f"expected {expected_path}, got {actual_path}"
            )
        runner._cs_effective_qemu_base_image = str(actual_path)

    def _apply_qemu_env_overrides(self, env: Any, env_dir: str) -> None:
        """Apply local QEMU fields gym-anything may ignore in older installs.

        Some pinned gym-anything revisions parse generic fields such as ssh but
        hardcode Linux QEMU to their stock Ubuntu image and ga/password123. The
        OSWorld import needs a prebuilt qcow2 with user/password instead.
        """
        runner = getattr(env, "_runner", None)
        if runner is None:
            return

        env_config = self._load_env_config(env_dir)
        image_path = self._configured_qemu_base_image(env_dir, env_config)
        if image_path is not None:
            if not hasattr(runner, "base_qcow2"):
                raise RuntimeError(
                    "configured qemu_base_image cannot be applied because this "
                    "runner does not expose base_qcow2"
                )
            runner.base_qcow2 = image_path
            runner._cs_configured_qemu_base_image = str(image_path)
            self._salt_qemu_checkpoint_key(runner, image_path)
            self._verify_qemu_base_image(runner, image_path)

        x11_display = env_config.get("qemu_x11_display")
        if x11_display:
            runner._cs_qemu_x11_display = str(x11_display)
            os.environ["GYM_ANYTHING_QEMU_X11_DISPLAY"] = str(x11_display)

        ssh_cfg = env_config.get("ssh")
        if isinstance(ssh_cfg, dict):
            user = ssh_cfg.get("user")
            password = ssh_cfg.get("password")
            if user:
                runner._ssh_user = str(user)
            if password is not None:
                runner._ssh_password = str(password)
            self._patch_qemu_password_sudo(runner)
            self._patch_qemu_file_transfer(runner)

    def _patch_qemu_password_sudo(self, runner: Any) -> None:
        if getattr(runner, "_cs_password_sudo_patched", False):
            return

        user = str(getattr(runner, "_ssh_user", "") or "")
        password = getattr(runner, "_ssh_password", None)
        if not user or password is None:
            return
        password_text = str(password)
        sudo_password_format = shlex.quote("%s\\n")

        def rewrite_command(command: str) -> str:
            rewritten = str(command)
            if user != "ga":
                rewritten = rewritten.replace("sudo chown ga:ga ", f"sudo chown {user}:{user} ")
                rewritten = rewritten.replace("/home/ga/", f"/home/{user}/")
            x11_display = getattr(runner, "_cs_qemu_x11_display", None)
            if x11_display:
                rewritten = rewritten.replace("DISPLAY=:1", f"DISPLAY={x11_display}")
            stripped = rewritten.lstrip()
            leading = rewritten[: len(rewritten) - len(stripped)]
            if stripped.startswith("sudo "):
                rest = stripped[len("sudo "):]
                return (
                    f"{leading}printf {sudo_password_format} {shlex.quote(password_text)} | "
                    f"sudo -S -p '' {rest}"
                )
            return rewritten

        original_run_ssh_cmd = getattr(runner, "_run_ssh_cmd", None)
        if callable(original_run_ssh_cmd):

            def run_ssh_cmd_with_password_sudo(
                port: int, cmd: str, *args: Any, **kwargs: Any
            ) -> Any:
                return original_run_ssh_cmd(port, rewrite_command(cmd), *args, **kwargs)

            runner._run_ssh_cmd = run_ssh_cmd_with_password_sudo

        original_ssh_command = getattr(runner, "_ssh_command", None)
        if callable(original_ssh_command):

            def ssh_command_with_password_sudo(
                cmd: str, *args: Any, **kwargs: Any
            ) -> Any:
                return original_ssh_command(rewrite_command(cmd), *args, **kwargs)

            runner._ssh_command = ssh_command_with_password_sudo

        runner._cs_password_sudo_patched = True

    def _patch_qemu_file_transfer(self, runner: Any) -> None:
        if getattr(runner, "_cs_file_transfer_patched", False):
            return

        user = str(getattr(runner, "_ssh_user", "") or "")
        password = getattr(runner, "_ssh_password", None)
        if not user or password is None or user == "ga":
            return
        password_text = str(password)

        original_sftp_connect = getattr(runner, "_sftp_connect", None)
        if callable(original_sftp_connect):

            def sftp_connect_with_configured_user() -> Any:
                import paramiko

                last_err = None
                for attempt in range(4):
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    try:
                        client.connect(
                            "localhost",
                            port=runner.ssh_port,
                            username=user,
                            password=password_text,
                            timeout=30,
                            banner_timeout=30,
                            auth_timeout=30,
                            look_for_keys=False,
                        )
                        return client
                    except Exception as exc:
                        last_err = exc
                        try:
                            client.close()
                        except Exception:
                            pass
                        if attempt < 3:
                            time.sleep(2 * (attempt + 1))
                raise last_err

            runner._sftp_connect = sftp_connect_with_configured_user

        original_copy_from = getattr(runner, "copy_from", None)
        sftp_copy_from = getattr(runner, "_sftp_copy_from", None)
        if callable(original_copy_from) and callable(sftp_copy_from):

            def copy_from_with_configured_user(container_src: str, host_dst: str) -> None:
                if getattr(runner, "is_android", False) or getattr(runner, "is_windows", False):
                    original_copy_from(container_src, host_dst)
                    return
                Path(host_dst).parent.mkdir(parents=True, exist_ok=True)
                if getattr(runner, "ssh_port", None):
                    sftp_copy_from(container_src, host_dst)

            runner.copy_from = copy_from_with_configured_user

        original_copy_to = getattr(runner, "copy_to", None)
        sftp_copy_to = getattr(runner, "_sftp_copy_to", None)
        if callable(original_copy_to) and callable(sftp_copy_to):

            def copy_to_with_configured_user(host_src: str, container_dst: str) -> None:
                if getattr(runner, "is_android", False) or getattr(runner, "is_windows", False):
                    original_copy_to(host_src, container_dst)
                    return
                if Path(host_src).exists() and getattr(runner, "ssh_port", None):
                    sftp_copy_to(host_src, container_dst)

            runner.copy_to = copy_to_with_configured_user

        runner._cs_file_transfer_patched = True

    def _salt_qemu_checkpoint_key(self, runner: Any, image_path: Path) -> None:
        env_hash = getattr(runner, "_cs_original_env_hash", None)
        if env_hash is None:
            env_hash = str(getattr(runner, "env_hash", ""))
            runner._cs_original_env_hash = env_hash

        stat = image_path.stat()
        salt_src = f"{image_path}:{stat.st_size}:{stat.st_mtime_ns}"
        salt = hashlib.sha256(salt_src.encode("utf-8")).hexdigest()[:16]
        runner.env_hash = f"{env_hash}_{salt}" if env_hash else salt

        checkpoint = getattr(runner, "env_checkpoint", None)
        if checkpoint is not None:
            checkpoint_dir = Path(checkpoint).parent
            runner.env_checkpoint = checkpoint_dir / f"checkpoint_{runner.env_hash}.qcow2"

    def _apply_qemu_volume_size(
        self,
        env: Any,
        requested_gb: Any,
        workdir: Path,
    ) -> None:
        if requested_gb is None:
            return
        requested = int(requested_gb)
        if requested <= 0:
            raise ValueError("qemu_volume_size_gb must be positive")
        runner = getattr(env, "_runner", None)
        base = Path(getattr(runner, "base_qcow2", ""))
        qemu_img = getattr(runner, "_run_qemu_img", None)
        if not base.is_file() or not callable(qemu_img):
            raise RuntimeError(
                "qemu_volume_size_gb requires a QEMU runner with a base image"
            )
        info_result = qemu_img(["info", "--output=json", str(base)])
        if info_result.returncode != 0:
            raise RuntimeError(f"qemu-img info failed: {info_result.stderr}")
        info = json.loads(info_result.stdout)
        requested_bytes = requested * 1024**3
        runner._cs_requested_qemu_volume_size_gb = requested
        if int(info["virtual-size"]) >= requested_bytes:
            return

        expanded = Path(workdir) / f"qemu-base-{requested}g.qcow2"
        expanded.unlink(missing_ok=True)
        create_result = qemu_img(
            [
                "create",
                "-f",
                "qcow2",
                "-b",
                str(base.resolve()),
                "-F",
                str(info["format"]),
                str(expanded),
                f"{requested}G",
            ]
        )
        if create_result.returncode != 0:
            raise RuntimeError(f"qemu-img resize overlay failed: {create_result.stderr}")
        runner.base_qcow2 = expanded
        runner._cs_effective_qemu_base_image = str(expanded)
        self._salt_qemu_checkpoint_key(runner, expanded)

    def prepare(self, env_spec: dict[str, Any], seed: int, workdir: Path) -> PreparedEnv:
        try:
            from gym_anything import from_config
        except ImportError as exc:
            raise RuntimeError(
                "the gym-anything backend needs gym-anything installed: "
                "pip install -e /path/to/gym-anything-primerl"
            ) from exc

        for key in ("env_dir", "task_id"):
            if key not in env_spec:
                raise ValueError(f"gym-anything env block needs '{key}'")

        # env_dir may use environment variables, e.g. ${GYM_ANYTHING_ROOT}/...,
        # so benchmark files stay free of machine-specific absolute paths.
        env_dir = os.path.expanduser(os.path.expandvars(str(env_spec["env_dir"])))

        t0 = time.monotonic()
        # Runner selection happens inside the env constructor, so it must be
        # steered BEFORE from_config. The backend's forced runner wins; a task
        # may only pin one when the backend did not.
        runner = self.runner or env_spec.get("runner")
        if not runner:
            raise RuntimeError(
                f"{self.name} needs an explicit gym-anything runner; use "
                "'gym-anything-local' for the QEMU family or "
                "'gym-anything-qemu-native' to force native QEMU"
            )
        os.environ["GYM_ANYTHING_RUNNER"] = runner
        config_overrides = env_spec.get("config_overrides")
        if config_overrides is not None and not isinstance(config_overrides, dict):
            raise ValueError("gym-anything config_overrides must be an object")
        env = from_config(
            env_dir,
            task_id=env_spec["task_id"],
            overrides=config_overrides,
        )
        actual_runner = env.runner_name
        self._record_runner(actual_runner)
        if self.require_runner and actual_runner != self.require_runner:
            try:
                env.close()
            finally:
                pass
            raise RuntimeError(
                f"{self.name} expected gym-anything runner {self.require_runner}, "
                f"but selected {actual_runner}"
            )
        try:
            if self.runner == "modal_native":
                from cua_speedrun.envs.modal_native_runtime import prepare

                prepare(env._runner, required_devices=env_spec.get('required_devices', []),
                        native_image=env_spec.get('native_image'))
            owner = os.environ.get("CS_NATIVE_OWNER")
            if self.runner == "modal_native" and owner:
                runner_obj = env._runner
                create_options = runner_obj._sandbox_create_kwargs

                def owned_options():
                    options = create_options()
                    options.setdefault("tags", {})["cs-parent"] = owner
                    return options

                runner_obj._sandbox_create_kwargs = owned_options
            self._apply_qemu_env_overrides(env, env_dir)
            self._apply_qemu_volume_size(
                env,
                env_spec.get("qemu_volume_size_gb"),
                workdir,
            )
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            raise
        if actual_runner == "LocalRunner":
            try:
                env.close()
            finally:
                pass
            raise RuntimeError(
                f"{self.name} selected gym-anything's LocalRunner, which is a "
                "blank smoke-test stub, not a VM. Install QEMU/KVM prerequisites "
                "or choose a real runner explicitly."
            )
        if (
            self.runner == "qemu_native"
            and sys.platform == "linux"
            and not os.environ.get("CS_ALLOW_QEMU_TCG")
            and getattr(getattr(env, "_runner", None), "_accel_type", None) != "kvm"
        ):
            try:
                env.close()
            finally:
                pass
            raise RuntimeError(
                "gym-anything-local selected QemuNativeRunner but could not "
                "use KVM. Restore /dev/kvm access for this user or set "
                "CS_ALLOW_QEMU_TCG=1 to allow very slow software emulation."
            )
        # Keep episode artifacts (frames, recordings, gym-anything's own
        # trajectory log) inside this run's directory. The env spec default
        # is a cwd-relative path that would scatter output across the repo.
        env.env_spec.recording.output_dir = str(workdir / "episode")
        try:
            # Optional environment-owned runner preparation, before reset can
            # load/create a checkpoint. No benchmark-specific behavior here.
            entrypoint = env_spec.get("prepare_entrypoint")
            if entrypoint:
                module_name, separator, function_name = str(entrypoint).partition(":")
                if not separator or not function_name.isidentifier():
                    raise ValueError("prepare_entrypoint must be 'module:function'")
                prepare_runner = getattr(importlib.import_module(module_name), function_name)
                prepare_runner(env)
            from cua_speedrun.envs.setup_hooks import checked_setup

            with checked_setup(env):
                env.reset(
                    seed=seed,
                    use_cache=bool(env_spec.get("use_cache", True)),
                    cache_level=env_spec.get("cache_level", "post_task"),
                )
            # The gateway owns wall-clock timing from arm.  This reset happens
            # while the environment may still wait in the untimed ready pool.
            max_steps = env_spec.get("max_steps", env.max_steps)
            env.set_episode_limits(max_steps=max_steps, timeout_sec=None)
            scrubbed_mounts = scrub_guest_privileged_material(env)
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            raise
        # CUA-World tasks put the agent-facing text in natural_language,
        # which may be a plain string or a dict with a "prompt" key.
        nl = getattr(env.task_spec, "natural_language", None)
        if isinstance(nl, dict):
            description = nl.get("prompt") or getattr(env.task_spec, "description", "")
        elif isinstance(nl, str) and nl.strip():
            description = nl
        else:
            description = getattr(env.task_spec, "description", "")
        # Per-task settle after each action batch, in seconds. Defaults low so
        # the environment step time reflects real UI latency, not a fixed tax;
        # a task that needs longer to settle can raise it in its env block.
        settle_sec = float(env_spec.get("action_settle_ms", 300)) / 1000.0
        try:
            # Patch the runner, not a benchmark adapter: extension adapters
            # (including OSWorld2) must use the same keyboard implementation.
            patch_runner_keyboard(env._runner)
            extension = self._load_adapter_extension(
                env=env,
                env_spec=env_spec,
                env_dir=env_dir,
                workdir=workdir,
                settle_sec=settle_sec,
            )
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            raise
        adapter = (
            extension.adapter
            if extension is not None
            else GymAnythingAdapter(env, settle_sec=settle_sec)
        )
        try:
            initial_observation = adapter.observe()
            if not initial_observation.png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(
                    "gym-anything environment did not produce a valid initial "
                    "PNG screenshot after reset"
                )
        except Exception:
            try:
                adapter.close()
            except Exception:
                pass
            raise
        return PreparedEnv(
            adapter=adapter,
            description=(
                extension.description if extension is not None else description
            ),
            prepare_time_sec=time.monotonic() - t0,
            checker=(extension.checker if extension is not None else None),
            info={
                "env_dir": str(env_spec["env_dir"]),
                "task_id": env_spec["task_id"],
                # The actual runner class that got selected, from the env
                # itself, so the run log records what really executed rather
                # than what the backend intended.
                "runner": env.runner_name,
                "privileged_mounts_scrubbed": scrubbed_mounts,
                "qemu_base_image": getattr(
                    getattr(env, "_runner", None), "_cs_effective_qemu_base_image", None
                ),
                **(extension.info if extension is not None else {}),
            },
        )
