#!/usr/bin/env python3
"""Import OSWorld's test_all.json into cua-speedrun benchmark format.

The generated benchmark keeps the upstream JSON for every task. A shared
transport attaches the live desktop to the pinned upstream
``DesktopEnv.evaluate`` implementation.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import textwrap
from pathlib import Path


# Pin the source commit so repeated imports produce the same benchmark.
OSWORLD_COMMIT = "315a7603173feadf1b8a85cbc006c93ffe1dc1a1"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark-assets" / "osworld"
ENV = OUT / "environment"
SHARED_SRC = ROOT / "scripts" / "osworld_shared"
TASK_PATCHES = ROOT / "scripts" / "osworld_task_patches"
OSWORLD_IMAGE_CONTRACT = json.loads(
    (ROOT / "benchmarks" / "osworld-image.json").read_text(encoding="utf-8")
)
OSWORLD_IMAGE_SOURCE = OSWORLD_IMAGE_CONTRACT["official_source"]
OSWORLD_IMAGE_GUEST = OSWORLD_IMAGE_CONTRACT["guest"]


def task_verifier(desktop_user: str, desktop_home: str, x11_display: str) -> str:
    return f'''import os
from pathlib import Path
import importlib.util

os.environ.setdefault("OSWORLD_DESKTOP_USER", {json.dumps(desktop_user)})
os.environ.setdefault("OSWORLD_DESKTOP_HOME", {json.dumps(desktop_home)})
os.environ.setdefault("OSWORLD_X11_DISPLAY", {json.dumps(x11_display)})

_shared = Path(__file__).resolve().parents[1] / "_shared" / "osworld_verifier.py"
_spec = importlib.util.spec_from_file_location("_osworld_verifier", _shared)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def check(traj, env_info, task_info):
    return _mod.check_with_source(Path(__file__).with_name("source.json"), traj, env_info, task_info)
'''


def task_sources() -> tuple[list[str], dict[str, dict]]:
    bundle = json.loads((ROOT / "benchmarks" / "osworld-tasks.json").read_text())
    if bundle["source_commit"] != OSWORLD_COMMIT:
        raise ValueError("bundled OSWorld tasks do not match the pinned source commit")
    names = [task_name(domain, uid) for domain, ids in bundle["test_all"].items() for uid in ids]
    sources = bundle["sources"]
    if len(names) != 369 or len(set(names)) != len(names) or set(names) != set(sources):
        raise ValueError("bundled OSWorld task index is incomplete")
    return names, sources


def q(value: str) -> str:
    return json.dumps(value)


def task_name(domain: str, uid: str) -> str:
    return f"{domain}__{uid}"


def task_id(domain: str, uid: str) -> str:
    return f"osworld_{domain}_{uid}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def copy_shared_file(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SHARED_SRC / name, dest)


def main() -> None:
    names, sources = task_sources()
    if OUT.exists():
        shutil.rmtree(OUT)

    manifest = [
        "name: osworld",
        'version: "0.1"',
        f"source: https://github.com/xlang-ai/OSWorld/blob/{OSWORLD_COMMIT}/evaluation_examples/test_all.json",
        f"source_commit: {OSWORLD_COMMIT}",
        "tasks:",
    ]
    manifest.extend(f"  - tasks/{name}" for name in names)
    write(OUT / "manifest.yaml", "\n".join(manifest) + "\n")

    # Keep the generated benchmark portable across evaluator hosts. The local
    # backend expands this variable at execution time and fails closed if the
    # prepared OSWorld image is absent; it must never fall back to the stock
    # gym-anything Ubuntu desktop.
    qemu_base_image = "${OSWORLD_QEMU_BASE_IMAGE}"
    default_desktop_user = str(OSWORLD_IMAGE_GUEST["ssh_user"])
    desktop_user = default_desktop_user
    desktop_home = f"/home/{desktop_user}"
    x11_display = os.environ.get("OSWORLD_X11_DISPLAY", ":0")
    ssh_password = str(OSWORLD_IMAGE_GUEST["ssh_password"])

    env_json = {
        "id": "osworld.full@0.1",
        "version": "0.1",
        "description": (
            "Full OSWorld test_all.json imported from xlang-ai/OSWorld. "
            "Each generated task stores the upstream source JSON. The shared "
            "transport runs the pinned upstream DesktopEnv evaluator."
        ),
        "base": "ubuntu-gnome-systemd_highres",
        "resources": {"cpu": 4, "mem_gb": 6, "gpu": 0, "net": True},
        "observation": [{"type": "rgb_screen", "resolution": [1920, 1080]}],
        "action": [{"type": "mouse"}, {"type": "keyboard"}],
        "mounts": [{"source": "tasks", "target": "/workspace/tasks", "mode": "ro"}],
        # A cold QEMU boot leaves the path assumed by upstream's unchanged
        # osworld_server.service empty; restore the live GDM cookie there.
        "hooks": {
            "post_start": (
                "sudo install -o user -g user -m 600 "
                "/run/user/1000/gdm/Xauthority /home/user/.Xauthority && "
                "sudo systemctl restart osworld_server.service"
            )
        },
        "ssh": {"user": desktop_user, "password": ssh_password},
        "vnc": {
            "enable": True,
            "host_port": -1,
            "view_only": True,
            "password": "password",
        },
    }
    env_json["qemu_base_image"] = qemu_base_image
    env_json["qemu_base_format"] = os.environ.get("OSWORLD_QEMU_BASE_FORMAT", "qcow2")
    env_json["qemu_require_base_image"] = True
    env_json["qemu_base_image_provenance"] = {
        "schema_version": OSWORLD_IMAGE_CONTRACT["provenance_schema_version"],
        "recipe": OSWORLD_IMAGE_CONTRACT["recipe"],
        "source_revision": OSWORLD_IMAGE_SOURCE["revision"],
        "archive_sha256": OSWORLD_IMAGE_SOURCE["archive_sha256"],
        "source_image_sha256": OSWORLD_IMAGE_SOURCE["image_sha256"],
        "ssh_prepared": True,
        "tools_prepared": True,
        "nopasswd_sudo": True,
        "ssh_user": desktop_user,
        "guest_auth_prepared": True,
    }
    env_json["qemu_x11_display"] = x11_display
    write(ENV / "env.json", json.dumps(env_json, indent=2) + "\n")
    shared_dir = ENV / "tasks" / "_shared"
    copy_shared_file("osworld_setup.py", shared_dir / "osworld_setup.py")
    copy_shared_file("osworld_verifier.py", shared_dir / "osworld_verifier.py")

    for name in names:
        domain, uid = name.split("__", 1)
        source = sources[name]
        prompt = source.get("instruction") or ""
        snapshot = source.get("snapshot") or domain
        evaluator = source.get("evaluator") or {}
        result = evaluator.get("result") or {}
        if isinstance(result, list):
            result_type = ",".join(str(item.get("type")) for item in result if isinstance(item, dict))
        elif isinstance(result, dict):
            result_type = str(result.get("type"))
        else:
            result_type = ""

        write(
            OUT / "tasks" / name / "task.yaml",
            textwrap.dedent(
                f"""\
                task_id: {task_id(domain, uid)}
                description: {q("OSWorld " + domain + " task " + uid + " (instruction comes from the environment task spec)")}
                timeout_sec: 39600
                env:
                  kind: gym-anything
                  env_dir: ${{BENCHMARK_DIR}}/environment
                  task_id: {name}
                  use_cache: false
                metadata:
                  osworld_id: {uid}
                  osworld_domain: {domain}
                  osworld_snapshot: {snapshot}
                """
            ),
        )

        task_json = {
            "id": f"{name}@1",
            "env_id": "osworld.full@0.1",
            "difficulty": "unknown",
            "natural_language": {"prompt": prompt},
            "init": {"timeout_sec": 39600, "max_steps": 100, "reward_type": "sparse"},
            "hooks": {
                "pre_task": (
                    f"OSWORLD_DESKTOP_USER={shlex.quote(desktop_user)} "
                    f"OSWORLD_DESKTOP_HOME={shlex.quote(desktop_home)} "
                    f"OSWORLD_X11_DISPLAY={shlex.quote(x11_display)} "
                    f"python3 /workspace/tasks/_shared/osworld_setup.py "
                    f"/workspace/tasks/{name}/source.json"
                ),
                "pre_task_timeout": 600,
                "fail_on_error": True,
            },
            "success": {"mode": "program", "spec": {"program": "verifier.py::check"}},
            "metadata": {
                "osworld_id": uid,
                "osworld_domain": domain,
                "osworld_snapshot": snapshot,
                "osworld_source": source.get("source"),
                "osworld_evaluator_func": evaluator.get("func"),
                "osworld_result_type": result_type,
            },
        }
        write(ENV / "tasks" / name / "task.json", json.dumps(task_json, indent=2) + "\n")
        write(ENV / "tasks" / name / "source.json", json.dumps(source, indent=2) + "\n")
        patch = TASK_PATCHES / f"{name}.json"
        if patch.is_file():
            shutil.copy2(patch, ENV / "tasks" / name / "setup-patch.json")
        write(ENV / "tasks" / name / "verifier.py", task_verifier(desktop_user, desktop_home, x11_display))

    print(f"imported {len(names)} OSWorld tasks into {OUT}")


if __name__ == "__main__":
    main()
