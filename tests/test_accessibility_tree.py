"""An agent can ask for the front window's accessibility tree with an observation."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from cua_speedrun.client import Computer
from cua_speedrun.envs.base import EnvAdapter, Observation, Verdict
from cua_speedrun.gateway import Gateway
from cua_speedrun.runlog import RunLogWriter, load_runlog

TREE = {"app": "gedit", "window": "Untitled Document 1", "truncated": False, "nodes": [
    {"role": "push button", "name": "Save", "text": "", "x": 10, "y": 10, "w": 40, "h": 20,
     "focused": False, "editable": False, "checked": False, "selected": False, "enabled": True},
]}


class Desktop(EnvAdapter):
    def observe(self) -> Observation:
        output = BytesIO()
        Image.new("RGB", (2, 2)).save(output, format="PNG")
        return Observation(output.getvalue(), {"resolution": [2, 2]})

    def step(self, actions):
        return {}

    def finalize(self) -> Verdict:
        return Verdict(True, 100.0, "complete")

    def close(self) -> None:
        return None


def run(adapter: EnvAdapter, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(adapter, log, tmp_path, timeout_sec=10, grace_sec=0)
    computer = Computer(gateway.start())
    gateway.arm()
    return gateway, computer, log


def test_the_tree_comes_only_when_asked_and_is_kept_with_its_frame(tmp_path):
    class TreeDesktop(Desktop):
        def accessibility_tree(self):
            return TREE

    gateway, computer, log = run(TreeDesktop(), tmp_path)
    try:
        assert "accessibility_tree" not in computer.observe()["meta"]
        meta = computer.observe(accessibility_tree=True)["meta"]
        assert meta == {"resolution": [2, 2], "frame": "frame_00001.png", "accessibility_tree": TREE}
        computer.done()
        gateway.join(timeout=2)
        observes = [e for e in load_runlog(log.path) if e["event"] == "observe"]
        assert "accessibility_tree" not in observes[0]
        assert observes[1]["accessibility_tree"] == "frame_00001.a11y.json"
        assert json.loads((tmp_path / "frame_00001.a11y.json").read_text()) == TREE
    finally:
        gateway.shutdown()
        log.close()


def test_a_missing_or_failed_tree_is_not_an_environment_failure(tmp_path):
    class BrokenDesktop(Desktop):
        def accessibility_tree(self):
            raise RuntimeError("the AT-SPI bus is not running")

    for name, adapter in (("unsupported", Desktop()), ("broken", BrokenDesktop())):
        gateway, computer, log = run(adapter, tmp_path / name)
        try:
            tree = computer.observe(accessibility_tree=True)["meta"]["accessibility_tree"]
            assert "error" in tree and tree.get("unsupported", False) == (name == "unsupported")
            assert "infrastructure_error" not in gateway.status()
            assert not gateway.status()["finished"]
            computer.done()
            gateway.join(timeout=2)
            assert gateway.verdict.passed
        finally:
            gateway.shutdown()
            log.close()


def test_the_guest_walk_ships_as_source_that_compiles_on_the_host():
    source = (Path(__file__).resolve().parents[1] / "src/cua_speedrun/envs/_atspi_tree.py").read_text()
    compile(source + "\nmain()\n", "_atspi_tree", "exec")


def test_a_step_can_return_the_next_observation(tmp_path):
    class TreeDesktop(Desktop):
        def accessibility_tree(self):
            return TREE

    gateway, computer, log = run(TreeDesktop(), tmp_path)
    try:
        result = computer.step([{"keyboard": {"keys": ["esc"]}}], observe=True, accessibility_tree=True)
        assert result["ok"] is True and result["observation"]["png"].startswith(b"\x89PNG")
        assert result["observation"]["meta"]["accessibility_tree"] == TREE
        assert "observation" not in computer.step([{"keyboard": {"keys": ["esc"]}}])
        computer.done()
        gateway.join(timeout=2)
        events = [e for e in load_runlog(log.path) if e["event"] in ("step", "observe")]
        assert [e["event"] for e in events] == ["step", "observe", "step"]
        assert events[1]["accessibility_tree"] == "frame_00000.a11y.json"
        assert events[0]["t_mono_complete"] <= events[1]["t_mono_arrive"]
        assert gateway.status()["num_steps"] == 2
    finally:
        gateway.shutdown()
        log.close()
