"""Notebook dashboard helpers, checked against the bundled agents and task sets."""

from pathlib import Path

import pytest

from cua_speedrun.commands.benchmark import dataset_path
from cua_speedrun.notebook import (launch_args, parse_variables, save_upload, startup_line, status_html,
                                   variables_for)

AGENTS = Path(__file__).resolve().parents[1] / "agents"


def test_variables_come_from_agent_and_task_set():
    required, optional = variables_for(AGENTS / "claude", dataset_path("osworld-50"))
    assert required == ["ANTHROPIC_API_KEY"]
    required, optional = variables_for(AGENTS / "yutori_n2", dataset_path("my-pc-bench"))
    assert required == ["YUTORI_API_KEY", "MYPCBENCH_JUDGE_API_KEY"]
    assert "YUTORI_REASONING_EFFORT" in optional
    required, optional = variables_for(None, dataset_path("cua-world-26"))
    assert required == [] and "GEMINI_API_KEY" in optional


def test_upload_becomes_a_valid_agent(tmp_path):
    agent = (AGENTS / "claude" / "agent.py").read_bytes()
    folder = save_upload({"agent.py": agent}, tmp_path)
    assert (folder / "agent.py").read_bytes() == agent
    assert (folder / "init.py").read_text() == 'print("ready")\n'


@pytest.mark.parametrize("files", [{"init.py": b"x = 1\n"}, {"agent.py": b"", "notes.txt": b""},
                                   {"agent.py": b"def broken(:\n"}])
def test_bad_uploads_are_refused(files, tmp_path):
    with pytest.raises(ValueError):
        save_upload(files, tmp_path)


def test_other_variables():
    assert parse_variables("HF_TOKEN=abc\n\nMODEL = x=y\n") == {"HF_TOKEN": "abc", "MODEL": "x=y"}
    with pytest.raises(ValueError):
        parse_variables("HF_TOKEN")


def test_launch_uses_the_hosted_benchmark_command():
    args = launch_args("claude", "osworld-50", 2, ["HF_TOKEN"])
    assert (args.command, args.host, args.agent, args.dataset) == ("benchmark", "modal", "claude", "osworld-50")
    assert args.parallel_evaluations == 2 and args.env == ["HF_TOKEN"]


def test_status_uses_the_terminal_view():
    html = status_html({"run_id": "m-0123456789abcdef", "stage": "running", "benchmark": "osworld-50",
                        "submission": {"name": "<agent>"}, "progress": {"finished": 3, "total": 50, "passed": 2},
                        "tasks": [], "elapsed_sec": 65})
    assert html.startswith("<pre") and "m-0123456789abcdef" in html
    assert "<agent>" not in html


def test_launch_progress_reads_as_plain_steps():
    steps = {}
    assert startup_line("[prepare] import_desktop_image.py", steps) is None
    running = startup_line('[startup] {"key": "desktop", "label": "Prepare desktop image", "state": "running", "at": 1}', steps)
    assert running == "Prepare desktop image: running…"
    done = startup_line('[startup] {"key": "desktop", "label": "Prepare desktop image", "state": "done", "at": 2, "elapsed_sec": 47.2}', steps)
    assert done == "Prepare desktop image: done in 47 s"
