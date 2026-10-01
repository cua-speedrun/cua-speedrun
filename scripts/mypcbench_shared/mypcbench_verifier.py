"""Input/output adapter around the unchanged canonical MyPCBench Gemini judge."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CANONICAL = Path(__file__).with_name("canonical")


class EvaluatorInfrastructure(RuntimeError):
    pass


class JudgeUnavailable(EvaluatorInfrastructure):
    pass


def canonical_judge() -> Path:
    path = CANONICAL / "osworld_full_traj_judge.py"
    expected = json.loads((CANONICAL / "provenance.json").read_text())["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise EvaluatorInfrastructure("canonical MyPCBench judge hash mismatch")
    return path


def native_trajectory(workdir: Path, traj: dict) -> dict:
    """Pair existing gateway screenshots with actions the native runner applied."""
    events = [
        json.loads(line)
        for line in (workdir / "runlog.jsonl").read_text().splitlines()
        if line.strip()
    ]
    timeline = [
        (event["t_mono_complete"], {"screenshot": str((workdir / event["frame"]).resolve())})
        for event in events if event.get("event") == "observe"
    ]
    # The gateway can also log a rejected concurrent step request. Use the
    # adapter's applied-action records, not requests, as the action evidence.
    timeline.extend(
        (step["t_mono"], {"action": step["action"]})
        for step in traj.get("steps", []) if step.get("event") == "step"
    )
    timeline.extend(
        (
            event.get("t_mono_complete", event.get("t_mono_arrive", 0.0)),
            {"raw_response_text": event["raw_response_text"]},
        )
        for event in events
        if isinstance(event.get("raw_response_text"), str)
        and event["raw_response_text"].strip()
    )
    evidence = []
    last_screenshot = None
    for _timestamp, entry in sorted(timeline, key=lambda item: item[0]):
        if "screenshot" in entry:
            last_screenshot = entry["screenshot"]
            evidence.append(entry)
        elif "raw_response_text" in entry and evidence:
            evidence[-1]["raw_response_text"] = entry["raw_response_text"]
        elif evidence and "action" not in evidence[-1]:
            evidence[-1]["action"] = entry["action"]
        else:
            if last_screenshot is not None:
                entry["screenshot"] = last_screenshot
            evidence.append(entry)
    return {"evidence": evidence}


def build_bundle(source: dict, traj: dict) -> dict:
    """Translate evidence and apply MyPCBench's uniform rubric aggregation."""
    grading = copy.deepcopy(source.get("grading") or {})
    rubrics = grading.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise EvaluatorInfrastructure("MyPCBench task has no grading rubrics")
    if any(not isinstance(rubric, dict) for rubric in rubrics):
        raise EvaluatorInfrastructure("MyPCBench task has an invalid grading rubric")
    uniform_weight = 1.0 / len(rubrics)
    for rubric in rubrics:
        rubric["weight"] = uniform_weight
    rows = [r for r in traj.get("steps", []) if r.get("event") == "step"]
    frames = list(traj.get("frames") or [])
    steps = []
    for index, row in enumerate(rows):
        entry = {"step_num": index, "parsed_action_obj": {"action": row.get("action")}}
        path = row.get("screenshot") or (frames[index] if index < len(frames) else None)
        if path:
            entry["screenshot"] = str(Path(path).resolve())
        steps.append(entry)
    # Keep additional observations and the final screen without inventing actions.
    extra = frames[len(rows) :]
    if traj.get("final_screenshot"):
        extra.append(traj["final_screenshot"])
    for path in extra:
        absolute = str(Path(path).resolve())
        if not any(row.get("screenshot") == absolute for row in steps):
            steps.append({"step_num": len(steps), "screenshot": absolute})
    if traj.get("evidence"):
        steps = [
            {
                "step_num": index,
                **({"screenshot": row["screenshot"]} if row.get("screenshot") else {}),
                **(
                    {"raw_response_text": row["raw_response_text"]}
                    if row.get("raw_response_text")
                    else {}
                ),
                "parsed_action_obj": {"action": row.get("action")},
            }
            for index, row in enumerate(traj["evidence"])
        ]
    paths = [Path(row["screenshot"]) for row in steps if row.get("screenshot")]
    if not paths or any(not path.is_file() for path in paths):
        raise EvaluatorInfrastructure(
            "MyPCBench trajectory screenshot evidence is missing"
        )
    return {
        "task": {"task_id": source["id"], "instruction": source["instruction"]},
        "grading_manifest": grading,
        "scoring_contract": {
            "rule": "uniform-rubric-mean@1",
            "rubric_count": len(rubrics),
            "weight_per_rubric": uniform_weight,
            "pass_condition": "all-rubrics",
        },
        "artifacts": {"steps": steps},
    }


def uniform_result(report: dict, rubric_count: int) -> dict:
    """Validate and expose the canonical labels under uniform aggregation."""
    if rubric_count < 1:
        raise EvaluatorInfrastructure(
            "uniform rubric scoring requires at least one rubric"
        )
    if not isinstance(report, dict):
        raise EvaluatorInfrastructure("canonical judge returned an invalid report")
    rubric_results = report.get("rubric_results", [])
    if (
        not isinstance(rubric_results, list)
        or len(rubric_results) != rubric_count
        or any(not isinstance(result, dict) for result in rubric_results)
    ):
        raise EvaluatorInfrastructure(
            "canonical judge returned an incomplete rubric report"
        )
    if any(not isinstance(result.get("success"), bool) for result in rubric_results):
        raise EvaluatorInfrastructure(
            "canonical judge returned an invalid rubric verdict"
        )
    expected_weight = 1.0 / rubric_count
    try:
        weights_are_uniform = all(
            abs(float(result.get("weight", 0.0)) - expected_weight) <= 1e-9
            for result in rubric_results
        )
    except (TypeError, ValueError):
        weights_are_uniform = False
    if not weights_are_uniform:
        raise EvaluatorInfrastructure(
            "canonical judge did not apply uniform rubric weights"
        )
    passed_rubrics = sum(bool(result.get("success")) for result in rubric_results)
    expected_score = int(round(100.0 * passed_rubrics / rubric_count))
    reported_score = report.get("score")
    if isinstance(reported_score, bool) or not isinstance(reported_score, (int, float)):
        raise EvaluatorInfrastructure("canonical judge returned an invalid score")
    if float(reported_score) != expected_score:
        raise EvaluatorInfrastructure(
            "canonical judge returned a non-uniform rubric score"
        )
    expected_passed = passed_rubrics == rubric_count
    if (
        not isinstance(report.get("passed"), bool)
        or report["passed"] != expected_passed
    ):
        raise EvaluatorInfrastructure(
            "canonical judge returned an inconsistent pass verdict"
        )
    return {
        "passed": expected_passed,
        "score": expected_score,
        "feedback": (
            f"{passed_rubrics}/{rubric_count} rubrics passed; "
            f"uniform fraction={passed_rubrics / rubric_count:.4f}"
        ),
        "metrics": rubric_results,
    }


def check_with_source(
    source_json: Path,
    traj: dict,
    env_info: dict,
    task_info: dict,
) -> dict:
    del task_info
    # Use a benchmark-scoped evaluator credential so Gemini-based submissions
    # can independently receive GEMINI_API_KEY.
    key = os.environ.get("MYPCBENCH_JUDGE_API_KEY", "").strip()
    if not key:
        raise JudgeUnavailable(
            "MYPCBENCH_JUDGE_API_KEY is required before MyPCBench runs"
        )
    judge = canonical_judge()
    source = json.loads(Path(source_json).read_text())
    if env_info.get("artifact_dir"):
        workdir = Path(env_info["artifact_dir"])
        traj = native_trajectory(workdir, traj)
        output = workdir / "episode" / "judge"
        output.mkdir(parents=True, exist_ok=True)
        final_image = output / "final.png"
        final_image.write_bytes(env_info["observe"]().png)
        traj["evidence"].append({"screenshot": str(final_image.resolve())})
        env_info = {**env_info, "judge_output_dir": str(output)}
    bundle = build_bundle(source, traj)
    first_image = next(
        row["screenshot"]
        for row in bundle["artifacts"]["steps"]
        if row.get("screenshot")
    )
    output = Path(
        env_info.get("judge_output_dir") or Path(first_image).parent / "judge"
    )
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "rubric_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    # Do not inherit mock controls, alternate models, agent keys, or Python hooks.
    environment = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "LANG", "SSL_CERT_FILE")
        if name in os.environ
    }
    environment.update(
        {
            "GEMINI_API_KEY": key,
            "MYPCBENCH_RUBRIC_JUDGE_MODEL": "gemini-3.1-flash-lite-preview",
            "MYPCBENCH_RUBRIC_BUNDLE_PATH": str(bundle_path.resolve()),
            "MYPCBENCH_RUBRIC_SAVE_DIR": str(output.resolve()),
        }
    )
    result_path = output / "osworld_full_traj_result.json"
    result_path.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, str(judge)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=1200,
        )
    except subprocess.TimeoutExpired as exc:
        raise JudgeUnavailable("canonical MyPCBench judge timed out") from exc
    if proc.returncode or not result_path.is_file():
        raise JudgeUnavailable("canonical MyPCBench judge failed; no verdict recorded")
    try:
        report = json.loads(result_path.read_text())
    except (ValueError, OSError) as exc:
        raise EvaluatorInfrastructure(
            "canonical judge returned an unreadable report"
        ) from exc
    rubric_count = len(source["grading"]["rubrics"])
    result = uniform_result(report, rubric_count)
    # Preserve the canonical report while classifying provider errors according
    # to the platform contract: infrastructure failures cannot score the agent.
    failures = [
        metric
        for metric in result["metrics"]
        if str(metric.get("reasoning", "")).startswith("Error judging rubric ")
    ]
    if failures:
        result_path.write_text(result_path.read_text().replace(key, "[REDACTED]"))
        raise JudgeUnavailable("canonical judge reported a provider error")
    if env_info.get("artifact_dir"):
        result["feedback"] = json.dumps(
            {"feedback": result["feedback"], "metrics": result["metrics"]}
        )
    return result
