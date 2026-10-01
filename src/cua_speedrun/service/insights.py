"""Read-only diagnostics for result and comparison pages.

Scoring remains the responsibility of :mod:`cua_speedrun.scoring`.  This
module turns stored card facts into a presentation model without changing a
verdict, recomputing a score, or comparing across a track/season boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cua_speedrun.service.frontier import pareto_entry_ids, parse_season_key


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seed_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _placement_label(placement: Mapping[str, Any]) -> str | None:
    mode = str(placement.get("mode") or "").lower()
    provider = placement.get("provider")
    if mode == "local":
        return "Local"
    if mode == "remote":
        return f"Remote · {str(provider).title()}" if provider else "Remote"
    return None


def clock_profile(agent_time_sec: Any, env_time_sec: Any) -> dict[str, Any]:
    """Describe clock attribution without claiming a causal bottleneck."""
    if agent_time_sec is None and env_time_sec is None:
        return {"available": False, "label": "not reported"}
    agent = max(_number(agent_time_sec) or 0.0, 0.0)
    environment = max(_number(env_time_sec) or 0.0, 0.0)
    total = agent + environment
    if total <= 0:
        return {"available": False, "label": "not reported"}
    agent_percent = round(100 * agent / total, 1)
    env_percent = round(100 * environment / total, 1)
    if agent_percent >= 60:
        label = "agent-heavy"
        detail = "Most attributed clock time was inside the agent loop."
    elif env_percent >= 60:
        label = "interaction-heavy"
        detail = "Most attributed clock time was act-and-observe latency."
    else:
        label = "balanced split"
        detail = "Agent and environment attribution are of similar size."
    return {
        "available": True,
        "label": label,
        "detail": detail,
        "agent_time_sec": agent,
        "env_time_sec": environment,
        "agent_percent": agent_percent,
        "env_percent": env_percent,
    }


def result_contract(
    result: Mapping[str, Any], *, fallback_track: str | None = None
) -> dict[str, Any]:
    """Expose the frozen comparison contract, including legacy gaps."""
    plan = dict(result.get("run_plan") or {})
    parts = parse_season_key(str(result.get("season_key") or ""))
    track = dict(plan.get("track") or {})
    benchmark = dict(plan.get("benchmark") or {})
    execution = dict(plan.get("execution") or {})
    topology = dict(execution.get("topology") or {})
    target = dict(execution.get("target") or {})
    scoring = dict(plan.get("scoring") or {})
    parallelism = dict(result.get("parallelism") or {})
    seeds = dict(plan.get("seeds") or {})
    agent_runtime = dict(plan.get("agent_runtime") or {})
    base_image = dict(agent_runtime.get("base_image") or {})

    benchmark_name = benchmark.get("name")
    benchmark_version = benchmark.get("version")
    benchmark_label = None
    if benchmark_name:
        benchmark_label = str(benchmark_name)
        if benchmark_version is not None:
            benchmark_label += f"@{benchmark_version}"
    benchmark_label = benchmark_label or parts.get("benchmark")
    contract_id = result.get("run_plan_hash") or parts.get("contract")
    success_bar = scoring.get("success_bar")
    if success_bar is None:
        success_bar = (result.get("rules") or {}).get("success_bar")

    compute = dict(topology.get("compute") or {})
    environment = dict(topology.get("environment") or {})
    if not compute or not environment:
        # Older contracts stored one combined placement. Use only its mode and
        # provider; a machine/site name is not part of the current UI model.
        legacy_placement = {
            "mode": target.get("mode"),
            "provider": target.get("provider"),
        }
        backend = str(plan.get("backend") or "")
        if not legacy_placement["mode"]:
            legacy_placement = (
                {"mode": "remote", "provider": "modal"}
                if backend == "modal-remote"
                else {"mode": "local", "provider": None}
            )
        compute = compute or legacy_placement
        environment = environment or legacy_placement

    fields = [
        ("Track", track.get("name") or parts.get("track") or fallback_track),
        ("Benchmark", benchmark_label),
        ("Model / agent", _placement_label(compute)),
        ("Environment VMs", _placement_label(environment)),
        (
            "Hardware",
            track.get("gpu")
            or ("CPU (no GPU)" if plan else parts.get("hardware")),
        ),
        ("Agent runtime", agent_runtime.get("recipe")),
        (
            "Agent base image",
            (
                f"{base_image.get('builder')} · Python "
                f"{base_image.get('observed_python_version')}"
            )
            if base_image.get("builder") and base_image.get("observed_python_version")
            else None,
        ),
        (
            "System packages",
            ", ".join(agent_runtime.get("system_packages") or ()),
        ),
        (
            "Python packages",
            ", ".join(agent_runtime.get("python_packages") or ()),
        ),
        (
            "Modal client",
            agent_runtime.get("modal_client_version")
            if agent_runtime.get("modal_client_version") != "not-used" else None,
        ),
        ("Eval algorithm", execution.get("algorithm") or parts.get("algorithm")),
        (
            "Agents / evaluation",
            execution.get("agents_per_evaluation")
            or execution.get("concurrency"),
        ),
        ("Parallel evaluations", parallelism.get("parallel_evaluations")),
        ("Active agents", parallelism.get("agent_concurrency")),
        ("Environment pool capacity", parallelism.get("env_pool_size")),
        ("Runs / task", execution.get("runs_per_task")),
        ("Seed policy", seeds.get("policy")),
        ("Network", track.get("network_policy")),
        ("Ranking", track.get("ranking_rule")),
        ("Success bar", f"{float(success_bar) * 100:.0f}%" if success_bar is not None else None),
        ("Contract hash", contract_id),
    ]
    return {
        "season_key": result.get("season_key"),
        "track_name": track.get("name") or parts.get("track") or fallback_track,
        "contract_id": contract_id,
        "legacy": not bool(plan),
        "fields": [
            {"label": label, "value": value}
            for label, value in fields
            if value is not None and value != ""
        ],
    }


def result_insight(
    result: Mapping[str, Any],
    peers: list[dict[str, Any]],
    *,
    success_bar: float,
    published_entry_id: int | None = None,
) -> dict[str, Any]:
    """Place one stored result against published peers in its exact season."""
    candidate = dict(result)
    candidate_id = published_entry_id if published_entry_id is not None else -1
    candidate["entry_id"] = candidate_id
    candidate.setdefault("entry_name", "private result")
    candidate_time = _number(candidate.get("total_time_sec"))
    success = _number(candidate.get("success_rate")) or 0.0
    reference_only = bool(
        candidate.get("reference_only")
        or ((candidate.get("run_plan") or {}).get("track") or {}).get("reference_only")
    )
    meets_bar = bool(candidate.get("meets_success_bar", success >= success_bar))

    competitive_peers = [peer for peer in peers if not peer.get("reference_only")]
    qualifying_peers = [
        peer for peer in competitive_peers
        if peer.get("meets_success_bar") and _number(peer.get("total_time_sec")) is not None
    ]
    fastest = min(
        qualifying_peers,
        key=lambda peer: float(peer["total_time_sec"]),
        default=None,
    )
    time_gap = None
    if candidate_time is not None and fastest is not None:
        time_gap = candidate_time - float(fastest["total_time_sec"])
    field = list(peers)
    if not any(int(peer["entry_id"]) == candidate_id for peer in field):
        field.append(candidate)
    frontier = (
        candidate_time is not None
        and candidate_id in pareto_entry_ids(field)
        and not reference_only
    )
    equivalent = any(
        int(peer["entry_id"]) != candidate_id
        and _number(peer.get("total_time_sec")) == candidate_time
        and _number(peer.get("success_rate")) == success
        for peer in competitive_peers
    )

    if reference_only:
        status = "reference"
        label = "Reference measurement"
        detail = "Visible for calibration, but excluded from competitive ranking."
    elif not meets_bar:
        status = "below-bar"
        label = "Below success bar"
        detail = (
            f"Needs {(success_bar - success) * 100:.1f} more percentage points "
            "before speed ranking applies."
        )
    elif published_entry_id is None and not competitive_peers:
        status = "establishes"
        label = "Would establish frontier"
        detail = "No competitive result has been published in this exact season."
    elif frontier and published_entry_id is None and equivalent:
        status = "equivalent"
        label = "Frontier-equivalent"
        detail = "Matches an existing frontier point at the stored precision."
    elif frontier and published_entry_id is None:
        status = "moves-frontier"
        label = "Would move frontier"
        detail = "This qualifying time-success point is not dominated by a published peer."
    elif frontier:
        status = "frontier"
        label = "Pareto frontier"
        detail = "No published peer is both faster and at least as successful."
    else:
        status = "qualifying"
        label = "Qualifies · off frontier"
        detail = "Clears the success bar, but a published point dominates it."

    if time_gap is None:
        gap_label = "No qualifying peer"
    elif abs(time_gap) < 0.005:
        gap_label = "Speed leader"
    elif time_gap > 0:
        gap_label = f"+{time_gap:.2f}s to fastest"
    else:
        gap_label = f"{abs(time_gap):.2f}s ahead of fastest"

    return {
        "status": status,
        "label": label,
        "detail": detail,
        "frontier": frontier,
        "meets_success_bar": meets_bar,
        "success_bar": success_bar,
        "success_gap_pp": max(0.0, (success_bar - success) * 100),
        "fastest": fastest,
        "time_gap_sec": time_gap,
        "gap_label": gap_label,
        "clock": clock_profile(
            candidate.get("agent_time_sec"), candidate.get("env_time_sec")
        ),
    }


def _task_cell(task: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    task_time = _number(task.get("task_time_sec"))
    baseline_time = _number(baseline.get("task_time_sec")) if baseline else None
    return {
        "present": True,
        "passed": task.get("passed"),
        "time_sec": task_time,
        "agent_time_sec": _number(task.get("agent_time_sec")),
        "env_time_sec": _number(task.get("env_time_sec")),
        "num_steps": task.get("num_steps"),
        "delta_sec": (
            task_time - baseline_time
            if task_time is not None and baseline_time is not None
            else None
        ),
    }


def build_comparison(
    entries: list[dict[str, Any]],
    tasks_by_entry: Mapping[int, list[dict[str, Any]]],
    peers: list[dict[str, Any]],
    *,
    success_bar: float,
) -> dict[str, Any]:
    """Build a 2–4 entry comparison with seed-safe task deltas."""
    if not 2 <= len(entries) <= 4:
        raise ValueError("select between two and four entries")
    if len({entry.get("season_key") for entry in entries}) != 1:
        raise ValueError("entries must share one exact season")
    if len({entry.get("track_name") for entry in entries}) != 1:
        raise ValueError("entries must share one exact track")

    qualifying = [
        entry for entry in entries
        if entry.get("meets_success_bar") and not entry.get("reference_only")
    ]
    baseline = min(
        qualifying or entries,
        key=lambda entry: float(entry["total_time_sec"]),
    )
    baseline_id = int(baseline["entry_id"])
    compared_entries = []
    for entry in entries:
        total = float(entry["total_time_sec"])
        insight = result_insight(
            entry,
            peers,
            success_bar=success_bar,
            published_entry_id=int(entry["entry_id"]),
        )
        task_rows = tasks_by_entry.get(int(entry["entry_id"]), [])
        compared_entries.append({
            **entry,
            "baseline": int(entry["entry_id"]) == baseline_id,
            "delta_sec": total - float(baseline["total_time_sec"]),
            "clock": clock_profile(entry.get("agent_time_sec"), entry.get("env_time_sec")),
            "steps": sum(
                int(task["num_steps"])
                for task in task_rows
                if task.get("num_steps") is not None
            ),
            "insight": insight,
        })

    task_maps: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    task_ids: set[str] = set()
    for entry in entries:
        entry_id = int(entry["entry_id"])
        per_task: dict[str, dict[str, dict[str, Any]]] = {}
        for task in tasks_by_entry.get(entry_id, []):
            task_id = str(task["task_id"])
            seed = str(task["seed"])
            per_task.setdefault(task_id, {})[seed] = dict(task)
            task_ids.add(task_id)
        task_maps[entry_id] = per_task

    task_groups = []
    mismatch_count = 0
    entry_ids = [int(entry["entry_id"]) for entry in entries]
    for task_id in sorted(task_ids):
        seed_sets = [
            tuple(sorted(task_maps[entry_id].get(task_id, {}), key=_seed_sort_key))
            for entry_id in entry_ids
        ]
        comparable = bool(seed_sets[0]) and all(seeds == seed_sets[0] for seeds in seed_sets)
        rows = []
        if comparable:
            for seed in seed_sets[0]:
                baseline_task = task_maps[baseline_id][task_id][seed]
                cells = []
                for entry_id in entry_ids:
                    cell = _task_cell(
                        task_maps[entry_id][task_id][seed], baseline_task
                    )
                    cell["baseline"] = entry_id == baseline_id
                    cells.append(cell)
                rows.append({
                    "seed": seed,
                    "cells": cells,
                })
        else:
            mismatch_count += 1
            cells = []
            for entry_id in entry_ids:
                task_rows = list(task_maps[entry_id].get(task_id, {}).values())
                times = [
                    float(task["task_time_sec"])
                    for task in task_rows
                    if task.get("task_time_sec") is not None
                ]
                cells.append({
                    "present": bool(task_rows),
                    "baseline": entry_id == baseline_id,
                    "seed_list": ", ".join(
                        sorted(
                            (str(task["seed"]) for task in task_rows),
                            key=_seed_sort_key,
                        )
                    ) or "—",
                    "time_sec": sum(times) if times else None,
                    "passed": sum(task.get("passed") is True for task in task_rows),
                    "runs": len(task_rows),
                    "delta_sec": None,
                })
            rows.append({"seed": None, "cells": cells})
        task_groups.append({
            "task_id": task_id,
            "seed_comparable": comparable,
            "rows": rows,
        })

    return {
        "entries": compared_entries,
        "baseline": baseline,
        "season_key": entries[0]["season_key"],
        "track_name": entries[0]["track_name"],
        "task_groups": task_groups,
        "seed_mismatch_count": mismatch_count,
        "success_bar": success_bar,
        "baseline_rule": (
            "Fastest selected entry that clears the success bar."
            if qualifying
            else "No selected entry clears the bar; fastest selected point shown as reference."
        ),
    }
