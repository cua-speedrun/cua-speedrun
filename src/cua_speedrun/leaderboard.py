"""Result cards and the leaderboard.

The flow matches the design's Geekbench-style model: a finished run yields a
private result card the submitter reviews, and only if they accept does
`publish` append it to the leaderboard. The leaderboard groups entries by
season (a frozen benchmark+harness+backend+hardware key) and ranks within a
season, because times are never comparable across seasons.

Everything here is a pure function over stored run logs and result.json, so
nothing here touches an environment and any card or ranking is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cua_speedrun.scoring import ScoringRules, score

LEADERBOARD_DIR = Path("leaderboard")
ENTRIES_FILE = LEADERBOARD_DIR / "entries.jsonl"


def season_key(season: dict[str, Any]) -> str:
    fields = ["benchmark", "harness", "backend", "hardware"]
    # New runs add the human track/algorithm plus the canonical contract
    # hash.  Old four-field seasons remain byte-for-byte compatible.
    fields.extend(k for k in ("track", "algorithm", "contract") if k in season)
    return " | ".join(f"{k}={season[k]}" for k in fields)


def build_card(run_dir: Path, rules: ScoringRules | None = None) -> dict[str, Any]:
    """Consolidate one run into a result card: the private result."""
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    frozen_scoring = (result.get("run_plan") or {}).get("scoring")
    if rules is not None and frozen_scoring:
        expected = {
            "success_bar": float(frozen_scoring["success_bar"]),
            "failure_costs_timeout": bool(
                frozen_scoring.get("failure_costs_timeout", False)
            ),
        }
        if rules.to_dict() != expected:
            raise ValueError(
                "cannot build or publish a card with scoring rules that "
                "differ from its frozen run plan; use score() for research "
                "re-scoring"
            )
    # Runs recorded before the season key existed synthesize one from the
    # fields they do carry, so old run directories remain publishable.
    season = result.get("season") or {
        "benchmark": f"{result['benchmark']['name']}@{result['benchmark']['version']}",
        "harness": result.get("harness_version", "unknown"),
        "backend": result.get("backend", "unknown"),
        "hardware": "unspecified",
    }
    scored = score(run_dir, rules)
    return {
        "run_id": result["run_id"],
        "season": season,
        "season_key": season_key(season),
        "submission_fingerprint": result["submission_fingerprint"],
        "num_runs": scored["num_runs"],
        "num_passed": scored["num_passed"],
        "mean_score": scored["mean_score"],
        "success_rate": scored["success_rate"],
        "meets_success_bar": scored["meets_success_bar"],
        "total_time_sec": scored["total_time_sec"],
        "measured_time_sec": scored.get("measured_time_sec"),
        "median_task_time_sec": scored["median_task_time_sec"],
        "env_time_sec": scored["env_time_sec"],
        "agent_time_sec": scored["agent_time_sec"],
        "rules": scored["rules"],
        "reference_only": bool(
            (result.get("run_plan") or {}).get("track", {}).get("reference_only", False)
        ),
        "run_plan_hash": result.get("run_plan_hash"),
        "run_plan": result.get("run_plan"),
        # Operational scale is useful evidence but deliberately does not
        # split a season: each compute replica is isolated.
        "parallelism": result.get("parallelism"),
    }


def format_card(card: dict[str, Any], entry_name: str | None = None) -> str:
    lines = []
    title = entry_name or card["submission_fingerprint"]
    lines.append(f"Result card: {title}")
    lines.append(f"  season:        {card['season_key']}")
    bar = "meets" if card["meets_success_bar"] else "BELOW"
    lines.append(
        f"  success:       {card['num_passed']}/{card['num_runs']} "
        f"({card['success_rate']:.0%}, {bar} the bar)"
    )
    lines.append(
        f"  mean score:    {card.get('mean_score', card['success_rate']):.1%}"
    )
    lines.append(f"  total time:    {card['total_time_sec']:.2f}s")
    if card["median_task_time_sec"] is not None:
        lines.append(f"  median/task:   {card['median_task_time_sec']:.2f}s")
    if card["agent_time_sec"]:
        lines.append(
            f"  time split:    agent {card['agent_time_sec']:.2f}s, "
            f"env {card['env_time_sec']:.2f}s"
        )
    return "\n".join(lines)


def publish(run_dir: Path, entry_name: str, rules: ScoringRules | None = None) -> dict[str, Any]:
    """Append a run's result card to the leaderboard. The Geekbench-style
    accept step: only called when the submitter chooses to publish."""
    card = build_card(run_dir, rules)
    card["entry_name"] = entry_name
    LEADERBOARD_DIR.mkdir(exist_ok=True)
    with open(ENTRIES_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(card, default=str) + "\n")
    return card


def load_entries() -> list[dict[str, Any]]:
    if not ENTRIES_FILE.is_file():
        return []
    entries = []
    for line in ENTRIES_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def format_leaderboard(entries: list[dict[str, Any]]) -> str:
    """Rank entries within each season by total time, qualifying entries first.
    Entries that miss the success bar are shown but never rank above qualifying
    ones."""
    if not entries:
        return "leaderboard is empty"
    by_season: dict[str, list[dict]] = {}
    for e in entries:
        by_season.setdefault(e["season_key"], []).append(e)

    out = []
    for skey, es in by_season.items():
        out.append(f"Season: {skey}")
        es.sort(key=lambda e: (
            bool(e.get("reference_only")),
            not e["meets_success_bar"],
            e["total_time_sec"],
        ))
        out.append(f"  {'rank':<5}{'entry':<24}{'time':>10}{'success':>10}")
        out.append("  " + "-" * 47)
        rank = 0
        for e in es:
            if e["meets_success_bar"] and not e.get("reference_only"):
                rank += 1
                r = str(rank)
            else:
                r = "-"
            out.append(
                f"  {r:<5}{e['entry_name'][:23]:<24}"
                f"{e['total_time_sec']:>9.2f}s{e['success_rate']:>9.0%}"
            )
        out.append("")
    return "\n".join(out)
