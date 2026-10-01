"""Presentation model for the track-local Pareto frontier.

The measurement core produces immutable cards and entries.  This module only
organizes those stored facts for the web UI: first by track, then by exact
season, and finally into rankings and time-versus-success geometry.  It never
scores a run and never compares entries across a season boundary.
"""

from __future__ import annotations

import math
import re
from itertools import groupby
from typing import Any


_EFFORT_SUFFIX = re.compile(
    r"(?:\s*[-–—·/:]\s*|\s+)"
    r"(minimal|low|medium|high|xhigh|max|thinking|default)"
    r"(?:\s*\(default\))?\s*$",
    re.IGNORECASE,
)
_EFFORT_RANK = {
    "minimal": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "xhigh": 4,
    "max": 5,
    "thinking": 6,
    "default": 6,
}


def _dashboard_identity(entry: dict[str, Any]) -> tuple[str, str]:
    """Return optional display metadata without changing result identity.

    Imported entries may already carry explicit model metadata. Older entries
    only have their public name, so a conventional trailing effort label is
    separated for presentation while the complete entry name stays intact.
    """
    name = str(entry.get("entry_name") or "Unnamed result").strip()
    explicit_model = entry.get("model_family") or entry.get("model_name")
    explicit_effort = entry.get("reasoning_effort")
    if explicit_model:
        return str(explicit_model), str(explicit_effort or "default").lower()

    match = _EFFORT_SUFFIX.search(name)
    if not match:
        return name, str(explicit_effort or "default").lower()
    family = name[:match.start()].strip(" -–—·/:") or name
    return family, str(explicit_effort or match.group(1)).lower()


def dashboard_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Small JSON-safe records consumed by the interactive results UI."""
    records = []
    for entry in entries:
        model, effort = _dashboard_identity(entry)
        num_runs = max(1, int(entry.get("num_runs") or 1))
        median_task_time = entry.get("median_task_time_sec")
        average_task_time = entry.get("average_task_time_sec")
        if average_task_time is None and entry.get("total_time_sec") is not None:
            average_task_time = float(entry["total_time_sec"]) / num_runs
        task_time = (
            average_task_time
            if entry.get("average_task_time_sec") is not None
            else median_task_time
        )
        if task_time is None:
            task_time = average_task_time
        if average_task_time is None:
            average_task_time = task_time
        if median_task_time is None:
            median_task_time = task_time
        performance = entry.get("mean_score")
        if performance is None:
            performance = entry.get("success_rate")
        cost_per_task = entry.get("cost_per_task_usd")
        if cost_per_task is None and entry.get("cost_usd") is not None:
            cost_per_task = float(entry["cost_usd"]) / num_runs
        model_type = str(entry.get("model_type") or "unclassified").lower()
        if model_type not in {"open", "closed"}:
            model_type = "unclassified"
        effort_rank = _EFFORT_RANK.get(
            effort.removesuffix(" (default)").strip(), -1
        )
        records.append({
            "entry_id": int(entry["entry_id"]),
            "entry_name": str(entry.get("entry_name") or model),
            "model": model,
            "effort": effort,
            "effort_rank": effort_rank,
            "model_type": model_type,
            "performance": float(performance or 0.0),
            "time_per_task_sec": float(task_time or 0.0),
            "average_time_per_task_sec": float(average_task_time or 0.0),
            "median_time_per_task_sec": float(median_task_time or 0.0),
            "cost_usd": (
                float(cost_per_task) if cost_per_task is not None else None
            ),
            "turns_per_task": (
                float(entry["average_turns_per_task"])
                if entry.get("average_turns_per_task") is not None
                else None
            ),
            "release_date": entry.get("release_date"),
            "average_output_tokens": (
                float(entry["average_output_tokens"])
                if entry.get("average_output_tokens") is not None
                else None
            ),
            "frontier": bool(entry.get("frontier")),
            "qualification_known": entry.get("meets_success_bar") is not None,
            "qualifying": bool(entry.get("meets_success_bar")),
            "reference": bool(entry.get("reference_only")),
            "provisional": bool(entry.get("provisional")),
        })
    return sorted(
        records,
        key=lambda record: (
            -float(record["performance"]),
            float(record["time_per_task_sec"]),
        ),
    )


def parse_season_key(key: str) -> dict[str, str]:
    """Parse both legacy four-field and content-addressed season keys."""
    parts: dict[str, str] = {}
    for part in str(key).split(" | "):
        name, separator, value = part.partition("=")
        if separator:
            parts[name] = value
    return parts


def track_label(name: str) -> str:
    """Human label without turning the canonical track name into identity."""
    acronyms = {
        "api": "API",
        "cpu": "CPU",
        "gpu": "GPU",
        "h100": "H100",
        "h200": "H200",
        "l4": "L4",
        "l40s": "L40S",
        "rl": "RL",
        "vllm": "vLLM",
    }
    return " ".join(acronyms.get(token.lower(), token.capitalize())
                    for token in str(name).replace("_", "-").split("-") if token)


def season_label(key: str) -> str:
    parts = parse_season_key(key)
    benchmark = parts.get("benchmark", key)
    hardware = parts.get("hardware")
    return " · ".join(value for value in (benchmark, hardware) if value)


def pareto_entry_ids(entries: list[dict[str, Any]]) -> set[int]:
    """Competitive, non-dominated ids in O(n log n).

    Entries at identical objective coordinates are equivalent and all remain
    on the frontier. Reference-only measurements are visible in the plot but
    never define the competitive frontier.
    """
    competitive = sorted(
        (entry for entry in entries
        if entry.get("total_time_sec") is not None
        and entry.get("success_rate") is not None
        and not entry.get("reference_only")),
        key=lambda entry: (
            float(entry["total_time_sec"]),
            -float(entry["success_rate"]),
        ),
    )
    frontier: set[int] = set()
    best_success_at_faster_time = -math.inf
    for _time, grouped_entries in groupby(
        competitive, key=lambda entry: float(entry["total_time_sec"])
    ):
        group = list(grouped_entries)
        group_best = float(group[0]["success_rate"])
        if group_best > best_success_at_faster_time:
            frontier.update(
                int(entry["entry_id"])
                for entry in group
                if float(entry["success_rate"]) == group_best
            )
        best_success_at_faster_time = max(
            best_success_at_faster_time, group_best
        )
    return frontier


def _nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _format_seconds(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}s"
    if value >= 10:
        return f"{value:.0f}s"
    return f"{value:g}s"


def pareto_geometry(
    entries: list[dict[str, Any]],
    *,
    success_bar: float,
    width: int = 1120,
    height: int = 430,
) -> dict[str, Any] | None:
    """SVG geometry for one season, including its frozen success bar."""
    measured = [
        entry for entry in entries
        if entry.get("total_time_sec") is not None
        and entry.get("success_rate") is not None
    ]
    if not measured:
        return None

    margin_left, margin_right, margin_top, margin_bottom = 68, 30, 28, 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    maximum_time = max(float(entry["total_time_sec"]) for entry in measured)
    tick_step = _nice_step(max(maximum_time * 1.08, 1.0) / 5)
    time_ceiling = max(tick_step, math.ceil(maximum_time * 1.08 / tick_step) * tick_step)

    def x(value: float) -> float:
        return margin_left + (value / time_ceiling) * plot_width

    def y(value: float) -> float:
        return margin_top + (1.0 - value) * plot_height

    frontier_ids = pareto_entry_ids(measured)
    points = []
    show_all_labels = len(measured) <= 12
    for entry in measured:
        point_x = round(x(float(entry["total_time_sec"])), 1)
        point_y = round(y(float(entry["success_rate"])), 1)
        label_on_left = point_x > width - 210
        points.append({
            "entry_id": entry["entry_id"],
            "label": entry["entry_name"],
            "cx": point_x,
            "cy": point_y,
            "label_x": round(point_x - 12 if label_on_left else point_x + 12, 1),
            "label_anchor": "end" if label_on_left else "start",
            "label_y": round(max(margin_top + 12, point_y - 12), 1),
            "show_label": show_all_labels or entry["entry_id"] in frontier_ids,
            "time": float(entry["total_time_sec"]),
            "success": float(entry["success_rate"]),
            "qualifying": bool(entry.get("meets_success_bar")),
            "frontier": entry["entry_id"] in frontier_ids,
            "reference": bool(entry.get("reference_only")),
        })

    # Labels are deliberately sparse for large seasons, but two frontier
    # points can still be nearly identical (the exact case the noise floor is
    # meant to explain).  Stagger nearby labels vertically instead of letting
    # the names overwrite each other in the static SVG.
    placed_labels: list[tuple[float, float]] = []
    for point in sorted(points, key=lambda item: (item["cy"], item["cx"])):
        if not point["show_label"]:
            continue
        candidates = (
            point["label_y"],
            point["cy"] + 19,
            point["cy"] + 34,
            point["cy"] - 27,
        )
        for candidate in candidates:
            candidate = round(
                min(margin_top + plot_height - 6, max(margin_top + 12, candidate)),
                1,
            )
            if all(
                abs(point["cx"] - other_x) >= 150 or abs(candidate - other_y) >= 14
                for other_x, other_y in placed_labels
            ):
                point["label_y"] = candidate
                break
        placed_labels.append((point["cx"], point["label_y"]))

    frontier_coordinates = sorted({
        (float(entry["total_time_sec"]), float(entry["success_rate"]))
        for entry in measured
        if entry["entry_id"] in frontier_ids
    })
    path = ""
    if frontier_coordinates:
        first_time, first_success = frontier_coordinates[0]
        segments = [f"M {x(first_time):.1f} {y(first_success):.1f}"]
        for next_time, next_success in frontier_coordinates[1:]:
            segments.append(
                f"L {x(next_time):.1f} {y(first_success):.1f} "
                f"L {x(next_time):.1f} {y(next_success):.1f}"
            )
            first_success = next_success
        path = " ".join(segments)

    tick_count = int(round(time_ceiling / tick_step))
    x_ticks = [
        {
            "value": round(index * tick_step, 6),
            "label": _format_seconds(index * tick_step),
            "x": round(x(index * tick_step), 1),
        }
        for index in range(tick_count + 1)
    ]
    y_ticks = [
        {"value": value, "label": f"{int(value * 100)}%", "y": round(y(value), 1)}
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    return {
        "width": width,
        "height": height,
        "ml": margin_left,
        "mt": margin_top,
        "pw": plot_width,
        "ph": plot_height,
        "points": points,
        "path": path,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "success_bar": success_bar,
        "success_bar_y": round(y(success_bar), 1),
        "frontier_count": len(frontier_ids),
        "single_point": len(measured) == 1,
    }


def _success_bar(entries: list[dict[str, Any]], track: dict[str, Any]) -> float:
    for entry in entries:
        rules = entry.get("rules") or {}
        if rules.get("success_bar") is not None:
            return float(rules["success_bar"])
        run_plan = entry.get("run_plan") or {}
        scoring = run_plan.get("scoring") or {}
        if scoring.get("success_bar") is not None:
            return float(scoring["success_bar"])
    return float(track.get("success_bar") if track.get("success_bar") is not None else 0.9)


def _rank_entries(
    entries: list[dict[str, Any]],
    *,
    frontier_ids: set[int],
    provisional_threshold: int,
) -> list[dict[str, Any]]:
    qualifying = sorted(
        (
            entry for entry in entries
            if entry.get("meets_success_bar") and not entry.get("reference_only")
        ),
        key=lambda entry: float(entry["total_time_sec"]),
    )
    rank = 0
    group_time: float | None = None
    ranked: list[dict[str, Any]] = []
    for entry in qualifying:
        entry_time = float(entry["total_time_sec"])
        tied = bool(
            group_time is not None
            and entry_time == group_time
        )
        if not tied:
            rank += 1
            group_time = entry_time
        ranked.append({**entry, "rank": rank, "tied": tied})

    below_bar = sorted(
        (
            entry for entry in entries
            if not entry.get("meets_success_bar") and not entry.get("reference_only")
        ),
        key=lambda entry: (-float(entry["success_rate"]), float(entry["total_time_sec"])),
    )
    references = sorted(
        (entry for entry in entries if entry.get("reference_only")),
        key=lambda entry: (-float(entry["success_rate"]), float(entry["total_time_sec"])),
    )
    ranked.extend({**entry, "rank": None, "tied": False} for entry in below_bar)
    ranked.extend({**entry, "rank": None, "tied": False} for entry in references)

    out = []
    for entry in ranked:
        agent_time = float(entry.get("agent_time_sec") or 0.0)
        env_time = float(entry.get("env_time_sec") or 0.0)
        split_total = agent_time + env_time
        out.append({
            **entry,
            "frontier": entry["entry_id"] in frontier_ids,
            "provisional": int(entry.get("num_runs") or 0) < provisional_threshold,
            "agent_percent": round(100 * agent_time / split_total, 1) if split_total else None,
            "env_percent": round(100 * env_time / split_total, 1) if split_total else None,
        })
    return out


def build_frontier_context(
    entries: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    seasons: dict[str, dict[str, Any]],
    *,
    selected_track: str | None = None,
    selected_season: str | None = None,
    provisional_threshold: int = 3,
) -> dict[str, Any]:
    """Build a comparison-safe view model for the Frontier page."""
    track_by_name = {str(track["name"]): dict(track) for track in tracks}
    for entry in entries:
        track_by_name.setdefault(entry["track_name"], {"name": entry["track_name"]})

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {} for name in track_by_name
    }
    for entry in entries:
        grouped.setdefault(entry["track_name"], {}).setdefault(
            entry["season_key"], []
        ).append(entry)

    season_owner = {
        season_key: track_name
        for track_name, track_seasons in grouped.items()
        for season_key in track_seasons
    }
    if selected_track not in track_by_name and selected_season in season_owner:
        selected_track = season_owner[selected_season]
    if selected_track not in track_by_name:
        selected_track = max(
            track_by_name,
            key=lambda name: (
                sum(len(rows) for rows in grouped.get(name, {}).values()),
                name,
            ),
            default=None,
        )

    track_options = []
    for name, track in track_by_name.items():
        entry_count = sum(len(rows) for rows in grouped.get(name, {}).values())
        track_options.append({
            **track,
            "name": name,
            "label": track_label(name),
            "entry_count": entry_count,
            "season_count": len(grouped.get(name, {})),
            "active": name == selected_track,
        })
    track_options.sort(key=lambda track: (-track["entry_count"], track["name"]))

    selected_track_data = track_by_name.get(selected_track or "", {})
    track_seasons = grouped.get(selected_track or "", {})
    if selected_season not in track_seasons:
        selected_season = max(
            track_seasons,
            key=lambda key: (
                len(track_seasons[key]),
                max(str(entry.get("published_at") or "") for entry in track_seasons[key]),
                key,
            ),
            default=None,
        )

    season_options = []
    for key, season_entries in track_seasons.items():
        season_data = seasons.get(key, {})
        parts = parse_season_key(key)
        season_options.append({
            "key": key,
            "label": season_label(key),
            "benchmark": parts.get("benchmark"),
            "hardware": parts.get("hardware"),
            "algorithm": parts.get("algorithm"),
            "contract": parts.get("contract"),
            "entry_count": len(season_entries),
            "status": season_data.get("status", "legacy"),
            "active": key == selected_season,
        })
    season_options.sort(key=lambda season: (-season["entry_count"], season["label"]))

    current_entries = list(track_seasons.get(selected_season or "", []))
    season_data = seasons.get(selected_season or "", {})
    success_bar = _success_bar(current_entries, selected_track_data)
    frontier_ids = pareto_entry_ids(current_entries)
    ranked = _rank_entries(
        current_entries,
        frontier_ids=frontier_ids,
        provisional_threshold=provisional_threshold,
    )
    competitive = [entry for entry in current_entries if not entry.get("reference_only")]
    qualifying = [entry for entry in competitive if entry.get("meets_success_bar")]
    fastest = min(qualifying, key=lambda entry: float(entry["total_time_sec"]), default=None)
    best_success = min(
        competitive,
        key=lambda entry: (-float(entry["success_rate"]), float(entry["total_time_sec"])),
        default=None,
    )
    parts = parse_season_key(selected_season or "")
    benchmark = parts.get("benchmark")
    benchmark_name = benchmark.rsplit("@", 1)[0] if benchmark else None
    selected_season_data = next(
        (season for season in season_options if season["active"]), None
    )

    return {
        "tracks": track_options,
        "track": {
            **selected_track_data,
            "name": selected_track,
            "label": track_label(selected_track) if selected_track else None,
        } if selected_track else None,
        "seasons": season_options,
        "season": selected_season_data,
        "season_key": selected_season,
        "season_spec": season_data.get("spec") or {},
        "season_contract_hash": season_data.get("contract_hash"),
        "current": ranked,
        "dashboard_entries": dashboard_entries(ranked),
        "pareto": pareto_geometry(ranked, success_bar=success_bar),
        "success_bar": success_bar,
        "fastest": fastest,
        "best_success": best_success,
        "frontier_count": len(frontier_ids),
        "entry_count": len(current_entries),
        "qualifying_count": len(qualifying),
        "benchmark_name": benchmark_name,
        "provisional_threshold": provisional_threshold,
    }
