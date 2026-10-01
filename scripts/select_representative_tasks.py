#!/usr/bin/env python3
"""Choose a representative benchmark subset from historical model scores.

The script implements the locked unweighted best-of-100 raw-energy method:

1. Run leave-one-model-out (LOMO) selection for every task count K.
2. Choose the smallest K whose K-1/K/K+1 neighborhoods all preserve both
   partial-score and exact-completion model rankings at Spearman rho >= 0.95.
3. Refit one K-task subset with all historical models and print/save its IDs.

Only NumPy is required.  Run ``python select_representative_tasks.py --help``
for accepted JSON/CSV layouts and options.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_STARTS = 100
DEFAULT_SEED_BASE = 7_000
DEFAULT_RANK_THRESHOLD = 0.95
PAIRWISE_DIAGNOSTIC_THRESHOLD = 0.90


@dataclass(frozen=True)
class Panel:
    model_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    scores: np.ndarray
    dropped_task_ids: tuple[str, ...]
    score_scale: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_panel(payload: dict[str, Any]) -> tuple[list[str], list[str], Any]:
    """Extract a matrix from the compact or OSWorld2 JSON schema."""

    if "complete_case_matrix" in payload:
        matrix = payload["complete_case_matrix"]
    elif "matrix" in payload:
        matrix = payload["matrix"]
    else:
        matrix = payload
    model_ids = matrix.get("model_ids") or matrix.get("models")
    task_ids = matrix.get("task_ids") or matrix.get("tasks")
    values = matrix.get("values")
    if values is None:
        values = matrix.get("scores")
    if model_ids is None or task_ids is None or values is None:
        raise ValueError("not a recognized matrix JSON object")
    return list(map(str, model_ids)), list(map(str, task_ids)), values


def _records_panel(payload: dict[str, Any]) -> tuple[list[str], list[str], Any]:
    """Extract ``{model: [{task_id, score}, ...]}`` or nested dictionaries."""

    if not payload:
        raise ValueError("empty JSON object")
    model_ids = list(map(str, payload.keys()))
    first = payload[model_ids[0]]
    if isinstance(first, dict):
        task_ids = list(map(str, first.keys()))
        values = [
            [payload[model].get(task) for task in task_ids]
            for model in model_ids
        ]
        return model_ids, task_ids, values
    if isinstance(first, list) and (not first or isinstance(first[0], dict)):
        if not first:
            raise ValueError("the first model has no task records")
        task_key = "task_id" if "task_id" in first[0] else "task"
        score_key = "score" if "score" in first[0] else "value"
        task_ids = [str(record[task_key]) for record in first]
        values: list[list[Any]] = []
        for model in model_ids:
            by_task = {
                str(record[task_key]): record.get(score_key)
                for record in payload[model]
            }
            values.append([by_task.get(task) for task in task_ids])
        return model_ids, task_ids, values
    raise ValueError("not a recognized model-record JSON object")


def _load_json(path: Path) -> tuple[list[str], list[str], Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    try:
        return _matrix_panel(payload)
    except (AttributeError, ValueError):
        return _records_panel(payload)


def _find_header(fieldnames: Sequence[str], choices: Sequence[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for choice in choices:
        if choice in normalized:
            return normalized[choice]
    return None


def _load_csv(path: Path) -> tuple[list[str], list[str], Any]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        rows = list(reader)
        fields = list(reader.fieldnames)
    if not rows:
        raise ValueError("CSV has no data rows")

    model_key = _find_header(fields, ("model", "model_id", "model_name"))
    task_key = _find_header(fields, ("task_id", "task", "instance_id"))
    score_key = _find_header(fields, ("score", "partial_score", "value"))

    if model_key and task_key and score_key:  # Long-form CSV.
        model_ids: list[str] = []
        task_ids: list[str] = []
        seen_models: set[str] = set()
        seen_tasks: set[str] = set()
        entries: dict[tuple[str, str], str] = {}
        for row in rows:
            model, task = row[model_key].strip(), row[task_key].strip()
            if not model or not task:
                raise ValueError("CSV model and task IDs must be nonempty")
            key = (model, task)
            if key in entries:
                raise ValueError(f"duplicate CSV entry for model={model}, task={task}")
            entries[key] = row[score_key]
            if model not in seen_models:
                seen_models.add(model)
                model_ids.append(model)
            if task not in seen_tasks:
                seen_tasks.add(task)
                task_ids.append(task)
        values = [[entries.get((model, task)) for task in task_ids] for model in model_ids]
        return model_ids, task_ids, values

    # Wide CSV: first/model column identifies rows; all other columns are tasks.
    model_key = model_key or fields[0]
    task_ids = [field for field in fields if field != model_key]
    model_ids = [row[model_key].strip() for row in rows]
    values = [[row[task] for task in task_ids] for row in rows]
    return model_ids, task_ids, values


def load_panel(path: Path, requested_scale: str) -> Panel:
    if path.suffix.lower() == ".csv":
        model_ids, task_ids, raw_values = _load_csv(path)
    else:
        model_ids, task_ids, raw_values = _load_json(path)
    if len(model_ids) != len(set(model_ids)) or any(not item for item in model_ids):
        raise ValueError("model IDs must be nonempty and unique")
    if len(task_ids) != len(set(task_ids)) or any(not item for item in task_ids):
        raise ValueError("task IDs must be nonempty and unique")

    def number(value: Any) -> float:
        if value is None or (isinstance(value, str) and not value.strip()):
            return math.nan
        return float(value)

    scores = np.asarray(
        [[number(value) for value in row] for row in raw_values],
        dtype=np.float64,
    )
    expected = (len(model_ids), len(task_ids))
    if scores.shape != expected:
        raise ValueError(f"score matrix must have shape {expected}; found {scores.shape}")

    complete = np.all(np.isfinite(scores), axis=0)
    dropped = tuple(task for task, keep in zip(task_ids, complete) if not keep)
    scores = np.ascontiguousarray(scores[:, complete], dtype=np.float64)
    task_ids = [task for task, keep in zip(task_ids, complete) if keep]
    if scores.shape[0] < 3 or scores.shape[1] < 3:
        raise ValueError("at least three models and three complete tasks are required")

    minimum, maximum = float(scores.min()), float(scores.max())
    scale = requested_scale
    if scale == "auto":
        scale = "unit" if maximum <= 1.0 else "percent"
    if scale == "percent":
        scores = scores / 100.0
        minimum, maximum = minimum / 100.0, maximum / 100.0
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "scores must lie in [0,1] (or [0,100] with --score-scale percent/auto)"
        )
    return Panel(
        model_ids=tuple(model_ids),
        task_ids=tuple(task_ids),
        scores=scores,
        dropped_task_ids=dropped,
        score_scale=scale,
    )


def principal_coordinate(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if not singular_values.size or singular_values[0] <= np.finfo(float).eps:
        coordinate = features.mean(axis=1).copy()
    else:
        coordinate = left[:, 0] * singular_values[0]
    difficulty = features.mean(axis=1)
    orientation = float(
        np.dot(coordinate - coordinate.mean(), difficulty - difficulty.mean())
    )
    if orientation < 0.0:
        coordinate = -coordinate
    elif orientation == 0.0:
        pivot = int(np.argmax(np.abs(coordinate)))
        if coordinate[pivot] < 0.0:
            coordinate = -coordinate
    return np.round(coordinate, decimals=12)


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    norms = np.sum(features * features, axis=1)
    squared = norms[:, None] + norms[None, :] - 2.0 * (features @ features.T)
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared, out=squared)


def fractional_start(order: np.ndarray, count: int, phase: float) -> np.ndarray:
    spacing = order.size / float(count)
    positions = np.floor(
        phase * spacing + np.arange(count, dtype=np.float64) * spacing
    ).astype(np.int64)
    np.minimum(positions, order.size - 1, out=positions)
    for index in range(count - 2, -1, -1):
        if positions[index] >= positions[index + 1]:
            positions[index] = positions[index + 1] - 1
    selected = np.sort(order[positions])
    if selected.size != count or np.unique(selected).size != count:
        raise AssertionError("fractional start did not return exactly K unique tasks")
    return selected


def energy_objective(
    selected: np.ndarray,
    distances: np.ndarray,
    population_sums: np.ndarray,
    total_sum: float,
    mean_distance: float,
) -> float:
    if mean_distance <= np.finfo(float).eps:
        return 0.0
    count, n_tasks = selected.size, population_sums.size
    return float(
        (
            2.0 * population_sums[selected].sum() / (count * n_tasks)
            - distances[np.ix_(selected, selected)].sum() / count**2
            - total_sum / n_tasks**2
        )
        / mean_distance
    )


def optimize_one_swap(
    initial: np.ndarray,
    distances: np.ndarray,
    population_sums: np.ndarray,
    total_sum: float,
    mean_distance: float,
) -> np.ndarray:
    """Strict best-improvement one-swap descent with deterministic tie breaks."""

    selected = np.sort(np.asarray(initial, dtype=np.int64))
    n_tasks, count = population_sums.size, selected.size
    if count == n_tasks or mean_distance <= np.finfo(float).eps:
        return selected
    while True:
        selected_distances = distances[np.ix_(selected, selected)]
        current = energy_objective(
            selected, distances, population_sums, total_sum, mean_distance
        )
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(current))
        mask = np.zeros(n_tasks, dtype=bool)
        mask[selected] = True
        outside = np.flatnonzero(~mask)
        outside_to_selected = distances[np.ix_(outside, selected)]
        outside_selected_sums = outside_to_selected.sum(axis=1)
        selected_population_sum = float(population_sums[selected].sum())
        selected_within_sum = float(selected_distances.sum())
        best_value, best_pair = current, None

        for slot, removed in enumerate(selected):
            candidate_population = (
                selected_population_sum
                - population_sums[removed]
                + population_sums[outside]
            )
            retained_within = (
                selected_within_sum - 2.0 * float(selected_distances[slot].sum())
            )
            candidate_within = retained_within + 2.0 * (
                outside_selected_sums - outside_to_selected[:, slot]
            )
            values = (
                2.0 * candidate_population / (count * n_tasks)
                - candidate_within / count**2
                - total_sum / n_tasks**2
            ) / mean_distance
            improving = np.flatnonzero(values < current - tolerance)
            if not improving.size:
                continue
            local_value = float(values[improving].min())
            tied_positions = improving[values[improving] == local_value]
            added = int(outside[int(tied_positions[0])])
            pair = (int(removed), added)
            if local_value < best_value - tolerance or (
                abs(local_value - best_value) <= tolerance
                and (best_pair is None or pair < best_pair)
            ):
                best_value, best_pair = local_value, pair
        if best_pair is None:
            return selected
        removed, added = best_pair
        selected = np.sort(
            np.append(selected[selected != removed], added).astype(np.int64)
        )


def geometry(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    joint = np.vstack([scores, (scores == 1.0).astype(np.float64)])
    features = np.ascontiguousarray(joint.T, dtype=np.float64)
    distances = pairwise_distances(features)
    population_sums = distances.sum(axis=1)
    total_sum = float(population_sums.sum())
    n_tasks = scores.shape[1]
    mean_distance = total_sum / float(n_tasks * (n_tasks - 1))
    order = np.argsort(principal_coordinate(features), kind="stable")
    return order, distances, population_sums, total_sum, mean_distance


def best_of_n(
    order: np.ndarray,
    count: int,
    phases: np.ndarray,
    distances: np.ndarray,
    population_sums: np.ndarray,
    total_sum: float,
    mean_distance: float,
) -> tuple[np.ndarray, float, int]:
    unique_initials: dict[tuple[int, ...], None] = {}
    for phase in phases:
        initial = fractional_start(order, count, float(phase))
        unique_initials.setdefault(tuple(map(int, initial)), None)

    final_objectives: dict[tuple[int, ...], float] = {}
    for initial_tuple in unique_initials:
        selected = optimize_one_swap(
            np.asarray(initial_tuple, dtype=np.int64),
            distances,
            population_sums,
            total_sum,
            mean_distance,
        )
        final_tuple = tuple(map(int, selected))
        final_objectives.setdefault(
            final_tuple,
            energy_objective(
                selected, distances, population_sums, total_sum, mean_distance
            ),
        )
    best_value = min(final_objectives.values())
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(best_value))
    best_tuple = min(
        subset
        for subset, value in final_objectives.items()
        if abs(value - best_value) <= tolerance
    )
    return (
        np.asarray(best_tuple, dtype=np.int64),
        float(final_objectives[best_tuple]),
        len(final_objectives),
    )


def evaluate_fold(arguments: tuple[int, np.ndarray, np.ndarray]) -> tuple[Any, ...]:
    held_out, scores, phases = arguments
    calibration = scores[np.arange(scores.shape[0]) != held_out]
    order, distances, population_sums, total_sum, mean_distance = geometry(calibration)
    n_tasks = scores.shape[1]
    partial = np.empty(n_tasks, dtype=np.float64)
    binary = np.empty(n_tasks, dtype=np.float64)
    objectives = np.empty(n_tasks, dtype=np.float64)
    unique_counts = np.empty(n_tasks, dtype=np.int64)
    for count in range(1, n_tasks + 1):
        selected, objective, unique_count = best_of_n(
            order,
            count,
            phases,
            distances,
            population_sums,
            total_sum,
            mean_distance,
        )
        partial[count - 1] = float(scores[held_out, selected].mean())
        binary[count - 1] = float((scores[held_out, selected] == 1.0).mean())
        objectives[count - 1] = objective
        unique_counts[count - 1] = unique_count
    return held_out, partial, binary, objectives, unique_counts


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(truth: np.ndarray, estimate: np.ndarray) -> float:
    left, right = average_ranks(truth), average_ranks(estimate)
    left, right = left - left.mean(), right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator else math.nan


def pairwise_accuracy(truth: np.ndarray, estimate: np.ndarray) -> float:
    correct = total = 0
    for left in range(truth.size):
        for right in range(left + 1, truth.size):
            delta = truth[left] - truth[right]
            if abs(delta) <= 1e-12:
                continue
            correct += int(np.sign(estimate[left] - estimate[right]) == np.sign(delta))
            total += 1
    return float(correct / total) if total else 1.0


def metric_row(
    count: int,
    true_partial: np.ndarray,
    true_binary: np.ndarray,
    pred_partial: np.ndarray,
    pred_binary: np.ndarray,
    objectives: np.ndarray,
) -> dict[str, Any]:
    return {
        "k": count,
        "partial_mae_pp": 100.0 * float(np.mean(np.abs(pred_partial - true_partial))),
        "binary_mae_pp": 100.0 * float(np.mean(np.abs(pred_binary - true_binary))),
        "partial_spearman": spearman(true_partial, pred_partial),
        "binary_spearman": spearman(true_binary, pred_binary),
        "partial_pairwise_accuracy": pairwise_accuracy(true_partial, pred_partial),
        "binary_pairwise_accuracy": pairwise_accuracy(true_binary, pred_binary),
        "mean_training_energy": float(objectives.mean()),
    }


def choose_k(rows: list[dict[str, Any]], threshold: float) -> int | None:
    by_k = {int(row["k"]): row for row in rows}
    for center in range(2, len(rows)):
        neighborhood = [by_k[count] for count in (center - 1, center, center + 1)]
        if all(
            math.isfinite(float(row[metric])) and float(row[metric]) >= threshold
            for row in neighborhood
            for metric in ("partial_spearman", "binary_spearman")
        ):
            return center
    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_panel(args.scores, args.score_scale)
    scores = panel.scores
    n_models, n_tasks = scores.shape
    workers = min(args.workers, n_models)
    phase_matrix = np.asarray(
        [
            np.random.default_rng(args.seed_base + draw).random(n_models)
            for draw in range(args.starts)
        ],
        dtype=np.float64,
    )
    partial = np.empty((n_tasks, n_models), dtype=np.float64)
    binary = np.empty_like(partial)
    objectives = np.empty_like(partial)
    unique_counts = np.empty((n_tasks, n_models), dtype=np.int64)
    jobs = [
        (held_out, scores, phase_matrix[:, held_out])
        for held_out in range(n_models)
    ]
    if workers == 1:
        results = map(evaluate_fold, jobs)
        for result in results:
            held_out, fold_partial, fold_binary, fold_objectives, fold_counts = result
            partial[:, held_out] = fold_partial
            binary[:, held_out] = fold_binary
            objectives[:, held_out] = fold_objectives
            unique_counts[:, held_out] = fold_counts
            print(f"LOMO complete: {panel.model_ids[held_out]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_fold, job): job[0] for job in jobs}
            for future in as_completed(futures):
                held_out, fold_partial, fold_binary, fold_objectives, fold_counts = (
                    future.result()
                )
                partial[:, held_out] = fold_partial
                binary[:, held_out] = fold_binary
                objectives[:, held_out] = fold_objectives
                unique_counts[:, held_out] = fold_counts
                print(f"LOMO complete: {panel.model_ids[held_out]}", flush=True)

    true_partial = scores.mean(axis=1)
    true_binary = (scores == 1.0).mean(axis=1)
    rows = [
        metric_row(
            count,
            true_partial,
            true_binary,
            partial[count - 1],
            binary[count - 1],
            objectives[count - 1],
        )
        for count in range(1, n_tasks + 1)
    ]
    selected_k = choose_k(rows, args.rank_threshold)
    common = {
        "input": str(args.scores.resolve()),
        "input_sha256": _sha256(args.scores),
        "model_ids": list(panel.model_ids),
        "n_models": n_models,
        "n_input_tasks": n_tasks + len(panel.dropped_task_ids),
        "n_complete_tasks": n_tasks,
        "dropped_incomplete_task_ids": list(panel.dropped_task_ids),
        "input_score_scale": panel.score_scale,
        "method": "unweighted_raw_energy_best_of_n",
        "starts": args.starts,
        "seed_base": args.seed_base,
        "binary_definition": "score == 1.0 after conversion to [0,1]",
        "selection_features": "historical partial scores concatenated with exact-completion indicators",
        "new_model_readout": "ordinary unweighted mean over selected tasks",
        "k_rule": (
            "smallest center K for which partial and binary LOMO Spearman rho "
            f"are >= {args.rank_threshold} at K-1, K, and K+1"
        ),
        "all_k_lomo_metrics": rows,
    }
    if selected_k is None:
        return {
            **common,
            "status": "no_qualifying_k",
            "selected_k": None,
            "selected_task_ids": [],
            "message": "No three-budget neighborhood satisfies the requested rank threshold.",
        }

    order, distances, population_sums, total_sum, mean_distance = geometry(scores)
    final_phases = np.asarray(
        [
            np.random.default_rng(args.seed_base + draw).random()
            for draw in range(args.starts)
        ],
        dtype=np.float64,
    )
    selected, final_energy, final_unique = best_of_n(
        order,
        selected_k,
        final_phases,
        distances,
        population_sums,
        total_sum,
        mean_distance,
    )
    neighborhood = rows[selected_k - 2 : selected_k + 1]
    pairwise_passes = all(
        float(row[metric]) >= PAIRWISE_DIAGNOSTIC_THRESHOLD
        for row in neighborhood
        for metric in ("partial_pairwise_accuracy", "binary_pairwise_accuracy")
    )
    per_model = []
    for model_index, model in enumerate(panel.model_ids):
        per_model.append(
            {
                "model_id": model,
                "partial_truth": float(true_partial[model_index]),
                "partial_lomo_estimate": float(partial[selected_k - 1, model_index]),
                "binary_truth": float(true_binary[model_index]),
                "binary_lomo_estimate": float(binary[selected_k - 1, model_index]),
            }
        )
    return {
        **common,
        "status": "ok",
        "selected_k": selected_k,
        "selected_fraction": selected_k / n_tasks,
        "task_count_reduction": n_tasks / selected_k,
        "selected_zero_based_indices": [int(index) for index in selected],
        "selected_task_ids": [panel.task_ids[int(index)] for index in selected],
        "final_training_energy": final_energy,
        "final_unique_local_optima": final_unique,
        "selected_neighborhood_metrics": neighborhood,
        "pairwise_diagnostic_threshold": PAIRWISE_DIAGNOSTIC_THRESHOLD,
        "pairwise_diagnostic_passes": pairwise_passes,
        "selected_k_per_model_lomo": per_model,
        "mean_unique_local_optima_at_selected_k_across_folds": float(
            unique_counts[selected_k - 1].mean()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select K and a representative task subset from model scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Accepted inputs:
  * OSWorld2 JSON with complete_case_matrix/model_ids/task_ids/values
  * JSON with model_ids, task_ids, and values (or scores)
  * JSON mapping each model to {task_id: score}
  * JSON mapping each model to [{"task_id": ..., "score": ...}, ...]
  * wide CSV (one model row, one task per column)
  * long CSV with model, task_id, and score columns

Example:
  python select_representative_tasks.py scores.json --output selection.json
""",
    )
    parser.add_argument("scores", type=Path, help="historical model-by-task scores")
    parser.add_argument("--output", type=Path, help="output JSON path")
    parser.add_argument(
        "--score-scale",
        choices=("auto", "unit", "percent"),
        default="auto",
        help="input score scale (default: infer [0,1] versus [0,100])",
    )
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument(
        "--rank-threshold", type=float, default=DEFAULT_RANK_THRESHOLD
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 8)),
        help="parallel LOMO processes (default: up to 8)",
    )
    args = parser.parse_args()
    if not args.scores.is_file():
        parser.error(f"input does not exist: {args.scores}")
    if args.starts < 1:
        parser.error("--starts must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not -1.0 <= args.rank_threshold <= 1.0:
        parser.error("--rank-threshold must lie in [-1,1]")
    if args.output is None:
        args.output = args.scores.with_name(
            args.scores.stem + ".representative-subset.json"
        )
    return args


def main() -> None:
    args = parse_args()
    payload = json_safe(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if payload["status"] == "ok":
        print(f"Selected K={payload['selected_k']} of {payload['n_complete_tasks']} tasks")
        print("Selected task IDs:")
        print(json.dumps(payload["selected_task_ids"], indent=2))
    else:
        print(payload["message"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
