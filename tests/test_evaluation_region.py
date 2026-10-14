"""Offline contract checks against the repository's real OSWorld benchmark."""

from pathlib import Path

import pytest

from cua_speedrun.runplan import RunPlan
from cua_speedrun.service.db import BenchmarkRow, Track
from cua_speedrun.service.plans import plan_for_catalog_rows


def catalog_rows():
    root = Path(__file__).resolve().parents[1]
    return (
        Track(
            name="open-l40s-shared-agents1",
            gpu="L40S",
            eval_algorithm="shared-agent-vllm@2",
            agents_per_evaluation=1,
            server_config={"region": "us-east", "extra_pip": []},
        ),
        BenchmarkRow(
            name="osworld-50", version="0.1",
            path=str(root / "benchmarks" / "osworld-50"),
        ),
    )


def test_region_is_frozen_per_evaluation_without_mutating_track():
    track, benchmark = catalog_rows()
    baseline = plan_for_catalog_rows(track, benchmark, allocate_gpu=False)
    requested = plan_for_catalog_rows(
        track, benchmark, allocate_gpu=False, region="  us-west  "
    )
    expected = baseline.to_dict()
    expected["track"]["server_config"]["region"] = "us-west"
    assert requested.to_dict() == expected
    assert requested.contract_hash != baseline.contract_hash
    assert RunPlan.from_dict(requested.to_dict()).contract_hash == requested.contract_hash
    assert track.server_config == {"region": "us-east", "extra_pip": []}
    assert baseline.server_config["region"] == "us-east"
    assert plan_for_catalog_rows(track, benchmark, allocate_gpu=False) == baseline


@pytest.mark.parametrize("region", ["", "   ", 123, ["us-east"]])
def test_invalid_region_is_rejected_before_queueing(region):
    track, benchmark = catalog_rows()
    with pytest.raises(ValueError, match="non-empty string"):
        plan_for_catalog_rows(track, benchmark, region=region)


def test_region_cannot_be_silently_ignored_by_local_execution():
    track, benchmark = catalog_rows()
    with pytest.raises(ValueError, match="requires remote placement"):
        plan_for_catalog_rows(
            track, benchmark, "local", "local", region="us-east"
        )
