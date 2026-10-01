from pathlib import Path

import yaml

from cua_speedrun.benchmark_sources import benchmark_source_metadata
from cua_speedrun.commands.install_assets import _BENCHMARK_RESOURCES
from cua_speedrun.service.templates_catalog import list_templates


ROOT = Path(__file__).resolve().parents[1]
SELECTED = """ardour_env/broadcast_podcast_stem_delivery
dhis2_env/rmncah_scorecard_dashboard
docker_desktop_env/diagnose_broken_microservices_stack
gpredict_env/poes_downlink_schedule_setup
gvsig_desktop_env/vulnerability_map_remote_communities
jstock_env/quarterly_portfolio_rebalance
librehealth_ehr_env/implement_lab_workflow_and_process_patient
moodle_env/configure_tiered_assessment_pathway
nextgen_connect_integration_engine_env/adt_census_lab_validation_pipeline
nosh_env/care_quality_remediation
odoo_inventory_env/pharma_lot_recall_quarantine
openclinic_ga_env/insured_consultation_billing
oracle_database_env/claims_pipeline_reconciliation
project_libre_env/schedule_recovery_rebaseline
pymol_env/kinase_selectivity_comparison
redmine_env/q1_milestone_reconciliation
rocket_chat_env/compliance_audit_remediation
snap_env/multicriteria_suitability_mapping
splunk_env/threat_intel_enrichment_pipeline
sumo_env/optimize_network_signal_timing
thunderbird_env/litigation_email_triage
wireshark_env/web_app_breach_investigation
wondershare_edrawmax_env/healthcare_it_architecture_review
woo_commerce_env/launch_coffee_product_line
wordpress_env/launch_woocommerce_coffee_roastery
wps_presentation_env/rebrand_restructure_pitch_deck""".splitlines()


def test_cua_world_26_preserves_task_membership_and_order():
    parent = yaml.safe_load(
        (ROOT / "benchmarks/cua-world-offline/benchmark-source.yaml").read_text()
    )
    subset = yaml.safe_load(
        (ROOT / "benchmarks/cua-world-26/benchmark-source.yaml").read_text()
    )
    canonical = {
        entry["env_name"] + "/" + entry["task_name"]: entry
        for entry in parent["tasks"]
    }
    assert len(SELECTED) == len(set(SELECTED)) == 26
    assert subset["name"] == "cua-world-26"
    assert subset["tasks"] == [canonical[key] for key in SELECTED]
    assert subset["selection"]["selected_task_count"] == 26
    assert subset["selection"]["platform_counts"] == {
        "windows": 0, "linux": 26, "android": 0
    }
    assert len({entry["id"] for entry in subset["tasks"]}) == 26
    assert subset["source_benchmark"] == parent["source_benchmark"]
    assert subset["protocol"] == {
        **parent["protocol"],
        "instruction_prefix": (
            "Interact with the task computer only through its graphical user interface (GUI). "
            "Do not use terminal commands, scripts, or direct APIs to bypass the GUI."
        ),
        "verifier": {
            "mode": "vlm_checklist",
            "spec": {
                "backend": "gemini",
                "model": "gemini-3-flash-preview",
                "frame_strategy": "all",
                "max_frames": -1,
                "completion_threshold": 100,
                "integrity_threshold": 1,
            },
        },
    }
    assert subset["host_runtime"] == {
        "native_runner": "gym-anything", "forward_env": ["GEMINI_API_KEY"]
    }
    assert subset["prepare"]["default_environment"] == "modal-native"
    assert subset["required_devices"] == parent["required_devices"] == {
        "docker_desktop_env": ["/dev/kvm"]
    }
    assert subset["version"] == "0.5"
    for key in ("path", "function"):
        assert subset["materializer"][key] == parent["materializer"][key]
    assert set(subset["setup_patches"]) == {
        "jstock_env", "odoo_inventory_env", "dhis2_env",
        "oracle_database_env", "rocket_chat_env", "splunk_env",
        "openclinic_ga_env", "project_libre_env", "pymol_env", "wps_presentation_env",
    }
    assert subset["materializer"]["inputs"] == [
        "scripts/import_cua_world_long.py", "benchmarks/desktop-images.json",
        *subset["setup_patches"].values()
    ]
    assert parent["setup_patches"] == subset["setup_patches"]
    assert set(parent["materializer"]["inputs"]) == set(subset["materializer"]["inputs"])
    environments = {entry["env_name"] for entry in subset["tasks"]}
    assert subset["asset_setup"] == {
        key: [name for name in names if name in environments]
        for key, names in parent["asset_setup"].items()
    }


def test_cua_world_26_is_discoverable_and_bundled():
    metadata = benchmark_source_metadata(ROOT / "benchmarks/cua-world-26")
    assert metadata["name"] == "cua-world-26"
    assert metadata["task_count"] == 26
    assert "cua-world-26" in _BENCHMARK_RESOURCES
    assert all("cua-world-26" in agent["compatible_benchmarks"] for agent in list_templates())
