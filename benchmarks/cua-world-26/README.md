# CUA-World Long: K26

This benchmark contains 26 Linux tasks from the pinned CUA-World
`long_horizon` split. Task membership, source revision, protocol, and setup
patches are defined in `benchmark-source.yaml`.

## Execution and grading

Tasks use GUI-only interaction, a 500-step environment limit, and a six-hour
agent timeout. Agent templates may also require their own maximum-step
setting.

Grading uses Gym-Anything's VLM checklist evaluator with
`gemini-3-flash-preview` and evaluator-side `GEMINI_API_KEY`. It receives all
trajectory frames and requires 100% completion and all integrity checks.

Setup patches are applied to copied environments during materialization;
upstream task instructions, checklists, and source files are unchanged.
`SOURCE.json` records patch hashes and effective environment hashes.

On Modal, native and QEMU Linux filesystem caches stop after `pre_start`
installation. The `post_start` and task-setup hooks run after every boot.
The Docker Desktop task requires a KVM-enabled runtime.

## Tasks

- `ardour_env/broadcast_podcast_stem_delivery`
- `dhis2_env/rmncah_scorecard_dashboard`
- `docker_desktop_env/diagnose_broken_microservices_stack`
- `gpredict_env/poes_downlink_schedule_setup`
- `gvsig_desktop_env/vulnerability_map_remote_communities`
- `jstock_env/quarterly_portfolio_rebalance`
- `librehealth_ehr_env/implement_lab_workflow_and_process_patient`
- `moodle_env/configure_tiered_assessment_pathway`
- `nextgen_connect_integration_engine_env/adt_census_lab_validation_pipeline`
- `nosh_env/care_quality_remediation`
- `odoo_inventory_env/pharma_lot_recall_quarantine`
- `openclinic_ga_env/insured_consultation_billing`
- `oracle_database_env/claims_pipeline_reconciliation`
- `project_libre_env/schedule_recovery_rebaseline`
- `pymol_env/kinase_selectivity_comparison`
- `redmine_env/q1_milestone_reconciliation`
- `rocket_chat_env/compliance_audit_remediation`
- `snap_env/multicriteria_suitability_mapping`
- `splunk_env/threat_intel_enrichment_pipeline`
- `sumo_env/optimize_network_signal_timing`
- `thunderbird_env/litigation_email_triage`
- `wireshark_env/web_app_breach_investigation`
- `wondershare_edrawmax_env/healthcare_it_architecture_review`
- `woo_commerce_env/launch_coffee_product_line`
- `wordpress_env/launch_woocommerce_coffee_roastery`
- `wps_presentation_env/rebrand_restructure_pitch_deck`
