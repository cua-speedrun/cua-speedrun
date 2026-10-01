"""Check the real pinned source package; never launch or simulate a desktop."""

import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib

import pytest
import yaml

from cua_speedrun.benchmark_sources import materialize_benchmark


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/cua-world-26/benchmark-source.yaml"
PREFIX = "benchmarks/cua_world/environments/"


@pytest.fixture(scope="module")
def package():
    return materialize_benchmark(SOURCE)


def test_materialized_package_changes_only_declared_scripts_and_configuration(package):
    source = yaml.safe_load(SOURCE.read_text())
    provenance = json.loads((package / "SOURCE.json").read_text())["setup_patches"]
    assert set(provenance) == set(source["setup_patches"])
    environments = {entry["env_name"] for entry in source["tasks"]}
    selected_tasks = {
        (entry["env_name"], f"tasks/{entry['task_name']}/task.json")
        for entry in source["tasks"]
    }
    recipes = {}
    for env, path in source["setup_patches"].items():
        recipe = ROOT / path
        recipes[env] = yaml.safe_load(recipe.read_text())["replacements"]
        assert provenance[env]["patch_sha256"] == hashlib.sha256(recipe.read_bytes()).hexdigest()

    process = subprocess.Popen(
        ["git", "-C", str(ROOT / "third_party/gym-anything"), "archive",
         source["source_benchmark"]["commit"],
         *[PREFIX + env for env in sorted(environments)]],
        stdout=subprocess.PIPE,
    )
    checked_scripts = set()
    checked_environments = set()
    checked_tasks = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                relative = Path(member.name).relative_to(PREFIX)
                env, name = relative.parts[0], Path(*relative.parts[1:]).as_posix()
                expected = archive.extractfile(member).read()
                actual = (package / "gym-anything" / member.name).read_bytes()
                if (env, name) in selected_tasks:
                    task = json.loads(expected)
                    task["success"] = source["protocol"]["verifier"]
                    task["description"] = (
                        source["protocol"]["instruction_prefix"] + "\n\n" + task["description"]
                    )
                    assert json.loads(actual) == task
                    checked_tasks.add((env, name))
                    continue
                if name == "env.json":
                    config = json.loads(expected)

                    def portable(value):
                        if isinstance(value, dict):
                            return {key: portable(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [portable(item) for item in value]
                        prefix = PREFIX + env + "/"
                        return value[len(prefix):] if isinstance(value, str) and value.startswith(prefix) else value

                    config = portable(config)
                    if env in provenance:
                        config["hooks"]["pre_start"] += (
                            "\n# cua-speedrun-setup-sha256=" + provenance[env]["setup_sha256"]
                        )
                    assert json.loads(actual) == config
                    checked_environments.add(env)
                    continue
                for change in recipes.get(env, []):
                    if change["path"] == name:
                        before, after = change["old"].encode(), change["new"].encode()
                        assert expected.count(before) == change.get("count", 1)
                        expected = expected.replace(before, after)
                        checked_scripts.add((env, name))
                assert actual == expected, (env, name)
                if (env, name) in checked_scripts:
                    subprocess.run(["bash", "-n"], input=actual, check=True)
        assert process.wait() == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdout.close()
    assert checked_environments == environments
    assert checked_tasks == selected_tasks
    assert checked_scripts == {
        (env, change["path"])
        for env, changes in recipes.items() for change in changes
    }


def test_all_cua_world_26_tasks_load_native_prompt_and_gemini_checklist_configuration(package):
    from gym_anything.config.loading import _load_taskspec
    from gym_anything.verification.vlm_checklist import VLMChecklistConfig
    from cua_speedrun.remote.modal_env import _host_runtime

    source = yaml.safe_load(SOURCE.read_text())
    for entry in source["tasks"]:
        environment = package / "gym-anything" / PREFIX / entry["env_name"]
        task_root = environment / "tasks" / entry["task_name"]
        task = _load_taskspec(task_root / "task.json")
        # These pinned tasks use description, not a natural_language override.
        assert not task.natural_language
        expected_instruction = source["protocol"]["instruction_prefix"] + "\n\n" + entry["description"]
        assert task.description.strip() == expected_instruction
        task_yaml = package / "tasks" / f"{entry['env_name']}__{entry['task_name']}" / "task.yaml"
        assert yaml.safe_load(task_yaml.read_text())["description"] == expected_instruction
        assert task.success.mode == "vlm_checklist"
        config = VLMChecklistConfig.from_spec(task.success.spec)
        assert (config.backend, config.model) == ("gemini", "gemini-3-flash-preview")
        assert (config.frame_strategy, config.max_frames) == ("all", -1)
        assert (config.completion_threshold, config.integrity_threshold) == (100, 1)
        assert json.loads((task_root / config.checklist).read_text())["task_completion"]
        assert _host_runtime(environment) == {
            "native_runner": "gym-anything", "forward_env": ["GEMINI_API_KEY"]
        }


def test_corrected_financial_seed_and_dbeaver_configuration_syntax(package):
    environments = package / "gym-anything" / PREFIX
    script = (environments / "jstock_env/tasks/quarterly_portfolio_rebalance/setup_task.sh").read_text()
    deposit = script.split('cat > "$PORTFOLIO_DIR/depositsummary.csv" << \'CSVEOF\'\n')[1].split("\nCSVEOF")[0]
    rows = list(csv.DictReader(io.StringIO(deposit)))
    assert list(rows[0]) == ["Date", "Cash", "Comment"]
    assert rows[0]["Cash"] == "100000.0"
    assert "XAUTHORITY=/run/user/1000/gdm/Xauthority" not in script
    oracle = (environments / "oracle_database_env/scripts/setup_oracle.sh").read_text()
    config = oracle.split("<<'PRODUCT_CONFIG' || exit 1\n")[1].split("\nPRODUCT_CONFIG")[0]
    ast.parse(config)
    assert "DBEAVER_DATA=/home/ga/.local/share" in oracle
    assert "product-config.json" in config


def test_cua_world_26_and_all_patch_inputs_are_bundled():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    targets = config["tool"]["hatch"]["build"]["targets"]
    name = "benchmarks/cua-world-26"
    assert targets["wheel"]["force-include"]["benchmarks"] == "cua_speedrun/_resources/benchmarks"
    assert "/benchmarks" in targets["sdist"]["include"]
    source = yaml.safe_load(SOURCE.read_text())
    for path in source["setup_patches"].values():
        assert (ROOT / path).is_file()
        assert path.startswith(name + "/")


def test_wps_setup_uses_installed_versions_first_run_preferences(package):
    script = (package / "gym-anything" / PREFIX /
              "wps_presentation_env/scripts/setup_wps.sh").read_text()
    settings = script.split("Office.conf << 'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    import configparser

    config = configparser.ConfigParser(interpolation=None)
    config.read_string(settings)
    assert config.getboolean("6.0", r"common\AcceptedEULA")
    assert config.getboolean("6.0", r"common\UserInfo\ACUPI")


def test_jstock_export_normalizes_real_seed_without_changing_amount(package, tmp_path):
    task = package / "gym-anything" / PREFIX / "jstock_env/tasks/quarterly_portfolio_rebalance"
    script = (task / "setup_task.sh").read_text()
    deposit = script.split('cat > "$PORTFOLIO_DIR/depositsummary.csv" << \'CSVEOF\'\n')[1].split("\nCSVEOF")[0]
    native_csv = tmp_path / "depositsummary.csv"
    native_csv.write_text(deposit)
    exporter = (task / "export_result.sh").read_text().split("python3 << 'PYEOF'\n")[1].split("\nPYEOF")[0]
    module = ast.parse(exporter)
    reader = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "read_csv_entries")
    # Execute only the pure CSV reader against the real task seed, never the
    # exporter shell (which would stop the application), or a simulated VM.
    import os
    namespace = {"os": os, "csv": csv, "result": {}}
    exec(compile(ast.Module(body=[reader], type_ignores=[]), "export_result.sh", "exec"), namespace)
    rows = namespace["read_csv_entries"](str(native_csv), amount_column="Cash")
    assert rows[0]["Cash"] == rows[0]["Amount"] == "100000.0"
    assert native_csv.read_text() == deposit
    assert "Amount" not in namespace["read_csv_entries"](str(native_csv))[0]
    assert namespace["result"] == {}
    assert 'amount_column="Dividend"' in exporter


def test_jstock_companion_csvs_use_pinned_native_column_order(package):
    environment = package / "gym-anything" / PREFIX / "jstock_env"
    expected = {
        "sellportfolio.csv": [
            "Code", "Symbol", "Purchase Date", "Date", "Units", "Selling Price",
            "Purchase Price", "Selling Value", "Purchase Value", "Purchase Broker",
            "Purchase Clearing Fee", "Purchase Stamp Duty", "Gain/Loss Price",
            "Gain/Loss Value", "Gain/Loss %", "Broker", "Clearing Fee", "Stamp Duty",
            "Net Selling Value", "Net Gain/Loss Value", "Net Gain/Loss %", "Comment",
        ],
        "depositsummary.csv": ["Date", "Cash", "Comment"],
        "dividendsummary.csv": ["Date", "Code", "Symbol", "Dividend", "Comment"],
    }
    for name in ("scripts/setup_jstock.sh", "tasks/quarterly_portfolio_rebalance/setup_task.sh"):
        script = (environment / name).read_text()
        for filename, columns in expected.items():
            sections = script.split(f'/{filename}" << \'CSVEOF\'\n')[1:]
            assert sections, (name, filename)
            for section in sections:
                rows = list(csv.reader(io.StringIO(section.split("\nCSVEOF")[0])))
                assert rows[0] == columns
                if filename != "depositsummary.csv":
                    assert len(rows) == 1  # Never pre-fill the agent's trades/dividends.
    setup = (environment / "scripts/setup_jstock.sh").read_text()
    preference = setup.split("<<'JSTOCK_OPTIONS' || exit 1\n")[1].split("\nJSTOCK_OPTIONS")[0]
    ast.parse(preference)
    assert "path.read_text" in preference  # Require native saved options; do not invent a config.
    assert "<isAutoUpdateNewsEnabled>false</isAutoUpdateNewsEnabled>" in preference
    assert "count != 1" in preference


def test_odoo_installs_only_missing_modules_and_checks_application(package):
    setup = (package / "gym-anything" / PREFIX / "odoo_inventory_env/scripts/setup_odoo.sh").read_text()
    correction = setup.split("# Every database path must satisfy")[1].split("# Set up Firefox profile")[0]
    assert "installed.state='installed' WHERE installed.id IS NULL" in correction
    assert '-i "$MISSING_MODULES"' in correction
    assert '-i "$REQUIRED_MODULES"' not in correction
    assert "--without-demo=False" not in correction
    assert "docker-compose stop web || exit 1" in correction
    assert "set -o pipefail" in correction
    assert "wait_for_database 180 || exit 1" in correction
    assert 'if [ "$MODULE_COUNT" != "4" ]; then' in correction


def test_jstock_warmup_requires_normal_close_and_native_options(package):
    setup = (package / "gym-anything" / PREFIX / "jstock_env/scripts/setup_jstock.sh").read_text()
    shutdown = setup.split("# Linux saves native options")[1].split('echo "JStock warm-up complete"')[0]
    assert "wmctrl -F -c 'JStock News'" in shutdown
    assert "wmctrl -F -c 'JStock - Free Stock Market Software'" in shutdown
    assert shutdown.index("-c 'JStock News'") < shutdown.index("-c 'JStock - Free")
    assert "pgrep -u ga -f 'jstock[.]jar'" in shutdown
    assert "if [ ! -s /home/ga/.jstock/1.0.7/config/options.xml ]; then" in shutdown
    assert "attempt<60" in shutdown
    assert "pkill" not in shutdown
    assert "xdotool" not in shutdown
    assert setup.index("JStock exited normally and saved native options") < setup.index("<<'JSTOCK_OPTIONS'")


def test_jstock_export_requires_normal_exit_before_reading_native_state(package):
    task = package / "gym-anything" / PREFIX / "jstock_env/tasks/quarterly_portfolio_rebalance"
    exporter = (task / "export_result.sh").read_text()
    shutdown = exporter.split("# 1. Close JStock normally")[1].split("# 2. Read task start timestamp")[0]
    assert "wmctrl -F -c 'JStock News'" in shutdown
    assert "wmctrl -F -c 'JStock - Free Stock Market Software'" in shutdown
    assert shutdown.index("-c 'JStock News'") < shutdown.index("-c 'JStock - Free")
    assert shutdown.count("pgrep -u ga -f 'jstock[.]jar'") == 2
    assert "attempt<60" in shutdown
    assert "sleep 1" in shutdown
    assert "pkill" not in exporter
    assert "xdotool" not in shutdown
    # Inspect the real script without executing desktop commands or replacing
    # them with fakes. A failed normal close must stop before CSV extraction.
    after_wait = shutdown.split("done\n", 1)[1]
    assert "if pgrep -u ga -f 'jstock[.]jar' > /dev/null; then" in after_wait
    assert 'echo "ERROR: JStock did not exit normally; refusing to export stale state" >&2' in after_wait
    assert "exit 1\nfi" in after_wait
    assert exporter.index("exit 1") < exporter.index("python3 << 'PYEOF'")
    subprocess.run(["bash", "-n"], input=exporter, text=True, check=True)


def test_checkpoint_keys_include_real_base_presets(package):
    from gym_anything.config.loading import _load_envspec
    from gym_anything.runtime.runners.qemu_apptainer import _get_env_hash

    source = yaml.safe_load(SOURCE.read_text())
    recipe = (ROOT / "src/cua_speedrun/envs/cua_world_runtime.py").read_bytes()

    def key(path):
        # Loading/composing a spec does not construct a runner or launch a VM.
        raw = _get_env_hash(_load_envspec(path / "env.json"))
        return hashlib.sha256(raw.encode() + b"\0" + recipe).hexdigest()[:16]

    changed = set()
    for task in source["tasks"]:
        env = task["env_name"]
        original = ROOT / "third_party/gym-anything" / PREFIX / env
        effective = package / "gym-anything" / PREFIX / env
        if key(original) != key(effective):
            changed.add(env)
    assert changed == set(source["setup_patches"])
