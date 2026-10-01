"""User-code starters for the submit page.

Each folder contains init.py, agent.py, and optional agent.json metadata.
The metadata declares hardware defaults and required environment-variable names.
It never declares or selects a track: the submitted scripts decide whether to
run a local model, call an external API, use both, or use neither.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from cua_speedrun.resources import resource_root


TEMPLATES_DIR: Path | None = None
def agent_metadata(path: Path) -> dict:
    metadata = path / "agent.json"
    if not metadata.is_file():
        return {}
    import json

    data = json.loads(metadata.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{metadata}: expected an object")
    allowed = {"description", "gpu", "required_environment_variables", "optional_environment_variables", "compatible_benchmarks"}
    if set(data) - allowed:
        raise ValueError(f"{metadata}: unknown fields {sorted(set(data) - allowed)}")
    if data.get("gpu") is not None and (
        not isinstance(data["gpu"], str) or not data["gpu"].strip()
    ):
        raise ValueError(f"{metadata}: gpu must be a nonempty string or null")
    for field in ("required_environment_variables", "optional_environment_variables", "compatible_benchmarks"):
        if field in data and (
            not isinstance(data[field], list)
            or not all(isinstance(value, str) and value for value in data[field])
        ):
            raise ValueError(f"{metadata}: {field} must be a list of names")
    return data


def list_templates() -> list[dict]:
    from cua_speedrun.benchmark_sources import benchmark_catalog_paths
    import yaml

    benchmarks = []
    for path in benchmark_catalog_paths(resource_root()):
        definition = path / "benchmark-source.yaml"
        if not definition.is_file():
            definition = path / "manifest.yaml"
        benchmarks.append(str(yaml.safe_load(definition.read_text())["name"]))
    out = []
    for path in sorted((TEMPLATES_DIR or resource_root() / "agents").iterdir()):
        if (path / "init.py").is_file() and (path / "agent.py").is_file():
            item = {"name": path.name, **agent_metadata(path)}
            item.setdefault("description", path.name)
            item.setdefault("required_environment_variables", [])
            item.setdefault("compatible_benchmarks", benchmarks)
            item["compatible_benchmark"] = next(iter(item["compatible_benchmarks"]), "")
            out.append(item)
    return out


def template_required_environment_variables(name: str) -> list[str]:
    path = template_dir(name)
    meta = agent_metadata(path) if path else {}
    return list(meta.get("required_environment_variables") or ())


def _refuse_stale_installed_copy(path: Path, name: str) -> None:
    """Fail closed when the installed template copy has drifted from a
    development checkout.

    ``cua-speedrun install`` copies templates into CUA_SPEEDRUN_HOME and
    submissions resolve against that copy, so a template edited in the
    checkout must be re-installed before using the installed copy.
    """
    from cua_speedrun.resources import source_checkout_root

    checkout = source_checkout_root()
    if checkout is None:
        return
    source = checkout / "agents" / name
    if source.resolve() == path.resolve() or not source.is_dir():
        return
    for script in ("init.py", "agent.py", "agent.json"):
        source_file = source / script
        installed_file = path / script
        if not source_file.is_file():
            continue
        if not installed_file.is_file() or source_file.read_bytes() != installed_file.read_bytes():
            raise RuntimeError(
                f"installed template {name!r} is stale: {installed_file} "
                f"differs from the checkout at {source_file}; rerun "
                f"`cua-speedrun setup` (or `cua-speedrun install`) to refresh the installed copy "
                f"before submitting"
            )


def template_dir(name: str) -> Path | None:
    """Resolve a template name to its folder, refusing path escapes."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    root = (TEMPLATES_DIR or resource_root() / "agents").resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        return None
    if (path / "init.py").is_file() and (path / "agent.py").is_file():
        _refuse_stale_installed_copy(path, name)
        return path
    return None


def template_zip(name: str) -> bytes | None:
    """Zip a template's two scripts (init.py + agent.py at the root)."""
    path = template_dir(name)
    if path is None:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as zf:
        zf.write(path / "init.py", "init.py")
        zf.write(path / "agent.py", "agent.py")
    return buf.getvalue()
