"""Installation paths and process configuration for operator commands."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cua_speedrun.config import load_dotenv
from cua_speedrun.resources import bundled_resource_root


def default_home() -> Path:
    configured = os.environ.get("CUA_SPEEDRUN_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local/share"
    return (base / "cua-speedrun").resolve()


@dataclass(frozen=True)
class InstallationPaths:
    home: Path

    @classmethod
    def resolve(cls, value: str | Path | None = None) -> "InstallationPaths":
        home = Path(value).expanduser().resolve() if value else default_home()
        return cls(home=home)

    @property
    def config_file(self) -> Path:
        return self.home / "config.env"

    @property
    def database(self) -> Path:
        return self.home / "platform.db"

    @property
    def store(self) -> Path:
        return self.home / "store"

    @property
    def runs(self) -> Path:
        return self.home / "runs"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def runtime(self) -> Path:
        return self.home / "runtime"

    @property
    def resource_root(self) -> Path:
        return self.runtime / "cua-speedrun"

    @property
    def agent_runtime(self) -> Path:
        return self.runtime / "agent-python"

    @property
    def agent_python(self) -> Path:
        return self.agent_runtime / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )

    @property
    def gym_anything_root(self) -> Path:
        return self.runtime / "gym-anything"

    @property
    def osworld_image(self) -> Path:
        return self.cache / "gym-anything/qemu/osworld_ubuntu.qcow2"

    @property
    def install_record(self) -> Path:
        return self.home / "install.json"

    def create_directories(self) -> None:
        for path in (
            self.home,
            self.store,
            self.runs,
            self.cache,
            self.logs,
            self.runtime,
        ):
            path.mkdir(parents=True, exist_ok=True)


def configure_process(paths: InstallationPaths) -> dict[str, str]:
    """Load user settings, then install deterministic home-derived defaults."""
    # Process variables always win because load_dotenv and setdefault never
    # replace them.  The selected home's stable secret/config comes next,
    # followed by home-derived paths.  A checkout-local .env is only a final
    # compatibility source for values the installation did not define.
    os.environ["CUA_SPEEDRUN_HOME"] = str(paths.home)
    load_dotenv(paths.config_file)
    active_resources = (
        paths.resource_root
        if (paths.resource_root / "catalog" / "tracks.yaml").is_file()
        else bundled_resource_root()
    )
    defaults = {
        "CS_DATABASE_URL": f"sqlite:///{paths.database}",
        "CS_STORE_ROOT": str(paths.store),
        "CS_BENCHMARK_CACHE": str(paths.cache / "benchmarks"),
        "CS_RESOURCE_ROOT": str(active_resources),
        "GYM_ANYTHING_ROOT": str(paths.gym_anything_root),
        "GYM_ANYTHING_QEMU_CACHE": str(paths.cache / "gym-anything/qemu"),
        "OSWORLD_QEMU_BASE_IMAGE": str(paths.osworld_image),
        "OSWORLD_APPTAINER_WORK": str(paths.runtime / "osworld-apptainer"),
        "CUA_OSWORLD_CACHE": str(paths.cache / "osworld"),
        "CS_AGENT_PYTHON": str(paths.agent_python),
        "PIP_CACHE_DIR": str(paths.cache / "pip"),
        "UV_CACHE_DIR": str(paths.cache / "uv"),
        "CS_DEV_LOGIN": "1",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    load_dotenv()
    return {
        "CUA_SPEEDRUN_HOME": os.environ["CUA_SPEEDRUN_HOME"],
        **{name: os.environ[name] for name in defaults},
    }


def add_home_argument(parser, *, help_text: bool = True) -> None:
    parser.add_argument(
        "--home",
        metavar="PATH",
        help=(
            "installation state directory; overrides CUA_SPEEDRUN_HOME"
            if help_text
            else None
        ),
    )


__all__ = [
    "InstallationPaths",
    "add_home_argument",
    "configure_process",
    "default_home",
]
