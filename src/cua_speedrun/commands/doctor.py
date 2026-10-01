"""Capability-oriented diagnostics for one cua-speedrun installation."""

from __future__ import annotations

import argparse
import json

from .diagnostics import Capability
from .doctor_dashboard import dashboard_capability, modal_capability
from .doctor_local import (
    local_agent_capability,
    local_gpu_capability,
    local_vm_capability,
    osworld_capability,
)
from .paths import InstallationPaths, add_home_argument, configure_process


_REQUIRE_CHOICES = (
    "dashboard",
    "local-vms",
    "osworld",
    "local-agent",
    "local-gpu",
    "modal",
)


def register_doctor_command(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "doctor", help="show which execution capabilities this machine has"
    )
    add_home_argument(parser)
    parser.add_argument(
        "--require",
        action="append",
        choices=_REQUIRE_CHOICES,
        help="exit nonzero unless this capability is ready; repeat as needed",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.set_defaults(_operator_handler=run_doctor)


def collect_capabilities(paths: InstallationPaths) -> list[Capability]:
    configure_process(paths)
    return [
        dashboard_capability(paths),
        local_vm_capability(),
        osworld_capability(paths),
        local_agent_capability(paths),
        local_gpu_capability(),
        modal_capability(),
    ]


def _render(capabilities: list[Capability], paths: InstallationPaths) -> str:
    lines = [f"cua-speedrun home: {paths.home}", ""]
    for capability in capabilities:
        status = "READY" if capability.ready else "NOT READY"
        lines.append(f"{capability.title}: {status}")
        for check in capability.checks:
            marker = "ok" if check.ok else ("--" if not check.required else "!!")
            lines.append(f"  [{marker}] {check.name}: {check.detail}")
        lines.append("")
    lines.append(
        "Capabilities are independent: API agents can run without a GPU, and "
        "remote runs do not require local KVM."
    )
    return "\n".join(lines)


def run_doctor(args: argparse.Namespace) -> int:
    paths = InstallationPaths.resolve(args.home)
    capabilities = collect_capabilities(paths)
    if args.json:
        print(
            json.dumps(
                {
                    "home": str(paths.home),
                    "capabilities": [item.to_dict() for item in capabilities],
                },
                indent=2,
            )
        )
    else:
        print(_render(capabilities, paths))
    by_key = {capability.key: capability for capability in capabilities}
    required = args.require or ["dashboard"]
    return 0 if all(by_key[key].ready for key in required) else 1


__all__ = ["collect_capabilities", "register_doctor_command", "run_doctor"]
