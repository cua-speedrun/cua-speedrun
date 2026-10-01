"""Validate network fields retained by already-frozen legacy run plans.

New plans use ``host-network`` and never construct a model-API domain list.
"""

from __future__ import annotations

from collections.abc import Iterable

NETWORK_GATEWAY_ONLY = "gateway-only"
NETWORK_GATEWAY_PLUS_API = "gateway+api"
NETWORK_HOST = "host-network"


def normalize_api_domains(domains: Iterable[str] | None) -> tuple[str, ...]:
    clean = []
    for domain in domains or ():
        value = str(domain).strip().lower().rstrip(".")
        if not value or "://" in value or "/" in value or " " in value:
            raise ValueError(f"API allowlist entries must be bare domains, got {domain!r}")
        clean.append(value)
    return tuple(sorted(set(clean)))


def api_domains_for_policy(policy: str, domains: Iterable[str] | None) -> tuple[str, ...]:
    normalized = (policy or NETWORK_GATEWAY_ONLY).strip().lower()
    allowed = normalize_api_domains(domains)
    if normalized == NETWORK_GATEWAY_ONLY:
        if allowed:
            raise ValueError("gateway-only plans cannot declare API domains")
        return ()
    if normalized == NETWORK_GATEWAY_PLUS_API:
        if not allowed:
            raise ValueError("gateway+api plans need at least one API domain")
        return allowed
    if normalized == NETWORK_HOST:
        if allowed:
            raise ValueError("host-network plans do not use an API allowlist")
        return ()
    raise ValueError(
        f"unknown network policy {policy!r}; expected "
        f"{NETWORK_GATEWAY_ONLY!r}, {NETWORK_GATEWAY_PLUS_API!r}, or "
        f"{NETWORK_HOST!r}"
    )
