"""Env-plane entrypoint for gym-anything environments: runs INSIDE the env
Modal sandbox.

A transport shim, deliberately free of preparation logic: the environment is
constructed by the same registered Backend the local executor uses
(``get_backend(CS_ENV_BACKEND).prepare``), so runner guards, image
provenance, SSH overrides, cache defaults, scrubbing, and the initial
screenshot check are one implementation, not two. This plane only resolves
the optional seeded generator (whose expected answer must stay env-side) and
serves the gateway; the gateway holds the single monotonic clock, so only
the agent-to-gateway hop crosses the network.

Configuration comes from the CS_* variables documented in ``plane_common``,
plus:

    CS_GENERATOR                    optional path to a seeded generator.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    seed = int(os.environ.get("CS_SEED", "0"))
    generator_path = os.environ.get("CS_GENERATOR") or None

    from cua_speedrun.envs import get_backend
    from cua_speedrun.envs.base import Verdict
    from cua_speedrun.remote.plane_common import (
        WORKDIR,
        read_env_spec,
        serve_env_plane,
    )

    workdir = WORKDIR
    workdir.mkdir(parents=True, exist_ok=True)

    print("[env_plane] booting environment...", flush=True)
    backend = get_backend(os.environ["CS_ENV_BACKEND"])
    prepared = backend.prepare(read_env_spec(), seed, workdir)
    print(
        f"[env_plane] runner={prepared.info.get('runner')} "
        f"qemu_base_image={prepared.info.get('qemu_base_image')} "
        f"scrubbed_mounts={len(prepared.info.get('privileged_mounts_scrubbed') or [])}",
        flush=True,
    )

    # Seeded task: the generator derives the instruction and the privileged
    # expected answer from the seed. Both stay in this sandbox; the agent
    # only ever sees the instruction through the gateway.
    instruction = None
    checker = None
    if generator_path:
        import importlib.util

        gspec = importlib.util.spec_from_file_location("cs_gen", generator_path)
        gen = importlib.util.module_from_spec(gspec)
        gspec.loader.exec_module(gen)
        gres = gen.generate(seed)
        instruction = gres["instruction"]
        expected = gres["expected"]

        def checker() -> Verdict:
            res = gen.check(prepared.adapter, expected)
            return Verdict(bool(res["passed"]), 100.0 if res["passed"] else 0.0,
                           str(res.get("detail", "")))

    serve_env_plane(
        prepared,
        workdir=workdir,
        plane="remote",
        instruction=instruction,
        checker=checker,
        seeded=bool(generator_path),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
