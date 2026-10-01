"""Modal-native env-plane entrypoint: runs INSIDE the env sandbox.

A transport shim, exactly like ``env_plane``: the environment is constructed
by ``ModalNativeBackend`` (registered as ``modal-native`` in
``cua_speedrun.envs``), which owns the pivot_root boot, pre-task setup,
privileged-material scrub, evaluator warmup, and the OSWorld checker. This
module only wires the launcher's CS_* variables to the backend and serves
the gateway, which holds the single monotonic clock.

Configuration comes from the CS_* variables documented in ``plane_common``,
plus the modal-native knobs the backend reads (CS_DESKTOP_USER,
and CS_DESKTOP_SETTLE_SEC).
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    from cua_speedrun.remote.native_realtime import isolate_controller_clock

    isolate_controller_clock()
    seed = int(os.environ.get("CS_SEED", "0"))

    from cua_speedrun.envs import get_backend
    from cua_speedrun.remote.plane_common import (
        WORKDIR,
        read_env_spec,
        serve_env_plane,
    )

    workdir = WORKDIR
    workdir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(os.environ.get("CS_ENV_BACKEND", "modal-native"))
    prepared = backend.prepare(read_env_spec(), seed, workdir)

    serve_env_plane(prepared, workdir=workdir, plane="modal-native")
    return 0


if __name__ == "__main__":
    sys.exit(main())
