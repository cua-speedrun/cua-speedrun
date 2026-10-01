"""Best-effort cleanup for Modal resources recorded before cancellation."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor


def terminate_sandboxes(sandbox_ids: Iterable[str]) -> list[str]:
    """Terminate known Modal sandboxes and return IDs that could not be stopped."""
    import modal

    def stop(sandbox_id):
        failed = []
        owner = None
        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
            owner = sandbox.get_tags().get("cs-native-owner")
            sandbox.terminate()
        except modal.exception.NotFoundError:
            pass
        except Exception:
            failed.append(sandbox_id)
        if owner:
            try:
                for child in modal.Sandbox.list(tags={"cs-parent": owner}):
                    try:
                        child.terminate()
                    except Exception:
                        failed.append(child.object_id)
            except Exception:
                failed.append(sandbox_id)
        return failed

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [failed for result in pool.map(stop, sorted(set(sandbox_ids))) for failed in result]


__all__ = ["terminate_sandboxes"]
