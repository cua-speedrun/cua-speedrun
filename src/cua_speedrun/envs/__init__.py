"""Environment backends.

A backend turns a task's `env` block plus a seed into a live environment
the agent can drive. Backends register here by name, so the executor and
CLI stay generic.
"""

from __future__ import annotations

from cua_speedrun.envs.base import Backend


def get_backend(name: str) -> Backend:
    if name == "gym-anything-modal-native":
        from cua_speedrun.envs.gym_anything import GymAnythingBackend

        return GymAnythingBackend(runner="modal_native", name=name,
                                  require_runner="ModalNativeRunner")
    if name in ("gym-anything", "gym-anything-local"):
        from cua_speedrun.envs.gym_anything import GymAnythingBackend

        return GymAnythingBackend(
            # "qemu" is gym-anything's QEMU-family selector. Gym-anything,
            # not this harness, chooses its native or Apptainer launcher.
            runner="qemu",
            name="gym-anything-local",
        )
    if name in ("gym-anything-qemu", "gym-anything-qemu-native"):
        from cua_speedrun.envs.gym_anything import GymAnythingBackend

        return GymAnythingBackend(runner="qemu_native", name="gym-anything-local",
                                  require_runner="QemuNativeRunner")
    if name == "gym-anything-qemu-apptainer":
        from cua_speedrun.envs.gym_anything import GymAnythingBackend

        # For hosts without native QEMU (e.g. cluster nodes): QEMU runs inside
        # an Apptainer container. Not to be confused with gym-anything's
        # ApptainerDirectRunner (containers without a VM). gym-anything has no
        # direct key for this runner; its "qemu" key auto-selects
        # QemuApptainerRunner when apptainer is installed, and require_runner
        # fails closed if the auto-selection lands anywhere else.
        return GymAnythingBackend(runner="qemu", name="gym-anything-qemu-apptainer",
                                  require_runner="QemuApptainerRunner")
    if name == "gym-anything-avd-native":
        from cua_speedrun.envs.gym_anything import GymAnythingBackend

        # Android environments inside the Modal env sandbox: the AVD emulator
        # runs natively on the sandbox kernel.
        return GymAnythingBackend(runner="avd_native", name="gym-anything-avd-native",
                                  require_runner="AVDNativeRunner")
    if name == "modal-native":
        from cua_speedrun.envs.modal_native import ModalNativeBackend

        # The OSWorld desktop booted directly on a Modal sandbox kernel (no
        # KVM). Only functions inside the env sandbox; registered here so
        # every env plane constructs its environment through this factory.
        return ModalNativeBackend()
    raise ValueError(
        f"unknown backend '{name}' "
        "(known: gym-anything-local, gym-anything-qemu, "
        "gym-anything-qemu-apptainer, gym-anything-avd-native, modal-native)"
    )
