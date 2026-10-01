"""CLI transport for one independent Modal controller per evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import uuid

from cua_speedrun import __version__
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.commands.setup import initialize, modal_credentials
from cua_speedrun.hosted import APP_NAME, VOLUME_NAME, REMOTE_HOME, TERMINAL, is_hosted, initial_status


def controller_image():
    import modal
    import cua_speedrun
    from cua_speedrun.resources import bundled_resource_root, bundled_gym_anything_root

    package = Path(cua_speedrun.__file__).parent
    target = "/opt/controller/cua_speedrun"
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("rsync", "curl", "git", "procps")
             .pip_install(f"cua-speedrun=={__version__}")
             .env({"PYTHONPATH": "/opt/controller", "PYTHONUNBUFFERED": "1"})
             .add_local_dir(package, target,
                            ignore=["**/__pycache__/**", "**/*.pyc", "_resources/**"]))
    resources = bundled_resource_root()
    for name in ("agents", "benchmarks", "catalog", "scripts"):
        image = image.add_local_dir(resources / name, f"{target}/_resources/{name}",
                                    ignore=["**/__pycache__/**", "**/*.pyc", "**/.env"])
    gym = bundled_gym_anything_root() / "benchmarks/cua_world/environments/preset_gnome_systemd"
    return image.add_local_dir(gym, f"{target}/_resources/gym-anything/benchmarks/cua_world/environments/preset_gnome_systemd")


class HostedEvaluations:
    def __init__(self, paths):
        self.paths = paths
        self._sandboxes = {}

    @classmethod
    def open(cls, home=None):
        paths = InstallationPaths.resolve(home)
        initialize(paths)
        token_id, token_secret = modal_credentials()
        if not token_id or not token_secret:
            raise ValueError("Modal credentials are missing; run cua-speedrun setup")
        os.environ.update(MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=token_secret)
        return cls(paths)

    def _volume(self):
        import modal
        return modal.Volume.from_name(VOLUME_NAME)

    def _json(self, run_id, name):
        import modal
        if not is_hosted(run_id):
            raise ValueError("invalid hosted evaluation ID")
        try:
            return json.loads(b"".join(self._volume().read_file(f"{run_id}/{name}")))
        except (FileNotFoundError, modal.exception.NotFoundError) as exc:
            raise ValueError(f"hosted evaluation {run_id} was not found in this Modal environment") from exc

    def _sandbox(self, run_id):
        import modal
        if run_id not in self._sandboxes:
            record = self._json(run_id, "controller.json")
            self._sandboxes[run_id] = modal.Sandbox.from_id(record["sandbox_id"])
        return self._sandboxes[run_id]

    def launch(self, args):
        import modal
        from cua_speedrun.hosted.inputs import pack_inputs
        from cua_speedrun.commands.benchmark import dataset_path
        from cua_speedrun.benchmark_preparation import independent_environment_preparation, prepare_environment
        from cua_speedrun.startup import phase

        request, archive, credentials = pack_inputs(args)
        run_id = "m-" + uuid.uuid4().hex[:16]
        request.update(run_id=run_id, version=__version__, created_at=time.time())
        credentials.update(MODAL_TOKEN_ID=os.environ["MODAL_TOKEN_ID"],
                           MODAL_TOKEN_SECRET=os.environ["MODAL_TOKEN_SECRET"])
        if os.environ.get("MODAL_ENVIRONMENT"):
            credentials["MODAL_ENVIRONMENT"] = os.environ["MODAL_ENVIRONMENT"]
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        app = modal.App.lookup(APP_NAME, create_if_missing=True)

        def start_controller():
            with phase("controller", "Start hosted controller"), modal.enable_output():
                sandbox = modal.Sandbox.create(
                    "python", "-m", "cua_speedrun.hosted.worker", run_id,
                    app=app, image=controller_image(), cpu=2, memory=4096,
                    timeout=24 * 60 * 60, volumes={"/data": volume},
                    secrets=[modal.Secret.from_dict(credentials)],
                    env={"CUA_SPEEDRUN_HOME": REMOTE_HOME},
                    tags={"cua-speedrun-hosted": run_id},
                )
                try:
                    record = {"run_id": run_id, "sandbox_id": sandbox.object_id,
                              "version": __version__, "created_at": request["created_at"]}
                    import io
                    with volume.batch_upload() as upload:
                        upload.put_file(io.BytesIO(json.dumps(record).encode()), f"/{run_id}/controller.json")
                        upload.put_file(io.BytesIO(json.dumps(initial_status(request)).encode()), f"/{run_id}/status.json")
                    sandbox.reload_volumes()
                    sandbox.filesystem.write_bytes(archive, "/tmp/inputs.zip")
                    sandbox.filesystem.write_text(json.dumps(request), "/tmp/request.pending")
                    process = sandbox.exec("mv", "/tmp/request.pending", "/tmp/request.json")
                    process.wait()
                    if process.returncode:
                        raise RuntimeError("could not deliver hosted evaluation request")
                    return sandbox, record
                except BaseException:
                    sandbox.terminate()
                    raise

        dataset = dataset_path(args.dataset)
        environment = request["environment"]
        # Only independent, bundled image recipes can run before the remote
        # task folder exists. Custom preparation stays with its controller.
        overlap = not Path(args.dataset).expanduser().is_dir() and independent_environment_preparation(dataset, environment)
        if overlap:
            sandbox = None

            def prepare_desktop():
                with phase("desktop", "Prepare desktop image"):
                    prepare_environment(dataset, dataset, environment, self.paths)

            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    desktop = pool.submit(prepare_desktop)
                    sandbox, record = start_controller()
                    desktop.result()
            except BaseException:
                if sandbox is not None:
                    sandbox.terminate()
                raise
        else:
            sandbox, record = start_controller()
        self._sandboxes[run_id] = sandbox
        try:
            sandbox.detach()
            self._sandboxes.pop(run_id, None)
            # Record only location metadata. Modal remains the source of truth.
            registry = self.paths.home / "hosted"
            registry.mkdir(exist_ok=True)
            (registry / f"{run_id}.json").write_text(json.dumps(record))
            return {"type": "queued", "run_id": run_id, "host": "modal",
                    "sandbox_id": sandbox.object_id}
        except BaseException as exc:
            sandbox.terminate()
            if isinstance(exc, Exception):
                raise RuntimeError(f"hosted launch failed: {exc}") from exc
            raise

    def status(self, run_id):
        import modal
        try:
            sandbox = self._sandbox(run_id)
            if sandbox.poll() is None:
                try:
                    payload = json.loads(sandbox.filesystem.read_text("/work/status.json"))
                    if payload["stage"] in TERMINAL:
                        sandbox.wait()
                        return self._json(run_id, "status.json")
                    return payload
                except modal.exception.SandboxFilesystemError:
                    if sandbox.poll() is not None:
                        sandbox = None
                    # The durable checkpoint remains available while the
                    # controller filesystem starts up or shuts down.
            else:
                sandbox = None
        except modal.exception.NotFoundError:
            sandbox = None
        payload = self._json(run_id, "status.json")
        if sandbox is None and payload["stage"] not in TERMINAL:
            payload.update(stage="failed", error="Modal controller stopped before completing the evaluation")
        return payload

    def cancel(self, run_id):
        payload = self.status(run_id)
        if payload["stage"] in TERMINAL:
            return payload
        sandbox = self._sandbox(run_id)
        try:
            sandbox.filesystem.write_text("cancel", "/work/cancel")
        except Exception as exc:
            if sandbox.poll() is not None:
                return self.status(run_id)
            raise RuntimeError(f"could not request cancellation: {exc}") from exc
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            payload = self.status(run_id)
            if payload["stage"] in TERMINAL:
                sandbox.wait()
                return payload
            time.sleep(1)
        raise RuntimeError(f"cancellation is still in progress; check cua-speedrun status {run_id}")

    def evaluations(self, *, active_only=False, limit=None):
        import modal
        try:
            records = [self._json(entry.path.strip('/'), "controller.json")
                       for entry in self._volume().iterdir("/", recursive=False)
                       if is_hosted(entry.path.strip('/'))]
        except modal.exception.NotFoundError:
            return []
        records.sort(key=lambda r: r["created_at"], reverse=True)
        rows = []
        for record in records:
            payload = self.status(record["run_id"])
            if active_only and payload["stage"] in TERMINAL:
                continue
            def mean(field):
                values = [t[field] for t in payload.get("tasks", []) if t.get(field) is not None]
                return sum(values) / len(values) if values else None
            rows.append({**payload, "name": (payload.get("submission") or {}).get("name"),
                         "benchmark": payload.get("benchmark", "?"),
                         "track": (payload.get("submission") or {}).get("track", "?"),
                         "score": payload.get("result"), "topology": "hosted",
                         "progress": {**payload.get("progress", {}),
                                      "failed": sum(t.get("passed") is False for t in payload.get("tasks", []))},
                         "per_task": {"time_sec": mean("task_time_sec"),
                                      "agent_time_sec": mean("agent_time_sec"),
                                      "env_time_sec": mean("env_time_sec")}})
            if limit is not None and len(rows) >= limit:
                break
        return rows

    def export(self, run_id, destination):
        payload = self.status(run_id)
        if payload["stage"] not in TERMINAL:
            raise ValueError("export the evaluation after it stops")
        with Path(destination).open("wb") as file:
            for chunk in self._volume().read_file(f"{run_id}/artifacts.zip"):
                file.write(chunk)
