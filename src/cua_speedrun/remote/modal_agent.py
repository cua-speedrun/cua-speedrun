"""Agent plane on Modal: init once, snapshot, then warm for execution.

The submission's init.py runs once (untimed, with internet) before any task
is revealed; the sandbox filesystem is then snapshotted. Each task gets a
sandbox created from that snapshot. New runs have normal outbound internet:
the submitted files decide whether they use local services, external APIs,
both, or neither.

Modal filesystem snapshots preserve disk, not processes, and GPU sandboxes
cannot take memory snapshots at all. So a
model server started by init.py does not survive into the task sandbox.
The storage-checkpointing contract handles that: init.py is re-run inside
each task sandbox as an untimed warmup, BEFORE the clock arms and BEFORE
the task is revealed.

A second measured Modal behavior dictates the process layout: sandbox execs are isolated from the main process and from each
other in separate PID and network namespaces; only the filesystem is
shared. A server started in one exec is unreachable from any other, so
init.py and agent.py cannot run as separate execs. Instead the sandbox's
MAIN process is a harness-owned supervisor: it runs init.py at boot (the
submitter's server lives on in the main namespace), prints a readiness
sentinel on the sandbox's stdout, then waits for a go-file on the shared
filesystem. The go-file, written by the executor immediately after the
gateway arms, carries the run URL and instruction; the supervisor spawns
agent.py in the same namespace as the server. The two-script contract is
unchanged.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cua_speedrun.evaluation_runtime import ComputeInfrastructureError
from cua_speedrun.remote.network_policy import NETWORK_HOST

_CS_PACKAGE = Path(__file__).resolve().parents[1]


class AgentWarmupError(RuntimeError):
    """The submission's init.py rerun failed deterministically during warmup.

    Mirrors the Slurm runner's classification of a failed worker: a nonzero
    init.py exit reported through the supervisor sentinel is the submission's
    own failure and must fail the run, never enter an infrastructure retry
    loop. Sandbox deaths and boot timeouts are replaceable infrastructure and
    are raised as ComputeInfrastructureError instead.
    """


# The sandbox main process. Everything task-related happens under it, in
# one namespace: init.py (and whatever servers it leaves behind), then,
# once the go-file appears, agent.py.
_SUPERVISOR = r'''
import json, os, subprocess, sys, time
print("__CS_SUP_START", flush=True)
os.chdir("/submission")
rc = subprocess.call([sys.executable, "-u", "init.py"])
print(f"__CS_INIT_EXIT rc={rc}", flush=True)
if rc != 0:
    sys.exit(rc)
# Task dispatch: one go-file per task under /tmp/cs_go, each spawning an
# agent.py in this namespace (the only namespace that can reach the server
# init.py started). Concurrent go-files run concurrently, which is what
# lets one warm sandbox serve a whole run in shared mode; per-task mode
# simply ever writes one file.
GO_DIR = "/tmp/cs_go"
os.makedirs(GO_DIR, exist_ok=True)
seen, procs = set(), {}
while True:
    for name in sorted(os.listdir(GO_DIR)):
        if not name.endswith(".json") or name in seen:
            continue
        seen.add(name)
        tid = name[:-5]
        go_path = os.path.join(GO_DIR, name)
        with open(go_path) as fh:
            go = json.load(fh)
        # The instruction is privileged until arm and should not remain as a
        # readable dispatch artifact for later processes in shared mode.
        os.unlink(go_path)
        out = open(f"/tmp/cs_agent.{tid}.stdout", "w")
        err = open(f"/tmp/cs_agent.{tid}.stderr", "w")
        procs[tid] = (subprocess.Popen(
            [sys.executable, "-u", "agent.py", go["run_url"], go["instruction"]],
            stdout=out, stderr=err), out, err)
    for tid, (proc, out, err) in list(procs.items()):
        code = proc.poll()
        if code is None:
            continue
        out.close(); err.close()
        with open(f"/tmp/cs_agent.{tid}.exit.tmp", "w") as fh:
            json.dump({"rc": code}, fh)
        os.replace(f"/tmp/cs_agent.{tid}.exit.tmp", f"/tmp/cs_agent.{tid}.exit")
        print(f"__CS_AGENT_EXIT {tid} rc={code}", flush=True)
        del procs[tid]
    time.sleep(0.05)
'''
_SENTINEL = re.compile(r"__CS_INIT_EXIT rc=(\d+)")
# Printed by the supervisor as its very first line: marks the moment the
# container is actually running. Everything before it is image build,
# scheduling (GPU queue), and boot, which would otherwise be one silent blob.
_START_SENTINEL = "__CS_SUP_START"


@dataclass
class AgentImage:
    snapshot: Any          # a modal.Image (filesystem snapshot after init)
    gpu: str | None
    runtime_env: dict[str, str] = field(default_factory=dict)


@dataclass
class AgentSandbox:
    """One task's warmed agent sandbox, ready to receive the go-signal."""

    sandbox: Any
    region: str | None
    warmup_sec: float
    warmup_log: str
    _stdout_reader: Any = field(default=None, repr=False)


def build_agent_base_image(
    submission_dir: Path,
    extra_pip: list[str] | None = None,
    agent_runtime: Mapping[str, Any] | None = None,
):
    import modal

    from cua_speedrun.remote.agent_runtime import (
        current_agent_runtime_contract,
        validate_agent_runtime_contract,
    )
    from cua_speedrun.remote.snapshot_cache import AGENT_RUNTIME_FILES

    runtime = dict(agent_runtime or current_agent_runtime_contract())
    validate_agent_runtime_contract(runtime)
    base = runtime["base_image"]
    system_packages = list(runtime["system_packages"])
    pip = list(runtime["python_packages"]) + (extra_pip or [])
    expected_python = base["observed_python_version"]
    image = (
        modal.Image.debian_slim(python_version=base["python_version"])
        .apt_install(*system_packages)
        .pip_install(*pip)
        .run_commands(
            "python -c \"import platform; "
            f"assert platform.python_version() == '{expected_python}', "
            "platform.python_version()\""
        )
        .env({"PYTHONPATH": "/opt/cs"})
    )
    # Hash and ship only agent runtime files so dashboard or executor changes
    # do not invalidate the initialized agent snapshot.
    for name in AGENT_RUNTIME_FILES:
        image = image.add_local_file(
            str(_CS_PACKAGE / name),
            remote_path=f"/opt/cs/cua_speedrun/{name}",
        )
    return image.add_local_dir(str(submission_dir), remote_path="/submission")


def _resources(gpu: str | None) -> tuple[int, int]:
    """CPU cores and memory MiB; GPU agents need more RAM to load weights."""
    return (4, 32768) if gpu else (1, 8192)


def _wait_init_sentinel(
    sb: Any,
    timeout_sec: float,
    on_line: Callable[[str], None],
    on_start: Callable[[], None] | None = None,
    what: str = "init.py",
) -> int:
    """Stream the sandbox's stdout until the supervisor's init sentinel.
    Returns init.py's exit code. Raises on timeout or early sandbox death.
    on_start fires at the supervisor's start sentinel, i.e. the moment the
    container is actually running; the time before it is image build plus
    scheduling plus boot."""
    result: dict[str, int] = {}

    def _reader() -> None:
        for line in sb.stdout:
            if _START_SENTINEL in line:
                if on_start:
                    on_start()
                continue
            m = _SENTINEL.search(line)
            if m:
                result["rc"] = int(m.group(1))
                return
            on_line(line)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    reader.join(timeout_sec)
    if "rc" not in result:
        if sb.poll() is not None:
            raise RuntimeError(
                f"agent sandbox exited during {what} (code {sb.returncode})"
            )
        raise TimeoutError(f"{what} did not finish within {timeout_sec:.0f}s")
    return result["rc"]


def _sh(sb: Any, cmd: str, timeout: int = 60, env: dict | None = None) -> tuple[int, str]:
    proc = sb.exec("bash", "-lc", cmd, timeout=timeout, env=env)
    out = proc.stdout.read()
    proc.wait()
    return (proc.returncode if proc.returncode is not None else -1), out


def _read_remote_file(sb: Any, path: str) -> tuple[str, bool]:
    """Read a sandbox file with its length verified against wc -c.

    A plain exec'd cat intermittently returns only the first 8KB (observed
    once in 48 pulls: an agent.stderr cut at exactly 8192 bytes mid-episode,
    unrecoverable because the file was deleted right after). Returns
    (content, verified); the caller must not delete the remote file unless
    verified is True.
    """
    content = ""
    for _attempt in range(3):
        rc, size_text = _sh(sb, f"wc -c < {path} 2>/dev/null")
        if rc != 0 or not size_text.strip().isdigit():
            return content, False
        expected = int(size_text.strip())
        rc, content = _sh(sb, f"cat {path} 2>/dev/null")
        if rc == 0 and len(content.encode("utf-8", "replace")) == expected:
            return content, True
    return content, False


def init_and_snapshot(submission_dir: Path, *, gpu: str | None = None,
                      extra_pip: list[str] | None = None,
                      agent_runtime: Mapping[str, Any] | None = None,
                      runtime_env: dict[str, str] | None = None,
                      init_timeout_sec: int = 2700,
                      log_path: Path | None = None,
                      on_line: Callable[[str], None] | None = None,
                      on_event: Callable[..., None] | None = None) -> AgentImage:
    """Run init.py once (as the supervisor's boot phase) and snapshot the
    filesystem. The init output is streamed live, line by line, to on_line
    (default: printed to stdout) and to log_path when given, so a long init
    is observable and a failing one leaves its reason behind instead of a
    bare exit code. on_event(kind, **payload) brackets every phase that used
    to be silent: image build, sandbox creation, queue+boot, init.py itself,
    and the snapshot."""
    import modal

    ev = on_event or (lambda kind, **payload: None)
    app = modal.App.lookup("cua-speedrun-agent", create_if_missing=True)
    image = build_agent_base_image(
        submission_dir, extra_pip, agent_runtime=agent_runtime
    )
    cpu, memory = _resources(gpu)
    # Sandbox.create blocks on the image build (the container itself comes
    # up asynchronously and no GPU is held during the build), so create_sec
    # is effectively the image build and queue_boot_sec is the GPU queue
    # plus container boot. Modal offers no way to force a build on its own
    # (Image.hydrate refuses on demand), so these two numbers are the split.
    t0 = time.monotonic()
    sb = modal.Sandbox.create(
        "python3", "-u", "-c", _SUPERVISOR,
        app=app, image=image, gpu=gpu, cpu=cpu, memory=memory,
        timeout=init_timeout_sec,
        env=runtime_env or {},
    )
    ev("init_sandbox_created", sandbox_id=getattr(sb, "object_id", None),
       create_sec=round(time.monotonic() - t0, 3), gpu=gpu)
    t_created = time.monotonic()
    started: dict[str, float] = {}

    def _on_start() -> None:
        # First output from the container: queue + boot are over.
        started["t"] = time.monotonic()
        ev("init_sandbox_started",
           queue_boot_sec=round(started["t"] - t_created, 3))

    emit = on_line or (lambda line: print(f"[init] {line.rstrip()}", flush=True))
    try:
        log_fh = open(log_path, "w") if log_path else None
        tail: list[str] = []

        def handle_line(line: str) -> None:
            tail.append(line)
            del tail[:-50]
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
            emit(line)

        try:
            rc = _wait_init_sentinel(sb, init_timeout_sec, handle_line,
                                     on_start=_on_start, what="init.py")
        finally:
            if log_fh:
                log_fh.close()
        ev("init_py_done", rc=rc,
           init_py_sec=round(time.monotonic() - started.get("t", t_created), 3))
        if rc != 0:
            raise RuntimeError(
                f"agent init.py exited with code {rc}; output tail:\n" + "".join(tail)
            )
        # GPU submissions can leave tens of GB (packages + weights) on disk,
        # so give the snapshot a generous budget.
        ev("snapshot_started")
        t0 = time.monotonic()
        snapshot = sb.snapshot_filesystem(timeout=1800)
        ev("snapshot_done", snapshot_sec=round(time.monotonic() - t0, 3),
           image_id=getattr(snapshot, "object_id", None))
    finally:
        sb.terminate()
    return AgentImage(snapshot=snapshot, gpu=gpu,
                      runtime_env=dict(runtime_env or {}))


def agent_image_from_cache(
    image_id: str,
    gpu: str | None,
    runtime_env: dict[str, str] | None = None,
) -> AgentImage | None:
    """Rebuild an AgentImage from a cached snapshot id, validated by creating
    and immediately terminating a probe sandbox from it. Returns None when
    the snapshot is unusable, so the caller falls back to a fresh init
    instead of failing per task. Unusable means NotFoundError (expired TTL,
    deleted) or PermissionDeniedError (the id belongs to another workspace,
    the case a Modal token rotation leaves behind).

    The probe is the one supported validation: Image.from_id is lazy and
    Image.hydrate() refuses on-demand hydration (both measured on modal
    1.5.1), so creating a sandbox is what actually resolves the image. It
    also hydrates the handle, which the real task sandboxes then reuse."""
    import modal

    app = modal.App.lookup("cua-speedrun-agent", create_if_missing=True)
    image = modal.Image.from_id(image_id)
    try:
        probe = modal.Sandbox.create("true", app=app, image=image, timeout=60)
    except (modal.exception.NotFoundError, modal.exception.PermissionDeniedError):
        return None
    try:
        probe.terminate()
    except Exception:
        pass
    return AgentImage(snapshot=image, gpu=gpu,
                      runtime_env=dict(runtime_env or {}))


def spawn_agent_sandbox(
    agent_image: AgentImage,
    gateway_host: str | None,
    *,
    network_policy: str = NETWORK_HOST,
    api_domains: Sequence[str] = (),
    region: str | None = None,
    task_timeout_sec: float = 600.0,
    warmup_timeout_sec: int = 1200,
    on_line: Callable[[str], None] | None = None,
    on_event: Callable[..., None] | None = None,
    on_started: Callable[[str | None], None] | None = None,
    on_sandbox: Callable[[Any], None] | None = None,
) -> AgentSandbox:
    """Create a task's sandbox from the init snapshot and warm it, untimed.

    The warmup is the supervisor's re-run of init.py: no task information
    exists here. Raises if the warmup exits nonzero or times out. The caller
    arms the clock only after this returns.

    ``gateway_host`` and ``api_domains`` are used only when replaying an older
    frozen restricted-network plan. New plans use host-network and omit an
    outbound allowlist entirely.

    Warmup output streams to on_line as it happens (it used to be buffered
    until the warmup finished, so a stuck warmup was a black box and a
    timeout lost every line). on_event brackets sandbox creation and
    queue+boot so a GPU capacity wait is distinguishable from a slow server
    start. on_started fires as soon as the container is running, with the
    region Modal actually placed it in, so a caller can start bringing the
    environment to the GPU while the server is still warming.
    """
    import modal

    ev = on_event or (lambda kind, **payload: None)
    app = modal.App.lookup("cua-speedrun-agent", create_if_missing=True)
    cpu, memory = _resources(agent_image.gpu)
    sandbox_options = {
        "app": app,
        "image": agent_image.snapshot,
        "gpu": agent_image.gpu,
        "cpu": cpu,
        "memory": memory,
        # Modal rejects sandbox timeouts above 86400s; a shared replica's
        # summed task budget can exceed that (48 tasks x 3600s), so clamp.
        "timeout": min(int(warmup_timeout_sec + task_timeout_sec + 300), 86400),
        "block_network": False,
        "region": region,
        "env": agent_image.runtime_env,
    }
    if network_policy != NETWORK_HOST:
        sandbox_options["outbound_domain_allowlist"] = list(dict.fromkeys(
            ([gateway_host] if gateway_host else ["placeholder.invalid"])
            + list(api_domains)
        ))
    t0 = time.monotonic()
    try:
        sb = modal.Sandbox.create(
            "python3", "-u", "-c", _SUPERVISOR,
            **sandbox_options,
        )
    except Exception as exc:
        # A failed creation is replaceable infrastructure, the same
        # classification the Slurm runner gives a failed submit command. The
        # original message is preserved so the caller's region-fallback
        # substring check ("worker type supports") still matches.
        raise ComputeInfrastructureError(
            f"failed to create the agent sandbox: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if on_sandbox is not None:
        # Hand the raw sandbox to the caller immediately so it can be
        # terminated even if this warmup never returns (the env dying while
        # the agent warms leaked one sandbox per retry otherwise).
        on_sandbox(sb)
    ev("warmup_sandbox_created", sandbox_id=getattr(sb, "object_id", None),
       create_sec=round(time.monotonic() - t0, 3), region=region)
    t_created = time.monotonic()
    lines: list[str] = []
    region_seen: dict[str, str | None] = {}

    def _collect(line: str) -> None:
        lines.append(line)
        if on_line:
            on_line(line)

    def _on_start() -> None:
        ev("warmup_sandbox_started",
           queue_boot_sec=round(time.monotonic() - t_created, 3))
        if on_started is not None:
            _, out = _sh(sb, "printenv MODAL_REGION", timeout=30)
            region_seen["v"] = out.strip() or None
            on_started(region_seen["v"])

    try:
        try:
            rc = _wait_init_sentinel(sb, warmup_timeout_sec, _collect,
                                     on_start=_on_start,
                                     what="agent warmup (init.py rerun)")
        except (RuntimeError, TimeoutError) as exc:
            # The sandbox died mid-warmup or never finished booting: this is
            # replaceable infrastructure (Modal preemption, OOM kill, capacity
            # stall), the same classification the Slurm runner gives its
            # signal-killed or never-starting workers. A signal-killed init.py
            # cannot print the sentinel, so every genuine submission failure
            # arrives through the nonzero-rc branch below instead.
            raise ComputeInfrastructureError(
                f"agent warmup infrastructure failed: {exc}",
                resource_id=getattr(sb, "object_id", None),
            ) from exc
        if rc != 0:
            raise AgentWarmupError(
                f"agent warmup (init.py rerun) exited with code {rc}; "
                f"log tail: {''.join(lines)[-800:]}"
            )
        # The region Modal actually placed us in, not just what was asked
        # (reuse the value read at start when on_started already fetched it).
        if "v" in region_seen:
            actual_region = region_seen["v"] or ""
        else:
            _, out = _sh(sb, "printenv MODAL_REGION", timeout=30)
            actual_region = out.strip()
    except BaseException:
        try:
            sb.terminate()
        except Exception:
            pass
        raise
    return AgentSandbox(
        sandbox=sb,
        region=actual_region or region,
        warmup_sec=time.monotonic() - t_created,
        warmup_log="".join(lines),
    )


def grant_gateway_access(
    agent_sandbox: AgentSandbox,
    gateway_hosts: str | list[str],
    *,
    network_policy: str = NETWORK_HOST,
    api_domains: Sequence[str] = (),
) -> None:
    """Finish networking for a frozen restricted-network plan.

    Host-network runs already have access and need no mutation. The update
    path remains solely so older plans can retain their recorded behavior.
    """
    if network_policy == NETWORK_HOST:
        return
    gateways = [gateway_hosts] if isinstance(gateway_hosts, str) else list(gateway_hosts)
    hosts = list(dict.fromkeys(gateways + list(api_domains)))
    agent_sandbox.sandbox._experimental_set_outbound_network_policy(
        outbound_domain_allowlist=hosts)


def pull_agent_logs(agent_sandbox: AgentSandbox, task_dir: Path) -> None:
    """Pull the model server's own log (/root/vllm.log, if the submission wrote
    one) out of the agent sandbox. Best effort; a scripted submission has none."""
    try:
        proc = agent_sandbox.sandbox.exec(
            "bash", "-lc", "cat /root/vllm.log 2>/dev/null | tail -c 200000",
            timeout=60)
        out = proc.stdout.read()
        proc.wait()
    except Exception:
        return
    if out and out.strip():
        (task_dir / "vllm.log").write_text(out)


def exec_agent(
    agent_sandbox: AgentSandbox,
    run_url: str,
    instruction: str,
    *,
    task_id: str = "t0",
    timeout_sec: int = 600,
    on_event: Callable[..., None] | None = None,
) -> tuple[int, str, str]:
    """The go-signal: reveal one task and let the supervisor run agent.py.

    Called immediately after that task's gateway is armed. Writes the task's
    go-file atomically on the sandbox's shared filesystem; the supervisor
    (polling at 50ms) spawns agent.py in the same namespace as the
    submitter's server. Concurrent calls with distinct task_ids run
    concurrently against the one sandbox (shared mode). Blocks until the
    agent process exits; returns (exit_code, stdout, stderr). The go-file
    write is the only harness action on the timed clock, so its duration is
    reported via on_event.
    """
    ev = on_event or (lambda kind, **payload: None)
    sb = agent_sandbox.sandbox
    payload = json.dumps({"run_url": run_url, "instruction": instruction})
    go = f"/tmp/cs_go/{task_id}.json"
    t0 = time.monotonic()
    rc, _ = _sh(
        sb,
        f'mkdir -p /tmp/cs_go && printf %s "$CS_GO" > {go}.tmp && mv {go}.tmp {go}',
        env={"CS_GO": payload},
    )
    if rc != 0:
        raise RuntimeError(f"failed to deliver go-signal (exit {rc})")
    ev("go_delivered", go_sec=round(time.monotonic() - t0, 3))

    exit_file = f"/tmp/cs_agent.{task_id}.exit"
    deadline = time.monotonic() + timeout_sec
    exit_info: dict | None = None
    while time.monotonic() < deadline:
        rc, out = _sh(sb, f"cat {exit_file} 2>/dev/null")
        if rc == 0 and out.strip():
            exit_info = json.loads(out)
            break
        if sb.poll() is not None:
            raise RuntimeError(
                f"agent sandbox exited mid-task (code {sb.returncode})"
            )
        time.sleep(2)

    agent_out, out_ok = _read_remote_file(sb, f"/tmp/cs_agent.{task_id}.stdout")
    agent_err, err_ok = _read_remote_file(sb, f"/tmp/cs_agent.{task_id}.stderr")
    if exit_info is None:
        return -1, agent_out, agent_err + "\n[harness] agent did not exit within budget"
    if out_ok and err_ok:
        # Only reap the remote copies once both reads are length-verified;
        # a truncated read with the source deleted is unrecoverable.
        _sh(
            sb,
            f"rm -f /tmp/cs_agent.{task_id}.stdout "
            f"/tmp/cs_agent.{task_id}.stderr {exit_file}",
        )
    else:
        agent_err += "\n[harness] agent log read could not be length-verified"
    return int(exit_info["rc"]), agent_out, agent_err
