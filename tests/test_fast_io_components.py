"""Real image codecs, files, queues and record reduction; no VM or fake adapter."""

import io
import json
import struct
import threading

import pytest
from PIL import Image
from gym_anything.utils.jsonl import JSONLWriter

from cua_speedrun.envs.gym_anything import observation_from_gym
from cua_speedrun.fast_io import decode_observation, encode_observation
from cua_speedrun.recording import AsyncJSONL, OrderedRecorder
from cua_speedrun.runlog import RunLogWriter, load_runlog, summarize


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_fast_png_pixels_metadata_and_binary_roundtrip(mode):
    source = Image.new(mode, (1920, 1080), "#bc912f")
    for x in range(0, 1920, 13):
        source.putpixel((x, x % 1080), (x % 255, 93, 29) + ((71,) if mode == "RGBA" else ()))
    observation = observation_from_gym(
        {"screen": {"image": source, "resolution": [1920, 1080]},
         "ui_tree": "text " * 20000}, fast_io=True, image_format="png",
    )
    metadata = {"ok": True, "info": {"done": False}, "meta": observation.meta}
    result = decode_observation(encode_observation(observation.image, metadata))
    assert result == {**metadata, "image": observation.image, "mime_type": "image/png"}
    with Image.open(io.BytesIO(result["image"])) as image:
        assert image.size == source.size
        assert image.mode == source.mode
        assert image.tobytes() == source.tobytes()


@pytest.mark.parametrize("data", [b"", b"abc", struct.pack("!I", 100),
    struct.pack("!I", 2) + b"[]", struct.pack("!I", 2) + b"{}not png",
    encode_observation(b"", {"ok": True})])
def test_binary_rejects_invalid_frames(data):
    with pytest.raises(ValueError):
        decode_observation(data)


def test_failed_action_can_return_metadata_without_a_frame():
    result = decode_observation(encode_observation(b"", {"ok": False, "error": "outcome unknown"}))
    assert result == {"ok": False, "error": "outcome unknown", "image": b"", "mime_type": None}


def test_archives_are_ordered_and_closed_completely(tmp_path):
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.enable_async()
    for index in range(80):
        path = tmp_path / f"frame_{index:05d}.png"
        data = f"frame {index}".encode()
        writer.artifact(path, data)
        writer.event("observe", frame=path.name, index=index)
    writer.close()
    events = load_runlog(writer.path)
    assert [record["index"] for record in events] == list(range(80))
    for record in events:
        assert (tmp_path / record["frame"]).read_bytes() == f"frame {record['index']}".encode()


def test_mutable_records_are_snapshotted_on_caller_and_written_off_thread(tmp_path):
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.enable_async()
    actions = [{"keyboard": {"text": "original"}}]
    writer.event("step", actions=actions)
    actions[0]["keyboard"]["text"] = "changed"
    writer.flush()
    assert load_runlog(writer.path)[0]["actions"][0]["keyboard"]["text"] == "original"
    writer.close()
    recorder = OrderedRecorder(capacity=2)
    identities = []
    recorder.submit(lambda: identities.append(threading.get_ident()))
    recorder.close()
    assert identities and identities[0] != threading.get_ident()


def test_real_archival_failure_is_not_silent(tmp_path):
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.enable_async()
    writer.artifact(tmp_path, b"cannot write a directory")
    with pytest.raises(RuntimeError, match="background recording failed"):
        writer.flush()
    with pytest.raises(RuntimeError, match="background recording failed"):
        writer.close()


def test_dependency_jsonl_writer_is_drained_before_read_and_close(tmp_path):
    writer = AsyncJSONL(JSONLWriter(tmp_path / "traj.jsonl"))
    for index in range(100):
        writer.write({"step": index})
    writer.flush()
    assert len(load_runlog(writer.path)) == 100
    writer.close()
    writer.flush()  # Gym can close it itself on a terminal action.
    writer.close()
    assert load_runlog(writer.path)[-1] == {"step": 99}


def test_fast_timing_uses_clipped_interval_union_and_counts_combined_observations():
    events = [
        {"event": "armed", "t_mono": 10.0},
        {"event": "finished", "t_mono": 20.0},
        {"event": "io_contract", "version": "fast-io-v2", "timing": "gateway-response-interval-union"},
        {"event": "step", "dur": 2.0},
        {"event": "observe", "dur": 0.0, "combined": True},
        *({"event": "gateway_response", "t_mono_arrive": a, "t_mono_sent": b}
          for a, b in [(9, 11), (12, 14), (13, 16), (19, 21)]),
    ]
    result = summarize(events)
    assert result["env_time_sec"] == 6.0
    assert result["agent_time_sec"] == 4.0
    assert result["num_steps"] == result["num_observes"] == 1
    legacy = summarize([event for event in events if event["event"] != "io_contract"])
    assert legacy["env_time_sec"] == 2.0
    assert legacy["agent_time_sec"] == 8.0


def test_closed_queue_rejects_new_work():
    recorder = OrderedRecorder()
    recorder.close()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.submit(json.dumps, {})


def test_queue_saturation_fails_instead_of_dropping_records():
    recorder = OrderedRecorder(capacity=1, timeout=0.1)
    entered, release = threading.Event(), threading.Event()
    completed = []

    def blocked_write():
        entered.set()
        release.wait(5)
        completed.append(1)

    try:
        recorder.submit(blocked_write)
        assert entered.wait(5)
        recorder.submit(completed.append, 2)
        with pytest.raises(TimeoutError, match="queue did not drain"):
            recorder.submit(completed.append, 3)
    finally:
        release.set()
        recorder.close()
    assert completed == [1, 2]


def test_fast_prompt_and_new_measurement_files_are_in_the_contract():
    from cua_speedrun.client import Computer
    from cua_speedrun.runplan import _MEASUREMENT_PATHS
    from agents.codex_cli.agent import ActionGateway, prompt

    assert {"fast_io.py", "fast_io_client.c", "images.py", "recording.py"} <= set(_MEASUREMENT_PATHS)
    gateway = ActionGateway(Computer("http://127.0.0.1", fast_io=True), 100)
    try:
        text = prompt("Example task", gateway)
        assert "step-observe" in text
        assert "-m cua_speedrun.fast_io" not in text
        assert "fast_io_client" in text
        assert "native client" in text
        assert "do not retry or follow redirects" in text
        assert "step does not append a screenshot" not in text
        assert "ONE request" in text
        assert "JPEG quality 95" in text
        assert "returned path" in text
    finally:
        gateway.stop()


def test_remote_runtime_imports_without_checkout_or_installed_package(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import cua_speedrun
    from cua_speedrun.remote.snapshot_cache import AGENT_RUNTIME_FILES

    source = Path(cua_speedrun.__file__).parent
    assert "fast_io_client.c" in AGENT_RUNTIME_FILES
    package = tmp_path / "cua_speedrun"
    package.mkdir()
    for name in AGENT_RUNTIME_FILES:
        shutil.copyfile(source / name, package / name)
    # -S removes site-packages. Requests is the only external dependency;
    # include its installed tree explicitly without exposing the checkout.
    import requests
    import urllib3
    paths = [str(tmp_path), str(Path(requests.__file__).parent.parent),
             str(Path(urllib3.__file__).parent.parent)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "from cua_speedrun.client import Computer; "
         "from cua_speedrun.recording import OrderedRecorder; "
         "import sys; assert 'PIL' not in sys.modules; "
         "import cua_speedrun; print(cua_speedrun.__file__)"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(package) in result.stdout


def test_fast_child_can_start_native_curl(tmp_path):
    import subprocess
    from agents.codex_cli.agent import child_environment
    env = child_environment(tmp_path, "http://127.0.0.1", "unused", fast_io=True)
    result = subprocess.run(
        ["curl", "-q", "--fail-with-body", "--version"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "curl" in result.stdout


def test_capture_defaults_to_dbus_and_rejects_typos():
    from cua_speedrun.envs.fast_capture import capture_backend, profiling_enabled
    assert capture_backend({}) == "dbus"
    assert capture_backend({"GYM_ANYTHING_QEMU_FAST_IO_BACKEND": " DBUS "}) == "dbus"
    assert capture_backend({"GYM_ANYTHING_QEMU_FAST_IO_BACKEND": "qmp"}) == "qmp"
    with pytest.raises(ValueError):
        capture_backend({"GYM_ANYTHING_QEMU_FAST_IO_BACKEND": "dbu"})
    assert not profiling_enabled({})
    assert profiling_enabled({"CS_FAST_IO_PROFILE": "true"})
    with pytest.raises(ValueError):
        profiling_enabled({"CS_FAST_IO_PROFILE": "yes"})


def test_capture_timer_wraps_real_image_copy_and_keeps_threads_separate():
    from concurrent.futures import ThreadPoolExecutor
    from cua_speedrun.envs.fast_capture import CaptureTimer
    source = Image.new("RGB", (80, 60), "green")
    timer = CaptureTimer(source.copy)

    def copy_frames(count):
        metrics = timer.begin()
        try:
            for _ in range(count):
                assert timer().tobytes() == source.tobytes()
        finally:
            timer.end()
        assert metrics["capture_calls"] == count
        assert metrics["capture_sec"] > 0
        timer()  # An untimed call must not change the completed record.
        assert metrics["capture_calls"] == count

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(copy_frames, [1, 2, 3, 4, 5]))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_jpeg_direct_encoding_quality_sampling_and_metadata(mode, tmp_path):
    from PIL import JpegImagePlugin
    from cua_speedrun.fast_io import save_screenshot

    source = Image.new(mode, (1920, 1080), "#b185ca")
    for x in range(20, 1900, 11):
        source.putpixel((x, x % 1080), (x % 255, 70, 22) + ((99,) if mode == "RGBA" else ()))
    obs = observation_from_gym(
        {"screen": {"image": source}, "ui_tree": "native coordinates"},
        fast_io=True, image_format="jpeg",
    )
    expected = io.BytesIO()
    source.convert("RGB").save(expected, format="JPEG", quality=95, subsampling=0,
                               optimize=False, progressive=False)
    assert obs.image == expected.getvalue()
    assert obs.meta["resolution"] == [1920, 1080]
    assert obs.meta["ui_tree"] == "native coordinates"
    assert obs.meta["encoding"] == {
        "format": "jpeg", "quality": 95, "subsampling": "4:4:4", "lossless": False,
    }
    payload = decode_observation(encode_observation(obs.image, {"ok": True, "meta": obs.meta}))
    assert payload["mime_type"] == "image/jpeg"
    assert "png" not in payload
    saved = save_screenshot(tmp_path / "wrong.png", payload)
    assert saved.name == "wrong.jpg"
    assert saved.read_bytes() == obs.image
    assert not (tmp_path / "wrong.png").exists()
    with Image.open(saved) as decoded:
        decoded.load()
        assert decoded.format == "JPEG"
        assert decoded.mode == "RGB" and decoded.size == source.size
        assert JpegImagePlugin.get_sampling(decoded) == 0
        assert decoded.tobytes() != source.convert("RGB").tobytes()
        # The legacy interface must contain a real PNG, never relabeled JPEG.
        with Image.open(io.BytesIO(obs.png)) as legacy:
            assert legacy.format == "PNG"
            assert legacy.tobytes() == decoded.tobytes()


def test_jpeg_default_png_opt_in_and_invalid_selection():
    from cua_speedrun.images import fast_image_format
    from cua_speedrun.runtime_environment import normalize_environment_name
    option = normalize_environment_name("GYM_ANYTHING_FAST_IO_IMAGE_FORMAT")
    assert fast_image_format({}) == "jpeg"
    assert fast_image_format({option: " PNG "}) == "png"
    for value in ("jpg", "webp", "", "raw"):
        with pytest.raises(ValueError):
            fast_image_format({option: value})


def test_wire_rejects_mislabeled_jpeg():
    from cua_speedrun.images import encode_image
    data, _ = encode_image(Image.new("RGB", (8, 8), "red"), "jpeg")
    with pytest.raises(ValueError, match="MIME"):
        encode_observation(data, {"ok": True, "mime_type": "image/png"})
    header = json.dumps({"ok": True, "mime_type": "image/png"}).encode()
    with pytest.raises(ValueError, match="MIME"):
        decode_observation(struct.pack("!I", len(header)) + header + data)


def test_jpeg_archive_matches_received_bytes_after_drain(tmp_path):
    from cua_speedrun.images import encode_image
    data, _ = encode_image(Image.new("RGB", (1920, 1080), "purple"), "jpeg")
    wire = encode_observation(data, {"ok": True})
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.enable_async()
    writer.artifact(tmp_path / "frame_00000.jpg", data)
    writer.event("observe", frame="frame_00000.jpg")
    writer.close()
    event = load_runlog(writer.path)[0]
    assert (tmp_path / event["frame"]).read_bytes() == decode_observation(wire)["image"]


def test_observation_keeps_legacy_constructor_and_png_access():
    from cua_speedrun.envs.base import Observation
    from cua_speedrun.images import encode_image
    data, _ = encode_image(Image.new("RGB", (8, 8), "red"), "png")
    obs = Observation(png=data, meta={"format": "png"})
    assert obs.image == obs.png == data
    with pytest.raises(ValueError):
        Observation(image=data, png=data)


@pytest.mark.parametrize("image_format", ["jpeg", "png"])
def test_actual_artifact_tar_includes_either_image_format(tmp_path, image_format):
    import subprocess
    import tarfile
    from cua_speedrun.images import encode_image, image_suffix, image_mime_type
    from cua_speedrun.remote.modal_env import _artifact_archive_command
    data, _ = encode_image(Image.new("RGB", (64, 48), "green"), image_format)
    name = "frame_00000" + image_suffix(image_mime_type(data))
    (tmp_path / name).write_bytes(data)
    (tmp_path / "env_plane.log").write_text("image archive component check\n")
    (tmp_path / "episode").mkdir()
    archive = tmp_path / "_artifacts.tgz"
    result = subprocess.run(["/bin/sh", "-c", _artifact_archive_command(str(archive), str(tmp_path))])
    assert result.returncode == 0
    with tarfile.open(archive) as contents:
        assert contents.extractfile(name).read() == data


@pytest.mark.parametrize("image_format", ["jpeg", "png"])
def test_persistent_proxy_saves_exact_images_before_returning_paths(tmp_path, image_format):
    import time
    from pathlib import Path
    from cua_speedrun.client import Computer
    from cua_speedrun.images import encode_image, image_suffix, image_mime_type
    from agents.codex_cli.agent import ActionGateway
    data, _ = encode_image(Image.new("RGB", (320, 240), "purple"), image_format)
    wire = encode_observation(data, {"ok": True, "meta": {"resolution": [320, 240]}})
    gateway = ActionGateway(Computer("http://127.0.0.1", fast_io=True), 100)
    directory = gateway._screenshots.path
    try:
        first = gateway.screenshot_reply(wire, time.perf_counter())
        second = gateway.screenshot_reply(wire, time.perf_counter())
        assert first["screenshot"] != second["screenshot"]
        for result in (first, second):
            path = Path(result["screenshot"])
            assert path.is_absolute() and path.parent == directory
            assert path.suffix == image_suffix(image_mime_type(data))
            assert path.read_bytes() == data
            assert "image" not in result and "png" not in result
            assert result["proxy_file_ready_ms"] >= 0
            assert result["meta"]["resolution"] == [320, 240]
            assert result["steps_remaining"] == 100
        failed = gateway.screenshot_reply(
            encode_observation(b"", {"ok": False, "error": "outcome unknown"}),
            time.perf_counter(),
        )
        assert failed["screenshot"] is None
        assert len(list(directory.iterdir())) == 2
        with pytest.raises(ValueError):
            gateway.screenshot_reply(b"invalid wire", time.perf_counter())
        assert len(list(directory.iterdir())) == 2
    finally:
        gateway.stop()
    assert not directory.exists()


def test_proxy_file_storage_supports_concurrent_real_writes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from cua_speedrun.fast_io import ScreenshotFiles
    from cua_speedrun.images import encode_image
    data, _ = encode_image(Image.new("RGB", (32, 32), "green"), "jpeg")
    wire = encode_observation(data, {"ok": True})
    files = ScreenshotFiles(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=8) as threads:
            replies = list(threads.map(files.receive, [wire] * 24))
        paths = {Path(reply["screenshot"]) for reply in replies}
        assert len(paths) == 24
        assert all(path.read_bytes() == data for path in paths)
    finally:
        files.close()


def test_native_curl_cannot_inherit_retry_or_redirect_settings(tmp_path):
    from cua_speedrun.fast_io import CURL_OPTIONS
    from agents.codex_cli.init import ensure_curl
    assert CURL_OPTIONS[0] == "-q"
    assert CURL_OPTIONS[CURL_OPTIONS.index("--retry") + 1] == "0"
    assert CURL_OPTIONS[CURL_OPTIONS.index("--noproxy") + 1] == "*"
    assert "--location" not in CURL_OPTIONS and "-L" not in CURL_OPTIONS
    ensure_curl()  # Actual installed binary; no server, model, or VM.


@pytest.mark.parametrize("client", ["curl", "shell"])
def test_native_client_preserves_real_proxy_errors_without_environment_calls(tmp_path, client):
    import shlex
    import subprocess
    from cua_speedrun.client import Computer
    from cua_speedrun.fast_io import CURL_OPTIONS, FILE_ACCEPT, build_shell_client
    from agents.codex_cli.agent import ActionGateway, child_environment
    # Real production HTTP handler and curl, but every request is rejected
    # locally. No task environment, fake adapter, or outbound request.
    gateway = ActionGateway(Computer("http://127.0.0.1:1", fast_io=True), 0)
    url = gateway.start()
    native = build_shell_client(tmp_path) if client == "shell" else None
    try:
        cases = (
            ("invalid", '{"actions":[]}', "invalid gateway token"),
            (gateway.token, '{}', "body must contain an actions list"),
            (gateway.token, '{"actions":[]}', "step budget exhausted"),
        )
        for token, body, error in cases:
            env = child_environment(tmp_path, url, token, fast_io=True)
            command = (
                "curl " + shlex.join(CURL_OPTIONS)
                + ' -H "X-Gateway-Token: $CS_COMPUTER_TOKEN"'
                + " -H " + shlex.quote("Accept: " + FILE_ACCEPT)
                + " -H 'Content-Type: application/json' --data " + shlex.quote(body)
                + ' "$CS_COMPUTER_URL/step-observe"'
            )
            if client == "shell":
                command = shlex.join([str(native), "step-observe", "[]" if body != '{}' else "{}"])
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-ec", command], env=env, cwd=tmp_path,
                capture_output=True, text=True, timeout=10,
            )
            assert result.returncode == 22
            assert json.loads(result.stdout)["error"] == error
            assert gateway.steps == 0 and gateway.fatal_error is None
        assert not list(gateway._screenshots.path.iterdir())
    finally:
        gateway.stop()
