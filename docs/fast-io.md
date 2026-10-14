# FastIO transport and sandbox placement

FastIO reduces the work between an agent's action and its returned screenshot.
It combines action and observation into one request, reuses HTTP connections,
prepares images from the live framebuffer, and records artifacts asynchronously.
Modal evaluations pin their agent and environment sandboxes to one region and
verify their actual placement before starting the task clock.

## Enable FastIO

FastIO is an explicit per-evaluation option. For the bundled Codex CLI agent
with a QEMU desktop on Modal:

```bash
export GYM_ANYTHING_FAST_IO=true
cua-speedrun benchmark --dataset osworld-50 --agent codex_cli \
  --environment modal --env GYM_ANYTHING_FAST_IO
```

The normal credential setup and benchmark preparation still apply. The agent
initialization compiles the small C client before task timing starts; the
initialized sandbox snapshot retains the executable.

The environment options are:

| Variable | Default | Effect |
| --- | --- | --- |
| `GYM_ANYTHING_FAST_IO` | `false` | Enable the fast capture and transport path. |
| `GYM_ANYTHING_QEMU_FAST_IO_BACKEND` | `dbus` | Select D-Bus framebuffer capture or `qmp`. |
| `GYM_ANYTHING_FAST_IO_IMAGE_FORMAT` | `jpeg` | Select JPEG quality 95 with 4:4:4 sampling, or lossless `png`. |
| `GYM_ANYTHING_FAST_IO_PREPARE_IMAGES` | `true` | Prepare D-Bus framebuffer encodings in a background thread. |

Additional options travel through the same `--env NAME` mechanism. FastIO
preserves native image dimensions and action coordinates. It adds no fixed
post-action settling delay; explicit agent wait actions remain unchanged.
Without the opt-in, the existing PNG observation and task settling behavior
remain available.

## Action and screenshot delivery

The image-aware Python client exposes `observe_image()` and
`step_observe(actions)`. Both return encoded image bytes under `image`, their
`mime_type`, and observation metadata. The combined method reuses the
screenshot already captured by the Gym-Anything step.

The binary response contains a four-byte metadata length, UTF-8 JSON metadata,
and the original PNG or JPEG bytes. It avoids base64 image conversion.
`observe_wire()` and `step_observe_wire(actions)` expose this response directly
for pass-through proxies. Existing `observe()` clients retain the PNG interface.

The Codex CLI agent runs an authenticated loopback proxy. Its native C client
sends commands to that proxy, which keeps the remote HTTP connection open and
saves each screenshot under a unique local filename before returning the path.
There is no per-action Python startup or image re-encoding. An ambiguous failed
action response is not replayed automatically.

## Capture and recording

D-Bus display updates identify framebuffer versions. Background image
preparation encodes those versions, while each observation selects the current
framebuffer version under the display lock. It does not substitute an older
frame just because that frame has already been encoded.

JPEG encoding uses a file-descriptor-backed Pillow encoder so background work
can release the Python GIL. Image preparation and archival recording are
separate: the returned image bytes are also the bytes queued for the archive.

The bounded recorder preserves write order and reports write failures. The
gateway drains accepted responses, trajectory records, and frame writes before
reporting completion and collecting artifacts. PNG and JPEG frames are both
included in exported trajectories.

## Sandbox placement and timing

The Modal executor resolves one actual region for the evaluation and uses it
for subsequent replicas, environments, and replacement allocations. Unknown
or mismatched actual regions fail before task timing starts. Initial compute
allocation retains the existing bounded infrastructure retries.

An explicit region can be supplied through `cua-speedrun run --region REGION`
or the `region` keyword of `LocalEvaluations.submit`. The request is frozen in
the run contract; it does not change the shared track.

Task time remains measured by the environment gateway's clock. FastIO's
environment-time accounting uses the union of gateway response intervals,
clipped to the task window, so overlapping requests are not double-counted.
The remaining agent time includes network and client work, not only inference.
Evaluator-controlled `CS_FAST_IO_PROFILE=true` additionally records capture,
image preparation, response-header, response-body, and connection-reuse timing.

## Source and dependency

| Component | Source |
| --- | --- |
| Region pinning and verification | [`remote/run.py`](../src/cua_speedrun/remote/run.py) |
| Combined gateway requests | [`gateway.py`](../src/cua_speedrun/gateway.py) |
| Binary transport and native client | [`fast_io.py`](../src/cua_speedrun/fast_io.py), [`fast_io_client.c`](../src/cua_speedrun/fast_io_client.c) |
| Framebuffer preparation | [`prepared_capture.py`](../src/cua_speedrun/envs/prepared_capture.py), [`images.py`](../src/cua_speedrun/images.py) |
| Ordered background recording | [`recording.py`](../src/cua_speedrun/recording.py), [`runlog.py`](../src/cua_speedrun/runlog.py) |
| Reference agent integration | [`agents/codex_cli`](../agents/codex_cli) |

The Gym-Anything submodule pins
[`ed89d9a`](https://github.com/cmu-l3/gym-anything/commit/ed89d9a10d72184176a21fca03ba4051e5f482fc),
the compatibility commit used by the FastIO evaluations. It applies upstream
[PR 53](https://github.com/cmu-l3/gym-anything/pull/53) to the speedrun-compatible
base and is available on the public `speedrun-fast-io-compat` branch.

For a source checkout, dependency initialization is:

```bash
git submodule update --init --recursive
pip install -e '.[dev]'
```

## Component checks

These checks exercise real image codecs, filesystem writes, recorder threads,
the native client, local HTTP transfers, and immutable run contracts:

```bash
python -m pytest -q tests/test_fast_io_components.py \
  tests/test_gym_anything_fast_io.py tests/test_prepared_capture.py \
  tests/test_transport_profile_and_region.py tests/test_evaluation_region.py \
  tests/test_codex_http_proxy.py
```
