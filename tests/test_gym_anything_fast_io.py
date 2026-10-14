"""Pure option/data-conversion checks; these do not boot or simulate a VM."""

import base64
import io
import json

import pytest
from PIL import Image

from cua_speedrun.envs.gym_anything import (
    action_settle_seconds,
    fast_io_enabled,
    observation_from_gym,
)
from cua_speedrun.runtime_environment import normalize_environment_name


def test_fast_io_defaults_off():
    assert not fast_io_enabled({})
    assert normalize_environment_name("GYM_ANYTHING_FAST_IO") == "GYM_ANYTHING_FAST_IO"


@pytest.mark.parametrize("value", ["true", "1", " TRUE "])
def test_fast_io_explicit_opt_in(value):
    assert fast_io_enabled({"GYM_ANYTHING_FAST_IO": value})


@pytest.mark.parametrize("value", ["false", "0", " False "])
def test_fast_io_explicit_off(value):
    assert not fast_io_enabled({"GYM_ANYTHING_FAST_IO": value})


@pytest.mark.parametrize("value", ["", "tru", "yes", "2"])
def test_fast_io_invalid_value_fails_closed(value):
    with pytest.raises(ValueError, match="GYM_ANYTHING_FAST_IO"):
        fast_io_enabled({"GYM_ANYTHING_FAST_IO": value})


@pytest.mark.parametrize("env_spec", [{}, {"action_settle_ms": 300}, {"action_settle_ms": 1500}])
def test_fast_io_has_no_fixed_settling_even_with_task_override(env_spec):
    assert action_settle_seconds(env_spec, fast_io=True) == 0.0


@pytest.mark.parametrize("env_spec,expected", [
    ({}, 0.3),
    ({"action_settle_ms": 0}, 0.0),
    ({"action_settle_ms": 125}, 0.125),
    ({"action_settle_ms": 1500}, 1.5),
])
def test_normal_io_settling_is_unchanged(env_spec, expected):
    assert action_settle_seconds(env_spec, fast_io=False) == expected


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_fast_image_becomes_lossless_png_and_json_metadata(mode):
    source = Image.new(mode, (13, 7), "#32a8e3")
    obs = {"screen": {"image": source, "format": "pil", "resolution": [13, 7], "mode": mode},
           "ui_tree": {"text": "desktop"}}
    result = observation_from_gym(obs)
    with Image.open(io.BytesIO(result.png)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.mode == source.mode
        assert decoded.size == source.size
        assert decoded.tobytes() == source.tobytes()
    assert json.loads(json.dumps(result.meta)) == {
        "format": "png", "resolution": [13, 7], "mode": mode,
        "ui_tree": {"text": "desktop"},
    }
    assert obs["screen"]["image"] is source
    assert obs["screen"]["format"] == "pil"


def test_legacy_png_bytes_and_metadata_unchanged(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (13, 7), "#32a8e3").save(buffer, format="PNG")
    png = buffer.getvalue()
    path = tmp_path / "screen.png"
    path.write_bytes(png)
    for payload in ({"path": str(path)}, {"png_b64": base64.b64encode(png).decode()}):
        result = observation_from_gym({"screen": {**payload, "format": "png", "resolution": [13, 7]}})
        assert result.png == png
        assert result.meta == {"format": "png", "resolution": [13, 7]}


def test_legacy_payload_precedence_unchanged(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    original = buffer.getvalue()
    result = observation_from_gym({"screen": {
        "png_b64": base64.b64encode(original).decode(),
        "path": str(tmp_path / "nonexistent.png"),
        "image": Image.new("RGB", (1, 1)),
    }})
    assert result.png == original
    assert result.meta == {}


def test_missing_screen_stays_empty():
    result = observation_from_gym({})
    assert result.png == b""
    assert result.meta == {}
