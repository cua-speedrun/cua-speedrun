"""Real image/codec/thread checks, without a VM or simulated environment."""

from concurrent.futures import ThreadPoolExecutor
import io

import pytest
from PIL import Image

from cua_speedrun.envs.prepared_capture import PreparedFrame, prepared_encoding_enabled
from cua_speedrun.images import encode_image


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("image_format", ["jpeg", "png"])
def test_threaded_file_encoder_preserves_exact_bytes(mode, image_format):
    image = Image.new(mode, (1920, 1080), "#537ca9")
    assert encode_image(image, image_format, threaded=True) == encode_image(image, image_format)


@pytest.mark.parametrize("image_format", ["jpeg", "png"])
def test_prepared_bytes_equal_fresh_encoding(image_format):
    image = Image.new("RGB", (1920, 1080), "#5b279d")
    for x in range(0, 1920, 17):
        image.putpixel((x, x % 1080), (x % 256, 191, 75))
    frame = PreparedFrame(image, 17, image_format)
    assert not frame.ready
    expected = encode_image(image, image_format)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: frame.encoded(), range(12)))
    assert all(result == expected for result in results)
    assert all(result[0] is results[0][0] for result in results)
    assert frame.ready and frame.encode_sec > 0
    for result in results:
        with Image.open(io.BytesIO(result[0])) as decoded:
            decoded.load()
            assert decoded.size == (1920, 1080)


def test_new_versions_cannot_reuse_previous_pixels_or_bytes():
    old = PreparedFrame(Image.new("RGB", (128, 128), "red"), 11, "jpeg")
    new = PreparedFrame(Image.new("RGB", (128, 128), "blue"), 12, "jpeg")
    assert old.encoded()[0] != new.encoded()[0]
    assert old.version != new.version
    observation = new.observation_image()
    assert observation._cua_prepared_frame is new
    assert observation.tobytes() == new.image.tobytes()
    observation.putpixel((0, 0), (0, 0, 0))
    assert new.image.getpixel((0, 0)) == (0, 0, 255)
    first = new.encoded()
    first[1]["quality"] = 1
    assert new.encoded()[1]["quality"] == 95


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("false", False), ("0", False)])
def test_preparation_flag(value, expected):
    assert prepared_encoding_enabled({"GYM_ANYTHING_FAST_IO_PREPARE_IMAGES": value}) is expected


@pytest.mark.parametrize("value", ["", "yes", "invalid"])
def test_invalid_preparation_flag(value):
    with pytest.raises(ValueError):
        prepared_encoding_enabled({"GYM_ANYTHING_FAST_IO_PREPARE_IMAGES": value})


def test_preparation_defaults_on():
    assert prepared_encoding_enabled({})
