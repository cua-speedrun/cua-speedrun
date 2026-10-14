"""Screenshot formats shared by the environment, wire client, and recorder.

Inspecting image headers needs no Pillow in the agent sandbox. Encoding and
legacy PNG conversion run only where a Pillow image is already available.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Mapping


def image_mime_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise ValueError("screenshot must be PNG or JPEG")


def image_suffix(mime_type: str) -> str:
    return {"image/png": ".png", "image/jpeg": ".jpg"}[mime_type]


def fast_image_format(environment: Mapping[str, str]) -> str:
    value = environment.get("GYM_ANYTHING_FAST_IO_IMAGE_FORMAT", "jpeg").strip().lower()
    if value not in {"jpeg", "png"}:
        raise ValueError("GYM_ANYTHING_FAST_IO_IMAGE_FORMAT must be jpeg or png")
    return value


def encode_image(image, image_format: str, *, threaded: bool = False) -> tuple[bytes, dict]:
    if threaded and image_format == "jpeg":
        # Pillow's BytesIO encoder holds the GIL. Its native file-descriptor
        # encoder releases it. A Linux memfd is anonymous RAM, not disk I/O;
        # other hosts use the portable file-descriptor fallback.
        output = (os.fdopen(os.memfd_create("cua-jpeg", os.MFD_CLOEXEC), "w+b")
                  if hasattr(os, "memfd_create") else tempfile.TemporaryFile())
    else:
        output = io.BytesIO()
    with output:
        if image_format == "jpeg":
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(output, format="JPEG", quality=95, subsampling=0,
                       optimize=False, progressive=False)
            encoding = {"format": "jpeg", "quality": 95, "subsampling": "4:4:4", "lossless": False}
        elif image_format == "png":
            image.save(output, format="PNG", compress_level=1)
            encoding = {"format": "png", "compress_level": 1, "lossless": True}
        else:
            raise ValueError(f"unsupported screenshot format: {image_format}")
        if isinstance(output, io.BytesIO):
            data = output.getvalue()
        else:
            output.seek(0)
            data = output.read()
        return data, encoding


def as_png(data: bytes) -> bytes:
    """Compatibility conversion only; the image-aware fast path never calls it."""
    if not data or image_mime_type(data) == "image/png":
        return data
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        return encode_image(image, "png")[0]
