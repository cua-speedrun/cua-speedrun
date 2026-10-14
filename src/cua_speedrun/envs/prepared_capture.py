"""Prepare exact framebuffer versions off the observation request path.

The D-Bus listener remains authoritative. An observation selects its version
under the same lock as display updates, never the last *encoded* version.
"""

from __future__ import annotations

import threading
import time

from cua_speedrun.images import encode_image


class PreparedFrame:
    """One private, immutable pixel snapshot and its shared encoding."""

    def __init__(self, image, version, image_format):
        self.image = image
        self.version = version
        self.image_format = image_format
        self._lock = threading.Lock()
        self._encoded = None
        self.encode_sec = 0.0

    @property
    def ready(self):
        return self._encoded is not None

    def encoded(self):
        with self._lock:
            if self._encoded is None:
                started = time.perf_counter()
                self._encoded = encode_image(self.image, self.image_format, threaded=True)
                self.encode_sec = time.perf_counter() - started
            data, metadata = self._encoded
            return data, dict(metadata)

    def observation_image(self):
        # Do not expose the private snapshot to verifiers or other PIL users.
        image = self.image.copy()
        image._cua_prepared_frame = self
        return image


class PreparedDbusCapture:
    def __init__(self, display, image_format):
        self.display = display
        self.image_format = image_format
        self._original_capture = display.capture_image
        self._original_ready_set = display._ready.set
        self._snapshot_lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._capture_callback = self.capture_image
        self._notify_callback = self._notify
        display.capture_image = self._capture_callback
        # The pinned listener calls set after every Scanout/Update, after
        # releasing its framebuffer lock. Preserve the original Event too.
        display._ready.set = self._notify_callback
        self._thread = threading.Thread(target=self._run, name="frame-encoder", daemon=True)
        self._thread.start()
        self._wake.set()

    def _notify(self):
        self._original_ready_set()
        self._wake.set()

    def _snapshot(self):
        if self._error is not None:
            raise RuntimeError("background frame encoding failed") from self._error
        if self.display._frame is None:
            self.display._ready.wait(1.0)
        with self.display._lock:
            if self.display._frame is None:
                raise RuntimeError("D-Bus display has not delivered a frame")
            # Listener updates pixels, dimensions and this generation stamp
            # together under _lock. The object is per-display/episode.
            version = self.display._last_update_ns
            with self._snapshot_lock:
                if self._latest is not None and self._latest.version == version:
                    return self._latest
            # Match the dependency's snapshot operation, but do conversion
            # outside BOTH locks. Holding the display lock during conversion
            # blocks its updates and can stall foreground captures behind
            # speculative work on a rapidly changing desktop.
            pixels = bytes(self.display._frame)
            size = self.display._width, self.display._height
            stride = self.display._stride
        from PIL import Image
        image = Image.frombytes("RGB", size, pixels, "raw", "BGRX", stride, 1)
        frame = PreparedFrame(image, version, self.image_format)
        with self._snapshot_lock:
            if self._latest is not None and self._latest.version == version:
                return self._latest
            if self._latest is None or self._latest.version < version:
                self._latest = frame
            return frame

    def capture_image(self):
        frame = self._snapshot()
        ready = frame.ready
        image = frame.observation_image()
        image._cua_encoding_ready_at_capture = ready
        return image

    def encode(self, image, image_format):
        frame = getattr(image, "_cua_prepared_frame", None)
        if not isinstance(frame, PreparedFrame) or frame.image_format != image_format:
            return encode_image(image, image_format)
        return frame.encoded()

    def _run(self):
        try:
            while True:
                self._wake.wait()
                self._wake.clear()
                if self._stop.is_set():
                    return
                self._snapshot().encoded()
                # Updates arriving during encoding set _wake again. Only
                # the newest display version is selected on the next pass.
        except BaseException as exc:
            self._error = exc

    def close(self):
        self.display._ready.set = self._original_ready_set
        self.display.capture_image = self._original_capture
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("frame encoder did not stop")
        if self._error is not None:
            raise RuntimeError("background frame encoding failed") from self._error


def prepared_encoding_enabled(environment):
    value = environment.get("GYM_ANYTHING_FAST_IO_PREPARE_IMAGES", "true").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("GYM_ANYTHING_FAST_IO_PREPARE_IMAGES must be true or false")
    return value in {"true", "1"}
