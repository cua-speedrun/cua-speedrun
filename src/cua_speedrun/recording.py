"""Bounded, ordered archival writes; failures surface at submit/drain/close."""

from __future__ import annotations

import copy
import queue
import threading
from typing import Any, Callable


class OrderedRecorder:
    def __init__(self, capacity: int = 16, timeout: float = 60.0):
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._timeout = timeout
        self._error: BaseException | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="artifact-recorder", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                fn, args = job
                fn(*args)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("background recording failed") from self._error

    def submit(self, fn: Callable, *args: Any) -> None:
        with self._lock:
            self._check()
            if self._closed:
                raise RuntimeError("recorder is closed")
            self._put((fn, args))

    def _put(self, item: Any) -> None:
        try:
            self._queue.put(item, timeout=self._timeout)
        except queue.Full as exc:
            raise TimeoutError("background recording queue did not drain") from exc

    def flush(self) -> None:
        barrier = threading.Event()
        with self._lock:
            closed = self._closed
            if not closed:
                self._put((barrier.set, ()))
        if closed:
            self._thread.join(self._timeout)
            if self._thread.is_alive():
                raise TimeoutError("background recording did not stop")
        elif not barrier.wait(self._timeout):
            raise TimeoutError("background recording did not finish")
        self._check()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._put(None)
                self._closed = True
        self._thread.join(self._timeout)
        if self._thread.is_alive():
            raise TimeoutError("background recording did not stop")
        self._check()


class AsyncJSONL:
    """Wrap an existing JSONL writer without changing its file or record format."""

    def __init__(self, writer: Any):
        self.path = writer.path
        self._writer = writer
        self._recorder = OrderedRecorder()

    def write(self, record: dict) -> None:
        self._recorder.submit(self._writer.write, copy.deepcopy(record))

    def flush(self) -> None:
        self._recorder.flush()

    def close(self) -> None:
        try:
            self._recorder.close()
        finally:
            self._writer.close()
