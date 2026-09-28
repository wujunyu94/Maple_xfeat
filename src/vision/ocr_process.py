"""Run map-title OCR outside the latency-sensitive vision process."""

from __future__ import annotations

import multiprocessing
import os
import threading
from multiprocessing.connection import Connection
from typing import Any, Callable


def _configure_worker() -> tuple[int, str]:
    """Give OCR limited CPU capacity without changing the main process."""
    threads = 1
    details = []
    try:
        import psutil

        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            details.append("below-normal")
        else:
            os.nice(10)
            details.append("nice+10")
        allowed = process.cpu_affinity()
        if len(allowed) >= 8:
            selected = allowed[-2:]
        elif len(allowed) >= 4:
            selected = allowed[-1:]
        else:
            selected = []
        if selected:
            process.cpu_affinity(selected)
            threads = len(selected)
            details.append(f"cpu={selected}")
    except Exception as exc:
        # Priority/affinity are optimizations; OCR must still work if the OS
        # refuses them (for example, inside a restricted process group).
        details.append(f"limits-unavailable:{type(exc).__name__}")
    return threads, ",".join(details)


def _worker_main(connection: Connection) -> None:
    try:
        threads, limits = _configure_worker()
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR(
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
        )
        connection.send(("ready", limits))
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if request is None:
                break
            operation, image = request
            if operation != "ocr":
                connection.send(("error", "unknown OCR operation"))
                continue
            try:
                connection.send(("ok", engine(image)))
            except Exception as exc:
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        try:
            connection.send(("fatal", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class IsolatedMapOCR:
    """A callable RapidOCR-compatible proxy with a local fail-safe."""

    def __init__(self, status_callback: Callable[[str], None] | None = None) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._status_callback = status_callback
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._fallback: Any = None
        self._closed = False

    def _notify(self, message: str) -> None:
        if self._status_callback is not None:
            try:
                self._status_callback(message)
                return
            except Exception:
                pass
        print(message)

    def _start_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main, args=(child,), name="MapTitleOCR", daemon=True,
        )
        try:
            process.start()
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        self._process = process
        self._connection = parent
        if not parent.poll(12.0):
            raise TimeoutError("OCR worker initialization exceeded 12 seconds")
        state, details = parent.recv()
        if state != "ready":
            raise RuntimeError(f"OCR worker initialization failed: {details}")
        self._notify(f"[地图OCR] 独立进程已启动 pid={process.pid}，{details}")

    def _stop_worker(self) -> None:
        connection = self._connection
        process = self._process
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=0.5)
            except (OSError, ValueError):
                pass

    def __call__(self, image: Any) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("OCR worker has been closed")
            if self._fallback is not None:
                return self._fallback(image)
            try:
                self._start_worker()
                assert self._connection is not None
                self._connection.send(("ocr", image))
                if not self._connection.poll(8.0):
                    raise TimeoutError("OCR inference exceeded 8 seconds")
                state, result = self._connection.recv()
                if state != "ok":
                    raise RuntimeError(f"OCR worker failed: {result}")
                return result
            except Exception as exc:
                self._stop_worker()
                if self._closed:
                    raise RuntimeError("OCR worker has been closed") from exc
                self._notify(f"[地图OCR] 独立进程不可用，回退本地 OCR：{exc}")
                from rapidocr_onnxruntime import RapidOCR

                self._fallback = RapidOCR(
                    intra_op_num_threads=1, inter_op_num_threads=1,
                )
                return self._fallback(image)

    def close(self) -> None:
        self._closed = True
        self._stop_worker()
