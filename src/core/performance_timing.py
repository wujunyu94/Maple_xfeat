"""Low-overhead, out-of-band timing log for capture and monster scans."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time


def _summary(values: list[float]) -> str:
    if not values:
        return "-/-/-"
    ordered = sorted(values)
    p50 = ordered[(len(ordered) - 1) // 2]
    p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
    return f"{p50:.1f}/{p95:.1f}/{ordered[-1]:.1f}"


class PerformanceTimingLog:
    """Collect samples on hot threads; format and write on a slow worker."""

    def __init__(self, path: str | Path, interval_sec: float = 5.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._file.write(
            "# Every 5s; timing columns are milliseconds as p50/p95/max. "
            "FPS counts newly published capture frames, not tracker iterations.\n"
        )
        self._interval_sec = max(0.1, float(interval_sec))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._window_start = time.perf_counter()
        self._capture: list[tuple[str, float, float, float, float | None]] = []
        self._scan: list[tuple[
            str, float, float, float, float, float, float | None,
            float | None, float | None, float | None,
        ]] = []
        self._ocr: list[tuple[float, str, str, float | None, str | None]] = []
        self._last_capture_at: float | None = None
        self._thread = threading.Thread(
            target=self._run, name="PerformanceTimingLog", daemon=True
        )
        self._thread.start()

    def record_capture(
        self, backend: str, acquire_ms: float, publish_ms: float, cycle_ms: float
    ) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._closed:
                return
            cadence_ms = (
                None if self._last_capture_at is None
                else (now - self._last_capture_at) * 1000.0
            )
            self._last_capture_at = now
            self._capture.append((backend, acquire_ms, publish_ms, cycle_ms, cadence_ms))

    def record_scan(
        self,
        mode: str,
        schedule_lag_ms: float,
        frame_wait_ms: float,
        backend_lock_wait_ms: float,
        execute_ms: float,
        total_ms: float,
        start_gap_ms: float | None,
        *,
        template_ms: float | None = None,
        hp_bar_ms: float | None = None,
        candidate_filter_ms: float | None = None,
    ) -> None:
        with self._lock:
            if not self._closed:
                self._scan.append((
                    mode, schedule_lag_ms, frame_wait_ms,
                    backend_lock_wait_ms, execute_ms, total_ms, start_gap_ms,
                    template_ms, hp_bar_ms, candidate_filter_ms,
                ))

    def record_ocr(
        self, phase: str, pass_name: str,
        elapsed_ms: float | None = None, outcome: str | None = None,
    ) -> None:
        """Record exact OCR boundaries; the writer thread handles disk I/O."""
        wall_time = time.time()
        with self._lock:
            if not self._closed:
                self._ocr.append((wall_time, phase, pass_name, elapsed_ms, outcome))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_sec):
            self.flush()

    def flush(self) -> None:
        now = time.perf_counter()
        with self._lock:
            capture, scan, ocr = self._capture, self._scan, self._ocr
            self._capture, self._scan, self._ocr = [], [], []
            elapsed = max(0.001, now - self._window_start)
            self._window_start = now
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        backend = capture[-1][0] if capture else "idle"
        capture_line = (
            f"[{stamp}] CAPTURE backend={backend} fps={len(capture) / elapsed:.1f} "
            f"frames={len(capture)} window_s={elapsed:.2f} "
            f"cycle_ms={_summary([row[3] for row in capture])} "
            f"acquire_ms={_summary([row[1] for row in capture])} "
            f"publish_ms={_summary([row[2] for row in capture])} "
            f"cadence_ms={_summary([row[4] for row in capture if row[4] is not None])} "
            f"cadence_gt20={sum(1 for row in capture if row[4] is not None and row[4] > 20.0)}\n"
        )
        scan_lines = []
        for mode in sorted({row[0] for row in scan}):
            rows = [row for row in scan if row[0] == mode]
            gaps = [row[6] for row in rows if row[6] is not None]
            scan_lines.append(
                f"[{stamp}] SCAN mode={mode} fps={len(rows) / elapsed:.1f} "
                f"scans={len(rows)} schedule_lag_ms={_summary([row[1] for row in rows])} "
                f"frame_wait_ms={_summary([row[2] for row in rows])} "
                f"backend_lock_wait_ms={_summary([row[3] for row in rows])} "
                f"execute_ms={_summary([row[4] for row in rows])} "
                f"template_ms={_summary([row[7] for row in rows if row[7] is not None])} "
                f"hp_bar_ms={_summary([row[8] for row in rows if row[8] is not None])} "
                f"filter_ms={_summary([row[9] for row in rows if row[9] is not None])} "
                f"total_ms={_summary([row[5] for row in rows])} "
                f"start_gap_ms={_summary(gaps)} "
                f"gap_gt70={sum(1 for gap in gaps if gap > 70.0)}\n"
            )
        if not scan_lines:
            scan_lines.append(f"[{stamp}] SCAN mode=idle fps=0.0 scans=0\n")
        ocr_lines = []
        for wall_time, phase, pass_name, elapsed_ms, outcome in ocr:
            event_stamp = datetime.fromtimestamp(wall_time).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            line = f"[{event_stamp}] OCR phase={phase} pass={pass_name}"
            if elapsed_ms is not None:
                line += f" elapsed_ms={elapsed_ms:.1f}"
            if outcome is not None:
                line += f" outcome={outcome}"
            ocr_lines.append(line + "\n")
        try:
            self._file.write(capture_line + "".join(scan_lines) + "".join(ocr_lines))
        except OSError:
            pass  # Diagnostics must never interrupt capture or control.

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.flush()
        self._file.close()
