"""Create a bounded, shareable snapshot of the current diagnostic logs."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


LOG_NAMES = (
    "runtime_actions.log",
    "performance_timing.log",
    "patrol_fsm.jsonl",
    "topology_world_debug.csv",
    "coordinate_trace.csv",
)


def save_log_snapshot(log_dir: str | Path, now: datetime | None = None) -> Path:
    """Save the bytes present at click time without blocking log writers."""
    source_dir = Path(log_dir)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    destination_dir = source_dir / "saved"
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive = destination_dir / f"current_logs_{stamp}.zip"
    included = 0
    with ZipFile(archive, mode="x", compression=ZIP_DEFLATED, compresslevel=3) as bundle:
        for name in LOG_NAMES:
            source = source_dir / name
            if not source.is_file():
                continue
            with source.open("rb") as reader:
                # Limit each copy to its size when opened. A running log may
                # grow during compression, but the snapshot remains finite.
                remaining = os.fstat(reader.fileno()).st_size
                with bundle.open(name, "w") as writer:
                    while remaining:
                        chunk = reader.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        writer.write(chunk)
                        remaining -= len(chunk)
            included += 1
        bundle.writestr("snapshot_time.txt", stamp + "\n")
    if not included:
        archive.unlink(missing_ok=True)
        raise FileNotFoundError(f"No current logs in {source_dir}")
    return archive
