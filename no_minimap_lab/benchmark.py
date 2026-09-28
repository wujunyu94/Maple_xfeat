"""Wall-time/CPU/RSS benchmark. Measures actual loop throughput, not 1/flow_ms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
import psutil

from .atlas import load_atlas
from .pipeline import Pipeline
from .run import live_capture, image_read


def percentiles(values):
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                max=float(max(values))) if values else None


def measure(atlas, seconds, asynchronous, image=None, backend="sift-cpu"):
    pipeline = Pipeline(atlas, asynchronous=asynchronous, backend=backend)
    capture = None if image else live_capture()
    frozen = image_read(image) if image else None
    process = psutil.Process()
    records, costs, captured = [], [], []
    cpu_start = process.cpu_times()
    start = time.perf_counter()
    try:
        while time.perf_counter()-start < seconds:
            tick = time.perf_counter()
            frame = frozen if frozen is not None else capture.capture_frame()
            stamp = time.perf_counter()
            if frame is None:
                raise RuntimeError("No captured frame during benchmark")
            r = pipeline.update(frame, stamp)
            captured.append((stamp-tick)*1000)
            costs.append((time.perf_counter()-tick)*1000)
            records.append(r)
            # Same target cadence as the GUI, but excludes GUI rendering cost.
            time.sleep(max(0, 1/60-(time.perf_counter()-tick)))
        elapsed = time.perf_counter()-start
        cpu_end = process.cpu_times()
        cpu_seconds = cpu_end.user+cpu_end.system-cpu_start.user-cpu_start.system
        cpu = cpu_seconds/elapsed*100
        measured = [r for r in records if r["comparison"] and r["comparison"]["kind"] == "measurement_comparison"]
        result = dict(mode="async_full_pipeline" if asynchronous else "sync_full_pipeline", backend=backend,
                      map_id=atlas.meta["map_id"],
                      source="frozen_frame" if image else "live_client", includes_gui_rendering=False,
                      frames=len(records), wall_seconds=elapsed, hz=len(records)/elapsed,
                      cpu_percent_one_core=cpu, cpu_percent_machine=cpu/(psutil.cpu_count() or 1),
                      logical_cpus=psutil.cpu_count(), rss_mb=process.memory_info().rss/1024**2,
                      total_ms=percentiles(costs), capture_ms=percentiles(captured),
                      frames_within_16_67ms=sum(x <= 1000/60 for x in costs),
                      camera_status=dict(Counter(r["status"] for r in records)),
                      valid_camera_fraction=sum(r["camera"] is not None for r in records)/len(records),
                      player_status=dict(Counter(r["player_state"]["status"] for r in records)),
                      comparison_distance_px=percentiles([r["comparison"]["distance"] for r in measured]),
                      stage_ms={name: percentiles([r["timings_ms"][name] for r in records])
                                for name in ("camera", "player", "yellow", "ladders", "background_anchor")},
                      first_valid_seconds=next((r["timestamp"]-start for r in records if r["player_world"]), None))
        if backend == "xfeat-cuda":
            import torch
            result["cuda"] = dict(device=torch.cuda.get_device_name(),
                                  allocated_mb=torch.cuda.memory_allocated()/1024**2,
                                  reserved_mb=torch.cuda.memory_reserved()/1024**2,
                                  peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2)
        return result, records
    finally:
        pipeline.close()
        if capture:
            capture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", type=int, default=103000201)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--image")
    parser.add_argument("--backend", choices=("sift-cpu", "xfeat-cpu", "xfeat-cuda"), default="sift-cpu")
    parser.add_argument("--mode", choices=("both", "sync", "async"), default="both")
    parser.add_argument("--output", default="no_minimap_lab/output/performance_v2")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    cv2.setNumThreads(4)
    atlas = load_atlas(args.map_id)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for asynchronous in ([False, True] if args.mode == "both" else [args.mode == "async"]):
        report, records = measure(atlas, args.seconds, asynchronous, args.image, args.backend)
        reports.append(report)
        (output / f"{report['mode']}.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    (output / "report.json").write_text(json.dumps(dict(opencv=cv2.__version__,
        cuda_devices=cv2.cuda.getCudaEnabledDeviceCount(), opencv_threads=cv2.getNumThreads(),
        reports=reports), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
