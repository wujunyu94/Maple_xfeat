"""Identical-input accuracy/latency checks for CPU and CUDA world anchoring."""
import json
from pathlib import Path

import cv2
import numpy as np

from .atlas import ROOT, load_atlas
from .localizer import Localizer


def main():
    cv2.setNumThreads(4)
    output = ROOT / "no_minimap_lab/output/backends_v3"
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for map_id in (103000201, 101000000):
        atlas = load_atlas(map_id)
        cases = []
        if map_id == 103000201:
            for x, y in ((650, 180), (1800, 270)):
                frame = cv2.warpAffine(atlas.bgr, np.float32([[1, 0, -x], [0, 1, -y]]), (1280, 720))
                cases.append((f"known_crop_{x}_{y}", frame, (atlas.origin+[x, y]).tolist()))
            occluded = cases[0][1].copy()
            occluded[250:500, 450:850] = 140
            cases.append(("known_crop_occluded", occluded, cases[0][2]))
            paths = ("no_minimap_lab/cache/comparison_probe.png", "no_minimap_lab/cache/live_probe.png")
        else:
            paths = ("real_game_frame.png",)
        for path in paths:
            image = cv2.imread(str(ROOT/path))
            if image is not None:
                cases.append((Path(path).stem, image, None))
        reference = {}
        for backend in ("sift-cpu", "xfeat-cpu", "xfeat-cuda"):
            model = Localizer(atlas, backend=backend)
            for name, image, expected in cases:
                times = []
                for run in range(4):
                    model.reset()
                    pose = model.update(image, timestamp=0, force=True)
                    if run:
                        times.append(pose.elapsed_ms)
                if backend == "sift-cpu":
                    reference[name] = pose.camera
                baseline = reference.get(name)
                report = dict(map_id=map_id, backend=backend, case=name, status=pose.status,
                              camera=pose.camera, expected=expected, inliers=pose.inliers,
                              median_ms=float(np.median(times)), max_ms=max(times),
                              true_error_px=float(np.linalg.norm(np.array(pose.camera)-expected)) if pose.camera and expected else None,
                              sift_delta_px=float(np.linalg.norm(np.array(pose.camera)-baseline)) if pose.camera and baseline else None)
                reports.append(report)
                print(json.dumps(report), flush=True)
    (output/"report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
