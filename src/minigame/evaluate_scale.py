"""Replay the guarded baseline and scale-aware tracker against external labels.

Usage: python -m src.minigame.evaluate_scale --data-root path/to/data
Labels are used only for scoring after update; the runtime never reads them.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .guarded_tracker import CausalShapeTracker as Baseline
from .scale_tracker import CausalShapeTracker as Candidate


def metrics(rows):
    errors = np.asarray([r['error'] for r in rows if r['error'] is not None])
    return dict(samples=len(rows), tracked=len(errors),
                mean=round(float(errors.mean()), 2) if len(errors) else None,
                p95=round(float(np.percentile(errors, 95)), 2) if len(errors) else None,
                maximum=round(float(errors.max()), 2) if len(errors) else None,
                over100=int(np.sum(errors > 100)),
                wrong_mouse_frames=sum(r['error'] is not None and r['error'] > 100
                                       and r['confidence'] >= .45 for r in rows))


def evaluate(root, name, scale, capture_fps):
    matches = sorted((root / 'calibration_data').glob(name + '_*.json'))
    if not matches:
        raise FileNotFoundError('No calibration for ' + name)
    annotation = json.loads(matches[-1].read_text(encoding='utf-8'))
    truth = {int(s['frame']): s for s in annotation['samples']}
    cap = cv2.VideoCapture(str(root / 'testVideo' / (name + '.mp4')))
    if not cap.isOpened():
        raise FileNotFoundError(name)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = capture_fps or source_fps
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trackers = {'before': Baseline(), 'after': Candidate()}
    rows = {k: [] for k in trackers}
    elapsed = {k: [] for k in trackers}
    source_index = -1
    frame = None
    normalized_frames = 0
    try:
        for tick in range(int(np.ceil(count / source_fps * fps))):
            index = min(count - 1, int(tick * source_fps / fps + 1e-6))
            if index != source_index:
                while source_index < index:
                    ok, raw = cap.read()
                    if not ok:
                        return None
                    source_index += 1
                h, w = raw.shape[:2]
                frame = cv2.resize(raw, (round(w * scale), round(h * scale)),
                                   interpolation=cv2.INTER_AREA) if scale != 1 else raw
                sx, sy = frame.shape[1] / w, frame.shape[0] / h
            label = truth.get(index, {})
            results = {}
            for key, tracker in trackers.items():
                start = time.perf_counter()
                results[key] = tracker.update(frame, tick / fps)
                elapsed[key].append(time.perf_counter() - start)
            normalized_frames += trackers['after'].normalized
            # Both versions are scored on the same motion-phase ticks, including
            # a missing result as a coverage failure, rather than omitting it.
            if not any(t.pre_game_finished for t in trackers.values()):
                continue
            if label.get('video_x') is None or label.get('video_y') is None:
                continue
            for key, result in results.items():
                error = float(np.hypot(result.x / sx - label['video_x'],
                                       result.y / sy - label['video_y'])) if result.initialized else None
                rows[key].append(dict(frame=index, time=tick/fps, error=error,
                                      x=result.x/sx, y=result.y/sy,
                                      confidence=result.confidence))
    finally:
        cap.release()
    return dict(video=name, scale=scale, capture_fps=fps,
                normalization_frames=normalized_frames,
                duplicate_frames=trackers['after'].duplicate_frames,
                summary={k: dict(**metrics(v),
                                update_ms_mean=round(float(np.mean(elapsed[k]))*1000, 2),
                                update_ms_p95=round(float(np.percentile(elapsed[k],95))*1000, 2))
                         for k, v in rows.items()}, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--videos', nargs='+', default=['sample_special3'])
    p.add_argument('--scales', type=float, nargs='+', default=[1.0, .72])
    p.add_argument('--fps', type=float, nargs='+', default=[30.0])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if any(s <= 0 for s in args.scales) or any(f < 0 for f in args.fps):
        p.error('scales must be positive and fps nonnegative (0 = source fps)')
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.videos:
        for scale in args.scales:
            for fps in args.fps:
                result = evaluate(args.data_root, name, scale, fps)
                if result is None:
                    raise RuntimeError('Video decode failed: ' + name)
                filename = f'{name}_scale{scale:g}_fps{fps:g}.json'
                (args.output / filename).write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
