"""CLI and independent desktop entry point. Never sends game inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .atlas import load_atlas
from .display import annotate, overview, add_ladder_overlay
from .pipeline import Pipeline


def live_capture():
    from src.core.window import WindowManager
    from src.vision.capture import ScreenCapture
    wm = WindowManager()
    if not wm.find_game_window():
        raise RuntimeError("未找到游戏窗口，请先打开游戏")
    return ScreenCapture(wm, use_async_thread=False, prefer_wgc=False)


def image_read(path):
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def run_offline(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atlas = load_atlas(args.map_id, args.rebuild)
    pipeline = Pipeline(atlas, args.screen_scale, asynchronous=bool(args.live and getattr(args, "asynchronous", False)),
                        backend=getattr(args, "backend", "sift-cpu"))
    model = pipeline.model
    capture = live_capture() if args.live else None
    video = cv2.VideoCapture(args.video) if args.video else None
    if video is not None and not video.isOpened():
        raise ValueError(f"无法打开录像：{args.video}")
    fps = max(1, video.get(cv2.CAP_PROP_FPS)) if video else 0
    records = []
    frame = None
    try:
        with (output / "poses.jsonl").open("w", encoding="utf-8") as log:
            for index in range(args.max_frames):
                if args.image:
                    frame = image_read(args.image)
                    timestamp = 0.0
                elif video:
                    ok, candidate = video.read()
                    if not ok:
                        break
                    frame = candidate
                    timestamp = index/fps
                else:
                    frame = capture.capture_frame()
                    timestamp = time.perf_counter()
                    if frame is None:
                        raise RuntimeError("无法捕获游戏窗口")
                record = pipeline.update(frame, timestamp)
                record["frame"] = index
                player, pose = record["player"], pipeline.pose
                log.write(json.dumps(record, ensure_ascii=False)+"\n")
                records.append(record)
                if args.image:
                    break
                if capture:
                    time.sleep(1/60)
        if frame is not None and records:
            rendered = add_ladder_overlay(annotate(frame, model, pose, player), record["ladder"], record["ladders"])
            Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)).save(output / "last_frame.png")
            Image.fromarray(cv2.cvtColor(overview(atlas, frame.shape, pose, args.screen_scale, player, yellow=record["yellow"]), cv2.COLOR_BGR2RGB)).save(output / "world_overview.png")
        summary = dict(frames=len(records), locked=sum(r["status"] == "LOCKED" for r in records),
                       coasting=sum(r["status"] == "COASTING" for r in records),
                       lost=sum(r["camera"] is None for r in records),
                       median_ms=float(np.median([r["elapsed_ms"] for r in records])) if records else None,
                       median_pipeline_ms=float(np.median([r["timings_ms"]["pipeline"] for r in records])) if records else None,
                       last=records[-1] if records else None)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    finally:
        if capture:
            capture.release()
        if video:
            video.release()
        pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="无小地图视觉定位实验")
    parser.add_argument("--map-id", type=int, default=103000201)
    parser.add_argument("--screen-scale", type=float, default=1.0, help="屏幕像素/地图像素")
    parser.add_argument("--backend", choices=("sift-cpu", "xfeat-cpu", "xfeat-cuda"), default="sift-cpu")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image")
    source.add_argument("--video")
    source.add_argument("--live", action="store_true", help="无界面实时采样")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output", default="no_minimap_lab/output/latest")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--asynchronous", action="store_true", help="--live 使用后台世界匹配；截图和录像保持同步以便复现")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    cv2.setNumThreads(4)
    if args.image or args.video or args.live:
        run_offline(args)
    else:
        from .gui import App
        App(args.map_id, args.screen_scale, args.backend).run()


if __name__ == "__main__":
    main()
