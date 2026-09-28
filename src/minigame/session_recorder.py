"""Archive mini-game tracks in the web monitor's JSON and CSV schema."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import time
from typing import Any


CSV_COLUMNS = (
    "frame_idx", "time_offset_sec", "roi_x", "roi_y", "roi_w", "roi_h",
    "roi_bl_x", "roi_bl_y", "roi_tl_x", "roi_tl_y",
    "roi_tr_x", "roi_tr_y", "roi_br_x", "roi_br_y",
    "target_x", "target_y", "diff_x", "diff_y",
    "global_target_x", "global_target_y", "confidence", "initialized",
)


def roi_corners(roi: list[int] | None) -> dict[str, list[int]] | None:
    if roi is None or len(roi) != 4:
        return None
    x, y, w, h = map(int, roi)
    return {
        "bottom_left": [x, y + h],
        "top_left": [x, y],
        "top_right": [x + w, y],
        "bottom_right": [x + w, y + h],
    }


class MiniGameSessionRecorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.current_session: dict[str, Any] | None = None
        self._started_at = 0.0
        self._session_counter = 0

    def start(self, title: str, dialog_roi, target_shape: str, now: float | None = None) -> None:
        if self.current_session is not None:
            self.finish()
        os.makedirs(self.output_dir, exist_ok=True)
        self._session_counter += 1
        started = dt.datetime.now()
        stem = f"session_{started:%Y%m%d_%H%M%S}_{self._session_counter}"
        while os.path.exists(os.path.join(self.output_dir, stem + ".json")):
            self._session_counter += 1
            stem = f"session_{started:%Y%m%d_%H%M%S}_{self._session_counter}"
        initial_roi = list(map(int, dialog_roi)) if dialog_roi else None
        self._started_at = time.perf_counter() if now is None else float(now)
        self.current_session = {
            "id": stem,
            "target_title": str(title or ""),
            "start_time": started.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "end_time": None,
            "duration_sec": 0.0,
            "coordinate_system": (
                "Origin (0,0) is at dialog_roi bottom-left corner. "
                "target_x goes right (+), target_y goes up (+). "
                "Four corners are recorded in dialog_roi_corners."
            ),
            "initial_roi": initial_roi,
            "initial_roi_corners": roi_corners(initial_roi),
            "target_shape": str(target_shape or "Unknown"),
            "total_frames": 0,
            "frames": [],
        }

    def record(self, track_result, now: float | None = None) -> None:
        session = self.current_session
        if session is None:
            return
        timestamp = time.perf_counter() if now is None else float(now)
        roi = getattr(track_result, "dialog_roi", None) if track_result else None
        roi_list = list(map(int, roi)) if roi else None
        if session["initial_roi"] is None and roi_list is not None:
            session["initial_roi"] = roi_list
            session["initial_roi_corners"] = roi_corners(roi_list)
        reference_roi = roi_list or session["initial_roi"]
        corners = roi_corners(reference_roi)
        initialized = bool(track_result and getattr(track_result, "initialized", False))
        global_x = round(float(track_result.x), 2) if initialized else None
        global_y = round(float(track_result.y), 2) if initialized else None
        global_dx = global_dy = None
        if track_result and track_result.diff_x > 0 and track_result.diff_y > 0:
            global_dx = round(float(track_result.diff_x), 2)
            global_dy = round(float(track_result.diff_y), 2)

        target_x = target_y = diff_x = diff_y = None
        if reference_roi:
            origin_x, origin_y = reference_roi[0], reference_roi[1] + reference_roi[3]
            if global_x is not None:
                target_x = round(global_x - origin_x, 2)
                target_y = round(origin_y - global_y, 2)
            if global_dx is not None:
                diff_x = round(global_dx - origin_x, 2)
                diff_y = round(origin_y - global_dy, 2)
        session["frames"].append({
            "frame_idx": len(session["frames"]),
            "time_offset_sec": round(max(0.0, timestamp - self._started_at), 3),
            "dialog_roi": roi_list,
            "dialog_roi_corners": corners,
            "target_x": target_x,
            "target_y": target_y,
            "diff_x": diff_x,
            "diff_y": diff_y,
            "global_target_x": global_x,
            "global_target_y": global_y,
            "global_diff_x": global_dx,
            "global_diff_y": global_dy,
            "confidence": round(float(track_result.confidence), 3) if track_result else 0.0,
            "initialized": initialized,
        })
        session["total_frames"] = len(session["frames"])

    def finish(self) -> tuple[str, str] | None:
        session = self.current_session
        if session is None:
            return None
        session["end_time"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        if session["frames"]:
            session["duration_sec"] = session["frames"][-1]["time_offset_sec"]
        base = os.path.join(self.output_dir, session["id"])
        json_path, csv_path = base + ".json", base + ".csv"
        self._write_atomic(json_path, json.dumps(session, ensure_ascii=False, indent=2))
        self._write_atomic(csv_path, self._to_csv(session), encoding="utf-8-sig")
        self.current_session = None
        return json_path, csv_path

    @staticmethod
    def _write_atomic(path: str, content: str, encoding: str = "utf-8") -> None:
        temporary = path + ".tmp"
        with open(temporary, "w", encoding=encoding, newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)

    @staticmethod
    def _to_csv(session: dict[str, Any]) -> str:
        output = io.StringIO(newline="")
        roi = session["initial_roi"]
        initial = f"[{','.join(map(str, roi))}]" if roi else "None"
        comments = [
            f"# Session ID: {session['id']}",
            f"# Target Window: {session['target_title']}",
            f"# Start Time: {session['start_time']}",
            f"# End Time: {session['end_time']}",
            f"# Initial Mini-game Arena ROI [x,y,w,h]: {initial}",
        ]
        corners = session["initial_roi_corners"]
        if corners:
            comments.append(
                "# Initial ROI Corners: "
                + ", ".join(f"{key}={corners[name]}" for key, name in (
                    ("BL", "bottom_left"), ("TL", "top_left"),
                    ("TR", "top_right"), ("BR", "bottom_right")
                ))
            )
        comments += [
            f"# Target Shape: {session['target_shape']}",
            f"# Total Recorded Frames: {session['total_frames']}",
            "# Coordinate System: target_x/target_y are relative to dialog_roi "
            "bottom-left corner (0,0), X right (+), Y up (+).",
            "",
        ]
        output.write("\n".join(comments) + "\n")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for frame in session["frames"]:
            roi = frame["dialog_roi"] or ("",) * 4
            corners = frame["dialog_roi_corners"] or {}
            row = [frame["frame_idx"], frame["time_offset_sec"], *roi]
            for name in ("bottom_left", "top_left", "top_right", "bottom_right"):
                row.extend(corners.get(name, ("", "")))
            row.extend(
                frame[key] for key in (
                    "target_x", "target_y", "diff_x", "diff_y",
                    "global_target_x", "global_target_y", "confidence",
                )
            )
            row.append(int(frame["initialized"]))
            writer.writerow(row)
        return output.getvalue()
