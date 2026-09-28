"""Timed test-video compositor for exercising the live mini-game pipeline."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np


class MiniGameVideoTest:
    """Inject one timed video into frames returned by ScreenCapture."""

    def __init__(self, status_callback=None):
        self.status_callback = status_callback or (lambda _message: None)
        self._lock = threading.Lock()
        self._capture = None
        self._session_active = False
        self._playback_done = False
        self._video_path = ""
        self._delay_sec = 0.0
        self._session_started_at = 0.0
        self._video_started_at = 0.0
        self._video_fps = 30.0
        self._video_index = -1
        self._video_frame: Optional[np.ndarray] = None
        self._last_capture_seq = -1
        self._last_composited_frame: Optional[np.ndarray] = None

    @property
    def session_active(self) -> bool:
        with self._lock:
            return self._session_active

    @property
    def video_visible(self) -> bool:
        """Whether the delayed inset is currently being composited."""
        with self._lock:
            return bool(
                self._session_active
                and self._capture is not None
                and not self._playback_done
                and self._session_started_at > 0.0
                and time.perf_counter() - self._session_started_at >= self._delay_sec
            )

    def start_f6_session(self, video_path: str, delay_sec: float) -> bool:
        """Arm a new timed overlay when F6 starts; repeated calls keep it alive."""
        with self._lock:
            if self._session_active:
                return True
            path = os.path.abspath(os.path.expanduser(str(video_path or "")))
            if not os.path.isfile(path):
                self.status_callback(f"⚠️ [小游戏测试] 视频文件不存在：{path}")
                return False
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                capture.release()
                self.status_callback(f"⚠️ [小游戏测试] 无法打开视频：{path}")
                return False
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self._capture = capture
            self._session_active = True
            self._playback_done = False
            self._video_path = path
            self._delay_sec = max(0.0, float(delay_sec))
            self._session_started_at = time.perf_counter()
            self._video_started_at = 0.0
            self._video_fps = fps if 1.0 <= fps <= 240.0 else 30.0
            self._video_index = -1
            self._video_frame = None
            self._last_capture_seq = -1
            self._last_composited_frame = None

        self.status_callback(
            f"🎞️ [小游戏测试] 已排程：{os.path.basename(path)}，"
            f"F6启动 {self._delay_sec:g} 秒后叠加"
        )
        return True

    def stop_f6_session(self, reason: str = "F6停止") -> None:
        with self._lock:
            capture = self._capture
            was_active = self._session_active
            self._capture = None
            self._session_active = False
            self._playback_done = True
            self._video_frame = None
            self._last_composited_frame = None
        if capture is not None:
            capture.release()
        if was_active:
            self.status_callback(f"⏹️ [小游戏测试] 回放已停止（{reason}）。")

    def composite(self, frame: np.ndarray, capture_seq: int) -> np.ndarray:
        """Composite the current source-video frame as a centered, aspect-fit inset."""
        if frame is None or frame.size == 0:
            return frame

        with self._lock:
            if not self._session_active or self._capture is None:
                return frame
            if capture_seq == self._last_capture_seq and self._last_composited_frame is not None:
                return self._last_composited_frame.copy()

            now = time.perf_counter()
            elapsed = now - self._session_started_at
            if elapsed < self._delay_sec:
                self._last_capture_seq = capture_seq
                self._last_composited_frame = frame.copy()
                return frame

            if self._playback_done:
                self._last_capture_seq = capture_seq
                self._last_composited_frame = frame.copy()
                return frame

            if self._video_started_at <= 0.0:
                self._video_started_at = now
            target_index = max(0, int((now - self._video_started_at) * self._video_fps))
            if target_index > self._video_index:
                if target_index - self._video_index > 8:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)
                    self._video_index = target_index - 1
                while self._video_index < target_index:
                    ok, next_frame = self._capture.read()
                    if not ok:
                        self._finish_playback_locked()
                        break
                    self._video_index += 1
                    self._video_frame = next_frame

            if self._playback_done or self._video_frame is None:
                result = frame
            else:
                result = self._draw_centered_inset(frame, self._video_frame)
            self._last_capture_seq = capture_seq
            self._last_composited_frame = result.copy()
            return result

    def _finish_playback_locked(self) -> None:
        capture = self._capture
        self._capture = None
        self._playback_done = True
        self._video_frame = None
        if capture is not None:
            capture.release()
        self.status_callback("✅ [小游戏测试] 视频已播放完毕；F6测试会话保持到手动停止。")

    @staticmethod
    def _draw_centered_inset(frame: np.ndarray, video_frame: np.ndarray) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        video_h, video_w = video_frame.shape[:2]
        if frame_w < 4 or frame_h < 4 or video_w < 1 or video_h < 1:
            return frame

        max_w = max(1, int(frame_w * 0.84))
        max_h = max(1, int(frame_h * 0.84))
        scale = min(max_w / video_w, max_h / video_h)
        out_w = max(1, min(max_w, int(round(video_w * scale))))
        out_h = max(1, min(max_h, int(round(video_h * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(video_frame, (out_w, out_h), interpolation=interpolation)
        x = (frame_w - out_w) // 2
        y = (frame_h - out_h) // 2
        frame[y : y + out_h, x : x + out_w] = resized
        cv2.rectangle(frame, (x, y), (x + out_w - 1, y + out_h - 1), (0, 220, 255), 2)
        return frame
