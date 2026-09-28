"""Arena-scale adaptation for large targets, outside the guarded tracker.

Acquisition still uses the original full capture. Only large irregular
shapes in small arenas and oversized circles are resampled. Other shapes
retain their original processing scale.
"""
from __future__ import annotations

from dataclasses import replace
import math
import cv2
import numpy as np

from .guarded_tracker import CausalShapeTracker as GuardedTracker
from .baseline_tracker import TrackerConfig, TrackResult


class CausalShapeTracker:
    CANONICAL_HEIGHT = 704
    CANVAS_SIZE = (1920, 1080)
    STALE_AFTER_SEC = 0.15

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._tracker = GuardedTracker(self.config)
        self._source_roi = None
        self._normalization_checked = False
        self._normalization_height = self.CANONICAL_HEIGHT
        self._canvas = None
        self._placement = None
        self._input_shape = None
        self._last_timestamp = None
        self._last_unique_time = None
        self._previous_crop = None
        self._last_result = None
        self._logical_time = 0.0
        self.duplicate_frames = 0

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    @property
    def dialog_roi(self):
        return self._last_result.dialog_roi if self._last_result is not None else None

    @property
    def normalized(self):
        return self._source_roi is not None

    def reset(self):
        self.__init__(self.config)

    def _needs_normalization(self, frame, result):
        # Use the acquired white footprint, not the input resolution or video ID.
        engine = self._tracker.engine
        input_scale = max(1.0, frame.shape[1] / 1920, frame.shape[0] / 1080)
        if engine.is_circle:
            observed_diameter = engine.rel_dim * result.dialog_roi[3] / input_scale
            kernel_diameter = 2 * getattr(engine, 'target_radius', 48.0)
            # The legacy circle kernel is clamped to radius 46--49. Only a
            # substantial mismatch warrants resampling: small circles retain
            # their existing path, as upsampling them regressed old samples.
            if observed_diameter > 1.3 * kernel_diameter:
                self._normalization_height = round(96.0 / engine.rel_dim)
                return True
            return False
        self._normalization_height = self.CANONICAL_HEIGHT
        return (not engine.is_circle and engine.circ < 0.52
                and engine.rel_dim >= 0.26 and engine.rel_area >= 0.02
                and result.dialog_roi[3] / input_scale < 650)

    def _make_canvas(self, frame):
        x, y, w, h = self._source_roi
        ox, oy, nw, nh = self._placement
        self._canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(
            frame[y:y + h, x:x + w], (nw, nh), interpolation=cv2.INTER_CUBIC
        )
        return self._canvas

    def _map_result(self, result):
        x, y, w, h = self._source_roi
        ox, oy, nw, nh = self._placement
        # A zero diff coordinate means "no candidate", not canvas origin.
        has_diff = result.diff_x != 0.0 or result.diff_y != 0.0
        return replace(
            result, x=x + (result.x - ox) * w / nw,
            y=y + (result.y - oy) * h / nh,
            diff_x=x + (result.diff_x - ox) * w / nw if has_diff else 0.0,
            diff_y=y + (result.diff_y - oy) * h / nh if has_diff else 0.0,
            dialog_roi=self._source_roi if result.dialog_roi is not None else None,
        )

    def update(self, frame: np.ndarray, timestamp: float | None = None) -> TrackResult:
        if timestamp is not None:
            if not math.isfinite(timestamp):
                raise ValueError('Capture timestamp must be finite')
            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                raise ValueError('Capture timestamps must increase')
        if self._input_shape is not None and frame.shape != self._input_shape:
            # Coordinates and image history are invalid after a capture resize.
            self.reset()
        self._input_shape = frame.shape
        self._last_timestamp = timestamp
        now = float(timestamp) if timestamp is not None else self._logical_time
        self._logical_time += 1.0 / 30.0

        roi = self.dialog_roi
        if (roi is not None and self._last_result.initialized
                and getattr(self._tracker.engine, '_close_confirm_count', 0) == 0):
            x, y, w, h = roi
            crop = frame[y:y + h, x:x + w]
            if self._previous_crop is not None and np.array_equal(crop, self._previous_crop):
                # Do not integrate velocity or overwrite the temporal reference
                # for a repeated video frame. The next observation uses its real dt.
                self.duplicate_frames += 1
                stale = now - self._last_unique_time > self.STALE_AFTER_SEC
                return replace(self._last_result,
                               confidence=min(self._last_result.confidence, 0.3)
                               if stale else self._last_result.confidence)

        if self.normalized:
            result = self._map_result(self._tracker.update(self._make_canvas(frame), timestamp))
        else:
            result = self._tracker.update(frame, timestamp)
            if (result.initialized and not self._normalization_checked
                    and self._needs_normalization(frame, result)):
                self._normalization_checked = True
                self._source_roi = result.dialog_roi
                x, y, w, h = self._source_roi
                nh = self._normalization_height
                nw = round(w * nh / h)
                cw, ch = self.CANVAS_SIZE
                self._placement = ((cw - nw) // 2, (ch - nh) // 2, nw, nh)
                self._canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
                candidate = GuardedTracker(self.config)
                normalized_result = candidate.update(self._make_canvas(frame), timestamp)
                if normalized_result.initialized:
                    self._tracker = candidate
                    result = self._map_result(normalized_result)
                else:
                    # Never publish an uninitialized center as a new target.
                    self._source_roi = self._placement = self._canvas = None

        self._last_result = result
        self._last_unique_time = now
        if result.initialized and result.dialog_roi is not None:
            x, y, w, h = result.dialog_roi
            # Own the pixels; callers may reuse their capture buffer.
            self._previous_crop = frame[y:y + h, x:x + w].copy()
        else:
            self._normalization_checked = False
            self._previous_crop = None
            if self.normalized:
                self._source_roi = self._placement = self._canvas = None
                self._tracker = GuardedTracker(self.config)
        return result


__all__ = ['CausalShapeTracker', 'TrackerConfig', 'TrackResult']
