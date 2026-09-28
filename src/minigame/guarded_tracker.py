"""Causal checkpoint replay and one-frame flash quarantine; no evaluation I/O."""
from collections import deque
from dataclasses import replace
import cv2
import numpy as np
from .baseline_tracker import CausalShapeTracker as OriginalTracker, TrackerConfig, TrackResult


def checkpoint(engine):
    state = engine.__dict__.copy()
    for key, value in state.items():
        if isinstance(value, np.ndarray) and value.size <= 16:
            state[key] = value.copy()
        elif isinstance(value, deque):
            state[key] = deque(value, maxlen=value.maxlen)
    return state


class CausalShapeTracker:
    def __init__(self, config=None):
        self.engine = OriginalTracker(config)
        self._trusted = None
        self._last_result = None
        self._last_timestamp = None
        self._held = False
        self._active_steps = 0
        self._previous_white = 0
        self.recovery_events = []

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def reset(self):
        self.engine.reset()
        self._trusted = None
        self._last_result = None
        self._last_timestamp = None
        self._held = False
        self._active_steps = 0
        self._previous_white = 0

    def _flash(self, frame):
        if self._last_result is None or self._last_result.dialog_roi is None:
            return False
        x, y, w, h = self._last_result.dialog_roi
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] < 22) & (hsv[:, :, 2] > 235)).astype(np.uint8)
        count = int(mask.sum())
        previous = self._previous_white
        self._previous_white = count
        if self._active_steps < 20:
            return False
        scale = max(1.0, frame.shape[1] / 1920, frame.shape[0] / 1080)
        radius = self.engine.target_radius * scale
        area = np.pi * radius ** 2
        if count - previous <= max(700, area * 0.65):
            return False
        # A closing dialog / success banner is not a sensor impulse. Require
        # one compact new blob in the still-visible local rock arena.
        rock = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] >= 20)
        if rock.mean() < 0.70:
            return False
        n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        p = np.array([self._last_result.x - x, self._last_result.y - y])
        for k in range(1, n):
            a = stats[k, 4]
            bw, bh = stats[k, 2:4]
            distance = np.linalg.norm(centers[k] - p)
            if (area * 0.65 < a < area * 2.0 and a / (bw * bh) > 0.65 and
                    radius * 1.5 < distance < radius * 3.5):
                return True
        return False

    def update(self, frame, timestamp=None):
        if timestamp is not None and self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError('Capture timestamps must increase')
        old = self._trusted
        active = old is not None and old.get('acquired') and old.get('pre_game_finished')
        self._active_steps = self._active_steps + 1 if active else 0
        flash = self._flash(frame) if active else False
        if flash and not self._held and self._last_result is not None:
            self._held = True
            self._last_timestamp = timestamp
            self.recovery_events.append(dict(time=timestamp, kind='one_frame_flash_quarantine'))
            return replace(self._last_result, confidence=min(0.3, self._last_result.confidence))
        self._held = False
        result = self.engine.update(frame, timestamp)
        if active and result.initialized and self.engine.dialog_roi == old.get('dialog_roi'):
            displacement = float(np.linalg.norm(self.engine.position - old['position']))
            bound = max(65.0, float(old['target_radius']) * 1.4)
            if displacement > bound:
                retry = OriginalTracker(self.engine.config)
                retry.__dict__.update(old)
                retry.__dict__.update(checkpoint(retry))
                candidate = retry.update(frame, timestamp)
                if (candidate.initialized and retry.dialog_roi == old['dialog_roi'] and
                        np.linalg.norm(retry.position - old['position']) <= bound):
                    self.recovery_events.append(dict(time=timestamp, kind='innovation_replay', jump=displacement))
                    self.engine = retry
                    result = candidate
        self._trusted = checkpoint(self.engine)
        self._last_result = result
        self._last_timestamp = timestamp
        return result
