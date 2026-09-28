"""Background map anchoring with timestamped reprojection onto the current frame."""
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

from .localizer import Localizer, Pose, scene_mask, spread_points


class AsyncLocalizer(Localizer):
    def __init__(self, *args, **kwargs):
        self.generation = 0
        self.pending = None
        super().__init__(*args, **kwargs)
        # CPU inference regularly needs 450–700ms. Two consecutive jobs can
        # exceed the old 800ms source-anchor lifetime even on a still scene.
        # Keep a bounded CPU budget; old results still require current-frame
        # forward/backward flow validation before they can establish a pose.
        self.max_result_age = 2.0 if self.backend == 'xfeat-cpu' else .65
        if self.backend == 'xfeat-cpu' and 'max_coast' not in kwargs:
            self.max_coast = 2.5
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="world-anchor")
        self.worker_ms = 0.0

    def reset(self):
        super().reset()
        self.generation += 1
        self.last_quality = (0, 0, None)
        self.track_residual = None
        self.reanchor_candidate = None

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _job(self, gray, mask, timestamp, generation, prior_camera=None):
        start = time.perf_counter()
        found, reason = self._global(gray, mask, prior_camera=prior_camera) if self.learned is not None else self._global(gray, mask)
        return gray, timestamp, generation, found, reason, (time.perf_counter()-start)*1000

    def update(self, frame, timestamp=None, exclude=None, force=False):
        start = time.perf_counter()
        now = start if timestamp is None else float(timestamp)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (self.previous is not None and self.previous.shape != gray.shape) or (
                self.last_timestamp is not None and (now < self.last_timestamp or now-self.last_timestamp > 1)):
            self.reset()
        mask = scene_mask(gray.shape, exclude)
        flowed = self.camera is not None and self._flow(gray, mask)
        pose = Pose(reason="Waiting for background world anchor")
        if self.pending is not None and self.pending.done():
            source, stamp, generation, found, reason, self.worker_ms = self.pending.result()
            self.pending = None
            if generation == self.generation and 0 <= now-stamp <= self.max_result_age:
                pose.reason = reason
                if found is not None and found[0] is not None:
                    cam, count, runner, points, error = found
                    # Never apply old image coordinates directly to a new frame.
                    # Validate and reproject the anchor's landmarks by optical flow.
                    saved = (self.camera, self.previous, self.track_points, self.track_residual)
                    self.camera, self.previous, self.track_points = cam.copy(), source, spread_points(points)
                    rebased = self._flow(gray, mask)
                    if rebased and flowed and saved[0] is not None:
                        innovation = self.camera-saved[0]
                        # A healthy tracked scene should agree with a fresh map
                        # anchor. Require another independently captured keyframe
                        # before accepting a large, contradictory map correction.
                        if np.linalg.norm(innovation) > 8:
                            previous = self.reanchor_candidate
                            confirmed = previous is not None and stamp > previous[1] and np.linalg.norm(innovation-previous[0]) < 6
                            if not confirmed:
                                self.reanchor_candidate = (innovation, stamp)
                                rebased = False
                        else:
                            self.reanchor_candidate = None
                    if rebased:
                        self.last_anchor = stamp
                        self.last_quality = (count, runner, error)
                        pose = Pose("LOCKED", tuple(map(float, self.camera)), count, runner,
                                    error, now-stamp, reason="Background anchor reprojected onto this frame")
                    else:
                        self.camera, self.previous, self.track_points, self.track_residual = saved
                        pose.reason = "Anchor rejected: reprojection failed or camera correction awaits confirmation"
                elif found is not None:
                    pose.status = "AMBIGUOUS"
                    pose.inliers, pose.runner_up = found[1:]
                    self.camera = self.track_points = None
            elif generation == self.generation:
                pose.reason = f"Background anchor expired: {now-stamp:.2f}s > {self.max_result_age:.2f}s"
        age = now-self.last_anchor
        if pose.camera is None and pose.status != "AMBIGUOUS":
            if flowed and self.camera is not None and age <= self.max_coast:
                count, runner, error = self.last_quality
                pose = Pose("COASTING", tuple(map(float, self.camera)), len(self.track_points),
                            runner, error, age, reason="Flow tracking; independent world refresh pending")
            else:
                self.camera = self.track_points = None
        if self.pending is None and (force or now-self.last_attempt >= (.2 if self.camera is None else self.anchor_interval)):
            self.last_attempt = now
            prior = self.camera.copy() if self.camera is not None and age <= self.max_coast else None
            self.pending = self.pool.submit(self._job, gray.copy(), mask, now, self.generation, prior)
        self.previous = gray
        self.last_timestamp = now
        pose.elapsed_ms = (time.perf_counter()-start)*1000
        pose.anchor_ms = self.worker_ms
        pose.track_residual = self.track_residual
        return pose
