"""Static landmark localization plus short, map-anchored optical-flow tracking."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time

import cv2
import numpy as np


@dataclass
class Pose:
    status: str = "LOST"
    camera: tuple | None = None
    inliers: int = 0
    runner_up: int = 0
    residual: float | None = None
    anchor_age: float | None = None
    elapsed_ms: float = 0
    reason: str = "No static landmark match"
    anchor_ms: float = 0
    track_residual: float | None = None

    def to_dict(self):
        return asdict(self)


def scene_mask(shape, exclude=None):
    """Configured screen exclusions; no compulsory minimap rectangle."""
    from .ignore_regions import rectangles
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    for x, y, rw, rh in rectangles(shape)+list(exclude or []):
        mask[max(0, y):min(h, y+rh), max(0, x):min(w, x+rw)] = 0
    return mask


class Localizer:
    def __init__(self, atlas, screen_scale=1.0, anchor_interval=.35, max_coast=.8, backend="sift-cpu"):
        if not .25 <= screen_scale <= 4:
            raise ValueError("screen_scale must be between 0.25 and 4")
        self.atlas = atlas
        self.screen_scale = screen_scale
        self.work_scale = .5
        self.anchor_interval = anchor_interval
        self.max_coast = max_coast
        self.backend = backend
        self.learned = None
        self.feature_detector = cv2.SIFT_create(nfeatures=18000, contrastThreshold=.025, edgeThreshold=12)
        gray = cv2.cvtColor(atlas.bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=self.work_scale, fy=self.work_scale)
        # Erosion keeps synthetic transparent edges out of landmark descriptors.
        mask = cv2.erode(atlas.mask, np.ones((11, 11), np.uint8))
        mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        if backend.startswith("xfeat-"):
            from .xfeat_backend import XFeatIndex
            self.learned = XFeatIndex(small, mask, "cuda" if backend == "xfeat-cuda" else "cpu")
            self.world_points = self.learned.points/self.work_scale+atlas.origin
            self.feature_count = len(self.world_points)
            if self.feature_count < 16:
                raise ValueError("Too few learned map features")
            self.reset()
            return
        if backend != "sift-cpu":
            raise ValueError(f"Unknown feature backend: {backend}")
        keys, desc = self.feature_detector.detectAndCompute(small, mask)
        if desc is None or len(keys) < 16:
            raise ValueError("Too few visible static landmarks in the rendered map")
        self.world_points = np.float32([k.pt for k in keys]) / self.work_scale + atlas.origin
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        self.matcher.add([desc])
        self.matcher.train()
        self.feature_count = len(keys)
        self.reset()

    def reset(self):
        self.camera = None
        self.previous = None
        self.track_points = None
        self.last_anchor = -float("inf")
        self.last_attempt = -float("inf")
        self.last_timestamp = None
        self.match_points = np.empty((0, 2), np.float32)

    def _global(self, gray, mask, prior_camera=None):
        # Normalize client scale first; coordinates below remain world-pixel units.
        f = self.work_scale / self.screen_scale
        small = cv2.resize(gray, None, fx=f, fy=f)
        mask_small = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.learned is not None:
            points, queries, targets = self.learned.match(small, mask_small)
            self.last_candidates = []
            result = self._vote(points/self.work_scale, queries, targets, gray, mask)
            if (result[0] is None or result[0][0] is None) and prior_camera is not None:
                # A timestamped, map-anchored flow prior proposes a search area;
                # actual opaque world pixels must still independently confirm it.
                proposals = [(0,np.asarray(prior_camera),None,0)]
                proposals += [(n,np.asarray(c),None,e) for n,c,e in self.last_candidates[:3]]
                result = self.learned.refine(gray,mask,proposals,self.atlas,self.screen_scale)
            return result
        keys, desc = self.feature_detector.detectAndCompute(small, mask_small)
        if desc is None or len(keys) < 12:
            return None, "Too few unmasked scene features"
        frame_points = np.float32([k.pt for k in keys]) / self.work_scale
        pairs = self.matcher.knnMatch(desc, k=min(8, self.feature_count))
        queries, targets = [], []
        ceiling, floor = 250, 100
        for neighbors in pairs:
            # Retain repeated-tile alternatives: a strict nearest-neighbor ratio
            # discards exactly the platform/ladder features this experiment needs.
            if not neighbors or neighbors[0].distance > ceiling:
                continue
            limit = min(ceiling, max(floor, neighbors[0].distance * 1.25))
            for m in neighbors:
                if m.distance <= limit:
                    queries.append(m.queryIdx)
                    targets.append(m.trainIdx)
        return self._vote(frame_points, np.array(queries), targets)

    def _vote(self, frame_points, queries, targets, gray=None, mask=None):
        if len(queries) < 12:
            return None, "No matching static landmarks"
        queries = np.array(queries)
        offsets = self.world_points[targets] - frame_points[queries]
        hypotheses = []
        # Two overlapping grids reduce sensitivity to translation-bin boundaries.
        for shift in (0, 8):
            bins = np.floor((offsets + shift) / 16).astype(np.int32)
            _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
            for index in np.argsort(counts)[-(64 if self.learned is not None else 16):]:
                center = np.median(offsets[inverse == index], axis=0)
                for _ in range(2):
                    good = np.linalg.norm(offsets-center, axis=1) < 6
                    if not good.any():
                        break
                    center = np.median(offsets[good], axis=0)
                good = np.linalg.norm(offsets-center, axis=1) < 5
                ids = np.unique(queries[good])
                # Multiple orientations at one SIFT keypoint are not independent evidence.
                if len(ids):
                    _, distinct = np.unique(np.round(frame_points[ids]/6), axis=0, return_index=True)
                    ids = ids[distinct]
                # Learned descriptors see a different background around atlas
                # transparency. Sparse proposals are usable because they must
                # still pass the independent >=100-pixel verification below.
                if len(ids) < (4 if self.learned is not None else 8):
                    continue
                points = frame_points[ids]
                spread = np.ptp(points, axis=0)
                cells = len(np.unique(np.floor(points/80), axis=0))
                if (max(spread) < (40 if self.learned is not None else 120) or
                    min(spread) < (0 if self.learned is not None else 35) or
                    cells < (2 if self.learned is not None else 4)):
                    continue
                if any(np.linalg.norm(center-h[1]) < 20 for h in hypotheses):
                    continue
                error = float(np.median(np.linalg.norm(offsets[good]-center, axis=1)))
                hypotheses.append((len(ids), center, points, error))
        if not hypotheses:
            return None, "Insufficient spatially distributed landmarks"
        hypotheses.sort(key=lambda h: h[0], reverse=True)
        self.last_candidates = [(h[0], h[1].tolist(), h[3]) for h in hypotheses[:16]]
        if self.learned is not None:
            result = self.learned.refine(gray, mask, hypotheses[:3], self.atlas, self.screen_scale)
            if (result[0] is None or result[0][0] is None) and len(hypotheses) > 3:
                # Repeated bridges may push the true translation below rank 3.
                # Spend more only after rejection; keep verification thresholds.
                result = self.learned.refine(gray, mask, hypotheses[:16], self.atlas, self.screen_scale)
            return result
        best = hypotheses[0]
        runner = hypotheses[1][0] if len(hypotheses) > 1 else 0
        if best[0] < 12 or runner >= best[0]*.80:
            return (None, best[0], runner), "Repeated scenery: ambiguous global position"
        return (best[1], best[0], runner, best[2]*self.screen_scale, best[3]), "Static world landmarks verified"

    def _flow(self, gray, mask):
        if self.previous is None or self.track_points is None or len(self.track_points) < 8:
            return False
        # A spatially distributed subset and half-size pyramids avoid spending
        # the 60 Hz budget tracking hundreds of redundant platform corners.
        points = spread_points(self.track_points)
        small_old = cv2.resize(self.previous, None, fx=.5, fy=.5)
        small_new = cv2.resize(gray, None, fx=.5, fy=.5)
        old = (points*.5).reshape(-1, 1, 2)
        new, ok, _ = cv2.calcOpticalFlowPyrLK(small_old, small_new, old, None,
                                            winSize=(25, 25), maxLevel=3)
        if new is None:
            return False
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(small_new, small_old, new, None,
                                                  winSize=(25, 25), maxLevel=3)
        if back is None:
            return False
        a, b = old[:, 0]*2, new[:, 0]*2
        valid = (ok[:, 0] > 0) & (back_ok[:, 0] > 0) & (np.linalg.norm(back[:, 0]*2-a, axis=1) < 1.2)
        h, w = gray.shape
        valid &= (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
        xy = b.astype(int)
        valid &= mask[np.clip(xy[:, 1], 0, h-1), np.clip(xy[:, 0], 0, w-1)] > 0
        if valid.sum() < 8:
            return False
        delta = np.median((b-a)[valid], axis=0)
        valid &= np.linalg.norm((b-a)-delta, axis=1) < 2.5
        if valid.sum() < 8 or valid.sum() < len(old)*.45:
            return False
        if len(np.unique(np.floor(b[valid]/80), axis=0)) < 3:
            return False
        self.camera -= np.median((b-a)[valid], axis=0) / self.screen_scale
        self.track_residual = float(np.median(np.linalg.norm((b-a)[valid]-delta, axis=1)))
        self.track_points = b[valid]
        return True

    def update(self, frame, timestamp=None, exclude=None, force=False):
        started = time.perf_counter()
        now = started if timestamp is None else float(timestamp)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (self.previous is not None and self.previous.shape != gray.shape) or (
                self.last_timestamp is not None and (now < self.last_timestamp or now-self.last_timestamp > 1)):
            self.reset()
        mask = scene_mask(gray.shape, exclude)
        flowed = self.camera is not None and self._flow(gray, mask)
        pose = Pose()
        if force or not flowed or now-self.last_attempt >= self.anchor_interval:
            self.last_attempt = now
            found, reason = self._global(gray, mask)
            pose.reason = reason
            if found is not None and found[0] is not None:
                camera, count, runner, points, error = found
                self.camera = camera.copy()
                self.track_points = points
                self.match_points = points
                self.last_anchor = now
                pose = Pose("LOCKED", tuple(map(float, camera)), count, runner, error, 0,
                            reason=reason)
            elif found is not None:
                pose.inliers, pose.runner_up = found[1:]
                # Conflicting global evidence must not be hidden by a prior pose.
                self.camera = None
                self.track_points = None
                pose.status = "AMBIGUOUS"
        age = now-self.last_anchor
        if pose.camera is None and pose.status != "AMBIGUOUS":
            if flowed and age <= self.max_coast:
                pose.status = "COASTING"
                pose.camera = tuple(map(float, self.camera))
                pose.inliers = len(self.track_points)
                pose.anchor_age = age
                pose.reason = "Short optical-flow estimate; waiting for a fresh world match"
            else:
                self.camera = None
                self.track_points = None
        self.previous = gray
        self.last_timestamp = now
        pose.elapsed_ms = (time.perf_counter()-started)*1000
        return pose


def spread_points(points, limit=160):
    points = np.asarray(points, np.float32)
    if len(points) <= limit:
        return points
    # First cover different image cells, then add points up to the budget.
    _, index = np.unique(np.floor(points/55), axis=0, return_index=True)
    index = index[np.linspace(0, len(index)-1, min(limit, len(index))).astype(int)]
    if len(index) < limit:
        rest = np.setdiff1d(np.arange(len(points)), index)
        index = np.r_[index, rest[np.linspace(0, len(rest)-1, limit-len(index)).astype(int)]]
    return points[index]
