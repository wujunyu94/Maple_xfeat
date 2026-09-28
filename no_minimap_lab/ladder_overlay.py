"""All visible IMG ladder/rope segments, with optional local pixel evidence."""
import cv2
import numpy as np

from .localizer import scene_mask
from .ignore_regions import rectangles


def project_ladders(atlas, camera, shape, scale=1.0):
    if camera is None:
        return []
    h, w = shape[:2]
    allowed = scene_mask(shape)
    output = []
    for node in atlas.meta.get("ladder_nodes", []):
        x = (node["x"]-camera[0])*scale
        y1, y2 = sorted(((node["y1"]-camera[1])*scale, (node["y2"]-camera[1])*scale))
        if x < 0 or x >= w or y2 < 0 or y1 >= h:
            continue
        a, b = max(0, int(np.ceil(y1))), min(h-1, int(np.floor(y2)))
        if b <= a:
            continue
        col = allowed[a:b+1, min(w-1, round(x))] > 0
        boundaries = np.flatnonzero(np.diff(np.r_[False, col, False].astype(int)))
        segments = [[float(x), float(a+s), float(a+e-1)] for s, e in zip(boundaries[::2], boundaries[1::2]) if e-s >= 8]
        if not segments:
            continue
        output.append(dict(id=node["id"], kind=node["kind"], world=[node["x"], node["y1"], node["y2"]],
                           screen_line=[float(x), float(y1), float(y2)], segments=segments,
                           upper_exit=node["upper_exit"], status="projected", score=None,
                           correction_x_px=0.0, verification_age=None))
    return output


class LadderOverlay:
    def __init__(self, atlas, scale=1.0):
        self.atlas = atlas
        self.scale = scale
        self.reset()

    def reset(self):
        self.last_check = -float("inf")
        self.camera = None
        self.evidence = {}

    def verify(self, frame, entry):
        h, w = frame.shape[:2]
        expected_x, y1, y2 = max(entry["segments"], key=lambda s: s[2]-s[1])
        if y2-y1 < 44*self.scale:
            return None
        results = []
        # Two spatially separated portions must agree on the horizontal center.
        for fraction in (.28, .72):
            sy = y1+(y2-y1)*fraction
            half_w, half_h = 12, 10
            wx = entry["world"][0]
            wy = self.camera[1]+sy/self.scale
            ax, ay = np.rint([wx-self.atlas.origin[0]-half_w,
                             wy-self.atlas.origin[1]-half_h]).astype(int)
            if ax < 0 or ay < 0:
                continue
            rgba = self.atlas.rgba[ay:ay+2*half_h, ax:ax+2*half_w]
            if rgba.shape[:2] != (2*half_h, 2*half_w):
                continue
            template = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            mask = np.uint8(rgba[:, :, 3] > 240)*255
            tw, th = max(5, round(2*half_w*self.scale)), max(8, round(2*half_h*self.scale))
            template = cv2.resize(template, (tw, th))
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(mask) < tw*th*.20 or template[mask>0].std() < 12:
                continue
            margin_x, margin_y = max(4, round(8*self.scale)), max(3, round(4*self.scale))
            left = round(expected_x-half_w*self.scale)
            top = round(sy-half_h*self.scale)
            rx, ry = left-margin_x, top-margin_y
            if rx < 0 or ry < 0 or rx+tw+2*margin_x > w or ry+th+2*margin_y > h:
                continue
            if any(rx < ex+ew and rx+tw+2*margin_x > ex and
                   ry < ey+eh and ry+th+2*margin_y > ey
                   for ex,ey,ew,eh in rectangles(frame.shape)):
                continue
            roi = frame[ry:ry+th+2*margin_y, rx:rx+tw+2*margin_x]
            scores = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
            _, best, _, (mx, my) = cv2.minMaxLoc(scores)
            rivals = scores.max(axis=0)
            rivals[max(0, mx-3):mx+4] = -1
            if best < .85 or best-float(rivals.max()) < .025:
                continue
            dx = float(rx+mx-left)
            if abs(dx) <= 6*self.scale:
                results.append((dx, float(best)))
        if len(results) != 2 or abs(results[0][0]-results[1][0]) > 2*self.scale:
            return None
        return dict(dx=float(np.mean([r[0] for r in results])), score=min(r[1] for r in results))

    def update(self, frame, pose, now):
        if pose.camera is None:
            self.reset()
            return []
        if self.camera is not None and np.linalg.norm(np.array(pose.camera)-self.camera) > 100:
            self.reset()
        self.camera = np.array(pose.camera)
        entries = project_ladders(self.atlas, pose.camera, frame.shape, self.scale)
        if now-self.last_check >= .15:
            self.evidence = {}
            for entry in entries:
                result = self.verify(frame, entry)
                if result:
                    self.evidence[entry["id"]] = dict(**result, timestamp=now)
            self.last_check = now
        for entry in entries:
            evidence = self.evidence.get(entry["id"])
            if evidence and 0 <= now-evidence["timestamp"] <= .2:
                entry.update(status="verified" if now == evidence["timestamp"] else "recently_verified",
                             score=evidence["score"], correction_x_px=evidence["dx"],
                             verification_age=now-evidence["timestamp"])
        return entries
