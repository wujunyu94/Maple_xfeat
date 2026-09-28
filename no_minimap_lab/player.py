"""Optional screen-space nameplate anchor. No yellow-dot dependency."""
import json

import cv2
import numpy as np
from PIL import Image

from .atlas import ROOT
from .localizer import scene_mask

CALIBRATION = ROOT / "no_minimap_lab" / "calibration"


class PlayerAnchor:
    def __init__(self):
        self.template = None
        self.offset = None
        self.calibrated = False
        self.frame_size = None
        custom = CALIBRATION / "player.png"
        metadata = CALIBRATION / "player.json"
        fallback = ROOT / "assets" / "templates" / "player_nametag.png"
        if custom.exists() and metadata.exists():
            self.template = np.array(Image.open(custom).convert("RGB"))[:, :, ::-1].copy()
            data = json.loads(metadata.read_text(encoding="utf-8"))
            self.offset = np.float32(data["offset"])
            self.frame_size = data["frame_size"]
            self.calibrated = True
        elif fallback.exists():
            self.template = np.array(Image.open(fallback).convert("RGB"))[:, :, ::-1].copy()
            self.offset = np.float32([self.template.shape[1]/2, 0])

    def save(self, frame, rectangle, feet):
        x, y, w, h = rectangle
        self.template = frame[y:y+h, x:x+w].copy()
        self.offset = np.float32(feet) - [x, y]
        self.frame_size = list(frame.shape[:2])
        self.calibrated = True
        CALIBRATION.mkdir(exist_ok=True)
        Image.fromarray(cv2.cvtColor(self.template, cv2.COLOR_BGR2RGB)).save(CALIBRATION / "player.png")
        (CALIBRATION / "player.json").write_text(json.dumps(dict(
            offset=self.offset.tolist(), frame_size=self.frame_size)), encoding="utf-8")

    def detect(self, frame):
        if self.template is None:
            return None
        if self.frame_size is not None and list(frame.shape[:2]) != self.frame_size:
            return None
        h, w = self.template.shape[:2]
        if h >= frame.shape[0] or w >= frame.shape[1] or self.template.std() < 5:
            return None
        result = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        allowed = scene_mask(frame.shape)
        # Require the entire nameplate to be inside the allowed scene.
        allowed = cv2.erode(allowed, np.ones((h, w), np.uint8), anchor=(0, 0))
        result[allowed[:result.shape[0], :result.shape[1]] == 0] = -1
        _, score, _, xy = cv2.minMaxLoc(result)
        x, y = xy
        result[max(0, y-h//2):y+h//2+1, max(0, x-w//2):x+w//2+1] = -1
        runner = float(result.max())
        if score < .80 or score-runner < .06:
            return None
        return dict(screen=(np.float32(xy)+self.offset).tolist(), score=float(score),
                    calibrated=self.calibrated, bbox=[x, y, w, h])
