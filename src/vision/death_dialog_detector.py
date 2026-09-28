"""Conservative visual detection of the in-game death/respawn dialog."""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class DeathDialogDetector:
    """Require the respawn instruction itself, not merely an empty HP bar."""

    def __init__(self) -> None:
        self._ocr = None
        self._ocr_lock = threading.Lock()

    def _get_ocr(self):
        if self._ocr is None:
            with self._ocr_lock:
                if self._ocr is None:
                    from rapidocr_onnxruntime import RapidOCR
                    self._ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        return self._ocr

    def _recognize(self, crop: np.ndarray) -> list[tuple[str, float]]:
        result, _ = self._get_ocr()(crop)
        return [(str(item[1]), float(item[2])) for item in (result or [])
                if len(item) >= 3]

    @staticmethod
    def _center_crop(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.ndim != 3:
            return None
        h, w = frame.shape[:2]
        if h < 300 or w < 500:
            return None
        # The dialog is horizontally centred and above the character, but its
        # pixel size does not necessarily grow with a 1080p/4K client window.
        half_width = min(int(w * 0.28), 360)
        x0, x1 = w // 2 - half_width, w // 2 + half_width
        y0, y1 = int(h * 0.08), min(h, int(h * 0.50))
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _has_blue_modal(crop: np.ndarray) -> bool:
        # The modal uses a broad *muted* blue panel.  Bright blue sky is very
        # common in maps; a generic blue HSV gate would run OCR on every frame
        # and waste roughly half a CPU second per check in those maps.
        mask = cv2.inRange(crop, (125, 75, 35), (225, 195, 145))
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        return any(
            stats[index, cv2.CC_STAT_AREA] >= 5000
            and stats[index, cv2.CC_STAT_WIDTH] >= 140
            and stats[index, cv2.CC_STAT_HEIGHT] >= 45
            for index in range(1, count)
        )

    def detect(self, frame: np.ndarray) -> bool:
        crop = self._center_crop(frame)
        if crop is None or not self._has_blue_modal(crop):
            return False
        try:
            lines = self._recognize(crop)
        except Exception:
            return False
        # Require the death-specific sentence; a generic blue Confirm dialog
        # or the unrelated minigame popup must never trigger a respawn press.
        text = "".join(line.replace(" ", "") for line, confidence in lines
                       if confidence >= 0.55)
        return "复活" in text and ("确定键" in text or "安全的村落" in text)

    def find_confirm_button(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return the OCR-located Confirm button centre in client coordinates.

        The location is used only while the death modal is still present, so
        an unrelated Confirm button elsewhere cannot be clicked by accident.
        """
        crop = self._center_crop(frame)
        if crop is None or not self._has_blue_modal(crop):
            return None
        try:
            items, _ = self._get_ocr()(crop)
        except Exception:
            return None
        items = [item for item in (items or []) if len(item) >= 3]
        text = "".join(str(item[1]).replace(" ", "") for item in items
                       if float(item[2]) >= 0.55)
        if "复活" not in text or ("确定键" not in text and "安全的村落" not in text):
            return None
        h, w = frame.shape[:2]
        half_width = min(int(w * 0.28), 360)
        x0, y0 = w // 2 - half_width, int(h * 0.08)
        for box, label, confidence in items:
            if str(label).strip() not in ("确定", "确认") or float(confidence) < 0.65:
                continue
            if len(box) < 4:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            if max(xs) - min(xs) > 85 or max(ys) - min(ys) > 45:
                continue
            cx = int(round(x0 + (min(xs) + max(xs)) / 2.0))
            cy = int(round(y0 + (min(ys) + max(ys)) / 2.0))
            if abs(cx - w / 2.0) <= min(180, w * 0.18):
                return cx, cy
        return None
