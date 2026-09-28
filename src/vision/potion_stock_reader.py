"""Read item quantities from the two-row in-game shortcut bar.

Only a labelled, visible slot produces a number.  A missing HUD, unsupported
key, or uncertain OCR result is unknown (None), never an inferred zero.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


# Column positions are relative to the 461x75 shortcut bar at the lower right.
# Keeping the key label in the check prevents reading the neighbour's item count
# if a client uses a different shortcut layout.
_SLOTS = {
    "insert": (39, "ins", False),
    "home": (74, "hm", False),
    "pageup": (109, "pup", False),
    "delete": (39, "del", True),
    "end": (74, "end", True),
    "pagedown": (109, "pdn", True),
}
_KEY_ALIASES = {"prior": "pageup", "next": "pagedown", "pgup": "pageup",
                "pgdn": "pagedown", "ins": "insert", "del": "delete"}
_LABEL_KEYS = {
    "ins": "insert", "insert": "insert", "hm": "home", "home": "home",
    "pup": "pageup", "pgup": "pageup", "del": "delete", "delete": "delete",
    "end": "end", "pdn": "pagedown", "pgdn": "pagedown",
    "shift": "shift", "ctrl": "ctrl", "control": "ctrl",
}
_LABEL_KEYS.update({key: key for key in "abcdefghijklmnopqrstuvwxyz0123456789"})
_KEY_ALIASES.update({"control": "ctrl", "return": "enter"})


def normalize_stock_key(key: str) -> str:
    key = str(key or "").strip().lower().replace(" ", "")
    return _KEY_ALIASES.get(key, key)


def supports_stock_key(key: str) -> bool:
    return normalize_stock_key(key) in set(_LABEL_KEYS.values())


@dataclass(frozen=True)
class PotionStockReading:
    timestamp: float
    hp_key: str
    mp_key: str
    hp_count: Optional[int] = None
    mp_count: Optional[int] = None
    layout_valid: bool = False


class PotionStockReader:
    """Use the dark row divider to align tiny quantity glyphs before OCR."""

    def __init__(self) -> None:
        self._ocr = None
        self._ocr_lock = threading.Lock()
        self._slot_cache: dict[str, tuple[int, int]] = {}

    def _get_ocr(self):
        if self._ocr is None:
            with self._ocr_lock:
                if self._ocr is None:
                    from rapidocr_onnxruntime import RapidOCR
                    self._ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        return self._ocr

    def _recognize(self, crop: np.ndarray, scale: int = 8) -> tuple[str, float]:
        if crop.size == 0:
            return "", 0.0
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=(cv2.INTER_CUBIC if scale > 8
                                             else cv2.INTER_NEAREST))
        # The recognizer's angle/classification preprocessor materially helps
        # the single tiny "0" glyph in this HUD; omitting it reduced confidence
        # on the same live frame from 0.83 to 0.50.
        result, _ = self._get_ocr()(enlarged, use_det=False)
        if not result or not result[0] or len(result[0]) < 2:
            return "", 0.0
        try:
            return str(result[0][0]), float(result[0][1])
        except (TypeError, ValueError):
            return "", 0.0

    @staticmethod
    def _divider_y(panel: np.ndarray) -> Optional[int]:
        if panel.shape[0] < 72 or panel.shape[1] < 140:
            return None
        # Use the horizontal line across the whole bar, not the old PUp cell:
        # that cell may contain a different icon after the user moves potions.
        gray = cv2.cvtColor(panel[30:43], cv2.COLOR_BGR2GRAY)
        darkness = np.mean(gray, axis=1)
        index = int(np.argmin(darkness))
        if darkness[index] > 35.0:
            return None
        return 30 + index - 2

    @staticmethod
    def _dark_groups(values: np.ndarray, threshold: float, max_width: int) -> list[float]:
        dark = np.asarray(values) <= threshold
        groups = []
        start = None
        for i, active in enumerate(np.r_[dark, False]):
            if active and start is None:
                start = i
            elif not active and start is not None:
                if i - start <= max_width:
                    groups.append((start + i - 1) / 2.0)
                start = None
        return groups

    @classmethod
    def _extract_panel(cls, crop: np.ndarray) -> Optional[np.ndarray]:
        """Find the two horizontal rows and repeated vertical slot borders in a loose ROI."""
        if crop is None or crop.ndim != 3 or crop.shape[0] < 65 or crop.shape[1] < 145:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        rows = cls._dark_groups(np.mean(gray, axis=1), 46.0, 8)
        layouts = []
        for top in rows:
            for middle in rows:
                gap = middle - top
                if not 27.0 <= gap <= 80.0:
                    continue
                for bottom in rows:
                    second_gap = bottom - middle
                    if abs(second_gap - gap) > max(3.0, gap * 0.12):
                        continue
                    scale = (gap + second_gap) / 65.0
                    if not 0.80 <= scale <= 2.45:
                        continue
                    y1, y2 = int(round(top + 2 * scale)), int(round(bottom - scale))
                    if y2 <= y1:
                        continue
                    columns = cls._dark_groups(
                        np.mean(gray[y1:y2], axis=0),
                        42.0, max(5, int(round(6 * scale))),
                    )
                    step = 35.0 * scale
                    for first in columns:
                        origin_x = first - 36.0 * scale
                        if origin_x < -8 * scale:
                            continue
                        chain = 1
                        for index in range(1, 13):
                            target = first + index * step
                            if any(abs(col - target) <= max(2.0, 2.6 * scale)
                                   for col in columns):
                                chain += 1
                            else:
                                break
                        if chain >= 3:
                            layouts.append((chain, -abs(gap - second_gap), top,
                                            origin_x, scale))
        if not layouts:
            return None
        _chain, _gap_score, top, origin_x, scale = max(layouts)
        origin_y = top - 6.0 * scale
        pw, ph = int(round(461 * scale)), int(round(75 * scale))
        x0, y0 = int(round(origin_x)), int(round(origin_y))
        canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
        sx1, sy1 = max(0, x0), max(0, y0)
        sx2, sy2 = min(crop.shape[1], x0 + pw), min(crop.shape[0], y0 + ph)
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        canvas[sy1 - y0:sy2 - y0, sx1 - x0:sx2 - x0] = crop[sy1:sy2, sx1:sx2]
        if (pw, ph) != (461, 75):
            canvas = cv2.resize(canvas, (461, 75), interpolation=cv2.INTER_CUBIC)
        return canvas

    def _label_key(self, panel: np.ndarray, divider: int, row: int, col: int) -> tuple[str, float]:
        x = 4 + col * 35
        y = divider + 3 if row else divider - 29
        text, confidence = self._recognize(panel[y:y + 13, x:x + 30], 6)
        normalized = re.sub(r"[^a-z0-9]", "", text.lower())
        return _LABEL_KEYS.get(normalized, ""), confidence

    def _find_slot(self, panel: np.ndarray, divider: int, key: str) -> Optional[tuple[int, int]]:
        cached = self._slot_cache.get(key)
        if cached is not None:
            observed, score = self._label_key(panel, divider, *cached)
            if observed == key and score >= 0.50:
                return cached
            self._slot_cache.pop(key, None)

        # Known keys get an early candidate, but this is only a search hint:
        # the slot is accepted solely after its visible key label matches.
        hinted = _SLOTS.get(key)
        candidates = []
        if hinted is not None:
            candidates.append((int(hinted[2]), int(round((hinted[0] - 4) / 35))))
        candidates.extend((row, col) for row in (0, 1) for col in range(13)
                          if (row, col) not in candidates)
        for row, col in candidates:
            observed, score = self._label_key(panel, divider, row, col)
            if observed == key and score >= 0.50:
                self._slot_cache[key] = row, col
                return row, col
        return None

    def _read_slot(self, panel: np.ndarray, divider: int, key: str) -> Optional[int]:
        key = normalize_stock_key(key)
        slot = self._find_slot(panel, divider, key)
        if slot is None:
            return None
        row, col = slot
        x = 4 + col * 35
        count_y = divider + 23 if row else divider - 10
        count_crop = panel[count_y:count_y + 14, x:x + 26]
        # Tiny outlined HUD digits vary by glyph: the normal 8x OCR can read
        # 25 yet reject 26 in the same slot. Retry only an unaccepted count,
        # using a different 16x interpolation. Never turn OCR failure into 0.
        for scale in (8, 16):
            text, confidence = self._recognize(count_crop, scale)
            text = text.strip().replace(" ", "")
            # The cell's dark lower border can be recognized as one trailing
            # dash: the live PAGEUP=26 crop produced "26-" at both 8x and 16x.
            # Permit only a *single trailing* mark, never punctuation between
            # digits or an arbitrary OCR word.
            if re.fullmatch(r"\d{1,4}[.,-]?", text) and confidence >= 0.65:
                return int(text.rstrip(".,-"))
        return None

    def read_panel(self, panel: np.ndarray, hp_key: str, mp_key: str) -> PotionStockReading:
        now = time.perf_counter()
        hp_key = normalize_stock_key(hp_key)
        mp_key = normalize_stock_key(mp_key)
        divider = self._divider_y(panel)
        if divider is None:
            return PotionStockReading(now, hp_key, mp_key)
        hp = self._read_slot(panel, divider, hp_key)
        mp = hp if hp_key == mp_key else self._read_slot(panel, divider, mp_key)
        return PotionStockReading(now, hp_key, mp_key, hp, mp, True)

    def read(self, frame: np.ndarray, hp_key: str, mp_key: str,
             roi: Optional[dict[str, Any]] = None) -> PotionStockReading:
        now = time.perf_counter()
        hp_key = normalize_stock_key(hp_key)
        mp_key = normalize_stock_key(mp_key)
        if frame is None or frame.ndim != 3 or frame.shape[0] < 75 or frame.shape[1] < 461:
            return PotionStockReading(now, hp_key, mp_key)
        h, w = frame.shape[:2]
        best = PotionStockReading(now, hp_key, mp_key)

        def include(result: PotionStockReading) -> None:
            nonlocal best
            if result.layout_valid:
                # 血药和蓝药可能放在不同的快捷栏；每个数字都必须由
                # 该栏的可见键位标签独立核验，不能凭旧ROI推断。
                best = PotionStockReading(
                    now, hp_key, mp_key,
                    best.hp_count if best.hp_count is not None else result.hp_count,
                    best.mp_count if best.mp_count is not None else result.mp_count,
                    True,
                )

        if roi is not None:
            from src.vision.status_bar_reader import StatusBarReader
            box = StatusBarReader.resolve_roi(frame.shape, roi)
            if box is not None and box[2] >= 145 and box[3] >= 65:
                x, y, width, height = box
                panel = self._extract_panel(frame[y:y + height, x:x + width])
                if panel is not None:
                    include(self.read_panel(panel, hp_key, mp_key))
                    if best.hp_count is not None and best.mp_count is not None:
                        return best
        # Some clients keep the HUD at native pixels; others scale it with the
        # viewport.  A previously saved ROI may point at the status bar after
        # resolution changes, so try both automatic layouts as a fallback.
        scales = [1.0]
        scaled = min(w / 1280.0, h / 720.0)
        if scaled >= 1.15 and abs(scaled - 1.0) > 0.05:
            scales.append(scaled)
        for scale in scales:
            pw, ph = int(round(461 * scale)), int(round(75 * scale))
            if pw > w or ph > h:
                continue
            panel = frame[h - ph:h, w - pw:w]
            if scale != 1.0:
                panel = cv2.resize(panel, (461, 75), interpolation=cv2.INTER_AREA)
            include(self.read_panel(panel, hp_key, mp_key))
            if best.hp_count is not None and best.mp_count is not None:
                return best
        # 旧版客户端常用右下角4×2紧凑快捷栏，位置在底部状态栏上方。
        # 它只有约149×75px，不能用461×75长栏的右下角裁切读取。
        for scale in scales:
            # 横向只保留贴近右边缘的160px；更宽时地图背景中的竖线
            # 可能被误选为快捷栏左边框，使所有键位整体偏移约一格。
            x1 = max(0, w - int(round(160 * scale)))
            y1 = max(0, h - int(round(188 * scale)))
            y2 = max(0, h - int(round(68 * scale)))
            if y2 <= y1 or w - x1 < 145:
                continue
            panel = self._extract_panel(frame[y1:y2, x1:w])
            if panel is not None:
                include(self.read_panel(panel, hp_key, mp_key))
                if best.hp_count is not None and best.mp_count is not None:
                    return best
        return best


class PotionStockMonitor:
    """Require two matching visual samples; never carry a count across key changes."""

    def __init__(self) -> None:
        self._keys = {"hp": "", "mp": ""}
        self._candidate = {"hp": None, "mp": None}
        self._hits = {"hp": 0, "mp": 0}
        self._confirmed = {"hp": None, "mp": None}
        self._observed_at = {"hp": 0.0, "mp": 0.0}

    def update(self, reading: PotionStockReading) -> None:
        for kind in ("hp", "mp"):
            key = getattr(reading, f"{kind}_key")
            count = getattr(reading, f"{kind}_count")
            if key != self._keys[kind]:
                self._keys[kind] = key
                self._candidate[kind] = None
                self._hits[kind] = 0
                self._confirmed[kind] = None
            if count is None:
                self._candidate[kind] = None
                self._hits[kind] = 0
                self._confirmed[kind] = None
                continue
            if count != self._candidate[kind]:
                self._candidate[kind] = count
                self._hits[kind] = 1
                self._confirmed[kind] = None
            else:
                self._hits[kind] += 1
                if self._hits[kind] >= 2:
                    self._confirmed[kind] = count
                    self._observed_at[kind] = reading.timestamp

    def confirmed(self, kind: str, key: str, now: Optional[float] = None) -> Optional[int]:
        now = time.perf_counter() if now is None else float(now)
        kind = str(kind)
        if normalize_stock_key(key) != self._keys.get(kind):
            return None
        if now - self._observed_at.get(kind, 0.0) > 2.5:
            return None
        return self._confirmed.get(kind)

    def note_use(self, kind: str) -> None:
        count = self._confirmed.get(kind)
        if count is not None and count > 0:
            self._confirmed[kind] = count - 1
            # Do not restore the pre-keypress number from a single buffered
            # frame; require two fresh matching observations again.
            self._candidate[kind] = None
            self._hits[kind] = 0
