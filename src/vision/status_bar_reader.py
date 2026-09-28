"""Visual reader for the classic-game HP/MP/EXP status bar.

The reader deliberately separates fast colour-bar measurements from slower OCR.
Potion decisions can therefore react to the bar at 10 Hz without running an OCR
model at the same rate.  OCR is used for the exact values and EXP accounting.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StatusBarReading:
    timestamp: float
    hp_bar_percent: Optional[float] = None
    mp_bar_percent: Optional[float] = None
    hp_current: Optional[int] = None
    hp_max: Optional[int] = None
    mp_current: Optional[int] = None
    mp_max: Optional[int] = None
    exp_current: Optional[int] = None
    exp_percent: Optional[float] = None
    ocr_confidence: float = 0.0
    ocr_lines: Tuple[str, ...] = ()
    roi_valid: bool = False
    layout_confident: bool = False

    @property
    def hp_percent(self) -> Optional[float]:
        if self.hp_current is not None and self.hp_max:
            return max(0.0, min(100.0, self.hp_current * 100.0 / self.hp_max))
        return self.hp_bar_percent

    @property
    def mp_percent(self) -> Optional[float]:
        if self.mp_current is not None and self.mp_max:
            return max(0.0, min(100.0, self.mp_current * 100.0 / self.mp_max))
        return self.mp_bar_percent


class StatusBarReader:
    """Read a user-selected status-bar ROI from a BGR game frame."""

    def __init__(self) -> None:
        self._ocr = None
        self._ocr_init_lock = threading.Lock()
        # 状态栏布局固定为 HP / MP / EXP 三段。正常路径直接把三段文字送
        # 入识别模型，跳过耗时最大的文本检测网络；只有快速路径缺字段时
        # 才低频回退一次完整检测，避免异常 ROI 令 CPU 长期满载。
        self._next_full_ocr_fallback_at = 0.0
        self._last_ocr_text_band: Optional[np.ndarray] = None
        self._last_ocr_values: Dict[str, Any] = {}
        self._last_ocr_confidence = 0.0
        self._last_ocr_lines: Tuple[str, ...] = ()
        self._last_ocr_mode = ""

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        with self._ocr_init_lock:
            if self._ocr is None:
                # Import lazily so an unused status card adds no startup cost.
                from rapidocr_onnxruntime import RapidOCR

                self._ocr = RapidOCR(
                    intra_op_num_threads=1,
                    inter_op_num_threads=1,
                )
        return self._ocr

    @staticmethod
    def resolve_roi(
        frame_shape: Sequence[int], roi: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[int, int, int, int]]:
        if not roi or len(frame_shape) < 2:
            return None
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if fw <= 0 or fh <= 0:
            return None
        try:
            if all(k in roi for k in ("nx", "ny", "nw", "nh")):
                x = int(round(float(roi["nx"]) * fw))
                y = int(round(float(roi["ny"]) * fh))
                w = int(round(float(roi["nw"]) * fw))
                h = int(round(float(roi["nh"]) * fh))
            else:
                source_w = int(roi.get("frame_w", fw) or fw)
                source_h = int(roi.get("frame_h", fh) or fh)
                sx = fw / max(1, source_w)
                sy = fh / max(1, source_h)
                x = int(round(float(roi["x"]) * sx))
                y = int(round(float(roi["y"]) * sy))
                w = int(round(float(roi["w"]) * sx))
                h = int(round(float(roi["h"]) * sy))
        except (KeyError, TypeError, ValueError):
            return None
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))
        w = max(0, min(fw - x, w))
        h = max(0, min(fh - y, h))
        if w < 90 or h < 18:
            return None
        return x, y, w, h

    @staticmethod
    def _track_boxes_with_confidence(
        crop: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, int, int]], bool]:
        """Locate the three long framed bars in the lower half of the ROI."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright = (gray > 60).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
        boxes: List[Tuple[int, int, int, int]] = []
        for idx in range(1, count):
            x, y, bw, bh, _area = (int(v) for v in stats[idx])
            if (
                y >= int(h * 0.35)
                and bw >= int(w * 0.18)
                and bw <= int(w * 0.45)
                and bh >= max(6, int(h * 0.18))
                and bh <= int(h * 0.75)
            ):
                boxes.append((x, y, bw, bh))
        # Suppress nested components created by the coloured fill.
        boxes.sort(key=lambda b: (-b[2], b[0]))
        selected: List[Tuple[int, int, int, int]] = []
        for candidate in boxes:
            cx = candidate[0] + candidate[2] / 2.0
            if any(abs(cx - (b[0] + b[2] / 2.0)) < min(candidate[2], b[2]) * 0.5 for b in selected):
                continue
            selected.append(candidate)
        selected.sort(key=lambda b: b[0])
        if len(selected) >= 3:
            return selected[:3], True

        # Tight user crops retain stable thirds even if a theme/scale changes
        # the border contrast enough to defeat connected components.
        thirds = (0.0, 1.0 / 3.0, 0.65, 1.0)
        fallback = []
        for left, right in zip(thirds, thirds[1:]):
            x1 = int(round(w * left))
            x2 = int(round(w * right))
            fallback.append((x1, int(h * 0.48), max(1, x2 - x1), max(1, int(h * 0.42))))
        return fallback, False

    @staticmethod
    def _track_boxes(crop: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Compatibility wrapper for callers that only need the rectangles."""
        boxes, _confident = StatusBarReader._track_boxes_with_confidence(crop)
        return boxes

    @staticmethod
    def _colour_fill_percent(
        crop: np.ndarray, box: Tuple[int, int, int, int], kind: str
    ) -> Optional[float]:
        x, y, w, h = box
        pad_x = max(1, int(round(w * 0.018)))
        pad_y = max(1, int(round(h * 0.14)))
        inner = crop[
            y + pad_y : y + h - pad_y,
            x + pad_x : x + w - pad_x,
        ]
        if inner.size == 0 or inner.shape[1] < 8:
            return None
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        if kind == "hp":
            mask = (((hue <= 10) | (hue >= 170)) & (sat >= 120) & (val >= 110))
        elif kind == "mp":
            mask = ((hue >= 90) & (hue <= 135) & (sat >= 90) & (val >= 85))
        else:
            mask = ((hue >= 15) & (hue <= 42) & (sat >= 90) & (val >= 100))

        # A column counts as filled when colour appears in at least a quarter
        # of its interior pixels.  Use the right-most filled column so bevels
        # and sparse shine pixels do not depress the estimate.
        occupied = np.mean(mask, axis=0) >= 0.25
        positions = np.flatnonzero(occupied)
        if positions.size == 0:
            return 0.0
        rightmost = int(positions[-1]) + 1
        # Ignore isolated colour noise far away from the leading contiguous run.
        gaps = np.flatnonzero(~occupied[:rightmost])
        if gaps.size:
            long_gap_start = None
            run = 0
            for i, is_on in enumerate(occupied[:rightmost]):
                if is_on:
                    run = 0
                else:
                    run += 1
                    if run >= 4:
                        long_gap_start = i - run + 1
                        break
            if long_gap_start is not None:
                rightmost = long_gap_start
        return max(0.0, min(100.0, rightmost * 100.0 / inner.shape[1]))

    @staticmethod
    def _normalise_ocr_text(text: str) -> str:
        return (
            str(text)
            .upper()
            .replace("[", "(")
            .replace("]", ")")
            .replace("，", ".")
            .replace(",", ".")
            .replace(" ", "")
        )

    @classmethod
    def parse_ocr_lines(cls, lines: Sequence[str]) -> Dict[str, Any]:
        joined = " ".join(cls._normalise_ocr_text(x) for x in lines)
        values: Dict[str, Any] = {}
        hp = re.search(r"HP\D*(\d+)\D+(\d+)", joined)
        mp = re.search(r"MP\D*(\d+)\D+(\d+)", joined)
        exp = re.search(r"EXP\D*(\d+).*?(\d+(?:\.\d+)?)%", joined)
        if hp:
            current, maximum = int(hp.group(1)), int(hp.group(2))
            if 0 <= current <= maximum and maximum > 0:
                values.update(hp_current=current, hp_max=maximum)
        if mp:
            current, maximum = int(mp.group(1)), int(mp.group(2))
            if 0 <= current <= maximum and maximum > 0:
                values.update(mp_current=current, mp_max=maximum)
        if exp:
            current, percent = int(exp.group(1)), float(exp.group(2))
            if current >= 0 and 0.0 <= percent < 100.0:
                values.update(exp_current=current, exp_percent=percent)
        return values

    def read(
        self,
        frame_bgr: np.ndarray,
        roi: Optional[Dict[str, Any]],
        *,
        run_ocr: bool = False,
        ocr_hp_mp: bool = True,
    ) -> StatusBarReading:
        now = time.perf_counter()
        resolved = self.resolve_roi(frame_bgr.shape, roi)
        if resolved is None:
            return StatusBarReading(timestamp=now)
        x, y, w, h = resolved
        crop = frame_bgr[y : y + h, x : x + w]
        boxes, layout_confident = self._track_boxes_with_confidence(crop)
        hp_bar = self._colour_fill_percent(crop, boxes[0], "hp")
        mp_bar = self._colour_fill_percent(crop, boxes[1], "mp")
        reading = StatusBarReading(
            timestamp=now,
            hp_bar_percent=hp_bar,
            mp_bar_percent=mp_bar,
            roi_valid=True,
            layout_confident=layout_confident,
        )
        if not run_ocr:
            return reading

        # 数值文字位于三条色条上方。内容未发生视觉变化时直接复用上次
        # 精确值；血条颜色仍由上面的快速路径逐次读取，不受此缓存影响。
        left_pad = max(4, int(round(w * 0.035)))
        ocr_indices = list(range(min(3, len(boxes)))) if ocr_hp_mp else [2]
        text_crops = []
        for index in ocr_indices:
            if index >= len(boxes):
                continue
            bx, by, bw, _bh = boxes[index]
            text_crop = crop[
                0:min(h, by + 2),
                max(0, bx - left_pad):min(w, bx + bw + 2),
            ]
            if text_crop.size:
                text_crops.append(text_crop)
        if ocr_hp_mp:
            text_bottom = min(h, max((box[1] + 2 for box in boxes[:3]), default=h))
            text_band = cv2.cvtColor(crop[:text_bottom], cv2.COLOR_BGR2GRAY)
            ocr_mode = "full"
        elif text_crops:
            text_band = cv2.cvtColor(text_crops[-1], cv2.COLOR_BGR2GRAY)
            ocr_mode = "exp_only"
        else:
            text_band = np.empty((0, 0), dtype=np.uint8)
            ocr_mode = "exp_only"
        if (
            self._last_ocr_text_band is not None
            and self._last_ocr_mode == ocr_mode
            and self._last_ocr_text_band.shape == text_band.shape
            and self._last_ocr_values
        ):
            changed_ratio = float(np.mean(
                cv2.absdiff(self._last_ocr_text_band, text_band) > 18
            ))
            # 单个数字字形变化通常只占整条文字带约 0.3%~0.8%；阈值
            # 必须低于这个比例，同时用 18 灰度差过滤捕获抖动。
            if changed_ratio <= 0.0015:
                return replace(
                    reading,
                    **self._last_ocr_values,
                    ocr_confidence=self._last_ocr_confidence,
                    ocr_lines=self._last_ocr_lines,
                )

        ocr = self._get_ocr()
        lines: List[str] = []
        confidences: List[float] = []

        def append_recognition_result(result, *, detection_mode: bool) -> None:
            for item in result or []:
                if not item:
                    continue
                # RapidOCR 完整检测输出 [box, text, score]；关闭检测后对
                # 单行图直接输出 [text, score]。
                if detection_mode and len(item) >= 3:
                    text, confidence = item[1], item[2]
                elif not detection_mode and len(item) >= 2:
                    text, confidence = item[0], item[1]
                else:
                    continue
                lines.append(str(text))
                try:
                    confidences.append(float(confidence))
                except (TypeError, ValueError):
                    pass

        # HP/MP只需初次安全校验和低频最大值复核；稳定运行时仅识别EXP，
        # 百分比和自动补给始终读取上面的颜色条快路径。
        for text_crop in text_crops:
            result, _elapsed = ocr(
                text_crop,
                use_det=False,
                use_cls=False,
                use_rec=True,
            )
            append_recognition_result(result, detection_mode=False)

        parsed = self.parse_ocr_lines(lines)
        required = (
            {
                "hp_current", "hp_max", "mp_current", "mp_max",
                "exp_current", "exp_percent",
            }
            if ocr_hp_mp else {"exp_current", "exp_percent"}
        )
        if not required.issubset(parsed) and now >= self._next_full_ocr_fallback_at:
            # 非标准皮肤/框选偏差仍保留原完整 OCR 作为兼容兜底，但至多
            # 每 10 秒一次，绝不能再次形成 0.5 秒周期追赶 1 秒推理的死循环。
            self._next_full_ocr_fallback_at = now + 10.0
            enlarged = cv2.resize(
                crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
            )
            result, _elapsed = ocr(enlarged)
            fallback_lines: List[str] = []
            fallback_confidences: List[float] = []
            for item in result or []:
                if item and len(item) >= 3:
                    fallback_lines.append(str(item[1]))
                    try:
                        fallback_confidences.append(float(item[2]))
                    except (TypeError, ValueError):
                        pass
            fallback_parsed = self.parse_ocr_lines(fallback_lines)
            if len(fallback_parsed) > len(parsed):
                parsed.update(fallback_parsed)
                lines = fallback_lines
                confidences = fallback_confidences

        confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        if required.issubset(parsed):
            self._last_ocr_text_band = text_band.copy()
            self._last_ocr_values = dict(parsed)
            self._last_ocr_confidence = confidence
            self._last_ocr_lines = tuple(lines)
            self._last_ocr_mode = ocr_mode

        return replace(
            reading,
            **parsed,
            ocr_confidence=confidence,
            ocr_lines=tuple(lines),
        )


class ExperienceRateTracker:
    """Accumulate observed EXP gains and expose a session hourly rate."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.started_at: Optional[float] = None
        self.last_exp: Optional[int] = None
        self.last_percent: Optional[float] = None
        self.total_gain = 0

    def update(self, exp_value: int, exp_percent: float, timestamp: Optional[float] = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        if exp_value < 0 or not 0.0 <= exp_percent < 100.0:
            return
        if self.started_at is None:
            self.started_at = now
            self.last_exp = exp_value
            self.last_percent = exp_percent
            return
        assert self.last_exp is not None
        delta = 0
        if exp_value >= self.last_exp:
            delta = exp_value - self.last_exp
        elif self.last_percent is not None and exp_percent < self.last_percent:
            # Level-up rollover: infer the previous level's required EXP from
            # its displayed value and percentage, then add the new-level EXP.
            if self.last_percent >= 0.01:
                estimated_max = int(round(self.last_exp * 100.0 / self.last_percent))
                delta = max(0, estimated_max - self.last_exp) + exp_value
        else:
            # EXP cannot normally move backwards inside one level.  Treat this
            # as an OCR outlier and retain the last trusted sample.
            return
        # Reject OCR catastrophes while still allowing large quest rewards.
        if delta <= max(10_000_000, self.last_exp * 20 + 1):
            self.total_gain += max(0, int(delta))
            self.last_exp = int(exp_value)
            self.last_percent = float(exp_percent)

    def per_hour(self, timestamp: Optional[float] = None) -> Optional[float]:
        if self.started_at is None:
            return None
        now = time.perf_counter() if timestamp is None else float(timestamp)
        elapsed = now - self.started_at
        if elapsed <= 0.0:
            return None
        return self.total_gain * 3600.0 / elapsed
