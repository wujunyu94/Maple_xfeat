"""Detect and dismiss the mini-game completion dialog before F6 resumes."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ResultDialogMatch:
    bbox: tuple[int, int, int, int]
    button_center: tuple[int, int]
    score: float


class ResultDialogDetector:
    """Match the supplied fixed completion dialog, not arbitrary game popups."""

    def __init__(self, reference_path: Optional[str] = None, threshold: float = 0.82):
        if reference_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            reference_path = os.path.join(project_root, "testphoto", "小游戏结束提示_窗口.png")
        self.reference_path = reference_path
        self.threshold = float(threshold)
        self.reference = None
        try:
            # OpenCV's imread is not reliable with Chinese paths on Windows.
            encoded = np.fromfile(reference_path, dtype=np.uint8)
            self.reference = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        except (OSError, ValueError):
            pass

    @property
    def ready(self) -> bool:
        return self.reference is not None and self.reference.size > 0

    def detect(self, frame: np.ndarray) -> Optional[ResultDialogMatch]:
        if not self.ready or frame is None or frame.size == 0:
            return None
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        half = cv2.resize(gray, (max(1, fw // 2), max(1, fh // 2)), interpolation=cv2.INTER_AREA)
        hh, hw = half.shape[:2]
        # The dialog is centered in the client area. Small cropped test images
        # are searched in full, while real game frames use a bounded ROI.
        if fw >= 640 and fh >= 400:
            x0, x1 = int(hw * 0.12), int(hw * 0.88)
            y0, y1 = int(hh * 0.12), int(hh * 0.88)
        else:
            x0, x1, y0, y1 = 0, hw, 0, hh
        search = half[y0:y1, x0:x1]

        base_scale = fh / 1080.0
        scales = sorted({
            round(scale, 3) for scale in (
                1.0, base_scale, base_scale * 0.75, base_scale * 1.25,
                0.75, 1.25, 1.5, 2.0,
            ) if 0.45 <= scale <= 3.0
        })
        best = None
        rh, rw = self.reference.shape[:2]
        for scale in scales:
            tw, th = max(8, round(rw * scale / 2)), max(8, round(rh * scale / 2))
            if tw > search.shape[1] or th > search.shape[0]:
                continue
            template = cv2.resize(self.reference, (tw, th), interpolation=cv2.INTER_AREA)
            _, score, _, point = cv2.minMaxLoc(
                cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            )
            if best is None or score > best[0]:
                best = (float(score), x0 + point[0], y0 + point[1], tw, th)
        if best is None or best[0] < self.threshold:
            return None

        score, hx, hy, tw, th = best
        x, y, width, height = hx * 2, hy * 2, tw * 2, th * 2
        # The orange confirm button is at x=336..392, y=169..199 in the
        # 398x203 reference; derive its center from the detected window.
        bx = x + round(width * 364 / 398)
        by = y + round(height * 184 / 203)
        button = frame[
            max(0, y + round(height * 169 / 203)):min(fh, y + round(height * 199 / 203)),
            max(0, x + round(width * 336 / 398)):min(fw, x + round(width * 393 / 398)),
        ]
        if button.size == 0:
            return None
        hsv = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)
        orange = (
            (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 30)
            & (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 65)
        )
        if float(np.mean(orange)) < 0.12:
            return None
        return ResultDialogMatch((x, y, width, height), (bx, by), score)


class ResultDialogConfirmer:
    """Press the dialogue key once, then click Confirm until the popup vanishes."""

    def __init__(
        self,
        detector: ResultDialogDetector,
        driver,
        log: Callable[[str], None],
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.detector = detector
        self.driver = driver
        self.log = log
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.started_at = 0.0
        self.next_scan_at = 0.0
        self.next_action_at = 0.0
        self.candidate_hits = 0
        self.absent_hits = 0
        self.seen_dialog = False
        self.key_sent = False
        self.click_count = 0
        self.limit_logged = False

    def begin(self) -> None:
        self.reset()
        self.active = True
        self.started_at = self.clock()
        if not self.detector.ready:
            self.log("🛑 [测谎结束确认] 缺少结束弹框模板，保持 F6 暂停")

    def update(self, frame: np.ndarray, dialog_key: str = "y", dialog_vk: Optional[int] = None) -> bool:
        """Return True only after a detected dialog disappears, or no dialog appears."""
        if not self.active:
            return False
        if not self.detector.ready:
            return False
        now = self.clock()
        if now < self.next_scan_at:
            return False
        self.next_scan_at = now + 0.12
        match = self.detector.detect(frame)
        if match is not None:
            self.absent_hits = 0
            self.candidate_hits += 1
            if self.candidate_hits < 2:
                return False
            if not self.seen_dialog:
                self.seen_dialog = True
                self.log(
                    f"🔎 [测谎结束确认] 检测到结束弹框，匹配={match.score:.3f}，"
                    f"按钮={match.button_center}；继续保持 F6 暂停"
                )
            if now < self.next_action_at:
                return False
            if not self.key_sent:
                self.driver.press_key(dialog_key or "y", duration_ms=65, vk_code=dialog_vk)
                self.key_sent = True
                self.next_action_at = self.clock() + 0.55
                self.log(f"⌨️ [测谎结束确认] 已先按对话键 {(dialog_key or 'y').upper()}")
            elif self.click_count < 5:
                self.click_count += 1
                if self.driver.click_client(*match.button_center):
                    self.log(
                        f"🖱️ [测谎结束确认] 弹框仍在，点击确认按钮 "
                        f"{match.button_center}（第{self.click_count}次）"
                    )
                else:
                    self.log(
                        f"⚠️ [测谎结束确认] 第{self.click_count}次点击未送达游戏窗口，"
                        "继续保持 F6 暂停"
                    )
                self.next_action_at = self.clock() + 0.75
            elif not self.limit_logged:
                self.limit_logged = True
                self.log("🛑 [测谎结束确认] 5次点击后弹框仍在，停止点击并保持 F6 暂停；人工关闭后可恢复")
            return False

        self.candidate_hits = 0
        if self.seen_dialog:
            self.absent_hits += 1
            if self.absent_hits >= 3:
                self.log("✅ [测谎结束确认] 弹框连续3次检测不到，允许恢复 F6")
                self.active = False
                return True
        elif now - self.started_at >= 3.0:
            self.log("ℹ️ [测谎结束确认] 结束后3秒未出现提示弹框，允许恢复 F6")
            self.active = False
            return True
        return False
