"""F6 death handling: release input, respawn, then stop or route home."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class DeathRecoveryController:
    def __init__(
        self, *, detector, frame_getter: Callable, map_id_getter: Callable,
        map_ready: Callable[[int, Optional[int]], bool],
        expected_town_getter: Callable[[Optional[int]], Optional[int]],
        hp_zero_getter: Callable,
        stock_getter: Callable,
        threshold_getter: Callable[[], int], f6_running: Callable[[], bool],
        can_monitor: Callable[[], bool], reconnecting: Callable[[], bool],
        interrupt: Callable[[], bool], click_confirm: Callable[[], Optional[bool]],
        wake_map_ocr: Callable[[], None], resume: Callable[[int, Optional[int]], bool],
        halt: Callable[[], None], status: Callable[[bool, str], None],
        log: Callable[[str], None], stop_event: threading.Event,
    ) -> None:
        self.detector = detector
        self.frame_getter = frame_getter
        self.map_id_getter = map_id_getter
        self.map_ready = map_ready
        self.expected_town_getter = expected_town_getter
        self.hp_zero_getter = hp_zero_getter
        self.stock_getter = stock_getter
        self.threshold_getter = threshold_getter
        self.f6_running = f6_running
        self.can_monitor = can_monitor
        self.reconnecting = reconnecting
        self.interrupt = interrupt
        self.click_confirm = click_confirm
        self.wake_map_ocr = wake_map_ocr
        self.resume = resume
        self.halt = halt
        self.status = status
        self.log = log
        self.stop_event = stop_event
        self._state = "idle"
        self._cancel = threading.Event()
        self._source_map: Optional[int] = None
        self._expected_town: Optional[int] = None
        self._stock: Optional[int] = None
        self._threshold = 0
        self._presses = 0
        self._click_failures = 0
        self._last_press_at = 0.0
        self._town_candidate: Optional[int] = None
        self._town_hits = 0
        self._state_since = 0.0
        self._suppress_input = False
        self._epoch = 0
        self._last_hp_reject_log_at = 0.0

    @property
    def active(self) -> bool:
        return self._state not in ("idle", "blocked", "halted")

    @property
    def input_suppressed(self) -> bool:
        return self._suppress_input

    def cancel(self) -> None:
        self._epoch += 1
        self._cancel.set()
        self._state = "idle"
        self.status(False, "")

    def arm(self) -> None:
        """Allow monitoring again only after an explicit new F6 start."""
        self._epoch += 1
        self._cancel.clear()
        self._state = "idle"
        self._suppress_input = False
        self.status(False, "")

    def _set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        self._state_since = time.perf_counter()
        self.status(state not in ("idle", "blocked", "halted"), detail)

    def _block(self, reason: str) -> None:
        self.log(f"⛔ [死亡恢复] {reason}；已停止自动输入，等待人工处理")
        self._set_state("blocked", reason)

    def _begin(self, frame) -> None:
        epoch = self._epoch
        self._cancel.clear()
        self._suppress_input = True
        self._source_map = self.map_id_getter()
        self._set_state("interrupting", "检测到死亡弹窗，正在急停输入")
        self.log(f"💀 [死亡弹窗] 已识别复活提示，死亡地图={self._source_map}，立即停止 F6 输入")
        if not self.interrupt():
            self._block("F6 工作线程未退出，未发送复活键")
            return
        if epoch != self._epoch or self._cancel.is_set() or self.reconnecting():
            self._set_state("idle")
            return
        self._expected_town = self.expected_town_getter(self._source_map)
        self.log(f"🏠 [死亡复活] 预计回城 MapID {self._expected_town}")
        self._threshold = max(0, int(self.threshold_getter()))
        self._stock = self.stock_getter(frame)
        self.log(
            f"🧪 [死亡库存判定] 血药="
            f"{self._stock if self._stock is not None else '未知'}，"
            f"停机阈值≤{self._threshold}；"
            f"{'返程后停机' if self._stock is None or self._stock <= self._threshold else '返程后恢复F6'}"
        )
        if epoch != self._epoch or self._cancel.is_set() or self.reconnecting():
            self._set_state("idle")
            return
        self._presses = 0
        self._click_failures = 0
        self._click_respawn(epoch)

    def _click_respawn(self, expected_epoch: Optional[int] = None) -> None:
        if (
            self._cancel.is_set() or self.reconnecting()
            or (expected_epoch is not None and expected_epoch != self._epoch)
        ):
            return
        clicked = self.click_confirm()
        if clicked is None:
            self._set_state("waiting_town", "复活弹窗已消失，等待主城地图确认")
            self.wake_map_ocr()
            return
        if not clicked:
            self._click_failures += 1
            if self._click_failures >= 3:
                self._block("连续3次找不到或点不到复活确认按钮")
            else:
                self._last_press_at = time.perf_counter()
                self._set_state("confirming", "等待复活确认按钮重新定位")
            return
        self._click_failures = 0
        self._presses += 1
        self._last_press_at = time.perf_counter()
        self._set_state("confirming", f"已点击复活确认按钮（第{self._presses}次）")
        self.log(f"🖱️ [死亡复活] 点击确认按钮，第{self._presses}次")
        self.wake_map_ocr()

    def _complete(self, town_map: int) -> None:
        if self._cancel.is_set() or self._state != "waiting_town":
            return
        if self._stock is None or self._stock <= self._threshold:
            self.halt()
            self.log(
                f"🛑 [死亡后停机] 已确认从 {self._source_map} 到主城 {town_map}；"
                f"血药={self._stock if self._stock is not None else '未知'}，"
                f"阈值={self._threshold}，停止 F6 与全部键盘输入"
            )
            self._set_state("halted", "血药不足或库存未知，已回城停机")
            return
        if self.resume(town_map, self._source_map):
            self._suppress_input = False
            self.log(
                f"▶️ [死亡后返程] 已确认主城 MapID {town_map}；"
                f"血药={self._stock}>{self._threshold}，接入异常回城路线"
            )
            self._set_state("idle")
        else:
            self._block("无法启动返回死亡前巡逻地图的路线")

    def tick(self) -> None:
        if self.stop_event.is_set() or self._cancel.is_set():
            return
        if self.reconnecting():
            if self.active:
                self._set_state("idle")
            return
        if self._state == "idle":
            if not self.f6_running() or not self.can_monitor():
                return
            frame = self.frame_getter()
            if frame is not None and self.detector.detect(frame):
                hp_zero = self.hp_zero_getter(frame)
                if hp_zero is True:
                    self._begin(frame)
                elif hp_zero is None:
                    self._suppress_input = True
                    self._set_state("interrupting", "死亡弹窗已出现，但 HP 读数未知")
                    self.interrupt()
                    self._block("HP 读数未知，不能安全确认复活")
                else:
                    now = time.perf_counter()
                    if now - self._last_hp_reject_log_at >= 5.0:
                        self._last_hp_reject_log_at = now
                        self.log("⚠️ [死亡弹窗候选] HP 未确认归零，不接管键盘")
            return
        if self._state in ("blocked", "halted"):
            return
        if self._state == "confirming":
            # Do not hammer the dialog key while a load screen is appearing.
            if time.perf_counter() - self._last_press_at < 0.75:
                return
            frame = self.frame_getter()
            if frame is None:
                return
            if self.detector.detect(frame):
                if self._presses >= 4:
                    self._block("连续4次鼠标点击后复活弹窗仍在")
                elif time.perf_counter() - self._last_press_at >= 1.25:
                    self._click_respawn(self._epoch)
                return
            self._set_state("waiting_town", "弹窗已消失，等待主城地图确认")
            self.wake_map_ocr()
        if self._state == "waiting_town":
            current = self.map_id_getter()
            if (current is not None and current != self._source_map
                    and self.map_ready(int(current), self._expected_town)):
                if current == self._town_candidate:
                    self._town_hits += 1
                else:
                    self._town_candidate, self._town_hits = int(current), 1
                if self._town_hits >= 2:
                    self._complete(int(current))
                    return
            else:
                self._town_candidate, self._town_hits = None, 0
            if time.perf_counter() - self._state_since >= 45.0:
                self._block("45秒内未确认复活后的新地图")

    def run(self) -> None:
        last_error = 0.0
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:
                now = time.perf_counter()
                if now - last_error > 5.0:
                    last_error = now
                    self.log(f"⚠️ [死亡恢复异常] {type(exc).__name__}: {exc}")
                if self.active:
                    self._block("恢复流程异常")
            # The non-dialog prefilter costs under 1ms on an ordinary map;
            # a 250ms cadence avoids missing a manually dismissed popup.
            self.stop_event.wait(0.5 if self.active else 0.25)
