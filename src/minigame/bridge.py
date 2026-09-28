"""Run the lie-detector tracker on frames captured by the main application."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable, Optional

import numpy as np

from .state_machine import DetectorState, LieDetectorStateMachine, StateMachineResult


class MiniGameBridge:
    """Tracks the mini-game on the existing capture stream and follows its target."""

    def __init__(self, on_ownership_change: Callable[[bool], None]):
        self.state_machine = LieDetectorStateMachine(enable_dialog_sentinel=True)
        self.on_ownership_change = on_ownership_change
        self._lock = threading.Lock()
        self.owns_input = False
        self.awaiting_result_confirmation = False
        self.mouse_control_enabled = True
        self._last_hwnd = 0
        self._last_cursor_pos: tuple[int, int] | None = None
        self._user32 = ctypes.windll.user32

    def set_mouse_control_enabled(self, enabled: bool) -> None:
        """Enable or immediately revoke this bridge's ability to move the cursor."""
        with self._lock:
            self.mouse_control_enabled = bool(enabled)
            if not self.mouse_control_enabled:
                self._last_cursor_pos = None

    def reset(self) -> None:
        self._set_ownership(False)
        with self._lock:
            self.state_machine = LieDetectorStateMachine(enable_dialog_sentinel=True)
            self.awaiting_result_confirmation = False
            self._last_hwnd = 0

    def finish_result_confirmation(self) -> None:
        """Release the F6 handoff only after the completion popup is gone."""
        with self._lock:
            self.awaiting_result_confirmation = False
            game_active = self.state_machine.state in (DetectorState.COUNTDOWN, DetectorState.ACTIVE)
        self._set_ownership(game_active)

    def _set_ownership(self, owns_input: bool) -> None:
        callback = None
        with self._lock:
            if self.owns_input != owns_input:
                self.owns_input = owns_input
                callback = self.on_ownership_change
            if not owns_input:
                self._last_cursor_pos = None
        if callback is not None:
            callback(owns_input)

    def process(
        self, frame: np.ndarray, hwnd: int, hold_result_confirmation: bool = False
    ) -> Optional[StateMachineResult]:
        if frame is None or frame.size == 0:
            return None

        with self._lock:
            result = self.state_machine.update(frame)
            if result.just_exited_active and hold_result_confirmation:
                self.awaiting_result_confirmation = True
            awaiting_result = self.awaiting_result_confirmation
        active_window = result.state in (DetectorState.COUNTDOWN, DetectorState.ACTIVE) or awaiting_result
        self._set_ownership(active_window)

        if result.just_entered_active:
            self._last_hwnd = int(hwnd or 0)

        if (
            result.state == DetectorState.ACTIVE
            and result.track_result is not None
            and result.track_result.initialized
            and result.track_result.confidence >= 0.45
            and self._is_point_in_dialog(result.track_result)
        ):
            self._move_cursor(
                int(hwnd or self._last_hwnd),
                result.track_result.x,
                result.track_result.y,
                frame.shape[1],
                frame.shape[0],
            )
        elif result.state == DetectorState.IDLE:
            self._last_hwnd = 0
        return result

    @staticmethod
    def _is_point_in_dialog(track_result) -> bool:
        roi = getattr(track_result, "dialog_roi", None)
        if roi is None:
            return False
        x, y, width, height = roi
        return (
            width > 0
            and height > 0
            and x <= track_result.x < x + width
            and y <= track_result.y < y + height
        )

    def _move_cursor(
        self, hwnd: int, x: float, y: float, frame_width: int, frame_height: int
    ) -> bool:
        """Move the desktop cursor to the tracked point in client coordinates."""
        with self._lock:
            if not self.mouse_control_enabled:
                return False
            user32 = self._user32
            if not hwnd or not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
                return False

            rect = wintypes.RECT()
            if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
                return False
            client_width = rect.right - rect.left
            client_height = rect.bottom - rect.top
            if client_width <= 0 or client_height <= 0 or frame_width <= 0 or frame_height <= 0:
                return False

            point = wintypes.POINT(
                int(round(x * client_width / frame_width)),
                int(round(y * client_height / frame_height)),
            )
            if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
                return False
            target = (point.x, point.y)
            previous = self._last_cursor_pos
            if previous is not None and abs(previous[0] - point.x) < 3 and abs(previous[1] - point.y) < 3:
                return True
            moved = bool(user32.SetCursorPos(point.x, point.y))
            if moved:
                self._last_cursor_pos = target
            return moved
