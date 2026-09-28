"""Record physical keyboard edges sent to the game and replay their timing.

The recorder uses ``WH_KEYBOARD_LL`` instead of polling ``GetAsyncKeyState`` so
short Down/Alt overlaps are preserved.  Events marked ``LLKHF_INJECTED`` are
ignored; consequently a replay can never record itself.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from ctypes import wintypes
from typing import Dict, List, Optional, Tuple


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
LLKHF_EXTENDED = 0x01
LLKHF_INJECTED = 0x10
LLKHF_ALTDOWN = 0x20


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
user32.UnhookWindowsHookEx.restype = wintypes.BOOL
user32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
user32.GetMessageW.restype = wintypes.BOOL
user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.PostThreadMessageW.restype = wintypes.BOOL
kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
kernel32.GetModuleHandleW.restype = wintypes.HMODULE


VK_NAMES = {
    0x12: "alt",
    0xA4: "alt",
    0xA5: "alt_r",
    0x28: "down",
    0x26: "up",
    0x25: "left",
    0x27: "right",
    0x20: "space",
    0x11: "ctrl",
    0xA2: "ctrl",
    0xA3: "ctrl_r",
    0x10: "shift",
    0xA0: "shift",
    0xA1: "shift_r",
}


def key_name_for_vk(vk_code: int) -> str:
    if int(vk_code) in VK_NAMES:
        return VK_NAMES[int(vk_code)]
    if 0x30 <= int(vk_code) <= 0x5A:
        return chr(int(vk_code)).lower()
    return f"vk_{int(vk_code):02x}"


class ManualInputRecorder:
    """Global physical-key recorder restricted to the target game window."""

    def __init__(self, target_hwnd: Optional[int], save_path: str):
        self.target_hwnd = int(target_hwnd or 0)
        self.save_path = os.path.abspath(save_path)
        self._lock = threading.Lock()
        self._events: List[Dict] = []
        self._thread: Optional[threading.Thread] = None
        self._thread_id = 0
        self._hook = None
        self._hook_proc = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._error = ""
        self._ignored_injected = 0
        self._ignored_outside_game = 0

    @property
    def is_recording(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._hook)

    @property
    def error(self) -> str:
        return self._error

    def start(self) -> Tuple[bool, str]:
        if self.is_recording:
            return False, "已经在录制"
        with self._lock:
            self._events = []
        self._error = ""
        self._ignored_injected = 0
        self._ignored_outside_game = 0
        self._ready.clear()
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._hook_loop,
            name="ManualKeyboardRecorder",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=1.5)
        if not self._hook:
            return False, self._error or "低级键盘钩子启动超时"
        return True, "物理键盘录制已启动"

    def _hook_loop(self) -> None:
        self._thread_id = int(kernel32.GetCurrentThreadId())

        @HOOKPROC
        def hook_proc(code, wparam, lparam):
            if code >= 0 and int(wparam) in (
                WM_KEYDOWN,
                WM_SYSKEYDOWN,
                WM_KEYUP,
                WM_SYSKEYUP,
            ):
                data = ctypes.cast(
                    lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)
                ).contents
                flags = int(data.flags)
                if flags & LLKHF_INJECTED:
                    self._ignored_injected += 1
                elif (
                    self.target_hwnd
                    and int(user32.GetForegroundWindow() or 0) != self.target_hwnd
                ):
                    self._ignored_outside_game += 1
                else:
                    is_down = int(wparam) in (WM_KEYDOWN, WM_SYSKEYDOWN)
                    now_ns = time.perf_counter_ns()
                    event = {
                        "timestamp_ns": now_ns,
                        "vk": int(data.vkCode),
                        "scan": int(data.scanCode),
                        "edge": "down" if is_down else "up",
                        "key": key_name_for_vk(int(data.vkCode)),
                        "extended": bool(flags & LLKHF_EXTENDED),
                        "alt_context": bool(flags & LLKHF_ALTDOWN),
                        "hook_flags": flags,
                        "foreground_hwnd": int(user32.GetForegroundWindow() or 0),
                    }
                    with self._lock:
                        self._events.append(event)
            return user32.CallNextHookEx(self._hook, code, wparam, lparam)

        self._hook_proc = hook_proc
        module = kernel32.GetModuleHandleW(None)
        self._hook = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._hook_proc, module, 0
        )
        if not self._hook:
            self._error = f"SetWindowsHookExW 失败，WinError={ctypes.get_last_error()}"
            self._ready.set()
            return
        self._ready.set()
        message = wintypes.MSG()
        try:
            while not self._stop_requested.is_set():
                result = int(user32.GetMessageW(ctypes.byref(message), None, 0, 0))
                if result <= 0:
                    break
        finally:
            if self._hook:
                user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            self._hook_proc = None

    def stop(self, *, save: bool = True) -> List[Dict]:
        self._stop_requested.set()
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        with self._lock:
            events = [dict(item) for item in self._events]
        events = self._normalize(events)
        if save and events:
            self._save(events)
        return events

    @staticmethod
    def _normalize(events: List[Dict]) -> List[Dict]:
        if not events:
            return []
        base_ns = int(events[0]["timestamp_ns"])
        previous_ns = base_ns
        normalized = []
        for event in events:
            row = dict(event)
            timestamp_ns = int(row.pop("timestamp_ns"))
            row["offset_ms"] = (timestamp_ns - base_ns) / 1_000_000.0
            row["delta_ms"] = (timestamp_ns - previous_ns) / 1_000_000.0
            previous_ns = timestamp_ns
            normalized.append(row)
        return normalized

    def _save(self, events: List[Dict]) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        payload = {
            "format": 1,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_hwnd": self.target_hwnd,
            "ignored_injected": self._ignored_injected,
            "ignored_outside_game": self._ignored_outside_game,
            "down_jump_attempts": self.extract_down_jump_attempts(events),
            "events": events,
        }
        temp_path = self.save_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.save_path)

    def load(self) -> List[Dict]:
        if not os.path.isfile(self.save_path):
            return []
        with open(self.save_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        events = payload.get("events") if isinstance(payload, dict) else None
        return [dict(item) for item in events] if isinstance(events, list) else []

    @staticmethod
    def extract_down_jump_attempts(events: List[Dict]) -> List[Dict]:
        """Split a continuous physical recording into Down+Alt attempts."""
        attempts: List[Dict] = []
        down_active = False
        alt_active = False
        last_alt_down_ms: Optional[float] = None
        current: Optional[Dict] = None
        for index, event in enumerate(events):
            key = str(event.get("key") or "").lower()
            edge = str(event.get("edge") or "").lower()
            offset_ms = float(event.get("offset_ms", 0.0))
            is_alt = (
                key in ("alt", "alt_l", "alt_r", "lalt", "ralt")
                or int(event.get("vk", 0) or 0) in (0x12, 0xA4, 0xA5)
            )
            if key == "down" and edge == "down":
                # Ignore keyboard auto-repeat while the same physical press is
                # still held; it is preserved in raw events but is not a new
                # manual down-jump attempt.
                if not down_active:
                    down_active = True
                    current = {
                        "attempt": len(attempts) + 1,
                        "start_event_index": index,
                        "down_down_ms": offset_ms,
                    }
                    if alt_active and last_alt_down_ms is not None:
                        current["alt_down_ms"] = last_alt_down_ms
            elif is_alt and edge == "down":
                if not alt_active:
                    alt_active = True
                    last_alt_down_ms = offset_ms
                if down_active and current is not None and "alt_down_ms" not in current:
                    current["alt_down_ms"] = offset_ms
            elif is_alt and edge == "up":
                alt_active = False
                if current is not None and "alt_down_ms" in current:
                    current.setdefault("alt_up_ms", offset_ms)
            elif key == "down" and edge == "up":
                down_active = False
                if current is not None:
                    current.setdefault("down_up_ms", offset_ms)
            # Human fingers can release Down first or Alt first.  Finalize only
            # after both physical key-up edges arrive; otherwise a negative
            # release order is incorrectly reported as a shortened Alt hold.
            if (
                current is not None
                and "alt_down_ms" in current
                and "alt_up_ms" in current
                and "down_up_ms" in current
            ):
                current["end_event_index"] = index
                current["down_to_alt_ms"] = (
                    float(current["alt_down_ms"])
                    - float(current["down_down_ms"])
                )
                current["alt_hold_ms"] = (
                    float(current["alt_up_ms"])
                    - float(current["alt_down_ms"])
                )
                current["overlap_ms"] = (
                    min(float(current["alt_up_ms"]), float(current["down_up_ms"]))
                    - float(current["alt_down_ms"])
                )
                # Positive: Alt released first. Negative: Down released first.
                current["alt_up_to_down_up_ms"] = (
                    float(current["down_up_ms"])
                    - float(current["alt_up_ms"])
                )
                current["total_ms"] = (
                    max(float(current["down_up_ms"]), float(current["alt_up_ms"]))
                    - float(current["down_down_ms"])
                )
                attempts.append(current)
                current = None
        return attempts

    @classmethod
    def timing_summary(cls, events: List[Dict]) -> str:
        if not events:
            return "没有录制到游戏前台的物理键盘事件"
        attempts = cls.extract_down_jump_attempts(events)
        if not attempts:
            return f"共{len(events)}个边沿，未发现完整DOWN+ALT下跳组合"
        rows = [f"共{len(events)}个边沿，识别{len(attempts)}次下跳"]
        for attempt in attempts:
            detail = (
                f"第{attempt['attempt']}次: DOWN→ALT="
                f"{float(attempt['down_to_alt_ms']):.1f}ms"
            )
            if "alt_hold_ms" in attempt:
                detail += f"/ALT保持={float(attempt['alt_hold_ms']):.1f}ms"
            if "overlap_ms" in attempt:
                detail += f"/真实重叠={float(attempt['overlap_ms']):.1f}ms"
            if "alt_up_to_down_up_ms" in attempt:
                release_delta = float(attempt["alt_up_to_down_up_ms"])
                if release_delta >= 0:
                    detail += f"/ALT先松{release_delta:.1f}ms"
                else:
                    detail += f"/DOWN先松{abs(release_delta):.1f}ms"
            rows.append(detail)
        return "；".join(rows)

    def replay(self, driver, events: Optional[List[Dict]] = None) -> Dict:
        sequence = [dict(item) for item in (events if events is not None else self.load())]
        if not sequence:
            raise ValueError("没有可回放的手动按键记录")
        driver.release_all_keys()
        if not driver.ensure_focus():
            raise RuntimeError("无法将输入焦点切换到游戏窗口")
        time.sleep(0.35)
        started_ns = time.perf_counter_ns()
        replay_rows = []
        try:
            for event in sequence:
                target_ns = started_ns + int(float(event.get("offset_ms", 0.0)) * 1_000_000)
                while True:
                    remaining_ns = target_ns - time.perf_counter_ns()
                    if remaining_ns <= 0:
                        break
                    if remaining_ns > 2_000_000:
                        time.sleep((remaining_ns - 1_000_000) / 1_000_000_000.0)
                    else:
                        time.sleep(0)
                actual_ns = time.perf_counter_ns()
                delivery = driver.send_recorded_key_event(
                    key_name=str(event.get("key") or key_name_for_vk(int(event.get("vk", 0)))),
                    vk_code=int(event.get("vk", 0)),
                    scan_code=int(event.get("scan", 0)),
                    is_up=str(event.get("edge")) == "up",
                    extended=bool(event.get("extended", False)),
                    notify=False,
                )
                replay_rows.append(
                    {
                        **event,
                        "actual_offset_ms": (actual_ns - started_ns) / 1_000_000.0,
                        "drift_ms": (actual_ns - target_ns) / 1_000_000.0,
                        "delivery": delivery,
                    }
                )
        finally:
            driver.release_all_keys()
        return {
            "events": replay_rows,
            "max_abs_drift_ms": max(
                (abs(float(item["drift_ms"])) for item in replay_rows),
                default=0.0,
            ),
            "transport_failures": sum(
                not bool((item.get("delivery") or {}).get("transport_ok"))
                for item in replay_rows
            ),
        }
