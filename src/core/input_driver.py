"""
input_driver.py - 游戏底层输入模拟驱动 (DirectInput / ScanCode / MapVirtualKey)
支持万能硬件扫描码动态映射与按键录制，支持拟人化按键延时、组合键与全局急停保护。
"""

import time
import random
import ctypes
import sys
from ctypes import wintypes
from typing import Optional, List, Dict, Tuple

user32 = ctypes.windll.user32

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_CHAR = 0x0102
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEWHEEL = 0x020A
MK_LBUTTON = 0x0001


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTUNION(ctypes.Union):
    # INPUT is a union of mouse, keyboard and hardware payloads.  Preserve the
    # 32-byte union size on 64-bit Windows even though this driver only emits
    # keyboard events; otherwise SendInput rejects the 32-byte structure.
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("_raw", ctypes.c_byte * 32),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT

# 静态映射优先表 (DirectInput Scan Codes)
SCAN_CODES = {
    "escape": (0x01, False), "esc": (0x01, False),
    "1": (0x02, False), "2": (0x03, False), "3": (0x04, False), "4": (0x05, False), "5": (0x06, False),
    "6": (0x07, False), "7": (0x08, False), "8": (0x09, False), "9": (0x0A, False), "0": (0x0B, False),
    "minus": (0x0C, False), "equals": (0x0D, False), "backspace": (0x0E, False), "tab": (0x0F, False),
    "q": (0x10, False), "w": (0x11, False), "e": (0x12, False), "r": (0x13, False), "t": (0x14, False),
    "y": (0x15, False), "u": (0x16, False), "i": (0x17, False), "o": (0x18, False), "p": (0x19, False),
    "a": (0x1E, False), "s": (0x1F, False), "d": (0x20, False), "f": (0x21, False), "g": (0x22, False),
    "h": (0x23, False), "j": (0x24, False), "k": (0x25, False), "l": (0x26, False),
    "z": (0x2C, False), "x": (0x2D, False), "c": (0x2E, False), "v": (0x2F, False), "b": (0x30, False),
    "n": (0x31, False), "m": (0x32, False),
    "enter": (0x1C, False), "return": (0x1C, False),
    "space": (0x39, False),
    "lctrl": (0x1D, False), "ctrl": (0x1D, False), "control_l": (0x1D, False), "control_r": (0x1D, True),
    "lshift": (0x2A, False), "shift": (0x2A, False), "shift_l": (0x2A, False), "shift_r": (0x36, False),
    "lalt": (0x38, False), "alt": (0x38, False), "alt_l": (0x38, False), "alt_r": (0x38, True),
    "up": (0x48, True),
    "left": (0x4B, True),
    "right": (0x4D, True),
    "down": (0x50, True),
    "insert": (0x52, True), "ins": (0x52, True),
    "delete": (0x53, True), "del": (0x53, True),
    "home": (0x47, True), "hm": (0x47, True),
    "end": (0x4F, True),
    "pageup": (0x49, True), "pup": (0x49, True), "prior": (0x49, True),
    "pagedown": (0x51, True), "pdn": (0x51, True), "next": (0x51, True),
    # F1 - F12
    "f1": (0x3B, False), "f2": (0x3C, False), "f3": (0x3D, False), "f4": (0x3E, False),
    "f5": (0x3F, False), "f6": (0x40, False), "f7": (0x41, False), "f8": (0x42, False),
    "f9": (0x43, False), "f10": (0x44, False), "f11": (0x57, False), "f12": (0x58, False),
    # 小键盘区
    "numpad0": (0x52, False), "kp_0": (0x52, False), "kp_insert": (0x52, False),
    "numpad1": (0x4F, False), "kp_1": (0x4F, False), "kp_end": (0x4F, False),
    "numpad2": (0x50, False), "kp_2": (0x50, False), "kp_down": (0x50, False),
    "numpad3": (0x51, False), "kp_3": (0x51, False), "kp_next": (0x51, False),
    "numpad4": (0x4B, False), "kp_4": (0x4B, False), "kp_left": (0x4B, False),
    "numpad5": (0x4C, False), "kp_5": (0x4C, False), "kp_begin": (0x4C, False),
    "numpad6": (0x4D, False), "kp_6": (0x4D, False), "kp_right": (0x4D, False),
    "numpad7": (0x47, False), "kp_7": (0x47, False), "kp_home": (0x47, False),
    "numpad8": (0x48, False), "kp_8": (0x48, False), "kp_up": (0x48, False),
    "numpad9": (0x49, False), "kp_9": (0x49, False), "kp_prior": (0x49, False),
    "kp_add": (0x4E, False), "kp_subtract": (0x4A, False), "kp_multiply": (0x37, False), "kp_divide": (0x35, True)
}


VK_MAP = {
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "ctrl": 0x11, "lctrl": 0x11, "control_l": 0x11,
    "alt": 0x12, "lalt": 0x12, "alt_l": 0x12,
    "shift": 0x10, "lshift": 0x10, "shift_l": 0x10,
    "space": 0x20, "enter": 0x0D, "escape": 0x1B, "tab": 0x09,
    "z": 0x5A, "x": 0x58, "c": 0x43, "v": 0x56, "a": 0x41, "s": 0x53, "d": 0x44, "f": 0x46,
    "q": 0x51, "w": 0x57, "e": 0x45, "r": 0x52,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39, "0": 0x30,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77, "f9": 0x78
}


kernel32 = ctypes.windll.kernel32
advapi32 = ctypes.windll.advapi32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_ELEVATION_CLASS = 20


class TOKEN_ELEVATION(ctypes.Structure):
    _fields_ = [("TokenIsElevated", wintypes.DWORD)]


kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
kernel32.CloseHandle.restype = wintypes.BOOL
advapi32.OpenProcessToken.argtypes = (
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
)
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.GetTokenInformation.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
)
advapi32.GetTokenInformation.restype = wintypes.BOOL


class InputDriver:
    def __init__(self, target_hwnd: Optional[int] = None, input_mode: str = "background"):
        self.target_hwnd = target_hwnd
        self.input_mode = "background"
        self.set_mode(input_mode)
        self.active_keys = set()
        self.current_facing = "right"
        self.key_event_callback = None
        self.last_event_source = "unknown"
        self.delivery_audit_enabled = False
        self.last_delivery_result: Optional[Dict] = None

    def set_mode(self, mode: str):
        """设置输入模式：'background' (PostMessage) 或 'foreground' (SendInput)"""
        m = str(mode).lower().strip()
        if m in ("background", "postmessage", "post", "bg", "后台"):
            self.input_mode = "background"
        elif m in ("foreground", "sendinput", "fg", "前台"):
            self.input_mode = "foreground"
        else:
            self.input_mode = "background"

    @staticmethod
    def _process_is_elevated(pid: int) -> Optional[bool]:
        """Return a process elevation state, or ``None`` if it cannot be queried."""
        process_handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not process_handle:
            return None
        token_handle = wintypes.HANDLE()
        try:
            if not advapi32.OpenProcessToken(
                process_handle, TOKEN_QUERY, ctypes.byref(token_handle)
            ):
                return None
            elevation = TOKEN_ELEVATION()
            returned = wintypes.DWORD()
            if not advapi32.GetTokenInformation(
                token_handle,
                TOKEN_ELEVATION_CLASS,
                ctypes.byref(elevation),
                ctypes.sizeof(elevation),
                ctypes.byref(returned),
            ):
                return None
            return bool(elevation.TokenIsElevated)
        finally:
            if token_handle:
                kernel32.CloseHandle(token_handle)
            kernel32.CloseHandle(process_handle)

    def check_input_readiness(self, *, focus: bool = False) -> Tuple[bool, str]:
        """Validate HWND, Windows integrity level and optional foreground focus.

        Windows UIPI silently discards both SendInput and PostMessage when a
        normal process targets an elevated game.  Reporting that mismatch here
        prevents movement tests from logging fake key presses for several
        seconds while the character never moves.
        """
        if not self.target_hwnd or not user32.IsWindow(self.target_hwnd):
            return False, "游戏窗口句柄已失效，请重新启动程序并重新绑定游戏窗口。"
        target_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(self.target_hwnd, ctypes.byref(target_pid))
        current_pid = int(kernel32.GetCurrentProcessId())
        current_elevated = self._process_is_elevated(current_pid)
        target_elevated = self._process_is_elevated(int(target_pid.value))
        if target_elevated is True and current_elevated is False:
            return False, (
                "游戏客户端正在以管理员权限运行，但当前 Python/main.py 不是管理员。"
                "Windows 会拦截所有模拟按键；请关闭当前程序后，以管理员身份重新运行 main.py。"
            )
        if focus and self.input_mode == "foreground" and not self.ensure_focus():
            return False, "无法将输入焦点切换到游戏窗口，请确认游戏没有最小化或被权限拦截。"
        return True, "输入通道可用。"

    @staticmethod
    def _input_caller_source() -> str:
        """返回真正请求按键的上层调用点，供实机按键审计。"""
        try:
            frame = sys._getframe(2)
            if frame.f_code.co_name == "press_key" and frame.f_back is not None:
                frame = frame.f_back
            module = frame.f_globals.get("__name__", "unknown")
            return f"{module}.{frame.f_code.co_name}:{frame.f_lineno}"
        except Exception:
            return "unknown"

    def ensure_focus(self) -> bool:
        """确保游戏窗口获得焦点。如果是后台模式，无需抢夺前台焦点，直接返回 True。"""
        if self.input_mode == "background":
            return bool(self.target_hwnd and user32.IsWindow(self.target_hwnd))

        if not self.target_hwnd or not user32.IsWindow(self.target_hwnd):
            return False

        fg_hwnd = user32.GetForegroundWindow()
        if fg_hwnd == self.target_hwnd:
            return True

        # 1. 恢复最小化窗口
        if user32.IsIconic(self.target_hwnd):
            user32.ShowWindow(self.target_hwnd, 9) # SW_RESTORE
        else:
            user32.ShowWindow(self.target_hwnd, 5) # SW_SHOW

        # 2. 挂接线程输入队列以安全获取置顶与焦点切换权限
        cur_tid = kernel32.GetCurrentThreadId()
        fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
        target_tid = user32.GetWindowThreadProcessId(self.target_hwnd, None)

        if fg_tid and cur_tid != fg_tid:
            user32.AttachThreadInput(cur_tid, fg_tid, True)
        if target_tid and cur_tid != target_tid:
            user32.AttachThreadInput(cur_tid, target_tid, True)

        user32.BringWindowToTop(self.target_hwnd)
        user32.SetForegroundWindow(self.target_hwnd)
        user32.SetFocus(self.target_hwnd)

        if fg_tid and cur_tid != fg_tid:
            user32.AttachThreadInput(cur_tid, fg_tid, False)
        if target_tid and cur_tid != target_tid:
            user32.AttachThreadInput(cur_tid, target_tid, False)

        time.sleep(0.03)
        return user32.GetForegroundWindow() == self.target_hwnd

    def get_key_scancode(self, key_name: str, vk_code: Optional[int] = None) -> Tuple[int, bool]:
        """动态获取任意按键的硬件扫描码"""
        key_lower = key_name.lower().replace(" ", "").replace("_", "")
        if key_lower in SCAN_CODES:
            return SCAN_CODES[key_lower]

        # 动态通过 MapVirtualKey 转换
        if vk_code:
            scan = user32.MapVirtualKeyW(vk_code, 0)
            if scan > 0:
                is_ext = vk_code in (0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E)
                return scan, is_ext

        return 0, False

    def _send_scan_code(self, key_name: str, is_up: bool, vk_code: Optional[int] = None):
        k_clean = key_name.lower().replace(" ", "").replace("_", "")
        vk = vk_code or VK_MAP.get(k_clean)
        if not vk and len(k_clean) == 1:
            vk = ord(k_clean.upper())

        scan_code, is_ext = self.get_key_scancode(key_name, vk)
        if not scan_code and vk:
            scan_code = user32.MapVirtualKeyW(vk, 0)

        flags = 0
        if is_ext:
            flags |= KEYEVENTF_EXTENDEDKEY
        if is_up:
            flags |= KEYEVENTF_KEYUP

        if scan_code > 0:
            flags |= KEYEVENTF_SCANCODE
            # 对于 Alt/Menu，bVk 传入 0，彻底防止 Windows 系统菜单激活拦截
            actual_vk = 0 if k_clean in ("alt", "lalt", "ralt", "menu", "lmenu", "rmenu") else (vk or 0)
            # Older game clients may ignore a scan-only Alt event.
            # Preserve VK_MENU for Alt while keeping scan-code injection for
            # the rest of the keyboard.
            send_vk = 0x12 if k_clean in ("alt", "lalt", "alt_l", "menu") else 0
            inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(send_vk, scan_code, flags, 0, 0))
            kernel32.SetLastError(0)
            accepted = int(user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)))
            return {
                "requested_packets": 1,
                "accepted_packets": accepted,
                "win_error": int(kernel32.GetLastError()) if accepted != 1 else 0,
                "vk": int(vk or 0),
                "scan": int(scan_code),
                "flags": int(flags),
            }
        elif vk:
            inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(vk, 0, flags, 0, 0))
            kernel32.SetLastError(0)
            accepted = int(user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)))
            return {
                "requested_packets": 1,
                "accepted_packets": accepted,
                "win_error": int(kernel32.GetLastError()) if accepted != 1 else 0,
                "vk": int(vk),
                "scan": 0,
                "flags": int(flags),
            }
        return {
            "requested_packets": 1,
            "accepted_packets": 0,
            "win_error": int(kernel32.GetLastError()),
            "vk": int(vk or 0),
            "scan": int(scan_code or 0),
            "flags": int(flags),
        }

    @staticmethod
    def _send_mouse_packet(
        dx: int,
        dy: int,
        flags: int,
        mouse_data: int = 0,
    ) -> bool:
        packet = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                int(dx), int(dy), int(mouse_data) & 0xFFFFFFFF,
                int(flags), 0, 0,
            ),
        )
        return bool(user32.SendInput(1, ctypes.byref(packet), ctypes.sizeof(INPUT)))

    @classmethod
    def _move_mouse_absolute(cls, screen_x: int, screen_y: int) -> bool:
        """用本模块自己的 INPUT 结构移动鼠标，避免污染 pydirectinput 类型。"""
        virtual_x = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        virtual_y = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        virtual_w = max(1, int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
        virtual_h = max(1, int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
        absolute_x = round((int(screen_x) - virtual_x) * 65535 / max(1, virtual_w - 1))
        absolute_y = round((int(screen_y) - virtual_y) * 65535 / max(1, virtual_h - 1))
        absolute_x = max(0, min(65535, absolute_x))
        absolute_y = max(0, min(65535, absolute_y))
        return cls._send_mouse_packet(
            absolute_x,
            absolute_y,
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        )

    @staticmethod
    def _send_unicode_text(value: str, interval_sec: float) -> bool:
        """使用 KEYEVENTF_UNICODE 输入文本，不依赖第三方 INPUT 结构。"""
        encoded = str(value).encode("utf-16-le")
        ok = True
        for offset in range(0, len(encoded), 2):
            code_unit = int.from_bytes(encoded[offset:offset + 2], "little")
            down = INPUT(
                type=INPUT_KEYBOARD,
                ki=KEYBDINPUT(0, code_unit, KEYEVENTF_UNICODE, 0, 0),
            )
            up = INPUT(
                type=INPUT_KEYBOARD,
                ki=KEYBDINPUT(0, code_unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0),
            )
            ok = bool(user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))) and ok
            ok = bool(user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))) and ok
            time.sleep(max(0.0, float(interval_sec)))
        return ok

    def _send_text_keystrokes(self, value: str, interval_sec: float) -> bool:
        """按当前键盘布局发送文本扫描码，同时绕开按键审计以保护密码。"""
        ok = True
        for char in str(value):
            key_info = int(user32.VkKeyScanW(ord(char)))
            if key_info == -1 or (key_info & 0xFFFF) == 0xFFFF:
                ok = self._send_unicode_text(char, 0.0) and ok
                time.sleep(max(0.0, float(interval_sec)))
                continue
            vk = key_info & 0xFF
            modifier_bits = (key_info >> 8) & 0xFF
            modifiers = []
            if modifier_bits & 0x01:
                modifiers.append("shift")
            if modifier_bits & 0x02:
                modifiers.append("ctrl")
            if modifier_bits & 0x04:
                modifiers.append("alt")
            try:
                for modifier in modifiers:
                    self._send_scan_code(modifier, is_up=False)
                self._send_scan_code(char, is_up=False, vk_code=vk)
                time.sleep(0.012)
                self._send_scan_code(char, is_up=True, vk_code=vk)
            finally:
                for modifier in reversed(modifiers):
                    self._send_scan_code(modifier, is_up=True)
            time.sleep(max(0.0, float(interval_sec)))
        return ok

    def _send_post_message(self, key_name: str, is_up: bool, vk_code: Optional[int] = None):
        """使用 Windows 消息队列向目标窗口后台投递按键 (WM_KEYDOWN / WM_KEYUP / WM_SYSKEYDOWN / WM_SYSKEYUP)"""
        if not self.target_hwnd or not user32.IsWindow(self.target_hwnd):
            return {
                "requested_packets": 1,
                "accepted_packets": 0,
                "win_error": int(kernel32.GetLastError()),
                "vk": 0,
                "scan": 0,
                "flags": 0,
            }

        k_clean = key_name.lower().replace(" ", "").replace("_", "")
        vk = vk_code or VK_MAP.get(k_clean)
        if not vk and len(k_clean) == 1:
            vk = ord(k_clean.upper())

        scan_code, is_ext = self.get_key_scancode(key_name, vk)
        if not scan_code and vk:
            scan_code = user32.MapVirtualKeyW(vk, 0)

        is_alt = k_clean in ("alt", "lalt", "ralt", "menu", "lmenu", "rmenu", "alt_l", "alt_r")
        ext_flag = 1 if is_ext else 0

        if not is_up:
            lparam = 1 | (scan_code << 16) | (ext_flag << 24)
            if is_alt:
                lparam |= (1 << 29)  # Context code (Alt down)
                accepted = int(bool(user32.PostMessageW(self.target_hwnd, WM_SYSKEYDOWN, vk or 0x12, lparam)))
                accepted += int(bool(user32.PostMessageW(self.target_hwnd, WM_KEYDOWN, vk or 0x12, 1 | (scan_code << 16) | (ext_flag << 24))))
                requested = 2
            else:
                accepted = int(bool(user32.PostMessageW(self.target_hwnd, WM_KEYDOWN, vk or 0, lparam)))
                requested = 1
        else:
            lparam = 1 | (scan_code << 16) | (ext_flag << 24) | (1 << 30) | (1 << 31)
            if is_alt:
                lparam |= (1 << 29)  # Context code
                accepted = int(bool(user32.PostMessageW(self.target_hwnd, WM_SYSKEYUP, vk or 0x12, lparam)))
                accepted += int(bool(user32.PostMessageW(self.target_hwnd, WM_KEYUP, vk or 0x12, 1 | (scan_code << 16) | (ext_flag << 24) | (1 << 30) | (1 << 31))))
                requested = 2
            else:
                accepted = int(bool(user32.PostMessageW(self.target_hwnd, WM_KEYUP, vk or 0, lparam)))
                requested = 1
        return {
            "requested_packets": requested,
            "accepted_packets": accepted,
            "win_error": int(kernel32.GetLastError()) if accepted != requested else 0,
            "vk": int(vk or 0),
            "scan": int(scan_code or 0),
            "flags": int(lparam),
        }

    def _finalize_delivery_audit(
        self, key: str, is_down: bool, transport_result: Optional[Dict]
    ) -> None:
        if not self.delivery_audit_enabled:
            self.last_delivery_result = None
            return
        expected_active = sorted(self.active_keys)
        os_down = []
        os_state = {}
        if self.input_mode == "foreground":
            for active_key in expected_active:
                vk = VK_MAP.get(active_key)
                if not vk:
                    continue
                down = bool(user32.GetAsyncKeyState(int(vk)) & 0x8000)
                os_state[active_key] = down
                if down:
                    os_down.append(active_key)
        released_vk = VK_MAP.get(key)
        released_still_down = bool(
            self.input_mode == "foreground"
            and
            released_vk
            and not is_down
            and (user32.GetAsyncKeyState(int(released_vk)) & 0x8000)
        )
        result = dict(transport_result or {})
        result.update(
            {
                "mode": self.input_mode,
                "key": key,
                "edge": "down" if is_down else "up",
                "foreground_hwnd": int(user32.GetForegroundWindow() or 0),
                "target_hwnd": int(self.target_hwnd or 0),
                "foreground_matches": bool(
                    self.target_hwnd
                    and user32.GetForegroundWindow() == self.target_hwnd
                ),
                "expected_active": expected_active,
                "os_down": os_down,
                "os_state": os_state,
                "released_still_down": released_still_down,
                "monotonic_ns": time.perf_counter_ns(),
            }
        )
        result["transport_ok"] = bool(
            int(result.get("accepted_packets", 0))
            == int(result.get("requested_packets", 1))
        )
        result["active_state_ok"] = (
            bool(
                all(
                    os_state.get(item, False)
                    for item in expected_active
                    if item in VK_MAP
                )
                and not released_still_down
            )
            if self.input_mode == "foreground"
            else None
        )
        self.last_delivery_result = result

    def key_down(self, key: str, vk_code: Optional[int] = None):
        """按下按键并保持"""
        self.last_event_source = self._input_caller_source()
        key_clean = key.lower()
        if self.input_mode == "foreground":
            self.ensure_focus()
            delivery = self._send_scan_code(key_clean, is_up=False, vk_code=vk_code)
        else:
            delivery = self._send_post_message(key_clean, is_up=False, vk_code=vk_code)

        self.active_keys.add(key_clean)
        self._finalize_delivery_audit(key_clean, True, delivery)
        if key_clean == "left":
            self.current_facing = "left"
        elif key_clean == "right":
            self.current_facing = "right"
        if self.key_event_callback:
            try:
                self.key_event_callback(key_clean, True, time.perf_counter())
            except Exception:
                pass

    def key_up(self, key: str, vk_code: Optional[int] = None):
        """松开按键"""
        self.last_event_source = self._input_caller_source()
        key_clean = key.lower()
        if self.input_mode == "foreground":
            delivery = self._send_scan_code(key_clean, is_up=True, vk_code=vk_code)
        else:
            delivery = self._send_post_message(key_clean, is_up=True, vk_code=vk_code)

        self.active_keys.discard(key_clean)
        self._finalize_delivery_audit(key_clean, False, delivery)
        if self.key_event_callback:
            try:
                self.key_event_callback(key_clean, False, time.perf_counter())
            except Exception:
                pass

    def send_recorded_key_event(
        self,
        *,
        key_name: str,
        vk_code: int,
        scan_code: int,
        is_up: bool,
        extended: bool,
        notify: bool = False,
    ) -> Dict:
        """Replay one recorded physical key edge without remapping its scan code.

        The caller schedules edges against an absolute clock.  ``notify`` is
        disabled during high-fidelity replay so GUI logging cannot stretch a
        short manual Down/Alt interval; delivery results are returned and can
        be logged after the complete sequence.
        """
        key_clean = str(key_name).lower()
        flags = KEYEVENTF_SCANCODE if int(scan_code) > 0 else 0
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        if is_up:
            flags |= KEYEVENTF_KEYUP
        # Match the proven foreground path: legacy clients can ignore a
        # scan-only Alt edge, while other keys should retain wVk=0 when a scan
        # code is supplied.
        send_vk = (
            (int(vk_code) or 0x12)
            if key_clean in ("alt", "lalt", "alt_l", "alt_r", "ralt", "menu")
            else (int(vk_code) if int(scan_code) <= 0 else 0)
        )
        packet = INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(send_vk, int(scan_code), int(flags), 0, 0),
        )
        kernel32.SetLastError(0)
        accepted = int(user32.SendInput(1, ctypes.byref(packet), ctypes.sizeof(INPUT)))
        delivery = {
            "requested_packets": 1,
            "accepted_packets": accepted,
            "win_error": int(kernel32.GetLastError()) if accepted != 1 else 0,
            "vk": int(vk_code),
            "scan": int(scan_code),
            "flags": int(flags),
            "recorded_replay": True,
        }
        if is_up:
            self.active_keys.discard(key_clean)
        else:
            self.active_keys.add(key_clean)
            if key_clean == "left":
                self.current_facing = "left"
            elif key_clean == "right":
                self.current_facing = "right"
        self.last_event_source = "manual_input_recording.replay"
        self._finalize_delivery_audit(key_clean, not is_up, delivery)
        result = dict(self.last_delivery_result or delivery)
        if notify and self.key_event_callback:
            try:
                self.key_event_callback(key_clean, not is_up, time.perf_counter())
            except Exception:
                pass
        return result

    def press_key(self, key: str, duration_ms: Optional[int] = None, vk_code: Optional[int] = None):
        """单次按下并释放按键，自带 45~75ms 拟人化随机延时"""
        if duration_ms is None:
            duration_ms = random.randint(45, 75)
        self.key_down(key, vk_code=vk_code)
        time.sleep(duration_ms / 1000.0)
        self.key_up(key, vk_code=vk_code)

    def click_client(self, x: int, y: int) -> bool:
        """点击目标窗口客户区坐标，兼容后台 PostMessage 与前台模式。"""
        if not self.target_hwnd or not user32.IsWindow(self.target_hwnd):
            return False
        x = max(0, int(round(x)))
        y = max(0, int(round(y)))
        if self.input_mode == "background":
            lparam = (y << 16) | (x & 0xFFFF)
            user32.PostMessageW(self.target_hwnd, WM_MOUSEMOVE, 0, lparam)
            user32.PostMessageW(self.target_hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
            time.sleep(0.045)
            user32.PostMessageW(self.target_hwnd, WM_LBUTTONUP, 0, lparam)
            return True

        if not self.ensure_focus():
            return False
        point = wintypes.POINT(x, y)
        if not user32.ClientToScreen(self.target_hwnd, ctypes.byref(point)):
            return False
        moved = self._move_mouse_absolute(int(point.x), int(point.y))
        time.sleep(0.02)
        pressed = self._send_mouse_packet(0, 0, MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.045)
        released = self._send_mouse_packet(0, 0, MOUSEEVENTF_LEFTUP)
        return bool(moved and pressed and released)

    def type_text(self, text: str, interval_sec: float = 0.025) -> bool:
        """向当前输入框输入文本；调用方不得把 ``text`` 写入日志。"""
        if not self.target_hwnd or not user32.IsWindow(self.target_hwnd):
            return False
        value = str(text)
        if self.input_mode == "background":
            for char in value:
                user32.PostMessageW(self.target_hwnd, WM_CHAR, ord(char), 1)
                time.sleep(max(0.0, float(interval_sec)))
            return True
        if not self.ensure_focus():
            return False
        return self._send_text_keystrokes(value, interval_sec)

    def replace_text(self, text: str) -> bool:
        """清空当前编辑框后输入新内容，兼容不支持 Ctrl+A 的游戏框。"""
        self.key_down("ctrl")
        try:
            self.press_key("a", duration_ms=45)
        finally:
            self.key_up("ctrl")
        time.sleep(0.04)
        # 游戏输入框可能忽略 Ctrl+A；显式删除选中内容，再从
        # 末尾退格清空历史密码。此方法仅在重连密码框被聚焦后调用。
        self.press_key("backspace", duration_ms=45)
        self.press_key("end", duration_ms=35)
        for _ in range(48):
            self.key_down("backspace")
            time.sleep(0.012)
            self.key_up("backspace")
            time.sleep(0.012)
        return self.type_text(text)

    def mouse_wheel_client(self, x: int, y: int, steps: int) -> bool:
        """在客户区指定位置滚动；正数向上、负数向下。"""
        if not self.target_hwnd or not user32.IsWindow(self.target_hwnd):
            return False
        steps = int(steps)
        if steps == 0:
            return True
        point = wintypes.POINT(max(0, int(x)), max(0, int(y)))
        if not user32.ClientToScreen(self.target_hwnd, ctypes.byref(point)):
            return False
        if self.input_mode == "background":
            lparam = ((int(point.y) & 0xFFFF) << 16) | (int(point.x) & 0xFFFF)
            direction = 120 if steps > 0 else -120
            for _ in range(abs(steps)):
                wparam = (direction & 0xFFFF) << 16
                user32.PostMessageW(self.target_hwnd, WM_MOUSEWHEEL, wparam, lparam)
                time.sleep(0.04)
            return True
        if not self.ensure_focus():
            return False
        ok = self._move_mouse_absolute(int(point.x), int(point.y))
        direction = WHEEL_DELTA if steps > 0 else -WHEEL_DELTA
        for _ in range(abs(steps)):
            ok = self._send_mouse_packet(
                0, 0, MOUSEEVENTF_WHEEL, direction
            ) and ok
            time.sleep(0.04)
        return bool(ok)

    def release_tracked_keys(self):
        """只释放本输入驱动实际按下且尚未释放的键。

        常规动作收束不应该广播所有方向键的 KEYUP：如果玩家此时
        正手动按住方向键，高频广播会把物理输入不断抬起，表现为“连
        手动也动不了”。完整急停仍由 :meth:`release_all_keys` 执行。
        """
        for k in list(self.active_keys):
            try:
                if self.input_mode == "foreground":
                    delivery = self._send_scan_code(k, is_up=True)
                else:
                    delivery = self._send_post_message(k, is_up=True)
                self.active_keys.discard(k)
                self._finalize_delivery_audit(k, False, delivery)
                if self.key_event_callback:
                    self.key_event_callback(k, False, time.perf_counter())
            except Exception:
                pass
        self.active_keys.clear()

    def release_all_keys(self):
        """急停释放所有按键 (包含所有修饰键与方向键，彻底杜绝按键残留与键盘失效)"""
        # 1. 先释放动态跟踪的按键
        self.release_tracked_keys()

        # 2. 全局强制重置所有核心按键与修饰键 (Alt/Ctrl/Shift/方向键)
        essential_vks = [
            (0x12, "alt"),
            (0x11, "ctrl"),
            (0x10, "shift"),
            (0x25, "left"),
            (0x26, "up"),
            (0x27, "right"),
            (0x28, "down"),
            (0x20, "space"),
            (0x5A, "z"),
            (0x58, "x"),
            (0x43, "c"),
            (0x56, "v"),
        ]
        if self.input_mode == "foreground":
            for vk, _ in essential_vks:
                try:
                    scan = user32.MapVirtualKeyW(vk, 0)
                    # 对 Alt/Ctrl/Shift 保留 VK 码，防止游戏客户端忽略 pure scancode KEYUP
                    send_vk = vk if vk in (0x12, 0x11, 0x10) else 0
                    for flags in (KEYEVENTF_KEYUP | KEYEVENTF_SCANCODE,
                                  KEYEVENTF_KEYUP | KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY):
                        inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(send_vk, scan, flags, 0, 0))
                        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
                except Exception:
                    pass
        else:
            for vk, name in essential_vks:
                try:
                    self._send_post_message(name, is_up=True, vk_code=vk)
                except Exception:
                    pass
