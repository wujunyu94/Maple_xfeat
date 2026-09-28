"""
window.py - 游戏窗口管理与坐标定位模块
负责探测目标游戏客户端窗口、获取客户区绝对坐标及窗口状态控制。
"""

import sys
import os
import ctypes
from ctypes import wintypes
import psutil
from typing import Optional, Tuple, Dict, Any

# 设置进程 DPI 感知以获取准确的真实屏幕像素坐标
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # Per-Monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class WindowManager:
    def __init__(self, target_process_name: str = "Maplestory_Classic.exe", target_window_title: str = "冒险岛怀旧服"):
        self.target_process_name = target_process_name.lower()
        self.target_window_title = target_window_title
        self.hwnd: Optional[int] = None
        self.pid: Optional[int] = None
        self._ensure_desktop_access()

    def _ensure_desktop_access(self):
        """确保在各种会话环境下具备交互式桌面的访问权限"""
        try:
            h_input_desk = user32.OpenInputDesktop(0, False, 0x01FF)
            if h_input_desk:
                user32.SetThreadDesktop(h_input_desk)
        except Exception:
            pass

    def find_game_window(self) -> Optional[int]:
        """
        探测并返回游戏窗口句柄 (HWND)。
        按进程名称、窗口类名以及窗口标题进行多重匹配。
        """
        self._ensure_desktop_access()
        matched_hwnd = None

        def enum_cb(hwnd, lparam):
            nonlocal matched_hwnd
            if not user32.IsWindowVisible(hwnd):
                return 1

            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            
            pname = ""
            try:
                pname = psutil.Process(pid.value).name().lower()
            except Exception:
                pass

            title_buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title_buf, 256)
            title = title_buf.value

            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls_buf, 256)
            cls_name = cls_buf.value

            # 排除自身 GUI 进程、当前 PID、以及 IDE / 调试器 / 浏览器窗口
            current_pid = os.getpid()
            if pid.value == current_pid or "python" in pname or "antigravity" in pname or "code" in pname or "chrome" in pname or "edge" in pname:
                return 1
            if "vision center" in title.lower() or cls_name == "TkTopLevel" or "antigravity" in title.lower():
                return 1

            # 优先 1: 精确匹配目标游戏进程名
            if self.target_process_name in pname:
                rect = (ctypes.c_long * 4)()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w >= 400 and h >= 300:
                    matched_hwnd = hwnd
                    self.pid = pid.value
                    return 0 # 精确锁定游戏主进程窗口

            # 优先 2: 匹配游戏主窗口类名或标题
            is_target_title = bool(self.target_window_title and self.target_window_title.lower() in title.lower())
            is_proc_match = ("classic" in pname or "maple" in pname or "mxd" in pname)
            is_title_match = (is_target_title or "冒险岛" in title or "Maple" in title)
            is_unity_class = (cls_name == "UnityWndClass" or cls_name == "MapleStoryClass")

            if (is_target_title or is_proc_match or is_title_match or is_unity_class) and not "setup" in pname:
                rect = (ctypes.c_long * 4)()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w >= 400 and h >= 300:
                    matched_hwnd = hwnd
                    self.pid = pid.value
                    return 0 # 找到目标，终止枚举
            return 1

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        self.hwnd = matched_hwnd
        return self.hwnd

    def list_visible_windows(self) -> list[Dict[str, Any]]:
        """返回可供手动选择的可见窗口，避免依赖固定游戏标题。"""
        windows: list[Dict[str, Any]] = []

        def enum_cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return 1
            rect = (ctypes.c_long * 4)()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            width, height = rect[2] - rect[0], rect[3] - rect[1]
            if width < 400 or height < 300:
                return 1
            title_buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title_buf, 256)
            title = title_buf.value.strip()
            if not title:
                return 1
            pid_buf = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buf))
            pid = int(pid_buf.value)
            pname = ""
            try:
                pname = psutil.Process(pid).name()
            except Exception:
                pass
            cls_buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls_buf, 256)
            windows.append({
                "hwnd": int(hwnd), "title": title, "pid": pid,
                "process": pname, "class_name": cls_buf.value,
                "width": int(width), "height": int(height),
            })
            return 1

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        return windows

    def select_window(self, hwnd: int, title: Optional[str] = None) -> bool:
        """绑定用户选择的窗口句柄，并同步 WGC 所需标题。"""
        if not hwnd or not user32.IsWindow(hwnd):
            return False
        self.hwnd = int(hwnd)
        pid_buf = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buf))
        self.pid = int(pid_buf.value)
        if title is None:
            title_buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title_buf, 256)
            title = title_buf.value
        self.target_window_title = title or self.target_window_title
        return True

    def is_valid(self) -> bool:
        """检查当前窗口句柄是否依然有效且可见"""
        if not self.hwnd:
            return False
        return bool(user32.IsWindow(self.hwnd) and user32.IsWindowVisible(self.hwnd))

    def get_client_rect(self) -> Optional[Dict[str, int]]:
        """
        获取游戏窗口客户区在屏幕上的绝对坐标与尺寸。
        返回格式: {'left': int, 'top': int, 'width': int, 'height': int}
        """
        if not self.is_valid():
            if not self.find_game_window():
                return None

        # 获取客户区尺寸 (0, 0, width, height)
        crect = (ctypes.c_long * 4)()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(crect)):
            return None

        width = crect[2] - crect[0]
        height = crect[3] - crect[1]

        # 将客户区原点 (0, 0) 转换为屏幕物理坐标
        pt = (ctypes.c_long * 2)(0, 0)
        if not user32.ClientToScreen(self.hwnd, ctypes.byref(pt)):
            return None

        return {
            "left": int(pt[0]),
            "top": int(pt[1]),
            "width": int(width),
            "height": int(height),
            "right": int(pt[0] + width),
            "bottom": int(pt[1] + height)
        }

    def focus(self) -> bool:
        """安全激活并聚焦游戏窗口 (使用 AttachThreadInput，杜绝模拟 Alt 导致角色误跳)"""
        if not self.is_valid():
            if not self.find_game_window():
                return False

        try:
            fg_hwnd = user32.GetForegroundWindow()
            if fg_hwnd == self.hwnd:
                return True

            if user32.IsIconic(self.hwnd):
                user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
            else:
                user32.ShowWindow(self.hwnd, 5)  # SW_SHOW

            cur_tid = kernel32.GetCurrentThreadId()
            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
            target_tid = user32.GetWindowThreadProcessId(self.hwnd, None)

            if fg_tid and cur_tid != fg_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, True)
            if target_tid and cur_tid != target_tid:
                user32.AttachThreadInput(cur_tid, target_tid, True)

            user32.BringWindowToTop(self.hwnd)
            user32.SetForegroundWindow(self.hwnd)
            user32.SetFocus(self.hwnd)

            if fg_tid and cur_tid != fg_tid:
                user32.AttachThreadInput(cur_tid, fg_tid, False)
            if target_tid and cur_tid != target_tid:
                user32.AttachThreadInput(cur_tid, target_tid, False)

            return user32.GetForegroundWindow() == self.hwnd
        except Exception:
            return False
