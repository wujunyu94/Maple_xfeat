"""Shared adaptive sizing helpers for Tk windows.

The GUI runs on Windows most of the time, where ``winfo_screenheight`` includes
the taskbar and may describe a different monitor than the window's parent.
These helpers always prefer the work area of the monitor containing the parent
and keep both the initial size and the minimum size inside that area.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Optional, Tuple
import tkinter as tk


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", _Rect),
        ("rcWork", _Rect),
        ("dwFlags", ctypes.c_ulong),
    ]


def get_work_area(widget: tk.Misc) -> Tuple[int, int, int, int]:
    """Return ``(x, y, width, height)`` for the widget's nearest monitor."""
    try:
        top = widget.winfo_toplevel()
        top.update_idletasks()
        hwnd = int(top.winfo_id())
        user32 = ctypes.windll.user32
        user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
        user32.MonitorFromWindow.restype = wintypes.HANDLE
        user32.GetMonitorInfoW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_MonitorInfo))
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        monitor = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        info = _MonitorInfo()
        info.cbSize = ctypes.sizeof(_MonitorInfo)
        if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            rect = info.rcWork
            width = int(rect.right - rect.left)
            height = int(rect.bottom - rect.top)
            if width > 0 and height > 0:
                return int(rect.left), int(rect.top), width, height
    except Exception:
        pass

    try:
        x = int(widget.winfo_vrootx())
        y = int(widget.winfo_vrooty())
        width = int(widget.winfo_vrootwidth())
        height = int(widget.winfo_vrootheight())
        if width > 0 and height > 0:
            return x, y, width, height
    except Exception:
        pass
    return 0, 0, int(widget.winfo_screenwidth()), int(widget.winfo_screenheight())


def fit_window_to_work_area(
    window: tk.Toplevel | tk.Tk,
    preferred: Tuple[int, int],
    minimum: Tuple[int, int] = (480, 320),
    *,
    parent: Optional[tk.Misc] = None,
    width_fraction: float = 0.94,
    height_fraction: float = 0.90,
) -> Tuple[int, int]:
    """Size and center a window without allowing it under the taskbar.

    The returned tuple is the actual client size selected for the window.  A
    minimum is still installed, but it is clamped as well so a small monitor is
    never trapped by an impossible ``minsize``.
    """
    anchor = parent or window
    work_x, work_y, work_w, work_h = get_work_area(anchor)
    max_w = max(320, min(work_w, int(round(work_w * width_fraction))))
    max_h = max(240, min(work_h, int(round(work_h * height_fraction))))

    width = max(320, min(int(preferred[0]), max_w))
    height = max(240, min(int(preferred[1]), max_h))
    min_w = max(240, min(int(minimum[0]), width, max_w))
    min_h = max(180, min(int(minimum[1]), height, max_h))
    window.minsize(min_w, min_h)

    center_x = work_x + work_w // 2
    center_y = work_y + work_h // 2
    if parent is not None:
        try:
            parent.update_idletasks()
            pw = int(parent.winfo_width())
            ph = int(parent.winfo_height())
            if pw > 1 and ph > 1:
                center_x = int(parent.winfo_rootx()) + pw // 2
                center_y = int(parent.winfo_rooty()) + ph // 2
        except Exception:
            pass

    x = max(work_x, min(center_x - width // 2, work_x + work_w - width))
    y = max(work_y, min(center_y - height // 2, work_y + work_h - height))
    window.geometry(f"{width}x{height}+{x}+{y}")
    return width, height


def fit_image_to_work_area(
    parent: tk.Misc,
    image_width: int,
    image_height: int,
    *,
    reserved_width: int = 48,
    reserved_height: int = 160,
    width_fraction: float = 0.94,
    height_fraction: float = 0.90,
    allow_upscale: bool = False,
) -> Tuple[int, int, float]:
    """Fit an image into a dialog's available body while preserving aspect."""
    _, _, work_w, work_h = get_work_area(parent)
    available_w = max(240, int(work_w * width_fraction) - reserved_width)
    available_h = max(180, int(work_h * height_fraction) - reserved_height)
    scale = min(
        available_w / max(1.0, float(image_width)),
        available_h / max(1.0, float(image_height)),
    )
    if not allow_upscale:
        scale = min(1.0, scale)
    scale = max(0.05, scale)
    width = max(1, int(round(image_width * scale)))
    height = max(1, int(round(image_height * scale)))
    return width, height, scale
