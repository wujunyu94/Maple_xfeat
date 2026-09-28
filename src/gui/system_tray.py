"""Windows notification-area icon for hiding and restoring the main Tk window."""

import queue
import threading

import win32api
import win32con
import win32gui


class SystemTray:
    _CALLBACK_MESSAGE = win32con.WM_USER + 41
    _RESTORE_ID = 1001
    _EXIT_ID = 1002

    def __init__(self, events: queue.Queue):
        self.events = events
        self._thread = None
        self._hwnd = 0
        self._window_proc = self._on_message  # Keep the WNDPROC callable alive.

    def show(self) -> None:
        if self._thread and self._thread.is_alive():
            if self._hwnd:
                self.events.put(("ready", ""))
            return
        self._thread = threading.Thread(target=self._run, name="SystemTray", daemon=True)
        self._thread.start()

    def close(self) -> None:
        hwnd = self._hwnd
        if hwnd:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    def _run(self) -> None:
        hwnd = 0
        added = False
        try:
            instance = win32api.GetModuleHandle(None)
            window_class = win32gui.WNDCLASS()
            window_class.hInstance = instance
            window_class.lpszClassName = f"MapleVisionTray_{id(self)}"
            window_class.lpfnWndProc = self._window_proc
            class_atom = win32gui.RegisterClass(window_class)
            hwnd = win32gui.CreateWindow(
                class_atom, "Maple Vision Tray", 0, 0, 0, 0, 0,
                0, 0, instance, None,
            )
            self._hwnd = hwnd
            icon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
            notify = (
                hwnd, 0,
                win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP,
                self._CALLBACK_MESSAGE, icon, "游戏视觉中枢",
            )
            win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, notify)
            added = True
            self.events.put(("ready", ""))
            win32gui.PumpMessages()
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if added:
                try:
                    win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (hwnd, 0))
                except Exception:
                    pass
            self._hwnd = 0

    def _on_message(self, hwnd, message, wparam, lparam):
        if message == self._CALLBACK_MESSAGE:
            if lparam == win32con.WM_LBUTTONDBLCLK:
                self.events.put(("restore", ""))
            elif lparam in (win32con.WM_RBUTTONUP, win32con.WM_CONTEXTMENU):
                self._show_menu(hwnd)
            return 0
        if message == win32con.WM_COMMAND:
            command = wparam & 0xFFFF
            if command == self._RESTORE_ID:
                self.events.put(("restore", ""))
            elif command == self._EXIT_ID:
                self.events.put(("exit", ""))
            return 0
        if message == win32con.WM_CLOSE:
            win32gui.DestroyWindow(hwnd)
            return 0
        if message == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, message, wparam, lparam)

    def _show_menu(self, hwnd) -> None:
        menu = win32gui.CreatePopupMenu()
        try:
            win32gui.AppendMenu(menu, win32con.MF_STRING, self._RESTORE_ID, "恢复窗口")
            win32gui.AppendMenu(menu, win32con.MF_STRING, self._EXIT_ID, "退出程序")
            x, y = win32gui.GetCursorPos()
            win32gui.SetForegroundWindow(hwnd)
            win32gui.TrackPopupMenu(
                menu, win32con.TPM_LEFTALIGN | win32con.TPM_BOTTOMALIGN,
                x, y, 0, hwnd, None,
            )
            win32gui.PostMessage(hwnd, win32con.WM_NULL, 0, 0)
        finally:
            win32gui.DestroyMenu(menu)
