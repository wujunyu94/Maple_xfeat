"""Asynchronous client capture with a reusable DIB and independent BGR frames.

PrintWindow is preferred where supported; legacy clients fall back to a
cached client-DC BitBlt. Background visibility depends on the selected mode.
"""

import time
import threading
import ctypes
from ctypes import wintypes
import numpy as np
import cv2
from typing import Callable, Optional, Tuple, Dict
from src.core.window import WindowManager

try:
    from windows_capture import WindowsCapture
except Exception:
    WindowsCapture = None

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002  # Win8.1+ 支持，捕获 DWM 渲染内容


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


class DIBSectionSurface:
    """持久化零拷贝 DIBSection 显存表面"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # 负值 = 自上而下
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0  # BI_RGB
        
        self.screen_dc = user32.GetDC(0)
        self.mem_dc = gdi32.CreateCompatibleDC(self.screen_dc)
        self.p_bits = ctypes.c_void_p()
        self.hbmp = gdi32.CreateDIBSection(
            self.screen_dc, ctypes.byref(bmi), 0, ctypes.byref(self.p_bits), None, 0
        )
        self.old_bmp = gdi32.SelectObject(self.mem_dc, self.hbmp)
        
        # 将 DIB 内存指针直接包装为 numpy 视图 (Zero-Copy)
        self.raw_view = np.ctypeslib.as_array(
            ctypes.cast(self.p_bits.value, ctypes.POINTER(ctypes.c_uint8)),
            shape=(height, width, 4)
        )
        # PrintWindow is unsupported by some DirectX clients. Remember that
        # per HWND instead of paying for three failed calls on every frame.
        self._capture_hwnd = None
        self._capture_mode = "auto"
        self._print_flags = None

    def _bitblt_client(self, hwnd: int) -> bool:
        client_dc = user32.GetDC(hwnd)
        if not client_dc:
            return False
        try:
            return bool(gdi32.BitBlt(
                self.mem_dc, 0, 0, self.width, self.height,
                client_dc, 0, 0, 0x00CC0020,
            ))
        finally:
            user32.ReleaseDC(hwnd, client_dc)

    def capture(self, hwnd: int) -> Optional[np.ndarray]:
        """Capture a client frame and return a view valid until the next capture.

        Cache the first successful method per HWND to avoid repeating failed
        PrintWindow calls on clients that only support BitBlt.
        """
        if hwnd != self._capture_hwnd:
            self._capture_hwnd = hwnd
            self._capture_mode = "auto"
            self._print_flags = None
        if self._capture_mode == "bitblt":
            if self._bitblt_client(hwnd):
                return self.raw_view[:, :, :3]
            self._capture_mode = "auto"

        flags = (PW_CLIENTONLY | PW_RENDERFULLCONTENT, PW_CLIENTONLY, 0)
        if self._print_flags is not None:
            flags = (self._print_flags,) + tuple(
                flag for flag in flags if flag != self._print_flags
            )
        ok = False
        for flag in flags:
            if user32.PrintWindow(hwnd, self.mem_dc, flag):
                self._capture_mode = "print"
                self._print_flags = flag
                ok = True
                break
        # DirectX clients may reject every PrintWindow variant. Once proven,
        # use the same successful client-DC fallback directly on future frames.
        if not ok:
            ok = self._bitblt_client(hwnd)
            if ok:
                self._capture_mode = "bitblt"
                self._print_flags = None
        if not ok:
            return None
        return self.raw_view[:, :, :3]

    def capture_screen(self, left: int, top: int) -> Optional[np.ndarray]:
        """Capture the composited desktop client area for visible login overlays."""
        if not gdi32.BitBlt(
            self.mem_dc, 0, 0, self.width, self.height,
            self.screen_dc, int(left), int(top), 0x40CC0020,
        ):
            return None
        return self.raw_view[:, :, :3]

    def release(self):
        try:
            if self.mem_dc:
                gdi32.SelectObject(self.mem_dc, self.old_bmp)
                gdi32.DeleteObject(self.hbmp)
                gdi32.DeleteDC(self.mem_dc)
            if self.screen_dc:
                user32.ReleaseDC(0, self.screen_dc)
        except Exception:
            pass


def capture_hwnd(hwnd: int) -> Optional[np.ndarray]:
    """单次快速捕获窗口"""
    if not hwnd or not user32.IsWindow(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None
    
    surface = DIBSectionSurface(w, h)
    frame = surface.capture(hwnd)
    result = cv2.cvtColor(surface.raw_view, cv2.COLOR_BGRA2BGR) if frame is not None else None
    surface.release()
    return result


class ScreenCapture:
    def __init__(self, window_mgr: Optional[WindowManager] = None, use_async_thread: bool = True,
                 prefer_wgc: bool = True, max_fps: float = 60.0):
        self.window_mgr = window_mgr or WindowManager()
        self.use_async_thread = use_async_thread
        self.prefer_wgc = prefer_wgc
        self.max_fps = max(1.0, float(max_fps))
        self.backend = "pending"
        self._foreground_login_capture = False
        self._frame_overlay_provider: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
        self._overlay_provider_lock = threading.Lock()
        self._timing_callback: Optional[Callable[[str, float, float, float], None]] = None

        self._latest_frame: Optional[np.ndarray] = None
        self.frame_seq: int = 0
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # 每次切换窗口都递增。旧 WGC/GDI 线程只要发现代际不一致就自行
        # 退出，切换操作不在 Tk 主线程 join，避免窗口选择后“未响应”。
        self._capture_epoch = 0
        self._wgc_control = None
        # ``start_free_threaded`` returns immediately, so retain the wrapper;
        # otherwise its native capture may be finalized while callbacks run.
        # Old generations are deliberately retained because stopping WGC while
        # rebinding has crashed some legacy DirectX clients in practice.
        self._wgc_capture_refs = []
        self._wgc_raw_frame: Optional[np.ndarray] = None
        self._wgc_raw_lock = threading.Lock()
        self._wgc_raw_seq = 0
        self._wgc_converted_seq = 0
        self._wgc_last_accept = 0.0
        self.wgc_accepted_count = 0
        self.wgc_converted_count = 0
        self.wgc_last_error = ""
        self._wgc_convert_thread: Optional[threading.Thread] = None

        self._frame_count: int = 0
        self._fps_timer = time.perf_counter()
        self.fps: float = 0.0

        if self.use_async_thread:
            self._start_thread()

    def _start_thread(self):
        self._running = True
        self._capture_epoch += 1
        epoch = self._capture_epoch
        self._thread = threading.Thread(target=self._capture_worker, args=(epoch,), daemon=True)
        self._thread.start()

    def start_async_capture(self) -> None:
        """Start the capture worker on demand.

        The GUI constructs the capture object before Tk enters ``mainloop``.
        Delaying the WGC/GDI worker until the first UI paint prevents capture
        initialization from competing with widget construction on startup.
        """
        if self._running and self._thread is not None and self._thread.is_alive():
            self.use_async_thread = True
            return
        self.use_async_thread = True
        self._start_thread()

    def _capture_worker(self, epoch: int):
        """后台捕获线程：优先 WGC，失败后回退到 GDI PrintWindow/BitBlt。"""
        if self.prefer_wgc and WindowsCapture is not None:
            try:
                self._capture_worker_wgc(epoch)
                return
            except Exception as e:
                if epoch != self._capture_epoch:
                    return
                self.backend = "gdi-fallback"
                print(f"[ScreenCapture] WGC 初始化失败，回退 GDI: {e}")
        self._capture_worker_gdi(epoch)

    def _capture_worker_wgc(self, epoch: int):
        """Windows Graphics Capture 回调模式，始终只保留最新帧。"""
        title = getattr(self.window_mgr, "target_window_title", "冒险岛怀旧服")
        self.backend = "wgc"
        capture = WindowsCapture(cursor_capture=False, window_name=title)
        self._wgc_convert_thread = threading.Thread(
            target=self._wgc_conversion_worker, args=(epoch,), daemon=True
        )
        self._wgc_convert_thread.start()

        @capture.event
        def on_frame_arrived(frame, capture_control):
            if not self._running or epoch != self._capture_epoch:
                # 手动切窗后由新的 GDI worker 接管。这里不调用 stop：
                # 某些旧版 DirectX 客户端的 WGC stop 会导致宿主进程崩溃。
                return
            self._wgc_control = capture_control
            try:
                raw = np.asarray(frame.frame_buffer, dtype=np.uint8)
                bgra = raw.reshape((frame.height, frame.width, 4))
                # WindowsCapture 返回的通常是整个窗口（含标题栏/边框），
                # 而现有视觉模块统一使用客户区坐标。动态裁掉非客户区，
                # 使 WGC 与原 PrintWindow 的坐标系一致。
                try:
                    outer = wintypes.RECT()
                    user32.GetWindowRect(self.window_mgr.hwnd, ctypes.byref(outer))
                    client = self.window_mgr.get_client_rect()
                    if client:
                        client_w = int(client["width"])
                        client_h = int(client["height"])
                        frame_w = bgra.shape[1]
                        frame_h = bgra.shape[0]

                        # 仅当帧尺寸与客户区尺寸差值在合理窗口边框范围 (0~40px) 时才执行边框裁剪；
                        # 若差值过大，说明处于不同 DPI 缩放系，按完整画面呈现，避免严重切边
                        diff_w = frame_w - client_w
                        diff_h = frame_h - client_h
                        if 0 < diff_w <= 40 and 0 <= diff_h <= 60:
                            ox = max(0, diff_w // 2)
                            oy_rect = int(client["top"] - outer.top)
                            oy = max(0, min(oy_rect, frame_h - client_h))
                            cw = min(client_w, frame_w - ox)
                            ch = min(client_h, frame_h - oy)
                            if cw > 0 and ch > 0:
                                bgra = bgra[oy:oy+ch, ox:ox+cw]
                except Exception:
                    pass
                # 回调只保存最新 BGRA；颜色转换由独立线程限速完成。
                with self._wgc_raw_lock:
                    self._wgc_raw_frame = bgra.copy()
                    self._wgc_raw_seq += 1
                    self.wgc_accepted_count += 1
            except Exception as e:
                # 单帧损坏不应中止整个捕获线程。
                self.wgc_last_error = repr(e)

        @capture.event
        def on_closed():
            if epoch == self._capture_epoch:
                self._running = False

        # The blocking ``start()`` keeps the GIL inside the native message loop
        # on current windows-capture builds.  Wrapping it in a Python Thread is
        # insufficient: Tk and every other Python worker then stop responding.
        # Use the library's native free-threaded entry point instead.
        self._wgc_capture_refs.append(capture)
        self._wgc_control = capture.start_free_threaded()
        while self._running and epoch == self._capture_epoch:
            time.sleep(0.05)

    def _wgc_conversion_worker(self, epoch: int):
        """独立颜色转换线程；只转换最新一帧，最高 max_fps。"""
        next_convert = time.perf_counter()
        while self._running and epoch == self._capture_epoch:
            now = time.perf_counter()
            if now < next_convert:
                time.sleep(min(0.002, next_convert - now))
                continue
            with self._wgc_raw_lock:
                seq = self._wgc_raw_seq
                raw = self._wgc_raw_frame
            if raw is None or seq == self._wgc_converted_seq:
                time.sleep(0.001)
                continue
            try:
                acquire_started_at = time.perf_counter()
                bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                acquire_finished_at = time.perf_counter()
                with self._frame_lock:
                    self._latest_frame = bgr
                    self.frame_seq += 1
                published_at = time.perf_counter()
                self._wgc_converted_seq = seq
                self.wgc_converted_count += 1
                self._update_fps()
                self._record_capture_timing(
                    "wgc", acquire_finished_at - acquire_started_at,
                    published_at - acquire_finished_at,
                    published_at - acquire_started_at,
                )
                next_convert = max(next_convert + (1.0 / self.max_fps), time.perf_counter())
            except Exception:
                time.sleep(0.002)

    def _capture_worker_gdi(self, epoch: int):
        """后台持续零拷贝 DIBSection 高频抓帧工作线程"""
        if self.backend not in ("gdi-fallback", "gdi-manual"):
            self.backend = "gdi"
        surface: Optional[DIBSectionSurface] = None
        last_find_time = 0.0
        cur_w, cur_h = 0, 0
        target_interval = 1.0 / self.max_fps
        next_capture_at = time.perf_counter()

        while self._running and epoch == self._capture_epoch:
            now = time.perf_counter()
            if now < next_capture_at:
                sleep_sec = next_capture_at - now
                if sleep_sec > 0.001:
                    time.sleep(min(0.002, sleep_sec))
                continue

            cycle_start = now

            # 动态探测游戏窗口
            if not self.window_mgr.hwnd or not self.window_mgr.is_valid():
                if now - last_find_time > 1.0:
                    self.window_mgr.find_game_window()
                    last_find_time = now

            hwnd = self.window_mgr.hwnd
            if not hwnd:
                time.sleep(0.05)
                next_capture_at = time.perf_counter()
                continue

            rect = wintypes.RECT()
            if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
                time.sleep(0.05)
                next_capture_at = time.perf_counter()
                continue

            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w <= 0 or h <= 0:
                time.sleep(0.05)
                next_capture_at = time.perf_counter()
                continue

            # 若尺寸改变或表面未初始化，重建持久化 DIBSectionSurface
            if surface is None or cur_w != w or cur_h != h:
                if surface:
                    surface.release()
                surface = DIBSectionSurface(w, h)
                cur_w, cur_h = w, h

            acquire_started_at = time.perf_counter()
            frame = surface.capture(hwnd)
            acquire_finished_at = time.perf_counter()
            if frame is not None:
                # The DIB is BGRA and reused next frame. OpenCV's SIMD
                # conversion produces an independent contiguous BGR buffer
                # much faster than copying the strided [:, :, :3] view.
                published_frame = cv2.cvtColor(
                    surface.raw_view, cv2.COLOR_BGRA2BGR
                )
                with self._frame_lock:
                    self._latest_frame = published_frame
                    self.frame_seq += 1
                published_at = time.perf_counter()
                self._update_fps()
                self._record_capture_timing(
                    self.backend, acquire_finished_at - acquire_started_at,
                    published_at - acquire_finished_at,
                    published_at - cycle_start,
                )
                next_capture_at = max(cycle_start + target_interval, time.perf_counter())
            else:
                time.sleep(0.01)
                next_capture_at = time.perf_counter()

        if surface:
            surface.release()

    def set_foreground_login_capture(self, enabled: bool) -> None:
        """Show composited login UI only while reconnect is using it."""
        self._foreground_login_capture = bool(enabled)

    def set_timing_callback(
        self, callback: Optional[Callable[[str, float, float, float], None]]
    ) -> None:
        """Receive timing for each newly published frame, outside frame locks."""
        self._timing_callback = callback

    def _record_capture_timing(
        self, backend: str, acquire_sec: float, publish_sec: float,
        cycle_sec: float,
    ) -> None:
        callback = self._timing_callback
        if callback is not None:
            try:
                callback(
                    backend, acquire_sec * 1000.0, publish_sec * 1000.0,
                    cycle_sec * 1000.0,
                )
            except Exception:
                pass  # Diagnostic callbacks must not stop capture.

    def capture_visible_client(self) -> Optional[np.ndarray]:
        """Read the on-screen client; never read an occluding app as game UI."""
        hwnd = self.window_mgr.hwnd
        if not hwnd or not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
            return None
        foreground = user32.GetForegroundWindow()
        if foreground != hwnd:
            game_pid = wintypes.DWORD()
            foreground_pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(game_pid))
            user32.GetWindowThreadProcessId(foreground, ctypes.byref(foreground_pid))
            if not game_pid.value or game_pid.value != foreground_pid.value:
                return None
        rect = self.window_mgr.get_client_rect()
        if not rect or rect["width"] <= 0 or rect["height"] <= 0:
            return None
        surface = DIBSectionSurface(rect["width"], rect["height"])
        try:
            frame = surface.capture_screen(rect["left"], rect["top"])
            return frame.copy() if frame is not None else None
        finally:
            surface.release()

    def set_frame_overlay_provider(
        self, provider: Optional[Callable[[np.ndarray, int], np.ndarray]]
    ) -> None:
        """Set a compositing hook applied to returned frames, after capture."""
        with self._overlay_provider_lock:
            self._frame_overlay_provider = provider

    def capture_frame(
        self,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        copy: bool = True,
        include_overlay: bool = True,
    ) -> Optional[np.ndarray]:
        """获取当前最新游戏画面 (默认安全返回深拷贝，避免多线程绘图竞争)"""
        visible = self.capture_visible_client() if self._foreground_login_capture else None
        if visible is not None:
            frame = visible
            frame_seq = int(self.frame_seq)
        elif self.use_async_thread:
            with self._frame_lock:
                if self._latest_frame is None:
                    return None
                frame = self._latest_frame.copy() if copy else self._latest_frame
                frame_seq = int(self.frame_seq)
        else:
            # 同步模式
            if not self.window_mgr.hwnd or not self.window_mgr.is_valid():
                self.window_mgr.find_game_window()
            hwnd = self.window_mgr.hwnd
            if hwnd:
                frame = capture_hwnd(hwnd)
                if frame is not None:
                    self._update_fps()
                    self.frame_seq += 1
                    frame_seq = int(self.frame_seq)
                else:
                    return None
            else:
                return None

        with self._overlay_provider_lock:
            overlay_provider = self._frame_overlay_provider if include_overlay else None
        if overlay_provider is not None:
            try:
                composited = overlay_provider(frame.copy(), frame_seq)
                if composited is not None:
                    frame = composited
            except Exception as exc:
                print(f"[ScreenCapture] 画面叠加失败，继续使用原始采集帧: {exc}")

        if roi_box is not None:
            rx, ry, rw, rh = roi_box
            fh, fw = frame.shape[:2]
            x1 = max(0, min(rx, fw))
            y1 = max(0, min(ry, fh))
            x2 = max(x1, min(x1 + rw, fw))
            y2 = max(y1, min(y1 + rh, fh))
            return frame[y1:y2, x1:x2]
        return frame
    def rebind_window(self, hwnd: int) -> bool:
        """切换到用户选定窗口，安全地改用独立 GDI 捕获线程。

        WGC 的窗口选择/停止在部分旧 DirectX 客户端上会直接拖垮宿主
        进程；人工切窗后固定用 HWND 驱动的 GDI，不重启 WindowsCapture。
        """
        if not hwnd or not user32.IsWindow(hwnd):
            return False
        self._foreground_login_capture = False
        # 递增 epoch 使旧 GDI/WGC worker 自动失效；不碰旧 WGC control。
        self._capture_epoch += 1
        epoch = self._capture_epoch
        self._wgc_control = None
        self._wgc_convert_thread = None
        self._wgc_raw_frame = None
        self._wgc_raw_seq = 0
        self._wgc_converted_seq = 0
        self.window_mgr.hwnd = int(hwnd)
        with self._frame_lock:
            self._latest_frame = None
            self.frame_seq += 1
        if self.use_async_thread:
            self._running = True
            self.backend = "gdi-manual"
            self._thread = threading.Thread(
                target=self._capture_worker_gdi, args=(epoch,), daemon=True
            )
            self._thread.start()
        return True

    def _update_fps(self):
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_timer
        if elapsed >= 0.5:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

    def release(self):
        self._running = False
        if self._wgc_control is not None:
            try:
                self._wgc_control.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._wgc_convert_thread and self._wgc_convert_thread.is_alive():
            self._wgc_convert_thread.join(timeout=0.5)
