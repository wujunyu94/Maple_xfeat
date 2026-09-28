"""Single-owner capture thread, bounded latest-frame mailbox (no frame backlog)."""
import threading
import time


class PreviewCapture:
    def __init__(self, factory, stop):
        self.factory, self.stop = factory, stop
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True, name='preview-capture')

    def start(self):
        self.thread.start()
        return self

    def get(self, after=None, timeout=0):
        with self.condition:
            if after == self.sequence and timeout:
                self.condition.wait(timeout)
            return self.sequence, self.latest

    def run(self):
        capture = None
        try:
            capture = self.factory()
            while not self.stop.is_set():
                begin = time.perf_counter()
                import ctypes
                manager = getattr(capture, 'window_mgr', None)
                minimized = bool(manager and manager.hwnd and ctypes.windll.user32.IsIconic(manager.hwnd))
                frame = None if minimized else capture.capture_frame()
                received = time.perf_counter()
                # Some capture backends reuse their buffer; own this frame.
                packet = (frame.copy(), received, (received-begin)*1000) if frame is not None else None
                with self.condition:
                    self.sequence += 1
                    self.latest = packet
                    self.condition.notify_all()
                self.stop.wait(max(0, 1/60-(time.perf_counter()-begin)))
        except Exception as exc:
            self.error = exc
        finally:
            if capture is not None:
                capture.release()

    def close(self):
        self.stop.set()
        self.thread.join(3)
