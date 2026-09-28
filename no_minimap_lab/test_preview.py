import queue
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from .display import overview
from .gui import App
from .preview_capture import PreviewCapture


class PreviewTests(unittest.TestCase):
    def test_slow_consumer_does_not_block_capture_or_reuse_buffer(self):
        released = threading.Event()
        buffer = np.zeros((16, 16, 3), np.uint8)
        def grab():
            buffer[:] += 1
            return buffer
        fake = SimpleNamespace(capture_frame=grab, release=released.set)
        capture = PreviewCapture(lambda:fake, threading.Event()).start()
        try:
            deadline = time.perf_counter()+2
            while capture.get()[1] is None and time.perf_counter()<deadline:
                time.sleep(.005)
            first_seq, first = capture.get()
            original = first[0].copy()
            time.sleep(.2)  # Simulate a slow detector; capture keeps running.
            last_seq, last = capture.get()
            self.assertGreater(last_seq-first_seq, 3)
            self.assertGreater(last[1], first[1])
            np.testing.assert_array_equal(first[0], original)
        finally:
            capture.close()
        self.assertTrue(released.is_set())
        self.assertFalse(capture.thread.is_alive())

    def test_only_latest_result_is_retained_and_errors_are_not_dropped(self):
        app = App.__new__(App)
        app.frame_lock = threading.Lock()
        app.events = queue.SimpleQueue()
        app.pending_frame = None
        app.emit('error', 'test error')
        for i in range(100):
            app.emit('frame', i)
        self.assertEqual(app.pending_frame, 99)
        self.assertEqual(app.events.get_nowait(), ('error', 'test error'))
        self.assertTrue(app.events.empty())

    def test_cached_map_matches_uncached_and_is_not_mutated(self):
        atlas = SimpleNamespace(bgr=np.zeros((100, 200, 3), np.uint8), origin=np.array([0,0]),
                                meta={'ladder_nodes':[dict(kind='rope', x=25, y1=5, y2=80)]})
        base = overview(atlas, (40,60), SimpleNamespace(camera=None), 1, size=(200,100))
        original = base.copy()
        pose = SimpleNamespace(camera=(20,30))
        cached = overview(atlas, (40,60), pose, 1, size=(200,100), background=base)
        plain = overview(atlas, (40,60), pose, 1, size=(200,100))
        np.testing.assert_array_equal(cached, plain)
        np.testing.assert_array_equal(base, original)


if __name__ == '__main__':
    unittest.main()
