import threading
import unittest
from types import SimpleNamespace
import numpy as np
from .async_reference import AsyncReference


class ReferenceTests(unittest.TestCase):
    def test_slow_reference_does_not_block_and_comparison_keeps_original_player(self):
        release = threading.Event()
        def detect(frame):
            release.wait(2)
            return dict(detected=True,raw_world=[1,2],snapped_world=[1,2],elapsed_ms=500)
        worker = AsyncReference(SimpleNamespace(detect=detect),45)
        frame = np.zeros((10,10,3),np.uint8)
        try:
            yellow, comparison = worker.update(frame,10,dict(world=[1,47],status='MEASURED'))
            self.assertFalse(yellow['detected'])
            self.assertIsNone(comparison)
            for _ in range(10):
                self.assertFalse(worker.update(frame,10.1,dict(world=[900,900],status='MEASURED'))[0]['detected'])
            release.set();worker.pending.result(timeout=2)
            yellow, comparison = worker.update(frame,10.2,dict(world=[900,900],status='MEASURED'))
            self.assertTrue(yellow['detected'])
            self.assertEqual(comparison['distance'],0)
            self.assertEqual(yellow['timestamp'],10)
            worker.reset()
            self.assertIsNone(worker.last)
        finally:
            release.set();worker.close()


if __name__ == '__main__':
    unittest.main()
