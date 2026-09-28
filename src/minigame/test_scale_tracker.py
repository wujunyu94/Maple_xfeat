"""Coordinate, cadence and lifecycle contracts for the live scale adapter."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from . import scale_tracker
from .baseline_tracker import TrackResult


class FakeGuarded:
    instances = []
    geometry = dict(is_circle=False, circ=.45, rel_dim=.28, rel_area=.025)

    def __init__(self, config=None):
        self.engine = SimpleNamespace(**self.geometry)
        self.calls = []
        self.index = len(self.instances)
        self.instances.append(self)
        self.result = (TrackResult(475, 310, .8, True, dialog_roi=(100, 60, 750, 500))
                       if self.index == 0 else
                       TrackResult(960, 540, .9, True, 970, 530, (432, 188, 1056, 704)))

    def update(self, frame, timestamp=None):
        self.calls.append(timestamp)
        return self.result


class ScaleTrackerTests(unittest.TestCase):
    def setUp(self):
        FakeGuarded.instances = []
        FakeGuarded.geometry = dict(is_circle=False, circ=.45, rel_dim=.28, rel_area=.025)
        self.patch = patch.object(scale_tracker, 'GuardedTracker', FakeGuarded)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.tracker = scale_tracker.CausalShapeTracker()

    def test_normalized_coordinates_and_roi_are_in_capture_space(self):
        result = self.tracker.update(self.frame, 1.0)
        self.assertTrue(self.tracker.normalized)
        self.assertEqual(result.dialog_roi, (100, 60, 750, 500))
        self.assertAlmostEqual(result.x, 475)
        self.assertAlmostEqual(result.y, 310)
        self.assertAlmostEqual(result.diff_x, 475 + 10 * 750 / 1056)
        self.assertAlmostEqual(result.diff_y, 310 - 10 * 500 / 704)

    def test_no_candidate_sentinel_is_not_translated(self):
        self.tracker.update(self.frame, 0)
        FakeGuarded.instances[-1].result = TrackResult(960, 540, .4, True,
                                                       dialog_roi=(432,188,1056,704))
        self.frame[100,100] = 1
        result = self.tracker.update(self.frame, .03)
        self.assertEqual((result.diff_x, result.diff_y), (0, 0))

    def test_repeated_frame_does_not_advance_and_uses_next_capture_time(self):
        self.tracker.update(self.frame, 1)
        for t in [1.01, 1.02, 1.03]:
            self.tracker.update(self.frame, t)
        self.assertEqual(FakeGuarded.instances[-1].calls, [1])
        # Mutating the caller's reused buffer must not mutate our previous image.
        self.frame[100,100] = 1
        self.tracker.update(self.frame, 1.04)
        self.assertEqual(FakeGuarded.instances[-1].calls, [1, 1.04])
        self.assertEqual(self.tracker.duplicate_frames, 3)

    def test_frozen_capture_revokes_mouse_confidence(self):
        first = self.tracker.update(self.frame, 0)
        held = self.tracker.update(self.frame, .2)
        self.assertTrue(held.initialized)
        self.assertLess(held.confidence, .45)
        self.assertEqual((held.x, held.y), (first.x, first.y))
        self.frame[100,100] = 1
        self.assertGreater(self.tracker.update(self.frame, .21).confidence, .45)

    def test_reset_clears_geometry_and_timestamp(self):
        self.tracker.update(self.frame, 5)
        self.tracker.reset()
        self.assertFalse(self.tracker.normalized)
        self.assertIsNone(self.tracker.dialog_roi)
        self.assertEqual(self.tracker.duplicate_frames, 0)
        self.tracker.update(self.frame, 1)

    def test_window_resize_reacquires_instead_of_reusing_crop(self):
        self.tracker.update(self.frame, 0)
        count = len(FakeGuarded.instances)
        self.tracker.update(np.zeros((1080,1920,3), np.uint8), .03)
        self.assertGreater(len(FakeGuarded.instances), count)
        self.assertFalse(self.tracker.normalized)

    def test_close_confirmation_is_not_skipped_for_duplicate_pixels(self):
        self.tracker.update(self.frame, 0)
        guard = FakeGuarded.instances[-1]
        guard.engine._close_confirm_count = 1
        guard.result = TrackResult(960,540,0,False)
        result = self.tracker.update(self.frame, .03)
        self.assertEqual(guard.calls, [0, .03])
        self.assertFalse(result.initialized)
        self.assertFalse(self.tracker.normalized)

    def test_failed_normalization_keeps_original_target_and_does_not_retry(self):
        original_update = FakeGuarded.update
        def fail_second(guard, frame, timestamp=None):
            if guard.index > 0:
                guard.calls.append(timestamp)
                return TrackResult(960,540,0,False)
            return original_update(guard, frame, timestamp)
        with patch.object(FakeGuarded, 'update', fail_second):
            result = self.tracker.update(self.frame, 0)
            self.assertFalse(self.tracker.normalized)
            self.assertEqual((result.x,result.y), (475,310))
            self.frame[100,100] = 1
            self.tracker.update(self.frame, .03)
            self.assertEqual(len(FakeGuarded.instances), 2)

    def test_invalid_or_non_increasing_timestamp_is_rejected(self):
        self.tracker.update(self.frame, 1)
        for value in [1, .9, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.tracker.update(self.frame, value)

    def test_existing_shape_paths_keep_native_scale(self):
        for geometry in [dict(is_circle=True, rel_dim=.19), dict(circ=.6),
                         dict(rel_dim=.20), dict(rel_area=.01)]:
            with self.subTest(geometry=geometry):
                FakeGuarded.instances = []
                self.tracker = scale_tracker.CausalShapeTracker()
                for key,value in geometry.items():
                    setattr(self.tracker.engine, key, value)
                result = self.tracker.update(self.frame, 0)
                self.assertFalse(self.tracker.normalized)
                self.assertEqual(len(FakeGuarded.instances), 1)
                self.assertEqual(result.x, 475)

    def test_dialog_close_drops_mapping_and_allows_new_acquisition(self):
        self.tracker.update(self.frame, 0)
        FakeGuarded.instances[-1].result = TrackResult(960,540,0,False)
        self.frame[100,100] = 1
        result = self.tracker.update(self.frame, .03)
        self.assertFalse(result.initialized)
        self.assertIsNone(result.dialog_roi)
        self.assertFalse(self.tracker.normalized)
        self.assertIsNone(self.tracker._previous_crop)

    def test_oversized_circle_uses_observed_footprint(self):
        self.tracker.engine.is_circle=True
        self.tracker.engine.rel_dim=.28
        self.tracker.engine.target_radius=49.
        result=self.tracker.update(self.frame,0.)
        self.assertTrue(self.tracker.normalized)
        self.assertEqual(self.tracker._placement[3],round(96/.28))
        self.assertEqual(result.dialog_roi,(100,60,750,500))

    def test_moderate_circle_mismatch_and_small_circle_stay_native(self):
        for diameter in (65.,96.,120.):
            with self.subTest(diameter=diameter):
                FakeGuarded.instances=[]
                t=scale_tracker.CausalShapeTracker();t.engine.is_circle=True
                t.engine.rel_dim=diameter/500;t.engine.target_radius=49.
                t.update(self.frame,0.)
                self.assertFalse(t.normalized)

    def test_reset_clears_circle_target_height(self):
        self.tracker.engine.is_circle=True;self.tracker.engine.rel_dim=.28
        self.tracker.update(self.frame,0.)
        self.assertNotEqual(self.tracker._normalization_height,704)
        self.tracker.reset()
        self.assertEqual(self.tracker._normalization_height,704)


if __name__ == '__main__':
    unittest.main()
