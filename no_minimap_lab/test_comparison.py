import unittest
from concurrent.futures import Future
import time

import cv2
import numpy as np

from .atlas import ROOT, load_atlas
from .async_localizer import AsyncLocalizer
from .localizer import Pose
from .player_state import PlayerState, compare_coordinates, geometry_advice


class PlayerPredictionTests(unittest.TestCase):
    def pose(self, x=0, y=0):
        return Pose(status="LOCKED", camera=(x, y))

    def obs(self, x, y=100):
        return dict(screen=[x, y], calibrated=True, source="test")

    def test_velocity_prediction_accounts_for_camera_scroll(self):
        state = PlayerState()
        state.update(self.obs(100), self.pose(), 0)
        state.update(self.obs(110), self.pose(), .1)
        result = state.update(None, self.pose(30), .2)
        self.assertEqual(result["status"], "PREDICTED")
        np.testing.assert_allclose(result["world"], [120, 100])
        np.testing.assert_allclose(result["screen"], [90, 100])
        self.assertGreater(result["uncertainty_px"], 3)

    def test_camera_alone_does_not_create_player_position(self):
        state = PlayerState()
        self.assertIsNone(state.update(None, self.pose(50), 0)["world"])
        state.update(self.obs(100), self.pose(), .1)
        self.assertIsNone(state.update(None, self.pose(80), .2)["world"])

    def test_expiry_and_camera_loss_stop_prediction(self):
        state = PlayerState()
        state.update(self.obs(100), self.pose(), 0)
        state.update(self.obs(110), self.pose(), .1)
        self.assertIsNone(state.update(None, self.pose(), .36)["world"])
        self.assertIsNone(state.update(self.obs(120), Pose(), .4)["world"])

    def test_large_false_detection_is_rejected_until_confirmed(self):
        state = PlayerState()
        state.update(self.obs(100), self.pose(), 0)
        state.update(self.obs(110), self.pose(), .1)
        result = state.update(self.obs(900), self.pose(), .12)
        self.assertEqual(result["status"], "PREDICTED")
        self.assertLess(result["world"][0], 150)
        result = state.update(self.obs(902), self.pose(), .14)
        self.assertEqual(result["status"], "REACQUIRED")
        self.assertEqual(result["velocity"], [0, 0])

    def test_coordinate_reference_difference_is_explicit(self):
        player = dict(world=[2504, -59], status="MEASURED")
        yellow = dict(detected=True, raw_world=[2504, -106])
        result = compare_coordinates(player, yellow)
        self.assertEqual(result["feet_minus_yellow_raw"], [0, 47])
        self.assertEqual(result["visual_body_minus_yellow_raw"], [0, 2])
        self.assertEqual(player["world"], [2504, -59])
        self.assertEqual(yellow["raw_world"], [2504, -106])

    def test_geometry_advice_does_not_authorize_controls(self):
        class AtlasStub:
            meta = dict(ladders=[[130, 0, 120]])
        advice = geometry_advice(AtlasStub(), self.pose(30),
                                 dict(world=[100, 100], status="PREDICTED"), 1)
        self.assertEqual(advice["dx_world"], 30)
        self.assertFalse(advice["control_ready"])


class MainPlayerFreshnessTests(unittest.TestCase):
    def test_main_detectors_held_result_is_not_a_fresh_measurement(self):
        from .main_adapters import MainPlayerAnchor
        class Detector:
            last_player_observed_time = 1.0
            def detect_player(self, frame):
                return True, (100, 100), (80, 40, 40, 60)
        anchor = MainPlayerAnchor.__new__(MainPlayerAnchor)
        frame = np.zeros((100, 100, 3), np.uint8)
        anchor.shape = frame.shape
        anchor.detector = Detector()
        anchor.feature_rescue = False
        self.assertIsNone(anchor.detect(frame))


@unittest.skipUnless((ROOT / "Map" / "Tile").exists(), "local WZ assets required")
class AsyncAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.atlas = load_atlas(103000201, progress=lambda _: None)

    def setUp(self):
        self.model = AsyncLocalizer(self.atlas)
        self.source = cv2.warpAffine(self.atlas.bgr, np.float32([[1, 0, -650], [0, 1, -180]]), (1280, 720))

    def tearDown(self):
        self.model.close()

    def test_late_anchor_is_reprojected_to_current_camera(self):
        from .localizer import scene_mask
        gray = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
        found, reason = self.model._global(gray, scene_mask(gray.shape))
        future = Future()
        future.set_result((gray, 1.0, self.model.generation, found, reason, 100))
        self.model.pending = future
        current = cv2.warpAffine(self.atlas.bgr, np.float32([[1, 0, -672], [0, 1, -187]]), (1280, 720))
        pose = self.model.update(current, timestamp=1.1)
        self.assertEqual(pose.status, "LOCKED")
        self.assertLess(np.linalg.norm(np.array(pose.camera)-(self.atlas.origin+[672, 187])), 1.5)
        self.assertAlmostEqual(pose.anchor_age, .1)

    def test_old_generation_and_expired_results_are_rejected(self):
        gray = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
        for stamp, generation in ((1.0, self.model.generation-1), (0, self.model.generation)):
            future = Future()
            future.set_result((gray, stamp, generation, (np.array([1, 2]), 20, 0, np.zeros((20, 2)), 0), "fake", 100))
            self.model.pending = future
            pose = self.model.update(self.source, timestamp=1.1)
            self.assertIsNone(pose.camera)
            # Consume any new submitted work before swapping in another future.
            if self.model.pending:
                self.model.pending.result(timeout=10)
                self.model.pending = None


@unittest.skipUnless((ROOT / "no_minimap_lab/cache/comparison_probe.png").exists(), "real client sample required")
class IndependentPipelineTests(unittest.TestCase):
    def test_blacked_minimap_cannot_change_visual_coordinates(self):
        from .pipeline import Pipeline
        # Character templates are user-owned and may have been recalibrated for
        # a different character since the old subway screenshot was captured.
        current = ROOT / "no_minimap_lab/cache/goal_start.png"
        frame = cv2.imread(str(current if current.exists() else ROOT / "no_minimap_lab/cache/comparison_probe.png"))
        atlas = load_atlas(101000000 if current.exists() else 103000201, progress=lambda _: None)
        output = []
        for black_minimap in (False, True):
            image = frame.copy()
            if black_minimap:
                image[:int(image.shape[0]*.38), :int(image.shape[1]*.30)] = 0
            pipeline = Pipeline(atlas, asynchronous=False)
            try:
                output.append(pipeline.update(image, 1.0))
            finally:
                pipeline.close()
        self.assertTrue(output[0]["yellow"]["detected"])
        self.assertFalse(output[1]["yellow"]["detected"])
        self.assertIsNotNone(output[0]["player_world"])
        np.testing.assert_allclose(output[0]["player_world"], output[1]["player_world"], atol=.1)


if __name__ == "__main__":
    unittest.main()
