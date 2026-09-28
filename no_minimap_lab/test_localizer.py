"""Regression tests with known world poses, UI contamination and tracking loss."""
import unittest

import cv2
import numpy as np

from .atlas import Atlas, ROOT, load_atlas
from .localizer import Localizer
from .player import PlayerAnchor


@unittest.skipUnless((ROOT / "Map" / "Tile").exists(), "requires local extracted WZ assets")
class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.atlas = load_atlas(103000201, progress=lambda _: None)
        cls.model = Localizer(cls.atlas)

    def setUp(self):
        self.model.reset()

    def crop(self, offset=(650, 180), scale=1.0):
        transform = np.float32([[scale, 0, -offset[0]*scale], [0, scale, -offset[1]*scale]])
        frame = cv2.warpAffine(self.atlas.bgr, transform, (round(1280*scale), round(720*scale)))
        return frame, self.atlas.origin+offset

    def test_wz_canvas_and_all_static_instances_decode(self):
        self.assertEqual(self.atlas.meta["warnings"], [])
        self.assertEqual(self.atlas.meta["objects"], 712)
        self.assertGreater(self.model.feature_count, 2000)

    def test_known_global_pose(self):
        frame, expected = self.crop()
        pose = self.model.update(frame, timestamp=0, force=True)
        self.assertEqual(pose.status, "LOCKED")
        self.assertLess(np.linalg.norm(np.array(pose.camera)-expected), 1)

    def test_minimap_pixels_cannot_change_pose(self):
        frame, expected = self.crop()
        frame[:273, :384] = np.random.default_rng(11).integers(0, 256, (273, 384, 3), np.uint8)
        pose = self.model.update(frame, timestamp=0, force=True)
        self.assertEqual(pose.status, "LOCKED")
        self.assertLess(np.linalg.norm(np.array(pose.camera)-expected), 1)

    def test_large_foreground_occlusion(self):
        frame, expected = self.crop()
        frame[280:540, 480:850] = 170
        pose = self.model.update(frame, timestamp=0, force=True)
        self.assertEqual(pose.status, "LOCKED")
        self.assertLess(np.linalg.norm(np.array(pose.camera)-expected), 1)

    def test_camera_scroll_sign_and_scale(self):
        first, _ = self.crop()
        self.model.update(first, timestamp=0, force=True)
        second, expected = self.crop((672, 187))
        pose = self.model.update(second, timestamp=.1)
        self.assertEqual(pose.status, "COASTING")
        self.assertLess(np.linalg.norm(np.array(pose.camera)-expected), 1.5)

    def test_scaled_client(self):
        model = Localizer(self.atlas, screen_scale=1.5)
        frame, expected = self.crop(scale=1.5)
        pose = model.update(frame, timestamp=0, force=True)
        self.assertEqual(pose.status, "LOCKED")
        self.assertLess(np.linalg.norm(np.array(pose.camera)-expected), 1.5)

    def test_blank_scene_drops_old_pose(self):
        frame, _ = self.crop()
        self.model.update(frame, timestamp=0, force=True)
        pose = self.model.update(np.zeros_like(frame), timestamp=.1)
        self.assertIsNone(pose.camera)
        self.assertEqual(pose.status, "LOST")

    def test_odometry_expires_without_map_anchor(self):
        frame, _ = self.crop()
        self.model.update(frame, timestamp=0, force=True)
        original = self.model._global
        self.model._global = lambda *_: (None, "Simulated absent map evidence")
        try:
            self.assertEqual(self.model.update(frame, timestamp=.4).status, "COASTING")
            self.assertIsNone(self.model.update(frame, timestamp=.9).camera)
        finally:
            self.model._global = original

    def test_wrong_map_does_not_lock(self):
        frame = cv2.imread(str(ROOT / "real_game_frame.png"))
        if frame is None:
            self.skipTest("historic game frame not installed")
        self.assertIsNone(self.model.update(frame, timestamp=0, force=True).camera)


class AmbiguityTests(unittest.TestCase):
    def test_two_identical_scene_copies_remain_ambiguous(self):
        rng = np.random.default_rng(4)
        tile = np.zeros((720, 1280, 3), np.uint8)
        for _ in range(250):
            x, y = rng.integers([10, 10], [1260, 700])
            color = tuple(map(int, rng.integers(50, 255, 3)))
            cv2.circle(tile, (int(x), int(y)), int(rng.integers(5, 20)), color, -1)
        atlas_bgr = np.concatenate([tile, tile], axis=1)
        rgba = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGBA)
        model = Localizer(Atlas(rgba, {"origin": [0, 0]}))
        pose = model.update(tile, timestamp=0, force=True)
        self.assertEqual(pose.status, "AMBIGUOUS")
        self.assertIsNone(pose.camera)

    def test_duplicated_nameplate_is_rejected(self):
        rng = np.random.default_rng(9)
        template = rng.integers(0, 256, (25, 70, 3), np.uint8)
        anchor = PlayerAnchor()
        anchor.template = template
        anchor.offset = np.float32([35, 0])
        anchor.frame_size = None
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[400:425, 600:670] = template
        frame[500:525, 900:970] = template
        self.assertIsNone(anchor.detect(frame))


if __name__ == "__main__":
    unittest.main()
