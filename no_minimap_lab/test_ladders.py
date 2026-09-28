import unittest

import cv2
import numpy as np

from .atlas import Atlas, ROOT, load_atlas
from .ladder_overlay import LadderOverlay, project_ladders
from .localizer import Localizer, Pose


class ProjectionTests(unittest.TestCase):
    def atlas(self):
        return Atlas(np.zeros((10, 10, 4), np.uint8), dict(origin=[0, 0], ladder_nodes=[
            dict(id="7", kind="ladder", x=500, y1=100, y2=500, upper_exit=True),
            dict(id="8", kind="rope", x=900, y1=-100, y2=900, upper_exit=False),
            dict(id="9", kind="rope", x=4000, y1=0, y2=900, upper_exit=False)]))

    def test_all_visible_types_and_ids_without_player(self):
        result = project_ladders(self.atlas(), [100, 0], (720, 1280, 3))
        self.assertEqual([n["id"] for n in result], ["7", "8"])
        self.assertEqual(result[1]["kind"], "rope")
        self.assertEqual(result[0]["screen_line"], [400, 100, 500])
        self.assertGreaterEqual(result[1]["segments"][0][1], int(720*.08))
        self.assertLess(result[1]["segments"][-1][2], int(720*.86))

    def test_no_camera_no_projection(self):
        self.assertEqual(project_ladders(self.atlas(), None, (720, 1280, 3)), [])

    def test_camera_and_scale_transform(self):
        result = project_ladders(self.atlas(), [100, 30], (1080, 1920, 3), 1.5)
        self.assertEqual(result[0]["screen_line"], [600, 105, 705])


@unittest.skipUnless((ROOT / "Map/Tile").exists(), "requires WZ assets")
class PixelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.atlas = load_atlas(103000201, progress=lambda _: None)

    def test_img_metadata_retains_ladder_ids_and_types(self):
        nodes = self.atlas.meta["ladder_nodes"]
        self.assertEqual(len(nodes), len(self.atlas.meta["ladders"]))
        self.assertEqual(len({n["id"] for n in nodes}), len(nodes))
        self.assertEqual(nodes[0]["kind"], "ladder")
        self.assertEqual(nodes[0]["x"], 197)

    def test_opaque_ladder_verified_blank_not_verified_and_loss_clears(self):
        frame = cv2.warpAffine(self.atlas.bgr, np.float32([[1, 0, -650], [0, 1, -180]]), (1280, 720))
        pose = Pose(status="LOCKED", camera=tuple(self.atlas.origin+[650, 180]))
        overlay = LadderOverlay(self.atlas)
        detected = overlay.update(frame, pose, 0)
        self.assertGreater(sum(n["status"] == "verified" for n in detected), 0)
        black = overlay.update(np.zeros_like(frame), pose, .3)
        self.assertTrue(all(n["status"] == "projected" for n in black))
        self.assertEqual(overlay.update(frame, Pose(), .4), [])
        self.assertEqual(overlay.evidence, {})


@unittest.skipUnless((ROOT / "no_minimap_lab/third_party/xfeat/weights/xfeat.pt").exists(), "XFeat not installed")
class LearnedBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cv2.setNumThreads(4)
        cls.atlas = load_atlas(103000201, progress=lambda _: None)
        cls.model = Localizer(cls.atlas, backend="xfeat-cuda" if torch.cuda.is_available() else "xfeat-cpu")

    def test_known_pose_and_minimap_independence(self):
        frame = cv2.warpAffine(self.atlas.bgr, np.float32([[1, 0, -650], [0, 1, -180]]), (1280, 720))
        poses = []
        for contaminated in (False, True):
            image = frame.copy()
            if contaminated:
                image[:273, :384] = np.random.default_rng(8).integers(0, 256, (273, 384, 3), np.uint8)
            self.model.reset()
            poses.append(self.model.update(image, force=True))
        self.assertTrue(all(p.status == "LOCKED" for p in poses))
        np.testing.assert_allclose(poses[0].camera, self.atlas.origin+[650, 180], atol=1)
        np.testing.assert_allclose(poses[0].camera, poses[1].camera, atol=.1)

    def test_wrong_scene_is_not_accepted(self):
        self.model.reset()
        frame = cv2.imread(str(ROOT / "real_game_frame.png"))
        if frame is None:
            self.skipTest("reference screenshot unavailable")
        self.assertIsNone(self.model.update(frame, force=True).camera)

    def test_blank_scene_loses_position(self):
        self.model.reset()
        self.assertIsNone(self.model.update(np.zeros((720, 1280, 3), np.uint8), force=True).camera)

    def test_repeated_forest_expands_rejected_candidates(self):
        path = ROOT / "no_minimap_lab/cache/live_v3_after.png"
        if not path.exists():
            self.skipTest("optional captured forest regression fixture unavailable")
        atlas = load_atlas(101010101, progress=lambda _: None)
        model = Localizer(atlas, backend=self.model.backend)
        pose = model.update(cv2.imread(str(path)), force=True)
        self.assertEqual(pose.status, "LOCKED")
        # Recorded visual/SIFT reference, not independently surveyed truth.
        np.testing.assert_allclose(pose.camera, [-1663, 914], atol=1)


if __name__ == "__main__":
    unittest.main()
