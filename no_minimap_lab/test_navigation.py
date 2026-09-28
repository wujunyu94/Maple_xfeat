import json
from collections import deque
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from .navigation import Keys, Observer
from .atlas import ROOT, load_atlas
from .localizer import Localizer


class ControlGuards(unittest.TestCase):
    def test_teleport_and_arbitrary_keys_cannot_be_sent(self):
        keys=Keys()
        for name in ("c", "enter", "f1", "powershell"):
            with self.assertRaises(ValueError):
                keys.set(name)

    def test_command_is_short_lived(self):
        folder=ROOT / "no_minimap_lab/output/test_control_commands"
        folder.mkdir(parents=True,exist_ok=True)
        with patch("no_minimap_lab.navigation.STATE",folder):
            Keys().set("left")
            command=json.loads((Path(folder)/"command.json").read_text())
            self.assertLessEqual(command["ttl"],.25)
            self.assertEqual(command["keys"],["left"])

    def test_old_or_missing_position_stops_control(self):
        obs=Observer.__new__(Observer)
        obs.latest=dict(time=time.perf_counter()-.31,world=[0,0])
        self.assertIsNone(obs.get())
        obs.latest=dict(time=time.perf_counter(),world=None)
        self.assertIsNone(obs.get())

    def test_goal_requires_continuous_standing(self):
        obs=Observer.__new__(Observer)
        now=time.perf_counter()
        obs.history=deque(dict(time=now-d,platform=87) for d in (.30,.20,.10,0))
        self.assertTrue(obs.stable(87))
        obs.history[1]["platform"]=None
        self.assertFalse(obs.stable(87))


class MinimapIndependence(unittest.TestCase):
    def test_temporal_prior_needs_fresh_map_pixels(self):
        path=ROOT/"no_minimap_lab/output/final_suite/prepare_xfeat/latest.jpg"
        if not path.exists():
            self.skipTest("optional upper-bridge regression fixture unavailable")
        import torch
        cv2.setNumThreads(4)
        from .localizer import scene_mask
        atlas=load_atlas(101000000,progress=lambda _:None)
        frame=cv2.imread(str(path))
        pose=Localizer(atlas).update(frame,force=True)
        self.assertIsNotNone(pose.camera)
        model=Localizer(atlas,backend="xfeat-cuda" if torch.cuda.is_available() else "xfeat-cpu")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask=scene_mask(gray.shape)
        found,_=model._global(gray,mask,prior_camera=np.asarray(pose.camera))
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found[0],pose.camera,atol=1)
        failed,_=model._global(np.zeros_like(gray),mask,prior_camera=np.asarray(pose.camera))
        self.assertTrue(failed is None or failed[0] is None)

    def test_pet_occlusion_rescue_needs_unique_two_frame_feature(self):
        from .main_adapters import MainPlayerAnchor
        path=ROOT/"no_minimap_lab/cache/player_pet_occlusion.png"
        if not path.exists():
            self.skipTest("optional pet occlusion fixture unavailable")
        image=cv2.imread(str(path))
        anchor=MainPlayerAnchor(feature_rescue=True)
        try:
            self.assertIsNone(anchor.rescue_feature(image))
            found=anchor.rescue_feature(image)
            self.assertIsNotNone(found)
            self.assertGreater(found['score'],.92)
            duplicate=image.copy()
            x,y,w,h=found['feature_bbox']
            duplicate[420:420+h,260:260+w]=image[y:y+h,x:x+w]
            self.assertIsNone(anchor.rescue_feature(duplicate))
        finally:
            anchor.close()

    def test_sparse_learned_proposal_is_verified_across_scene(self):
        path=ROOT/"no_minimap_lab/cache/xfeat_stall_p41.png"
        if not path.exists() or not (ROOT/"third_party/accelerated_features/weights/xfeat.pt").exists():
            self.skipTest("optional learned regression fixture unavailable")
        import torch
        cv2.setNumThreads(4)
        model=Localizer(load_atlas(101000000,progress=lambda _:None),
                        backend="xfeat-cuda" if torch.cuda.is_available() else "xfeat-cpu")
        pose=model.update(cv2.imread(str(path)),force=True)
        self.assertEqual(pose.status,"LOCKED")
        # Independent SIFT/image alignment reference; not survey ground truth.
        np.testing.assert_allclose(pose.camera,[-25,-1704],atol=1)
        self.assertGreaterEqual(pose.inliers,100)

    def test_actual_start_and_finish_do_not_need_minimap(self):
        cv2.setNumThreads(4)
        atlas=load_atlas(101000000,progress=lambda _:None)
        paths=[ROOT/"no_minimap_lab/cache/goal_start.png",
               ROOT/"no_minimap_lab/output/trial_opencv_01/latest.jpg"]
        if not all(p.exists() for p in paths):
            self.skipTest("captured live fixtures unavailable")
        backends=["sift-cpu"]
        if (ROOT/"third_party/accelerated_features/weights/xfeat.pt").exists():
            import torch
            backends.append("xfeat-cuda" if torch.cuda.is_available() else "xfeat-cpu")
        for backend in backends:
            model=Localizer(atlas,backend=backend)
            for path in paths:
                image=cv2.imread(str(path))
                hidden=image.copy()
                hidden[:int(image.shape[0]*.38),:int(image.shape[1]*.30)]=0
                poses=[]
                for frame in (image,hidden):
                    model.reset()
                    poses.append(model.update(frame,force=True))
                self.assertTrue(all(p.camera is not None for p in poses),(backend,path.name))
                np.testing.assert_allclose(poses[0].camera,poses[1].camera,atol=1)


if __name__ == "__main__":
    unittest.main()
