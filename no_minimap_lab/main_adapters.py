"""Read-only adapters around the main program's recognition algorithms.

Yellow-dot results never enter the scene localizer or player predictor.
"""
import json
import time

import cv2
import numpy as np

from .atlas import ROOT
from .player import PlayerAnchor


def recognition_config():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    keys = ("player_feature_entry_threshold", "enable_two_stage_feature",
            "enable_costume_verification", "recognition_exclusion_regions", "yellow_dot_candidate_sizes")
    return {k: config[k] for k in keys if k in config}


class MainPlayerAnchor:
    def __init__(self, feature_rescue=False):
        from src.vision.main_view_detector import MainViewDetector
        from .calibration import TEMPLATES
        self.config = recognition_config()
        self.detector = MainViewDetector(
            template_dir=str(TEMPLATES) if TEMPLATES.exists() else None,
            monster_compute_device="cpu", monster_hp_bar_compute_device="cpu",
            player_feature_threshold=float(self.config.get("player_feature_entry_threshold", .7)))
        self.detector.enable_monster_detection = False
        self.detector.enable_monster_hp_bar_detection = False
        self.detector.enable_two_stage_feature = self.config.get("enable_two_stage_feature", True)
        self.detector.enable_costume_verification = self.config.get("enable_costume_verification", True)
        self.custom = PlayerAnchor()
        self.shape = None
        self.feature_rescue = feature_rescue
        self.rescue_candidate = None

    def reset(self):
        self.detector.reset_player_tracking(clear_monsters=False)
        self.rescue_candidate = None

    def close(self):
        self.detector.executor.shutdown(wait=False, cancel_futures=True)

    def detect(self, frame):
        from .localizer import scene_mask
        from .ignore_regions import load, rectangles
        regions = load()
        if frame.shape != self.shape or regions != getattr(self, 'regions', None):
            self.reset()
            self.shape = frame.shape
            self.regions = regions
            exclusions = [dict(x=x,y=y,w=w,h=h,player=True) for x,y,w,h in rectangles(frame.shape)]
            self.detector.set_exclusion_regions(exclusions)
        frame = frame.copy()
        frame[scene_mask(frame.shape) == 0] = 0
        if self.feature_rescue and self.rescue_candidate is not None and time.perf_counter()-self.rescue_candidate[1]<.4:
            rescued=self.rescue_feature(frame)
            if rescued:
                return rescued
        before = self.detector.last_player_observed_time
        found, position, bbox = self.detector.detect_player(frame)
        # The original API may return a one-second held position on a miss.
        # Only a new real observation is admitted as a player measurement.
        if not found or position is None or self.detector.last_player_observed_time <= before:
            return self.rescue_feature(frame) if self.feature_rescue else None
        self.rescue_candidate = None
        source = "main_name_and_feature"
        custom = self.custom.detect(frame) if self.custom.calibrated else None
        if custom and np.linalg.norm(np.array(custom["screen"])-position) < 35:
            position = custom["screen"]
            source += "+lab_feet_calibration"
        return dict(screen=list(map(float, position)), bbox=list(bbox), calibrated=True,
                    feature_bbox=self.detector.last_feature_bbox, source=source,
                    evidence="feature" if self.detector.last_feature_bbox else "nameplate_votes")

    def rescue_feature(self, frame):
        """Strict full-resolution reacquisition when pet occlusion defeats coarse search."""
        from .localizer import scene_mask
        profile = self.detector.two_stage_profile
        if not profile:
            return None
        allowed = scene_mask(frame.shape)
        for rect in self.detector.exclusion_regions:
            if rect.get("player"):
                x,y,w,h = (int(rect.get(k,0)) for k in ("x","y","w","h"))
                allowed[max(0,y):max(0,y)+h,max(0,x):max(0,x)+w] = 0
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray[allowed==0] = 0
        candidates=[]
        rivals=[]
        for side in ("r","l"):
            tpl=profile.get(f"feature_{side}_gray")
            mask=profile.get(f"feature_{side}_mask")
            if tpl is None:
                continue
            scores=self.detector._match_player_feature(gray,tpl,mask)
            scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
            _,score,_,(x,y)=cv2.minMaxLoc(scores)
            h,w=tpl.shape
            patch=frame[y:y+h,x:x+w]
            reference=profile[f"feature_{side}_bgr"]
            pixels=(mask>0) if mask is not None else np.ones((h,w),bool)
            color_error=float(np.mean(np.abs(patch[pixels].astype(float)-reference[pixels])))
            candidates.append((score,x,y,w,h,side,color_error))
            scores[max(0,y-h):y+h+1,max(0,x-w):x+w+1]=-1
            rivals.append(float(scores.max()))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        score,x,y,w,h,side,error=candidates[0]
        rival=max(rivals+[c[0] for c in candidates[1:] if abs(c[1]-x)>w or abs(c[2]-y)>h])
        if score<.92 or score-rival<.10 or error>35:
            self.rescue_candidate=None
            return None
        offset=float(profile["offset_from_center_x"])
        point=np.array([x+w/2+(-offset if side=="r" else offset),
                        y+h+float(profile["offset_from_feet_y"])])
        now=time.perf_counter()
        prior=self.rescue_candidate
        self.rescue_candidate=(point,now)
        if prior is None or now-prior[1]>.4 or np.linalg.norm(point-prior[0])>16:
            return None
        self.detector.last_player_pos=tuple(int(round(v)) for v in point)
        self.detector.last_player_time=now
        self.detector.last_player_observed_time=now
        return dict(screen=point.tolist(),bbox=[int(point[0]-30),int(point[1]-85),61,85],
                    feature_bbox=[x,y,w,h],calibrated=True,source="main_feature_full_resolution_rescue",
                    evidence="unique_feature_two_frames",score=score,rival_score=rival,color_error=error)


class YellowWorldReference:
    def __init__(self, map_id):
        from src.vision.tracker import MinimapTracker
        from src.vision.wz_map_reader import WzMapReader
        from src.engine.platform_graph import PlatformGraphBuilder
        data = WzMapReader(str(ROOT / "Map")).load_map(map_id)
        self.graph = PlatformGraphBuilder.build_from_map_dict(data)
        self.tracker = MinimapTracker(enable_template_tracking=False)
        self.tracker.set_yellow_candidate_sizes(recognition_config().get("yellow_dot_candidate_sizes"))
        self.tracker.set_expected_canvas_size(self.graph.get_minimap_canvas_size())
        self.tracker.set_expected_canvas_image(self.graph.get_minimap_canvas_gray(), map_id,
                                              self.graph.get_minimap_canvas_gray_variants())

    def detect(self, frame):
        start = time.perf_counter()
        result = self.tracker.detect(frame)
        output = dict(detected=False, raw_world=None, snapped_world=None,
                      body_reference_offset_y=45.0, inner_box=result.inner_box)
        if result.is_detected and result.norm_pos is not None and result.inner_box:
            x, y, w, h = result.inner_box
            crop = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            debug = {}
            raw = self.graph.minimap_norm_to_world(*result.norm_pos, crop_gray_frame=crop, debug_out=debug)
            snapped = self.graph.get_snapped_player_world_pos(*raw)
            output.update(detected=True, raw_world=list(map(float, raw)),
                          snapped_world=list(map(float, snapped)), conversion=debug,
                          dot_pixel=list(result.subpixel_pos or result.pixel_pos))
        output["elapsed_ms"] = (time.perf_counter()-start)*1000
        return output
