"""Pinned XFeat inference + tensor cosine matching on CPU or CUDA.

OpenCV is used only for image preparation by the caller, not feature inference
or descriptor matching here. No automatic downloads during startup.
"""
from pathlib import Path

import numpy as np
import sys


class XFeatIndex:
    def __init__(self, image, mask, device="cuda"):
        from .check_environment import require
        require('xfeat-'+device)
        import torch
        xfeat_root = Path(__file__).resolve().parents[1] / "third_party" / "accelerated_features"
        if not xfeat_root.is_dir():
            raise FileNotFoundError("XFeat submodule is missing. Run: git submodule update --init --recursive")
        if str(xfeat_root) not in sys.path:
            sys.path.insert(0, str(xfeat_root))
        from modules.xfeat import XFeat, InterpolateSparse2d, XFeatModel
        self.torch = torch
        self.device = torch.device(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA unavailable; choose xfeat-cpu or sift-cpu")
        torch.set_num_threads(4)
        self.extractor = XFeat.__new__(XFeat)
        torch.nn.Module.__init__(self.extractor)
        self.extractor.dev = self.device
        self.extractor.top_k = 6000
        self.extractor.detection_threshold = .05
        self.extractor.interpolator = InterpolateSparse2d("bicubic")
        self.extractor.net = XFeatModel().to(self.device).eval()
        weights = xfeat_root / "weights" / "xfeat.pt"
        if not weights.is_file():
            raise FileNotFoundError("XFeat checkpoint is missing from the initialized submodule.")
        self.extractor.net.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.points, self.descriptors = self.extract(image, mask, 8000)

    def extract(self, image, mask, limit):
        t = self.torch
        # Pad to a multiple of 32 instead of resizing anisotropically. The
        # official preprocessor then preserves this image's physical scale.
        h, w = image.shape
        padded = np.pad(image, ((0, (-h) % 32), (0, (-w) % 32)), mode="constant")
        tensor = t.from_numpy(np.ascontiguousarray(padded)).to(self.device).float()[None, None]/255
        with t.inference_mode():
            result = self.extractor.detectAndCompute(tensor, top_k=limit)[0]
            points = result["keypoints"].cpu().numpy()
            x, y = np.rint(points).astype(int).T
            keep = (x >= 0) & (y >= 0) & (x < w) & (y < h)
            keep &= mask[np.clip(y, 0, h-1), np.clip(x, 0, w-1)] > 0
            return points[keep], result["descriptors"][t.as_tensor(keep, device=self.device)].contiguous()

    def match(self, image, mask):
        # CNN receptive fields extend beyond the selected keypoint. Zero UI
        # pixels before inference as well as excluding their keypoint centers.
        image = np.where(mask > 0, image, 0).astype(np.uint8)
        points, desc = self.extract(image, mask, 2500)
        if not len(points) or not len(self.points):
            return points, np.empty(0, int), np.empty(0, int)
        with self.torch.inference_mode():
            cosine = desc @ self.descriptors.T
            scores, indices = cosine.topk(min(8, len(self.points)), dim=1)
            # Preserve close alternatives on repeated scenery for world voting.
            valid = (scores >= .78) & (scores >= scores[:, :1]-.08)
            queries, positions = valid.nonzero(as_tuple=True)
            return points, queries.cpu().numpy(), indices[queries, positions].cpu().numpy()

    def refine(self, gray, allowed, hypotheses, atlas, scale):
        """Verify coarse learned proposals against actual opaque map pixels.

        Pixel-art repetition can move learned correspondences by one rung or
        brick. Search ±24 world pixels using GPU tensor gathers, with a runner-up
        margin; never accept the learned feature count alone as an exact pose.
        """
        t = self.torch
        h, w = gray.shape
        if not hasattr(self, "pixel_world"):
            import cv2
            ag = cv2.cvtColor(atlas.bgr, cv2.COLOR_BGR2GRAY)
            texture = np.abs(ag.astype(np.int16)-np.roll(ag, 2, 0)) + np.abs(ag.astype(np.int16)-np.roll(ag, 2, 1))
            yy, xx = np.mgrid[3:ag.shape[0]-3:6, 3:ag.shape[1]-3:6]
            good = (atlas.mask[yy, xx] > 0) & (texture[yy, xx] > 25)
            self.pixel_world = np.column_stack([xx[good], yy[good]]).astype(np.float32)+atlas.origin
            self.pixel_values = ag[yy[good], xx[good]].astype(np.float32)/255
        image = t.as_tensor(np.ascontiguousarray(gray), device=self.device).float().flatten()/255
        usable = t.as_tensor(allowed > 0, device=self.device).flatten()
        d = t.arange(-24, 25, device=self.device)
        dy, dx = t.meshgrid(d, d, indexing="ij")
        shifts = t.stack([dx.flatten(), dy.flatten()], 1).float()
        solutions = []
        with t.inference_mode():
            for _, camera, _, _ in hypotheses:
                screen = (self.pixel_world-camera)*scale
                keep = np.flatnonzero((screen[:, 0] > 26*scale) & (screen[:, 0] < w-26*scale) &
                                      (screen[:, 1] > 26*scale) & (screen[:, 1] < h-26*scale))
                if len(keep) < 100:
                    continue
                keep = keep[np.linspace(0, len(keep)-1, min(768, len(keep))).astype(int)]
                positions = t.as_tensor(screen[keep], device=self.device)
                reference = t.as_tensor(self.pixel_values[keep], device=self.device)
                xy = (positions[None]-shifts[:, None]*scale).round().long()
                index = xy[:, :, 1].clamp(0, h-1)*w+xy[:, :, 0].clamp(0, w-1)
                valid = usable[index] & (xy[:, :, 0] >= 0) & (xy[:, :, 0] < w) & (xy[:, :, 1] >= 0) & (xy[:, :, 1] < h)
                error = (image[index]-reference).abs()
                counts = valid.sum(1)
                scores = ((error < .06) & valid).sum(1)/counts.clamp(min=1)
                scores[counts < 100] = -1
                # Keep both local winner and a spatially distinct alternative.
                for _ in range(2):
                    best = int(scores.argmax().item())
                    score = float(scores[best].item())
                    if score < 0:
                        break
                    refined = camera+shifts[best].cpu().numpy()
                    accepted = valid[best] & (error[best] < .06)
                    pts = xy[best][accepted].float().cpu().numpy()
                    solutions.append((score, refined, pts))
                    scores[t.linalg.vector_norm(shifts-shifts[best], dim=1) < 5] = -1
        if not solutions:
            return None, "No opaque world pixels available to verify learned features"
        solutions.sort(key=lambda s: s[0], reverse=True)
        score, camera, points = solutions[0]
        rivals = [s for s in solutions[1:] if np.linalg.norm(s[1]-camera) >= 5]
        rival_score = max((s[0] for s in rivals), default=0)
        self.last_pixel_quality = dict(score=score, rival_score=rival_score)
        spread = np.ptp(points, axis=0) if len(points) else np.zeros(2)
        cells = len(np.unique(np.floor(points/80), axis=0)) if len(points) else 0
        if (score < .60 or score-rival_score < .06 or len(points) < 100 or
                max(spread) < 120 or min(spread) < 35 or cells < 4):
            return (None, len(points), round(rival_score*len(points)/max(score, .01))), "Learned proposal rejected by opaque-pixel verification"
        return (camera, len(points), round(rival_score*len(points)/score), points, None), "XFeat proposal refined and verified against opaque world pixels"
