"""按怪物 ID 文件夹加载的多尺度模板匹配后端。"""

from __future__ import annotations

import os
import cv2
from typing import Any, Dict, List, Tuple

try:
    from multi_scale_monster_tracker import MonsterTemplateMatcher
except Exception:
    MonsterTemplateMatcher = None


RawCandidate = Tuple[int, int, int, int, str, float]


class MultiScaleMonsterBackend:
    """只匹配当前地图 Mob ID 对应的 ``templates/<MobID>/`` 文件夹。"""

    def __init__(self, template_root: str, threshold: float = 0.52,
                 template_scale: float = 1.0,
                 compute_device: str = "auto", coarse_scale: float = 0.6):
        self.template_root = template_root
        self.threshold = float(threshold)
        self.template_scale = float(template_scale)
        self.compute_device = compute_device
        self.coarse_scale = float(coarse_scale)
        self.matchers: List[Any] = []
        # 兼容旧检测器：调用方只需判断该列表是否为空。
        self.template_items: List[Any] = []

    @property
    def available(self) -> bool:
        return bool(self.matchers and self.template_items)

    def clear(self) -> None:
        for matcher in self.matchers:
            close = getattr(matcher, "close", None)
            if callable(close):
                close()
        self.matchers = []
        self.template_items = []

    def load_for_mobs(self, mobs: List[Dict[str, Any]]) -> List[str]:
        self.clear()
        if MonsterTemplateMatcher is None:
            return []
        loaded_ids: List[str] = []
        seen_ids = set()
        for mob in mobs or []:
            mob_id = str(mob.get("id", "")).strip()
            if not mob_id or mob_id in seen_ids:
                continue
            seen_ids.add(mob_id)
            mob_dir = os.path.join(self.template_root, mob_id)
            if not os.path.isdir(mob_dir):
                continue
            try:
                matcher = MonsterTemplateMatcher(
                    template_dir=mob_dir,
                    # 模板文件保留原始尺寸；实测 1080p 游戏内怪物为
                    # 90x63px，而对应源模板为 67x46px，联合拟合比例约
                    # 1.41x（避免旧版下载阶段 1.80x 与匹配尺度叠加）。
                    scale=self.template_scale,
                    match_threshold=self.threshold,
                    nms_threshold=0.35,
                    # 11px 高的眼部特征在 0.30x 下会小于 4px 而被跳过。
                    # CUDA 承担较高分辨率的粗检，最终分数仍由原图精检决定。
                    coarse_scale=self.coarse_scale,
                    max_workers=4,
                    compute_device=self.compute_device,
                )
            except Exception:
                continue
            # 文件可统一叫 mob.png；类别由父目录 Mob ID 提供。
            for item in matcher.templates:
                item.name = f"mob_{mob_id}"
            self.matchers.append(matcher)
            self.template_items.extend(matcher.templates)
            loaded_ids.append(mob_id)
            print(f"[MonsterBackend] Mob {mob_id}: {matcher.compute_device.upper()}, "
                  f"coarse={matcher.coarse_scale:g}, threshold={matcher.match_threshold:g}")
        return loaded_ids

    @staticmethod
    def _allowed_rectangles(width: int, height: int, exclusions) -> List[Tuple[int, int, int, int]]:
        """
        仅按 Y 轴处理屏蔽区，X 方向强制拉满全宽 (0~width)，
        彻底保证最终送入匹配器的未屏蔽区域始终是 1 个单一矩形，绝不产生左右碎片。
        """
        rects = [(0, 0, int(width), int(height))]
        for item in exclusions or []:
            try:
                # 强制 X 全宽，仅在 Y 轴上进行扣减
                ex1 = 0
                ex2 = int(width)
                ey1 = max(0, int(item.get("y", 0)))
                ey2 = min(height, ey1 + max(0, int(item.get("h", 0))))
            except Exception:
                continue
            if ey2 <= ey1:
                continue
            next_rects = []
            for x, y, w, h in rects:
                ry2 = y + h
                iy1, iy2 = max(y, ey1), min(ry2, ey2)
                if iy2 <= iy1:
                    next_rects.append((x, y, w, h))
                    continue
                # 仅在 Y 方向上切割，保留未屏蔽部分
                if iy1 > y:
                    next_rects.append((0, y, int(width), iy1 - y))
                if iy2 < ry2:
                    next_rects.append((0, iy2, int(width), ry2 - iy2))
            rects = next_rects
        res = [(x, y, w, h) for x, y, w, h in rects if w >= 8 and h >= 8]
        return res if res else [(0, 0, int(width), int(height))]

    def detect(self, frame, threshold: float, exclusion_regions=None) -> List[RawCandidate]:
        if frame is None or not self.available:
            return []
        candidates: List[RawCandidate] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # 将整帧拆成未屏蔽 ROI；粗匹配和细匹配都只在这些区域执行。
        active_exclusions = [r for r in (exclusion_regions or []) if r.get("monster", True)]
        fh, fw = gray.shape[:2]
        allowed = self._allowed_rectangles(fw, fh, active_exclusions)
        for ox, oy, rw, rh in allowed:
            roi_gray = gray[oy:oy + rh, ox:ox + rw]
            roi_source = frame[oy:oy + rh, ox:ox + rw]
            pyramids = {}
            cuda_frames = {}
            for matcher in self.matchers:
                matcher.match_threshold = float(threshold)
                cs = matcher.coarse_scale
                if cs not in pyramids:
                    pyramids[cs] = (cv2.resize(roi_gray, (0, 0), fx=cs, fy=cs,
                                              interpolation=cv2.INTER_AREA) if cs < 0.99 else roi_gray)
                shared_frame = None
                cuda_peaks = getattr(matcher, "_cuda_peaks", None)
                if cuda_peaks is not None and cs < 0.99:
                    if cs not in cuda_frames:
                        try:
                            cuda_frames[cs] = cuda_peaks.prepare_frame(pyramids[cs])
                        except Exception:
                            # The matcher's existing CUDA-to-CPU fallback handles
                            # device errors; sharing is only an optimization.
                            cuda_frames[cs] = None
                    shared_frame = cuda_frames[cs]
                kwargs = {"cuda_frame_context": shared_frame} if shared_frame is not None else {}
                for det in matcher.detect(
                    roi_source, coarse_frame=pyramids[cs], gray_frame=roi_gray,
                    **kwargs,
                ):
                    x, y, w, h = det["bbox"]
                    candidates.append((int(x + ox), int(y + oy), int(w), int(h),
                                       str(det["label"]), float(det["score"])))
        return candidates
