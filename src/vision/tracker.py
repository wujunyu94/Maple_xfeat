"""
tracker.py - 小地图识别与角色定位追踪引擎
实现对小地图内部平台区域的自适应定位、角色黄点的高精度提取、归一化坐标与运动速度估计，
以及其他玩家红点、传送门等实体的检测与调试画面渲染。
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2


@dataclass
class TrackerResult:
    is_detected: bool = False
    pixel_pos: Optional[Tuple[int, int]] = None       # 小地图内部像素坐标 (x, y)
    subpixel_pos: Optional[Tuple[float, float]] = None # 黄色连通域重心亚像素坐标
    norm_pos: Optional[Tuple[float, float]] = None    # 归一化坐标 (0.0~1.0, 0.0~1.0)
    velocity: Tuple[float, float] = (0.0, 0.0)        # 归一化速度 (vx, vy) 单位: 1/s
    speed: float = 0.0                                # 标量速率
    other_players: List[Tuple[int, int]] = field(default_factory=list) # 其他玩家红点坐标
    portals: List[Tuple[int, int]] = field(default_factory=list)       # 传送门蓝圈坐标
    inner_box: Optional[Tuple[int, int, int, int]] = None # (x, y, w, h) 内部小地图边界
    timestamp: float = 0.0


class MinimapTracker:
    DEFAULT_YELLOW_CANDIDATE_SIZES = {
        (4, 4), (4, 5), (5, 4), (5, 5), (6, 5), (5, 6), (6, 6),
        (6, 7), (7, 6), (7, 7),
    }

    def __init__(
        self,
        # 默认小地图在客户区左上角的完整搜索范围 (rel_x, rel_y, width, height) - 宽幅自适应支持地铁/废都/神木等超长超宽小地图
        search_roi: Tuple[int, int, int, int] = (0, 0, 450, 360),
        # 历史轨迹缓存长度（用于绘制运动轨迹尾迹）
        history_length: int = 40,
        # 调试/原始读数场景应逐帧走 HSV 连通域，不能被上一帧模板位置
        # 限制在一个小搜索半径里。
        enable_template_tracking: bool = True,
    ):
        self.search_roi = search_roi
        self.history_length = history_length
        self.enable_template_tracking = enable_template_tracking
        self.yellow_candidate_sizes = set(self.DEFAULT_YELLOW_CANDIDATE_SIZES)
        self.position_history: deque = deque(maxlen=history_length)

        # 缓存的小地图内部平台矩形区域 (x_rel, y_rel, w, h)
        self.cached_inner_box: Optional[Tuple[int, int, int, int]] = None
        # 手动框由用户负责确认，不参与自动可信度淘汰；切图或恢复自动时清除。
        self._manual_inner_box_locked = False
        self._last_detect_time: float = 0.0
        self._last_norm_pos: Optional[Tuple[float, float]] = None

        # EMA 平滑系数 (0~1, 越大越偏向新值)
        self.smooth_alpha: float = 0.85
        self.smoothed_norm_pos: Optional[Tuple[float, float]] = None
        self._consecutive_misses: int = 0
        # 首次确认黄点后缓存其真实像素模板。角色在相邻地图纹理旁移动时，
        # HSV 连通域可能合并变形，模板可继续锁定原本的 5~6px 黄点。
        self._yellow_dot_template: Optional[np.ndarray] = None
        self._yellow_dot_template_mask: Optional[np.ndarray] = None
        # 来自当前地图 JSON Canvas 的实际像素尺寸。只用来剔除检测框
        # 额外吞入的边框；卷轴轴的可视尺寸小于 Canvas 时绝不强行放大。
        self.expected_canvas_size: Optional[Tuple[int, int]] = None
        # 从当前地图 IMG miniMap.canvas 解码的原始背景。它不含人物黄点、
        # 传送点与玩家标记，可用于排除地图内部纹理形成的假边框。
        self.expected_canvas_gray: Optional[np.ndarray] = None
        self.expected_canvas_grays: Tuple[np.ndarray, ...] = ()
        self._expected_canvas_identity: Optional[Any] = None
        # 自动框不能只做几何复检。切图加载帧中的旧地图/过渡纹理也可能
        # 形成位置、尺寸都合法的矩形；若不再核对 WZ 背景，它会一直污染
        # 黄点归一化坐标。背景复检本身只有一次小尺寸 templateMatch，远比
        # 重新跑完整轮廓枚举便宜，因此按间隔执行并以连续两次失配淘汰。
        self._last_auto_box_background_check: float = 0.0
        self._auto_box_background_misses: int = 0
        self._auto_box_background_check_interval: float = 0.45
        self._auto_box_stable_confirmations: int = 0
        # 切图后的前几帧通常仍是旧地图或黑屏。此时当前 WZ 背景必然
        # 匹配失败，继续跑数百个 trim 组合只会把 60Hz 黄点线程阻塞
        # 约 0.4s。先让廉价的精确候选承担快速命中；昂贵穷举只在画面
        # 稳定一小段时间后、且按低频率兜底一次。
        self._canvas_exhaustive_search_not_before: float = 0.0
        self._last_canvas_exhaustive_search_at: float = 0.0
        self._canvas_exhaustive_warmup_sec: float = 0.75
        self._canvas_exhaustive_cooldown_sec: float = 1.0

    def _defer_canvas_exhaustive_search(self) -> None:
        self._canvas_exhaustive_search_not_before = (
            time.perf_counter() + self._canvas_exhaustive_warmup_sec
        )
        self._last_canvas_exhaustive_search_at = 0.0

    def _allow_canvas_exhaustive_search(self) -> bool:
        """限制破损边框兜底穷举，避免切图过渡帧连续阻塞追踪线程。"""
        now = time.perf_counter()
        if now < self._canvas_exhaustive_search_not_before:
            return False
        if (
            self._last_canvas_exhaustive_search_at > 0.0
            and now - self._last_canvas_exhaustive_search_at
            < self._canvas_exhaustive_cooldown_sec
        ):
            return False
        self._last_canvas_exhaustive_search_at = now
        return True

    def set_yellow_candidate_sizes(self, sizes) -> None:
        """更新黄点连通域允许的宽×高尺寸集合，兼容字符串 '4x5' 和元组/列表 (4, 5)。"""
        parsed = set()
        for item in sizes or []:
            try:
                if isinstance(item, str):
                    clean = item.lower().replace("*", "x").replace(",", "x")
                    parts = clean.split("x")
                    if len(parts) == 2:
                        w, h = int(parts[0].strip()), int(parts[1].strip())
                    else:
                        continue
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    w, h = int(item[0]), int(item[1])
                else:
                    continue

                if 2 <= w <= 12 and 2 <= h <= 12:
                    parsed.add((w, h))
            except (TypeError, ValueError):
                continue
        if parsed:
            self.yellow_candidate_sizes = parsed
            # 尺寸规则变化后，旧模板可能来自已被禁用的候选尺寸，立即丢弃。
            self._yellow_dot_template = None
            self._yellow_dot_template_mask = None

    def set_expected_canvas_size(self, size: Optional[Tuple[int, int]]) -> None:
        """设置当前地图 JSON Canvas 尺寸，None 表示不施加尺寸约束。"""
        previous_size = self.expected_canvas_size
        if size is None:
            self.expected_canvas_size = None
        else:
            try:
                width, height = int(size[0]), int(size[1])
                self.expected_canvas_size = (
                    (width, height) if width > 0 and height > 0 else None
                )
            except (TypeError, ValueError, IndexError):
                self.expected_canvas_size = None

        # 地图识别完成到拓扑/IMG 载入完成之间存在一个短窗口：追踪器可能
        # 已经用通用兜底框缓存了错误范围。Canvas 尺寸稍后到达时必须让
        # 下一帧重新框选，否则错误框会一直保留到下一次切图。
        if self.expected_canvas_size != previous_size:
            self._defer_canvas_exhaustive_search()
            self.cached_inner_box = None
            self._manual_inner_box_locked = False
            self._last_auto_box_background_check = 0.0
            self._auto_box_background_misses = 0
            self._auto_box_stable_confirmations = 0
            self._last_detect_time = 0.0
            self._last_norm_pos = None
            self.smoothed_norm_pos = None
            self._consecutive_misses = 0
            self._yellow_dot_template = None
            self._yellow_dot_template_mask = None
            self.position_history.clear()

    def set_expected_canvas_image(
        self,
        image: Optional[np.ndarray],
        identity: Optional[Any] = None,
        image_variants: Optional[Tuple[np.ndarray, ...]] = None,
    ) -> None:
        """设置当前地图的 WZ 小地图背景，供自动框选做反向验证。"""
        gray = None
        if image is not None and getattr(image, "size", 0) > 0:
            if image.ndim == 2:
                gray = np.ascontiguousarray(image.astype(np.uint8, copy=False))
            else:
                gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        variants = [gray] if gray is not None else []
        for variant in image_variants or ():
            if variant is None or getattr(variant, "size", 0) == 0:
                continue
            variant_gray = (
                np.ascontiguousarray(variant.astype(np.uint8, copy=False))
                if variant.ndim == 2 else cv2.cvtColor(variant[:, :, :3], cv2.COLOR_BGR2GRAY)
            )
            if gray is not None and variant_gray.shape != gray.shape:
                continue
            if not any(np.array_equal(variant_gray, existing) for existing in variants):
                variants.append(variant_gray)
        new_identity = (
            identity,
            None if gray is None else (int(gray.shape[1]), int(gray.shape[0])),
        )
        if new_identity == self._expected_canvas_identity:
            self.expected_canvas_gray = gray
            self.expected_canvas_grays = tuple(variants)
            return

        self._expected_canvas_identity = new_identity
        self.expected_canvas_gray = gray
        self.expected_canvas_grays = tuple(variants)
        self._defer_canvas_exhaustive_search()
        if gray is not None:
            self.expected_canvas_size = (int(gray.shape[1]), int(gray.shape[0]))
        # 即使两张地图 Canvas 尺寸相同，背景也完全不同；MapID/identity
        # 变化时必须淘汰上一张地图缓存的框和黄点模板。
        self.cached_inner_box = None
        self._manual_inner_box_locked = False
        self._last_auto_box_background_check = 0.0
        self._auto_box_background_misses = 0
        self._auto_box_stable_confirmations = 0
        self._last_detect_time = 0.0
        self._last_norm_pos = None
        self.smoothed_norm_pos = None
        self._consecutive_misses = 0
        self._yellow_dot_template = None
        self._yellow_dot_template_mask = None
        self.position_history.clear()

    def set_manual_inner_box(self, box: Optional[Tuple[int, int, int, int]]) -> None:
        """手动设定小地图内画布范围 (x, y, w, h)，None 表示恢复自动探测。"""
        if box is None:
            self.cached_inner_box = None
            self._manual_inner_box_locked = False
            self._last_auto_box_background_check = 0.0
            self._auto_box_background_misses = 0
            self._auto_box_stable_confirmations = 0
        else:
            try:
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if w >= 10 and h >= 10:
                    self.cached_inner_box = (x, y, w, h)
                    self._manual_inner_box_locked = True
                    self._last_auto_box_background_check = 0.0
                    self._auto_box_background_misses = 0
                    self._auto_box_stable_confirmations = 0
            except (TypeError, ValueError, IndexError):
                self.cached_inner_box = None
                self._manual_inner_box_locked = False

    def reset_map_calibration(self) -> None:
        """地图切换后清除上一张地图的小地图框与轨迹状态。"""
        self.cached_inner_box = None
        self._manual_inner_box_locked = False
        self._last_auto_box_background_check = 0.0
        self._auto_box_background_misses = 0
        self._auto_box_stable_confirmations = 0
        self._last_detect_time = 0.0
        self._last_norm_pos = None
        self.smoothed_norm_pos = None
        self._consecutive_misses = 0
        self._yellow_dot_template = None
        self._yellow_dot_template_mask = None
        self.expected_canvas_size = None
        self.expected_canvas_gray = None
        self.expected_canvas_grays = ()
        self._expected_canvas_identity = None
        self._defer_canvas_exhaustive_search()
        self.position_history.clear()

    @staticmethod
    def _refine_pure_dark_client_rect(roi_img: np.ndarray, outer_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        从外部小地图边框中，像素级内缩收敛，精确提取 100% 纯暗色地图物理画布（完全剔除四周灰白边框与分割线）
        """
        x, y, w, h = outer_box
        sub = roi_img[y:y+h, x:x+w]
        if sub.size == 0:
            return outer_box
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        sh, sw = gray.shape

        # 1. 优先扫描顶部与底部浅色横线 (边框/标题装饰横条)
        # 必须先剔除上下浅色横条，否则左右列均值会被上下整行的浅色像素拉高，
        # 导致把本来属于深色地图画布的最左/最右列误当成边框过度切除。
        edge_scan_y = min(12, max(2, sh // 10))
        top_in = 0
        while top_in < edge_scan_y:
            if np.mean(gray[top_in, :]) < 90:
                break
            top_in += 1

        bot_in = 0
        while bot_in < edge_scan_y:
            if np.mean(gray[sh - 1 - bot_in, :]) < 90:
                break
            bot_in += 1

        # 2. 在纯净的内部有效行高范围内，扫描左右两侧浅色垂直边框
        valid_rows = gray[top_in:max(top_in + 5, sh - bot_in), :]
        edge_scan_x = min(12, max(2, sw // 15))
        left_in = 0
        while left_in < edge_scan_x:
            if np.mean(valid_rows[:, left_in]) < 90:
                break
            left_in += 1

        right_in = 0
        while right_in < edge_scan_x:
            if np.mean(valid_rows[:, sw - 1 - right_in]) < 90:
                break
            right_in += 1

        pure_x = x + left_in
        pure_y = y + top_in
        pure_w = max(10, w - left_in - right_in)
        pure_h = max(10, h - top_in - bot_in)

        return (pure_x, pure_y, pure_w, pure_h)

    def _refine_minimap_inner_box(self, roi_img: np.ndarray, outer_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """剥离嵌套的小地图边框，直到边缘已是实际暗色画布。"""
        box = outer_box
        # 有些小地图有“外框 + 内侧浅色描边”两层边缘。旧实现只剥一层，
        # 会保留两侧各 3px 的亮边，导致 110px Canvas 被报成 116px。
        for _ in range(3):
            refined = self._refine_pure_dark_client_rect(roi_img, box)
            if refined == box:
                break
            box = refined
        return box

    def _canvas_backgrounds(self) -> Tuple[np.ndarray, ...]:
        return self.expected_canvas_grays or (
            (self.expected_canvas_gray,) if self.expected_canvas_gray is not None else ()
        )

    def _find_full_canvas_box(
        self, roi_img: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """完整 WZ 画布可见时，直接匹配整图，不依赖内部轮廓。"""
        backgrounds = self._canvas_backgrounds()
        if not backgrounds or roi_img is None or roi_img.size == 0:
            return None
        canvas_h, canvas_w = backgrounds[0].shape[:2]
        roi_h, roi_w = roi_img.shape[:2]
        x0, y0 = 2, 55
        x1 = min(24, roi_w - canvas_w)
        y1 = min(145, roi_h - canvas_h)
        if x1 < x0 or y1 < y0:
            return None
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        search = gray[y0:y1 + canvas_h, x0:x1 + canvas_w]
        best_score, best_xy = -1.0, None
        for background in backgrounds:
            try:
                response = cv2.matchTemplate(search, background, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(response)
            except cv2.error:
                continue
            if np.isfinite(score) and score > best_score:
                best_score, best_xy = float(score), (x0 + location[0], y0 + location[1])
        # 整图含动态黄点仍应高度相似。低分时宁可等待下一帧，也不能
        # 将切图过渡纹理误识别成完整小地图。
        if best_xy is None or best_score < 0.78:
            return None
        return best_xy[0], best_xy[1], canvas_w, canvas_h

    def _find_canvas_backed_inner_box(
        self,
        roi_img: np.ndarray,
        contours,
    ) -> Optional[Tuple[int, int, int, int]]:
        """用 WZ miniMap.canvas 背景验证并选择面积最大的真实可视框。

        人物黄点等动态前景只占极少像素。真实屏幕画布必定能作为 WZ
        原图的一个子区域获得较高匹配分；内部纹理小框即使局部得分更高，
        也会因为覆盖面积较小在综合分中落后。
        """
        backgrounds = self._canvas_backgrounds()
        if not backgrounds:
            return None
        direct_box = self._find_full_canvas_box(roi_img)
        if direct_box is not None:
            return direct_box
        full_gray = backgrounds[0]
        canvas_h, canvas_w = full_gray.shape[:2]
        if canvas_w < 40 or canvas_h < 25:
            return None

        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        raw_boxes = set()
        for contour in contours:
            x, y, bw, bh = map(int, cv2.boundingRect(contour))
            if (
                0 <= x <= 30
                and 45 <= y <= 155
                and 70 <= bw <= roi_img.shape[1] - 2
                and 30 <= bh <= roi_img.shape[0] - 2
            ):
                raw_boxes.add((x, y, bw, bh))
        if not raw_boxes:
            return None

        def score_box(candidate):
            sx, sy, view_w, view_h = map(int, candidate)
            if (
                not 2 <= sx <= 24
                or not 55 <= sy <= 145
                or view_w < 70
                or view_h < 30
                or view_w > canvas_w
                or view_h > canvas_h
            ):
                return None
            # 非卷轴小地图的 WZ 画布本身很小。内部纹理裁下的一条窄带
            # 不能当作整幅画布；103000201 曾把 247x79 框成 241x33。
            if canvas_h <= 100 and view_h < canvas_h * 0.80:
                return None
            crop = roi_gray[sy:sy + view_h, sx:sx + view_w]
            if crop.shape != (view_h, view_w) or float(np.std(crop)) < 3.0:
                return None
            try:
                matches = [
                    cv2.minMaxLoc(cv2.matchTemplate(background, crop, cv2.TM_CCOEFF_NORMED))
                    for background in backgrounds
                ]
                match_score, match_loc = max(
                    ((item[1], item[3]) for item in matches), key=lambda item: item[0]
                )
            except cv2.error:
                return None
            if not np.isfinite(match_score) or match_score < 0.18:
                return None
            area = float(view_w * view_h)
            return (
                float(match_score) * math.sqrt(area),
                float(match_score),
                area,
                (sx, sy, view_w, view_h),
                match_loc,
            )

        ordered_raw_boxes = sorted(
            raw_boxes, key=lambda box: box[2] * box[3], reverse=True
        )[:12]
        # 正常稳定画面可以直接由浅色外框剥离到真实画布，通常只需 1~2
        # 次背景匹配；下面的穷举仅保留给边框破损/被 UI 遮挡的少数情况。
        fast_scored = []
        seen_refined = set()
        for raw_box in ordered_raw_boxes:
            refined = self._refine_minimap_inner_box(roi_img, raw_box)
            if refined in seen_refined:
                continue
            seen_refined.add(refined)
            scored_item = score_box(refined)
            if scored_item is not None:
                fast_scored.append(scored_item)
        if fast_scored:
            fast_scored.sort(
                key=lambda item: (item[0], item[1], item[2]), reverse=True
            )
            return fast_scored[0][3]

        # 过渡帧与当前 WZ 背景不一致时，快速候选失败是预期行为。不要
        # 每帧立即进入昂贵的边框 trim 穷举；真正存在破损边框的地图在
        # 画面稳定后仍会得到低频兜底机会。
        if not self._allow_canvas_exhaustive_search():
            return None

        def axis_variants(raw_len: int, canvas_len: int, minimum: int):
            delta = raw_len - canvas_len
            if 0 <= delta <= 16:
                # 非卷轴轴：边框最多比 WZ Canvas 多十余像素，枚举其
                # 分配到两端的方式，输出长度直接锁定为 Canvas 长度。
                return [(lead, canvas_len) for lead in range(delta + 1)]
            # 卷轴轴：Canvas 大于屏幕视口，枚举外框两侧 0~6px 描边。
            variants = []
            for lead in range(7):
                for tail in range(7):
                    length = raw_len - lead - tail
                    if minimum <= length <= canvas_len:
                        variants.append((lead, length))
            return variants

        scored = []
        # 仅保留面积最大的若干闭合轮廓，防止复杂小地图产生无意义组合。
        for x, y, bw, bh in ordered_raw_boxes:
            x_variants = axis_variants(bw, canvas_w, 70)
            y_variants = axis_variants(bh, canvas_h, 30)
            for left_trim, view_w in x_variants:
                sx = x + left_trim
                if not 2 <= sx <= 24:
                    continue
                for top_trim, view_h in y_variants:
                    sy = y + top_trim
                    if not 55 <= sy <= 145:
                        continue
                    scored_item = score_box((sx, sy, view_w, view_h))
                    if scored_item is not None:
                        scored.append(scored_item)
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return scored[0][3]

    def _cached_box_background_score(
        self,
        roi_img: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[float]:
        """核对已缓存可视框是否仍是当前 WZ 小地图背景的一部分。"""
        backgrounds = self._canvas_backgrounds()
        if not backgrounds or roi_img is None or roi_img.size == 0:
            return None

        try:
            x, y, width, height = map(int, box)
        except (TypeError, ValueError):
            return None
        roi_h, roi_w = roi_img.shape[:2]
        if x < 0 or y < 0 or width < 30 or height < 20:
            return None
        width = min(width, roi_w - x)
        height = min(height, roi_h - y)
        if width < 30 or height < 20:
            return None

        # 与 detect() 的 Canvas 尺寸收敛保持一致，避免缓存框只多吞了几
        # 像素边框时因为模板比 WZ 原图稍大而被误判为背景失配。
        canvas_h, canvas_w = backgrounds[0].shape[:2]
        if canvas_w <= width and (width - canvas_w) <= max(12, int(canvas_w * 0.12)):
            x += (width - canvas_w) // 2
            width = canvas_w
        if canvas_h <= height and (height - canvas_h) <= max(12, int(canvas_h * 0.12)):
            y += (height - canvas_h) // 2
            height = canvas_h
        if width > canvas_w or height > canvas_h:
            return None

        crop = roi_img[y:y + height, x:x + width]
        if crop.shape[:2] != (height, width):
            return None
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.std(crop_gray)) < 3.0:
            return None
        try:
            score = max(
                cv2.minMaxLoc(cv2.matchTemplate(background, crop_gray, cv2.TM_CCOEFF_NORMED))[1]
                for background in backgrounds
            )
        except cv2.error:
            return None
        return float(score) if np.isfinite(score) else None

    def find_minimap_inner_box(self, roi_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        全自适应像素级小地图内画布边框探测器：
        1. 优先利用小地图内画布四周清晰闭合的浅灰/白色封闭矩形外框（Canny + RETR_TREE 轮廓分析）；
        2. 结合内部深色背景比例与均值亮度校验，像素级严密包裹深色地图画布；
        3. 自适应 1 行、2 行或 3 行文字标题栏，并通过 _refine_pure_dark_client_rect 彻底剥离所有灰色外边框与分割线。
        """
        if roi_img is None or roi_img.size == 0:
            return None

        h, w, _ = roi_img.shape
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        # 1. 精确封闭矩形边框探测
        edges = cv2.Canny(gray, 30, 100)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 已载入 Map IMG 时，以原始 miniMap.canvas 为最高优先级事实源。
        # 这一步同时完成背景滤除和边界选择，不再依赖地图内部暗色比例。
        canvas_backed_box = self._find_canvas_backed_inner_box(roi_img, cnts)
        if canvas_backed_box is not None:
            return canvas_backed_box
        # 已有当前地图 WZ 背景时，匹配失败意味着此帧仍在切图/展开动画，
        # 或候选框根本不是小地图。此时宁可等待下一帧，也不能退回仅凭
        # 几何形状的候选并把错误范围永久缓存。
        if self.expected_canvas_gray is not None:
            return None

        canvas_candidates = []
        for cnt in cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 内画布位置与尺寸约束：锚定左上角 (x in [3..18], y in [45..110])
            # 标题/任务栏通常高度很小；地图画布位于其下方。部分地图的
            # Canvas 很矮，例如 100020000 只有 151x69，不能再用 80px
            # 作为硬下限，否则会落入固定兜底框并截错整个小地图。
            if 3 <= x <= 18 and 60 <= y <= 120 and 100 <= bw <= (w - 6) and 35 <= bh <= (h - 10):
                aspect = bw / float(bh)
                if 0.4 <= aspect <= 5.0:
                    sub = roi_img[y+2:y+bh-2, x+2:x+bw-2]
                    if sub.size > 0:
                        mean_bgr = float(np.mean(sub))
                        dark_ratio = float(np.mean((sub[:, :, 0] < 85) & (sub[:, :, 1] < 85) & (sub[:, :, 2] < 85)))
                        # 彩色平台铺满矮 Canvas 时，真实小地图暗像素占比会
                        # 降到约 0.30～0.35。结合左上锚点、矩形闭合轮廓与
                        # 平均亮度即可安全放宽，避免把平台纹理当成外框。
                        if dark_ratio > 0.25 and mean_bgr < 120:
                            refined = self._refine_minimap_inner_box(
                                roi_img, (x, y, bw, bh)
                            )
                            refined_x, refined_y, refined_w, refined_h = refined
                            # 真正的小地图画布紧贴窗口左侧。加载中的地图纹理
                            # 也可能形成大闭合矩形，并从合法的外轮廓出发被
                            # refine 连续向内剥到 x=30~50；这种内部纹理框
                            # 一旦缓存便会让全部黄点 X 带上固定偏移。
                            if (
                                not (2 <= refined_x <= 24)
                                or not (55 <= refined_y <= 145)
                                or (refined_x - x) > 18
                            ):
                                continue
                            expected_fit = 0.0
                            if self.expected_canvas_size:
                                expected_w, expected_h = self.expected_canvas_size
                                expected_fit = -(
                                    abs(refined_w - expected_w) / max(1.0, float(expected_w))
                                    + abs(refined_h - expected_h) / max(1.0, float(expected_h))
                                )
                            # 地图右侧可能包含平台、装饰和传送点，暗色比例反而
                            # 低于左侧子区域。若单纯按 dark_ratio 排序，会把
                            # 左半边的暗色子框误选成完整小地图。面积优先，
                            # 暗色比例仅作为次级评分；已有 WZ Canvas 时则先
                            # 选择尺寸最接近的闭合框。
                            score = (
                                1 if self.expected_canvas_size else 0,
                                expected_fit,
                                refined_w * refined_h,
                                dark_ratio,
                                -mean_bgr,
                            )
                            canvas_candidates.append((score, refined))

        if canvas_candidates:
            canvas_candidates.sort(key=lambda it: it[0], reverse=True)
            return canvas_candidates[0][1]

        # 2. 备选方案：深色连通域闭运算
        dark_mask = (roi_img[:, :, 0] < 75) & (roi_img[:, :, 1] < 75) & (roi_img[:, :, 2] < 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(dark_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cnts_dark, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_candidates = []
        for cnt in cnts_dark:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if 3 <= x <= 18 and 60 <= y <= 120 and 100 <= bw <= (w - 6) and 35 <= bh <= (h - 10):
                dark_candidates.append((bw * bh, (x, y, bw, bh)))

        if dark_candidates:
            dark_candidates.sort(key=lambda it: it[0], reverse=True)
            refined = self._refine_minimap_inner_box(
                roi_img, dark_candidates[0][1]
            )
            if 2 <= refined[0] <= 24 and 55 <= refined[1] <= 145:
                return refined

        # 加载帧中没有可靠闭合外框时宁可本帧返回未定位并在下一帧重试，
        # 不能缓存固定默认框；默认框落在内部纹理上会持续污染世界坐标。
        return None

    def detect(self, full_game_frame: np.ndarray) -> TrackerResult:
        """
        处理单帧游戏画面，提取角色位置及小地图状态。
        """
        now = time.perf_counter()
        res = TrackerResult(timestamp=now)

        if full_game_frame is None or full_game_frame.size == 0:
            return res

        # 1. 自适应裁剪小地图搜索区域 (支持任意屏幕分辨率与 DPI)
        fh, fw, _ = full_game_frame.shape
        sw = min(fw, max(360, int(fw * 0.40)))
        sh = min(fh, max(320, int(fh * 0.35)))
        sx, sy = 0, 0

        roi_img = full_game_frame[sy:sy+sh, sx:sx+sw]
        if roi_img.size == 0:
            return res

        # 2. 定位内部小地图边界。自动缓存也要做轻量几何复检：切图
        # 加载帧可能曾把内部纹理误当画布，不能让该错误保持到下次切图。
        if self.cached_inner_box is not None and not self._manual_inner_box_locked:
            cbx, cby, cbw, cbh = self.cached_inner_box
            if not (
                2 <= int(cbx) <= 24
                and 55 <= int(cby) <= 145
                and int(cbw) >= 100
                and int(cbh) >= 35
            ):
                self.cached_inner_box = None
                self._yellow_dot_template = None
                self._yellow_dot_template_mask = None
                self.position_history.clear()
            elif (
                self.expected_canvas_gray is not None
                and now - self._last_auto_box_background_check
                >= self._auto_box_background_check_interval
            ):
                self._last_auto_box_background_check = now
                full_box = self._find_full_canvas_box(roi_img)
                if full_box is not None and full_box != self.cached_inner_box:
                    # 子区域也可能与 WZ 高分匹配，不能只靠缓存背景分维持
                    # 错误小框；完整画布恢复后立即提升为真实边界。
                    self.cached_inner_box = full_box
                    self._auto_box_background_misses = 0
                    self._auto_box_stable_confirmations = 2
                    self._yellow_dot_template = None
                    self._yellow_dot_template_mask = None
                    self.position_history.clear()
                score = self._cached_box_background_score(roi_img, self.cached_inner_box)
                if score is None or score < 0.18:
                    self._auto_box_background_misses += 1
                else:
                    self._auto_box_background_misses = 0
                if self._auto_box_background_misses >= 2:
                    self.cached_inner_box = None
                    self._auto_box_background_misses = 0
                    self._auto_box_stable_confirmations = 0
                    self._yellow_dot_template = None
                    self._yellow_dot_template_mask = None
                    self.position_history.clear()
                elif score is not None and score >= 0.18 and self._auto_box_stable_confirmations < 2:
                    # 初次框选可能发生在小地图展开/切图动画中。画面背景已
                    # 恢复后再独立重找两次；只有边界连续一致才转为稳定缓存。
                    fresh_box = self.find_minimap_inner_box(roi_img)
                    if fresh_box is not None:
                        if fresh_box != self.cached_inner_box:
                            self.cached_inner_box = fresh_box
                            self._auto_box_stable_confirmations = 0
                            self._yellow_dot_template = None
                            self._yellow_dot_template_mask = None
                            self.position_history.clear()
                        else:
                            self._auto_box_stable_confirmations += 1
        if self.cached_inner_box is None:
            self.cached_inner_box = self.find_minimap_inner_box(roi_img)
            if self.cached_inner_box is not None:
                self._last_auto_box_background_check = now
                self._auto_box_background_misses = 0
                self._auto_box_stable_confirmations = 0

        if not self.cached_inner_box:
            return res

        bx, by, bw, bh = self.cached_inner_box
        # 边界防溢出
        bx = max(0, min(bx, roi_img.shape[1] - 10))
        by = max(0, min(by, roi_img.shape[0] - 10))
        bw = min(bw, roi_img.shape[1] - bx)
        bh = min(bh, roi_img.shape[0] - by)

        # JSON Canvas 是地图本身的可信物理尺寸。检测框偶尔会把两侧/上下
        # 的浅色边框一起吞进来（如 100050000: 116x135 而 Canvas 为
        # 110x135）；此时必须从两侧均分裁除，而不是只裁右/下。若 Canvas
        # 比当前视口大，说明该轴可卷轴，保留当前较小的可视窗口，不能反向扩张。
        if self.expected_canvas_size:
            expected_w, expected_h = self.expected_canvas_size
            if expected_w <= bw and (bw - expected_w) <= max(12, int(expected_w * 0.12)):
                bx += (bw - expected_w) // 2
                bw = expected_w
            if expected_h <= bh and (bh - expected_h) <= max(12, int(expected_h * 0.12)):
                by += (bh - expected_h) // 2
                bh = expected_h

        inner_map = roi_img[by:by+bh, bx:bx+bw]
        res.inner_box = (sx + bx, sy + by, bw, bh)

        if inner_map.size == 0:
            return res

        # 3. 原始左上角黄点算法：HSV -> 连通域 -> 颜色/形状综合评分。
        # 角色点边缘会受抗锯齿、透明度和捕获后端影响。实机 P77 样本仍是
        # 标准 6x6 黄点，但其中一个边缘像素令 B 通道跨度达到 51；旧的
        # ``B range <= 48`` 硬门槛会让它在静止后永久消失。
        hsv = cv2.cvtColor(inner_map, cv2.COLOR_BGR2HSV)
        # 低饱和的亮黄抗锯齿也属于角色点。色相仍保持在很窄的黄色范围，
        # 再由下方的尺寸、紧凑度与颜色一致性排除地图线条。
        mask = cv2.inRange(hsv, np.array([22, 80, 185], np.uint8),
                           np.array([32, 255, 255], np.uint8))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detected_xy = None
        # 先以首次确认到的黄点掩膜模板跟踪。只比较黄点自身的掩膜像素，
        # 不比较四周地图背景，避免相邻纹理导致连通域合并后丢失目标。
        if (self.enable_template_tracking
                and self._yellow_dot_template is not None
                and self._yellow_dot_template_mask is not None
                and self.position_history):
            template = self._yellow_dot_template
            template_mask = self._yellow_dot_template_mask
            th, tw = template.shape[:2]
            last_x, last_y = self.position_history[-1][0]
            # 角色接近地图黄色综合特征时，HSV 连通域可能已经合并；
            # 模板搜索必须覆盖一帧内的正常位移，不能被 12px 半径限制住。
            radius = 20
            search_x1 = max(0, int(last_x - radius - tw // 2))
            search_y1 = max(0, int(last_y - radius - th // 2))
            search_x2 = min(bw, int(last_x + radius + tw // 2 + 1))
            search_y2 = min(bh, int(last_y + radius + th // 2 + 1))
            search = inner_map[search_y1:search_y2, search_x1:search_x2]
            if search.shape[0] >= th and search.shape[1] >= tw:
                match = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED, mask=template_mask)
                _, score, _, loc = cv2.minMaxLoc(match)
                # 只在黄点自身掩膜上评分，地图装饰不会因为连通而改变模板；
                # 允许轻微缩放/DPI/抗锯齿差异，但仍保留较高置信度门槛。
                if score >= 0.92:
                    detected_xy = (
                        float(search_x1 + loc[0] + (tw - 1) / 2.0),
                        float(search_y1 + loc[1] + (th - 1) / 2.0),
                    )

        # 模板尚未建立或模板得分不足时，使用原有 HSV 连通域逻辑获取新模板。
        if detected_xy is None:
            all_candidates = []

            def add_candidate(x0, y0, rw, rh, component, extracted=False):
                """评分一个完整组件，或从粘连组件中裁出的候选窗口。"""
                area = int(np.count_nonzero(component))
                if (rw, rh) not in self.yellow_candidate_sizes or not (5 <= area <= 60):
                    return
                density = area / float(max(1, rw * rh))
                if density < 0.28:
                    return
                component_bgr = inner_map[y0:y0 + rh, x0:x0 + rw][component]
                component_hsv = hsv[y0:y0 + rh, x0:x0 + rw][component]
                if component_bgr.size == 0 or component_hsv.size == 0:
                    return
                bgr_spread = np.ptp(component_bgr.astype(np.int16), axis=0)
                saturated_core = component_hsv[:, 1] >= 235
                saturated_core_ratio = float(np.mean(saturated_core))
                core_hues = component_hsv[saturated_core, 0].astype(np.float32)
                if core_hues.size >= 5:
                    hue_lo, hue_hi = np.percentile(core_hues, [10.0, 90.0])
                    core_hue_spread = float(hue_hi - hue_lo)
                    dominant_hue = float(np.median(core_hues))
                    dominant_hue_ratio = float(np.mean(np.abs(core_hues - dominant_hue) <= 3.0))
                else:
                    core_hue_spread = 180.0
                    dominant_hue_ratio = 0.0
                uniform_capture = bool(np.all(bgr_spread <= np.array([48, 112, 96])))
                antialiased_dot = bool(
                    saturated_core_ratio >= 0.55
                    and (
                        core_hue_spread <= 6.0
                        or dominant_hue_ratio >= 0.72
                    )
                )
                if not (uniform_capture or antialiased_dot):
                    return

                ys, xs = np.nonzero(component)
                cx, cy = float(x0 + np.mean(xs)), float(y0 + np.mean(ys))
                channel_std = float(np.mean(np.std(component_bgr.astype(np.float32), axis=0)))
                area_fit = max(0.0, 1.0 - abs(float(area) - 22.0) / 22.0)
                square_fit = 1.0 - abs(float(rw - rh)) / max(1.0, float(max(rw, rh)))
                density_fit = max(0.0, 1.0 - abs(density - 0.64) / 0.50)
                # 真正黄点近似上下、左右对称。粘连平台线的窗口即便颜色
                # 相同，其翻转重合度通常也明显更低。
                sym_lr = 1.0 - float(np.mean(component != np.fliplr(component)))
                sym_ud = 1.0 - float(np.mean(component != np.flipud(component)))
                symmetry = max(0.0, (sym_lr + sym_ud) * 0.5)
                dot_score = (
                    1.35 * min(1.0, saturated_core_ratio / 0.55)
                    + 1.00 * max(0.0, 1.0 - channel_std / 28.0)
                    + 0.55 * area_fit
                    + 0.35 * square_fit
                    + 0.45 * density_fit
                    + 0.55 * symmetry
                    - (0.08 if extracted else 0.0)
                )
                all_candidates.append({
                    "cx": cx, "cy": cy, "area": area,
                    "rw": int(rw), "rh": int(rh), "x": int(x0), "y": int(y0),
                    "mask": component.astype(np.uint8) * 255,
                    "score": float(dot_score),
                })

            for i in range(1, num_labels):
                x0, y0, rw, rh, area = stats[i]
                component = labels[y0:y0 + rh, x0:x0 + rw] == i
                add_candidate(int(x0), int(y0), int(rw), int(rh), component)

                # 黄点可能与一小段同色平台线连成 8x6 等超尺寸组件。
                # 只对有限范围的粘连块扫描用户允许的标准窗口，避免在
                # 整张黄色地图纹理中穷举并制造大量伪候选。
                if (
                    (rw, rh) not in self.yellow_candidate_sizes
                    and 5 <= area <= 120 and rw <= 24 and rh <= 16
                ):
                    for cw, ch in self.yellow_candidate_sizes:
                        if cw > rw or ch > rh:
                            continue
                        for oy in range(0, int(rh - ch + 1)):
                            for ox in range(0, int(rw - cw + 1)):
                                window = component[oy:oy + ch, ox:ox + cw]
                                add_candidate(
                                    int(x0 + ox), int(y0 + oy), int(cw), int(ch),
                                    window, extracted=True,
                                )
            if all_candidates:
                all_candidates.sort(
                    key=lambda item: (item["score"], -abs(item["area"] - 22)),
                    reverse=True,
                )
                best_cand = all_candidates[0]
                detected_xy = (float(best_cand["cx"]), float(best_cand["cy"]))
                x0, y0 = best_cand["x"], best_cand["y"]
                self._yellow_dot_template = inner_map[
                    y0:y0 + best_cand["rh"], x0:x0 + best_cand["rw"]
                ].copy()
                self._yellow_dot_template_mask = best_cand["mask"]

        if detected_xy:
            cx_f, cy_f = detected_xy
            res.is_detected = True
            res.subpixel_pos = (cx_f, cy_f)
            res.pixel_pos = (int(round(cx_f)), int(round(cy_f)))
            # 纯物理画布内零偏移归一化换算 (0.0 ~ 1.0)
            norm_x = np.clip(cx_f / float(bw), 0.0, 1.0)
            norm_y = np.clip(cy_f / float(bh), 0.0, 1.0)

            # norm_pos 保持为当帧的原始亚像素坐标，供世界坐标和卷轴视口换算。
            # 平滑值只用于速度估计，避免 EMA 滞后造成卷轴地图坐标漂移。
            res.norm_pos = (norm_x, norm_y)

            # 平滑滤波 (EMA)
            if self.smoothed_norm_pos is None:
                self.smoothed_norm_pos = (norm_x, norm_y)
            else:
                sx_pos = self.smooth_alpha * norm_x + (1 - self.smooth_alpha) * self.smoothed_norm_pos[0]
                sy_pos = self.smooth_alpha * norm_y + (1 - self.smooth_alpha) * self.smoothed_norm_pos[1]
                self.smoothed_norm_pos = (sx_pos, sy_pos)

            # 计算速度矢量
            if self._last_norm_pos is not None and self._last_detect_time > 0:
                dt = now - self._last_detect_time
                if 0.001 < dt < 1.0:
                    vx = (self.smoothed_norm_pos[0] - self._last_norm_pos[0]) / dt
                    vy = (self.smoothed_norm_pos[1] - self._last_norm_pos[1]) / dt
                    res.velocity = (vx, vy)
                    res.speed = float(np.hypot(vx, vy))

            self._last_norm_pos = self.smoothed_norm_pos
            self._last_detect_time = now
            self._consecutive_misses = 0

            # 记录历史轨迹
            self.position_history.append((res.pixel_pos, now))
        else:
            self._consecutive_misses += 1
            # 黄点短暂被特效、遮挡或截图帧异常影响时，不能连带重置已确认的
            # 小地图画布边界；画布边界只在切图/显式 reset_map_calibration 时重找。

        # 4. 提取其他玩家红点 (Red Dot: R > 180, G < 70, B < 70)
        mask_red = (inner_map[:, :, 2] > 180) & (inner_map[:, :, 1] < 75) & (inner_map[:, :, 0] < 75)
        red_pts = np.argwhere(mask_red)
        if len(red_pts) >= 2:
            # 简单连通域或聚类提取红点中心
            red_mask_u8 = mask_red.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(red_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if 2 <= cv2.contourArea(c) <= 60:
                    rx, ry, rw, rh = cv2.boundingRect(c)
                    res.other_players.append((rx + rw // 2, ry + rh // 2))

        # 5. 提取传送门蓝圈 (Blue Portal: B > 180, G > 140, R < 100)
        mask_portal = (inner_map[:, :, 0] > 180) & (inner_map[:, :, 1] > 130) & (inner_map[:, :, 2] < 100)
        portal_u8 = mask_portal.astype(np.uint8) * 255
        cnts_p, _ = cv2.findContours(portal_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cp in cnts_p:
            if 4 <= cv2.contourArea(cp) <= 120:
                px, py, pw, ph = cv2.boundingRect(cp)
                res.portals.append((px + pw // 2, py + ph // 2))

        return res

    def render_debug_view(self, full_game_frame: np.ndarray, result: TrackerResult, fps: float = 0.0) -> np.ndarray:
        """
        生成高质量的可视化调试图像，包含放大的小地图、准星十字瞄准线、历史轨迹尾迹与状态仪表盘。
        """
        if full_game_frame is None:
            return np.zeros((360, 400, 3), dtype=np.uint8)

        # 提取小地图及其周边区域
        sx, sy, sw, sh = self.search_roi
        fh, fw, _ = full_game_frame.shape
        roi = full_game_frame[sy:min(sy+sh, fh), sx:min(sx+sw, fw)].copy()

        if result.inner_box:
            ix, iy, iw, ih = result.inner_box
            # 绘制小地图内部有效区域绿色边框
            rel_ix = ix - sx
            rel_iy = iy - sy
            cv2.rectangle(roi, (rel_ix, rel_iy), (rel_ix + iw, rel_iy + ih), (0, 255, 128), 2)

            # 绘制历史轨迹尾迹
            if len(self.position_history) >= 2:
                pts_list = list(self.position_history)
                for i in range(1, len(pts_list)):
                    pt1 = (rel_ix + pts_list[i-1][0][0], rel_iy + pts_list[i-1][0][1])
                    pt2 = (rel_ix + pts_list[i][0][0], rel_iy + pts_list[i][0][1])
                    alpha = i / float(len(pts_list))
                    thickness = max(1, int(alpha * 3))
                    color = (0, int(200 * alpha), int(255 * alpha))
                    cv2.line(roi, pt1, pt2, color, thickness)

            # 绘制其他玩家红点标记
            for r_pos in result.other_players:
                r_screen = (rel_ix + r_pos[0], rel_iy + r_pos[1])
                cv2.circle(roi, r_screen, 5, (0, 0, 255), -1)
                cv2.circle(roi, r_screen, 8, (0, 0, 255), 1)

            # 绘制传送门标记
            for p_pos in result.portals:
                p_screen = (rel_ix + p_pos[0], rel_iy + p_pos[1])
                cv2.circle(roi, p_screen, 6, (255, 200, 0), 2)

            # 绘制角色当前准星十字与坐标
            if result.is_detected and result.pixel_pos:
                cx, cy = result.pixel_pos
                px_screen = rel_ix + cx
                py_screen = rel_iy + cy

                # 准星外圈与中心点
                cv2.circle(roi, (px_screen, py_screen), 9, (0, 255, 255), 2) # 黄色准星外环
                cv2.circle(roi, (px_screen, py_screen), 2, (0, 0, 255), -1)   # 红色准星中心
                
                # 十字线
                cv2.line(roi, (px_screen - 14, py_screen), (px_screen + 14, py_screen), (0, 255, 255), 1)
                cv2.line(roi, (px_screen, py_screen - 14), (px_screen, py_screen + 14), (0, 255, 255), 1)

                # 速度方向指示箭头
                if result.speed > 0.05:
                    vx, vy = result.velocity
                    arrow_len = min(40, int(result.speed * 80))
                    if arrow_len > 5:
                        norm_v = np.hypot(vx, vy)
                        dx = int((vx / norm_v) * arrow_len)
                        dy = int((vy / norm_v) * arrow_len)
                        cv2.arrowedLine(roi, (px_screen, py_screen), (px_screen + dx, py_screen + dy),
                                       (0, 165, 255), 2, tipLength=0.3)

        # 放大画面以便于观察调试 (缩放到 1.5x)
        scale = 1.3
        resized_view = cv2.resize(roi, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        rh, rw, _ = resized_view.shape

        # 底部信息面板 (HUD Panel)
        panel_h = 100
        hud_canvas = np.zeros((rh + panel_h, rw, 3), dtype=np.uint8)
        hud_canvas[:rh, :rw] = resized_view

        # 背景面板填充
        cv2.rectangle(hud_canvas, (0, rh), (rw, rh + panel_h), (25, 25, 28), -1)
        cv2.line(hud_canvas, (0, rh), (rw, rh), (60, 60, 65), 1)

        # 渲染 HUD 文本
        status_color = (0, 255, 128) if result.is_detected else (0, 0, 255)
        status_text = "TRACKING ACTIVE" if result.is_detected else "SEARCHING..."
        cv2.putText(hud_canvas, f"STATUS: {status_text}", (12, rh + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, status_color, 1)

        if result.is_detected and result.norm_pos:
            nx, ny = result.norm_pos
            px, py = result.pixel_pos
            coord_str = f"Pos: ({nx:.3f}, {ny:.3f}) | Pixel: ({px:3d}, {py:3d})"
            speed_str = f"Speed: {result.speed:5.2f} (Vx:{result.velocity[0]:+.2f}, Vy:{result.velocity[1]:+.2f})"
            cv2.putText(hud_canvas, coord_str, (12, rh + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.putText(hud_canvas, speed_str, (12, rh + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

        # 帧率与实体信息
        fps_str = f"FPS: {fps:4.1f} | Players: {len(result.other_players)} | Portals: {len(result.portals)}"
        cv2.putText(hud_canvas, fps_str, (12, rh + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        return hud_canvas
