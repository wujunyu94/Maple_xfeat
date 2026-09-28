"""
tracker.py - 小地图识别与角色定位追踪引擎
实现对小地图内部平台区域的自适应定位、角色黄点的高精度提取、归一化坐标与运动速度估计，
以及其他玩家红点、传送门等实体的检测与调试画面渲染。
"""

import time
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
        self.position_history: deque = deque(maxlen=history_length)

        # 缓存的小地图内部平台矩形区域 (x_rel, y_rel, w, h)
        self.cached_inner_box: Optional[Tuple[int, int, int, int]] = None
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

    def set_expected_canvas_size(self, size: Optional[Tuple[int, int]]) -> None:
        """设置当前地图 JSON Canvas 尺寸，None 表示不施加尺寸约束。"""
        if size is None:
            self.expected_canvas_size = None
            return
        try:
            width, height = int(size[0]), int(size[1])
            self.expected_canvas_size = (width, height) if width > 0 and height > 0 else None
        except (TypeError, ValueError, IndexError):
            self.expected_canvas_size = None

    def reset_map_calibration(self) -> None:
        """地图切换后清除上一张地图的小地图框与轨迹状态。"""
        self.cached_inner_box = None
        self._last_detect_time = 0.0
        self._last_norm_pos = None
        self.smoothed_norm_pos = None
        self._consecutive_misses = 0
        self._yellow_dot_template = None
        self._yellow_dot_template_mask = None
        self.expected_canvas_size = None
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

        # 1. 优先扫描左右两侧的浅色外边框/图标列 (按整列均值)
        # 轮廓框本身已经锁定了地图画布；这里只剔除少量描边像素。
        # 不能扫描到 w//3：地图右侧常有亮色平台/装饰，会被误当成
        # “非暗色边框”，从而把完整画布错误收缩成左侧子区域。
        edge_scan = min(8, max(1, w // 20))
        left_in = 0
        while left_in < edge_scan:
            if np.mean(gray[:, left_in]) < 90:
                break
            left_in += 1

        right_in = 0
        while right_in < edge_scan:
            if np.mean(gray[:, w - 1 - right_in]) < 90:
                break
            right_in += 1

        # 2. 在纯净的内部有效列宽范围内，扫描顶部与底部浅色横线
        col_s = left_in
        col_e = max(col_s + 10, w - right_in)

        top_in = 0
        edge_scan_y = min(8, max(1, h // 20))
        while top_in < edge_scan_y:
            if np.mean(gray[top_in, col_s:col_e]) < 90:
                break
            top_in += 1

        bot_in = 0
        while bot_in < edge_scan_y:
            if np.mean(gray[h - 1 - bot_in, col_s:col_e]) < 90:
                break
            bot_in += 1

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

        canvas_candidates = []
        for cnt in cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 内画布位置与尺寸约束：锚定左上角 (x in [3..18], y in [45..110])
            # 标题/任务栏通常高度很小；地图画布位于其下方且高度明显更大。
            if 3 <= x <= 18 and 60 <= y <= 120 and 100 <= bw <= (w - 6) and 80 <= bh <= (h - 10):
                aspect = bw / float(bh)
                if 0.4 <= aspect <= 3.0:
                    sub = roi_img[y+2:y+bh-2, x+2:x+bw-2]
                    if sub.size > 0:
                        mean_bgr = float(np.mean(sub))
                        dark_ratio = float(np.mean((sub[:, :, 0] < 85) & (sub[:, :, 1] < 85) & (sub[:, :, 2] < 85)))
                        # 确保内部是深色地图
                        if dark_ratio > 0.40 and mean_bgr < 110:
                            # 地图右侧可能包含平台、装饰和传送点，暗色比例反而
                            # 低于左侧子区域。若单纯按 dark_ratio 排序，会把
                            # 左半边的暗色子框误选成完整小地图。面积优先，
                            # 暗色比例仅作为次级评分。
                            score = (bw * bh, dark_ratio, mean_bgr)
                            canvas_candidates.append((score, (x, y, bw, bh)))

        if canvas_candidates:
            canvas_candidates.sort(key=lambda it: it[0], reverse=True)
            return self._refine_minimap_inner_box(roi_img, canvas_candidates[0][1])

        # 2. 备选方案：深色连通域闭运算
        dark_mask = (roi_img[:, :, 0] < 75) & (roi_img[:, :, 1] < 75) & (roi_img[:, :, 2] < 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(dark_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cnts_dark, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_candidates = []
        for cnt in cnts_dark:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if 3 <= x <= 18 and 60 <= y <= 120 and 100 <= bw <= (w - 6) and 80 <= bh <= (h - 10):
                dark_candidates.append((bw * bh, (x, y, bw, bh)))

        if dark_candidates:
            dark_candidates.sort(key=lambda it: it[0], reverse=True)
            return self._refine_minimap_inner_box(roi_img, dark_candidates[0][1])

        # 兜底默认
        return self._refine_minimap_inner_box(roi_img, (3, 67, 121, 111))

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

        # 2. 定位内部小地图边界
        if self.cached_inner_box is None:
            self.cached_inner_box = self.find_minimap_inner_box(roi_img)

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

        # 3. 原始左上角黄点算法：单一 HSV 阈值 -> 连通域 -> 最大面积候选。
        hsv = cv2.cvtColor(inner_map, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([22, 140, 185], np.uint8),
                           np.array([30, 255, 255], np.uint8))
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
            radius = 12
            search_x1 = max(0, int(last_x - radius - tw // 2))
            search_y1 = max(0, int(last_y - radius - th // 2))
            search_x2 = min(bw, int(last_x + radius + tw // 2 + 1))
            search_y2 = min(bh, int(last_y + radius + th // 2 + 1))
            search = inner_map[search_y1:search_y2, search_x1:search_x2]
            if search.shape[0] >= th and search.shape[1] >= tw:
                match = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED, mask=template_mask)
                _, score, _, loc = cv2.minMaxLoc(match)
                if score >= 0.96:
                    detected_xy = (
                        float(search_x1 + loc[0] + (tw - 1) / 2.0),
                        float(search_y1 + loc[1] + (th - 1) / 2.0),
                    )

        # 模板尚未建立或模板得分不足时，使用原有 HSV 连通域逻辑获取新模板。
        if detected_xy is None:
            all_candidates = []
            for i in range(1, num_labels):
                x0, y0, rw, rh, area = stats[i]
                if (rw, rh) not in {
                    (4, 4), (4, 5), (5, 4),
                    (5, 5), (6, 5), (5, 6), (6, 6),
                } or not (5 <= area <= 60):
                    continue
                component = labels[y0:y0 + rh, x0:x0 + rw] == i
                component_bgr = inner_map[y0:y0 + rh, x0:x0 + rw][component]
                if component_bgr.size == 0 or np.any(
                    np.ptp(component_bgr.astype(np.int16), axis=0) > np.array([48, 112, 96])
                ):
                    continue
                cx, cy = centroids[i]
                all_candidates.append({"cx": cx, "cy": cy, "area": int(area),
                                       "rw": int(rw), "rh": int(rh), "x": int(x0), "y": int(y0),
                                       "mask": component.astype(np.uint8) * 255})
            if all_candidates:
                all_candidates.sort(key=lambda item: item["area"], reverse=True)
                best_cand = all_candidates[0]
                detected_xy = (float(best_cand["cx"]), float(best_cand["cy"]))
                x0, y0 = best_cand["x"], best_cand["y"]
                self._yellow_dot_template = inner_map[y0:y0 + best_cand["rh"], x0:x0 + best_cand["rw"]].copy()
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
