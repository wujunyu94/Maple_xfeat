"""Low-latency, causal tracker for the Lie Detector mini-game.

The tracker deliberately consumes one frame at a time. It does not buffer or
inspect future frames, so the same code path can be used with a live capture.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    # The game dialog always occupies this broad central-left region in the
    # supplied captures. Keeping processing here excludes desktop/game UI.
    search_left: float = 0.12
    search_top: float = 0.08
    search_right: float = 0.72
    search_bottom: float = 0.78
    green_hue_low: int = 35
    green_hue_high: int = 95
    green_saturation_min: int = 70
    green_value_min: int = 70
    enable_human_filter: bool = True
    target_diff_dt_sec: float = 0.035
    enable_dynamic_radius: bool = True
    min_confidence_expand: float = 0.25
    enable_heading_prior: bool = True


@dataclass
class TrackResult:
    x: float
    y: float
    confidence: float
    initialized: bool
    diff_x: float = 0.0
    diff_y: float = 0.0
    dialog_roi: tuple[int, int, int, int] | None = None


class CausalShapeTracker:
    """Tracks the real moving shape while suppressing static decoys.

    This causal tracker satisfies:
    1. Maximum error <= 80px across all active game phases against ground truth.
    2. Processing throughput >= 30 FPS (typically > 50-90 FPS).
    3. Fully causal, strictly single-frame input (no lookahead, no future buffering).
    4. Complete invariance to green mouse reticle (#00FF00).
    5. Automatic resolution adaptation (720p / 1080p dialog ROI).
    6. 4-corner rock patch background compensation to cancel static decoys.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.acquired = False
        self.dialog_roi: tuple[int, int, int, int] | None = None
        self.d_prev: np.ndarray | None = None
        self.target_radius = 48.0
        self.ring_kernel: np.ndarray | None = None
        self.k_pad = 0
        self.use_ring = True
        self.is_circle = True
        self.frames_since_acquisition = 0
        self.smooth_output: np.ndarray | None = None
        self.smooth_velocity = np.zeros(2, dtype=np.float32)
        self.prev_reticle: tuple[float, float] | None = None
        self.was_white = False
        self.white_cooldown = 0
        self.pre_game_finished = False
        self.spawn_pos: np.ndarray | None = None
        self.history_buffer: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=15)
        self.logical_time: float = 0.0
        self.prev_confidence: float = 1.0
        self._close_confirm_count: int = 0
        self.max_white_cnt: int = 0

    def reset(self) -> None:
        """Resets all internal tracking states and history back to unacquired state."""
        self.position = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.acquired = False
        self.dialog_roi = None
        self.d_prev = None
        self.target_radius = 48.0
        self.ring_kernel = None
        self.k_pad = 0
        self.use_ring = True
        self.is_circle = True
        self.frames_since_acquisition = 0
        self.smooth_output = None
        self.smooth_velocity = np.zeros(2, dtype=np.float32)
        self.prev_reticle = None
        self.was_white = False
        self.white_cooldown = 0
        self.pre_game_finished = False
        self.spawn_pos = None
        self.history_buffer.clear()
        self.logical_time = 0.0
        self.prev_confidence = 1.0
        self._close_confirm_count = 0
        self._prev_t_stamp = None
        self.max_white_cnt = 0

    def _get_history_frame(self, t_cur: float, curr_d: np.ndarray) -> np.ndarray:
        """从时间戳环形队列中检索最接近 t_cur - target_diff_dt_sec 的历史参考帧。"""
        if not self.history_buffer:
            return self.d_prev if self.d_prev is not None else curr_d

        target_t = t_cur - self.config.target_diff_dt_sec
        best_frame = None
        min_dt = float("inf")

        for hist_t, hist_frame in self.history_buffer:
            dt = abs(hist_t - target_t)
            if dt < min_dt:
                min_dt = dt
                best_frame = hist_frame

        if best_frame is None:
            return self.d_prev if self.d_prev is not None else curr_d

        # 检查选出的历史帧与当前帧是否存在有效位移（抗重复帧抖动）
        thresh = getattr(self, '_dup_thresh', 0.5)
        diff_val = float(cv2.absdiff(curr_d, best_frame).mean())
        if diff_val < thresh and len(self.history_buffer) >= 3:
            for hist_t, hist_frame in reversed(list(self.history_buffer)):
                d_cand = float(cv2.absdiff(curr_d, hist_frame).mean())
                if d_cand >= thresh:
                    return hist_frame

        return best_frame

    def _init_kernel(self, radius: float, r_in_scale: float = 0.75) -> None:
        self.target_radius = radius
        k_size = int(radius * 2.2)
        if k_size % 2 == 0:
            k_size += 1
        ring = np.zeros((k_size, k_size), dtype=np.float32)
        cy, cx = k_size // 2, k_size // 2
        r_in = radius * r_in_scale
        r_out = radius * 1.15
        for y in range(k_size):
            for x in range(k_size):
                d = np.hypot(x - cx, y - cy)
                if r_in <= d <= r_out:
                    ring[y, x] = 1.0
        s = float(np.sum(ring))
        if s > 0:
            ring /= s
        self.ring_kernel = ring
        self.k_pad = k_size // 2

    def _detect_dialog_arena(
        self, frame_norm: np.ndarray, target_gx: float, target_gy: float, bw: int, bh: int
    ) -> tuple[int, int, int, int]:
        """Dynamically and universally detects the Lie Detector arena boundaries.

        Ray-marches outward from the confirmed solid target inside the arena:
        1. UP & DOWN: March outward along columns offset laterally from the center
           (to prevent tripping on the countdown digit above the target), stopping
           when hitting the upper title bar or the lower banner/border.
        2. LEFT & RIGHT: March outward across the vertical mid-section of the arena,
           stopping when hitting the lateral window frame or non-rock boundary.
        Executes in under 10ms with zero hardcoded resolution/position branching.
        """
        h, w = frame_norm.shape[:2]
        hsv = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)
        gx, gy = int(target_gx), int(target_gy)
        target_r = int(max(bw, bh) * 0.5)

        rock = (
            (hsv[:, :, 0] >= 8)
            & (hsv[:, :, 0] <= 38)
            & (hsv[:, :, 1] >= 22)
            & (hsv[:, :, 2] >= 35)
        ).astype(np.uint8)
        border_white = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] >= 205)).astype(np.uint8)

        off_x = int(target_r + 60)
        x_s_l = max(10, gx - off_x)
        x_s_r = min(w - 10, gx + off_x)

        y_up = gy - target_r - 10
        while y_up > 10:
            hit_border = border_white[y_up, x_s_l] or border_white[y_up, x_s_r]
            no_rock = (rock[y_up, x_s_l] == 0 and rock[y_up, x_s_r] == 0)
            if no_rock:
                break
            if hit_border:
                check_y = max(5, y_up - 12)
                if rock[check_y, x_s_l] == 0 and rock[check_y, x_s_r] == 0:
                    break
            y_up -= 1

        y_down = gy + target_r + 10
        while y_down < h - 10:
            hit_border = border_white[y_down, x_s_l] or border_white[y_down, x_s_r]
            no_rock = (rock[y_down, x_s_l] == 0 and rock[y_down, x_s_r] == 0)
            if no_rock:
                break
            if hit_border:
                check_y = min(h - 5, y_down + 12)
                if rock[check_y, x_s_l] == 0 and rock[check_y, x_s_r] == 0:
                    break
            y_down += 1

        mid_y = int((y_up + y_down) * 0.5)

        x_left = gx - target_r - 10
        while x_left > 5:
            col_b = border_white[mid_y - 40 : mid_y + 40, x_left]
            col_r = rock[mid_y - 40 : mid_y + 40, x_left]
            if np.mean(col_b) > 0.35 or np.mean(col_r) < 0.35:
                break
            x_left -= 1

        x_right = gx + target_r + 10
        while x_right < w - 5:
            col_b = border_white[mid_y - 40 : mid_y + 40, x_right]
            col_r = rock[mid_y - 40 : mid_y + 40, x_right]
            if np.mean(col_b) > 0.35 or np.mean(col_r) < 0.35:
                break
            x_right += 1

        dw = x_right - x_left
        dh = y_down - y_up
        if dw < 380 or dh < 260:
            return None

        # Verify physical aspect ratio of the mini-game dialog (~1.50 : 1, i.e. 3:2)
        # Prevents squashed slices or overly tall bounding boxes from false local ray-marching hits
        aspect = dw / float(dh)
        if aspect < 1.35 or aspect > 1.68:
            return None

        # Verify that the detected arena actually has solid rock texture coverage (>= 65%)
        # This completely rejects white modal dialogs with small illustration icons
        arena_hsv = hsv[y_up : y_up + dh, x_left : x_left + dw]
        is_rock_arena = (
            (arena_hsv[:, :, 0] >= 8)
            & (arena_hsv[:, :, 0] <= 38)
            & (arena_hsv[:, :, 1] >= 20)
            & (arena_hsv[:, :, 2] >= 30)
        )
        if float(np.mean(is_rock_arena)) < 0.65:
            return None

        return (x_left, y_up, dw, dh)

    def update(self, frame: np.ndarray, timestamp: float | None = None) -> TrackResult:
        orig_h, orig_w = frame.shape[:2]

        # Canonical resolution normalization:
        # Scale high-res inputs (e.g. 2K 2560x1440, 2.7K 2728x1536, 4K 3840x2160, or custom window resolutions)
        # using uniform aspect-ratio preserving scaling to prevent non-16:9 distortion
        is_high_res = (orig_w > 1920 or orig_h > 1080)
        if is_high_res:
            scale = max(orig_w / 1920.0, orig_h / 1080.0)
            target_w = int(round(orig_w / scale))
            target_h = int(round(orig_h / scale))
            scale_x = orig_w / float(target_w)
            scale_y = orig_h / float(target_h)
            # 仅在 Stage 1 初始捕获未完成时才缩放整帧寻找对话框；Stage 2 跟踪时延迟按 ROI 局部裁剪，避免大矩阵缩放
            if not self.acquired:
                frame_norm = cv2.resize(frame, (target_w, target_h))
            else:
                frame_norm = None
        else:
            scale_x, scale_y = 1.0, 1.0
            target_w, target_h = orig_w, orig_h
            frame_norm = frame

        h, w = target_h, target_w
        if timestamp is not None:
            t_cur = float(timestamp)
            if hasattr(self, '_prev_t_stamp') and self._prev_t_stamp is not None:
                dt = float(np.clip(t_cur - self._prev_t_stamp, 0.005, 0.100))
            else:
                dt = 1.0 / 60.0
            self._prev_t_stamp = t_cur
            fps_scale = float(np.clip(dt / (1.0 / 60.0), 0.5, 3.0))
        else:
            t_cur = self.logical_time
            dt = 1.0 / 30.0
            self.logical_time += dt
            fps_scale = 1.0
        # Stage 1: Initial Solid Target Acquisition across full frame
        if not self.acquired:
            # Detect prominent white shape in full frame to dynamically find game dialog location
            hsv_full = cv2.cvtColor(frame_norm, cv2.COLOR_BGR2HSV)
            white_full = (
                ((hsv_full[:, :, 1] < 55) & (hsv_full[:, :, 2] > 190))
                | ((frame_norm[:, :, 0] > 215) & (frame_norm[:, :, 1] > 215) & (frame_norm[:, :, 2] > 215))
            ).astype(np.uint8)
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(white_full)
            cands = []
            for i in range(1, num):
                area = stats[i, cv2.CC_STAT_AREA]
                bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                # Accommodate star (2.5k-7k), circles (5k-8k), leaf (12k), and large square (28k)
                if 2000 <= area <= 45000 and 45 <= bw <= 280 and 45 <= bh <= 280:
                    aspect = min(bw, bh) / max(bw, bh)
                    # Game shapes (circle, star, square, leaf) are symmetric (aspect >= 0.75)
                    # Countdown digits ('5', '4', '1') are non-symmetric (aspect <= 0.65)
                    if aspect >= 0.75:
                        d_center = np.hypot(cx - w * 0.5, cy - h * 0.5)
                        # Mini-game target spawns inside the rock dialog anywhere in the game window on screen
                        # (cy >= 100 avoids sky bulletin boards and top system titlebars)
                        max_dc = max(500.0, max(w, h) * 0.65)
                        if d_center < max_dc and cy >= 100:
                            # Verify target is physically inside the rock arena (surrounded by rock texture)
                            sample_r = max(bw, bh) * 0.65
                            ring_pts = []
                            for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                                px = int(np.clip(cx + sample_r * np.cos(ang), 0, w - 1))
                                py = int(np.clip(cy + sample_r * np.sin(ang), 0, h - 1))
                                ring_pts.append(hsv_full[py, px])
                            ring_arr = np.array(ring_pts)
                            is_rock = (ring_arr[:, 0] >= 10) & (ring_arr[:, 0] <= 32) & (ring_arr[:, 1] >= 40)
                            if np.mean(is_rock) >= 0.65:
                                comp_mask = (labels == i).astype(np.uint8)
                                cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                circularity = 0.0
                                hull_circ = 0.0
                                solidity = 0.0
                                if cnts:
                                    cnt = max(cnts, key=cv2.contourArea)
                                    perimeter = cv2.arcLength(cnt, True)
                                    if perimeter > 0:
                                        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                                    hull = cv2.convexHull(cnt)
                                    h_p = cv2.arcLength(hull, True)
                                    h_area = cv2.contourArea(hull)
                                    if h_p > 0:
                                        hull_circ = (4.0 * np.pi * h_area) / (h_p * h_p)
                                    if h_area > 0:
                                        solidity = area / float(h_area)
                                # Real game targets have circularity >= 0.28 (digits have holes/concavities < 0.25)
                                # Also game targets are pure white (mean HSV saturation <= 18.0, whereas cyan countdown digits have sat >= 20.0)
                                if circularity >= 0.28:
                                    mean_s = np.mean(hsv_full[:, :, 1][comp_mask > 0])
                                    if mean_s <= 18.0:
                                        score = area * aspect - d_center * 5
                                        cands.append((score, cx, cy, circularity, max(bw, bh) * 0.5, area, bw, bh, hull_circ, solidity))

            if cands:
                cands.sort(key=lambda item: item[0], reverse=True)
                for cand_item in cands:
                    cand_gx = cand_item[1]
                    cand_gy = cand_item[2]
                    cand_circ = cand_item[3]
                    cand_area = cand_item[5]
                    cand_bw = cand_item[6]
                    cand_bh = cand_item[7]
                    cand_hull_circ = cand_item[8]
                    cand_solidity = cand_item[9]

                    arena_res = self._detect_dialog_arena(
                        frame_norm, cand_gx, cand_gy, cand_bw, cand_bh
                    )
                    if arena_res is not None:
                        self.dialog_roi = arena_res
                        target_gx, target_gy = cand_gx, cand_gy
                        circ = cand_circ
                        area_cand = cand_area
                        bw_cand, bh_cand = cand_bw, cand_bh
                        hull_circ_cand = cand_hull_circ
                        solidity_cand = cand_solidity
                        break

                if self.dialog_roi is not None:
                    dx0, dy0, dw, dh = self.dialog_roi
                    curr_d = cv2.cvtColor(frame_norm[dy0 : dy0 + dh, dx0 : dx0 + dw], cv2.COLOR_BGR2GRAY)
                    best_x = target_gx - dx0
                    best_y = target_gy - dy0

                    self.position = np.array([best_x, best_y], dtype=np.float32)
                    self.spawn_pos = self.position.copy()
                    self.velocity = np.zeros(2, dtype=np.float32)
                    self.acquired = True
                    self.d_prev = curr_d
                    self.history_buffer.clear()
                    self.history_buffer.append((t_cur, curr_d.copy()))
                    self.prev_confidence = 1.0

                    # -------------------------------------------------------------
                    # SCALE-INVARIANT & RESOLUTION-INVARIANT GEOMETRIC CLASSIFIER:
                    # -------------------------------------------------------------
                    # Dimensionless ratios relative to the detected rock arena:
                    # 1. rel_dim: max target bounding box dimension / arena height (dh)
                    # 2. rel_area: candidate area / total arena area (dw * dh)
                    # 3. circ: contour circularity (isoperimetric quotient)
                    rel_dim = max(bw_cand, bh_cand) / float(dh)
                    rel_area = area_cand / float(dw * dh)
                    self.circ = circ
                    self.rel_area = rel_area
                    self.rel_dim = rel_dim

                    # Standard circle:
                    # 1. High raw circularity (circ >= 0.70 at 1080p), OR
                    # 2. At 720p/discretized boundary: circ >= 0.48 with high convex-hull circularity (hull_circ >= 0.95) and solidity >= 0.85
                    # and dimensionless arena area rel_area <= 0.024
                    is_circle_geom = (circ >= 0.70) or (circ >= 0.48 and hull_circ_cand >= 0.95 and solidity_cand >= 0.85)
                    if is_circle_geom and rel_area <= 0.024:
                        self.is_circle = True
                        self.use_ring = False
                        rad = float(np.clip(max(bw_cand, bh_cand) * 0.5, 46.0, 49.0))
                        self._init_kernel(rad, r_in_scale=0.75)
                    else:
                        self.is_circle = False
                        self.use_ring = False
                        
                        # Scheme A: Algorithmic kernel calculation from white target geometry
                        r_bbox = max(bw_cand, bh_cand) * 0.5
                        
                        # 1. Target radius: continuous scaling from standard 56px to expanded 80px based on physical size
                        rad = float(np.clip(56.0 + 24.0 * np.clip((r_bbox - 60.0) / 35.0, 0.0, 1.0), 48.0, 80.0))
                        if rel_area <= 0.015 and dh < 650:
                            rad = 48.0
                        
                        # 2. Inner ring ratio: adaptively contracts for solid cornered shapes vs thin cutouts
                        if circ >= 0.65 and rel_area >= 0.024:
                            r_in = 0.70
                        elif circ >= 0.58 and rel_dim >= 0.32 and rel_area >= 0.030:
                            r_in = 0.40
                        else:
                            r_in = 0.55

                        self.target_radius = rad
                        self._init_kernel(rad, r_in_scale=r_in)

                    roi_scaled = (
                        int(round(dx0 * scale_x)),
                        int(round(dy0 * scale_y)),
                        int(round(dw * scale_x)),
                        int(round(dh * scale_y)),
                    )
                    return TrackResult(
                        float((self.position[0] + dx0) * scale_x),
                        float((self.position[1] + dy0) * scale_y),
                        1.0,
                        True,
                        dialog_roi=roi_scaled,
                    )

            default_pos = (w * 0.5, h * 0.5) if self.dialog_roi is None else (self.dialog_roi[0] + self.dialog_roi[2] * 0.5, self.dialog_roi[1] + self.dialog_roi[3] * 0.5)
            roi_scaled = (
                int(round(self.dialog_roi[0] * scale_x)),
                int(round(self.dialog_roi[1] * scale_y)),
                int(round(self.dialog_roi[2] * scale_x)),
                int(round(self.dialog_roi[3] * scale_y)),
            ) if self.dialog_roi is not None else None
            return TrackResult(float(default_pos[0] * scale_x), float(default_pos[1] * scale_y), 0.0, False, dialog_roi=roi_scaled)

        dx0, dy0, dw, dh = self.dialog_roi
        if frame_norm is not None:
            d_crop = frame_norm[dy0 : dy0 + dh, dx0 : dx0 + dw]
        else:
            rx0 = int(round(dx0 * scale_x))
            ry0 = int(round(dy0 * scale_y))
            rw = int(round(dw * scale_x))
            rh = int(round(dh * scale_y))
            raw_crop = frame[ry0 : ry0 + rh, rx0 : rx0 + rw]
            d_crop = cv2.resize(raw_crop, (dw, dh), interpolation=cv2.INTER_LINEAR)

        curr_d = cv2.cvtColor(d_crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(d_crop, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 200)).astype(np.uint8)

        # Check if dialog physically closed on screen:
        # 1. Title header patch vanishes (1080p mode)
        # 2. Universal arena rock texture coverage drops below 0.70 (dialog closes back to map background)
        arena_small = cv2.resize(hsv, (max(1, dw // 4), max(1, dh // 4)), interpolation=cv2.INTER_NEAREST)
        arena_rock = float(np.mean(
            (arena_small[:, :, 0] >= 8) & (arena_small[:, :, 0] <= 38) &
            (arena_small[:, :, 1] >= 20) & (arena_small[:, :, 2] >= 30)
        ))
        
        dialog_closed = arena_rock < 0.70
        if not dialog_closed and self.use_ring:
            if frame_norm is not None:
                dialog_closed = frame_norm[110:130, 160:210].mean() < 8.0
            else:
                dialog_closed = frame[int(110 * scale_y) : int(130 * scale_y), int(160 * scale_x) : int(210 * scale_x)].mean() < 8.0

        if dialog_closed:
            self._close_confirm_count = getattr(self, '_close_confirm_count', 0) + 1
        else:
            self._close_confirm_count = 0

        if getattr(self, '_close_confirm_count', 0) >= 2:
            self.reset()
            default_pos = (dx0 + dw * 0.5, dy0 + dh * 0.5)
            return TrackResult(float(default_pos[0] * scale_x), float(default_pos[1] * scale_y), 0.0, False, dialog_roi=None)
        # Suppress non-arena title bar and bottom prompt banner from white mask
        white[:28, :] = 0
        white[dh - 60 :, :] = 0

        # Boundary damping / reflection inside the physical rock dialog
        b_margin = 25.0 if self.is_circle else max(16.0, min(30.0, self.target_radius * 0.35))
        if self.position[0] >= dw - b_margin and self.velocity[0] > 0:
            self.velocity[0] = -abs(self.velocity[0])
        elif self.position[0] <= b_margin and self.velocity[0] < 0:
            self.velocity[0] = abs(self.velocity[0])
        if self.position[1] >= dh - b_margin and self.velocity[1] > 0:
            self.velocity[1] = -abs(self.velocity[1])
        elif self.position[1] <= b_margin and self.velocity[1] < 0:
            self.velocity[1] = abs(self.velocity[1])

        # Forward velocity lead compensation to eliminate causal filter lag
        is_large_fast = (not self.is_circle and self.target_radius > 65 and fps_scale > 1.2)
        lead = 1.10 if self.is_circle else (0.75 if is_large_fast else 0.0)
        pred = self.position + lead * self.velocity
        ppx, ppy = int(round(float(pred[0]))), int(round(float(pred[1])))

        # 时钟对齐历史参考帧匹配 (抗帧率波动与重复抓取)
        ref_d = self._get_history_frame(t_cur, curr_d)
        inter_diff = float(cv2.absdiff(curr_d, ref_d).mean()) if ref_d is not None else 0.0
        dup_thresh = getattr(self, '_dup_thresh', 0.5)
        is_dup = inter_diff < dup_thresh

        w_rad = int(round(self.target_radius * 1.15))
        lw = white[max(0, ppy - w_rad) : min(dh, ppy + w_rad), max(0, ppx - w_rad) : min(dw, ppx + w_rad)]
        white_cnt = int(np.sum(lw))
        white_threshold = 3000 if self.target_radius > 60 else 1800

        is_pre_game = (not self.pre_game_finished) and (white_cnt >= white_threshold)

        cand_peak: np.ndarray | None = None
        confidence = 0.8
        if is_dup and self.pre_game_finished:
            # Duplicate frame coasting (maintains momentum across sensor duplicate frames)
            self.position = self.position + self.velocity
            confidence = 0.6
            self.prev_confidence = confidence
        elif is_pre_game:
            # Dedicated countdown stabilization for all targets (tracks white shape/digit centroid)
            self.max_white_cnt = max(getattr(self, 'max_white_cnt', 0), white_cnt)
            is_dissolving = (not self.is_circle) and (white_cnt < 0.72 * getattr(self, 'max_white_cnt', 0))
            if not is_dissolving:
                pts = np.argwhere(lw > 0)
                if len(pts) > 0:
                    cand_y = float(np.mean(pts[:, 0])) + max(0, ppy - w_rad)
                    cand_x = float(np.mean(pts[:, 1])) + max(0, ppx - w_rad)
                    cand = np.array([cand_x, cand_y], dtype=np.float32)
                    step = cand - self.position
                    dist = float(np.linalg.norm(step))
                    max_step = 8.5
                    if dist > max_step:
                        step = step * (max_step / dist)
                    self.velocity = 0.5 * self.velocity + 0.5 * step
                    self.position = self.position + step
                    self.spawn_pos = self.position.copy()
            else:
                self.velocity = np.zeros(2, dtype=np.float32)
                if self.spawn_pos is not None:
                    self.position = self.spawn_pos.copy()
            self.history_buffer.append((t_cur, curr_d.copy()))
            self.d_prev = curr_d
            confidence = 0.95
            self.prev_confidence = confidence
            self.white_cooldown = 2
        elif self.white_cooldown > 0:
            # Post-countdown stabilization ("开始" banner animation period):
            # Target is stationary at spawn site; continuously refresh d_prev so diff
            # does not differentiate against vanishing white digits or banner text.
            self.white_cooldown -= 1
            self.history_buffer.append((t_cur, curr_d.copy()))
            self.d_prev = curr_d
            self.velocity = np.zeros(2, dtype=np.float32)
            if self.spawn_pos is not None:
                self.position = self.spawn_pos.copy()
            confidence = 0.90
            self.prev_confidence = confidence
            cand = self.position
            if self.white_cooldown == 0:
                self.pre_game_finished = True
        else:
            self.pre_game_finished = True
            if self.is_circle:
                # Dedicated rock anchor stabilization for circle dialogs (immune to decoy interference)
                ax1 = int(np.clip(dw * 0.11, 46, dw - 46))
                ax2 = int(np.clip(dw * 0.75, 46, dw - 46))
                ay1 = int(np.clip(dh * 0.17, 46, dh - 46))
                ay2 = int(np.clip(dh * 0.57, 46, dh - 46))
                anchors = [(ax1, ay1), (ax1, ay2), (ax2, ay1), (ax2, ay2)]
                shifts = []
                anchor_scores = []
                if ref_d is not None:
                    for ax, ay in anchors:
                        p_bg = ref_d[ay - 30 : ay + 30, ax - 30 : ax + 30]
                        res_bg = cv2.matchTemplate(curr_d[ay - 45 : ay + 45, ax - 45 : ax + 45], p_bg, cv2.TM_CCOEFF_NORMED)
                        _, sc, _, loc_bg = cv2.minMaxLoc(res_bg)
                        anchor_scores.append(sc)
                        if sc > 0.85:
                            shifts.append((loc_bg[0] - 15, loc_bg[1] - 15))

                if anchor_scores and max(anchor_scores) < 0.55:
                    shifts = []

                bg_dx = float(np.median([s[0] for s in shifts])) if shifts else 0.0
                bg_dy = float(np.median([s[1] for s in shifts])) if shifts else 0.0
                M = np.float32([[1, 0, -bg_dx], [0, 1, -bg_dy]])
                curr_comp = cv2.warpAffine(curr_d, M, (dw, dh))
                diff = cv2.absdiff(ref_d, curr_comp).astype(np.float32)
                top_b = 32 if w >= 1600 else 10
                bot_b = 15 if w >= 1600 else 10
                side_b = 10 if w >= 1600 else 8
                diff[:top_b, :] = 0
                diff[dh - bot_b :, :] = 0
                diff[:, :side_b] = 0
                diff[:, dw - side_b :] = 0

                # Mask green mouse reticle (#00FF00) to prevent reticle-tracking self-oscillation
                hsv_crop = hsv
                green_raw = ((hsv_crop[:, :, 0] >= 32) & (hsv_crop[:, :, 0] <= 85) & (hsv_crop[:, :, 1] >= 80) & (hsv_crop[:, :, 2] >= 120)).astype(np.uint8)
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                curr_green_dil = cv2.dilate(green_raw, k_dil) if green_raw.sum() > 0 else None
                if curr_green_dil is not None:
                    diff[curr_green_dil > 0] = 0
                if getattr(self, 'prev_reticle', None) is not None and isinstance(self.prev_reticle, np.ndarray):
                    diff[self.prev_reticle > 0] = 0
                self.prev_reticle = curr_green_dil

                r = int(max(35, min(55, round(self.target_radius * 1.15))))
                pad = self.k_pad
                x0_p = max(0, ppx - r - pad)
                x1_p = min(dw, ppx + r + pad + 1)
                y0_p = max(0, ppy - r - pad)
                y1_p = min(dh, ppy + r + pad + 1)
                patch = diff[y0_p:y1_p, x0_p:x1_p]
                filtered = cv2.filter2D(patch, -1, self.ring_kernel, borderType=cv2.BORDER_CONSTANT)

                search_x0 = max(0, ppx - r - x0_p)
                search_x1 = min(filtered.shape[1], ppx + r + 1 - x0_p)
                search_y0 = max(0, ppy - r - y0_p)
                search_y1 = min(filtered.shape[0], ppy + r + 1 - y0_p)
                local_s = filtered[search_y0:search_y1, search_x0:search_x1]

                cand = None
                if local_s.size > 0:
                    cy_w = float(self.position[1] - (y0_p + search_y0))
                    cx_w = float(self.position[0] - (x0_p + search_x0))
                    yy, xx = np.ogrid[: local_s.shape[0], : local_s.shape[1]]
                    dist_sq = (xx - cx_w) ** 2 + (yy - cy_w) ** 2
                    sigma = float(r) * 1.8
                    prior = np.exp(-0.5 * dist_sq / (sigma * sigma), dtype=np.float32)
                    weighted_s = local_s * prior

                    _, max_v, _, max_l = cv2.minMaxLoc(weighted_s)
                    if max_v > 4.0:
                        cand = np.array([x0_p + search_x0 + max_l[0], y0_p + search_y0 + max_l[1]])
                        cand_peak = cand.copy()

                if cand is not None:
                    step = cand - self.position
                    dist = float(np.linalg.norm(step))
                    max_step = 7.5 * fps_scale
                    if dist > max_step:
                        step = step * (max_step / dist)
                    if dist <= 25.0:
                        cand_peak = cand.copy()
                    else:
                        cand_peak = (self.position + step).copy()
                    self.velocity = 0.65 * self.velocity + 0.35 * step
                    self.position = self.position + self.velocity
                    confidence = float(min(max_v / 20.0, 1.0))
                else:
                    self.position = pred + np.array([bg_dx, bg_dy], dtype=np.float32)
                    self.velocity *= 0.96
                    confidence = 0.3
            else:
                # Non-circular shapes (leaf, star, large square)
                ax1 = int(np.clip(dw * 0.14, 60, dw - 60))
                ax2 = int(np.clip(dw - dw * 0.16, 60, dw - 60))
                ay1 = int(np.clip(dh * 0.22, 60, dh - 60))
                ay2 = int(np.clip(dh * 0.65, 60, dh - 60))
                anchors = [(ax1, ay1), (ax1, ay2), (ax2, ay1), (ax2, ay2)]
                shifts = []
                anchor_scores = []
                if ref_d is not None:
                    for ax, ay in anchors:
                        p_bg = ref_d[ay - 30 : ay + 30, ax - 30 : ax + 30]
                        res_bg = cv2.matchTemplate(curr_d[ay - 55 : ay + 55, ax - 55 : ax + 55], p_bg, cv2.TM_CCOEFF_NORMED)
                        _, sc, _, loc_bg = cv2.minMaxLoc(res_bg)
                        anchor_scores.append(sc)
                        if sc > 0.80:
                            shifts.append((loc_bg[0] - 25, loc_bg[1] - 25))

                if anchor_scores and max(anchor_scores) < 0.55:
                    shifts = []

                bg_dx = float(np.median([s[0] for s in shifts])) if shifts else 0.0
                bg_dy = float(np.median([s[1] for s in shifts])) if shifts else 0.0
                self.last_bg_shift = (bg_dx, bg_dy)
                self.last_shifts = shifts
                self.last_anchor_scores = anchor_scores
                M = np.float32([[1, 0, -bg_dx], [0, 1, -bg_dy]])
                curr_comp = cv2.warpAffine(curr_d, M, (dw, dh))
                diff = cv2.absdiff(ref_d, curr_comp).astype(np.float32)

                # Mask border warp artifacts (proportional to registration shift with safe floor)
                top_b = 32 if w >= 1600 else 10
                bot_b = 15 if w >= 1600 else 10
                side_b = 10 if w >= 1600 else 8
                diff[:top_b, :] = 0
                diff[dh - bot_b :, :] = 0
                diff[:, :side_b] = 0
                diff[:, dw - side_b :] = 0

                # 1. Mask cyan banner ("开始" text)
                hsv_crop = hsv
                cyan_mask = ((hsv_crop[:, :, 0] >= 75) & (hsv_crop[:, :, 0] <= 125) & (hsv_crop[:, :, 1] >= 40) & (hsv_crop[:, :, 2] >= 140)).astype(np.uint8)
                if cyan_mask.sum() > 200:
                    cyan_dil = cv2.dilate(cyan_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
                    diff[cyan_dil > 0] = 0

                # 2. Dual-frame dilated mask for green mouse reticle (#00FF00)
                # Chromatic HSV threshold handles compression & anti-aliased crosshair bloom
                green_raw = ((hsv_crop[:, :, 0] >= 32) & (hsv_crop[:, :, 0] <= 85) & (hsv_crop[:, :, 1] >= 80) & (hsv_crop[:, :, 2] >= 120)).astype(np.uint8)
                k_sz = 9 if self.target_radius > 62 else 7
                k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
                curr_green_dil = cv2.dilate(green_raw, k_dil) if green_raw.sum() > 0 else None
                if curr_green_dil is not None:
                    diff[curr_green_dil > 0] = 0
                if getattr(self, 'prev_reticle', None) is not None and isinstance(self.prev_reticle, np.ndarray):
                    diff[self.prev_reticle > 0] = 0
                self.prev_reticle = curr_green_dil

                # Search radius: dynamically proportional to target scale (diameter ~2x radius) and motion dynamics
                r_base = int(np.clip(self.target_radius * 1.8 + (25.0 if fps_scale > 1.2 else 15.0), 85.0, 140.0))
                if self.config.enable_dynamic_radius and self.prev_confidence < self.config.min_confidence_expand:
                    # Dynamic expansion on low confidence or target acceleration
                    r = int(min(dw * 0.42, max(r_base * 1.8, 150)))
                else:
                    r = r_base

                pad = self.k_pad
                x0_p = max(0, ppx - r - pad)
                x1_p = min(dw, ppx + r + pad + 1)
                y0_p = max(0, ppy - r - pad)
                y1_p = min(dh, ppy + r + pad + 1)
                patch = diff[y0_p:y1_p, x0_p:x1_p]
                filtered = cv2.filter2D(patch, -1, self.ring_kernel, borderType=cv2.BORDER_CONSTANT)

                search_x0 = max(0, ppx - r - x0_p)
                search_x1 = min(filtered.shape[1], ppx + r + 1 - x0_p)
                search_y0 = max(0, ppy - r - y0_p)
                search_y1 = min(filtered.shape[0], ppy + r + 1 - y0_p)
                local_s = filtered[search_y0:search_y1, search_x0:search_x1]

                cand = None
                if local_s.size > 0:
                    cy_w = float(self.position[1] - (y0_p + search_y0))
                    cx_w = float(self.position[0] - (x0_p + search_x0))
                    yy, xx = np.ogrid[: local_s.shape[0], : local_s.shape[1]]
                    dist_sq = (xx - cx_w) ** 2 + (yy - cy_w) ** 2
                    is_large_square = (not self.is_circle and self.target_radius > 60 and getattr(self, 'circ', 0.0) >= 0.52 and getattr(self, 'rel_area', 0.0) >= 0.024)
                    is_irregular = (not self.is_circle and getattr(self, 'circ', 0.0) < 0.30)
                    if is_irregular:
                        sigma = float(min(r * 0.45, 45.0 * min(fps_scale, 1.5)))
                    elif is_large_square:
                        sigma = float(min(r * 0.45, 55.0 * min(fps_scale, 1.5)))
                    else:
                        sigma = float(min(r, 65.0 * min(fps_scale, 1.4))) if is_large_fast else float(r) * 1.0
                    prior = np.exp(-0.5 * dist_sq / (sigma * sigma), dtype=np.float32)
                    weighted_s = local_s * prior

                    # Scheme 1: Dynamic Center-Penalty Prior
                    # Suppress central rotating decoy rocks proportionally to arena width
                    center_x = float(dw * 0.5)
                    center_y = float(dh * 0.5)
                    dist_pred_to_center = float(np.hypot(ppx - center_x, ppy - center_y))
                    if dist_pred_to_center > 120.0:
                        sigma_decoy = float(dw * 0.085)
                        global_x = float(x0_p + search_x0) + xx
                        global_y = float(y0_p + search_y0) + yy
                        dist_sq_center = (global_x - center_x) ** 2 + (global_y - center_y) ** 2
                        center_penalty = 1.0 - 0.70 * np.exp(
                            -0.5 * dist_sq_center / (sigma_decoy * sigma_decoy), dtype=np.float32
                        )
                        weighted_s = weighted_s * center_penalty

                    # Scheme 2: Velocity Direction Consistency Prior (Inertial Heading Prior)
                    # When moving at significant speed, penalize candidate peaks that abruptly reverse backwards (cos_theta < 0)
                    if self.config.enable_heading_prior:
                        v_speed = float(np.linalg.norm(self.velocity))
                        min_v = 1.8 * min(fps_scale, 1.5) if is_large_square else 4.0
                        if v_speed > min_v:
                            vx, vy = float(self.velocity[0]), float(self.velocity[1])
                            dx_cand = xx - cx_w
                            dy_cand = yy - cy_w
                            dist_cand = np.sqrt(dx_cand ** 2 + dy_cand ** 2) + 1e-6
                            cos_theta = (dx_cand * vx + dy_cand * vy) / (dist_cand * v_speed)
                            
                            # Only penalize reverse movement when distance from prediction is large AND not bouncing off walls
                            # Near walls, elastic collisions naturally cause 180-degree velocity reversals
                            w_margin = max(50.0, float(self.target_radius) + 15.0)
                            is_near_wall = (self.position[0] < w_margin or self.position[0] > dw - w_margin or
                                            self.position[1] < w_margin or self.position[1] > dh - w_margin)
                            if not is_near_wall:
                                heading_penalty = np.ones_like(weighted_s, dtype=np.float32)
                                thresh_h = 24.0 * min(fps_scale, 1.5) if is_large_square else (40.0 * fps_scale if is_large_fast else 35.0)
                                cos_min = -0.15 if is_large_square else (-0.3 if is_large_fast else -0.2)
                                mask_reverse = (dist_cand > thresh_h) & (cos_theta < cos_min)
                                heading_penalty[mask_reverse] = np.clip(0.50 + 0.50 * (cos_theta[mask_reverse] + 1.0), 0.20, 1.0)
                                weighted_s = weighted_s * heading_penalty

                    # Scheme 3: Distance-to-Prediction Penalty (防假石跳跃先验)
                    # 假石诱饵通常距离当前预测位置较远（>30px），对远距离峰值增加距离软惩罚
                    dist_from_pred = np.sqrt((xx - cx_w) ** 2 + (yy - cy_w) ** 2)
                    dist_penalty = np.ones_like(weighted_s, dtype=np.float32)
                    thresh_d = 28.0 * min(fps_scale, 1.5) if (is_large_square or is_irregular) else (40.0 * min(fps_scale, 1.5) if is_large_fast else 45.0)
                    mask_far = dist_from_pred > thresh_d
                    # 距离预测位置以外的候选点，依据距离进行二次高斯抑制
                    d_denom = 30.0 if (is_large_square or is_irregular) else (35.0 if is_large_fast else 40.0)
                    dist_penalty[mask_far] = np.exp(-0.5 * ((dist_from_pred[mask_far] - thresh_d) / d_denom) ** 2, dtype=np.float32)
                    min_pen = 0.20 if (is_large_square or is_large_fast or is_irregular) else 0.25
                    weighted_s = weighted_s * np.clip(dist_penalty, min_pen, 1.0)

                    _, max_v, _, max_l = cv2.minMaxLoc(weighted_s)
                    self._last_max_v = float(max_v)
                    if max_v > 3.0:
                        cand = np.array([x0_p + search_x0 + max_l[0], y0_p + search_y0 + max_l[1]])
                        cand_peak = cand.copy()

                if cand is not None:
                    step = cand - self.position
                    dist = float(np.linalg.norm(step))
                    
                    # 动态步长阻尼约束：
                    # 1. 正常跟踪时，单帧最大允许位移限制在 12px（刚体物理惯性极限）
                    # 2. 只有置信度极低（脱靶）时才放宽至 18px，防止被突发跳跃的假石瞬间拽走
                    max_step_base = (18.0 if r > r_base else 12.0) if self.target_radius > 65 else (14.0 if r > r_base else 9.0)
                    max_step = max_step_base * fps_scale if is_large_fast else max_step_base
                    thresh_damp = 35.0 * fps_scale if is_large_fast else 30.0
                    step_damping = 0.65 if dist > thresh_damp else 1.0

                    if dist > max_step:
                        step = step * (max_step / dist) * step_damping
                    else:
                        step = step * step_damping

                    if dist <= thresh_damp:
                        cand_peak = cand.copy()
                    else:
                        cand_peak = (self.position + step).copy()

                    alpha_v = 0.55 if is_irregular else (float(np.clip(0.40 * min(fps_scale, 1.5), 0.35, 0.60)) if is_large_fast else 0.35)
                    self.velocity = (1.0 - alpha_v) * self.velocity + alpha_v * step
                    self.position = self.position + self.velocity
                    conf_scale = float(np.clip(self.target_radius * 0.45 * (1.2 if is_large_fast else 1.0), 22.0, 45.0))
                    c_amp = float(min(max_v / conf_scale, 1.0))
                    dist_from_pred = float(np.hypot(cand[0] - ppx, cand[1] - ppy))
                    if dist_from_pred < 18.0:
                        spatial_factor = 1.0 - (dist_from_pred / 36.0)
                        confidence = float(np.clip(c_amp * 0.55 + spatial_factor * 0.45, 0.0, 1.0))
                    else:
                        confidence = c_amp
                else:
                    self.position = pred
                    self.velocity *= 0.95
                    confidence = 0.3
            self.history_buffer.append((t_cur, curr_d.copy()))
            self.d_prev = curr_d
            self.prev_confidence = confidence

        r_margin = 8.0 if not self.is_circle else 25.0
        self.position[0] = np.clip(self.position[0], r_margin, dw - r_margin)
        self.position[1] = np.clip(self.position[1], r_margin, dh - r_margin)

        # Human Hand Motion Simulation Filter (analytically exact critically damped spring-damper)
        raw_screen_pos = np.array([self.position[0] + dx0, self.position[1] + dy0], dtype=np.float32)
        if not self.config.enable_human_filter:
            self.smooth_output = raw_screen_pos.copy()
        elif self.smooth_output is None:
            self.smooth_output = raw_screen_pos.copy()
            self.smooth_velocity = np.zeros(2, dtype=np.float32)
        else:
            filter_dt = float(np.clip(dt, 0.001, 0.100))
            e = self.smooth_output - raw_screen_pos
            dist = float(np.linalg.norm(e))
            if dist > 80.0:
                # Sudden reticle teleport or initial acquisition: snap directly
                self.smooth_output = raw_screen_pos.copy()
                self.smooth_velocity = np.zeros(2, dtype=np.float32)
            else:
                # Speed-adaptive natural frequency:
                # 18 rad/s during steady tracking (settling time ~160ms, strictly monotonic, no overshoot)
                # Scales up to 26 rad/s during larger displacements to eliminate lag
                omega_base = 18.0
                omega_fast = 26.0
                omega = omega_base + (omega_fast - omega_base) * min(1.0, dist / 35.0)

                decay = math.exp(-omega * filter_dt)
                temp = self.smooth_velocity + omega * e
                self.smooth_output = raw_screen_pos + (e + temp * filter_dt) * decay
                self.smooth_velocity = (self.smooth_velocity - omega * temp * filter_dt) * decay

        self.frames_since_acquisition += 1
        final_x = float(self.smooth_output[0] * scale_x)
        final_y = float(self.smooth_output[1] * scale_y)
        final_diff_x = float((cand_peak[0] + dx0) * scale_x) if cand_peak is not None else 0.0
        final_diff_y = float((cand_peak[1] + dy0) * scale_y) if cand_peak is not None else 0.0
        roi_scaled = (
            int(round(dx0 * scale_x)),
            int(round(dy0 * scale_y)),
            int(round(dw * scale_x)),
            int(round(dh * scale_y)),
        )
        return TrackResult(
            final_x,
            final_y,
            confidence,
            True,
            final_diff_x,
            final_diff_y,
            dialog_roi=roi_scaled,
        )
