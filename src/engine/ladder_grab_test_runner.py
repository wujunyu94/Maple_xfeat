"""主程序内的梯/绳单次走位跳抓压力测试。"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import cv2


@dataclass(frozen=True)
class LadderGrabTestConfig:
    start_platform_id: int
    ladder_id: int
    start_x: float
    rounds: int = 10
    closed_loop_walk: bool = False
    run_jump_grab: bool = False
    disturbance_x_min_px: float = 10.0
    disturbance_x_max_px: float = 10.0
    disturbance_y_min_px: float = 10.0
    disturbance_y_max_px: float = 10.0
    disturbance_duration_sec: float = 1.5
    calibrate_raw_landing_tolerance: bool = False
    landing_sample_seconds: float = 0.8


class LadderGrabTestRunner:
    """使用 F6 真实单边执行器的绳梯抓取与往返压力测试。"""

    def __init__(
        self,
        driver,
        motion,
        motion_model,
        get_graph: Callable[[], object],
        get_raw_position: Callable[[], Optional[Tuple[float, float]]],
        get_world_position: Optional[Callable[[], Optional[Tuple[float, float]]]],
        capture_frame: Callable[[], object],
        log: Callable[[str], None],
        status_callback: Optional[Callable[[str, bool], None]] = None,
        get_pixel_world_scale: Optional[Callable[[], Tuple[float, float]]] = None,
        get_intra_map_portal_enabled: Optional[Callable[[], bool]] = None,
        f6_edge_executor: Optional[Callable[..., object]] = None,
        begin_f6_test_session: Optional[Callable[[], Tuple[bool, str]]] = None,
        end_f6_test_session: Optional[Callable[[], None]] = None,
    ):
        self.driver = driver
        self.motion = motion
        self.motion_model = motion_model
        self.get_graph = get_graph
        self.get_raw_position = get_raw_position
        self.get_world_position = get_world_position or get_raw_position
        self.capture_frame = capture_frame
        self.log = log
        self.status_callback = status_callback or (lambda _text, _active: None)
        self.get_pixel_world_scale = get_pixel_world_scale
        self.get_intra_map_portal_enabled = get_intra_map_portal_enabled
        self.f6_edge_executor = f6_edge_executor
        self.begin_f6_test_session = begin_f6_test_session
        self.end_f6_test_session = end_f6_test_session
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self._active_disk_log: Optional[Callable[[str], None]] = None
        self._disturbance_lock = threading.Lock()
        self._disturbance_until = 0.0
        self._disturbance_offset_x: Optional[float] = None
        self._disturbance_offset_y: Optional[float] = None
        self._disturbance_x_min_px = 10.0
        self._disturbance_x_max_px = 10.0
        self._disturbance_y_min_px = 10.0
        self._disturbance_y_max_px = 10.0
        self._disturbance_duration_sec = 1.5
        self._top_exit_transition_samples = []

    @property
    def disturbance_duration_sec(self) -> float:
        return self._disturbance_duration_sec

    def inject_position_disturbance(self) -> bool:
        """在当前稳定性测试中注入一次临时的黄点坐标偏移。"""
        if not self.active or self.stop_event.is_set():
            return False
        duration = self._disturbance_duration_sec
        # 在小地图黄点的像素坐标 X/Y 轴分别注入固定偏移，模拟短时识别漂移；
        # 每次按键重新随机方向和幅度；持续时间使用测试窗口中的设置。
        offset_x_px = random.choice((-1.0, 1.0)) * random.uniform(
            self._disturbance_x_min_px, self._disturbance_x_max_px
        )
        offset_y_px = random.choice((-1.0, 1.0)) * random.uniform(
            self._disturbance_y_min_px, self._disturbance_y_max_px
        )
        # 稳定性测试设置使用小地图黄点像素；测试内部位置回调使用世界坐标，
        # 因此按当前地图 X/Y 各自的“世界坐标/小地图像素”比例换算后再注入。
        try:
            scale_x, scale_y = self.get_pixel_world_scale() if self.get_pixel_world_scale else (0.0, 0.0)
            scale_x = float(scale_x)
            scale_y = float(scale_y)
        except (TypeError, ValueError, AttributeError):
            scale_x = scale_y = 0.0
        if scale_x <= 0.0:
            scale_x = float(getattr(self.motion_model, "measurement_step_px", 1.0) or 1.0)
        if scale_y <= 0.0:
            scale_y = scale_x
        offset_x = offset_x_px * max(1.0, scale_x)
        offset_y = offset_y_px * max(1.0, scale_y)
        with self._disturbance_lock:
            self._disturbance_offset_x = offset_x
            self._disturbance_offset_y = offset_y
            self._disturbance_until = time.perf_counter() + duration
        self._trace(
            f"🧪 [黄点扰动] 已注入黄点像素误差 "
            f"dx={offset_x_px:+.1f}px, dy={offset_y_px:+.1f}px "
            f"(世界换算={offset_x:+.1f},{offset_y:+.1f})，"
            f"持续{duration:.2f}s"
        )
        return True

    def _apply_position_disturbance(
        self, position: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """将 F10 扰动同步施加到 F6 的 raw 与吸附世界坐标观测。"""
        if position is None:
            return None
        now = time.perf_counter()
        with self._disturbance_lock:
            if (
                now >= self._disturbance_until
                or self._disturbance_offset_x is None
                or self._disturbance_offset_y is None
            ):
                return position
            offset_x = self._disturbance_offset_x
            offset_y = self._disturbance_offset_y
        return (float(position[0]) + offset_x, float(position[1]) + offset_y)

    def _get_test_position(self) -> Optional[Tuple[float, float]]:
        """返回测试专用 raw 坐标；正式运行坐标不会被修改。"""
        return self._apply_position_disturbance(self.get_raw_position())

    def _get_test_world_position(self) -> Optional[Tuple[float, float]]:
        """返回测试专用的 F6 吸附/预测世界坐标。"""
        return self._apply_position_disturbance(self.get_world_position())

    def _clear_position_disturbance(self) -> None:
        with self._disturbance_lock:
            self._disturbance_until = 0.0
            self._disturbance_offset_x = None
            self._disturbance_offset_y = None

    def _trace(self, message: str) -> None:
        """测试遥测同时显示在 GUI 并落入当前 run.log。"""
        if self._active_disk_log is not None:
            self._active_disk_log(message)
        else:
            self.log(message)

    def start(self, cfg: LadderGrabTestConfig) -> Tuple[bool, str]:
        if self.active:
            return False, "测试正在运行。"
        readiness_check = getattr(self.driver, "check_input_readiness", None)
        if callable(readiness_check):
            try:
                ready, reason = readiness_check(focus=True)
            except Exception as exc:
                return False, f"输入通道检查失败：{exc}"
            if not ready:
                self.log(f"⛔ [跳抓测试未启动] {reason}")
                return False, reason
        try:
            disturbance_x_min_px = max(0.0, float(cfg.disturbance_x_min_px))
            disturbance_x_max_px = max(0.0, float(cfg.disturbance_x_max_px))
            disturbance_y_min_px = max(0.0, float(cfg.disturbance_y_min_px))
            disturbance_y_max_px = max(0.0, float(cfg.disturbance_y_max_px))
            disturbance_duration_sec = float(cfg.disturbance_duration_sec)
        except (TypeError, ValueError):
            return False, "黄点扰动误差和持续时间必须是有效数字。"
        if disturbance_x_max_px < disturbance_x_min_px:
            return False, "黄点扰动 X 轴最大误差不能小于最小误差。"
        if disturbance_y_max_px < disturbance_y_min_px:
            return False, "黄点扰动 Y 轴最大误差不能小于最小误差。"
        if not 0.05 <= disturbance_duration_sec <= 30.0:
            return False, "黄点扰动持续时间必须在 0.05–30 秒之间。"
        graph = self.get_graph()
        if graph is None:
            return False, "当前地图拓扑尚未加载完成。"
        start = graph.get_node(cfg.start_platform_id)
        ladder = graph.get_ladder_rope(cfg.ladder_id)
        if start is None:
            return False, f"不存在起始平台 P{cfg.start_platform_id}。"
        if ladder is None:
            return False, f"不存在绳梯 #{cfg.ladder_id}。"
        if ladder.bottom_platform_id != start.id:
            return False, (
                f"绳梯 #{cfg.ladder_id} 的底端属于 P{ladder.bottom_platform_id}，"
                f"不是 P{start.id}。"
            )
        if graph.get_node(ladder.top_platform_id) is None:
            return False, f"绳梯 #{cfg.ladder_id} 缺少上端平台。"
        if not callable(self.f6_edge_executor):
            return False, "F6 共享单边执行器尚未就绪。"
        climb_edges = [
            edge for edge in graph.get_edges_from(start.id)
            if int(getattr(edge, "to_id", -1)) == int(ladder.top_platform_id)
            and int(getattr(edge, "ladder_id", -1) or -1) == int(ladder.id)
            and "CLIMB" in str(getattr(edge, "action", ""))
        ]
        if not climb_edges:
            return False, (
                f"F6 拓扑中不存在 P{start.id} 经绳梯#{ladder.id} "
                f"到 P{ladder.top_platform_id} 的可执行边。"
            )

        if callable(self.begin_f6_test_session):
            try:
                session_ok, session_message = self.begin_f6_test_session()
            except Exception as exc:
                return False, f"F6 共享导航测试会话启动失败：{exc}"
            if not session_ok:
                return False, session_message

        self.stop_event.clear()
        self._disturbance_x_min_px = disturbance_x_min_px
        self._disturbance_x_max_px = disturbance_x_max_px
        self._disturbance_y_min_px = disturbance_y_min_px
        self._disturbance_y_max_px = disturbance_y_max_px
        self._disturbance_duration_sec = disturbance_duration_sec
        self.active = True
        self.thread = threading.Thread(target=self._run, args=(cfg, graph), daemon=True)
        try:
            self.thread.start()
        except Exception:
            self.active = False
            if callable(self.end_f6_test_session):
                self.end_f6_test_session()
            raise
        return True, "测试已启动。"

    def stop(self) -> None:
        self.stop_event.set()
        self._clear_position_disturbance()
        try:
            self.motion.stop()
            self.driver.release_all_keys()
        except Exception:
            pass

    def _status(self, text: str, active: bool = True) -> None:
        try:
            self.status_callback(text, active)
        except Exception:
            pass

    def _wait(self, seconds: float) -> bool:
        return not self.stop_event.wait(max(0.0, seconds))

    def _one_shot_move(self, target_x: float) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """一次方向键按住＋自然刹停；不根据落点补按。"""
        # 与 F6 原地跳抓共用同一个“推力 + 松键阻力”反推器，避免测试
        # 已验证而自动导航仍走另一套 PID 微调的情况。
        return self.motion.move_to_x_by_motion_model(
            target_x,
            self._get_test_position,
            stop_event=self.stop_event,
        )

    def _move_to_x(
        self,
        cfg: LadderGrabTestConfig,
        target_x: float,
        platform_bounds: Optional[Tuple[float, float]] = None,
        tolerance: int = 18,
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """按测试选项选择开环单次走位或主导航闭环走位。"""
        # 开环模式的所有原地跳抓都使用同一套一次性运动学反推；不再
        # 针对低净空梯子悄悄退回 PID 微调，保证测试和 F6 可直接对照。
        if not cfg.closed_loop_walk:
            return self._one_shot_move(target_x)

        before = self._get_test_position()
        if before is None:
            return False, 0.0, None
        distance = abs(float(target_x) - float(before[0]))
        timeout = min(12.0, max(2.5, distance / max(1.0, float(self.motion_model.v_max)) + 2.0))
        self.log(
            f"🧪 [闭环走位] X={before[0]:.0f} -> {target_x:.0f}，"
            f"每次 raw 更新重算剩余距离与刹停"
        )
        started = time.perf_counter()
        ok = self.motion.walk_to_x(
            int(round(target_x)),
            get_player_pos=self._get_test_position,
            tolerance=max(1, int(tolerance)),
            timeout_sec=timeout,
            stop_event=self.stop_event,
            platform_bounds=platform_bounds,
            safe_margin=12,
        )
        return ok, time.perf_counter() - started, self._get_test_position()

    def _grab_and_climb(
        self,
        graph,
        top_platform,
        run_direction: Optional[str] = None,
        keep_direction_sec: float = 0.0,
    ) -> Tuple[bool, bool, Optional[Tuple[float, float]]]:
        before = self._get_test_position()
        if before is None:
            return False, False, None
        launch = self._get_test_position()
        min_y = float(before[1])
        min_y_pos = before
        self._trace(f"🧪 [跑跳遥测/起跳] raw={launch}，抓取基准={before}，UP+Alt 前")
        self.driver.key_down("up")
        try:
            if not self._wait(0.02):
                return False, False, self._get_test_position()
            self.driver.press_key("alt", duration_ms=80)
            # 跑跳抓取在 Alt 后继续保持原方向，直至预测的绳梯底端
            # 相交时刻；之后立即松开横向键，避免带着速度错过绳梯。
            if run_direction in ("left", "right"):
                self._wait(keep_direction_sec)
                self.driver.key_up(run_direction)
                self._trace(
                    f"🧪 [跑跳遥测/松横向] 方向={run_direction}，保持={keep_direction_sec:.3f}s，"
                    f"raw={self._get_test_position()}"
                )
            deadline = time.perf_counter() + 4.8
            grabbed = False
            top_hits = 0
            while time.perf_counter() < deadline and not self.stop_event.is_set():
                now = self._get_test_position()
                if now is not None and float(now[1]) < min_y:
                    min_y = float(now[1])
                    min_y_pos = now
                current = graph.find_player_platform(*(now or (0, 0)))
                # 只有当角色高度持续高于起跳点 25px 以上（脱离了普通起跳落地周期），
                # 或者已经爬上顶部平台时，才确认真正抓住了梯绳。
                if now is not None and (now[1] < before[1] - 25 or (current is not None and current.id == top_platform.id)):
                    grabbed = True
                if grabbed and current is not None and current.id == top_platform.id:
                    top_hits += 1
                    if top_hits >= 3:
                        break
                else:
                    top_hits = 0
                if not self._wait(0.08):
                    break
            if grabbed and not self.stop_event.is_set():
                # 保证角色从绳梯顶端完整翻上平台。
                self._wait(1.5)
        finally:
            self.driver.key_up("up")

        self._wait(0.20)
        after = self._get_test_position()
        current = graph.find_player_platform(*(after or (0, 0)))
        topped = bool(current is not None and current.id == top_platform.id)
        # 如果最终角色掉回起跳层且未登顶，判定抓取失败
        if not topped and after is not None and after[1] >= before[1] - 15:
            grabbed = False
        self._trace(
            f"🧪 [跑跳遥测/结束] raw={after}，最高点raw={min_y_pos}，"
            f"Y最小={min_y:.0f}，有效抓取={grabbed}，登顶={topped}"
        )
        return grabbed, topped, after

    @staticmethod
    def _run_jump_plan(ladder, current_pos, v_max: float) -> Tuple[str, float, float, float]:
        """以梯绳底端高度反推起跳后保持方向的时间与起跳准备点。"""
        current_x, current_y = float(current_pos[0]), float(current_pos[1])
        direction = "right" if ladder.x > current_x else "left"
        sign = 1.0 if direction == "right" else -1.0
        # 梯绳 y2 已是底端的世界 Y；它与 raw 黄点使用同一世界坐标
        # 基准。
        capture_y = float(max(ladder.y1, ladder.y2))
        ladder_top_y = float(min(ladder.y1, ladder.y2))

        # 当绳梯底端齐平或低于脚下地面时（如废弃都市穿过地面的排水管道），
        # 绳梯在角色站立高度已有实体。跑跳不是为了“摸到底端”，
        # 而是以舒适的跳跃抛物线上升期（0.15~0.18s，离地上升约35~45px）扑入绳梯中段。
        if capture_y >= current_y - 10.0:
            target_grab_y = max(ladder_top_y + 30.0, current_y - 42.0)
            rise_needed = max(0.0, current_y - target_grab_y)
            disc = max(0.0, 555.0 * 555.0 - 4000.0 * rise_needed)
            capture_sec = (555.0 - math.sqrt(disc)) / 2000.0 if rise_needed > 0.0 else 0.17
            capture_sec = max(0.15, min(0.22, capture_sec))
            # 空中位移以奔跑速度覆盖，并留足穿过绳梯中心线的余量，
            # 保证起跳点距离绳梯 60~75px，在跳跃上升期稳稳挂上。
            air_distance = max(55.0, float(v_max) * capture_sec + 8.0)
            prep_x = float(ladder.x) - sign * air_distance
            return direction, prep_x, capture_sec, air_distance

        # 悬空梯子：梯子底端明显高于脚下地面，需要跳起指定高度以接触底端
        rise_needed = max(0.0, current_y - capture_y)
        disc = max(0.0, 555.0 * 555.0 - 4000.0 * rise_needed)
        capture_sec = (555.0 - math.sqrt(disc)) / 2000.0 if rise_needed > 0.0 else 0.08
        capture_sec = max(0.08, min(0.26, capture_sec))
        air_distance = max(25.0, float(v_max) * capture_sec + 8.0)
        prep_x = float(ladder.x) - sign * air_distance
        return direction, prep_x, capture_sec, air_distance

    def _run_jump_and_grab(self, cfg: LadderGrabTestConfig, graph, start, ladder, top):
        """跑跳抓：移动到由高度反推的准备线时不中断方向，直接触发跳抓。"""
        current = self._get_test_position()
        if current is None:
            return False, False, None, 0.0, None
        # 与 F6 同步的实机标定档案。仅在用户勾选“跑跳抓绳/梯”时会
        # 调用本函数；原地测试仍走下方的 _move_to_x + _grab_and_climb。
        if getattr(graph, "map_id", None) == 101000102 and int(ladder.id) == 1:
            direction = "right" if float(ladder.x) > float(current[0]) else "left"
            sign = 1.0 if direction == "right" else -1.0
            air_distance, capture_sec = 74.0, 0.180
            prep_x = float(ladder.x) - sign * air_distance
            prep_x = max(float(start.x_min + 6), min(float(start.x_max - 6), prep_x))
            self._trace(
                f"🧪 [跑跳实机标定] 101000102 梯#1：准备X={prep_x:.1f}，"
                f"方向={direction}，Alt后保持={capture_sec:.3f}s"
            )
            self.driver.ensure_focus()
            self.driver.key_down(direction)
            run_started = time.perf_counter()
            next_direction_refresh = run_started + 0.15
            movement_probe_at = run_started + 0.80
            deadline = run_started + min(10.0, max(1.2, abs(prep_x - float(current[0])) / max(1.0, self.motion_model.v_max) + 2.0))
            while time.perf_counter() < deadline and not self.stop_event.is_set():
                # 部分客户端的后台 PostMessage 不会自动产生系统按键重复事件，
                # 只发送一次 key_down 时可能只移动一个很小的量。定期刷新
                # 同一方向的 key_down，保持长距离跑跳输入连续。
                now_refresh = time.perf_counter()
                if now_refresh >= next_direction_refresh:
                    self.driver.key_down(direction)
                    next_direction_refresh = now_refresh + 0.15
                now = self._get_test_position()
                if (
                    now is not None
                    and time.perf_counter() >= movement_probe_at
                    and abs(float(now[0]) - float(current[0])) < 1.0
                ):
                    self.driver.key_up(direction)
                    elapsed = time.perf_counter() - run_started
                    self._trace(
                        f"⛔ [输入未生效] 按住{direction.upper()} {elapsed:.2f}s后 "
                        f"raw X仍为{float(now[0]):.0f}；已停止本轮，请检查管理员权限/游戏焦点"
                    )
                    self.stop_event.set()
                    return False, False, now, elapsed, prep_x
                if now is not None and (
                    (direction == "right" and now[0] >= prep_x)
                    or (direction == "left" and now[0] <= prep_x)
                ):
                    break
                self._wait(0.012)
            run_sec = time.perf_counter() - run_started
            self._trace(f"🧪 [跑跳标定起跳] raw={self._get_test_position()}，助跑={run_sec:.3f}s")
            grabbed, topped, after = self._grab_and_climb(
                graph, top, run_direction=direction, keep_direction_sec=capture_sec
            )
            return grabbed, topped, after, run_sec, prep_x
        bottom_y = float(max(ladder.y1, ladder.y2))
        # 梯绳底端与脚下高度只差一个小地图量化步长时，没有可用于
        # 跑跳的垂直窗口。实机 P1 -> 1 号梯证实：raw X=-674 可抓，
        # X=-658（相邻量化格）必然抓空。因此必须先精确停在梯绳格点。
        low_clearance = 0.0 <= float(current[1]) - bottom_y <= 16.5
        if low_clearance:
            self._trace(
                f"🧪 [低净空抓取] 脚下Y={current[1]:.0f}，底端Y={bottom_y:.0f}，"
                f"差={float(current[1]) - bottom_y:.1f}px；改为精确对位 X={ladder.x:.0f}"
            )
            moved, hold_sec, aligned = self._move_to_x(
                cfg, float(ladder.x), (start.x_min, start.x_max), tolerance=3
            )
            if aligned is None or abs(float(aligned[0]) - float(ladder.x)) > 4.0:
                self._trace(
                    f"🧪 [低净空拒跳] 对位未命中：raw={aligned}，目标X={ladder.x:.0f}"
                )
                return False, False, aligned, hold_sec, float(ladder.x)
            self._trace(f"🧪 [低净空起跳] raw={aligned}，精确对位完成，执行UP+Alt")
            grabbed, topped, after = self._grab_and_climb(graph, top)
            return grabbed, topped, after, hold_sec, float(ladder.x)
        direction, prep_x, capture_sec, air_distance = self._run_jump_plan(
            ladder, current, float(self.motion_model.v_max)
        )
        prep_x = max(float(start.x_min + 6), min(float(start.x_max - 6), prep_x))
        distance = abs(prep_x - float(current[0]))
        self._trace(
            f"🧪 [跑跳遥测/计划] 起始raw={current}，绳梯X={ladder.x:.0f}，"
            f"准备X={prep_x:.1f}，方向={direction}，空中保持={capture_sec:.3f}s"
        )
        # 开环测试使用预测到准备线的时间；闭环测试则以 raw 坐标跨过
        # 准备线为触发条件。两者都不在准备线停下。
        self.driver.ensure_focus()
        self.driver.key_down(direction)
        run_started = time.perf_counter()
        next_direction_refresh = run_started + 0.15
        if cfg.closed_loop_walk:
            movement_probe_at = run_started + 0.80
            deadline = time.perf_counter() + min(10.0, max(1.2, distance / max(1.0, self.motion_model.v_max) + 2.0))
            while time.perf_counter() < deadline and not self.stop_event.is_set():
                # 定期刷新方向键，兼容后台输入模式下没有自动重复键的客户端。
                now_refresh = time.perf_counter()
                if now_refresh >= next_direction_refresh:
                    self.driver.key_down(direction)
                    next_direction_refresh = now_refresh + 0.15
                now = self._get_test_position()
                if (
                    now is not None
                    and time.perf_counter() >= movement_probe_at
                    and abs(float(now[0]) - float(current[0])) < 1.0
                ):
                    self.driver.key_up(direction)
                    elapsed = time.perf_counter() - run_started
                    self._trace(
                        f"⛔ [输入未生效] 按住{direction.upper()} {elapsed:.2f}s后 "
                        f"raw X仍为{float(now[0]):.0f}；已停止本轮，请检查管理员权限/游戏焦点"
                    )
                    self.stop_event.set()
                    return False, False, now, elapsed, prep_x
                if now is not None and (
                    (direction == "right" and now[0] >= prep_x)
                    or (direction == "left" and now[0] <= prep_x)
                ):
                    break
                self._wait(0.015)
            run_sec = time.perf_counter() - run_started
        else:
            run_sec = distance / max(1.0, float(self.motion_model.v_max))
            self._wait(run_sec)
        self._trace(
            f"🏃 [跑跳物理计划] 底端Y={max(ladder.y1, ladder.y2)}，上升相交={capture_sec:.3f}s，"
            f"空中位移={air_distance:.1f}px，准备X={prep_x:.1f}，助跑={run_sec:.3f}s"
        )
        self._trace(f"🧪 [跑跳遥测/准备线] raw={self._get_test_position()}，即将连续执行UP+Alt")
        grabbed, topped, after = self._grab_and_climb(
            graph, top, run_direction=direction, keep_direction_sec=capture_sec
        )
        self._trace(
            f"🧪 [跑跳遥测/时序] Alt至相交总计={capture_sec:.3f}s，"
            f"Alt后额外保持={capture_sec:.3f}s"
        )
        return grabbed, topped, after, run_sec, prep_x

    @staticmethod
    def _edge_label(edge) -> str:
        ladder = (
            f" ladder#{edge.ladder_id}"
            if getattr(edge, "ladder_id", None) is not None else ""
        )
        return f"P{edge.from_id}->P{edge.to_id} {edge.action}{ladder}"

    def _test_location(self, graph):
        position = self._get_test_world_position()
        if position is None:
            return None, None, False
        try:
            return graph.find_player_location(float(position[0]), float(position[1]))
        except Exception:
            platform = graph.find_player_platform(float(position[0]), float(position[1]))
            return platform, None, False

    def _test_platform(self, graph):
        return self._test_location(graph)[0]

    def _test_is_climbing(self, graph) -> bool:
        return bool(self._test_location(graph)[2])

    @staticmethod
    def _edge_verify_timeout(edge) -> float:
        action = str(getattr(edge, "action", ""))
        if "CLIMB" in action:
            return 1.40
        if "LONG_DROP" in action or action == "PORTAL":
            return 2.20
        if "DROP" in action or action == "DOWN_JUMP":
            return 1.60
        return 1.35

    def _wait_for_stable_platform(self, graph, timeout_sec: float):
        deadline = time.perf_counter() + max(0.0, float(timeout_sec))
        stable_id = None
        stable_hits = 0
        latest = None
        while time.perf_counter() < deadline and not self.stop_event.is_set():
            latest = self._test_platform(graph)
            platform_id = getattr(latest, "id", None)
            if platform_id is not None and platform_id == stable_id:
                stable_hits += 1
            elif platform_id is not None:
                stable_id = platform_id
                stable_hits = 1
            else:
                stable_id = None
                stable_hits = 0
            if stable_hits >= 2:
                return latest
            self._wait(0.06)
        return latest

    def _execute_shared_f6_edge(self, cfg, graph, edge, macro_target_id: int):
        position = self._get_test_world_position()
        if position is None:
            self._trace(f"⚠️ [F6共享边] {self._edge_label(edge)} 执行前无坐标")
            return False, None, "missing_position", 0.0
        self._trace(
            f"▶️ [F6共享边] {self._edge_label(edge)}，"
            f"宏观目标=P{macro_target_id}，当前={position}"
        )
        started = time.perf_counter()
        calibration_tolerance = None
        calibration_callback = None
        if cfg.calibrate_raw_landing_tolerance and "CLIMB" in str(edge.action):
            try:
                _scale_x, scale_y = (
                    self.get_pixel_world_scale()
                    if self.get_pixel_world_scale else (1.0, 1.0)
                )
                scale_y = max(1.0, float(scale_y))
            except (TypeError, ValueError, AttributeError):
                scale_y = 1.0
            calibration_tolerance = max(64.0, min(120.0, scale_y * 4.0))
            calibration_callback = self._top_exit_transition_samples.append
        outcome = self.f6_edge_executor(
            edge,
            int(macro_target_id),
            float(position[0]),
            float(position[1]),
            world_position_getter=self._get_test_world_position,
            raw_position_getter=self._get_test_position,
            platform_getter=lambda: self._test_platform(graph),
            is_climbing_getter=lambda: self._test_is_climbing(graph),
            stop_event=self.stop_event,
            run_jump_enabled=bool(cfg.run_jump_grab),
            log_callback=self._trace,
            top_exit_landing_sample_callback=calibration_callback,
            top_exit_landing_tolerance_override=calibration_tolerance,
        )
        elapsed = time.perf_counter() - started
        observed = self._wait_for_stable_platform(
            graph, self._edge_verify_timeout(edge)
        )
        arrived = bool(
            outcome != "retry"
            and observed is not None
            and int(observed.id) == int(edge.to_id)
        )
        self._trace(
            f"{'✅' if arrived else '⚠️'} [F6共享边验证] {self._edge_label(edge)} "
            f"结果={outcome}，观测P{getattr(observed, 'id', None)}，"
            f"耗时={elapsed:.2f}s"
        )
        return arrived, observed, outcome, elapsed

    def _sample_stable_landing_error(self, cfg, graph, top) -> list:
        """落到目标台并停止输入后，采集真实 raw Y 与理论站立 Y 的误差。"""
        if not cfg.calibrate_raw_landing_tolerance:
            return []
        samples = []
        deadline = time.perf_counter() + max(
            0.20, min(3.0, float(cfg.landing_sample_seconds))
        )
        while time.perf_counter() < deadline and not self.stop_event.is_set():
            raw = self._get_test_position()
            platform = self._test_platform(graph)
            if (
                raw is not None
                and platform is not None
                and int(platform.id) == int(top.id)
                and (top.x_min + 2) <= float(raw[0]) <= (top.x_max - 2)
            ):
                expected_y = float(top.surface_y_at(float(raw[0]))) - 45.0
                samples.append(
                    {
                        "raw_x": float(raw[0]),
                        "raw_y": float(raw[1]),
                        "expected_y": expected_y,
                        "y_error": float(raw[1]) - expected_y,
                    }
                )
            self._wait(0.025)
        return samples

    @staticmethod
    def _percentile(values, quantile: float) -> Optional[float]:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return None
        index = max(
            0,
            min(len(ordered) - 1, int(math.ceil(quantile * len(ordered))) - 1),
        )
        return ordered[index]

    def _write_landing_calibration_summary(
        self, out_dir: str, disk_log, transition_samples: list, stable_samples: list
    ) -> None:
        eligible = [
            item
            for item in transition_samples
            if item.get("in_platform")
            and item.get("clear_of_rope")
            and int(item.get("landing_samples", 0)) > 0
        ]
        transition_errors = [abs(float(item["y_error"])) for item in eligible]
        stable_errors = [abs(float(item["y_error"])) for item in stable_samples]
        all_errors = transition_errors + stable_errors
        try:
            _scale_x, scale_y = (
                self.get_pixel_world_scale()
                if self.get_pixel_world_scale else (1.0, 1.0)
            )
            scale_y = max(1.0, float(scale_y))
        except (TypeError, ValueError, AttributeError):
            scale_y = 1.0
        p95 = self._percentile(all_errors, 0.95)
        max_error = max(all_errors) if all_errors else None
        # 加半个黄点 Y 量化格作为下一次落台可能落在格子另一侧的保护量，
        # 再向上取整为偶数，便于作为 UI 参数使用。
        recommended = None
        if p95 is not None and max_error is not None:
            basis = max(float(p95), float(max_error)) + max(2.0, scale_y * 0.5)
            recommended = int(math.ceil(basis / 2.0) * 2)
        payload = {
            "transition_sample_count": len(transition_errors),
            "stable_sample_count": len(stable_errors),
            "measurement_step_y": scale_y,
            "absolute_error_p95": p95,
            "absolute_error_max": max_error,
            "recommended_raw_landing_y_tolerance_px": recommended,
            "transition_samples": transition_samples,
            "stable_samples": stable_samples,
        }
        result_path = os.path.join(out_dir, "landing_tolerance_calibration.json")
        with open(result_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        if recommended is None:
            disk_log("[CALIBRATION] 未取得有效落台样本，无法计算建议容差")
        else:
            disk_log(
                f"[CALIBRATION] 转场样本={len(transition_errors)}，"
                f"静止样本={len(stable_errors)}，Y量化步长={scale_y:.2f}px，"
                f"|误差| P95={p95:.2f}px，最大={max_error:.2f}px，"
                f"建议raw落台Y容差={recommended}px"
            )
        disk_log(f"[CALIBRATION_FILE] {result_path}")

    def _navigate_by_f6_shortest_path(
        self, cfg: LadderGrabTestConfig, graph, destination_id: int, reason: str
    ) -> Tuple[bool, list]:
        """与 F6 相同：find_path 规划，共享单边执行器，按实际落台重规划。"""
        route_log = []
        failures = {}
        max_steps = max(12, min(80, len(getattr(graph, "nodes", {})) * 4))
        for step_index in range(1, max_steps + 1):
            if self.stop_event.is_set() or self.get_graph() is not graph:
                return False, route_log
            current = self._test_platform(graph)
            if current is None:
                current = self._wait_for_stable_platform(graph, 1.20)
            if current is None:
                self._trace(f"⚠️ [{reason}] 无法定位当前承重平台")
                return False, route_log
            if int(current.id) == int(destination_id):
                return True, route_log

            path = graph.find_path(
                int(current.id), int(destination_id),
                allow_run_jump=bool(cfg.run_jump_grab),
                allow_portal=(
                    bool(self.get_intra_map_portal_enabled())
                    if self.get_intra_map_portal_enabled is not None else True
                ),
            )
            if not path:
                self._trace(
                    f"⛔ [{reason}] F6 拓扑无路径：P{current.id}->P{destination_id}"
                )
                return False, route_log
            labels = [self._edge_label(item) for item in path]
            metrics = graph.path_metrics(path) if hasattr(graph, "path_metrics") else {}
            self._trace(
                f"🗺️ [{reason}] 第{step_index}次规划 P{current.id}->P{destination_id} "
                f"cost={metrics.get('estimated_cost', '?')} ropes={metrics.get('rope_count', '?')} | "
                + " | ".join(labels)
            )
            edge = path[0]
            route_log.append(self._edge_label(edge))
            arrived, observed, outcome, _elapsed = self._execute_shared_f6_edge(
                cfg, graph, edge, int(destination_id)
            )
            if arrived:
                failures.pop((edge.from_id, edge.to_id, edge.action), None)
                continue
            if outcome == "retry":
                self._trace(f"🔁 [{reason}] 起跳未放行，保留最短路径原边重试")
                self._wait(0.30)
                continue
            if observed is not None and int(observed.id) != int(current.id):
                self._trace(
                    f"↪️ [{reason}] 实际落到P{observed.id}，从实际平台重规划"
                )
                continue
            key = (edge.from_id, edge.to_id, edge.action)
            failures[key] = failures.get(key, 0) + 1
            if failures[key] >= 4:
                self._trace(
                    f"⛔ [{reason}] {self._edge_label(edge)} 连续4次未离开源平台，结束本轮"
                )
                return False, route_log
            self._wait(0.30)
        self._trace(f"⛔ [{reason}] 超过最大拓扑执行步数 {max_steps}")
        return False, route_log

    def _run_shared_climb_edge(self, cfg, graph, start, ladder, top):
        candidates = [
            edge for edge in graph.get_edges_from(start.id)
            if int(getattr(edge, "to_id", -1)) == int(top.id)
            and int(getattr(edge, "ladder_id", -1) or -1) == int(ladder.id)
            and "CLIMB" in str(getattr(edge, "action", ""))
        ]
        if not candidates:
            return False, None, 0.0, []
        edge = min(candidates, key=lambda item: float(getattr(item, "cost", 0.0)))
        attempts = []
        total_elapsed = 0.0
        for attempt in range(1, 5):
            current = self._test_platform(graph)
            if current is not None and int(current.id) == int(top.id):
                return True, self._get_test_position(), total_elapsed, attempts
            if current is None or int(current.id) != int(start.id):
                self._trace(
                    f"⚠️ [F6跳抓验收] 第{attempt}次前已不在源平台P{start.id}，"
                    f"当前P{getattr(current, 'id', None)}"
                )
                break
            attempts.append(self._edge_label(edge))
            arrived, _observed, outcome, elapsed = self._execute_shared_f6_edge(
                cfg, graph, edge, int(top.id)
            )
            total_elapsed += elapsed
            if arrived:
                return True, self._get_test_position(), total_elapsed, attempts
            if outcome == "retry":
                self._wait(0.30)
                continue
            self._wait(0.30)
        return False, self._get_test_position(), total_elapsed, attempts

    def _return_to_start_x(self, cfg, start) -> Tuple[bool, Optional[Tuple[float, float]]]:
        current = self._get_test_world_position()
        if current is None:
            return False, None
        tolerance = max(4, min(20, int(float(start.length) * 0.20)))
        distance = abs(float(current[0]) - float(cfg.start_x))
        arrived = self.motion.walk_to_x(
            target_x=float(cfg.start_x),
            get_player_pos=self._get_test_world_position,
            tolerance=tolerance,
            timeout_sec=max(2.0, min(7.0, distance / 100.0 + 1.5)),
            stop_event=self.stop_event,
            platform_bounds=(start.x_min, start.x_max),
            safe_margin=max(8, min(25, int(float(start.length) * 0.10))),
            speed_scale=1.0,
        )
        return arrived, self._get_test_position()

    def _run(self, cfg: LadderGrabTestConfig, graph) -> None:
        ladder = graph.get_ladder_rope(cfg.ladder_id)
        start = graph.get_node(cfg.start_platform_id)
        top = graph.get_node(ladder.top_platform_id)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(root, "logs", "ladder_grab_tests", f"{graph.map_id}_ladder_{ladder.id}")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")
        csv_path = os.path.join(out_dir, "results.csv")
        rows = []
        self._top_exit_transition_samples = []
        stable_landing_samples = []

        def disk_log(message: str) -> None:
            stamp = time.strftime("%H:%M:%S")
            line = f"[{stamp}] {message}"
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")
            self.log(line)

        with open(log_path, "w", encoding="utf-8") as fp:
            fp.write(f"=== ladder grab test: {asdict(cfg)} ===\n")
        self._active_disk_log = disk_log

        try:
            if not self.driver.ensure_focus():
                disk_log("[ABORT] 无法聚焦游戏窗口")
                return
            disk_log(
                f"[START] MapID={graph.map_id} P{start.id}@{cfg.start_x:.0f} "
                f"-> {ladder.kind_name}#{ladder.id}@X={ladder.x} -> P{top.id}"
            )
            for n in range(1, cfg.rounds + 1):
                if self.stop_event.is_set() or self.get_graph() is not graph:
                    disk_log("[STOP] 用户停止或地图已切换")
                    break
                current = self._get_test_position()
                curr_platform = self._test_platform(graph)
                if curr_platform is None or curr_platform.id != start.id:
                    disk_log(f"[STOP] 第{n}轮起点不在 P{start.id}：raw={current}，平台={getattr(curr_platform, 'id', None)}")
                    break

                self._status(f"第 {n}/{cfg.rounds} 轮：F6共享执行 {ladder.kind_name} #{ladder.id}")
                topped, after_grab, hold_sec, outbound_edges = self._run_shared_climb_edge(
                    cfg, graph, start, ladder, top
                )
                grabbed = topped
                after_move = after_grab
                disk_log(
                    f"[ROUND {n}] f6_edges={outbound_edges} after={after_move} "
                    f"grabbed={grabbed} topped={topped}"
                )
                if topped and cfg.calibrate_raw_landing_tolerance:
                    round_samples = self._sample_stable_landing_error(
                        cfg, graph, top
                    )
                    stable_landing_samples.extend(round_samples)
                    if round_samples:
                        errors = [
                            abs(float(item["y_error"])) for item in round_samples
                        ]
                        disk_log(
                            f"[CALIBRATION ROUND {n}] 静止落台样本={len(errors)}，"
                            f"|Y误差|={min(errors):.2f}..{max(errors):.2f}px"
                        )

                self._status(f"第 {n}/{cfg.rounds} 轮：F6最短路径返回 P{start.id}")
                return_path_ok, return_edges = self._navigate_by_f6_shortest_path(
                    cfg, graph, int(start.id), "F6返程最短路径"
                )
                positioned, final_pos = (
                    self._return_to_start_x(cfg, start)
                    if return_path_ok else (False, self._get_test_position())
                )
                final_platform = self._test_platform(graph)
                returned = bool(
                    return_path_ok
                    and positioned
                    and final_platform is not None
                    and final_platform.id == start.id
                    and final_pos is not None and abs(float(final_pos[0]) - cfg.start_x) <= 40.0
                )
                screenshot = os.path.join(out_dir, f"round_{n:02d}.png")
                frame = self.capture_frame()
                if frame is not None:
                    cv2.imwrite(screenshot, frame)
                row = {
                    "round": n, "start_raw": current, "move_hold_sec": round(hold_sec, 4),
                    "after_move": after_move, "grabbed": grabbed, "topped": topped,
                    "after_grab": after_grab, "outbound_edges": " | ".join(outbound_edges),
                    "return_edges": " | ".join(return_edges), "return_path_ok": return_path_ok,
                    "returned": returned,
                    "final_raw": final_pos, "final_platform": getattr(final_platform, "id", None),
                }
                rows.append(row)
                disk_log(f"[ROUND_END {n}] returned={returned} final={final_pos}")
                self._status(f"第 {n}/{cfg.rounds} 轮完成：爬顶={topped}，回程={returned}")
                if not self._wait(0.50):
                    break
        except Exception as exc:
            disk_log(f"[FATAL] {exc!r}")
        finally:
            self._clear_position_disturbance()
            self._active_disk_log = None
            try:
                self.motion.stop()
                self.driver.release_all_keys()
            except Exception:
                pass
            if callable(self.end_f6_test_session):
                try:
                    self.end_f6_test_session()
                except Exception as exc:
                    self.log(f"⚠️ [跳抓测试] 恢复 F6 导航状态失败：{exc}")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
                fields = list(rows[0].keys()) if rows else ["round"]
                writer = csv.DictWriter(fp, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            if cfg.calibrate_raw_landing_tolerance:
                self._write_landing_calibration_summary(
                    out_dir,
                    disk_log,
                    list(self._top_exit_transition_samples),
                    stable_landing_samples,
                )
            success = sum(bool(r["grabbed"] and r["topped"]) for r in rows)
            returned = sum(bool(r["returned"]) for r in rows)
            summary = f"完成 {len(rows)} 轮：爬顶 {success}/{len(rows)}，回程 {returned}/{len(rows)}"
            self.log(f"🧪 [跳抓测试结果] {summary}；日志：{out_dir}")
            self._status(summary, False)
            self.active = False
