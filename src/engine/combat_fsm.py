"""
combat_fsm.py - 自动挂机战斗状态机 (CombatFSM & Auto-Grind Engine)
驱动视觉中枢、物理运动控制器与航点系统，实现全自动索敌、走位逼近、转向攻击与卡死脱困。
"""

import time
import math
import random
import threading
from enum import Enum
from typing import Optional, Tuple, List, Dict, Callable, Any

from src.core.input_driver import InputDriver
from src.vision.main_view_detector import (
    MainViewDetector,
    MonsterTarget,
    is_fresh_attack_observation,
    monster_observation_age,
)
from src.engine.motion_controller import MotionController
from src.engine.platform_graph import PlatformGraphBuilder
from src.engine.platform_patrol_fsm import PatrolPhase, PlatformPatrolFSM
from src.engine.waypoint_manager import WaypointManager, Waypoint
from src.engine.attack_skills import (
    attack_box as skill_attack_box, contains as target_in_box,
    eligible_skills, mob_id_of, rear_box as skill_rear_box, skills_from_config,
    skirmish_boxes as skill_skirmish_boxes,
    choose_skill_for_target,
)
from src.engine.attack_stall_guard import AttackStallGuard


class BotState(Enum):
    IDLE = "空闲"
    PATROLLING = "巡逻中"
    APPROACHING = "靠近怪物"
    ATTACKING = "攻击歼灭"
    LOOTING = "拾取物品"
    UNSTUCK = "卡死脱困"
    ATTACK_ONLY_WAIT = "仅攻击待命"


class PlatformTransitionState(Enum):
    """一条长平台边从规划到落地确认的执行阶段。"""
    IDLE = "空闲"
    PLANNING = "规划平台路径"
    APPROACHING = "接近短段起跳区"
    READY = "起跳就绪"
    EXECUTING = "执行平台转移"
    VERIFYING = "确认目标平台"
    RECOVERING = "失败恢复"


class CombatFSM:
    def __init__(
        self,
        detector: MainViewDetector,
        input_driver: InputDriver,
        motion_controller: MotionController,
        waypoint_manager: WaypointManager,
        config: Dict,
        log_callback: Optional[Callable[[str], None]] = None,
        get_platform_graph: Optional[Callable[[], Any]] = None,
        get_patrol_platforms: Optional[Callable[[], List[int]]] = None,
        get_current_platform: Optional[Callable[[], Any]] = None,
        get_is_climbing: Optional[Callable[[], bool]] = None,
        get_player_world_pos: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
        get_player_raw_world_pos: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
        get_current_frame: Optional[Callable[[], Optional[Any]]] = None,
        on_single_step_complete: Optional[Callable[[], None]] = None,
        reset_motion_prediction: Optional[Callable[[], None]] = None,
        get_enable_run_jump: Optional[Callable[[], bool]] = None,
        get_enable_run_jump_fallback: Optional[Callable[[], bool]] = None,
        get_run_jump_failure_limit: Optional[Callable[[], int]] = None,
        get_enable_monster_detection: Optional[Callable[[], bool]] = None,
        get_patrol_dwell_range: Optional[Callable[[], Tuple[float, float]]] = None,
        get_single_patrol_positions: Optional[Callable[[], List[float]]] = None,
        get_platform_patrol_positions: Optional[Callable[[], Dict[int, Tuple[float, ...]]]] = None,
        get_patrol_position_random: Optional[Callable[[], float]] = None,
        get_patrol_arrival_tolerance: Optional[Callable[[], float]] = None,
        get_rest_settings: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        get_global_rest_due: Optional[Callable[[], bool]] = None,
        cross_map_tick: Optional[Callable[[], bool]] = None,
        patrol_target_completed_callback: Optional[Callable[[int], None]] = None,
        rest_task_completed_callback: Optional[Callable[[int], None]] = None,
    ):
        self.detector = detector
        self.driver = input_driver
        self.motion = motion_controller
        self.waypoints = waypoint_manager
        self.config = config
        self.log_fn = log_callback or print
        self.get_platform_graph = get_platform_graph
        self.get_patrol_platforms = get_patrol_platforms
        self.get_current_platform = get_current_platform
        self.get_is_climbing = get_is_climbing
        self.get_player_world_pos = get_player_world_pos
        self.get_player_raw_world_pos = get_player_raw_world_pos
        self.get_current_frame = get_current_frame
        self.on_single_step_complete = on_single_step_complete
        self.reset_motion_prediction = reset_motion_prediction
        self.get_enable_run_jump = get_enable_run_jump
        self.get_enable_run_jump_fallback = get_enable_run_jump_fallback
        self.get_run_jump_failure_limit = get_run_jump_failure_limit
        self.get_enable_monster_detection = get_enable_monster_detection
        self.get_patrol_dwell_range = get_patrol_dwell_range
        self.get_single_patrol_positions = get_single_patrol_positions
        self.get_platform_patrol_positions = get_platform_patrol_positions
        self.get_patrol_position_random = get_patrol_position_random
        self.get_patrol_arrival_tolerance = get_patrol_arrival_tolerance
        self.get_rest_settings = get_rest_settings
        self.get_global_rest_due = get_global_rest_due
        self.cross_map_tick = cross_map_tick
        self._patrol_target_completed_callback = patrol_target_completed_callback
        self._rest_task_completed_callback = rest_task_completed_callback

        self.state = BotState.IDLE
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self._attack_input_lock = threading.Lock()
        self._attack_paused = threading.Event()
        self.f6_session_id = 0
        self.f6_started_at = 0.0

        # 战斗参数
        self.attack_key = self.config.get("attack_key", "ctrl")
        self.jump_key = self.config.get("jump_key", "alt")
        self.pick_key = self.config.get("pick_key", "z")
        self.attack_reach_x = self.config.get("attack_reach_x", 260)
        legacy_reach_y = self.config.get("attack_reach_y", 140)
        self.attack_reach_y_up = self.config.get("attack_reach_y_up", legacy_reach_y)
        self.attack_reach_y_down = self.config.get("attack_reach_y_down", legacy_reach_y)
        self.behind_reach_x = self.config.get("behind_reach_x", 40)
        self.attack_two_way = bool(self.config.get("attack_two_way", False))

        # 平台巡航状态
        self.patrol_target_idx = 0
        self.platform_stay_time = 0.0
        self.patrol_dir = 1
        self._last_patrol_platform_id: Optional[int] = None
        self._short_platform_centered_id: Optional[int] = None
        self.platform_transition_state = PlatformTransitionState.IDLE
        self.active_platform_transition = None
        self._transition_started_at = 0.0
        # 每条跨层绳梯边独立记录跑跳失败次数；避免 FSM 重进同一段
        # 后又从第 1 次跑跳开始，永远不会切换到原地跳抓。
        self._run_jump_failures: Dict[Tuple[int, int, Optional[int]], int] = {}
        # 卡死监控
        self.last_state_change = time.perf_counter()
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.stuck_duration = 0.0
        # 启动挂机时如果角色正挂在绳梯上，先完成脱绳，避免把
        # “无平台”状态误交给普通巡航逻辑而原地卡住。
        self._last_climb_nudge = 0.0
        self._was_climbing = False
        self._climb_finish_hold_until = 0.0
        self._climb_up_held = False
        self._down_climb_anti_up_until = 0.0
        self._last_downward_verify_attempt = None
        self._climb_watch_y: Optional[float] = None
        self._climb_watch_progress_at = 0.0
        self._climb_false_ignore_until = 0.0
        self._startup_climb_active = False
        self._startup_climb_clear_pending = False
        self._startup_climb_clear_done = False
        self._patrol_forced_target: Optional[int] = None
        # 新巡逻FSM已经选定的边必须原样交给动作执行器。若这里只保留
        # target，旧执行器会再次 find_path，并把带失败惩罚选出的备用边
        # 悄悄换回基准最短边。
        self._patrol_forced_edge = None
        self._active_down_jump_target: Optional[Tuple[int, int, float, Tuple[float, float]]] = None
        # 独立绳梯压力测试会复用 F6 的同一条边执行器。测试期间临时
        # 替换坐标/平台回调与 stop_event，因此必须串行化并保存 F6 状态。
        self._external_edge_lock = threading.RLock()
        self._external_test_active = False
        self._external_saved_navigation_state: Optional[Dict[str, Any]] = None
        self._last_vertical_source_lost = False
        self._last_worker_exception_log_at = 0.0
        self._last_ignored_patrol_target_log_at = 0.0
        # 战斗误触发诊断：分别限制地面中断检查与实际攻击日志的频率，
        # 既保留候选怪物/人物的完整坐标证据，又避免 50Hz 主循环刷爆日志。
        self._last_attack_debug_log_at: Dict[str, float] = {}
        self._last_attack_output_latency_log_at = 0.0
        self._has_attacked_monster = False
        self._forward_clear_since: Optional[float] = None
        self._rear_turn_pending_direction: Optional[str] = None
        self._rear_turn_pending_until = 0.0
        self._last_rear_wait_log_at = 0.0
        # 某个平台的某一侧已到达游击安全边界时，锁存该方向。
        # 目标仍在同一侧时不再每帧重发 stop/走位，防止 KEYUP 风暴
        # 压制玩家的手动输入。目标消失或换平台后自动解除。
        self._skirmish_boundary_blocks: Dict[Tuple[int, str], float] = {}
        # 直接攻击前的内撤保护独立于游击追怪；走位失败后限频重试，
        # 避免红框持续存在时反复按方向键。
        self._attack_edge_retry_key: Optional[Tuple[int, str]] = None
        self._attack_edge_retry_after = 0.0

        # 仅攻击键介入模式 (移动由玩家控制，进攻击范围只自动按攻击)
        self.attack_only_mode: bool = bool(self.config.get("attack_only_mode", False))
        self._attack_only_last_locked_id: Optional[int] = None
        self._attack_only_last_locked_name: Optional[str] = None
        self._attack_only_lock_start_t: float = 0.0
        self._attack_only_last_log_t: float = 0.0
        self._attack_only_last_climb_log_t: float = 0.0
        self._attack_only_next_attack_at: float = 0.0
        self._normal_attack_next_at: float = 0.0
        self._attack_stall_guard = AttackStallGuard()
        self._normal_attack_last_climb_log_t: float = 0.0
        self._dwell_attack_boxes_occupied = False
        # 休息调度与普通巡逻共用同一个平台状态机，但以独立目标覆盖当前
        # 循环。这样仍能复用全部长平台、绳梯和失败恢复逻辑。
        self._rest_phase = "idle"
        self._rest_settings_signature: Tuple[Any, ...] = ()
        self._rest_settings: Optional[Dict[str, Any]] = None
        self._rest_normal_elapsed = 0.0
        self._rest_interval_sec = 0.0
        self._rest_last_tick_at = 0.0
        self._rest_platform_id: Optional[int] = None
        self._rest_return_platform_id: Optional[int] = None
        self._rest_settle_until = 0.0
        self._rest_until = 0.0
        self._forced_rest_settings: Optional[Dict[str, Any]] = None
        self._rest_run_started_at = 0.0
        self._rest_last_completed_at = 0.0
        self.platform_patrol = PlatformPatrolFSM(
            graph_getter=lambda: self.get_platform_graph() if self.get_platform_graph else None,
            patrol_getter=self._active_patrol_targets,
            platform_getter=lambda: self.get_current_platform() if self.get_current_platform else None,
            world_position_getter=(
                lambda: self.get_player_world_pos() if self.get_player_world_pos else None
            ),
            edge_executor=self._execute_patrol_edge,
            motion=self.motion,
            stop_event=self.stop_event,
            log_callback=self.log_fn,
            run_jump_enabled_getter=self.get_enable_run_jump,
            intra_map_portal_enabled_getter=lambda: not bool(
                self.config.get("disable_intra_map_portals", False)
            ),
            dwell_range_getter=self.get_patrol_dwell_range,
            single_positions_getter=self.get_single_patrol_positions,
            platform_positions_getter=self.get_platform_patrol_positions,
            position_random_getter=self.get_patrol_position_random,
            arrival_tolerance_getter=self.get_patrol_arrival_tolerance,
            dwell_extension_getter=lambda: self.config.get(
                "patrol_combat_dwell_extension_sec", 2.0
            ),
            dwell_threat_getter=lambda: self._dwell_attack_boxes_occupied,
            failure_replan_enabled_getter=lambda: bool(
                self.config.get("patrol_failure_replan_enabled", False)
            ),
            rest_navigation_getter=lambda: self._rest_phase == "travel",
            target_completed_callback=self._on_platform_target_completed,
            target_arrival_override=self._handle_rest_target_arrival,
            inplace_dwell_safe_margin_getter=lambda: float(
                self.config.get("inplace_dwell_safe_margin_px", 50.0)
            ),
            fatal_block_callback=self.stop,
        )
        self.motion.external_force_callback = self.platform_patrol.report_external_force
        # 地面导航是阻塞闭环；让它逐帧检查红色攻击框，才能在长平台
        # 助跑途中及时把输入权交还战斗主循环。
        try:
            self.motion.priority_interrupt_checker = self._attack_priority_pending
        except Exception:
            pass

    def _active_patrol_targets(self) -> List[int]:
        if self._rest_phase in ("travel", "settling", "resting"):
            return [self._rest_platform_id] if self._rest_platform_id is not None else []
        if self._rest_phase == "returning":
            return (
                [self._rest_return_platform_id]
                if self._rest_return_platform_id is not None else []
            )
        return self.get_patrol_platforms() if self.get_patrol_platforms else []

    def _on_platform_target_completed(self, platform_id: int) -> None:
        # 休息去程/回程属于内部任务，绝不能让跨地图控制器把它计为
        # 当前地图的正常巡逻目标完成。
        if self._rest_phase != "idle":
            return
        if self._patrol_target_completed_callback is not None:
            self._patrol_target_completed_callback(int(platform_id))

    @staticmethod
    def _normalized_rest_settings(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not raw or raw.get("platform_id") is None:
            return None
        try:
            platform_id = int(raw["platform_id"])
            duration_low = max(0.0, float(raw.get("duration_min_sec", 30.0)))
            duration_high = max(duration_low, float(raw.get("duration_max_sec", 60.0)))
            interval_low = max(1.0, float(raw.get("interval_min_sec", 600.0)))
            interval_high = max(interval_low, float(raw.get("interval_max_sec", 900.0)))
            map_id = raw.get("map_id")
            map_id = int(map_id) if map_id is not None else None
        except (TypeError, ValueError):
            return None
        return {
            "map_id": map_id,
            "platform_id": platform_id,
            "duration_min_sec": min(86400.0, duration_low),
            "duration_max_sec": min(86400.0, duration_high),
            "interval_min_sec": min(86400.0, interval_low),
            "interval_max_sec": min(86400.0, interval_high),
        }

    @staticmethod
    def _rest_signature(settings: Optional[Dict[str, Any]]) -> Tuple[Any, ...]:
        if settings is None:
            return ()
        return (
            settings.get("map_id"), settings["platform_id"],
            round(settings["duration_min_sec"], 3),
            round(settings["duration_max_sec"], 3),
            round(settings["interval_min_sec"], 3),
            round(settings["interval_max_sec"], 3),
        )

    def _sample_rest_interval(self, settings: Dict[str, Any]) -> float:
        low = float(settings["interval_min_sec"])
        high = float(settings["interval_max_sec"])
        return random.uniform(low, high) if high > low else low

    def _reset_rest_scheduler(self, reason: str) -> None:
        was_active = self._rest_phase != "idle"
        self._rest_phase = "idle"
        self._rest_settings_signature = ()
        self._rest_settings = None
        self._rest_normal_elapsed = 0.0
        self._rest_interval_sec = 0.0
        self._rest_last_tick_at = time.perf_counter()
        self._rest_platform_id = None
        self._rest_return_platform_id = None
        self._rest_settle_until = 0.0
        self._rest_until = 0.0
        self._forced_rest_settings = None
        self._rest_last_completed_at = 0.0
        if was_active:
            self.log_fn(f"🪑 [休息任务取消] {reason}")

    def request_forced_rest(self, settings: Dict[str, Any]) -> None:
        """跨地图全局计时到点后，由地图控制器派发一次休息任务。"""
        self._reset_rest_scheduler("跨地图休息任务接管")
        self._forced_rest_settings = dict(settings)
        self.log_fn(
            f"🪑 [全局休息派发] MapID {settings.get('map_id')} "
            f"P{settings.get('platform_id')}，立即前往休息平台"
        )

    def _begin_rest_task(self, current_platform: Any) -> None:
        assert self._rest_settings is not None
        self.motion.stop()
        self._clear_route_attack_hold()
        self._forward_clear_since = None
        self._rear_turn_pending_direction = None
        self._dwell_attack_boxes_occupied = False
        self._rest_platform_id = int(self._rest_settings["platform_id"])
        self._rest_return_platform_id = int(current_platform.id)
        self._rest_phase = "travel"
        self.platform_patrol.reset("rest_due")
        self.platform_patrol._set_phase(
            PatrolPhase.REST_TRAVEL,
            f"P{self._rest_return_platform_id}→休息P{self._rest_platform_id}",
        )
        self.log_fn(
            f"🪑 [休息任务触发] 正常巡逻{self._rest_normal_elapsed:.1f}s/"
            f"{self._rest_interval_sec:.1f}s；中止当前行为，从P{self._rest_return_platform_id}"
            f"前往休息P{self._rest_platform_id}"
        )

    def _finish_rest_task(self, platform_id: int, *, global_rest: bool = False) -> None:
        self._rest_phase = "idle"
        now = time.perf_counter()
        self._rest_last_tick_at = now
        self._rest_normal_elapsed = (
            max(0.0, now - self._rest_last_completed_at)
            if not global_rest and self._rest_last_completed_at > 0.0 else 0.0
        )
        self._rest_platform_id = None
        self._rest_return_platform_id = None
        self._rest_settle_until = 0.0
        self._rest_until = 0.0
        self.platform_patrol.reset("rest_complete")
        self.platform_patrol._set_phase(PatrolPhase.OBSERVE, "休息结束，恢复原巡逻")
        if global_rest:
            self.log_fn(f"✅ [休息任务完成] P{platform_id}休息结束，交给跨地图返程控制器")
        else:
            self.log_fn(
                f"✅ [休息任务完成] 已返回P{platform_id}；恢复原巡逻，"
                f"下次休息间隔={self._rest_interval_sec:.1f}s"
            )

    def _handle_rest_target_arrival(
        self, platform: Any, position: Tuple[float, float], now: float
    ) -> bool:
        """覆盖休息任务的到站行为，跳过普通站位百分比和怪物行为。"""
        if self._rest_phase == "idle":
            return False

        if self._rest_phase in ("travel", "settling", "resting"):
            if self._rest_platform_id is None or int(platform.id) != self._rest_platform_id:
                return True
            center_x = float(platform.center_x)
            tolerance = max(3, min(20, int(float(platform.length) * 0.20)))
            if self._rest_phase == "travel":
                if abs(float(position[0]) - center_x) > tolerance:
                    arrived = self.motion.walk_to_x(
                        target_x=center_x,
                        get_player_pos=self.get_player_world_pos,
                        tolerance=tolerance,
                        timeout_sec=max(
                            2.0,
                            min(10.0, abs(float(position[0]) - center_x) / 100.0 + 1.5),
                        ),
                        stop_event=self.stop_event,
                        platform_bounds=(platform.x_min, platform.x_max),
                        safe_margin=max(8, min(25, int(platform.length * 0.10))),
                        speed_scale=1.0,
                    )
                    latest_platform = self.get_current_platform() if self.get_current_platform else None
                    if not arrived or latest_platform is None or int(latest_platform.id) != int(platform.id):
                        return True
                self.motion.stop()
                self._rest_phase = "settling"
                self._rest_settle_until = time.perf_counter() + 5.0
                self.platform_patrol._set_phase(
                    PatrolPhase.REST_SETTLING,
                    f"休息P{platform.id}中点等待5.00s",
                )
                self.log_fn(
                    f"📍 [休息点到达] P{platform.id}中点X={center_x:.1f}，"
                    "站定等待5.00s后按椅子键"
                )
                return True

            if self._rest_phase == "settling":
                latest_position = self.get_player_world_pos()
                if latest_position is None:
                    self._rest_settle_until = time.perf_counter() + 5.0
                    return True
                if abs(float(latest_position[0]) - center_x) > tolerance:
                    self._rest_phase = "travel"
                    self._rest_settle_until = 0.0
                    self.log_fn(f"↩️ [休息点站位丢失] P{platform.id}，重新到中点后再等待5秒")
                    return True
                if now < self._rest_settle_until:
                    return True
                settings = self._rest_settings or {}
                low = float(settings.get("duration_min_sec", 30.0))
                high = float(settings.get("duration_max_sec", 60.0))
                duration = random.uniform(low, high) if high > low else low
                chair_key = str(self.config.get("chair_key", "end"))
                chair_vk = self.config.get("chair_vk")
                self.driver.press_key(chair_key, duration_ms=70, vk_code=chair_vk)
                self._rest_phase = "resting"
                self._rest_until = time.perf_counter() + duration
                self.platform_patrol._set_phase(
                    PatrolPhase.RESTING,
                    f"P{platform.id}休息{duration:.2f}s",
                )
                self.log_fn(
                    f"🪑 [椅子按键] key={chair_key.upper()}，P{platform.id}休息"
                    f"{duration:.2f}s（范围{low:.2f}～{high:.2f}s）"
                )
                return True

            if now < self._rest_until:
                return True
            if self._forced_rest_settings is not None:
                self._forced_rest_settings = None
                self._finish_rest_task(int(platform.id), global_rest=True)
                if self._rest_task_completed_callback is not None:
                    self._rest_task_completed_callback(int(platform.id))
                return True
            # 第二轮间隔从椅子休息结束起算，而不是从返程落到原平台起算。
            self._rest_last_completed_at = now
            if self._rest_settings is not None:
                self._rest_interval_sec = self._sample_rest_interval(self._rest_settings)
            self._rest_normal_elapsed = 0.0
            return_id = self._rest_return_platform_id
            self._rest_phase = "returning"
            self.platform_patrol.reset("rest_duration_complete")
            self.platform_patrol._set_phase(
                PatrolPhase.REST_RETURN,
                f"休息结束，返回P{return_id}",
            )
            self.log_fn(f"↩️ [休息返回] 休息结束，从P{platform.id}返回P{return_id}")
            return True

        if self._rest_phase == "returning":
            if (
                self._rest_return_platform_id is not None
                and int(platform.id) == self._rest_return_platform_id
            ):
                self._finish_rest_task(int(platform.id))
            return True
        return True

    def _service_rest_task(
        self, world_position: Optional[Tuple[float, float]]
    ) -> bool:
        """更新正常巡逻计时；活动中的休息任务独占输入。"""
        now = time.perf_counter()
        try:
            raw = (
                self._forced_rest_settings
                if self._forced_rest_settings is not None else
                (self.get_rest_settings() if self.get_rest_settings else None)
            )
        except Exception:
            raw = None
        settings = self._normalized_rest_settings(raw)
        signature = self._rest_signature(settings)
        if signature != self._rest_settings_signature:
            if self._rest_phase != "idle":
                self.motion.stop()
                self.platform_patrol.reset("rest_map_or_settings_changed")
            self._rest_phase = "idle"
            self._rest_settings = settings
            self._rest_settings_signature = signature
            anchor = self._rest_last_completed_at or self._rest_run_started_at
            self._rest_normal_elapsed = (
                0.0 if self._forced_rest_settings is not None
                else (
                    max(0.0, now - anchor)
                    if anchor > 0.0 else 0.0
                )
            )
            self._rest_last_tick_at = now
            self._rest_platform_id = None
            self._rest_return_platform_id = None
            self._rest_interval_sec = (
                0.0 if self._forced_rest_settings is not None else
                (self._sample_rest_interval(settings) if settings is not None else 0.0)
            )
            if settings is not None:
                self.log_fn(
                    f"⏱️ [休息调度] P{settings['platform_id']}，"
                    f"本轮正常巡逻间隔={self._rest_interval_sec:.2f}s"
                )
        elif settings is not None and self._rest_phase == "idle":
            self._rest_normal_elapsed += max(0.0, now - self._rest_last_tick_at)
            self._rest_last_tick_at = now
        else:
            self._rest_last_tick_at = now

        if settings is None:
            return False
        if self._rest_phase == "idle" and self._rest_normal_elapsed >= self._rest_interval_sec:
            current = self.get_current_platform() if self.get_current_platform else None
            if current is None:
                return False
            self._begin_rest_task(current)

        if self._rest_phase == "idle":
            return False
        old_checker = getattr(self.motion, "priority_interrupt_checker", None)
        try:
            # 休息去程和回程不因怪物红框让行；整个任务期间不攻击。
            self.motion.priority_interrupt_checker = None
            if world_position is None:
                self._do_patrol(None, None)
            else:
                self._do_patrol(world_position[0], world_position[1])
        finally:
            self.motion.priority_interrupt_checker = old_checker
        return True

    def _set_platform_transition_state(self, state: PlatformTransitionState, edge=None):
        """更新短段级平台转移状态，并保存当前宏观边。"""
        self.platform_transition_state = state
        if edge is not None:
            self.active_platform_transition = edge
        if state == PlatformTransitionState.EXECUTING:
            self._transition_started_at = time.perf_counter()
        elif state in (PlatformTransitionState.IDLE, PlatformTransitionState.PLANNING):
            self.active_platform_transition = None
            self._transition_started_at = 0.0

    def start(self, f6_started_at: Optional[float] = None):
        """开启自动挂机引擎"""
        if self.is_running:
            return
        now = time.perf_counter()
        self.f6_session_id += 1
        self.f6_started_at = (
            min(now, max(0.0, float(f6_started_at)))
            if f6_started_at is not None else now
        )
        self._attack_paused.clear()
        self.is_running = True
        self.stop_event.clear()
        self.state = BotState.ATTACK_ONLY_WAIT if self.attack_only_mode else BotState.PATROLLING
        self._set_platform_transition_state(PlatformTransitionState.IDLE)
        self._attack_only_last_locked_id = None
        self._attack_only_last_locked_name = None
        self._attack_only_lock_start_t = 0.0
        self._attack_only_last_log_t = 0.0
        self._attack_only_last_climb_log_t = 0.0
        self._attack_only_next_attack_at = 0.0
        self._normal_attack_next_at = 0.0
        self._attack_stall_guard.reset()
        self._normal_attack_last_climb_log_t = 0.0
        if not self.attack_only_mode:
            self.platform_patrol.start()
        # 若按下 F6 时角色已经在绳/梯上，先走“启动时绳上收束”流程。
        self._startup_climb_active = False
        self._startup_climb_clear_pending = False
        self._startup_climb_clear_done = False
        self._climb_watch_y = None
        self._climb_watch_progress_at = 0.0
        self._climb_false_ignore_until = 0.0
        self._down_climb_anti_up_until = 0.0
        self._last_downward_verify_attempt = None
        self._has_attacked_monster = False
        self._forward_clear_since = None
        self._rear_turn_pending_direction = None
        self._rear_turn_pending_until = 0.0
        self._dwell_attack_boxes_occupied = False
        self._skirmish_boundary_blocks.clear()
        self._attack_edge_retry_key = None
        self._attack_edge_retry_after = 0.0
        self._clear_route_attack_hold()
        self._reset_rest_scheduler("F6启动")
        self._rest_run_started_at = (
            min(now, max(0.0, float(f6_started_at)))
            if f6_started_at is not None else now
        )
        if not self.attack_only_mode and self.get_is_climbing is not None:
            try:
                self._startup_climb_active = bool(self.get_is_climbing())
            except Exception:
                self._startup_climb_active = False
        self.thread = threading.Thread(target=self._fsm_worker, daemon=True)
        self.thread.start()
        if self.attack_only_mode:
            self.log_fn("⚔️ [挂机引擎] 【仅攻击键介入模式】已启动！移动由玩家自行控制，怪物进入攻击范围时自动攻击。")
        else:
            self.log_fn("🚀 [挂机引擎] 自动挂机系统已启动！")

    def stop(self):
        """停止自动挂机引擎并急停按键"""
        # stop() 会被窗口关闭、F6、断线急停等多个入口调用。未运行时不能
        # 再次操作 stop_event：独立导航验收会临时把它替换成测试自己的
        # Event，旧行为会误停测试并且每帧重复打印“自动挂机已停止”。
        if not self.is_running:
            return False
        was_attack_only = self.attack_only_mode
        self.is_running = False
        self.stop_event.set()
        self.state = BotState.IDLE
        self._set_platform_transition_state(PlatformTransitionState.IDLE)
        self.platform_patrol.stop()
        self._reset_rest_scheduler("F6停止")
        self._clear_route_attack_hold()
        if not was_attack_only:
            self.motion.stop()
        with self._attack_input_lock:
            self.driver.release_all_keys()
        self._attack_paused.clear()
        self._normal_attack_next_at = 0.0
        self._attack_stall_guard.reset()
        self._skirmish_boundary_blocks.clear()
        self._attack_edge_retry_key = None
        self._attack_edge_retry_after = 0.0
        self._attack_only_last_locked_id = None
        self._attack_only_last_locked_name = None
        if was_attack_only:
            self.log_fn("🛑 [挂机引擎] 仅攻击键介入已停止。")
        else:
            self.log_fn("🛑 [挂机引擎] 自动挂机已停止。")
        return True

    def feed_pet_safely(self, key: str, vk_code: Optional[int] = None,
                        session_id: Optional[int] = None) -> bool:
        """Pause automated attack input, release it, then press the feed key."""
        key = str(key or "").strip().lower()
        if (not key or not self.config.get("enable_auto_pet_feed", False)
                or not self.is_running or self.stop_event.is_set()):
            return False
        if session_id is not None and session_id != self.f6_session_id:
            return False
        attack_bindings = {
            str(skill["key"]).lower(): skill["vk"]
            for skill in skills_from_config(self.config)
        }
        if key in attack_bindings:
            self.log_fn(f"⚠️ [宠物喂食] 喂食键 {key.upper()} 与攻击技能键冲突，跳过")
            return False
        self._attack_paused.set()
        try:
            with self._attack_input_lock:
                if (not self.config.get("enable_auto_pet_feed", False)
                        or not self.is_running or self.stop_event.is_set()
                        or (session_id is not None and session_id != self.f6_session_id)):
                    return False
                # The attack loop normally releases its own key before the
                # lock becomes available. This also handles a stale tracked
                # key without lifting unrelated movement keys.
                active = getattr(self.driver, "active_keys", set())
                for attack_key, attack_vk in attack_bindings.items():
                    if attack_key in active:
                        self.driver.key_up(attack_key, vk_code=attack_vk)
                time.sleep(0.04)
                if (not self.config.get("enable_auto_pet_feed", False)
                        or not self.is_running or self.stop_event.is_set()):
                    return False
                self.driver.press_key(key, duration_ms=70, vk_code=vk_code)
                return True
        finally:
            self._attack_paused.clear()

    def reset_navigation_failures(self):
        """切换地图时清空绳梯跑跳失败计数。"""
        self._run_jump_failures.clear()
        self._clear_route_attack_hold()
        self._set_platform_transition_state(PlatformTransitionState.IDLE)
        self.platform_patrol.reset("navigation_failures_reset")

    def _random_climb_clear(self, platform, get_player_pos, reason: str) -> None:
        """脱绳后随机侧移一小段，避免固定目标点触发预测刹停。"""
        if platform is None or get_player_pos is None or self.stop_event.is_set():
            return
        pos = get_player_pos()
        if pos is None:
            return
        x = float(pos[0])
        margin = 8.0
        available = []
        if x - (float(platform.x_min) + margin) > 12.0:
            available.append("left")
        if (float(platform.x_max) - margin) - x > 12.0:
            available.append("right")
        if not available:
            return
        direction = random.choice(available)
        duration = random.uniform(0.40, 0.50)
        self.log_fn(
            f"🚶 [{reason}] 随机向{('左' if direction == 'left' else '右')}"
            f"移动 {duration:.2f}s，脱离绳梯后再继续导航"
        )
        self.motion.stop()
        self.driver.key_down(direction)
        try:
            time.sleep(duration)
        finally:
            self.driver.key_up(direction)
            self.motion.stop()

    def run_single_navigation_step(self) -> bool:
        """仅执行当前拓扑路径的第一段跨层动作，不启动连续挂机循环。"""
        if self.is_running:
            self.log_fn("⚠️ [单步跨层] 请先停止自动挂机，再执行单步动作。")
            return False

        def worker():
            try:
                graph = self.get_platform_graph() if self.get_platform_graph else None
                patrol = self.get_patrol_platforms() if self.get_patrol_platforms else []
                pos = self.get_player_world_pos() if self.get_player_world_pos else None
                curr = self.get_current_platform() if self.get_current_platform else None
                if graph is None or not patrol or pos is None or curr is None:
                    self.log_fn("⚠️ [单步跨层] 缺少地图、循环平台或当前角色平台信息。")
                    return

                # 若正站在循环列表中的一个目标平台，则下一个列表项才是
                # 本次单步的宏观目标；中间平台则沿用现有目标继续下一段。
                if curr.id in patrol:
                    self.patrol_target_idx = (patrol.index(curr.id) + 1) % len(patrol)

                target_id = patrol[self.patrol_target_idx % len(patrol)]
                enabled = True
                if self.get_enable_run_jump is not None:
                    try:
                        enabled = bool(self.get_enable_run_jump())
                    except Exception:
                        enabled = True
                path = graph.find_path(
                    curr.id,
                    target_id,
                    allow_run_jump=enabled,
                    allow_portal=not bool(
                        self.config.get("disable_intra_map_portals", False)
                    ),
                )
                if not path:
                    self.log_fn(f"⚠️ [单步跨层] P{curr.id} -> P{target_id} 未找到可执行路径。")
                    return

                self.stop_event.clear()
                self.log_fn(
                    f"▶️ [单步跨层] P{curr.id} -> P{target_id}：仅执行第 1 段 "
                    f"{path[0].action}"
                )
                self._do_patrol(pos[0], pos[1])
            except Exception as exc:
                self.log_fn(f"⚠️ [单步跨层异常] {exc}")
            finally:
                self.motion.stop()
                self.stop_event.set()
                if self.on_single_step_complete is not None:
                    try:
                        self.on_single_step_complete()
                    except Exception:
                        pass

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _fsm_attack_only_tick(self):
        """仅攻击键介入模式：不控制任何移动/转向/跳跃，仅当怪物在攻击框内时自动执行攻击键并输出毫秒级日志。"""
        if not self.is_running or self.stop_event.is_set():
            return

        now = time.perf_counter()
        s_pos = getattr(self.detector, "last_player_pos", None)
        spx, spy = s_pos if s_pos is not None else (400, 300)
        facing = getattr(self.detector, "current_facing", "right")

        monsters = self._fresh_monster_tracks()
        best_target, in_attack_box = self._evaluate_best_target(
            spx, spy, facing, monsters
        )

        # 1. 目标脱离或无目标状态处理
        if best_target is None or not in_attack_box:
            if self._attack_only_last_locked_id is not None:
                duration_ms = (now - self._attack_only_lock_start_t) * 1000.0
                last_name = self._attack_only_last_locked_name or "未知"
                last_id = self._attack_only_last_locked_id
                self.log_fn(
                    f"💨 [目标脱离/击杀] 目标=[{last_name}]#{last_id} "
                    f"脱离攻击判定范围或已消灭，在框总时长={duration_ms:.1f}ms"
                )
                self._attack_only_last_locked_id = None
                self._attack_only_last_locked_name = None
            self.state = BotState.ATTACK_ONLY_WAIT
            return

        # 2. 目标在攻击范围内
        tid = getattr(best_target, "track_id", "?")
        tname = getattr(best_target, "name", "怪物")
        tscore = getattr(best_target, "score", 0.0)
        tcx = getattr(best_target, "center_x", getattr(best_target, "cx", 0))
        tcy = getattr(best_target, "center_y", getattr(best_target, "cy", 0))
        tbbox = getattr(best_target, "bbox", (0, 0, 0, 0))
        dist = float(math.hypot(tcx - spx, tcy - spy))

        # 3. 怪物识别日志输出 (初次进入或切换目标，或间隔超过1.0秒)
        is_new_target = (tid != self._attack_only_last_locked_id)
        if is_new_target or (now - self._attack_only_last_log_t >= 1.0):
            obs_age_ms = self._track_real_age(best_target, now) * 1000.0
            attack_box_func = getattr(self.detector, "calculate_attack_box", None)
            abox_info = ""
            if callable(attack_box_func):
                try:
                    frame_w = int(getattr(self.detector, "last_frame_width", 1920))
                    abox = attack_box_func((spx, spy), facing, frame_w=frame_w)
                    abox_info = f", 攻击框={abox}"
                except Exception:
                    pass

            self.log_fn(
                f"🎯 [怪物识别] 目标进入攻击范围: [{tname}] (ID:{tid}, 置信度:{tscore:.2f}, "
                f"中心:({tcx:.0f},{tcy:.0f}), BBox:{tbbox}, 距角色:{dist:.1f}px, "
                f"朝向:{facing}, 识别延迟:{obs_age_ms:.1f}ms{abox_info})"
            )
            self._attack_only_last_locked_id = tid
            self._attack_only_last_locked_name = tname
            if is_new_target:
                self._attack_only_lock_start_t = now
            self._attack_only_last_log_t = now

        # 4. 绳梯状态检测：如果在绳梯上，严禁执行攻击键
        is_climbing = False
        if self.get_is_climbing is not None:
            try:
                is_climbing = bool(self.get_is_climbing())
            except Exception:
                is_climbing = False

        if is_climbing:
            if now - getattr(self, "_attack_only_last_climb_log_t", 0.0) >= 1.0:
                self.log_fn(
                    f"🪜 [绳梯抑制] 目标=[{tname}]#{tid} 在攻击范围内，"
                    f"但角色处于绳梯状态，跳过攻击按键"
                )
                self._attack_only_last_climb_log_t = now
            self.state = BotState.ATTACK_ONLY_WAIT
            return

        # 5. 攻击冷却与执行
        if now < getattr(self, "_attack_only_next_attack_at", 0.0):
            return

        self.state = BotState.ATTACKING
        skill = self._skill_for_target(best_target, spx, spy, facing, monsters)
        if skill is None:
            return
        atk_key, atk_vk = skill["key"], skill["vk"]

        # 锁定朝向保护（防止挥刀动作影响朝向检测）
        lock_facing = getattr(self.detector, "lock_facing_during_attack", None)
        if callable(lock_facing):
            lock_facing(facing, duration_sec=0.50)

        # 执行 1~2 次攻击按键
        rounds = random.randint(1, 2)
        for press_idx in range(rounds):
            if (not self.is_running or self.stop_event.is_set()
                    or self._attack_paused.is_set()):
                break

            # 绳梯状态复核（若连击间隙角色攀上了绳梯，立即中断后续攻击）
            if self.get_is_climbing is not None:
                try:
                    if bool(self.get_is_climbing()):
                        self.log_fn("🪜 [绳梯抑制] 检测到角色进入绳梯，立即中断后续攻击")
                        break
                except Exception:
                    pass

            # 复核目标存活与在框
            if press_idx > 0:
                current_tracks = self._fresh_monster_tracks()
                live_target, live_in_box = self._evaluate_best_target(
                    spx, spy, facing, current_tracks
                )
                if live_target is None or not live_in_box:
                    break
                next_skill = self._skill_for_target(live_target, spx, spy, facing, current_tracks)
                if next_skill is None:
                    break
                skill = next_skill
                atk_key, atk_vk = skill["key"], skill["vk"]
                tname = getattr(live_target, "name", "怪物")
                tid = getattr(live_target, "track_id", "?")

            press_ms = random.randint(55, 75)
            t_press_start = time.perf_counter()
            self.log_fn(
                f"⚔️ [攻击执行] 按下攻击键: 技能={skill['name']} key='{atk_key}' (vk=0x{atk_vk:02X}), "
                f"计划时长={press_ms}ms, 连段={press_idx + 1}/{rounds}, 目标=[{tname}]#{tid}"
            )
            with self._attack_input_lock:
                if self._attack_paused.is_set() or not self.is_running or self.stop_event.is_set():
                    break
                try:
                    self.driver.key_down(atk_key, vk_code=atk_vk)
                    time.sleep(press_ms / 1000.0)
                finally:
                    self.driver.key_up(atk_key, vk_code=atk_vk)
            actual_press_ms = (time.perf_counter() - t_press_start) * 1000.0

            cd_ms = random.randint(60, 90) if press_idx < rounds - 1 else random.randint(150, 220)
            self.log_fn(
                f"⚔️ [攻击执行] 释放攻击键: key='{atk_key}', "
                f"实际按压={actual_press_ms:.1f}ms, 后摇CD={cd_ms}ms"
            )
            time.sleep(cd_ms / 1000.0)

        self._attack_only_next_attack_at = time.perf_counter() + 0.05

    def _fsm_worker(self):
        """FSM 决策工作主循环"""
        while self.is_running and not self.stop_event.is_set():
            try:
                if self._attack_paused.is_set():
                    time.sleep(0.01)
                    continue
                if self.attack_only_mode:
                    self._fsm_attack_only_tick()
                    time.sleep(0.01)
                    continue

                # 1. 优先获取高精度物理世界坐标 (降级使用视口像素)
                w_pos = self.get_player_world_pos() if self.get_player_world_pos else None
                s_pos = getattr(self.detector, 'last_player_pos', None)
                p_pos = w_pos or s_pos
                facing = getattr(self.detector, 'current_facing', 'right')

                # 跨地图编排只在“对齐传送门/等待切图/等待新图定位”阶段
                # 独占输入；刷怪和前往出口仍完全复用现有平台与战斗逻辑。
                if self.cross_map_tick is not None:
                    try:
                        if self.cross_map_tick():
                            self.state = BotState.PATROLLING
                            time.sleep(0.03)
                            continue
                    except Exception as exc:
                        now_log = time.perf_counter()
                        if now_log - self._last_worker_exception_log_at >= 1.0:
                            self._last_worker_exception_log_at = now_log
                            self.log_fn(f"⚠️ [跨地图编排异常] {exc}")

                # 角色可能在按下 F6 的瞬间仍处于绳梯/绳子状态。
                # 此时不执行平台寻路或攻击，短暂保持 UP 让角色先翻上最近平台。
                is_climbing = False
                if self.get_is_climbing is not None:
                    try:
                        is_climbing = bool(self.get_is_climbing())
                    except Exception:
                        is_climbing = False
                now = time.perf_counter()
                active_attempt = getattr(self.platform_patrol, "attempt", None)
                active_edge = getattr(active_attempt, "edge", None)
                downward_verify = (
                    self.platform_patrol.phase == PatrolPhase.VERIFY
                    and active_edge is not None
                    and str(getattr(active_edge, "action", "")).startswith("CLIMB_")
                    and str(getattr(active_edge, "action", "")).endswith("_DOWN")
                )
                if downward_verify:
                    # The generic climbing guard below always holds UP. It
                    # must never reverse an active P16->P13 descent while the
                    # platform FSM is still verifying that downward edge.
                    self._down_climb_anti_up_until = now + 1.5
                    if self._climb_up_held:
                        self.driver.key_up("up")
                        self._climb_up_held = False
                    self._was_climbing = False
                    self._climb_finish_hold_until = 0.0
                    self._climb_watch_y = None
                    self._climb_watch_progress_at = 0.0
                    if self._last_downward_verify_attempt != active_attempt.number:
                        self._last_downward_verify_attempt = active_attempt.number
                        self.log_fn(
                            f"⬇️ [下爬方向保护] P{active_edge.from_id}->P{active_edge.to_id} "
                            f"#{active_attempt.number} 核验期间禁止通用UP，只按实际落台推进"
                        )
                    self.state = BotState.PATROLLING
                    if w_pos is None:
                        self._do_patrol(None, None, verify_only=True)
                    else:
                        self._do_patrol(w_pos[0], w_pos[1], verify_only=True)
                    time.sleep(0.03)
                    continue
                if now < self._down_climb_anti_up_until:
                    # Just-landed characters can still geometrically overlap
                    # the ladder for a moment. Let the next patrol edge move
                    # them away rather than pressing UP back onto the ladder.
                    is_climbing = False
                if is_climbing and now < self._climb_false_ignore_until:
                    # 看门狗刚确认过一次静止的几何误判；短暂忽略同一个
                    # 结果，让平台巡逻器重新派发抓取，而不是立即再次按死UP。
                    is_climbing = False
                if is_climbing:
                    current_y = float(p_pos[1]) if p_pos is not None else None
                    if self._climb_watch_y is None or current_y is None:
                        self._climb_watch_y = current_y
                        self._climb_watch_progress_at = now
                    elif current_y <= self._climb_watch_y - 6.0:
                        self._climb_watch_y = current_y
                        self._climb_watch_progress_at = now
                    elif now - self._climb_watch_progress_at >= 1.20:
                        # 真实向上攀爬会持续改变Y。若几何状态一直说“在绳上”
                        # 而Y超过1.2秒没有任何向上进度，释放UP并交还巡逻重试。
                        if self._climb_up_held:
                            self.driver.key_up("up")
                            self._climb_up_held = False
                        self._was_climbing = False
                        self._climb_finish_hold_until = 0.0
                        self._climb_false_ignore_until = now + 1.0
                        self._climb_watch_y = None
                        self._climb_watch_progress_at = 0.0
                        is_climbing = False
                        self.log_fn(
                            "⚠️ [攀爬无进度恢复] 状态命中绳梯但Y连续1.20s未上升，"
                            "判为几何误吸附；已松开UP并交还巡逻重试"
                        )
                if is_climbing:
                    self.state = BotState.PATROLLING
                    # 绳梯攀爬必须连续保持 UP；周期性 press_key 会在每次
                    # 脉冲之间松键，造成明显的“一卡一卡”。
                    if not self._climb_up_held:
                        self.motion.stop()
                        self.driver.key_down("up")
                        self._climb_up_held = True
                    self._was_climbing = True
                    if self._startup_climb_active:
                        self._startup_climb_clear_done = False
                    time.sleep(0.03)
                    continue

                if now >= self._climb_false_ignore_until:
                    self._climb_watch_y = None
                    self._climb_watch_progress_at = 0.0

                # 图像/平台状态已经判定脱绳后，角色还需要完成翻台动画。
                # 固定补按 UP 1.0 秒，避免身体尚未完全站上平台就开始下一段导航。
                if self._was_climbing:
                    self._was_climbing = False
                    self._climb_finish_hold_until = now + 1.00
                    if self._startup_climb_active:
                        self._startup_climb_clear_pending = True
                if now < self._climb_finish_hold_until:
                    self.state = BotState.PATROLLING
                    time.sleep(0.03)
                    continue

                if self._climb_up_held:
                    self.driver.key_up("up")
                    self._climb_up_held = False

                # 特殊启动场景：F6 按下时角色就在绳上（例如 3 号绳）。
                # 该绳可能上下端连接同一平台，平台识别完成后若立即执行
                # DOWN_JUMP，会再次从绳子位置触发下跳。先离绳约 50px。
                if (
                    self._startup_climb_clear_pending
                    and not self._startup_climb_clear_done
                ):
                    self._startup_climb_clear_pending = False
                    self._startup_climb_clear_done = True
                    self._startup_climb_active = False
                    graph = self.get_platform_graph() if self.get_platform_graph else None
                    curr = self.get_current_platform() if self.get_current_platform else None
                    cur_pos = self.get_player_world_pos() if self.get_player_world_pos else None
                    if graph is not None and curr is not None and cur_pos is not None:
                        nearby = [
                            rope for rope in getattr(graph, "ladder_ropes", {}).values()
                            if abs(float(rope.x) - float(cur_pos[0])) <= 80.0
                        ]
                        if nearby:
                            self._random_climb_clear(
                                curr,
                                self.get_player_world_pos or (lambda: getattr(self.detector, "last_player_pos", None)),
                                "启动脱绳避让",
                            )

                # 休息任务高于索敌、攻击、普通站位和跨层派发。已经离地或
                # 正在爬绳时会先由上面的安全收束完成，落台后立即接管。
                if self._service_rest_task(w_pos):
                    self.state = BotState.PATROLLING
                    time.sleep(0.02)
                    continue

                monster_enabled = True
                if self.get_enable_monster_detection is not None:
                    try:
                        monster_enabled = bool(self.get_enable_monster_detection())
                    except Exception:
                        monster_enabled = True
                if monster_enabled:
                    monsters = self._fresh_monster_tracks()
                else:
                    monsters = []

                if p_pos is None:
                    # 尝试从平台图当前节点估算
                    curr_node = self.get_current_platform() if self.get_current_platform else None
                    if curr_node:
                        p_pos = (curr_node.center_x, curr_node.y)

                if p_pos is None:
                    # 位置完全丢失也必须推进巡逻 FSM，让它记录观测宽限、
                    # 超时并在存在最后可信平台时执行安全重新定位。旧逻辑
                    # 在这里直接 continue，造成 F6 已启动却永久静默。
                    self._do_patrol(None, None)
                    time.sleep(0.05)
                    continue

                px, py = p_pos
                # 视口屏幕中心角色锚点 (若未开启视口模板，默认以视口中轴 400, 300 对齐)
                spx, spy = s_pos if s_pos is not None else (400, 300)

                best_target, in_attack_box = self._evaluate_best_target(
                    spx, spy, facing, monsters
                )

                if in_attack_box and self.platform_patrol.phase == PatrolPhase.VERIFY:
                    # Attack still owns input priority, but must not starve
                    # confirmation of the edge that already landed.
                    if w_pos is None:
                        self._do_patrol(None, None, verify_only=True)
                    else:
                        self._do_patrol(w_pos[0], w_pos[1], verify_only=True)

                # 3. 状态分发与决策执行
                # 站位/停留只独占“方向走位”，不独占攻击。红色攻击框内
                # 有目标时允许原地攻击；范围外目标仍不会触发追怪移动。
                if self._service_exclusive_patrol_phase(
                    w_pos, best_target, in_attack_box, spx, spy, facing, monsters
                ):
                    time.sleep(0.02)
                    continue
                self._dispatch_combat_or_patrol(
                    best_target, in_attack_box, spx, spy, facing, w_pos, monsters
                )

            except Exception as e:
                print(f"[CombatFSM 异常] {e}")
                now = time.perf_counter()
                if now - self._last_worker_exception_log_at >= 1.0:
                    self._last_worker_exception_log_at = now
                    self.log_fn(f"⚠️ [CombatFSM工作线程异常] {type(e).__name__}: {e}")
                time.sleep(0.05)

            time.sleep(0.02)

        # stop() 调用方已经执行过一次完整急停。工作线程退出
        # 时只收束它自己跟踪的动作，避免玩家刚停 F6 并开始手动
        # 移动时，被延后的第二次全键 KEYUP 打断。
        self.motion.stop()

    def _evaluate_best_target(
        self,
        spx: int,
        spy: int,
        facing: str,
        monsters: List[Any]
    ) -> Tuple[Optional[Any], bool]:
        """
        根据距离、层高与朝向打分，选出最适合击杀的目标怪物 (视口屏幕像素坐标体系):
        :return: (最佳怪物, 是否已经在攻击盒内)
        """
        if not monsters:
            return None, False

        # 筛选存活怪 (兼容 TrackedMob 与 MonsterTarget)
        now = time.perf_counter()
        stall_guard = getattr(self, "_attack_stall_guard", None)
        alive_mobs = [m for m in monsters
                      if not getattr(m, 'is_ghost', False)
                      and not getattr(m, 'is_dead', False)
                      and (stall_guard is None or not stall_guard.is_suppressed(
                          str(mob_id_of(m) or getattr(m, 'name', '?')),
                          (float(spx), float(spy)),
                          (float(getattr(m, 'center_x', getattr(m, 'cx', 0))),
                           float(getattr(m, 'center_y', getattr(m, 'cy', 0)))), now))]
        if not alive_mobs:
            return None, False

        frame_w = int(getattr(self.detector, "last_frame_width", 1920))
        all_skills = skills_from_config(self.config)
        boxes_by_skill = {
            skill["id"]: skill_attack_box(skill, (spx, spy), facing, frame_w)
            for skill in all_skills
        }
        allowed_by_mob = {
            id(mob): eligible_skills(self.config, mob_id_of(mob), all_skills)
            for mob in alive_mobs
        }
        area_counts = {
            skill["id"]: sum(
                skill in allowed_by_mob[id(mob)]
                and target_in_box(boxes_by_skill[skill["id"]], mob)
                for mob in alive_mobs
            )
            for skill in all_skills if skill["area"]
        }

        scored_mobs = []
        for m in alive_mobs:
            mx = getattr(m, 'center_x', getattr(m, 'cx', m.bbox[0] + m.bbox[2]//2 if hasattr(m, 'bbox') else 0))
            my = getattr(m, 'center_y', getattr(m, 'cy', m.bbox[1] + m.bbox[3]//2 if hasattr(m, 'bbox') else 0))
            dx = mx - spx
            dy = my - spy

            allowed = allowed_by_mob[id(m)]
            if not allowed:
                continue
            boxes = [boxes_by_skill[skill["id"]] for skill in allowed]
            in_box = any(target_in_box(box, m) for box in boxes)
            is_same_platform = any(box[1] <= my <= box[1] + box[3] for box in boxes)

            # 打分机制：在攻击盒内的最高优先级 (1000分)，同平台次之 (500分)，其余按水平距离递减
            score = 0.0
            if in_box:
                score += 1000.0 - abs(dx)
                score += max((350.0 * (area_counts[skill["id"]] - 1)
                              for skill in allowed if skill["area"]
                              and area_counts[skill["id"]] >= 2
                              and target_in_box(boxes_by_skill[skill["id"]], m)), default=0.0)
            elif is_same_platform:
                score += 500.0 - abs(dx)
            else:
                score += 100.0 - (abs(dx) + abs(dy) * 2.0)

            scored_mobs.append((m, in_box, score))

        if not scored_mobs:
            return None, False

        scored_mobs.sort(key=lambda x: x[2], reverse=True)
        best = scored_mobs[0]
        return best[0], best[1]

    def _skill_for_target(self, target: Any, spx: int, spy: int, facing: str,
                          monsters: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        """Prefer an area skill only when it can hit at least two eligible mobs."""
        return choose_skill_for_target(
            self.config, target, (spx, spy), facing,
            int(getattr(self.detector, "last_frame_width", 1920)), monsters,
        )

    def _evaluate_rear_target(
        self,
        spx: int,
        spy: int,
        facing: str,
        monsters: List[Any],
    ) -> Optional[Any]:
        """返回淡橙色身后框内最近的新鲜目标；双向技能不需要回身。"""
        if not monsters:
            return None
        frame_w = int(getattr(self.detector, "last_frame_width", 1920))
        candidates = []
        for mob in monsters:
            if getattr(mob, 'is_ghost', False) or getattr(mob, 'is_dead', False):
                continue
            mx = float(getattr(mob, 'center_x', getattr(mob, 'cx', 0.0)))
            my = float(getattr(mob, 'center_y', getattr(mob, 'cy', 0.0)))
            if any(
                box is not None and target_in_box(box, mob)
                for box in (
                    skill_rear_box(skill, (spx, spy), facing, frame_w)
                    for skill in eligible_skills(self.config, mob_id_of(mob))
                )
            ):
                candidates.append((abs(mx - spx) + abs(my - spy) * 0.5, mob))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _evaluate_skirmish_targets(
        self,
        spx: int,
        spy: int,
        facing: str,
        monsters: List[Any],
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """按前向、后向顺序返回攻击盒外侧游击带内的最近目标。"""
        extra = int(getattr(self.detector, "skirmish_range_x", 0))
        if not monsters or extra <= 0:
            return None, None
        frame_w = int(getattr(self.detector, "last_frame_width", 1920))

        def nearest(which: int) -> Optional[Any]:
            candidates = []
            for mob in monsters:
                if getattr(mob, 'is_ghost', False) or getattr(mob, 'is_dead', False):
                    continue
                mx = float(getattr(mob, 'center_x', getattr(mob, 'cx', 0.0)))
                my = float(getattr(mob, 'center_y', getattr(mob, 'cy', 0.0)))
                in_band = any(
                    target_in_box(
                        skill_skirmish_boxes(skill, (spx, spy), facing, frame_w, extra)[which], mob
                    )
                    for skill in eligible_skills(self.config, mob_id_of(mob))
                )
                if in_band:
                    candidates.append((abs(mx - spx) + abs(my - spy) * 0.5, mob))
            return min(candidates, key=lambda item: item[0])[1] if candidates else None

        return nearest(0), nearest(1)

    def _prune_skirmish_boundary_blocks(
        self, active_keys: Optional[set] = None
    ) -> None:
        """清理已不再需要的游击边界锁存。

        同平台的 ARRIVED/DWELLING 可能在短时间内往返，不能因一帧
        阶段切换就重新放行不可达的追击。只有换了平台，或该侧目标
        持续消失 0.5s，才解除。
        """
        if not self._skirmish_boundary_blocks:
            return
        now = time.perf_counter()
        active = active_keys or set()
        try:
            current = self.get_current_platform() if self.get_current_platform else None
            current_id = int(current.id) if current is not None else None
        except Exception:
            current_id = None
        for key, last_seen in list(self._skirmish_boundary_blocks.items()):
            if current_id is not None and key[0] != current_id:
                self._skirmish_boundary_blocks.pop(key, None)
            elif key not in active and now - float(last_seen) >= 0.50:
                self._skirmish_boundary_blocks.pop(key, None)

    def _do_skirmish_approach(
        self,
        target: Any,
        direction: str,
        spx: int,
        world_position: Optional[Tuple[float, float]],
        *,
        group_skill: Optional[Dict[str, Any]] = None,
        completion_checker: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """在当前长平台安全边界内连续接近，进入红框后由攻击优先中断。"""
        if direction not in ("left", "right") or world_position is None:
            return False
        try:
            platform = self.get_current_platform() if self.get_current_platform else None
        except Exception:
            platform = None
        if platform is None:
            self.log_fn("⛔ [游击取消] 当前承重平台未知，不执行主动走位")
            return False

        guard_key = (int(platform.id), direction)
        now = time.perf_counter()
        if guard_key in self._skirmish_boundary_blocks:
            # 仍能进入此函数说明同侧游击带仍有目标。只更新“最后
            # 看到”时间，不发任何键盘事件，也不重复打日志。
            self._skirmish_boundary_blocks[guard_key] = now
            return False

        current_x = float(world_position[0])
        tx = float(getattr(target, 'center_x', getattr(target, 'cx', spx)))
        frame_w = int(getattr(self.detector, "last_frame_width", 1920))
        screen_scale = max(0.4, float(frame_w) / 1920.0)
        eligible = [group_skill] if group_skill is not None else eligible_skills(
            self.config, mob_id_of(target)
        )
        if group_skill is not None:
            player_screen = getattr(self.detector, "last_player_pos", None) or (spx, 0)
            screen_facing = getattr(self.detector, "current_facing", direction)
            box = skill_attack_box(
                group_skill, (spx, float(player_screen[1])),
                screen_facing, frame_w,
            )
            edge = box[0] + box[2] if direction == "right" else box[0]
            screen_gap = tx - edge if direction == "right" else edge - tx
        else:
            reach_screen = max(
                (float(skill["reach_x"]) for skill in eligible),
                default=float(self.attack_reach_x),
            ) * screen_scale
            screen_gap = (
                tx - (float(spx) + reach_screen)
                if direction == "right"
                else (float(spx) - reach_screen) - tx
            )
        required_world_move = max(6.0, screen_gap / screen_scale + 8.0)
        sign = 1.0 if direction == "right" else -1.0

        try:
            safe_margin = float(
                self.config.get("skirmish_platform_guard_px", 50.0)
            )
        except (TypeError, ValueError):
            safe_margin = 50.0
        safe_margin = max(1.0, min(300.0, safe_margin))
        if float(platform.length) <= safe_margin * 2.0:
            self._skirmish_boundary_blocks[guard_key] = now
            self.log_fn(
                f"⛔ [游击边界保护] P{platform.id} 宽度={platform.length:.1f}px，"
                f"不足以在两侧各保留{safe_margin:.1f}px；已锁存{direction}侧，"
                "目标消失或换平台前不再发键"
            )
            return False
        safe_min = float(platform.x_min) + safe_margin
        safe_max = float(platform.x_max) - safe_margin
        desired_x = current_x + sign * required_world_move
        target_x = max(safe_min, min(safe_max, desired_x))
        available_move = (target_x - current_x) * sign
        if available_move <= 3.0:
            self._skirmish_boundary_blocks[guard_key] = now
            self.log_fn(
                f"⛔ [游击边界保护] P{platform.id} 当前X={current_x:.1f}，"
                f"{direction}侧安全边界={target_x:.1f}，已锁存该侧；"
                "目标消失或换平台前不再发键"
            )
            return False

        clipped = abs(target_x - desired_x) > 0.5
        label = f"群攻聚怪:{group_skill['name']}" if group_skill is not None else "平台游击前出"
        self.log_fn(
            f"🏹 [{label}] P{platform.id} 向{direction}接近，"
            f"怪物屏幕X={tx:.1f}，需移动≈{required_world_move:.1f}px，"
            f"目标世界X={target_x:.1f}，安全区=[{safe_min:.1f},{safe_max:.1f}]，"
            f"边界裁剪={clipped}"
        )
        getter = self.get_player_world_pos or (lambda: world_position)
        timeout = max(1.0, min(5.0, available_move / 90.0 + 1.0))
        previous_checker = getattr(self.motion, "priority_interrupt_checker", None)
        try:
            if group_skill is not None:
                # 单只目标已在红框内；此处只在凑齐两只或目标消失时停下。
                self.motion.priority_interrupt_checker = None
            arrived = self.motion.walk_to_x(
                target_x=target_x,
                get_player_pos=getter,
                tolerance=8,
                timeout_sec=timeout,
                stop_event=self.stop_event,
                platform_bounds=(float(platform.x_min), float(platform.x_max)),
                safe_margin=int(safe_margin),
                speed_scale=1.0,
                completion_checker=completion_checker,
                prefer_teleport=group_skill is not None,
            )
        finally:
            self.motion.priority_interrupt_checker = previous_checker
        self.motion.stop()
        if getattr(self.motion, "last_walk_priority_interrupted", False):
            self.log_fn("🎯 [游击到射程] 目标已进入红框，停止前出并交还攻击")
            return True
        if clipped and not arrived:
            self._skirmish_boundary_blocks[guard_key] = time.perf_counter()
            self.log_fn(
                "⛔ [游击边界保护] 已在平台安全边界前停止并锁存该侧；"
                "目标消失或换平台前不再发键"
            )
        if (group_skill is not None and arrived and
                not self._group_attack_ready_or_lost(group_skill, direction)):
            self._skirmish_boundary_blocks[guard_key] = time.perf_counter()
            self.log_fn(
                f"⛔ [群攻聚怪受限] P{platform.id} 已到安全目标但仍不足两只，"
                "停止重复走位，回退为当前可用攻击"
            )
        if group_skill is not None and not arrived and guard_key not in self._skirmish_boundary_blocks:
            latest = getter()
            if latest is not None and abs(float(latest[0]) - current_x) < 6.0:
                self._skirmish_boundary_blocks[guard_key] = time.perf_counter()
                self.log_fn(
                    f"⛔ [群攻聚怪无进展] P{platform.id} 向{direction}走位未移动，"
                    "停止重复尝试，回退为当前可用攻击"
                )
        return True if group_skill is not None else bool(arrived)

    def _group_approach_candidate(
        self, monsters: List[Any], spx: int, spy: int, facing: str,
    ) -> Optional[Tuple[Dict[str, Any], Any, str]]:
        """一只在群攻框内、同侧游击带另有合格目标时选择聚怪方向。"""
        extra = int(getattr(self.detector, "skirmish_range_x", 0))
        if extra <= 0:
            return None
        frame_w = int(getattr(self.detector, "last_frame_width", 1920))
        alive = [mob for mob in monsters if not getattr(mob, "is_ghost", False)
                 and not getattr(mob, "is_dead", False)]
        candidates = []
        skills = skills_from_config(self.config)
        for skill in skills:
            if not skill["area"]:
                continue
            allowed = [mob for mob in alive
                       if skill in eligible_skills(self.config, mob_id_of(mob), skills)]
            box = skill_attack_box(skill, (spx, spy), facing, frame_w)
            inside = [mob for mob in allowed if target_in_box(box, mob)]
            if len(inside) != 1:
                continue
            direct_x = float(getattr(inside[0], "center_x", getattr(inside[0], "cx", spx)))
            direction = "right" if direct_x > spx else "left" if direct_x < spx else None
            if direction is None or (not skill["two_way"] and direction != facing):
                continue
            bands = skill_skirmish_boxes(skill, (spx, spy), facing, frame_w, extra)
            band = bands[0] if direction == facing else bands[1]
            outside = [mob for mob in allowed if mob is not inside[0]
                       and not target_in_box(box, mob) and target_in_box(band, mob)]
            for mob in outside:
                x = float(getattr(mob, "center_x", getattr(mob, "cx", spx)))
                edge = box[0] + box[2] if direction == "right" else box[0]
                candidates.append((abs(x - edge), skill, mob, direction))
        if not candidates:
            return None
        _, skill, mob, direction = min(candidates, key=lambda item: item[0])
        return skill, mob, direction

    def _group_attack_ready_or_lost(self, skill: Dict[str, Any], direction: str) -> bool:
        if not self.is_running or self.stop_event.is_set():
            return True
        if self._rest_deadline_due():
            return True
        player = getattr(self.detector, "last_player_pos", None)
        if player is None:
            return True
        facing = getattr(self.detector, "current_facing", direction)
        if not skill["two_way"] and facing != direction:
            return True
        box = skill_attack_box(skill, player, facing,
                               int(getattr(self.detector, "last_frame_width", 1920)))
        skills = skills_from_config(self.config)
        eligible = [mob for mob in self._fresh_monster_tracks()
                    if not getattr(mob, "is_ghost", False)
                    and not getattr(mob, "is_dead", False)
                    and skill in eligible_skills(self.config, mob_id_of(mob), skills)]
        hits = sum(target_in_box(box, mob) for mob in eligible)
        return hits >= 2 or hits == 0

    @staticmethod
    def _template_observation_timestamp(target: Any) -> Optional[float]:
        """返回模板真正命中的单调时钟；光流和缓存age都不能替代它。"""
        try:
            timestamp = float(getattr(target, "last_update_ts"))
        except (AttributeError, TypeError, ValueError):
            return None
        return timestamp if math.isfinite(timestamp) else None

    def _track_real_age(self, target: Any, now: Optional[float] = None) -> float:
        """按当前时钟计算命中年龄，避免检测线程卡顿时缓存age冻结。"""
        return monster_observation_age(target, now)

    def _fresh_monster_tracks(
        self, max_age: Optional[float] = None
    ) -> List[Any]:
        """获取满足连续漏检次数与真实命中硬上限的轨迹快照。"""
        tracked = getattr(
            self.detector,
            "tracked_monsters",
            getattr(self.detector, "active_tracks", []),
        ) or []
        now = time.perf_counter()
        return [
            mob for mob in list(tracked)
            if is_fresh_attack_observation(mob, now, max_age)
        ]

    def _log_attack_decision(
        self,
        source: str,
        target: Any,
        spx: int,
        spy: int,
        facing: str,
        in_attack_box: bool,
        candidate_count: Optional[int] = None,
    ) -> None:
        """记录导致中断/攻击的完整空间判定，供误识别复盘。"""
        now = time.perf_counter()
        min_interval = 0.25 if source == "walk_interrupt" else 0.15
        if now - self._last_attack_debug_log_at.get(source, 0.0) < min_interval:
            return
        self._last_attack_debug_log_at[source] = now

        tx = float(getattr(target, "center_x", getattr(target, "cx", 0.0)))
        ty = float(getattr(target, "center_y", getattr(target, "cy", 0.0)))
        dx = tx - float(spx)
        dy = ty - float(spy)
        bbox = getattr(target, "bbox", None)
        track_id = getattr(target, "track_id", getattr(target, "id", "?"))
        name = getattr(target, "name", getattr(target, "template_name", "?"))
        score = getattr(target, "score", getattr(target, "confidence", None))
        age = self._track_real_age(target, now)
        hits = getattr(target, "hits", None)
        full_scan_misses = getattr(target, "consecutive_full_scan_misses", None)
        hard_timeout_sec = getattr(
            target, "attack_observation_hard_timeout_sec", None
        )
        player_bbox = getattr(self.detector, "last_player_bbox", None)

        world_pos = None
        raw_world_pos = None
        current_platform = None
        try:
            world_pos = self.get_player_world_pos() if self.get_player_world_pos else None
        except Exception:
            pass
        try:
            raw_world_pos = (
                self.get_player_raw_world_pos()
                if self.get_player_raw_world_pos else None
            )
        except Exception:
            pass
        try:
            current_platform = self.get_current_platform() if self.get_current_platform else None
        except Exception:
            pass
        platform_id = getattr(current_platform, "id", None)

        actual_box = None
        calculate_box = getattr(self.detector, "calculate_attack_box", None)
        if callable(calculate_box):
            try:
                actual_box = calculate_box(
                    (spx, spy), facing,
                    frame_w=int(getattr(self.detector, "last_frame_width", 1920)),
                )
            except Exception:
                actual_box = None
        if actual_box is not None:
            ax, ay, aw, ah = actual_box
            box_x = (float(ax - spx), float(ax + aw - spx))
            box_y = (float(ay - spy), float(ay + ah - spy))
        else:
            if self.attack_two_way:
                box_x = (-float(self.attack_reach_x), float(self.attack_reach_x))
            elif facing == "right":
                box_x = (-float(self.behind_reach_x), float(self.attack_reach_x))
            else:
                box_x = (-float(self.attack_reach_x), float(self.behind_reach_x))
            box_y = (-float(self.attack_reach_y_up), float(self.attack_reach_y_down))

        def fmt_number(value: Any, digits: int = 2) -> str:
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return "?"

        count_text = "?" if candidate_count is None else str(candidate_count)
        self.log_fn(
            f"🎯 [攻击判定详情] source={source}，候选数={count_text}，"
            f"人物屏幕=({spx},{spy}) bbox={player_bbox}，"
            f"人物世界={world_pos} raw={raw_world_pos} P{platform_id}，朝向={facing}；"
            f"怪物=id:{track_id} name:{name} center=({tx:.1f},{ty:.1f}) "
            f"bbox={bbox} score={fmt_number(score, 3)} hits={hits} "
            f"age={fmt_number(age, 3)}s misses={full_scan_misses} "
            f"hard={fmt_number(hard_timeout_sec, 3)}s；"
            f"相对=(dx:{dx:.1f},dy:{dy:.1f})，"
            f"攻击框dx=[{box_x[0]:.0f},{box_x[1]:.0f}] "
            f"dy=[{box_y[0]:.0f},{box_y[1]:.0f}]，inBox={bool(in_attack_box)}"
        )

    def _ensure_safe_attack_position(self, facing: str) -> bool:
        """直接攻击前先从平台边缘内撤；True 表示本轮交还视觉重判。"""
        if self.attack_only_mode or self.get_player_world_pos is None or self.get_current_platform is None:
            return False
        if self.get_is_climbing is not None:
            try:
                if self.get_is_climbing():
                    return False  # 绳梯上的攻击由原有绳梯抑制处理。
            except Exception:
                return False
        try:
            platform = self.get_current_platform()
            position = self.get_player_world_pos()
            if platform is None or position is None:
                return False
            x = float(position[0])
            left = float(platform.x_min)
            right = float(platform.x_max)
            platform_id = int(platform.id)
            width = right - left
            configured = float(self.config.get("attack_edge_guard_px", 50.0))
        except (AttributeError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in (x, left, right, configured)) or width < 30.0:
            # 极窄平台没有可靠的内撤空间；禁止在此发盲走输入。
            return False
        margin = min(max(1.0, configured), width * 0.25)
        safe_left, safe_right = left + margin, right - margin
        if safe_left <= x <= safe_right:
            self._attack_edge_retry_key = None
            self._attack_edge_retry_after = 0.0
            return False
        if not left - 15.0 <= x <= right + 15.0:
            # 承重平台与世界坐标相互矛盾时不能向未知方向盲走。
            return False
        side = "left" if x < safe_left else "right"
        key = (platform_id, side)
        self.state = BotState.PATROLLING
        if self.platform_patrol.phase in (
            PatrolPhase.EXECUTE, PatrolPhase.VERIFY, PatrolPhase.OBSERVATION_GRACE,
        ):
            return True  # 尚未确认落台，不插入内撤，也不在边缘攻击。
        now = time.perf_counter()
        if self._attack_edge_retry_key == key and now < self._attack_edge_retry_after:
            return True
        self._attack_edge_retry_key = key
        center = (left + right) * 0.5
        tolerance = max(3.0, min(8.0, width * 0.10))
        inward_pad = max(tolerance + 6.0, min(20.0, width * 0.10))
        target_x = (
            min(center, safe_left + inward_pad)
            if side == "left" else max(center, safe_right - inward_pad)
        )
        self.log_fn(
            f"🛡️ [攻击前边缘保护] P{platform_id} 当前X={x:.1f}，"
            f"安全区=[{safe_left:.1f},{safe_right:.1f}]，先内撤到X={target_x:.1f}；"
            "此动作不是游击追怪"
        )
        previous_checker = getattr(self.motion, "priority_interrupt_checker", None)
        arrived = False
        try:
            self.motion.stop()
            # 红框本身不能中断内撤，否则下一帧又发起同一段内撤。
            # 休息截止及 F6 stop_event 仍保持最高优先级。
            self.motion.priority_interrupt_checker = (
                lambda: "休息时间已到" if self._rest_deadline_due() else False
            )
            arrived = self.motion.walk_to_x(
                target_x=target_x,
                get_player_pos=self.get_player_world_pos,
                tolerance=int(tolerance),
                timeout_sec=max(1.2, min(3.0, abs(target_x - x) / 100.0 + 1.0)),
                stop_event=self.stop_event,
                platform_bounds=(left, right),
                safe_margin=max(4, min(12, int(margin * 0.5))),
                speed_scale=1.0,
            )
        except Exception as exc:
            self.log_fn(f"⚠️ [攻击前边缘保护] 内撤异常：{type(exc).__name__}: {exc}")
        finally:
            self.motion.stop()
            self.motion.priority_interrupt_checker = previous_checker
        try:
            latest = self.get_player_world_pos()
            latest_platform = self.get_current_platform()
            safe_arrival = bool(
                arrived and latest is not None and latest_platform is not None
                and int(latest_platform.id) == platform_id
                and safe_left <= float(latest[0]) <= safe_right
            )
        except (AttributeError, TypeError, ValueError):
            safe_arrival = False
        if safe_arrival:
            # 内撤后留极短观测收束窗；若下一帧仍报旧的边缘坐标，
            # 不立即重复发方向键。读到真实安全坐标会在上方直接放行。
            self._attack_edge_retry_key = key
            self._attack_edge_retry_after = time.perf_counter() + 0.20
            # 内撤改变了朝向；先恢复原攻击方向，再让下一帧重新识别目标。
            if facing in ("left", "right") and not self.stop_event.is_set():
                try:
                    self.motion.face_direction(facing, current_facing=None, duration_ms=30)
                except Exception as exc:
                    self.log_fn(f"⚠️ [攻击前边缘保护] 恢复朝向失败：{exc}")
            self.log_fn(f"✅ [攻击前边缘保护] P{platform_id} 已退入安全区，重新确认怪物后攻击")
        else:
            self._attack_edge_retry_after = time.perf_counter() + 0.8
            self.log_fn(f"⏸️ [攻击前边缘保护] P{platform_id} 内撤未确认，暂停本轮攻击并限频重试")
        return True

    def _do_attack(self, target: Any, spx: int, spy: int, facing: str):
        """执行攻击技能输出（与仅攻击键介入模式完全一致的攻击节奏、按键与冷却控制）。"""
        if not self.is_running or self.stop_event.is_set():
            return

        now = time.perf_counter()
        # 1. 攻击冷却检查（与仅攻击键介入模式一致，防止按键洪水在客户端队列堆积）
        if now < getattr(self, "_normal_attack_next_at", 0.0):
            return

        if self._ensure_safe_attack_position(facing):
            return

        # 2. 绳梯状态检测：如果在绳梯上，严禁执行攻击键
        is_climbing = False
        if self.get_is_climbing is not None:
            try:
                is_climbing = bool(self.get_is_climbing())
            except Exception:
                is_climbing = False

        tid = getattr(target, "track_id", getattr(target, "id", "?"))
        tname = getattr(target, "name", "怪物")

        if is_climbing:
            if now - getattr(self, "_normal_attack_last_climb_log_t", 0.0) >= 1.0:
                self.log_fn(
                    f"🪜 [绳梯抑制] 目标=[{tname}]#{tid} 在攻击范围内，"
                    f"但角色处于绳梯状态，跳过攻击按键"
                )
                self._normal_attack_last_climb_log_t = now
            return

        self.state = BotState.ATTACKING
        self.motion.stop()
        skill = self._skill_for_target(target, spx, spy, facing, self._fresh_monster_tracks())
        if skill is None:
            return

        # 4. 执行 1~2 次攻击按键（完全对齐仅攻击键介入模式）
        rounds = random.randint(1, 2)
        pressed_any = False
        for press_idx in range(rounds):
            if not self.is_running or self.stop_event.is_set():
                break
            live_facing_for_guard = getattr(self.detector, "current_facing", facing)
            if press_idx > 0 and self._ensure_safe_attack_position(live_facing_for_guard):
                break

            if self.get_enable_monster_detection is not None:
                try:
                    if not bool(self.get_enable_monster_detection()):
                        self.motion.stop()
                        self.state = BotState.PATROLLING
                        break
                except Exception:
                    pass

            # 绳梯状态复核（若连击间隙角色攀上了绳梯，立即中断后续攻击）
            if self.get_is_climbing is not None:
                try:
                    if bool(self.get_is_climbing()):
                        self.log_fn("🪜 [绳梯抑制] 检测到角色进入绳梯，立即中断后续攻击")
                        break
                except Exception:
                    pass

            # Every press uses the current player position, actual detector
            # facing and fresh target list. The previous burst could have
            # switched target/skill, or knockback could have moved the player.
            live_pos = getattr(self.detector, "last_player_pos", None) or (spx, spy)
            live_px, live_py = live_pos
            live_facing = getattr(self.detector, "current_facing", facing)
            current_tracks = self._fresh_monster_tracks()
            live_target, live_in_box = self._evaluate_best_target(
                live_px, live_py, live_facing, current_tracks
            )
            if live_target is None or not live_in_box:
                break
            target = live_target
            skill = self._skill_for_target(target, live_px, live_py,
                                           live_facing, current_tracks)
            if skill is None:
                break
            box = skill_attack_box(skill, (live_px, live_py), live_facing,
                                   int(getattr(self.detector, "last_frame_width", 1920)))
            if not target_in_box(box, target):
                break
            tx = float(getattr(target, "center_x", getattr(target, "cx", 0)))
            ty = float(getattr(target, "center_y", getattr(target, "cy", 0)))
            target_side = "right" if tx > live_px else "left"
            tid = getattr(target, "track_id", getattr(target, "id", "?"))
            tname = getattr(target, "name", "怪物")
            atk_key, atk_vk = skill["key"], skill["vk"]
            if not skill["two_way"] and target_side != live_facing:
                self.log_fn(
                    f"🛑 [单向攻击朝向保护] 目标#{tid} 在{target_side}，"
                    f"视觉朝向={live_facing}，本次禁止按 {atk_key.upper()}"
                )
                self.motion.face_direction(target_side, current_facing=None,
                                           duration_ms=80)
                self.detector.facing_lock_until = 0.0
                self._normal_attack_next_at = time.perf_counter() + 0.18
                break

            if not skill["area"]:
                guard_action = self._attack_stall_guard.before_attack(
                    str(mob_id_of(target) or tname),
                    (float(live_px), float(live_py)), (tx, ty), time.perf_counter()
                )
                if guard_action == "reorient":
                    focus = self._attack_stall_guard.focus
                    streak = (f"累计{focus.presses}次/"
                              f"{time.perf_counter() - focus.first_at:.1f}s") if focus else ""
                    self.log_fn(
                        f"🧭 [持续攻击脱困] 同位置目标#{tid} {streak}，"
                        f"强制朝{target_side}转向并重新观察；玩家=({live_px},{live_py}) "
                        f"怪物=({tx:.0f},{ty:.0f})"
                    )
                    self.motion.face_direction(target_side, current_facing=None,
                                               duration_ms=90)
                    self.detector.facing_lock_until = 0.0
                    self._normal_attack_next_at = time.perf_counter() + 0.20
                    break
                if guard_action == "suppress":
                    focus = self._attack_stall_guard.blocked[-1][0] if self._attack_stall_guard.blocked else None
                    self.log_fn(
                        f"⏸️ [持续攻击脱困] 目标#{tid} 长时间仍可见，"
                        f"累计{focus.presses if focus else '?'}次/"
                        f"{time.perf_counter() - focus.first_at:.1f}s；"
                        "暂时让行巡逻3秒；没有怪物血量数据，不能据此断言未造成伤害"
                    )
                    self.state = BotState.PATROLLING
                    break

            # Only lock the *observed* direction for the animation window.
            # A 0.5s intent lock used to bridge successive casts indefinitely,
            # hiding a failed turn or knockback from fresh visual evidence.
            lock_facing = getattr(self.detector, "lock_facing_during_attack", None)
            if callable(lock_facing):
                lock_facing(live_facing, duration_sec=0.18)

            if press_idx == 0:
                now_log = time.perf_counter()
                if now_log - self._last_attack_output_latency_log_at >= 0.50:
                    self._last_attack_output_latency_log_at = now_log
                    c_tracks = self._fresh_monster_tracks()
                    self.log_fn(
                        "⚡ [攻击输出响应] "
                        f"track={tid}，"
                        f"真实模板命中至Ctrl={self._track_real_age(target, now_log) * 1000.0:.1f}ms，"
                        f"候选={len(c_tracks)}，phase={self.platform_patrol.phase.value}"
                    )

            press_ms = random.randint(55, 75)
            t_press_start = time.perf_counter()
            if (not self.is_running or self.stop_event.is_set()
                    or self._attack_paused.is_set()):
                break
            with self._attack_input_lock:
                if self._attack_paused.is_set() or not self.is_running or self.stop_event.is_set():
                    break
                try:
                    self.driver.key_down(atk_key, vk_code=atk_vk)
                    self.log_fn(
                        f"⚔️ [攻击执行] 按下攻击键: 技能={skill['name']} key='{atk_key}' (vk=0x{atk_vk:02X}), "
                        f"计划时长={press_ms}ms, 连段={press_idx + 1}/{rounds}, 目标=[{tname}]#{tid}；"
                        f"玩家=({live_px},{live_py}) 朝向={live_facing} "
                        f"怪物=({tx:.0f},{ty:.0f}) dx={tx-live_px:.0f} "
                        f"攻击盒={box} 技能ID={skill['id']}"
                    )
                    time.sleep(press_ms / 1000.0)
                finally:
                    self.driver.key_up(atk_key, vk_code=atk_vk)
            pressed_any = True
            actual_press_ms = (time.perf_counter() - t_press_start) * 1000.0

            cd_ms = random.randint(60, 90) if press_idx < rounds - 1 else random.randint(150, 220)
            self.log_fn(
                f"⚔️ [攻击执行] 释放攻击键: key='{atk_key}', "
                f"实际按压={actual_press_ms:.1f}ms, 后摇CD={cd_ms}ms"
            )
            time.sleep(cd_ms / 1000.0)

        if pressed_any:
            self._has_attacked_monster = True
            self._forward_clear_since = None
            self.platform_patrol.notify_attack_during_dwell()
            self._normal_attack_next_at = time.perf_counter() + 0.05

    def _clear_route_attack_hold(self) -> None:
        pass

    def _do_approach(self, target: Any, spx: int, spy: int):
        """走位逼近目标怪物 (视口微调走位)"""
        self.state = BotState.APPROACHING
        tx = getattr(target, 'center_x', getattr(target, 'cx', 0))
        walk_dir = "right" if tx > spx else "left"
        
        # 释放一次短暂的逼近脉冲按键 (150ms)，然后重新感知
        self.driver.press_key(walk_dir, duration_ms=random.randint(120, 180))
        time.sleep(0.05)

    def _has_configured_platform_patrol(self) -> bool:
        if self.get_patrol_platforms is None:
            return False
        try:
            return bool(self.get_patrol_platforms())
        except Exception:
            return False

    def _dispatch_combat_or_patrol(
        self,
        best_target: Optional[Any],
        in_attack_box: bool,
        spx: int,
        spy: int,
        facing: str,
        world_position: Optional[Tuple[float, float]],
        monsters: Optional[List[Any]] = None,
    ) -> None:
        """平台巡逻只打射程内目标；射程外候选不得抢方向键追逐。"""
        if best_target is not None and in_attack_box:
            if self._ensure_safe_attack_position(facing):
                return
            # 过路平台的稳定地面阶段也允许聚怪；起跳/落台验证中绝不
            # 强插横移，以免改写当前拓扑边的起跳位置。
            if (
                world_position is not None
                and self.platform_patrol.phase in (
                    PatrolPhase.OBSERVE, PatrolPhase.PLAN,
                    PatrolPhase.ARRIVED, PatrolPhase.RECOVER,
                )
            ):
                group = self._group_approach_candidate(
                    monsters or [], spx, spy, facing
                )
                if group is not None:
                    skill, outer_target, direction = group
                    if self._do_skirmish_approach(
                        outer_target, direction, spx, world_position,
                        group_skill=skill,
                        completion_checker=lambda: self._group_attack_ready_or_lost(
                            skill, direction
                        ),
                    ):
                        return
            self._do_attack(best_target, spx, spy, facing)
            return

        if best_target is not None and not self._has_configured_platform_patrol():
            self._do_approach(best_target, spx, spy)
            return

        if best_target is not None:
            now = time.perf_counter()
            if now - self._last_ignored_patrol_target_log_at >= 1.0:
                self._last_ignored_patrol_target_log_at = now
                tx = getattr(best_target, "center_x", getattr(best_target, "cx", None))
                ty = getattr(best_target, "center_y", getattr(best_target, "cy", None))
                self.log_fn(
                    f"👁️ [巡逻忽略范围外目标] target=({tx},{ty})，"
                    "不追怪，继续平台路线"
                )

        # 平台导航只接受世界坐标。传送点光标可能暂时遮住
        # 小地图黄点，此时主视口像素绝不能冒充世界坐标。
        if world_position is None:
            self._do_patrol(None, None)
        else:
            self._do_patrol(world_position[0], world_position[1])

    def _rest_deadline_due(self) -> bool:
        """休息计时不能等长平台的阻塞走位走到旧目标才检查。"""
        if not self.is_running or self.stop_event.is_set() or self._rest_phase != "idle":
            return False
        if self.get_global_rest_due is not None:
            try:
                if self.get_global_rest_due():
                    return True
            except Exception:
                pass
        return bool(
            self._rest_settings is not None
            and self._forced_rest_settings is None
            and self._rest_interval_sec > 0.0
            and self._rest_normal_elapsed
            + max(0.0, time.perf_counter() - self._rest_last_tick_at)
            >= self._rest_interval_sec
        )

    def _attack_priority_pending(self) -> bool | str:
        """阻塞地面走位优先响应休息截止，再检查有效攻击目标。"""
        if not self.is_running or self.stop_event.is_set():
            return False
        if self._rest_deadline_due():
            return "休息时间已到"
        if self.get_enable_monster_detection is not None:
            try:
                if not bool(self.get_enable_monster_detection()):
                    return False
            except Exception:
                return False
        player = getattr(self.detector, "last_player_pos", None)
        if player is None:
            return False
        fresh = self._fresh_monster_tracks()
        if not fresh:
            return False
        facing = getattr(self.detector, "current_facing", "right")
        target, in_attack_box = self._evaluate_best_target(
            int(player[0]), int(player[1]), facing, fresh
        )
        if target is not None and in_attack_box:
            self._log_attack_decision(
                "walk_interrupt", target, int(player[0]), int(player[1]),
                facing, True, len(fresh),
            )
        return bool(in_attack_box)

    def _current_action_failure_count(self, edge: Any) -> int:
        """读取独立动作重试序号，不与路线失败惩罚共用开关。"""
        try:
            return int(self.platform_patrol.action_failure_count(edge))
        except Exception:
            return 0

    @staticmethod
    def _reverse_runup_preparation(
        target_x: float,
        current_x: float,
        *,
        right_jump: bool,
        gate_width: float,
        measurement_step: float,
        tolerance: float,
        source_bounds: Tuple[float, float],
        safe_margin: float,
    ) -> Tuple[float, bool]:
        """窄起跳窗前需反向准备时，多留一个实际小地图量化格建立助跑。"""
        travel_sign = 1.0 if right_jump else -1.0
        step = max(1.0, float(measurement_step))
        reverse_approach = (
            gate_width < step
            and (float(current_x) - float(target_x)) * travel_sign > float(tolerance)
        )
        if not reverse_approach:
            return float(target_x), False
        source_lo, source_hi = sorted(map(float, source_bounds))
        margin = min(max(8.0, float(safe_margin)), max(0.0, (source_hi - source_lo) * 0.25))
        safe_lo, safe_hi = source_lo + margin, source_hi - margin
        adjusted = max(safe_lo, min(safe_hi, float(target_x) - travel_sign * step))
        return adjusted, True

    def _stage_failed_static_grab(
        self,
        curr_node: Any,
        trigger_x: float,
        pos_getter: Callable[[], Optional[Tuple[float, float]]],
        retry_index: int,
    ) -> Optional[str]:
        """原地跳抓失败后，从可用平台空间连续行走建立新助跑侧。"""
        current = pos_getter()
        if current is None:
            return None
        platform_len = max(0.0, float(curr_node.x_max) - float(curr_node.x_min))
        edge_margin = max(8.0, min(20.0, platform_len * 0.08))
        safe_lo = float(curr_node.x_min) + edge_margin
        safe_hi = float(curr_node.x_max) - edge_margin
        if safe_lo >= safe_hi:
            return None
        try:
            measurement_step = float(
                getattr(self.motion.motion_model, "measurement_step_px", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            measurement_step = 0.0
        try:
            capture_window = float(self.motion._run_jump_capture_window())
        except Exception:
            capture_window = 18.0
        # 跑道长度随当前小地图量化比例和真实吸附窗变化；不为某张地图
        # 写死起跳距离。其作用只是让角色离开刚才失败的静止采样格，真正
        # 起跳仍由 motion_controller 的实时速度相交门逐帧决定。
        runway = max(
            12.0,
            min(48.0, max(measurement_step * 0.85, capture_window * 0.80)),
        )
        candidates = []
        right_x = min(safe_hi, float(trigger_x) + runway)
        left_x = max(safe_lo, float(trigger_x) - runway)
        if right_x - float(trigger_x) >= 9.0:
            candidates.append((right_x, "left", safe_hi - float(trigger_x)))
        if float(trigger_x) - left_x >= 9.0:
            candidates.append((left_x, "right", float(trigger_x) - safe_lo))
        if not candidates:
            return None
        # 首次选空间更宽的一侧；仍失败时轮换另一侧，防止同样的接近轨迹
        # 永久重放。只有一侧可用时继续实时重算该侧准备点。
        candidates.sort(key=lambda item: item[2], reverse=True)
        chosen = candidates[max(0, int(retry_index) - 1) % len(candidates)]
        staging_x, run_direction, available_room = chosen
        # 与普通跑跳的源平台窄台门槛保持一致。长平台的远距准备
        # 必须持续按键；只有源平台不足90px才用占空比防止踏空。
        narrow_source = platform_len < 90.0
        staging_speed_scale = 0.65 if narrow_source else 1.0
        self.log_fn(
            f"🔧 [抓梯动作级重规划] 动作变体={retry_index}，"
            f"量化步长={measurement_step:.1f}px，吸附半窗={capture_window:.1f}px；"
            f"从P{curr_node.id}内侧X={staging_x:.1f}向"
            f"{('左' if run_direction == 'left' else '右')}重新建立跑跳轨迹；"
            f"源平台宽={platform_len:.1f}px，"
            f"控制={'窄台短按' if narrow_source else '长台连续行走'}"
        )
        arrived = self.motion.walk_to_x(
            target_x=staging_x,
            get_player_pos=pos_getter,
            tolerance=max(4.0, min(8.0, runway * 0.25)),
            timeout_sec=max(
                1.2,
                min(4.0, abs(float(current[0]) - staging_x) / 75.0 + 1.0),
            ),
            stop_event=self.stop_event,
            platform_bounds=(curr_node.x_min, curr_node.x_max),
            safe_margin=edge_margin,
            speed_scale=staging_speed_scale,
        )
        self.motion.stop()
        time.sleep(0.10)
        staged = pos_getter()
        if staged is None:
            return None
        staged_distance = abs(float(staged[0]) - float(trigger_x))
        if not arrived and abs(float(staged[0]) - staging_x) > max(10.0, runway * 0.45):
            self.log_fn(
                f"⏸️ [抓梯动作重规划未到位] 当前X={float(staged[0]):.1f}，"
                f"准备点={staging_x:.1f}"
            )
            return None
        if staged_distance < 8.0:
            self.log_fn(
                f"⏸️ [抓梯动作重规划跑道不足] 当前X={float(staged[0]):.1f}，"
                f"梯轴={float(trigger_x):.1f}"
            )
            return None
        if self.reset_motion_prediction:
            self.reset_motion_prediction()
        self.log_fn(
            f"✅ [抓梯动作重规划完成] 当前X={float(staged[0]):.1f}，"
            f"可用空间={available_room:.1f}px，交给实时速度相交门"
        )
        return run_direction

    def _prepare_vertical_jump(
        self,
        curr_node,
        landing_node,
        pos_getter,
        fallback_x: float,
        transition_edge=None,
        action_retry_count: int = 0,
    ) -> bool:
        """按短 foothold 动作计划准备直跳；旧边退回目标平台范围。"""
        current_pos = pos_getter()
        current_x = float(current_pos[0]) if current_pos else float(fallback_x)
        foothold_range = getattr(transition_edge, 'takeoff_x_range', None)
        is_foothold_plan = bool(
            transition_edge is not None
            and getattr(transition_edge, 'source_foothold_id', None) is not None
            and foothold_range is not None
        )
        if is_foothold_plan:
            landing_lo = float(foothold_range[0])
            landing_hi = float(foothold_range[1])
            # 起跳范围和 motion.walk_to_x 使用的是同一套连续世界坐标。
            # raw 黄点只用于诊断量化格，不能再用 raw-shadow 偏差反推
            # 控制目标；两者一混用会让目标在安全区两侧来回翻转。
            physical_target_x = (landing_lo + landing_hi) * 0.5
            raw_pos = (
                self.get_player_raw_world_pos()
                if self.get_player_raw_world_pos is not None else None
            )
            raw_x = float(raw_pos[0]) if raw_pos is not None else None
            coordinate_bias = (
                raw_x - current_x if raw_x is not None else 0.0
            )
            trigger_x = physical_target_x
            trigger_x = max(
                float(curr_node.x_min) + 6.0,
                min(float(curr_node.x_max) - 6.0, trigger_x),
            )
            # trigger_x_range 在拓扑构建时已经从真实重叠区两端各扣除了
            # 6~12px。这里不能再次缩窄：本图 raw X 每次跨约16~20px，
            # 二次缩边后的区间里可能不存在任何可观测坐标，日志中的
            # -750/-730 来回振荡正由此产生。
            physical_lo = landing_lo
            physical_hi = landing_hi
            observed_x = current_x

            def takeoff_range_is_covered(
                shadow_x: float,
                candidate_raw_x: Optional[float],
            ) -> bool:
                """Accept a narrow gate when either coordinate sensor covers it.

                The shadow estimator remains the primary control coordinate.  On
                very narrow foothold gates it can settle one pixel outside while
                the same frame's unfiltered physical coordinate is already inside
                (for example shadow=625, raw=619, gate=[616, 624]).  Re-driving in
                that state creates an endless left/right alignment loop.  Raw X is
                therefore only a positive in-range witness; it never changes the
                target and cannot widen the topology gate.
                """
                if physical_lo <= float(shadow_x) <= physical_hi:
                    return True
                return bool(
                    candidate_raw_x is not None
                    and physical_lo <= float(candidate_raw_x) <= physical_hi
                )

            def source_is_still_standing() -> bool:
                """Reject a vertical jump after alignment has already fallen through."""
                raw = (
                    self.get_player_raw_world_pos()
                    if self.get_player_raw_world_pos is not None else None
                )
                if raw is None:
                    return True
                source_x, source_y = float(raw[0]), float(raw[1])
                if not (float(curr_node.x_min) <= source_x <= float(curr_node.x_max)):
                    self._last_vertical_source_lost = True
                    self.log_fn(
                        f"↩️ [垂直起跳源平台丢失] P{curr_node.id} 对齐后rawX="
                        f"{source_x:.1f} 已离开平台范围"
                        f"[{curr_node.x_min},{curr_node.x_max}]；按实况重规划"
                    )
                    return False
                expected_y = float(curr_node.surface_y_at(source_x)) - 45.0
                try:
                    tolerance_y = float(
                        self.config.get("jump_source_raw_y_tolerance_px", 65.0)
                    )
                except (TypeError, ValueError):
                    tolerance_y = 65.0
                standing = abs(source_y - expected_y) <= tolerance_y
                if not standing:
                    self._last_vertical_source_lost = True
                    self.log_fn(
                        f"↩️ [垂直起跳源平台丢失] P{curr_node.id} 对齐后raw="
                        f"({source_x:.1f},{source_y:.1f})，源平台站立Y={expected_y:.1f}"
                        f"±{tolerance_y:.0f}px；本轮不发Alt并按实况重规划"
                    )
                return standing
            # 短段计划的范围本身已经扣除了端点安全边距，必须按这个
            # 局部范围准备，不能重新放宽为整个合并目标平台。
            is_narrow = True
            if takeoff_range_is_covered(observed_x, raw_x):
                if not source_is_still_standing():
                    return False
                coordinate_label = (
                    "影子+raw"
                    if physical_lo <= observed_x <= physical_hi
                    else "raw兼容证据"
                )
                self.log_fn(
                    f"✅ [短段起跳区已覆盖] fh{transition_edge.source_foothold_id} -> "
                    f"fh{transition_edge.target_foothold_id}，影子X={observed_x:.1f}，"
                    f"rawX={raw_x if raw_x is not None else 'None'}，证据={coordinate_label}，"
                    f"安全区=[{physical_lo:.1f},{physical_hi:.1f}]"
                )
                return True
            if raw_x is not None:
                self.log_fn(
                    f"🎯 [短段物理坐标校准] 第{action_retry_count + 1}轮，"
                    f"影子X={current_x:.1f}，rawX={raw_x:.1f}，"
                    f"量化差={coordinate_bias:+.1f}px；安全区中点={physical_target_x:.1f}，"
                    f"控制目标={trigger_x:.1f}"
                )
            # 合并长平台上的 foothold 计划已经给出安全区，不再进入旧的
            # “窄目标平台25%短按”流程。连续行走交给运动模型预测刹停，
            # 到位后只判断是否进入安全区，不强求命中单个像素。
            plan_tolerance = max(
                5,
                min(12, int((landing_hi - landing_lo) * 0.45)),
            )
            # Never stop the walk outside the true foothold gate merely
            # because the control target is within a wider arrival tolerance.
            # P26->P31 had X=1580 vs gate [1581,1589], but 5px was accepted.
            gate_walk_tolerance = max(
                1, min(7, int(min(trigger_x - physical_lo, physical_hi - trigger_x)) - 1)
            )
            timeout = max(
                2.0,
                min(10.0, abs(trigger_x - current_x) / 105.0 + 1.5),
            )
            self.log_fn(
                f"🏃 [短段起跳连续对齐] fh{transition_edge.source_foothold_id}"
                f"->fh{transition_edge.target_foothold_id}，连续走向X={trigger_x:.1f}，"
                f"安全区=[{landing_lo:.1f},{landing_hi:.1f}]，远距保持连续输入"
            )
            latest_x = current_x
            latest_raw_x = raw_x
            # 最多两轮连续坐标闭环。只有起跳源平台本身是窄台，且第二轮
            # 已靠近起跳区，才允许低占空比收束；长源平台始终连续行走。
            for calibration_pass in range(2):
                remaining_x = abs(trigger_x - latest_x)
                fine_approach = (
                    calibration_pass > 0 and remaining_x <= 80.0
                    and float(curr_node.length) < 90.0
                )
                if calibration_pass > 0 and not fine_approach:
                    self.log_fn(
                        f"🚶 [短段校准远距续走] 距起跳目标{remaining_x:.1f}px，"
                        "保持方向键连续按住，不启用低占空比"
                    )
                arrived = self.motion.walk_to_x(
                    target_x=trigger_x,
                    get_player_pos=pos_getter,
                    tolerance=min(plan_tolerance, gate_walk_tolerance),
                    timeout_sec=timeout,
                    stop_event=self.stop_event,
                    platform_bounds=(curr_node.x_min, curr_node.x_max),
                    safe_margin=6,
                    speed_scale=0.65 if fine_approach else 1.0,
                )
                self.motion.stop()
                if bool(getattr(self.motion, "last_walk_priority_interrupted", False)):
                    # 攻击优先已接管，不能在第二轮再次走位/等待；交还 FSM
                    # 攻击后从最新实际位置重试原边。
                    self.log_fn("⚔️ [短段校准让行攻击] 停止本轮起跳点对齐")
                    return False
                time.sleep(0.20)
                settled = pos_getter()
                latest_x = float(settled[0]) if settled else latest_x
                settled_raw = (
                    self.get_player_raw_world_pos()
                    if self.get_player_raw_world_pos is not None else None
                )
                latest_raw_x = (
                    float(settled_raw[0]) if settled_raw is not None else None
                )
                if takeoff_range_is_covered(latest_x, latest_raw_x):
                    if not source_is_still_standing():
                        return False
                    coordinate_label = (
                        "影子+raw"
                        if physical_lo <= latest_x <= physical_hi
                        else "raw兼容证据"
                    )
                    self.log_fn(
                        f"✅ [短段起跳点确认] 影子X={latest_x:.1f}，"
                        f"rawX={latest_raw_x if latest_raw_x is not None else 'None'}，"
                        f"证据={coordinate_label}，"
                        f"安全区=[{physical_lo:.1f},{physical_hi:.1f}]，"
                        f"校准轮次={calibration_pass + 1}"
                    )
                    return True
                if calibration_pass >= 1:
                    break
                remaining_x = abs(trigger_x - latest_x)
                self.log_fn(
                    f"🔧 [瞬移动作级收束] 影子X={latest_x:.1f} 尚未进入"
                    f"[{physical_lo:.1f},{physical_hi:.1f}]，"
                    f"距目标{remaining_x:.1f}px；"
                    + ("近距低速收束" if remaining_x <= 80.0 else "远距保持全速连续行走")
                )
                timeout = (
                    max(1.0, min(3.0, remaining_x / 70.0 + 0.8))
                    if remaining_x <= 80.0 else
                    max(2.0, min(10.0, remaining_x / 105.0 + 1.5))
                )
            self.log_fn(
                f"⏸️ [短段对齐未到位] 影子X={latest_x:.1f}，"
                f"rawX={latest_raw_x if latest_raw_x is not None else 'None'}，"
                f"安全区=[{physical_lo:.1f},{physical_hi:.1f}]，"
                "本轮不执行跳跃/瞬移"
            )
            return False
        else:
            margin = min(12.0, max(4.0, float(landing_node.length) * 0.12))
            landing_lo = float(landing_node.x_min) + margin
            landing_hi = float(landing_node.x_max) - margin
            trigger_x = float(landing_node.center_x)
            is_narrow = float(landing_node.length) < 60.0

        # 宽平台：只要起跳 X 已位于目标的有效落地区间，立即跳。
        # 若不在范围内，只走到最近的边界，不强制去中心点，也不短按。
        if not is_narrow:
            if landing_lo <= current_x <= landing_hi:
                self.log_fn(
                    f"✅ [垂直跳落点已覆盖] 当前X={current_x:.1f} 在目标P{landing_node.id} "
                    f"有效范围=[{landing_lo:.1f},{landing_hi:.1f}]"
                )
                return True
            target_x = landing_lo if current_x < landing_lo else landing_hi
            timeout = max(2.0, min(10.0, abs(target_x - current_x) / 105.0 + 1.5))
            self.log_fn(
                f"🚶 [垂直跳范围准备] P{curr_node.id} 持续行走至目标P{landing_node.id} "
                f"有效范围=[{landing_lo:.1f},{landing_hi:.1f}] 的最近点X={target_x:.1f} "
                f"(当前X={current_x:.1f})"
            )
            arrived = self.motion.walk_to_x(
                target_x=target_x,
                get_player_pos=pos_getter,
                tolerance=12,
                timeout_sec=timeout,
                stop_event=self.stop_event,
                platform_bounds=(curr_node.x_min, curr_node.x_max),
                safe_margin=12,
                speed_scale=1.0,
            )
            latest = pos_getter()
            latest_x = float(latest[0]) if latest else current_x
            if arrived and (landing_lo - 12 <= latest_x <= landing_hi + 12):
                return True
            self.log_fn(
                f"⏸️ [垂直跳范围准备未到位] 当前X={latest_x:.1f}，"
                f"目标有效范围=[{landing_lo:.1f},{landing_hi:.1f}]"
            )
            return False

        # 窄目标台：以中点作为精确落点；是否短按仍只由源平台宽度决定。
        distance = abs(current_x - trigger_x)
        if distance <= 9:
            return True

        # 远距离阶段只走到起跳点前 55px；绝不把整段路程变成短按。
        if distance > 55:
            sign = 1.0 if float(trigger_x) > current_x else -1.0
            approach_x = float(trigger_x) - sign * 55.0
            approach_x = max(curr_node.x_min + 12, min(curr_node.x_max - 12, approach_x))
            coarse_timeout = max(2.0, min(10.0, abs(approach_x - current_x) / 105.0 + 1.5))
            plan_label = (
                f"fh{transition_edge.source_foothold_id}->fh{transition_edge.target_foothold_id}"
                if is_foothold_plan else f"P{landing_node.id}"
            )
            self.log_fn(
                f"🚶 [垂直跳远距准备] {plan_label}，P{curr_node.id} 持续行走至"
                f"X={approach_x:.1f}，再对齐至X={trigger_x} (当前X={current_x:.1f})"
            )
            coarse_ok = self.motion.walk_to_x(
                target_x=approach_x,
                get_player_pos=pos_getter,
                tolerance=12,
                timeout_sec=coarse_timeout,
                stop_event=self.stop_event,
                platform_bounds=(curr_node.x_min, curr_node.x_max),
                safe_margin=12,
                speed_scale=1.0,
            )
            after_coarse = pos_getter()
            current_x = float(after_coarse[0]) if after_coarse else current_x
            if not coarse_ok and abs(current_x - approach_x) > 15:
                self.log_fn(
                    f"⏸️ [垂直跳远距准备未到位] 当前X={current_x:.1f}，目标X={approach_x:.1f}"
                )
                return False

        # 只有窄源平台的最后55px才改用短按；长源平台即使目标
        # 很短，也由运动模型连续行走和预测刹停完成定位。
        narrow_source = float(curr_node.length) < 90.0
        fine_speed_scale = 0.25 if narrow_source else 1.0
        fine_timeout = max(2.0, min(4.0, abs(current_x - float(trigger_x)) / 24.0 + 1.5))
        self.log_fn(
            f"🚶 [垂直跳微调] P{curr_node.id} "
            f"{'窄源短按' if narrow_source else '长源连续行走'}收束至起跳X={trigger_x} "
            f"(当前X={current_x:.1f})"
        )
        fine_ok = self.motion.walk_to_x(
            target_x=trigger_x,
            get_player_pos=pos_getter,
            tolerance=9,
            timeout_sec=fine_timeout,
            stop_event=self.stop_event,
            platform_bounds=(curr_node.x_min, curr_node.x_max),
            safe_margin=12,
            speed_scale=fine_speed_scale,
        )
        latest = pos_getter()
        latest_x = float(latest[0]) if latest else current_x
        if not fine_ok or abs(latest_x - float(trigger_x)) > 9:
            self.log_fn(
                f"⏸️ [垂直跳微调未到位] 当前X={latest_x:.1f}，目标X={trigger_x}，本轮不启动跳跃"
            )
            return False
        return True

    def _pick_nearest_safe_down_jump_x(
        self,
        curr_node: Any,
        target_node: Optional[Any],
        first_step: Any,
        graph: Any,
        current_x: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """在安全重叠区间内选择离当前位置最近的下跳 X。"""
        if target_node is not None:
            overlap_min = max(float(curr_node.x_min), float(target_node.x_min))
            overlap_max = min(float(curr_node.x_max), float(target_node.x_max))
        elif getattr(first_step, "trigger_x_range", None):
            overlap_min = float(first_step.trigger_x_range[0]) - 10.0
            overlap_max = float(first_step.trigger_x_range[1]) + 10.0
        else:
            overlap_min = float(curr_node.x_min)
            overlap_max = float(curr_node.x_max)
        if getattr(first_step, "trigger_x_range", None):
            planned_lo, planned_hi = sorted(map(float, first_step.trigger_x_range))
            overlap_min = max(overlap_min, planned_lo)
            overlap_max = min(overlap_max, planned_hi)

        overlap_width = max(0.0, overlap_max - overlap_min)
        # 边缘安全距离：平台两端尽量保留 35px 边距，避免擦边落空（窄重叠平台自适应缩减，至少保留 10px）
        edge_margin = min(35.0, max(10.0, (overlap_width - 20.0) / 2.0))
        safe_lo = overlap_min + edge_margin
        safe_hi = overlap_max - edge_margin
        if safe_lo > safe_hi:
            safe_lo = safe_hi = (overlap_min + overlap_max) / 2.0

        # 梯绳禁跳区 (绳梯中轴两侧各 48px 内禁止触发下跳，避免误吸附或在绳上按DOWN+JUMP再次挂绳)
        ladder_xs: List[float] = []
        if graph and getattr(graph, "ladder_ropes", None):
            to_id = getattr(first_step, "to_id", None)
            for lr in graph.ladder_ropes.values():
                is_connected = (
                    lr.bottom_platform_id in (curr_node.id, to_id)
                    or lr.top_platform_id in (curr_node.id, to_id)
                )
                if is_connected:
                    ladder_xs.append(float(lr.x))
                elif target_node is not None:
                    top_y = min(float(curr_node.y), float(target_node.y))
                    bot_y = max(float(curr_node.y), float(target_node.y))
                    lr_top = min(float(lr.y1), float(lr.y2))
                    lr_bot = max(float(lr.y1), float(lr.y2))
                    if not (lr_bot < top_y or lr_top > bot_y):
                        if overlap_min <= float(lr.x) <= overlap_max:
                            ladder_xs.append(float(lr.x))

        ladder_buffer = 48.0
        intervals: List[Tuple[float, float]] = [(safe_lo, safe_hi)]
        blocked_ranges = [
            (lx - ladder_buffer, lx + ladder_buffer)
            for lx in sorted(set(ladder_xs))
        ]
        if target_node is not None and graph is not None:
            for middle in getattr(graph, "nodes", {}).values():
                if middle.id in (curr_node.id, target_node.id):
                    continue
                if float(curr_node.y) + 5 < float(middle.y) < float(target_node.y) - 5:
                    blocked_ranges.append(
                        (float(middle.x_min) - 10.0, float(middle.x_max) + 10.0)
                    )
        for bad_lo, bad_hi in blocked_ranges:
            new_intervals: List[Tuple[float, float]] = []
            for s, e in intervals:
                if bad_hi <= s or bad_lo >= e:
                    new_intervals.append((s, e))
                else:
                    if s < bad_lo:
                        new_intervals.append((s, bad_lo))
                    if e > bad_hi:
                        new_intervals.append((bad_hi, e))
            intervals = new_intervals

        valid_intervals = [(s, e) for (s, e) in intervals if e - s >= 12.0]
        if not valid_intervals:
            if blocked_ranges:
                steps = 20
                samples = [safe_lo + (safe_hi - safe_lo) * (i / steps) for i in range(steps + 1)]
                best_x = max(
                    samples,
                    key=lambda x: min(
                        max(lo - x, 0.0, x - hi) for lo, hi in blocked_ranges
                    ),
                )
                return float(round(best_x, 1)), (safe_lo, safe_hi)
            return float(round((safe_lo + safe_hi) / 2.0, 1)), (safe_lo, safe_hi)

        current_x = float(current_x)
        # 已处于无绳梯、远离边缘的安全段时直接原地下跳；否则只移动到
        # 最近安全段的最近点。旧版在几千像素宽的重叠区随机抽点，会让
        # 一条 0.42 cost 的直达下跳实际横穿整张平台。
        candidates = [max(s, min(e, current_x)) for s, e in valid_intervals]
        chosen_x = min(candidates, key=lambda value: abs(value - current_x))
        return float(round(chosen_x, 1)), (safe_lo, safe_hi)

    def _do_patrol(self, px: Optional[float], py: Optional[float],
                   *, verify_only: bool = False):
        """新巡逻编排入口；每次调用只推进持久状态机的一个阶段。"""
        supplied = (float(px), float(py)) if px is not None and py is not None else None
        self.platform_patrol.tick(supplied, verification_only=verify_only)

    def _service_exclusive_patrol_phase(
        self,
        world_position: Optional[Tuple[float, float]],
        best_target: Optional[Any] = None,
        in_attack_box: bool = False,
        spx: int = 400,
        spy: int = 300,
        facing: str = "right",
        monsters: Optional[List[Any]] = None,
    ) -> bool:
        """独占到站走位；停留时允许正面攻击和延迟回身攻击。"""
        phase = self.platform_patrol.phase
        if phase not in (PatrolPhase.POSITIONING, PatrolPhase.DWELLING):
            active = set()
            if (
                phase in (PatrolPhase.OBSERVE, PatrolPhase.PLAN,
                          PatrolPhase.ARRIVED, PatrolPhase.RECOVER)
                and best_target is not None and in_attack_box
                and world_position is not None
            ):
                group = self._group_approach_candidate(
                    monsters or [], spx, spy, facing
                )
                if group is not None:
                    platform = self.get_current_platform() if self.get_current_platform else None
                    if platform is not None:
                        active.add((int(platform.id), group[2]))
            self._prune_skirmish_boundary_blocks(active)
            self._forward_clear_since = None
            self._rear_turn_pending_direction = None
            self._dwell_attack_boxes_occupied = False
            return False

        rear_target = None
        front_skirmish = None
        rear_skirmish = None
        group_choice = None
        if best_target is not None and in_attack_box and world_position is not None:
            group_choice = self._group_approach_candidate(
                monsters or [], spx, spy, facing
            )
        if phase == PatrolPhase.DWELLING:
            rear_target = self._evaluate_rear_target(
                spx, spy, facing, monsters or []
            )
            front_skirmish, rear_skirmish = self._evaluate_skirmish_targets(
                spx, spy, facing, monsters or []
            )
            try:
                platform = self.get_current_platform() if self.get_current_platform else None
                platform_id = int(platform.id) if platform is not None else None
            except Exception:
                platform_id = None
            active_guard_keys = set()
            if platform_id is not None and front_skirmish is not None:
                active_guard_keys.add((platform_id, facing))
            if platform_id is not None and rear_skirmish is not None:
                active_guard_keys.add(
                    (platform_id, "left" if facing == "right" else "right")
                )
            if platform_id is not None and group_choice is not None:
                active_guard_keys.add((platform_id, group_choice[2]))
            self._prune_skirmish_boundary_blocks(active_guard_keys)
            # 截止续期只看正、反向直接攻击框。游击框仍可在停留期间
            # 驱动前出，但不会阻止停留结束。真正续期仍只在截止点发生。
            self._dwell_attack_boxes_occupied = bool(
                in_attack_box
                or rear_target is not None
            )
        else:
            active_guard_keys = set()
            if group_choice is not None:
                platform = self.get_current_platform() if self.get_current_platform else None
                if platform is not None:
                    active_guard_keys.add((int(platform.id), group_choice[2]))
            self._prune_skirmish_boundary_blocks(active_guard_keys)
            self._dwell_attack_boxes_occupied = False

        if best_target is not None and in_attack_box:
            if self._ensure_safe_attack_position(facing):
                return True
            if group_choice is not None:
                skill, outer_target, direction = group_choice
                if self._do_skirmish_approach(
                    outer_target, direction, spx, world_position,
                    group_skill=skill,
                    completion_checker=lambda: self._group_attack_ready_or_lost(
                        skill, direction
                    ),
                ):
                    # 闭环行走已完成或正在收束；下一轮用实时怪物框选技攻击。
                    return True
            self._forward_clear_since = None
            self._rear_turn_pending_direction = None
            self._do_attack(best_target, spx, spy, facing)
            return True

        # 回身攻击严格限定在目标平台的停留阶段。站位和任何路径边执行
        # 阶段都不会因为身后目标改变方向。
        if phase == PatrolPhase.POSITIONING:
            self._forward_clear_since = None
            self._rear_turn_pending_direction = None

        # 进入停留时 PlatformPatrolFSM 已经 stop 一次。这里不得每帧
        # 重复急停：前台 SendInput 下的高频 KEYUP 会抬起玩家正在按的
        # 物理方向键，使角色在边界保护后连手动也无法移动。
        if phase == PatrolPhase.DWELLING:
            now = time.perf_counter()
            if self._rear_turn_pending_direction is not None:
                if facing == self._rear_turn_pending_direction:
                    self._rear_turn_pending_direction = None
                elif now < self._rear_turn_pending_until:
                    if world_position is None:
                        self._do_patrol(None, None)
                    else:
                        self._do_patrol(world_position[0], world_position[1])
                    return True
                else:
                    self._rear_turn_pending_direction = None

            if self._forward_clear_since is None:
                self._forward_clear_since = now
            try:
                delay = float(self.config.get("rear_attack_turn_delay_sec", 2.0))
            except (TypeError, ValueError):
                delay = 2.0
            delay = max(0.0, min(30.0, delay))
            waited = now - self._forward_clear_since
            delay_satisfied = (not self._has_attacked_monster) or waited >= delay
            if rear_target is not None and delay_satisfied:
                turn_to = "left" if facing == "right" else "right"
                tx = getattr(rear_target, 'center_x', getattr(rear_target, 'cx', None))
                ty = getattr(rear_target, 'center_y', getattr(rear_target, 'cy', None))
                self.log_fn(
                    f"↩️ [停留回身攻击] 正面已空{waited:.2f}s，"
                    f"身后目标=({tx},{ty})，转向{turn_to}；延迟={delay:.2f}s"
                )
                self.motion.face_direction(turn_to, current_facing=facing)
                self._rear_turn_pending_direction = turn_to
                self._rear_turn_pending_until = now + 0.35
            elif (
                rear_target is not None and not delay_satisfied
                and now - self._last_rear_wait_log_at >= 0.5
            ):
                self._last_rear_wait_log_at = now
                self.log_fn(
                    f"⏳ [回身攻击等待] 正面无怪{waited:.2f}/{delay:.2f}s，"
                    "保持当前朝向"
                )

            # 身后橙框内的直接攻击始终高于任何游击行为；即使尚在等待
            # 回身冷却，也不为更低优先级目标离开当前站位。
            if rear_target is not None:
                if world_position is None:
                    self._do_patrol(None, None)
                else:
                    self._do_patrol(world_position[0], world_position[1])
                return True

            if front_skirmish is not None:
                self._do_skirmish_approach(
                    front_skirmish, facing, spx, world_position
                )
            elif rear_skirmish is not None:
                rear_direction = "left" if facing == "right" else "right"
                if any(
                    skill["two_way"]
                    for skill in eligible_skills(self.config, mob_id_of(rear_skirmish))
                ):
                    # 双向技能无需转身，直接在平台安全范围内向目标接近。
                    self._do_skirmish_approach(
                        rear_skirmish, rear_direction, spx, world_position
                    )
                elif delay_satisfied:
                    tx = getattr(rear_skirmish, 'center_x', getattr(rear_skirmish, 'cx', None))
                    ty = getattr(rear_skirmish, 'center_y', getattr(rear_skirmish, 'cy', None))
                    self.log_fn(
                        f"↩️ [后向游击转身] 正面已空{waited:.2f}s，"
                        f"低优先级目标=({tx},{ty})，转向{rear_direction}"
                    )
                    self.motion.face_direction(rear_direction, current_facing=facing)
                    self._rear_turn_pending_direction = rear_direction
                    self._rear_turn_pending_until = now + 0.35
                elif now - self._last_rear_wait_log_at >= 0.5:
                    self._last_rear_wait_log_at = now
                    self.log_fn(
                        f"⏳ [后向游击等待] 正面无怪{waited:.2f}/{delay:.2f}s，"
                        "暂不回身追击"
                    )

        if world_position is None:
            self._do_patrol(None, None)
        else:
            self._do_patrol(world_position[0], world_position[1])
        return True

    def _execute_patrol_edge(self, target_id: int, px: float, py: float):
        """复用已实机调校的单边动作执行器，目标与生命周期由新FSM管理。"""
        attempt = getattr(self.platform_patrol, "attempt", None)
        return self._execute_forced_patrol_edge(
            target_id, px, py, getattr(attempt, "edge", None)
        )

    def _execute_forced_patrol_edge(
        self, target_id: int, px: float, py: float, forced_edge=None
    ):
        """F6 与独立压力测试共用的、唯一的单条拓扑边执行入口。"""
        previous_target = self._patrol_forced_target
        previous_edge = self._patrol_forced_edge
        self._patrol_forced_target = int(target_id)
        self._patrol_forced_edge = forced_edge
        # 旧执行器内部残留的局部验证状态不能跨越新FSM的边界。
        self._set_platform_transition_state(PlatformTransitionState.IDLE)
        try:
            self.motion.last_walk_priority_interrupted = False
            self.motion.last_jump_gate_aborted = False
            self.motion.last_top_exit_failed = False
            self._last_vertical_source_lost = False
        except Exception:
            pass
        try:
            self._execute_patrol_step_legacy(px, py)
            if self._last_vertical_source_lost:
                return "relocalize"
            if bool(getattr(self.motion, "last_top_exit_failed", False)):
                return "top_exit_failed"
            if bool(getattr(self.motion, "last_jump_gate_aborted", False)):
                return "retry"
            return not bool(
                getattr(self.motion, "last_walk_priority_interrupted", False)
            )
        finally:
            self._patrol_forced_target = previous_target
            self._patrol_forced_edge = previous_edge

    def begin_external_navigation_test(self) -> Tuple[bool, str]:
        """隔离一次独立测试，防止测试失败计数污染下一次 F6。"""
        with self._external_edge_lock:
            if self.is_running:
                return False, "F6 自动挂机正在运行，不能同时启动导航测试。"
            if self._external_test_active:
                return False, "已有 F6 导航验收测试正在运行。"
            self._external_saved_navigation_state = {
                "run_jump_failures": dict(self._run_jump_failures),
                "state": self.state,
                "last_patrol_platform_id": self._last_patrol_platform_id,
                "short_platform_centered_id": self._short_platform_centered_id,
                "patrol_target_idx": self.patrol_target_idx,
                "patrol_dir": self.patrol_dir,
                "active_down_jump_target": self._active_down_jump_target,
                "motion_walk_interrupted": self.motion.last_walk_priority_interrupted,
                "motion_jump_gate_aborted": self.motion.last_jump_gate_aborted,
                "motion_landing_confirmed": self.motion.last_landing_confirmed,
            }
            self._run_jump_failures.clear()
            self._active_down_jump_target = None
            self._set_platform_transition_state(PlatformTransitionState.IDLE)
            self._external_test_active = True
            return True, "F6 共享导航执行器已进入测试模式。"

    def end_external_navigation_test(self) -> None:
        """恢复测试前的 F6 导航状态。"""
        with self._external_edge_lock:
            if not self._external_test_active and self._external_saved_navigation_state is None:
                return
            saved = self._external_saved_navigation_state or {}
            self._run_jump_failures.clear()
            self._run_jump_failures.update(saved.get("run_jump_failures", {}))
            if "state" in saved:
                self.state = saved["state"]
            self._last_patrol_platform_id = saved.get("last_patrol_platform_id")
            self._short_platform_centered_id = saved.get("short_platform_centered_id")
            self.patrol_target_idx = int(saved.get("patrol_target_idx", self.patrol_target_idx))
            self.patrol_dir = int(saved.get("patrol_dir", self.patrol_dir))
            self._active_down_jump_target = saved.get("active_down_jump_target")
            if "motion_walk_interrupted" in saved:
                self.motion.last_walk_priority_interrupted = saved["motion_walk_interrupted"]
            if "motion_jump_gate_aborted" in saved:
                self.motion.last_jump_gate_aborted = saved["motion_jump_gate_aborted"]
            if "motion_landing_confirmed" in saved:
                self.motion.last_landing_confirmed = saved["motion_landing_confirmed"]
            self._external_saved_navigation_state = None
            self._external_test_active = False
            self._set_platform_transition_state(PlatformTransitionState.IDLE)

    def execute_patrol_edge_for_test(
        self,
        edge,
        target_id: int,
        px: float,
        py: float,
        *,
        world_position_getter,
        raw_position_getter,
        platform_getter,
        is_climbing_getter,
        stop_event: threading.Event,
        run_jump_enabled: bool,
        log_callback,
        top_exit_landing_sample_callback=None,
        top_exit_landing_tolerance_override=None,
    ):
        """以测试观测源运行 F6 的真实单边执行器，不复制动作逻辑。"""
        with self._external_edge_lock:
            if self.is_running:
                raise RuntimeError("F6 已启动，独立导航测试必须停止")
            if not self._external_test_active:
                raise RuntimeError("F6 共享导航测试会话尚未启动")

            previous = {
                "stop_event": self.stop_event,
                "world": self.get_player_world_pos,
                "raw": self.get_player_raw_world_pos,
                "platform": self.get_current_platform,
                "climbing": self.get_is_climbing,
                "run_jump": self.get_enable_run_jump,
                "log": self.log_fn,
                "motion_raw": self.motion.raw_position_getter,
                "motion_log": self.motion.log_fn,
                "motion_priority": self.motion.priority_interrupt_checker,
                "top_exit_sample_callback": getattr(
                    self, "_top_exit_landing_sample_callback", None
                ),
                "top_exit_tolerance_override": getattr(
                    self, "_top_exit_landing_tolerance_override", None
                ),
            }
            self.stop_event = stop_event
            self.get_player_world_pos = world_position_getter
            self.get_player_raw_world_pos = raw_position_getter
            self.get_current_platform = platform_getter
            self.get_is_climbing = is_climbing_getter
            self.get_enable_run_jump = lambda: bool(run_jump_enabled)
            self.log_fn = log_callback
            self.motion.raw_position_getter = raw_position_getter
            self.motion.log_fn = log_callback
            # 独立验收没有战斗主循环可接管输入；若保留红框中断，动作会
            # 让行后无人攻击/续跑。这里只隔离战斗抢占，移动执行器本身不变。
            self.motion.priority_interrupt_checker = None
            # 专项测试可以采集绳顶 raw 落台误差，并临时使用较宽的确认带。
            # 两项状态均严格限制在这一条测试边内，正式 F6 不受影响。
            self._top_exit_landing_sample_callback = top_exit_landing_sample_callback
            self._top_exit_landing_tolerance_override = (
                float(top_exit_landing_tolerance_override)
                if top_exit_landing_tolerance_override is not None
                else None
            )
            try:
                return self._execute_forced_patrol_edge(
                    int(target_id), float(px), float(py), edge
                )
            finally:
                self.stop_event = previous["stop_event"]
                self.get_player_world_pos = previous["world"]
                self.get_player_raw_world_pos = previous["raw"]
                self.get_current_platform = previous["platform"]
                self.get_is_climbing = previous["climbing"]
                self.get_enable_run_jump = previous["run_jump"]
                self.log_fn = previous["log"]
                self.motion.raw_position_getter = previous["motion_raw"]
                self.motion.log_fn = previous["motion_log"]
                self.motion.priority_interrupt_checker = previous["motion_priority"]
                self._top_exit_landing_sample_callback = previous[
                    "top_exit_sample_callback"
                ]
                self._top_exit_landing_tolerance_override = previous[
                    "top_exit_tolerance_override"
                ]

    def _execute_patrol_step_legacy(self, px: int, py: int):
        """单条动作执行器；不再拥有路线、目标轮换或失败生命周期。"""
        self.state = BotState.PATROLLING

        # 1. 优先检查多平台循环巡航
        patrol_list = (
            [self._patrol_forced_target]
            if self._patrol_forced_target is not None
            else (self.get_patrol_platforms() if self.get_patrol_platforms else [])
        )
        g = self.get_platform_graph() if self.get_platform_graph else None
        curr_node = self.get_current_platform() if self.get_current_platform else None

        if g and g.nodes and curr_node is None:
            curr_node = g.find_player_platform(px, py)

        # 短段跳跃发出后不能在下一帧立即重复规划。先等待视觉确认真正
        # 落到目标长平台；超时或落到其他平台才进入恢复并从实况重规划。
        active_transition = self.active_platform_transition
        if (
            self.platform_transition_state == PlatformTransitionState.VERIFYING
            and active_transition is not None
        ):
            elapsed = time.perf_counter() - self._transition_started_at
            if curr_node is not None and curr_node.id == active_transition.to_id:
                self.log_fn(
                    f"✅ [平台转移确认] P{active_transition.from_id} -> "
                    f"P{active_transition.to_id}，已落到目标平台"
                )
                self._set_platform_transition_state(PlatformTransitionState.IDLE)
            elif elapsed < 1.25:
                return
            else:
                observed_id = getattr(curr_node, 'id', None)
                self._set_platform_transition_state(
                    PlatformTransitionState.RECOVERING, active_transition
                )
                self.log_fn(
                    f"↩️ [平台转移恢复] 预期P{active_transition.to_id}，"
                    f"当前P{observed_id}，等待{elapsed:.2f}s仍未确认，按当前位置重新规划"
                )

        # 进入新平台时解除“已居中”标记。这样每次重新落到窄平台都会
        # 先回中点一次，而不会沿用上一次经过该平台的状态。
        previous_platform_id = self._last_patrol_platform_id
        if curr_node and curr_node.id != previous_platform_id:
            self._last_patrol_platform_id = curr_node.id
            self._short_platform_centered_id = None
            # 角色离开后又回到某个起点平台，说明上一轮跨层动作
            # 已失败/坠落返回；该起点上的绳梯应重新从跑跳第 1 次开始。
            if previous_platform_id is not None:
                self._run_jump_failures = {
                    key: count for key, count in self._run_jump_failures.items()
                    if key[0] != curr_node.id
                }

        pos_getter = self.get_player_world_pos or (lambda: getattr(self.detector, 'last_player_pos', None))

        if len(patrol_list) >= 1 and g and g.nodes:
            target_pid = patrol_list[self.patrol_target_idx % len(patrol_list)]
            target_node = g.get_node(target_pid)
            run_jump_enabled = True
            run_jump_fallback_enabled = bool(
                self.config.get("run_jump_fallback_to_static_grab", False)
            )
            run_jump_failure_limit = max(
                1, int(self.config.get("run_jump_failure_limit", 2) or 2)
            )
            if self.get_enable_run_jump_fallback is not None:
                try:
                    run_jump_fallback_enabled = bool(self.get_enable_run_jump_fallback())
                except Exception:
                    pass
            if self.get_run_jump_failure_limit is not None:
                try:
                    run_jump_failure_limit = max(1, int(self.get_run_jump_failure_limit()))
                except Exception:
                    pass
            if self.get_enable_run_jump is not None:
                try:
                    run_jump_enabled = bool(self.get_enable_run_jump())
                except Exception:
                    run_jump_enabled = True
            allow_intra_map_portal = not bool(
                self.config.get("disable_intra_map_portals", False)
            )

            if curr_node and curr_node.id == target_pid:
                # 到达循环目标平台，下一段属于新一轮导航，清空上一轮
                # 的跑跳失败次数，确保前两次仍使用跑跳抓取。
                self._run_jump_failures.clear()
                self._active_down_jump_target = None
                # 已经在目标平台上了：在当前平台安全范围内游走巡逻 (带防跌落边界)
                safe_min_x = curr_node.x_min + max(25, min(40, curr_node.length // 5))
                safe_max_x = curr_node.x_max - max(25, min(40, curr_node.length // 5))

                if safe_min_x < safe_max_x:
                    target_x = safe_max_x if self.patrol_dir > 0 else safe_min_x
                    if abs(px - target_x) <= 30:
                        self.patrol_dir *= -1
                        target_x = safe_max_x if self.patrol_dir > 0 else safe_min_x

                    self.log_fn(f"🏃 [平台巡逻] 平台 #{curr_node.id} 巡航 -> 走向 X={target_x} (当前 X={px})")
                    arrived = self.motion.walk_to_x(
                        target_x=target_x,
                        get_player_pos=pos_getter,
                        tolerance=25,
                        timeout_sec=2.0,
                        stop_event=self.stop_event,
                        platform_bounds=(curr_node.x_min, curr_node.x_max),
                        safe_margin=25
                    )
                    self.patrol_dir *= -1
                    self.patrol_target_idx += 1
                    next_pid = patrol_list[self.patrol_target_idx % len(patrol_list)]
                    self.log_fn(f"🔄 [平台循环] 平台 #{target_pid} 扫荡完成，前往平台 #{next_pid}...")
                else:
                    self.patrol_target_idx += 1
                return
            elif curr_node and target_node:
                # 不在目标平台上：调用拓扑图寻路跨层
                self._set_platform_transition_state(PlatformTransitionState.PLANNING)
                path = g.find_path(
                    curr_node.id,
                    target_pid,
                    allow_run_jump=run_jump_enabled,
                    allow_portal=allow_intra_map_portal,
                )
                if path:
                    # 跑跳关闭时，不能执行“绕过中间平台”的直达抓绳边。
                    # 例如 2 号绳实际位于 P36 上方，P33->P67 是专门给
                    # 跑跳模式生成的捷径；关闭跑跳应先走 P33->P36，
                    # 下一轮再从 P36 用普通跳抓绳。
                    first_candidate = path[0]
                    if (
                        not run_jump_enabled
                        and "JUMP_CLIMB" in first_candidate.action
                        and first_candidate.ladder_id
                    ):
                        lr_candidate = g.get_ladder_rope(first_candidate.ladder_id)
                        if lr_candidate is not None:
                            outside = (
                                lr_candidate.x < curr_node.x_min - 20
                                or lr_candidate.x > curr_node.x_max + 20
                            )
                            bottom_id = lr_candidate.bottom_platform_id
                            if outside and bottom_id and bottom_id != curr_node.id:
                                detour = g.find_path(
                                    curr_node.id,
                                    bottom_id,
                                    allow_run_jump=run_jump_enabled,
                                    allow_portal=allow_intra_map_portal,
                                )
                                if detour:
                                    self.log_fn(
                                        f"↪️ [跑跳已关闭] 绳梯 #{lr_candidate.id} 位于 "
                                        f"P{bottom_id} 上方，先绕行至 P{bottom_id} 再普通跳抓"
                                    )
                                    path = detour
                    # FSM可能因连续失败给某条边加权，并选出一条非默认边。
                    # 当前动作必须执行该边；后续路径只用于落台方向预判。
                    forced_edge = self._patrol_forced_edge
                    if (
                        forced_edge is not None
                        and int(getattr(forced_edge, "from_id", -1)) == int(curr_node.id)
                    ):
                        suffix = []
                        if int(forced_edge.to_id) != int(target_pid):
                            suffix = g.find_path(
                                forced_edge.to_id,
                                target_pid,
                                allow_run_jump=run_jump_enabled,
                                allow_portal=allow_intra_map_portal,
                            ) or []
                        path = [forced_edge, *suffix]
                    first_step = path[0]
                    action_failure_count = self._current_action_failure_count(first_step)
                    if (
                        action_failure_count > 0
                        and first_step.action == "TELEPORT_UP"
                    ):
                        retry_source = g.get_node(first_step.from_id)
                        retry_target = g.get_node(first_step.to_id)
                        retry_edge = None
                        if retry_source is not None and retry_target is not None:
                            retry_edge = PlatformGraphBuilder._best_vertical_foothold_jump(
                                g,
                                retry_source,
                                retry_target,
                                enable_teleport=bool(
                                    getattr(self.motion, "enable_teleport", False)
                                ),
                                teleport_distance_px=float(
                                    self.config.get("teleport_distance_px", 150.0)
                                ),
                                candidate_rank=action_failure_count,
                            )
                        if retry_edge is not None:
                            self.log_fn(
                                f"🔀 [瞬移候选重选] 原动作已失败{action_failure_count}次，"
                                f"改用 fh{retry_edge.source_foothold_id}->"
                                f"fh{retry_edge.target_foothold_id}，"
                                f"安全区={retry_edge.trigger_x_range}，"
                                f"高度差方案={retry_edge.description}"
                            )
                            first_step = retry_edge
                    self.log_fn(f"🪜 [跨层导航] P{curr_node.id} -> P{target_pid}: {first_step.description}")

                    dest_node = g.get_node(first_step.to_id)
                    rope_node = (
                        g.get_ladder_rope(first_step.ladder_id)
                        if first_step.ladder_id else None
                    )
                    landing_samples = 0

                    def raw_landing_confirmed() -> bool:
                        """用raw黄点连续命中确认人物已物理离开绳轴并落台。"""
                        nonlocal landing_samples
                        raw_pos = (
                            self.get_player_raw_world_pos()
                            if self.get_player_raw_world_pos else None
                        )
                        if raw_pos is None or dest_node is None:
                            landing_samples = 0
                            return False
                        raw_x, raw_y = raw_pos
                        in_platform = (
                            (dest_node.x_min + 2) <= raw_x <= (dest_node.x_max - 2)
                        )
                        expected_y = float(dest_node.surface_y_at(float(raw_x))) - 45.0
                        try:
                            measurement_step = float(
                                getattr(
                                    self.motion.motion_model,
                                    "measurement_step_px",
                                    0.0,
                                ) or 0.0
                            )
                        except (TypeError, ValueError):
                            measurement_step = 0.0
                        rope_clearance = max(
                            8.0, min(30.0, measurement_step * 0.60)
                        )
                        if rope_node is None:
                            clear_of_rope = True
                        elif step_dir == "right":
                            # 目标平台在右侧时，绳轴左侧的量化偏差绝不是
                            # 成功脱绳；必须实际越过绳轴右侧安全线。
                            clear_of_rope = (
                                float(raw_x) > float(rope_node.x) + rope_clearance
                            )
                        elif step_dir == "left":
                            clear_of_rope = (
                                float(raw_x) < float(rope_node.x) - rope_clearance
                            )
                        else:
                            clear_of_rope = (
                                abs(float(raw_x) - float(rope_node.x)) > rope_clearance
                            )
                        tolerance_override = getattr(
                            self, "_top_exit_landing_tolerance_override", None
                        )
                        if tolerance_override is not None:
                            standing_tolerance = max(
                                1.0, float(tolerance_override)
                            )
                        else:
                            try:
                                standing_tolerance = max(
                                    1.0,
                                    float(
                                        self.config.get(
                                            "top_exit_raw_y_tolerance_px", 36.0
                                        )
                                    ),
                                )
                            except (TypeError, ValueError):
                                standing_tolerance = 36.0
                        y_error = float(raw_y) - expected_y
                        standing_band = abs(y_error) <= standing_tolerance
                        if in_platform and standing_band and clear_of_rope:
                            landing_samples += 1
                        else:
                            landing_samples = 0
                        sample_callback = getattr(
                            self, "_top_exit_landing_sample_callback", None
                        )
                        if callable(sample_callback):
                            try:
                                sample_callback(
                                    {
                                        "raw_x": float(raw_x),
                                        "raw_y": float(raw_y),
                                        "expected_y": expected_y,
                                        "y_error": y_error,
                                        "surface_y": expected_y + 45.0,
                                        "in_platform": bool(in_platform),
                                        "clear_of_rope": bool(clear_of_rope),
                                        "standing_band": bool(standing_band),
                                        "standing_tolerance": standing_tolerance,
                                        "measurement_step": measurement_step,
                                        "landing_samples": landing_samples,
                                        "confirmed": landing_samples >= 3,
                                    }
                                )
                            except Exception:
                                pass
                        return landing_samples >= 3

                    step_dir = "right" if dest_node and dest_node.center_x > (first_step.trigger_x or curr_node.center_x) else ("left" if dest_node and dest_node.center_x < (first_step.trigger_x or curr_node.center_x) else None)
                    if (step_dir is None and dest_node is not None
                            and str(first_step.action).endswith("_DOWN")):
                        rope_x = float(first_step.trigger_x or curr_node.center_x)
                        step_dir = (
                            "right" if dest_node.x_max - rope_x >= rope_x - dest_node.x_min
                            else "left"
                        )
                    # 若爬绳/爬梯后还要继续跳跃，落台方向应服务于下一跳的
                    # 蓄跑方向，而不是被触发点与平台中心的 1px 取整误差决定。
                    next_step_dir = step_dir
                    if dest_node is not None and dest_node.id != target_pid:
                        lookahead = g.find_path(
                            dest_node.id,
                            target_pid,
                            allow_run_jump=run_jump_enabled,
                            allow_portal=allow_intra_map_portal,
                        )
                        if lookahead:
                            next_action = lookahead[0].action
                            if "RIGHT" in next_action or "WALK_RIGHT" in next_action:
                                next_step_dir = "left"
                            elif "LEFT" in next_action or "WALK_LEFT" in next_action:
                                next_step_dir = "right"

                    # 1. 悬空起跳抓梯/抓绳 (JUMP_CLIMB_UP)
                    if first_step.action == "PORTAL":
                        trigger_x = (
                            first_step.trigger_x
                            if first_step.trigger_x is not None
                            else curr_node.center_x
                        )

                        def portal_completed() -> bool:
                            if self.get_current_platform is None:
                                return False
                            try:
                                observed = self.get_current_platform()
                                return bool(
                                    observed is not None
                                    and int(observed.id) == int(first_step.to_id)
                                )
                            except Exception:
                                return False

                        self.log_fn(
                            f"🌀 [传送点] P{curr_node.id} 向X={trigger_x}连续行走，"
                            f"进入触发区前按住UP，目标P{target_pid}"
                        )
                        self.motion.walk_through_portal(
                            trigger_x=trigger_x,
                            get_player_pos=pos_getter,
                            stop_event=self.stop_event,
                            platform_bounds=(curr_node.x_min, curr_node.x_max),
                            completion_checker=portal_completed,
                            approach_lead_px=60.0,
                            pass_through_px=16.0,
                            timeout_sec=4.0,
                        )
                        return

                    if "JUMP_CLIMB" in first_step.action:
                        trigger_x = first_step.trigger_x or curr_node.center_x
                        t_type = "rope" if ("ROPE" in first_step.action or "绳" in first_step.description) else "ladder"
                        grab_tolerance = 6 if t_type == "rope" else 12

                        def raw_still_on_grab_source() -> bool:
                            """最终发出抓绳按键前，确认仍站在本边源平台。"""
                            if self.get_is_climbing is not None:
                                try:
                                    if self.get_is_climbing():
                                        return False
                                except Exception:
                                    pass
                            observed = (
                                self.get_current_platform()
                                if self.get_current_platform is not None else None
                            )
                            if observed is not None and int(observed.id) != int(curr_node.id):
                                self._last_vertical_source_lost = True
                                return False
                            raw = (
                                self.get_player_raw_world_pos()
                                if self.get_player_raw_world_pos is not None else None
                            )
                            if raw is None:
                                return observed is None or int(observed.id) == int(curr_node.id)
                            raw_x, raw_y = float(raw[0]), float(raw[1])
                            if not (curr_node.x_min <= raw_x <= curr_node.x_max):
                                self._last_vertical_source_lost = True
                                return False
                            expected_foot_y = float(curr_node.surface_y_at(raw_x)) - 45.0
                            try:
                                raw_y_tolerance = float(
                                    self.config.get("jump_source_raw_y_tolerance_px", 65.0)
                                )
                            except (TypeError, ValueError):
                                raw_y_tolerance = 65.0
                            standing = abs(raw_y - expected_foot_y) <= raw_y_tolerance
                            if not standing:
                                self._last_vertical_source_lost = True
                            return standing

                        def abandon_old_grab_edge_if_source_lost(stage: str) -> bool:
                            if raw_still_on_grab_source():
                                # 前一帧的守卫可能因平台识别瞬时空缺而拒绝；
                                # 当前已重新确认源平台时允许本轮重新对齐。
                                try:
                                    self.motion.last_jump_gate_aborted = False
                                except Exception:
                                    pass
                                return False
                            try:
                                self.motion.last_jump_gate_aborted = True
                            except Exception:
                                pass
                            # 这不是“起跳线尚未放行”，而是人物已经物理掉到
                            # 另一平台。向外层明确报告 relocalize，禁止随机测试
                            # 和 F6 用旧源平台保护继续派发同一条边。
                            self._last_vertical_source_lost = True
                            self.motion.stop()
                            observed = (
                                self.get_current_platform()
                                if self.get_current_platform is not None else None
                            )
                            raw = (
                                self.get_player_raw_world_pos()
                                if self.get_player_raw_world_pos is not None else None
                            )
                            self.log_fn(
                                f"↩️ [抓绳旧边取消] {stage}前已离开源平台P{curr_node.id}，"
                                f"当前P{getattr(observed, 'id', None)}，raw={raw}；"
                                "不再发送Alt，按当前位置重规划"
                            )
                            return True

                        self.log_fn(f"🧗 [起跳抓梯] P{curr_node.id} 走向 X={trigger_x} 执行 [跳+上] 空中吸附攀爬 ({t_type})...")
                        # 起跑方向与起跳距离统一使用当前 Kalman 连续世界坐标。
                        # raw 只参与下方源平台/落台物理保险。
                        initial_grab_pos = pos_getter()
                        initial_grab_dx = (
                            abs(float(trigger_x) - float(initial_grab_pos[0]))
                            if initial_grab_pos is not None else 0.0
                        )
                        run_jump_direction = None
                        run_jump_enabled = True
                        if self.get_enable_run_jump is not None:
                            try:
                                run_jump_enabled = bool(self.get_enable_run_jump())
                            except Exception:
                                run_jump_enabled = True
                        # 上一轮若已在同一点原地跳抓失败，本轮先从平台内侧
                        # 重新生成跑道，不能再次把相同量化坐标送进原地跳。
                        # 这属于动作参数重规划，不会改变当前拓扑边。
                        if action_failure_count > 0 and run_jump_enabled:
                            run_jump_direction = self._stage_failed_static_grab(
                                curr_node,
                                float(trigger_x),
                                pos_getter,
                                action_failure_count + 1,
                            )
                            if bool(getattr(self.motion, "last_walk_priority_interrupted", False)):
                                self.log_fn(
                                    "⚔️ [跑跳准备让行攻击] 准备走位已中断，"
                                    "本轮禁止继续助跑或发送UP+Alt"
                                )
                                return
                            if run_jump_direction in ("left", "right"):
                                initial_grab_pos = pos_getter()
                                initial_grab_dx = (
                                    abs(float(trigger_x) - float(initial_grab_pos[0]))
                                    if initial_grab_pos is not None else 0.0
                                )
                        # 勾选跑跳抓时，所有“跳+上”边都优先使用跑跳，而不是只
                        # 在旧版的距离 >150px 时才启用。跑跳控制器会根据绳梯
                        # 底端高度反推准备线，并在准备线不停顿地衔接 UP+Alt。
                        # 仅当角色已几乎压在立柱正下方（<=8px）时，跑道长度不足，
                        # 才保留原地抓取；这避免在无实际水平输入的情况下伪装成跑跳。
                        if (
                            run_jump_direction not in ("left", "right")
                            and run_jump_enabled
                            and initial_grab_pos is not None
                            and initial_grab_dx > 8.0
                        ):
                            run_jump_direction = "right" if trigger_x > initial_grab_pos[0] else "left"
                            route_kind = "直接跨台" if "绕过中间平台" in first_step.description else "常规"
                            self.log_fn(
                                f"🏃 [跑跳抓取已选中] {route_kind}边，起始X={initial_grab_pos[0]:.0f}，"
                                f"绳梯X={trigger_x:.0f}，距离={initial_grab_dx:.1f}px，"
                                f"向{('右' if run_jump_direction == 'right' else '左')}连续跑跳"
                            )
                        elif run_jump_enabled and run_jump_direction not in ("left", "right"):
                            self.log_fn(
                                f"ℹ️ [跑跳抓取条件不足] 起始X={initial_grab_pos[0] if initial_grab_pos else '未知'}，"
                                f"绳梯X={trigger_x:.0f}，距离={initial_grab_dx:.1f}px <= 8px，改用原地跳抓"
                            )
                        run_jump_key = (curr_node.id, target_pid, first_step.ladder_id)
                        run_jump_failures = self._run_jump_failures.get(run_jump_key, 0)
                        if (
                            run_jump_fallback_enabled
                            and run_jump_direction in ("left", "right")
                            and run_jump_failures >= run_jump_failure_limit
                        ):
                            self.log_fn(
                                f"🔁 [跑跳降级] P{curr_node.id} -> P{target_pid} "
                                f"跑跳已失败 {run_jump_failures} 次，改为绳梯下方原地跳抓"
                            )
                            run_jump_direction = None
                        # 稳定性测试与 F6 共用同一个实时速度相交门。禁止再
                        # 注入固定准备距离；起跳位置由当前 Vx、加速度、移动
                        # 上限、脚下高度和绳底高度逐帧预测。
                        run_jump_profile = None
                        if run_jump_direction in ("left", "right"):
                            self.log_fn("🧭 [跑跳兼容模式] 使用实时速度+绳底高度相交门，无固定起跳距离")
                        static_grab_pos_getter = pos_getter
                        grabbed = self.motion.jump_and_grab_ladder(
                            target_ladder_x=trigger_x,
                            get_player_pos=static_grab_pos_getter,
                            # 横向对齐、准备线与起跳触发统一使用 Kalman X。
                            align_player_pos=static_grab_pos_getter,
                            jump_key=self.jump_key,
                            target_y=first_step.target_y,
                            step_off_direction=next_step_dir,
                            platform_bounds=(curr_node.x_min, curr_node.x_max),
                            approach_tolerance=grab_tolerance,
                            timeout_sec=4.0,
                            stop_event=self.stop_event,
                            get_raw_player_pos=self.get_player_raw_world_pos,
                            target_type=t_type,
                            enable_jump_steering=False,
                            landing_confirm_fn=raw_landing_confirmed,
                            source_standing_check=raw_still_on_grab_source,
                            run_jump_direction=run_jump_direction,
                            run_jump_sec=0.15,
                            ladder_bottom_y=(max(rope_node.y1, rope_node.y2) if rope_node is not None else None),
                            run_jump_profile=run_jump_profile,
                            use_model_static_approach=True,
                            top_hold_sec=float(self.config.get("climb_top_hold_sec", 1.0)),
                        )
                        if bool(getattr(self.motion, "last_top_exit_failed", False)):
                            self.log_fn(
                                "⛔ [绳顶恢复待续] 抓取已成功但物理脱绳尚未确认；"
                                "本轮禁止重跳和失败计数"
                            )
                            return
                        if (
                            not grabbed
                            and bool(getattr(self.motion, "last_jump_gate_aborted", False))
                        ):
                            self.log_fn(
                                "🔁 [跑跳起跳线原边重规划] Alt未发出，不计抓绳失败；"
                                "按当前实时坐标重新派发原边"
                            )
                            return
                        if grabbed:
                            self._run_jump_failures.pop(run_jump_key, None)
                        elif run_jump_direction in ("left", "right"):
                            if self.reset_motion_prediction:
                                self.reset_motion_prediction()
                            self._run_jump_failures[run_jump_key] = run_jump_failures + 1
                        # 视觉平台状态是离散的；若已经连续确认落到本段的
                        # 直接目标平台（dest_node），即使原始脱绳判定漏帧，
                        # 也不得再次起跳重试。
                        platform_landed = False
                        if not grabbed and dest_node is not None and not self.stop_event.is_set():
                            stable_hits = 0
                            for _ in range(6):
                                observed = self.get_current_platform() if self.get_current_platform else None
                                climbing = self.get_is_climbing() if self.get_is_climbing else False
                                if observed is not None and observed.id == dest_node.id and not climbing:
                                    stable_hits += 1
                                    if stable_hits >= 2:
                                        platform_landed = True
                                        break
                                else:
                                    stable_hits = 0
                                time.sleep(0.08)
                            if platform_landed:
                                grabbed = True
                                self.log_fn(f"✅ [平台状态确认] 已稳定站在直接目标平台 P{dest_node.id}，跳过抓绳重试")

                        if not grabbed and not platform_landed and not self.stop_event.is_set():
                            if abandon_old_grab_edge_if_source_lost("抓梯重试"):
                                return
                            self.log_fn("⚠️ [抓梯重试] 首次未确认攀爬，保持拓扑 X 坐标重新尝试一次")
                            # 首次跑跳失败时角色通常仍处于下落帧；不能沿用
                            # 第一次的方向立刻再跑，否则一旦已经越过立柱，就会
                            # 向错误的一侧越跑越远。等待 raw 坐标在落地后稳定，
                            # 再从最新位置重算本次跑跳的方向与准备线。
                            retry_raw = None
                            retry_last_y = None
                            retry_stable_samples = 0
                            retry_deadline = time.perf_counter() + 1.6
                            while time.perf_counter() < retry_deadline and not self.stop_event.is_set():
                                candidate = (
                                    self.get_player_raw_world_pos()
                                    if self.get_player_raw_world_pos is not None
                                    else None
                                )
                                if candidate is not None:
                                    retry_raw = candidate
                                    if retry_last_y is not None and abs(float(candidate[1]) - retry_last_y) <= 2.5:
                                        retry_stable_samples += 1
                                    else:
                                        retry_stable_samples = 0
                                    retry_last_y = float(candidate[1])
                                    if retry_stable_samples >= 3:
                                        break
                                time.sleep(0.08)

                            retry_run_jump_direction = run_jump_direction
                            retry_world = pos_getter()
                            if (
                                retry_run_jump_direction not in ("left", "right")
                                and run_jump_enabled
                            ):
                                # 同一次边执行内第一次原地跳已经是一次真实
                                # 失败；第二次先换准备点与接近方式，不再原样
                                # 连按第二次 Alt。
                                retry_run_jump_direction = self._stage_failed_static_grab(
                                    curr_node,
                                    float(trigger_x),
                                    pos_getter,
                                    max(1, action_failure_count + 1),
                                )
                                if bool(getattr(self.motion, "last_walk_priority_interrupted", False)):
                                    self.log_fn(
                                        "⚔️ [跑跳重试准备让行攻击] 走位已中断，"
                                        "本轮禁止继续发送UP+Alt"
                                    )
                                    return
                                retry_world = pos_getter()
                            if retry_run_jump_direction in ("left", "right") and retry_world is not None:
                                retry_run_jump_direction = "right" if trigger_x > float(retry_world[0]) else "left"
                                self.log_fn(
                                    f"🏃 [跑跳重规划] 失败后Kalman坐标="
                                    f"({retry_world[0]:.1f}, {retry_world[1]:.1f})，"
                                    f"绳梯X={trigger_x:.0f}，二次改为向"
                                    f"{('右' if retry_run_jump_direction == 'right' else '左')}跑跳"
                                )
                            grabbed = self.motion.jump_and_grab_ladder(
                                target_ladder_x=trigger_x,
                                get_player_pos=pos_getter,
                                align_player_pos=pos_getter,
                                jump_key=self.jump_key,
                                target_y=first_step.target_y,
                                step_off_direction=next_step_dir,
                                platform_bounds=(curr_node.x_min, curr_node.x_max),
                                approach_tolerance=max(4, grab_tolerance // 2),
                                timeout_sec=4.0,
                                stop_event=self.stop_event,
                                get_raw_player_pos=self.get_player_raw_world_pos,
                                target_type=t_type,
                                enable_jump_steering=False,
                                landing_confirm_fn=raw_landing_confirmed,
                                source_standing_check=raw_still_on_grab_source,
                                # 第一次跑跳失败后的重试仍保持跑跳；若第二次
                                # 也失败，失败计数达到 2，下一轮自动降级。
                                run_jump_direction=retry_run_jump_direction,
                                run_jump_sec=0.15,
                                ladder_bottom_y=(max(rope_node.y1, rope_node.y2) if rope_node is not None else None),
                                run_jump_profile=run_jump_profile,
                                use_model_static_approach=True,
                                top_hold_sec=float(self.config.get("climb_top_hold_sec", 1.0)),
                            )
                            if bool(getattr(self.motion, "last_top_exit_failed", False)):
                                self.log_fn(
                                    "⛔ [绳顶恢复待续] 二次抓取已到顶但尚未物理落台；"
                                    "禁止继续降级或重跳"
                                )
                                return
                            if (
                                not grabbed
                                and bool(getattr(self.motion, "last_jump_gate_aborted", False))
                            ):
                                self.log_fn(
                                    "🔁 [跑跳重试起跳线重规划] Alt未发出，不计抓绳失败；"
                                    "按当前实时坐标重新派发原边"
                                )
                                return
                            if grabbed:
                                self._run_jump_failures.pop(run_jump_key, None)
                            elif run_jump_direction in ("left", "right"):
                                if self.reset_motion_prediction:
                                    self.reset_motion_prediction()
                                self._run_jump_failures[run_jump_key] = run_jump_failures + 1

                            if (
                                not grabbed
                                and abandon_old_grab_edge_if_source_lost("原地降级")
                            ):
                                return

                        # 同一次抵达起点平台内，跑跳连续失败两次后立即
                        # 执行第三次原地跳抓；不能等下一轮回到起点再降级，
                        # 因为回到起点会按策略清零计数。
                        if (
                            not grabbed
                            and not self.stop_event.is_set()
                            and run_jump_fallback_enabled
                            and run_jump_direction in ("left", "right")
                            and self._run_jump_failures.get(run_jump_key, 0) >= run_jump_failure_limit
                        ):
                            if abandon_old_grab_edge_if_source_lost("原地降级"):
                                return
                            self.log_fn(
                                f"🔁 [跑跳降级] P{curr_node.id} -> P{target_pid} "
                                f"跑跳已失败 {run_jump_failure_limit} 次，本次改为绳梯下方原地跳抓"
                            )
                            # 外侧梯绳无法站到立柱 X；原地抓取的物理起跳点
                            # 必须是当前平台朝向绳梯的安全边缘，而不是把
                            # 预测坐标/旧的绳坐标直接当作可站立目标。
                            self.log_fn(
                                f"📍 [原地抓取对齐] 绳梯X={trigger_x}，"
                                "由控制器夹取可站起跳点并一次反推按键时间"
                            )
                            grabbed = self.motion.jump_and_grab_ladder(
                                target_ladder_x=trigger_x,
                                get_player_pos=pos_getter,
                                align_player_pos=pos_getter,
                                jump_key=self.jump_key,
                                target_y=first_step.target_y,
                                step_off_direction=next_step_dir,
                                platform_bounds=(curr_node.x_min, curr_node.x_max),
                                approach_tolerance=max(4, grab_tolerance // 2),
                                timeout_sec=4.0,
                                stop_event=self.stop_event,
                                get_raw_player_pos=self.get_player_raw_world_pos,
                                target_type=t_type,
                                enable_jump_steering=False,
                                landing_confirm_fn=raw_landing_confirmed,
                                source_standing_check=raw_still_on_grab_source,
                                run_jump_direction=None,
                                use_model_static_approach=True,
                                top_hold_sec=float(self.config.get("climb_top_hold_sec", 1.0)),
                            )
                            if bool(getattr(self.motion, "last_top_exit_failed", False)):
                                self.log_fn(
                                    "⛔ [绳顶恢复待续] 原地抓取已到顶但尚未物理落台；"
                                    "禁止继续发送Alt"
                                )
                                return
                            if grabbed:
                                self._run_jump_failures.pop(run_jump_key, None)
                        # A confirmed grab is not yet a completed platform
                        # transition. Wait until the minimap/platform resolver
                        # reports the destination platform before the FSM can
                        # select the next patrol edge.
                        if not self.stop_event.is_set():
                            if grabbed:
                                # 攀爬结束后不要立刻让路线规划器执行下一段。
                                # 角色刚翻上平台时身体仍贴着绳子，若下一段是
                                # DOWN_JUMP，极易再次从绳子位置触发下跳并回到
                                # 原绳。先沿平台离开绳子约 50px，再交给导航。
                                next_is_down_jump = (
                                    len(path) > 1
                                    and str(getattr(path[1], "action", "")) == "DOWN_JUMP"
                                )
                                if next_is_down_jump and rope_node is not None and dest_node is not None:
                                    self._random_climb_clear(dest_node, pos_getter, "脱绳避让")
                                landing_pid = getattr(dest_node, "id", target_pid)
                                if getattr(self.motion, "last_landing_confirmed", False):
                                    self.log_fn(f"✅ [平台确认] 已落到本段平台 P{landing_pid}，允许继续下一段")
                                else:
                                    self.log_fn("⏳ [抓取确认] 等待平台状态收敛，下一轮不重复起跳")
                            else:
                                landing_pid = getattr(dest_node, "id", target_pid)
                                self.log_fn(f"⚠️ [落地未确认] 尚未稳定落入本段平台 P{landing_pid}，下一轮保持当前跨层目标")
                        return

                    # 2. 地面贴合梯绳直接攀爬 (CLIMB_UP / CLIMB_DOWN)
                    elif "CLIMB" in first_step.action:
                        trigger_x = first_step.trigger_x or curr_node.center_x
                        climb_dir = "up" if "UP" in first_step.action else "down"
                        self.log_fn(f"🪜 [梯绳攀爬] P{curr_node.id} 走向 X={trigger_x} 吸附攀爬 ({climb_dir})...")
                        direct_climbed = self.motion.walk_and_grab_ladder(
                            target_ladder_x=trigger_x,
                            get_player_pos=pos_getter,
                            climb_direction=climb_dir,
                            target_y=first_step.target_y,
                            # 下爬只按目标平台的实际安全侧脱绳；面向下一跳
                            # 的 lookahead 方向可能把人推向梯轴另一边。
                            step_off_direction=(step_dir if climb_dir == "down" else next_step_dir),
                            platform_bounds=(curr_node.x_min, curr_node.x_max),
                            approach_tolerance=18,
                            timeout_sec=4.0,
                            stop_event=self.stop_event,
                            is_climbing_fn=self.get_is_climbing,
                            top_hold_sec=float(self.config.get("climb_top_hold_sec", 1.0)),
                            get_raw_player_pos=self.get_player_raw_world_pos,
                            landing_confirm_fn=raw_landing_confirmed,
                        )
                        if bool(getattr(self.motion, "last_top_exit_failed", False)):
                            self.log_fn(
                                "⛔ [绳顶恢复待续] 直爬已到顶但尚未物理落台；"
                                "禁止降级为跳抓"
                            )
                            return
                        if not direct_climbed and climb_dir == "down":
                            self.log_fn(
                                f"⚠️ [下爬未完成] P{curr_node.id}->P{target_pid} "
                                "未取得目标平台落台证据；禁止改为跳抓，交还巡逻核验"
                            )
                            return
                        if not direct_climbed and not self.stop_event.is_set():
                            # 直爬越过两格小地图像素仍未吸附，说明底端空隙
                            # 或实际碰撞高度比图数据更大；立刻降级为跳+上抓取。
                            # 降级动作必须继续遵循 UI 的“启用跑跳抓”选项；
                            # 旧代码没有传 run_jump_direction，导致这里无条件
                            # 退回原地跳抓。
                            t_type = "rope" if "ROPE" in first_step.action else "ladder"
                            fallback_run_direction = None
                            fallback_profile = None
                            fallback_pos = pos_getter()
                            if (
                                climb_dir == "up"
                                and run_jump_enabled
                                and fallback_pos is not None
                                and abs(float(trigger_x) - float(fallback_pos[0])) > 8.0
                            ):
                                fallback_run_direction = (
                                    "right" if float(trigger_x) > float(fallback_pos[0]) else "left"
                                )
                            self.log_fn(
                                f"🔁 [直爬降级跳抓] P{curr_node.id} -> P{target_pid} "
                                f"直爬未吸附，改为"
                                f"{'跑跳' if fallback_run_direction else '原地跳'}抓{t_type}"
                            )
                            direct_climbed = self.motion.jump_and_grab_ladder(
                                target_ladder_x=trigger_x,
                                get_player_pos=pos_getter,
                                align_player_pos=pos_getter,
                                jump_key=self.jump_key,
                                target_y=first_step.target_y,
                                step_off_direction=next_step_dir,
                                platform_bounds=(curr_node.x_min, curr_node.x_max),
                                approach_tolerance=6 if t_type == "rope" else 12,
                                timeout_sec=4.0,
                                stop_event=self.stop_event,
                                get_raw_player_pos=self.get_player_raw_world_pos,
                                target_type=t_type,
                                enable_jump_steering=False,
                                run_jump_direction=fallback_run_direction,
                                run_jump_sec=0.15,
                                ladder_bottom_y=(max(rope_node.y1, rope_node.y2) if rope_node is not None else None),
                                run_jump_profile=fallback_profile,
                                use_model_static_approach=True,
                                top_hold_sec=float(self.config.get("climb_top_hold_sec", 1.0)),
                                landing_confirm_fn=raw_landing_confirmed,
                            )
                            if bool(getattr(self.motion, "last_top_exit_failed", False)):
                                self.log_fn(
                                    "⛔ [绳顶恢复待续] 降级跳抓已到顶但尚未物理落台；"
                                    "禁止继续发送Alt"
                                )
                                return
                        # 已经吸附并持续上绳后，如果下一段就是 DOWN_JUMP，
                        # 必须先离开绳子约 50px；否则下跳触发点仍贴着绳子，
                        # 会再次落回同一根绳子。
                        next_is_down_jump = (
                            len(path) > 1
                            and str(getattr(path[1], "action", "")) == "DOWN_JUMP"
                        )
                        climb_rope = (
                            g.get_ladder_rope(first_step.ladder_id)
                            if first_step.ladder_id else None
                        )
                        landed_pos = pos_getter()
                        if (climb_dir == "up" and direct_climbed and next_is_down_jump
                                and climb_rope is not None and landed_pos is not None):
                            self._random_climb_clear(dest_node, pos_getter, "脱绳避让")
                        return

                    # 3. 同层相邻平台步行连接 (WALK_LEFT / WALK_RIGHT)
                    # WALK_LEFT_DROP / WALK_RIGHT_DROP 虽以 WALK_ 开头，
                    # 但属于“走出边缘自然下落”，必须交给后面的 DROP
                    # 执行器，不能在这里被当作普通同层步行截获。
                    if first_step.action.startswith("WALK_") and "DROP" not in first_step.action:
                        if dest_node is None:
                            return
                        walk_dir = "right" if first_step.action == "WALK_RIGHT" else "left"
                        # 合并连续 WALK 边：相邻小平台只是拓扑切分，不应每个
                        # 节点都停一次。一直走到下一条非 WALK 动作前的平台。
                        walk_target_node = dest_node
                        walk_path_nodes = [curr_node, dest_node]
                        for follow in path[1:]:
                            # 只有同方向的连续步行边才能合并；遇到 WALK_LEFT
                            # -> WALK_RIGHT 方向反转必须在中间节点重新规划。
                            if follow.action != first_step.action:
                                break
                            follow_node = g.get_node(follow.to_id)
                            if follow_node is None:
                                break
                            walk_target_node = follow_node
                            walk_path_nodes.append(follow_node)
                        target_x = walk_target_node.center_x
                        walk_bounds = (
                            min(n.x_min for n in walk_path_nodes),
                            max(n.x_max for n in walk_path_nodes),
                        )
                        self.log_fn(
                            f"🚶 [同层步行] P{curr_node.id} -> P{walk_target_node.id} "
                            f"方向={walk_dir}，目标X={target_x}"
                        )
                        self.motion.walk_to_x(
                            target_x=target_x,
                            get_player_pos=pos_getter,
                            # 绳梯底端平台的最终对齐需要小于黄点量化格，
                            # 由预测松键刹停收束到目标附近，而不是 12px 提前停住。
                            tolerance=5,
                            timeout_sec=3.0,
                            stop_event=self.stop_event,
                            platform_bounds=walk_bounds,
                            safe_margin=8,
                        )
                        return

                    # 4. 水平助跑跳跃与垂直上跳/瞬移 (JUMP_UP / TELEPORT_UP)
                    elif first_step.action in ("JUMP_UP", "TELEPORT_UP"):
                        # 垂直跳/向上瞬移也必须先到达拓扑计算出的起跳点。
                        # 旧逻辑直接 jump_step(None)，角色刚落回起点时
                        # 会在当前位置反复原地起跳，完全绕过起跳准备。
                        landing_node = g.get_node(first_step.to_id)
                        if landing_node is None:
                            return
                        has_foothold_plan = first_step.source_foothold_id is not None
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.APPROACHING, first_step
                            )
                        if not self._prepare_vertical_jump(
                            curr_node, landing_node, pos_getter, px, first_step,
                            action_retry_count=action_failure_count,
                        ):
                            # 仅对齐未完成时没有发出技能键，不能把它计作
                            # 一次真实动作失败；下一帧按新 raw 偏差继续闭环。
                            self.motion.last_jump_gate_aborted = True
                            return
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.READY, first_step
                            )
                        use_teleport = (first_step.action == "TELEPORT_UP")
                        if not use_teleport and getattr(self.motion, "enable_teleport", False):
                            source_y = curr_node.surface_y_at(px)
                            target_y = landing_node.surface_y_at(px)
                            if source_y - target_y > 75.0:
                                use_teleport = True

                        act_label = "向上瞬移" if use_teleport else "垂直上跳"
                        self.log_fn(
                            f"🚀 [{act_label}] P{curr_node.id} -> P{dest_node.id if dest_node else '?'} "
                            f"落点平台P{landing_node.id}"
                        )
                        raw_takeoff = (
                            self.get_player_raw_world_pos()
                            if self.get_player_raw_world_pos else None
                        )
                        self.log_fn(
                            f"📐 [{act_label}诊断] P{curr_node.id}->P{landing_node.id} "
                            f"计划X={first_step.trigger_x}，"
                            f"安全区={first_step.trigger_x_range}，动作前raw={raw_takeoff}"
                        )
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.EXECUTING, first_step
                            )
                        if use_teleport:
                            self.motion.teleport("up")
                            time.sleep(0.40)
                        else:
                            jump_started = self.motion.jump_step(
                                direction=None,
                                jump_key=self.jump_key,
                                wait_land_sec=0.50,
                            )
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.VERIFYING, first_step
                            )
                        return

                    elif any(k in first_step.action for k in ("JUMP_RIGHT", "JUMP_LEFT", "DROP")):
                        is_right = ("RIGHT" in first_step.action)
                        is_drop = ("DROP" in first_step.action)
                        has_foothold_plan = first_step.source_foothold_id is not None

                        def raw_still_on_jump_source() -> bool:
                            # 承重平台编号可能在坠落后滞后一帧以上。Alt 的
                            # 最终保险直接用 raw 脚底高度核对源平台曲面，
                            # 防止已经落到下层平台仍执行上一条普通跳。
                            raw = (
                                self.get_player_raw_world_pos()
                                if self.get_player_raw_world_pos else None
                            )
                            if raw is None:
                                return True
                            raw_x, raw_y = float(raw[0]), float(raw[1])
                            # 普通横跳的计划起跳线本来就在边缘内约10px。
                            # raw 一旦已越过真实边界，再补发 Alt 会直接落到
                            # 下层（P24->P26 的 -710、P24->P27 的 -454）。
                            # 因此这里只允许仍在源平台真实 X 边界内的样本。
                            if not (curr_node.x_min <= raw_x <= curr_node.x_max):
                                return False
                            surface_x = min(curr_node.x_max, max(curr_node.x_min, raw_x))
                            expected_foot_y = float(curr_node.surface_y_at(surface_x)) - 45.0
                            # 107000100 的 raw 小地图 Y 会以约27px整格量化：
                            # P26 实际站立中心 -824 可能报告为 -851。旧的
                            # 26px 门槛恰好差1px，导致人物明明静止在 P26，
                            # 每轮都在 Alt 前被判为“源平台丢失”。允许一个
                            # 完整量化格；X 仍须位于源平台范围，超过一格的
                            # 起跳/下落位移仍会被拒绝。容差由“特殊参数”
                            # 实时读取，默认50px。
                            try:
                                raw_y_tolerance = float(
                                    self.config.get("jump_source_raw_y_tolerance_px", 65.0)
                                )
                            except (TypeError, ValueError):
                                raw_y_tolerance = 65.0
                            return abs(raw_y - expected_foot_y) <= raw_y_tolerance

                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.APPROACHING, first_step
                            )
                        self.log_fn(
                            f"🔎 [动作诊断] 当前P{curr_node.id}边界="
                            f"[{curr_node.x_min},{curr_node.x_max}]，"
                            f"当前X={px:.1f}，本步目标P{getattr(dest_node, 'id', target_pid)}，"
                            f"巡逻目标P{target_pid}，动作={first_step.action}，"
                            f"触发区={first_step.trigger_x_range or first_step.trigger_x}"
                        )
                        if first_step.action.startswith("WALK_"):
                            dir_name = "right" if is_right else "left"
                            if dest_node is None:
                                return
                            def landed_on_drop_target() -> bool:
                                observed = self.get_current_platform() if self.get_current_platform else None
                                return bool(observed is not None and observed.id == dest_node.id)
                            hold_sec, landing_x = self.motion.walk_off_drop(
                                dir_name,
                                start_x=px,
                                source_bounds=(curr_node.x_min, curr_node.x_max),
                                source_y=curr_node.y,
                                target_bounds=(dest_node.x_min, dest_node.x_max),
                                target_y=dest_node.y,
                                stop_event=self.stop_event,
                                landed_on_target_fn=landed_on_drop_target,
                                # P22 -> P9 一类：自动循环会在下一帧继续
                                # 规划，窄目标平台必须先消掉残余横向速度。
                                brake_on_landing=dest_node.length <= 180,
                                get_player_pos=pos_getter,
                            )
                            self.log_fn(
                                f"🚶 [自然下落] P{curr_node.id} -> P{dest_node.id} "
                                f"方向={dir_name}，预测按键={hold_sec:.3f}s，落点 X={landing_x:.1f}"
                            )
                            return
                        plen = curr_node.length
                        offset = min(50, max(15, int(plen * 0.35)))
                        safe_margin = max(12, min(25, int(offset * 0.5)))
                        launch_margin = 8  # 给视觉/输入延迟预留的边缘安全余量
                        # 长距离跨层跳需要更长的助跑；普通窄平台跳保持较短
                        # 助跑以免在起跳前越过边缘。
                        planned_run_up_sec = 0.20 if "LONG_DROP" in first_step.action else 0.22
                        # 普通跳准备区额外预留 5px 助跑距离，确保起跳时
                        # 已建立足够的水平速度；LONG_DROP 后续会使用
                        # 独立的动态助跑计算，不重复叠加该偏移。
                        planned_run_distance = 125.0 * planned_run_up_sec + (
                            5.0 if ("LONG_DROP" not in first_step.action and plen < 60) else 0.0
                        )
                        # 窄台、绳梯前的准备点需要精确；普通的宽台→宽台跑跳
                        # 则不应被小地图的一格量化坐标卡死。后者只需处于一格
                        # 原始黄点的可助跑区内，起跳时的连续助跑会补足余量。
                        # 否则像 P61→P62 这种正常跑跳会在距准备点十几像素处
                        # 退化成 15% 短按，长期无法进入原先的 ±9px 硬门槛。
                        narrow_source_for_prep = curr_node.length < 90
                        narrow_dest_for_prep = bool(
                            dest_node is not None and dest_node.length < 90
                        )
                        # 窄台→窄台本身没有足够空间容纳完整 0.22s 的
                        # 反向准备距离。保留跳跃时的助跑时序，但把“先反向
                        # 走到准备点”的距离压到 24px，避免 P11→P17 这类
                        # 相邻窄台先退得过远、再折返。
                        prep_backtrack_distance = planned_run_distance
                        if (
                            narrow_source_for_prep
                            and narrow_dest_for_prep
                            and "LONG_DROP" not in first_step.action
                        ):
                            # 16px 会让 P9→P13 这类窄台跳跃没有足够的
                            # 起跳前速度；24px 仍比原 32.5px 明显更短。
                            prep_backtrack_distance = min(24.0, planned_run_distance)
                            self.log_fn(
                                f"🏃 [窄台短准备] P{curr_node.id} -> P{getattr(dest_node, 'id', target_pid)} "
                                f"反向准备 {planned_run_distance:.1f}px -> {prep_backtrack_distance:.1f}px"
                            )
                        # 极深的 LONG_DROP（例如 P22 -> P1，ΔY=660）只需
                        # 从窄台建立连续水平速度，不需要、也不应该在准备点
                        # 以 15% 脉冲精调。保留 P26 -> P10 等中距离窄台
                        # 长跳（约 ΔY=420）的精调；P22 -> P9 是独立自然
                        # 下落边，不受本规则影响。
                        vertical_drop_for_prep = abs(
                            float((first_step.target_y if first_step.target_y is not None else (dest_node.y if dest_node is not None else curr_node.y)))
                            - float(curr_node.y)
                        )
                        relaxed_deep_long_drop = bool(
                            "LONG_DROP" in first_step.action and vertical_drop_for_prep >= 500.0
                        )
                        # “窄平台30%短按”是未合并短平台时代的兼容策略。
                        # 只有源平台本身很短时才保留；长源平台即使目标平台
                        # 很短（如 107000100 的 P7->P11），也应满速连续走到
                        # 助跑点，不能因目标短而把整段准备移动变成脉冲短按。
                        # 新的 foothold 计划已有安全起跳区，同样不走旧短按。
                        precision_prep = bool(
                            not has_foothold_plan
                            and narrow_source_for_prep
                            and not relaxed_deep_long_drop
                        )
                        if relaxed_deep_long_drop:
                            self.log_fn(
                                f"🏃 [深长跳连续助跑] P{curr_node.id} -> P{getattr(dest_node, 'id', target_pid)} "
                                f"ΔY={vertical_drop_for_prep:.0f}px，取消窄台短按精调"
                            )
                        if has_foothold_plan and first_step.takeoff_x_range is not None:
                            takeoff_width = max(
                                0.0,
                                float(first_step.takeoff_x_range[1])
                                - float(first_step.takeoff_x_range[0]),
                            )
                            # 准备点位于实际起跳区反方向一个助跑距离处。
                            # 容差不能大于起跳区半宽，否则连续助跑后可能在
                            # 安全区外按下跳跃键。
                            prep_tolerance = max(6, min(12, int(takeoff_width * 0.45)))
                        else:
                            prep_tolerance = 9 if precision_prep else 18
                        planning_pos = pos_getter()
                        planning_x = float(planning_pos[0]) if planning_pos is not None else float(px)

                        if is_right:
                            # 优先采用拓扑边计算出的实际起跳触发线；固定的
                            # “距边缘 19px”会让 P26->P10 提前约 14px 起跳。
                            if "LONG_DROP" in first_step.action:
                                # 大跳先回平台左半段的中值蓄跑。完全由平台自身
                                # 边界和中点推导，不依赖某个地图/平台的离散表。
                                target_run_x = (curr_node.x_min + curr_node.center_x) / 2.0 + 8.0
                            else:
                                # 宽平台跳到窄平台时，以目标窄平台中点为
                                # 落点，反推 Alt 起跳点和助跑准备点。
                                narrow_target = bool(
                                    not has_foothold_plan
                                    and
                                    dest_node is not None
                                    and dest_node.length < 60
                                    and curr_node.length > dest_node.length
                                )
                                if narrow_target:
                                    post_alt_sec = 0.08 + 0.20
                                    launch_x = dest_node.center_x - 125.0 * post_alt_sec
                                    target_run_x = launch_x - prep_backtrack_distance
                                    self.log_fn(
                                        f"🎯 [窄平台中点反推] P{curr_node.id} -> P{target_pid} "
                                        f"落点中值={dest_node.center_x}，起跳X={launch_x:.1f}，准备X={target_run_x:.1f}"
                                    )
                                else:
                                    target_run_x = ((first_step.trigger_x - prep_backtrack_distance)
                                                    if first_step.trigger_x is not None
                                                    else (curr_node.x_max - offset + 12))
                            target_run_x = min(curr_node.x_max - 4, max(curr_node.x_min + 4, target_run_x))
                            need_walk = (not relaxed_deep_long_drop) and abs(planning_x - target_run_x) > prep_tolerance

                        else:
                            if "LONG_DROP" in first_step.action:
                                # 向左大跳对称地先回平台右半段的中值。
                                target_run_x = (curr_node.center_x + curr_node.x_max) / 2.0 - 8.0
                            else:
                                narrow_target = bool(
                                    not has_foothold_plan
                                    and
                                    dest_node is not None
                                    and dest_node.length < 60
                                    and curr_node.length > dest_node.length
                                )
                                if narrow_target:
                                    post_alt_sec = 0.08 + 0.20
                                    launch_x = dest_node.center_x + 125.0 * post_alt_sec
                                    target_run_x = launch_x + prep_backtrack_distance
                                    self.log_fn(
                                        f"🎯 [窄平台中点反推] P{curr_node.id} -> P{target_pid} "
                                        f"落点中值={dest_node.center_x}，起跳X={launch_x:.1f}，准备X={target_run_x:.1f}"
                                    )
                                else:
                                    target_run_x = (first_step.trigger_x + prep_backtrack_distance) if first_step.trigger_x is not None else (curr_node.x_min + offset)
                            target_run_x = min(curr_node.x_max - 4, max(curr_node.x_min + 4, target_run_x))
                            need_walk = (not relaxed_deep_long_drop) and abs(planning_x - target_run_x) > prep_tolerance

                        reverse_narrow_prep = False
                        if has_foothold_plan and first_step.takeoff_x_range is not None:
                            gate_width = abs(
                                float(first_step.takeoff_x_range[1])
                                - float(first_step.takeoff_x_range[0])
                            )
                            step_px = float(
                                getattr(getattr(self.motion, "motion_model", None),
                                        "measurement_step_px", 16.0) or 16.0
                            )
                            revised_x, reverse_narrow_prep = self._reverse_runup_preparation(
                                target_run_x, planning_x,
                                right_jump=is_right,
                                gate_width=gate_width,
                                measurement_step=step_px,
                                tolerance=prep_tolerance,
                                source_bounds=(curr_node.x_min, curr_node.x_max),
                                safe_margin=max(safe_margin, 12),
                            )
                            if reverse_narrow_prep:
                                self.log_fn(
                                    f"🔁 [窄窗反向备跑] 准备点X={target_run_x:.1f}"
                                    f"→{revised_x:.1f}，量化步长={step_px:.1f}px，"
                                    f"起跳窗宽={gate_width:.1f}px；先刹停再同向助跑"
                                )
                                target_run_x = revised_x
                                need_walk = abs(planning_x - target_run_x) > prep_tolerance

                        if relaxed_deep_long_drop:
                            self.log_fn(
                                f"🏃 [深长跳跳过准备区] P{curr_node.id} -> P{target_pid}："
                                "不执行窄台对齐/短按，直接连续助跑起跳"
                            )

                        # 普通跳与 LONG_DROP 都先定位到起跳准备区；普通跳
                        # 到位后再由 jump_step 执行连续助跑和起跳。
                        prep_approach_distance = abs(float(planning_x) - float(target_run_x))
                        # 无缝交接只适合已经靠近准备区的短距离助跑。若从平台
                        # 另一端赶来仍一路按住方向键，角色会以最高速度穿过准备
                        # 点，甚至在 jump_step 接管前直接冲出短平台边缘。P27->P24
                        # 与 P26->P24 都属于这种数百像素的远距离接近。
                        max_seamless_approach = 160.0
                        seamless_runup = bool(
                            not precision_prep
                            and not reverse_narrow_prep
                            and "LONG_DROP" not in first_step.action
                            and not is_drop
                            and prep_approach_distance <= max_seamless_approach
                        )
                        if (
                            not precision_prep
                            and "LONG_DROP" not in first_step.action
                            and not is_drop
                            and prep_approach_distance > max_seamless_approach
                        ):
                            self.log_fn(
                                f"🛑 [长距离助跑分段] 距准备点{prep_approach_distance:.1f}px > "
                                f"{max_seamless_approach:.0f}px；先在X={target_run_x:.1f}刹停，"
                                "再执行短距离连续助跑"
                            )
                        prep_handoff = False
                        if need_walk:
                            # 低占空比仅用于窄平台，或已接近准备点的最终
                            # 微调。旧逻辑把所有普通跳都设为低占空比，像
                            # P55->P59 这类宽平台、相距百余像素的正常跑跳
                            # 会变成连续短按，既慢也攒不出助跑速度。
                            narrow_source = narrow_source_for_prep
                            # 30% 脉冲只属于窄平台的防过冲微调。宽台之间
                            # 以及长台跳短台的普通跑跳必须保持满速，才能
                            # 连续建立助跑速度。
                            prep_speed_scale = (
                                0.30
                                if (
                                    not has_foothold_plan
                                    and narrow_source
                                    and not relaxed_deep_long_drop
                                )
                                else 1.0
                            )
                            if has_foothold_plan:
                                self.log_fn(
                                    f"🏃 [短段计划连续助跑] fh{first_step.source_foothold_id}"
                                    f"->fh{first_step.target_foothold_id}，准备点X={target_run_x:.1f}，"
                                    f"容差={prep_tolerance}，不使用旧窄台短按"
                                )
                            # 低占空比准备区的有效速度显著低于满速，超时必须
                            # 与本次实际占空比一致，避免远距离准备点反复超时。
                            prep_timeout = max(
                                2.0,
                                min(
                                    7.0,
                                    abs(float(target_run_x) - planning_x)
                                    / max(20.0, 125.0 * prep_speed_scale)
                                    + 1.0,
                                ),
                            )
                            self.log_fn(
                                f"🚶 [助跑准备] 走向起跳蓄力区 X={target_run_x} "
                                f"(预测X={px:.1f}，控制rawX={planning_x:.1f}，"
                                f"差值={target_run_x - planning_x:+.1f}，"
                                f"平台边界=[{curr_node.x_min},{curr_node.x_max}]，"
                                f"容差={prep_tolerance}，占空比={prep_speed_scale:.2f}，"
                                f"超时={prep_timeout:.1f}s)"
                            )
                            prep_arrived = self.motion.walk_to_x(
                                target_x=target_run_x,
                                get_player_pos=pos_getter,
                                tolerance=prep_tolerance,
                                timeout_sec=prep_timeout,
                                stop_event=self.stop_event,
                                platform_bounds=(curr_node.x_min, curr_node.x_max),
                                safe_margin=max(safe_margin, 12),
                                # 准备区采用更慢的脉冲，降低超调概率。
                                speed_scale=prep_speed_scale,
                                # 宽平台普通左右跳不在准备点松键，直接把
                                # 已建立的同向速度交给 jump_step。
                                preserve_direction_on_arrival=seamless_runup,
                            )
                            latest_prep = pos_getter()
                            prep_confirmed = bool(
                                latest_prep is not None
                                and abs(latest_prep[0] - target_run_x) <= prep_tolerance
                            )
                            prep_ready = bool(prep_arrived and prep_confirmed)
                            if not prep_ready:
                                if seamless_runup:
                                    self.motion.stop()
                                self.log_fn(
                                    f"⏸️ [准备未到位] 当前 X={latest_prep[0] if latest_prep else None}，"
                                    f"目标 X={target_run_x}，预测到位={prep_arrived}，"
                                    f"Kalman确认={prep_confirmed}(±{prep_tolerance:.0f}px)，"
                                    f"本轮不启动跳跃"
                                )
                                return
                            prep_handoff = seamless_runup

                        # 没有从 walk_to_x 带着方向键交接时，角色此刻应为静止。
                        # 用最新 raw 黄点显式重置连续运动模型，清除长距离赶路、
                        # 战斗击退或攻击中断留下的 18~20px 预测漂移。否则连续 X
                        # 会先于/晚于人物跨过起跳线，表现为过早 Alt 或原地死等。
                        if (
                            not prep_handoff
                            and "LONG_DROP" not in first_step.action
                            and not is_drop
                            and self.reset_motion_prediction is not None
                        ):
                            self.reset_motion_prediction()
                            reset_raw = (
                                self.get_player_raw_world_pos()
                                if self.get_player_raw_world_pos else None
                            )
                            self.log_fn(
                                f"📍 [起跳坐标重锚] 非无缝起跑，按最新raw={reset_raw} "
                                "清零旧水平速度后重新助跑"
                            )

                        dir_name = "right" if is_right else "left"
                        height_drop = max(0, (first_step.target_y or curr_node.y) - curr_node.y)
                        if "LONG_DROP" in first_step.action:
                            # 按完整起跳-上升-下落模型计算方向键保持时间：
                            # 先以 -555px/s 上升，再加速下落并受 670px/s
                            # 终端速度限制，不能把角色当作从静止开始自由落体。
                            gravity = 2000.0
                            jump_v0 = 555.0
                            terminal_v = 670.0
                            peak_time = jump_v0 / gravity
                            peak_height = (jump_v0 * jump_v0) / (2.0 * gravity)
                            terminal_t = terminal_v / gravity
                            terminal_y = 0.5 * gravity * terminal_t * terminal_t
                            descend_from_peak = peak_height + height_drop
                            if descend_from_peak <= terminal_y:
                                airborne_sec = peak_time + math.sqrt(2.0 * descend_from_peak / gravity)
                            else:
                                airborne_sec = (
                                    peak_time + terminal_t
                                    + (descend_from_peak - terminal_y) / terminal_v
                                )
                            self.log_fn(
                                f"🎯 [长跳物理参数] P{curr_node.id} -> P{target_pid} "
                                f"方向保持={airborne_sec:.3f}s (ΔY={height_drop})"
                            )
                        else:
                            # 普通左右跳：Alt 起跳后继续保持方向 0.2s；
                            # 斜向下落跳仍使用独立的 0.45s 参数。
                            airborne_sec = 0.45 if is_drop else 0.20
                        run_up_sec = planned_run_up_sec
                        if "LONG_DROP" in first_step.action and first_step.trigger_x is not None:
                            prep_start_x = latest_prep[0] if 'latest_prep' in locals() and latest_prep else px
                            # 助跑只计算到安全起跳线（触发线前预留余量），
                            # 不能跑到拓扑触发线本身，否则 Alt 触发前可能
                            # 已越过平台边缘。
                            launch_x = first_step.trigger_x - launch_margin if is_right else first_step.trigger_x + launch_margin
                            run_distance = abs(launch_x - prep_start_x)
                            run_up_sec = max(0.24, min(0.34, run_distance / 125.0 + 0.04))
                            self.log_fn(
                                f"🏃 [动态助跑] 准备 X={prep_start_x} -> 安全起跳线 X={launch_x}，"
                                f"助跑={run_up_sec:.3f}s"
                            )
                        self.log_fn(
                            f"🚀 [跨层起跳] P{curr_node.id} -> "
                            f"P{getattr(dest_node, 'id', target_pid)} "
                            f"(巡逻目标=P{target_pid}，方向={dir_name}, 坠落={is_drop})"
                        )
                        def log_takeoff_sample() -> None:
                            actual = (
                                self.get_player_raw_world_pos()
                                if self.get_player_raw_world_pos else None
                            )
                            planned = (
                                first_step.takeoff_x
                                if first_step.takeoff_x is not None
                                else first_step.trigger_x
                            )
                            sample_x = float(planned if planned is not None else planning_x)
                            travel_sign = 1.0 if is_right else -1.0
                            behind_x = min(
                                curr_node.x_max,
                                max(curr_node.x_min, sample_x - travel_sign * 20.0),
                            )
                            ahead_x = min(
                                curr_node.x_max,
                                max(curr_node.x_min, sample_x + travel_sign * 20.0),
                            )
                            behind_y = curr_node.surface_y_at(behind_x)
                            ahead_y = curr_node.surface_y_at(ahead_x)
                            directional_dy = float(ahead_y - behind_y)
                            terrain = (
                                "下坡" if directional_dy > 2.0
                                else "上坡" if directional_dy < -2.0
                                else "近似平地"
                            )
                            actual_x = float(actual[0]) if actual is not None else None
                            overshoot = (
                                (actual_x - sample_x) * travel_sign
                                if actual_x is not None else None
                            )
                            self.log_fn(
                                f"📐 [起跳坡度诊断] P{curr_node.id}->P{target_pid} "
                                f"计划X={planned}，安全区={first_step.takeoff_x_range or first_step.trigger_x_range}，"
                                f"准备X={target_run_x:.1f}，Alt前raw={actual}，"
                                f"沿{dir_name}40px ΔY={directional_dy:+.1f}({terrain})，"
                                f"越过计划点={overshoot:+.1f}px" if overshoot is not None else
                                f"📐 [起跳坡度诊断] P{curr_node.id}->P{target_pid} "
                                f"计划X={planned}，安全区={first_step.takeoff_x_range or first_step.trigger_x_range}，"
                                f"准备X={target_run_x:.1f}，Alt前raw=None，"
                                f"沿{dir_name}40px ΔY={directional_dy:+.1f}({terrain})"
                            )
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.EXECUTING, first_step
                            )
                        jump_started = self.motion.jump_step(
                            dir_name,
                            jump_key=self.jump_key,
                            run_up_sec=run_up_sec,
                            airborne_hold_sec=airborne_sec,
                            # 普通相邻平台跳不需要固定等待半秒以上；P59→P61
                            # 这类连续台阶在实际落地后会白白停顿。空中方向键
                            # 仍按用户设定保持 0.20s，仅将落地解析缓冲缩短。
                            wait_land_sec=0.85 if "LONG_DROP" in first_step.action else 0.30,
                            direction_already_held=prep_handoff,
                            before_jump_callback=log_takeoff_sample,
                            takeoff_x=(
                                None if "LONG_DROP" in first_step.action
                                else (
                                    first_step.takeoff_x
                                    if first_step.takeoff_x is not None
                                    else first_step.trigger_x
                                )
                            ),
                            takeoff_x_range=(
                                None if "LONG_DROP" in first_step.action
                                else first_step.takeoff_x_range
                            ),
                            # 无短段安全区的旧边仍由连续坐标命中计划线；有
                            # foothold 安全区时则由连续坐标与量化 raw 共同确认
                            # 已进入区间，避免斜坡加速时预测领先而提前起跳。
                            get_player_pos=pos_getter,
                            source_standing_check=raw_still_on_jump_source,
                        )
                        if jump_started is False:
                            if has_foothold_plan:
                                self._set_platform_transition_state(
                                    PlatformTransitionState.IDLE
                                )
                            return
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.VERIFYING, first_step
                            )
                        if "LONG_DROP" in first_step.action:
                            before_relocate = self.get_player_world_pos() if self.get_player_world_pos else None
                            raw_before_relocate = self.get_player_raw_world_pos() if self.get_player_raw_world_pos else None
                            observed_after_jump = self.get_current_platform() if self.get_current_platform else None
                            self.log_fn(
                                f"🔎 [大跳落地诊断] 预期目标P{target_pid}，"
                                f"当前预测={before_relocate}，raw={raw_before_relocate}，"
                                f"当前平台={getattr(observed_after_jump, 'id', None)}，"
                                f"预期直接落台={getattr(dest_node, 'id', None)}"
                            )
                            reverse_dir = "left" if is_right else "right"
                            self.log_fn(
                                f"🔄 [大跳落地重定位] 先向{('左' if is_right else '右')}"
                                f"、再向{('右' if is_right else '左')}微动，占空比=25%"
                            )
                            # 先反向释放落地时的横向残余速度，再向下一跳方向
                            # 轻微刷新视口/坐标；两段合计保持原来的微动时长。
                            self.motion.micro_nudge(
                                reverse_dir,
                                duration_sec=0.175,
                                speed_scale=0.25,
                                stop_event=self.stop_event,
                            )
                            self.motion.micro_nudge(
                                dir_name,
                                duration_sec=0.175,
                                speed_scale=0.25,
                                stop_event=self.stop_event,
                            )
                            self.log_fn("⏳ [大跳落地重定位] 等待 2.0s 让坐标与平台检测收束")
                            self.stop_event.wait(2.0)
                            after_relocate = self.get_player_world_pos() if self.get_player_world_pos else None
                            raw_after_relocate = self.get_player_raw_world_pos() if self.get_player_raw_world_pos else None
                            observed_after_wait = self.get_current_platform() if self.get_current_platform else None
                            self.log_fn(
                                f"🔎 [重定位后诊断] 预测={after_relocate}，raw={raw_after_relocate}，"
                                f"当前平台={getattr(observed_after_wait, 'id', None)}，"
                                f"预期直接落台={getattr(dest_node, 'id', None)}"
                            )
                        return

                    # 4. 垂直台阶连环直跳 / 向上瞬移 (JUMP_UP / TELEPORT_UP)
                    elif first_step.action in ("JUMP_UP", "TELEPORT_UP"):
                        landing_node = g.get_node(first_step.to_id)
                        if landing_node is None:
                            return
                        has_foothold_plan = first_step.source_foothold_id is not None
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.APPROACHING, first_step
                            )
                        if not self._prepare_vertical_jump(
                            curr_node, landing_node, pos_getter, px, first_step,
                            action_retry_count=action_failure_count,
                        ):
                            self.motion.last_jump_gate_aborted = True
                            return
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.READY, first_step
                            )
                        use_teleport = (first_step.action == "TELEPORT_UP")
                        if not use_teleport and getattr(self.motion, "enable_teleport", False):
                            source_y = curr_node.surface_y_at(px)
                            target_y = landing_node.surface_y_at(px)
                            if source_y - target_y > 75.0:
                                use_teleport = True

                        if use_teleport:
                            self.log_fn(f"⚡ [垂直台阶瞬移] P{curr_node.id} 向上瞬移至P{landing_node.id}...")
                        else:
                            self.log_fn(f"🦘 [垂直台阶跳] P{curr_node.id} 向上直跳至P{landing_node.id}...")
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.EXECUTING, first_step
                            )
                        if use_teleport:
                            self.motion.teleport("up")
                            time.sleep(0.40)
                        else:
                            self.motion.jump_step(None, jump_key=self.jump_key, wait_land_sec=0.45)
                        if has_foothold_plan:
                            self._set_platform_transition_state(
                                PlatformTransitionState.VERIFYING, first_step
                            )
                        return

                    # 5. 平台垂直下跳 (DOWN_JUMP)
                    elif first_step.action == "DOWN_JUMP":
                        dest_node = g.get_node(first_step.to_id) if g else None
                        down_key = (int(curr_node.id), int(first_step.to_id))
                        current_pos = pos_getter()
                        current_x = float(current_pos[0]) if current_pos else float(px)
                        if (
                            self._active_down_jump_target is not None
                            and self._active_down_jump_target[0] == down_key[0]
                            and self._active_down_jump_target[1] == down_key[1]
                        ):
                            trigger_x = self._active_down_jump_target[2]
                            lo, hi = self._active_down_jump_target[3]
                        else:
                            trigger_x, (lo, hi) = self._pick_nearest_safe_down_jump_x(
                                curr_node, dest_node, first_step, g, current_x
                            )
                            self._active_down_jump_target = (down_key[0], down_key[1], trigger_x, (lo, hi))

                        if abs(current_x - trigger_x) <= 20.0:
                            self.log_fn(
                                f"✅ [下跳安全点已覆盖] 当前X={current_x:.1f}，"
                                f"最近安全X={trigger_x:.1f}，安全范围=[{lo:.1f},{hi:.1f}]"
                            )
                            self._active_down_jump_target = None
                            self.motion.down_jump(jump_key=self.jump_key)
                            time.sleep(0.5)
                            return

                        self.log_fn(
                            f"🚶 [下跳安全点准备] 当前X={current_x:.1f}，安全范围=[{lo:.1f},{hi:.1f}]，"
                            f"走向最近安全点X={trigger_x:.1f}"
                        )
                        arrived = self.motion.walk_to_x(
                            target_x=trigger_x,
                            get_player_pos=pos_getter,
                            tolerance=10,
                            timeout_sec=max(2.0, min(6.0, abs(trigger_x - current_x) / 105.0 + 1.5)),
                            stop_event=self.stop_event,
                            platform_bounds=(curr_node.x_min, curr_node.x_max),
                            safe_margin=15,
                            speed_scale=1.0,
                        )
                        after = pos_getter()
                        after_x = float(after[0]) if after else current_x
                        if not arrived or abs(after_x - trigger_x) > 30.0:
                            self.log_fn(
                                f"⏸️ [下跳安全点准备未到位] 当前X={after_x:.1f}，"
                                f"目标X={trigger_x:.1f}，安全范围=[{lo:.1f},{hi:.1f}]"
                            )
                            return

                        self._active_down_jump_target = None
                        self.motion.down_jump(jump_key=self.jump_key)
                        time.sleep(0.5)
                        return

        # 2. 次级使用录制航点管理器
        wp = self.waypoints.get_current_waypoint()
        if wp is not None:
            if wp.action == "DOWN_JUMP":
                self.motion.down_jump(jump_key=self.jump_key)
                self.waypoints.advance_next()
                return
            elif wp.action == "JUMP_RIGHT":
                self.motion.jump_step("right", jump_key=self.jump_key)
                self.waypoints.advance_next()
                return
            elif wp.action == "JUMP_LEFT":
                self.motion.jump_step("left", jump_key=self.jump_key)
                self.waypoints.advance_next()
                return
            else:
                # 平走至航点
                arrived = self.motion.walk_to_x(
                    target_x=wp.x,
                    get_player_pos=pos_getter,
                    tolerance=getattr(wp, 'tolerance', 20),
                    timeout_sec=1.5,
                    stop_event=self.stop_event
                )
                if arrived or abs(wp.x - px) <= getattr(wp, 'tolerance', 20):
                    if getattr(wp, 'wait_time_sec', 0) > 0:
                        time.sleep(wp.wait_time_sec)
                    self.waypoints.advance_next()
                return

        # 3. 单平台巡逻 (如果未指定多平台巡逻列表)
        if curr_node:
            safe_min_x = curr_node.x_min + min(25, max(5, curr_node.length // 4))
            safe_max_x = curr_node.x_max - min(25, max(5, curr_node.length // 4))
            if safe_min_x < safe_max_x:
                target_x = safe_max_x if self.patrol_dir > 0 else safe_min_x
                arrived = self.motion.walk_to_x(
                    target_x=target_x,
                    get_player_pos=pos_getter,
                    tolerance=20,
                    timeout_sec=1.2,
                    stop_event=self.stop_event
                )
                if arrived or abs(px - target_x) <= 30:
                    self.patrol_dir *= -1
                return

        # 4. 兜底方案：在当前位置附近安全游走巡逻
        patrol_target_x = px + (120 * self.patrol_dir)
        arrived = self.motion.walk_to_x(
            target_x=patrol_target_x,
            get_player_pos=pos_getter,
            tolerance=25,
            timeout_sec=0.6,
            stop_event=self.stop_event
        )
        if arrived or abs(px - patrol_target_x) <= 30:
            self.patrol_dir *= -1
        if arrived:
            self.patrol_dir *= -1
