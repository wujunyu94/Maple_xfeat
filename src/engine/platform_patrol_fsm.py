"""Persistent platform patrol state machine built on merged-platform routes.

The combat FSM owns target selection and combat.  This class owns only platform
route lifecycle: observe -> plan -> dispatch one edge -> verify -> recover.
It deliberately never uses viewport pixel coordinates as world coordinates.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ACTION_PENALTY_AFTER = 3
ACTION_EXCLUDE_AFTER = 4
CLIMB_FAILURE_LIMIT = ACTION_EXCLUDE_AFTER


class PatrolPhase(Enum):
    IDLE = "idle"
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    OBSERVATION_GRACE = "observation_grace"
    RECOVER = "recover"
    ARRIVED = "arrived"
    POSITIONING = "positioning"
    DWELLING = "dwelling"
    REST_TRAVEL = "rest_travel"
    REST_SETTLING = "rest_settling"
    RESTING = "resting"
    REST_RETURN = "rest_return"
    BLOCKED = "blocked"


@dataclass
class EdgeAttempt:
    number: int
    edge: Any
    macro_target_id: int
    dispatched_at: float
    verify_started_at: float = 0.0
    source_position: Optional[Tuple[float, float]] = None
    external_force_reason: Optional[str] = None
    last_raw_x: Optional[float] = None
    last_raw_t: float = 0.0
    unheld_origin_x: Optional[float] = None
    unheld_origin_t: float = 0.0
    reverse_run_px: float = 0.0
    reverse_run_sign: int = 0
    reverse_run_t: float = 0.0


class PlatformPatrolFSM:
    """One-shot edge dispatcher with persistent route and observation state."""

    def __init__(
        self,
        *,
        graph_getter: Callable[[], Any],
        patrol_getter: Callable[[], Sequence[int]],
        platform_getter: Callable[[], Any],
        world_position_getter: Callable[[], Optional[Tuple[float, float]]],
        edge_executor: Callable[[int, float, float], None],
        motion: Any,
        stop_event: Any,
        log_callback: Callable[[str], None],
        run_jump_enabled_getter: Optional[Callable[[], bool]] = None,
        intra_map_portal_enabled_getter: Optional[Callable[[], bool]] = None,
        dwell_range_getter: Optional[Callable[[], Tuple[float, float]]] = None,
        single_positions_getter: Optional[Callable[[], Sequence[float]]] = None,
        platform_positions_getter: Optional[Callable[[], Dict[int, Sequence[float]]]] = None,
        position_random_getter: Optional[Callable[[], float]] = None,
        arrival_tolerance_getter: Optional[Callable[[], float]] = None,
        dwell_extension_getter: Optional[Callable[[], float]] = None,
        dwell_threat_getter: Optional[Callable[[], bool]] = None,
        failure_replan_enabled_getter: Optional[Callable[[], bool]] = None,
        rest_navigation_getter: Optional[Callable[[], bool]] = None,
        target_completed_callback: Optional[Callable[[int], None]] = None,
        target_arrival_override: Optional[
            Callable[[Any, Tuple[float, float], float], bool]
        ] = None,
        inplace_dwell_safe_margin_getter: Optional[Callable[[], float]] = None,
        fatal_block_callback: Optional[Callable[[], None]] = None,
        debug_log_path: Optional[str] = None,
    ):
        self.graph_getter = graph_getter
        self.patrol_getter = patrol_getter
        self.platform_getter = platform_getter
        self.world_position_getter = world_position_getter
        self.edge_executor = edge_executor
        self.motion = motion
        self.stop_event = stop_event
        self.log = log_callback
        self.run_jump_enabled_getter = run_jump_enabled_getter
        self.intra_map_portal_enabled_getter = intra_map_portal_enabled_getter
        self.dwell_range_getter = dwell_range_getter
        self.single_positions_getter = single_positions_getter
        self.platform_positions_getter = platform_positions_getter
        self.position_random_getter = position_random_getter
        self.arrival_tolerance_getter = arrival_tolerance_getter
        self.dwell_extension_getter = dwell_extension_getter
        self.dwell_threat_getter = dwell_threat_getter
        self.failure_replan_enabled_getter = failure_replan_enabled_getter
        self.rest_navigation_getter = rest_navigation_getter
        self.target_completed_callback = target_completed_callback
        self.target_arrival_override = target_arrival_override
        self.inplace_dwell_safe_margin_getter = inplace_dwell_safe_margin_getter
        self.fatal_block_callback = fatal_block_callback
        self._arrival_behavior_enabled = bool(
            dwell_range_getter is not None or single_positions_getter is not None
        )

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.debug_log_path = (
            os.path.join(project_root, "logs", "patrol_fsm.jsonl")
            if debug_log_path is None else debug_log_path
        )
        self._debug_lock = threading.Lock()
        self._attempt_motion_lock = threading.Lock()

        self.phase = PatrolPhase.IDLE
        self.target_index = 0
        self.target_id: Optional[int] = None
        self.route: List[Any] = []
        self.attempt: Optional[EdgeAttempt] = None
        self.attempt_sequence = 0
        self.failure_counts: Dict[Tuple[int, int, str, Optional[int]], int] = {}
        self.rope_failure_counts: Dict[int, int] = {}
        # 动作级失败与“多次到达失败后重规划”完全独立。后者决定是否
        # 给 Dijkstra 边加惩罚；这里仅用于防止同一条边以完全相同的
        # 起跳点/接近方式无限重放，并把重试序号交给动作执行器。
        self.action_failure_counts: Dict[
            Tuple[int, int, str, Optional[int]], int
        ] = {}
        self._last_failed_action_key: Optional[
            Tuple[int, int, str, Optional[int]]
        ] = None
        # 仅真正的拓扑无路径可阻塞；动作候选耗尽会重置失败权重重试。
        self._blocked_route_signature: Optional[Tuple[int, int, int]] = None
        self._preflight_retry_key = None
        self._preflight_retry_anchor = None
        self._preflight_retry_count = 0
        # 起跳门返回 retry 表示动作键根本没有发出。短暂的平台量化抖动
        # 不能在下一帧把角色推进到目标平台，否则会从错误平台继续规划。
        self._deferred_source_guard_id: Optional[int] = None
        self._deferred_source_guard_until = 0.0
        self._deferred_source_guard_logged = False
        # 跳跃上升途中也可能短暂命中目标平台编号。必须在目标站立高度
        # 稳定一小段时间，才能把这条空中动作边判为成功。
        self._landing_candidate_id: Optional[int] = None
        self._landing_candidate_since = 0.0
        self._landing_candidate_last_x: Optional[float] = None
        self._landing_candidate_last_y: Optional[float] = None
        self._patrol_signature: Tuple[Any, ...] = ()
        self._last_good_position: Optional[Tuple[float, float]] = None
        self._last_good_platform: Any = None
        self._last_good_observation_at = 0.0
        self._observation_missing_since: Optional[float] = None
        self._observation_timeout_reported = False
        self._relocalization_attempted = False
        self._observation_relocalization_allowed = False
        self._platform_id: Optional[int] = None
        self._platform_since = 0.0
        self._last_stall_log_at = 0.0
        self._arrival_hold_until = 0.0
        self._dwell_until = 0.0
        self._dwell_started_at = 0.0
        self._dwell_duration = 0.0
        self._last_dwell_attack_at: Optional[float] = None
        self._arrival_position_x: Optional[float] = None
        self._arrival_random_fraction: Optional[float] = None
        # 当前目标平台内的站位序号。一个平台必须走完全部站位后才切换
        # 到下一个循环平台：P7@30 -> P7@60 -> P5@30 -> P5@60。
        self._arrival_station_index = 0
        self._recovery_until = 0.0
        self._last_phase_log: Optional[PatrolPhase] = None
        self._phase_since = time.perf_counter()

    @staticmethod
    def _edge_key(edge: Any) -> Tuple[int, int, str, Optional[int]]:
        return (
            int(edge.from_id), int(edge.to_id), str(edge.action),
            getattr(edge, "ladder_id", None),
        )

    @staticmethod
    def _edge_label(edge: Any) -> str:
        footholds = ""
        if getattr(edge, "source_foothold_id", None) is not None:
            footholds = (
                f" fh{edge.source_foothold_id}->fh{edge.target_foothold_id}"
                f" takeoff={getattr(edge, 'takeoff_x', None)}"
            )
        ladder = f" rope={edge.ladder_id}" if getattr(edge, "ladder_id", None) else ""
        return f"P{edge.from_id}->P{edge.to_id} {edge.action}{footholds}{ladder}"

    def _event(self, event: str, **payload: Any) -> None:
        if not self.debug_log_path:
            return
        record = {
            "ts": round(time.time(), 3),
            "mono": round(time.perf_counter(), 3),
            "event": event,
            "phase": self.phase.value,
            "target": self.target_id,
            **payload,
        }
        try:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._debug_lock:
                with open(self.debug_log_path, "a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        except Exception:
            pass

    def _set_phase(self, phase: PatrolPhase, reason: str = "") -> None:
        changed = phase != self.phase
        self.phase = phase
        if changed:
            self._phase_since = time.perf_counter()
            self._event("phase", new=phase.value, reason=reason)
            # OBSERVE/VERIFY run every frame; log only actual phase changes.
            self.log(f"🧭 [巡逻状态] {phase.value}" + (f"：{reason}" if reason else ""))
            self._last_phase_log = phase

    def start(self) -> None:
        self.reset("start")
        self._set_phase(PatrolPhase.OBSERVE, "F6启动")

    def stop(self) -> None:
        self._event("stop")
        self.phase = PatrolPhase.IDLE
        self.route.clear()
        self.attempt = None

    def reset(self, reason: str = "reset") -> None:
        self.route.clear()
        self.attempt = None
        self.target_id = None
        self.target_index = 0
        self._patrol_signature = ()
        self._observation_missing_since = None
        self._observation_timeout_reported = False
        self._relocalization_attempted = False
        self._observation_relocalization_allowed = False
        self._platform_id = None
        self._platform_since = time.perf_counter()
        self._arrival_hold_until = 0.0
        self._dwell_until = 0.0
        self._dwell_started_at = 0.0
        self._dwell_duration = 0.0
        self._last_dwell_attack_at = None
        self._arrival_position_x = None
        self._arrival_random_fraction = None
        self._arrival_station_index = 0
        self._recovery_until = 0.0
        self._phase_since = time.perf_counter()
        self.action_failure_counts.clear()
        self.rope_failure_counts.clear()
        self._last_failed_action_key = None
        self._deferred_source_guard_id = None
        self._blocked_route_signature = None
        self._preflight_retry_key = None
        self._preflight_retry_anchor = None
        self._preflight_retry_count = 0
        self._deferred_source_guard_until = 0.0
        self._deferred_source_guard_logged = False
        self._landing_candidate_id = None
        self._landing_candidate_since = 0.0
        self._landing_candidate_last_x = None
        self._landing_candidate_last_y = None
        self._event("reset", reason=reason)

    def _edge_arrival_is_stable(
        self,
        edge: Any,
        platform: Any,
        position: Tuple[float, float],
        now: float,
    ) -> bool:
        """Require a stationary landing instead of a transient platform hit."""
        action = str(getattr(edge, "action", ""))
        airborne = (
            "JUMP" in action
            or "DROP" in action
            or action.startswith("TELEPORT_")
        )
        if not airborne:
            self._landing_candidate_id = None
            self._landing_candidate_since = 0.0
            self._landing_candidate_last_x = None
            self._landing_candidate_last_y = None
            return True
        x, y = float(position[0]), float(position[1])
        expected_y = float(platform.surface_y_at(x)) - 45.0
        # The yellow dot is quantized by the WZ minimap canvas. A map whose
        # world Y scale is ~16 px per minimap pixel can show a stationary
        # character 24 px away from surface-45 (103000102/P23 does this).
        # Keep the old 20 px gate for high-resolution maps and cap the wider
        # gate so a passing jump cannot claim a distant platform.
        tolerance_y = 20.0
        try:
            graph = self.graph_getter()
            minimap = getattr(graph, "minimap_meta", {}) or {}
            world_height = float(minimap.get("height", 0) or 0)
            canvas_height = float(minimap.get("canvasHeight", 0) or 0)
            if world_height > 0 and canvas_height > 0:
                world_per_pixel = world_height / canvas_height
                if 0 < world_per_pixel <= 100:
                    tolerance_y = min(36.0, max(20.0, 2.0 * world_per_pixel))
        except (TypeError, ValueError, OverflowError):
            pass
        height_error = abs(y - expected_y)
        if height_error > tolerance_y:
            self._landing_candidate_id = None
            self._landing_candidate_since = 0.0
            self._landing_candidate_last_x = None
            self._landing_candidate_last_y = None
            return False
        platform_id = int(platform.id)
        if self._landing_candidate_id != platform_id:
            self._landing_candidate_id = platform_id
            self._landing_candidate_since = now
            self._landing_candidate_last_x = x
            self._landing_candidate_last_y = y
            return False
        if (
            self._landing_candidate_last_x is None
            or abs(x - self._landing_candidate_last_x) > 3.0
            or
            self._landing_candidate_last_y is None
            or abs(y - self._landing_candidate_last_y) > 3.0
        ):
            self._landing_candidate_since = now
        self._landing_candidate_last_x = x
        self._landing_candidate_last_y = y
        # A candidate outside the former strict gate needs longer stillness.
        # This covers minimap quantization without accepting a brief airborne
        # hit on the destination platform's broad classifier range.
        stable_sec = 0.35 if height_error > 20.0 else 0.18
        stable = (now - self._landing_candidate_since) >= stable_sec
        if stable and height_error > 20.0:
            self.log(
                f"🟢 [量化落台确认] P{platform_id} 观测Y={y:.1f}，"
                f"基准Y={expected_y:.1f}，偏差={height_error:.1f}px，"
                f"地图容差={tolerance_y:.1f}px，静止{stable_sec:.2f}s"
            )
        return stable

    def action_failure_count(self, edge: Optional[Any] = None) -> int:
        """返回当前边连续动作失败数；不受路线重规划开关影响。"""
        candidate = edge
        if candidate is None and self.attempt is not None:
            candidate = self.attempt.edge
        if candidate is None:
            return 0
        key = self._edge_key(candidate)
        preparation_failures = int(self.action_failure_counts.get(key, 0))
        ladder_id = getattr(candidate, "ladder_id", None)
        if ladder_id is None:
            return preparation_failures
        return max(
            preparation_failures,
            int(self.rope_failure_counts.get(int(ladder_id), 0)),
        )

    def report_external_force(self, reason: str) -> None:
        """记录运动控制器确认的动作期间受击；其它尝试不受影响。"""
        with self._attempt_motion_lock:
            if self.attempt is not None and self.attempt.external_force_reason is None:
                self.attempt.external_force_reason = str(reason)

    def observe_raw_motion(self, position: Tuple[float, float], *, timestamp: Optional[float] = None) -> None:
        """用 60Hz 原始黄点记录与输入不符的横移，供本次动作失败判定。"""
        if self.attempt is None:
            return
        now = time.perf_counter() if timestamp is None else float(timestamp)
        raw_x = float(position[0])
        with self._attempt_motion_lock:
            attempt = self.attempt
            if attempt is None or attempt.external_force_reason is not None:
                return
            previous_x, previous_t = attempt.last_raw_x, attempt.last_raw_t
            attempt.last_raw_x, attempt.last_raw_t = raw_x, now
            if previous_x is None or not 0.0 < now - previous_t <= 0.35:
                attempt.unheld_origin_x = None
                return
            model = getattr(self.motion, "motion_model", None)
            step = max(8.0, float(getattr(model, "measurement_step_px", 16.0) or 16.0))
            driver = getattr(self.motion, "driver", None)
            active = getattr(driver, "active_keys", ()) if driver is not None else ()
            held = getattr(self.motion, "current_held_key", None)
            left = held == "left" or "left" in active
            right = held == "right" or "right" in active
            sign = -1 if left and not right else 1 if right and not left else 0
            delta = raw_x - previous_x
            # 水平瞬移本身就是合法大位移。
            tp_at = float(getattr(self.motion, "last_teleport_command_at", 0.0) or 0.0)
            tp_dir = getattr(self.motion, "last_teleport_direction", None)
            if tp_dir in ("left", "right") and 0 <= now - tp_at <= 0.35:
                attempt.unheld_origin_x = None
                return
            if abs(delta) >= max(60.0, step * 2.25):
                attempt.external_force_reason = f"原始坐标单帧突跳{delta:+.1f}px"
            elif sign:
                attempt.unheld_origin_x = None
                if attempt.reverse_run_sign != sign or now - attempt.reverse_run_t > 0.35:
                    attempt.reverse_run_px = 0.0
                attempt.reverse_run_sign = sign
                attempt.reverse_run_t = now
                if delta * sign < -0.5:
                    attempt.reverse_run_px += -delta * sign
                    if attempt.reverse_run_px >= max(32.0, step * 2.0):
                        attempt.external_force_reason = (
                            f"按{held or ('left' if sign < 0 else 'right')}时"
                            f"连续逆向位移{attempt.reverse_run_px:.1f}px"
                        )
                elif delta * sign > 0.5:
                    # 换向后的单格惯性/黄点量化抖动不能累计成受击。
                    attempt.reverse_run_px = 0.0
            else:
                attempt.reverse_run_px = 0.0
                attempt.reverse_run_sign = 0
                action = str(getattr(attempt.edge, "action", ""))
                if attempt.unheld_origin_x is None:
                    attempt.unheld_origin_x = previous_x
                    attempt.unheld_origin_t = previous_t
                if now - attempt.unheld_origin_t > 0.85:
                    attempt.unheld_origin_x = previous_x
                    attempt.unheld_origin_t = previous_t
                else:
                    unheld_delta = raw_x - attempt.unheld_origin_x
                    if action in (
                        "JUMP_UP", "TELEPORT_UP", "DOWN_JUMP",
                        "CLIMB_LADDER_UP", "CLIMB_LADDER_DOWN",
                    ) and abs(unheld_delta) >= max(48.0, step * 2.5):
                        attempt.external_force_reason = f"未按方向键时横移{unheld_delta:+.1f}px"
                    elif "LEFT" in action and unheld_delta >= max(48.0, step * 2.5):
                        attempt.external_force_reason = f"向左动作松键后反向横移{unheld_delta:+.1f}px"
                    elif "RIGHT" in action and unheld_delta <= -max(48.0, step * 2.5):
                        attempt.external_force_reason = f"向右动作松键后反向横移{unheld_delta:+.1f}px"

    def _run_jump_enabled(self) -> bool:
        if self.run_jump_enabled_getter is None:
            return True
        try:
            return bool(self.run_jump_enabled_getter())
        except Exception:
            return True

    def _intra_map_portal_enabled(self) -> bool:
        if self.intra_map_portal_enabled_getter is None:
            return True
        try:
            return bool(self.intra_map_portal_enabled_getter())
        except Exception:
            return True

    def _dwell_range(self) -> Tuple[float, float]:
        try:
            values = self.dwell_range_getter() if self.dwell_range_getter else (0.0, 0.0)
            low, high = float(values[0]), float(values[1])
        except (TypeError, ValueError, IndexError):
            low, high = 0.0, 0.0
        low = max(0.0, min(600.0, low))
        high = max(0.0, min(600.0, high))
        return (min(low, high), max(low, high))

    def _dwell_extension(self) -> float:
        try:
            value = float(
                self.dwell_extension_getter()
                if self.dwell_extension_getter else 2.0
            )
        except (TypeError, ValueError):
            value = 2.0
        return max(0.0, min(60.0, value))

    def _failure_replan_enabled(self) -> bool:
        try:
            return bool(
                self.failure_replan_enabled_getter()
                if self.failure_replan_enabled_getter else False
            )
        except Exception:
            return False

    def notify_attack_during_dwell(self, at: Optional[float] = None) -> None:
        """记录真实攻击按键时间，供停留截止点检查最近 n 秒窗口。"""
        if self.phase == PatrolPhase.DWELLING and self._dwell_until > 0.0:
            self._last_dwell_attack_at = (
                time.perf_counter() if at is None else float(at)
            )

    def _position_overrides(self) -> Dict[int, Tuple[float, ...]]:
        try:
            raw = self.platform_positions_getter() if self.platform_positions_getter else {}
        except Exception:
            raw = {}
        result = {}
        for pid, values in (raw or {}).items():
            try:
                parsed = tuple(float(value) for value in values)
                if parsed and all(0.0 <= value <= 1.0 for value in parsed):
                    result[int(pid)] = parsed
            except (TypeError, ValueError):
                continue
        return result

    def _single_positions(self, platform_id: Optional[int] = None) -> Tuple[float, ...]:
        if platform_id is not None:
            override = self._position_overrides().get(int(platform_id))
            if override:
                return override
        try:
            raw = self.single_positions_getter() if self.single_positions_getter else (0.30, 0.60)
        except Exception:
            raw = (0.30, 0.60)
        parsed: List[float] = []
        for value in raw or ():
            try:
                fraction = float(value)
            except (TypeError, ValueError):
                continue
            fraction = max(0.0, min(1.0, fraction))
            if not parsed or abs(fraction - parsed[-1]) > 1e-6:
                parsed.append(fraction)
        return tuple(parsed or (0.30, 0.60))

    def _settings_signature(
        self, patrol: Tuple[int, ...]
    ) -> Tuple[Any, ...]:
        dwell = self._dwell_range()
        positions = self._single_positions()
        overrides = self._position_overrides()
        return (
            patrol,
            round(dwell[0], 3),
            round(dwell[1], 3),
            tuple(round(value, 4) for value in positions),
            tuple(sorted((pid, tuple(round(value, 4) for value in values))
                         for pid, values in overrides.items())),
            round(self._position_random_fraction(), 4),
            round(self._configured_arrival_tolerance(), 3),
            self._failure_replan_enabled(),
        )

    def _position_random_fraction(self) -> float:
        try:
            value = self.position_random_getter() if self.position_random_getter else 0.0
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _configured_arrival_tolerance(self) -> float:
        try:
            value = (
                self.arrival_tolerance_getter()
                if self.arrival_tolerance_getter is not None
                else 20.0
            )
            return max(1.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return 20.0

    def _inplace_dwell_safe_margin(self) -> float:
        if self.inplace_dwell_safe_margin_getter is not None:
            try:
                return max(0.0, float(self.inplace_dwell_safe_margin_getter()))
            except Exception:
                pass
        if hasattr(self, "motion") and hasattr(self.motion, "config") and isinstance(self.motion.config, dict):
            try:
                return max(0.0, float(self.motion.config.get("inplace_dwell_safe_margin_px", 50.0)))
            except Exception:
                pass
        return 50.0

    def _cancel_arrival_behavior(self, reason: str) -> None:
        if self._dwell_until > 0.0 or self._arrival_position_x is not None:
            self._event("arrival_cancelled", reason=reason)
        self._dwell_until = 0.0
        self._dwell_started_at = 0.0
        self._dwell_duration = 0.0
        self._last_dwell_attack_at = None
        self._arrival_position_x = None
        self._arrival_random_fraction = None

    def _desired_arrival_x(
        self, patrol: Tuple[int, ...], platform: Any
    ) -> Tuple[float, Optional[float]]:
        positions = self._single_positions(int(platform.id))
        base_fraction = positions[self._arrival_station_index % len(positions)]
        if self._arrival_random_fraction is None:
            radius = self._position_random_fraction()
            self._arrival_random_fraction = max(
                0.0,
                min(1.0, base_fraction + random.uniform(-radius, radius)),
            ) if radius > 0.0 else base_fraction
        fraction = self._arrival_random_fraction
        target_x = float(platform.x_min) + fraction * (
            float(platform.x_max) - float(platform.x_min)
        )
        safe_lo = float(platform.x_min) + min(8.0, max(0.0, platform.length / 4.0))
        safe_hi = float(platform.x_max) - min(8.0, max(0.0, platform.length / 4.0))
        if safe_lo <= safe_hi:
            target_x = min(safe_hi, max(safe_lo, target_x))
        return target_x, fraction

    def _handle_target_arrival(
        self,
        patrol: Tuple[int, ...],
        platform: Any,
        position: Tuple[float, float],
        now: float,
    ) -> None:
        """目标平台专属的站位与随机停留；路径中的过路平台不会调用。"""
        if self.target_arrival_override is not None:
            try:
                if self.target_arrival_override(platform, position, now):
                    return
            except Exception as exc:
                self._event("arrival_override_error", error=str(exc))
                self.log(f"⚠️ [到站覆盖逻辑异常] {type(exc).__name__}: {exc}")
        if not self._arrival_behavior_enabled:
            self._advance_target(patrol, now)
            return

        if self._dwell_until > 0.0:
            if now < self._dwell_until:
                self._set_phase(
                    PatrolPhase.DWELLING,
                    f"P{platform.id} 剩余{self._dwell_until - now:.2f}s",
                )
                return
            extension = self._dwell_extension()
            attack_age = (
                now - self._last_dwell_attack_at
                if self._last_dwell_attack_at is not None else None
            )
            # 攻击只在截止点之前的 n 秒滑动窗口内有效；更早的攻击
            # 不得因为“本次停留曾攻击过”而继续续期。
            recently_attacked = bool(
                extension > 0.0
                and attack_age is not None
                and 0.0 <= attack_age <= extension
            )
            try:
                occupied = bool(
                    self.dwell_threat_getter()
                    if self.dwell_threat_getter else False
                )
            except Exception:
                occupied = False
            if extension > 0.0 and (recently_attacked or occupied):
                # 只在截止点增加一次，不能在高频检测循环中每帧叠加。
                # 一次攻击最多触发下一次截止点的一次续期；延长期内若
                # 再次真实攻击，会写入新时间戳供下个截止点判断。
                if recently_attacked:
                    self._last_dwell_attack_at = None
                self._dwell_until = now + extension
                self._dwell_duration += extension
                reasons = []
                if recently_attacked:
                    reasons.append(
                        f"截止前{extension:.2f}s内发生攻击(age={attack_age:.2f}s)"
                    )
                if occupied:
                    reasons.append("正反向攻击框仍有怪")
                reason = "、".join(reasons)
                self._set_phase(
                    PatrolPhase.DWELLING,
                    f"P{platform.id} 战斗续留{extension:.2f}s",
                )
                self._event(
                    "dwell_extended_for_combat",
                    platform=int(platform.id), extension=round(extension, 3),
                    recently_attacked=recently_attacked,
                    attack_age=(round(attack_age, 3) if attack_age is not None else None),
                    occupied=occupied,
                )
                self.log(
                    f"⏳ [平台战斗续留] P{platform.id} 停留截止时检测到{reason}，"
                    f"延长{extension:.2f}s"
                )
                return
            duration = self._dwell_duration
            platform_id = int(platform.id)
            positions = self._single_positions(int(platform.id))
            station_index = self._arrival_station_index
            self._event(
                "dwell_complete", platform=platform_id,
                duration=round(duration, 3),
                station_index=station_index,
            )
            self._dwell_until = 0.0
            self._dwell_started_at = 0.0
            self._dwell_duration = 0.0
            self._last_dwell_attack_at = None
            self._arrival_position_x = None
            self._arrival_random_fraction = None
            next_station_index = station_index + 1
            if next_station_index < len(positions):
                self._arrival_station_index = next_station_index
                next_fraction = positions[next_station_index]
                self.route.clear()
                self.attempt = None
                self._arrival_hold_until = now + 0.05
                self._set_phase(
                    PatrolPhase.ARRIVED,
                    f"P{platform.id} 停留完成，下一站位{next_fraction * 100:.0f}%",
                )
                return
            # 当前平台的全部百分比站位已完成，下一目标平台从第一个
            # 站位重新开始；单平台时 advance_target 会自然回到自身。
            self._arrival_station_index = 0
            self._advance_target(patrol, now)
            return

        target_x, fraction = self._desired_arrival_x(patrol, platform)
        # 到站目标是“平台中部区域”，不是必须命中单一像素。小地图世界
        # 坐标存在量化与平滑跳动；固定点配 8px 死区会在目标两侧反复
        # 翻转方向。长平台允许中心 ±20px，短平台按长度收窄以保留安全边距。
        configured_tolerance = self._configured_arrival_tolerance()
        tolerance = max(
            1.0,
            min(configured_tolerance, float(platform.length) * 0.20),
        )

        # 方案 1: 就地停留 (In-Place Dwell) 保护判定
        safe_margin = self._inplace_dwell_safe_margin()
        plat_len = float(platform.length)
        eff_margin = min(safe_margin, max(0.0, plat_len * 0.25)) if plat_len < (2.0 * safe_margin) else safe_margin
        safe_x_min = float(platform.x_min) + eff_margin
        safe_x_max = float(platform.x_max) - eff_margin

        # 若此前走位曾被战斗打断（或怪撞），且当前所处位置已在安全区内，直接就地停留
        was_interrupted = bool(getattr(self.motion, "last_walk_priority_interrupted", False))
        cur_pos_x = float(position[0])
        if was_interrupted and (safe_x_min <= cur_pos_x <= safe_x_max):
            if hasattr(self.motion, "last_walk_priority_interrupted"):
                self.motion.last_walk_priority_interrupted = False
            self.log(
                f"🛡️ [就地停留] 战斗打断让行后，当前P{platform.id}位于安全区 "
                f"(X={cur_pos_x:.1f}，距边缘>{eff_margin:.0f}px)，就地开始停留"
            )
            self._event(
                "inplace_dwell_adopted", platform=int(platform.id),
                original_target_x=round(target_x, 2), adopted_x=round(cur_pos_x, 2),
                safe_margin=round(eff_margin, 1), reason="combat_interrupted_pre",
            )
            target_x = cur_pos_x

        if abs(float(position[0]) - target_x) > tolerance:
            if self._arrival_position_x != target_x:
                self._arrival_position_x = target_x
                self._event(
                    "arrival_positioning", platform=int(platform.id),
                    target_x=round(target_x, 2),
                    tolerance=round(tolerance, 2),
                    fraction=(round(fraction, 4) if fraction is not None else None),
                )
                station = (
                    f"{fraction * 100:.0f}%站位" if fraction is not None else "平台中点"
                )
                self.log(
                    f"📍 [巡逻目标站位] P{platform.id} 走向{station} "
                    f"X={target_x:.1f}±{tolerance:.0f}px"
                )
            self._set_phase(PatrolPhase.POSITIONING, f"P{platform.id} X={target_x:.1f}")
            distance = abs(float(position[0]) - target_x)
            arrived = self.motion.walk_to_x(
                target_x=target_x,
                get_player_pos=self.world_position_getter,
                tolerance=int(round(tolerance)),
                timeout_sec=max(2.0, min(7.0, distance / 100.0 + 1.5)),
                stop_event=self.stop_event,
                platform_bounds=(platform.x_min, platform.x_max),
                safe_margin=max(8, min(25, int(platform.length * 0.10))),
                speed_scale=1.0,
            )
            latest = self.world_position_getter()
            current_platform = self.platform_getter()
            if (
                latest is None
                or current_platform is None
                or int(current_platform.id) != int(platform.id)
            ):
                self._event(
                    "arrival_position_failed", platform=int(platform.id),
                    target_x=round(target_x, 2), position=latest,
                    observed_platform=getattr(current_platform, "id", None),
                )
                return

            if not arrived:
                latest_x = float(latest[0])
                if safe_x_min <= latest_x <= safe_x_max:
                    interrupted = bool(getattr(self.motion, "last_walk_priority_interrupted", False))
                    if hasattr(self.motion, "last_walk_priority_interrupted"):
                        self.motion.last_walk_priority_interrupted = False
                    reason = "战斗中断" if interrupted else "走位未达原点但处于安全区"
                    self.log(
                        f"🛡️ [就地停留] {reason}，当前P{platform.id}位于安全区 "
                        f"(X={latest_x:.1f}，距边缘>{eff_margin:.0f}px)，就地开始停留"
                    )
                    self._event(
                        "inplace_dwell_adopted", platform=int(platform.id),
                        original_target_x=round(target_x, 2), adopted_x=round(latest_x, 2),
                        safe_margin=round(eff_margin, 1), reason="walk_interrupted_post",
                    )
                    target_x = latest_x
                else:
                    self._event(
                        "arrival_position_failed", platform=int(platform.id),
                        target_x=round(target_x, 2), position=latest,
                        observed_platform=getattr(current_platform, "id", None),
                    )
                    return
            now = time.perf_counter()

        self.motion.stop()
        low, high = self._dwell_range()
        duration = random.uniform(low, high) if high > low else low
        self._dwell_started_at = now
        self._dwell_duration = duration
        self._dwell_until = now + duration
        self._last_dwell_attack_at = None
        self._arrival_position_x = target_x
        self._set_phase(PatrolPhase.DWELLING, f"P{platform.id} 停留{duration:.2f}s")
        self._event(
            "dwell_started", platform=int(platform.id), duration=round(duration, 3),
            target_x=round(target_x, 2),
            fraction=(round(fraction, 4) if fraction is not None else None),
            station_index=self._arrival_station_index,
        )
        self.log(
            f"⏸️ [平台随机停留] P{platform.id} X={target_x:.1f}，"
            f"本次{duration:.2f}s（范围{low:.2f}–{high:.2f}s）"
        )

    def _capture_observation(
        self, graph: Any, supplied_position: Optional[Tuple[float, float]]
    ) -> Tuple[Optional[Tuple[float, float]], Any]:
        now = time.perf_counter()
        try:
            position = self.world_position_getter()
        except Exception:
            position = None
        if position is None:
            position = supplied_position
        try:
            platform = self.platform_getter()
        except Exception:
            platform = None
        if platform is None and position is not None:
            try:
                platform = graph.find_player_platform(position[0], position[1])
            except Exception:
                platform = None

        if position is not None:
            position = (float(position[0]), float(position[1]))
            self._last_good_position = position
            self._last_good_observation_at = now
        if platform is not None:
            self._last_good_platform = platform
            self._last_good_observation_at = now
        return position, platform

    def _handle_missing_observation(self, now: float) -> bool:
        if self._observation_missing_since is None:
            self._observation_missing_since = now
            action = getattr(getattr(self.attempt, "edge", None), "action", "")
            # 盲走脱离遮挡只适用于传送点，或根本没有正在验证的动作。
            # 跳跃/攀爬后坐标短暂丢失时，最后可信平台已经是旧位置，
            # 按它的“内部方向”移动反而会把角色从绳上或落点推出去。
            self._observation_relocalization_allowed = action in ("", "PORTAL")
            self._event("observation_lost", action=action)
            self.log(
                "🟡 [巡逻观测丢失] 黄点/承重平台暂时不可用，进入宽限期；"
                "不使用屏幕坐标替代世界坐标，也不重复派发动作"
            )
        missing_for = now - self._observation_missing_since
        active_action = str(getattr(getattr(self.attempt, "edge", None), "action", ""))
        grace = 2.20 if active_action == "PORTAL" else 1.20
        if missing_for <= grace:
            self._set_phase(PatrolPhase.OBSERVATION_GRACE, f"丢失{missing_for:.2f}s/{grace:.2f}s")
            return True

        if not self._observation_timeout_reported:
            # 从宽限进入超时时只停止一次。观测长时间丢失时
            # tick 仍会继续调用本函数，若每帧发送全键 KEYUP，会让
            # 玩家无法手动移动自救。
            self.motion.stop()
            self.route.clear()
            self.attempt = None
            self._recovery_until = now + 0.35
            self._observation_timeout_reported = True
            self._set_phase(PatrolPhase.RECOVER, "观测超时，停止输入并等待重新定位")
            self._event("observation_timeout", missing_for=round(missing_for, 3))

        # 传送门光标会持续盖住小地图黄点；完全静止时观测无法自行恢复。
        # 只依据最后可信的世界平台与位置，向平台内部做一次短距离连续
        # 行走来脱离遮挡。没有安全余量时保持停止，绝不拿主视口像素
        # 冒充世界坐标，也不在一次丢失期间反复盲走。
        if (
            not self._relocalization_attempted
            and self._observation_relocalization_allowed
            and self._last_good_platform is not None
            and self._last_good_position is not None
        ):
            platform = self._last_good_platform
            x = float(self._last_good_position[0])
            left_room = x - float(platform.x_min)
            right_room = float(platform.x_max) - x
            direction = None
            if right_room >= 65.0 or left_room >= 65.0:
                direction = "right" if right_room >= left_room else "left"
            if direction is not None:
                self._relocalization_attempted = True
                self._event(
                    "observation_relocalize", direction=direction,
                    duration=0.25, platform=int(platform.id), x=round(x, 1),
                )
                self.log(
                    f"🧭 [巡逻重新定位] 仅按最后可信P{platform.id}向平台内部"
                    f"连续移动0.25s（{direction}），尝试脱离传送门遮挡"
                )
                try:
                    self.motion.timed_walk(direction, 0.25, self.stop_event)
                except Exception as exc:
                    self._event("observation_relocalize_error", error=str(exc))
        return True

    def _observe_platform_stall(self, platform: Any, position: Tuple[float, float], now: float) -> None:
        platform_id = int(platform.id)
        if platform_id != self._platform_id:
            previous = self._platform_id
            self._platform_id = platform_id
            self._platform_since = now
            self._last_stall_log_at = 0.0
            self._event(
                "platform_enter", previous=previous, platform=platform_id,
                x=round(position[0], 1), y=round(position[1], 1),
            )
            return
        stayed = now - self._platform_since
        # 正常站位停留和休息本来就会有意保持在同一平台，不能把配置的
        # 10~15 秒随机停留误报成“卡死”。开始停留时已有明确的时长日志，
        # 此处只监控本应产生导航进展的阶段。
        intentional_hold_phases = {
            PatrolPhase.DWELLING,
            PatrolPhase.REST_SETTLING,
            PatrolPhase.RESTING,
            PatrolPhase.BLOCKED,
        }
        if self.phase in intentional_hold_phases:
            return
        if stayed >= 8.0 and now - self._last_stall_log_at >= 4.0:
            self._last_stall_log_at = now
            edge = getattr(self.attempt, "edge", None)
            phase_elapsed = max(0.0, now - self._phase_since)
            self.log(
                f"⏱️ [导航滞留] P{platform_id} 平台总驻留{stayed:.1f}s，"
                f"当前阶段={self.phase.value}已持续{phase_elapsed:.1f}s，"
                f"当前边={self._edge_label(edge) if edge else 'None'}"
            )
            self._event(
                "platform_stall", platform=platform_id, stayed=round(stayed, 2),
                phase_elapsed=round(phase_elapsed, 2),
                edge=self._edge_label(edge) if edge else None,
                failures=self.failure_counts.get(self._edge_key(edge), 0) if edge else 0,
            )

    def _select_initial_target(
        self, patrol: Tuple[int, ...], platform: Any,
        graph: Any, position: Tuple[float, float],
    ) -> None:
        current_id = int(platform.id)
        selection = "current_platform"
        estimated_sec = 0.0
        if current_id in patrol:
            # F6 begins with this platform's own position/dwell cycle. Skipping
            # directly to the next stop made a launch on P3 ignore P3 entirely.
            self.target_index = patrol.index(current_id)
        else:
            # Compare the same position-aware Dijkstra metric used by normal
            # navigation. Straight-line distance can favor a nearby-looking
            # platform that requires a long ladder or a one-way detour.
            selection = "nearest_reachable_route"
            best: Optional[Tuple[Tuple[int, int, int], int]] = None
            for index, candidate_id in enumerate(patrol):
                route, metric = graph.find_path_from_position(
                    current_id, float(position[0]), int(candidate_id),
                    allow_run_jump=self._run_jump_enabled(),
                    allow_portal=self._intra_map_portal_enabled(),
                    return_metric=True,
                )
                if not route or metric is None:
                    continue
                ranked = (metric, index)
                if best is None or ranked < best:
                    best = ranked
            if best is not None:
                self.target_index = best[1]
                estimated_sec = best[0][0] / 100.0
            else:
                # Keep the existing blocked-route behavior when no configured
                # stop is reachable; never mistake a visually close platform
                # for a valid first destination.
                self.target_index = 0
                selection = "no_reachable_patrol_platform"
        self.target_id = patrol[self.target_index]
        self._event(
            "target_selected", target=self.target_id, patrol=list(patrol),
            source=current_id, selection=selection,
            estimated_sec=round(estimated_sec, 2),
        )
        self.log(
            f"🎯 [巡逻初始目标] 当前P{current_id}，先到P{self.target_id}"
            f"（{selection}，预计{estimated_sec:.2f}s），之后按设定顺序循环"
        )

    def _advance_target(self, patrol: Tuple[int, ...], now: float) -> None:
        arrived = self.target_id
        if arrived is not None and self.target_completed_callback is not None:
            try:
                self.target_completed_callback(int(arrived))
            except Exception as exc:
                self._event(
                    "target_completed_callback_error",
                    arrived=arrived,
                    error=str(exc),
                )
                self.log(f"⚠️ [巡逻目标回调异常] P{arrived}：{exc}")
        self.target_index = (self.target_index + 1) % len(patrol)
        self.target_id = patrol[self.target_index]
        self.route.clear()
        self.attempt = None
        self._arrival_hold_until = now + 0.30
        self._set_phase(PatrolPhase.ARRIVED, f"到达P{arrived}，下一目标P{self.target_id}")
        self._event("target_arrived", arrived=arrived, next_target=self.target_id)

    def _plan(self, graph: Any, platform: Any,
              position: Optional[Tuple[float, float]] = None) -> bool:
        self._set_phase(PatrolPhase.PLAN, f"P{platform.id}->P{self.target_id}")
        rest_navigation = bool(
            self.rest_navigation_getter() if self.rest_navigation_getter else False
        )
        # The current X matters even on ordinary F6 routes. Treating every
        # landing on a long platform as the same state made the planner prefer
        # a far-away 1000px ledge over the ladder already on the way.
        origin = position if position is not None else self.world_position_getter()
        origin_x = float(origin[0]) if origin is not None else None
        # 用户开启“多次到达失败后重规划”时，才把失败反馈给下一次
        # Dijkstra；默认关闭时始终保持基准最短路径，不记录失败次数。
        def reliability_penalty(edge: Any) -> float:
            key = self._edge_key(edge)
            penalty = 0.0
            # The raw edge cost only counts the nominal action. Landing on a
            # tiny *transit* foothold also needs precise alignment and often
            # costs another recovery attempt. Prefer a direct, wide-platform
            # ladder route over a chain of 30-80 px steps when it is faster
            # in real execution (103000102 P20->P10 is one such case).
            if int(edge.to_id) != int(self.target_id):
                transit = graph.get_node(int(edge.to_id))
                if transit is not None:
                    width = max(0.0, float(transit.x_max) - float(transit.x_min))
                    penalty += min(2.0, max(0.0, (160.0 - width) * 0.015))
            source = graph.get_node(int(edge.from_id))
            destination = graph.get_node(int(edge.to_id))
            goal = graph.get_node(int(self.target_id))
            if source is not None and destination is not None and goal is not None:
                upward_backtrack = float(source.y) - float(destination.y)
                if upward_backtrack > 0 and float(goal.y) > float(source.y) + 100.0:
                    # A nominal 0.38s upward teleport is not a shortcut when
                    # the actual destination is far below. It incurs an
                    # extra alignment/return-descent action in real play.
                    penalty += 1.0 + upward_backtrack / 125.0
            if self._failure_replan_enabled():
                failures = self.failure_counts.get(key, 0)
                if failures >= 2:
                    penalty += 4.0 + min(
                        8.0, float(failures - 2) * 0.75
                    )
            # 第三次失败后大幅降权，第四次本轮排除；候选全部排除后
            # 整轮清零，重新从原始最优动作边开始。
            action_failures = self.action_failure_counts.get(key, 0)
            if rest_navigation and action_failures >= 1:
                # 临时休息目标不能在同一条失败边上再赌一次。先尝试
                # 其它几何可达边；普通 F6 路线仍保留原有重试次数。
                penalty += 2.0
            ladder_id = getattr(edge, "ladder_id", None)
            climb_failures = (
                self.rope_failure_counts.get(int(ladder_id), 0)
                if ladder_id is not None else 0
            )
            if action_failures >= ACTION_EXCLUDE_AFTER or climb_failures >= CLIMB_FAILURE_LIMIT:
                return float("inf")
            if action_failures >= ACTION_PENALTY_AFTER or climb_failures >= ACTION_PENALTY_AFTER:
                penalty += 60.0
            return penalty

        path_kwargs = dict(
            allow_run_jump=self._run_jump_enabled(),
            edge_penalty_fn=reliability_penalty,
            allow_portal=self._intra_map_portal_enabled(),
        )
        if origin_x is not None and hasattr(graph, "find_path_from_position"):
            route = graph.find_path_from_position(
                int(platform.id), origin_x, int(self.target_id), **path_kwargs,
            )
        else:
            route = graph.find_path(
                int(platform.id), int(self.target_id), **path_kwargs,
            )
        if not route:
            self.motion.stop()
            # 只有基准图可达、但本轮动作全部被第四次失败排除，才重置
            # 权重并从原始首边重新尝试；真正无路径仍保持 BLOCKED。
            exhausted_actions = [
                key for key, failures in self.action_failure_counts.items()
                if failures >= ACTION_EXCLUDE_AFTER
            ]
            exhausted_ropes = [
                rope_id for rope_id, failures in self.rope_failure_counts.items()
                if failures >= CLIMB_FAILURE_LIMIT
            ]
            if exhausted_actions or exhausted_ropes:
                baseline_kwargs = dict(path_kwargs)
                baseline_kwargs.pop("edge_penalty_fn", None)
                if origin_x is not None and hasattr(graph, "find_path_from_position"):
                    baseline_route = graph.find_path_from_position(
                        int(platform.id), origin_x, int(self.target_id),
                        **baseline_kwargs,
                    )
                else:
                    baseline_route = graph.find_path(
                        int(platform.id), int(self.target_id), **baseline_kwargs,
                    )
                if baseline_route:
                    self.route.clear()
                    self.action_failure_counts.clear()
                    self.rope_failure_counts.clear()
                    self.failure_counts.clear()
                    self._last_failed_action_key = None
                    self._preflight_retry_key = None
                    self._preflight_retry_anchor = None
                    self._preflight_retry_count = 0
                    self._blocked_route_signature = None
                    self._recovery_until = time.perf_counter() + 0.45
                    self._set_phase(PatrolPhase.RECOVER, "本轮候选耗尽，权重清零后重试")
                    self._event(
                        "action_route_cycle_reset",
                        source=int(platform.id),
                        destination=int(self.target_id),
                        exhausted_actions=[str(key) for key in exhausted_actions],
                        exhausted_ropes=exhausted_ropes,
                        baseline_route=[self._edge_label(edge) for edge in baseline_route],
                    )
                    self.log(
                        f"🔄 [巡逻动作轮次重置] P{platform.id}->P{self.target_id} "
                        f"候选均已失败{ACTION_EXCLUDE_AFTER}次；清零动作/梯绳/路线权重，"
                        f"冷却0.45秒后从原始首边重试。动作边={exhausted_actions}，梯绳={exhausted_ropes}"
                    )
                    return False
            self.route.clear()
            self._set_phase(PatrolPhase.BLOCKED, f"P{platform.id}->P{self.target_id}无路径")
            self._event("route_missing", source=platform.id, destination=self.target_id)
            self.log(f"⛔ [巡逻无路径] P{platform.id} 无法到达 P{self.target_id}")
            # 图未变化时每帧重跑相同 Dijkstra 没有意义，还会把日志和
            # CPU 撑满。短暂等待后再试，地图/巡逻设置变更仍会 reset。
            self._recovery_until = time.perf_counter() + 1.0
            return False
        self.route = list(route)
        metrics = graph.path_metrics(route) if hasattr(graph, "path_metrics") else {
            "estimated_cost": round(sum(float(edge.cost) for edge in route), 2),
            "rope_count": sum(bool(getattr(edge, "is_rope", False)) for edge in route),
            "hop_count": len(route),
        }
        metrics = dict(metrics)
        applied_penalty = round(sum(reliability_penalty(edge) for edge in route), 2)
        action_penalty = round(
            sum(
                60.0
                for failures in self.action_failure_counts.values()
                if ACTION_PENALTY_AFTER <= failures < ACTION_EXCLUDE_AFTER
            )
            + sum(
                60.0
                for failures in self.rope_failure_counts.values()
                if ACTION_PENALTY_AFTER <= failures < CLIMB_FAILURE_LIMIT
            ),
            2,
        )
        metrics["runtime_penalty"] = applied_penalty
        metrics["action_replan_penalty"] = action_penalty
        metrics["failure_replan_enabled"] = self._failure_replan_enabled()
        if origin_x is not None:
            metrics["route_origin_x"] = round(origin_x, 1)
            metrics["position_aware"] = True
        metrics["effective_cost"] = round(
            float(metrics["estimated_cost"]) + applied_penalty, 2
        )
        labels = [self._edge_label(edge) for edge in route]
        self._event(
            "route_planned", source=platform.id, destination=self.target_id,
            metrics=metrics, route=labels,
        )
        self.log(
            f"🗺️ [巡逻路径] P{platform.id}->P{self.target_id} "
            f"cost={metrics['estimated_cost']} hops={metrics['hop_count']} "
            f"ropes={metrics['rope_count']} runtimePenalty={applied_penalty} "
            f"actionPenalty={action_penalty} "
            f"failureReplan={'on' if metrics['failure_replan_enabled'] else 'off'} | "
            + (f"originX={origin_x:.1f} positionAware=on | " if origin_x is not None else "")
            + " | ".join(labels)
        )
        exhausted = [
            (key, failures)
            for key, failures in self.action_failure_counts.items()
            if failures >= ACTION_PENALTY_AFTER
        ]
        if exhausted and route:
            failed_key, failed_count = exhausted[0]
            if self._edge_key(route[0]) != failed_key:
                self.log(
                    f"↪️ [动作候选耗尽局部绕行] P{failed_key[0]}->P{failed_key[1]} "
                    f"{failed_key[2]} 已失败{failed_count}次；"
                    f"本轮改走 {self._edge_label(route[0])}"
                )
                self._event(
                    "action_fallback_route",
                    failed_edge=str(failed_key),
                    failures=failed_count,
                    fallback_edge=self._edge_label(route[0]),
                )
        return True

    def _verification_timeout(self, edge: Any) -> float:
        action = str(edge.action)
        if "CLIMB" in action:
            # 攀爬控制器内部已经完成登顶保持与最长0.70s落地确认；
            # 返回后这里只需给平台解析一次短收敛窗口。
            return 1.20
        if action == "PORTAL":
            return 2.20
        if "LONG_DROP" in action:
            return 2.20
        if "DROP" in action or action == "DOWN_JUMP":
            return 1.60
        return 1.35

    def _mark_edge_success(self, platform: Any, position: Tuple[float, float], now: float) -> None:
        assert self.attempt is not None
        edge = self.attempt.edge
        elapsed = now - self.attempt.dispatched_at
        self._event(
            "edge_success", attempt=self.attempt.number, edge=self._edge_label(edge),
            elapsed=round(elapsed, 3), platform=platform.id,
            x=round(position[0], 1), y=round(position[1], 1),
        )
        self.log(
            f"✅ [巡逻边完成 #{self.attempt.number}] {self._edge_label(edge)}，"
            f"耗时{elapsed:.2f}s"
        )
        if self.route and self._edge_key(self.route[0]) == self._edge_key(edge):
            self.route.pop(0)
        if self.rest_navigation_getter and self.rest_navigation_getter():
            # 休息是临时目标；每次真正落台后，从实测 X 重新选下一条
            # 边，不继续沿用以预测落点编排的整段旧路线。
            self.route.clear()
        if self._failure_replan_enabled():
            self.failure_counts[self._edge_key(edge)] = 0
        ladder_id = getattr(edge, "ladder_id", None)
        if ladder_id is not None:
            ladder_id = int(ladder_id)
            self.rope_failure_counts.pop(ladder_id, None)
            # 梯绳失败按实体ID共享。任意入口成功后，该实体的失败与
            # 准备失败记录全部清零，下一轮重新拥有完整四次容错。
            for failed_key in list(self.action_failure_counts):
                if failed_key[3] == ladder_id:
                    self.action_failure_counts.pop(failed_key, None)
        # 只清除真正成功的这一条动作边。不能因“掉到下层后成功走回
        # 起点”就清掉原失败边，否则 P73->P74 成功会反复复活失败的
        # P74->绳31，形成跨边的大范围来回走。
        succeeded_key = self._edge_key(edge)
        self.action_failure_counts.pop(succeeded_key, None)
        if self._last_failed_action_key == succeeded_key:
            self._last_failed_action_key = None
        self._landing_candidate_id = None
        self._landing_candidate_since = 0.0
        self._landing_candidate_last_x = None
        self._landing_candidate_last_y = None
        self.attempt = None
        self._preflight_retry_key = None
        self._preflight_retry_anchor = None
        self._preflight_retry_count = 0
        self._set_phase(PatrolPhase.OBSERVE, "继续下一条边")

    def _discard_external_attempt(self, platform: Any, position: Tuple[float, float],
                                  now: float, reason: str) -> bool:
        with self._attempt_motion_lock:
            attempt = self.attempt
            force_reason = getattr(attempt, "external_force_reason", None)
        if attempt is None or force_reason is None:
            return False
        self.motion.stop()
        self.route.clear()
        self.attempt = None
        self._preflight_retry_key = None
        self._preflight_retry_anchor = None
        self._preflight_retry_count = 0
        self._landing_candidate_id = None
        self._landing_candidate_since = 0.0
        self._landing_candidate_last_x = None
        self._landing_candidate_last_y = None
        self._recovery_until = now + 0.25
        self._set_phase(PatrolPhase.RECOVER, "外力打断，不计失败")
        self._event(
            "edge_external_force_uncounted",
            attempt=attempt.number, edge=self._edge_label(attempt.edge),
            force=force_reason, original_reason=reason,
            platform=getattr(platform, "id", None),
            x=round(position[0], 1), y=round(position[1], 1),
        )
        self.log(
            f"💥 [动作外力打断] {self._edge_label(attempt.edge)}，"
            f"证据={force_reason}，原结果={reason}；本次不计失败，按实时平台重新规划"
        )
        return True

    def _mark_edge_failure(self, platform: Any, position: Tuple[float, float], now: float, reason: str) -> bool:
        if self._discard_external_attempt(platform, position, now, reason):
            return False
        self._preflight_retry_key = None
        self._preflight_retry_anchor = None
        self._preflight_retry_count = 0
        assert self.attempt is not None
        edge = self.attempt.edge
        key = self._edge_key(edge)
        ladder_id = getattr(edge, "ladder_id", None)
        if ladder_id is None:
            action_count = self.action_failure_counts.get(key, 0) + 1
            self.action_failure_counts[key] = action_count
            self._last_failed_action_key = key
            rope_count = 0
        else:
            ladder_id = int(ladder_id)
            # 真正发出攀爬动作后的失败只记在梯绳实体上，不能再同时进入
            # 普通动作独立计数，否则梯绳无法获得四次容错。
            rope_count = self.rope_failure_counts.get(ladder_id, 0) + 1
            self.rope_failure_counts[ladder_id] = rope_count
            action_count = rope_count
        self._landing_candidate_id = None
        self._landing_candidate_since = 0.0
        self._landing_candidate_last_x = None
        self._landing_candidate_last_y = None
        track_failures = self._failure_replan_enabled()
        if track_failures:
            count = self.failure_counts.get(key, 0) + 1
            self.failure_counts[key] = count
        else:
            count = 0
            self.failure_counts.pop(key, None)
        rope_id = ladder_id
        self.motion.stop()
        self._event(
            "edge_failure", attempt=self.attempt.number, edge=self._edge_label(edge),
            reason=reason, count=count, action_failures=action_count,
            rope_failures=rope_count,
            observed_platform=getattr(platform, "id", None),
            x=round(position[0], 1), y=round(position[1], 1),
        )
        self.log(
            f"❌ [巡逻边失败 #{self.attempt.number}] {self._edge_label(edge)}，"
            f"原因={reason}，"
            + (f"路线连续失败={count}" if track_failures else "路线失败计数已关闭")
            + f"，动作连续失败={action_count}"
        )
        if action_count >= 1:
            self.log(
                f"🔧 [动作级重规划待命] {self._edge_label(edge)} 已连续失败"
                f"{action_count}次；下轮保留拓扑路线，但重新计算动作参数"
            )
        if rope_id is not None:
            self.log(
                f"🪢 [梯绳容错] 梯绳#{rope_id} 本次失败，"
                f"连续{rope_count}/{CLIMB_FAILURE_LIMIT}次；"
                + (
                    "达到上限，本轮排除该绳；候选耗尽后整轮清零重试"
                    if rope_count >= CLIMB_FAILURE_LIMIT
                    else "未达上限，允许按实时坐标重试同一梯绳"
                )
            )
        if rope_id is not None and rope_count >= 2:
            self.log(
                f"🧪 [绳索失败诊断] 绳#{rope_id} 已连续失败{rope_count}次；"
                f"当前位置=({position[0]:.1f},{position[1]:.1f})，"
                f"源平台=P{edge.from_id}，目标平台=P{edge.to_id}，"
                f"触发X={getattr(edge, 'trigger_x', None)}"
            )
            self._event(
                "rope_repeated_failure", rope=rope_id, count=rope_count,
                trigger_x=getattr(edge, "trigger_x", None),
            )
        self.route.clear()
        self.attempt = None
        delay_count = count if track_failures else action_count
        self._recovery_until = now + min(0.75, 0.15 + min(4, delay_count) * 0.10)
        self._set_phase(PatrolPhase.RECOVER, reason)
        return True

    def tick(self, supplied_position: Optional[Tuple[float, float]] = None,
             *, verification_only: bool = False) -> None:
        if self.stop_event.is_set():
            return
        now = time.perf_counter()
        graph = self.graph_getter()
        patrol = tuple(int(value) for value in (self.patrol_getter() or ()) if value is not None)
        if graph is None or not getattr(graph, "nodes", None) or not patrol:
            self._set_phase(PatrolPhase.IDLE, "缺少拓扑或循环平台")
            return
        missing = [platform_id for platform_id in patrol if graph.get_node(platform_id) is None]
        if missing:
            self._set_phase(PatrolPhase.BLOCKED, f"平台不存在:{missing}")
            return

        settings_signature = self._settings_signature(patrol)
        if settings_signature != self._patrol_signature:
            self.motion.stop()
            if not self._failure_replan_enabled():
                self.failure_counts.clear()
                self.rope_failure_counts.clear()
            self.reset("patrol_changed")
            self._patrol_signature = settings_signature

        position, platform = self._capture_observation(graph, supplied_position)
        if position is None or platform is None:
            self._handle_missing_observation(now)
            return
        guarded_source_id = self._deferred_source_guard_id
        if guarded_source_id is not None:
            if now > self._deferred_source_guard_until:
                self._deferred_source_guard_id = None
                self._deferred_source_guard_until = 0.0
                self._deferred_source_guard_logged = False
            elif int(platform.id) != int(guarded_source_id):
                guarded_source = graph.get_node(int(guarded_source_id))
                if guarded_source is not None:
                    if not self._deferred_source_guard_logged:
                        self.log(
                            f"🛡️ [未发键源平台保护] 动作未发出，忽略P{platform.id}"
                            f"瞬时观测，仍按源平台P{guarded_source_id}重试"
                        )
                        self._event(
                            "deferred_source_guard",
                            observed_platform=int(platform.id),
                            forced_platform=int(guarded_source_id),
                            x=round(position[0], 1),
                            y=round(position[1], 1),
                        )
                        self._deferred_source_guard_logged = True
                    platform = guarded_source
        if self._observation_missing_since is not None:
            lost_for = now - self._observation_missing_since
            self._event(
                "observation_recovered", lost_for=round(lost_for, 3),
                platform=platform.id, x=round(position[0], 1), y=round(position[1], 1),
            )
            self.log(f"🟢 [巡逻观测恢复] 丢失{lost_for:.2f}s后恢复，当前P{platform.id}")
            self._observation_missing_since = None
            self._observation_timeout_reported = False
            self._relocalization_attempted = False
            self._observation_relocalization_allowed = False

        self._observe_platform_stall(platform, position, now)
        # Combat may take priority for many consecutive frames. Keep the
        # already-dispatched edge's landing verification alive, but never
        # dispatch another navigation edge from this combat-side tick.
        if verification_only and (
            self.phase != PatrolPhase.VERIFY or self.attempt is None
        ):
            return
        if (
            self.target_id is not None
            and int(platform.id) != int(self.target_id)
            and (self._dwell_until > 0.0 or self._arrival_position_x is not None)
        ):
            # 站位或停留期间被击退/掉出目标平台，取消本次计时并按当前
            # 实际平台重新规划返回同一目标；过路平台不获得停留时间。
            self._cancel_arrival_behavior(
                f"离开目标P{self.target_id}，当前P{platform.id}"
            )
            self.route.clear()
            self.attempt = None
            self._set_phase(
                PatrolPhase.RECOVER,
                f"掉出目标P{self.target_id}，重新规划返回",
            )
        if now < self._recovery_until or now < self._arrival_hold_until:
            return
        if self.target_id is None:
            self._select_initial_target(patrol, platform, graph, position)

        if self._blocked_route_signature is not None:
            current_signature = (
                id(graph), int(platform.id), int(self.target_id)
            )
            if current_signature == self._blocked_route_signature:
                # No callback was provided (e.g. an isolated test runner).
                # Stay safely idle rather than re-running the same Dijkstra.
                return
            self._blocked_route_signature = None

        if int(platform.id) == int(self.target_id):
            if (
                self.attempt is not None
                and int(platform.id) == int(self.attempt.edge.to_id)
            ):
                if not self._edge_arrival_is_stable(
                    self.attempt.edge, platform, position, now
                ):
                    self._set_phase(
                        PatrolPhase.VERIFY,
                        f"P{platform.id} 等待站稳确认",
                    )
                    return
                self._mark_edge_success(platform, position, now)
            if verification_only:
                return
            self._handle_target_arrival(patrol, platform, position, now)
            return

        if self.attempt is not None:
            edge = self.attempt.edge
            if int(platform.id) == int(edge.to_id):
                if not self._edge_arrival_is_stable(edge, platform, position, now):
                    self._set_phase(
                        PatrolPhase.VERIFY,
                        f"P{platform.id} 等待站稳确认",
                    )
                    return
                self._mark_edge_success(platform, position, now)
                return
            self._landing_candidate_id = None
            self._landing_candidate_since = 0.0
            self._landing_candidate_last_x = None
            self._landing_candidate_last_y = None
            verify_elapsed = now - self.attempt.verify_started_at
            timeout = self._verification_timeout(edge)
            if verify_elapsed < timeout:
                self._set_phase(
                    PatrolPhase.VERIFY,
                    f"{self._edge_label(edge)} {verify_elapsed:.2f}/{timeout:.2f}s",
                )
                return
            reason = (
                "仍停留源平台"
                if int(platform.id) == int(edge.from_id)
                else f"落到非预期P{platform.id}"
            )
            self._mark_edge_failure(platform, position, now, reason)
            return

        if not self.route or int(self.route[0].from_id) != int(platform.id):
            if not self._plan(graph, platform, position):
                return

        edge = self.route[0]
        takeoff = getattr(edge, "takeoff_x", None)
        if takeoff is None:
            takeoff = getattr(edge, "trigger_x", None)
        slope_payload: Dict[str, Any] = {}
        action = str(getattr(edge, "action", ""))
        if takeoff is not None and ("RIGHT" in action or "LEFT" in action):
            travel_sign = 1.0 if "RIGHT" in action else -1.0
            sample_x = float(takeoff)
            behind_x = min(platform.x_max, max(platform.x_min, sample_x - travel_sign * 20.0))
            ahead_x = min(platform.x_max, max(platform.x_min, sample_x + travel_sign * 20.0))
            slope_payload = {
                "planned_takeoff_x": round(sample_x, 1),
                "takeoff_x_range": getattr(edge, "takeoff_x_range", None),
                "directional_surface_dy_40px": round(
                    float(platform.surface_y_at(ahead_x) - platform.surface_y_at(behind_x)), 2
                ),
            }
        self.attempt_sequence += 1
        self.attempt = EdgeAttempt(
            number=self.attempt_sequence,
            edge=edge,
            macro_target_id=int(self.target_id),
            dispatched_at=now,
            source_position=position,
        )
        failures = self.failure_counts.get(self._edge_key(edge), 0)
        action_failures = self.action_failure_count(edge)
        self._set_phase(PatrolPhase.EXECUTE, self._edge_label(edge))
        self._event(
            "edge_dispatch", attempt=self.attempt.number,
            edge=self._edge_label(edge), failures=failures,
            action_failures=action_failures,
            x=round(position[0], 1), y=round(position[1], 1),
            **slope_payload,
        )
        self.log(
            f"▶️ [巡逻边派发 #{self.attempt.number}] {self._edge_label(edge)}，"
            f"路线历史失败={failures}，动作连续失败={action_failures}"
        )
        try:
            executed = self.edge_executor(int(self.target_id), position[0], position[1])
        except Exception as exc:
            self._mark_edge_failure(platform, position, time.perf_counter(), f"执行异常:{exc}")
            return
        if executed in ("retry", "relocalize") and self._discard_external_attempt(
            platform, position, time.perf_counter(), str(executed)
        ):
            return
        if executed == "retry":
            deferred_attempt = self.attempt
            self.attempt = None
            # No key was sent, but repeating at one X is a failed preparation
            # strategy. A sub-minimap-pixel gate can otherwise loop forever.
            try:
                current_position = self.world_position_getter()
            except Exception:
                current_position = None
            if current_position is None:
                current_position = position
            current_position = (
                float(current_position[0]), float(current_position[1])
            )
            key = self._edge_key(edge)
            anchor = self._preflight_retry_anchor
            if (
                key != self._preflight_retry_key
                or anchor is None
                or abs(current_position[0] - anchor[0]) >= 12.0
                or abs(current_position[1] - anchor[1]) >= 12.0
            ):
                self._preflight_retry_key = key
                self._preflight_retry_anchor = current_position
                self._preflight_retry_count = 1
            else:
                self._preflight_retry_count += 1
            if self._preflight_retry_count >= 3:
                count = self.action_failure_counts.get(key, 0) + 1
                self.action_failure_counts[key] = count
                self._last_failed_action_key = key
                self._preflight_retry_count = 0
                self._preflight_retry_anchor = current_position
                self.route.clear()
                self._deferred_source_guard_id = None
                self._deferred_source_guard_until = 0.0
                self._recovery_until = time.perf_counter() + 0.30
                self._set_phase(PatrolPhase.RECOVER, "起跳准备连续无进度，重新选动作")
                self._event(
                    "takeoff_gate_no_progress", edge=self._edge_label(edge),
                    x=round(current_position[0], 1),
                    y=round(current_position[1], 1),
                    preparation_failures=count,
                )
                self.log(
                    f"⛔ [起跳准备无进度] {self._edge_label(edge)} "
                    f"连续3次未发键且位置未推进，动作准备失败={count}；"
                    "重选候选或绕行"
                )
                return
            # Alt 尚未发出，不能算作这条边的实机失败。保留当前直达路线，
            # 下一帧重新观测后从源平台再次对齐，避免失败惩罚把 P8->P10
            # 改规划成 P8->P18->P20->P10。
            self._recovery_until = time.perf_counter() + 0.30
            self._deferred_source_guard_id = int(edge.from_id)
            self._deferred_source_guard_until = time.perf_counter() + 1.20
            self._deferred_source_guard_logged = False
            self._landing_candidate_id = None
            self._landing_candidate_since = 0.0
            self._landing_candidate_last_x = None
            self._landing_candidate_last_y = None
            self._set_phase(PatrolPhase.OBSERVE, "起跳未放行，重新确认落台后原边无惩罚重试")
            self._event(
                "edge_deferred_for_takeoff_gate",
                attempt=getattr(deferred_attempt, "number", None),
                edge=self._edge_label(edge),
            )
            self.log(
                f"🔁 [起跳线原边重试] {self._edge_label(edge)}，"
                "Alt未发出，不计失败、不切换绕行路线"
            )
            return
        if executed == "relocalize":
            relocalized_attempt = self.attempt
            self.attempt = None
            self.route.clear()
            failed_key = self._edge_key(edge)
            relocalize_count = self.action_failure_counts.get(failed_key, 0) + 1
            self.action_failure_counts[failed_key] = relocalize_count
            self._last_failed_action_key = failed_key
            self._deferred_source_guard_id = None
            self._deferred_source_guard_until = 0.0
            self._deferred_source_guard_logged = False
            self._landing_candidate_id = None
            self._landing_candidate_since = 0.0
            self._landing_candidate_last_x = None
            self._landing_candidate_last_y = None
            self._recovery_until = time.perf_counter() + 0.08
            self._set_phase(PatrolPhase.OBSERVE, "源平台已丢失，按实况立即重规划")
            self._event(
                "edge_relocalize_without_penalty",
                attempt=getattr(relocalized_attempt, "number", None),
                edge=self._edge_label(edge),
                action_failures=relocalize_count,
            )
            self.log(
                f"↩️ [源平台丢失重定位] {self._edge_label(edge)}，"
                f"动作未发出、不计路线失败，动作准备连续失败={relocalize_count}，"
                "解除源平台保护"
            )
            if relocalize_count >= ACTION_PENALTY_AFTER:
                self.log(
                    f"↪️ [动作准备失败换边] {self._edge_label(edge)} 连续"
                    f"{relocalize_count}次在发键前丢失源平台；下轮降权并优先绕行"
                )
                self._event(
                    "relocalize_action_replan",
                    edge=self._edge_label(edge),
                    action_failures=relocalize_count,
                )
            return
        if executed == "top_exit_failed":
            # 抓取本身已经成功；失败点是该梯绳边的登顶/横移方案。
            # 抓取或登顶失败按梯绳实体ID累计。第三次降权，第四次
            # 本轮排除该绳；任一入口成功后清零。
            counted = self._mark_edge_failure(
                platform,
                position,
                time.perf_counter(),
                "抓取成功但绳顶脱离失败",
            )
            if not counted:
                return
            ladder_id = getattr(edge, "ladder_id", None)
            climb_failures = (
                self.rope_failure_counts.get(int(ladder_id), 0)
                if ladder_id is not None else 0
            )
            self.log(
                f"{'↪️' if climb_failures >= CLIMB_FAILURE_LIMIT else '🔁'} "
                f"[绳顶失败容错] {self._edge_label(edge)}，"
                f"梯绳#{ladder_id} 连续失败{climb_failures}/{CLIMB_FAILURE_LIMIT}；"
                + (
                    "达到上限，按当前平台选择备用边"
                    if climb_failures >= CLIMB_FAILURE_LIMIT
                    else "保留该梯绳，下轮重新定位后重试"
                )
            )
            self._event(
                "top_exit_failure_replan",
                edge=self._edge_label(edge),
                ladder_id=ladder_id,
                climb_failures=climb_failures,
                retry_limit=CLIMB_FAILURE_LIMIT,
            )
            return
        self._deferred_source_guard_id = None
        self._deferred_source_guard_until = 0.0
        self._deferred_source_guard_logged = False
        if executed is False:
            deferred_attempt = self.attempt
            self.attempt = None
            self._set_phase(PatrolPhase.OBSERVE, "攻击优先，原路线保留待重试")
            self._event(
                "edge_deferred_for_attack",
                attempt=getattr(deferred_attempt, "number", None),
                edge=self._edge_label(edge),
            )
            self.log(f"⚔️ [巡逻让行攻击] {self._edge_label(edge)}，不计失败，攻击后重试")
            return
        if self.attempt is not None:
            self.attempt.verify_started_at = time.perf_counter()
            self._set_phase(PatrolPhase.VERIFY, self._edge_label(edge))
