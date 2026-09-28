"""
motion_controller.py - 2D 物理微动作控制器 (MotionController)
负责将路径规划/战斗决策层的高层指令转化为精确的平走、下跳、小跳、瞬时转向等键盘序列。
"""

import time
import random
import threading
import math
from typing import Optional, Tuple, Callable, Any, Dict
import numpy as np
from src.core.input_driver import InputDriver


class MotionController:
    def __init__(
        self,
        input_driver: InputDriver,
        jump_key: str = "alt",
        ladder_aligner: Optional[Any] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        motion_model: Optional[Any] = None,
        raw_position_getter: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
        takeoff_gate_tolerance_getter: Optional[Callable[[], float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.driver = input_driver
        self.jump_key = jump_key
        self.ladder_aligner = ladder_aligner
        self.log_fn = log_callback or (lambda msg: None)
        self.motion_model = motion_model
        self.raw_position_getter = raw_position_getter
        self.takeoff_gate_tolerance_getter = takeoff_gate_tolerance_getter
        self.config = config if isinstance(config, dict) else {}
        self.is_moving = False
        self.current_held_key: Optional[str] = None
        self.last_landing_confirmed = False
        # 抓绳已经成功、但绳顶主动脱离仍未取得物理落台证据。调用方可
        # 据此区分普通抓取失败，避免在绳顶再次发送 Alt。
        self.last_top_exit_failed = False
        self.priority_interrupt_checker = None
        self.last_walk_priority_interrupted = False
        # 区分“动作已经执行但落台失败”和“起跳线闭环尚未放行”。后者
        # 没有按下 Alt，不应被巡逻 FSM 计作边失败并触发绕路惩罚。
        self.last_jump_gate_aborted = False
        self.external_force_callback: Optional[Callable[[str], None]] = None

        # 法师瞬移配置与状态
        self.enable_teleport: bool = bool(self.config.get("enable_teleport", False))
        self.teleport_key: str = str(self.config.get("teleport_key", "shift"))
        self.teleport_vk: int = int(self.config.get("teleport_vk", 16) or 16)
        self.teleport_cd_min_ms: float = float(self.config.get("teleport_cd_min_ms", 300.0))
        self.teleport_cd_max_ms: float = float(self.config.get("teleport_cd_max_ms", 500.0))
        self.teleport_distance_px: float = float(self.config.get("teleport_distance_px", 150.0))
        self.next_teleport_ts: float = 0.0
        self.last_teleport_command_at: float = 0.0
        self.last_teleport_direction: Optional[str] = None
        self.get_current_platform_bounds: Optional[Callable[[], Optional[Tuple[int, int]]]] = None

    def log(self, msg: str):
        if self.log_fn:
            try:
                self.log_fn(msg)
            except Exception:
                pass

    def _report_external_force(self, reason: str) -> None:
        callback = self.external_force_callback
        if callback is not None:
            try:
                callback(reason)
            except Exception:
                pass

    def walk_to_x(
        self,
        target_x: int,
        get_player_pos: Callable[[], Optional[Tuple[int, int]]],
        tolerance: int = 18,
        timeout_sec: float = 3.5,
        stop_event = None,
        platform_bounds: Optional[Tuple[int, int]] = None,
        safe_margin: int = 25,
        early_takeoff_x: Optional[int] = None,
        speed_scale: float = 1.0,
        preserve_direction_on_arrival: bool = False,
        completion_checker: Optional[Callable[[], bool]] = None,
        prefer_teleport: bool = False,
        observation_timeout_sec: Optional[float] = None,
    ) -> bool:
        """
        闭环控制角色平走至物理世界目标 X 坐标 (集成平台防跌落保护与边缘起跳):
        :param target_x: 物理世界目标 X 坐标
        :param get_player_pos: 实时坐标回调函数
        :param tolerance: 目标到位容差半径 (默认 18px)
        :param timeout_sec: 最大行进保护超时
        :param stop_event: 外部中断事件 (threading.Event)
        :param platform_bounds: 当前所处平台 X 物理边界 (x_min, x_max)，启用防跌落急停
        :param safe_margin: 防跌落安全距离 (默认距边缘 25px 提前刹车)
        :param early_takeoff_x: 起跳触发线，到达即刻返回 True 衔接跳跃动作
        :param speed_scale: 行走占空比 (0~1)，用于窄平台等需要慢速微调的位置
        :param preserve_direction_on_arrival: 到位后保留当前方向键，供普通跑跳
            无缝接力使用；调用方必须立即进入 jump_step。
        :param observation_timeout_sec: 连续失去坐标后提前停止的可选超时。
            默认保持旧行为；无小地图过渡使用较短超时以进入找人状态。
        """
        start_time = time.perf_counter()
        self.last_walk_priority_interrupted = False
        stuck_check_time = start_time
        stuck_start_x = None
        braking = False
        brake_started = 0.0
        brake_cooldown_until = 0.0
        # raw 黄点每一次跨格都代表新的真实位置。正常同向跨格由运动模型
        # 连续积分；若它突然与当前行进方向相反，通常是被怪击退/推走，
        # 必须立即丢弃旧速度后重新计算剩余距离与刹停点。
        last_raw_x = None
        last_raw_t = None
        # 低占空比精确对齐时输出节流后的控制遥测。它只记录决策依据，
        # 不改变输入时序；用于区分坐标源、方向计算或按键发送三类问题。
        debug_next_log = start_time
        observation_lost_at = None

        while True:
            if stop_event is not None and stop_event.is_set():
                self.stop()
                return False

            if completion_checker is not None:
                try:
                    if bool(completion_checker()):
                        self.stop()
                        return True
                except Exception:
                    pass

            checker = self.priority_interrupt_checker
            if checker is not None:
                try:
                    interrupt_reason = checker()
                    if bool(interrupt_reason):
                        self.stop()
                        self.last_walk_priority_interrupted = True
                        if isinstance(interrupt_reason, str):
                            self.log(f"🪑 [休息优先中断] {interrupt_reason}，停止当前地面走位")
                        else:
                            self.log("⚔️ [攻击优先中断] 攻击框变红，停止地面走位并让行攻击")
                        return False
                except Exception:
                    pass

            now = time.perf_counter()
            if (now - start_time) > timeout_sec:
                self.stop()
                return False

            pos = get_player_pos()
            if pos is None:
                # 视觉暂失，短暂等待
                if observation_lost_at is None:
                    observation_lost_at = now
                if (observation_timeout_sec is not None
                        and now - observation_lost_at >= observation_timeout_sec):
                    self.stop()
                    return False
                time.sleep(0.02)
                continue
            observation_lost_at = None

            raw_pos = self.raw_position_getter() if self.raw_position_getter else None
            cur_x, cur_y = pos
            # 方向、到站、起跳线、边缘保护统一使用输入感知 Kalman 的
            # 连续世界 X（由 get_player_pos 提供）。raw 黄点只保留给外力
            # 突跳/反向跨格等安全诊断，不能再抢占主控制坐标。
            control_x = float(cur_x)
            if raw_pos is not None:
                current_raw_x = float(raw_pos[0])
                if last_raw_x is not None and last_raw_t is not None:
                    raw_delta = current_raw_x - last_raw_x
                    held_sign = 1 if self.current_held_key == "right" else -1 if self.current_held_key == "left" else 0
                    # 反向准备后单格回跳可由惯性或黄点量化造成；只有至少
                    # 两格的突跳才立即重锚。多帧持续逆移由巡逻观测器确认。
                    step = float(getattr(self.motion_model, "measurement_step_px", 16.0) or 16.0)
                    if held_sign and raw_delta * held_sign <= -max(32.0, step * 2.0):
                        self._report_external_force(
                            f"走位按{self.current_held_key}时反向位移{raw_delta:+.1f}px"
                        )
                        if self.motion_model is not None:
                            try:
                                self.motion_model.explicit_reanchor(current_raw_x, reset_velocity=True)
                            except Exception:
                                pass
                        braking = False
                        brake_cooldown_until = now
                        self.log(
                            f"⚠️ [外力位移重规划] rawX {last_raw_x:.1f} -> {current_raw_x:.1f} "
                            f"与当前{self.current_held_key}相反；重算剩余走位与刹停"
                        )
                last_raw_x = current_raw_x
                last_raw_t = now
            dx = target_x - control_x

            if speed_scale <= 0.30 and now >= debug_next_log:
                model = self.motion_model
                model_x = None
                model_v = 0.0
                model_dir = 0
                model_blocked = False
                if model is not None:
                    try:
                        model_x = getattr(model, "x", None)
                        model_v = float(getattr(model, "vx", 0.0))
                        model_dir = int(getattr(model, "direction", 0))
                        model_blocked = bool(getattr(model, "blocked", False))
                    except Exception:
                        pass
                planned_key = "right" if dx > 0 else "left"
                self.log(
                    f"🔬 [对齐遥测] 目标X={target_x:.1f} | predictedX={float(cur_x):.1f} | "
                    f"rawX={float(raw_pos[0]):.1f}" if raw_pos is not None
                    else f"🔬 [对齐遥测] 目标X={target_x:.1f} | predictedX={float(cur_x):.1f} | rawX=None"
                )
                # 单独一行保留，避免上面含条件表达式时遗漏真实控制来源。
                self.log(
                    f"🔬 [对齐决策] controlX={control_x:.1f} dx={dx:+.1f} -> {planned_key} | "
                    f"held={self.current_held_key} | modelX={model_x} v={model_v:+.1f} "
                    f"dir={model_dir} blocked={model_blocked} | scale={float(speed_scale):.2f}"
                )
                debug_next_log = now + 0.35

            # 1. 起跳触发线判定 (用于跨平台跳跃无缝衔接，绝不踏空)
            if early_takeoff_x is not None:
                if dx > 0 and control_x >= early_takeoff_x:
                    return True
                elif dx < 0 and control_x <= early_takeoff_x:
                    return True

            # 2. 达到连续 Kalman 坐标容差死区 -> 立即刹车到位。
            if abs(dx) <= tolerance:
                if not preserve_direction_on_arrival:
                    self.stop()
                return True

            # 2.5 预测性松键刹停：按当前速度和地面阻力计算滑行距离，
            # 在到达目标前松开方向键，让角色自然停在目标附近。
            model = self.motion_model
            req_sign = 1.0 if dx > 0 else -1.0
            if model is not None and not preserve_direction_on_arrival:
                try:
                    signed_v = float(getattr(model, "vx", 0.0)) * req_sign
                    drag = max(1.0, float(getattr(model, "drag_accel", 900.0)))
                    stop_dist = (max(0.0, signed_v) ** 2) / (2.0 * drag)
                    if (
                        not braking
                        and now >= brake_cooldown_until
                        and self.current_held_key == ("right" if req_sign > 0 else "left")
                    ):
                        if abs(dx) <= stop_dist + max(4.0, float(tolerance)):
                            self.stop()
                            braking = True
                            brake_started = now
                            # 松键后的短暂惯性阶段内禁止立即重新触发
                            # 同一刹停判断，否则会出现“松键—再按—再松键”
                            # 的导航卡死。
                            brake_cooldown_until = now + 0.25
                            self.log(
                                f"🛑 [预测松键刹停] 目标X={target_x}，当前X={cur_x:.1f}，"
                                f"v={getattr(model, 'vx', 0.0):+.1f}，预测滑行={stop_dist:.1f}px"
                            )
                            time.sleep(0.02)
                            continue
                    elif braking:
                        # 等待速度耗尽，若仍未进入容差才允许继续微调。
                        if abs(float(getattr(model, "vx", 0.0))) > 5.0 and (now - brake_started) < 0.50:
                            time.sleep(0.02)
                            continue
                        braking = False
                except Exception:
                    braking = False

            # 3. 平台物理防跌落硬保护 (Anti-Falloff Guard)
            if platform_bounds is not None:
                p_min, p_max = platform_bounds
                if dx > 0 and control_x >= (p_max - safe_margin):
                    # 向右移动且已逼近右边缘危险区 -> 紧急刹车！
                    self.stop()
                    return False
                elif dx < 0 and control_x <= (p_min + safe_margin):
                    # 向左移动且已逼近左边缘危险区 -> 紧急刹车！
                    self.stop()
                    return False

            # 4. 卡死防死锁检测：若持续 1.0 秒移动但 X 坐标位移 < 8px -> 判定撞墙或被阻挡
            if stuck_start_x is None:
                stuck_start_x = cur_x
                stuck_check_time = now
            elif (now - stuck_check_time) > 1.0:
                if abs(cur_x - stuck_start_x) < 8:
                    self.stop()
                    return False
                stuck_start_x = cur_x
                stuck_check_time = now

            # 4.5 瞬移寻路介入判定 (Teleport Pathing):
            # 常规寻路保留两倍瞬移距离门槛；聚怪可请求优先瞬移，但
            # 预测落点不能越过目标和平台安全线。
            req_key = "right" if dx > 0 else "left"
            if (
                self.enable_teleport
                and speed_scale >= 0.8
                and abs(dx) > (
                    (self.teleport_distance_px + max(8.0, float(tolerance)))
                    if prefer_teleport else 2.0 * self.teleport_distance_px
                )
                and now >= self.next_teleport_ts
            ):
                bounds = platform_bounds
                if bounds is None and self.get_current_platform_bounds is not None:
                    try:
                        bounds = self.get_current_platform_bounds()
                    except Exception:
                        bounds = None

                # 安全性检测：确保瞬移后的预测落点不越出平台安全范围
                tp_safe = True
                predicted_next_x = control_x + (self.teleport_distance_px if dx > 0 else -self.teleport_distance_px)
                if bounds is not None:
                    p_min, p_max = bounds
                    if dx > 0 and predicted_next_x > (p_max - safe_margin):
                        tp_safe = False
                    elif dx < 0 and predicted_next_x < (p_min + safe_margin):
                        tp_safe = False
                if prefer_teleport and (
                    (dx > 0 and predicted_next_x > target_x + tolerance)
                    or (dx < 0 and predicted_next_x < target_x - tolerance)
                ):
                    tp_safe = False

                if tp_safe:
                    self.log(
                        f"⚡ [寻路瞬移] 距目标 {dx:+.1f}px (> 2x{self.teleport_distance_px:.0f}px)，"
                        f"向{req_key}瞬移，预计到达 X={predicted_next_x:.1f}"
                    )
                    self.teleport(req_key)
                    stuck_start_x = None
                    stuck_check_time = time.perf_counter()
                    time.sleep(0.06)
                    continue

            # 5. 决定行进方向与保持按键
            req_key = "right" if dx > 0 else "left"
            if self.current_held_key != req_key:
                if self.current_held_key is not None:
                    self.driver.key_up(self.current_held_key)
                self.driver.key_down(req_key)
                self.current_held_key = req_key
                if speed_scale <= 0.30:
                    self.log(f"⌨️ [对齐输入] key_down({req_key})")

            # 窄平台居中使用短按+松开，避免全速跨过小平台中点。
            # 当前窄台档为100ms控制周期、30%占空比，即按30ms/松70ms。
            scale = max(0.15, min(1.0, float(speed_scale)))
            if scale < 0.99:
                period_sec = 0.10
                time.sleep(max(0.012, period_sec * scale))
                self.driver.key_up(req_key)
                self.current_held_key = None
                if scale <= 0.30:
                    self.log(f"⌨️ [对齐输入] key_up({req_key})")
                time.sleep(max(0.005, period_sec * (1.0 - scale)))
                if 0.295 <= scale <= 0.305:
                    # 窄平台低占空比微调：给角色和坐标采样留出
                    # 稳定时间，避免连续脉冲叠加造成过冲。
                    time.sleep(0.10)
                continue

            # 控制循环频率 (约 30~50 Hz)
            time.sleep(0.02)

    def walk_through_portal(
        self,
        *,
        trigger_x: float,
        get_player_pos: Callable[[], Optional[Tuple[float, float]]],
        stop_event: Any = None,
        platform_bounds: Optional[Tuple[float, float]] = None,
        completion_checker: Optional[Callable[[], bool]] = None,
        approach_lead_px: float = 70.0,
        pass_through_px: float = 18.0,
        timeout_sec: float = 12.0,
    ) -> bool:
        """保持行走方向并按住 UP 穿过传送门触发区。

        传送门是一个有宽度的触发区，不是需要精确停靠的单点。远处
        先走到门前，到达接力线时不松方向键，再按住 UP 继续穿过
        门轴。穿越目标会被裁剪在平台安全区内。
        """
        start = get_player_pos()
        if start is None:
            return False
        current_x = float(start[0])
        trigger = float(trigger_x)
        safe_min = safe_max = None
        if platform_bounds is not None:
            safe_min = float(min(platform_bounds)) + 5.0
            safe_max = float(max(platform_bounds)) - 5.0
            if safe_min > safe_max:
                safe_min = safe_max = (float(platform_bounds[0]) + float(platform_bounds[1])) * 0.5

        delta = trigger - current_x
        if abs(delta) > 1.0:
            sign = 1.0 if delta > 0.0 else -1.0
        elif safe_min is not None and safe_max is not None:
            # 恰好站在门轴时，从平台内侧向靠近的边缘穿越。
            sign = -1.0 if trigger <= (safe_min + safe_max) * 0.5 else 1.0
        else:
            sign = 1.0

        def clamp_x(value: float) -> float:
            if safe_min is None or safe_max is None:
                return value
            return max(safe_min, min(safe_max, value))

        entry_x = clamp_x(trigger - sign * max(20.0, float(approach_lead_px)))
        pass_x = clamp_x(trigger + sign * max(8.0, float(pass_through_px)))
        # 18px 是门内穿越上限，不是到达后还能继续走的距离。预留
        # 6px 给观测延迟与松键惯性，避免切图较慢时走出触发区。
        stop_offset = max(0.0, (pass_x - trigger) * sign - 6.0)
        stop_x = clamp_x(trigger + sign * stop_offset)
        direction = "right" if sign > 0.0 else "left"
        started_at = time.perf_counter()

        self.stop()
        remaining_to_portal = (trigger - current_x) * sign
        if remaining_to_portal > max(20.0, float(approach_lead_px)):
            coarse_timeout = max(
                2.0,
                min(float(timeout_sec), abs(entry_x - current_x) / 80.0 + 2.0),
            )
            self.log(
                f"🚪 [传送门连续接近] 先到接力线X={entry_x:.1f}，"
                f"到位后保持{direction}不松键"
            )
            if not self.walk_to_x(
                target_x=entry_x,
                get_player_pos=get_player_pos,
                tolerance=10,
                timeout_sec=coarse_timeout,
                stop_event=stop_event,
                platform_bounds=platform_bounds,
                safe_margin=5,
                preserve_direction_on_arrival=True,
                completion_checker=completion_checker,
            ):
                self.stop()
                return False

        if completion_checker is not None:
            try:
                if bool(completion_checker()):
                    self.stop()
                    return True
            except Exception:
                pass
        if stop_event is not None and stop_event.is_set():
            self.stop()
            return False

        # 粗定位后方向键通常仍按着；若一开始就在接力线内，也要
        # 明确先按方向再按UP，不依赖 walk_to_x 的副作用。
        if self.current_held_key != direction:
            if self.current_held_key in ("left", "right"):
                self.driver.key_up(self.current_held_key)
            self.driver.key_down(direction)
            self.current_held_key = direction
        self.driver.key_down("up")
        self.log(
            f"🚪 [传送门连续穿越] 按住UP+{direction}，"
            f"门轴X={trigger:.1f}，门内止步线X={stop_x:.1f}"
        )
        # 只朝进门方向走，不再用 walk_to_x 在越过门轴后反向追逐单点。
        # 进入门轴附近刷新一次 UP 边沿，接着在门内止步线松开方向。
        axis_refreshed = False
        lost_at = None
        cross_deadline = time.perf_counter() + min(
            5.0, max(0.6, float(timeout_sec) - (time.perf_counter() - started_at))
        )
        while time.perf_counter() < cross_deadline:
            if stop_event is not None and stop_event.is_set():
                self.stop()
                return False
            if completion_checker is not None:
                try:
                    if bool(completion_checker()):
                        self.stop()
                        return True
                except Exception:
                    pass
            latest = get_player_pos()
            if latest is None:
                if lost_at is None:
                    lost_at = time.perf_counter()
                elif time.perf_counter() - lost_at >= 0.25:
                    self.log("🛡️ [传送门视觉暂失] 已到门前，停止横向输入等待切图")
                    break
                time.sleep(0.015)
                continue
            lost_at = None
            latest_x = float(latest[0])
            progress = (latest_x - trigger) * sign
            if progress >= -5.0 and not axis_refreshed:
                self.driver.key_up("up")
                time.sleep(0.012)
                self.driver.key_down("up")
                axis_refreshed = True
                self.log(
                    f"🚪 [传送门轴UP刷新] X={latest_x:.1f}，"
                    f"保持{direction}并重新触发UP"
                )
            braking_distance = 0.0
            if self.motion_model is not None:
                try:
                    speed = max(0.0, float(self.motion_model.vx) * sign)
                    drag = max(1.0, float(self.motion_model.drag_accel))
                    braking_distance = min(18.0, speed * speed / (2.0 * drag))
                except (AttributeError, TypeError, ValueError):
                    pass
            if (progress >= stop_offset or
                    (axis_refreshed and progress + braking_distance >= stop_offset)):
                self.log(
                    f"🛡️ [传送门范围止步] X={latest_x:.1f}，"
                    f"门轴后{progress:.1f}px，预测滑行={braking_distance:.1f}px；"
                    "松开方向，原地保持UP"
                )
                break
            if safe_min is not None and safe_max is not None and (
                (sign > 0.0 and latest_x >= safe_max) or
                (sign < 0.0 and latest_x <= safe_min)
            ):
                self.log(f"🛡️ [传送门平台边界] X={latest_x:.1f}，立即松开方向")
                break
            time.sleep(0.015)
        else:
            self.log("⚠️ [传送门接近超时] 未到门内止步线，停止横向输入")
            self.stop()
            return False

        if self.current_held_key == direction:
            self.driver.key_up(direction)
            self.current_held_key = None
        # 横向已经停在传送点内；只刷新 UP 等待切图，不能继续把人物
        # 从门的另一侧带出去。
        hold_deadline = time.perf_counter() + 0.45
        next_up_refresh = time.perf_counter() + 0.16
        while time.perf_counter() < hold_deadline:
            if stop_event is not None and stop_event.is_set():
                break
            if completion_checker is not None:
                try:
                    if bool(completion_checker()):
                        break
                except Exception:
                    pass
            now = time.perf_counter()
            if now >= next_up_refresh:
                self.driver.key_up("up")
                time.sleep(0.012)
                self.driver.key_down("up")
                next_up_refresh = now + 0.16
            time.sleep(0.015)
        self.stop()
        return stop_event is None or not stop_event.is_set()

    def down_jump(
        self,
        jump_key: Optional[str] = None,
        wait_land_sec: Optional[float] = None,
        timings_override: Optional[Dict[str, Any]] = None,
    ):
        """
        平台下跳 (Down + Jump组合键):
        自动下落至下一层平台踏板。支持 6 阶段精细化可配置时序。
        """
        self.stop()
        jk = jump_key or self.jump_key

        cfg = dict(self.config or {})
        if timings_override and isinstance(timings_override, dict):
            cfg.update(timings_override)

        # 6 个阶段时序参数 (单位: 秒)
        pre_neutral_sec = max(0.0, float(cfg.get("down_jump_pre_neutral_ms", 50.0))) / 1000.0
        down_prep_sec = max(0.0, float(cfg.get("down_jump_down_prep_ms", 109.0))) / 1000.0
        jump_hold_sec = max(0.01, float(cfg.get("down_jump_jump_hold_ms", 124.0))) / 1000.0
        retry_enabled = bool(cfg.get("down_jump_retry_enabled", False))
        retry_jump_hold_sec = max(0.01, float(cfg.get("down_jump_retry_jump_hold_ms", 100.0))) / 1000.0
        adaptive_full_retry_enabled = bool(
            cfg.get("down_jump_adaptive_full_retry_enabled", True)
        )
        post_down_hold_sec = max(0.0, float(cfg.get("down_jump_post_down_hold_ms", 1.0))) / 1000.0
        post_neutral_sec = max(0.0, float(cfg.get("down_jump_post_neutral_ms", 100.0))) / 1000.0

        if wait_land_sec is not None:
            effective_wait_land_sec = max(0.0, float(wait_land_sec))
        else:
            effective_wait_land_sec = max(0.0, float(cfg.get("down_jump_wait_land_ms", 450.0))) / 1000.0

        # 阶段 1：起跳前中性缓冲时间 (确保所有按键松开与窗口焦点)
        self.driver.ensure_focus()
        if pre_neutral_sec > 0:
            time.sleep(pre_neutral_sec)

        started_at = time.perf_counter()
        down_at = started_at
        alt_down_at = None
        alt_up_at = None
        down_up_at = None
        jump_retry_down_at = None
        jump_retry_up_at = None
        adaptive_retry_attempted = False
        adaptive_retry_confirmed = False
        first_downward_confirmed = False
        downward_confirmed = False
        down_held = False
        jump_held = False
        baseline_pos = None
        try:
            measurement_step = float(
                getattr(self.motion_model, "measurement_step_px", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            measurement_step = 0.0
        # 单个黄点量化格或斜坡横移可让 raw Y 抖动20余像素，不能把它
        # 当成穿台成功；真实下跳在确认窗内会产生更明显的向下位移。
        down_confirm_delta = max(30.0, measurement_step * 1.25)

        def moved_down(reference_pos=None) -> bool:
            if reference_pos is None or self.raw_position_getter is None:
                return False
            try:
                current_pos = self.raw_position_getter()
                return bool(
                    current_pos is not None
                    and float(current_pos[1])
                    > float(reference_pos[1]) + down_confirm_delta
                )
            except Exception:
                return False

        try:
            # 阶段 2：按下方向下键 (DOWN) 预压
            self.driver.key_down("down")
            down_held = True
            if down_prep_sec > 0:
                time.sleep(down_prep_sec)

            baseline_pos = self.raw_position_getter() if self.raw_position_getter else None

            # 阶段 3：在保持 DOWN 的同时按下 Jump 键重叠
            alt_down_at = time.perf_counter()
            self.driver.key_down(jk)
            jump_held = True
            if jump_hold_sec > 0:
                time.sleep(jump_hold_sec)
            alt_up_at = time.perf_counter()
            self.driver.key_up(jk)
            jump_held = False

            first_downward_confirmed = moved_down(baseline_pos)
            downward_confirmed = first_downward_confirmed

            # 阶段 4：未检测到下落时的补按跳跃
            if retry_enabled and not downward_confirmed:
                jump_retry_down_at = time.perf_counter()
                self.driver.key_down(jk)
                jump_held = True
                if retry_jump_hold_sec > 0:
                    time.sleep(retry_jump_hold_sec)
                jump_retry_up_at = time.perf_counter()
                self.driver.key_up(jk)
                jump_held = False
                downward_confirmed = moved_down(baseline_pos)

            # 阶段 5：Jump 松开后 DOWN 保持与释放等待
            confirm_deadline = time.perf_counter() + post_down_hold_sec
            while not downward_confirmed and time.perf_counter() < confirm_deadline:
                remaining = confirm_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                # 手动录制的中位数是 Alt 与 Down 几乎同时松开。
                # 旧的15ms固定轮询会把UI中的1ms强行拉长到15ms；
                # 小窗口要按真实剩余时间睡眠，不得改写按键时序。
                time.sleep(min(0.015, remaining))
                downward_confirmed = moved_down(baseline_pos)
            down_up_at = time.perf_counter()
            self.driver.key_up("down")
            down_held = False
            if post_neutral_sec > 0:
                time.sleep(post_neutral_sec)
        finally:
            # 即使窗口或输入驱动临时异常，也不能留下 DOWN/ALT 按死
            if jump_held:
                self.driver.key_up(jk)
            if down_held:
                self.driver.key_up("down")

        # 固定时序发完并不代表游戏已经接受了下跳。raw Y 在成功穿台后会
        # 很快向下增大；若首轮完整序列结束仍没有任何下落证据，先彻底
        # 松键，再使用“短DOWN预压 + 80ms跳键”的完整组合重发一次。
        # 这与上面的“保持DOWN只补按Jump”是两种机制，后者即使被用户
        # 关闭，F6仍可依靠本闭环从一次漏键/趴下中恢复。
        downward_confirmed = bool(downward_confirmed or moved_down(baseline_pos))
        if (
            adaptive_full_retry_enabled
            and not downward_confirmed
            and baseline_pos is not None
            and self.raw_position_getter is not None
        ):
            adaptive_retry_attempted = True
            retry_baseline = self.raw_position_getter()
            adaptive_down_prep_sec = min(0.025, max(0.012, down_prep_sec))
            adaptive_jump_hold_sec = 0.080
            adaptive_confirm_sec = max(0.12, min(0.25, post_down_hold_sec))
            self.log(
                "🔁 [下跳时序闭环] 首轮raw Y未出现下落，全部松键后改用完整组合重试："
                f"DOWN预压={adaptive_down_prep_sec * 1000:.0f}ms，"
                f"DOWN+{jk.upper()}={adaptive_jump_hold_sec * 1000:.0f}ms，"
                f"确认位移>{down_confirm_delta:.1f}px"
            )
            self.stop()
            time.sleep(0.060)
            adaptive_down_held = False
            adaptive_jump_held = False
            try:
                self.driver.key_down("down")
                adaptive_down_held = True
                time.sleep(adaptive_down_prep_sec)
                self.driver.key_down(jk)
                adaptive_jump_held = True
                time.sleep(adaptive_jump_hold_sec)
                self.driver.key_up(jk)
                adaptive_jump_held = False
                confirm_until = time.perf_counter() + adaptive_confirm_sec
                while time.perf_counter() < confirm_until:
                    if moved_down(retry_baseline):
                        adaptive_retry_confirmed = True
                        break
                    time.sleep(0.015)
            finally:
                if adaptive_jump_held:
                    self.driver.key_up(jk)
                if adaptive_down_held:
                    self.driver.key_up("down")
            downward_confirmed = bool(
                adaptive_retry_confirmed or moved_down(retry_baseline)
            )
            self.log(
                f"{'✅' if downward_confirmed else '⚠️'} [下跳时序闭环] "
                f"完整组合重试后raw下落确认={'是' if downward_confirmed else '否'}"
            )
            time.sleep(0.080)

        if alt_down_at is not None and alt_up_at is not None and down_up_at is not None:
            self.log(
                "⌨️ [下跳按键时序] "
                f"阶段1起跳前缓冲={pre_neutral_sec * 1000:.0f}ms, "
                f"阶段2DOWN预压={(alt_down_at - down_at) * 1000:.0f}ms, "
                f"阶段3DOWN+{jk.upper()}重叠={(alt_up_at - alt_down_at) * 1000:.0f}ms, "
                f"阶段4首跳确认={'是' if first_downward_confirmed else '否'}, "
                f"补按Jump={'是(' + f'{(jump_retry_up_at - jump_retry_down_at) * 1000:.0f}ms)' if jump_retry_down_at is not None else ('跳过' if not retry_enabled else '否')}, "
                f"阶段5ALT松开后DOWN保持={(down_up_at - (jump_retry_up_at or alt_up_at)) * 1000:.0f}ms, "
                f"DOWN松开后等待={post_neutral_sec * 1000:.0f}ms, "
                f"自适应完整重试={'是' if adaptive_retry_attempted else '否'}, "
                f"raw下落确认={'是' if downward_confirmed else '否'}, "
                f"阶段6下落着陆等待={effective_wait_land_sec * 1000:.0f}ms"
            )

        # 阶段 6：重力下落着陆等待
        if effective_wait_land_sec > 0:
            time.sleep(effective_wait_land_sec)
        return downward_confirmed

    def walk_off_drop(
        self,
        direction: str,
        start_x: float,
        source_bounds: Tuple[int, int],
        source_y: float,
        target_bounds: Tuple[int, int],
        target_y: float,
        walk_speed: float = 125.0,
        gravity: float = 2000.0,
        terminal_fall_speed: float = 670.0,
        stop_event=None,
        landed_on_target_fn: Optional[Callable[[], bool]] = None,
        brake_on_landing: bool = False,
        get_player_pos: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
    ) -> Tuple[float, float]:
        """按 v83 运动学模型走出边缘并自然下落；不发送跳跃键。

        返回 (预测总按键秒数, 预测落点 X)。先走到边缘；进入空中后
        仅保持到目标平台安全落点附近就松键，不能整段下落一直按方向，
        否则在相邻窄平台上会直接越过目标。
        """
        key = "right" if direction == "right" else "left"
        edge_x = float(source_bounds[1] if direction == "right" else source_bounds[0])
        drop_y = max(0.0, float(target_y) - float(source_y))

        # y(t)=1/2*g*t^2，达到终端速度后改为匀速下落。
        terminal_t = terminal_fall_speed / gravity
        terminal_y = 0.5 * gravity * terminal_t * terminal_t
        if drop_y <= terminal_y:
            fall_sec = (2.0 * drop_y / gravity) ** 0.5
        else:
            fall_sec = terminal_t + (drop_y - terminal_y) / terminal_fall_speed

        to_edge = max(0.0, (edge_x - start_x) if direction == "right" else (start_x - edge_x))
        edge_sec = to_edge / walk_speed
        # 选择目标平台靠近起跳边缘的一侧作为安全落点，留出 8px 边距。
        # 这样 P44->P32 这类只有数像素间隙的边会“走出即松”，
        # 而不是在约 1.7 秒的下落全程继续横向加速。
        margin = min(16.0, max(8.0, (target_bounds[1] - target_bounds[0]) * 0.20))
        desired_landing_x = (
            float(target_bounds[0]) + margin
            if direction == "right"
            else float(target_bounds[1]) - margin
        )
        air_distance = max(
            0.0,
            (desired_landing_x - edge_x) if direction == "right" else (edge_x - desired_landing_x),
        )
        air_hold_sec = min(fall_sec, max(0.05, air_distance / walk_speed))
        hold_sec = edge_sec + air_hold_sec
        landing_x = edge_x + (walk_speed * air_hold_sec if direction == "right" else -walk_speed * air_hold_sec)
        if stop_event is not None and stop_event.is_set():
            self.stop()
            return (hold_sec, landing_x)
        self.driver.key_down(key)
        try:
            if get_player_pos is None:
                deadline = time.perf_counter() + max(0.05, hold_sec)
                while time.perf_counter() < deadline:
                    if stop_event is not None and stop_event.is_set():
                        self.stop()
                        return (hold_sec, landing_x)
                    time.sleep(0.005)
            else:
                # 长平台不能用固定 125px/s 开环推算到边缘。实际速度只要
                # 低几个百分点，跑数百像素后就会刚好停在边缘内（实机
                # P81 左缘停在 -507，而边缘是 -508），永远不会下落。
                # 先闭环确认角色中心确实越过边缘，再计算空中保持时间。
                crossing_margin = 4.0
                edge_deadline = time.perf_counter() + max(1.0, edge_sec + 1.25)
                crossed_at = None
                while time.perf_counter() < edge_deadline:
                    if stop_event is not None and stop_event.is_set():
                        self.stop()
                        return (hold_sec, landing_x)
                    pos = get_player_pos()
                    if pos is not None:
                        cur_x = float(pos[0])
                        crossed = (
                            cur_x >= edge_x + crossing_margin
                            if direction == "right"
                            else cur_x <= edge_x - crossing_margin
                        )
                        if crossed:
                            crossed_at = time.perf_counter()
                            break
                    time.sleep(0.01)

                if crossed_at is not None:
                    air_deadline = crossed_at + air_hold_sec
                    while time.perf_counter() < air_deadline:
                        if stop_event is not None and stop_event.is_set():
                            self.stop()
                            return (hold_sec, landing_x)
                        time.sleep(0.005)
        finally:
            self.driver.key_up(key)
        # F6 连续导航不能只凭预估睡眠返回：角色可能刚触及窄平台仍
        # 带横向速度，下一帧就会继续滑出平台。等待平台状态确认，确认
        # 后立即给极短反向刹车，单步和自动运行使用同一落地语义。
        landed = False
        if landed_on_target_fn is not None:
            # 单帧平台命中只是黄点穿过承重层，不能代表角色已经站稳。
            # 连续 3 次（约 240ms）命中后才交给下一条导航边，防止 P22
            # 落 P9 后带着惯性立即执行 P9 的下一跳而直接掉穿。
            wait_until = time.perf_counter() + max(0.45, fall_sec + 0.55)
            stable_hits = 0
            while time.perf_counter() < wait_until:
                if stop_event is not None and stop_event.is_set():
                    self.stop()
                    return (hold_sec, landing_x)
                try:
                    landed = bool(landed_on_target_fn())
                except Exception:
                    landed = False
                if landed:
                    stable_hits += 1
                    if stable_hits >= 3:
                        break
                else:
                    stable_hits = 0
                if landed and stable_hits >= 3:
                    break
                time.sleep(0.08)
            landed = stable_hits >= 3
        if landed and brake_on_landing:
            brake_key = "left" if key == "right" else "right"
            self.driver.press_key(brake_key, duration_ms=28)
            self.log(f"🛑 [自然下落落地刹车] 已确认目标平台，反向{brake_key} 28ms")
        # 刹车后仍给角色一个完整的小地图采样间隔，确保下一段读取的是
        # 已停止在平台上的位置，而不是落地过程中的旧点。
        time.sleep(0.16 if landed else 0.08)
        return (hold_sec, landing_x)

    def jump_step(
        self,
        direction: Optional[str] = None,
        jump_key: Optional[str] = None,
        run_up_sec: float = 0.22,
        airborne_hold_sec: float = 0.48,
        wait_land_sec: float = 0.55,
        direction_already_held: bool = False,
        before_jump_callback: Optional[Callable[[], None]] = None,
        takeoff_x: Optional[float] = None,
        takeoff_x_range: Optional[Tuple[float, float]] = None,
        get_player_pos: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
        source_standing_check: Optional[Callable[[], bool]] = None,
    ):
        """
        真正的连贯助跑大跳 / 跨越平台跳跃 (带起跳前全速冲刺 + 空中全长惯性保持):
        1. 提前按住方向键 run_up_sec (默认 220ms) 建立最高行走初速度；
        2. 在方向键持续按住的状态下，按下 Jump 键起跳 (80ms)；
        3. 在空中持续保持方向键按住 airborne_hold_sec (默认 480ms)，维持抛物线最大飞行跨度；
        4. 落地后松开方向键，等待物理落地缓冲。
        """
        self.last_jump_gate_aborted = False
        jk = jump_key or self.jump_key
        # 宽平台普通跳可由 walk_to_x 保持同向按键无缝衔接；其他
        # 动作仍先急停，防止沿用上段残余输入。
        reuse_direction = bool(
            direction_already_held
            and direction in ("left", "right")
            and self.current_held_key == direction
        )
        if not reuse_direction:
            self.stop()
        if direction in ("left", "right"):
            self.driver.ensure_focus()
            if source_standing_check is not None:
                try:
                    source_ready = bool(source_standing_check())
                except Exception:
                    source_ready = False
                if not source_ready:
                    self.last_jump_gate_aborted = True
                    self.log("⚠️ [起跳源平台丢失] Alt前已不在源平台，本轮禁止跳跃并重新定位")
                    self.stop()
                    return False
            if not reuse_direction:
                self.driver.key_down(direction)
                self.current_held_key = direction
            if takeoff_x is not None and get_player_pos is not None:
                # 固定助跑时间无法保证跨过拓扑起跳线：起步加速、斜坡和
                # 小地图量化都会改变实际位移。保持方向直到调用方提供的
                # 连续控制 X 到达/跨过起跳线，再按 Alt；设置短超时防止
                # 撞墙死等。raw X 只适合校验，不能命中位于两格之间的线。
                initial_gate_pos = get_player_pos()
                initial_remaining = 0.0
                if initial_gate_pos is not None:
                    initial_remaining = max(
                        0.0,
                        (float(takeoff_x) - float(initial_gate_pos[0]))
                        if direction == "right"
                        else (float(initial_gate_pos[0]) - float(takeoff_x)),
                    )
                # 0.4s 固定门槛只够理想的 27.5px 助跑；受斜坡、刚刹停
                # 或一帧坐标延迟影响时会在距线数像素处超时。按当前剩余距离
                # 给出有限的动态预算，源平台守卫仍会在踏空时立即终止。
                gate_budget = min(
                    1.05,
                    max(0.55, run_up_sec + 0.25, initial_remaining / 70.0 + 0.22),
                )
                gate_deadline = time.perf_counter() + gate_budget
                # 连续 Kalman 坐标在斜坡加速时一帧可跨过约5~8px。容差由
                # “特殊参数”实时读取；源平台 raw 高度保险仍负责防止踏空，
                # 但量化 raw X 不再否决已经到线的连续坐标。
                try:
                    gate_tolerance = float(
                        self.takeoff_gate_tolerance_getter()
                        if self.takeoff_gate_tolerance_getter is not None
                        else 20.0
                    )
                except (TypeError, ValueError):
                    gate_tolerance = 20.0
                gate_tolerance = max(1.0, min(100.0, gate_tolerance))
                safe_gate_range = None
                if takeoff_x_range is not None:
                    try:
                        safe_lo = float(min(takeoff_x_range))
                        safe_hi = float(max(takeoff_x_range))
                        if math.isfinite(safe_lo) and math.isfinite(safe_hi) and safe_hi >= safe_lo:
                            safe_gate_range = (safe_lo, safe_hi)
                    except (TypeError, ValueError):
                        safe_gate_range = None
                crossed = False
                last_gate_pos = None
                last_raw_gate_pos = None
                source_lost_samples = 0
                while time.perf_counter() < gate_deadline:
                    if source_standing_check is not None:
                        try:
                            still_standing = bool(source_standing_check())
                        except Exception:
                            still_standing = False
                        source_lost_samples = 0 if still_standing else source_lost_samples + 1
                        if source_lost_samples >= 2:
                            self.last_jump_gate_aborted = True
                            self.log(
                                "⚠️ [起跳源平台丢失] 等待起跳线时已离开源平台，"
                                "立即松开方向且不发送Alt"
                            )
                            self.driver.key_up(direction)
                            if self.current_held_key == direction:
                                self.current_held_key = None
                            return False
                    last_gate_pos = get_player_pos()
                    last_raw_gate_pos = (
                        self.raw_position_getter()
                        if self.raw_position_getter else None
                    )
                    if last_gate_pos is not None:
                        gate_x = float(last_gate_pos[0])
                        if safe_gate_range is not None:
                            # foothold 规划器已经根据源/目标短段斜率和落点
                            # 算出了真实安全区，不能再用中心±全局容差向助跑
                            # 来向外扩张。P16->P17 的安全区右界是 -759，旧
                            # ±20 会在 -746 就放行，直接丢掉了斜台几何信息。
                            entry_x = (
                                safe_gate_range[0]
                                if direction == "right"
                                else safe_gate_range[1]
                            )
                            continuous_crossed = (
                                gate_x >= entry_x
                                if direction == "right"
                                else gate_x <= entry_x
                            )
                            # raw 坐标虽有量化，但“是否越过区间入口”是单向
                            # 判定，不要求命中某个像素。两套坐标同时入区才
                            # 发 Alt，避免 Kalman 在斜坡加速时预测领先。
                            raw_crossed = True
                            if last_raw_gate_pos is not None:
                                raw_x = float(last_raw_gate_pos[0])
                                raw_crossed = (
                                    raw_x >= entry_x
                                    if direction == "right"
                                    else raw_x <= entry_x
                                )
                            crossed = continuous_crossed and raw_crossed
                        else:
                            continuous_crossed = (
                                gate_x >= float(takeoff_x) - gate_tolerance
                                if direction == "right"
                                else gate_x <= float(takeoff_x) + gate_tolerance
                            )
                            crossed = continuous_crossed
                        if crossed:
                            break
                    time.sleep(0.01)
                if crossed:
                    if source_standing_check is not None:
                        try:
                            still_standing = bool(source_standing_check())
                        except Exception:
                            still_standing = False
                        if not still_standing:
                            self.last_jump_gate_aborted = True
                            self.log(
                                "⚠️ [起跳源平台丢失] 已到起跳线但raw高度不在源平台，"
                                "禁止发送Alt"
                            )
                            self.driver.key_up(direction)
                            if self.current_held_key == direction:
                                self.current_held_key = None
                            return False
                    self.log(
                        f"🎯 [起跳线闭环] controlX={float(last_gate_pos[0]):.1f} "
                        f"rawX={float(last_raw_gate_pos[0]):.1f} "
                        if last_raw_gate_pos is not None else
                        f"🎯 [起跳线闭环] controlX={float(last_gate_pos[0]):.1f} rawX=None "
                    )
                    self.log(
                        f"🎯 [起跳线触发] 计划X={float(takeoff_x):.1f}，"
                        + (
                            f"foothold安全区=[{safe_gate_range[0]:.1f},"
                            f"{safe_gate_range[1]:.1f}]"
                            if safe_gate_range is not None
                            else f"兼容容差=±{gate_tolerance:.1f}px"
                        )
                        + f"，预算={gate_budget:.2f}s，触发Alt"
                    )
                else:
                    self.last_jump_gate_aborted = True
                    self.log(
                        f"⚠️ [起跳线等待超时] control={last_gate_pos}，"
                        f"raw={last_raw_gate_pos}，计划X={float(takeoff_x):.1f}，"
                        f"预算={gate_budget:.2f}s，取消本轮Alt并原边重试"
                    )
                    self.driver.key_up(direction)
                    if self.current_held_key == direction:
                        self.current_held_key = None
                    return False
            else:
                time.sleep(run_up_sec) # 1. 助跑加速 (220ms 建立最高初速度)
            # 在 Alt 真正按下前采样一次位置。调用方可据此区分“规划
            # 起跳点不合理”和“下坡助跑导致越过起跳点”，不改变输入时序。
            if before_jump_callback is not None:
                try:
                    before_jump_callback()
                except Exception:
                    pass
            self.driver.key_down(jk) # 2. 奔跑中起跳 (不松开方向键)
            time.sleep(0.08)
            self.driver.key_up(jk)
            time.sleep(airborne_hold_sec) # 3. 空中保持全长水平推进惯性 (480ms)
            self.driver.key_up(direction)
            if self.current_held_key == direction:
                self.current_held_key = None
        else:
            self.driver.press_key(jk, duration_ms=60)

        if wait_land_sec > 0:
            time.sleep(wait_land_sec)
        return True

    def teleport(
        self,
        direction: str = "right",
        override_key: Optional[str] = None,
        override_vk: Optional[int] = None,
    ) -> bool:
        """
        执行法师瞬移技能:
        :param direction: "up", "left", "right", "down" (down 自动替换为下跳)
        :param override_key: 临时覆盖瞬移按键
        :param override_vk: 临时覆盖虚拟键码
        :return: 是否成功触发瞬移
        """
        if direction == "down":
            self.down_jump()
            return True

        if direction not in ("up", "left", "right"):
            return False

        if not self.enable_teleport:
            return False

        now = time.perf_counter()
        if now < self.next_teleport_ts:
            return False

        tp_key = override_key or self.teleport_key
        tp_vk = override_vk if override_vk is not None else self.teleport_vk

        # 方向键处理：若当前没有按住该方向键，则按下方向键；瞬移按键释放后再释放方向键
        need_release_dir = False
        if direction in ("left", "right"):
            if self.current_held_key != direction:
                self.driver.key_down(direction)
                time.sleep(0.025)
                need_release_dir = True
        elif direction == "up":
            self.driver.key_down("up")
            time.sleep(0.030)
            need_release_dir = True

        self.last_teleport_command_at = time.perf_counter()
        self.last_teleport_direction = direction
        try:
            self.driver.press_key(tp_key, duration_ms=random.randint(55, 75), vk_code=tp_vk)
            if direction == "up":
                time.sleep(0.030)
        finally:
            if need_release_dir:
                self.driver.key_up(direction)

        # 随机抽取下次 CD 间隔并在冷却完成后方可下一次瞬移
        cd_min = min(self.teleport_cd_min_ms, self.teleport_cd_max_ms)
        cd_max = max(self.teleport_cd_min_ms, self.teleport_cd_max_ms)
        cd_ms = random.uniform(cd_min, cd_max)
        self.next_teleport_ts = time.perf_counter() + (cd_ms / 1000.0)
        self.log(f"⚡ [法师瞬移] 方向={direction} 按键={tp_key.upper()}，下次CD随机={cd_ms:.1f}ms")
        return True

    def face_direction(self, target_dir: str, current_facing: Optional[str] = None,
                       duration_ms: int = 30):
        """
        瞬时角色朝向调整；通常轻点 30ms，攻击脱困可用稍长脉冲。
        """
        if target_dir not in ("left", "right"):
            return
        if current_facing is not None and current_facing == target_dir:
            return

        self.driver.press_key(target_dir, duration_ms=max(30, min(120, int(duration_ms))))
        time.sleep(0.02)

    def micro_nudge(
        self,
        direction: str,
        duration_sec: float = 0.35,
        speed_scale: float = 0.15,
        stop_event=None,
    ):
        """低占空比微动，用于落地后刷新小地图视口而不依赖世界坐标。"""
        if direction not in ("left", "right"):
            return
        self.stop()
        end_time = time.perf_counter() + max(0.05, duration_sec)
        on_sec = max(0.012, 0.05 * max(0.15, min(1.0, speed_scale)))
        off_sec = max(0.005, 0.05 - on_sec)
        while time.perf_counter() < end_time:
            if stop_event is not None and stop_event.is_set():
                break
            self.driver.key_down(direction)
            time.sleep(on_sec)
            self.driver.key_up(direction)
            time.sleep(off_sec)
        self.stop()

    def timed_walk(self, direction: str, duration_sec: float, stop_event=None) -> None:
        """持续按住一个方向一小段时间，用于坐标观测遮挡后的安全脱离。"""
        if direction not in ("left", "right"):
            return
        self.stop()
        self.driver.key_down(direction)
        self.current_held_key = direction
        try:
            deadline = time.perf_counter() + max(0.0, float(duration_sec))
            while time.perf_counter() < deadline:
                if stop_event is not None and stop_event.is_set():
                    break
                time.sleep(0.01)
        finally:
            self.driver.key_up(direction)
            if self.current_held_key == direction:
                self.current_held_key = None

    def _adaptive_top_exit(
        self,
        *,
        target_ladder_x: float,
        get_raw_player_pos: Optional[
            Callable[[], Optional[Tuple[float, float]]]
        ],
        landing_confirm_fn: Optional[Callable[[], bool]],
        step_off_direction: Optional[str],
        top_hold_sec: float,
        stop_event=None,
        is_landed_fn: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """以真实水平响应闭环确认绳顶脱离，而不是固定等待后猜测成功。

        到达目标高度时角色仍可能挂在绳轴上；此时拓扑平台和吸附后的
        Kalman X 都不能证明已经落台。这里交替执行短 UP 推进与向平台
        内侧探测，只接受 raw 黄点离开绳轴后的连续落台证据。

        ``top_hold_sec`` 保留原配置兼容性，但只作为恢复预算基准，不再
        是一次固定 sleep。一般落台会在第一次探测后立即返回。
        """
        self.last_landing_confirmed = False
        self.last_top_exit_failed = False

        direction = step_off_direction if step_off_direction in ("left", "right") else None
        base_sec = max(0.10, min(2.0, float(top_hold_sec)))
        recovery_budget = max(3.0, min(8.0, base_sec * 6.0))
        started_at = time.perf_counter()

        try:
            measurement_step = float(
                getattr(self.motion_model, "measurement_step_px", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            measurement_step = 0.0
        # raw 黄点一格在不同地图对应的世界距离不同。离绳阈值跟随当前
        # 地图比例，不能写死为某个世界像素值。
        rope_clearance = max(8.0, min(30.0, measurement_step * 0.60))
        entry_raw = get_raw_player_pos() if get_raw_player_pos else None
        entry_raw_x = float(entry_raw[0]) if entry_raw is not None else None
        # 平台拓扑可能在人物仍挂绳时提前命中。脱绳闭环必须观察到按下
        # 内侧方向之后新增的定向水平位移，不能把进入闭环前已经存在的
        # 黄点量化偏差当作离绳证据。
        exit_progress_gate = max(
            4.0,
            min(12.0, measurement_step * 0.50 if measurement_step > 0.0 else 6.0),
        )

        def directional_progress() -> float:
            raw_pos = get_raw_player_pos() if get_raw_player_pos else None
            if raw_pos is None or entry_raw_x is None or direction is None:
                return 0.0
            delta = float(raw_pos[0]) - entry_raw_x
            return delta if direction == "right" else -delta

        def physical_landing_confirmed() -> bool:
            if directional_progress() < exit_progress_gate:
                return False
            return bool(landing_confirm_fn and landing_confirm_fn())

        if direction is None or landing_confirm_fn is None:
            # 旧调用方没有提供物理落台回调时无法完成严格闭环；仍将固定
            # 长等待缩成一次基础推进，并明确记录为兼容降级。
            self.log(
                "⚠️ [绳顶闭环降级] 缺少平台内侧方向或raw落台回调，"
                f"仅保持UP {base_sec:.2f}s"
            )
            time.sleep(base_sec)
            self.driver.key_up("up")
            if direction is not None:
                self.driver.press_key(direction, duration_ms=120)
                time.sleep(0.15)
            self.last_landing_confirmed = True
            return True

        self.log(
            "🔄 [自适应绳顶脱离] 开始UP推进+内侧横移闭环，"
            f"raw离绳阈值={rope_clearance:.1f}px，"
            f"定向位移门槛={exit_progress_gate:.1f}px，起点rawX={entry_raw_x}，"
            f"最长{recovery_budget:.2f}s"
        )
        attempt = 0
        try:
            while (time.perf_counter() - started_at) < recovery_budget:
                if stop_event is not None and stop_event.is_set():
                    return False

                attempt += 1
                self.driver.ensure_focus()

                # 无缝接力：到达绳顶时 UP 仍处于按下状态，先压住目标平台
                # 内侧方向，再在短暂重叠后只松开 UP。这样人物完成翻台的
                # 同一帧即可向平台内部走，不会出现“全键松开”的机械停顿。
                up_pulse_sec = min(0.28, 0.10 + 0.035 * (attempt - 1))
                before_raw = get_raw_player_pos() if get_raw_player_pos else None
                probe_ms = min(200, 80 + (attempt - 1) * 30)
                raw_clear = False

                self.driver.key_down("up")
                self.driver.key_down(direction)
                up_deadline = time.perf_counter() + up_pulse_sec
                while time.perf_counter() < up_deadline:
                    if stop_event is not None and stop_event.is_set():
                        return False
                    if physical_landing_confirmed():
                        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                        after_raw = get_raw_player_pos() if get_raw_player_pos else None
                        self.last_landing_confirmed = True
                        self.log(
                            "✅ [无缝脱绳确认] UP与内侧方向接力期间已连续落台，"
                            f"轮次={attempt}，耗时={elapsed_ms:.0f}ms，"
                            f"定向位移={directional_progress():.1f}px，raw={after_raw}"
                        )
                        return True
                    time.sleep(0.015)
                self.driver.key_up("up")

                # UP 松开后方向键仍连续保持一小段时间，同时进行落台采样；
                # 不再等横移脉冲全部结束后才检查，确认后可立即交还FSM。
                direction_deadline = time.perf_counter() + probe_ms / 1000.0
                while time.perf_counter() < direction_deadline:
                    if stop_event is not None and stop_event.is_set():
                        return False
                    if physical_landing_confirmed():
                        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                        after_raw = get_raw_player_pos() if get_raw_player_pos else None
                        self.last_landing_confirmed = True
                        self.log(
                            "✅ [无缝脱绳确认] 横向键未中断并已连续落台，"
                            f"轮次={attempt}，探测={probe_ms}ms，耗时={elapsed_ms:.0f}ms，"
                            f"定向位移={directional_progress():.1f}px，raw={after_raw}"
                        )
                        return True
                    raw_pos = get_raw_player_pos() if get_raw_player_pos else None
                    if raw_pos is not None:
                        raw_clear = abs(float(raw_pos[0]) - float(target_ladder_x)) > rope_clearance
                    time.sleep(0.015)
                self.driver.key_up(direction)

                # 方向脉冲结束后只留很短的物理收束窗口。大多数正常翻台会
                # 在上面的按键保持期间确认，不再产生肉眼可见的停顿。
                probe_deadline = min(
                    started_at + recovery_budget,
                    time.perf_counter() + 0.18,
                )
                while time.perf_counter() < probe_deadline:
                    if stop_event is not None and stop_event.is_set():
                        return False
                    if physical_landing_confirmed():
                        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                        after_raw = get_raw_player_pos() if get_raw_player_pos else None
                        self.last_landing_confirmed = True
                        self.log(
                            "✅ [自适应脱绳确认] raw坐标已离开绳轴并连续落台，"
                            f"轮次={attempt}，探测={probe_ms}ms，耗时={elapsed_ms:.0f}ms，"
                            f"定向位移={directional_progress():.1f}px，raw={after_raw}"
                        )
                        return True

                    raw_pos = get_raw_player_pos() if get_raw_player_pos else None
                    if raw_pos is not None:
                        raw_clear = abs(float(raw_pos[0]) - float(target_ladder_x)) > rope_clearance
                    time.sleep(0.025)

                after_raw = get_raw_player_pos() if get_raw_player_pos else None
                raw_delta = None
                if before_raw is not None and after_raw is not None:
                    raw_delta = float(after_raw[0]) - float(before_raw[0])
                self.log(
                    "↻ [绳顶仍未脱离] "
                    f"第{attempt}轮UP={up_pulse_sec:.3f}s/内侧={probe_ms}ms，"
                    f"rawΔX={raw_delta if raw_delta is not None else '未知'}，"
                    f"离绳={raw_clear}；继续闭环恢复"
                )

                # 无 landing_confirm_fn 的兼容路径上方已处理。这里的
                # is_landed_fn 只可作诊断，绝不能覆盖 raw 物理证据。
                if is_landed_fn is not None:
                    try:
                        if is_landed_fn() and not raw_clear:
                            self.log(
                                "🟡 [绳顶平台假阳性] 拓扑已命中目标平台，"
                                "但raw仍贴绳轴，拒绝提前结束"
                            )
                    except Exception:
                        pass
        finally:
            self.driver.key_up("up")
            self.driver.key_up(direction)

        self.last_top_exit_failed = True
        self.log(
            "⛔ [绳顶脱离超时] 已确认抓绳但未取得raw物理落台证据；"
            "标记为绳顶恢复失败，禁止重新起跳"
        )
        return False

    def _top_exit_entry_gate(self, target_surface_y: float) -> Tuple[float, float]:
        """返回允许横移探测的角色Y门槛及动态容差。

        拓扑边的 ``target_y`` 表示 foothold 平台表面，而黄点世界Y表示
        角色站立中心，正常承重位置固定在表面上方约45px。旧逻辑直接用
        ``target_y + 15``，会在角色仍位于绳顶下方约60px时提前停爬。
        """
        try:
            measurement_step = float(
                getattr(self.motion_model, "measurement_step_px", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            measurement_step = 0.0
        # 容差只用于覆盖raw黄点量化边界，不能大到重新提前一个人物高度。
        y_tolerance = max(
            2.0,
            min(6.0, measurement_step * 0.25 if measurement_step > 0.0 else 4.0),
        )
        standing_center_y = float(target_surface_y) - 45.0
        return standing_center_y + y_tolerance, y_tolerance

    def walk_and_grab_ladder(
        self,
        target_ladder_x: int,
        get_player_pos: Callable[[], Optional[Tuple[int, int]]],
        climb_direction: str = "up",
        target_y: Optional[int] = None,
        step_off_direction: Optional[str] = None,
        platform_bounds: Optional[Tuple[int, int]] = None,
        approach_tolerance: int = 18,
        timeout_sec: float = 6.0,
        stop_event = None,
        is_climbing_fn: Optional[Callable[[], bool]] = None,
        top_hold_sec: float = 1.0,
        get_raw_player_pos: Optional[
            Callable[[], Optional[Tuple[float, float]]]
        ] = None,
        landing_confirm_fn: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """平走抓梯/抓绳控制器 (带防跌落保护与自动登顶移出踏板)"""
        self.last_landing_confirmed = False
        self.last_top_exit_failed = False
        start_time = time.perf_counter()
        climb_key = "up" if climb_direction == "up" else "down"

        # 边缘安全夹紧
        eff_target_x = target_ladder_x
        if platform_bounds:
            p_min, p_max = platform_bounds
            eff_target_x = max(p_min + 18, min(p_max - 18, target_ladder_x))

        # 根据初始距离自适应放大超时时间 (按 100px/s 估算并保留裕量)
        init_p = get_player_pos()
        eff_timeout = max(timeout_sec, (abs(eff_target_x - init_p[0]) / 90.0 + 4.0) if init_p else timeout_sec)
        # 宏观世界坐标来自小地图黄点，精度受当前地图的比例限制。
        # 用 2 个小地图像素作为吸附窗口，而不是固定 ±18 世界像素。
        # 例：1 mini-px = 20 world-px 时，窗口即 ±40 world-px。
        minimap_step = None
        if self.motion_model is not None:
            try:
                minimap_step = max(1.0, float(getattr(self.motion_model, "measurement_step_px", 0.0)))
            except (TypeError, ValueError):
                minimap_step = None
        if minimap_step is not None:
            approach_tolerance = 2.0 * minimap_step
            self.log(
                f"📏 [梯绳吸附容差] 小地图1px≈{minimap_step:.1f}世界px，"
                f"使用±{approach_tolerance:.1f}px"
            )

        # 直爬向上时，在预计抵达绳梯前 0.1 秒预先按住 UP；客户端会在
        # 横向穿过梯绳判定线的瞬间吸附，不能等到已走进容差才开始按。
        prehold_distance = 0.0
        if climb_direction == "up":
            speed_ratio = 1.0
            if self.motion_model is not None:
                try:
                    speed_ratio = max(0.5, float(getattr(self.motion_model, "speed_percent", 100.0)) / 100.0)
                except (TypeError, ValueError):
                    pass
            prehold_distance = 125.0 * speed_ratio * 0.10
        up_preheld = False

        while True:
            if stop_event is not None and stop_event.is_set():
                self.stop()
                return False

            now = time.perf_counter()
            if (now - start_time) > eff_timeout:
                self.stop()
                return False

            pos = get_player_pos()
            if pos is None:
                time.sleep(0.02)
                continue

            cur_x, cur_y = pos
            dx = eff_target_x - cur_x

            # 保持原本的横向行走键，提前 0.1 秒压下 UP；到达容差后才
            # 进入吸附确认循环。下爬不采用该时序。
            if (
                climb_direction == "up"
                and not up_preheld
                and abs(dx) <= approach_tolerance + prehold_distance
            ):
                self.driver.key_down(climb_key)
                up_preheld = True
                self.log(
                    f"🧗 [直爬预按] 距绳梯 {abs(dx):.1f}px，提前0.10s按住{climb_key.upper()}"
                )

            # 对齐梯绳
            if abs(dx) <= approach_tolerance:
                # 保持刚才靠近绳梯的横向方向，同时按 UP。若先松方向，
                # 角色可能在吸附判定前失去横向惯性而错过绳梯。
                approach_key = self.current_held_key
                if approach_key not in ("left", "right"):
                    approach_key = "right" if dx >= 0 else "left"
                    self.driver.key_down(approach_key)
                    self.current_held_key = approach_key
                if not up_preheld:
                    self.driver.key_down(climb_key)
                time.sleep(0.02)
                self.log(
                    f"🧗 [攀爬触发] X误差={dx:+.1f}px，保持{approach_key}并按{climb_key.upper()}，"
                    f"等待绳梯吸附"
                )

                # 连续观察是否已进入梯绳状态。若仍未吸附且沿原方向
                # 越过绳梯两格小地图像素，立即交还上层改走跳抓。
                overshoot_px = 2.0 * (minimap_step if minimap_step is not None else 18.0)
                attach_deadline = time.perf_counter() + 1.2
                attached = False
                origin_pos = (
                    self.raw_position_getter()
                    if self.raw_position_getter else None
                ) or (cur_x, cur_y)
                # 起点与后续进度必须使用同一坐标源；不能用 snapped Y
                # 对比 raw Y，否则两者的固定偏移也会被误算成攀爬位移。
                attach_origin_y = float(origin_pos[1])
                attach_progress_hits = 0
                geometric_candidate_logged = False
                # 仅凭 X/Y 落入绳梯几何盒会把贴近绳脚的平台站立误认成
                # 已吸附。至少看到约 1.25 个小地图采样格的定向 Y 位移，
                # 并连续确认两次，才承认游戏角色真正开始攀爬。
                required_climb_progress = max(
                    10.0,
                    1.25 * (minimap_step if minimap_step is not None else 8.0),
                )
                while time.perf_counter() < attach_deadline:
                    if stop_event is not None and stop_event.is_set():
                        break
                    try:
                        geometric_attached = bool(is_climbing_fn and is_climbing_fn())
                    except Exception:
                        geometric_attached = False
                    raw_pos = self.raw_position_getter() if self.raw_position_getter else get_player_pos()
                    climb_progress = 0.0
                    if raw_pos is not None:
                        observed_y = float(raw_pos[1])
                        climb_progress = (
                            attach_origin_y - observed_y
                            if climb_direction == "up"
                            else observed_y - attach_origin_y
                        )
                    if geometric_attached and climb_progress >= required_climb_progress:
                        attach_progress_hits += 1
                    else:
                        attach_progress_hits = 0
                    attached = attach_progress_hits >= 2
                    if geometric_attached and not attached and not geometric_candidate_logged:
                        geometric_candidate_logged = True
                        self.log(
                            "🟡 [吸附候选待确认] 几何状态已命中，但尚未检测到真实攀爬位移；"
                            f"需要{required_climb_progress:.1f}px，当前{climb_progress:.1f}px"
                        )
                    if attached:
                        self.driver.key_up(approach_key)
                        self.current_held_key = None
                        self.log(
                            "✅ [直爬吸附确认] 几何状态与定向Y位移连续确认，"
                            f"已移动{climb_progress:.1f}px；松开横向键并持续攀爬"
                        )
                        break
                    if raw_pos is not None:
                        raw_x = float(raw_pos[0])
                        crossed = (
                            (approach_key == "right" and raw_x >= eff_target_x + overshoot_px)
                            or (approach_key == "left" and raw_x <= eff_target_x - overshoot_px)
                        )
                        if crossed:
                            self.driver.key_up(approach_key)
                            self.current_held_key = None
                            self.driver.key_up(climb_key)
                            self.log(
                                f"⚠️ [直爬越位未吸附] 已越过绳梯 {overshoot_px:.1f}px 仍未上绳，"
                                f"交由跳抓重试"
                            )
                            return False
                    time.sleep(0.03)

                if not attached:
                    self.driver.key_up(approach_key)
                    self.current_held_key = None
                    self.driver.key_up(climb_key)
                    self.log(
                        "⚠️ [直爬未吸附] 等待1.2s仍未检测到真实定向Y位移，"
                        "交由跳抓重试"
                    )
                    return False

                # 持续攀爬并检测目标高度。下爬的目标是平台面上方约45px的
                # 人物中心，不是平台面本身；固定4秒会在长梯中途提前松DOWN。
                climb_start = time.perf_counter()
                reached_target_height = False
                down_center_y = float(target_y) - 45.0 if target_y is not None else None
                down_tolerance = max(6.0, min(18.0, (minimap_step or 12.0) * 0.5))
                climb_budget = 4.0
                if climb_direction == "down" and down_center_y is not None:
                    remaining_y = max(0.0, down_center_y - float(cur_y))
                    climb_budget = min(16.0, max(5.0, remaining_y / 70.0 + 2.0))
                    self.log(
                        f"⬇️ [下爬目标闭环] 当前Y={float(cur_y):.1f}，"
                        f"平台面Y={float(target_y):.1f}，人物目标Y≈{down_center_y:.1f}，"
                        f"门槛Y={down_center_y - down_tolerance:.1f}，"
                        f"动态上限={climb_budget:.1f}s"
                    )
                while (time.perf_counter() - climb_start) < climb_budget:
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(0.08)
                    cp = get_player_pos()
                    if cp and target_y is not None:
                        physical_cp = (
                            get_raw_player_pos()
                            if get_raw_player_pos is not None else cp
                        )
                        physical_y = float(physical_cp[1]) if physical_cp else float(cp[1])
                        top_gate_y, top_gate_tolerance = self._top_exit_entry_gate(target_y)
                        if climb_direction == "up" and physical_y <= top_gate_y:
                            reached_target_height = True
                            self.log(
                                "📍 [绳顶入口确认] "
                                f"rawY={physical_y:.1f} 已到平台站立高度，"
                                f"平台面Y={float(target_y):.1f}，"
                                f"门槛Y={top_gate_y:.1f}(容差{top_gate_tolerance:.1f}px)"
                            )
                            break
                        elif (climb_direction == "down" and down_center_y is not None
                              and float(cp[1]) >= down_center_y - down_tolerance):
                            reached_target_height = True
                            self.log(
                                f"📍 [下爬目标高度确认] 观测Y={float(cp[1]):.1f}，"
                                f"目标人物Y≈{down_center_y:.1f}"
                            )
                            break

                if climb_direction == "up" and reached_target_height:
                    return self._adaptive_top_exit(
                        target_ladder_x=target_ladder_x,
                        get_raw_player_pos=(get_raw_player_pos or get_player_pos),
                        landing_confirm_fn=landing_confirm_fn,
                        step_off_direction=step_off_direction,
                        top_hold_sec=top_hold_sec,
                        stop_event=stop_event,
                        is_landed_fn=(
                            (lambda: not bool(is_climbing_fn()))
                            if is_climbing_fn is not None else None
                        ),
                    )
                if climb_direction == "down" and not reached_target_height:
                    self.driver.key_up(climb_key)
                    self.log(
                        f"⚠️ [下爬未达目标] {climb_budget:.1f}s内未到"
                        f"目标人物Y≈{down_center_y}；禁止中途侧移和报告成功"
                    )
                    return False
                if stop_event is not None and stop_event.is_set():
                    self.driver.key_up(climb_key)
                    return False
                if climb_direction == "down" and landing_confirm_fn is None:
                    self.driver.key_up(climb_key)
                    self.log("⚠️ [下爬缺少落台核验] 不允许仅凭目标高度报告成功")
                    return False
                if climb_direction != "down":
                    self.driver.key_up(climb_key)
                # 登顶脱离梯绳踏上平台
                if step_off_direction in ("left", "right"):
                    time.sleep(0.05)
                    self.driver.press_key(step_off_direction, duration_ms=180)
                    time.sleep(0.15)
                if climb_direction == "down":
                    confirm_deadline = time.perf_counter() + 1.2
                    while time.perf_counter() < confirm_deadline:
                        if stop_event is not None and stop_event.is_set():
                            self.driver.key_up(climb_key)
                            return False
                        if landing_confirm_fn():
                            self.driver.key_up(climb_key)
                            self.last_landing_confirmed = True
                            self.log("✅ [下爬落台确认] 目标平台raw脚点连续命中，允许结束本边")
                            return True
                        time.sleep(0.06)
                    self.driver.key_up(climb_key)
                    self.log("⚠️ [下爬落台未确认] 已到目标高度但未稳定离绳落台，不报告成功")
                    return False
                return True

            # 保持向目标方向平走
            req_key = "right" if dx > 0 else "left"
            if self.current_held_key != req_key:
                if self.current_held_key is not None:
                    self.driver.key_up(self.current_held_key)
                self.driver.key_down(req_key)
                self.current_held_key = req_key

            time.sleep(0.02)

    def align_to_x_pid(
        self,
        target_x: float,
        get_player_pos: Callable[[], Optional[Tuple[float, float]]],
        tolerance: float = 4.0,
        timeout_sec: float = 3.5,
        platform_bounds: Optional[Tuple[float, float]] = None,
        stop_event: Optional[threading.Event] = None,
        slow_zone: float = 36.0,
        brake_ms: int = 14,
        pulse_scale: float = 1.5,
        pulse_min_ms: Optional[int] = None,
        pulse_max_ms: Optional[int] = None,
        predictive_pulse_max_ms: Optional[int] = None,
    ) -> bool:
        """
        闭环比例微脉冲 (Proportional PID-like) 走位对齐控制器:
        - 彻底解决固定时间/开环移动导致的超调 (Overshoot)、滑落平台或失准问题；
        - 距离 > 36px: 高速连跑并在提前 36px 处减速；
        - 距离 <= 36px: 使用短微脉冲 (可配置，默认 12ms ~ 35ms) 并持续复核，减少停键惯性超调。
        """
        eff_pulse_min = int(float(self.config.get("align_pulse_min_ms", 12.0))) if pulse_min_ms is None else int(pulse_min_ms)
        eff_pulse_max = int(float(self.config.get("align_pulse_max_ms", 35.0))) if pulse_max_ms is None else int(pulse_max_ms)
        pulse_min_ms = eff_pulse_min
        pulse_max_ms = eff_pulse_max
        eff_target_x = target_x
        if platform_bounds:
            p_min, p_max = platform_bounds
            eff_target_x = max(p_min + 18, min(p_max - 18, target_x))

        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < timeout_sec:
            if stop_event is not None and stop_event.is_set():
                self.stop()
                return False

            pos = get_player_pos()
            if not pos:
                time.sleep(0.02)
                continue

            cur_x, _ = pos
            dx = eff_target_x - cur_x
            if abs(dx) <= tolerance:
                self.stop()
                return True

            direction = "right" if dx > 0 else "left"
            if abs(dx) > slow_zone:
                # 粗调阶段：提前进入减速区，避免角色带惯性冲过梯绳。
                self.driver.key_down(direction)
                t_run = time.perf_counter()
                while (time.perf_counter() - t_run) < 1.8:
                    if stop_event is not None and stop_event.is_set():
                        break
                    p2 = get_player_pos()
                    if p2 and abs(eff_target_x - p2[0]) <= slow_zone:
                        break
                    time.sleep(0.02)
                self.stop()
                # 极短反向刹车只用于消除横向惯性，不改变目标方向。
                if brake_ms > 0:
                    self.driver.press_key("left" if direction == "right" else "right", duration_ms=brake_ms)
                self.stop()
                time.sleep(0.08)
            else:
                # 精调阶段：更短的闭环微脉冲，避免最后一次脉冲越过目标。
                pulse_ms = int(max(pulse_min_ms, min(pulse_max_ms, abs(dx) * pulse_scale)))
                # 有连续运动模型时，按“按键加速 + 松键阻力”反推
                # 一次性按键时长，使角色在目标点附近自然刹停，
                # 不再把 raw 黄点的离散跳变当成可到达位置。
                model = self.motion_model
                if model is not None:
                    try:
                        sign = 1.0 if dx > 0 else -1.0
                        v = float(getattr(model, "vx", 0.0)) * sign
                        accel = max(1.0, float(getattr(model, "push_accel", 1500.0)))
                        drag = max(1.0, float(getattr(model, "drag_accel", 900.0)))
                        distance = abs(float(dx))
                        # 若当前惯性已朝向目标，先扣除松键刹停距离。
                        stop_dist = (max(0.0, v) ** 2) / (2.0 * drag)
                        remaining = max(0.0, distance - stop_dist)
                        if v >= 0.0:
                            # 二分求解：按键 t 秒的加速位移 + 松键刹停位移。
                            lo, hi = 0.0, 0.35
                            for _ in range(24):
                                mid = (lo + hi) * 0.5
                                vm = min(float(getattr(model, "v_max", 130.0)), v + accel * mid)
                                travel = v * mid + 0.5 * accel * mid * mid + (vm * vm) / (2.0 * drag)
                                if travel < distance:
                                    lo = mid
                                else:
                                    hi = mid
                            # 预测解也必须遵守动作类型的单次脉冲上限。
                            # 绳索原地抓取需要细小行程；若绕过该限制会
                            # 出现 80~160ms 一脚走过头的情况。
                            predictive_cap = (
                                float(predictive_pulse_max_ms)
                                if predictive_pulse_max_ms is not None
                                else 180.0
                            )
                            pulse_ms = int(max(pulse_min_ms, min(predictive_cap, hi * 1000.0)))
                            self.log(
                                f"🎯 [预测刹车] dx={dx:+.1f}px vx={v*sign:+.1f} "
                                f"停距={stop_dist:.1f}px 按键={pulse_ms}ms"
                            )
                    except Exception:
                        pass
                self.driver.press_key(direction, duration_ms=pulse_ms)
                self.stop()
                # 原地跳抓绳梯的微脉冲后留出足够的物理刹停与小地图
                # 采样时间，避免下一次脉冲叠加造成横向过冲。
                time.sleep(0.20)

        self.stop()
        return False

    def move_to_x_by_motion_model(
        self,
        target_x: float,
        get_player_pos: Callable[[], Optional[Tuple[float, float]]],
        stop_event: Optional[threading.Event] = None,
        platform_bounds: Optional[Tuple[float, float]] = None,
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """一次按键、一次自然刹停的模型走位；不做 PID 或补按。"""
        before = get_player_pos()
        if before is None:
            return False, 0.0, None
        target = float(target_x)
        if platform_bounds is not None:
            low, high = map(float, platform_bounds)
            target = max(low + 6.0, min(high - 6.0, target))
        try:
            if self.motion_model is not None:
                reset_velocity = getattr(self.motion_model, "reset_velocity", None)
                if callable(reset_velocity):
                    reset_velocity()
                else:
                    self.motion_model.explicit_reanchor(float(before[0]), reset_velocity=True)
        except Exception:
            pass
        dx = target - float(before[0])
        if abs(dx) <= 6.0:
            return True, 0.0, before

        model = self.motion_model
        accel = max(1.0, float(getattr(model, "push_accel", 1500.0)))
        drag = max(1.0, float(getattr(model, "drag_accel", 900.0)))
        vmax = max(1.0, float(getattr(model, "v_max", 130.0)))
        distance = abs(dx)
        direction = "right" if dx > 0 else "left"

        def travel_distance(held_sec: float) -> float:
            cap = vmax / accel
            if held_sec <= cap:
                release_v = accel * held_sec
                powered = 0.5 * accel * held_sec * held_sec
            else:
                release_v = vmax
                powered = 0.5 * accel * cap * cap + vmax * (held_sec - cap)
            return powered + release_v * release_v / (2.0 * drag)

        lo, hi = 0.0, max(0.10, distance / vmax + 1.0)
        for _ in range(32):
            mid = (lo + hi) * 0.5
            if travel_distance(mid) < distance:
                lo = mid
            else:
                hi = mid
        self.log(
            f"🎯 [原地抓取模型走位] rawX={before[0]:.1f} -> 起跳X={target:.1f}，"
            f"方向={direction}，按住={hi:.3f}s（无微调）"
        )
        self.driver.key_down(direction)
        try:
            deadline = time.perf_counter() + hi
            while time.perf_counter() < deadline:
                if stop_event is not None and stop_event.is_set():
                    return False, hi, get_player_pos()
                time.sleep(0.005)
        finally:
            self.driver.key_up(direction)
        # 上述解包括松键后的摩擦滑行；等待其收束，才触发原地 UP+Alt。
        settle_deadline = time.perf_counter() + min(0.50, vmax / drag + 0.05)
        while time.perf_counter() < settle_deadline:
            if stop_event is not None and stop_event.is_set():
                return False, hi, get_player_pos()
            time.sleep(0.005)
        return True, hi, get_player_pos()

    @staticmethod
    def _integrate_powered_horizontal(
        velocity_x: float,
        direction_sign: float,
        duration_sec: float,
        push_accel: float,
        max_speed: float,
    ) -> float:
        """预测持续按住一个方向后的水平位移。

        这里积分的是当前实时速度，而不是用 ``距离 / 固定跑速`` 反推一条
        固定起跳线。这样刚掉头、受击后重新加速、不同移动速度和左右方向
        都会得到不同的起跳位置。
        """
        remaining = max(0.0, float(duration_sec))
        sign = 1.0 if direction_sign >= 0.0 else -1.0
        accel = max(1.0, float(push_accel))
        vmax = max(1.0, float(max_speed))
        velocity = max(-vmax, min(vmax, float(velocity_x)))
        displacement = 0.0

        # 方向键施加恒定推力，达到同向速度上限后匀速。若当前速度与输入
        # 相反，这一段积分自然包含先制动、再反向加速的过程。
        signed_velocity = velocity * sign
        time_to_cap = max(0.0, (vmax - signed_velocity) / accel)
        accelerated = min(remaining, time_to_cap)
        displacement += sign * (
            signed_velocity * accelerated + 0.5 * accel * accelerated * accelerated
        )
        remaining -= accelerated
        if remaining > 0.0:
            displacement += sign * vmax * remaining
        return displacement

    def _run_jump_capture_window(self) -> float:
        """按当前小地图量化精度估计梯绳的可吸附水平半窗。"""
        try:
            measurement_step = float(
                getattr(self.motion_model, "measurement_step_px", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            measurement_step = 0.0
        # 不是起跳距离：这是实时预测落点允许进入的吸附窗。量化越粗，
        # 必须给连续世界坐标留下更大的观测不确定度，但仍限制在角色身体宽度内。
        return max(12.0, min(20.0, measurement_step * 0.75 + 6.0))

    def _run_jump_gate_window(self, capture_window: float) -> float:
        """返回真正允许起跳的核心窗，必须小于完整物理吸附窗。

        ``capture_window`` 描述角色仍有机会吸附的外边界，不能直接当作
        起跳目标。否则预测点刚擦到 ±18px 外缘就会放行，任何一帧量化
        或调度误差都会越过绳轴。核心窗仍随当前黄点量化步长变化，但
        限制在 8–12px，目标始终靠近吸附区中央。
        """
        try:
            measurement_step = float(
                getattr(self.motion_model, "measurement_step_px", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            measurement_step = 0.0
        candidates = [12.0, max(1.0, float(capture_window)) * 0.67]
        if measurement_step > 0.0:
            candidates.append(measurement_step * 0.75)
        return max(8.0, min(candidates))

    def _run_jump_capture_time(
        self,
        source_y: Optional[float],
        ladder_bottom_y: Optional[float],
        fallback_sec: float,
    ) -> float:
        """由脚下高度与梯绳底端反推跳跃上升段的相交时间。"""
        if source_y is None or ladder_bottom_y is None:
            return max(0.08, min(0.26, float(fallback_sec)))

        source = float(source_y)
        capture_y = float(ladder_bottom_y)
        vertical_gap = max(0.0, source - capture_y)
        # 与上面的低净空诊断统一处理近绳区。旧模型在 gap=10px 两侧
        # 会从 0.150s 突然跌到 0.080s，P74 的32号梯、P78 的35号绳
        # 因此经常“看似已经跳到绳轴却没抓住”。低净空时真正要相交的
        # 是绳梯中段而非埋在地面的底端。实机中35号绳在约20~30px
        # 间距下 0.134s 失败、0.150s 成功，因此近绳区保持 0.150s；
        # 32~48px 再连续过渡回普通物理相交时间，避免另一条硬断层。
        if vertical_gap <= 32.0:
            rise_needed = 42.0
            lower, upper = 0.15, 0.22
        else:
            rise_needed = vertical_gap
            blend = min(1.0, (vertical_gap - 32.0) / (48.0 - 32.0))
            lower = 0.15 + (0.08 - 0.15) * blend
            upper = 0.22 + (0.26 - 0.22) * blend
        discriminant = max(0.0, 555.0 ** 2 - 4000.0 * rise_needed)
        crossing = (
            (555.0 - math.sqrt(discriminant)) / 2000.0
            if rise_needed > 0.0
            else float(fallback_sec)
        )
        return max(lower, min(upper, crossing))

    def _predict_run_jump_capture_x(
        self,
        current_x: float,
        direction: str,
        capture_sec: float,
    ) -> Tuple[float, float, float]:
        """返回 ``(相交时X, 当前Vx, 从当前到相交的水平位移)``。"""
        sign = 1.0 if direction == "right" else -1.0
        model = self.motion_model
        velocity_x = sign * float(getattr(model, "v_max", 125.0))
        if model is not None:
            snapshot = getattr(model, "snapshot", None)
            try:
                if callable(snapshot):
                    velocity_x = float(snapshot().get("vx", velocity_x))
                else:
                    velocity_x = float(getattr(model, "vx", velocity_x))
            except (TypeError, ValueError, AttributeError):
                pass
        accel = float(getattr(model, "push_accel", 1500.0)) if model is not None else 1500.0
        vmax = float(getattr(model, "v_max", 125.0)) if model is not None else 125.0
        # UP 预压 20ms 后才按下跳键；该段角色仍持续水平运动。
        horizon = 0.02 + max(0.0, float(capture_sec))
        displacement = self._integrate_powered_horizontal(
            velocity_x, sign, horizon, accel, vmax
        )
        return float(current_x) + displacement, velocity_x, displacement

    def jump_and_grab_ladder(
        self,
        target_ladder_x: float,
        get_player_pos: Callable[[], Optional[Tuple[float, float]]],
        align_player_pos: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
        jump_key: Optional[str] = None,
        target_y: Optional[float] = None,
        step_off_direction: Optional[str] = None,
        platform_bounds: Optional[Tuple[float, float]] = None,
        approach_tolerance: float = 4.0,
        timeout_sec: float = 4.0,
        stop_event: Optional[threading.Event] = None,
        get_raw_player_pos: Optional[Callable[[], Optional[Tuple[float, float]]]] = None,
        target_type: Optional[str] = None,
        enable_jump_steering: bool = False,
        is_landed_fn: Optional[Callable[[], bool]] = None,
        landing_confirm_fn: Optional[Callable[[], bool]] = None,
        source_standing_check: Optional[Callable[[], bool]] = None,
        run_jump_direction: Optional[str] = None,
        run_jump_sec: float = 0.15,
        ladder_bottom_y: Optional[float] = None,
        run_jump_profile: Optional[Tuple[float, float]] = None,
        use_model_static_approach: bool = False,
        top_hold_sec: float = 1.0,
    ) -> bool:
        """
        起跳抓梯/抓绳控制器 (带闭环PID对齐、防滑落保护、跳出登顶与自愈检测):
        1. 采用闭环比例微脉冲对齐梯绳 X 轴 (小地图宏观对齐)；
        2. 原地起跳并在空中上升顶点吸附梯绳 (80ms 跳跃 + 40ms 空中延时 + 按住 UP)；
        3. 持续攀爬至目标层高度 target_y 或到达物理顶端；
        4. 登顶轻点横移踏入平台表面。
        """
        jk = jump_key or self.jump_key
        self.last_landing_confirmed = False
        self.last_top_exit_failed = False
        self.last_jump_gate_aborted = False

        # 1. 严格计算安全停靠点 (绝不超出平台边缘)
        eff_target_x = target_ladder_x
        # 直接跨过中间平台的跑跳抓绳，目标绳子可能位于当前平台边界之外；
        # 此时不能把绳坐标夹回平台内部，否则永远对不准真正的抓取点。
        if platform_bounds and run_jump_direction not in ("left", "right"):
            p_min, p_max = platform_bounds
            eff_target_x = max(p_min + 6, min(p_max - 6, target_ladder_x))

        init_p = get_player_pos()
        eff_timeout = max(timeout_sec, (abs(eff_target_x - init_p[0]) / 90.0 + 4.0) if init_p else timeout_sec)

        approach_target_x = eff_target_x
        run_capture_sec = float(run_jump_sec)
        empirical_run_profile = run_jump_profile if run_jump_profile is not None else None
        # 连续 Kalman 世界坐标负责准备点、回撤和起跳线；raw 回调仅负责
        # 检测不可能由正常跑速产生的突跳/击退。
        run_position_getter = get_player_pos
        run_raw_getter = get_raw_player_pos
        run_initial_pos = run_position_getter() if run_jump_direction in ("left", "right") else init_p
        if run_jump_direction in ("left", "right"):
            if empirical_run_profile is not None:
                run_capture_sec = float(empirical_run_profile[1])
                self.log(
                    f"ℹ️ [旧跑跳档案兼容] 忽略固定准备距离={float(empirical_run_profile[0]):.1f}px；"
                    f"只保留相交时间={run_capture_sec:.3f}s，起跳位置改由实时速度预测"
                )
            # 是否跑跳由 UI 的“跑跳抓绳”开关决定。低净空只记录诊断，
            # 绝不能在控制器内部悄悄改成原地抓取；原地模式由调用方
            # 显式传入 run_jump_direction=None，或在两次失败后降级。
            if ladder_bottom_y is not None and run_initial_pos is not None:
                vertical_gap = float(run_initial_pos[1]) - float(ladder_bottom_y)
                if 0.0 <= vertical_gap <= 16.5:
                    self.log(
                        f"📏 [低净空跑跳诊断] 脚下Y={run_initial_pos[1]:.0f}，底端Y={ladder_bottom_y:.0f}，"
                        f"差={vertical_gap:.1f}px；保持 UI 选择的跑跳模式"
                    )
            # 先由绳梯底端高度反推上升相交时刻。水平起跳位置不在这里
            # 固定下来，而会在助跑循环中按实时 Vx 逐帧预测。
            source_y = float(run_initial_pos[1]) if run_initial_pos is not None else None
            if empirical_run_profile is None:
                run_capture_sec = self._run_jump_capture_time(
                    source_y, ladder_bottom_y, run_jump_sec
                )
            capture_window = self._run_jump_capture_window()
            gate_window = self._run_jump_gate_window(capture_window)
            sign = 1.0 if run_jump_direction == "right" else -1.0
            preview_x, preview_vx, preview_displacement = self._predict_run_jump_capture_x(
                float(run_initial_pos[0]) if run_initial_pos is not None else eff_target_x,
                run_jump_direction,
                run_capture_sec,
            )
            # 仅用于日志、回撤和超时预算的预览线；真正放行条件在循环中
            # 依据实时 Vx 重算，受击、掉头和移动速度变化不会沿用旧值。
            preview_offset = abs(preview_displacement) + gate_window
            approach_target_x = (
                eff_target_x - preview_offset
                if run_jump_direction == "right"
                else eff_target_x + preview_offset
            )
            if platform_bounds:
                p_min, p_max = platform_bounds
                approach_target_x = max(p_min + 6, min(p_max - 6, approach_target_x))
            self.log(
                f"🏃 [自适应跑跳计划] 绳梯X={eff_target_x:.1f}，上升相交={run_capture_sec:.3f}s，"
                f"当前Vx={preview_vx:+.1f}px/s，预览起跳X={approach_target_x:.1f}，"
                f"核心放行窗=±{gate_window:.1f}px/物理吸附窗=±{capture_window:.1f}px "
                f"(方向={run_jump_direction})"
            )
            # 上方的跑跳计划在低净空判定后仅用于记录；实际控制点必须
            # 回到梯绳本体，不能保留计算出的准备线。
            if run_jump_direction not in ("left", "right"):
                approach_target_x = eff_target_x
                self.log(f"🚶 [宏观对齐] 低净空精确目标X={eff_target_x}")
        else:
            self.log(f"🚶 [宏观对齐] 目标梯绳世界X={eff_target_x} (初始世界坐标={init_p})")

        # Phase 1a: 闭环比例微脉冲 (P-Control) 小地图宏观对齐
        # Rope and ladder have different physical capture windows. Ropes need
        # earlier, gentler braking; ladders tolerate a stronger final pulse.
        cfg_pulse_min = int(float(self.config.get("align_pulse_min_ms", 12.0)))
        cfg_pulse_max = int(float(self.config.get("align_pulse_max_ms", 35.0)))
        if target_type == "rope":
            align_profile = dict(
                slow_zone=50.0, brake_ms=10, pulse_scale=1.15,
                pulse_min_ms=cfg_pulse_min, pulse_max_ms=cfg_pulse_max,
                predictive_pulse_max_ms=cfg_pulse_max,
            )
        else:
            align_profile = dict(
                slow_zone=18.0, brake_ms=0, pulse_scale=1.2,
                pulse_min_ms=cfg_pulse_min, pulse_max_ms=cfg_pulse_max,
            )

        if run_jump_direction in ("left", "right") and run_initial_pos is not None:
            # 真正的跑跳：持续按方向移动，到准备线立即衔接跳跃；
            # 读取实时坐标触发而非固定 sleep，击退或卷轴更新时会随
            # 新位置自动延后/提前触发，期间绝不在准备线停车。
            travel_sign = 1.0 if run_jump_direction == "right" else -1.0
            passed_rope_by = (
                float(run_initial_pos[0]) - float(eff_target_x)
            ) * travel_sign
            if passed_rope_by > capture_window:
                # 只有确实越过绳轴及其吸附窗才回撤。人在动态预览线和绳子
                # 之间时仍可按当前速度直接起跳，无需退回某个固定准备点。
                retreat_direction = "left" if run_jump_direction == "right" else "right"
                retreat_overshoot = 40.0
                retreat_target_x = float(approach_target_x) - travel_sign * retreat_overshoot
                if platform_bounds:
                    p_min, p_max = map(float, platform_bounds)
                    safe_edge = max(10.0, min(25.0, (p_max - p_min) * 0.08))
                    retreat_target_x = max(p_min + safe_edge, min(p_max - safe_edge, retreat_target_x))

                retreat_dist = abs(float(run_initial_pos[0]) - retreat_target_x)
                self.log(
                    f"↩️ [跑跳回撤规划] 当前X={float(run_initial_pos[0]):.1f} "
                    f"已越过绳轴吸附窗{passed_rope_by:.1f}px；"
                    f"直接平走回撤至外侧X={retreat_target_x:.1f} (重新建立实时速度轨迹)"
                )
                retreat_timeout = max(1.5, min(4.0, retreat_dist / 85.0 + 1.2))
                self.walk_to_x(
                    target_x=retreat_target_x,
                    get_player_pos=run_position_getter,
                    tolerance=15,
                    timeout_sec=retreat_timeout,
                    stop_event=stop_event,
                    platform_bounds=platform_bounds,
                    safe_margin=10,
                    speed_scale=1.0,
                )
                self.stop()
                time.sleep(0.08)
                run_initial_pos = run_position_getter()
                if run_initial_pos is None:
                    self.log("⚠️ [跑跳回撤失败] 停稳后坐标不可用，本次不触发UP+Alt")
                    return False
                remaining = (
                    float(approach_target_x) - float(run_initial_pos[0])
                ) * travel_sign
                if remaining < 6.0:
                    self.log(
                        f"⚠️ [跑跳回撤未到位] 当前X={float(run_initial_pos[0]):.1f}，"
                        f"起跳X={float(approach_target_x):.1f}，助跑距离仅{remaining:.1f}px不足，本次放弃起跳"
                    )
                    return False
                self.log(
                    f"✅ [跑跳回撤完成] 当前X={float(run_initial_pos[0]):.1f}，"
                    f"预览门X={float(approach_target_x):.1f}，"
                    f"助跑加速距离={remaining:.1f}px；立即掉头并由实时相交门决定起跳！"
                )
            self.driver.ensure_focus()
            self.driver.key_down(run_jump_direction)
            distance_to_ref = max(0.0, abs(float(approach_target_x) - float(run_initial_pos[0])))
            # 长平台可能离绳近千像素；旧版把预算强行封顶 8 秒，受击稍微
            # 反向位移后尚未到起跳线便超时，并错误穿透到后面的 UP+Alt。
            # 现在按保守速度计算完整路程，仍以 30 秒作为卡死上限。
            run_deadline = time.perf_counter() + max(
                2.0, min(30.0, distance_to_ref / 70.0 + 3.0)
            )
            run_gate_reached = False
            run_gate_prediction = None
            last_run_raw_x = None
            if run_raw_getter is not None:
                initial_raw = run_raw_getter()
                if initial_raw is not None:
                    last_run_raw_x = float(initial_raw[0])
            recovering_from_knockback = False
            try:
                measurement_step = float(
                    getattr(self.motion_model, "measurement_step_px", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                measurement_step = 0.0
            # raw 黄点通常每格跳约 20–30 世界像素；超过两格的单帧位移
            # 不可能来自正常跑速。无论方向是否与助跑一致，都按受击/扰动
            # 处理，绝不能让一次同向突跳直接越过准备线并触发 Alt。
            external_jump_threshold = max(60.0, measurement_step * 2.25)
            while time.perf_counter() < run_deadline:
                if stop_event is not None and stop_event.is_set():
                    self.stop()
                    return False
                run_pos = run_position_getter()
                if run_pos is not None:
                    now_run = time.perf_counter()
                    if run_raw_getter is not None:
                        raw_run_pos = run_raw_getter()
                        if raw_run_pos is not None:
                            current_raw_x = float(raw_run_pos[0])
                            if last_run_raw_x is not None:
                                raw_delta = current_raw_x - last_run_raw_x
                                signed_progress = raw_delta * travel_sign
                                if abs(raw_delta) >= external_jump_threshold:
                                    self._report_external_force(
                                        f"跑跳助跑单帧突跳{raw_delta:+.1f}px"
                                    )
                                    if self.motion_model is not None:
                                        try:
                                            self.motion_model.explicit_reanchor(
                                                current_raw_x, reset_velocity=True
                                            )
                                        except Exception:
                                            pass
                                    self.last_jump_gate_aborted = True
                                    self.stop()
                                    jump_direction = "同向" if signed_progress > 0.0 else "反向"
                                    self.log(
                                        f"⚠️ [跑跳外力突跳] rawX {last_run_raw_x:.1f} -> "
                                        f"{current_raw_x:.1f}，单帧Δ={raw_delta:+.1f}px "
                                        f"({jump_direction}，阈值={external_jump_threshold:.1f}px)；"
                                        "本轮禁止起跳，交还原边按新坐标重建助跑"
                                    )
                                    return False
                                if signed_progress < -4.0:
                                    self._report_external_force(
                                        f"跑跳按{run_jump_direction}时反向位移{raw_delta:+.1f}px"
                                    )
                                    recovering_from_knockback = True
                                    if self.motion_model is not None:
                                        try:
                                            self.motion_model.explicit_reanchor(
                                                current_raw_x, reset_velocity=True
                                            )
                                        except Exception:
                                            pass
                                    remaining = abs(
                                        float(approach_target_x) - float(run_pos[0])
                                    )
                                    run_deadline = max(
                                        run_deadline,
                                        now_run + max(2.0, min(10.0, remaining / 70.0 + 2.0)),
                                    )
                                    self.log(
                                        f"⚠️ [跑跳受击重规划] rawX {last_run_raw_x:.1f} -> "
                                        f"{current_raw_x:.1f} 与{run_jump_direction}助跑相反；"
                                        f"重置速度并按Kalman剩余{remaining:.1f}px重算起跳预算"
                                    )
                                elif signed_progress > 4.0 and recovering_from_knockback:
                                    recovering_from_knockback = False
                                    self.log(
                                        f"✅ [跑跳受击恢复] 已重新朝{run_jump_direction}移动，"
                                        "继续按Kalman坐标等待起跳线"
                                    )
                            last_run_raw_x = current_raw_x
                    live_capture_sec = run_capture_sec
                    if empirical_run_profile is None and ladder_bottom_y is not None:
                        live_capture_sec = self._run_jump_capture_time(
                            float(run_pos[1]), ladder_bottom_y, run_jump_sec
                        )
                    predicted_capture_x, live_vx, predicted_displacement = (
                        self._predict_run_jump_capture_x(
                            float(run_pos[0]), run_jump_direction, live_capture_sec
                        )
                    )
                    remaining_now = (float(eff_target_x) - float(run_pos[0])) * travel_sign
                    remaining_at_capture = (
                        float(eff_target_x) - predicted_capture_x
                    ) * travel_sign
                    if remaining_now < -capture_window:
                        self.last_jump_gate_aborted = True
                        self.stop()
                        self.log(
                            f"⚠️ [自适应起跳门越界] 当前X={float(run_pos[0]):.1f} 已越过"
                            f"绳轴吸附窗，预测相交X={predicted_capture_x:.1f}；"
                            "本轮不盲跳，按当前位置重规划方向"
                        )
                        return False
                    if abs(remaining_at_capture) <= gate_window:
                        # 若这一格恰好是击退造成的反向跨线，不能当作正常
                        # 助跑到位；至少等到一次重新朝绳移动的真实采样。
                        if not recovering_from_knockback:
                            run_gate_reached = True
                            run_gate_prediction = (
                                float(run_pos[0]), live_vx, predicted_capture_x,
                                predicted_displacement, remaining_at_capture,
                                live_capture_sec,
                            )
                            break
                time.sleep(0.015)
            if not run_gate_reached:
                self.last_jump_gate_aborted = True
                latest = run_position_getter()
                self.stop()
                self.log(
                    f"⚠️ [跑跳起跳线超时] 当前={latest}，计划X={approach_target_x:.1f}，"
                    "未到起跳线，禁止发送UP+Alt并交还原边实时重规划"
                )
                return False
            if run_gate_prediction is not None:
                (
                    gate_x, gate_vx, predicted_x, predicted_dx,
                    predicted_error, run_capture_sec,
                ) = run_gate_prediction
                approach_target_x = gate_x
                self.log(
                    f"✅ [自适应起跳门放行] 当前X={gate_x:.1f}，Vx={gate_vx:+.1f}px/s，"
                    f"预测位移={predicted_dx:+.1f}px，预测相交X={predicted_x:.1f}，"
                    f"距绳轴={predicted_error:+.1f}px（核心允许±{gate_window:.1f}px，"
                    f"物理吸附±{capture_window:.1f}px），"
                    f"实时相交={run_capture_sec:.3f}s"
                )
        else:
            if use_model_static_approach:
                self.log(
                    f"🎯 [原地抓取计划] 绳梯X={target_ladder_x:.1f}，"
                    f"可站起跳X={approach_target_x:.1f}，按运动模型反推一次按键时长"
                )
                aligned, _held, _after = self.move_to_x_by_motion_model(
                    approach_target_x,
                    get_player_pos,
                    stop_event=stop_event,
                    platform_bounds=platform_bounds,
                )
            else:
                aligned = self.align_to_x_pid(
                    target_x=approach_target_x,
                    get_player_pos=align_player_pos or get_player_pos,
                    tolerance=approach_tolerance,
                    timeout_sec=eff_timeout,
                    platform_bounds=platform_bounds,
                    stop_event=stop_event,
                    **align_profile,
                )
            if not aligned:
                self.stop()
                if stop_event is None or not stop_event.is_set():
                    self.log(
                        f"⚠️ [跳抓对齐未完成] 目标X={approach_target_x:.1f}，"
                        "本次不触发跳跃，交由下一次闭环重试"
                    )
                return False

            # 普通跳抓需要停稳后再起跳。
            self.stop()
            time.sleep(0.35)

        mid_p = (
            run_position_getter()
            if run_jump_direction in ("left", "right")
            else get_player_pos()
        )
        self.log(
            f"📍 [宏观到位] 当前世界坐标={mid_p} "
            f"(距{('跑跳准备点' if run_jump_direction else '目标X')}误差="
            f"{mid_p[0] - approach_target_x if mid_p else 0:.1f}px)"
        )

        # 梯绳对齐仅使用拓扑与实时世界坐标。旧的主画面 PNG 梯绳
        # 模板视觉伺服已移除，避免无效资源加载和背景误匹配。
        final_tx, final_px, final_dx = None, None, 0

        # 稳定性测试的跑跳抓在跨过准备线后会立刻执行 UP+Alt；若在
        # 此处再等待一帧，角色会继续跑向绳梯，导致“跑到绳子正下方
        # 才跳”。普通原地抓仍保留这段短暂稳定等待。
        if run_jump_direction not in ("left", "right"):
            time.sleep(0.03)

        # 跑跳模式下，此处就是实时相交门放行后的起跳快照。
        pre_jump_p = (
            run_position_getter()
            if run_jump_direction in ("left", "right")
            else get_player_pos()
        )
        servo_summary = f"屏幕X: 角色={final_px}, 梯绳={final_tx}, 残余dx={final_dx:+d}px" if final_tx else "基于宏观对齐"
        pre_jump_label = "自适应起跳门" if run_jump_direction else "起跳前坐标"
        self.log(f"🚀 [起跳抓梯] {pre_jump_label}: 世界坐标={pre_jump_p} | {servo_summary}")

        # Phase 2: v83 requires UP before the jump so the character is already
        # requesting ladder/rope attachment while crossing its lower hit line.
        # 走位、回撤和失败后的坐标收敛都可能令角色离开原平台。最终
        # 发出 UP+Alt 前重新核对承重面，禁止在下层平台继续执行旧绳边。
        if source_standing_check is not None:
            try:
                source_ready = bool(source_standing_check())
            except Exception:
                source_ready = False
            if not source_ready:
                self.last_jump_gate_aborted = True
                self.stop()
                self.log(
                    "⚠️ [抓绳源平台丢失] UP+Alt前已离开原平台，"
                    "取消旧抓绳动作并交还状态机重规划"
                )
                return False

        self.driver.ensure_focus()
        jump_dir = run_jump_direction if run_jump_direction in ("left", "right") else None
        if enable_jump_steering:
            if jump_dir is None:
                jump_dir = "right" if (final_dx and final_dx > 4) else ("left" if (final_dx and final_dx < -4) else None)
        if jump_dir:
            self.driver.key_down(jump_dir)
        actual_launch_p = (
            run_position_getter()
            if run_jump_direction in ("left", "right")
            else get_player_pos()
        )
        self.log(f"🚀 [起跳触发] 世界坐标={actual_launch_p} -> 按 [上+跳] 吸附！")
        self.driver.key_down("up")
        time.sleep(0.02)
        self.driver.key_down(jk)
        time.sleep(0.08)
        self.driver.key_up(jk)
        # run_capture_sec 是“跳键按下 -> 上升相交”的总时间。跳键本身已
        # 按住 80ms，这里只补足剩余时间，不能再把整段时间重复等待一次。
        if run_jump_direction in ("left", "right") and jump_dir:
            hold_after_jump = max(0.0, min(0.42, float(run_capture_sec) - 0.08))
            self.log(
                f"🏃 [跑跳相交时序] UP预压=0.020s，跳键至相交={run_capture_sec:.3f}s，"
                f"跳键松开后继续按{jump_dir} {hold_after_jump:.3f}s"
            )
            if hold_after_jump > 0.0:
                time.sleep(hold_after_jump)
        if jump_dir:
            self.driver.key_up(jump_dir)

        # Phase 3: 持续爬升循环并检测高度到达与顶部停滞。
        # 固定 4.5s 对长绳不成立：101000000 的35号绳从 P77 到 P81
        # 约需爬升689px，旧控制器会在绳中松开 UP，返回 FSM 后才由
        # 外层重新按下，形成肉眼可见的停顿。按剩余高度用保守 70px/s
        # 计算预算，确保同一个控制器连续持有 UP 直到登顶。
        climb_start = time.perf_counter()
        last_y = None
        stuck_count = 0
        stuck_logged = False
        grab_confirmed = False
        grab_lost = False
        last_observed_p = actual_launch_p
        last_observed_raw_p = actual_launch_p
        below_launch_samples = 0
        near_rope_rise_samples = 0
        off_rope_samples = 0
        post_jump_guard_y = None
        grab_confirm_grace_until = 1.0
        grab_confirm_grace_logged = False
        reached_target_top = False
        # raw 小地图坐标通常以 16~20px 为一格。吸附后的角色 X 应贴着
        # 绳轴；留出两格量化余量，但不能接受失败跳跃后越过绳子上百像素。
        rope_x_tolerance = 45.0
        # 抓取确认和攀爬距离必须以真正按下 UP+Alt 时的位置为基准，
        # 不能用走向绳子之前的起始 Y。长斜坡上横向移动会带来很大的
        # 地面高度差：P77 从 X=-618 走到34号绳时约相差125px，旧基准
        # 会把尚未抓绳误判为已经高于目标平台。
        init_py = actual_launch_p[1] if actual_launch_p is not None else (
            init_p[1] if init_p else None
        )
        jump_peak_y = float(init_py) if init_py is not None else None
        if init_py is not None and target_y is not None:
            climb_distance = max(0.0, float(init_py) - float(target_y))
            # 使用保守的60px/s并额外预留4秒给量化停顿和翻台动画；
            # 已确认抓住长绳后不能因过紧预算在绳中松开UP。
            climb_timeout = max(5.0, min(25.0, climb_distance / 60.0 + 4.0))
        else:
            climb_timeout = 6.0
        self.log(
            f"⏱️ [连续攀爬预算] 距离="
            f"{(max(0.0, float(init_py) - float(target_y)) if init_py is not None and target_y is not None else -1):.0f}px，"
            f"最长{climb_timeout:.2f}s，期间不交还FSM"
        )

        while (time.perf_counter() - climb_start) < climb_timeout:
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(0.08)
            cur_p = get_player_pos()
            if cur_p is not None:
                last_observed_p = cur_p
                _, cur_py = cur_p
                raw_p = get_raw_player_pos() if get_raw_player_pos is not None else cur_p
                last_observed_raw_p = raw_p
                raw_x = float(raw_p[0]) if raw_p is not None else float(cur_p[0])
                raw_y = float(raw_p[1]) if raw_p is not None else float(cur_py)
                rope_dx = abs(raw_x - float(target_ladder_x))
                confirm_elapsed = time.perf_counter() - climb_start

                # 起跳后的前0.35秒仍可能只是普通跳跃上升。冻结这段时间
                # 观察到的最高点；真正抓住绳子后必须继续爬过这个自然跳跃
                # 顶点，不能把“贴着绳轴停在跳跃顶点”当成抓取成功。
                if jump_peak_y is not None and confirm_elapsed < 0.35:
                    jump_peak_y = min(jump_peak_y, float(cur_py))

                # 跳跃本身也会让 Y 上升，不能单凭 Y:75->-14 就宣布抓绳。
                # 必须连续两次同时满足“已上升、X贴近绳轴”，排除高速越过
                # 绳子后在空中达到跳跃顶点的假成功。
                if not grab_confirmed and init_py is not None:
                    continued_above_jump_peak = (
                        confirm_elapsed >= 0.35
                        and jump_peak_y is not None
                        and float(cur_py) <= (jump_peak_y - 6.0)
                    )
                    # 另一类可靠证据：自然跳跃在约0.28秒到顶，0.40秒后
                    # 应只会下降。若此后角色仍贴着绳轴，并相对下降阶段的
                    # 最低观测点重新持续上升至少10px，只可能是已经吸附并
                    # 在攀爬。旧逻辑只认“超过最初跳跃顶点”，量化Y更新较
                    # 慢时会在绳中于1秒处松开UP，造成肉眼可见的停顿。
                    continued_post_jump_rise = False
                    if confirm_elapsed >= 0.40 and rope_dx <= rope_x_tolerance:
                        if post_jump_guard_y is None:
                            post_jump_guard_y = float(cur_py)
                        else:
                            post_jump_guard_y = max(post_jump_guard_y, float(cur_py))
                            continued_post_jump_rise = (
                                float(cur_py) <= post_jump_guard_y - 10.0
                            )
                    elif rope_dx > rope_x_tolerance:
                        post_jump_guard_y = None

                    has_climb_evidence = (
                        continued_above_jump_peak or continued_post_jump_rise
                    )
                    if has_climb_evidence and rope_dx <= rope_x_tolerance:
                        near_rope_rise_samples += 1
                        # 已出现一帧攀爬证据时，不要恰好在1.0秒边界松开
                        # UP；仅给第二帧确认预留350ms。完全没有证据的失败
                        # 起跳仍维持原来的1秒快速重试。
                        grab_confirm_grace_until = max(grab_confirm_grace_until, 1.35)
                        if not grab_confirm_grace_logged:
                            grab_confirm_grace_logged = True
                            self.log(
                                "🟡 [抓取确认续持] 已检测到绳轴内持续上升，"
                                "继续保持UP等待第二帧确认"
                            )
                        if near_rope_rise_samples >= 2:
                            grab_confirmed = True
                            evidence_name = (
                                "超过自然跳跃顶点"
                                if continued_above_jump_peak
                                else "自然跳跃后仍持续上升"
                            )
                            self.log(
                                f"🧗 [抓取成功] 已连续贴合绳轴并上升【{target_type}】"
                                f"(Y: {init_py} -> {cur_py}, 跳跃顶点={jump_peak_y:.0f}, "
                                f"|dx|={rope_dx:.1f}px，证据={evidence_name})，关闭靶向检测！"
                            )
                    else:
                        near_rope_rise_samples = 0

                # 抓取确认后仍持续校验绳轴。若角色连续三帧离开绳轴，说明
                # 刚才只是跳跃轨迹短暂穿过吸附区，或已经从绳上滑落；无需
                # 等完整条长绳的攀爬预算。
                if grab_confirmed and (time.perf_counter() - climb_start) >= 0.35:
                    if rope_dx > rope_x_tolerance:
                        off_rope_samples += 1
                        if off_rope_samples >= 3:
                            grab_lost = True
                            self.log(
                                f"⚠️ [绳轴脱离] rawX={raw_x:.0f}，绳轴X={target_ladder_x:.0f}，"
                                f"偏差={rope_dx:.1f}px；提前结束并立即重试"
                            )
                            break
                    else:
                        off_rope_samples = 0

                # 跳跃本身也会造成短暂上升，旧逻辑只凭“上升10px”就把
                # 它当成抓绳成功。一旦随后落回起跳平台，仍会空等完整的
                # 6~8 秒攀爬预算。连续三个采样明显低于起跳脚点，说明已
                # 从绳上掉回地面，可立即结束并重新对齐。
                if grab_confirmed and init_py is not None and cur_py >= (init_py + 8):
                    below_launch_samples += 1
                    if below_launch_samples >= 3:
                        grab_lost = True
                        self.log(
                            f"⚠️ [抓绳后回落] Y={cur_py} 已低于起跳Y={init_py}，"
                            "提前结束本次尝试并立即重试"
                        )
                        break
                else:
                    below_launch_samples = 0

                top_gate_y = None
                top_gate_tolerance = None
                if target_y is not None:
                    top_gate_y, top_gate_tolerance = self._top_exit_entry_gate(target_y)
                if (
                    grab_confirmed
                    and top_gate_y is not None
                    and raw_y <= top_gate_y
                ):
                    # 到达目标高度只表示进入绳顶区域，不能证明已踏上平台。
                    # 保持 UP 状态交给后面的主动横移闭环来取得物理证据。
                    self.log(
                        "📍 [绳顶入口确认] "
                        f"rawY={raw_y:.1f} 已到平台站立高度，"
                        f"平台面Y={float(target_y):.1f}，"
                        f"门槛Y={top_gate_y:.1f}(容差{top_gate_tolerance:.1f}px)"
                    )
                    reached_target_top = True
                    break
                if last_y is not None and abs(cur_py - last_y) < 2:
                    stuck_count += 1
                    # A successful jump-grab can briefly report the same
                    # snapped minimap Y while the character settles onto the
                    # ladder. Do not release UP after only ~0.3 s of quantized
                    # coordinates; allow roughly one second before declaring
                    # the climb stalled.
                    if stuck_count >= 12:
                        if not grab_confirmed:
                            # 尚未确认抓住时，持续静止说明起跳失败；及时
                            # 退出，让状态机按失败策略重新对齐。
                            time.sleep(0.55)
                            break
                        # 已经确认在绳上后，小地图 Y 量化或短暂遮挡不能
                        # 成为松开 UP 的依据。保持按键，等待坐标恢复或
                        # target_y 真正到达。
                        if not stuck_logged:
                            stuck_logged = True
                            self.log(
                                "🟡 [攀爬坐标暂稳] 已确认在绳上，继续保持UP，"
                                "不因量化Y短暂停滞而交还FSM"
                            )
                else:
                    stuck_count = 0
                    stuck_logged = False
                last_y = cur_py

            # 长绳预算只用于“已经抓住后的连续攀爬”，不能让一次根本
            # 没抓住的起跳也等待十几秒。起跳后 1.0 秒仍未同时满足上升
            # 与绳轴贴合，立即按失败返回，由状态机重新对齐重试。
            if (
                not grab_confirmed
                and (time.perf_counter() - climb_start) >= grab_confirm_grace_until
            ):
                self.log(
                    f"⚠️ [抓取确认超时] {grab_confirm_grace_until:.2f}s内未连续检测到"
                    "上升与绳轴贴合，提前结束并重试"
                )
                break

        if not grab_confirmed or grab_lost:
            self.driver.key_up("up")
            time.sleep(0.08)
            jump_rise = (
                max(0.0, float(init_py) - float(jump_peak_y))
                if init_py is not None and jump_peak_y is not None
                else 0.0
            )
            # 这里不用一个很大的固定阈值判断“有没有跳”：只要观测到
            # 6px 以上的上升就标记为 Alt 已产生视觉响应，同时保留完整跳高
            # 供日志分析。它只是输入验证证据，不参与抓绳成功判定。
            visual_jump_seen = jump_rise >= 6.0
            final_raw_x = None
            final_raw_y = None
            final_rope_dx = None
            if last_observed_raw_p is not None:
                try:
                    final_raw_x = float(last_observed_raw_p[0])
                    final_raw_y = float(last_observed_raw_p[1])
                    final_rope_dx = abs(final_raw_x - float(target_ladder_x))
                except (TypeError, ValueError, IndexError):
                    pass
            final_raw_text = (
                f"({final_raw_x:.1f},{final_raw_y:.1f})"
                if final_raw_x is not None and final_raw_y is not None
                else str(last_observed_raw_p)
            )
            rope_dx_text = (
                f"{final_rope_dx:.1f}px" if final_rope_dx is not None else "N/A"
            )
            init_y_text = f"{float(init_py):.1f}" if init_py is not None else "N/A"
            peak_y_text = (
                f"{float(jump_peak_y):.1f}" if jump_peak_y is not None else "N/A"
            )
            self.log(
                f"❌ [抓取失败] 跳起后未能吸附【{target_type}】；"
                f"Alt视觉起跳={'是' if visual_jump_seen else '否'}，"
                f"起跳Y={init_y_text}，顶点Y={peak_y_text}，"
                f"观测跳高={jump_rise:.1f}px，最终raw={final_raw_text}，"
                f"绳轴误差={rope_dx_text}，最终世界坐标={last_observed_p}"
            )
            return False

        if reached_target_top:
            return self._adaptive_top_exit(
                target_ladder_x=target_ladder_x,
                get_raw_player_pos=get_raw_player_pos,
                landing_confirm_fn=landing_confirm_fn,
                step_off_direction=step_off_direction,
                top_hold_sec=top_hold_sec,
                stop_event=stop_event,
                is_landed_fn=is_landed_fn,
            )

        self.driver.key_up("up")
        time.sleep(0.08)

        # Phase 4: 登顶轻点横移踏入平台内部
        if step_off_direction in ("left", "right"):
            # A very short tap can leave the character attached to the rope;
            # use a controlled lateral pulse long enough to cross the rope
            # attachment window, then allow the landing state to settle.
            step_off_started_at = time.perf_counter()
            self.driver.press_key(step_off_direction, duration_ms=80)
        else:
            step_off_started_at = time.perf_counter()

        # The navigation coordinate is snapped to nearby platforms, so it
        # cannot prove a rope exit by itself. Confirm landing with raw
        # minimap/world coordinates after the step-off pulse.
        if landing_confirm_fn is not None:
            confirm_deadline = time.perf_counter() + 0.70
            while time.perf_counter() < confirm_deadline:
                if landing_confirm_fn():
                    settle_ms = (time.perf_counter() - step_off_started_at) * 1000.0
                    self.log(
                        "✅ [脱绳确认] 原始坐标已连续稳定在目标平台内部，"
                        f"横移至交还FSM={settle_ms:.0f}ms"
                    )
                    self.last_landing_confirmed = True
                    return True
                # 不再先固定空等250ms；从横移结束立即采样，三次稳定
                # 命中后马上交还FSM，使已落台目标可以及时触发攻击。
                time.sleep(0.025)
            # Y 已确认爬升即代表抓取动作成功；这里只是目标平台的
            # 原始坐标确认尚未收敛。绝不能据此再执行一次起跳，否则会
            # 在绳顶/相邻平台被当作起点重新跳落。
            self.log("⚠️ [脱绳未确认] 已确认抓取，等待平台状态收敛，禁止重复起跳")
            return True

        # 没有物理落台回调的旧调用方仍保留原先的短暂动画缓冲。
        if step_off_direction in ("left", "right"):
            time.sleep(0.25)
        self.last_landing_confirmed = True
        return True

    def stop(self):
        """停止本控制器发出的动作，不干扰玩家的物理键盘。"""
        tracked_keys = getattr(self.driver, "active_keys", None)
        had_tracked_keys = bool(tracked_keys) if tracked_keys is not None else True
        if self.current_held_key is not None:
            self.driver.key_up(self.current_held_key)
            self.current_held_key = None
        # InputDriver 能精确知道哪些键是程序按下的。常规 stop
        # 只释放这些键；F6 停止/窗口退出仍会在上层调用
        # release_all_keys() 执行一次完整急停。
        if had_tracked_keys:
            release_tracked = getattr(self.driver, "release_tracked_keys", None)
            if callable(release_tracked):
                release_tracked()
            else:
                # 兼容测试替身和旧的自定义 InputDriver。
                self.driver.release_all_keys()
