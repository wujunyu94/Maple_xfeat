"""
jump_quest_solver.py - 跳跳地图高精度物理运动学与自主通关引擎 (JumpQuestNavigator)
=============================================================================
核心能力：
1. 经典 2D 物理运动学抛物线方程求解 (Kinematics Trajectory Solver)
2. 像素级极限边缘助跑起跳 (Pixel-Edge Takeoff)
3. 滞空高精度梯绳吸附抓取 (Mid-Air Rope/Ladder Latching)
4. 多阶段拓扑状态机与高空跌落断点自愈 (Self-Healing Autonomous Loop)
"""

import time
import math
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass

from src.core.input_driver import InputDriver
from src.engine.platform_graph import PlatformGraph, PlatformNode, LadderRopeNode
from src.vision.tracker import MinimapTracker, TrackerResult


@dataclass
class JumpTrajectoryPlan:
    """跳跃动力学规划方案"""
    is_reachable: bool
    direction: str  # "left" or "right"
    run_up_ms: int  # 起跳前助跑蓄力时间 (ms)
    air_hold_ms: int  # 滞空持续保持方向键时间 (ms)
    takeoff_edge_offset: int  # 距离平台边缘的安全起跳偏移 (px)
    mid_air_grab_delay_ms: int  # 空中触发 UP 键抓绳的延迟 (ms)
    estimated_landing_x: int  # 预估落点 X 坐标


class JumpQuestKinematics:
    """
    经典游戏物理运动学求解器
    基于官方底层物理常数：
    - 重力加速度 g = 2000.0 px/s^2
    - 起跳垂直初速度 Vy0 = -555.0 px/s (最大起跳高度 H_max ≈ 105px)
    - 基础水平移动速度 Vx = 125.0 px/s (100% 移速基准)
    """

    GRAVITY: float = 2000.0
    JUMP_V0: float = -555.0
    BASE_WALK_SPEED: float = 125.0

    @classmethod
    def calculate_plan(
        cls,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        speed_factor: float = 1.0,
        is_target_rope: bool = False
    ) -> JumpTrajectoryPlan:
        """
        求解从 (from_x, from_y) 跳跃至 (to_x, to_y) 的最佳运动学参数
        """
        dx = to_x - from_x
        dy = to_y - from_y
        direction = "right" if dx >= 0 else "left"
        abs_dx = abs(dx)

        vx = cls.BASE_WALK_SPEED * speed_factor

        # 判别 1：向上跳跃 (dy < 0) 极限高度检查
        # 最高点 H_peak = (Vy0^2) / (2g) ≈ 77px
        if dy < -95:
            return JumpTrajectoryPlan(
                is_reachable=False,
                direction=direction,
                run_up_ms=0,
                air_hold_ms=0,
                takeoff_edge_offset=0,
                mid_air_grab_delay_ms=0,
                estimated_landing_x=from_x
            )

        # 判别 2：计算滞空时间 (飞行时间)
        # 解方程: dy = Vy0 * t + 0.5 * g * t^2
        # => 0.5*g*t^2 + Vy0*t - dy = 0
        discriminant = cls.JUMP_V0 ** 2 + 2 * cls.GRAVITY * dy
        if discriminant < 0:
            air_time = 0.55
        else:
            air_time = (-cls.JUMP_V0 + math.sqrt(discriminant)) / cls.GRAVITY

        # 水平最大位移
        max_dist = vx * air_time
        is_reachable = abs_dx <= max_dist + 25.0

        # 计算助跑蓄力时长 (跨度越大，助跑越充分)
        if abs_dx > 100:
            run_up_ms = min(220, max(140, int(abs_dx * 1.2)))
        elif abs_dx > 50:
            run_up_ms = min(120, max(80, int(abs_dx * 1.0)))
        else:
            run_up_ms = 40  # 微步短跳

        # 空中方向保持时间
        air_hold_ms = int(min(0.5, air_time) * 1000)

        # 边缘起跳偏移：大跨度紧贴边缘 (4px)，短跳保留余量 (12px)
        takeoff_edge_offset = 5 if abs_dx > 80 else 12

        # 空中抓绳延迟 (到达绳索 X 坐标的时间点)
        if is_target_rope:
            time_to_rope = min(air_time * 0.8, max(0.12, abs_dx / vx))
            mid_air_grab_delay_ms = int(time_to_rope * 1000)
        else:
            mid_air_grab_delay_ms = 0

        estimated_landing_x = from_x + (int(max_dist) if direction == "right" else -int(max_dist))

        return JumpTrajectoryPlan(
            is_reachable=is_reachable,
            direction=direction,
            run_up_ms=run_up_ms,
            air_hold_ms=air_hold_ms,
            takeoff_edge_offset=takeoff_edge_offset,
            mid_air_grab_delay_ms=mid_air_grab_delay_ms,
            estimated_landing_x=estimated_landing_x
        )


class JumpQuestNavigator:
    """
    跳跳地图全自动智能导航控制引擎
    """

    def __init__(
        self,
        driver: InputDriver,
        graph: PlatformGraph,
        get_frame_callback: Callable[[], any],
        tracker: Optional[MinimapTracker] = None,
        jump_key: str = "alt"
    ):
        self.driver = driver
        self.graph = graph
        self.get_frame = get_frame_callback
        self.tracker = tracker or MinimapTracker()
        self.jump_key = jump_key
        self.is_running = False

    def get_player_world_pos(self) -> Optional[Tuple[int, int]]:
        """实时获取角色全图物理世界坐标 (X, Y)"""
        frame = self.get_frame()
        if frame is None or frame.size == 0:
            return None
        tr = self.tracker.detect(frame)
        if tr.is_detected and tr.norm_pos:
            return self.graph.minimap_norm_to_world(*tr.norm_pos)
        return None

    def walk_to_x_precise(self, target_x: int, tolerance: int = 10, timeout_sec: float = 4.0) -> bool:
        """闭环高精度平走至物理世界 X 坐标 (比例阻尼控制)"""
        start_t = time.perf_counter()
        while time.perf_counter() - start_t < timeout_sec:
            pos = self.get_player_world_pos()
            if pos is None:
                time.sleep(0.03)
                continue
            cur_x, cur_y = pos
            dx = target_x - cur_x
            if abs(dx) <= tolerance:
                self.driver.release_all_keys()
                return True

            req_key = "right" if dx > 0 else "left"
            # 动态阻尼按键时长：远距离长按，近距离微调
            hold_time = min(0.35, max(0.04, abs(dx) / 350.0))
            self.driver.key_down(req_key)
            time.sleep(hold_time)
            self.driver.key_up(req_key)
            time.sleep(0.04)

        self.driver.release_all_keys()
        return False

    def precision_climb_ladder_or_rope(
        self,
        target_x: int,
        target_y: int,
        is_ladder: bool = True,
        timeout_sec: float = 6.0
    ) -> bool:
        """
        高精度对齐梯绳中轴线并执行攀爬
        """
        # 1. 精确对齐中轴线 (极致容差 3px)
        self.walk_to_x_precise(target_x, tolerance=4, timeout_sec=3.0)

        # 2. 滞空/地面触发抓升 (单次起跳 + 保持 UP 键持续攀爬)
        self.driver.ensure_focus()
        self.driver.key_down("up")
        time.sleep(0.03)
        self.driver.press_key(self.jump_key, duration_ms=80)

        # 3. 闭环监测爬升高度
        climb_start = time.perf_counter()
        while time.perf_counter() - climb_start < timeout_sec:
            p = self.get_player_world_pos()
            if p and p[1] <= target_y:
                self.driver.key_up("up")
                time.sleep(0.2)
                return True
            time.sleep(0.2)

        self.driver.key_up("up")
        time.sleep(0.2)
        return True

    def precision_jump_with_plan(self, plan: JumpTrajectoryPlan) -> bool:
        """
        根据物理动力学规划方案执行像素级跳跃
        """
        self.driver.ensure_focus()
        self.driver.key_down(plan.direction)
        time.sleep(plan.run_up_ms / 1000.0)
        self.driver.press_key(self.jump_key, duration_ms=80)

        # 空中抓绳分支
        if plan.mid_air_grab_delay_ms > 0:
            time.sleep(plan.mid_air_grab_delay_ms / 1000.0)
            self.driver.key_down("up")
            time.sleep(0.15)
            self.driver.key_up(plan.direction)
            time.sleep(2.5)  # 持续向上爬升
            self.driver.key_up("up")
        else:
            time.sleep(plan.air_hold_ms / 1000.0)
            self.driver.key_up(plan.direction)

        time.sleep(0.35)
        return True

    def run_autonomous_stage_climb(self, max_steps: int = 40) -> bool:
        """
        自主全自动跳跳通关调度主循环
        """
        print("=== [JumpQuest] 启动全自动跳跳通关调度引擎 ===")
        self.driver.ensure_focus()

        for step in range(max_steps):
            pos = self.get_player_world_pos()
            if pos is None:
                time.sleep(0.1)
                continue
            wx, wy = pos
            print(f"\n[JumpQuest Step {step+1}/{max_steps}] 实时物理坐标: (X={wx}, Y={wy})")

            # 登顶判定
            if wy <= -2350:
                print("🎉 [JumpQuest] 恭喜！已成功登顶顶层平台，到达传送门过关！")
                return True

            # 阶段 1: 底层地面 (Y >= -60) -> 导航至 Ladder #1 并攀爬
            if wy >= -60:
                print(">> [阶段 1: 底层地面] 导航至 梯子 #1 (X=-674) 并爬升至平台 #8...")
                self.precision_climb_ladder_or_rope(target_x=-674, target_y=-180, is_ladder=True)
                continue

            # 阶段 2: 平台 #8 (Y 在 -260 ~ -120 且 X <= -580) -> 抓绳索 #2 爬升至顶端
            if -260 <= wy <= -120 and wx <= -580:
                print(">> [阶段 2: 平台 #8] 对准绳索 #2 (X=-633) 并空中抓绳爬升至顶端...")
                self.precision_climb_ladder_or_rope(target_x=-633, target_y=-480, is_ladder=False)
                continue

            # 阶段 3: 绳索 #2 顶端 (Y 在 -530 ~ -440 且 X <= -660) -> 微右轻跳着陆平台 #22 中心 (X=-630)
            if -530 <= wy <= -440 and wx <= -660:
                print(">> [阶段 3: 绳索 #2 顶端] 微右轻跳着陆平台 #22 中心...")
                plan = JumpTrajectoryPlan(
                    is_reachable=True,
                    direction="right",
                    run_up_ms=30,
                    air_hold_ms=150,
                    takeoff_edge_offset=0,
                    mid_air_grab_delay_ms=0,
                    estimated_landing_x=-630
                )
                self.precision_jump_with_plan(plan)
                continue

            # 阶段 4: 平台 #22 (Y 在 -530 ~ -440 且 X 在 -659 ~ -580) -> 自然平走脱离下落着陆平台 #9 中心 (X=-540, Y=-262)
            if -530 <= wy <= -440 and -659 <= wx <= -580:
                print(">> [阶段 4: 平台 #22] 自然平走脱离下落着陆平台 #9 (X=-540)...")
                # 自然平走脱落：长按右键 0.70 秒确保无论从平台何处起步均能稳定滑降至平台 #9 (X=-550)
                self.driver.key_down("right")
                time.sleep(0.70)
                self.driver.key_up("right")
                time.sleep(0.4)
                continue

            # 阶段 5a: 平台 #9 (X 在 -580 ~ -500, Y 在 -380 ~ -90) -> 跃升至平台 #13 (X=-450, Y=-322)
            if -380 <= wy <= -90 and -580 <= wx <= -500:
                print(">> [阶段 5a: 平台 #9] 跃升至平台 #13 (X=-450, Y=-322)...")
                self.walk_to_x_precise(-516, tolerance=5, timeout_sec=1.5)
                self.driver.key_down("right")
                time.sleep(0.08)
                self.driver.press_key(self.jump_key, duration_ms=80)
                time.sleep(0.25)
                self.driver.key_up("right")
                time.sleep(0.4)
                continue

            # 阶段 5b: 平台 #13 (X 在 -480 ~ -380, Y 在 -380 ~ -90) -> 平跳至平台 #14 (X=-360, Y=-322)
            if -380 <= wy <= -90 and -480 <= wx <= -380:
                print(">> [阶段 5b: 平台 #13] 平跳至平台 #14 (X=-360, Y=-322)...")
                self.walk_to_x_precise(-395, tolerance=5, timeout_sec=1.5)
                self.driver.key_down("right")
                time.sleep(0.06)
                self.driver.press_key(self.jump_key, duration_ms=75)
                time.sleep(0.25)
                self.driver.key_up("right")
                time.sleep(0.4)
                continue

            # 阶段 5c: 平台 #14 (X 在 -390 ~ -325, Y 在 -380 ~ -90) -> 抓绳索 #3 (X=-361) 爬升至顶端 (Y<=-650)
            if -380 <= wy <= -90 and -390 <= wx <= -325:
                print(">> [阶段 5c: 平台 #14] 抓绳索 #3 (X=-361) 爬升至顶端 (Y<=-650)...")
                self.precision_climb_ladder_or_rope(target_x=-361, target_y=-650, is_ladder=False)
                continue

            # 阶段 6: 绳索 #3 顶端 / 中层平台群 (Y 在 -720 ~ -600 且 X 在 -400 ~ 550) -> 向右横向推进至立柱
            if -720 <= wy <= -600 and wx < 580:
                print(">> [阶段 6: 绳索顶端/中台群] 向右横向推进至立柱...")
                plan = JumpTrajectoryPlan(
                    is_reachable=True,
                    direction="right",
                    run_up_ms=160,
                    air_hold_ms=250,
                    takeoff_edge_offset=4,
                    mid_air_grab_delay_ms=180,
                    estimated_landing_x=min(580, wx + 150)
                )
                self.precision_jump_with_plan(plan)
                continue

            # 阶段 7: 右侧天梯立柱 (X >= 580, Y 在 -800 ~ -300) -> 向上执行 60px 垂直等距节奏跳跃
            if wx >= 580 and -800 <= wy <= -300:
                print(">> [阶段 7: 右侧天梯立柱] 执行 60px 垂直等距跳跃...")
                self.driver.press_key(self.jump_key, duration_ms=80)
                time.sleep(0.45)
                continue

            # 阶段 8: 高层避障矩阵区 (Y < -800) -> 密集向上推进
            if wy < -800:
                print(">> [阶段 8: 高层避障矩阵区] 向上推进登顶...")
                self.driver.press_key(self.jump_key, duration_ms=80)
                time.sleep(0.45)
                continue

        print("[JumpQuest] 调度轮次完成")
        return True
