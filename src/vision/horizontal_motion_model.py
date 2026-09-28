"""基于方向键输入的水平运动预测模型。"""

import threading
import time
from typing import Optional


class HorizontalMotionModel:
    def __init__(self, speed_percent: float = 103.0,
                 push_accel: float = 1500.0, drag_accel: float = 900.0):
        self.speed_percent = float(speed_percent)
        self.push_accel = float(push_accel)
        self.drag_accel = float(drag_accel)
        self.v_max = 125.0 * self.speed_percent / 100.0
        # 当前小地图黄点 1px 对应的世界坐标距离，由地图/视口动态设置。
        self.measurement_step_px = 40.0
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.x: Optional[float] = None
            self.vx = 0.0
            self.direction = 0
            self.key_events = []
            self.last_t: Optional[float] = None
            self.last_measurement_t: Optional[float] = None
            self.last_raw_measurement: Optional[float] = None
            self.last_raw_change_t: Optional[float] = None
            self.raw_still_since: Optional[float] = None
            self.blocked = False
            # 当操作系统无法读取手动按键时，用相邻黄点量化格的变化
            # 推断短时运动方向；它只作为键盘输入的后备来源。
            self.inferred_direction = 0
            self.inferred_motion_until = 0.0
            # 自动控制刚松开方向键后，后续几个量化测量格仍会继续移动。
            # 它们是先前速度的惯性，不应误当成“玩家重新按住方向键”。
            self.explicit_release_until = 0.0

    def set_measurement_step(self, world_px_per_minimap_px: float) -> None:
        """设置当前地图量化步长，禁止使用固定世界坐标阈值。"""
        try:
            value = float(world_px_per_minimap_px)
            if value > 0.0:
                with self._lock:
                    self.measurement_step_px = value
        except (TypeError, ValueError):
            pass

    def set_speed_percent(self, speed_percent: float) -> float:
        """运行时更新人物移速，并同步更新水平速度上限。"""
        value = max(1.0, min(200.0, float(speed_percent)))
        with self._lock:
            self._advance(time.perf_counter())
            self.speed_percent = value
            self.v_max = 125.0 * value / 100.0
            self.vx = max(-self.v_max, min(self.v_max, self.vx))
        return self.speed_percent

    def set_direction(self, direction: int, timestamp: Optional[float] = None) -> None:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._advance(now)
            direction = int(max(-1, min(1, direction)))
            if direction != self.direction:
                self.key_events.append((now, direction))
            self.direction = direction
            if direction != 0:
                self.inferred_direction = 0
                self.inferred_motion_until = 0.0
            if direction == 0:
                # 松开方向键后解除边界/障碍阻塞锁存。
                self.blocked = False
                self.explicit_release_until = now + 0.70

    def _advance(self, now: float) -> None:
        if self.last_t is None:
            self.last_t = now
            return
        dt = max(0.0, min(0.25, now - self.last_t))
        self.last_t = now
        if self.direction and not self.blocked:
            self.vx += self.direction * self.push_accel * dt
        elif now < self.inferred_motion_until:
            # 观测到黄点刚跨过一个量化格：在下一格到来前维持观测速度，
            # 从而填补 17/36px 等离散测量之间的连续世界坐标。
            pass
        elif self.vx > 0.0:
            self.vx = max(0.0, self.vx - self.drag_accel * dt)
        elif self.vx < 0.0:
            self.vx = min(0.0, self.vx + self.drag_accel * dt)
        self.vx = max(-self.v_max, min(self.v_max, self.vx))
        if self.x is not None:
            self.x += self.vx * dt

    def correct_measurement(self, measured_x: float, timestamp: Optional[float] = None) -> float:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._advance(now)
            measured_x = float(measured_x)
            if self.x is None:
                self.x = measured_x
                self.last_measurement_t = now
                self.last_raw_measurement = measured_x
                self.last_raw_change_t = now
                self.raw_still_since = now
            else:
                # 预测只负责填补相邻黄点量化格之间的连续坐标，不能跨越
                # 多个格点继续相信旧状态。出现这类大创新量通常意味着
                # 小地图卷轴/Canvas 匹配已切换到新的坐标视口；若不立即
                # 重定位，拓扑图会显示旧 X（例如 -139）而 F8 已是 120。
                reanchor_limit = max(1.0, self.measurement_step_px * 2.0)
                if abs(float(self.x) - measured_x) > reanchor_limit:
                    self.x = measured_x
                    self.vx = 0.0
                    self.direction = 0
                    self.inferred_direction = 0
                    self.inferred_motion_until = 0.0
                    self.blocked = False
                    self.last_raw_measurement = measured_x
                    self.last_raw_change_t = now
                    self.raw_still_since = now
                    self.last_measurement_t = now
                    return float(self.x)

                # 小地图世界坐标是量化值。短时间不变不代表角色没动，
                # 只有在持续按键、原始测量长期不变、且预测值已经越过
                # 一个量化步长时，才认为撞到地图边界/障碍物。
                if self.last_raw_measurement is None or abs(measured_x - self.last_raw_measurement) >= 1e-6:
                    previous_raw = self.last_raw_measurement
                    previous_change_t = self.last_raw_change_t
                    self.last_raw_measurement = measured_x
                    self.last_raw_change_t = now
                    self.raw_still_since = now
                    self.blocked = False
                    # 键盘状态未读到时，使用真实黄点跨格位移校正模型。
                    # 这不是每帧硬重置：仅在量化测量真正变化时更新，
                    # 其余时间由模型在格点间连续外推。
                    if (
                        self.direction == 0
                        and now >= self.explicit_release_until
                        and previous_raw is not None
                        and previous_change_t is not None
                    ):
                        observed_dt = max(0.03, now - previous_change_t)
                        observed_v = (measured_x - previous_raw) / observed_dt
                        if abs(observed_v) >= 1.0:
                            self.inferred_direction = 1 if observed_v > 0 else -1
                            self.inferred_motion_until = now + 0.45
                            self.vx = max(-self.v_max, min(self.v_max, observed_v))
                            self.x = measured_x
                elif self.raw_still_since is None:
                    self.raw_still_since = now

                raw_still_sec = now - self.raw_still_since
                prediction_error = abs(float(self.x) - measured_x)
                # 方向键已松开且 raw 黄点保持不变时，角色视为静止。
                # 清零残余速度但保留当前预测位置，避免静止后 predicted X
                # 继续因惯性漂移；这里不把预测值拉回 raw 测量值。
                if self.direction == 0 and raw_still_sec >= 0.15:
                    self.vx = 0.0
                if (
                    self.direction != 0
                    and not self.blocked
                    and raw_still_sec >= 0.35
                    and prediction_error >= max(1.0, self.measurement_step_px)
                ):
                    self.blocked = True
                    self.vx = 0.0
                    # 碰撞时允许回到当前量化测量点；这是边界修正，
                    # 不是正常停止时的周期性重校准。
                    self.x = measured_x
            # 重要：不能按固定周期把预测值重置为量化后的小地图坐标。
            # 小地图世界坐标的最小步进可能达到数十像素，周期性重锚定会
            # 在角色停止或准备跳抓时把亚像素预测重新拉回错误整数格点。
            # 因此正常运行期间只推进模型；硬校准由 explicit_reanchor()
            # 在切图、镜头重定位或人工确认时调用。
            return float(self.x)

    def explicit_reanchor(self, measured_x: float,
                          timestamp: Optional[float] = None,
                          reset_velocity: bool = False) -> float:
        """显式重定位模型，不在普通帧循环中自动调用。"""
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._advance(now)
            self.x = float(measured_x)
            self.last_measurement_t = now
            if reset_velocity:
                self.vx = 0.0
            return self.x

    def predict(self, timestamp: Optional[float] = None) -> Optional[float]:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._advance(now)
            return None if self.x is None else float(self.x)
