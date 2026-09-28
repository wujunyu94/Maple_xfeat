"""世界坐标 Kalman 滤波器。

``WorldCoordinateKalman`` 是早期的二维常速度滤波器。当前导航实验使用
``InputAwareHorizontalKalman``：它把方向键产生的加速/减速作为控制输入，
把小地图黄点换算后的世界 X 当作量化区间观测；当前导航与拓扑均使用
它的连续 X 输出。
"""

import threading
import time
from typing import Optional, Tuple

import numpy as np


class WorldCoordinateKalman:
    """对 [x, y, vx, vy] 做二维常速度模型滤波。"""

    def __init__(self, measurement_noise: float = 36.0, process_noise: float = 180.0):
        self.measurement_noise = float(measurement_noise)
        self.process_noise = float(process_noise)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._state: Optional[np.ndarray] = None
            self._cov = np.eye(4, dtype=np.float64) * 1000.0
            self._last_t: Optional[float] = None

    def update(self, measurement: Tuple[float, float], timestamp: Optional[float] = None) -> Tuple[float, float]:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        z = np.asarray(measurement, dtype=np.float64).reshape(2)
        with self._lock:
            if self._state is None:
                self._state = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
                self._last_t = now
                return float(z[0]), float(z[1])

            dt = float(np.clip(now - (self._last_t or now), 1e-3, 0.25))
            self._last_t = now
            F = np.array([[1.0, 0.0, dt, 0.0],
                          [0.0, 1.0, 0.0, dt],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            q = self.process_noise
            Q = q * np.array([[dt**4 / 4, 0.0, dt**3 / 2, 0.0],
                              [0.0, dt**4 / 4, 0.0, dt**3 / 2],
                              [dt**3 / 2, 0.0, dt**2, 0.0],
                              [0.0, dt**3 / 2, 0.0, dt**2]], dtype=np.float64)
            H = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
            R = np.eye(2, dtype=np.float64) * self.measurement_noise

            pred = F @ self._state
            pred_cov = F @ self._cov @ F.T + Q
            innovation = z - H @ pred
            S = H @ pred_cov @ H.T + R
            K = pred_cov @ H.T @ np.linalg.inv(S)
            self._state = pred + K @ innovation
            self._cov = (np.eye(4) - K @ H) @ pred_cov
            return float(self._state[0]), float(self._state[1])


class InputAwareHorizontalKalman:
    """输入感知的一维水平 Kalman 影子模型。

    状态向量为 ``[x, vx]``。方向键不是观测，而是已知控制输入：按住方向
    时按人物推力加速，松键时按地面阻力减速。黄点世界 X 是量化观测；同一
    个量化格的 60Hz 重复值不会被当作 60 份独立证据，避免把格内运动强行
    吸回格点中心。

    该类是当前 F6、平台定位和拓扑 YOU 的主水平坐标模型。
    """

    def __init__(
        self,
        speed_percent: float = 103.0,
        push_accel: float = 1500.0,
        drag_accel: float = 900.0,
        process_accel_noise: float = 520.0,
        measurement_noise_floor: float = 4.0,
    ):
        self.speed_percent = float(speed_percent)
        self.push_accel = float(push_accel)
        self.drag_accel = float(drag_accel)
        self.process_accel_noise = float(process_accel_noise)
        self.measurement_noise_floor = float(measurement_noise_floor)
        self.v_max = 125.0 * self.speed_percent / 100.0
        self.measurement_step_px = 40.0
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._state: Optional[np.ndarray] = None
            self._cov = np.diag([1600.0, 10000.0]).astype(np.float64)
            self._last_t: Optional[float] = None
            self._last_measurement: Optional[float] = None
            self._last_measurement_update_t: Optional[float] = None
            self._last_measurement_change_t: Optional[float] = None
            self._direction_zero_since: Optional[float] = None
            self.direction = 0
            self.last_innovation = 0.0
            self.last_gain_x = 0.0
            self.last_reanchored = False

    @property
    def x(self) -> Optional[float]:
        with self._lock:
            return None if self._state is None else float(self._state[0])

    @property
    def vx(self) -> float:
        with self._lock:
            return 0.0 if self._state is None else float(self._state[1])

    @property
    def position_variance(self) -> float:
        with self._lock:
            return float(self._cov[0, 0])

    def set_measurement_step(self, world_px_per_minimap_px: float) -> None:
        try:
            value = float(world_px_per_minimap_px)
        except (TypeError, ValueError):
            return
        if value > 0.0:
            with self._lock:
                self.measurement_step_px = value

    def set_speed_percent(self, speed_percent: float) -> float:
        value = max(1.0, min(200.0, float(speed_percent)))
        with self._lock:
            self._predict_locked(time.perf_counter())
            self.speed_percent = value
            self.v_max = 125.0 * value / 100.0
            if self._state is not None:
                self._state[1] = np.clip(self._state[1], -self.v_max, self.v_max)
        return self.speed_percent

    def set_direction(self, direction: int, timestamp: Optional[float] = None) -> None:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._predict_locked(now)
            new_direction = int(max(-1, min(1, direction)))
            if new_direction == 0 and self.direction != 0:
                self._direction_zero_since = now
            elif new_direction != 0:
                self._direction_zero_since = None
            self.direction = new_direction

    def _control_acceleration(self, vx: float, dt: float) -> float:
        if self.direction:
            # 已达到同向速度上限后不继续施加推力；反向键则先制动再反向。
            if vx * self.direction >= self.v_max:
                return 0.0
            return self.direction * self.push_accel
        if vx > 0.0:
            return -min(self.drag_accel, vx / max(dt, 1e-6))
        if vx < 0.0:
            return min(self.drag_accel, -vx / max(dt, 1e-6))
        return 0.0

    def _predict_locked(self, now: float) -> None:
        if self._last_t is None:
            self._last_t = now
            return
        dt = max(0.0, min(0.25, now - self._last_t))
        self._last_t = now
        if self._state is None or dt <= 0.0:
            return

        old_v = float(self._state[1])
        accel = self._control_acceleration(old_v, dt)
        new_v = float(np.clip(old_v + accel * dt, -self.v_max, self.v_max))
        self._state[0] += 0.5 * (old_v + new_v) * dt
        self._state[1] = new_v

        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = self.process_accel_noise ** 2
        q_mat = q * np.array(
            [[dt ** 4 / 4.0, dt ** 3 / 2.0],
             [dt ** 3 / 2.0, dt ** 2]],
            dtype=np.float64,
        )
        self._cov = f @ self._cov @ f.T + q_mat

    def correct_measurement(
        self, measured_x: float, timestamp: Optional[float] = None
    ) -> float:
        """预测到当前时刻并吸收一次黄点世界 X 观测。"""
        now = float(timestamp if timestamp is not None else time.perf_counter())
        measured_x = float(measured_x)
        with self._lock:
            self._predict_locked(now)
            self.last_reanchored = False
            if self._state is None:
                self._state = np.array([measured_x, 0.0], dtype=np.float64)
                self._cov = np.diag([
                    max(1.0, self.measurement_step_px ** 2 / 12.0),
                    max(100.0, self.v_max ** 2),
                ]).astype(np.float64)
                self._last_measurement = measured_x
                self._last_measurement_update_t = now
                self._last_measurement_change_t = now
                return measured_x

            half_step = max(0.5, self.measurement_step_px * 0.5)
            interval_low = measured_x - half_step
            interval_high = measured_x + half_step
            predicted_x = float(self._state[0])
            if predicted_x < interval_low:
                interval_innovation = interval_low - predicted_x
            elif predicted_x > interval_high:
                interval_innovation = interval_high - predicted_x
            else:
                interval_innovation = 0.0
            self.last_innovation = interval_innovation
            reanchor_limit = max(60.0, self.measurement_step_px * 2.25)
            if abs(interval_innovation) >= reanchor_limit:
                self._state[:] = (measured_x, 0.0)
                self._cov = np.diag([
                    max(1.0, self.measurement_step_px ** 2 / 12.0),
                    max(100.0, self.v_max ** 2),
                ]).astype(np.float64)
                self._last_measurement = measured_x
                self._last_measurement_update_t = now
                self._last_measurement_change_t = now
                self.last_gain_x = 1.0
                self.last_reanchored = True
                return measured_x

            changed = (
                self._last_measurement is None
                or abs(measured_x - self._last_measurement) >= 1e-6
            )
            if changed:
                self._last_measurement = measured_x
                self._last_measurement_change_t = now

            # 量化黄点只说明真实 X 落在一个区间内，而不是精确等于格点
            # 中心。预测仍在区间内时绝不移动均值；只有越界时才把最近的
            # 区间边界作为约束观测。这样停在格内 +7px 的估计不会被同一
            # 个 15px 黄点格反复吸回 0px。
            if interval_innovation != 0.0:
                h = np.array([[1.0, 0.0]], dtype=np.float64)
                boundary_measurement = (
                    interval_low if interval_innovation > 0.0 else interval_high
                )
                # 区间边界本身来自量化规则，不再叠加整格方差；保留一个
                # 小噪声下限，避免协方差数值退化。
                r = max(1.0, self.measurement_noise_floor)
                innovation = boundary_measurement - float((h @ self._state)[0])
                s = float(h @ self._cov @ h.T) + r
                k = (self._cov @ h.T) / max(s, 1e-9)
                self._state = self._state + k[:, 0] * innovation
                self._cov = (np.eye(2, dtype=np.float64) - k @ h) @ self._cov
                # 数值增益小于1时均值可能仍残留在区间外；区间约束必须
                # 始终成立，因此最终投影到最近边界。
                self._state[0] = np.clip(self._state[0], interval_low, interval_high)
                self._state[1] = np.clip(self._state[1], -self.v_max, self.v_max)
                self.last_innovation = float(innovation)
                self.last_gain_x = float(k[0, 0])
                self._last_measurement_update_t = now
            else:
                self.last_gain_x = 0.0

            # 松键且黄点量化格连续稳定后，“人物停止”提供的是 Vx=0
            # 约束，而不是 X=格点中心。直接清零速度和 X-V 相关性，保留
            # 当前格内 X 均值。
            stable_from = max(
                self._direction_zero_since or now,
                self._last_measurement_change_t or now,
            )
            if self.direction == 0 and now - stable_from >= 0.15:
                self._state[1] = 0.0
                self._cov[0, 1] = 0.0
                self._cov[1, 0] = 0.0
                self._cov[1, 1] = min(float(self._cov[1, 1]), 4.0)
            return float(self._state[0])

    def explicit_reanchor(
        self,
        measured_x: float,
        timestamp: Optional[float] = None,
        reset_velocity: bool = False,
    ) -> float:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._predict_locked(now)
            measured_x = float(measured_x)
            if self._state is None:
                self._state = np.array([measured_x, 0.0], dtype=np.float64)
            else:
                self._state[0] = measured_x
                if reset_velocity:
                    self._state[1] = 0.0
            position_variance = max(1.0, self.measurement_step_px ** 2 / 12.0)
            if reset_velocity:
                self._cov = np.diag([
                    position_variance, max(100.0, self.v_max ** 2)
                ]).astype(np.float64)
            else:
                # X 已由外部明确重锚，旧的 X-V 相关性不应跨越坐标基准。
                self._cov[0, :] = 0.0
                self._cov[:, 0] = 0.0
                self._cov[0, 0] = position_variance
            self._last_measurement = measured_x
            self._last_measurement_update_t = now
            self._last_measurement_change_t = now
            self.last_reanchored = True
            return measured_x

    def reset_velocity(self, timestamp: Optional[float] = None) -> Optional[float]:
        """只清除水平速度，保留连续 X 与最后一个真实量化观测区间。"""
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._predict_locked(now)
            if self._state is None:
                return None
            self._state[1] = 0.0
            self._cov[0, 1] = 0.0
            self._cov[1, 0] = 0.0
            self._cov[1, 1] = min(float(self._cov[1, 1]), 4.0)
            return float(self._state[0])

    def predict(self, timestamp: Optional[float] = None) -> Optional[float]:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._predict_locked(now)
            return None if self._state is None else float(self._state[0])

    def snapshot(self, timestamp: Optional[float] = None) -> dict:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        with self._lock:
            self._predict_locked(now)
            measurement = self._last_measurement
            half_step = max(0.5, self.measurement_step_px * 0.5)
            return {
                "x": None if self._state is None else float(self._state[0]),
                "vx": 0.0 if self._state is None else float(self._state[1]),
                "direction": int(self.direction),
                "position_variance": float(self._cov[0, 0]),
                "innovation": float(self.last_innovation),
                "gain_x": float(self.last_gain_x),
                "reanchored": bool(self.last_reanchored),
                "measurement": measurement,
                "interval_low": None if measurement is None else measurement - half_step,
                "interval_high": None if measurement is None else measurement + half_step,
            }
