"""
waypoint_manager.py - 多平台航点与巡逻路线管理器 (WaypointManager)
支持用户在任意地图交互式录制/保存/加载刷怪巡逻航点，支持平走、下跳、跳台与闭环循环。
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Waypoint:
    x: int
    y: int
    action: str = "WALK"          # "WALK", "DOWN_JUMP", "JUMP_LEFT", "JUMP_RIGHT", "WAIT"
    tolerance: int = 25           # 到位容差
    wait_time_sec: float = 0.0    # 到位后停留时间
    note: str = ""


class WaypointManager:
    def __init__(self, storage_dir: str = "data/routes"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.waypoints: List[Waypoint] = []
        self.current_idx: int = 0
        self.is_recording: bool = False
        self.last_recorded_pos: Optional[Tuple[int, int]] = None
        self.current_map_id: Optional[int] = None

    def start_recording(self, map_id: int):
        """开启航点录制"""
        self.current_map_id = map_id
        self.waypoints = []
        self.is_recording = True
        self.last_recorded_pos = None

    def stop_recording(self, auto_save: bool = True) -> int:
        """停止录制并可选自动保存"""
        self.is_recording = False
        if auto_save and self.current_map_id:
            self.save_route(self.current_map_id)
        return len(self.waypoints)

    def record_step(self, player_pos: Tuple[int, int], action: str = "WALK", force: bool = False) -> bool:
        """
        录制一个移动足迹点 (根据位移距离自动稀疏采样，默认每隔 80px 采一个点)
        """
        if not self.is_recording or player_pos is None:
            return False

        px, py = player_pos
        if not force and self.last_recorded_pos is not None:
            lx, ly = self.last_recorded_pos
            # 水平位移或垂直跃迁达到阈值
            if abs(px - lx) < 80 and abs(py - ly) < 40 and action == "WALK":
                return False

        wp = Waypoint(x=int(px), y=int(py), action=action)
        self.waypoints.append(wp)
        self.last_recorded_pos = (px, py)
        return True

    def add_manual_action(self, action: str, player_pos: Optional[Tuple[int, int]] = None):
        """手动插入特殊动作航点 (如 下跳、跳台、攀爬)"""
        if player_pos is not None:
            px, py = player_pos
        elif self.last_recorded_pos is not None:
            px, py = self.last_recorded_pos
        else:
            px, py = 0, 0

        wp = Waypoint(x=int(px), y=int(py), action=action)
        self.waypoints.append(wp)

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """获取当前目标航点"""
        if not self.waypoints:
            return None
        if self.current_idx >= len(self.waypoints):
            self.current_idx = 0
        return self.waypoints[self.current_idx]

    def advance_next(self) -> Optional[Waypoint]:
        """到达当前航点，切换到下一航点 (闭环循环)"""
        if not self.waypoints:
            return None
        self.current_idx = (self.current_idx + 1) % len(self.waypoints)
        return self.waypoints[self.current_idx]

    def reset_index(self):
        self.current_idx = 0

    def save_route(self, map_id: int, filename: Optional[str] = None) -> bool:
        """保存当前路线到本地 JSON"""
        fname = filename or f"route_map_{map_id}.json"
        fpath = os.path.join(self.storage_dir, fname)
        try:
            data = {
                "map_id": map_id,
                "saved_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "waypoints": [asdict(w) for w in self.waypoints]
            }
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[WaypointManager] 保存路线失败: {e}")
            return False

    def load_route(self, map_id: int, filename: Optional[str] = None) -> bool:
        """从本地加载该地图的路线"""
        fname = filename or f"route_map_{map_id}.json"
        fpath = os.path.join(self.storage_dir, fname)
        if not os.path.exists(fpath):
            return False
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.waypoints = [Waypoint(**w) for w in data.get("waypoints", [])]
            self.current_idx = 0
            self.current_map_id = map_id
            return len(self.waypoints) > 0
        except Exception as e:
            print(f"[WaypointManager] 加载路线失败: {e}")
            return False
