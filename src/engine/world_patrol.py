"""跨地图刷怪巡逻编排器。

支持 WZ ``type=1/2`` 门及逐条确认落点的脚本门。地图内移动继续交给
PlatformPatrolFSM；本模块只负责地图级路径、出口选择、进门和切图确认。
"""

from __future__ import annotations

import heapq
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.engine.platform_graph import PlatformGraph, PlatformGraphBuilder


CROSS_MAP_PORTAL_TYPES = frozenset((1, 2))
# WZ 的 type=7 只给脚本名，toMap=999999999，无法靠 IMG 自行推断落点。
# 这条映射来自本客户端的实测：售票处 subway_in2 直接进入地铁一号线第1地区。
CONFIRMED_SCRIPT_PORTAL_TARGETS = {(103000100, "subway_in2"): (103000101, "out00")}


@dataclass(frozen=True)
class WorldPatrolStop:
    map_id: int
    platforms: Tuple[int, ...]
    dwell_min_sec: Optional[float] = None
    dwell_max_sec: Optional[float] = None
    positions: Tuple[float, ...] = ()
    position_random: Optional[float] = None
    rest_platform_id: Optional[int] = None
    rest_duration_min_sec: Optional[float] = None
    rest_duration_max_sec: Optional[float] = None
    rest_interval_min_sec: Optional[float] = None
    rest_interval_max_sec: Optional[float] = None
    position_overrides: Tuple[Tuple[int, Tuple[float, ...]], ...] = ()

    @classmethod
    def from_saved_spec(
        cls, item: Dict[str, Any], platforms: Sequence[int],
        positions: Sequence[float], fallback_dwell: Tuple[float, float],
    ) -> "WorldPatrolStop":
        """启动时恢复已应用的逐地图设置，包括唯一休息点。"""
        rest_id = item.get("rest_platform_id")
        overrides = tuple(sorted(
            (int(pid), tuple(float(value) for value in values))
            for pid, values in (item.get("position_overrides") or {}).items()
            if isinstance(values, (list, tuple)) and values
        ))
        return cls(
            int(item["map_id"]), tuple(platforms),
            float(item.get("dwell_min_sec", fallback_dwell[0])),
            float(item.get("dwell_max_sec", fallback_dwell[1])),
            tuple(positions),
            float(item.get("position_random_percent", 0.0)) / 100.0,
            rest_platform_id=int(rest_id) if rest_id not in (None, "") else None,
            rest_duration_min_sec=float(item.get("rest_duration_min_sec", 30.0)),
            rest_duration_max_sec=float(item.get("rest_duration_max_sec", 60.0)),
            rest_interval_min_sec=float(item.get("rest_interval_min_sec", 30.0)),
            rest_interval_max_sec=float(item.get("rest_interval_max_sec", 40.0)),
            position_overrides=overrides,
        )


@dataclass(frozen=True)
class VisiblePortalEdge:
    source_map_id: int
    target_map_id: int
    portal_name: str
    target_portal_name: str
    source_platform_id: int
    x: int
    y: int
    portal_type: int = 2


class WorldRoutePlanner:
    """从本地地图数据构建普通门及已确认脚本门的有向地图图。

    ``map_loader`` 存在时优先读取 Map.wz IMG；原 ``maps_dir`` JSON
    路径完整保留为后备来源。
    """

    def __init__(
        self,
        maps_dir: str,
        merge_short_platforms: bool = True,
        map_loader: Optional[Callable[[int], Dict[str, Any]]] = None,
        map_exists: Optional[Callable[[int], bool]] = None,
        map_ids: Optional[Callable[[], Iterable[int]]] = None,
    ):
        self.maps_dir = os.path.abspath(maps_dir)
        self.merge_short_platforms = bool(merge_short_platforms)
        self._external_map_loader = map_loader
        self._external_map_exists = map_exists
        self._external_map_ids = map_ids
        self._map_data: Dict[int, Dict[str, Any]] = {}
        self._graphs: Dict[int, PlatformGraph] = {}
        self._portal_edges: Dict[int, List[VisiblePortalEdge]] = {}
        self._lock = threading.RLock()

    def set_merge_short_platforms(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._lock:
            if enabled != self.merge_short_platforms:
                self.merge_short_platforms = enabled
                self._graphs.clear()
                self._portal_edges.clear()

    def map_path(self, map_id: int) -> str:
        return os.path.join(self.maps_dir, f"{int(map_id)}.json")

    def has_local_map(self, map_id: int) -> bool:
        if self._external_map_exists is not None:
            try:
                if self._external_map_exists(int(map_id)):
                    return True
            except Exception:
                pass
        return os.path.isfile(self.map_path(map_id))

    def load_map_data(self, map_id: int) -> Dict[str, Any]:
        map_id = int(map_id)
        with self._lock:
            cached = self._map_data.get(map_id)
            if cached is not None:
                return cached
            payload = None
            external_error = None
            if self._external_map_loader is not None:
                try:
                    payload = self._external_map_loader(map_id)
                except Exception as exc:
                    external_error = exc
            path = self.map_path(map_id)
            if payload is None:
                if not os.path.isfile(path):
                    if external_error is not None:
                        raise FileNotFoundError(
                            f"本地地图 IMG 与 JSON 均不可用：IMG={external_error}; JSON={path}"
                        ) from external_error
                    raise FileNotFoundError(f"缺少本地地图 JSON：{path}")
                with open(path, "r", encoding="utf-8") as stream:
                    payload = json.load(stream)
            if not isinstance(payload, dict):
                raise ValueError(f"地图 JSON 不是对象：{path}")
            actual_id = int(payload.get("id", map_id) or map_id)
            if actual_id != map_id:
                raise ValueError(f"地图 JSON 的 id={actual_id}，与文件名 {map_id} 不一致")
            self._map_data[map_id] = payload
            return payload

    def graph(self, map_id: int) -> PlatformGraph:
        map_id = int(map_id)
        with self._lock:
            graph = self._graphs.get(map_id)
            if graph is None:
                graph = PlatformGraphBuilder.build_from_map_dict(
                    self.load_map_data(map_id),
                    merge_short_platforms=self.merge_short_platforms,
                )
                self._graphs[map_id] = graph
            return graph

    def _local_map_ids(self) -> Iterable[int]:
        result = set()
        if self._external_map_ids is not None:
            try:
                result.update(int(value) for value in self._external_map_ids())
            except Exception:
                pass
        if os.path.isdir(self.maps_dir):
            for name in os.listdir(self.maps_dir):
                stem, ext = os.path.splitext(name)
                if ext.lower() == ".json" and stem.isdigit():
                    result.add(int(stem))
        return sorted(result)

    def _portal_edges_for_map(
        self, map_id: int, force_refresh: bool = False
    ) -> List[VisiblePortalEdge]:
        """按需解析一张地图，避免启动时扫描数千个 WZ IMG。"""
        map_id = int(map_id)
        with self._lock:
            if map_id in self._portal_edges and not force_refresh:
                return self._portal_edges[map_id]
            data = self.load_map_data(map_id)
            visible_portals = []
            for portal in data.get("portals", []) or []:
                if not isinstance(portal, dict):
                    continue
                try:
                    portal_type = int(portal.get("type", -1))
                    target_map = int(portal.get("toMap", 999999999))
                except (TypeError, ValueError):
                    continue
                target_portal_name = str(portal.get("toName", "") or "")
                confirmed = None
                if portal_type == 7 and target_map == 999999999:
                    confirmed = CONFIRMED_SCRIPT_PORTAL_TARGETS.get(
                        (map_id, str(portal.get("script", "") or ""))
                    )
                    if confirmed is not None:
                        target_map, target_portal_name = confirmed
                if portal_type not in CROSS_MAP_PORTAL_TYPES and confirmed is None:
                    continue
                if target_map in (map_id, 999999999):
                    continue
                if self.has_local_map(target_map):
                    visible_portals.append((portal, target_map, portal_type, target_portal_name))

            result: List[VisiblePortalEdge] = []
            if visible_portals:
                graph = self.graph(map_id)
                for portal, target_map, portal_type, target_portal_name in visible_portals:
                    source_platform = PlatformGraphBuilder._portal_platform(graph, portal)
                    if source_platform is None:
                        continue
                    result.append(VisiblePortalEdge(
                        source_map_id=map_id,
                        target_map_id=target_map,
                        portal_name=str(portal.get("portalName", "")),
                        target_portal_name=target_portal_name,
                        source_platform_id=int(source_platform.id),
                        x=int(portal.get("x", source_platform.center_x)),
                        y=int(portal.get("y", source_platform.center_y)),
                        portal_type=portal_type,
                    ))
            result.sort(key=lambda item: (item.target_map_id, item.source_platform_id, item.x))
            self._portal_edges[map_id] = result
            return result

    def portal_edges(self, force_refresh: bool = False) -> Dict[int, List[VisiblePortalEdge]]:
        with self._lock:
            if force_refresh:
                self._portal_edges.clear()
            # 外部 WZ 集合可能有数千张图，不能为了查看一个出口全部解析。
            # 无外部 loader 时保留旧 JSON 全量行为；WZ 模式由寻路 BFS
            # 逐张调用 _portal_edges_for_map。
            if self._external_map_loader is None:
                for map_id in self._local_map_ids():
                    try:
                        self._portal_edges_for_map(map_id)
                    except Exception:
                        continue
            return dict(self._portal_edges)

    def find_map_route(
        self, source_map_id: int, target_map_id: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[int]:
        """按传送次数求最短有向地图路径。"""
        source_map_id, target_map_id = int(source_map_id), int(target_map_id)
        if source_map_id == target_map_id:
            return [source_map_id]
        queue: List[Tuple[int, int, List[int]]] = [(0, source_map_id, [source_map_id])]
        best = {source_map_id: 0}
        while queue:
            if cancel_event is not None and cancel_event.is_set():
                return []
            hops, map_id, route = heapq.heappop(queue)
            if hops != best.get(map_id):
                continue
            try:
                outgoing = self._portal_edges_for_map(map_id)
            except Exception:
                outgoing = []
            for edge in outgoing:
                next_id = edge.target_map_id
                next_hops = hops + 1
                if next_hops >= best.get(next_id, 1 << 30):
                    continue
                next_route = route + [next_id]
                if next_id == target_map_id:
                    return next_route
                best[next_id] = next_hops
                heapq.heappush(queue, (next_hops, next_id, next_route))
        return []

    def choose_exit(
        self,
        source_map_id: int,
        target_map_id: int,
        current_platform_id: int,
        allow_run_jump: bool = True,
        allow_intra_map_portal: bool = True,
    ) -> Tuple[Optional[VisiblePortalEdge], List[Any]]:
        """选择从当前平台可达的跨地图出口。"""
        graph = self.graph(source_map_id)
        candidates = []
        for edge in self._portal_edges_for_map(int(source_map_id)):
            if edge.target_map_id != int(target_map_id):
                continue
            if int(current_platform_id) == edge.source_platform_id:
                route = []
            else:
                route = graph.find_path(
                    int(current_platform_id), edge.source_platform_id,
                    allow_run_jump=bool(allow_run_jump),
                    allow_portal=bool(allow_intra_map_portal),
                )
                if not route:
                    continue
            metrics = graph.path_metrics(route)
            key = (
                float(metrics["estimated_cost"]),
                int(metrics["rope_count"]),
                int(metrics["hop_count"]),
                abs(edge.x - graph.get_node(edge.source_platform_id).center_x),
            )
            candidates.append((key, edge, route))
        if not candidates:
            return None, []
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], list(candidates[0][2])

    def validate_stops(
        self,
        stops: Sequence[WorldPatrolStop],
        allow_run_jump: bool = True,
        allow_intra_map_portal: bool = True,
    ) -> List[str]:
        errors: List[str] = []
        if len(stops) < 2:
            return ["跨地图巡逻至少需要两个地图目标。"]
        if len({stop.map_id for stop in stops}) < 2:
            errors.append("跨地图巡逻至少需要两个不同的 MapID。")
        rest_stops = [stop for stop in stops if stop.rest_platform_id is not None]
        if len(rest_stops) > 1:
            errors.append("多地图巡逻同时只能配置一个休息点。")
        for stop in stops:
            if not self.has_local_map(stop.map_id):
                errors.append(f"MapID {stop.map_id} 缺少本地 IMG/JSON。")
                continue
            try:
                graph = self.graph(stop.map_id)
            except Exception as exc:
                errors.append(f"MapID {stop.map_id} 拓扑构建失败：{exc}")
                continue
            missing = [pid for pid in stop.platforms if graph.get_node(pid) is None]
            if missing:
                errors.append(
                    f"MapID {stop.map_id} 不存在平台："
                    + ", ".join(f"P{pid}" for pid in missing)
                )
                continue
            if stop.rest_platform_id is not None:
                rest_pid = int(stop.rest_platform_id)
                if graph.get_node(rest_pid) is None:
                    errors.append(f"MapID {stop.map_id} 不存在休息平台 P{rest_pid}。")
                else:
                    for pid in stop.platforms:
                        for source, target in ((pid, rest_pid), (rest_pid, pid)):
                            if source != target and not graph.find_path(
                                source, target,
                                allow_run_jump=bool(allow_run_jump),
                                allow_portal=bool(allow_intra_map_portal),
                            ):
                                errors.append(
                                    f"MapID {stop.map_id}：P{source}→休息相关P{target} 不可达。"
                                )
            for source, target in zip(stop.platforms, stop.platforms[1:] + stop.platforms[:1]):
                if source != target and not graph.find_path(
                    source,
                    target,
                    allow_run_jump=bool(allow_run_jump),
                    allow_portal=bool(allow_intra_map_portal),
                ):
                    errors.append(f"MapID {stop.map_id}：P{source}→P{target} 不可达。")
        for source, target in zip(stops, stops[1:] + stops[:1]):
            route = self.find_map_route(source.map_id, target.map_id)
            if not route:
                errors.append(
                    f"没有由普通门或已确认脚本门组成的路线："
                    f"{source.map_id}→{target.map_id}。"
                )
        if len(rest_stops) == 1:
            rest_map = rest_stops[0].map_id
            for stop in stops:
                if stop.map_id == rest_map:
                    continue
                for source, target in ((stop.map_id, rest_map), (rest_map, stop.map_id)):
                    if not self.find_map_route(source, target):
                        errors.append(f"休息往返地图路线不可用：{source}→{target}。")
        return errors


class WorldPatrolPhase(Enum):
    IDLE = "idle"
    FARM = "farm"
    REST_FARM = "rest_farm"
    REST_RETURN_PLATFORM = "rest_return_platform"
    GO_EXIT = "go_exit"
    ENTER_PORTAL = "enter_portal"
    WAIT_MAP_CHANGE = "wait_map_change"
    ARRIVAL_CLEAR = "arrival_clear"
    VISUAL_TRANSIT = "visual_transit"
    RELOCALIZE = "relocalize"
    BLOCKED = "blocked"


class WorldPatrolController:
    """在 CombatFSM 外层编排地图目标，复用其平台巡逻执行器。"""

    def __init__(
        self,
        *,
        planner: WorldRoutePlanner,
        motion: Any,
        driver: Any,
        stop_event: Any,
        map_id_getter: Callable[[], Optional[int]],
        graph_getter: Callable[[], Optional[PlatformGraph]],
        platform_getter: Callable[[], Any],
        position_getter: Callable[[], Optional[Tuple[float, float]]],
        allow_run_jump_getter: Callable[[], bool],
        prepare_transition: Callable[[int], None],
        log_callback: Callable[[str], None],
        allow_intra_map_portal_getter: Optional[Callable[[], bool]] = None,
        recovery_completed_callback: Optional[Callable[[], None]] = None,
        recovery_state_callback: Optional[Callable[[bool, str], None]] = None,
        arrival_reset_callback: Optional[Callable[[], None]] = None,
        arrival_visual_getter: Optional[
            Callable[[], Optional[Tuple[float, float, float]]]
        ] = None,
        visual_frame_size_getter: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
        arm_known_portal_ocr: Optional[Callable[[VisiblePortalEdge], None]] = None,
        begin_rest_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        unexpected_map_confirm_sec: float = 0.8,
        unexpected_map_callback: Optional[Callable[[], None]] = None,
    ):
        self.planner = planner
        self.motion = motion
        self.driver = driver
        self.stop_event = stop_event
        self.map_id_getter = map_id_getter
        self.graph_getter = graph_getter
        self.platform_getter = platform_getter
        self.position_getter = position_getter
        self.allow_run_jump_getter = allow_run_jump_getter
        self.allow_intra_map_portal_getter = allow_intra_map_portal_getter
        self.prepare_transition = prepare_transition
        self.log = log_callback
        self.recovery_completed_callback = recovery_completed_callback
        self.recovery_state_callback = recovery_state_callback
        self.arrival_reset_callback = arrival_reset_callback
        self.arrival_visual_getter = arrival_visual_getter
        self.visual_frame_size_getter = visual_frame_size_getter
        self.arm_known_portal_ocr = arm_known_portal_ocr
        self.begin_rest_callback = begin_rest_callback
        self.unexpected_map_confirm_sec = max(0.0, float(unexpected_map_confirm_sec))
        self.unexpected_map_callback = unexpected_map_callback
        self.enabled = False
        self.running = False
        # 单地图 F6 也需要地图级守护。正常停留在 home_map 时本控制器
        # 完全让行；一旦 OCR 稳定确认进入其它地图，就复用同一套 type=1/2
        # 多跳规划与传送门执行器返回 home_map。
        self.recovery_armed = False
        self.recovery_active = False
        self._recovery_home_stop: Optional[WorldPatrolStop] = None
        self._recovery_ui_active = False
        self._recovery_ui_detail = ""
        self._unexpected_map_id: Optional[int] = None
        self._unexpected_map_since = 0.0
        self.stops: Tuple[WorldPatrolStop, ...] = ()
        self.phase = WorldPatrolPhase.IDLE
        self.stop_index = 0
        self._farm_completed: List[int] = []
        self._map_route: List[int] = []
        self._map_route_index = 0
        self._destination_stop_index = 0
        self._exit: Optional[VisiblePortalEdge] = None
        self._portal_attempts = 0
        self._wait_started_at = 0.0
        # 新地图入口光圈会暂时遮挡角色特征。此期间保持跨图控制器独占
        # 输入，先离开入口，再丢弃旧视觉轨迹并等待真实角色重新定位。
        self._arrival_edge: Optional[VisiblePortalEdge] = None
        self._arrival_started_at = 0.0
        self._arrival_portal_x: Optional[float] = None
        self._arrival_reset_at = 0.0
        self._arrival_last_visual_ts = 0.0
        self._arrival_last_visual_pos: Optional[Tuple[float, float]] = None
        self._arrival_stable_samples = 0
        self._visual_transit_anchor: Optional[Tuple[float, float, float, float]] = None
        self._visual_transit_entry_portal: Optional[VisiblePortalEdge] = None
        self._visual_transit_started_at = 0.0
        self._visual_transit_last_sample: Optional[Tuple[float, float, float]] = None
        self._visual_transit_stable_samples = 0
        self._visual_transit_stage = ""
        self._visual_transit_attempts = 0
        self._visual_transit_last_jump_log_at = 0.0
        self._visual_transit_viewport_anchor = False
        self._visual_transit_waypoints: Optional[Tuple[float, float, float]] = None
        self._visual_transit_settle_observed_at = 0.0
        self._visual_transit_missing_since = 0.0
        self._visual_recovery_probe_index = 0
        self._visual_recovery_round = 0
        self._visual_recovery_next_at = 0.0
        self._visual_recovery_observed_at = 0.0
        self._visual_recovery_stable_samples = 0
        self._visual_recovery_last_valid_pos: Optional[Tuple[float, float]] = None
        self._visual_recovery_blocked = False
        self._visual_recovery_goal_y: Optional[float] = None
        self._last_phase_log: Optional[Tuple[WorldPatrolPhase, str]] = None
        self._rest_stop_index: Optional[int] = None
        self._rest_interval_sec = 0.0
        self._rest_due_at = 0.0
        self._f6_started_at = 0.0
        self._rest_stage = "idle"
        self._rest_resume: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

    def configure(self, enabled: bool, stops: Sequence[WorldPatrolStop]) -> None:
        with self._lock:
            self._set_recovery_ui_state(False, "")
            self.enabled = bool(enabled)
            self.stops = tuple(stops)
            self._rest_stop_index = None
            self._rest_due_at = 0.0
            self._f6_started_at = 0.0
            self._rest_stage = "idle"
            self._rest_resume = None
            self.recovery_armed = False
            self.recovery_active = False
            self._recovery_home_stop = None
            self._clear_unexpected_map_candidate()
            if not self.enabled:
                self.running = False
                self.phase = WorldPatrolPhase.IDLE

    def arm_home_recovery(self, home_stop: WorldPatrolStop) -> None:
        """为单地图 F6 设置异常换图后的自动返程目标。"""
        with self._lock:
            self.enabled = False
            self.running = False
            self.stops = (home_stop,)
            self.stop_index = 0
            self._destination_stop_index = 0
            self.recovery_armed = True
            self.recovery_active = False
            self._recovery_home_stop = home_stop
            self._clear_unexpected_map_candidate()
            self._set_phase(
                WorldPatrolPhase.IDLE,
                f"异常换图恢复待命，目标 MapID {home_stop.map_id}",
            )

    def begin_death_recovery(self, actual_map: int, death_map: Optional[int]) -> bool:
        """Resume the armed F6 route from a respawn town, not from a new home.

        CombatFSM is stopped while the death modal is handled, but this
        controller deliberately keeps its pre-death stop list and index.
        """
        with self._lock:
            self.motion.stop()
            self.driver.release_all_keys()
            self._clear_unexpected_map_candidate()
            if self.recovery_armed and self._recovery_home_stop is not None:
                if death_map is not None and self._recovery_home_stop.map_id != int(death_map):
                    self.log("⚠️ [死亡返程] 单地图恢复目标与死亡地图不一致，已停键")
                    return False
                self._start_home_recovery(int(actual_map))
                return self._routing_active() and self.phase != WorldPatrolPhase.BLOCKED
            if not self.enabled or not self.stops:
                self.log("⚠️ [死亡返程] 没有已保存的跨图巡逻目标，已停键")
                return False
            target_index = min(max(0, self.stop_index), len(self.stops) - 1)
            self.running = True
            self._farm_completed = []
            self._portal_attempts = 0
            self._set_recovery_ui_state(
                True,
                f"死亡复活后从 MapID {actual_map} 返回巡逻地图 "
                f"{self.stops[target_index].map_id}",
            )
            self.log(
                f"🏠 [死亡返程] MapID {death_map} 死亡后进入 {actual_map}，"
                f"恢复原巡逻目标 MapID {self.stops[target_index].map_id}"
            )
            self._begin_travel(target_index)
            # Do not restart CombatFSM if even a map route cannot be planned.
            return self._routing_active() and self.phase != WorldPatrolPhase.BLOCKED

    def _clear_unexpected_map_candidate(self) -> None:
        self._unexpected_map_id = None
        self._unexpected_map_since = 0.0

    def _notify_unexpected_map(self) -> None:
        if self.unexpected_map_callback is not None:
            try:
                self.unexpected_map_callback()
            except Exception:
                pass

    @property
    def recovery_ui_active(self) -> bool:
        with self._lock:
            return bool(self._recovery_ui_active)

    def _set_recovery_ui_state(self, active: bool, detail: str) -> None:
        active = bool(active)
        detail = str(detail or "")
        changed = (
            active != self._recovery_ui_active
            or detail != self._recovery_ui_detail
        )
        self._recovery_ui_active = active
        self._recovery_ui_detail = detail
        if changed and self.recovery_state_callback is not None:
            try:
                self.recovery_state_callback(active, detail)
            except Exception:
                # UI 通知失败不能打断跨地图返程。
                pass

    def _routing_active(self) -> bool:
        return bool(self.running and (self.enabled or self.recovery_active))

    def rest_deadline_due(self) -> bool:
        """只读地检查全局休息是否到点，供正在进行的地面走位及时让行。"""
        with self._lock:
            return bool(
                self._routing_active()
                and self._rest_stop_index is not None
                and self._rest_stage in ("idle", "pending")
                and self.phase in (WorldPatrolPhase.FARM, WorldPatrolPhase.GO_EXIT)
                and time.perf_counter() >= self._rest_due_at
            )

    def _unexpected_map_stable(self, actual_map: int) -> bool:
        """地图名 OCR 偶发误识别不能立刻驱动人物跨图。"""
        now = time.perf_counter()
        actual_map = int(actual_map)
        if self._unexpected_map_id != actual_map:
            self._unexpected_map_id = actual_map
            self._unexpected_map_since = now
            return self.unexpected_map_confirm_sec <= 0.0
        return now - self._unexpected_map_since >= self.unexpected_map_confirm_sec

    def _start_home_recovery(self, actual_map: int) -> bool:
        home = self._recovery_home_stop
        if home is None:
            return False
        self.running = True
        self.recovery_active = True
        self.stop_index = 0
        self._destination_stop_index = 0
        self._portal_attempts = 0
        self._farm_completed = []
        self.motion.stop()
        self.driver.release_all_keys()
        self._set_recovery_ui_state(
            True, f"从 MapID {actual_map} 返回 {home.map_id}"
        )
        self.log(
            f"🏠 [异常回城恢复] 检测到 F6 从 MapID {home.map_id} "
            f"异常进入 {actual_map}，停止战斗与移动并规划返程"
        )
        if not self._begin_travel(0):
            # BLOCKED 状态继续独占输入，避免恢复失败后旧巡逻状态机在
            # 错误地图上误按键。用户停止 F6 或补齐地图数据后再处理。
            self.running = True
            self.recovery_active = True
            return False
        return True

    def _set_phase(self, phase: WorldPatrolPhase, detail: str = "") -> None:
        self.phase = phase
        marker = (phase, detail)
        if marker != self._last_phase_log:
            self._last_phase_log = marker
            self.log(f"🌐 [跨图状态] {phase.value}" + (f"：{detail}" if detail else ""))

    def start(
        self, initial_route: Optional[Sequence[int]] = None,
        f6_started_at: Optional[float] = None,
    ) -> Tuple[bool, str]:
        with self._lock:
            if not self.enabled:
                self.running = False
                return True, "单地图模式"
            if len(self.stops) < 2:
                return False, "跨地图目标不足两个"
            rest_indices = [
                index for index, stop in enumerate(self.stops)
                if stop.rest_platform_id is not None
            ]
            if len(rest_indices) > 1:
                return False, "多地图巡逻同时只能配置一个休息点"
            now = time.perf_counter()
            self._f6_started_at = (
                min(now, max(0.0, float(f6_started_at)))
                if f6_started_at is not None else now
            )
            current_map = self.map_id_getter()
            current_platform = self.platform_getter()
            if current_map is None:
                return False, "当前地图尚未定位"
            self.running = True
            self._portal_attempts = 0
            matching = [i for i, stop in enumerate(self.stops) if stop.map_id == int(current_map)]
            if matching:
                self.stop_index = matching[0]
                self._enter_farm(self.stop_index)
                self._arm_global_rest(rest_indices[0] if rest_indices else None, initial=True)
                if current_platform is None:
                    return True, f"从 MapID {current_map} 启动；等待角色脱离绳梯并落台后刷怪"
                return True, f"从 MapID {current_map} 开始刷怪"
            self.stop_index = 0
            if not self._begin_travel(self.stop_index, route_override=initial_route):
                self.running = False
                return False, f"当前地图 {current_map} 无法到达首个目标地图 {self.stops[0].map_id}"
            self._arm_global_rest(rest_indices[0] if rest_indices else None, initial=True)
            return True, f"前往首个目标地图 {self.stops[0].map_id}"

    def stop(self) -> None:
        with self._lock:
            self._set_recovery_ui_state(False, "")
            self.running = False
            self._rest_stop_index = None
            self._rest_due_at = 0.0
            self._f6_started_at = 0.0
            self._rest_stage = "idle"
            self._rest_resume = None
            self.recovery_armed = False
            self.recovery_active = False
            self._recovery_home_stop = None
            self._clear_unexpected_map_candidate()
            self._exit = None
            self._clear_arrival_state()
            self._clear_visual_transit_state()
            self._map_route = []
            self._set_phase(WorldPatrolPhase.IDLE, "停止")

    def _arm_global_rest(self, stop_index: Optional[int], *, initial: bool = False) -> None:
        """首轮以按下 F6 为零点；后续轮次以椅子休息结束为零点。"""
        self._rest_stop_index = stop_index
        self._rest_stage = "idle"
        self._rest_resume = None
        if stop_index is None:
            self._rest_due_at = 0.0
            return
        self._schedule_next_rest_deadline(
            self._f6_started_at if initial else time.perf_counter(),
            "F6启动" if initial else "上次休息结束",
        )

    def _schedule_next_rest_deadline(self, anchor: float, source: str) -> None:
        """只更新下一次截止时间，不改当前返程状态。"""
        stop_index = self._rest_stop_index
        if stop_index is None:
            return
        stop = self.stops[stop_index]
        low = max(1.0, float(
            stop.rest_interval_min_sec
            if stop.rest_interval_min_sec is not None else 30.0
        ))
        high = max(low, float(
            stop.rest_interval_max_sec
            if stop.rest_interval_max_sec is not None else 40.0
        ))
        self._rest_interval_sec = random.uniform(low, high) if high > low else low
        self._rest_due_at = anchor + self._rest_interval_sec
        self.log(
            f"⏱️ [全局休息计时] {source}起计{self._rest_interval_sec:.2f}s，"
            f"到点前往 MapID {stop.map_id} P{stop.rest_platform_id}"
        )

    def _enter_rest_farm(self) -> None:
        index = self._rest_stop_index
        if index is None or self.begin_rest_callback is None:
            self._set_phase(WorldPatrolPhase.BLOCKED, "全局休息派发器不可用")
            return
        stop = self.stops[index]
        self.stop_index = index
        self._map_route = []
        self._exit = None
        self._rest_stage = "resting"
        self._set_phase(
            WorldPatrolPhase.REST_FARM,
            f"MapID {stop.map_id} P{stop.rest_platform_id}，战斗让行",
        )
        self.begin_rest_callback({
            "map_id": stop.map_id,
            "platform_id": stop.rest_platform_id,
            "duration_min_sec": float(
                stop.rest_duration_min_sec
                if stop.rest_duration_min_sec is not None else 30.0
            ),
            "duration_max_sec": float(
                stop.rest_duration_max_sec
                if stop.rest_duration_max_sec is not None else 60.0
            ),
            "interval_min_sec": 1.0,
            "interval_max_sec": 1.0,
        })

    def _begin_rest_detour(self, current_map: int, platform: Any) -> bool:
        index = self._rest_stop_index
        if index is None:
            return False
        self.motion.stop()
        self.driver.release_all_keys()
        self._rest_resume = {
            "map_id": int(current_map),
            "platform_id": int(platform.id),
            "stop_index": int(self.stop_index),
            "phase": self.phase,
            "destination_index": int(self._destination_stop_index),
            "farm_completed": list(self._farm_completed),
        }
        self._rest_stage = "to_rest_map"
        stop = self.stops[index]
        self.log(
            f"🪑 [全局休息到点] 当前MapID {current_map} P{platform.id}，"
            f"立即前往 MapID {stop.map_id} P{stop.rest_platform_id}"
        )
        if int(current_map) == stop.map_id:
            self._enter_rest_farm()
            return True
        if not self._begin_travel(index):
            self._set_phase(WorldPatrolPhase.BLOCKED, "到休息地图的路线不可用，已停键")
            return False
        return True

    def on_rest_completed(self, platform_id: int) -> None:
        """战斗状态机已完成坐椅等待；返程在下一次地图 tick 中进行。"""
        with self._lock:
            if self._rest_stage != "resting":
                return
            self._schedule_next_rest_deadline(time.perf_counter(), "上次休息结束")
            self._rest_stage = "rest_complete"
            self.log(f"✅ [全局休息完成] P{platform_id}，准备返回原地图和平台")

    def _enter_rest_return_platform(self) -> None:
        resume = self._rest_resume
        if resume is None:
            self._set_phase(WorldPatrolPhase.BLOCKED, "休息返程目标丢失")
            return
        self.stop_index = int(resume["stop_index"])
        self._map_route = []
        self._exit = None
        self._rest_stage = "returning_platform"
        self._set_phase(
            WorldPatrolPhase.REST_RETURN_PLATFORM,
            f"MapID {resume['map_id']} 返回P{resume['platform_id']}",
        )

    def _start_rest_return(self) -> None:
        resume = self._rest_resume
        if resume is None:
            self._set_phase(WorldPatrolPhase.BLOCKED, "休息返程状态丢失")
            return
        self._rest_stage = "returning_map"
        if not self._begin_travel(int(resume["stop_index"])):
            self._set_phase(WorldPatrolPhase.BLOCKED, "休息返程地图路线不可用，已停键")

    def _complete_rest_return(self) -> None:
        resume = self._rest_resume
        if resume is None:
            return
        prior_phase = resume["phase"]
        self.stop_index = int(resume["stop_index"])
        self._farm_completed = list(resume["farm_completed"])
        # 下一轮截止时间在椅子休息结束时已经生成；返程耗时也计入间隔。
        self._rest_stage = "idle"
        self._rest_resume = None
        self.log(
            f"↩️ [全局休息返程完成] MapID {resume['map_id']} "
            f"P{resume['platform_id']}，恢复休息前路线"
        )
        if prior_phase == WorldPatrolPhase.GO_EXIT:
            self._begin_travel(int(resume["destination_index"]))
        else:
            self._set_phase(
                WorldPatrolPhase.FARM,
                f"MapID {resume['map_id']} 平台 {list(self.stops[self.stop_index].platforms)}",
            )

    def status_text(self) -> str:
        with self._lock:
            if self.recovery_active and self._recovery_home_stop is not None:
                if self._exit is not None:
                    return (
                        f"异常回城恢复：{self._exit.source_map_id}→"
                        f"{self._exit.target_map_id}（目标{self._recovery_home_stop.map_id}）"
                    )
                return f"异常回城恢复中：返回 MapID {self._recovery_home_stop.map_id}"
            if self.recovery_armed and self._recovery_home_stop is not None:
                return f"异常回城恢复待命：MapID {self._recovery_home_stop.map_id}"
            if not self.enabled:
                return "跨地图巡逻未启用"
            if not self.running:
                return f"跨地图路线已配置（{len(self.stops)}个刷怪地图），等待F6"
            if self.phase == WorldPatrolPhase.FARM and self.stops:
                stop = self.stops[self.stop_index]
                return f"刷怪 MapID {stop.map_id}：{','.join('P' + str(x) for x in stop.platforms)}"
            if self._exit is not None:
                return (
                    f"{self.phase.value}：{self._exit.source_map_id}→"
                    f"{self._exit.target_map_id}（{self._exit.portal_name}）"
                )
            return self.phase.value

    def current_patrol(self, fallback: Sequence[int]) -> List[int]:
        with self._lock:
            if not self._routing_active():
                return list(fallback)
            if self.phase == WorldPatrolPhase.REST_RETURN_PLATFORM and self._rest_resume:
                return [int(self._rest_resume["platform_id"])]
            if self.phase == WorldPatrolPhase.FARM:
                return list(self.stops[self.stop_index].platforms)
            if self.phase == WorldPatrolPhase.GO_EXIT and self._exit is not None:
                return [self._exit.source_platform_id]
            return []

    def current_dwell_range(self, fallback: Tuple[float, float]) -> Tuple[float, float]:
        with self._lock:
            if not self._routing_active():
                return fallback
            if self.phase != WorldPatrolPhase.FARM:
                return (0.0, 0.0)
            stop = self.stops[self.stop_index]
            if stop.dwell_min_sec is None or stop.dwell_max_sec is None:
                return fallback
            return (float(stop.dwell_min_sec), float(stop.dwell_max_sec))

    def current_positions(self, fallback: Sequence[float]) -> List[float]:
        with self._lock:
            if not self._routing_active():
                return list(fallback)
            if self.phase != WorldPatrolPhase.FARM:
                return [0.5]
            stop = self.stops[self.stop_index]
            return list(stop.positions) if stop.positions else list(fallback)

    def current_position_overrides(self) -> Dict[int, Tuple[float, ...]]:
        with self._lock:
            if not self._routing_active() or self.phase != WorldPatrolPhase.FARM:
                return {}
            return dict(self.stops[self.stop_index].position_overrides)

    def current_position_random(self, fallback: float) -> float:
        with self._lock:
            if not self._routing_active():
                return float(fallback)
            if self.phase != WorldPatrolPhase.FARM:
                return 0.0
            value = self.stops[self.stop_index].position_random
            return float(fallback) if value is None else float(value)

    def current_rest_settings(
        self, fallback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._routing_active():
                return fallback
            # 跨地图时只有 WorldPatrolController 的 F6 全局时钟能派发
            # 休息；CombatFSM 的旧地图本地计时必须完全停用。
            return None

    def on_target_completed(self, platform_id: int) -> None:
        with self._lock:
            if not self._routing_active():
                return
            platform_id = int(platform_id)
            if self.phase == WorldPatrolPhase.REST_RETURN_PLATFORM:
                if self._rest_resume and platform_id == int(self._rest_resume["platform_id"]):
                    self._complete_rest_return()
                return
            if self.phase == WorldPatrolPhase.GO_EXIT:
                if self._exit is not None and platform_id == self._exit.source_platform_id:
                    self._set_phase(
                        WorldPatrolPhase.ENTER_PORTAL,
                        f"MapID {self._exit.source_map_id} 门 {self._exit.portal_name}",
                    )
                return
            if self.phase != WorldPatrolPhase.FARM:
                return
            if platform_id not in self._farm_completed:
                self._farm_completed.append(platform_id)
            required = set(self.stops[self.stop_index].platforms)
            if required.issubset(self._farm_completed):
                next_index = (self.stop_index + 1) % len(self.stops)
                self.log(
                    f"🌐 [目标地图完成] MapID {self.stops[self.stop_index].map_id} "
                    f"已完成平台 {self._farm_completed}，前往 {self.stops[next_index].map_id}"
                )
                self._begin_travel(next_index)

    def _enter_farm(self, stop_index: int) -> None:
        if self._rest_stage == "to_rest_map" and stop_index == self._rest_stop_index:
            self._enter_rest_farm()
            return
        if self._rest_stage == "returning_map" and self._rest_resume is not None:
            if stop_index == int(self._rest_resume["stop_index"]):
                self._enter_rest_return_platform()
                return
        self.stop_index = int(stop_index)
        self._farm_completed = []
        self._map_route = []
        self._map_route_index = 0
        self._exit = None
        stop = self.stops[self.stop_index]
        if self.recovery_active:
            self.recovery_active = False
            self.running = False
            self._clear_unexpected_map_candidate()
            self._set_recovery_ui_state(False, "")
            self._set_phase(
                WorldPatrolPhase.IDLE,
                f"已返回 MapID {stop.map_id}，交还原 F6 巡逻",
            )
            self.log(
                f"✅ [异常回城恢复] 已回到 MapID {stop.map_id} 并完成落台定位，"
                "恢复原平台巡逻与战斗"
            )
            if self.recovery_completed_callback is not None:
                try:
                    self.recovery_completed_callback()
                except Exception as exc:
                    self.log(f"⚠️ [异常回城恢复] 恢复原巡逻状态失败：{exc}")
            return
        self._set_phase(
            WorldPatrolPhase.FARM,
            f"MapID {stop.map_id} 平台 {list(stop.platforms)}",
        )

    def _begin_travel(
        self, destination_stop_index: int,
        route_override: Optional[Sequence[int]] = None,
    ) -> bool:
        current_map = self.map_id_getter()
        if current_map is None:
            self._set_phase(WorldPatrolPhase.BLOCKED, "当前 MapID 未识别")
            return False
        destination = self.stops[destination_stop_index]
        if int(current_map) == destination.map_id:
            self._enter_farm(destination_stop_index)
            return True
        route = (
            list(route_override) if route_override is not None else
            self.planner.find_map_route(int(current_map), destination.map_id)
        )
        if route and (route[0] != int(current_map) or route[-1] != destination.map_id):
            route = []
        if len(route) < 2:
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                f"无已支持的传送门路线 {current_map}→{destination.map_id}",
            )
            return False
        self._destination_stop_index = int(destination_stop_index)
        self._map_route = route
        self._map_route_index = 0
        self.log("🌐 [跨图路线] " + " → ".join(str(value) for value in route))
        return self._prepare_current_exit()

    def _prepare_current_exit(self) -> bool:
        current_map = self.map_id_getter()
        current_platform = self.platform_getter()
        graph = self.graph_getter()
        # 售票处没有小地图，无法产生承重平台；从售票处直接按 F6
        # 时也要进入主画面导航，不能永远等待黄点定位。
        if (
            current_map is not None and graph is not None
            and int(getattr(graph, "map_id", -1)) == int(current_map)
            and current_platform is None
            and self._begin_visual_transit(graph, None)
        ):
            return True
        if (
            current_map is not None and graph is not None
            and int(getattr(graph, "map_id", -1)) == int(current_map)
            and current_platform is None
            and int((graph.minimap_meta or {}).get("canvasWidth", 0) or 0) == 0
        ):
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                f"MapID {current_map} 无小地图，且没有可验证的地形导航路线；已停键",
            )
            return True
        if current_map is None or current_platform is None or graph is None:
            self._set_phase(WorldPatrolPhase.RELOCALIZE, "等待地图拓扑与承重平台")
            return True
        if self._map_route_index + 1 >= len(self._map_route):
            self._enter_farm(self._destination_stop_index)
            return True
        expected_source = self._map_route[self._map_route_index]
        next_map = self._map_route[self._map_route_index + 1]
        if int(current_map) != expected_source:
            return self._replan_from_actual_map(int(current_map))
        exit_edge, local_route = self.planner.choose_exit(
            expected_source,
            next_map,
            int(current_platform.id),
            allow_run_jump=bool(self.allow_run_jump_getter()),
            allow_intra_map_portal=(
                bool(self.allow_intra_map_portal_getter())
                if self.allow_intra_map_portal_getter is not None else True
            ),
        )
        if exit_edge is None:
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                f"MapID {expected_source} 当前P{current_platform.id}无法到达通往{next_map}的普通门",
            )
            return False
        self._exit = exit_edge
        self._portal_attempts = 0
        if int(current_platform.id) == exit_edge.source_platform_id:
            # 当前已经站在出口所在长平台时，不再把 [P1] 交回平台巡逻器。
            # 单平台站位会把角色拉向 50% 附近，恰好抵消前一次朝远端
            # portal 的移动，形成“走半程 -> 超时 -> 回中心”的死循环。
            self._set_phase(
                WorldPatrolPhase.ENTER_PORTAL,
                f"已在P{exit_edge.source_platform_id}，直接对齐 "
                f"{exit_edge.portal_name}@X={exit_edge.x} → {next_map}",
            )
        else:
            self._set_phase(
                WorldPatrolPhase.GO_EXIT,
                f"MapID {expected_source} P{exit_edge.source_platform_id} "
                f"{exit_edge.portal_name}@X={exit_edge.x} → {next_map}，"
                f"地图内{len(local_route)}步",
            )
        return True

    def _replan_from_actual_map(self, actual_map: int) -> bool:
        destination = self.stops[self._destination_stop_index]
        route = self.planner.find_map_route(actual_map, destination.map_id)
        if not route:
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                f"意外进入{actual_map}且无法重规划到{destination.map_id}",
            )
            return False
        self._map_route = route
        self._map_route_index = 0
        self._exit = None
        self._clear_arrival_state()
        self._clear_visual_transit_state()
        self._set_phase(WorldPatrolPhase.RELOCALIZE, f"从实际地图{actual_map}重新规划")
        return True

    def _clear_arrival_state(self) -> None:
        self._arrival_edge = None
        self._arrival_started_at = 0.0
        self._arrival_portal_x = None
        self._arrival_reset_at = 0.0
        self._arrival_last_visual_ts = 0.0
        self._arrival_last_visual_pos = None
        self._arrival_stable_samples = 0

    def _clear_visual_transit_state(self) -> None:
        self._visual_transit_anchor = None
        self._visual_transit_entry_portal = None
        self._visual_transit_entry_world = None
        self._visual_transit_started_at = 0.0
        self._visual_transit_last_sample = None
        self._visual_transit_stable_samples = 0
        self._visual_transit_stage = ""
        self._visual_transit_attempts = 0
        self._visual_transit_settle_y = None
        self._visual_transit_last_jump_log_at = 0.0
        self._visual_transit_viewport_anchor = False
        self._visual_transit_waypoints = None
        self._visual_transit_settle_observed_at = 0.0
        self._visual_transit_missing_since = 0.0
        self._visual_recovery_probe_index = 0
        self._visual_recovery_round = 0
        self._visual_recovery_next_at = 0.0
        self._visual_recovery_observed_at = 0.0
        self._visual_recovery_stable_samples = 0
        self._visual_recovery_last_valid_pos = None
        self._visual_recovery_blocked = False
        self._visual_recovery_goal_y = None

    @staticmethod
    def _ticket_booth_descent_waypoints(
        graph: PlatformGraph,
    ) -> Optional[Tuple[float, float, float]]:
        """从实际 foothold 生成售票处绕墙下楼的两个转向点。

        上层入口 fh5 右侧的 fh6 是竖直墙，不能朝地铁门 X=200
        直走。先向左离开 fh3 并落在 fh2，再向右越过 fh14
        落到底层 fh16。若 WZ 地形改变，拒绝使用旧路线。
        """
        by_foothold = {
            int(foothold_id): node
            for node in graph.nodes.values()
            for foothold_id in getattr(node, "foothold_ids", ())
        }
        upper_step = by_foothold.get(3)
        upper_landing = by_foothold.get(2)
        lower_last = by_foothold.get(14)
        floor = by_foothold.get(16)
        if any(node is None for node in (upper_step, upper_landing, lower_last, floor)):
            return None
        upper_x = float(upper_step.x_min) - 18.0
        lower_x = float(lower_last.x_max) + 35.0
        if not (
            upper_landing.x_min + 8 <= upper_x <= upper_landing.x_max - 20
            and upper_x <= upper_step.x_min - 12
            and floor.x_min + 20 <= lower_x <= floor.x_max - 20
            and lower_x >= lower_last.x_max + 25
            and upper_landing.y < lower_last.y < floor.y
        ):
            return None
        return upper_x, lower_x, float(upper_landing.y)

    def _reset_arrival_visuals(self) -> None:
        callback = self.arrival_reset_callback
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                self.log(f"⚠️ [换图人物重定位] 清理旧视觉轨迹失败：{exc}")

    def _begin_arrival_clear(self, edge: VisiblePortalEdge, current_map: int) -> None:
        """换图后先离开入口光圈；老调用方未提供回调时保持兼容。"""
        self._clear_visual_transit_state()
        if self.arrival_reset_callback is None and self.arrival_visual_getter is None:
            self.log(
                "⚠️ [换图战斗门禁未接入] 缺少人物视觉回调，"
                "仅执行旧版拓扑重定位"
            )
            self._set_phase(
                WorldPatrolPhase.RELOCALIZE,
                f"已进入 MapID {current_map}，等待拓扑和黄点",
            )
            return
        self._arrival_edge = edge
        self._arrival_started_at = time.perf_counter()
        self._arrival_portal_x = None
        self._arrival_reset_at = 0.0
        self._arrival_last_visual_ts = 0.0
        self._arrival_last_visual_pos = None
        self._arrival_stable_samples = 0
        # 先清掉上一张地图的角色/怪物框，黑屏及入口光圈期间即使后台
        # 尚未刷新，也不会把旧框交给战斗状态机。
        self.motion.stop()
        self.driver.release_all_keys()
        self._reset_arrival_visuals()
        self._set_phase(
            WorldPatrolPhase.ARRIVAL_CLEAR,
            f"已进入 MapID {current_map}，先脱离入口光圈并重新识别人",
        )

    @staticmethod
    def _find_arrival_portal_x(
        graph: PlatformGraph, edge: VisiblePortalEdge
    ) -> Optional[float]:
        wanted = str(edge.target_portal_name or "")
        matches = []
        for portal in getattr(graph, "portals", ()) or ():
            if not isinstance(portal, dict):
                continue
            name = str(portal.get("portalName", portal.get("pn", "")) or "")
            if wanted and name != wanted:
                continue
            try:
                x = float(portal.get("x", 0))
                to_map = int(portal.get("toMap", portal.get("tm", 999999999)))
            except (TypeError, ValueError):
                continue
            # toName 正常应精确命中；缺失时优先选返回来源地图的入口。
            score = 0 if wanted and name == wanted else 1
            if to_map == edge.source_map_id:
                score -= 1
            matches.append((score, x))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        return matches[0][1]

    def _begin_visual_transit(
        self, graph: PlatformGraph, arrival: Optional[VisiblePortalEdge]
    ) -> bool:
        """无小地图时，只在已确认的短图/脚本门上启用主画面位移闭环。"""
        if int(graph.map_id) != 103000100 or not self._map_route:
            return False
        if self._map_route_index + 1 >= len(self._map_route):
            return False
        next_map = self._map_route[self._map_route_index + 1]
        exits = [
            edge for edge in self.planner._portal_edges_for_map(graph.map_id)
            if edge.target_map_id == next_map and edge.portal_type == 7
        ]
        if not exits:
            return False
        entry = next((
            portal for portal in graph.portals
            if arrival is not None
            and str(portal.get("portalName", "")) == arrival.target_portal_name
        ), None)
        if (arrival is not None and entry is None) or self.arrival_visual_getter is None:
            self._set_phase(WorldPatrolPhase.BLOCKED, "无小地图过渡图缺少入口或人物视觉锚点")
            return True
        try:
            frame_size = self.visual_frame_size_getter() if self.visual_frame_size_getter else None
        except Exception:
            frame_size = None
        world_width = float((graph.vr_bounds or {}).get("width", 0) or 0)
        if frame_size is None or world_width <= 0 or frame_size[0] + 8 < world_width:
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                f"售票处无小地图，当前画面宽度不足以覆盖{world_width:.0f}px地图；禁止盲走",
            )
            return True
        bounds = graph.vr_bounds or {}
        world_height = float(bounds.get("height", 0) or 0)
        if arrival is None and (
            world_height <= 0
            or frame_size[0] + 8 < world_width
            or frame_size[1] + 8 < world_height
        ):
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                "售票处从当前画面启动时，视口不足以覆盖整张地图，无法安全标定人物位置",
            )
            return True
        waypoints = self._ticket_booth_descent_waypoints(graph)
        if waypoints is None or waypoints[1] >= exits[0].x - 50:
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                "售票处台阶地形与已验证路线不符，停止输入，禁止直线撞墙",
            )
            return True
        self._clear_visual_transit_state()
        self._visual_transit_waypoints = waypoints
        self._visual_transit_entry_portal = arrival
        if entry is not None:
            self._visual_transit_entry_world = (float(entry["x"]), float(entry["y"]))
        else:
            # 售票处完整画面固定在视口中央，无镜头卷动；实机
            # 1280x720 客户区中的 800x600 世界位于 x≈240..1040，
            # 因此画面左上角不是 WZ 的 vrLeft/vrTop。
            view_left = (float(frame_size[0]) - world_width) / 2.0
            view_top = (float(frame_size[1]) - world_height) / 2.0
            self._visual_transit_anchor = (
                view_left, view_top, float(bounds["left"]), float(bounds["top"])
            )
            self._visual_transit_viewport_anchor = True
        self._visual_transit_started_at = time.perf_counter()
        self._visual_transit_stage = "anchor"
        self._exit = exits[0]
        self._portal_attempts = 0
        self.motion.stop()
        self.driver.release_all_keys()
        if arrival is None:
            self.log(
                f"🧭 [无小地图过渡] 从售票处直接启动F6，"
                f"视口{frame_size[0]}x{frame_size[1]}，"
                f"地图画面偏移=({view_left:.1f},{view_top:.1f})；"
                f"台阶路径=左至{waypoints[0]:.0f}、右至{waypoints[1]:.0f}、"
                f"再进{self._exit.portal_name}@X={self._exit.x}"
            )
        else:
            self.log(
                f"🧭 [无小地图过渡] MapID {graph.map_id}，入口"
                f"{arrival.target_portal_name}@{self._visual_transit_entry_world}，"
                f"台阶路径=左至{waypoints[0]:.0f}、右至{waypoints[1]:.0f}，"
                f"脚本门{self._exit.portal_name}@X={self._exit.x}；等待人物视觉锚定"
            )
        self._set_phase(WorldPatrolPhase.VISUAL_TRANSIT, "售票处主画面定位")
        return True

    def _visual_transit_position(self) -> Optional[Tuple[float, float]]:
        anchor = self._visual_transit_anchor
        if anchor is None or self.arrival_visual_getter is None:
            return None
        edge = self._exit
        current_map = self.map_id_getter()
        if edge is None or current_map is None or int(current_map) != edge.source_map_id:
            self.motion.stop()
            return None
        observation = self.arrival_visual_getter()
        if observation is None:
            self.motion.stop()
            return None
        sx, sy, observed_at = map(float, observation)
        if time.perf_counter() - observed_at > 0.25:
            self.motion.stop()
            return None
        previous = self._visual_transit_last_sample
        if previous is not None and observed_at > previous[2]:
            dt = observed_at - previous[2]
            if abs(sx - previous[0]) > max(45.0, 350.0 * dt + 18.0):
                self.motion.stop()
                now = time.perf_counter()
                if now - self._visual_transit_last_jump_log_at >= 1.0:
                    self._visual_transit_last_jump_log_at = now
                    self.log("⚠️ [无小地图过渡] 人物视觉X突跳，停止输入等待重新定位")
                return None
        self._visual_transit_last_sample = (sx, sy, observed_at)
        anchor_sx, anchor_sy, anchor_wx, anchor_wy = anchor
        world_x = anchor_wx + sx - anchor_sx
        world_y = anchor_wy + sy - anchor_sy
        if not -430.0 <= world_x <= 430.0:
            self.motion.stop()
            return None
        return world_x, world_y

    def _ticket_booth_stage_for_y(self, world_y: float, door_y: float) -> str:
        waypoints = self._visual_transit_waypoints
        if waypoints is None:
            return "anchor"
        if world_y >= door_y - 35.0:
            return "portal"
        return "lower_descend" if world_y >= waypoints[2] - 7.0 else "upper_descend"

    def _visual_transit_walk_to(self, target_x: float, timeout_sec: float) -> bool:
        """分段闭环走位；任一段卡住立即松键，不沿直线顶墙。"""
        old_checker = getattr(self.motion, "priority_interrupt_checker", None)
        old_teleport = getattr(self.motion, "enable_teleport", False)
        try:
            self.motion.priority_interrupt_checker = None
            self.motion.enable_teleport = False
            return bool(self.motion.walk_to_x(
                target_x=int(round(target_x)),
                get_player_pos=self._visual_transit_position,
                tolerance=4,
                timeout_sec=timeout_sec,
                stop_event=self.stop_event,
                platform_bounds=(-399, 389),
                safe_margin=20,
                completion_checker=lambda: (
                    self.map_id_getter() is not None
                    and int(self.map_id_getter()) != 103000100
                ),
                observation_timeout_sec=0.45,
            ))
        finally:
            self.motion.priority_interrupt_checker = old_checker
            self.motion.enable_teleport = old_teleport
            self.motion.stop()

    def _begin_visual_recovery(self, detail: str, *, goal_y: Optional[float] = None) -> None:
        """视觉暂失或落地未确认时停键，进行有界短探测。"""
        self.motion.stop()
        self.driver.release_all_keys()
        self._visual_transit_stage = "recover_vision"
        self._visual_recovery_probe_index = 0
        self._visual_recovery_round = 0
        self._visual_recovery_next_at = time.perf_counter() + 0.15
        self._visual_recovery_observed_at = 0.0
        self._visual_recovery_stable_samples = 0
        self._visual_recovery_last_valid_pos = None
        self._visual_recovery_blocked = False
        self._visual_recovery_goal_y = goal_y
        self._visual_transit_missing_since = 0.0
        self._set_phase(WorldPatrolPhase.VISUAL_TRANSIT, f"售票处定位/落地未确认：{detail}，短探测找人")

    def _visual_recovery_down_jump_safe(self) -> bool:
        """只有最后可信脚点确在已知下跳通道内才试下跳。"""
        sample = self._visual_transit_last_sample
        anchor = self._visual_transit_anchor
        graph = self.graph_getter()
        if sample is None or anchor is None or graph is None:
            return False
        if time.perf_counter() - sample[2] > 3.0:
            return False
        world_x = anchor[2] + sample[0] - anchor[0]
        world_y = anchor[3] + sample[1] - anchor[1]
        for node in graph.nodes.values():
            if not (node.x_min + 8 <= world_x <= node.x_max - 8):
                continue
            if abs(float(node.surface_y_at(world_x)) - world_y) > 22.0:
                continue
            for edge in graph.get_edges_from(node.id):
                if edge.action != "DOWN_JUMP" or not edge.trigger_x_range:
                    continue
                lo, hi = sorted(map(float, edge.trigger_x_range))
                if lo + 5 <= world_x <= hi - 5:
                    return True
        return False

    def _visual_recovery_tick(self, edge: VisiblePortalEdge) -> bool:
        position = self._visual_transit_position()
        if position is not None and self._visual_transit_last_sample is not None:
            observed_at = self._visual_transit_last_sample[2]
            if observed_at > self._visual_recovery_observed_at:
                self._visual_recovery_observed_at = observed_at
                previous = self._visual_recovery_last_valid_pos
                self._visual_recovery_stable_samples = (
                    self._visual_recovery_stable_samples + 1
                    if previous is not None
                    and abs(position[0] - previous[0]) <= 8.0
                    and abs(position[1] - previous[1]) <= 6.0
                    else 1
                )
                self._visual_recovery_last_valid_pos = position
            goal_met = (
                self._visual_recovery_goal_y is None
                or position[1] >= self._visual_recovery_goal_y
            )
            if (self._visual_recovery_stable_samples >= 2 and goal_met
                    and time.perf_counter() >= self._visual_recovery_next_at):
                self._visual_transit_stage = self._ticket_booth_stage_for_y(position[1], edge.y)
                self._visual_recovery_blocked = False
                self._visual_recovery_goal_y = None
                self._set_phase(
                    WorldPatrolPhase.VISUAL_TRANSIT,
                    f"人物重新定位 X={position[0]:.0f},Y={position[1]:.0f}；从实际楼层续走",
                )
                return True
            # 仅能看到人物但Y仍未达到目标楼层，不等于已经落地；继续
            # 有界探测。达到目标高度后再等第二帧稳定观测，不盲按。
            if goal_met:
                return True
        else:
            self._visual_recovery_stable_samples = 0
            self._visual_recovery_last_valid_pos = None
        if self._visual_recovery_blocked:
            return True
        now = time.perf_counter()
        if now < self._visual_recovery_next_at:
            return True
        # 先垂直原地跳，再试有安全落点的下跳；避免最后一跳把刚下到
        # 下一层的角色又送回上层。
        probes = ("left", "right", "up_jump", "down_jump")
        if self._visual_recovery_probe_index >= len(probes):
            self._visual_recovery_round += 1
            if self._visual_recovery_round >= 2:
                self._visual_recovery_blocked = True
                self._set_phase(
                    WorldPatrolPhase.BLOCKED,
                    "售票处四方向短探测两轮仍未定位人物；已停键，持续等待视觉恢复",
                )
                return True
            self._visual_recovery_probe_index = 0
            self._visual_recovery_next_at = now + 0.5
            return True
        action = probes[self._visual_recovery_probe_index]
        self._visual_recovery_probe_index += 1
        if action == "down_jump" and not self._visual_recovery_down_jump_safe():
            self.log("⚠️ [售票处视觉找人] 最后脚点不在可靠下跳通道，跳过盲目下跳")
            self._visual_recovery_next_at = now + 0.15
            return True
        goal_text = (
            f"Y≥{self._visual_recovery_goal_y:.0f}"
            if self._visual_recovery_goal_y is not None else "重新识别人"
        )
        self.log(
            f"🔎 [售票处视觉找人] 第{self._visual_recovery_round + 1}轮"
            f"动作={action}，当前={position}，目标={goal_text}"
        )
        try:
            if action in ("left", "right"):
                self.driver.press_key(action, duration_ms=75)
            elif action == "down_jump":
                self.driver.key_down("down")
                time.sleep(0.09)
                self.driver.press_key(getattr(self.motion, "jump_key", "alt"), duration_ms=100)
            else:
                self.driver.press_key(getattr(self.motion, "jump_key", "alt"), duration_ms=100)
        finally:
            self.motion.stop()
            self.driver.release_all_keys()
        self._visual_recovery_next_at = time.perf_counter() + (
            0.55 if action in ("down_jump", "up_jump") else 0.2
        )
        return True

    def _visual_transit_settle_sample(self, target_min_y: float) -> bool:
        position = self._visual_transit_position()
        screen = self._visual_transit_last_sample
        if position is None or screen is None:
            return False
        if screen[2] > self._visual_transit_settle_observed_at:
            previous_y = getattr(self, "_visual_transit_settle_y", None)
            if previous_y is not None and abs(screen[1] - previous_y) <= 3.0:
                self._visual_transit_stable_samples += 1
            else:
                self._visual_transit_stable_samples = 1
            self._visual_transit_settle_y = screen[1]
            self._visual_transit_settle_observed_at = screen[2]
        return self._visual_transit_stable_samples >= 3 and position[1] >= target_min_y

    def _visual_transit_tick(self) -> bool:
        edge = self._exit
        if edge is None or not self.running or self.stop_event.is_set():
            return True
        current_map = self.map_id_getter()
        if current_map is None:
            self.motion.stop()
            return True
        if current_map is not None and int(current_map) != edge.source_map_id:
            self._wait_started_at = time.perf_counter()
            self._set_phase(WorldPatrolPhase.WAIT_MAP_CHANGE, "售票处脚本门已切图")
            return True
        now = time.perf_counter()
        if self._visual_transit_stage == "anchor":
            observation = self.arrival_visual_getter() if self.arrival_visual_getter else None
            if observation is not None:
                sx, sy, observed_at = map(float, observation)
                if observed_at > self._arrival_started_at + 0.05 and now - observed_at <= 0.25:
                    previous = self._visual_transit_last_sample
                    if previous is None or observed_at > previous[2]:
                        self._visual_transit_stable_samples = (
                            self._visual_transit_stable_samples + 1
                            if previous is not None
                            and abs(sx - previous[0]) <= 8.0
                            and abs(sy - previous[1]) <= 8.0 else 1
                        )
                        self._visual_transit_last_sample = (sx, sy, observed_at)
                        if self._visual_transit_stable_samples >= 3:
                            if self._visual_transit_viewport_anchor:
                                position = self._visual_transit_position()
                                if position is not None:
                                    self._visual_transit_stage = self._ticket_booth_stage_for_y(
                                        position[1], edge.y
                                    )
                                    self.log(
                                        f"✅ [无小地图过渡] 人物画面({sx:.1f},{sy:.1f})"
                                        f"=世界({position[0]:.1f},{position[1]:.1f})，"
                                        f"阶段={self._visual_transit_stage}"
                                    )
                            else:
                                wx, wy = self._visual_transit_entry_world
                                self._visual_transit_anchor = (sx, sy, wx, wy)
                                self._visual_transit_stage = self._ticket_booth_stage_for_y(
                                    wy, edge.y
                                )
                                self.log(f"✅ [无小地图过渡] 人物脚底视觉锚定：画面({sx:.1f},{sy:.1f})=世界({wx:.1f},{wy:.1f})")
            if self._visual_transit_stage == "anchor" and now - self._visual_transit_started_at > 6.0:
                self._set_phase(WorldPatrolPhase.BLOCKED, "售票处6秒内未稳定识别人，禁止盲走")
            return True
        if self._visual_transit_stage == "recover_vision":
            return self._visual_recovery_tick(edge)
        if self._visual_transit_stage == "upper_descend":
            target_x, _, upper_y = self._visual_transit_waypoints
            self._set_phase(
                WorldPatrolPhase.VISUAL_TRANSIT,
                f"上层楼梯先向左至X={target_x:.0f}，避开右侧竖墙",
            )
            reached = self._visual_transit_walk_to(target_x, 7.0)
            if not reached:
                position = self._visual_transit_position()
                if position is None:
                    self._begin_visual_recovery(f"上层下楼去X={target_x:.0f}途中")
                    return True
                self._set_phase(
                    WorldPatrolPhase.BLOCKED,
                    f"售票处上层下楼受阻，目标X={target_x:.0f}，实际={position}；已停键",
                )
                return True
            if self.map_id_getter() is not None and int(self.map_id_getter()) != edge.source_map_id:
                self._wait_started_at = time.perf_counter()
                self._set_phase(WorldPatrolPhase.WAIT_MAP_CHANGE, "售票处提前切图")
                return True
            self._visual_transit_stage = "upper_settle"
            self._visual_transit_started_at = time.perf_counter()
            self._visual_transit_stable_samples = 0
            self._visual_transit_settle_observed_at = 0.0
            self._visual_transit_settle_y = None
            self._set_phase(
                WorldPatrolPhase.VISUAL_TRANSIT,
                f"等待落到上层楼梯底端Y≈{upper_y:.0f}",
            )
            return True
        if self._visual_transit_stage == "upper_settle":
            upper_y = self._visual_transit_waypoints[2]
            if (
                now - self._visual_transit_started_at >= 0.3
                and self._visual_transit_settle_sample(upper_y - 7.0)
            ):
                self._visual_transit_stage = "lower_descend"
                self._set_phase(WorldPatrolPhase.VISUAL_TRANSIT, "上层楼梯落地，改向右下楼")
            elif now - self._visual_transit_started_at > 3.0:
                position = self._visual_transit_position()
                self._begin_visual_recovery(
                    f"上层落地确认超时，当前={position}，目标Y≥{upper_y - 7.0:.0f}，"
                    f"稳定帧={self._visual_transit_stable_samples}",
                    goal_y=upper_y - 7.0,
                )
            return True
        if self._visual_transit_stage == "lower_descend":
            target_x = self._visual_transit_waypoints[1]
            position = self._visual_transit_position()
            if position is None:
                if self._visual_transit_missing_since <= 0.0:
                    self._visual_transit_missing_since = now
                elif now - self._visual_transit_missing_since >= 0.45:
                    self._begin_visual_recovery("下层走位开始前")
                return True
            self._visual_transit_missing_since = 0.0
            if position[1] < self._visual_transit_waypoints[2] - 7.0:
                self._visual_transit_stage = "upper_descend"
                return True
            self._set_phase(
                WorldPatrolPhase.VISUAL_TRANSIT,
                f"下层楼梯向右至X={target_x:.0f}，再落到底层",
            )
            reached = self._visual_transit_walk_to(target_x, 7.0)
            if not reached:
                position = self._visual_transit_position()
                if position is None:
                    self._begin_visual_recovery(f"下层下楼去X={target_x:.0f}途中")
                    return True
                self._set_phase(
                    WorldPatrolPhase.BLOCKED,
                    f"售票处下层下楼受阻，目标X={target_x:.0f}，实际={position}；已停键",
                )
                return True
            self._visual_transit_stage = "settle"
            self._visual_transit_started_at = time.perf_counter()
            self._visual_transit_stable_samples = 0
            self._visual_transit_settle_observed_at = 0.0
            self._visual_transit_settle_y = None
            self._set_phase(WorldPatrolPhase.VISUAL_TRANSIT, "等待角色自然落到售票处底层")
            return True
        if self._visual_transit_stage == "settle":
            if (
                now - self._visual_transit_started_at >= 0.6
                and self._visual_transit_settle_sample(edge.y - 35.0)
            ):
                self._visual_transit_stage = "portal"
                self._set_phase(WorldPatrolPhase.VISUAL_TRANSIT, "已落到底层，前往脚本门")
            if self._visual_transit_stage == "settle" and now - self._visual_transit_started_at > 3.0:
                position = self._visual_transit_position()
                self._begin_visual_recovery(
                    f"底层落地确认超时，当前={position}，目标Y≥{edge.y - 35.0:.0f}，"
                    f"稳定帧={self._visual_transit_stable_samples}",
                    goal_y=edge.y - 35.0,
                )
            return True
        if self._visual_transit_stage == "portal":
            self._visual_transit_attempts += 1
            self.prepare_transition(edge.target_map_id)
            if self.arm_known_portal_ocr is not None:
                self.arm_known_portal_ocr(edge)
            old_checker = getattr(self.motion, "priority_interrupt_checker", None)
            old_teleport = getattr(self.motion, "enable_teleport", False)
            try:
                self.motion.priority_interrupt_checker = None
                self.motion.enable_teleport = False
                entered = self.motion.walk_through_portal(
                    trigger_x=edge.x,
                    get_player_pos=self._visual_transit_position,
                    stop_event=self.stop_event,
                    platform_bounds=(-399, 389),
                    completion_checker=lambda: (
                        self.map_id_getter() is not None
                        and int(self.map_id_getter()) != edge.source_map_id
                    ),
                    approach_lead_px=50.0,
                    timeout_sec=6.0,
                )
            finally:
                self.motion.priority_interrupt_checker = old_checker
                self.motion.enable_teleport = old_teleport
                self.motion.stop()
            if entered:
                self._portal_attempts = self._visual_transit_attempts
                self._wait_started_at = time.perf_counter()
                self._set_phase(WorldPatrolPhase.WAIT_MAP_CHANGE, "售票处脚本门已触发，等待103000101")
            elif self._visual_transit_attempts >= 3:
                self._set_phase(WorldPatrolPhase.BLOCKED, "售票处脚本门连续3次未进入，已停键")
            else:
                position = self._visual_transit_position()
                if position is None:
                    self._begin_visual_recovery("脚本门前")
                    return True
                self._visual_transit_stage = (
                    self._ticket_booth_stage_for_y(position[1], edge.y)
                    if position is not None else "portal"
                )
                self._set_phase(WorldPatrolPhase.VISUAL_TRANSIT, "脚本门未触发，从实际地形阶段重新接近")
            return True
        return True

    def _arrival_clear_tick(self) -> bool:
        current_map = self.map_id_getter()
        graph = self.graph_getter()
        platform = self.platform_getter()
        edge = self._arrival_edge
        if (
            current_map is not None and graph is not None and edge is not None
            and int(getattr(graph, "map_id", -1)) == int(current_map)
            and int((graph.minimap_meta or {}).get("canvasWidth", 0) or 0) == 0
            and self._begin_visual_transit(graph, edge)
        ):
            return True
        position = self.position_getter()
        if (
            current_map is None
            or graph is None
            or platform is None
            or position is None
            or edge is None
            or int(getattr(graph, "map_id", -1)) != int(current_map)
        ):
            return True

        now = time.perf_counter()
        portal_x = self._arrival_portal_x
        if portal_x is None:
            portal_x = self._find_arrival_portal_x(graph, edge)
        if portal_x is None:
            # 某些私服 WZ 没有正确填写 toName；以新图第一次稳定黄点作为
            # 入口轴兜底，仍然先横移再恢复战斗。
            portal_x = float(position[0])
        self._arrival_portal_x = portal_x

        clearance_px = 90.0
        distance = abs(float(position[0]) - portal_x)
        if self._arrival_reset_at <= 0.0 and distance < clearance_px:
            margin = min(35.0, max(10.0, float(platform.length) * 0.12))
            left_safe = float(platform.x_min) + margin
            right_safe = float(platform.x_max) - margin
            left_room = max(0.0, portal_x - left_safe)
            right_room = max(0.0, right_safe - portal_x)
            direction = 1.0 if right_room >= left_room else -1.0
            available = right_room if direction > 0 else left_room
            move_distance = min(max(clearance_px + 30.0, 120.0), available)
            target_x = portal_x + direction * move_distance
            self.log(
                f"🚪 [新图入口脱离] 入口X={portal_x:.1f}，当前X={position[0]:.1f}，"
                f"沿P{platform.id}走向X={target_x:.1f}，期间禁止战斗"
            )
            old_checker = getattr(self.motion, "priority_interrupt_checker", None)
            try:
                self.motion.priority_interrupt_checker = None
                self.motion.walk_to_x(
                    int(round(target_x)),
                    get_player_pos=self.position_getter,
                    tolerance=16,
                    timeout_sec=3.5,
                    stop_event=self.stop_event,
                    platform_bounds=(platform.x_min, platform.x_max),
                    safe_margin=max(10, int(round(margin))),
                )
            finally:
                self.motion.priority_interrupt_checker = old_checker
            if self.stop_event.is_set() or not self.running:
                return True
            position = self.position_getter()
            if position is None:
                return True
            distance = abs(float(position[0]) - portal_x)

        # 已横移到入口影响区之外（短平台则走到其安全极限）。此时再次
        # 清轨，确保光圈阶段误识别到的“人物”不能参与后续攻击几何。
        platform_capacity = max(
            abs(portal_x - (float(platform.x_min) + 10.0)),
            abs((float(platform.x_max) - 10.0) - portal_x),
        )
        # 极短出生平台可能根本没有90px空间；这时以该平台实际可用距离
        # 为准，不能因为永远达不到固定阈值而永久锁住跨图状态机。
        required_distance = min(clearance_px, max(0.0, platform_capacity - 5.0))
        if self._arrival_reset_at <= 0.0 and distance >= required_distance:
            self.motion.stop()
            self.driver.release_all_keys()
            self._reset_arrival_visuals()
            self._arrival_reset_at = time.perf_counter()
            self.log(
                f"🧍 [换图人物重定位] 已离入口{distance:.1f}px，"
                "清空光圈阶段人物/怪物轨迹，等待真实人物连续命中"
            )
            return True
        if self._arrival_reset_at <= 0.0:
            # 被怪推回或平台太窄时下一 tick 从实际位置继续闭环，不把输入
            # 交还战斗状态机。
            return True

        observation = None
        if self.arrival_visual_getter is not None:
            try:
                observation = self.arrival_visual_getter()
            except Exception:
                observation = None
        if observation is not None:
            ox, oy, observed_at = map(float, observation)
            if observed_at > max(self._arrival_reset_at, self._arrival_last_visual_ts):
                previous = self._arrival_last_visual_pos
                if previous is not None and abs(ox - previous[0]) <= 24.0 and abs(oy - previous[1]) <= 24.0:
                    self._arrival_stable_samples += 1
                else:
                    self._arrival_stable_samples = 1
                self._arrival_last_visual_pos = (ox, oy)
                self._arrival_last_visual_ts = observed_at

        settled = (now - self._arrival_reset_at) >= 0.18
        visual_ready = (
            self.arrival_visual_getter is None
            or self._arrival_stable_samples >= 3
        )
        timed_out = (now - self._arrival_reset_at) >= 2.5
        if settled and (visual_ready or timed_out):
            if timed_out and not visual_ready:
                self.log(
                    "⚠️ [换图人物重定位] 2.5s内未取得3次连续视觉命中；"
                    "旧攻击目标已清空，按新图当前定位继续"
                )
            else:
                self.log(
                    f"✅ [换图人物重定位] 连续{self._arrival_stable_samples}次命中稳定，"
                    "恢复路径与战斗"
                )
            self._clear_arrival_state()
            self._set_phase(WorldPatrolPhase.RELOCALIZE, "入口已脱离，重新规划新图")
        return True

    def tick(self) -> bool:
        """返回 True 表示本帧独占输入，CombatFSM 不再执行攻击/巡逻。"""
        with self._lock:
            current_map = self.map_id_getter()
            # 单地图 F6：正常在家地图时零干预；地图一旦变化，先急停，
            # 再经过短暂稳定确认后启动多跳返程。
            if self.recovery_armed and not self.recovery_active:
                home = self._recovery_home_stop
                if home is not None and current_map is not None:
                    if int(current_map) == home.map_id:
                        self._clear_unexpected_map_candidate()
                        self._set_recovery_ui_state(False, "")
                        return False
                    self.motion.stop()
                    self.driver.release_all_keys()
                    self._set_recovery_ui_state(
                        True,
                        f"确认异常地图 {current_map}，准备返回 {home.map_id}",
                    )
                    if self._unexpected_map_stable(int(current_map)):
                        self._notify_unexpected_map()
                        self._start_home_recovery(int(current_map))
                    return True
            if not self._routing_active():
                return False
            phase = self.phase
            if (
                self._rest_stop_index is not None
                and self._rest_stage == "idle"
                and time.perf_counter() >= self._rest_due_at
            ):
                self._rest_stage = "pending"
                elapsed = time.perf_counter() - self._f6_started_at
                late = time.perf_counter() - self._rest_due_at
                self.log(
                    f"⏰ [全局休息截止] F6已运行{elapsed:.2f}s，"
                    f"本轮到点延迟{late:.2f}s；安全收束后转去休息"
                )
            if self._rest_stage == "pending" and phase in (
                WorldPatrolPhase.FARM, WorldPatrolPhase.GO_EXIT,
            ):
                platform = self.platform_getter()
                expected_map = (
                    self.stops[self.stop_index].map_id
                    if phase == WorldPatrolPhase.FARM else
                    (self._exit.source_map_id if self._exit is not None else None)
                )
                if (
                    platform is not None and current_map is not None
                    and expected_map == int(current_map)
                ):
                    self._begin_rest_detour(int(current_map), platform)
                    return True
            if self._rest_stage == "rest_complete":
                self._start_rest_return()
                return True
            # 多地图巡逻在 FARM 阶段原先默认地图不会变化；异常回城后
            # phase 仍是 FARM，因而旧实现不会重规划。现在返回当时正在
            # 刷的目标图，不推进 stop_index。
            if phase == WorldPatrolPhase.FARM and current_map is not None and self.stops:
                expected_map = self.stops[self.stop_index].map_id
                if int(current_map) != expected_map:
                    self.motion.stop()
                    self.driver.release_all_keys()
                    self._set_recovery_ui_state(
                        True,
                        f"从 MapID {current_map} 返回 {expected_map}",
                    )
                    if self._unexpected_map_stable(int(current_map)):
                        self._notify_unexpected_map()
                        self._destination_stop_index = self.stop_index
                        self.log(
                            f"🏠 [跨图异常回城] MapID {expected_map} 刷怪期间进入"
                            f" {current_map}，返回原目标图"
                        )
                        self._begin_travel(self.stop_index)
                    return True
                self._clear_unexpected_map_candidate()
                self._set_recovery_ui_state(False, "")
        if phase == WorldPatrolPhase.FARM:
            # 异常返程到达目标图后，_prepare_current_exit 会重新进入 FARM。
            # 在这里统一恢复 F6 按钮，普通跨图切换不会触发该标记。
            if self._recovery_ui_active:
                self._set_recovery_ui_state(False, "")
            return False
        if phase in (
            WorldPatrolPhase.REST_FARM,
            WorldPatrolPhase.REST_RETURN_PLATFORM,
        ):
            return False
        if phase == WorldPatrolPhase.GO_EXIT:
            edge = self._exit
            platform = self.platform_getter()
            if (
                edge is not None
                and platform is not None
                and int(platform.id) == edge.source_platform_id
            ):
                # 平台巡逻器只负责把人物送上出口平台。落台后立即由跨图
                # 控制器接管并直奔 portal，不能再让单平台站位逻辑先走到
                # 该长平台的 50%/随机站位。
                self.motion.stop()
                self._set_phase(
                    WorldPatrolPhase.ENTER_PORTAL,
                    f"已落到出口P{edge.source_platform_id}，跳过平台站位，"
                    f"直奔 {edge.portal_name}@X={edge.x}",
                )
                return True
            return False
        if phase == WorldPatrolPhase.BLOCKED:
            self.motion.stop()
            if self._visual_recovery_blocked and self._exit is not None:
                current_map = self.map_id_getter()
                if (current_map is not None
                        and int(current_map) != self._exit.source_map_id):
                    self._wait_started_at = time.perf_counter()
                    self._set_phase(WorldPatrolPhase.WAIT_MAP_CHANGE, "视觉找人期间已切图")
                    return True
                # 探测次数耗尽后禁止继续盲按，但视觉稍后恢复时自动接回
                # 原路线，不要求用户重启整段 F6。
                return self._visual_recovery_tick(self._exit)
            return True
        if phase == WorldPatrolPhase.RELOCALIZE:
            current_map = self.map_id_getter()
            graph = self.graph_getter()
            platform = self.platform_getter()
            if (
                current_map is not None and graph is not None
                and (
                    platform is not None
                    or int((graph.minimap_meta or {}).get("canvasWidth", 0) or 0) == 0
                )
                and int(getattr(graph, "map_id", -1)) == int(current_map)
            ):
                with self._lock:
                    if (
                        self._map_route_index < len(self._map_route)
                        and int(current_map) == self._map_route[self._map_route_index]
                    ):
                        self._prepare_current_exit()
                    else:
                        self._replan_from_actual_map(int(current_map))
            return True
        if phase == WorldPatrolPhase.ARRIVAL_CLEAR:
            return self._arrival_clear_tick()
        if phase == WorldPatrolPhase.VISUAL_TRANSIT:
            return self._visual_transit_tick()
        # X闭环可能阻塞数秒，不能持有状态锁，否则GUI状态刷新与F6停止
        # 都会跟着卡住。相关函数只在本工作线程写入阶段字段。
        if phase == WorldPatrolPhase.ENTER_PORTAL:
            return self._enter_portal()
        if phase == WorldPatrolPhase.WAIT_MAP_CHANGE:
            return self._wait_for_map_change()
        return True

    def _enter_portal(self) -> bool:
        if not self._routing_active():
            return False
        edge = self._exit
        graph = self.graph_getter()
        platform = self.platform_getter()
        if edge is None or graph is None or platform is None:
            self._set_phase(WorldPatrolPhase.RELOCALIZE, "进门前定位丢失")
            return True
        if int(self.map_id_getter() or -1) != edge.source_map_id:
            self._set_phase(WorldPatrolPhase.RELOCALIZE, "进门前地图已变化")
            return True
        if int(platform.id) != edge.source_platform_id:
            self._set_phase(WorldPatrolPhase.GO_EXIT, "离开出口平台，重新返回")
            return False
        position = self.position_getter()
        if position is None:
            return True
        distance = abs(float(edge.x) - float(position[0]))
        # 入口可能位于数千像素长平台的最边缘。固定 5 秒只够移动约
        # 800~1000px；按保守 80px/s 给足时间，同时 walk_to_x 自带
        # 1 秒卡死检测和 stop_event，因此不会降低停止响应速度。
        align_timeout = max(5.0, min(30.0, distance / 80.0 + 3.0))
        self.log(
            f"🚪 [传送门穿越准备] {edge.portal_name} 当前X={position[0]:.1f}，"
            f"门轴X={edge.x}，距离={distance:.1f}px，超时={align_timeout:.1f}s"
        )
        self.prepare_transition(edge.target_map_id)
        old_checker = getattr(self.motion, "priority_interrupt_checker", None)
        try:
            self.motion.priority_interrupt_checker = None
            arrived = self.motion.walk_through_portal(
                trigger_x=edge.x,
                get_player_pos=self.position_getter,
                stop_event=self.stop_event,
                platform_bounds=(platform.x_min, platform.x_max),
                completion_checker=lambda: int(self.map_id_getter() or -1) != edge.source_map_id,
                # 传送门范围很大，UP 从原来的门前70px提前到120px，
                # 让人物保持方向键走入门轴时已经稳定压住UP。
                approach_lead_px=120.0,
                pass_through_px=18.0,
                timeout_sec=align_timeout,
            )
        finally:
            self.motion.priority_interrupt_checker = old_checker
        if self.stop_event.is_set() or not self.running:
            return True
        if not arrived:
            latest_platform = self.platform_getter()
            latest_position = self.position_getter()
            if (
                latest_platform is not None
                and int(latest_platform.id) == edge.source_platform_id
            ):
                # 仍在正确长平台就从当前位置续走，不能交还单平台站位逻辑。
                latest_x = latest_position[0] if latest_position is not None else None
                self._set_phase(
                    WorldPatrolPhase.ENTER_PORTAL,
                    f"X对齐未完成，仍在P{edge.source_platform_id}续走"
                    + (f"（X={latest_x:.1f}）" if latest_x is not None else ""),
                )
            else:
                self._set_phase(WorldPatrolPhase.GO_EXIT, "离开出口平台，重新返回")
            return True
        self._portal_attempts += 1
        self.log(
            f"🚪 [进入跨地图传送门 type={edge.portal_type}] MapID {edge.source_map_id} "
            f"{edge.portal_name}@X={edge.x} → {edge.target_map_id}，"
            f"第{self._portal_attempts}次"
        )
        self._wait_started_at = time.perf_counter()
        self._set_phase(
            WorldPatrolPhase.WAIT_MAP_CHANGE,
            f"等待 MapID {edge.target_map_id}",
        )
        return True

    def _wait_for_map_change(self) -> bool:
        if not self._routing_active():
            return False
        edge = self._exit
        if edge is None:
            self._set_phase(WorldPatrolPhase.BLOCKED, "缺少正在执行的传送门")
            return True
        current_map = self.map_id_getter()
        if current_map is not None and int(current_map) != edge.source_map_id:
            if int(current_map) == edge.target_map_id:
                self._map_route_index += 1
                self._exit = None
                self._begin_arrival_clear(edge, int(current_map))
            else:
                self.log(
                    f"⚠️ [跨图意外落点] 预期{edge.target_map_id}，实际{current_map}"
                )
                self._replan_from_actual_map(int(current_map))
            return True
        # 成功进门后标题变化/OCR加速通常在 1 秒内给出新 MapID；5 秒
        # 足够覆盖黑屏和加载波动，同时避免失败后原地等待 8 秒。
        if time.perf_counter() - self._wait_started_at < 5.0:
            return True
        if self._portal_attempts >= 3:
            self._set_phase(
                WorldPatrolPhase.BLOCKED,
                f"传送门{edge.portal_name}(type={edge.portal_type})连续3次未切图",
            )
            return True
        # 不再先盲走 0.28 秒“离门”。该动作结束后世界坐标可能尚未刷新，
        # 下一轮会依据旧坐标选择错误方向。直接从当前稳定坐标反向穿越，
        # walk_through_portal 会在门轴处重新触发 UP。
        position = self.position_getter()
        detail = "切图超时，从当前位置重新穿越"
        if position is not None:
            detail += f"（X={float(position[0]):.1f}）"
        self._set_phase(WorldPatrolPhase.ENTER_PORTAL, detail)
        return True


def parse_world_patrol_stops(raw: str) -> List[WorldPatrolStop]:
    """解析 ``107000100:7,11;107000200:5,8``。"""
    text = str(raw or "").replace("；", ";").replace("：", ":")
    stops: List[WorldPatrolStop] = []
    for section in (part.strip() for part in text.split(";")):
        if not section:
            continue
        if ":" not in section:
            raise ValueError(f"缺少冒号：{section}")
        map_text, platforms_text = section.split(":", 1)
        if not map_text.strip().isdigit():
            raise ValueError(f"MapID 必须是纯数字：{map_text.strip()}")
        tokens = platforms_text.replace("，", ",").replace(" ", ",").split(",")
        platforms = []
        for token in tokens:
            token = token.strip().upper()
            if token.startswith("P"):
                token = token[1:]
            if not token:
                continue
            if not token.isdigit():
                raise ValueError(f"平台编号无效：{token}")
            value = int(token)
            if value not in platforms:
                platforms.append(value)
        if not platforms:
            raise ValueError(f"MapID {map_text.strip()} 没有平台目标")
        stops.append(WorldPatrolStop(int(map_text.strip()), tuple(platforms)))
    return stops
