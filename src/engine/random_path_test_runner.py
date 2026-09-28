"""Randomized whole-map navigation acceptance test using the real F6 executor."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2


CLIMB_FAILURE_LIMIT = 3


class RandomPathTestRunner:
    """Cover as many merged platforms as possible and retain replayable telemetry."""

    def __init__(
        self,
        *,
        driver: Any,
        motion: Any,
        get_graph: Callable[[], Any],
        get_platform: Callable[[], Any],
        get_world_position: Callable[[], Optional[Tuple[float, float]]],
        get_raw_position: Callable[[], Optional[Tuple[float, float]]],
        get_is_climbing: Callable[[], bool],
        capture_frame: Callable[[], Any],
        get_run_jump_enabled: Callable[[], bool],
        get_intra_map_portal_enabled: Callable[[], bool],
        f6_edge_executor: Callable[..., Any],
        begin_f6_test_session: Callable[[], Tuple[bool, str]],
        end_f6_test_session: Callable[[], None],
        log: Callable[[str], None],
        status_callback: Optional[Callable[[str, bool], None]] = None,
    ) -> None:
        self.driver = driver
        self.motion = motion
        self.get_graph = get_graph
        self.get_platform = get_platform
        self.get_world_position = get_world_position
        self.get_raw_position = get_raw_position
        self.get_is_climbing = get_is_climbing
        self.capture_frame = capture_frame
        self.get_run_jump_enabled = get_run_jump_enabled
        self.get_intra_map_portal_enabled = get_intra_map_portal_enabled
        self.f6_edge_executor = f6_edge_executor
        self.begin_f6_test_session = begin_f6_test_session
        self.end_f6_test_session = end_f6_test_session
        self.log = log
        self.status_callback = status_callback or (lambda _text, _active: None)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.active = False
        self.last_output_dir: Optional[str] = None
        self._last_platform_wait_stable = False

    @staticmethod
    def _edge_key(edge: Any) -> Tuple[int, int, str, Optional[int]]:
        return (
            int(edge.from_id),
            int(edge.to_id),
            str(edge.action),
            getattr(edge, "ladder_id", None),
        )

    @staticmethod
    def _edge_label(edge: Any) -> str:
        ladder = (
            f" rope={getattr(edge, 'ladder_id', None)}"
            if getattr(edge, "ladder_id", None) is not None else ""
        )
        footholds = ""
        if getattr(edge, "source_foothold_id", None) is not None:
            footholds = (
                f" fh{edge.source_foothold_id}->fh{edge.target_foothold_id}"
                f" takeoff={getattr(edge, 'takeoff_x', None)}"
            )
        return f"P{edge.from_id}->P{edge.to_id} {edge.action}{footholds}{ladder}"

    @staticmethod
    def _edge_payload(edge: Any) -> Dict[str, Any]:
        if is_dataclass(edge):
            return asdict(edge)
        names = (
            "from_id", "to_id", "action", "cost", "trigger_x",
            "trigger_x_range", "landing_x", "landing_x_range", "target_y",
            "ladder_id", "is_rope", "description", "source_foothold_id",
            "target_foothold_id", "takeoff_x", "takeoff_x_range", "confidence",
        )
        return {name: getattr(edge, name, None) for name in names}

    def _location_payload(self) -> Dict[str, Any]:
        platform = self.get_platform()
        world = self.get_world_position()
        raw = self.get_raw_position()
        return {
            "platform_id": getattr(platform, "id", None),
            "world": list(world) if world is not None else None,
            "raw": list(raw) if raw is not None else None,
            "climbing": bool(self.get_is_climbing()),
        }

    @staticmethod
    def plan_coverage_targets(
        graph: Any,
        start_id: int,
        *,
        allow_run_jump: bool,
        allow_portal: bool,
        seed: int,
        trials: int = 96,
    ) -> Tuple[List[int], Set[int], Set[int]]:
        """Random-search an executable target order, maximizing visited nodes first."""
        node_ids = sorted(int(value) for value in graph.nodes)
        if int(start_id) not in graph.nodes:
            return [], set(), set(node_ids)
        cache: Dict[Tuple[int, int], List[Any]] = {}

        def path(source: int, target: int) -> List[Any]:
            key = (int(source), int(target))
            if key not in cache:
                cache[key] = list(graph.find_path(
                    key[0], key[1],
                    allow_run_jump=bool(allow_run_jump),
                    allow_portal=bool(allow_portal),
                ))
            return cache[key]

        rng = random.Random(int(seed))
        best_targets: List[int] = []
        best_covered: Set[int] = {int(start_id)}
        best_steps = 10 ** 9
        # Each trial changes target order. Intermediate nodes on shortest paths count
        # as covered, so the chosen result normally needs far fewer macro targets.
        for trial in range(max(1, int(trials))):
            order = [node for node in node_ids if node != int(start_id)]
            if trial:
                rng.shuffle(order)
            covered = {int(start_id)}
            current = int(start_id)
            targets: List[int] = []
            total_steps = 0
            while True:
                candidates = []
                for target in order:
                    if target in covered:
                        continue
                    route = path(current, target)
                    if not route:
                        continue
                    new_nodes = {int(edge.to_id) for edge in route} - covered
                    if new_nodes:
                        candidates.append((target, route, len(new_nodes)))
                if not candidates:
                    break
                # Prefer routes that cover more new platforms per action, but retain
                # randomness among the best few to escape directed dead ends.
                candidates.sort(
                    key=lambda item: (item[2] / max(1, len(item[1])), item[2]),
                    reverse=True,
                )
                pool = candidates[: min(5, len(candidates))]
                target, route, _ = pool[0] if trial == 0 else rng.choice(pool)
                targets.append(int(target))
                total_steps += len(route)
                covered.update(int(edge.to_id) for edge in route)
                current = int(target)
            score = (len(covered), -total_steps)
            best_score = (len(best_covered), -best_steps)
            if score > best_score:
                best_targets = targets
                best_covered = covered
                best_steps = total_steps
        return best_targets, best_covered, set(node_ids) - best_covered

    def start(self) -> Tuple[bool, str]:
        if self.active:
            return False, "随机全图行走测试正在运行。"
        graph = self.get_graph()
        platform = self.get_platform()
        if graph is None or not getattr(graph, "nodes", None):
            return False, "当前地图拓扑尚未加载完成。"
        if platform is None:
            return False, "当前承重平台尚未定位，请先站稳后再开始测试。"
        if not callable(self.f6_edge_executor):
            return False, "F6 共享单边执行器尚未就绪。"
        readiness_check = getattr(self.driver, "check_input_readiness", None)
        if callable(readiness_check):
            try:
                ready, reason = readiness_check(focus=True)
            except Exception as exc:
                return False, f"输入通道检查失败：{exc}"
            if not ready:
                return False, str(reason)
        try:
            session_ok, session_message = self.begin_f6_test_session()
        except Exception as exc:
            return False, f"F6 共享导航测试会话启动失败：{exc}"
        if not session_ok:
            return False, session_message
        self.stop_event.clear()
        self.active = True
        self.thread = threading.Thread(
            target=self._run,
            args=(graph, int(platform.id)),
            name="RandomPathTest",
            daemon=True,
        )
        try:
            self.thread.start()
        except Exception:
            self.active = False
            self.end_f6_test_session()
            raise
        return True, "随机全图行走测试已启动。"

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.motion.stop()
            self.driver.release_all_keys()
        except Exception:
            pass
        if self.active:
            self.status_callback("正在停止随机全图行走测试…", True)

    def _wait_for_stable_platform(self, timeout_sec: float = 1.6) -> Any:
        self._last_platform_wait_stable = False
        deadline = time.perf_counter() + max(0.0, float(timeout_sec))
        stable_id = None
        stable_since = 0.0
        stable_x = None
        stable_y = None
        latest = None
        while time.perf_counter() < deadline and not self.stop_event.is_set():
            latest = self.get_platform()
            current_id = getattr(latest, "id", None)
            world = self.get_world_position()
            standing = False
            current_y = None
            if current_id is not None and world is not None:
                try:
                    current_x, current_y = float(world[0]), float(world[1])
                    expected_y = float(latest.surface_y_at(current_x)) - 45.0
                    standing = abs(current_y - expected_y) <= 20.0
                except Exception:
                    standing = False
            now = time.perf_counter()
            if current_id is None or not standing:
                stable_id = None
                stable_since = 0.0
                stable_x = None
                stable_y = None
            elif current_id != stable_id:
                stable_id = current_id
                stable_since = now
                stable_x = current_x
                stable_y = current_y
            else:
                # 斜坡落点会在 Y 已经稳定时继续水平滑动。只看 Y 会过早
                # 派发下一条边，并把这段自然滑移误作受击/外力位移。
                if (
                    stable_x is None
                    or stable_y is None
                    or abs(float(current_x) - float(stable_x)) > 3.0
                    or abs(float(current_y) - float(stable_y)) > 3.0
                ):
                    stable_since = now
                stable_x = current_x
                stable_y = current_y
            if stable_id is not None and (now - stable_since) >= 0.18:
                self._last_platform_wait_stable = True
                return latest
            self.stop_event.wait(0.06)
        return latest

    @staticmethod
    def _verify_timeout(edge: Any) -> float:
        action = str(getattr(edge, "action", ""))
        if "CLIMB" in action:
            return 1.8
        if action == "PORTAL" or "LONG_DROP" in action:
            return 2.4
        if "DROP" in action:
            return 1.8
        return 1.5

    def _run(self, graph: Any, start_id: int) -> None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        run_name = time.strftime("%Y%m%d_%H%M%S") + f"_map_{graph.map_id}"
        out_dir = os.path.join(root, "logs", "random_path_tests", run_name)
        os.makedirs(out_dir, exist_ok=True)
        self.last_output_dir = out_dir
        log_path = os.path.join(out_dir, "run.log")
        events_path = os.path.join(out_dir, "events.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
        graph_path = os.path.join(out_dir, "graph_snapshot.json")
        seed = int(time.time_ns() & 0xFFFFFFFF)
        allow_run_jump = bool(self.get_run_jump_enabled())
        allow_portal = bool(self.get_intra_map_portal_enabled())
        started_wall = time.time()
        event_index = 0
        covered: Set[int] = {int(start_id)}
        failed_targets: Set[int] = set()
        edge_attempts: Counter = Counter()
        edge_successes: Counter = Counter()
        action_attempts: Counter = Counter()
        action_successes: Counter = Counter()
        failure_images = 0

        def text_log(message: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] {message}"
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            self.log(line)

        def event(kind: str, **payload: Any) -> None:
            nonlocal event_index
            event_index += 1
            record = {
                "index": event_index,
                "ts": round(time.time(), 6),
                "mono": round(time.perf_counter(), 6),
                "event": kind,
                **payload,
            }
            with open(events_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        def snapshot(reason: str, step: int) -> Optional[str]:
            nonlocal failure_images
            frame = self.capture_frame()
            if frame is None:
                return None
            failure_images += 1
            name = f"failure_{failure_images:03d}_step_{step:03d}_{reason}.png"
            path = os.path.join(out_dir, name)
            try:
                cv2.imwrite(path, frame)
                return name
            except Exception:
                return None

        with open(log_path, "w", encoding="utf-8") as stream:
            stream.write("=== randomized whole-map F6 navigation acceptance test ===\n")
        try:
            if not self.driver.ensure_focus():
                text_log("⛔ [随机路径测试] 无法聚焦游戏窗口")
                return
            targets, planned_covered, planned_uncovered = self.plan_coverage_targets(
                graph,
                start_id,
                allow_run_jump=allow_run_jump,
                allow_portal=allow_portal,
                seed=seed,
            )
            if hasattr(graph, "to_dict"):
                with open(graph_path, "w", encoding="utf-8") as stream:
                    json.dump(
                        (
                            graph.to_cache_dict()
                            if hasattr(graph, "to_cache_dict")
                            else graph.to_dict()
                        ),
                        stream,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
            all_edges = [edge for values in graph.edges.values() for edge in values]
            metadata = {
                "map_id": graph.map_id,
                "map_name": getattr(graph, "map_name", ""),
                "seed": seed,
                "start_platform": start_id,
                "node_count": len(graph.nodes),
                "edge_count": len(all_edges),
                "allow_run_jump": allow_run_jump,
                "allow_intra_map_portal": allow_portal,
                "planned_targets": targets,
                "planned_covered": sorted(planned_covered),
                "planned_uncovered": sorted(planned_uncovered),
                "start_location": self._location_payload(),
            }
            event("session_start", **metadata)
            text_log(
                f"🧪 [随机路径测试开始] MapID={graph.map_id}，seed={seed}，"
                f"起点=P{start_id}，节点={len(graph.nodes)}，边={len(all_edges)}，"
                f"跑跳={'开' if allow_run_jump else '关'}，"
                f"图内传送点={'启用' if allow_portal else '禁用'}"
            )
            text_log(
                f"🗺️ [覆盖计划] 宏观目标={' -> '.join('P' + str(x) for x in targets) or '无'}；"
                f"预计覆盖={len(planned_covered)}/{len(graph.nodes)}；"
                f"预计不可覆盖={sorted(planned_uncovered)}"
            )
            if not targets:
                text_log("ℹ️ [随机路径测试] 当前平台无法到达其他平台。")
                return

            max_steps = max(20, min(600, len(graph.nodes) * 10))
            step = 0
            deferred_source_id: Optional[int] = None
            deferred_source_until = 0.0

            def platform_with_deferred_source_guard() -> Any:
                """Keep no-key retries on their physical source, not an overlap ID."""
                observed = self.get_platform()
                if (
                    deferred_source_id is None
                    or time.perf_counter() > deferred_source_until
                ):
                    return observed
                source = graph.get_node(int(deferred_source_id))
                if source is None:
                    return observed
                raw = self.get_raw_position()
                if raw is None:
                    return source
                raw_x, raw_y = float(raw[0]), float(raw[1])
                if not (float(source.x_min) <= raw_x <= float(source.x_max)):
                    return observed
                expected_y = float(source.surface_y_at(raw_x)) - 45.0
                # Once the character has physically left the source height, stop
                # guarding immediately so landing/climbing checks see real state.
                if abs(raw_y - expected_y) > 65.0:
                    return observed
                return source

            # 梯绳容错跨宏观目标保留，并只按实体 ladder_id 计数；成功后
            # 才清零。这样同一根绳从不同平台入口尝试也不会各得三次。
            ladder_failures: Counter = Counter()
            for target_index, target_id in enumerate(targets, 1):
                same_edge_failures: Counter = Counter()
                while not self.stop_event.is_set() and step < max_steps:
                    if self.get_graph() is not graph:
                        text_log("⛔ [随机路径测试] 地图或拓扑已切换，立即停止。")
                        event("graph_changed", location=self._location_payload())
                        return
                    current = self.get_platform() or self._wait_for_stable_platform(1.2)
                    if current is None:
                        text_log(f"⛔ [目标{target_index}] 当前承重平台丢失超过1.2秒。")
                        event("platform_missing", target=target_id, location=self._location_payload())
                        snapshot("platform_missing", step)
                        failed_targets.add(int(target_id))
                        break
                    observed_current_id = int(current.id)
                    if (
                        deferred_source_id is not None
                        and time.perf_counter() <= deferred_source_until
                        and observed_current_id != int(deferred_source_id)
                    ):
                        guarded = graph.get_node(int(deferred_source_id))
                        if guarded is not None:
                            text_log(
                                f"🛡️ [测试未发键源平台保护] 忽略P{observed_current_id}"
                                f"瞬时观测，继续按源平台P{deferred_source_id}规划"
                            )
                            event(
                                "deferred_source_guard",
                                observed_platform=observed_current_id,
                                forced_platform=int(deferred_source_id),
                            )
                            current = guarded
                    elif (
                        deferred_source_id is not None
                        and time.perf_counter() > deferred_source_until
                    ):
                        deferred_source_id = None
                        deferred_source_until = 0.0
                    current_id = int(current.id)
                    covered.add(current_id)
                    if current_id == int(target_id):
                        text_log(f"✅ [覆盖目标] 到达P{target_id}（{target_index}/{len(targets)}）")
                        event("target_arrived", target=target_id, location=self._location_payload())
                        break
                    route = graph.find_path(
                        current_id,
                        int(target_id),
                        allow_run_jump=allow_run_jump,
                        allow_portal=allow_portal,
                        edge_penalty_fn=lambda item: (
                            1000.0
                            if (
                                same_edge_failures[self._edge_key(item)] >= 3
                                or (
                                    getattr(item, "ladder_id", None) is not None
                                    and ladder_failures[int(item.ladder_id)]
                                    >= CLIMB_FAILURE_LIMIT
                                )
                            )
                            else 0.0
                        ),
                    )
                    if not route:
                        text_log(f"⛔ [实时重规划] P{current_id}->P{target_id} 无可执行路径。")
                        event("route_missing", source=current_id, target=target_id)
                        failed_targets.add(int(target_id))
                        break
                    step += 1
                    edge = route[0]
                    key = self._edge_key(edge)
                    label = self._edge_label(edge)
                    metrics = graph.path_metrics(route)
                    before = self._location_payload()
                    text_log(
                        f"▶️ [随机路径边#{step}] 目标P{target_id}，{label}；"
                        f"剩余{len(route)}步，cost={metrics.get('estimated_cost')}，"
                        f"ropes={metrics.get('rope_count')}，before={before}"
                    )
                    event(
                        "edge_dispatch",
                        step=step,
                        target_index=target_index,
                        macro_target=target_id,
                        route_metrics=metrics,
                        route=[self._edge_payload(item) for item in route],
                        edge=self._edge_payload(edge),
                        before=before,
                        prior_failures=max(
                            same_edge_failures[key],
                            ladder_failures[int(edge.ladder_id)]
                            if getattr(edge, "ladder_id", None) is not None else 0,
                        ),
                    )
                    edge_attempts[label] += 1
                    action_attempts[str(edge.action)] += 1
                    position = self.get_world_position()
                    if position is None:
                        text_log(f"⛔ [随机路径边#{step}] 执行前世界坐标丢失。")
                        event("position_missing_before_edge", step=step, edge=label)
                        snapshot("position_missing", step)
                        failed_targets.add(int(target_id))
                        break
                    edge_started = time.perf_counter()
                    outcome = self.f6_edge_executor(
                        edge,
                        int(target_id),
                        float(position[0]),
                        float(position[1]),
                        world_position_getter=self.get_world_position,
                        raw_position_getter=self.get_raw_position,
                        platform_getter=platform_with_deferred_source_guard,
                        is_climbing_getter=self.get_is_climbing,
                        stop_event=self.stop_event,
                        run_jump_enabled=allow_run_jump,
                        log_callback=text_log,
                    )
                    elapsed = time.perf_counter() - edge_started
                    # retry 表示动作键尚未发出，无需再等 1.5~2.4 秒“落台”。
                    # 直接保留源平台语义进入下一轮，避免把量化抖到的邻台
                    # 当成真实落点后走一大段路再返回。
                    observed = (
                        self.get_platform()
                        if outcome in ("retry", "relocalize", "top_exit_failed")
                        else self._wait_for_stable_platform(self._verify_timeout(edge))
                    )
                    observed_id = getattr(observed, "id", None)
                    after = self._location_payload()
                    observed_target = (
                        observed_id is not None
                        and int(observed_id) == int(edge.to_id)
                    )
                    # The shared F6 executor returns "retry" only when the action
                    # key was never emitted. A transient platform-ID flip can never
                    # turn that deferred attempt into a success.
                    arrived = bool(
                        observed_target
                        and outcome not in ("retry", "relocalize", "top_exit_failed")
                        and self._last_platform_wait_stable
                    )
                    event(
                        "edge_result",
                        step=step,
                        edge=label,
                        outcome=outcome,
                        elapsed_sec=round(elapsed, 4),
                        arrived=arrived,
                        expected_platform=int(edge.to_id),
                        observed_platform=observed_id,
                        observed_target_during_retry=bool(
                            observed_target and outcome == "retry"
                        ),
                        after=after,
                    )
                    text_log(
                        f"{'✅' if arrived else '⚠️'} [随机路径边验证#{step}] {label}，"
                        f"outcome={outcome}，期望P{edge.to_id}，实际P{observed_id}，"
                        f"耗时={elapsed:.2f}s，after={after}"
                    )
                    if outcome == "retry":
                        deferred_source_id = int(edge.from_id)
                        deferred_source_until = time.perf_counter() + 3.0
                        if observed_target:
                            text_log(
                                f"🛡️ [未发键假落台拦截#{step}] 动作未发出但平台观测跳到"
                                f"P{observed_id}，不计成功并按源平台P{edge.from_id}语义重试"
                            )
                        event(
                            "edge_retry_without_penalty",
                            step=step,
                            edge=label,
                            expected_source_platform=int(edge.from_id),
                        )
                        self.stop_event.wait(0.20)
                        continue
                    if outcome == "relocalize":
                        deferred_source_id = None
                        deferred_source_until = 0.0
                        relocalize_count = same_edge_failures[key] + 1
                        # 两次都在同一动作准备阶段滑出源平台，说明不是一次
                        # 观测抖动。直接达到绕行阈值；成功走回源平台不能
                        # 清除此具体失败边的记录。
                        same_edge_failures[key] = (
                            3 if relocalize_count >= 2 else relocalize_count
                        )
                        text_log(
                            f"↩️ [测试源平台丢失重定位#{step}] {label} 动作未发出，"
                            f"解除源平台保护；当前观测P{observed_id}，"
                            f"动作准备连续失败={relocalize_count}"
                        )
                        event(
                            "edge_relocalize_without_penalty",
                            step=step,
                            edge=label,
                            observed_platform=observed_id,
                            action_failures=relocalize_count,
                        )
                        if relocalize_count >= 2:
                            text_log(
                                f"↪️ [测试动作准备失败换边#{step}] {label} 连续两次"
                                "在发键前丢失源平台；本轮禁用该动作边并重规划"
                            )
                            event(
                                "relocalize_action_replan",
                                step=step,
                                edge=label,
                                action_failures=relocalize_count,
                            )
                        self.stop_event.wait(0.08)
                        continue
                    if outcome == "top_exit_failed":
                        deferred_source_id = None
                        deferred_source_until = 0.0
                        ladder_id = getattr(edge, "ladder_id", None)
                        if ladder_id is not None:
                            ladder_id = int(ladder_id)
                            ladder_failures[ladder_id] += 1
                            climb_failures = ladder_failures[ladder_id]
                        else:
                            same_edge_failures[key] += 1
                            climb_failures = same_edge_failures[key]
                        text_log(
                            f"{'↪️' if climb_failures >= CLIMB_FAILURE_LIMIT else '🔁'} "
                            f"[测试绳顶失败容错#{step}] {label} 已抓住但脱绳失败；"
                            f"梯绳#{ladder_id} 连续{climb_failures}/{CLIMB_FAILURE_LIMIT}次，"
                            + (
                                "达到上限，禁用本轮该梯绳并重规划"
                                if climb_failures >= CLIMB_FAILURE_LIMIT
                                else "未达上限，允许重新定位后重试"
                            )
                        )
                        event(
                            "top_exit_failure_replan",
                            step=step,
                            edge=label,
                            observed_platform=observed_id,
                            ladder_id=ladder_id,
                            consecutive=climb_failures,
                            retry_limit=CLIMB_FAILURE_LIMIT,
                        )
                        snapshot("top_exit_failed", step)
                        self.stop_event.wait(0.08)
                        continue
                    deferred_source_id = None
                    deferred_source_until = 0.0
                    if observed_id is not None:
                        covered.add(int(observed_id))
                    if arrived:
                        edge_successes[label] += 1
                        action_successes[str(edge.action)] += 1
                        same_edge_failures.pop(key, None)
                        ladder_id = getattr(edge, "ladder_id", None)
                        if ladder_id is not None:
                            cleared = ladder_failures.pop(int(ladder_id), 0)
                            if cleared:
                                text_log(
                                    f"✅ [梯绳容错清零] 梯绳#{int(ladder_id)} 成功，"
                                    f"此前连续失败{cleared}次已清零"
                                )
                        continue
                    if observed_id is not None and int(observed_id) != current_id:
                        ladder_id = getattr(edge, "ladder_id", None)
                        if ladder_id is not None:
                            ladder_id = int(ladder_id)
                            ladder_failures[ladder_id] += 1
                            text_log(
                                f"🪢 [测试梯绳失败容错#{step}] {label} 落到非预期"
                                f"P{observed_id}；梯绳#{ladder_id} 连续"
                                f"{ladder_failures[ladder_id]}/{CLIMB_FAILURE_LIMIT}次"
                            )
                        event(
                            "unexpected_landing_replan",
                            step=step,
                            edge=label,
                            observed_platform=int(observed_id),
                        )
                        snapshot("unexpected_landing", step)
                        continue
                    ladder_id = getattr(edge, "ladder_id", None)
                    if ladder_id is not None:
                        ladder_id = int(ladder_id)
                        ladder_failures[ladder_id] += 1
                        failure_count = ladder_failures[ladder_id]
                    else:
                        same_edge_failures[key] += 1
                        failure_count = same_edge_failures[key]
                    image = snapshot("edge_failed", step)
                    event(
                        "edge_failure_same_source",
                        step=step,
                        edge=label,
                        consecutive=failure_count,
                        screenshot=image,
                    )
                    if failure_count >= 4:
                        text_log(
                            f"⛔ [动作连续失败] {label} 在同一源平台连续4次失败，"
                            f"放弃本宏观目标P{target_id}，继续其余覆盖计划。"
                        )
                        failed_targets.add(int(target_id))
                        break
                    self.stop_event.wait(0.25)
                if step >= max_steps:
                    text_log(f"⛔ [随机路径测试] 达到安全上限{max_steps}条边。")
                    break
                self.status_callback(
                    f"随机全图测试：已覆盖 {len(covered)}/{len(graph.nodes)} 个平台",
                    True,
                )
        except Exception as exc:
            text_log(f"💥 [随机路径测试异常] {exc!r}")
            event("fatal", error=repr(exc), location=self._location_payload())
            snapshot("fatal", 0)
        finally:
            stopped = self.stop_event.is_set()
            try:
                self.motion.stop()
                self.driver.release_all_keys()
            except Exception:
                pass
            try:
                self.end_f6_test_session()
            except Exception as exc:
                text_log(f"⚠️ [随机路径测试] 恢复F6导航状态失败：{exc}")
            all_nodes = set(int(value) for value in graph.nodes)
            summary = {
                "map_id": graph.map_id,
                "seed": seed,
                "started_at": started_wall,
                "duration_sec": round(time.time() - started_wall, 3),
                "stopped_by_user": stopped,
                "covered_platforms": sorted(covered),
                "uncovered_platforms": sorted(all_nodes - covered),
                "coverage_count": len(covered),
                "platform_count": len(all_nodes),
                "failed_targets": sorted(failed_targets),
                "edge_attempts": dict(edge_attempts),
                "edge_successes": dict(edge_successes),
                "action_attempts": dict(action_attempts),
                "action_successes": dict(action_successes),
                "failure_screenshot_count": failure_images,
                "final_location": self._location_payload(),
            }
            # Even if the disk becomes unavailable while writing the final report,
            # the UI must never remain stuck in the "test active" state.
            self.active = False
            with open(summary_path, "w", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, default=str)
            event("session_end", **summary)
            text_log(
                f"🏁 [随机路径测试结果] 覆盖{len(covered)}/{len(all_nodes)}，"
                f"未覆盖={sorted(all_nodes - covered)}，失败目标={sorted(failed_targets)}；"
                f"日志目录：{out_dir}"
            )
            self.status_callback(
                f"测试结束：覆盖 {len(covered)}/{len(all_nodes)}；日志已保存",
                False,
            )
