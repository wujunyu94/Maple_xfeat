"""
platform_graph.py - 通用多平台与梯绳拓扑图自动建模引擎 (PlatformGraphBuilder & PlatformGraph)
纯数据驱动，适用于任意地图 (footholds + ladderRopes)，零特调参数。
自动完成平台聚类合并、下跳/跳跃/攀爬动作边生成，并支持 A* / 循环巡航路径规划。
"""

import os
import json
import heapq
import math
import urllib.request
import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Set


@dataclass
class PlatformNode:
    """平台节点定义"""
    id: int                       # 平台唯一编号 (1-indexed)
    x_min: int                    # 平台左边界 X
    x_max: int                    # 平台右边界 X
    y: int                        # 平台平均高度 Y (物理坐标系)
    length: int                   # 平台水平宽度
    layer: int = 0                # 所在物理垂直层级 (自底向上从 1 开始)
    foothold_ids: List[int] = field(default_factory=list)
    raw_lines: List[Dict[str, int]] = field(default_factory=list)

    @property
    def center_x(self) -> int:
        return (self.x_min + self.x_max) // 2

    def surface_y_at(self, x: float) -> float:
        """返回合并平台在指定 X 的真实 foothold 表面高度。

        合并只改变拓扑节点数量，不能把台阶/斜坡压成 ``self.y`` 的
        平均水平线。若 X 位于多条线段上，选取最接近平台代表高度的一条；
        若落在链的空隙，则使用水平距离最近的端点作为稳定兜底。
        """
        candidates: List[float] = []
        nearest: Optional[Tuple[float, float]] = None
        for line in self.raw_lines:
            try:
                x1, y1 = float(line['x1']), float(line['y1'])
                x2, y2 = float(line['x2']), float(line['y2'])
            except (KeyError, TypeError, ValueError):
                continue
            dx = x2 - x1
            if abs(dx) < 1e-9:
                continue
            lo, hi = min(x1, x2), max(x1, x2)
            if lo <= x <= hi:
                ratio = (float(x) - x1) / dx
                candidates.append(y1 + ratio * (y2 - y1))
            for endpoint_x, endpoint_y in ((x1, y1), (x2, y2)):
                distance = abs(float(x) - endpoint_x)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, endpoint_y)
        if candidates:
            return min(candidates, key=lambda value: abs(value - self.y))
        return nearest[1] if nearest is not None else float(self.y)

    @property
    def center_y(self) -> int:
        """平台标签/显示使用的真实中点表面高度。"""
        return int(round(self.surface_y_at(self.center_x)))


@dataclass
class LadderRopeNode:
    """梯子与绳索节点定义 (官方物理通道)"""
    id: int                       # 梯绳唯一编号 (1-indexed)
    x: int                        # 梯绳垂直 X 坐标
    y1: int                       # 梯绳顶端 Y
    y2: int                       # 梯绳底端 Y
    is_ladder: bool = False       # True: 梯子 (Ladder), False: 绳索 (Rope)
    bottom_platform_id: Optional[int] = None  # 下端所连通的平台 ID
    top_platform_id: Optional[int] = None     # 上端所连通的平台 ID

    @property
    def kind_name(self) -> str:
        return "梯子" if self.is_ladder else "绳索"

    @property
    def label(self) -> str:
        prefix = "梯" if self.is_ladder else "绳"
        return f"{prefix}#{self.id}"

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def length(self) -> int:
        return abs(self.y2 - self.y1)


@dataclass
class PlatformEdge:
    """平台之间的动作转移边"""
    from_id: int                  # 源平台 ID
    to_id: int                    # 目标平台 ID
    action: str                   # "DOWN_JUMP", "JUMP_LEFT", "JUMP_RIGHT", "CLIMB_UP", "CLIMB_DOWN"
    cost: float                   # 转移代价/权重 (秒或距离)
    
    # 动作执行参数
    trigger_x: Optional[int] = None           # 出发动作触发 X 坐标 (如梯子中轴、跳跃点)
    trigger_x_range: Optional[Tuple[int, int]] = None  # 可触发动作的区间 (如可下跳区间)
    landing_x: Optional[float] = None          # 按当前运动模型估算的落点；None 表示使用目标平台中心
    target_y: Optional[int] = None            # 动作结束到达的目标 Y
    ladder_id: Optional[int] = None           # 梯子/绳索 ID (如果是爬梯边)
    is_rope: bool = False                     # 是否为绳索 (True: 绳索, False: 梯子)
    description: str = ""                     # 人类可读描述
    # 合并长平台只用于宏观寻路；真正执行跳跃时仍回到原始 foothold。
    # 以下字段描述一条“长平台边”展开后的短段级动作计划。旧边保持
    # None，状态机便会自动退回原有执行方式。
    source_foothold_id: Optional[int] = None
    target_foothold_id: Optional[int] = None
    takeoff_x: Optional[float] = None
    takeoff_x_range: Optional[Tuple[float, float]] = None
    landing_x_range: Optional[Tuple[float, float]] = None
    confidence: Optional[float] = None


class PlatformGraph:
    """平台有向拓扑图，提供路径规划与查询"""
    ladder_bottom_y_tolerance_px: float = 35.0
    def __init__(
        self,
        map_id: int = 0,
        map_name: str = "",
        vr_bounds: Optional[Dict[str, Any]] = None,
        minimap_meta: Optional[Dict[str, Any]] = None
    ):
        self.map_id = map_id
        self.map_name = map_name
        self.vr_bounds = vr_bounds or {}
        self.minimap_meta = minimap_meta or {}
        # 完整 foothold 拓扑（包含不可站立的墙和 1~3px 微型接缝）。
        # 平台节点只保存可站立线段，但 next/prev 遍历需要这些中间节点。
        self.foothold_lines_by_id: Dict[int, Dict[str, Any]] = {}
        # X 轴卷轴标定修正（单位：Canvas 像素）。同一地图可以累积多条
        # 已知绳梯样本，运行时取中位数，避免单次识别误差污染坐标。
        self.x_calibration_samples_px: List[float] = []
        # Y 轴同理使用水平平台标定；平台面与角色黄点存在固定角色高度差。
        self.y_calibration_samples_px: List[float] = []
        # 小地图 Canvas 与当前屏幕视口的最近一次可信卷轴偏移。
        # 纹理匹配偶发低置信度时保留它，避免退化公式在视口中部产生坐标突跳。
        self._last_minimap_offset_x: Optional[float] = None
        self._last_minimap_offset_y: Optional[float] = None
        # 部分 v83 小地图与 WZ Canvas 的灰度/色阶不完全一致，整体模板
        # 分数会长期低于 0.55，但真实候选位置在连续帧中非常稳定。分别
        # 记录轴向低分候选，达到连续性门槛后允许它替换陈旧 offset。
        self._low_conf_offset_x: Optional[float] = None
        self._low_conf_offset_y: Optional[float] = None
        self._low_conf_offset_x_hits: int = 0
        self._low_conf_offset_y_hits: int = 0
        self.nodes: Dict[int, PlatformNode] = {}
        self.ladder_ropes: Dict[int, LadderRopeNode] = {}  # id -> LadderRopeNode
        self.edges: Dict[int, List[PlatformEdge]] = {}  # from_id -> [PlatformEdge]
        # 原始 WZ portal 节点，供拓扑图显示传送点；type/pt 由地图数据保留。
        self.portals: List[Dict[str, Any]] = []

    @property
    def x_calibration_offset_px(self) -> float:
        if not self.x_calibration_samples_px:
            return 0.0
        return float(np.median(np.asarray(self.x_calibration_samples_px, dtype=np.float64)))

    def add_x_calibration_sample(self, offset_px: float) -> float:
        """追加一条 X 轴标定样本并返回当前中位数修正。"""
        self.x_calibration_samples_px.append(float(offset_px))
        self.x_calibration_samples_px = self.x_calibration_samples_px[-20:]
        return self.x_calibration_offset_px

    @property
    def y_calibration_offset_px(self) -> float:
        if not self.y_calibration_samples_px:
            return 0.0
        return float(np.median(np.asarray(self.y_calibration_samples_px, dtype=np.float64)))

    def add_y_calibration_sample(self, offset_px: float) -> float:
        """追加一条 Y 轴标定样本并返回当前中位数修正。"""
        self.y_calibration_samples_px.append(float(offset_px))
        self.y_calibration_samples_px = self.y_calibration_samples_px[-20:]
        return self.y_calibration_offset_px

    def reset_minimap_runtime_alignment(self) -> None:
        """清除只属于本次进入地图的小地图卷轴匹配状态。

        人工 X/Y 样本可能吸收标定当时小地图框相对真实 Canvas 的偏移，
        所以它只对当前框、本次进入地图有效。内存拓扑缓存再次启用时，
        标定样本与卷轴偏移都必须清空并依据新画面重新计算。
        """
        self.x_calibration_samples_px = []
        self.y_calibration_samples_px = []
        self._last_minimap_offset_x = None
        self._last_minimap_offset_y = None
        self._low_conf_offset_x = None
        self._low_conf_offset_y = None
        self._low_conf_offset_x_hits = 0
        self._low_conf_offset_y_hits = 0

    def add_node(self, node: PlatformNode):
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []

    def add_ladder_rope(self, lr: LadderRopeNode):
        self.ladder_ropes[lr.id] = lr

    def get_ladder_rope(self, lr_id: int) -> Optional[LadderRopeNode]:
        return self.ladder_ropes.get(lr_id)

    def add_edge(self, edge: PlatformEdge):
        if edge.from_id not in self.edges:
            self.edges[edge.from_id] = []
        self.edges[edge.from_id].append(edge)

    def get_node(self, node_id: int) -> Optional[PlatformNode]:
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: int) -> List[PlatformEdge]:
        return self.edges.get(node_id, [])

    def get_foothold_transition(
        self,
        from_id: int,
        to_id: int,
        action: Optional[str] = None,
    ) -> Optional[PlatformEdge]:
        """返回一条已展开到原始 foothold 的最高置信度平台转移。"""
        candidates = [
            edge for edge in self.get_edges_from(from_id)
            if edge.to_id == to_id
            and edge.source_foothold_id is not None
            and (action is None or edge.action == action)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda edge: float(edge.confidence or 0.0))

    def get_world_bounds(self) -> Tuple[int, int, int, int]:
        """获取地图物理世界坐标全范围 (x_min, x_max, y_min, y_max)"""
        if not self.nodes:
            return (-1000, 1000, -800, 300)
        # 必须基于原始 foothold 端点。合并平台的 node.y 是代表高度，
        # 会随链内台阶数量变化；若用它计算边界，切换“合并短平台”会让
        # 整张地图（尤其绳梯）在 Render 底图上发生整体 Y 偏移。
        xs: List[int] = []
        ys: List[int] = []
        for node in self.nodes.values():
            for line in node.raw_lines:
                try:
                    xs.extend((int(line['x1']), int(line['x2'])))
                    ys.extend((int(line['y1']), int(line['y2'])))
                except (KeyError, TypeError, ValueError):
                    continue
        if not xs:
            xs = [n.x_min for n in self.nodes.values()] + [n.x_max for n in self.nodes.values()]
            ys = [n.y for n in self.nodes.values()]
        return (min(xs), max(xs), min(ys), max(ys))

    @staticmethod
    def _decode_minimap_canvas_gray(
        canvas_str: str, background_gray: float = 80.0
    ) -> Optional[np.ndarray]:
        """解码 WZ 小地图，并按客户端的灰色底板合成透明像素。

        ``miniMap.canvas`` 是带 Alpha 的前景纹理。直接丢弃 Alpha 会把
        透明区错误变成纯黑色，而游戏实际把它绘制在约 80 灰度的面板上；
        对透明区域占比较高的地图，这会令正确屏幕框的背景匹配分从约
        0.76 降到 0.07，并在复检时错误删除黄框。
        """
        if not isinstance(canvas_str, str) or not canvas_str:
            return None
        try:
            import base64
            raw = base64.b64decode(canvas_str.split(",", 1)[-1])
            decoded = cv2.imdecode(
                np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED
            )
        except Exception:
            return None
        if decoded is None or decoded.size == 0:
            return None
        if decoded.ndim == 2:
            return np.ascontiguousarray(decoded.astype(np.uint8, copy=False))
        foreground_gray = cv2.cvtColor(decoded[:, :, :3], cv2.COLOR_BGR2GRAY)
        if decoded.shape[2] < 4:
            return foreground_gray
        alpha = decoded[:, :, 3].astype(np.float32) / 255.0
        # 不同客户端/小地图皮肤使用不同暗色底板；调用方可为同一 WZ
        # 前景生成多个背景假设，不依赖动态黄点/传送点图标。
        composited = (
            foreground_gray.astype(np.float32) * alpha
            + float(background_gray) * (1.0 - alpha)
        )
        return np.ascontiguousarray(np.rint(composited).astype(np.uint8))

    def get_minimap_canvas_size(self) -> Optional[Tuple[int, int]]:
        """返回 miniMap Canvas 的实际像素尺寸。"""
        mm = self.minimap_meta or {}
        full_gray = getattr(self, "_cached_full_canvas_gray", None)
        if full_gray is not None:
            return int(full_gray.shape[1]), int(full_gray.shape[0])
        try:
            explicit_width = int(mm.get("canvasWidth", 0) or 0)
            explicit_height = int(mm.get("canvasHeight", 0) or 0)
            if explicit_width > 0 and explicit_height > 0:
                return explicit_width, explicit_height
        except (TypeError, ValueError):
            pass
        canvas_str = mm.get("canvas")
        if isinstance(canvas_str, str) and canvas_str:
            gray = self._decode_minimap_canvas_gray(canvas_str)
            if gray is not None:
                self._cached_full_canvas_gray = gray
                return int(gray.shape[1]), int(gray.shape[0])
        try:
            mag = float(mm.get("magnification", 4) or 4)
            width = int(round(float(mm.get("width", 0)) / mag))
            height = int(round(float(mm.get("height", 0)) / mag))
            return (width, height) if width > 0 and height > 0 else None
        except (TypeError, ValueError):
            return None

    def get_minimap_canvas_gray(self) -> Optional[np.ndarray]:
        """返回解码后的 WZ 小地图灰度背景，供屏幕自动框选过滤前景。"""
        gray = getattr(self, "_cached_full_canvas_gray", None)
        if gray is None:
            canvas_str = (self.minimap_meta or {}).get("canvas")
            if isinstance(canvas_str, str) and canvas_str:
                gray = self._decode_minimap_canvas_gray(canvas_str)
                if gray is not None:
                    self._cached_full_canvas_gray = gray
        return gray if gray is not None and gray.size > 0 else None

    def get_minimap_canvas_gray_variants(self) -> Tuple[np.ndarray, ...]:
        """返回暗底板与灰底板两种 WZ 背景假设，供自动框选取最佳匹配。"""
        gray_80 = self.get_minimap_canvas_gray()
        if gray_80 is None:
            return ()
        cached = getattr(self, "_cached_minimap_gray_variants", None)
        if cached is None:
            canvas_str = (self.minimap_meta or {}).get("canvas")
            gray_20 = self._decode_minimap_canvas_gray(canvas_str, background_gray=20.0)
            cached = (gray_80, gray_20) if gray_20 is not None else (gray_80,)
            self._cached_minimap_gray_variants = cached
        return cached

    def minimap_norm_to_world(
        self,
        norm_x: float,
        norm_y: float,
        crop_gray_frame: Optional[np.ndarray] = None,
        debug_out: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        """
        官方 WZ miniMap 纹理画布 (Canvas) 视觉特征匹配与 2D 卷轴逆解引擎:
        - 具备 0.05ms 微秒级 WZ Canvas 纹理模板匹配能力，实时直读游戏小地图摄像机 Offset (ox, oy)
        - 无论是底层、在绳索上高频上下爬动、还是登顶天梯，均可实现 100% 精确的世界坐标直读
        - norm_x/norm_y 必须来自当帧黄点的原始亚像素质心；调用方不应传入
          EMA 平滑值，否则卷轴切换时会将历史视口位置带入 offset 换算。
        """
        mm = self.minimap_meta
        if mm and mm.get("width", 0) > 0 and mm.get("height", 0) > 0:
            cx = float(mm.get("centerX", 0))
            cy = float(mm.get("centerY", 0))
            w_wz = float(mm.get("width", 1))
            h_wz = float(mm.get("height", 1))

            # 1. 动态获取 WZ miniMap Canvas 纹理实际宽高与灰度缓存
            canvas_w, canvas_h = 0.0, 0.0
            full_gray = getattr(self, "_cached_full_canvas_gray", None)

            if full_gray is None:
                canvas_str = mm.get("canvas")
                if canvas_str and isinstance(canvas_str, str):
                    full_gray = self._decode_minimap_canvas_gray(canvas_str)
                    if full_gray is not None:
                        self._cached_full_canvas_gray = full_gray

            if full_gray is not None:
                canvas_h, canvas_w = float(full_gray.shape[0]), float(full_gray.shape[1])
            else:
                try:
                    canvas_w = float(mm.get("canvasWidth", 0) or 0)
                    canvas_h = float(mm.get("canvasHeight", 0) or 0)
                except (TypeError, ValueError):
                    canvas_w = canvas_h = 0.0
                if canvas_w <= 0 or canvas_h <= 0:
                    mag = float(mm.get("magnification", 4) or 4)
                    canvas_w = float(round(w_wz / mag))
                    canvas_h = float(round(h_wz / mag))

            # 可视小地图窗口尺寸应随当前地图/截屏动态确定。
            # 旧版固定 109x112 只适用于 101000102，会让其它地图的世界坐标比例失真。
            raw_view_w = raw_view_h = 0.0
            if crop_gray_frame is not None and crop_gray_frame.size > 0 and len(crop_gray_frame.shape) >= 2:
                visible_h, visible_w = crop_gray_frame.shape[:2]
                raw_view_w = float(visible_w)
                raw_view_h = float(visible_h)
                box_w = min(float(visible_w), canvas_w)
                box_h = min(float(visible_h), canvas_h)
            else:
                box_w = canvas_w
                box_h = canvas_h

            # norm_pos 是 Tracker 按屏幕内框原始尺寸归一化的，不能在
            # Canvas 比内框更矮/更窄时直接乘 box_*，否则会把黄点位置
            # 再压缩一次（例如 95px 视口映射到 86px Canvas 的 Y 轴）。
            # 先恢复黄点在当前屏幕视口中的实际像素，再交给卷轴/Canvas
            # 映射；没有传入截图时才退回 box_*。
            px = norm_x * (raw_view_w if raw_view_w > 0 else box_w)
            py = norm_y * (raw_view_h if raw_view_h > 0 else box_h)

            offset_x = 0.0
            offset_y = 0.0
            # 0 是合法的模板匹配位置，不能再用 offset==0 判断“未匹配”。
            matched_x = False
            matched_y = False
            sx = w_wz / float(canvas_w)
            sy = h_wz / float(canvas_h)

            # 2. 若传入当前小地图可视区域灰度帧，优先执行微秒级特征匹配直读精确 offset
            # X/Y 卷轴必须独立处理：横向卷轴地图不能让一次二维匹配的
            # Y 结果干扰 X 偏移；固定轴始终保持 offset=0。
            scroll_x = canvas_w > box_w + 5.0
            scroll_y = canvas_h > box_h + 5.0
            has_scroll = scroll_x or scroll_y
            if has_scroll and full_gray is not None and crop_gray_frame is not None and crop_gray_frame.size > 0:
                try:
                    # 屏幕端检测到的内框有时会比 WZ Canvas 多出 1~9px 的
                    # 底部/右侧描边。旧逻辑要求二维尺寸完全不超出，横向卷轴
                    # 图便会完全跳过模板匹配，退回“黄点位于中间就线性推 offset”
                    # 的粗略公式，角色经过视口中间时世界 X 因而突跳。
                    # 对齐左上有效画布，只裁掉超出的尾部边框；不改变黄点的
                    # 原始归一化 X，也不影响另一个固定轴的换算比例。
                    match_h = min(int(full_gray.shape[0]), int(crop_gray_frame.shape[0]))
                    match_w = min(int(full_gray.shape[1]), int(crop_gray_frame.shape[1]))
                    match_crop = crop_gray_frame[:match_h, :match_w]
                    if match_h >= 8 and match_w >= 8 and full_gray.shape[0] >= match_h and full_gray.shape[1] >= match_w:
                        match_res = cv2.matchTemplate(full_gray, match_crop, cv2.TM_CCOEFF_NORMED)
                        _, max_score, _, max_loc = cv2.minMaxLoc(match_res)
                        # 低分匹配在纵向长地图里很容易落到相似的暗色纹理段。
                        # 只有可信匹配才更新偏移；短暂失配沿用最近一次可信偏移，
                        # 不在视口中部退回线性估算而造成 Y 坐标跳变。
                        match_threshold = 0.55
                        low_conf_temporal_x = False
                        low_conf_temporal_y = False
                        if max_score >= match_threshold:
                            if scroll_x:
                                offset_x = float(max_loc[0])
                                matched_x = True
                                self._last_minimap_offset_x = offset_x
                                self._low_conf_offset_x = offset_x
                                self._low_conf_offset_x_hits = 0
                            if scroll_y:
                                offset_y = float(max_loc[1])
                                matched_y = True
                                self._last_minimap_offset_y = offset_y
                                self._low_conf_offset_y = offset_y
                                self._low_conf_offset_y_hits = 0
                        else:
                            # 低分候选本身不能单帧直接采信；但当同一轴的
                            # 偏移连续三帧稳定在 ±1px 内时，它比“停在地图
                            # 底部时留下的旧 offset”更可信。P26 时实测
                            # score≈0.23、max_loc.y=76 连续稳定，正属此例。
                            if scroll_x:
                                proposed_x = float(max_loc[0])
                                if (self._low_conf_offset_x is not None
                                        and abs(proposed_x - self._low_conf_offset_x) <= 1.0):
                                    self._low_conf_offset_x_hits += 1
                                else:
                                    self._low_conf_offset_x = proposed_x
                                    self._low_conf_offset_x_hits = 1
                                if self._low_conf_offset_x_hits >= 3:
                                    offset_x = proposed_x
                                    matched_x = True
                                    low_conf_temporal_x = True
                                    self._last_minimap_offset_x = offset_x
                            if scroll_x and self._last_minimap_offset_x is not None:
                                if not matched_x:
                                    offset_x = self._last_minimap_offset_x
                                    matched_x = True
                            if scroll_y:
                                proposed_y = float(max_loc[1])
                                if (self._low_conf_offset_y is not None
                                        and abs(proposed_y - self._low_conf_offset_y) <= 1.0):
                                    self._low_conf_offset_y_hits += 1
                                else:
                                    self._low_conf_offset_y = proposed_y
                                    self._low_conf_offset_y_hits = 1
                                if self._low_conf_offset_y_hits >= 3:
                                    offset_y = proposed_y
                                    matched_y = True
                                    low_conf_temporal_y = True
                                    self._last_minimap_offset_y = offset_y
                            if scroll_y and self._last_minimap_offset_y is not None:
                                if not matched_y:
                                    offset_y = self._last_minimap_offset_y
                                    matched_y = True
                        if debug_out is not None:
                            debug_out.update({
                                "offset_x": int(offset_x),
                                "offset_y": int(offset_y),
                                "match_score": max_score,
                                "match_reason": (
                                    "axis_independent_tm"
                                    if max_score >= match_threshold
                                    else ("low_confidence_temporal"
                                          if (low_conf_temporal_x or low_conf_temporal_y)
                                          else "low_confidence_use_last_good")
                                ),
                                "candidate_offset_x": int(max_loc[0]),
                                "candidate_offset_y": int(max_loc[1]),
                                "match_threshold": match_threshold,
                                "candidate_count": 1,
                                "scroll_x": scroll_x,
                                "scroll_y": scroll_y,
                                "matched_x": matched_x,
                                "matched_y": matched_y,
                                "canvas_w": int(canvas_w),
                                "canvas_h": int(canvas_h),
                                "view_w": int(box_w),
                                "view_h": int(box_h),
                            })
                except Exception:
                    pass

            # 3. 几何运动学连续插值兜底
            if not matched_x and canvas_w > box_w + 5.0:
                if px >= box_w * 0.55:
                    offset_x = canvas_w - box_w
                elif px <= box_w * 0.45:
                    offset_x = 0.0
                else:
                    ratio_x = (px - box_w * 0.45) / max(0.01, box_w * 0.10)
                    offset_x = (canvas_w - box_w) * ratio_x

            if not matched_y and canvas_h > box_h + 5.0:
                if py >= box_h * 0.55:
                    offset_y = canvas_h - box_h
                elif py <= box_h * 0.45:
                    offset_y = 0.0
                else:
                    ratio_y = (py - box_h * 0.45) / max(0.01, box_h * 0.10)
                    offset_y = (canvas_h - box_h) * ratio_y

            # 模板匹配得到的是当前镜头偏移；标定值修正的是该偏移的
            # 系统误差，因此在恢复世界坐标前叠加（不改变黄点识别）。
            canvas_x = offset_x + px + self.x_calibration_offset_px
            canvas_y = offset_y + py + self.y_calibration_offset_px

            # 世界坐标恢复为启用亚像素前的整数换算行为。
            # 亚像素仍保留在 TrackerResult/F8 调试界面，但不直接改变
            # 现有导航、平台吸附和路线控制使用的世界坐标离散值。
            world_x = int(round(canvas_x * sx - cx))
            world_y = int(round(canvas_y * sy - cy))
            if debug_out is not None and "match_reason" not in debug_out:
                debug_out.update({
                    "offset_x": int(offset_x),
                    "offset_y": int(offset_y),
                    "match_score": None,
                    "match_reason": "geometry_fallback",
                    "candidate_count": 0,
                })
            elif debug_out is not None:
                # 记录最终采用的偏移，便于诊断；匹配到 (0, y) 也应明确标记为成功。
                debug_out.update({
                    "offset_x": int(offset_x),
                    "offset_y": int(offset_y),
                    "matched_x": matched_x,
                    "matched_y": matched_y,
                })
            # 标定和诊断始终需要这些尺寸，不能只在模板匹配分支提供。
            if debug_out is not None:
                debug_out.update({
                    "canvas_w": int(canvas_w),
                    "canvas_h": int(canvas_h),
                    "view_w": int(raw_view_w if raw_view_w > 0 else box_w),
                    "view_h": int(raw_view_h if raw_view_h > 0 else box_h),
                    "scroll_x": bool(scroll_x),
                    "scroll_y": bool(scroll_y),
                    "calibration_offset_x": self.x_calibration_offset_px,
                    "calibration_offset_y": self.y_calibration_offset_px,
                })
            return world_x, world_y

        # 回退 1: 若无 miniMap，使用官方 vrBounds
        vr = self.vr_bounds
        if vr and not vr.get("isEmpty", True) and vr.get("width", 0) > 0:
            vx = vr.get("left", vr.get("x", 0))
            vy = vr.get("top", vr.get("y", 0))
            vw = vr.get("width", 1)
            vh = vr.get("height", 1)
            world_x = int(vx + norm_x * vw)
            world_y = int(vy + norm_y * vh)
            return world_x, world_y

        # 兜底 2: 使用 Foothold 物理外接包围盒
        x_min, x_max, y_min, y_max = self.get_world_bounds()
        world_x = int(x_min + norm_x * (x_max - x_min))
        world_y = int(y_min + norm_y * (y_max - y_min))
        return world_x, world_y

    def get_snapped_player_world_pos(
        self,
        raw_wx: int,
        raw_wy: int,
        preferred_platform_id: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        物理平台表面与梯绳中轴高精度吸附校准 (彻底消除小地图卷轴非线性 Y 轴漂移):
        1. 若水平落入某个物理平台，强制将 Y 吸附至平台承重线 (node.y - 45)
        2. 若处于梯子/绳索覆盖区，强制将 X 吸附至梯绳垂直中轴 (lr.x)
        """
        # 必须先用未吸附坐标判断攀爬状态。旧逻辑先匹配平台并把 Y 改成
        # surface-45，导致角色尚在梯顶翻越时提前跳到上层平台。
        p, lr, is_climbing = self.find_player_location(
            raw_wx,
            raw_wy,
            ladder_tolerance_x=30,
            preferred_platform_id=preferred_platform_id,
        )

        snapped_x = raw_wx
        snapped_y = raw_wy

        if is_climbing and lr is not None:
            # 攀爬及梯顶翻越阶段只吸附 X，中途绝不能改写连续上升的 Y。
            snapped_x = lr.x
            snapped_y = raw_wy
        elif p is not None:
            # 站立在平台上：X 为真实水平移动坐标 raw_wx (绝不可被梯子中轴吸附锁死！)，Y 严格锁定到平台表面上方 45px 角色重心
            snapped_x = raw_wx
            # 刚从梯顶退出时保留原始 Y，否则 surface-45 可能再次落入
            # 梯顶判定区并在下一次状态计算中反复切回“攀爬”。
            if lr is not None and p.id == lr.top_platform_id:
                snapped_y = raw_wy
            else:
                snapped_y = int(round(p.surface_y_at(raw_wx))) - 45
        elif lr is not None:
            # 正在梯绳上爬行：X 锁定梯绳中轴，Y 限制在梯绳长度范围内
            snapped_x = lr.x
            y_min = min(lr.y1, lr.y2)
            y_max = max(lr.y1, lr.y2)
            snapped_y = max(y_min, min(raw_wy, y_max))

        return (snapped_x, snapped_y)

    def find_player_platform(
        self,
        player_x: int,
        player_y: int,
        tolerance_y: int = 110,
        tolerance_x: int = 35,
        preferred_platform_id: Optional[int] = None,
        preferred_cost_slack: float = 18.0,
    ) -> Optional[PlatformNode]:
        """
        基于 2D 重力承重法则的物理平台判定:
        1. 严格使用游戏真实世界物理坐标 (x, y)
        2. 重力法则: 角色只能踩在自己脚底附近的承重水平面上 (node.y 在 player_y 下方，即 drop = node.y - player_y >= -15)
        3. 彻底排除头顶上方平台 (drop < -15)，杜绝在下层平台靠近上层平台下方时被误吸入上层天花板
        4. 优先选取垂直距离最贴近脚底站立线 (~45px) 的承重面
        """
        candidates = []

        for node in self.nodes.values():
            # 1. 水平包围检测 (允许边缘悬空 tolerance_x)
            if (node.x_min - tolerance_x) <= player_x <= (node.x_max + tolerance_x):
                # 2. 使用当前 X 对应的真实线段高度；合并台阶的 avg_y
                # 只能用于排序，不能用于玩家承重/吸附判断。
                surface_y = node.surface_y_at(player_x)
                drop = surface_y - player_y
                if drop < -15:
                    # 平台在角色身体上方 15px 以上，属于头顶天花板，一票否决！
                    continue

                if drop <= tolerance_y:
                    # 计算 X 轴边缘外溢惩罚 (优先选择角色真实身处其范围内的平台)
                    if player_x < node.x_min:
                        x_pen = (node.x_min - player_x) * 4.0
                    elif player_x > node.x_max:
                        x_pen = (player_x - node.x_max) * 4.0
                    else:
                        x_pen = 0.0

                    cost = abs(drop - 45) + x_pen
                    candidates.append((node, cost))

        if candidates:
            # 严格选择最贴合脚底的真实承重面
            candidates.sort(key=lambda item: item[1])
            # 多条合并长平台在斜坡/台阶末端可能上下重叠，黄点 Y 又是
            # 量化观测。若两个候选只差一格以内，单帧最小 cost 会让人物
            # 在完全没有执行跳跃时从源平台瞬移到目标平台。保留仍然物理
            # 合法的上一承重平台；一旦真实起跳/下落使它超出候选范围，
            # 便立即切换，不会阻挡真正的平台转移。
            if preferred_platform_id is not None:
                best_cost = float(candidates[0][1])
                for node, cost in candidates:
                    if (
                        int(node.id) == int(preferred_platform_id)
                        and float(cost) <= best_cost + max(0.0, float(preferred_cost_slack))
                    ):
                        return node
            return candidates[0][0]
        return None

    def find_player_ladder_rope(
        self,
        player_x: int,
        player_y: int,
        tolerance_x: int = 16,
        tolerance_y: Optional[float] = None,
        top_tolerance_y: int = 75,
    ) -> Optional[LadderRopeNode]:
        """
        判定角色当前是否对齐或攀爬在某根梯子/绳索上:
        1. X 轴贴合: abs(player_x - lr.x) <= tolerance_x (通常梯绳中轴 +-15px 内)
        2. Y 轴高度覆盖: 顶端额外覆盖翻越区，底端使用常规 tolerance_y
        """
        if tolerance_y is None:
            tolerance_y = float(self.ladder_bottom_y_tolerance_px)
        candidates = []
        for lr in self.ladder_ropes.values():
            dx = abs(player_x - lr.x)
            if dx <= tolerance_x:
                y_min = min(lr.y1, lr.y2)
                y_max = max(lr.y1, lr.y2)
                # 梯顶退出时角色中心需要继续上升一段距离才能真正落到
                # 上层平台；顶端容差必须大于底端容差，覆盖翻越过程。
                if (y_min - top_tolerance_y) <= player_y <= (y_max + tolerance_y):
                    # 代价: X 越贴合中轴且 Y 越居中越优先
                    cost = dx
                    candidates.append((lr, cost))

        if candidates:
            candidates.sort(key=lambda item: item[1])
            return candidates[0][0]
        return None

    def find_player_location(
        self,
        player_x: int,
        player_y: int,
        ladder_tolerance_x: int = 16,
        preferred_platform_id: Optional[int] = None,
    ) -> Tuple[Optional[PlatformNode], Optional[LadderRopeNode], bool]:
        """
        综合解算角色位置与状态:
        返回: (platform_node, ladder_rope_node, is_climbing)
        - 若踩在实体平台上: is_climbing = False, platform = node
        - 若攀爬在绳索/梯子上: is_climbing = True, ladder_rope = lr
        """
        p = self.find_player_platform(
            player_x,
            player_y,
            preferred_platform_id=preferred_platform_id,
        )
        lr = self.find_player_ladder_rope(
            player_x, player_y, tolerance_x=ladder_tolerance_x
        )

        if lr is not None:
            if p is not None:
                drop = p.surface_y_at(player_x) - player_y
                top_y = min(lr.y1, lr.y2)
                top_clearance = top_y - player_y
                if p.id == lr.top_platform_id:
                    # 上层平台：只要角色中心尚未充分越过梯顶，就仍是
                    # 攀爬/翻越状态。101000000 的梯8在 -688 到 -718
                    # 正属于此区间，约 -750 才真正站上 P20。
                    #
                    # 但必须先承认真实的站立高度。斜坡平台在梯轴处的
                    # surface_y 与合并后的 avg_y 可能相差很大；例如同图
                    # 35号绳顶端的 P81，角色中心 y=-3844、脚底距真实
                    # 表面约51px，已经稳定站立。旧的单一
                    # ``top_clearance < 55`` 会把它永久判成攀爬并令 FSM
                    # 一直按住 UP。
                    if 45 <= drop <= 68:
                        return (p, lr, False)
                    if top_clearance < 55:
                        return (p, lr, True)
                    return (p, lr, False)
                if p.id == lr.bottom_platform_id:
                    # 下层平台：正常站立高度优先，防止角色站在梯脚时
                    # 因 X 恰好对齐梯轴而被误认为正在攀爬。
                    if 25 <= drop <= 68:
                        return (p, lr, False)
                    return (p, lr, True)
                # 平台承重高度优先于“贴近绳轴”的几何关系。绳梯端点
                # 常常就在平台边缘/中部：例如 19 号绳的下端紧邻 P61，
                # 角色站在 P61 上时 X 可以恰好等于绳轴，但这不是攀爬。
                # 旧逻辑把 dx <= 10 无条件视作攀爬，导致 FSM 无限按 UP。
                if 25 <= drop <= 65:
                    return (p, lr, False)
                if drop < 20 or drop > 68:
                    # 已偏离正常站立高度，且仍处于梯绳有效范围，才认定
                    # 为真正攀爬。
                    return (p, lr, True)
            else:
                # 空中且在梯绳范围内 -> 100% 攀爬中
                return (None, lr, True)

        return (p, None, False)

    def find_path(
        self,
        from_id: int,
        to_id: int,
        allow_run_jump: bool = True,
        edge_penalty_fn: Optional[Any] = None,
        allow_portal: bool = True,
    ) -> List[PlatformEdge]:
        """
        使用 A* / Dijkstra 算法寻找从平台 A 到平台 B 的最优动作序列
        """
        if from_id == to_id or from_id not in self.nodes or to_id not in self.nodes:
            return []

        # 字典序 Dijkstra：首先严格最小化预计动作时间（所有边成本按
        # 0.01 单位整数化，消除浮点累计误差），同成本时依次选择更少
        # 的绳索和更少的动作段。这样不会为了少一根绳而接受明显绕路，
        # 但所有等长最短路径中一定选绳索最少的一条。
        # queue item: ((cost_centis, rope_count, hop_count), node_id, path)
        start_metric = (0, 0, 0)
        queue = [(start_metric, from_id, [])]
        visited_metrics = {from_id: start_metric}

        while queue:
            current_metric, curr_id, path = heapq.heappop(queue)

            if curr_id == to_id:
                return path

            if current_metric != visited_metrics.get(curr_id):
                continue

            for edge in self.get_edges_from(curr_id):
                if self._jump_landing_shadowed_by_overhead(edge):
                    continue
                # 这里只过滤地图内部的 PORTAL 动作边。跨地图 type=2
                # 出口由 WorldPatrolController 单独编排，不经过本图 A*。
                if not allow_portal and str(edge.action) == "PORTAL":
                    continue
                # 绳梯位于当前平台边界外的直达边，是专门给“跑跳抓取”
                # 使用的捷径。关闭跑跳时必须排除它，让寻路器回到绳梯
                # 底端平台，再使用普通跳抓，而不是执行必然失败的原地跳。
                if not allow_run_jump and edge.action.startswith("JUMP_CLIMB"):
                    src = self.get_node(edge.from_id)
                    if src is not None and edge.trigger_x is not None:
                        if edge.trigger_x < src.x_min - 20 or edge.trigger_x > src.x_max + 20:
                            continue
                next_id = edge.to_id
                runtime_penalty = 0.0
                if edge_penalty_fn is not None:
                    try:
                        runtime_penalty = max(0.0, float(edge_penalty_fn(edge)))
                    except Exception:
                        runtime_penalty = 0.0
                # An infinite runtime penalty means the action is temporarily
                # unavailable, not merely expensive. A finite penalty still
                # allows Dijkstra to replay it when every exit has failed.
                if not math.isfinite(runtime_penalty):
                    continue
                edge_cost = max(
                    0,
                    int(round((float(edge.cost) + runtime_penalty) * 100.0)),
                )
                uses_rope = int(bool(edge.is_rope or "ROPE" in edge.action))
                new_metric = (
                    current_metric[0] + edge_cost,
                    current_metric[1] + uses_rope,
                    current_metric[2] + 1,
                )

                if new_metric < visited_metrics.get(
                    next_id, (math.inf, math.inf, math.inf)
                ):
                    visited_metrics[next_id] = new_metric
                    heapq.heappush(queue, (new_metric, next_id, path + [edge]))

        return []  # 无法连通

    def find_path_from_position(
        self,
        from_id: int,
        from_x: float,
        to_id: int,
        allow_run_jump: bool = True,
        edge_penalty_fn: Optional[Any] = None,
        allow_portal: bool = True,
        return_metric: bool = False,
    ) -> Any:
        """按当前世界 X 和每条边的落点寻路，供 F6 巡逻及临时目标使用。

        同一长平台的不同落点不是同一状态；否则一次极远的横向
        移动会被错误地算作零成本，休息途中便会绕过整张地图。
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return ([], None) if return_metric else []
        if from_id == to_id:
            return ([], (0, 0, 0)) if return_metric else []
        speed = 250.0 if getattr(self, "enable_teleport", False) else 125.0
        sequence = 0
        origin_x = float(from_x)
        start = (0, 0, 0)
        queue = [(start, sequence, int(from_id), origin_x, [])]
        best = {(int(from_id), round(origin_x / 5.0)): start}
        target_node = self.get_node(to_id)

        while queue:
            metric, _sequence, node_id, current_x, route = heapq.heappop(queue)
            state = (node_id, round(current_x / 5.0))
            if metric != best.get(state):
                continue
            if node_id == to_id:
                return (route, metric) if return_metric else route
            for edge in self.get_edges_from(node_id):
                if self._jump_landing_shadowed_by_overhead(edge):
                    continue
                if not allow_portal and edge.action == "PORTAL":
                    continue
                if not allow_run_jump and edge.action.startswith("JUMP_CLIMB"):
                    source = self.get_node(edge.from_id)
                    if source is not None and edge.trigger_x is not None and (
                        edge.trigger_x < source.x_min - 20
                        or edge.trigger_x > source.x_max + 20
                    ):
                        continue
                destination = self.get_node(edge.to_id)
                if destination is None:
                    continue
                if edge.action == "DOWN_JUMP" and edge.trigger_x_range:
                    lo, hi = sorted(map(float, edge.trigger_x_range))
                    departure_x = max(lo, min(hi, current_x))
                else:
                    departure_x = float(
                        edge.takeoff_x if edge.takeoff_x is not None else
                        edge.trigger_x if edge.trigger_x is not None else current_x
                    )
                if edge.landing_x is not None:
                    arrival_x = float(edge.landing_x)
                elif edge.ladder_id is not None and edge.ladder_id in self.ladder_ropes:
                    arrival_x = float(self.ladder_ropes[edge.ladder_id].x)
                else:
                    arrival_x = max(float(destination.x_min), min(float(destination.x_max), departure_x))
                extra = 0.0
                if edge_penalty_fn is not None:
                    try:
                        extra = max(0.0, float(edge_penalty_fn(edge)))
                    except Exception:
                        pass
                if not math.isfinite(extra):
                    continue
                # 两个平台间动作成本之外，还要计入从实际落点到下一
                # 起跳线的行走，以及边自身的横向位移。
                travel = (abs(current_x - departure_x) + abs(arrival_x - departure_x)) / speed
                if edge.to_id == to_id:
                    travel += abs(arrival_x - float(target_node.center_x)) / speed
                cost = max(0, int(round((float(edge.cost) + extra + travel) * 100.0)))
                new_metric = (
                    metric[0] + cost,
                    metric[1] + int(bool(edge.is_rope or "ROPE" in edge.action)),
                    metric[2] + 1,
                )
                next_state = (int(edge.to_id), round(arrival_x / 5.0))
                if new_metric < best.get(next_state, (math.inf, math.inf, math.inf)):
                    best[next_state] = new_metric
                    sequence += 1
                    heapq.heappush(
                        queue, (new_metric, sequence, int(edge.to_id), arrival_x, route + [edge])
                    )
        return ([], None) if return_metric else []

    def _jump_landing_shadowed_by_overhead(self, edge: PlatformEdge) -> bool:
        """排除会先落到目标正上方整片平台的上跳边。

        例如环乘区 P17→P19：P19 仅比横跨其上方的 P20 低
        20px；实际两次跳跃都落回 P20。只拒绝上层平台完整覆盖
        目标并留有边距的情况，避免把普通局部交叠误判成遮挡。
        """
        if not str(edge.action).startswith("JUMP"):
            return False
        source = self.get_node(edge.from_id)
        target = self.get_node(edge.to_id)
        if source is None or target is None or target.y >= source.y:
            return False
        for overhead in self.nodes.values():
            if overhead.id in (source.id, target.id):
                continue
            if not (0 < target.y - overhead.y <= 30):
                continue
            if not (source.y - 130 <= overhead.y < source.y):
                continue
            if (
                overhead.x_min <= target.x_min - 12
                and overhead.x_max >= target.x_max + 12
            ):
                return True
        return False

    @staticmethod
    def path_metrics(path: List[PlatformEdge]) -> Dict[str, Any]:
        """返回与 find_path 目标函数一致的路线统计。"""
        return {
            "estimated_cost": round(sum(float(edge.cost) for edge in path), 2),
            "rope_count": sum(
                1 for edge in path if edge.is_rope or "ROPE" in edge.action
            ),
            "hop_count": len(path),
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典/JSON"""
        return {
            "map_id": self.map_id,
            "map_name": self.map_name,
            "total_platforms": len(self.nodes),
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for edges in self.edges.values() for e in edges],
            "portals": list(self.portals),
        }

    def to_cache_dict(self) -> Dict[str, Any]:
        """返回可持久化到 JSON 的完整拓扑缓存内容。"""
        return {
            "cache_schema": 5,
            "map_id": self.map_id,
            "map_name": self.map_name,
            "vr_bounds": self.vr_bounds,
            "minimap_meta": self.minimap_meta,
            "foothold_lines_by_id": {
                str(k): value for k, value in self.foothold_lines_by_id.items()
            },
            "nodes": [asdict(node) for node in self.nodes.values()],
            "ladder_ropes": [asdict(rope) for rope in self.ladder_ropes.values()],
            "edges": [asdict(edge) for edges in self.edges.values() for edge in edges],
            "portals": list(self.portals),
        }

    @classmethod
    def from_cache_dict(cls, payload: Dict[str, Any]) -> "PlatformGraph":
        """从 ``to_cache_dict`` 生成的 JSON 内容恢复完整拓扑对象。"""
        if not isinstance(payload, dict) or int(payload.get("cache_schema", 0)) != 5:
            raise ValueError("不支持的拓扑缓存版本")

        graph = cls(
            map_id=int(payload.get("map_id", 0) or 0),
            map_name=str(payload.get("map_name", "") or ""),
            vr_bounds=payload.get("vr_bounds") or {},
            minimap_meta=payload.get("minimap_meta") or {},
        )
        graph.portals = [
            dict(portal) for portal in (payload.get("portals", []) or [])
            if isinstance(portal, dict)
        ]

        foothold_lines = payload.get("foothold_lines_by_id", {}) or {}
        if isinstance(foothold_lines, dict):
            graph.foothold_lines_by_id = {
                int(key): value for key, value in foothold_lines.items()
                if isinstance(value, dict)
            }

        for node_data in payload.get("nodes", []) or []:
            if isinstance(node_data, dict):
                graph.add_node(PlatformNode(**node_data))

        for rope_data in payload.get("ladder_ropes", []) or []:
            if isinstance(rope_data, dict):
                graph.add_ladder_rope(LadderRopeNode(**rope_data))

        tuple_fields = {
            "trigger_x_range", "takeoff_x_range", "landing_x_range"
        }
        for edge_data in payload.get("edges", []) or []:
            if not isinstance(edge_data, dict):
                continue
            edge_data = dict(edge_data)
            for field_name in tuple_fields:
                value = edge_data.get(field_name)
                if isinstance(value, list):
                    edge_data[field_name] = tuple(value)
            graph.add_edge(PlatformEdge(**edge_data))
        return graph

    def world_to_render_pixel(self, wx: int, wy: int, img_w: int, img_h: int) -> Tuple[int, int]:
        """将物理世界坐标精确转换为全景拓扑图底图像素坐标"""
        if not self.nodes:
            return (0, 0)
        world_x_min, _world_x_max, _world_y_min, world_y_max = self.get_world_bounds()
        vr = self.vr_bounds
        if vr and not vr.get('isEmpty', True) and vr.get('width', 0) > 0:
            x_min = vr.get('left', vr.get('x', 0))
            y_min = vr.get('top', vr.get('y', 0))
        else:
            x_min = world_x_min
            y_min = world_y_max - (img_h - 107)
        px = int(wx - x_min)
        py = int(wy - y_min)
        return max(0, min(img_w - 1, px)), max(0, min(img_h - 1, py))

    def render_topology_image(
        self,
        player_pos: Optional[Tuple[int, int]] = None,
        active_path: Optional[List[PlatformEdge]] = None,
        highlight_platforms: Optional[List[int]] = None,
        title: str = "",
        show_platform_edges: bool = True,
        draw_labels: bool = True,
    ) -> Optional[np.ndarray]:
        """
        将当前地图的平台与梯绳拓扑结构叠加在官方 Render 高清全景大地图上渲染 (严密像素对齐)
        """
        if not self.nodes:
            return None
        world_x_min, world_x_max, world_y_min, world_y_max = self.get_world_bounds()

        # 1. 尝试加载/下载在线后备 render 全景实景大地图
        bg_render: Optional[np.ndarray] = None
        if self.map_id and self.map_id > 0:
            # 拓扑底图单独放在当前项目的本地缓存目录，避免依赖旧项目
            # Legacy absolute-path fallback removed; resolve local PNGs relative to the app root,
            # 只有缓存不存在时才访问网络。
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(project_root, "assets", "map_renders")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{self.map_id}_render.png")
            if not os.path.exists(cache_path):
                try:
                    url = f"https://maplestory.io/api/gms/83/map/{self.map_id}/render"
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        with open(cache_path, "wb") as f:
                            f.write(resp.read())
                except Exception:
                    pass

            if os.path.exists(cache_path):
                try:
                    loaded = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)
                    if loaded is not None:
                        if len(loaded.shape) == 3 and loaded.shape[2] == 4:
                            alpha = loaded[:, :, 3] / 255.0
                            bgr = loaded[:, :, :3]
                            bg_black = np.full_like(bgr, (25, 28, 35))
                            bg_render = (bgr * alpha[:, :, None] + bg_black * (1.0 - alpha[:, :, None])).astype(np.uint8)
                        else:
                            bg_render = loaded[:, :, :3]
                except Exception:
                    pass

        # 2. 精准计算物理世界坐标 -> Render 图像像素坐标映射 (严格保持 1:1 物理尺度，严禁非等比拉伸)
        if bg_render is not None:
            canvas = cv2.convertScaleAbs(bg_render, alpha=0.90, beta=-5)
            ih, iw = canvas.shape[:2]
            def to_cv(x, y):
                return self.world_to_render_pixel(x, y, iw, ih)
        else:
            x_min = world_x_min - 40
            y_min = world_y_min - 60
            width = int(world_x_max - world_x_min + 80)
            height = int(world_y_max - world_y_min + 120)
            canvas = np.full((height, width, 3), (22, 24, 30), dtype=np.uint8)
            def to_cv(x, y):
                return int(x - x_min), int(y - y_min)

        def draw_motion_curve(start_xy, end_xy, action, color, thickness):
            """绘制符合运动模型的二次贝塞尔轨迹，而不是直线连边。

            普通跳跃采用向上的抛物线；长距下落采用向下展开的弧线；
            DOWN_JUMP/JUMP_UP 保持近似垂直并只给出轻微弯曲。
            """
            x1, y1 = float(start_xy[0]), float(start_xy[1])
            x2, y2 = float(end_xy[0]), float(end_xy[1])

            # 垂直台阶跳/瞬移只改变 Y，不施加水平位移，使用直线表示。
            if action in ("JUMP_UP", "TELEPORT_UP"):
                pt1, pt2 = to_cv(x1, y1), to_cv(x2, y2)
                cv2.arrowedLine(canvas, pt1, pt2, color, thickness,
                                cv2.LINE_AA, tipLength=0.10)
                return

            dx, dy = x2 - x1, y2 - y1
            if "LONG_DROP" in action or "_DROP" in action or action == "DROP":
                # 自然下落/长距下落：先离开起跳边缘，再沿重力方向下坠。
                direction = 1.0 if dx >= 0 else -1.0
                cx = x1 + dx * 0.58 + direction * min(28.0, abs(dx) * 0.10)
                cy = y1 + dy * 0.22
            elif action == "DOWN_JUMP":
                cx = x1 + dx * 0.18
                cy = y1 + dy * 0.42
            elif action in ("JUMP_UP", "TELEPORT_UP"):
                cx = x1 + dx * 0.50
                cy = min(y1, y2) - max(24.0, abs(dx) * 0.18)
            else:
                # JUMP_LEFT/JUMP_RIGHT：起跳后上升，越过顶点后落到目标平台。
                cx = x1 + dx * 0.50
                cy = min(y1, y2) - max(28.0, abs(dx) * 0.34)

            points = []
            for i in range(25):
                t = i / 24.0
                omt = 1.0 - t
                wx = omt * omt * x1 + 2.0 * omt * t * cx + t * t * x2
                wy = omt * omt * y1 + 2.0 * omt * t * cy + t * t * y2
                points.append(to_cv(wx, wy))
            pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)
            # 只在末端画箭头，避免把整条曲线覆盖成直线。
            cv2.arrowedLine(canvas, points[-2], points[-1], color, thickness,
                            cv2.LINE_AA, tipLength=0.45)

        # 1. 绘制普通动作边 (下跳、平跳、瞬移)
        active_keys = None
        if active_path:
            active_keys = {(e.from_id, e.to_id, e.action, e.trigger_x) for e in active_path}
        for from_id, edges in (self.edges.items() if show_platform_edges else ()):
            p_from = self.get_node(from_id)
            if not p_from: continue
            for edge in edges:
                if active_keys is not None and (edge.from_id, edge.to_id, edge.action, edge.trigger_x) not in active_keys:
                    continue
                p_to = self.get_node(edge.to_id)
                if not p_to: continue

                is_active = False
                if active_path:
                    is_active = any(e.from_id == edge.from_id and e.to_id == edge.to_id for e in active_path)

                if edge.action == "PORTAL":
                    # 传送不是物理运动，用紫色虚线表示瞬移连接。
                    source_x = edge.trigger_x if edge.trigger_x is not None else p_from.center_x
                    p1 = to_cv(source_x, p_from.surface_y_at(source_x))
                    p2 = to_cv(p_to.center_x, p_to.center_y)
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    dist = max(1.0, float((dx * dx + dy * dy) ** 0.5))
                    for t in np.arange(0.0, 1.0, 10.0 / dist):
                        if int(t * dist / 10.0) % 2 == 0:
                            a = (int(p1[0] + dx * t), int(p1[1] + dy * t))
                            b = (int(p1[0] + dx * min(1.0, t + 5.0 / dist)),
                                 int(p1[1] + dy * min(1.0, t + 5.0 / dist)))
                            cv2.line(canvas, a, b, (255, 0, 255), 3 if is_active else 2, cv2.LINE_AA)
                    cv2.arrowedLine(canvas, (int(p1[0] + dx * 0.9), int(p1[1] + dy * 0.9)), p2,
                                    (255, 0, 255), 3 if is_active else 2, cv2.LINE_AA, tipLength=0.18)
                elif edge.action == "DOWN_JUMP":
                    color = (0, 140, 255) if not is_active else (0, 220, 255) # 橙色
                    thickness = 3 if is_active else 2
                    source_x = edge.trigger_x if edge.trigger_x is not None else p_from.center_x
                    source_y = p_from.surface_y_at(source_x)
                    landing_x = edge.landing_x if edge.landing_x is not None else source_x
                    target_y = p_to.surface_y_at(landing_x)
                    draw_motion_curve(
                        (source_x, source_y),
                        (landing_x, target_y),
                        edge.action, color, thickness,
                    )
                elif "JUMP" in edge.action or "DROP" in edge.action or "TELEPORT" in edge.action:
                    if "TELEPORT" in edge.action:
                        color = (255, 105, 180) if not is_active else (255, 180, 255) # 粉紫
                    else:
                        color = (0, 255, 128) if not is_active else (0, 255, 255) # 绿色
                    source_x = edge.trigger_x if edge.trigger_x is not None else p_from.center_x
                    source_y = p_from.surface_y_at(source_x)
                    if edge.action in ("JUMP_UP", "TELEPORT_UP"):
                        landing_x = source_x
                    else:
                        landing_x = edge.landing_x if edge.landing_x is not None else p_to.center_x
                    target_y = p_to.surface_y_at(landing_x)
                    draw_motion_curve(
                        (source_x, source_y),
                        (landing_x, target_y),
                        edge.action, color, 3 if is_active else 2,
                    )
                    # 预览正在执行的长平台路径时，把短 foothold 动作计划
                    # 明确画出来：青色安全起跳区、起跳/落点和原始 fh 编号。
                    if is_active and edge.source_foothold_id is not None:
                        takeoff_range = edge.takeoff_x_range or (
                            (float(edge.trigger_x_range[0]), float(edge.trigger_x_range[1]))
                            if edge.trigger_x_range else None
                        )
                        if takeoff_range is not None:
                            range_y1 = p_from.surface_y_at(takeoff_range[0])
                            range_y2 = p_from.surface_y_at(takeoff_range[1])
                            cv2.line(
                                canvas,
                                to_cv(takeoff_range[0], range_y1),
                                to_cv(takeoff_range[1], range_y2),
                                (255, 255, 0), 5, cv2.LINE_AA,
                            )
                        launch_pt = to_cv(source_x, source_y)
                        landing_pt = to_cv(landing_x, target_y)
                        cv2.circle(canvas, launch_pt, 7, (255, 255, 0), 2, cv2.LINE_AA)
                        cv2.circle(canvas, landing_pt, 7, (0, 255, 255), 2, cv2.LINE_AA)
                        label = (
                            f"fh{edge.source_foothold_id}->fh{edge.target_foothold_id} "
                            f"X={int(round(edge.takeoff_x or source_x))}"
                        )
                        cv2.putText(
                            canvas, label, (launch_pt[0] + 9, launch_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 2,
                            cv2.LINE_AA,
                        )

        # 2. 绘制 WZ 传送点。
        # type/pt=0 是出生点，不是可触发传送；type=1 是隐藏传送点，
        # type=2 是普通可见传送门。两类真正的传送点都显示，便于核对
        # 跨地图路线与实际地图结构。
        for portal in self.portals:
            try:
                portal_type = int(portal.get("type", portal.get("pt", -1)))
                if portal_type not in (1, 2):
                    continue
                px, py = to_cv(float(portal.get("x", 0)), float(portal.get("y", 0)))
                if portal_type == 2:
                    color = (255, 0, 255)       # 可见传送门：紫色
                    marker = "VISIBLE"
                else:
                    color = (0, 165, 255)       # 隐藏传送点：橙色
                    marker = "HIDDEN"
                cv2.circle(canvas, (px, py), 9, color, 2, cv2.LINE_AA)
                cv2.drawMarker(
                    canvas, (px, py), color, cv2.MARKER_CROSS,
                    16, 2, cv2.LINE_AA,
                )
                portal_name = str(portal.get("portalName", portal.get("pn", "")) or "")
                target_map = int(portal.get("toMap", portal.get("tm", 999999999)) or 999999999)
                target_name = str(portal.get("toName", portal.get("tn", "")) or "")
                target_text = target_name or (str(target_map) if target_map != 999999999 else "?")
                label = f"门 {portal_name or portal.get('id', '?')} → {target_text} [{marker}]"
                label_x = min(canvas.shape[1] - 4, px + 13)
                label_y = max(14, py - 12)
                cv2.putText(
                    canvas, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA,
                )
                cv2.putText(
                    canvas, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 25), 1, cv2.LINE_AA,
                )
            except (TypeError, ValueError):
                continue

        # 3. 判定角色当前所在平台与梯绳攀爬状态
        curr_p, curr_lr, is_climbing = None, None, False
        if player_pos:
            curr_p, curr_lr, is_climbing = self.find_player_location(player_pos[0], player_pos[1])

        # 4. 绘制所有梯子与绳索 (带独立编号 Badge 和攀爬发光高亮)
        for lr in self.ladder_ropes.values():
            pt_top = to_cv(lr.x, lr.y1)
            pt_bot = to_cv(lr.x, lr.y2)

            is_active_ladder = (is_climbing and curr_lr is not None and curr_lr.id == lr.id)

            if is_active_ladder:
                line_color = (0, 255, 0)
                glow_color = (180, 255, 180)
                cv2.line(canvas, pt_top, pt_bot, glow_color, 8, cv2.LINE_AA)
                cv2.line(canvas, pt_top, pt_bot, line_color, 4, cv2.LINE_AA)
            elif lr.is_ladder:
                # 梯子: 青蓝色
                line_color = (255, 200, 0)
                cv2.line(canvas, (pt_top[0]-3, pt_top[1]), (pt_bot[0]-3, pt_bot[1]), line_color, 2, cv2.LINE_AA)
                cv2.line(canvas, (pt_top[0]+3, pt_top[1]), (pt_bot[0]+3, pt_bot[1]), line_color, 2, cv2.LINE_AA)
                # 梯子横档
                step_y1 = min(pt_top[1], pt_bot[1])
                step_y2 = max(pt_top[1], pt_bot[1])
                for sy in range(step_y1, step_y2, 14):
                    cv2.line(canvas, (pt_top[0]-4, sy), (pt_top[0]+4, sy), line_color, 2)
            else:
                # 绳索: 暖橙金色
                line_color = (0, 165, 255)
                cv2.line(canvas, pt_top, pt_bot, line_color, 3, cv2.LINE_AA)

            # 梯绳编号标签 Badge (例如: 梯#1 或 绳#2)
            if not draw_labels:
                continue
            mid_pt = to_cv(lr.x, (lr.y1 + lr.y2) // 2)
            # 绳梯编号与世界 X 一并显示；文字绘制在底图上，因此会跟随
            # 拓扑图整体缩放。编号字号刻意小于 YOU 标牌。
            lbl_text = (
                f"{lr.label} X={lr.x}"
                if not is_active_ladder
                else f"{lr.label} X={lr.x} ★攀爬"
            )
            font_scale = 0.42 if not is_active_ladder else 0.48
            (lw, lh), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            lbx1 = mid_pt[0] - lw // 2 - 4
            lby1 = mid_pt[1] - lh // 2 - 3
            lbx2 = mid_pt[0] + lw // 2 + 4
            lby2 = mid_pt[1] + lh // 2 + 3

            bg_col = (0, 80, 0) if is_active_ladder else ((60, 40, 0) if lr.is_ladder else (0, 40, 70))
            border_col = (0, 255, 0) if is_active_ladder else ((255, 200, 0) if lr.is_ladder else (0, 165, 255))
            txt_col = (255, 255, 255) if is_active_ladder else border_col

            cv2.rectangle(canvas, (lbx1, lby1), (lbx2, lby2), bg_col, -1)
            cv2.rectangle(canvas, (lbx1, lby1), (lbx2, lby2), border_col, 1)
            cv2.putText(canvas, lbl_text, (lbx1 + 4, lby2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 1, cv2.LINE_AA)

        # 4. 绘制平台节点与高对比度特大号数字徽标 (Badge Bubble)
        for node in self.nodes.values():
            # 颜色策略: 当前角色所在平台(荧光绿) | 巡航目标平台(紫红) | 普通平台(亮青黄)
            is_curr = (curr_p is not None and curr_p.id == node.id and not is_climbing)
            is_target = bool(highlight_platforms and node.id in highlight_platforms)

            if is_curr:
                line_color = (0, 255, 0)
                border_color = (120, 255, 120)
                thickness = 7
            elif is_target:
                line_color = (255, 0, 255)
                border_color = (255, 180, 255)
                thickness = 6
            else:
                line_color = (0, 230, 255)
                border_color = (255, 255, 255)
                thickness = 4

            # 合并的是拓扑节点，不是几何形状。逐段绘制原始 foothold，
            # 从而保留台阶和斜坡；旧实现用 x_min/x_max + avg_y 画线，
            # 会把 P69/P60/P40/P17 一类连续台阶压成一条水平线。
            drawable_lines = node.raw_lines or [{
                'x1': node.x_min, 'y1': node.y,
                'x2': node.x_max, 'y2': node.y,
            }]
            for line in drawable_lines:
                try:
                    pt1 = to_cv(line['x1'], line['y1'])
                    pt2 = to_cv(line['x2'], line['y2'])
                except (KeyError, TypeError, ValueError):
                    continue
                cv2.line(canvas, (pt1[0], pt1[1]-1), (pt2[0], pt2[1]-1),
                         border_color, thickness, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, line_color,
                         max(2, thickness - 2), cv2.LINE_AA)

            if not draw_labels:
                continue

            # 绘制大号高对比度徽标 (深色胶囊底板 + 鲜艳大字，任何缩放下清晰锐利)
            tag = f"P{node.id}" if not is_curr else f"P{node.id} ★YOU"
            # 平台编号同样绘制在底图上，随拓扑图缩放；字号小于动态 YOU。
            font_scale = 0.68 if is_curr else 0.58
            font_thick = 2
            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thick)

            center_pt = to_cv(node.center_x, node.center_y)
            bx1 = center_pt[0] - tw // 2 - 6
            by1 = center_pt[1] - th - 12
            bx2 = center_pt[0] + tw // 2 + 6
            by2 = center_pt[1] - 2

            # 徽标底框
            bg_color = (0, 100, 0) if is_curr else ((100, 0, 100) if is_target else (20, 22, 28))
            edge_color = (0, 255, 0) if is_curr else ((255, 50, 255) if is_target else (220, 220, 220))
            text_color = (255, 255, 255) if is_curr else ((255, 255, 255) if is_target else (0, 255, 255))

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), edge_color, 2)
            cv2.putText(canvas, tag, (center_pt[0] - tw // 2, center_pt[1] - 6),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)

        # 5. 绘制角色实时位置光晕点与 YOU 标牌
        if player_pos:
            px, py = to_cv(player_pos[0], player_pos[1])
            cv2.circle(canvas, (px, py), 16, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 7, (0, 255, 255), -1, cv2.LINE_AA)

            # 角色头顶醒目标签
            if is_climbing and curr_lr:
                you_tag = f"YOU (攀爬 {curr_lr.label})"
                tag_bg = (0, 60, 90)
                tag_border = (0, 220, 255)
            elif curr_p:
                you_tag = f"YOU (P{curr_p.id})"
                tag_bg = (0, 80, 0)
                tag_border = (0, 255, 0)
            else:
                you_tag = "YOU"
                tag_bg = (50, 50, 50)
                tag_border = (200, 200, 200)

            (yw, yh), _ = cv2.getTextSize(you_tag, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            cv2.rectangle(canvas, (px + 10, py - 24), (px + 16 + yw, py + 4), tag_bg, -1)
            cv2.rectangle(canvas, (px + 10, py - 24), (px + 16 + yw, py + 4), tag_border, 2)
            cv2.putText(canvas, you_tag, (px + 13, py - 5), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # 6. 绘制标题与图例
        header = title or f"{self.map_name} (MapID: {self.map_id}) - 平台与梯绳拓扑图"
        cv2.putText(canvas, header, (24, 38), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        legend = "图例: 蓝色[梯#N]=梯子 | 橙色[绳#N]=绳索 | 橙箭=下跳 | 绿箭=平跳/直跳 | 紫箭=瞬移直上 | 绿框=当前平台"
        cv2.putText(canvas, legend, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

        return canvas


class PlatformGraphBuilder:
    """
    通用平台拓扑图构建器 (完全通用，零特调)
    从 Map.wz 碰撞物理数据 (footholds + ladderRopes) 中自适应构建平台与连接边
    """

    @staticmethod
    def _jump_model_landing_x(launch_x: float, target_y: float, source_y: float, direction: str,
                              horizontal_speed: float = 125.0) -> float:
        """按 v83 跳跃模型计算下降支到达目标高度时的水平落点。

        Y 轴向下为正，因此 ``dy`` 必须保留符号。旧实现把向上的
        ``dy`` 截成 0，导致所有上跳都套用了下落时间；同时 JUMP_UP
        实际不按左右键，却仍按 125 px/s 计算水平位移。
        """
        dy = float(target_y) - float(source_y)
        g = 2000.0
        vy0 = -555.0
        disc = vy0 * vy0 + 2.0 * g * dy
        if disc < 0.0:
            return float("nan")
        # Positive root is the descending crossing of the target height.
        air_time = (-vy0 + math.sqrt(disc)) / g
        # jump_step 的空中保持时间最多 0.5s；超出部分由落地等待而非水平推进。
        travel = float(horizontal_speed) * min(0.5, air_time)
        return float(launch_x + (travel if direction == "right" else -travel))

    @staticmethod
    def _jump_height_reachable(target_y: float, source_y: float) -> bool:
        """判断目标高度是否落在基础跳跃的轨迹范围内。

        对向上的目标，最大上升高度为 Vy0²/(2G)；对向下目标，
        只要不是无穷远都由终端速度模型处理。这里不使用平台编号
        或地图特例，避免把不可达的“直跳边”送进 A*。
        """
        dy = float(target_y) - float(source_y)
        if dy >= 0.0:
            return True
        return -dy <= (555.0 * 555.0) / (2.0 * 2000.0) + 1e-6

    @staticmethod
    def _line_y_at(line: Dict[str, Any], x: float) -> Optional[float]:
        """在线段水平范围内插值表面 Y；范围外返回 None。"""
        try:
            x1, y1 = float(line['x1']), float(line['y1'])
            x2, y2 = float(line['x2']), float(line['y2'])
        except (KeyError, TypeError, ValueError):
            return None
        lo, hi = min(x1, x2), max(x1, x2)
        if x < lo or x > hi or abs(x2 - x1) < 1e-9:
            return None
        return y1 + (y2 - y1) * ((float(x) - x1) / (x2 - x1))

    @staticmethod
    def _best_vertical_foothold_jump(
        graph: PlatformGraph,
        source: PlatformNode,
        target: PlatformNode,
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
        candidate_rank: int = 0,
    ) -> Optional[PlatformEdge]:
        """在两条合并长平台之间寻找最可靠的原始短段直跳方案（或瞬移直上方案）。

        Maple foothold 可从下方穿过、在下降阶段承重。因此这里只要求：
        原始短段拥有安全水平重叠、局部高度在基础跳跃或瞬移范围内，且下降
        到目标之前不会先碰到另一条更低的平台。平台平均 Y 不参与判断。
        """
        max_rise = (555.0 * 555.0) / (2.0 * 2000.0)
        effective_max_rise = max(max_rise, float(teleport_distance_px)) if enable_teleport else max_rise
        candidates: List[Tuple[float, PlatformEdge]] = []

        for source_line in source.raw_lines:
            for target_line in target.raw_lines:
                try:
                    source_lo = min(float(source_line['x1']), float(source_line['x2']))
                    source_hi = max(float(source_line['x1']), float(source_line['x2']))
                    target_lo = min(float(target_line['x1']), float(target_line['x2']))
                    target_hi = max(float(target_line['x1']), float(target_line['x2']))
                    overlap_lo = max(source_lo, target_lo)
                    overlap_hi = min(source_hi, target_hi)
                except (KeyError, TypeError, ValueError):
                    continue
                overlap_width = overlap_hi - overlap_lo
                # 角色位置存在约一格小地图量化误差；短于 20px 的重叠
                # 无法提供稳定起跳区，不作为自动巡逻动作。
                if overlap_width < 20.0:
                    continue

                takeoff_x = (overlap_lo + overlap_hi) * 0.5
                source_y = PlatformGraphBuilder._line_y_at(source_line, takeoff_x)
                target_y = PlatformGraphBuilder._line_y_at(target_line, takeoff_x)
                if source_y is None or target_y is None:
                    continue
                rise = source_y - target_y
                if rise < 16.0 or rise > effective_max_rise:
                    continue

                margin = min(12.0, max(6.0, overlap_width * 0.20))
                safe_lo = overlap_lo + margin
                safe_hi = overlap_hi - margin
                if safe_hi - safe_lo < 8.0:
                    continue
                takeoff_x = (safe_lo + safe_hi) * 0.5
                source_y = PlatformGraphBuilder._line_y_at(source_line, takeoff_x)
                target_y = PlatformGraphBuilder._line_y_at(target_line, takeoff_x)
                if source_y is None or target_y is None:
                    continue
                rise = source_y - target_y
                if rise < 16.0 or rise > effective_max_rise:
                    continue
                is_teleport = rise > max_rise
                action_reach = float(teleport_distance_px) if is_teleport else max_rise

                # An ordinary jump cannot be intercepted by a platform above
                # its jump apex, even when teleport is enabled for other edges.
                # Check at the final takeoff X, not the untrimmed overlap center.
                intercepted = False
                for middle in graph.nodes.values():
                    if middle.id in (source.id, target.id):
                        continue
                    middle_ys = [
                        value for value in (
                            PlatformGraphBuilder._line_y_at(line, takeoff_x)
                            for line in middle.raw_lines
                        )
                        if value is not None
                    ]
                    if any(
                        value < target_y - 3.0
                        and 16.0 <= source_y - value <= action_reach
                        for value in middle_ys
                    ):
                        intercepted = True
                        break
                if intercepted:
                    continue

                source_dx = max(1.0, abs(float(source_line['x2']) - float(source_line['x1'])))
                target_dx = max(1.0, abs(float(target_line['x2']) - float(target_line['x1'])))
                source_slope = abs(float(source_line['y2']) - float(source_line['y1'])) / source_dx
                target_slope = abs(float(target_line['y2']) - float(target_line['y1'])) / target_dx
                force_penalty = 0.08 if int(source_line.get('force', 0) or 0) != 0 else 0.0
                # 宽安全区、较小高度差、较平缓表面优先。置信度用于同一
                # 长平台对存在多个可跳短段时排序，而不是伪装成成功率。
                confidence = (
                    0.62
                    + min(0.20, (safe_hi - safe_lo) / 100.0)
                    + min(0.15, max(0.0, (effective_max_rise - rise) / effective_max_rise * 0.15))
                    - min(0.10, (source_slope + target_slope) * 0.05)
                    - force_penalty
                )
                confidence = max(0.05, min(0.99, confidence))
                score = confidence + min(0.12, overlap_width / 300.0)
                action_name = "TELEPORT_UP" if is_teleport else "JUMP_UP"
                act_text = "向上瞬移" if is_teleport else "原地上跳"
                edge = PlatformEdge(
                    from_id=source.id,
                    to_id=target.id,
                    action=action_name,
                    cost=round((0.55 if is_teleport else 0.68) - confidence * 0.20, 2),
                    trigger_x=int(round(takeoff_x)),
                    trigger_x_range=(int(math.ceil(safe_lo)), int(math.floor(safe_hi))),
                    landing_x=float(takeoff_x),
                    target_y=int(round(target_y)),
                    description=(
                        f"长平台 {source.id}->{target.id}：在 foothold "
                        f"{source_line.get('id')} 的 X=[{int(math.ceil(safe_lo))}~"
                        f"{int(math.floor(safe_hi))}] {act_text}，落到 foothold "
                        f"{target_line.get('id')}（局部高度差 {rise:.0f}px，"
                        f"置信度 {confidence:.2f}）"
                    ),
                    source_foothold_id=int(source_line.get('id', 0) or 0),
                    target_foothold_id=int(target_line.get('id', 0) or 0),
                    takeoff_x=float(takeoff_x),
                    takeoff_x_range=(float(safe_lo), float(safe_hi)),
                    landing_x_range=(float(safe_lo), float(safe_hi)),
                    confidence=round(confidence, 3),
                )
                candidates.append((score, edge))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        rank = max(0, int(candidate_rank))
        return candidates[rank][1] if rank < len(candidates) else None

    @staticmethod
    def _best_directional_foothold_jump(
        graph: PlatformGraph,
        source: PlatformNode,
        target: PlatformNode,
    ) -> Optional[PlatformEdge]:
        """从原始短段采样可落地的左右跑跳，而非使用长平台平均 Y。"""
        max_rise = (555.0 * 555.0) / (2.0 * 2000.0)
        best: Optional[Tuple[float, PlatformEdge]] = None

        for source_line in source.raw_lines:
            try:
                source_lo = min(float(source_line['x1']), float(source_line['x2']))
                source_hi = max(float(source_line['x1']), float(source_line['x2']))
            except (KeyError, TypeError, ValueError):
                continue
            source_width = source_hi - source_lo
            if source_width < 18.0:
                continue

            for target_line in target.raw_lines:
                try:
                    target_lo = min(float(target_line['x1']), float(target_line['x2']))
                    target_hi = max(float(target_line['x1']), float(target_line['x2']))
                except (KeyError, TypeError, ValueError):
                    continue
                target_width = target_hi - target_lo
                if target_width < 18.0:
                    continue

                if target_lo >= source_hi - 2.0:
                    direction = "right"
                    gap = max(0.0, target_lo - source_hi)
                elif target_hi <= source_lo + 2.0:
                    direction = "left"
                    gap = max(0.0, source_lo - target_hi)
                else:
                    # 有明显重叠时交给更稳的原地上跳规划器。
                    continue
                if gap > 70.0:
                    continue

                source_margin = min(10.0, max(4.0, source_width * 0.18))
                target_margin = min(12.0, max(6.0, target_width * 0.15))
                source_safe_lo = source_lo + source_margin
                source_safe_hi = source_hi - source_margin
                target_safe_lo = target_lo + target_margin
                target_safe_hi = target_hi - target_margin
                if source_safe_hi <= source_safe_lo or target_safe_hi <= target_safe_lo:
                    continue

                # 在真实源短段上采样起跳点，并使用与执行器相同的基础
                # 水平速度/跳跃重力模型反推下降支落点。
                sample_count = max(7, min(25, int(source_safe_hi - source_safe_lo) + 1))
                valid: List[Tuple[float, float, float, float]] = []
                target_mid = (target_safe_lo + target_safe_hi) * 0.5
                target_mid_y = PlatformGraphBuilder._line_y_at(target_line, target_mid)
                if target_mid_y is None:
                    continue
                for index in range(sample_count):
                    ratio = index / max(1, sample_count - 1)
                    launch_x = source_safe_lo + (source_safe_hi - source_safe_lo) * ratio
                    source_y = PlatformGraphBuilder._line_y_at(source_line, launch_x)
                    if source_y is None:
                        continue
                    target_y = target_mid_y
                    landing_x = PlatformGraphBuilder._jump_model_landing_x(
                        launch_x, target_y, source_y, direction
                    )
                    if math.isnan(landing_x):
                        continue
                    # 目标可能是斜坡：用第一次落点处的真实 Y 再迭代一次。
                    iterated_y = PlatformGraphBuilder._line_y_at(target_line, landing_x)
                    if iterated_y is not None:
                        target_y = iterated_y
                        landing_x = PlatformGraphBuilder._jump_model_landing_x(
                            launch_x, target_y, source_y, direction
                        )
                    landing_y = PlatformGraphBuilder._line_y_at(target_line, landing_x)
                    if landing_y is None or not (target_safe_lo <= landing_x <= target_safe_hi):
                        continue
                    rise = source_y - landing_y
                    if rise > max_rise or rise < -90.0:
                        continue
                    valid.append((launch_x, landing_x, source_y, landing_y))

                if not valid:
                    continue

                # 首选落点和起跳点都远离端点的样本，抵抗小地图量化和
                # 起跳帧延迟；所有有效样本的跨度成为动态安全区。
                source_center = (source_safe_lo + source_safe_hi) * 0.5
                target_center = (target_safe_lo + target_safe_hi) * 0.5
                preferred = max(
                    valid,
                    key=lambda item: (
                        min(item[1] - target_safe_lo, target_safe_hi - item[1])
                        + 0.35 * min(item[0] - source_safe_lo, source_safe_hi - item[0])
                        - 0.05 * abs(item[0] - source_center)
                    ),
                )
                launch_x, landing_x, source_y, landing_y = preferred
                valid_launch_lo = min(item[0] for item in valid)
                valid_launch_hi = max(item[0] for item in valid)
                valid_landing_lo = min(item[1] for item in valid)
                valid_landing_hi = max(item[1] for item in valid)
                launch_width = valid_launch_hi - valid_launch_lo
                landing_margin = min(
                    landing_x - target_safe_lo, target_safe_hi - landing_x
                )
                force = int(source_line.get('force', 0) or 0)
                force_penalty = 0.0
                if force and ((force > 0) != (direction == "right")):
                    force_penalty = 0.15
                confidence = (
                    0.66
                    + min(0.16, launch_width / 100.0)
                    + min(0.12, max(0.0, landing_margin) / 80.0)
                    - min(0.10, gap / 300.0)
                    - force_penalty
                )
                confidence = max(0.05, min(0.99, confidence))
                score = confidence + min(0.10, target_width / 400.0)
                action = "JUMP_RIGHT" if direction == "right" else "JUMP_LEFT"
                edge = PlatformEdge(
                    from_id=source.id,
                    to_id=target.id,
                    action=action,
                    cost=round(0.82 + gap / 220.0 - confidence * 0.10, 2),
                    trigger_x=int(round(launch_x)),
                    trigger_x_range=(
                        int(math.ceil(valid_launch_lo)), int(math.floor(valid_launch_hi))
                    ),
                    landing_x=float(landing_x),
                    target_y=int(round(landing_y)),
                    description=(
                        f"长平台 {source.id}->{target.id}：从 foothold "
                        f"{source_line.get('id')} 的 X={launch_x:.0f} 向"
                        f"{('右' if direction == 'right' else '左')}跑跳，落到 foothold "
                        f"{target_line.get('id')} 的 X={landing_x:.0f}"
                        f"（间隔 {gap:.0f}px，置信度 {confidence:.2f}）"
                    ),
                    source_foothold_id=int(source_line.get('id', 0) or 0),
                    target_foothold_id=int(target_line.get('id', 0) or 0),
                    takeoff_x=float(launch_x),
                    takeoff_x_range=(float(valid_launch_lo), float(valid_launch_hi)),
                    landing_x_range=(float(valid_landing_lo), float(valid_landing_hi)),
                    confidence=round(confidence, 3),
                )
                if best is None or score > best[0]:
                    best = (score, edge)
        return best[1] if best is not None else None

    @staticmethod
    def _generate_foothold_transition_edges(
        graph: PlatformGraph,
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
    ):
        """用原始短 foothold 补足合并长平台遗漏的局部跳跃边（及瞬移直上边）。"""
        nodes = list(graph.nodes.values())
        y_bounds: Dict[int, Tuple[float, float]] = {}
        for node in nodes:
            ys = [
                float(line[key])
                for line in node.raw_lines
                for key in ('y1', 'y2')
                if key in line
            ]
            y_bounds[node.id] = (
                (min(ys), max(ys)) if ys else (float(node.y), float(node.y))
            )
        max_up_reach = max(77.1, float(teleport_distance_px)) if enable_teleport else 77.1
        for source in nodes:
            for target in nodes:
                if source.id == target.id:
                    continue
                # 任意两条原始短段若要发生普通跳跃或瞬移，其长平台水平包围盒
                # 间距必不超过70px，局部高度差范围也必须在可达范围内。
                # 先做 O(1) 空间裁剪，避免对整图所有 raw_lines 进行笛卡尔积采样。
                horizontal_distance = max(
                    0.0,
                    float(target.x_min - source.x_max),
                    float(source.x_min - target.x_max),
                )
                if horizontal_distance > 70.0:
                    continue
                source_y_min, source_y_max = y_bounds[source.id]
                target_y_min, target_y_max = y_bounds[target.id]
                difference_min = source_y_min - target_y_max
                difference_max = source_y_max - target_y_min
                if difference_max < -90.0 or difference_min > max_up_reach:
                    continue
                plans: List[Optional[PlatformEdge]] = []
                # 有足够重叠时尝试更稳的原地直跳/向上瞬移。
                if min(source.x_max, target.x_max) - max(source.x_min, target.x_min) >= 20:
                    plans.append(PlatformGraphBuilder._best_vertical_foothold_jump(
                        graph, source, target,
                        enable_teleport=enable_teleport,
                        teleport_distance_px=teleport_distance_px,
                    ))
                # 长平台整体可能相交，但末端短段仍可能存在真实间隔；
                # 始终枚举短段级左右跑跳，覆盖 P83->P84 这类情况。
                plans.append(PlatformGraphBuilder._best_directional_foothold_jump(
                    graph, source, target
                ))

                for edge in (item for item in plans if item is not None):
                    existing = any(
                        item.to_id == target.id
                        and item.action == edge.action
                        and item.source_foothold_id is not None
                        for item in graph.get_edges_from(source.id)
                    )
                    if not existing:
                        graph.add_edge(edge)
                    if edge.action in ("JUMP_UP", "TELEPORT_UP"):
                        # The generic overlap rule uses the *long-platform*
                        # center to choose left/right. When a short-foothold
                        # vertical plan exists, that diagonal edge is a less
                        # reliable duplicate (103000201 P1->P5).
                        graph.edges[source.id] = [
                            candidate for candidate in graph.get_edges_from(source.id)
                            if not (
                                candidate.to_id == target.id
                                and candidate.action in ("JUMP_LEFT", "JUMP_RIGHT")
                                and candidate.source_foothold_id is None
                                and "重叠区斜向" in candidate.description
                            )
                        ]

    @staticmethod
    def _walk_drop_landing_x(launch_x: float, source_y: float, target_y: float, direction: str) -> float:
        """对应 MotionController.walk_off_drop 的重力下落落点。"""
        g = 2000.0
        terminal_speed = 670.0
        drop_y = max(0.0, float(target_y) - float(source_y))
        terminal_t = terminal_speed / g
        terminal_y = 0.5 * g * terminal_t * terminal_t
        if drop_y <= terminal_y:
            fall_sec = math.sqrt(2.0 * drop_y / g)
        else:
            fall_sec = terminal_t + (drop_y - terminal_y) / terminal_speed
        travel = 125.0 * fall_sec
        return float(launch_x + (travel if direction == "right" else -travel))

    @staticmethod
    def build_from_map_dict(
        map_data: Dict[str, Any],
        merge_short_platforms: bool = True,
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
    ) -> PlatformGraph:
        map_id = map_data.get('id') or map_data.get('info', {}).get('mapId', 0)
        street = map_data.get('streetName') or map_data.get('info', {}).get('streetName', '')
        name = map_data.get('name') or map_data.get('info', {}).get('mapName', '')
        map_name = f"{street} - {name}".strip(" -")
        vr_bounds = map_data.get('vrBounds') or map_data.get('info', {}).get('vrBounds')
        minimap_meta = map_data.get('miniMap') or map_data.get('info', {}).get('miniMap')

        graph = PlatformGraph(
            map_id=int(map_id or 0),
            map_name=map_name,
            vr_bounds=vr_bounds,
            minimap_meta=minimap_meta
        )
        graph.enable_teleport = bool(enable_teleport)
        graph.teleport_distance_px = float(teleport_distance_px)
        graph.portals = [
            dict(portal) for portal in (map_data.get("portals", []) or [])
            if isinstance(portal, dict)
        ]
        # X/Y 人工标定只对当前自动框选范围、本次进入地图有效，不从旧版
        # minimap_x_calibrations.json 或拓扑缓存恢复。

        # 1. 提取所有水平/可站立线段 (排除垂直阻挡墙)
        raw_lines = PlatformGraphBuilder._extract_foothold_lines(map_data.get('footholds', {}))
        for line in raw_lines:
            try:
                fh_id = int(line.get('id', 0) or 0)
            except (TypeError, ValueError):
                continue
            if fh_id > 0:
                graph.foothold_lines_by_id.setdefault(fh_id, line)
        standable_lines = [l for l in raw_lines if PlatformGraphBuilder._is_standable(l)]

        # 2. 聚类合并水平线段为独立平台
        platforms = PlatformGraphBuilder._cluster_and_merge_platforms(
            standable_lines,
            merge_short_platforms=merge_short_platforms,
            topology_lines=raw_lines,
        )
        for p in platforms:
            graph.add_node(p)

        # 3. 提取梯子和绳索
        ladder_ropes = map_data.get('ladderRopes', [])

        # 4. 生成平台间动作转移边 (下跳、跳跃、爬梯、瞬移)
        PlatformGraphBuilder._generate_edges(
            graph, ladder_ropes,
            enable_teleport=enable_teleport,
            teleport_distance_px=teleport_distance_px,
        )
        # 5. 将长平台之间的局部上跳展开到具体原始 foothold。该层使用
        # 短段真实高度，修复长斜坡被平均 Y 掩盖后漏边的问题。
        PlatformGraphBuilder._generate_foothold_transition_edges(
            graph,
            enable_teleport=enable_teleport,
            teleport_distance_px=teleport_distance_px,
        )
        # 6. 将当前地图内的成对传送点加入平台拓扑。
        PlatformGraphBuilder._generate_portal_edges(graph, map_data.get('portals', []))

        return graph

    @staticmethod
    def _portal_platform(graph: PlatformGraph, portal: Dict[str, Any]) -> Optional[PlatformNode]:
        """将传送点坐标绑定到其脚下平台。"""
        x = float(portal.get('x', 0))
        y = float(portal.get('y', 0))
        candidates = [
            n for n in graph.nodes.values()
            if n.x_min - 30 <= x <= n.x_max + 30 and abs(n.surface_y_at(x) - y) <= 30
        ]
        return min(
            candidates,
            key=lambda n: abs(n.surface_y_at(x) - y) + abs(n.center_x - x) * 0.05,
        ) if candidates else None

    @staticmethod
    def _generate_portal_edges(graph: PlatformGraph, portals: List[Dict[str, Any]]):
        """为同图成对 portal 建立双向 PORTAL 边，外部出口不纳入跨层寻路。"""
        by_name = {
            p.get('portalName'): p for p in portals
            if isinstance(p, dict) and p.get('portalName')
        }
        for portal in portals:
            if not isinstance(portal, dict) or portal.get('toMap') != graph.map_id:
                continue
            target_name = portal.get('toName')
            target = by_name.get(target_name)
            if not target:
                continue
            source_node = PlatformGraphBuilder._portal_platform(graph, portal)
            target_node = PlatformGraphBuilder._portal_platform(graph, target)
            if not source_node or not target_node or source_node.id == target_node.id:
                continue
            if any(e.to_id == target_node.id and e.action == 'PORTAL' for e in graph.get_edges_from(source_node.id)):
                continue
            graph.add_edge(PlatformEdge(
                from_id=source_node.id,
                to_id=target_node.id,
                action='PORTAL',
                cost=0.8,
                trigger_x=int(portal.get('x', source_node.center_x)),
                target_y=target_node.y,
                description=(f"从平台 {source_node.id} 走到传送点 {portal.get('portalName')}，"
                             f"传送至平台 {target_node.id} ({target_name})")
            ))

    @staticmethod
    def _extract_foothold_lines(footholds_data: Any) -> List[Dict[str, int]]:
        """递归展开所有嵌套的 footholds 数据"""
        lines = []
        if isinstance(footholds_data, dict):
            if 'x1' in footholds_data and 'y1' in footholds_data and 'x2' in footholds_data and 'y2' in footholds_data:
                lines.append(footholds_data)
            else:
                for v in footholds_data.values():
                    lines.extend(PlatformGraphBuilder._extract_foothold_lines(v))
        elif isinstance(footholds_data, list):
            for item in footholds_data:
                lines.extend(PlatformGraphBuilder._extract_foothold_lines(item))
        return lines

    @staticmethod
    def _is_standable(line: Dict[str, int]) -> bool:
        """判定线段是否为可站立表面 (排除垂直墙壁)"""
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        # 必须具备一定的水平跨度，且坡度不能是接近 90 度的垂直悬崖 (斜率 dy/dx < 1.8)
        if dx < 4:
            return False
        return (dy / max(1, dx)) < 1.8

    @staticmethod
    def _is_micro_topology_bridge(line: Dict[str, Any]) -> bool:
        """是否为只用于衔接 next/prev 的退化微型 foothold。

        WZ 地图会在两段可站地面之间插入 1~3px 的接缝。它们因水平
        跨度不足而不是独立平台，但也不是阻断行走的长垂直墙。
        """
        try:
            dx = abs(int(line['x2']) - int(line['x1']))
            dy = abs(int(line['y2']) - int(line['y1']))
        except (KeyError, TypeError, ValueError):
            return False
        return not PlatformGraphBuilder._is_standable(line) and max(dx, dy) <= 3

    @staticmethod
    def _linked_foothold(
        line: Dict[str, Any],
        direction: str,
        by_id: Dict[int, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """返回 next/prev 双方互指的直接邻居。"""
        try:
            source_id = int(line.get('id', 0) or 0)
            neighbor_id = int(line.get(direction, 0) or 0)
        except (TypeError, ValueError):
            return None
        neighbor = by_id.get(neighbor_id)
        if source_id <= 0 or neighbor is None:
            return None
        reverse = 'prev' if direction == 'next' else 'next'
        try:
            reverse_id = int(neighbor.get(reverse, 0) or 0)
        except (TypeError, ValueError):
            return None
        return neighbor if reverse_id == source_id else None

    @staticmethod
    def _linked_standable_foothold(
        line: Dict[str, Any],
        direction: str,
        by_id: Dict[int, Dict[str, Any]],
        standable_ids: Set[int],
        allow_four_px_bridge: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """沿互指拓扑穿过微型接缝，返回下一条可站 foothold。"""
        current = line
        visited: Set[int] = set()
        while True:
            neighbor = PlatformGraphBuilder._linked_foothold(current, direction, by_id)
            if neighbor is None:
                # WZ 允许多条 foothold 汇入同一个端点，而目标的反向字段
                # 只能记录其中一条。此时接受“同层/同组、端点完全重合”
                # 的单向连接；这不是旧式距离猜测，不允许任何坐标容差。
                try:
                    neighbor_id = int(current.get(direction, 0) or 0)
                except (TypeError, ValueError):
                    return None
                candidate = by_id.get(neighbor_id)
                if candidate is None:
                    return None
                source_endpoint = ('x2', 'y2') if direction == 'next' else ('x1', 'y1')
                target_endpoint = ('x1', 'y1') if direction == 'next' else ('x2', 'y2')
                same_endpoint = (
                    current.get(source_endpoint[0]) == candidate.get(target_endpoint[0])
                    and current.get(source_endpoint[1]) == candidate.get(target_endpoint[1])
                )
                same_surface = (
                    current.get('layerId') == candidate.get('layerId')
                    and current.get('groupId') == candidate.get('groupId')
                )
                if not (
                    same_endpoint
                    and same_surface
                    and PlatformGraphBuilder._is_standable(current)
                    and PlatformGraphBuilder._is_standable(candidate)
                ):
                    return None
                neighbor = candidate
            try:
                neighbor_id = int(neighbor.get('id', 0) or 0)
            except (TypeError, ValueError):
                return None
            if neighbor_id in visited:
                return None
            visited.add(neighbor_id)
            if neighbor_id in standable_ids:
                return neighbor
            # 4px 的短坡可供人物走过，却不能并入平台聚类：并入会
            # 改变整图 P 编号，破坏已保存的巡逻平台设置。
            is_walk_bridge = PlatformGraphBuilder._is_micro_topology_bridge(neighbor)
            if not is_walk_bridge and allow_four_px_bridge:
                try:
                    dx = abs(int(neighbor['x2']) - int(neighbor['x1']))
                    dy = abs(int(neighbor['y2']) - int(neighbor['y1']))
                    is_walk_bridge = (
                        not PlatformGraphBuilder._is_standable(neighbor)
                        and max(dx, dy) <= 4
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            if not is_walk_bridge:
                return None
            current = neighbor

    @staticmethod
    def _cluster_and_merge_platforms(
        lines: List[Dict[str, Any]],
        merge_short_platforms: bool = True,
        topology_lines: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PlatformNode]:
        """按 WZ ``next``/``prev`` 拓扑将 foothold 合并为平台。

        ``next`` 指向越过原始第二端点后的 foothold，``prev`` 指向越过
        原始第一端点后的 foothold。通常要求双方互相指认；对于 WZ 的
        多入口汇合点，也接受同层同组且端点完全重合的单向指认。除此
        之外不使用 Y 高度或 X 间隙猜测，避免误合并相邻独立平台。

        垂直墙在调用本方法前已被过滤，因此一条包含“墙 -> 地面 -> 墙”
        的 WZ 闭合轮廓会在墙处自然截断，只合并其中可站立的地面部分。
        ``merge_short_platforms=False`` 明确表示保留每一条原始 foothold。
        """
        if not lines:
            return []

        # 保留 next/prev/force/forbidFallDown 等全部原始属性。不能为了
        # x1 <= x2 而交换端点，因为 next/prev 的语义绑定原始端点方向。
        source_lines: List[Dict[str, Any]] = [dict(line) for line in lines]

        def foothold_id(line: Dict[str, Any]) -> int:
            try:
                return int(line.get('id', 0) or 0)
            except (TypeError, ValueError):
                return 0

        by_id: Dict[int, Dict[str, Any]] = {}
        for line in topology_lines or source_lines:
            fh_id = foothold_id(line)
            if fh_id > 0:
                by_id.setdefault(fh_id, line)

        line_index: Dict[int, int] = {}
        for index, line in enumerate(source_lines):
            fh_id = foothold_id(line)
            # 0 是 WZ 的断链标记，不是有效 foothold；重复 ID 也不参与
            # 自动连接，以免把损坏数据错误合并。
            if fh_id > 0 and fh_id not in line_index:
                line_index[fh_id] = index
        standable_ids = set(line_index)

        def linked_neighbor(line: Dict[str, Any], direction: str) -> Optional[Dict[str, Any]]:
            """穿过退化微型接缝，返回双方互指的下一条可站 foothold。"""
            return PlatformGraphBuilder._linked_standable_foothold(
                line, direction, by_id, standable_ids
            )

        if not merge_short_platforms:
            clusters: List[List[Dict[str, Any]]] = [[line] for line in source_lines]
        else:
            # 并查集只消费 WZ 拓扑，不再使用固定 X/Y/gap 阈值。
            parent = list(range(len(source_lines)))

            def find(index: int) -> int:
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(left: int, right: int) -> None:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[right_root] = left_root

            for index, line in enumerate(source_lines):
                for direction in ('prev', 'next'):
                    neighbor = linked_neighbor(line, direction)
                    if neighbor is not None:
                        union(index, line_index[foothold_id(neighbor)])

            components: Dict[int, List[Dict[str, Any]]] = {}
            for index, line in enumerate(source_lines):
                components.setdefault(find(index), []).append(line)

            def order_chain(component: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """按 prev -> next 排序，令 foothold_ids 可直接表达行走顺序。"""
                component_ids = {foothold_id(line) for line in component if foothold_id(line) > 0}
                component_by_id = {foothold_id(line): line for line in component if foothold_id(line) > 0}
                starts = []
                for line in component:
                    previous = linked_neighbor(line, 'prev')
                    if previous is None or foothold_id(previous) not in component_ids:
                        starts.append(line)
                starts.sort(key=lambda line: foothold_id(line) or id(line))

                ordered: List[Dict[str, Any]] = []
                visited: Set[int] = set()
                seeds = starts + sorted(component, key=lambda line: foothold_id(line) or id(line))
                for seed in seeds:
                    current = seed
                    while current is not None:
                        current_id = foothold_id(current)
                        current_key = current_id if current_id > 0 else id(current)
                        if current_key in visited:
                            break
                        visited.add(current_key)
                        ordered.append(current)
                        nxt = linked_neighbor(current, 'next')
                        if nxt is None or foothold_id(nxt) not in component_by_id:
                            break
                        current = nxt
                return ordered

            clusters = [order_chain(component) for component in components.values()]

        # 转换为 PlatformNode
        platforms: List[PlatformNode] = []
        # 按高度 Y 降序 (从最底层到最顶层) 排序
        clusters_sorted = sorted(clusters, key=lambda c: np.mean([(it['y1'] + it['y2'])/2 for it in c]), reverse=True)

        for idx, cluster in enumerate(clusters_sorted):
            x_min = min(min(c['x1'], c['x2']) for c in cluster)
            x_max = max(max(c['x1'], c['x2']) for c in cluster)
            avg_y = int(round(np.mean([(c['y1'] + c['y2'])/2 for c in cluster])))
            length = x_max - x_min
            
            # 过滤太短的噪点平台 (< 15px)
            if length < 15:
                continue

            fh_ids = [foothold_id(c) for c in cluster if foothold_id(c) > 0]
            platforms.append(PlatformNode(
                id=len(platforms) + 1,
                x_min=x_min,
                x_max=x_max,
                y=avg_y,
                length=length,
                layer=len(platforms) + 1,
                foothold_ids=fh_ids,
                raw_lines=cluster
            ))

        return platforms

    @staticmethod
    def _generate_edges(
        graph: PlatformGraph,
        ladder_ropes: List[Dict[str, Any]],
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
    ):
        """通用生成所有动作边 (下跳、平跳、高空落差跳、爬梯绳、阶梯立柱) 并对梯子/绳索进行统一编号"""
        nodes = list(graph.nodes.values())

        # 1. 梯绳统一物理编号排序 (按底端 Y 从底层到顶层排序，若同层按 X 从左到右)
        def lr_sort_key(lr_item):
            y_max = max(lr_item.get('y1', 0), lr_item.get('y2', 0))
            return (-y_max, lr_item.get('x', 0))

        ladder_ropes_sorted = sorted(ladder_ropes, key=lr_sort_key)

        for idx, lr in enumerate(ladder_ropes_sorted):
            lr_id = idx + 1
            lx = lr.get('x', 0)
            ly1 = min(lr.get('y1', 0), lr.get('y2', 0))
            ly2 = max(lr.get('y1', 0), lr.get('y2', 0))
            is_ladder = bool(lr.get('isLadder', lr.get('l', False)))
            action_type = "LADDER" if is_ladder else "ROPE"

            # 寻找梯子下端与上端所接触的平台 (Bottom / Top Platform)
            bottom_platforms = []
            top_platforms = []

            for p in nodes:
                # 检查梯子 X 是否落在平台水平跨度内 (允许 45px 容差)
                if (p.x_min - 45) <= lx <= (p.x_max + 45):
                    platform_y = p.surface_y_at(lx)
                    # 下端接触判定 (常规接触容差 60px，或悬空跳抓空隙容差 up to 110px)
                    if abs(platform_y - ly2) <= 60 or (ly1 < platform_y <= ly2 + 55) or (0 <= platform_y - ly2 <= 110):
                        bottom_platforms.append(p)
                    # 上端接触判定 (容差 60px)
                    if abs(platform_y - ly1) <= 60 or (ly1 - 55 <= platform_y < ly2):
                        top_platforms.append(p)

            # 按垂直距离排序，优先匹配最邻近梯绳端点的平台
            bottom_platforms.sort(key=lambda p: abs(p.surface_y_at(lx) - ly2))
            top_platforms.sort(key=lambda p: abs(p.surface_y_at(lx) - ly1))

            # 注册 LadderRopeNode 节点
            bp_id = bottom_platforms[0].id if bottom_platforms else None
            tp_id = top_platforms[0].id if top_platforms else None
            lr_node = LadderRopeNode(
                id=lr_id,
                x=lx,
                y1=ly1,
                y2=ly2,
                is_ladder=is_ladder,
                bottom_platform_id=bp_id,
                top_platform_id=tp_id
            )
            graph.add_ladder_rope(lr_node)

            # 一根实体梯绳的每个端点只连接距离端点最近的承重面。
            # 旧逻辑将 45px 水平容差内的所有平台做笛卡尔积，会把仅仅
            # 靠近绳顶的相邻平台也当作“直达”：101000000 的35号绳
            # 实际顶端是 P81，却额外生成 P77->P83，实机只会先落 P81。
            connected_bottoms = bottom_platforms[:1]
            connected_tops = top_platforms[:1]

            # 在真实上下端平台间建立双向攀爬边
            for bp in connected_bottoms:
                for tp in connected_tops:
                    bp_y = bp.surface_y_at(lx)
                    tp_y = tp.surface_y_at(lx)
                    if bp.id != tp.id and bp_y > tp_y:
                        height_diff = bp_y - tp_y
                        bottom_gap = bp_y - ly2 # 底端平台表面与梯子悬挂口之间的垂直空隙
                        
                        # 直爬不能只看绳底与“绳正下方一点”的 Y 差。角色按住
                        # UP 横穿绳索时有约 20px 的站立吸附余量；但在明显斜坡
                        # 上，可靠起点会沿坡提前，不能套用这段余量。因此同时
                        # 检查绳索两侧各 30px（一次正常接近步幅）的地面起伏：
                        # 平缓接近区允许走爬，起伏较大则只容忍 4px 端点误差。
                        # 这能区分 107000100 的 14/15 号绳（平地走爬）与
                        # 101000000 的 34 号绳（P77 下坡，必须跳抓）。
                        approach_surface_ys = [
                            bp.surface_y_at(lx + sample_dx)
                            for sample_dx in (-30, -20, -10, 0, 10, 20, 30)
                            if bp.x_min <= lx + sample_dx <= bp.x_max
                        ]
                        approach_relief = (
                            max(approach_surface_ys) - min(approach_surface_ys)
                            if approach_surface_ys else float("inf")
                        )
                        direct_walk_climb = (
                            bottom_gap <= 4
                            or (bottom_gap <= 20 and approach_relief <= 8)
                        )
                        if not direct_walk_climb:
                            climb_up_act = f"JUMP_CLIMB_{action_type}_UP"
                            climb_up_desc = f"从平台 {bp.id} 走至 X={lx} 触发 [跳+上] 起跳抓{lr_node.kind_name} #{lr_id} 并攀爬至平台 {tp.id}"
                        else:
                            climb_up_act = f"CLIMB_{action_type}_UP"
                            climb_up_desc = f"从平台 {bp.id} 走至 X={lx} 按住 [上] 攀爬{lr_node.kind_name} #{lr_id} 至平台 {tp.id}"

                        # 向上爬 (CLIMB_UP / JUMP_CLIMB_UP)
                        graph.add_edge(PlatformEdge(
                            from_id=bp.id,
                            to_id=tp.id,
                            action=climb_up_act,
                            cost=round(1.0 + height_diff / 120.0, 2),
                            trigger_x=lx,
                            target_y=int(round(tp_y)),
                            ladder_id=lr_id,
                            is_rope=not is_ladder,
                            description=climb_up_desc
                        ))
                        # 向下爬 (CLIMB_DOWN)
                        graph.add_edge(PlatformEdge(
                            from_id=tp.id,
                            to_id=bp.id,
                            action=f"CLIMB_{action_type}_DOWN",
                            cost=round(0.8 + height_diff / 150.0, 2),
                            trigger_x=lx,
                            target_y=int(round(bp_y)),
                            ladder_id=lr_id,
                            is_rope=not is_ladder,
                            description=f"从平台 {tp.id} 走至 X={lx} 按住 [下] 攀爬{lr_node.kind_name} #{lr_id} 至平台 {bp.id}"
                        ))

            # 允许从绳梯底部附近的平台直接跳抓，绕过过短的中间落脚平台。
            # 例如 P33 到 2 号绳：水平差约 47px、高度差在可接受范围内。
            for src in nodes:
                if src.id in {p.id for p in bottom_platforms}:
                    continue
                src_y = src.surface_y_at(lx)
                if abs(src_y - ly2) > 60:
                    continue
                # “绕过中间落脚平台”的横向跳抓只适用于源平台站立高度
                # 已经落在绳子的实体竖段内。若 src_y > ly2，绳底实际悬在
                # 源平台上方；角色必须向上越过绳底，而且会先被绳底所属的
                # 相邻平台承接，这不是一条可靠的直达跳抓边。
                # 例如 107000100：7号绳的 P6/P8 位于绳底之上，可以横跳
                # 入绳；17号绳的 P19/P20 与18号绳的 P18/P23 都在绳底
                # 之下，不能绕过 P23/P20 直接抓绳。
                if src_y > ly2:
                    continue
                horizontal_gap = (
                    0.0 if src.x_min <= lx <= src.x_max
                    else min(abs(lx - src.x_min), abs(lx - src.x_max))
                )
                # 这里量的是“源平台边缘到绳轴”的无承重距离，而不是
                # 角色当前位置到绳轴。实机跑跳档案虽然会提前约74px
                # 起跳，但角色离开平台以后仍需要在空中跨完这段空隙。
                # 107000100 的 P18 -> 16号绳空隙为128px，实际无法抓取；
                # 已验证可用的 P8 -> 7号绳为53px、P17 -> 13号绳为72px。
                # 不能再额外放宽到90px：101000000 的 P74 -> 31号绳为
                # 83px，运动模型会在人物走出 P74、落回 P73 后才命中起跳
                # 线，导致 P73/P74 间无限补跳。给72px样本保留少量坐标
                # 误差，将可靠旁路上限限制为75px；更远的绳应从真实下端
                # 平台起跳。
                if horizontal_gap > 75:
                    continue
                # 起跳侧至少保留 25px 空间，避免从平台边缘反向起跳。
                if lx > src.x_max and (src.x_max - src.x_min) < 25:
                    continue
                if lx < src.x_min and (src.x_max - src.x_min) < 25:
                    continue
                for tp in connected_tops:
                    tp_y = tp.surface_y_at(lx)
                    if src.id == tp.id or src_y <= tp_y:
                        continue
                    graph.add_edge(PlatformEdge(
                        from_id=src.id,
                        to_id=tp.id,
                        action=f"JUMP_CLIMB_{action_type}_UP",
                        cost=round(1.15 + horizontal_gap / 125.0 + max(0, src_y - ly2) / 300.0, 2),
                        trigger_x=lx,
                        target_y=int(round(tp_y)),
                        ladder_id=lr_id,
                        is_rope=not is_ladder,
                        description=f"从平台 {src.id} 直接跳抓{lr_node.kind_name} #{lr_id} 至平台 {tp.id}（绕过中间平台）"
                    ))

        # 2. 按 WZ next/prev 生成短 foothold 之间的步行连接。
        # merge_short_platforms=True 时，同链 foothold 已经属于同一平台，
        # 不需要 WALK 边；保留短平台模式下则用官方拓扑连接各节点。
        foothold_to_platform: Dict[int, PlatformNode] = {}
        foothold_lines: Dict[int, Dict[str, Any]] = dict(graph.foothold_lines_by_id)
        for platform in nodes:
            for line in platform.raw_lines:
                try:
                    fh_id = int(line.get('id', 0) or 0)
                except (TypeError, ValueError):
                    continue
                if fh_id > 0:
                    foothold_to_platform.setdefault(fh_id, platform)
                    foothold_lines.setdefault(fh_id, line)

        walk_edges: Set[Tuple[int, int, str]] = set()
        standable_ids = set(foothold_to_platform)
        for source_fh_id in standable_ids:
            source_line = foothold_lines[source_fh_id]
            source_platform = foothold_to_platform[source_fh_id]
            for direction, endpoint_key, other_endpoint_key in (
                ('next', 'x2', 'x1'),
                ('prev', 'x1', 'x2'),
            ):
                target_line = PlatformGraphBuilder._linked_standable_foothold(
                    source_line, direction, foothold_lines, standable_ids,
                    allow_four_px_bridge=True,
                )
                if target_line is None:
                    continue
                target_fh_id = int(target_line['id'])
                target_platform = foothold_to_platform.get(target_fh_id)
                if target_platform is None:
                    continue
                if target_platform.id == source_platform.id:
                    continue

                # next/prev 绑定原始端点；由“朝哪个端点离开”直接确定按键
                # 方向，不再通过两个平台的 X/Y 间隙猜测。
                endpoint_x = int(source_line[endpoint_key])
                other_x = int(source_line[other_endpoint_key])
                action = 'WALK_RIGHT' if endpoint_x >= other_x else 'WALK_LEFT'
                edge_key = (source_platform.id, target_platform.id, action)
                if edge_key in walk_edges:
                    continue
                walk_edges.add(edge_key)
                graph.add_edge(PlatformEdge(
                    from_id=source_platform.id,
                    to_id=target_platform.id,
                    action=action,
                    cost=0.15,
                    trigger_x=endpoint_x,
                    target_y=target_platform.y,
                    description=(
                        f"沿 foothold {source_fh_id} 的 {direction} 拓扑到 {target_fh_id}，"
                        f"从平台 {source_platform.id} 步行至平台 {target_platform.id}"
                    )
                ))

        # 3. 生成垂直下跳边 (DOWN_JUMP)
        for p_upper in nodes:
            for p_lower in nodes:
                if p_upper.id == p_lower.id:
                    continue
                height_drop = p_lower.y - p_upper.y
                # 30~34px 的近层台阶同样可以下跳。103000201 的 P14
                # 到 P7 相隔 33px，再沿 WZ 接缝步行即可到 P11。
                if 30 <= height_drop <= 320:
                    overlap_min = max(p_upper.x_min, p_lower.x_min)
                    overlap_max = min(p_upper.x_max, p_lower.x_max)
                    # 下跳必须有足够宽的共同踏板区域，避免在窄重叠
                    # 平台（如 P81->P80，仅 34px）上误触发 DOWN_JUMP。
                    if overlap_max - overlap_min >= 50:
                        safe_lo = overlap_min + 10
                        safe_hi = overlap_max - 10
                        # 目标平台中点位于共同安全区时直接采用；若目标中点
                        # 落在共同区外，不能把它硬夹到 safe_lo/safe_hi 端点。
                        # 端点在量化坐标和短 foothold 接缝处很容易导致
                        # DOWN+JUMP 偶发不触发。此时改用共同安全区的真中点。
                        if safe_lo <= p_lower.center_x <= safe_hi:
                            trigger_x = int(round(p_lower.center_x))
                        else:
                            trigger_x = int(round((safe_lo + safe_hi) / 2.0))
                        graph.add_edge(PlatformEdge(
                            from_id=p_upper.id,
                            to_id=p_lower.id,
                            action="DOWN_JUMP",
                            # [下+跳] 是低风险的确定性近层动作，应显著
                            # 低于走落/长落边，保证寻路优先逐层下跳。
                            cost=round(0.18 + height_drop / 1000.0, 2),
                            trigger_x=trigger_x,
                            trigger_x_range=(safe_lo, safe_hi),
                            landing_x=float(trigger_x),
                            target_y=p_lower.y,
                            description=f"从平台 {p_upper.id} 走至 X=[{overlap_min+10}~{overlap_max-10}] 触发 [下+跳] 跌落至平台 {p_lower.id}"
                        ))

        # A deep DOWN_JUMP is only valid in a vertical corridor free of every
        # intermediate platform. Looking at the globally nearest lower layer
        # is insufficient: a short platform elsewhere can hide an intervening
        # platform directly under the planned X (103000102 P26->P18).
        # Keep any genuinely open part of the source/target overlap, so a
        # partial blocker does not erase a valid deeper drop on another X.
        for p_upper in nodes:
            outgoing = graph.edges.get(p_upper.id, [])
            retained = []
            for edge in outgoing:
                if edge.action != "DOWN_JUMP" or edge.to_id not in graph.nodes:
                    retained.append(edge)
                    continue
                target = graph.nodes[edge.to_id]
                if edge.trigger_x_range is None:
                    retained.append(edge)
                    continue
                safe_intervals = [tuple(sorted(map(float, edge.trigger_x_range)))]
                has_intermediate_blocker = False
                for middle in nodes:
                    if middle.id in (p_upper.id, target.id):
                        continue
                    if not (p_upper.y + 5 < middle.y < target.y - 5):
                        continue
                    blocked_lo = float(middle.x_min) - 10.0
                    blocked_hi = float(middle.x_max) + 10.0
                    if any(blocked_lo < hi and blocked_hi > lo for lo, hi in safe_intervals):
                        has_intermediate_blocker = True
                    next_intervals = []
                    for lo, hi in safe_intervals:
                        if blocked_hi <= lo or blocked_lo >= hi:
                            next_intervals.append((lo, hi))
                        else:
                            if blocked_lo - lo >= 20.0:
                                next_intervals.append((lo, blocked_lo))
                            if hi - blocked_hi >= 20.0:
                                next_intervals.append((blocked_hi, hi))
                    safe_intervals = next_intervals
                    if not safe_intervals:
                        break
                if not safe_intervals:
                    continue
                # A bypass around an intervening platform needs enough room
                # for the player's width and minimap quantization.  A sliver
                # beside P16 (103000201 P25->P14) is not a reliable drop;
                # descend to P16 first.  This extra clearance is unnecessary
                # when there is no intervening platform at all.
                if has_intermediate_blocker:
                    safe_intervals = [
                        interval for interval in safe_intervals
                        if interval[1] - interval[0] >= 80.0
                    ]
                    if not safe_intervals:
                        continue
                # The edge has one trigger range; retain its widest clear
                # corridor rather than an unsafe union of disjoint spans.
                lo, hi = max(
                    safe_intervals,
                    key=lambda interval: (
                        interval[1] - interval[0],
                        -abs((interval[0] + interval[1]) / 2.0 - target.center_x),
                    ),
                )
                edge.trigger_x_range = (int(math.ceil(lo)), int(math.floor(hi)))
                original_trigger = float(
                    edge.trigger_x if edge.trigger_x is not None else target.center_x
                )
                edge.trigger_x = int(round(max(lo, min(hi, original_trigger))))
                edge.landing_x = float(edge.trigger_x)
                retained.append(edge)
            graph.edges[p_upper.id] = retained

        # 4. 生成跳跃动作边 (平跳、阶梯跳、斜向高空落差跳)
        for p1 in nodes:
            for p2 in nodes:
                if p1.id == p2.id:
                    continue
                dy = p2.y - p1.y
                gap_right = p2.x_min - p1.x_max
                gap_left = p1.x_min - p2.x_max
                center_dx = p2.center_x - p1.center_x
                overlap_min = max(p1.x_min, p2.x_min)
                overlap_max = min(p1.x_max, p2.x_max)

                # 3.1 水平平跳与助跑跳 (-70 <= dy <= 90)
                if -70 <= dy <= 90:
                    # 实机验证表明 124px 的错位（P14->P10）无法靠普通
                    # 平跳跨越；将普通跳的安全间隔收紧到 70px，保留
                    # P14->P13(44px)、P22->P9(34px) 等可行边。
                    normal_jump_gap_limit = 70
                    # 目标仅在侧下方一小段时，不需要先反向蓄跑再按跳跃。
                    # 持续走出源平台边缘并在空中保持方向即可自然落入目标。
                    # 此规则必须排在通用 JUMP_LEFT/RIGHT 前面；例如
                    # 107000100 的 P11->P7：左移24px、下落59px。
                    if p1.length <= 100 and 35 <= dy <= 90 and 0 < gap_right <= 45:
                        launch_x = p1.x_max + 5
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id, to_id=p2.id,
                            action="WALK_RIGHT_DROP",
                            cost=round(0.55 + dy / 350.0, 2),
                            trigger_x=p1.x_max - 5,
                            landing_x=PlatformGraphBuilder._walk_drop_landing_x(
                                launch_x, p1.y, p2.y, "right"),
                            target_y=p2.y,
                            description=f"从平台 {p1.id} 向右走出边缘自然落下至平台 {p2.id}"
                        ))
                    elif p1.length <= 100 and 35 <= dy <= 90 and 0 < gap_left <= 45:
                        launch_x = p1.x_min - 5
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id, to_id=p2.id,
                            action="WALK_LEFT_DROP",
                            cost=round(0.55 + dy / 350.0, 2),
                            trigger_x=p1.x_min + 5,
                            landing_x=PlatformGraphBuilder._walk_drop_landing_x(
                                launch_x, p1.y, p2.y, "left"),
                            target_y=p2.y,
                            description=f"从平台 {p1.id} 向左走出边缘自然落下至平台 {p2.id}"
                        ))
                    elif 0 < gap_right <= normal_jump_gap_limit:
                        # 合并长平台的平均Y会掩盖边缘处的陡坡。侧跳真正使用
                        # 源台右端与目标左端，必须按这两个局部表面检查跳跃
                        # 高度；101000000 的 P54->P56 平均只高48px，但跳口
                        # 实际高约100px，基础跳跃不可能到达。
                        source_edge_y = p1.surface_y_at(p1.x_max)
                        target_edge_y = p2.surface_y_at(p2.x_min)
                        if not PlatformGraphBuilder._jump_height_reachable(
                            target_edge_y, source_edge_y
                        ):
                            continue
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="JUMP_RIGHT",
                            cost=round(1.0 + gap_right / 180.0, 2),
                            trigger_x=p1.x_max - 10,
                            landing_x=float(p2.center_x),
                            target_y=int(round(target_edge_y)),
                            description=f"从平台 {p1.id} 右侧助跑起跳至平台 {p2.id}"
                        ))
                    elif 0 < gap_left <= normal_jump_gap_limit:
                        source_edge_y = p1.surface_y_at(p1.x_min)
                        target_edge_y = p2.surface_y_at(p2.x_max)
                        if not PlatformGraphBuilder._jump_height_reachable(
                            target_edge_y, source_edge_y
                        ):
                            continue
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="JUMP_LEFT",
                            cost=round(1.0 + gap_left / 180.0, 2),
                            trigger_x=p1.x_min + 10,
                            landing_x=float(p2.center_x),
                            target_y=int(round(target_edge_y)),
                            description=f"从平台 {p1.id} 左侧助跑起跳至平台 {p2.id}"
                        ))
                    elif (abs(center_dx) <= 35 and -75 <= dy <= -40
                          and PlatformGraphBuilder._jump_height_reachable(p2.y, p1.y)):
                        # 垂直台阶直跳 (如右侧连环立柱)
                        step_x = p1.center_x
                        source_step_y = p1.surface_y_at(step_x)
                        target_step_y = p2.surface_y_at(step_x)
                        # 合并长平台的平均 Y 不能代表起跳处高度。斜坡上
                        # 必须用同一 X 的真实局部表面判断最大上升高度。
                        if not PlatformGraphBuilder._jump_height_reachable(
                            target_step_y, source_step_y
                        ):
                            continue
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="JUMP_UP",
                            # 垂直直跳比绕经多个碎平台更稳定、更短；
                            # 代价必须低于连续 WALK，否则 A* 会选绕路。
                            cost=0.55,
                            trigger_x=step_x,
                            # JUMP_UP is executed with no horizontal key;
                            # its predicted landing must therefore stay at
                            # the launch X (not drift by walk speed).
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                step_x, target_step_y, source_step_y, "right",
                                horizontal_speed=0.0),
                            target_y=int(round(target_step_y)),
                            description=f"从平台 {p1.id} 向上起跳至台阶 {p2.id}"
                        ))
                    elif (enable_teleport and -teleport_distance_px <= dy < -75
                          and (abs(center_dx) <= 35 or (overlap_max - overlap_min) >= 20)):
                        # 低平台往垂直方向的高平台：普通上跳跳不到但瞬移可以到达时，使用上+瞬移替代普通上跳
                        step_x = (
                            (overlap_min + overlap_max) // 2
                            if (overlap_max - overlap_min) >= 20
                            else p1.center_x
                        )
                        source_step_y = p1.surface_y_at(step_x)
                        target_step_y = p2.surface_y_at(step_x)
                        rise = source_step_y - target_step_y
                        if 75.0 < rise <= teleport_distance_px:
                            graph.add_edge(PlatformEdge(
                                from_id=p1.id,
                                to_id=p2.id,
                                action="TELEPORT_UP",
                                cost=0.55,
                                trigger_x=step_x,
                                landing_x=float(step_x),
                                target_y=int(round(target_step_y)),
                                description=f"从平台 {p1.id} 向上瞬移至高平台 {p2.id}（高度差 {rise:.0f}px）"
                            ))
                    elif (overlap_max - overlap_min) >= 50 and -75 <= dy <= -40:
                        # 上层平台只覆盖下层的一端时，两个中心可能相差很
                        # 大，但只要有效重叠至少 50px，角色可走到重叠区后
                        # 斜向上跳。不能用中心差 <=80px 把这类台阶漏掉；
                        # 例如 100030000 的 P48 -> P49。
                        step_dir = "RIGHT" if center_dx > 0 else "LEFT"
                        # 重叠区斜向上跳不能机械取中点。角色在 Alt 后还会
                        # 持续保持方向约 0.20s：左跳若从中点起跳，会在起跳
                        # 前的助跑阶段已逼近源台左缘（P71->P74 即属此类）。
                        # 因此起跳线放在“出发反方向”的安全端：向右取重叠
                        # 左端、向左取重叠右端，均留 10px 余量。
                        step_x = (
                            overlap_min + 10 if step_dir == "RIGHT"
                            else overlap_max - 10
                        )
                        source_step_y = p1.surface_y_at(step_x)
                        target_step_y = p2.surface_y_at(step_x)
                        if not PlatformGraphBuilder._jump_height_reachable(
                            target_step_y, source_step_y
                        ):
                            continue
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action=f"JUMP_{step_dir}",
                            cost=1.0,
                            trigger_x=step_x,
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                step_x, target_step_y, source_step_y,
                                "right" if center_dx > 0 else "left"),
                            target_y=int(round(target_step_y)),
                            description=f"从平台 {p1.id} 重叠区斜向{('右' if center_dx > 0 else '左')}上跳至平台 {p2.id}"
                        ))
                    elif (overlap_max - overlap_min) >= 20 and abs(center_dx) <= 80 and -75 <= dy <= -40:
                        # 两个平台存在有效重叠、但中心错位稍大时，
                        # 仍可通过带方向的斜向上跳连接（例如 P80->P81）。
                        step_dir = "RIGHT" if center_dx > 0 else "LEFT"
                        step_action = f"JUMP_{step_dir}"
                        # 同上：以重叠区靠近助跑起点的一端作为 Alt 起跳线，
                        # 为起跳前连续助跑和 Alt 后 0.20s 横向保持留出空间。
                        step_x = (
                            overlap_min + 10 if step_dir == "RIGHT"
                            else overlap_max - 10
                        )
                        source_step_y = p1.surface_y_at(step_x)
                        target_step_y = p2.surface_y_at(step_x)
                        if not PlatformGraphBuilder._jump_height_reachable(
                            target_step_y, source_step_y
                        ):
                            continue
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action=step_action,
                            cost=1.0,
                            trigger_x=step_x,
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                step_x, target_step_y, source_step_y,
                                "right" if center_dx > 0 else "left"),
                            target_y=int(round(target_step_y)),
                            description=f"从平台 {p1.id} 斜向{('右' if center_dx > 0 else '左')}上跳至台阶 {p2.id}"
                        ))
                    elif -75 <= dy <= -40 and -2 <= gap_right <= 0:
                        # 边缘恰好相接的上台：间隙为 0 不是不可达。向右
                        # 起跳后落在目标左端附近，覆盖 P54 -> P57 的断边。
                        launch_x = p1.x_max - 8
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id, to_id=p2.id, action="JUMP_RIGHT",
                            cost=1.0, trigger_x=launch_x,
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                launch_x, p2.y, p1.y, "right"),
                            target_y=p2.y,
                            description=f"从平台 {p1.id} 右侧边缘上跳至相接平台 {p2.id}"
                        ))
                    elif -75 <= dy <= -40 and -2 <= gap_left <= 0:
                        # 与上项对称：向左上台后落在目标右端附近。
                        launch_x = p1.x_min + 8
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id, to_id=p2.id, action="JUMP_LEFT",
                            cost=1.0, trigger_x=launch_x,
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                launch_x, p2.y, p1.y, "left"),
                            target_y=p2.y,
                            description=f"从平台 {p1.id} 左侧边缘上跳至相接平台 {p2.id}"
                        ))
                    elif (35 <= dy <= 90 and (overlap_max - overlap_min) >= 20
                          and (p2.length >= p1.length + 20 or abs(center_dx) <= 80)):
                        # 下方平台重叠区不足 50px 时不使用 DOWN_JUMP，
                        # 但目标平台明显更宽时，允许通过带方向的普通跳返回；
                        # 此时不以两个平台中心距离作为可达性硬限制。
                        step_dir = "RIGHT" if center_dx > 0 else "LEFT"
                        step_action = f"JUMP_{step_dir}"
                        step_x = (overlap_min + overlap_max) // 2
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action=step_action,
                            cost=1.0,
                            trigger_x=step_x,
                            landing_x=float(p2.center_x),
                            target_y=p2.y,
                            description=f"从平台 {p1.id} 向{('右' if center_dx > 0 else '左')}跳回平台 {p2.id}"
                        ))

                # 3.2 高台向斜下方落差跳跃 (90 < dy <= 300)
                # 垂直落差超过 300px 时，必须使用完整滞空/落点模型，
                # 归入 LONG_DROP，而不是普通固定时长的 DROP。
                elif 90 < dy <= 300:
                    # 实机跳跃的水平射程明显小于“允许 160px 间隔”的旧经验值。
                    # 用高度差估算本次下落可覆盖的安全间隔：高度越大，空中
                    # 时间越长，但仍设置上限，避免把 P22->P13 这类实测不可达
                    # 的平台错误连成直达边（P22->P9 的 34px 间隔仍可保留）。
                    jump_gap_limit = min(150, max(45, int(dy * 0.40)))
                    # 长平台中心可能离目标很远，但靠近目标的边缘只隔
                    # 数十像素。自然下落是否可行应以真实边缘与预测落点
                    # 判断；103000201 P28→P25 的中心相差 315px，
                    # 边缘仅隔 25px，落点却稳在目标内。
                    right_fall_x = PlatformGraphBuilder._walk_drop_landing_x(
                        p1.x_max + 4, p1.y, p2.y, "right"
                    )
                    left_fall_x = PlatformGraphBuilder._walk_drop_landing_x(
                        p1.x_min - 4, p1.y, p2.y, "left"
                    )
                    right_fall_reaches = p2.x_min + 8 <= right_fall_x <= p2.x_max - 8
                    left_fall_reaches = p2.x_min + 8 <= left_fall_x <= p2.x_max - 8
                    # 极小间隔的情况实机表现为“走出边缘自然下落”，
                    # 不应发送跳跃键（例如本图 P22 -> P9）。
                    if 0 < gap_right <= 45 and (
                        0 < center_dx <= 190 or right_fall_reaches
                    ):
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id, to_id=p2.id,
                            action="WALK_RIGHT_DROP",
                            cost=round(0.9 + dy / 350.0, 2),
                            trigger_x=p1.x_max - 5, target_y=p2.y,
                            landing_x=PlatformGraphBuilder._walk_drop_landing_x(
                                p1.x_max - 5, p1.y, p2.y, "right"),
                            description=f"从平台 {p1.id} 右侧走出边缘自然下落至平台 {p2.id}"
                        ))
                    elif 0 < gap_left <= 45 and (
                        -190 <= center_dx < 0 or left_fall_reaches
                    ):
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id, to_id=p2.id,
                            action="WALK_LEFT_DROP",
                            cost=round(0.9 + dy / 350.0, 2),
                            trigger_x=p1.x_min + 5, target_y=p2.y,
                            landing_x=PlatformGraphBuilder._walk_drop_landing_x(
                                p1.x_min + 5, p1.y, p2.y, "left"),
                            description=f"从平台 {p1.id} 左侧走出边缘自然下落至平台 {p2.id}"
                        ))
                    elif 0 < center_dx <= 190 and 0 < gap_right <= jump_gap_limit:
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="JUMP_RIGHT_DROP",
                            cost=round(1.2 + dy / 300.0, 2),
                            trigger_x=p1.x_max - 5,
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                p1.x_max - 5, p2.y, p1.y, "right"),
                            target_y=p2.y,
                            description=f"从高台 {p1.id} 向右斜下方大跳跌落至平台 {p2.id}"
                        ))
                    elif -190 <= center_dx < 0 and 0 < gap_left <= jump_gap_limit:
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="JUMP_LEFT_DROP",
                            cost=round(1.2 + dy / 300.0, 2),
                            trigger_x=p1.x_min + 5,
                            landing_x=PlatformGraphBuilder._jump_model_landing_x(
                                p1.x_min + 5, p2.y, p1.y, "left"),
                            target_y=p2.y,
                            description=f"从高台 {p1.id} 向左斜下方大跳跌落至平台 {p2.id}"
                        ))

                # 3.3 长距离跨层下落：显式模拟下落轨迹是否会被中间平台截获。
                # 只有存在一条从起跳边缘到目标平台的轨迹，能在每个中间
                # 平台高度绕开其有效宽度时，才创建直达边。这样 A* 才能
                # 区分“必须经停中层”与“可以越过中层直达”。
                elif 300 < dy <= 1000:
                    # 当前平台已经存在合法的 [下+跳] 近层出口时，优先
                    # 走该安全出口，禁止再生成跨越多层的长落差捷径。
                    # 例如 P44 与 P43 有 120px 重叠且仅差 126px，必须
                    # 先 P44->P43 DOWN_JUMP，不能被 P44->P21/P32 的
                    # 自然落下边绕过。
                    has_local_down_jump = any(
                        edge.action == "DOWN_JUMP"
                        for edge in graph.edges.get(p1.id, [])
                    )
                    # 近层 DOWN_JUMP 只能否决“非常深”的捷径；不能据此
                    # 把仍在基础跳跃滞空范围内的斜向长跳一并删除。
                    # 例如 P78 -> P36 为 ΔY=360、左移约129px，轨迹模拟
                    # 通过且实测可达；P78 虽有垂直 P78->P72 的 DOWN_JUMP，
                    # 但不应阻止这条左向长跑跳。P44 那类深层（>=450px）
                    # 仍优先走相邻 DOWN_JUMP，避免绕过中层。
                    if has_local_down_jump and dy >= 450:
                        continue

                    # 下方宽平台可能直接覆盖当前窄台的一侧出口。例如
                    # P22[-658,-602] -> P1[-838,838]：从右缘 X=-598
                    # 走出后会直落到 P1，且不会碰到右侧的 P9。旧逻辑仅
                    # 接受 ``gap > 0`` 的自然下落，因而漏掉这种“目标已
                    # 覆盖出口”的情形，错误退化为 LONG_DROP + Alt。
                    #
                    # 对左右两个出口逐一沿竖直下落线检查中间踏板；只有
                    # 没有拦截时才建立自然下落边，避免把会落在中层平台上
                    # 的路线误标为直达。
                    vertical_exit_options = []
                    for direction, exit_x in (
                        ("left", p1.x_min - 4),
                        ("right", p1.x_max + 4),
                    ):
                        if not (p2.x_min + 2 <= exit_x <= p2.x_max - 2):
                            continue
                        intercepted = any(
                            mid.id not in (p1.id, p2.id)
                            and p1.y + 15 < mid.y < p2.y - 15
                            # 角色脚点/碰撞盒有宽度，且松开方向键后仍有
                            # 少量水平惯性。P84 左出口 X=228 虽比 P83
                            # 右端 X=213 多15px，实机仍会先被 P83 接住。
                            # 对中间踏板扩张24px后再判断，保留真正净空的
                            # 另一侧出口（该例会正确选择 P84 右侧）。
                            and (mid.x_min - 24) <= exit_x <= (mid.x_max + 24)
                            for mid in nodes
                        )
                        if not intercepted:
                            vertical_exit_options.append((direction, exit_x))

                    if vertical_exit_options:
                        # 两侧都可下落时优先选择离目标中点更近的一侧，保持
                        # 一般地图上的选择稳定；P22->P1 只有右侧可用。
                        direction, exit_x = min(
                            vertical_exit_options,
                            key=lambda item: abs(item[1] - p2.center_x),
                        )
                        action = "WALK_LEFT_DROP" if direction == "left" else "WALK_RIGHT_DROP"
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action=action,
                            # 整段竖直自然下落一旦已通过中层拦截检查，就不应
                            # 被“先落一块短台、再 DOWN_JUMP”的组合路线抢走。
                            # 后者会重新引入窄台精调，违背了直落的目的。
                            cost=round(0.45 + dy / 900.0, 2),
                            trigger_x=int(exit_x),
                            landing_x=float(exit_x),
                            target_y=p2.y,
                            description=(
                                f"从高台 {p1.id} 向{('左' if direction == 'left' else '右')}"
                                f"走出边缘垂直自然下落至平台 {p2.id}"
                            )
                        ))
                        continue

                    # 两个平台边缘间隙很小（如 P44[-208,748] ->
                    # P32[-293,-212]，仅 4px）时，正确操作是走出边缘
                    # 后自然下落，不是从平台另一侧助跑起跳。这里不再以
                    # 两个平台中心的距离作为限制，因为宽平台中心会严重
                    # 放大实际所需的横向位移。
                    # 宽平台的边缘可容许较大的“走出后自然下落”间隙；
                    # 但窄台不能照搬 200px。P78/P36 均为 56px 窄台、
                    # 间隙124px，实测必须使用 LONG_DROP 的跑跳，若标为
                    # WALK_DROP 会在未建立速度时直接落空。
                    edge_drop_gap_limit = 45 if p1.length < 90 else 200
                    if 0 < gap_right <= edge_drop_gap_limit:
                        safe_landing_x = float(min(p2.x_max - 8, max(p2.x_min + 8, p2.x_min + 16)))
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="WALK_RIGHT_DROP",
                            # 同样是自然下落时，优先选择边缘间隙更小的
                            # 落脚平台；否则 A* 会把 4px 的 P44->P32
                            # 与 168px 的 P44->P21 视为同价并随机绕远。
                            cost=round(0.9 + dy / 500.0 + gap_right / 100.0, 2),
                            trigger_x=p1.x_max - 4,
                            landing_x=safe_landing_x,
                            target_y=p2.y,
                            description=(
                                f"从高台 {p1.id} 右侧走出边缘自然下落至平台 {p2.id} "
                                f"(边缘间隙 {gap_right}px)"
                            )
                        ))
                        continue
                    if 0 < gap_left <= edge_drop_gap_limit:
                        safe_landing_x = float(max(p2.x_min + 8, min(p2.x_max - 8, p2.x_max - 16)))
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action="WALK_LEFT_DROP",
                            cost=round(0.9 + dy / 500.0 + gap_left / 100.0, 2),
                            trigger_x=p1.x_min + 4,
                            landing_x=safe_landing_x,
                            target_y=p2.y,
                            description=(
                                f"从高台 {p1.id} 左侧走出边缘自然下落至平台 {p2.id} "
                                f"(边缘间隙 {gap_left}px)"
                            )
                        ))
                        continue

                    # For the marginal 300~380px long jump, collision is
                    # determined by the character foot-point, not a
                    # conservative 10px platform shrink. Keep the full
                    # foothold interval so P78->P36 remains representable
                    # at its extreme edge. Preserve the older safety margin
                    # for deeper cross-layer drops.
                    clearance = 0 if dy <= 380 else 10
                    max_horizontal = min(280, int(70 + dy * 0.28))
                    best_option = None  # (horizontal_distance, direction, launch_x, landing_x)

                    for direction, launch_x in (
                        ("left", p1.x_min + 5),
                        ("right", p1.x_max - 5),
                    ):
                        reach_lo = launch_x - max_horizontal
                        reach_hi = launch_x + max_horizontal
                        target_lo = max(p2.x_min + clearance, reach_lo)
                        target_hi = min(p2.x_max - clearance, reach_hi)
                        if target_lo > target_hi:
                            continue

                        # Sample candidate landing Xs from nearest to farthest
                        # in the requested direction.  A square-root progress
                        # curve approximates horizontal travel during gravity
                        # fall more closely than linear interpolation.
                        sample_count = max(2, min(13, int((target_hi - target_lo) / 18) + 2))
                        samples = [
                            target_lo + (target_hi - target_lo) * i / (sample_count - 1)
                            for i in range(sample_count)
                        ]
                        if direction == "left":
                            samples.sort(reverse=True)
                        else:
                            samples.sort()

                        for landing_x in samples:
                            if direction == "left" and landing_x >= launch_x:
                                continue
                            if direction == "right" and landing_x <= launch_x:
                                continue

                            intercepted = False
                            for mid in nodes:
                                if mid.id in (p1.id, p2.id) or not (p1.y + 15 < mid.y < p2.y - 15):
                                    continue
                                # Use elapsed vertical-flight time rather than
                                # sqrt(height ratio). The latter advances X too
                                # far near the final intermediate platforms.
                                g = 2000.0
                                vy0 = -555.0
                                t_peak = abs(vy0) / g
                                h_peak = (vy0 * vy0) / (2.0 * g)
                                terminal_v = 670.0
                                terminal_t = terminal_v / g
                                terminal_y = 0.5 * g * terminal_t * terminal_t

                                def elapsed_to_drop(drop):
                                    descend = h_peak + max(0.0, float(drop))
                                    if descend <= terminal_y:
                                        return t_peak + math.sqrt(2.0 * descend / g)
                                    return (t_peak + terminal_t
                                            + (descend - terminal_y) / terminal_v)

                                progress = elapsed_to_drop(mid.y - p1.y) / elapsed_to_drop(dy)
                                x_at_mid = launch_x + (landing_x - launch_x) * progress
                                if (mid.x_min - clearance) <= x_at_mid <= (mid.x_max + clearance):
                                    intercepted = True
                                    break

                            if not intercepted:
                                option = (abs(landing_x - launch_x), direction, launch_x, landing_x)
                                if best_option is None or option[0] < best_option[0]:
                                    best_option = option
                                break

                    if best_option is not None:
                        _, direction, launch_x, landing_x = best_option
                        action = "JUMP_LEFT_LONG_DROP" if direction == "left" else "JUMP_RIGHT_LONG_DROP"
                        graph.add_edge(PlatformEdge(
                            from_id=p1.id,
                            to_id=p2.id,
                            action=action,
                            cost=round(1.3 + dy / 500.0, 2),
                            trigger_x=int(round(launch_x)),
                            landing_x=float(landing_x),
                            target_y=p2.y,
                            description=(
                                f"从高台 {p1.id} 向{('左' if direction == 'left' else '右')}长距下落，"
                                f"绕过中间平台直达 {p2.id} (预计落点 X={int(round(landing_x))})"
                            )
                        ))
