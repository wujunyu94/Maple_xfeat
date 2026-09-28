"""Read extracted Map.wz ``.img`` files as the map dictionary used by the app.

The application historically consumed an online map JSON.  This module
adapts an extracted WZ Property image to the same small, stable schema so the
platform builder and cross-map planner do not need a second implementation.
Network JSON remains a fallback in the callers.

The parser backend is the MIT-licensed ``wz-python`` source bundled in this
workspace under ``wz_python_tool``.  Canvas pixel payloads are never decoded.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple


class WzMapReadError(RuntimeError):
    """Raised when a local Map.wz image exists but cannot be decoded."""


def _numeric_sort_key(value: Any) -> Tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except (TypeError, ValueError):
        return (1, text)


class WzMapReader:
    """Load extracted map ``.img`` files using a four-byte WZ string IV."""

    REQUIRED_TOP_LEVEL = frozenset(
        {"info", "miniMap", "portal", "ladderRope", "foothold", "life"}
    )

    def __init__(
        self,
        map_root: str,
        iv_hex: str = "4D23C72B",
        parser_root: Optional[str] = None,
    ) -> None:
        self.map_root = os.path.abspath(map_root)
        project_root = os.path.dirname(self.map_root)
        self.parser_root = os.path.abspath(
            parser_root or os.path.join(project_root, "wz_python_tool")
        )
        clean_iv = "".join(str(iv_hex).split())
        try:
            self.iv = bytes.fromhex(clean_iv)
        except ValueError as exc:
            raise ValueError(f"无效的 WZ 字符串密钥：{iv_hex!r}") from exc
        if len(self.iv) != 4:
            raise ValueError("WZ 字符串密钥必须正好是 4 字节")

        self._backend: Optional[Tuple[Any, Any, Any]] = None
        self._cache: Dict[int, Tuple[Tuple[int, int], Dict[str, Any]]] = {}
        self._map_ids_cache: Optional[Tuple[int, ...]] = None
        self._lock = threading.RLock()

    def _load_backend(self) -> Tuple[Any, Any, Any]:
        if self._backend is not None:
            return self._backend
        if not os.path.isdir(self.parser_root):
            raise WzMapReadError(
                f"缺少 WZ 解析器目录：{self.parser_root}（将回退到地图 JSON）"
            )
        if self.parser_root not in sys.path:
            sys.path.insert(0, self.parser_root)
        try:
            from wzpy.crypto import WzKey
            from wzpy.canvas import decode_canvas
            from wzpy.wz_image import WzImage
        except Exception as exc:
            raise WzMapReadError(f"WZ 解析器加载失败：{exc}") from exc
        self._backend = (WzKey, WzImage, decode_canvas)
        return self._backend

    @staticmethod
    def _map_filename(map_id: int) -> str:
        return f"{int(map_id):09d}.img"

    def map_path(self, map_id: int) -> str:
        map_id = int(map_id)
        filename = self._map_filename(map_id)
        # Extracted Map.wz normally has Map/Map/Map0 ... Map9.
        category = filename[0]
        candidates = (
            os.path.join(self.map_root, "Map", f"Map{category}", filename),
            os.path.join(self.map_root, f"Map{category}", filename),
            os.path.join(self.map_root, filename),
        )
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def has_map(self, map_id: int) -> bool:
        return os.path.isfile(self.map_path(map_id))

    def source_signature(self, map_id: int) -> Dict[str, Optional[int]]:
        path = self.map_path(map_id)
        try:
            stat = os.stat(path)
            return {
                "wz_img_mtime_ns": int(stat.st_mtime_ns),
                "wz_img_size": int(stat.st_size),
            }
        except OSError:
            return {"wz_img_mtime_ns": None, "wz_img_size": None}

    def list_map_ids(self) -> Sequence[int]:
        """List available extracted maps; result is cached for route planning."""
        with self._lock:
            if self._map_ids_cache is not None:
                return self._map_ids_cache
            result = set()
            roots = (os.path.join(self.map_root, "Map"), self.map_root)
            for root in roots:
                if not os.path.isdir(root):
                    continue
                try:
                    category_names = os.listdir(root)
                except OSError:
                    continue
                for category_name in category_names:
                    category_path = os.path.join(root, category_name)
                    if not os.path.isdir(category_path) or not category_name.startswith("Map"):
                        continue
                    try:
                        names = os.listdir(category_path)
                    except OSError:
                        continue
                    for name in names:
                        stem, ext = os.path.splitext(name)
                        if ext.lower() == ".img" and stem.isdigit():
                            result.add(int(stem))
            self._map_ids_cache = tuple(sorted(result))
            return self._map_ids_cache

    @staticmethod
    def _children(node: Any) -> List[Any]:
        if node is None:
            return []
        try:
            return list(node.children())
        except Exception:
            return []

    @staticmethod
    def _child(node: Any, name: Any) -> Any:
        if node is None:
            return None
        try:
            return node.child(str(name))
        except Exception:
            return None

    @classmethod
    def _value(cls, node: Any, name: Any, default: Any = None) -> Any:
        child = cls._child(node, name)
        if child is None:
            return default
        try:
            value = child.value
        except Exception:
            return default
        return default if value is None else value

    @classmethod
    def _scalars(cls, node: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for child in cls._children(node):
            if cls._children(child):
                continue
            try:
                value = child.value
            except Exception:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[str(child.name)] = value
        return result

    @classmethod
    def _sorted_children(cls, node: Any) -> List[Any]:
        return sorted(cls._children(node), key=lambda item: _numeric_sort_key(item.name))

    @classmethod
    def _extract_footholds(cls, foothold_root: Any) -> List[Dict[str, Any]]:
        # 标准 Map.wz 的 foothold 编号在整张地图内唯一；必须保留它，
        # 因为 next/prev 会跨嵌套 group 串起同一条物理地面。极少数非标准
        # 导出若真的出现重号，才退化为稳定的全图编号。
        entries: List[Tuple[int, int, int, Any]] = []
        for layer_node in cls._sorted_children(foothold_root):
            try:
                layer_id = int(layer_node.name)
            except (TypeError, ValueError):
                continue
            for group_node in cls._sorted_children(layer_node):
                try:
                    group_id = int(group_node.name)
                except (TypeError, ValueError):
                    continue
                for line_node in cls._sorted_children(group_node):
                    try:
                        local_id = int(line_node.name)
                    except (TypeError, ValueError):
                        continue
                    required = ("x1", "y1", "x2", "y2")
                    if any(cls._child(line_node, key) is None for key in required):
                        continue
                    entries.append((layer_id, group_id, local_id, line_node))

        local_counts: Dict[int, int] = {}
        for _layer_id, _group_id, local_id, _node in entries:
            local_counts[local_id] = local_counts.get(local_id, 0) + 1
        globally_unique = all(count == 1 for count in local_counts.values())
        assigned_ids = {
            (layer_id, group_id, local_id): (local_id if globally_unique else index)
            for index, (layer_id, group_id, local_id, _node) in enumerate(entries, start=1)
        }
        unique_by_local = {
            local_id: assigned_ids[(layer_id, group_id, local_id)]
            for layer_id, group_id, local_id, _node in entries
            if local_counts.get(local_id) == 1
        }

        def linked_id(layer_id: int, group_id: int, raw_id: int) -> int:
            if raw_id == 0:
                return 0
            if globally_unique:
                return raw_id if raw_id in local_counts else 0
            return assigned_ids.get(
                (layer_id, group_id, raw_id), unique_by_local.get(raw_id, 0)
            )
        result: List[Dict[str, Any]] = []
        for layer_id, group_id, local_id, line_node in entries:
            raw_prev = int(cls._value(line_node, "prev", 0) or 0)
            raw_next = int(cls._value(line_node, "next", 0) or 0)
            item: Dict[str, Any] = {
                "id": assigned_ids[(layer_id, group_id, local_id)],
                "piece": local_id,
                "layerId": layer_id,
                "groupId": group_id,
                "x1": int(cls._value(line_node, "x1", 0)),
                "y1": int(cls._value(line_node, "y1", 0)),
                "x2": int(cls._value(line_node, "x2", 0)),
                "y2": int(cls._value(line_node, "y2", 0)),
                "prev": linked_id(layer_id, group_id, raw_prev),
                "next": linked_id(layer_id, group_id, raw_next),
            }
            for optional_name in ("force", "forbidFallDown"):
                optional_node = cls._child(line_node, optional_name)
                if optional_node is not None:
                    item[optional_name] = int(cls._value(line_node, optional_name, 0) or 0)
            result.append(item)
        return result

    @classmethod
    def _extract_ladder_ropes(cls, root: Any) -> List[Dict[str, Any]]:
        result = []
        for node in cls._sorted_children(root):
            values = cls._scalars(node)
            if not all(key in values for key in ("x", "y1", "y2")):
                continue
            is_ladder = bool(int(values.get("l", 0) or 0))
            result.append(
                {
                    "id": int(node.name) if str(node.name).isdigit() else len(result),
                    "x": int(values["x"]),
                    "y1": int(values["y1"]),
                    "y2": int(values["y2"]),
                    "isLadder": is_ladder,
                    "l": int(is_ladder),
                    "uf": int(values.get("uf", 0) or 0),
                    "page": int(values.get("page", 0) or 0),
                }
            )
        return result

    @classmethod
    def _extract_portals(cls, root: Any) -> List[Dict[str, Any]]:
        result = []
        aliases = {"pn": "portalName", "tn": "toName", "pt": "type", "tm": "toMap"}
        for node in cls._sorted_children(root):
            values = cls._scalars(node)
            if "x" not in values or "y" not in values or "pt" not in values:
                continue
            item = dict(values)
            for source_name, target_name in aliases.items():
                item[target_name] = values.get(source_name, "" if source_name in ("pn", "tn") else 0)
            item["id"] = int(node.name) if str(node.name).isdigit() else len(result)
            for int_name in ("x", "y", "type", "toMap"):
                item[int_name] = int(item.get(int_name, 0) or 0)
            result.append(item)
        return result

    @classmethod
    def _extract_life(cls, root: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        mobs: List[Dict[str, Any]] = []
        npcs: List[Dict[str, Any]] = []
        seen_mobs, seen_npcs = set(), set()
        for node in cls._sorted_children(root):
            values = cls._scalars(node)
            life_type = str(values.get("type", "")).lower()
            try:
                life_id = int(values.get("id"))
            except (TypeError, ValueError):
                continue
            target, seen = (mobs, seen_mobs) if life_type == "m" else (npcs, seen_npcs)
            if life_type not in ("m", "n") or life_id in seen:
                continue
            seen.add(life_id)
            target.append({"id": life_id})
        return mobs, npcs

    @classmethod
    def _adapt_tree(
        cls,
        root: Any,
        map_id: int,
        source_path: str,
        canvas_decoder: Optional[Any] = None,
    ) -> Dict[str, Any]:
        info_node = cls._child(root, "info")
        info = cls._scalars(info_node)
        minimap_node = cls._child(root, "miniMap")
        mini_values = cls._scalars(minimap_node)
        canvas_node = cls._child(minimap_node, "canvas")
        canvas_width = int(getattr(canvas_node, "width", 0) or 0)
        canvas_height = int(getattr(canvas_node, "height", 0) or 0)
        mini_map = {
            "width": int(mini_values.get("width", 0) or 0),
            "height": int(mini_values.get("height", 0) or 0),
            "centerX": int(mini_values.get("centerX", 0) or 0),
            "centerY": int(mini_values.get("centerY", 0) or 0),
            "magnification": int(mini_values.get("mag", 0) or 0),
            # Canvas 像素不需要解压或 Base64 化，但其真实尺寸必须保留。
            # Map.wz 的 width/height 是世界范围，不能用 width / mag 猜图像
            # 尺寸（例如 107000000 实际为 206x83，而非 826x332）。
            "canvasWidth": canvas_width,
            "canvasHeight": canvas_height,
        }
        # 大地图会滚动小地图视口，世界坐标需要用完整 Canvas 做模板匹配
        # 才能反推出摄像机 offset。Canvas 通常仅几百像素，按 PNG 缓存在
        # 地图字典中即可复用旧 JSON 路径的成熟坐标算法。
        if canvas_node is not None and canvas_decoder is not None:
            try:
                canvas_image = canvas_decoder(canvas_node, region="GMS")
                buffer = io.BytesIO()
                canvas_image.save(buffer, format="PNG")
                mini_map["canvas"] = base64.b64encode(buffer.getvalue()).decode("ascii")
            except Exception:
                # 罕见像素格式不能影响 foothold/portal 等结构读取；此时仍
                # 使用上面的真实 canvasWidth/canvasHeight 做几何换算。
                pass

        vr_keys = ("VRLeft", "VRRight", "VRTop", "VRBottom")
        vr_bounds = None
        if all(key in info for key in vr_keys):
            left, right = int(info["VRLeft"]), int(info["VRRight"])
            top, bottom = int(info["VRTop"]), int(info["VRBottom"])
            vr_bounds = {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom,
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            }

        mobs, npcs = cls._extract_life(cls._child(root, "life"))
        return {
            "id": int(map_id),
            "name": f"Map_{int(map_id)}",
            "streetName": "WZ",
            "backgroundMusic": info.get("bgm", ""),
            "returnMap": int(info.get("returnMap", 999999999) or 999999999),
            "isReturnMap": int(info.get("returnMap", -1) or -1) == int(map_id),
            "isTown": bool(int(info.get("town", 0) or 0)),
            "mapMark": info.get("mapMark", ""),
            "mobRate": float(info.get("mobRate", 1.0) or 1.0),
            "miniMap": mini_map,
            "vrBounds": vr_bounds,
            "footholds": cls._extract_footholds(cls._child(root, "foothold")),
            "ladderRopes": cls._extract_ladder_ropes(cls._child(root, "ladderRope")),
            "portals": cls._extract_portals(cls._child(root, "portal")),
            "mobs": mobs,
            "npcs": npcs,
            "_source": "wz_img",
            "_source_path": os.path.abspath(source_path),
        }

    def load_map(self, map_id: int) -> Dict[str, Any]:
        map_id = int(map_id)
        path = self.map_path(map_id)
        try:
            stat = os.stat(path)
        except OSError as exc:
            raise FileNotFoundError(f"缺少本地地图 IMG：{path}") from exc
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
        with self._lock:
            cached = self._cache.get(map_id)
            if cached is not None and cached[0] == signature:
                return cached[1]
            WzKey, WzImage, decode_canvas = self._load_backend()
            try:
                with open(path, "rb") as stream:
                    data = stream.read()
                image = WzImage.from_bytes(data, key=WzKey(self.iv), name=os.path.basename(path))
                root = image.parse_partial(only=self.REQUIRED_TOP_LEVEL)
                payload = self._adapt_tree(
                    root, map_id, path, canvas_decoder=decode_canvas
                )
                if getattr(image, "truncated", False):
                    warnings = "; ".join(getattr(image, "parse_warnings", []) or [])
                    raise WzMapReadError(f"地图 IMG 数据不完整：{path} {warnings}".strip())
            except WzMapReadError:
                raise
            except Exception as exc:
                raise WzMapReadError(f"地图 IMG 解析失败 {path}：{exc}") from exc
            self._cache[map_id] = (signature, payload)
            return payload
