"""
map_resolver.py - 游戏小地图名称 OCR 自动识别、本地自定义持久化优先与 GMS v83 官方怪物解析引擎
"""

import os
import json
import re
import math
import threading
import time
from typing import Optional, Tuple, Dict, List, Any, Callable
import numpy as np
import cv2
from src.vision.asset_downloader import AssetDownloader
from src.vision.ocr_process import IsolatedMapOCR

class MapResolver:
    def __init__(
        self,
        downloader: Optional[AssetDownloader] = None,
        ocr_roi: Optional[Dict[str, int]] = None,
        map_data_reader: Optional[Any] = None,
    ):
        # Construct the OCR worker lazily so model loading cannot block Tk.
        self.ocr = None
        self._ocr_lock = threading.Lock()
        self.ocr_timing_callback: Optional[
            Callable[[str, str, Optional[float], Optional[str]], None]
        ] = None
        self.ocr_status_callback: Optional[Callable[[str], None]] = None
        self.downloader = downloader or AssetDownloader(region="gms", version="83")
        # 本地 Map.wz IMG 是地图结构的主来源；在线接口只在本地
        # 文件缺失或解析失败时兜底。保持 Any 以避免让 OCR 模块硬依赖
        # 某一种 WZ 解析器实现。
        self.map_data_reader = map_data_reader
        self.current_map_name: str = "未知地图"
        self.current_map_id: Optional[int] = None
        self.current_map_info: Optional[Dict] = None
        self.current_map_mobs: List[Dict] = []
        self.last_unmatched_ocr_name: Optional[str] = None
        # OCR 识别范围 (x, y, w, h)，支持用户在 UI 界面自定义框选标定
        self.ocr_roi: Optional[Dict[str, int]] = ocr_roi
        # OCR 会周期运行。控制台只在首次结果或识别内容变化时打印，
        # 避免同一幅小地图标题每几秒刷屏。
        self._last_ocr_debug_key = None
        self._last_local_match_log_key = None
        # 地图标题切换期间不直接相信单帧 OCR。候选 MapID 必须连续命中
        # 两次后才提交，避免加载画面或小号字体的一次误识别覆盖当前地图。
        self._map_change_candidate_id: Optional[int] = None
        self._map_change_candidate_hits = 0
        self._last_expected_map_mismatch_key = None
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.custom_mobs_file = os.path.join(base_dir, "assets", "custom_map_mobs.json")
        self.local_maps_file = os.path.join(base_dir, "assets", "local_maps.json")

    def _get_ocr_engine(self):
        """Lazily create the isolated OCR backend for the map sentinel."""
        if self.ocr is not None:
            return self.ocr
        with self._ocr_lock:
            if self.ocr is None:
                self.ocr = IsolatedMapOCR(
                    status_callback=self.ocr_status_callback,
                )
        return self.ocr

    def close_ocr(self) -> None:
        engine = self.ocr
        if isinstance(engine, IsolatedMapOCR):
            engine.close()

    def set_ocr_roi(self, roi: Optional[Dict[str, int]]) -> None:
        """设置或更新 OCR 识别范围 (x, y, w, h)"""
        self.ocr_roi = roi

    def _extract_map_name_timed(
        self, frame: np.ndarray, *, high_precision: bool, pass_name: str,
    ) -> Optional[str]:
        callback = self.ocr_timing_callback
        if callback is None:
            return self.extract_map_name_from_frame(frame, high_precision=high_precision)

        def emit(phase: str, elapsed_ms=None, outcome=None) -> None:
            try:
                callback(phase, pass_name, elapsed_ms, outcome)
            except Exception:
                pass  # Timing diagnostics must not interrupt map recognition.

        emit("start")
        started_at = time.perf_counter()
        outcome = "error"
        try:
            name = self.extract_map_name_from_frame(
                frame, high_precision=high_precision
            )
            outcome = "text" if name else "empty"
            return name
        finally:
            emit("end", (time.perf_counter() - started_at) * 1000.0, outcome)

    def _log_ocr_result_if_changed(self, raw_lines: List[str], selected: str, tag: str = "") -> None:
        key = (tuple(raw_lines), str(selected), str(tag))
        if key == self._last_ocr_debug_key:
            return
        self._last_ocr_debug_key = key
        suffix = f"-{tag}" if tag else ""
        print(f"[OCR地图原文{suffix}] {raw_lines}")
        print(f"[OCR地图选用{suffix}] {selected!r}")

    def load_local_maps(self) -> List[Dict[str, Any]]:
        """读取本地地图表；地图名称和别名均由此文件提供。"""
        if not os.path.exists(self.local_maps_file):
            return []
        try:
            with open(self.local_maps_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            maps = payload.get("maps", []) if isinstance(payload, dict) else payload
            return [m for m in maps if isinstance(m, dict) and m.get("map_id") is not None]
        except Exception as e:
            print(f"[读取本地地图表失败] {e}")
            return []

    def update_local_map_id(self, map_name: str, map_id: int) -> bool:
        """将 UI 中确认的地图名/MapID 修改持久化到 local_maps.json。"""
        try:
            with open(self.local_maps_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            maps = payload.get("maps", []) if isinstance(payload, dict) else payload
            target = self.normalize_name(map_name)
            for entry in maps:
                names = [
                    entry.get("name_cn", ""),
                    entry.get("name_en", ""),
                ] + list(entry.get("ocr_aliases", []) or [])
                if any(self.normalize_name(str(n)) == target for n in names if n):
                    entry["map_id"] = int(map_id)
                    with open(self.local_maps_file, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    print(f"[MapResolver] 已更新本地地图映射：【{map_name}】 -> {map_id}")
                    return True
            # OCR 名称与既有标准名不完全一致时，保留用户确认的
            # 名称作为新的精确映射；不绑定任何具体地图。
            if isinstance(payload, dict):
                maps.append({
                    "map_id": int(map_id),
                    "name_cn": str(map_name).strip(),
                    "name_en": "",
                    "ocr_aliases": [],
                    "mob_ids": [],
                })
                payload["maps"] = maps
                with open(self.local_maps_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                print(f"[MapResolver] 已新增本地地图映射：【{map_name}】 -> {map_id}")
                return True
            return False
        except Exception as e:
            print(f"[MapResolver] 更新本地地图映射失败: {e}")
            return False

    def save_local_map_mobs(self, map_id: int, map_name: str, mobs: List[Dict]) -> bool:
        """将本图怪物编辑结果写回 local_maps.json 的 mob_ids。

        ``local_maps.json`` 是地图名称、MapID 与默认怪物的统一离线来源；
        因此这里仅保存去重后的 Mob ID，而不把 UI/下载状态等运行时字段
        混入地图表。若当前地图此前不在本地表中，则补建一条本地记录。
        """
        try:
            payload: Dict[str, Any] = {"version": 1, "maps": []}
            if os.path.exists(self.local_maps_file):
                with open(self.local_maps_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    payload = loaded
                elif isinstance(loaded, list):
                    payload = {"version": 1, "maps": loaded}

            maps = payload.setdefault("maps", [])
            if not isinstance(maps, list):
                raise ValueError("local_maps.json 的 maps 字段必须是列表")

            mob_ids: List[int] = []
            seen = set()
            for mob in mobs or []:
                raw_id = mob.get("id") if isinstance(mob, dict) else mob
                try:
                    mob_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if mob_id > 0 and mob_id not in seen:
                    seen.add(mob_id)
                    mob_ids.append(mob_id)

            entry = next(
                (item for item in maps
                 if isinstance(item, dict) and int(item.get("map_id", -1)) == int(map_id)),
                None,
            )
            if entry is None:
                entry = {
                    "map_id": int(map_id),
                    "name_cn": str(map_name).strip(),
                    "name_en": "",
                    "ocr_aliases": [],
                    "mob_ids": [],
                }
                maps.append(entry)
            else:
                # 当前地图的正式名称优先保留；空名称才用 UI 名称补齐。
                if not str(entry.get("name_cn", "")).strip() and map_name:
                    entry["name_cn"] = str(map_name).strip()

            entry["mob_ids"] = mob_ids
            # 只有通过“本图怪物”界面保存的列表才是用户明确覆盖；
            # 没有这个标记的旧列表可能来自错误的静态表，不能遮住 WZ IMG。
            entry["mob_ids_source"] = "manual"
            with open(self.local_maps_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(
                f"[MapResolver] 已更新本地地图怪物：【{entry.get('name_cn', map_name)}】"
                f"(ID: {map_id}) -> {mob_ids}"
            )
            return True
        except Exception as e:
            print(f"[MapResolver] 更新本地地图怪物失败: {e}")
            return False

    def load_custom_mobs_dict(self) -> Dict:
        """从本地 custom_map_mobs.json 读取所有用户自定义地图怪物配置"""
        if os.path.exists(self.custom_mobs_file):
            try:
                with open(self.custom_mobs_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[读取本地自定义怪物失败] {e}")
        return {}

    def save_local_map_disabled_mobs(
        self,
        map_id: int,
        disabled_ids: List[Any],
        map_name: Optional[str] = None,
    ) -> bool:
        """持久化当前地图被用户关闭的 Mob ID 到 local_maps.json。

        地图尚未存在本地表时也创建最小记录，避免勾选状态只能对
        已经手工录入 local_maps.json 的地图生效。
        """
        try:
            with open(self.local_maps_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            maps = payload.get("maps", []) if isinstance(payload, dict) else payload
            if not isinstance(maps, list):
                return False
            clean = []
            for raw in disabled_ids or []:
                try:
                    value = int(raw)
                    if value > 0 and value not in clean:
                        clean.append(value)
                except (TypeError, ValueError):
                    continue
            entry = next(
                (item for item in maps
                 if isinstance(item, dict) and int(item.get("map_id", -1)) == int(map_id)),
                None,
            )
            if entry is None:
                entry = {
                    "map_id": int(map_id),
                    "name_cn": str(map_name or "").strip(),
                    "name_en": "",
                    "ocr_aliases": [],
                    "mob_ids": [],
                }
                maps.append(entry)
            entry["disabled_mob_ids"] = clean
            with open(self.local_maps_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[MapResolver] 保存禁用怪物失败: {e}")
            return False

    def save_custom_map_mobs(self, map_id: int, map_name: str, mobs: List[Dict]) -> bool:
        """将用户手动修改的怪物配置持久化保存到本地 (下次启动优先直接读取，跳过网络请求)"""
        try:
            os.makedirs(os.path.dirname(self.custom_mobs_file), exist_ok=True)
            data = self.load_custom_mobs_dict()
            data[str(map_id)] = {
                "map_name": map_name,
                "mobs": mobs
            }
            with open(self.custom_mobs_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[MapResolver] 成功保存地图【{map_name}】(ID: {map_id}) 的自定义怪物配置到本地！")
            return True
        except Exception as e:
            print(f"[MapResolver] 保存本地自定义怪物失败: {e}")
            return False

    def extract_map_name_from_frame(
        self,
        frame: np.ndarray,
        high_precision: bool = False,
    ) -> Optional[str]:
        """
        高保真从小地图标题栏提取地图中文名 (扩展至 x=20~450, y=18~140，完整识别两行地图名，如: 金银岛 废都南方工地)
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        if self.ocr_roi and isinstance(self.ocr_roi, dict):
            rx = max(0, int(self.ocr_roi.get("x", 20)))
            ry = max(0, int(self.ocr_roi.get("y", 18)))
            rw = max(10, int(self.ocr_roi.get("w", 260)))
            rh = max(10, int(self.ocr_roi.get("h", 55)))
            crop_x1, crop_x2 = rx, min(w, rx + rw)
            crop_y1, crop_y2 = ry, min(h, ry + rh)
        else:
            crop_y1, crop_y2 = 18, min(75, h)
            crop_x1, crop_x2 = 20, min(300, w)

        title_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if title_crop.size == 0:
            return None

        try:
            known_names = set()
            for entry in self.load_local_maps():
                for field in ("name_cn", "name_en"):
                    name = str(entry.get(field, "")).strip()
                    if name:
                        known_names.add(self.normalize_name(name))
                for alias in entry.get("ocr_aliases", []) or []:
                    if alias:
                        known_names.add(self.normalize_name(str(alias)))

            def parse_ocr_lines(raw_results, coordinate_scale: float = 1.0) -> List[str]:
                """清理 RapidOCR 结果并按小地图标题的纵向行序重新拼接。"""
                valid_items = []
                for result in raw_results or []:
                    if not result or len(result) < 2:
                        continue
                    box, text = result[0], str(result[1]).strip()
                    score = float(result[2]) if len(result) > 2 else 1.0
                    clean = re.sub(r"[^\w\u4e00-\u9fa5IVX1-9lL<>]", "", text)
                    # RapidOCR 对地图名称中的小号罗马数字竖线经常识别成
                    # 小写 l。例如“黑森林打猎场Ⅱ”在 3x 图上输出 Il。
                    # 只修正“含中文的地图名末尾”，避免改坏英文单词正文。
                    if re.search(r"[\u4e00-\u9fa5]", clean):
                        clean = re.sub(
                            r"[lL]+$",
                            lambda match: "I" * len(match.group(0)),
                            clean,
                        )
                    # 过滤全局 UI 按钮等干扰词
                    if clean in ["小地图", "大地图", "NPC", "世界地图"]:
                        continue
                    if len(clean) >= 2 and box and len(box) >= 4:
                        y_center = (box[0][1] + box[2][1]) / 2.0
                        x_left = min(box[0][0], box[3][0])
                        valid_items.append({
                            "clean": clean,
                            "y_center": y_center,
                            "x_left": x_left,
                            "score": score,
                        })

                if not valid_items:
                    return []

                # 放大精识别时坐标也同比放大；行聚类阈值必须同步缩放。
                line_tolerance = 8.0 * max(1.0, float(coordinate_scale))
                sorted_items = sorted(valid_items, key=lambda item: item["y_center"])
                line_groups = []
                for item in sorted_items:
                    placed = False
                    for group in line_groups:
                        if abs(item["y_center"] - group["y_center"]) <= line_tolerance:
                            group["items"].append(item)
                            group["y_center"] = sum(
                                entry["y_center"] for entry in group["items"]
                            ) / len(group["items"])
                            placed = True
                            break
                    if not placed:
                        line_groups.append({"y_center": item["y_center"], "items": [item]})

                line_groups.sort(key=lambda group: group["y_center"])
                lines = []
                for group in line_groups:
                    group["items"].sort(key=lambda item: item["x_left"])
                    line_text = "".join(item["clean"] for item in group["items"])
                    if line_text:
                        lines.append(line_text)
                return lines

            def detect_trailing_roman_strokes(raw_results, source_image) -> int:
                """从最底层地图名右端数细竖条，区分像素字体 I/II/III。"""
                if source_image is None or source_image.size == 0:
                    return 0
                line_candidates = []
                for result in raw_results or []:
                    if not result or len(result) < 2 or not result[0]:
                        continue
                    text = str(result[1])
                    if not re.search(r"[\u4e00-\u9fa5]", text):
                        continue
                    box = result[0]
                    if len(box) < 4:
                        continue
                    xs = [float(point[0]) for point in box]
                    ys = [float(point[1]) for point in box]
                    line_candidates.append((sum(ys) / len(ys), min(xs), max(xs), min(ys), max(ys)))
                if not line_candidates:
                    return 0

                _, x1, x2, y1, y2 = max(line_candidates, key=lambda item: item[0])
                image_h, image_w = source_image.shape[:2]
                x1 = max(0, int(math.floor(x1)))
                x2 = min(image_w, int(math.ceil(x2)) + 1)
                y1 = max(0, int(math.floor(y1)))
                y2 = min(image_h, int(math.ceil(y2)) + 1)
                line_h = max(1, y2 - y1)
                # 只看 OCR 行框最右侧约 1.6 个字符，排除前面的中文结构。
                tail_w = max(12, int(round(line_h * 1.6)))
                tail_x1 = max(x1, x2 - tail_w)
                tail = source_image[y1:y2, tail_x1:x2]
                if tail.size == 0:
                    return 0
                if tail.ndim == 2:
                    white_mask = tail >= 220
                else:
                    bgr = tail[:, :, :3]
                    # 小地图字芯是纯白；使用三通道共同阈值可排除蓝灰背景。
                    white_mask = np.min(bgr, axis=2) >= 220
                count, _, stats, _ = cv2.connectedComponentsWithStats(
                    white_mask.astype(np.uint8), 8
                )
                strokes = []
                max_stroke_w = max(2, int(round(line_h * 0.22)))
                min_stroke_h = max(5, int(round(line_h * 0.35)))
                max_stroke_h = max(min_stroke_h, int(round(line_h * 0.75)))
                for sx, sy, sw, sh, area in stats[1:count]:
                    sx, sy, sw, sh, area = map(int, (sx, sy, sw, sh, area))
                    if (
                        1 <= sw <= max_stroke_w
                        and min_stroke_h <= sh <= max_stroke_h
                        and sh >= sw * 3
                        and area >= sw * min_stroke_h
                    ):
                        strokes.append((sx, sy, sw, sh))
                if not strokes:
                    return 0

                strokes.sort(key=lambda item: item[0])
                rightmost = strokes[-1]
                # 罗马数字必须紧贴 OCR 行框右端；否则可能是汉字内部竖画。
                if (tail.shape[1] - (rightmost[0] + rightmost[2])) > max(4, int(line_h * 0.25)):
                    return 0
                group = [rightmost]
                max_gap = max(2, int(round(line_h * 0.20)))
                for stroke in reversed(strokes[:-1]):
                    next_stroke = group[-1]
                    gap = next_stroke[0] - (stroke[0] + stroke[2])
                    if (
                        0 <= gap <= max_gap
                        and abs(stroke[1] - rightmost[1]) <= 2
                        and abs(stroke[3] - rightmost[3]) <= 2
                    ):
                        group.append(stroke)
                    else:
                        break
                return min(8, len(group))

            results, _ = self._get_ocr_engine()(title_crop)
            detected_lines = parse_ocr_lines(results)
            trailing_roman_strokes = detect_trailing_roman_strokes(results, title_crop)
            used_high_precision = False

            if not detected_lines:
                return None

            # 2. 小地图标题层级语义判定：
            # 若识别出两行或多行：
            #   - 顶行通常为大区域/街道名 streetName（如“迷宫”、“金银岛”、“魔法密林”、“废弃都市”）
            #   - 底行永远为当前所在具体子地图名 mapName（如“黑森林打猎场”、“北方工地第1地区”）
            #   - 绝不能将顶行大区域/城镇作为当前地图返回，否则会导致误判城镇安全区并清空野怪模板！
            # 若仅有一行：该行直接作为 mapName。
            if len(detected_lines) >= 2:
                street_cand = detected_lines[0]
                map_cand = detected_lines[-1]
            else:
                street_cand = None
                map_cand = detected_lines[0]
                # 无 miniMap 的过渡图可能把“当前地图名／区域名／按钮”
                # 横排在同一标题栏。逐行拼接会得到
                # “地铁售票处废弃都市地铁...”，虽然首个 OCR 文字块
                # 已准确识别为“地铁售票处”，最终却无法匹配本地地图表。
                # 仅在整行本身不是已知地图时，取唯一最长的已知
                # 文字块。右侧区域名偶尔也会被 OCR 单独切成已知
                # 地图名；长度相同则视为歧义，不贸然切图。
                # 普通双行标题与数字地图的复核逻辑不变。
                if self.normalize_name(map_cand) not in known_names:
                    exact_blocks = []
                    for item in results or []:
                        if not item or len(item) < 2 or not item[0]:
                            continue
                        clean = re.sub(
                            r"[^\w\u4e00-\u9fa5IVX1-9lL<>]", "", str(item[1]).strip()
                        )
                        if clean in ("小地图", "大地图", "NPC", "世界地图"):
                            continue
                        if self.normalize_name(clean) in known_names:
                            exact_blocks.append(clean)
                    if exact_blocks:
                        unique_blocks = {
                            self.normalize_name(block): block for block in exact_blocks
                        }
                        longest = max(len(name) for name in unique_blocks)
                        winners = [
                            block for name, block in unique_blocks.items()
                            if len(name) == longest
                        ]
                        if len(winners) == 1:
                            map_cand = winners[0]
                # 冷启动时 RapidOCR 偶尔把售票处标题与右侧区域名
                # 合成一个文字块，以上逐块规则便无从选出地图名。
                # 只有整行以该地图名开头、却仍未精确命中时，才把
                # 标题栏左侧 80px 单独 OCR 复核；不能仅凭前缀切图。
                booth_name = "地铁售票处"
                if (
                    self.normalize_name(map_cand) not in known_names
                    and self.normalize_name(booth_name) in known_names
                    and self.normalize_name(map_cand).startswith(
                        self.normalize_name(booth_name)
                    )
                ):
                    isolated = title_crop[:min(25, title_crop.shape[0]), :min(80, title_crop.shape[1])]
                    if isolated.shape[0] >= 12 and isolated.shape[1] >= 70:
                        isolated_results, _ = self._get_ocr_engine()(isolated)
                        exact_left = [
                            item for item in (isolated_results or [])
                            if len(item) >= 3
                            and float(item[2]) >= 0.90
                            and self.normalize_name(str(item[1]))
                            == self.normalize_name(booth_name)
                        ]
                        if len(exact_left) == 1:
                            map_cand = booth_name

            # 原尺寸 OCR 容易完全漏掉 16px 高字体末尾的“Ⅱ”，并把
            # “黑森林打猎场Ⅱ”退化成“黑森林打猎场”，随后错误默认到 I。
            # 只有当无后缀结果可对应两个以上已知数字地图时，才额外做
            # 一次 3x CUBIC 精识别；普通地图仍保持原来的一次 OCR 成本。
            norm_map = self.normalize_name(map_cand)
            numbered_base = norm_map
            numbered_suffix_match = re.match(r"^(.*?)(\d+)$", norm_map)
            if numbered_suffix_match:
                numbered_base = numbered_suffix_match.group(1)
            numbered_matches = {
                known for known in known_names
                if known.startswith(numbered_base)
                and known[len(numbered_base):].isdigit()
            }
            # 不仅复核“后缀完全缺失”，也复核已经被判成 I 的情况。
            # 实机上同一个Ⅱ字形会随 ROI 背景宽度不同在“缺失 / I / Il”
            # 之间跳变；若 I 已直接命中正式名称，旧条件会过早相信它。
            needs_numbered_review = (
                len(numbered_matches) >= 2
                and (norm_map == numbered_base or norm_map in numbered_matches)
            )
            if needs_numbered_review:
                visual_norm = numbered_base + str(trailing_roman_strokes)
                if trailing_roman_strokes > 0 and visual_norm in numbered_matches:
                    display_base = re.sub(
                        r"(?:[IVX]+|[ⅠⅡⅢⅣⅤⅥⅦⅧ]+|\d+)$", "", map_cand
                    )
                    map_cand = display_base + ("I" * trailing_roman_strokes)
                    detected_lines[-1] = map_cand
                else:
                    enhanced_scale = 3.0
                    enhanced_crop = cv2.resize(
                        title_crop,
                        None,
                        fx=enhanced_scale,
                        fy=enhanced_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    enhanced_results, _ = self._get_ocr_engine()(enhanced_crop)
                    enhanced_lines = parse_ocr_lines(enhanced_results, enhanced_scale)
                    if enhanced_lines:
                        enhanced_map = enhanced_lines[-1]
                        enhanced_norm = self.normalize_name(enhanced_map)
                        if enhanced_norm in numbered_matches:
                            detected_lines = enhanced_lines
                            map_cand = enhanced_map
                            street_cand = enhanced_lines[0] if len(enhanced_lines) >= 2 else None
                            used_high_precision = True

            # 切图后的首轮确认对所有地图名都做一次 3x 精识别。平时仍只跑
            # 原尺寸 OCR，避免把稳定地图的周期检测成本翻倍。只有原尺寸结果
            # 没有命中本地正式名称/别名、而放大结果能够精确命中时才替换，
            # 因此不会让一轮较差的放大识别覆盖本来正确的结果。
            if high_precision and not needs_numbered_review:
                base_norm = self.normalize_name(map_cand)
                base_exact = base_norm in known_names or (base_norm + "1") in known_names
                if not base_exact and street_cand:
                    combined_norm = self.normalize_name(street_cand + map_cand)
                    base_exact = (
                        combined_norm in known_names
                        or (combined_norm + "1") in known_names
                    )

                enhanced_scale = 3.0
                enhanced_crop = cv2.resize(
                    title_crop,
                    None,
                    fx=enhanced_scale,
                    fy=enhanced_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                enhanced_results, _ = self._get_ocr_engine()(enhanced_crop)
                enhanced_lines = parse_ocr_lines(enhanced_results, enhanced_scale)
                if enhanced_lines:
                    enhanced_map = enhanced_lines[-1]
                    enhanced_street = (
                        enhanced_lines[0] if len(enhanced_lines) >= 2 else None
                    )
                    enhanced_norm = self.normalize_name(enhanced_map)
                    enhanced_exact = (
                        enhanced_norm in known_names
                        or (enhanced_norm + "1") in known_names
                    )
                    if not enhanced_exact and enhanced_street:
                        enhanced_combined = self.normalize_name(
                            enhanced_street + enhanced_map
                        )
                        enhanced_exact = (
                            enhanced_combined in known_names
                            or (enhanced_combined + "1") in known_names
                        )
                    # 原图已经精确命中时保留原图结果；3x 仍真实执行，作为
                    # 切图复核，但不能用另一条碰巧合法的结果反向污染它。
                    if enhanced_exact and not base_exact:
                        detected_lines = enhanced_lines
                        map_cand = enhanced_map
                        street_cand = enhanced_street
                        used_high_precision = True

            # 3. 针对子地图名生成候选列表并评估优先级：
            # 优先级 1: map_cand 直接命中已知数据库
            # 优先级 2: map_cand 缺省罗马数字 I / 1 后命中（如“黑森林打猎场”+“I” -> “黑森林打猎场I”）
            # 优先级 3: 联合 street_cand + map_cand 命中（如“废弃都市”+“南方工地” -> “废都南方工地”）
            # 优先级 4: 后备直接返回 map_cand（绝不回退到 street_cand）
            selected = None
            norm_map = self.normalize_name(map_cand)

            if norm_map in known_names:
                selected = map_cand
            elif (norm_map + "1") in known_names:
                selected = map_cand + "I"
            elif street_cand:
                comb1 = street_cand + map_cand
                if self.normalize_name(comb1) in known_names:
                    selected = comb1
                elif (self.normalize_name(comb1) + "1") in known_names:
                    selected = comb1 + "I"

            if not selected:
                selected = map_cand

            self._log_ocr_result_if_changed(
                detected_lines,
                selected,
                "切图3x" if used_high_precision else "1",
            )
            return selected
        except Exception as e:
            print(f"[OCR 提取异常] {e}")

        return None


    @staticmethod
    def normalize_name(name: str) -> str:
        """
        全场景地图名称标准化：
        1. 过滤所有尖括号、圆括号、空格、标点符号 (如 < > ( ) [ ] 、 , . - _ 空格)；
        2. 将所有罗马数字 (I/II/III/IV) 与中文数字 (一/二/三/四/五/六/七/八/九/十) 统一映射为标准阿拉伯数字；
        3. 彻底解决《地铁一号线<第1地区>》因符号与汉字数字导致的字典匹配漂移。
        """
        if not name:
            return ""
        s = re.sub(r"[^\w\u4e00-\u9fa5]", "", str(name))
        # 罗马数字归一化
        s = s.replace("VIII", "8").replace("VII", "7").replace("VI", "6").replace("IV", "4").replace("V", "5")
        s = s.replace("III", "3").replace("II", "2").replace("I", "1")
        # 中文大写/小写数字归一化
        num_map = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
        }
        for k, v in num_map.items():
            s = s.replace(k, v)
        # OCR 有时会输出 Unicode 罗马数字（Ⅰ/Ⅱ/Ⅲ），而不是 ASCII 的 I/II/III。
        # 统一后，“沼泽地Ⅱ”可以直接命中正式名称，不会退化成短别名“沼泽地”。
        unicode_roman_map = {
            "Ⅷ": "8", "Ⅶ": "7", "Ⅵ": "6", "Ⅴ": "5",
            "Ⅳ": "4", "Ⅲ": "3", "Ⅱ": "2", "Ⅰ": "1",
        }
        for k, v in unicode_roman_map.items():
            s = s.replace(k, v)
        return s.lower()

    def resolve_map_id_by_name(self, map_name: str) -> Optional[Dict]:
        """
        将 OCR 地图名匹配至 local_maps.json (支持中英文、多级符号与数字归一化匹配)
        """
        if not map_name:
            return None

        target_norm = self.normalize_name(map_name)
        if not target_norm:
            return None

        local_maps = self.load_local_maps()

        def build_local_info(entry: Dict[str, Any]) -> Dict[str, Any]:
            canonical = str(entry.get("name_cn", entry.get("map_name", ""))).strip()
            english = str(entry.get("name_en", "")).strip()
            return {
                "map_id": int(entry["map_id"]),
                "street": entry.get("street", "Local"),
                "name": english or canonical,
                "chinese_name": canonical or map_name,
                "local_mob_ids": [int(x) for x in (entry.get("mob_ids", []) or [])],
                "mob_ids_source": str(entry.get("mob_ids_source", "")).strip().lower(),
                "disabled_mob_ids": [int(x) for x in (entry.get("disabled_mob_ids", []) or [])],
                "local_exact_match": True,
            }

        # 本地表的正式名称优先于别名。这样即使某个短别名与其它地图的
        # 名称前缀相同，也不会抢走 OCR 已经识别出的完整正式名称。
        canonical_matches = []
        english_match_ids = set()
        for entry in local_maps:
            canonical = str(entry.get("name_cn", entry.get("map_name", ""))).strip()
            english = str(entry.get("name_en", "")).strip()
            cn_match = bool(canonical and (
                map_name.strip() == canonical
                or self.normalize_name(canonical) == target_norm
            ))
            en_match = bool(english and self.normalize_name(english) == target_norm)
            if (canonical or english) and (
                cn_match or en_match
            ):
                canonical_matches.append(entry)
                if en_match:
                    english_match_ids.add(int(entry["map_id"]))

        if canonical_matches:
            matched_ids = sorted({int(entry["map_id"]) for entry in canonical_matches})
            if len(matched_ids) > 1:
                # WZ 中可能存在与普通地图完全同名的活动/GM/镜像地图。
                # OCR 别名是用户对具体 MapID 的显式绑定；正式名称重名时，
                # 允许与本次 OCR 原文完全相同的别名先完成消歧。
                explicit_alias_matches = []
                for entry in canonical_matches:
                    aliases = [
                        str(alias).strip()
                        for alias in (entry.get("ocr_aliases", []) or [])
                        if str(alias).strip()
                    ]
                    if any(
                        map_name.strip() == alias
                        or self.normalize_name(alias) == target_norm
                        for alias in aliases
                    ):
                        explicit_alias_matches.append(entry)
                explicit_ids = {
                    int(entry["map_id"]) for entry in explicit_alias_matches
                }
                if len(explicit_ids) == 1:
                    info = build_local_info(explicit_alias_matches[0])
                    local_key = (map_name, info["map_id"])
                    if local_key != self._last_local_match_log_key:
                        self._last_local_match_log_key = local_key
                        print(
                            "[MapResolver] 同名正式地图由显式 OCR 别名消歧命中："
                            f"{map_name!r} -> {info['map_id']}"
                        )
                    return info
                kind = "英文地图名" if english_match_ids else "地图正式名称"
                ambiguous_key = (map_name, tuple(matched_ids))
                if ambiguous_key != getattr(self, "_last_ambiguous_local_match_log_key", None):
                    self._last_ambiguous_local_match_log_key = ambiguous_key
                    print(
                        f"[MapResolver] 本地{kind}存在歧义：{map_name!r}，"
                        f"候选 MapID={matched_ids}，跳过匹配"
                    )
                return None
            info = build_local_info(canonical_matches[0])
            # 使用 repr 避免 OCR 带零宽字符/特殊符号时触发 Windows
            # 控制台 GBK 编码异常，导致本地命中流程被中断。
            local_key = (map_name, info["map_id"])
            if local_key != self._last_local_match_log_key:
                self._last_local_match_log_key = local_key
                print(f"[MapResolver] 本地地图表正式名称命中：{map_name!r} -> {info['map_id']}")
            return info

        # 正式名称没有命中时才尝试 OCR 别名。别名是用户对具体地图的
        # 显式绑定关系；例如只有“沼泽地Ⅰ”配置了“沼泽地”，识别到
        # “沼泽地”就必须命中“沼泽地Ⅰ”。只有多个地图同时配置同一个
        # 别名时，才因为无法确定具体地图而判为歧义。
        alias_matches = []
        for entry in local_maps:
            aliases = [str(a).strip() for a in (entry.get("ocr_aliases", []) or [])]
            if any(
                map_name.strip() == alias
                or self.normalize_name(alias) == target_norm
                for alias in aliases if alias
            ):
                alias_matches.append(entry)

        if alias_matches:
            if len(alias_matches) > 1:
                ambiguous_key = (map_name, tuple(sorted(int(e["map_id"]) for e in alias_matches)))
                if ambiguous_key != getattr(self, "_last_ambiguous_local_match_log_key", None):
                    self._last_ambiguous_local_match_log_key = ambiguous_key
                    ids = [int(e["map_id"]) for e in alias_matches]
                    print(f"[MapResolver] OCR 地图名存在歧义：{map_name!r}，候选 MapID={ids}，跳过别名匹配")
                return None

            info = build_local_info(alias_matches[0])
            local_key = (map_name, info["map_id"])
            if local_key != self._last_local_match_log_key:
                self._last_local_match_log_key = local_key
                print(f"[MapResolver] 本地地图表 OCR 别名命中：{map_name!r} -> {info['map_id']}")
            return info

        # 没有任何显式别名时，才对多个正式名称的共同前缀做歧义保护。
        # 因此“沼泽地Ⅰ”配置了“沼泽地”后，会在上面的别名分支直接命中，
        # 不会被这里的“沼泽地Ⅰ/Ⅱ”共同前缀规则拦截。
        prefix_candidates = []
        for entry in local_maps:
            canonical = str(entry.get("name_cn", entry.get("map_name", ""))).strip()
            canonical_norm = self.normalize_name(canonical)
            english = str(entry.get("name_en", "")).strip()
            english_norm = self.normalize_name(english)
            if (
                (canonical_norm and canonical_norm != target_norm and canonical_norm.startswith(target_norm))
                or (english_norm and english_norm != target_norm and english_norm.startswith(target_norm))
            ):
                prefix_candidates.append(entry)
        if len(prefix_candidates) > 1:
            # 地图序列缺省保护：当 OCR 丢失尾部细线罗马数字 I 时（如“黑森林打猎场”），
            # 若前缀候选列表中恰好有且仅有一张对应的 1 号地图（如“黑森林打猎场1”），
            # 则判定为第 1 号地图缺省命中，直达该地图并避免判为歧义。
            base_i_matches = [
                e for e in prefix_candidates
                if self.normalize_name(str(e.get("name_cn", e.get("map_name", "")))) == target_norm + "1"
            ]
            if len(base_i_matches) == 1:
                info = build_local_info(base_i_matches[0])
                local_key = (map_name, info["map_id"])
                if local_key != self._last_local_match_log_key:
                    self._last_local_match_log_key = local_key
                    print(f"[MapResolver] 本地地图表前缀第1地区缺省命中：{map_name!r} -> {info['map_id']} ({info['chinese_name']})")
                return info

            ambiguous_key = (map_name, tuple(sorted(int(e["map_id"]) for e in prefix_candidates)))
            if ambiguous_key != getattr(self, "_last_ambiguous_local_match_log_key", None):
                self._last_ambiguous_local_match_log_key = ambiguous_key
                ids = [int(e["map_id"]) for e in prefix_candidates]
                print(f"[MapResolver] OCR 地图名存在歧义：{map_name!r}，候选 MapID={ids}，跳过前缀匹配")
            return None
        elif len(prefix_candidates) == 1:
            info = build_local_info(prefix_candidates[0])
            local_key = (map_name, info["map_id"])
            if local_key != self._last_local_match_log_key:
                self._last_local_match_log_key = local_key
                print(f"[MapResolver] 本地地图表前缀唯一命中：{map_name!r} -> {info['map_id']} ({info['chinese_name']})")
            return info

        # OCR 地图名只能来自 local_maps.json；没有本地正式名称/别名时
        # 不猜测 MapID，也不再使用内置经典地图表或模糊匹配。
        return None

    def auto_detect_and_sync_map(
        self,
        frame: Optional[np.ndarray],
        manual_map_name: Optional[str] = None,
        manual_map_id: Optional[int] = None,
        force_sync: bool = False,
        high_precision_ocr: bool = False,
        confirm_map_change: bool = False,
        expected_map_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        全自动小地图探测与地图数据同步 (支持【手动输入 MapID】、【下拉选图】与【OCR 自动探测】)
        """
        map_info = None
        raw_map_data = None
        map_data_source = None
        if manual_map_id is not None:
            # 1. 优先使用本地地图表，避免手动应用后又被联网数据覆盖。
            for entry in self.load_local_maps():
                if int(entry.get("map_id", -1)) == int(manual_map_id):
                    map_info = {
                        "map_id": int(manual_map_id),
                        "street": entry.get("street", "Local"),
                        "name": entry.get("name_en", entry.get("name_cn", f"Map_{manual_map_id}")),
                        "chinese_name": entry.get("name_cn", f"地图 {manual_map_id}"),
                        "local_mob_ids": [int(x) for x in (entry.get("mob_ids", []) or [])],
                        "mob_ids_source": str(entry.get("mob_ids_source", "")).strip().lower(),
                        "disabled_mob_ids": [int(x) for x in (entry.get("disabled_mob_ids", []) or [])],
                        "local_exact_match": True,
                    }
                    print(f"[MapResolver] 本地地图表按 MapID 命中：{manual_map_id}")
                    break
            if not map_info:
                # IMG 本身不含 String.wz 的地图显示名，但只要结构数据存在，
                # 就不为名称阻塞或访问网络。显示名可继续由本地地图表补齐。
                if self.map_data_reader is not None:
                    try:
                        raw_map_data = self.map_data_reader.load_map(int(manual_map_id))
                        map_data_source = "wz_img"
                    except Exception as exc:
                        print(f"[MapResolver] 本地地图 IMG 不可用，将尝试后备来源：{exc}")
                net_info = None
                if raw_map_data is None:
                    # 本地 IMG 缺失时保留在线接口兼容后备。
                    net_info = self.downloader.get_map_details(manual_map_id)
                if net_info:
                    map_info = {
                        "map_id": manual_map_id,
                        "name": net_info.get("name", f"Map_{manual_map_id}"),
                        "street": net_info.get("streetName", "Maple World"),
                        "chinese_name": net_info.get("name", f"Map_{manual_map_id}")
                    }
                else:
                    map_info = {
                        "map_id": manual_map_id,
                        "name": f"Map_{manual_map_id}",
                        "street": "Custom",
                        "chinese_name": f"地图 {manual_map_id}"
                    }
            target_name = map_info["chinese_name"]
        elif manual_map_name:
            target_name = manual_map_name
            map_info = self.resolve_map_id_by_name(target_name)
        else:
            if frame is None:
                return None
            detected_name = self._extract_map_name_timed(
                frame,
                high_precision=high_precision_ocr,
                pass_name="precision3x" if high_precision_ocr else "normal",
            )
            if not detected_name:
                self.last_unmatched_ocr_name = None
                if confirm_map_change:
                    self._map_change_candidate_id = None
                    self._map_change_candidate_hits = 0
                return None
            target_name = detected_name
            map_info = self.resolve_map_id_by_name(target_name)
            if map_info is None and not high_precision_ocr:
                print(f"[MapResolver] 首轮OCR未匹配：{target_name!r}，立即尝试3x精识别")
                refined_name = self._extract_map_name_timed(
                    frame, high_precision=True, pass_name="fallback3x"
                )
                if refined_name:
                    target_name = refined_name
                    map_info = self.resolve_map_id_by_name(target_name)

        if not map_info:
            if manual_map_id is None and manual_map_name is None:
                self.last_unmatched_ocr_name = target_name
            if confirm_map_change:
                self._map_change_candidate_id = None
                self._map_change_candidate_hits = 0
            return None
        self.last_unmatched_ocr_name = None

        map_id = int(map_info["map_id"])
        chinese_name = map_info.get("chinese_name", target_name)

        # 已知传送门带有明确目标 MapID。加载画面中的旧标题或偶发误识别
        # 即使能命中另一张合法地图，也不能结束本次传送门切图确认。
        if expected_map_id is not None and map_id != int(expected_map_id):
            mismatch_key = (map_id, int(expected_map_id))
            if mismatch_key != self._last_expected_map_mismatch_key:
                self._last_expected_map_mismatch_key = mismatch_key
                print(
                    f"[MapResolver] 切图候选与传送门目标不符："
                    f"识别={map_id}，期望={int(expected_map_id)}，继续复核"
                )
            self._map_change_candidate_id = None
            self._map_change_candidate_hits = 0
            return None

        is_automatic = manual_map_id is None and manual_map_name is None
        is_real_change = self.current_map_id is not None and map_id != self.current_map_id
        if confirm_map_change and is_automatic and is_real_change:
            # 跨地图循环按下普通门前已经从 WZ 得到唯一目标 MapID。
            # OCR 首帧若精确命中该目标，前面的 expected_map_id 检查已经
            # 排除了其它合法地图，无需再执行一轮昂贵的 3x OCR。死亡
            # 回城、玩家手动换图等未知目标场景仍保留连续两帧确认。
            exact_expected_target = (
                expected_map_id is not None and map_id == int(expected_map_id)
            )
            if exact_expected_target:
                self._map_change_candidate_id = None
                self._map_change_candidate_hits = 0
                print(
                    f"[MapResolver] 传送门目标首帧确认："
                    f"{chinese_name}({map_id})"
                )
            else:
                if self._map_change_candidate_id == map_id:
                    self._map_change_candidate_hits += 1
                else:
                    self._map_change_candidate_id = map_id
                    self._map_change_candidate_hits = 1
                if self._map_change_candidate_hits < 2:
                    print(
                        f"[MapResolver] 切图候选首帧：{chinese_name}({map_id})，"
                        "等待下一帧确认"
                    )
                    return None
        else:
            self._map_change_candidate_id = None
            self._map_change_candidate_hits = 0

        self._last_expected_map_mismatch_key = None

        if raw_map_data is None and self.map_data_reader is not None:
            try:
                raw_map_data = self.map_data_reader.load_map(int(map_id))
                map_data_source = "wz_img"
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"[MapResolver] 本地地图 IMG 解析失败，将使用 JSON/API 后备：{exc}")

        if raw_map_data is not None:
            # 地图名称存放在 String.wz 而非 Map.img；沿用已识别的本地名称
            # 仅补显示元数据，不改变 IMG 中的物理、传送门或怪物数据。
            raw_map_data = dict(raw_map_data)
            raw_map_data["name"] = map_info.get("name") or chinese_name
            raw_map_data["streetName"] = map_info.get("street") or "WZ"

        # 自动探测模式下且未强制刷新：若 MapID 未改变，直接判定为同一张地图并跳过，绝不重复加载！
        if not force_sync and manual_map_id is None and manual_map_name is None:
            if self.current_map_id == map_id:
                return None

        print(f"[MapResolver] 自动识别地图: 【{chinese_name}】 -> GMS v83 MapID: {map_id} ({map_info.get('name')})")

        # ================= 优先层 1: 读取本地用户自定义配置 (完全跳过网络请求) =================
        custom_data = self.load_custom_mobs_dict()
        # 本地地图表精确命中时，以该表中的 mob_ids 为准；旧版
        # custom_map_mobs.json 仅作为未命中地图的兼容回退。
        if str(map_id) in custom_data and not map_info.get("local_exact_match", False):
            local_entry = custom_data[str(map_id)]
            local_mobs = local_entry.get("mobs", [])
            if local_mobs:
                print(f"[MapResolver] 命中本地自定义怪物配置【{chinese_name}】，直接从本地载入 {len(local_mobs)} 种怪物 (已跳过网络请求)！")
                for m in local_mobs:
                    m["ready"] = True
                result = {
                    "chinese_name": chinese_name,
                    "map_id": map_id,
                    "street": map_info.get("street"),
                    "gms_name": map_info.get("name"),
                    "mobs": local_mobs,
                    # 禁用项属于地图级设置，怪物来源切换到旧自定义表时
                    # 也必须随识别结果返回，否则 GUI 会重新默认全选。
                    "disabled_mob_ids": [
                        int(x) for x in (map_info.get("disabled_mob_ids", []) or [])
                    ],
                    "raw_map_data": raw_map_data,
                    "map_data_source": map_data_source,
                    "is_custom": True
                }
                self.current_map_mobs = local_mobs
                self.current_map_name = chinese_name
                self.current_map_id = map_id
                self.current_map_info = map_info
                return result

        # ================= 优先层 2: 以 WZ IMG 的 life 为实际刷怪来源 =================
        # local_maps.json 只负责地图名称、别名和 MapID；其中的 mob_ids
        # 仅代表用户通过界面明确保存的覆盖列表，不再作为预置怪物来源。
        # 自动怪物清单只从当前地图 WZ 的 life 节点取得。
        configured_mobs = [int(x) for x in (map_info.get("local_mob_ids", []) or [])]
        wz_mobs = []
        if raw_map_data:
            wz_mobs = [
                int(item.get("id")) for item in (raw_map_data.get("mobs", []) or [])
                if isinstance(item, dict) and item.get("id") is not None
            ]
            # Reader 已按 Mob ID 去重；这里再次去重以兼容其它地图数据读取器。
            wz_mobs = list(dict.fromkeys(wz_mobs))

        if wz_mobs and map_info.get("mob_ids_source") != "manual":
            selected_mob_ids = wz_mobs
            if configured_mobs and configured_mobs != wz_mobs:
                print(
                    f"[MapResolver] WZ 怪物清单覆盖旧预置：MapID={map_id} "
                    f"旧={configured_mobs}，WZ={wz_mobs}"
                )
        elif map_info.get("mob_ids_source") == "manual":
            selected_mob_ids = configured_mobs
        else:
            selected_mob_ids = wz_mobs
        mobs = []
        if selected_mob_ids:
            for mid in selected_mob_ids:
                paths = self.downloader.download_and_save_mob_template(mid)
                m_alias = "Lupin" if mid == 3210101 else ("Zombie Lupin" if mid == 3210100 else ("Evil Eye" if mid == 2230100 else f"Mob_{mid}"))
                mobs.append({
                    "id": mid,
                    "name": m_alias,
                    "file_prefix": f"mob_{mid}",
                    "templates": paths,
                    "ready": True
                })
        # WZ IMG 不存在或 life 节点没有怪物时保持空列表；
        # 不再调用在线接口自动猜测或补充怪物。
        if not selected_mob_ids:
            print(f"[MapResolver] MapID={map_id} 未找到怪物信息，不加载或显示怪物模板")

        result = {
            "chinese_name": chinese_name,
            "map_id": map_id,
            "street": map_info.get("street"),
            "gms_name": map_info.get("name"),
            "mobs": mobs,
            # resolve_map_id_by_name / 手动 MapID 已从 local_maps.json
            # 读到该字段；必须透传给 UI，才能让怪物列表恢复默认禁用项。
            "disabled_mob_ids": [
                int(x) for x in (map_info.get("disabled_mob_ids", []) or [])
            ],
            "raw_map_data": raw_map_data,
            "map_data_source": map_data_source,
            "is_custom": False
        }
        # 同步结果完整构造后才提交；中途异常不会把后续 OCR 错当成同图。
        self.current_map_mobs = mobs
        self.current_map_name = chinese_name
        self.current_map_id = map_id
        self.current_map_info = map_info
        return result
