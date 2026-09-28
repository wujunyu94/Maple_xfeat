"""
asset_downloader.py - 在线游戏资产下载与模板生成器 (GMS v83 专版)
通过在线后备接口按怪物名称或 ID 检索并下载纯透明 PNG 精灵图，
自动生成多尺度识别所需的全身/特征模板并保存在 templates/<MobID>/ 目录中。
"""

import os
import sys
import io
import time
import glob
import re
import zipfile
import requests
from PIL import Image
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple, Set

DEFAULT_REGION = "gms"
DEFAULT_VERSION = "83"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _public_error(exc: Exception) -> str:
    """Keep vendor names from network exceptions out of the console output."""
    text = re.sub(r"maplestory\.io", "在线资源接口", str(exc), flags=re.I)
    return re.sub("maplestory", "游戏客户端", text, flags=re.I)


class AssetDownloader:
    def __init__(
        self,
        region: str = DEFAULT_REGION,
        version: str = DEFAULT_VERSION,
        output_dir: Optional[str] = None,
        local_mob_dir: Optional[str] = None,
        parser_root: Optional[str] = None,
        wz_iv_hex: str = "4D23C72B",
    ):
        self.region = region
        self.version = version
        self.base_url = f"https://maplestory.io/api/{self.region}/{self.version}"

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if output_dir is None:
            self.output_dir = os.path.join(base_dir, "templates")
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Extracted v83 Mob.wz images are the authoritative first source.
        # The online API remains a fallback only when the IMG is absent or
        # completely unreadable.  A usable local IMG is never mixed with a
        # potentially different network revision.
        self.local_mob_dir = os.path.abspath(
            local_mob_dir or os.path.join(base_dir, "Mob")
        )
        self.parser_root = os.path.abspath(
            parser_root or os.path.join(base_dir, "wz_python_tool")
        )
        try:
            self.wz_iv = bytes.fromhex("".join(str(wz_iv_hex).split()))
        except ValueError as exc:
            raise ValueError(f"无效的 Mob WZ 字符串密钥：{wz_iv_hex!r}") from exc
        if len(self.wz_iv) != 4:
            raise ValueError("Mob WZ 字符串密钥必须正好是4字节")

        self._mob_cache: Optional[List[Dict]] = None
        self._map_cache: Optional[List[Dict]] = None

    def _local_mob_path(self, mob_id: int) -> str:
        return os.path.join(self.local_mob_dir, f"{int(mob_id):07d}.img")

    @staticmethod
    def _resolve_local_uol(node):
        """Resolve a same-IMG UOL chain relative to each UOL's parent."""
        seen = set()
        for _ in range(16):
            if node is None or getattr(node, "type_name", "") != "UOL":
                return node
            if id(node) in seen or getattr(node, "parent", None) is None:
                return None
            seen.add(id(node))
            value = getattr(node, "value", None)
            if not value:
                return None
            node = node.parent.get(str(value))
        return None

    @classmethod
    def _first_local_canvas(cls, action_node, seen=None):
        """Return frame 0 (or the first numeric frame) as a Canvas node."""
        if action_node is None:
            return None
        action_node = cls._resolve_local_uol(action_node)
        if action_node is None:
            return None
        if seen is None:
            seen = set()
        if id(action_node) in seen:
            return None
        seen.add(id(action_node))
        if getattr(action_node, "type_name", "") == "Canvas":
            return action_node
        children = list(action_node.children())
        children.sort(
            key=lambda item: (
                0 if str(item.name).isdigit() else 1,
                int(item.name) if str(item.name).isdigit() else str(item.name),
            )
        )
        for child in children:
            candidate = cls._resolve_local_uol(child)
            if getattr(candidate, "type_name", "") == "Canvas":
                return candidate
            if candidate is not None:
                nested = cls._first_local_canvas(candidate, seen)
                if nested is not None:
                    return nested
        return None

    def _load_local_mob_frames(
        self, mob_id: int, actions: Tuple[str, ...]
    ) -> Dict[str, Image.Image]:
        """Decode requested action frames from ``Mob/<id>.img``."""
        path = self._local_mob_path(mob_id)
        if not os.path.isfile(path):
            return {}
        if not os.path.isdir(self.parser_root):
            print(
                f"[警告] 找到本地 Mob#{mob_id} IMG，但缺少解析器目录："
                f"{self.parser_root}；将回退在线接口"
            )
            return {}
        if self.parser_root not in sys.path:
            sys.path.insert(0, self.parser_root)
        try:
            from wzpy.canvas import decode_canvas
            from wzpy.crypto import WzKey
            from wzpy.wz_image import WzImage

            with open(path, "rb") as stream:
                data = stream.read()
            image = WzImage.from_bytes(
                data, key=WzKey(self.wz_iv), name=os.path.basename(path)
            )
            root = image.parse()
            if getattr(image, "truncated", False):
                warnings = "; ".join(getattr(image, "parse_warnings", []) or [])
                raise ValueError(f"IMG数据不完整 {warnings}".strip())
            frames: Dict[str, Image.Image] = {}
            for action in actions:
                canvas = self._first_local_canvas(root.child(action))
                if canvas is None:
                    continue
                frames[action] = decode_canvas(canvas, region="GMS").convert("RGBA")
            if frames:
                missing = [action for action in actions if action not in frames]
                print(
                    f"[AssetDownloader] Mob#{mob_id} 优先使用本地IMG "
                    f"{os.path.basename(path)}：{', '.join(frames)}"
                    + (f"；IMG本身没有 {', '.join(missing)}，这些动作跳过" if missing else "")
                )
            return frames
        except Exception as exc:
            print(
                f"[警告] 本地 Mob#{mob_id} IMG解析失败: {_public_error(exc)}；"
                "将回退在线接口"
            )
            return {}

    @staticmethod
    def _extract_feature_sprite(bgra: np.ndarray) -> np.ndarray:
        """从完整 BGRA 怪物图提取高反差局部特征，作为 mob_* 模板。"""
        if bgra is None or bgra.size == 0:
            return bgra
        bgr = bgra[:, :, :3] if bgra.ndim == 3 and bgra.shape[2] >= 3 else bgra
        alpha = bgra[:, :, 3] if bgra.ndim == 3 and bgra.shape[2] == 4 else None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        h, w = gray.shape[:2]
        valid = (alpha > 50).astype(np.uint8) if alpha is not None else np.ones((h, w), np.uint8)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy = np.sqrt(gx * gx + gy * gy) * valid
        # 眼睛/嘴巴通常在上 3/4；局部模板保留足够上下文以降低误匹配。
        search_h = max(1, int(h * 0.75))
        fw = max(12, min(w, int(w * 0.38)))
        fh = max(12, min(h, int(h * 0.34)))
        score = cv2.boxFilter(energy[:search_h], -1, (fw, fh), normalize=False)
        _, _, _, loc = cv2.minMaxLoc(score)
        x1 = max(0, min(w - fw, loc[0] - fw // 2))
        y1 = max(0, min(h - fh, loc[1] - fh // 2))
        return bgra[y1:y1 + fh, x1:x1 + fw].copy()

    def get_all_mobs(self) -> List[Dict]:
        """获取 GMS v83 全量怪物列表"""
        if self._mob_cache is not None:
            return self._mob_cache
        try:
            url = f"{self.base_url}/mob"
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                self._mob_cache = resp.json()
                return self._mob_cache
        except Exception as e:
            print(f"[错误] 连接在线怪物列表失败: {_public_error(e)}")
        return []

    def get_all_maps(self) -> List[Dict]:
        """获取 GMS v83 全量地图列表"""
        if self._map_cache is not None:
            return self._map_cache
        try:
            url = f"{self.base_url}/map"
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                self._map_cache = resp.json()
                return self._map_cache
        except Exception as e:
            print(f"[错误] 连接在线地图列表失败: {_public_error(e)}")
        return []

    def get_map_details(self, map_id: int) -> Optional[Dict]:
        """获取指定 Map ID 的地图详情 (包含刷怪列表 mobs)"""
        url = f"{self.base_url}/map/{map_id}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[错误] 获取地图 {map_id} 详情失败: {_public_error(e)}")
        return None

    def get_mob_details(self, mob_id: int) -> Optional[Dict]:
        """获取指定 Mob ID 的详情 (名称、血量等)"""
        url = f"{self.base_url}/mob/{mob_id}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[错误] 获取怪物 {mob_id} 详情失败: {_public_error(e)}")
        return None

    def download_and_save_mob_template(
        self,
        mob_id: int,
        custom_name: Optional[str] = None,
        scale: float = 1.0,
        force_redownload: bool = False
    ) -> List[str]:
        """
        全动作下载：为指定怪物下载【stand (站立)】、【move (移动)】、【hit1 (受击)】与【jump (跳跃)】四套精灵图，
        自动处理透明度，保留原始像素尺寸并生成正向与镜像模板。
        :return: 生成的所有模板文件绝对路径列表
        """
        # custom_name 为旧调用兼容参数；新目录始终由 Mob ID 唯一确定。
        prefix = custom_name or f"mob_{mob_id}"
        saved_paths = []
        mob_dir = os.path.join(self.output_dir, str(mob_id))
        # 分目录模板库由用户维护。目录一旦存在，即使其中只有部分动作
        # 或手工调过的特征图，也绝不能在地图同步时自动补写/覆盖。
        # force_redownload 是显式维护操作时唯一允许突破该保护的开关。
        if os.path.isdir(mob_dir) and not force_redownload:
            existing = glob.glob(os.path.join(mob_dir, "*.png"))
            if existing:
                print(
                    f"[AssetDownloader] 保留已有 Mob#{mob_id} 模板目录，"
                    f"跳过自动下载/更新（{len(existing)} 张）。"
                )
                return existing
        os.makedirs(mob_dir, exist_ok=True)

        # 死亡帧会与尸体/掉落物产生误匹配，不纳入运行时怪物模板库。
        actions = ("stand", "move", "hit1", "jump")
        raw_frames: Dict[str, Image.Image] = self._load_local_mob_frames(
            mob_id, actions
        )
        using_local_img = bool(raw_frames)

        # 只有本地 IMG 不存在或完全无法解析时才访问 IO。只要本地文件
        # 成功提供了动作，就以它为唯一事实来源；某动作在同版本 IMG 中
        # 不存在通常意味着怪物本来就没有该动作，不能再混入网络版本。
        # /render/{act} 是服务端重绘预览图，会损失原始半透明边缘，因此
        # 后备仍使用 /download 压缩包里的原始动作帧。
        missing_actions = [] if raw_frames else list(actions)
        if missing_actions:
            try:
                zip_url = f"{self.base_url}/mob/{mob_id}/download"
                zip_resp = requests.get(zip_url, headers=HEADERS, timeout=60)
                zip_resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as archive:
                    names = [n for n in archive.namelist() if n.lower().endswith(".png")]
                    for act in missing_actions:
                        exact = next(
                            (n for n in names if os.path.basename(n).lower() == f"{act}_0.png"),
                            None,
                        )
                        source_name = exact or next(
                            (n for n in names if os.path.basename(n).lower().startswith(f"{act}_")),
                            None,
                        )
                        if source_name:
                            with Image.open(io.BytesIO(archive.read(source_name))) as source:
                                raw_frames[act] = source.convert("RGBA").copy()
                downloaded = [act for act in missing_actions if act in raw_frames]
                if downloaded:
                    print(
                        f"[AssetDownloader] Mob#{mob_id} 缺失动作使用IO原始帧："
                        f"{', '.join(downloaded)}"
                    )
            except Exception as e:
                print(f"[错误] 下载怪物 {mob_id} 原始动作压缩包失败: {_public_error(e)}")

        for act in actions:
            act_tag = "" if act == "stand" else f"_{act}"
            # mob_global 与 mob 的动作/朝向后缀必须一致，供多尺度匹配器
            # 自动配对：global 还原全身框/Alpha 掩膜，mob 做局部特征匹配。
            r_global = os.path.join(mob_dir, f"mob_global{act_tag}_right.png")
            l_global = os.path.join(mob_dir, f"mob_global{act_tag}_left.png")
            r_feature = os.path.join(mob_dir, f"mob{act_tag}_right.png")
            l_feature = os.path.join(mob_dir, f"mob{act_tag}_left.png")

            if not force_redownload and all(os.path.exists(p) for p in (r_global, l_global, r_feature, l_feature)):
                saved_paths.extend([r_global, l_global, r_feature, l_feature])
                continue

            try:
                source_frame = raw_frames.get(act)
                if source_frame is None:
                    if using_local_img:
                        print(
                            f"[警告] Mob#{mob_id} 本地IMG没有 {act}_0，跳过该动作"
                        )
                    else:
                        print(
                            f"[警告] Mob#{mob_id} 本地IMG与IO压缩包均未找到 "
                            f"{act}_0.png，跳过"
                        )
                    continue
                rgba_np = np.array(source_frame.convert("RGBA"))
                if scale != 1.0:
                    rgba_np = cv2.resize(rgba_np, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

                # 保留 BGRA 4 通道保存为 PNG (包含完整透明度)
                bgra = cv2.cvtColor(rgba_np, cv2.COLOR_RGBA2BGRA)

                left_bgra = cv2.flip(bgra, 1)
                cv2.imwrite(r_global, bgra)
                cv2.imwrite(l_global, left_bgra)
                cv2.imwrite(r_feature, self._extract_feature_sprite(bgra))
                cv2.imwrite(l_feature, self._extract_feature_sprite(left_bgra))
                saved_paths.extend([r_global, l_global, r_feature, l_feature])
            except Exception as e:
                print(f"[错误] 下载怪物 {mob_id} 动作 {act} 模板失败: {_public_error(e)}")

        return saved_paths

    def sync_map_mobs(self, map_id: int) -> List[Dict]:
        """
        全自动同步指定地图的所有怪物：
        查询地图中出现的所有 mob_id，获取怪物名称并自动下载 stand/move/hit1/die1/jump 模板
        :return: 怪物信息列表 [{"id": int, "name": str, "ready": bool}]
        """

        map_info = self.get_map_details(map_id)
        if not map_info:
            return []

        raw_mobs = map_info.get("mobs", [])
        unique_mob_ids: Set[int] = set()
        for m in raw_mobs:
            if isinstance(m, dict) and "id" in m:
                unique_mob_ids.add(m["id"])

        synced_mobs = []
        for mob_id in unique_mob_ids:
            mob_detail = self.get_mob_details(mob_id)
            mob_name = mob_detail.get("name", f"Mob_{mob_id}") if mob_detail else f"Mob_{mob_id}"
            tpls = self.download_and_save_mob_template(mob_id)
            synced_mobs.append({
                "id": mob_id,
                "name": mob_name,
                "file_prefix": f"mob_{mob_id}",
                "ready": len(tpls) > 0
            })

        return synced_mobs
