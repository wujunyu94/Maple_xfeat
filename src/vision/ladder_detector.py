"""
ladder_detector.py - 基于模板匹配与垂直共线聚类的主视觉梯子/绳索精准检测器
利用标准梯级/绳段模板库进行多尺度匹配，结合垂直空间聚类彻底过滤电线杆、墙角等背景误检，输出像素级中轴坐标。
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict


class LadderDetector:
    def __init__(self, templates_dir: Optional[str] = None, match_threshold: float = 0.70):
        """
        :param templates_dir: 梯子/绳索模板目录路径
        :param match_threshold: 模板匹配置信度阈值 (默认 0.70)
        """
        self.threshold = match_threshold
        if templates_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.tpl_dir = os.path.join(base_dir, "assets", "ladder_templates")
        else:
            self.tpl_dir = templates_dir

        self.templates: List[Tuple[str, np.ndarray]] = []
        self._load_templates()

    def _load_templates(self):
        """从模板目录加载所有梯级与绳索图片切片"""
        self.templates.clear()
        if not os.path.exists(self.tpl_dir):
            os.makedirs(self.tpl_dir, exist_ok=True)
            return

        for fname in os.listdir(self.tpl_dir):
            if fname.lower().endswith((".png", ".jpg", ".bmp")):
                fpath = os.path.join(self.tpl_dir, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    self.templates.append((fname, img))

    def detect_in_roi(
        self,
        frame: np.ndarray,
        player_pos: Tuple[int, int],
        search_direction: str = "left",
        is_ladder: bool = True,
        roi_width: int = 350,
        roi_height_up: int = 200,
        roi_height_down: int = 260
    ) -> Optional[Dict[str, Any]]:
        """
        在角色身侧局部感兴趣区域 (ROI) 中使用模板匹配精确提取真实梯子/绳索。

        :param frame: 游戏全图 (BGR 格式)
        :param player_pos: 角色脚底像素 (pfx, pfy)
        :param search_direction: 搜索方向 ("left" 或 "right")
        :param is_ladder: True 为梯子，False 为绳索
        :param roi_width: 水平搜索宽度
        :param roi_height_up: 角色上方搜索高度
        :param roi_height_down: 角色下方搜索深度
        :return: 检出信息字典或 None
        """
        pfx, pfy = player_pos
        h, w = frame.shape[:2]

        if search_direction == "left":
            rx1 = max(0, pfx - roi_width)
            rx2 = pfx
        else:
            rx1 = pfx
            rx2 = min(w, pfx + roi_width)

        ry1 = max(0, pfy - roi_height_up)
        ry2 = min(h, pfy + roi_height_down)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        roi = frame[ry1:ry2, rx1:rx2]
        if not self.templates:
            self._load_templates()
            if not self.templates:
                return None

        # 遍历所有可用模板并在 ROI 中执行匹配
        matched_boxes: List[Tuple[int, int, int, int, float]] = []  # (x, y, w, h, score)

        for tname, tpl in self.templates:
            th, tw = tpl.shape[:2]
            if roi.shape[0] < th or roi.shape[1] < tw:
                continue

            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= self.threshold)
            for pt_y, pt_x in zip(locs[0], locs[1]):
                score = float(res[pt_y, pt_x])
                matched_boxes.append((pt_x, pt_y, tw, th, score))

        if not matched_boxes:
            # 降级尝试稍低阈值
            for tname, tpl in self.templates:
                th, tw = tpl.shape[:2]
                if roi.shape[0] < th or roi.shape[1] < tw:
                    continue
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= 0.60)
                for pt_y, pt_x in zip(locs[0], locs[1]):
                    matched_boxes.append((pt_x, pt_y, tw, th, float(res[pt_y, pt_x])))

        if not matched_boxes:
            return None

        # 空间垂直聚类：将水平距离在 15px 以内的梯级聚合为同一根梯子
        clusters: List[List[Tuple[int, int, int, int, float]]] = []
        for box in matched_boxes:
            bx_center = box[0] + box[2] / 2.0
            found = False
            for cl in clusters:
                cl_mean_x = np.mean([b[0] + b[2] / 2.0 for b in cl])
                if abs(bx_center - cl_mean_x) <= 15:
                    cl.append(box)
                    found = True
                    break
            if not found:
                clusters.append([box])

        # 筛选垂直跨度最大、匹配点最多的最优梯子
        best_cluster = None
        best_ladder_score = -1.0

        for cl in clusters:
            # 梯子在 Y 方向应有多个梯级重复出现
            ys = [b[1] for b in cl]
            y_span = max(ys) - min(ys)
            total_score = sum(b[4] for b in cl) + y_span * 0.05
            if total_score > best_ladder_score:
                best_ladder_score = total_score
                best_cluster = cl

        if best_cluster is None:
            return None

        # 计算最优梯子的绝对主画面中轴 X
        ladder_roi_x = float(np.mean([b[0] + b[2] / 2.0 for b in best_cluster]))
        ladder_screen_x = rx1 + ladder_roi_x
        ladder_w = int(np.mean([b[2] for b in best_cluster]))
        dx = ladder_screen_x - pfx

        # 计算梯子的上下垂直跨度
        ladder_screen_y1 = ry1 + min(b[1] for b in best_cluster)
        ladder_screen_y2 = ry1 + max(b[1] + b[3] for b in best_cluster)

        return {
            "type": "ladder" if is_ladder else "rope",
            "screen_x": ladder_screen_x,
            "player_screen_x": pfx,
            "dx": dx,
            "rail1_screen_x": ladder_screen_x - ladder_w / 2.0,
            "rail2_screen_x": ladder_screen_x + ladder_w / 2.0,
            "rail_dist": ladder_w,
            "vertical_span": (ladder_screen_y1, ladder_screen_y2),
            "matched_count": len(best_cluster),
            "confidence": float(np.mean([b[4] for b in best_cluster])),
            "roi_bounds": (rx1, ry1, rx2, ry2),
            "matched_boxes": [(rx1 + b[0], ry1 + b[1], b[2], b[3]) for b in best_cluster],
        }
