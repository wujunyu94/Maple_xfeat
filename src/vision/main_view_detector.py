"""
main_view_detector.py - 高性能主视口角色定位、像素级视觉朝向识别与怪物检测引擎 (极速版 60+ FPS)
优化:
  1. GMS v83 Mob ID 到本地多帧特征全量智能扩展 (CLASSIC_MOB_ID_MAP)。
  2. 多人防误认 (唯一名字牌最高优先 + 摄像机中心优选)。
  3. 平衡置信度 (0.52) 与严格地图物种隔离。
"""

import os
import time
import json
import glob
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from src.vision.multi_scale_monster_backend import MultiScaleMonsterBackend
from src.vision.cuda_hp_bar_detector import CudaHpBarDetector
from src.engine.attack_skills import (
    attack_box as skill_attack_box, contains as target_in_box,
    choose_skill_for_target, eligible_skills, mob_id_from_name, mob_id_of, skills_from_config,
)

user32 = ctypes.windll.user32
VK_LEFT = 0x25
VK_RIGHT = 0x27

# 红色攻击框和 CombatFSM 必须共用同一条攻击有效性判定。
# 光流跟踪活跃时认可其连续性；全模扫描时容忍受击闪白与动作切换。
ATTACK_OBSERVATION_MAX_AGE_SEC = 0.50
ATTACK_OBSERVATION_MAX_MISSES = 20

# 朝向模板在挥刀帧会被武器、手臂和技能光效遮挡。低质量匹配仍可用于
# 画出特征位置，但绝不能据此翻转人物朝向；真正转身后的静态/走路帧
# 通常会迅速恢复到 0.8 以上。
# The blue FEATURE box means that the feature is genuinely visible in this
# frame.  Scores around 0.35~0.55 are common on ropes/attack effects even when
# the calibrated hat/body patch is fully occluded, so they may retain the last
# reliable direction but must never render a current-frame match box.
FACING_FEATURE_VISIBLE_SCORE = 0.60
FACING_SWITCH_MIN_SCORE = 0.60
FACING_SWITCH_MIN_MARGIN = 0.025
FACING_SWITCH_IMMEDIATE_SCORE = 0.75
FACING_SWITCH_IMMEDIATE_MARGIN = 0.10


def monster_observation_age(target: Any, now: Optional[float] = None) -> float:
    """返回目标距最近一次真实模板命中的时间；不采用可能冻结的缓存 age。"""
    current = time.perf_counter() if now is None else float(now)
    try:
        timestamp = float(getattr(target, "last_update_ts"))
        if np.isfinite(timestamp):
            return max(0.0, current - timestamp)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return max(0.0, float(getattr(target, "time_since_update_sec", 999.0)))
    except (TypeError, ValueError):
        return 999.0


def is_fresh_attack_observation(
    target: Any,
    now: Optional[float] = None,
    max_age: Optional[float] = None,
    hard_timeout_enabled: Optional[bool] = None,
) -> bool:
    """统一 HUD 红框与战斗线程使用的可攻击目标有效性判定。"""
    current = time.perf_counter() if now is None else float(now)
    if hard_timeout_enabled is None:
        hard_timeout_enabled = bool(
            getattr(target, "attack_observation_hard_timeout_enabled", True)
        )

    obs_age = monster_observation_age(target, current)

    if hard_timeout_enabled:
        if max_age is None:
            max_age = getattr(
                target,
                "attack_observation_hard_timeout_sec",
                ATTACK_OBSERVATION_MAX_AGE_SEC,
            )
        max_age_val = max(0.20, float(max_age))
        if obs_age > max_age_val:
            return False
    else:
        max_age_val = float("inf")

    # 1. 检查光流/轨迹跟踪活跃度：若光流仍在持续更新目标坐标，且模板命中间隔在合理范围内，坚决判定有效
    last_track_ts = getattr(target, "last_track_ts", 0.0)
    if last_track_ts and last_track_ts > 0:
        track_age = max(0.0, current - float(last_track_ts))
        if track_age <= 0.30 and (not hard_timeout_enabled or obs_age <= max_age_val):
            return True

    # 2. 全速模式/常规模式漏检次数校验：容忍受击闪白与数字遮挡
    try:
        misses = max(0, int(getattr(target, "consecutive_full_scan_misses", 0)))
    except (TypeError, ValueError):
        misses = 0
    return misses <= ATTACK_OBSERVATION_MAX_MISSES and (
        not hard_timeout_enabled or obs_age <= max_age_val
    )

# GMS v83 经典怪物 ID 到模板关键词全映射表
CLASSIC_MOB_ID_MAP = {
    3210800: ["lupin", "monkey", "angel_monkey"],
    4230101: ["zombie_lupin", "zombie", "lupin"],
    5300100: ["malady"],
    3210101: ["lupin", "angel_monkey", "monkey"],
    3210100: ["zombie_lupin", "lupin", "angel_monkey"],
    2230100: ["evil_eye"],
    2230101: ["curse_eye", "evil_eye"],
    2230102: ["cold_eye", "evil_eye"],
    2110200: ["horny_mushroom"],
    1110100: ["green_mushroom"],
    1110101: ["horny_mushroom", "green_mushroom"],
    2300100: ["zombie_mushroom"],
    210100:  ["slime"],
    1210100: ["pig"],
    1210101: ["ribbon_pig", "pig"],
    130100:  ["stump"],
    130101:  ["red_snail", "snail"],
    100100:  ["snail"],
    100101:  ["blue_snail", "snail"],
    1140100: ["ghost_stump", "stump"],
    1140130: ["smirking_ghost_stump", "ghost_stump"]
}

# 怪物名称别名池
MOB_ALIAS_MAP = {
    "lupin": ["lupin", "angel_monkey", "monkey", "zombie_lupin"],
    "zombie_lupin": ["zombie_lupin", "lupin", "angel_monkey"],
    "monkey": ["angel_monkey", "lupin", "zombie_lupin"],
    "evil_eye": ["evil_eye"],
    "curse_eye": ["evil_eye"],
    "horny_mushroom": ["horny_mushroom"],
    "green_mushroom": ["green_mushroom"],
    "zombie_mushroom": ["zombie_mushroom"],
    "slime": ["slime"],
    "pig": ["pig", "ribbon_pig"],
    "ribbon_pig": ["ribbon_pig", "pig"],
    "stump": ["stump", "dark_stump", "ghost_stump"],
    "snail": ["snail", "blue_snail", "red_snail"]
}


@dataclass
class TrackedMob:
    """
    通用多目标时序追踪对象 (MOT Track):
    具备真实物理时间（毫秒级）与速度惯性外推、自适应平滑滤波、丢失容忍缓冲 (Time-Based Lost Buffer)。
    严格基于真实时间 (0.22s) 维持生命周期，自适应 30FPS/60FPS/144FPS 任意帧率。
    """
    track_id: int
    name: str
    bbox: Tuple[int, int, int, int]
    cx: float
    cy: float
    w: float
    h: float
    score: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    time_since_update_sec: float = 0.0
    last_update_ts: float = field(default_factory=time.perf_counter)
    # 只在消费新的完整模板扫描批次时更新。轻量光流帧不得累加漏检。
    consecutive_full_scan_misses: int = 0
    attack_observation_hard_timeout_sec: float = ATTACK_OBSERVATION_MAX_AGE_SEC
    attack_observation_hard_timeout_enabled: bool = True
    # 与 last_update_ts 分离：前者只代表模板真正命中，后者允许光流
    # 追踪维持视觉框连续性，绝不用于攻击决策。
    last_track_ts: float = field(default_factory=time.perf_counter)

    def predict(self, dt_sec: float):
        # 惯性外推 (物理速度积分 + 阻尼衰减)
        decay = max(0.2, 1.0 - dt_sec * 3.0)
        self.cx += self.vx * dt_sec * 60.0 * decay
        self.cy += self.vy * dt_sec * 60.0 * decay
        self.bbox = (int(self.cx - self.w / 2), int(self.cy - self.h / 2), int(self.w), int(self.h))
        self.time_since_update_sec += dt_sec

    def update(self, det_bbox: Tuple[int, int, int, int], det_score: float, det_name: str, current_ts: float):
        new_w = float(det_bbox[2])
        new_h = float(det_bbox[3])
        new_cx = float(det_bbox[0]) + new_w / 2.0
        new_cy = float(det_bbox[1]) + new_h / 2.0

        dt = max(0.005, current_ts - self.last_update_ts)
        inst_vx = (new_cx - self.cx) / (dt * 60.0)
        inst_vy = (new_cy - self.cy) / (dt * 60.0)

        # 平滑速度与位置 (指数移动加权平滑 EMA)
        self.vx = 0.35 * self.vx + 0.65 * inst_vx
        self.vy = 0.35 * self.vy + 0.65 * inst_vy
        self.cx = 0.75 * new_cx + 0.25 * self.cx
        self.cy = 0.75 * new_cy + 0.25 * self.cy
        self.w = 0.7 * new_w + 0.3 * self.w
        self.h = 0.7 * new_h + 0.3 * self.h

        self.bbox = (int(self.cx - self.w / 2), int(self.cy - self.h / 2), int(self.w), int(self.h))
        self.score = det_score
        self.name = det_name
        self.hits += 1
        self.time_since_update_sec = 0.0
        self.last_update_ts = current_ts
        self.last_track_ts = current_ts
        self.consecutive_full_scan_misses = 0

    @property
    def is_ghost(self) -> bool:
        return False

    @property
    def is_dead(self) -> bool:
        return False

    @property
    def center_x(self) -> int:
        return int(round(self.cx))

    @property
    def center_y(self) -> int:
        return int(round(self.cy))

    @property
    def center(self) -> Tuple[int, int]:
        return (int(round(self.cx)), int(round(self.cy)))


@dataclass
class MonsterTarget:
    name: str
    bbox: Tuple[int, int, int, int]         # (x, y, w, h)
    center: Tuple[int, int]                  # (cx, cy)
    score: float                             # 置信度 (0.0~1.0)
    distance: float                          # 距离角色本体的像素距离
    is_in_attack_range: bool = False         # 是否位于攻击判定盒内
    relative_direction: str = "right"        # "left", "right"
    track_id: Optional[int] = None            # 对应时序追踪器 ID
    last_update_ts: float = 0.0               # 最近一次真实模板命中的单调时钟
    last_track_ts: float = 0.0                # 最近一次光流/轨迹跟踪更新的单调时钟
    consecutive_full_scan_misses: int = 0      # 连续完整扫描漏检次数
    attack_observation_hard_timeout_sec: float = ATTACK_OBSERVATION_MAX_AGE_SEC
    attack_observation_hard_timeout_enabled: bool = True
    is_fresh_for_attack: bool = False         # 本次快照是否满足统一攻击时效线
    mob_id: Optional[str] = None              # 模板目录中的真实 Mob ID；track_id 只是临时轨迹号


@dataclass
class MainViewResult:
    player_found: bool = False
    player_pos: Optional[Tuple[int, int]] = None    # 角色身体中心 (px, py)
    player_bbox: Optional[Tuple[int, int, int, int]] = None
    feature_bbox: Optional[Tuple[int, int, int, int]] = None # (x, y, w, h) 匹配到的服饰/帽子/局部特征框
    facing_direction: str = "right"                 # "left" 或 "right"
    facing_confidence: float = 1.0                  # 朝向置信度
    is_facing_locked: bool = False                  # 当前是否处于攻击动作保护锁中
    attack_box: Optional[Tuple[int, int, int, int]] = None # (x, y, w, h) 攻击判定盒
    rear_attack_box: Optional[Tuple[int, int, int, int]] = None # 单向技能的回身候选盒
    front_skirmish_box: Optional[Tuple[int, int, int, int]] = None
    rear_skirmish_box: Optional[Tuple[int, int, int, int]] = None
    skill_attack_boxes: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    locked_skill_id: Optional[str] = None
    monsters: List[MonsterTarget] = field(default_factory=list)
    locked_target: Optional[MonsterTarget] = None   # 当前锁定的最近可攻击怪物
    proc_time_ms: float = 0.0                       # 本帧算法耗时 (毫秒)
    timestamp: float = 0.0


@dataclass(frozen=True)
class MonsterDetectionBatch:
    """独立完整模板线程产出的不可变最新帧结果。"""
    frame_seq: int
    captured_at: float
    completed_at: float
    template_generation: int
    detections: Tuple[Tuple[int, int, int, int, str, float], ...]
    duration_sec: float
    backend_lock_wait_sec: float = 0.0
    template_match_sec: float = 0.0
    hp_bar_sec: float = 0.0
    candidate_filter_sec: float = 0.0


class MainViewDetector:
    def __init__(
        self,
        template_dir: Optional[str] = None,
        attack_reach_x: int = 260,
        attack_reach_y: int = 140,
        attack_reach_y_up: Optional[int] = None,
        attack_reach_y_down: Optional[int] = None,
        behind_reach_x: int = 40,
        skirmish_range_x: int = 0,
        attack_two_way: bool = False,
        monster_threshold: float = 0.52,
        scale_factor: float = 0.35,
        combat_roi_radius_x: int = 950,
        combat_roi_radius_y: int = 550,
        tracker_buffer_time_sec: float = 0.12,
        monster_template_scale: float = 1.0,
        monster_compute_device: str = "auto",
        monster_hp_bar_compute_device: str = "auto",
        monster_coarse_scale: float = 0.6,
        monster_redetect_interval: int = 4,
        player_feature_threshold: float = 0.58,
        attack_observation_hard_timeout_sec: float = ATTACK_OBSERVATION_MAX_AGE_SEC,
        attack_observation_hard_timeout_enabled: bool = True,
        ghost_box_safety_timeout_sec: float = 0.75,
        ghost_box_safety_timeout_enabled: bool = True,
    ):
        self.attack_reach_x = attack_reach_x
        # 保留旧字段兼容已有 config；Y+ 表示身体上方、Y− 表示身体下方。
        self.attack_reach_y_up = int(attack_reach_y if attack_reach_y_up is None else attack_reach_y_up)
        self.attack_reach_y_down = int(attack_reach_y if attack_reach_y_down is None else attack_reach_y_down)
        self.attack_reach_y = self.attack_reach_y_up  # 兼容旧调用方
        self.behind_reach_x = behind_reach_x
        self.skirmish_range_x = max(0, int(skirmish_range_x))
        self.attack_two_way = bool(attack_two_way)
        # GUI 可将共享配置字典设到这里；默认离线调用仍按旧攻击盒工作。
        self.attack_config: Optional[Dict[str, Any]] = None
        self.monster_threshold = monster_threshold
        self.scale_factor = scale_factor
        self.combat_roi_radius_x = combat_roi_radius_x
        self.combat_roi_radius_y = combat_roi_radius_y
        self.tracker_buffer_time_sec = tracker_buffer_time_sec  # 真实物理时间缓冲 (默认 120 毫秒，响应极其干脆)
        self.monster_template_scale = float(monster_template_scale)
        # 模板全屏重检频率。中间帧由光流/轨迹维持；0ms 缓冲时强制逐帧
        # 重检，避免用户明确关闭残留缓冲后出现旧框。
        self.monster_redetect_interval = max(1, int(monster_redetect_interval))
        self.player_feature_threshold = float(player_feature_threshold)
        self.attack_observation_hard_timeout_sec = max(
            0.0, float(attack_observation_hard_timeout_sec)
        )
        self.attack_observation_hard_timeout_enabled = bool(
            attack_observation_hard_timeout_enabled
        )
        self.ghost_box_safety_timeout_sec = max(0.0, float(ghost_box_safety_timeout_sec))
        self.ghost_box_safety_timeout_enabled = bool(ghost_box_safety_timeout_enabled)
        self._monster_frame_index = 0
        self.exclusion_regions: List[Dict[str, Any]] = []
        self.last_frame_timestamp = time.perf_counter()
        # CombatFSM 与 HUD 必须使用同一个分辨率缩放基准。process 每帧
        # 更新该值，避免状态机拿配置原值而 HUD 使用缩放值。
        self.last_frame_width = 1920
        # 目标锁定平滑保持 (Hold Latch)：防止受击硬直或1帧噪点造成红橙框闪烁
        self._locked_target_id: Optional[int] = None
        self._locked_target_until: float = 0.0
        self.enable_monster_detection: bool = True
        self.enable_monster_hp_bar_detection: bool = True
        requested_hp_device = str(monster_hp_bar_compute_device or "auto").lower()
        if requested_hp_device not in ("auto", "cpu", "cuda"):
            requested_hp_device = "auto"
        self.monster_hp_bar_compute_device = requested_hp_device
        # CUDA血条检测延迟到第一张实际怪物画面再初始化，避免安全区/城镇
        # 启动时仅因一个未使用功能导入PyTorch并创建CUDA上下文。
        self._cuda_hp_bar_detector: Optional[CudaHpBarDetector] = None
        self._cuda_hp_bar_init_attempted = False
        self._cuda_hp_bar_init_lock = threading.Lock()
        self._cuda_hp_bar_generation = 0
        self._cuda_hp_bar_error = ""
        self.last_hp_bar_backend = "cpu"
        self.last_hp_bar_cost_ms = 0.0



        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if template_dir is None:
            self.template_dir = os.path.join(base_dir, "assets", "templates")
        else:
            self.template_dir = template_dir

        self.player_templates_gray = {}
        self.facing_templates_gray = {}
        self.facing_template_masks = {}
        self.monster_templates_scaled_gray = {}
        self.monster_cached_scaled_templates = []
        # 新模板按 Mob ID 分目录：templates/<MobID>/mob*.png；其中
        # mob_global 是全身图/掩膜，mob 是局部特征图，由多尺度算法配对。
        self.multi_scale_template_root = os.path.join(base_dir, "templates")
        self.multi_scale_backend = MultiScaleMonsterBackend(
            self.multi_scale_template_root, monster_threshold,
            template_scale=self.monster_template_scale,
            compute_device=monster_compute_device, coarse_scale=monster_coarse_scale
        )
        # 完整模板匹配只允许一个后台线程进入；地图切换/模板重载使用
        # 同一把锁，并通过 generation 丢弃旧地图的晚到结果。
        self._monster_backend_lock = threading.RLock()
        self._monster_track_lock = threading.RLock()
        self._monster_template_generation = 0
        self._last_async_batch_sequence = -1
        self.tracked_monsters: List[TrackedMob] = []
        self.next_track_id: int = 1
        self.last_full_scan_candidate_count: int = 0
        self._track_flow_prev_gray = None
        # 相邻帧稀疏光流：模板匹配短暂失分时仍可让已确认目标平滑保框。
        self.costume_template_bgr = None
        self.costume_template_gray = None
        self.costume_template_flipped_gray = None
        self.costume_template_mask = None
        self.costume_template_flipped_mask = None
        self.costume_size = None
        self.costume_feet_offset_y = 55
        self.enable_costume_verification = True
        self.enable_two_stage_feature = True
        self.two_stage_profile = None
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._load_player_templates()

        # 默认加载猴子森林怪物
        self.current_facing: str = "right"
        self.facing_lock_until: float = 0.0
        self._facing_pending_direction: Optional[str] = None
        self._facing_pending_count: int = 0
        self.last_facing_scores: Tuple[float, float] = (0.0, 0.0)
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.last_player_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_feature_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_player_time: float = 0.0
        # 与 last_player_time 分开：process(reuse_player=True) 会刷新后者，
        # 只有本帧真实模板投票成功才更新 observed_time。换图后的视觉门禁
        # 必须依赖真实命中，不能把复用的旧坐标当作重新识别成功。
        self.last_player_observed_time: float = 0.0
        self.player_vx: float = 0.0
        self.player_vy: float = 0.0

    @property
    def active_tracks(self) -> List[TrackedMob]:
        return self.tracked_monsters



    def lock_facing_during_attack(self, facing: Optional[str] = None, duration_sec: float = 0.45):
        now = time.perf_counter()
        if facing in ("left", "right"):
            self.current_facing = facing
        self._facing_pending_direction = None
        self._facing_pending_count = 0
        self.facing_lock_until = max(
            self.facing_lock_until, now + max(0.0, float(duration_sec))
        )

    def _load_player_templates(self):
        """加载角色专属名字牌（多部件空间分解）、勋章与朝向特征"""
        os.makedirs(self.template_dir, exist_ok=True)
        self.player_parts = []
        self.facing_templates_gray = {}
        self.facing_template_masks = {}
        self.nametag_color_profile = None
        
        # 1. 加载名字牌/角色特征并分解为空间部件（精准对齐脚底 Foothold 锚点与真实身材尺寸）
        fpath_name = os.path.join(self.template_dir, "player_nametag.png")
        if os.path.exists(fpath_name):
            img = cv2.imread(fpath_name)
            if img is not None:
                h, w = img.shape[:2]
                g_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.player_templates_gray["nametag"] = g_full
                
                # 智能识别框选范围与真实脚底站立线 (Foothold Line):
                aspect = float(w) / float(max(1, h))
                if aspect >= 1.8 or h <= 50:
                    # 模式 1：名牌/勋章条模式 (包含单独名牌、勋章或名牌+勋章，w/h >= 1.8 或 h <= 50)
                    # 脚底就在名牌的最顶边 (feet_offset_y = 0)
                    feet_offset_y = 0
                    char_h = 55
                    char_w = int(max(45, min(65, int(w * 0.45))))
                    self.player_bbox_size = (char_w, char_h)
                else:
                    # 模式 2：角色全身大框模式 (包含帽子到脚底名牌，h > 50 且方形/立正竖框)
                    # 帽子在 y=0，脚底位于名牌上方 (距离框底约 45 像素)
                    feet_offset_y = max(45, h - 45)
                    char_h = feet_offset_y
                    char_w = w
                    self.player_bbox_size = (char_w, char_h)




                # 动态自适应提取名字牌主色调特征 (Color Fingerprint)，摆脱写死 85~115 青蓝色的局限
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # 排除边缘纯黑 (V < 35) 和中间白字文字笔画 (S < 30 且 V > 200)，提取真正的底色像素
                valid_bg_mask = (hsv[:, :, 2] >= 35) & ~((hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 200))
                if np.any(valid_bg_mask):
                    dom_h = float(np.median(hsv[:, :, 0][valid_bg_mask]))
                    dom_s = float(np.median(hsv[:, :, 1][valid_bg_mask]))
                    dom_v = float(np.median(hsv[:, :, 2][valid_bg_mask]))
                else:
                    dom_h, dom_s, dom_v = float(hsv[:, :, 0].mean()), float(hsv[:, :, 1].mean()), float(hsv[:, :, 2].mean())

                # 自适应容差范围 (H±18, S 动态下限, V 动态下限)
                h_low = max(0, int(dom_h - 18))
                h_high = min(179, int(dom_h + 18))
                s_low = max(25, int(dom_s - 75))
                v_low = max(35, int(dom_v - 75))

                tpl_bg_hits = np.count_nonzero(
                    cv2.inRange(hsv, np.array([h_low, s_low, v_low], dtype=np.uint8),
                                     np.array([h_high, 255, 255], dtype=np.uint8))
                )
                expected_ratio = float(tpl_bg_hits) / float(max(1, h * w))

                self.nametag_color_profile = {
                    "h_range": (h_low, h_high),
                    "s_range": (s_low, 255),
                    "v_range": (v_low, 255),
                    "expected_ratio": expected_ratio,
                    "dominant_h": dom_h,
                    "dominant_s": dom_s,
                    "dominant_v": dom_v,
                    "is_adaptive": True
                }

                # 登记多部件: (name, bgr_tpl, gray_tpl, offset_x_from_center, feet_offset_y)
                self.player_parts.append(("nametag_full", img, g_full, 0, feet_offset_y))
                if w >= 24:
                    g_left = g_full[:, :w//2]
                    g_right = g_full[:, w//2:]
                    g_mid = g_full[:, w//4: 3*w//4]
                    self.player_parts.append(("nametag_left", img[:, :w//2], g_left, -w//4, feet_offset_y))
                    self.player_parts.append(("nametag_right", img[:, w//2:], g_right, w//4, feet_offset_y))
                    self.player_parts.append(("nametag_mid", img[:, w//4: 3*w//4], g_mid, 0, feet_offset_y))

                # 遮挡鲁棒锚点：梯子、技能光效通常只遮住名字牌的局部。
                # 使用重叠的小块（而非只切左右两半），随后按同一个完整名字
                # 牌中心聚类投票。x/y 偏移必须保留，才能从小块还原人物位置。
                if w >= 48 and h >= 24:
                    anchor_specs = [
                        ("top_left", 0, max(30, w * 40 // 100), 0, h // 2),
                        ("top_mid", w * 22 // 100, w * 68 // 100, 0, h // 2),
                        ("top_right", w * 55 // 100, w, 0, h // 2),
                        ("bottom_left", 0, max(30, w * 40 // 100), h // 2, h),
                        ("bottom_mid", w * 22 // 100, w * 68 // 100, h // 2, h),
                        ("bottom_right", w * 55 // 100, w, h // 2, h),
                    ]
                    for label, x1, x2, y1, y2 in anchor_specs:
                        if x2 - x1 < 20 or y2 - y1 < 12:
                            continue
                        part_bgr = img[y1:y2, x1:x2]
                        part_gray = g_full[y1:y2, x1:x2]
                        center_offset_x = ((x1 + x2) / 2.0) - (w / 2.0)
                        self.player_parts.append((
                            f"nametag_anchor_{label}", part_bgr, part_gray,
                            center_offset_x, feet_offset_y - y1,
                        ))





        # 2. 加载勋章
        fpath_medal = os.path.join(self.template_dir, "player_medal.png")
        if os.path.exists(fpath_medal) and getattr(self, "is_nametag_strip", False):
            img = cv2.imread(fpath_medal)
            if img is not None:
                g_medal = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.player_templates_gray["medal"] = g_medal
                self.player_parts.append(("medal", img, g_medal, 0, 0))

        # 3. 加载朝向特征
        for fname, key in [("player_facing_right.png", "right"), ("player_facing_left.png", "left")]:
            fpath = os.path.join(self.template_dir, fname)
            if os.path.exists(fpath):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.facing_templates_gray[key] = img

        # 4. 加载服饰/魔法帽特征（双重保险）
        self.costume_template_bgr = None
        self.costume_template_gray = None
        self.costume_template_flipped_gray = None
        self.costume_template_mask = None
        self.costume_template_flipped_mask = None
        self.costume_size = None

        hat_paths = [
            os.path.join(self.template_dir, "player_hat.png"),
            os.path.join(self.template_dir, "player_costume.png")
        ]
        base_assets = os.path.dirname(self.template_dir)
        fallback_head = os.path.join(base_assets, "char_head_template.png")
        if not any(os.path.exists(p) for p in hat_paths) and os.path.exists(fallback_head):
            hat_paths.append(fallback_head)

        for fpath_hat in hat_paths:
            if os.path.exists(fpath_hat):
                img_hat = cv2.imread(fpath_hat)
                if img_hat is not None and img_hat.shape[0] >= 10 and img_hat.shape[1] >= 10:
                    self.costume_template_bgr = img_hat
                    g_hat = cv2.cvtColor(img_hat, cv2.COLOR_BGR2GRAY) if len(img_hat.shape) == 3 else img_hat
                    self.costume_template_gray = g_hat
                    self.costume_template_flipped_gray = cv2.flip(g_hat, 1)
                    self.costume_size = (img_hat.shape[1], img_hat.shape[0])
                    bh = getattr(self, "player_bbox_size", (55, 60))[1]
                    self.costume_feet_offset_y = max(30, min(75, int(bh * 0.75)))
                    print(f"[MainViewDetector] 载入服饰/魔法帽特征: {os.path.basename(fpath_hat)} ({self.costume_size[0]}x{self.costume_size[1]})")
                    break

        # 5. 加载两段式全身掩膜与局部特征 (两段式标定体系)
        self.two_stage_profile = None
        prof_path = os.path.join(self.template_dir, "player_profile.json")
        f_r_path = os.path.join(self.template_dir, "player_feature_right.png")
        f_l_path = os.path.join(self.template_dir, "player_feature_left.png")
        m_r_path = os.path.join(self.template_dir, "player_feature_mask_right.png")
        m_l_path = os.path.join(self.template_dir, "player_feature_mask_left.png")
        if os.path.exists(prof_path) and os.path.exists(f_r_path) and os.path.exists(f_l_path):
            try:
                with open(prof_path, "r", encoding="utf-8") as f:
                    self.two_stage_profile = json.load(f)

                img_fr = cv2.imread(f_r_path)
                img_fl = cv2.imread(f_l_path)
                if img_fr is not None and img_fl is not None:
                    g_fr = cv2.cvtColor(img_fr, cv2.COLOR_BGR2GRAY) if len(img_fr.shape) == 3 else img_fr
                    g_fl = cv2.cvtColor(img_fl, cv2.COLOR_BGR2GRAY) if len(img_fl.shape) == 3 else img_fl
                    mask_r = cv2.imread(m_r_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(m_r_path) else None
                    mask_l = cv2.imread(m_l_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(m_l_path) else None
                    if mask_r is not None and mask_r.shape != g_fr.shape:
                        mask_r = None
                    if mask_l is not None and mask_l.shape != g_fl.shape:
                        mask_l = None
                    if mask_r is not None:
                        mask_r = np.where(mask_r >= 128, 255, 0).astype(np.uint8)
                    if mask_l is not None:
                        mask_l = np.where(mask_l >= 128, 255, 0).astype(np.uint8)
                    self.two_stage_profile["feature_r_bgr"] = img_fr
                    self.two_stage_profile["feature_l_bgr"] = img_fl
                    self.two_stage_profile["feature_r_gray"] = g_fr
                    self.two_stage_profile["feature_l_gray"] = g_fl
                    self.two_stage_profile["feature_r_mask"] = mask_r
                    self.two_stage_profile["feature_l_mask"] = mask_l

                    # 提取色彩指纹 (HSV 主色调与饱和度分布)，用于严格色彩门控
                    fr_hsv = cv2.cvtColor(img_fr, cv2.COLOR_BGR2HSV) if len(img_fr.shape) == 3 else None
                    if fr_hsv is not None:
                        valid = mask_r > 0 if mask_r is not None else np.ones(fr_hsv.shape[:2], dtype=bool)
                        s_mean = float(np.mean(fr_hsv[:, :, 1][valid]))
                        h_mean = float(np.mean(fr_hsv[:, :, 0][valid]))
                        v_mean = float(np.mean(fr_hsv[:, :, 2][valid]))
                        bgr_mean = [float(np.mean(img_fr[:, :, c][valid])) for c in range(3)]
                        self.two_stage_profile["color_profile"] = {
                            "s_mean": s_mean,
                            "h_mean": h_mean,
                            "v_mean": v_mean,
                            "bgr_mean": bgr_mean,
                            "is_saturated": s_mean >= 60.0,
                        }

                    # 设为朝向模板与服饰模板
                    self.facing_templates_gray["right"] = g_fr
                    self.facing_templates_gray["left"] = g_fl
                    self.facing_template_masks["right"] = mask_r
                    self.facing_template_masks["left"] = mask_l
                    self.costume_template_bgr = img_fr
                    self.costume_template_gray = g_fr
                    self.costume_template_flipped_gray = g_fl
                    self.costume_template_mask = mask_r
                    self.costume_template_flipped_mask = mask_l
                    self.costume_size = (img_fr.shape[1], img_fr.shape[0])
                    char_w = self.two_stage_profile.get("char_w", 55)
                    char_h = self.two_stage_profile.get("char_h", 65)
                    self.player_bbox_size = (char_w, char_h)
                    self.costume_feet_offset_y = self.two_stage_profile.get("offset_from_feet_y", int(char_h * 0.7))
                    print(f"[MainViewDetector] 成功载入两段式标定特征! 全身: {char_w}x{char_h}, 特征: {img_fr.shape[1]}x{img_fr.shape[0]}")
            except Exception as e:
                print(f"[MainViewDetector] 加载两段式配置失败: {e}")

    def save_two_stage_calibration(
        self,
        whole_crop_bgr: np.ndarray,
        feature_crop_bgr: np.ndarray,
        feature_box_in_whole: Tuple[int, int, int, int],
        calibrated_facing: str = "right",
        feature_mask: Optional[np.ndarray] = None,
        feature_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """
        保存两段式角色标定（全身整体 + 核心特征 + 当前朝向）：
        1. whole_crop_bgr: 角色全身截图 (从头顶到脚底站立线)
        2. feature_crop_bgr: 核心特征截图 (如魔法帽、头部)
        3. feature_box_in_whole: (fx, fy, fw, fh) 特征在全身图内部的局部坐标
        4. calibrated_facing: 标定时角色朝向 ('left' 或 'right')
        """
        if whole_crop_bgr is None or feature_crop_bgr is None:
            return False

        char_h, char_w = whole_crop_bgr.shape[:2]
        fx, fy, fw, fh = feature_box_in_whole
        if feature_mask is not None:
            if feature_mask.shape[:2] != (fh, fw):
                return False
            feature_mask = np.where(feature_mask >= 128, 255, 0).astype(np.uint8)
            if int(np.count_nonzero(feature_mask)) < 80:
                return False
        else:
            # 兼容旧调用方：无mask即为完整矩形特征。
            feature_mask = np.full((fh, fw), 255, dtype=np.uint8)

        # 物理几何关系：脚底站立线在全身图最底部 (y = char_h)
        offset_from_feet_y = max(0, char_h - (fy + fh))
        feature_center_x = fx + fw / 2.0
        body_center_x = char_w / 2.0
        offset_from_center_x = feature_center_x - body_center_x

        if calibrated_facing == "left":
            feat_left = feature_crop_bgr
            feat_right = cv2.flip(feature_crop_bgr, 1)
            mask_left = feature_mask
            mask_right = cv2.flip(feature_mask, 1)
            norm_offset_x = -offset_from_center_x
        else:
            feat_right = feature_crop_bgr
            feat_left = cv2.flip(feature_crop_bgr, 1)
            mask_right = feature_mask
            mask_left = cv2.flip(feature_mask, 1)
            norm_offset_x = offset_from_center_x

        os.makedirs(self.template_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.template_dir, "player_body.png"), whole_crop_bgr)
        cv2.imwrite(os.path.join(self.template_dir, "player_feature_right.png"), feat_right)
        cv2.imwrite(os.path.join(self.template_dir, "player_feature_left.png"), feat_left)
        cv2.imwrite(os.path.join(self.template_dir, "player_feature_mask_right.png"), mask_right)
        cv2.imwrite(os.path.join(self.template_dir, "player_feature_mask_left.png"), mask_left)
        cv2.imwrite(os.path.join(self.template_dir, "player_hat.png"), feat_right)

        polygon_right = None
        polygon_left = None
        if feature_polygon:
            polygon_original = [
                [int(point[0]), int(point[1])] for point in feature_polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            polygon_flipped = [[int(fw - 1 - x), int(y)] for x, y in polygon_original]
            if calibrated_facing == "left":
                polygon_left, polygon_right = polygon_original, polygon_flipped
            else:
                polygon_right, polygon_left = polygon_original, polygon_flipped

        profile = {
            "char_w": int(char_w),
            "char_h": int(char_h),
            "feature_w": int(fw),
            "feature_h": int(fh),
            "offset_from_feet_y": int(offset_from_feet_y),
            "offset_from_center_x": float(norm_offset_x),
            "calibrated_facing": calibrated_facing,
            "feature_shape": "polygon" if feature_polygon else "rectangle",
            "feature_polygon_right": polygon_right,
            "feature_polygon_left": polygon_left,
            "feature_mask_pixels": int(np.count_nonzero(mask_right)),
            "created_at": time.time()
        }
        with open(os.path.join(self.template_dir, "player_profile.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        self._load_player_templates()
        print(f"[MainViewDetector] 成功保存两段式角色标定! 全身={char_w}x{char_h}, 特征={fw}x{fh}, 朝向={calibrated_facing}")
        return True

    @staticmethod
    def _auto_trim_nametag(crop_bgr: np.ndarray) -> np.ndarray:
        """
        自动检测并修剪名字牌截图周边的多余地图背景、树皮或人物鞋子残留。
        保持名字牌边框/底色条完整，消除多余背景造成的负相关。
        """
        if crop_bgr is None or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 30:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr

        # 1. 垂直方向修剪：寻找文字笔画和名牌框线所在的有效行
        row_vars = np.var(gray, axis=1)
        thresh_var = max(12.0, float(np.percentile(row_vars, 25)))
        valid_rows = np.where(row_vars >= thresh_var)[0]
        r_top, r_bot = 0, h
        if len(valid_rows) >= 12:
            r_top = max(0, int(valid_rows[0]) - 1)
            r_bot = min(h, int(valid_rows[-1]) + 2)

        sub = crop_bgr[r_top:r_bot, :]
        sub_gray = gray[r_top:r_bot, :]

        # 2. 水平方向修剪：去除左右边缘多余的树叶/背景
        col_vars = np.var(sub_gray, axis=0)
        thresh_col = max(8.0, float(np.percentile(col_vars, 15)))
        valid_cols = np.where(col_vars >= thresh_col)[0]
        c_left, c_right = 0, w
        if len(valid_cols) >= 20:
            c_left = max(0, int(valid_cols[0]) - 1)
            c_right = min(w, int(valid_cols[-1]) + 2)

        trimmed = sub[:, c_left:c_right]
        if trimmed.shape[0] >= 14 and trimmed.shape[1] >= 24:
            return trimmed
        return crop_bgr

    def set_manual_cropped_player(self, crop_bgr: np.ndarray) -> bool:
        """
        手动框选标定角色（支持任意名字、时装名牌、全身框）：
        保存用户框选的区域作为当前玩家的核心特征，并自动完成背景修剪、多部件解耦与热重载
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        # 自动智能去除边缘地图杂色背景
        crop_bgr = self._auto_trim_nametag(crop_bgr)
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 15:
            return False

        # 清除旧的单体勋章避免尺寸冲突
        fpath_medal = os.path.join(self.template_dir, "player_medal.png")
        if os.path.exists(fpath_medal):
            try:
                os.remove(fpath_medal)
            except Exception:
                pass

        fpath = os.path.join(self.template_dir, "player_nametag.png")
        cv2.imwrite(fpath, crop_bgr)
        self._load_player_templates()
        print(f"[MainViewDetector] 成功手动标定角色! 自动修剪后尺寸: {w}x{h}")
        return True

    def set_manual_cropped_hat(self, crop_bgr: np.ndarray) -> bool:
        """
        手动框选标定角色服饰/魔法帽特征 (双重保险)：
        保存用户框选的帽子或服饰区域，自动生成左右镜像以支持转向，并热重载
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return False

        fpath = os.path.join(self.template_dir, "player_hat.png")
        cv2.imwrite(fpath, crop_bgr)
        self._load_player_templates()
        print(f"[MainViewDetector] 成功手动标定服饰/魔法帽特征! 尺寸: {w}x{h}")
        return True

    def clear_costume_template(self) -> bool:
        """清除已标定的服饰/魔法帽特征"""
        for fname in ["player_hat.png", "player_costume.png"]:
            fpath = os.path.join(self.template_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
        self._load_player_templates()
        print("[MainViewDetector] 已清除服饰特征。")
        return True



    def _load_templates_by_keywords(self, keywords: List[str]):
        """
        加载指定怪物的原生高精度模板 (支持 RGBA 掩码与 BGR 彩色特征，配合 HSV 颜色过滤器)
        """
        new_dict = {}
        expanded = set()
        for kw in keywords:
            kw_clean = str(kw).lower()
            expanded.add(kw_clean)
            for k, aliases in MOB_ALIAS_MAP.items():
                if k in kw_clean or kw_clean in k:
                    expanded.update(aliases)

        for f in glob.glob(os.path.join(self.template_dir, "*.png")):
            base = os.path.basename(f).lower()
            if any(k in base for k in ["player", "nametag", "medal", "facing", "die", "dead"]):
                continue  # 严格过滤角色部件与死亡动作，仅保留活着怪物的动作切片

            if any(k in base for k in expanded):
                img_rgba = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img_rgba is not None:
                    # 检查是否有透明 Alpha 通道
                    if len(img_rgba.shape) == 3 and img_rgba.shape[2] == 4:
                        bgr = img_rgba[:, :, :3]
                        alpha = img_rgba[:, :, 3]
                        _, mask = cv2.threshold(alpha, 60, 255, cv2.THRESH_BINARY)
                    else:
                        bgr = img_rgba[:, :, :3] if len(img_rgba.shape) == 3 else cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2BGR)
                        mask = None

                    tag = os.path.splitext(os.path.basename(f))[0]
                    # 记录模板 (BGR, AlphaMask, 原始宽度, 原始高度)
                    new_dict[tag] = (bgr, mask, bgr.shape[1], bgr.shape[0])

        self.monster_templates_scaled_gray = new_dict
        self._update_cached_scaled_templates()
        print(f"[MainViewDetector] 载入了本图 {len(self.monster_templates_scaled_gray)} 个高精度怪物特征模板 (特征词: {expanded})")

    def _update_cached_scaled_templates(self):
        """预先计算并缓存 0.30x 降采样的灰度怪物动作模板，彻底消除运行期 CPU 缩放与色彩转换开销"""
        cached = []
        pyramid_scale = 0.30
        for mname, (tpl_bgr, mask, orig_w, orig_h) in self.monster_templates_scaled_gray.items():
            tpl_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
            scaled_tpl = cv2.resize(tpl_gray, (0, 0), fx=pyramid_scale, fy=pyramid_scale, interpolation=cv2.INTER_AREA)
            cached.append((mname, scaled_tpl, orig_w, orig_h))
        self.monster_cached_scaled_templates = cached




    def load_dynamic_monster_templates(self, synced_mobs: List[Dict]):
        """
        按当前地图 Mob ID 热加载多尺度怪物模板。

        新目录约定：templates/<MobID>/mob_global*.png（全身/掩膜）与
        templates/<MobID>/mob*.png（局部特征）。不再混用旧平铺模板算法。
        """
        with self._monster_backend_lock:
            self.multi_scale_backend.threshold = float(self.monster_threshold)
            loaded_ids = self.multi_scale_backend.load_for_mobs(synced_mobs or [])
            self.monster_templates_scaled_gray = {}
            self.monster_cached_scaled_templates = list(self.multi_scale_backend.template_items)
            self._monster_template_generation += 1
        # 切图、模板重载时绝不能带入上一张图的索敌对象。
        with self._monster_track_lock:
            self.tracked_monsters = []
            self._monster_frame_index = 0
            self._last_async_batch_sequence = -1
            self._track_flow_prev_gray = None

        if loaded_ids:
            print(
                f"[MainViewDetector] 多尺度识别载入本图 "
                f"{len(self.monster_cached_scaled_templates)} 个特征模板 "
                f"(Mob目录: {', '.join(loaded_ids)})"
            )
        elif not synced_mobs:
            print("[MainViewDetector] 本图为安全区/城镇 (无野怪)，已清空多尺度模板。")
        else:
            wanted = ", ".join(str(m.get("id", "?")) for m in synced_mobs)
            print(
                f"[MainViewDetector] 未找到本图多尺度模板："
                f"{self.multi_scale_template_root}\\<MobID> (需要: {wanted})"
            )

    def set_monster_template_scale(self, scale: float) -> None:
        """更新怪物模板匹配比例；重新加载当前模板后立即生效。"""
        value = float(scale)
        if not 0.1 <= value <= 3.0:
            raise ValueError("怪物模板比例必须在 0.1~3.0 之间")
        self.monster_template_scale = value
        with self._monster_backend_lock:
            self.multi_scale_backend.template_scale = value

    def clear_monster_tracks(self) -> None:
        """线程安全地清空视觉轨迹；地图切换和关闭识别时共用。"""
        with self._monster_track_lock:
            self.tracked_monsters = []
            self._track_flow_prev_gray = None
            self._last_async_batch_sequence = -1
            self._locked_target_id = None
            self._locked_target_until = 0.0

    def reset_player_tracking(self, clear_monsters: bool = True) -> None:
        """丢弃旧地图/遮挡阶段的角色位置与速度，不保留1秒坐标兜底。"""
        self.last_player_pos = None
        self.last_player_bbox = None
        self.last_feature_bbox = None
        self.last_player_time = 0.0
        self.last_player_observed_time = 0.0
        self.player_vx = 0.0
        self.player_vy = 0.0
        self._facing_pending_direction = None
        self._facing_pending_count = 0
        self.facing_lock_until = 0.0
        if clear_monsters:
            self.clear_monster_tracks()

    def set_monster_hp_bar_compute_device(self, device: str) -> None:
        """热切换怪物血条后端；CPU实现始终保留为参考与回退。"""
        requested = str(device or "auto").lower()
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError("monster HP-bar device must be auto, cpu or cuda")
        if requested == self.monster_hp_bar_compute_device:
            return
        self.monster_hp_bar_compute_device = requested
        with self._cuda_hp_bar_init_lock:
            self._cuda_hp_bar_generation += 1
            self._cuda_hp_bar_detector = None
            self._cuda_hp_bar_init_attempted = False
            self._cuda_hp_bar_error = ""

    def _get_cuda_hp_bar_detector(self) -> Optional[CudaHpBarDetector]:
        if self.monster_hp_bar_compute_device == "cpu":
            return None
        if self._cuda_hp_bar_detector is not None:
            return self._cuda_hp_bar_detector
        if self._cuda_hp_bar_init_attempted:
            return None
        with self._cuda_hp_bar_init_lock:
            if self._cuda_hp_bar_detector is not None:
                return self._cuda_hp_bar_detector
            if self._cuda_hp_bar_init_attempted:
                return None
            self._cuda_hp_bar_init_attempted = True
            generation = self._cuda_hp_bar_generation

        # 首次导入PyTorch/建立CUDA上下文可能耗时数秒。不能把这段黑延时
        # 放进55ms完整重检线程；初始化期间直接执行保留的CPU算法，后续帧
        # 检测器就绪后自然切到CUDA。
        def initialize() -> None:
            try:
                detector = CudaHpBarDetector()
                # CUDA上下文建立后，第一次池化/积分核仍可能有约200~300ms
                # 的惰性编译与显存分配。也放在后台完成；发布 detector 前
                # 先用标准720p空帧热身，实战首帧继续走CPU而不会卡住重检。
                warmup = np.zeros((720, 1280, 3), dtype=np.uint8)
                wx, wy = 100, 100
                warmup[wy, wx:wx + 50] = 255
                warmup[wy + 7, wx:wx + 50] = 255
                warmup[wy:wy + 8, wx] = 255
                warmup[wy:wy + 8, wx + 49] = 255
                warmup[wy + 2:wy + 5, wx + 2:wx + 48] = (0, 255, 0)
                detector.detect(warmup)
                detector.detect(warmup)
                detector.detect(np.zeros_like(warmup))
            except Exception as exc:
                with self._cuda_hp_bar_init_lock:
                    if generation == self._cuda_hp_bar_generation:
                        self._cuda_hp_bar_error = str(exc)
                print(
                    f"[MainViewDetector] 怪物血条CUDA不可用，回退CPU: {exc}",
                    flush=True,
                )
                return
            with self._cuda_hp_bar_init_lock:
                if (
                    generation != self._cuda_hp_bar_generation
                    or self.monster_hp_bar_compute_device == "cpu"
                ):
                    return
                self._cuda_hp_bar_detector = detector
            print(
                f"[MainViewDetector] 怪物血条后端: CUDA ({detector.device_name})",
                flush=True,
            )

        threading.Thread(
            target=initialize,
            daemon=True,
            name="hp-bar-cuda-init",
        ).start()
        return None

    def detect_monster_hp_bars(
        self,
        bgr_frame: np.ndarray,
        exclusion_regions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """按配置使用CUDA，任何异常均在同一帧回退既有CPU算法。"""
        started_at = time.perf_counter()
        cuda_detector = self._get_cuda_hp_bar_detector()
        if cuda_detector is not None:
            try:
                bars = cuda_detector.detect(
                    bgr_frame,
                    self.exclusion_regions
                    if exclusion_regions is None else exclusion_regions,
                )
                self.last_hp_bar_backend = "cuda"
                self.last_hp_bar_cost_ms = (
                    time.perf_counter() - started_at
                ) * 1000.0
                return bars
            except Exception as exc:
                # 一次运行期CUDA异常后不再每帧重试/刷日志；CPU路径仍保证
                # 识别不中断，用户也可通过热切换设备重新初始化。
                self._cuda_hp_bar_error = str(exc)
                self._cuda_hp_bar_detector = None
                self._cuda_hp_bar_init_attempted = True
                print(
                    f"[MainViewDetector] 怪物血条CUDA运行失败，已回退CPU: {exc}",
                    flush=True,
                )
        bars = self.detect_monster_hp_bars_cpu(bgr_frame, exclusion_regions)
        self.last_hp_bar_backend = "cpu"
        self.last_hp_bar_cost_ms = (time.perf_counter() - started_at) * 1000.0
        return bars

    def detect_monster_hp_bars_cpu(
        self,
        bgr_frame: np.ndarray,
        exclusion_regions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        极速向量化怪物血条检测器：
        支持完整血条 (50x8 白框) 与半遮挡血条 (被树木/绳索/伤害数字遮挡 20%~60%)。
        标准尺寸：外层 52x10 像素，包含 50x8 纯白外框与 1px 外黑边，
        内部包含高饱和绿色 (0, 255, 0) / 暗红色 (170, 0, 0) / 扣血纯黑 (0, 0, 0)。
        全流程采用 OpenCV 形态学向量运算，单帧耗时稳定在 1~5ms，杜绝 CPU 占满与掉帧。
        """
        if bgr_frame is None or bgr_frame.size == 0 or len(bgr_frame.shape) != 3:
            return []

        fh, fw = bgr_frame.shape[:2]
        if fh < 20 or fw < 60:
            return []

        # 1. 极速提取纯白外框像素掩膜
        white_mask = cv2.inRange(bgr_frame, (230, 230, 230), (255, 255, 255))

        # 排除游戏底部状态栏 (通常高度 >= 300px 的游戏窗口底部 45px 均为菜单/血条/经验条)
        if fh >= 300:
            white_mask[max(0, fh - 45):fh, :] = 0

        # 排除用户自定义屏蔽区 (小地图、聊天栏等)
        exclusions = exclusion_regions if exclusion_regions is not None else self.exclusion_regions
        if exclusions:
            for ex in exclusions:
                try:
                    if not ex.get("monster", True):
                        continue
                    ex_x = max(0, int(ex.get("x", 0)))
                    ex_y = max(0, int(ex.get("y", 0)))
                    ex_w = int(ex.get("w", 0))
                    ex_h = int(ex.get("h", 0))
                    if ex_w > 0 and ex_h > 0:
                        white_mask[ex_y:min(fh, ex_y + ex_h), ex_x:min(fw, ex_x + ex_w)] = 0
                except Exception:
                    pass

        # 快速短路：全屏白色像素不足一个完整血条外框 (约 116px) 时直接返回
        if cv2.countNonZero(white_mask) < 90:
            return []

        # 2. 向量化形态学滤波提取连续线段 (14px 水平白线与 8px 垂直白线)
        kernel_h14 = np.ones((1, 14), dtype=np.uint8)
        h14_lines = cv2.erode(white_mask, kernel_h14, anchor=(0, 0))
        # 一条恰好14px的白线腐蚀后只剩1个锚点，不是14个像素。
        # 顶/底各有一个锚点就足以进入半遮挡验证；旧的<14早退会漏掉
        # 正中间被伤害数字遮住、左右各剩约15px的真实血条。
        if cv2.countNonZero(h14_lines) < 2:
            return []

        kernel_v8 = np.ones((8, 1), dtype=np.uint8)
        v8_lines = cv2.erode(white_mask, kernel_v8, anchor=(0, 0))

        kernel_h37 = np.ones((1, 37), dtype=np.uint8)  # 14 + 37 - 1 = 50px 完整横线
        h50_lines = cv2.erode(h14_lines, kernel_h37, anchor=(0, 0))

        candidates = []
        found_full_boxes = []

        # 3. 第一阶段：向量化四边框几何锚点提取 (50x8 矩形框)
        # 上横线 + 下横线 + 左竖线 + 右竖线全部闭合，且框中心必须非纯白实心
        full_box_anchors = np.zeros_like(h50_lines)
        full_box_anchors[:-7, :-49] = (
            (h50_lines[:-7, :-49] > 0) &
            (h50_lines[7:, :-49] > 0) &
            (v8_lines[:-7, :-49] > 0) &
            (v8_lines[:-7, 49:] > 0) &
            (white_mask[3:-4, 25:-24] == 0)
        )
        y_full, x_full = np.where(full_box_anchors > 0)

        for y, x in zip(y_full, x_full):
            # 外围 1px 黑色边框
            if y - 1 >= 0 and np.mean(bgr_frame[y - 1, x:x+50]) > 65:
                continue
            if y + 8 < fh and np.mean(bgr_frame[y + 8, x:x+50]) > 65:
                continue

            inner_roi = bgr_frame[y+2:y+5, x+2:x+48]
            in_b, in_g, in_r = inner_roi[:, :, 0], inner_roi[:, :, 1], inner_roi[:, :, 2]
            green_px = np.sum((in_g > 160) & (in_r < 90) & (in_b < 90))
            red_px = np.sum((in_r > 130) & (in_g < 90) & (in_b < 90))
            black_px = np.sum((in_r < 50) & (in_g < 50) & (in_b < 50))
            tot = inner_roi.shape[0] * inner_roi.shape[1]
            if (green_px + red_px + black_px) / float(tot) >= 0.70:
                hx = max(0, int(x - 1))
                hy = max(0, int(y - 1))
                candidates.append((hx, hy, 52, 10))
                found_full_boxes.append((x, y, 50, 8))

        # 4. 第二阶段：向量化半遮挡血条检测 (左锚点 / 右锚点)
        # 左锚点：左竖边 + 顶部/底部水平白线 >= 14px，框内非实心白
        left_anchors = np.zeros_like(h14_lines)
        left_anchors[:-7, :-13] = (
            (h14_lines[:-7, :-13] > 0) &
            (h14_lines[7:, :-13] > 0) &
            (v8_lines[:-7, :-13] > 0) &
            (white_mask[3:-4, 7:-6] == 0)
        )
        if found_full_boxes:
            left_anchors[:-7, :-49] &= ~full_box_anchors[:-7, :-49]
        y_left, x_left = np.where(left_anchors > 0)

        for y, x in zip(y_left, x_left):
            if any(abs(y - fy) <= 3 and abs(x - fx) <= 15 for fx, fy, _, _ in found_full_boxes):
                continue
            rx = x + 13
            while rx + 1 < fw and white_mask[y, rx + 1] > 0 and white_mask[y + 7, rx + 1] > 0:
                rx += 1
            inferred_full_x = x
            if any(abs(y - fy) <= 3 and abs(inferred_full_x - fx) <= 15 for fx, fy, _, _ in found_full_boxes):
                continue
            if 0 <= inferred_full_x and inferred_full_x + 49 < fw:
                vis_x1 = max(x, inferred_full_x + 2)
                vis_x2 = min(rx, inferred_full_x + 47)
                if vis_x2 - vis_x1 >= 10:
                    inner_roi = bgr_frame[y+2:y+5, vis_x1:vis_x2+1]
                    in_b, in_g, in_r = inner_roi[:, :, 0], inner_roi[:, :, 1], inner_roi[:, :, 2]
                    green_px = np.sum((in_g > 160) & (in_r < 90) & (in_b < 90))
                    red_px = np.sum((in_r > 130) & (in_g < 90) & (in_b < 90))
                    black_px = np.sum((in_r < 50) & (in_g < 50) & (in_b < 50))
                    tot = inner_roi.shape[0] * inner_roi.shape[1]
                    if (green_px + red_px + black_px) / float(tot) >= 0.70:
                        hx = max(0, int(inferred_full_x - 1))
                        hy = max(0, int(y - 1))
                        candidates.append((hx, hy, 52, 10))

        # 右锚点：右竖边 + 顶部/底部水平白线 >= 14px
        right_anchors = np.zeros_like(h14_lines)
        right_anchors[:-7, 13:] = (
            (h14_lines[:-7, :-13] > 0) &
            (h14_lines[7:, :-13] > 0) &
            (v8_lines[:-7, 13:] > 0) &
            (white_mask[3:-4, 7:-6] == 0)
        )
        y_right, x_right = np.where(right_anchors > 0)

        for y, x in zip(y_right, x_right):
            rx = x
            if any(abs(y - fy) <= 3 and abs(rx - (fx + 49)) <= 15 for fx, fy, _, _ in found_full_boxes):
                continue
            lx = x - 13
            while lx - 1 >= 0 and white_mask[y, lx - 1] > 0 and white_mask[y + 7, lx - 1] > 0:
                lx -= 1
            inferred_full_x = rx - 49
            if any(abs(y - fy) <= 3 and abs(inferred_full_x - fx) <= 15 for fx, fy, _, _ in found_full_boxes):
                continue
            if 0 <= inferred_full_x and inferred_full_x + 49 < fw:
                vis_x1 = max(lx, inferred_full_x + 2)
                vis_x2 = min(rx, inferred_full_x + 47)
                if vis_x2 - vis_x1 >= 10:
                    inner_roi = bgr_frame[y+2:y+5, vis_x1:vis_x2+1]
                    in_b, in_g, in_r = inner_roi[:, :, 0], inner_roi[:, :, 1], inner_roi[:, :, 2]
                    green_px = np.sum((in_g > 160) & (in_r < 90) & (in_b < 90))
                    red_px = np.sum((in_r > 130) & (in_g < 90) & (in_b < 90))
                    black_px = np.sum((in_r < 50) & (in_g < 50) & (in_b < 50))
                    tot = inner_roi.shape[0] * inner_roi.shape[1]
                    if (green_px + red_px + black_px) / float(tot) >= 0.70:
                        hx = max(0, int(inferred_full_x - 1))
                        hy = max(0, int(y - 1))
                        candidates.append((hx, hy, 52, 10))

        # 5. 去重 (合并距离在 15px 以内的重复候选)
        final_bars: List[Tuple[int, int, int, int]] = []
        for cb in candidates:
            cx, cy, cw, ch = cb
            if not any(abs(cx - fx) <= 15 and abs(cy - fy) <= 3 for fx, fy, _, _ in final_bars):
                final_bars.append(cb)

        return final_bars

    def _detect_monsters_from_hp_bars(
        self,
        bgr_frame: np.ndarray,
        existing_candidates: List[Tuple[int, int, int, int, str, float]],
    ) -> List[Tuple[int, int, int, int, str, float]]:
        """
        血条补充检测与反推：
        - 若怪物本体已被检出，以怪物本体位置为主（忽略对应血条，不生成多余反推框）；
        - 若仅有血条被检出（本体受遮挡），通过血条反推出怪物识别框 (x, y, w, h, name, score)。
        """
        hp_bars = self.detect_monster_hp_bars(bgr_frame)
        if not hp_bars:
            return []

        # 确定反推怪物的默认尺寸与名称
        def_w, def_h = 86, 35
        def_name = "mob_hp"
        try:
            if self.multi_scale_backend and self.multi_scale_backend.template_items:
                primary_tpl = self.multi_scale_backend.template_items[0]
                scale = float(getattr(self.multi_scale_backend, "template_scale", 1.0))
                def_w = max(20, int(round(primary_tpl.orig_w * scale)))
                def_h = max(20, int(round(primary_tpl.orig_h * scale)))
                species = {mob_id_of(tpl) for tpl in self.multi_scale_backend.template_items}
                species.discard(None)
                if len(species) == 1:
                    def_name = primary_tpl.name
        except Exception:
            pass

        inferred_candidates: List[Tuple[int, int, int, int, str, float]] = []

        for hx, hy, hw, hh in hp_bars:
            hcx = hx + hw / 2.0
            hp_bottom = hy + hh

            # 检查是否有本体候选框与该血条空间匹配
            # 规则：水平中心差距 <= 45px，且本体顶部在血条下沿 0~60px 之间
            has_body_match = False
            for bx, by, bw, bh, bname, bscore in existing_candidates:
                bcx = bx + bw / 2.0
                if abs(bcx - hcx) <= 45.0 and 0 <= (by - hp_bottom) <= 60.0:
                    has_body_match = True
                    break

            if has_body_match:
                # 怪物本体已检出 -> 以怪物位置为主，消耗该血条，不生成重复框
                continue

            # 仅检出血条（本体遮挡） -> 反推出怪物识别框
            # 几何规律：怪物顶部在血条下沿下方 20px，水平中心对齐
            inf_top = hp_bottom + 20
            inf_x = int(round(hcx - def_w / 2.0))
            inf_y = int(round(inf_top))

            # 越界防守
            fh, fw = bgr_frame.shape[:2]
            inf_x = max(0, min(fw - def_w, inf_x))
            inf_y = max(0, min(fh - def_h, inf_y))

            inf_bbox = (inf_x, inf_y, def_w, def_h)
            if not self._bbox_excluded(inf_bbox, "monster"):
                inferred_candidates.append(
                    (inf_x, inf_y, def_w, def_h, def_name, 0.85)
                )

        return inferred_candidates

    def run_full_monster_detection(
        self,
        bgr_frame: np.ndarray,
        *,
        frame_seq: int = 0,
        captured_at: Optional[float] = None,
    ) -> MonsterDetectionBatch:
        """在独立线程对一张最新帧做完整模板扫描，不接触实时轨迹。"""
        started_at = time.perf_counter()
        captured_ts = started_at if captured_at is None else float(captured_at)
        detections: List[Tuple[int, int, int, int, str, float]] = []
        template_match_sec = 0.0
        hp_bar_sec = 0.0
        candidate_filter_sec = 0.0
        with self._monster_backend_lock:
            lock_acquired_at = time.perf_counter()
            generation = self._monster_template_generation
            if bgr_frame is not None and bgr_frame.size > 0:
                if self.enable_monster_detection and self.multi_scale_backend.available:
                    stage_started_at = time.perf_counter()
                    raw_candidates = self.multi_scale_backend.detect(
                        bgr_frame, self.monster_threshold, self.exclusion_regions
                    )
                    template_match_sec = time.perf_counter() - stage_started_at
                    stage_started_at = time.perf_counter()
                    detections = [
                        item for item in raw_candidates
                        if not self._bbox_excluded(item[:4], "monster")
                    ]
                    candidate_filter_sec = time.perf_counter() - stage_started_at
                if self.enable_monster_hp_bar_detection and len(bgr_frame.shape) == 3:
                    stage_started_at = time.perf_counter()
                    hp_inferred = self._detect_monsters_from_hp_bars(
                        bgr_frame, existing_candidates=detections
                    )
                    if hp_inferred:
                        detections.extend(hp_inferred)
                    hp_bar_sec = time.perf_counter() - stage_started_at
                else:
                    self.last_hp_bar_backend = "off"
                    self.last_hp_bar_cost_ms = 0.0
        completed_at = time.perf_counter()
        return MonsterDetectionBatch(
            frame_seq=int(frame_seq),
            captured_at=captured_ts,
            completed_at=completed_at,
            template_generation=generation,
            detections=tuple(detections),
            duration_sec=max(0.0, completed_at - started_at),
            backend_lock_wait_sec=max(0.0, lock_acquired_at - started_at),
            template_match_sec=template_match_sec,
            hp_bar_sec=hp_bar_sec,
            candidate_filter_sec=candidate_filter_sec,
        )

    def track_monsters_from_async_batch(
        self,
        gray_frame: np.ndarray,
        player_center: Optional[Tuple[int, int]],
        batch: Optional[MonsterDetectionBatch],
    ) -> List[MonsterTarget]:
        """60Hz轻量线程：光流推进并至多消费一次完整模板扫描结果。"""
        fh, fw = gray_frame.shape[:2]
        pcx, pcy = player_center if player_center else (fw // 2, fh // 2)
        with self._monster_track_lock:
            flow_propagated = self._propagate_tracks_with_optical_flow(gray_frame)
            detections: List[Tuple[int, int, int, int, str, float]] = []
            observation_ts: Optional[float] = None
            full_scan_completed = False
            if (
                batch is not None
                and batch.template_generation == self._monster_template_generation
                and batch.frame_seq > self._last_async_batch_sequence
            ):
                self._last_async_batch_sequence = batch.frame_seq
                detections = list(batch.detections)
                full_scan_completed = True
                # 新鲜度从完整扫描完成时开始计算；这是结果首次可供战斗
                # 使用的时刻，避免把计算耗时重复算成“识别后延迟”。
                observation_ts = batch.completed_at
            return self._update_tracker(
                detections,
                pcx,
                pcy,
                predict_tracks=not flow_propagated,
                observation_ts=observation_ts,
                full_scan_completed=full_scan_completed,
            )

    def set_player_feature_threshold(self, threshold: float) -> None:
        """更新人物识别基础准入门限（服饰核验、兜底救援、特征框激活线等同步绑定此门限）。"""
        val = float(threshold)
        if not 0.1 <= val <= 1.0:
            raise ValueError("人物基础准入门限必须在 0.1~1.0 之间")
        self.player_feature_threshold = val

    def set_attack_observation_hard_timeout_ms(
        self, timeout_ms: float, enabled: bool = True
    ) -> None:
        """实时更新红框/自动攻击共用的真实命中硬上限。"""
        timeout_sec = max(0.0, float(timeout_ms) / 1000.0)
        self.attack_observation_hard_timeout_sec = timeout_sec
        self.attack_observation_hard_timeout_enabled = bool(enabled)
        with self._monster_track_lock:
            for track in self.tracked_monsters:
                track.attack_observation_hard_timeout_sec = timeout_sec
                track.attack_observation_hard_timeout_enabled = bool(enabled)

    def set_ghost_box_safety_timeout_ms(
        self, timeout_ms: float, enabled: bool = True
    ) -> None:
        """实时更新光流防幽灵框安全上限。"""
        timeout_sec = max(0.0, float(timeout_ms) / 1000.0)
        self.ghost_box_safety_timeout_sec = timeout_sec
        self.ghost_box_safety_timeout_enabled = bool(enabled)

    def set_exclusion_regions(self, regions: Optional[List[Dict[str, Any]]]) -> None:
        """设置不参与怪物/人物识别的原始捕获帧区域。"""
        clean = []
        for item in regions or []:
            try:
                x, y, w, h = (int(round(float(item[k]))) for k in ("x", "y", "w", "h"))
                if w <= 0 or h <= 0:
                    continue
                clean.append({"x": x, "y": y, "w": w, "h": h,
                              "monster": bool(item.get("monster", True)),
                              "player": bool(item.get("player", True))})
            except Exception:
                continue
        self.exclusion_regions = clean

    def _point_excluded(self, x: float, y: float, kind: str) -> bool:
        return any(
            r.get(kind, False) and r["x"] <= x <= r["x"] + r["w"] and
            r["y"] <= y <= r["y"] + r["h"] for r in self.exclusion_regions
        )

    def _bbox_excluded(self, bbox: Tuple[int, int, int, int], kind: str) -> bool:
        x, y, w, h = bbox
        return any(
            r.get(kind, False) and x < r["x"] + r["w"] and x + w > r["x"] and
            y < r["y"] + r["h"] and y + h > r["y"] for r in self.exclusion_regions
        )

    @staticmethod
    def _match_player_feature(
        search_gray: np.ndarray,
        template_gray: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """人物局部特征匹配；新标定仅计算多边形mask内部像素。"""
        if mask is not None and mask.shape == template_gray.shape:
            result = cv2.matchTemplate(
                search_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
                mask=mask,
            )
        else:
            result = cv2.matchTemplate(
                search_gray, template_gray, cv2.TM_CCOEFF_NORMED
            )
        # 全色块候选可能令归一化分母为0；绝不能让NaN进入峰值比较。
        return np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)


    def _detect_facing_direction(
        self,
        gray_frame: np.ndarray,
        player_center: Tuple[int, int],
        update_direction: bool = True,
    ) -> Tuple[str, float, bool]:
        """Match the facing feature and optionally apply it to direction state.

        Climbing still needs current-frame feature matching for player
        verification/debug rendering, but a rope animation must not decide the
        left/right attack direction.  ``update_direction=False`` preserves the
        reliable ground direction while still refreshing the blue feature box.
        """
        now = time.perf_counter()
        if update_direction and now < self.facing_lock_until:
            return self.current_facing, 1.0, True

        tpl_r = self.facing_templates_gray.get("right")
        tpl_l = self.facing_templates_gray.get("left")
        mask_r = self.facing_template_masks.get("right")
        mask_l = self.facing_template_masks.get("left")

        if tpl_r is not None and tpl_l is not None and gray_frame is not None:
            pfx, pfy = player_center
            fh, fw = gray_frame.shape
            tw, th = tpl_r.shape[1], tpl_r.shape[0]

            # 搜索区域：优先在标定的特征相对脚底高度与中心附近进行高精度定位
            if self.two_stage_profile and "offset_from_feet_y" in self.two_stage_profile:
                off_feet_y = self.two_stage_profile["offset_from_feet_y"]
                expected_top_y = pfy - th - off_feet_y
                margin_y = max(10, int(th * 0.5))
                y1 = max(0, expected_top_y - margin_y)
                y2 = min(fh, expected_top_y + th + margin_y)
                # 水平范围：限定在身体中心左右 (tw//2 + 10) 像素内，坚决杜绝背景杂色峰值
                margin_x = max(12, int(tw * 0.35))
                x1 = max(0, pfx - tw // 2 - margin_x)
                x2 = min(fw, pfx + tw // 2 + margin_x)
            else:
                bh = getattr(self, "player_bbox_size", (55, 65))[1]
                y1 = max(0, pfy - bh - 10)
                y2 = min(fh, pfy - 5)
                x1 = max(0, pfx - max(30, tw // 2 + 10))
                x2 = min(fw, pfx + max(30, tw // 2 + 10))

            char_crop = gray_frame[y1:y2, x1:x2]
            if char_crop.shape[0] >= th and char_crop.shape[1] >= tw:
                res_r = self._match_player_feature(char_crop, tpl_r, mask_r)
                _, score_r, _, loc_r = cv2.minMaxLoc(res_r)
                res_l = self._match_player_feature(char_crop, tpl_l, mask_l)
                _, score_l, _, loc_l = cv2.minMaxLoc(res_l)

                diff = score_r - score_l
                best_s = max(score_r, score_l)
                best_loc = loc_r if score_r >= score_l else loc_l
                self.last_facing_scores = (float(score_r), float(score_l))

                if best_s >= FACING_FEATURE_VISIBLE_SCORE:
                    self.last_feature_bbox = (
                        x1 + best_loc[0], y1 + best_loc[1], tw, th
                    )

                candidate = "right" if diff > 0.0 else "left"
                margin = abs(diff)

                if not update_direction:
                    # Feature visibility is already recorded above.  Do not
                    # mutate current_facing or its pending debounce state.
                    return self.current_facing, best_s, False

                # 挥刀/受击帧的典型特征是两边都只有约 0.35~0.55，旧逻辑
                # 仅凭 0.005 的分差就会翻向。低分或近似平局一律维持最近
                # 一次可靠朝向，不累计候选。
                if (
                    best_s < FACING_SWITCH_MIN_SCORE
                    or margin < FACING_SWITCH_MIN_MARGIN
                ):
                    self._facing_pending_direction = None
                    self._facing_pending_count = 0
                    retained_score = (
                        score_r if self.current_facing == "right" else score_l
                    )
                    return self.current_facing, retained_score, False

                if candidate == self.current_facing:
                    self._facing_pending_direction = None
                    self._facing_pending_count = 0
                    return self.current_facing, best_s, False

                # 清晰转身帧立即生效；中等质量证据需连续两帧一致，既不
                # 拖慢正常转向，也不会被单帧技能光效翻转。
                if (
                    best_s >= FACING_SWITCH_IMMEDIATE_SCORE
                    and margin >= FACING_SWITCH_IMMEDIATE_MARGIN
                ):
                    self.current_facing = candidate
                    self._facing_pending_direction = None
                    self._facing_pending_count = 0
                else:
                    if self._facing_pending_direction == candidate:
                        self._facing_pending_count += 1
                    else:
                        self._facing_pending_direction = candidate
                        self._facing_pending_count = 1
                    if self._facing_pending_count >= 2:
                        self.current_facing = candidate
                        self._facing_pending_direction = None
                        self._facing_pending_count = 0

                current_score = (
                    score_r if self.current_facing == "right" else score_l
                )
                return self.current_facing, current_score, False

        # Feature-only callers must not receive a synthetic direction
        # confidence when no feature crop/template was actually matchable.
        return self.current_facing, (0.8 if update_direction else 0.0), False

    def detect_player(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
        """
        工业级抗遮挡多部件空间投票 + 色彩门控定位引擎 (统一标准脚底 Foothold 锚点):
          1. 采用名字牌/角色身体多部件空间投票
          2. 最终输出的 player_pos (pfx, pfy) 严格对齐角色站立脚底线 (Foothold Contact Line)
          3. player_bbox 底部贴合脚底，向上紧密包裹角色全身
        """
        # A feature box is evidence from exactly one frame.  Never let a box
        # found on the ground survive after the character climbs onto a rope
        # where the feature becomes occluded.
        self.last_feature_bbox = None
        if frame is None or frame.size == 0 or (not getattr(self, "player_parts", None) and not getattr(self, "two_stage_profile", None)):
            return False, None, None

        fh, fw = frame.shape[:2]
        bgr_frame = frame if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        gray_frame = None  # 延迟按需加载，局部追踪时不执行昂贵的全屏转灰度，耗时仅 0.01ms

        def _check_color_profile(patch_bgr, tpl_bgr):
            if not self.nametag_color_profile or patch_bgr.size == 0 or patch_bgr.shape != tpl_bgr.shape:
                return True
            prof = self.nametag_color_profile
            patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
            if prof.get("is_adaptive"):
                hl, hh = prof["h_range"]
                sl, sh = prof["s_range"]
                vl, vh = prof["v_range"]
                hits = np.count_nonzero(cv2.inRange(patch_hsv, np.array([hl, sl, vl], dtype=np.uint8),
                                                              np.array([hh, sh, vh], dtype=np.uint8)))
                ratio = float(hits) / float(patch_bgr.shape[0] * patch_bgr.shape[1])
                min_ratio = max(0.04, prof["expected_ratio"] * 0.30)
                return ratio >= min_ratio
            elif prof.get("is_cyan_blue", True):
                blue_cnt = np.count_nonzero(cv2.inRange(patch_hsv, np.array([85, 45, 120]), np.array([115, 255, 255])))
                blue_ratio = blue_cnt / float(patch_bgr.shape[0] * patch_bgr.shape[1])
                return blue_ratio >= 0.10
            else:
                diff = np.mean(np.abs(patch_bgr.astype(float) - tpl_bgr.astype(float)))
                return diff <= 60.0

        def _check_feature_color(
            patch_bgr: np.ndarray,
            feature_mask: Optional[np.ndarray] = None,
        ) -> bool:
            """
            两段式核心特征色彩门控：
            验证候选区域的色彩分布是否与标定的特征（如魔法帽、服饰、发色）相符，
            杜绝暗色树皮、木质平台或天空背景仅凭灰度边缘虚假匹配。
            """
            if not self.two_stage_profile or "color_profile" not in self.two_stage_profile:
                return True
            if patch_bgr is None or patch_bgr.size == 0:
                return False
            cp = self.two_stage_profile["color_profile"]
            patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV) if len(patch_bgr.shape) == 3 else None
            if patch_hsv is None:
                return True

            if feature_mask is not None and feature_mask.shape == patch_hsv.shape[:2]:
                valid = feature_mask > 0
                if int(np.count_nonzero(valid)) < 8:
                    return False
            else:
                valid = np.ones(patch_hsv.shape[:2], dtype=bool)

            p_s = float(np.mean(patch_hsv[:, :, 1][valid]))
            p_h = float(np.mean(patch_hsv[:, :, 0][valid]))

            if cp.get("is_saturated", False):
                min_s = max(30.0, cp["s_mean"] * 0.35)
                if p_s < min_s:
                    return False
                if abs(p_s - cp["s_mean"]) > 95.0:
                    return False
                h_dist = min(abs(p_h - cp["h_mean"]), 180.0 - abs(p_h - cp["h_mean"]))
                if h_dist > 35.0:
                    return False
            else:
                bgr_diff = sum(
                    abs(float(np.mean(patch_bgr[:, :, c][valid])) - cp["bgr_mean"][c])
                    for c in range(3)
                ) / 3.0
                if bgr_diff > 75.0:
                    return False
            return True

        def _verify_costume(pfx: int, pfy: int) -> Tuple[bool, float]:
            """
            服饰/魔法帽双重保险核验：
            在脚底 pfy 上方头部区域极速匹配帽子/服饰，并执行色彩门控。
            同时匹配原图和水平镜像（支持左右转向切换），耗时 < 0.05ms。
            """
            if self.costume_template_gray is None or not getattr(self, "enable_costume_verification", True):
                return True, 1.0

            c_w, c_h = self.costume_size
            bw, bh = getattr(self, "player_bbox_size", (55, 60))
            hy1 = max(0, pfy - bh - 25)
            hy2 = min(fh, pfy - 10)
            hx1 = max(0, pfx - max(35, c_w // 2 + 15))
            hx2 = min(fw, pfx + max(35, c_w // 2 + 15))

            if (hy2 - hy1) < c_h or (hx2 - hx1) < c_w:
                return False, 0.0

            head_roi = bgr_frame[hy1:hy2, hx1:hx2]
            head_gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY) if len(head_roi.shape) == 3 else head_roi

            res_norm = self._match_player_feature(
                head_gray, self.costume_template_gray, self.costume_template_mask
            )
            _, v_norm, _, loc_norm = cv2.minMaxLoc(res_norm)

            res_flip = self._match_player_feature(
                head_gray,
                self.costume_template_flipped_gray,
                self.costume_template_flipped_mask,
            )
            _, v_flip, _, loc_flip = cv2.minMaxLoc(res_flip)

            best_score = max(v_norm, v_flip)
            best_loc = loc_norm if v_norm >= v_flip else loc_flip
            best_mask = (
                self.costume_template_mask
                if v_norm >= v_flip else self.costume_template_flipped_mask
            )
            lx, ly = best_loc
            patch_head = head_roi[ly:ly+c_h, lx:lx+c_w]
            if not _check_feature_color(patch_head, best_mask):
                return False, 0.0

            # 特征框激活线与核验通过门限同步绑定基础准入门限
            th = self.player_feature_threshold
            if best_score >= th:
                self.last_feature_bbox = (hx1 + lx, hy1 + ly, c_w, c_h)

            return (best_score >= th), best_score

        def _scan_roi(rx1, ry1, rx2, ry2, allow_early_exit: bool = True):
            roi_bgr = bgr_frame[ry1:ry2, rx1:rx2]
            if roi_bgr.size == 0:
                return []
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
            votes = []

            # 1. 主力定位：优先以完整名牌做单模快速匹配 (Early-Exit)
            full_part = next((p for p in self.player_parts if p[0] == "nametag_full"), None)
            if full_part is not None:
                pname, tpl_bgr, tpl_g, off_x, off_y = full_part
                if tpl_g is not None and tpl_g.shape[0] <= roi_gray.shape[0] and tpl_g.shape[1] <= roi_gray.shape[1]:
                    res = cv2.matchTemplate(roi_gray, tpl_g, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    lx, ly = max_loc
                    if max_val >= 0.65:
                        patch = roi_bgr[ly:ly+tpl_g.shape[0], lx:lx+tpl_g.shape[1]]
                        if _check_color_profile(patch, tpl_bgr):
                            pfx = int(round(rx1 + lx + tpl_g.shape[1] / 2.0 - off_x))
                            pfy = int(round(ry1 + ly + off_y))
                            # 服饰/头部特征辅助核验 + 视觉朝向判定 (仅作为加权加分，不一票否决名牌)
                            c_ok, c_score = _verify_costume(pfx, pfy)
                            boost = 1.3 if c_ok else (1.1 if c_score >= self.player_feature_threshold else 1.0)
                            if allow_early_exit and max_val >= 0.72:
                                return [(pfx, pfy, max_val * 1.5 * boost, pname)]
                            votes.append((pfx, pfy, max_val * 1.3 * boost, pname))

            # 2. 完整名牌受阻时，启动多部件空间分解联合投票
            for pname, tpl_bgr, tpl_g, off_x, off_y in self.player_parts:
                if pname == "nametag_full" and votes:
                    continue
                if tpl_g is None or tpl_g.shape[0] > roi_gray.shape[0] or tpl_g.shape[1] > roi_gray.shape[1]:
                    continue
                res = cv2.matchTemplate(roi_gray, tpl_g, cv2.TM_CCOEFF_NORMED)

                # 多重峰值提取
                for _ in range(2):
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    threshold = 0.42 if pname == "nametag_full" else (0.44 if "nametag_anchor" in pname else (0.50 if "nametag" in pname else 0.70))
                    if max_val < threshold:
                        break

                    lx, ly = max_loc
                    patch = roi_bgr[ly:ly+tpl_g.shape[0], lx:lx+tpl_g.shape[1]]
                    if "nametag" in pname and not _check_color_profile(patch, tpl_bgr):
                        cv2.rectangle(res, (max(0, lx-4), max(0, ly-4)),
                                      (min(res.shape[1], lx+tpl_g.shape[1]+4), min(res.shape[0], ly+tpl_g.shape[0]+4)), 0, -1)
                        continue

                    # 计算脚底 Foothold 坐标
                    pfx = int(round(rx1 + lx + tpl_g.shape[1] / 2.0 - off_x))
                    pfy = int(round(ry1 + ly + off_y))
                    weight = 1.3 if "nametag" in pname else 0.8
                    votes.append((pfx, pfy, max_val * weight, pname))
                    break

            # 3. 辅助兜底救援：当名牌被遮挡或未匹配到时，启动两段式特征挽救定位 (需开启两段式特征开关)
            if not votes and self.two_stage_profile and getattr(self, "enable_two_stage_feature", True):
                f_r = self.two_stage_profile.get("feature_r_gray")
                f_l = self.two_stage_profile.get("feature_l_gray")
                f_r_mask = self.two_stage_profile.get("feature_r_mask")
                f_l_mask = self.two_stage_profile.get("feature_l_mask")
                if f_r is not None and f_l is not None and roi_gray.shape[0] >= f_r.shape[0] and roi_gray.shape[1] >= f_r.shape[1]:
                    res_r = self._match_player_feature(roi_gray, f_r, f_r_mask)
                    _, v_r, _, loc_r = cv2.minMaxLoc(res_r)
                    res_l = self._match_player_feature(roi_gray, f_l, f_l_mask)
                    _, v_l, _, loc_l = cv2.minMaxLoc(res_l)

                    best_v = max(v_r, v_l)
                    # 基础准入门限门控，杜绝地图树皮、岩石等杂色低分虚假匹配
                    if best_v >= self.player_feature_threshold:
                        facing = "right" if v_r >= v_l else "left"
                        lx, ly = loc_r if v_r >= v_l else loc_l
                        best_feature_mask = f_r_mask if v_r >= v_l else f_l_mask
                        fw_tpl, fh_tpl = f_r.shape[1], f_r.shape[0]
                        patch_feat = roi_bgr[ly:ly+fh_tpl, lx:lx+fw_tpl]

                        # 色彩门控过滤木质平台与背景
                        if _check_feature_color(patch_feat, best_feature_mask):
                            off_feet_y = self.two_stage_profile["offset_from_feet_y"]
                            off_center_x = self.two_stage_profile["offset_from_center_x"]

                            if facing == "right":
                                pfx = int(round(rx1 + lx + fw_tpl / 2.0 - off_center_x))
                            else:
                                pfx = int(round(rx1 + lx + fw_tpl / 2.0 + off_center_x))
                            pfy = int(round(ry1 + ly + fh_tpl + off_feet_y))

                            # 此处分数用于“人物在哪里”的兜底定位，不能同时
                            # 作为“人物朝哪边”的结论。攻击动画可能让反向模板
                            # 暂时略高；统一交给带质量门槛与防抖的朝向判定器。
                            self.last_feature_bbox = (rx1 + lx, ry1 + ly, fw_tpl, fh_tpl)
                            if best_v >= 0.75 and allow_early_exit:
                                return [(pfx, pfy, best_v * 1.8, f"two_stage_{facing}")]
                            votes.append((pfx, pfy, best_v * 1.5, f"two_stage_{facing}"))

            return votes

        player_exclusions = [r for r in self.exclusion_regions if r.get("player", False)]
        allowed_player_regions = MultiScaleMonsterBackend._allowed_rectangles(
            fw, fh, player_exclusions
        )

        def _scan_allowed(rx1, ry1, rx2, ry2):
            """只在未屏蔽的子区域执行角色模板扫描。"""
            result = []
            for ax, ay, aw, ah in allowed_player_regions:
                ix1, iy1 = max(rx1, ax), max(ry1, ay)
                ix2, iy2 = min(rx2, ax + aw), min(ry2, ay + ah)
                if ix2 > ix1 and iy2 > iy1:
                    result.extend(_scan_roi(ix1, iy1, ix2, iy2))
            return result

        # 1. 动态速度前瞻局部追踪窗口 (Velocity Anticipation ROI)：
        # 当已知历史位置且时效在 0.6s 内时，基于实时速度前瞻预判角色中心，并在前瞻点周围开窗
        now_t = time.perf_counter()
        has_recent_pos = (self.last_player_pos is not None and (now_t - self.last_player_time) < 0.6)
        if has_recent_pos:
            lx, ly = self.last_player_pos
            dt_pred = min(0.08, max(0.01, now_t - self.last_player_time))
            pred_x = int(round(lx + self.player_vx * dt_pred))
            pred_y = int(round(ly + self.player_vy * dt_pred))

            # 速度自适应窗口扩展：同时以 pred 点和上次确认点为中心做并集，杜绝速度惯性导致名牌被甩出视野
            speed = np.hypot(self.player_vx, self.player_vy)
            extra_w = min(80, int(speed * 0.06))
            extra_h = min(90, int(speed * 0.07))
            local_w = max(130, int(fw * 0.15)) + extra_w
            local_h = max(110, int(fh * 0.17)) + extra_h

            min_x = max(0, min(pred_x, lx) - local_w)
            max_x = min(fw, max(pred_x, lx) + local_w)
            min_y = max(0, min(pred_y, ly) - local_h)
            max_y = min(fh, max(pred_y, ly) + local_h)
            votes = _scan_allowed(min_x, min_y, max_x, max_y)
        else:
            votes = []

        # 2. 若无历史位置或局部丢失：使用 0.35x 金字塔全屏极速粗检 + 局部精修 (耗时仅 2ms，比 1080P 原图全扫快 70 倍！)
        if not votes:
            # 候选探测模板列表：优先使用名字牌，两段式特征也作为备用探测源 (需开启两段式开关)
            coarse_templates = []
            if self.player_templates_gray.get("nametag") is not None:
                coarse_templates.append(("nametag", self.player_templates_gray["nametag"]))
            if getattr(self, "enable_two_stage_feature", True) and self.two_stage_profile and self.two_stage_profile.get("feature_r_gray") is not None:
                f_tpl = self.two_stage_profile["feature_r_gray"]
                if not any(tpl is f_tpl for _, tpl in coarse_templates):
                    coarse_templates.append((
                        "feature",
                        f_tpl,
                        self.two_stage_profile.get("feature_r_mask"),
                    ))

            p_scale = 0.35
            inv_p = 1.0 / p_scale

            normalized_coarse_templates = [
                item if len(item) == 3 else (item[0], item[1], None)
                for item in coarse_templates
            ]
            for tpl_name, main_tpl_g, main_mask in normalized_coarse_templates:
                if votes:
                    break
                s_tpl = cv2.resize(main_tpl_g, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
                s_mask = None
                if main_mask is not None:
                    s_mask = cv2.resize(
                        main_mask,
                        (s_tpl.shape[1], s_tpl.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                # 各未屏蔽 ROI 分别做粗检，避免把角色模板匹配到屏蔽区。
                for ax, ay, aw, ah in allowed_player_regions:
                    roi_bgr = bgr_frame[ay:ay + ah, ax:ax + aw]
                    if roi_bgr.size == 0:
                        continue
                    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
                    s_gray = cv2.resize(roi_gray, (0, 0), fx=p_scale, fy=p_scale, interpolation=cv2.INTER_AREA)
                    if s_tpl.shape[0] >= s_gray.shape[0] or s_tpl.shape[1] >= s_gray.shape[1]:
                        continue
                    res_p = self._match_player_feature(s_gray, s_tpl, s_mask)
                    
                    candidate_votes = []
                    # 遍历粗检候选，交由原图精修和色彩门控过滤
                    for _ in range(8):
                        _, p_max_v, _, p_max_l = cv2.minMaxLoc(res_p)
                        if p_max_v < 0.25:
                            break
                        gx = ax + int(p_max_l[0] * inv_p)
                        gy = ay + int(p_max_l[1] * inv_p)
                        rx1 = max(ax, gx - 45)
                        ry1 = max(ay, gy - 45)
                        rx2 = min(ax + aw, gx + main_tpl_g.shape[1] + 45)
                        ry2 = min(ay + ah, gy + main_tpl_g.shape[0] + 45)
                        c_votes = _scan_allowed(rx1, ry1, rx2, ry2)
                        if c_votes:
                            candidate_votes.extend(c_votes)
                            # 如果已有强特征通过色彩门控确立，或者多部件聚类成功，提前早退
                            if any(v[2] >= 1.25 for v in c_votes):
                                break
                        px, py = p_max_l
                        cv2.rectangle(res_p, (max(0, px - 8), max(0, py - 8)),
                                      (min(res_p.shape[1], px + s_tpl.shape[1] + 8),
                                       min(res_p.shape[0], py + s_tpl.shape[0] + 8)), 0, -1)
                    if candidate_votes:
                        votes = candidate_votes
                        break

        # 3. 若名字牌被完全遮挡（如梯子、绳索、伤害数字），尝试用服饰/魔法帽在局部追踪窗口内挽救定位
        if not votes and self.costume_template_gray is not None and has_recent_pos and getattr(self, "enable_costume_verification", True) and getattr(self, "enable_two_stage_feature", True):
            c_w, c_h = self.costume_size
            local_roi_bgr = bgr_frame[min_y:max_y, min_x:max_x]
            if local_roi_bgr.shape[0] >= c_h and local_roi_bgr.shape[1] >= c_w:
                local_roi_gray = cv2.cvtColor(local_roi_bgr, cv2.COLOR_BGR2GRAY) if len(local_roi_bgr.shape) == 3 else local_roi_bgr
                res_norm = self._match_player_feature(
                    local_roi_gray,
                    self.costume_template_gray,
                    self.costume_template_mask,
                )
                _, v_norm, _, loc_norm = cv2.minMaxLoc(res_norm)
                res_flip = self._match_player_feature(
                    local_roi_gray,
                    self.costume_template_flipped_gray,
                    self.costume_template_flipped_mask,
                )
                _, v_flip, _, loc_flip = cv2.minMaxLoc(res_flip)

                best_v, best_loc = (v_norm, loc_norm) if v_norm >= v_flip else (v_flip, loc_flip)
                if best_v >= self.player_feature_threshold:
                    hx, hy = best_loc
                    pfx = int(round(min_x + hx + c_w / 2.0))
                    pfy = int(round(min_y + hy + c_h + self.costume_feet_offset_y))
                    votes.append((pfx, pfy, best_v * 1.6, "costume_rescue"))

        if votes:
            # 同一名字牌的多个小锚点会映射回几乎相同的 pfx/pfy。将它们
            # 聚类，优先选择“有多个独立部件支持”的候选，避免单一 UI
            # 纹理在低阈值下抢走结果。尤其是天空云层能偶然匹配到下方
            # 蓝色条，因此小锚点必须同时包含上、下两行的几何支持。
            def _supporters(candidate):
                cx, cy = candidate[0], candidate[1]
                return [v for v in votes if abs(v[0] - cx) <= 10 and abs(v[1] - cy) <= 10]

            def _is_structurally_valid(candidate):
                supporters = _supporters(candidate)
                names = [v[3] for v in supporters]
                if any("costume_rescue" in name or "two_stage_" in name for name in names):
                    return True
                has_top = any("nametag_anchor_top_" in name for name in names)
                has_bottom = any("nametag_anchor_bottom_" in name for name in names)
                has_large_template = any(name in ("nametag_full", "nametag_left", "nametag_right", "nametag_mid") for name in names)
                return has_large_template or (has_top and has_bottom)

            valid_votes = [v for v in votes if _is_structurally_valid(v)]
            votes = valid_votes

            # 服饰双重保险过滤与加权：若启用服饰特征，仅作增益加分/降权，不再一票否决肉眼真名牌
            if self.costume_template_gray is not None and getattr(self, "enable_costume_verification", True) and getattr(self, "enable_two_stage_feature", True):
                c_filtered = []
                for v in votes:
                    if v[3] == "costume_rescue":
                        c_filtered.append(v)
                        continue
                    c_ok, c_score = _verify_costume(v[0], v[1])
                    if c_ok:
                        c_filtered.append((v[0], v[1], v[2] * (1.2 + 0.3 * c_score), v[3]))
                    elif c_score >= self.player_feature_threshold:
                        c_filtered.append(v)
                    else:
                        # 服饰未对上时仅做适度降权 (乘 0.85)，绝不剔除高分名牌，防止攻击/遮挡动作时名牌被弃用
                        c_filtered.append((v[0], v[1], v[2] * 0.85, v[3]))
                if c_filtered:
                    votes = c_filtered

        if votes:
            def _vote_score(candidate):
                cx, cy = candidate[0], candidate[1]
                supporters = _supporters(candidate)
                score = sum(v[2] for v in supporters) + 0.14 * max(0, len(supporters) - 1)
                if self.last_player_pos is not None:
                    lx, ly = self.last_player_pos
                    score -= min(0.35, np.hypot(cx - lx, cy - ly) / float(fw * 0.35))
                return score

            best_vote = max(votes, key=_vote_score)

            pfx, pfy = best_vote[0], best_vote[1]
            now_hit = time.perf_counter()
            if self.last_player_pos is not None and self.last_player_time > 0:
                dt = max(0.005, min(0.5, now_hit - self.last_player_time))
                cur_vx = (pfx - self.last_player_pos[0]) / dt
                cur_vy = (pfy - self.last_player_pos[1]) / dt
                # EMA 指数平滑速度估计
                self.player_vx = 0.55 * self.player_vx + 0.45 * cur_vx
                self.player_vy = 0.55 * self.player_vy + 0.45 * cur_vy
            self.last_player_pos = (pfx, pfy)
            self.last_player_time = now_hit
            self.last_player_observed_time = now_hit
            bw, bh = getattr(self, "player_bbox_size", (int(fw * 0.04), int(fh * 0.08)))
            
            # 包围盒：底部对齐脚底 (pfy)，向上包裹全身高度 bh
            bbox = (pfx - bw // 2, pfy - bh, bw, bh)
            if self._point_excluded(pfx, pfy, "player"):
                return False, None, None
            return True, (pfx, pfy), bbox

        if self.last_player_pos is not None and (time.perf_counter() - self.last_player_time) < 1.0:
            px, py = self.last_player_pos
            bw, bh = getattr(self, "player_bbox_size", (int(fw * 0.04), int(fh * 0.08)))
            if self._point_excluded(px, py, "player"):
                return False, None, None
            return True, (px, py), (px - bw // 2, py - bh, bw, bh)

        return False, None, None


    def calibrate_player_from_frame(self, frame: np.ndarray, search_center: Optional[Tuple[int, int]] = None) -> bool:
        """
        一键重新标定当前角色（策略3）：
        在画面中心自动搜索天蓝底色名字牌与勋章，自适应任意分辨率
        """
        if frame is None or frame.size == 0:
            return False

        fh, fw = frame.shape[:2]
        cx = search_center[0] if search_center else fw // 2
        cy = search_center[1] if search_center else fh // 2

        # 在中心附近 40% 视口区域寻找天蓝色名字牌
        half_w = int(fw * 0.20)
        half_h = int(fh * 0.20)
        y1, y2 = max(0, cy - half_h), min(fh, cy + half_h)
        x1, x2 = max(0, cx - half_w), min(fw, cx + half_w)
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([85, 60, 140]), np.array([115, 255, 255]))

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask)
        best_rect = None
        min_w_thresh = max(20, int(fw * 0.02))
        for i in range(1, num_labels):
            sx, sy, sw, sh, s_area = stats[i]
            aspect = sw / max(1, sh)
            if 2.2 <= aspect <= 14.0 and 8 <= sh <= int(fh * 0.05) and sw >= min_w_thresh:
                best_rect = (x1 + sx, y1 + sy, sw, sh)
                break

        if best_rect is not None:
            rx, ry, rw, rh = best_rect
            nametag_crop = frame[ry:ry+rh, rx:rx+rw]
            medal_crop = frame[ry+rh:min(fh, ry+rh+int(rh*1.5)), max(0, rx-int(rw*0.1)):min(fw, rx+rw+int(rw*0.1))]

            cv2.imwrite(os.path.join(self.template_dir, "player_nametag.png"), nametag_crop)
            if medal_crop.size > 0:
                cv2.imwrite(os.path.join(self.template_dir, "player_medal.png"), medal_crop)

            self._load_player_templates()
            print(f"[MainViewDetector] 成功从画面标定角色! 名字牌尺寸: {nametag_crop.shape}")
            return True

        return False


    def _propagate_tracks_with_optical_flow(self, gray_frame: np.ndarray) -> bool:
        """以 LK 光流更新已确认怪物的屏幕位置，不延长其检测缓冲寿命。"""
        # 没有目标时没有任何可传播的点。旧实现仍会在 60Hz 下复制整张
        # 1280x720 灰度图，即使完整重检长期返回候选=0；这会制造约
        # 55MB/s 的无效内存带宽和额外 GC 压力。
        if not self.tracked_monsters:
            self._track_flow_prev_gray = None
            return False

        previous = self._track_flow_prev_gray
        # process() 每帧由 cvtColor 生成新的灰度 ndarray，后续不会原地
        # 改写，因此可以保留引用而不再复制整帧。
        self._track_flow_prev_gray = gray_frame
        if previous is None or previous.shape != gray_frame.shape:
            return False

        points = np.array(
            [[[track.cx, track.cy]] for track in self.tracked_monsters], dtype=np.float32
        )
        # LK 只需要怪物附近的纹理。对少量目标在整张画面建立三层金字塔
        # 是主要 CPU 热点之一；以所有追踪点的并集加 64px 边界裁切，仍
        # 覆盖现有允许的最大 45px 单帧位移，并保留 21x21 窗口余量。
        frame_h, frame_w = gray_frame.shape[:2]
        flat_points = points.reshape(-1, 2)
        margin = 64
        roi_x1 = max(0, int(np.floor(np.min(flat_points[:, 0]))) - margin)
        roi_y1 = max(0, int(np.floor(np.min(flat_points[:, 1]))) - margin)
        roi_x2 = min(frame_w, int(np.ceil(np.max(flat_points[:, 0]))) + margin + 1)
        roi_y2 = min(frame_h, int(np.ceil(np.max(flat_points[:, 1]))) + margin + 1)
        if roi_x2 - roi_x1 < 24 or roi_y2 - roi_y1 < 24:
            return False
        previous_roi = previous[roi_y1:roi_y2, roi_x1:roi_x2]
        current_roi = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        local_points = points.copy()
        local_points[:, :, 0] -= float(roi_x1)
        local_points[:, :, 1] -= float(roi_y1)
        try:
            new_points, status, errors = cv2.calcOpticalFlowPyrLK(
                previous_roi,
                current_roi,
                local_points,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
        except cv2.error:
            return False
        if new_points is None or status is None:
            return False
        new_points = new_points.copy()
        new_points[:, :, 0] += float(roi_x1)
        new_points[:, :, 1] += float(roi_y1)

        propagated = False
        for track, old_pt, new_pt, ok, error in zip(
            self.tracked_monsters,
            points.reshape(-1, 2),
            new_points.reshape(-1, 2),
            status.reshape(-1),
            (errors.reshape(-1) if errors is not None else np.zeros(len(points))),
        ):
            dx, dy = float(new_pt[0] - old_pt[0]), float(new_pt[1] - old_pt[1])
            # 失败光流/镜头突变不能把框拉飞；下一次模板检测会负责重定位。
            if not ok or float(error) > 25.0 or abs(dx) > 45.0 or abs(dy) > 45.0:
                continue
            track.cx += dx
            track.cy += dy
            track.bbox = (
                int(track.cx - track.w / 2),
                int(track.cy - track.h / 2),
                int(track.w),
                int(track.h),
            )
            track.last_track_ts = time.perf_counter()
            propagated = True
        return propagated

    def detect_monsters_classic(
        self,
        bgr_frame: np.ndarray,
        player_center: Optional[Tuple[int, int]],
        gray_frame: Optional[np.ndarray] = None,
        force_full_scan: bool = False,
        use_optical_flow: bool = True,
    ) -> List[MonsterTarget]:
        """
        超低延迟并发全屏怪物模板匹配 + 软抑制 (Soft-NMS) + 时序生命周期追踪器 (MOT Tracker):
          1. 采用常驻 ThreadPoolExecutor 并行匹配所有动作精灵图
          2. 使用微步长局部峰值抑制 (Micro-Step Peak Suppression)，杜绝重叠怪物相互吞峰值
          3. 采用自适应 Soft-NMS 重叠怪物分流聚类，双怪/多怪同屏紧挨时独立保留
          4. 引入多目标时序追踪器 (MOT Tracker，含 8 帧 ~200ms 丢失容忍缓冲)，彻底解决受击闪白与伤害数字遮挡闪烁
        """
        if bgr_frame is None or (not self.multi_scale_backend.available and not self.enable_monster_hp_bar_detection):
            # 无模板且无血条识别时不会进入 _update_tracker，因此这里必须主动清空
            # 上一帧/上一张地图遗留的追踪目标。
            self.clear_monster_tracks()
            return []

        fh, fw = bgr_frame.shape[:2]
        pcx, pcy = player_center if player_center else (fw // 2, fh // 2)
        if gray_frame is None:
            gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        if use_optical_flow:
            with self._monster_track_lock:
                flow_propagated = self._propagate_tracks_with_optical_flow(gray_frame)
        else:
            # “全速完整模板扫描”用每张新捕获帧的真实模板结果更新坐标，
            # 不在两次扫描之间使用光流推进，便于隔离验证完整匹配本身。
            flow_propagated = False

        self._monster_frame_index += 1
        force_redetect = (
            bool(force_full_scan)
            or self.monster_redetect_interval <= 1
            or self.tracker_buffer_time_sec <= 0.0
            or not self.tracked_monsters
            or (self._monster_frame_index % self.monster_redetect_interval == 0)
        )
        if not force_redetect:
            # 光流已经更新了轨迹位置；这里仅推进生命周期，不再重复预测
            # 一次，避免中间帧出现双倍位移。
            with self._monster_track_lock:
                return self._update_tracker(
                    [], pcx, pcy, predict_tracks=False, full_scan_completed=False
                )

        # 实体发现完全使用新多尺度匹配器；原有 _update_tracker 继续提供
        # 基于真实秒数的缓冲，因而 UI 的 Buffer=0ms 可严格禁用残留目标。
        with self._monster_backend_lock:
            raw_candidates = (
                self.multi_scale_backend.detect(
                    bgr_frame, self.monster_threshold, self.exclusion_regions
                ) if (self.enable_monster_detection and self.multi_scale_backend.available) else []
            )
        raw_candidates = [
            item for item in raw_candidates
            if not self._bbox_excluded(item[:4], "monster")
        ]
        if self.enable_monster_hp_bar_detection and bgr_frame is not None and len(bgr_frame.shape) == 3:
            hp_inferred = self._detect_monsters_from_hp_bars(
                bgr_frame, existing_candidates=raw_candidates
            )
            if hp_inferred:
                raw_candidates.extend(hp_inferred)
        else:
            self.last_hp_bar_backend = "off"
            self.last_hp_bar_cost_ms = 0.0
        self.last_full_scan_candidate_count = len(raw_candidates)
        with self._monster_track_lock:
            return self._update_tracker(
                raw_candidates,
                pcx,
                pcy,
                predict_tracks=(not flow_propagated and not force_full_scan),
            )

        # 100% 全屏扫描 (适应任意游戏分辨率)
        gray_roi = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        # 0.30x 金字塔快速全屏缩放 (极致性能 + 完整保留怪物轮廓细节)
        pyramid_scale = 0.30
        inv_scale = 1.0 / pyramid_scale
        small_gray_roi = cv2.resize(gray_roi, (0, 0), fx=pyramid_scale, fy=pyramid_scale, interpolation=cv2.INTER_AREA)

        # 内部并发工作函数 (C++ OpenCV 在多线程下释放 GIL)
        def _match_single_template(item):
            mname, scaled_tpl, orig_w, orig_h = item
            if scaled_tpl.shape[0] >= small_gray_roi.shape[0] or scaled_tpl.shape[1] >= small_gray_roi.shape[1]:
                return []

            match_res = cv2.matchTemplate(small_gray_roi, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            cands = []
            
            # 微步长峰值抑制半径 (仅消除同像素噪点，保留相邻重叠怪物的独立峰值)
            sup_rx = max(2, int(scaled_tpl.shape[1] * 0.25))
            sup_ry = max(2, int(scaled_tpl.shape[0] * 0.25))

            # 全屏提取前 6 个显著峰值
            for _ in range(6):
                _, max_v, _, max_l = cv2.minMaxLoc(match_res)
                if max_v < self.monster_threshold:
                    break
                gx = int(max_l[0] * inv_scale)
                gy = int(max_l[1] * inv_scale)
                cands.append((gx, gy, orig_w, orig_h, mname, max_v))
                
                # 精准局部峰值抑制
                cv2.rectangle(match_res, (max(0, max_l[0]-sup_rx), max(0, max_l[1]-sup_ry)), 
                              (min(match_res.shape[1], max_l[0]+sup_rx), 
                               min(match_res.shape[0], max_l[1]+sup_ry)), 0, -1)
            return cands


        # 高性能流水线匹配所有动作模板 (避免外部多线程与 OpenCV 内部多核争抢，大幅降低 CPU 占用)
        raw_candidates = []
        for tpl_item in self.monster_cached_scaled_templates:
            res = _match_single_template(tpl_item)
            if res:
                raw_candidates.extend(res)

        # 软非极大值抑制 (Soft-NMS) 重叠怪物分流
        picked_candidates = self._soft_nms(raw_candidates, iou_thresh=0.40, min_score=self.monster_threshold * 0.90)

        # 时序生命周期追踪器 (MOT Tracker，抗受击闪白与伤害数字遮挡)
        targets = self._update_tracker(picked_candidates, pcx, pcy)
        return targets


    def _soft_nms(self, raw_candidates: List[Tuple], iou_thresh: float = 0.40, min_score: float = 0.48) -> List[Tuple]:
        """
        自适应 Soft-NMS (软非极大值抑制) 与重叠怪物分流聚类:
        1. 对于重叠候选框，不直接暴力抹杀，而是根据重叠面积进行平滑衰减
        2. 当两只怪中心距超过物理阈值时，自动识别为紧挨着的多只独立怪物并同时保留
        """
        if not raw_candidates:
            return []

        cands = list(raw_candidates)
        cands.sort(key=lambda c: c[5], reverse=True)
        picked = []

        while cands:
            best = cands.pop(0)
            picked.append(best)
            bx, by, bw, bh, bname, bscore = best
            bcx, bcy = bx + bw / 2, by + bh / 2

            survived = []
            for item in cands:
                ix, iy, iw, ih, iname, iscore = item
                icx, icy = ix + iw / 2, iy + ih / 2

                # 计算 IoU
                xx1 = max(bx, ix)
                yy1 = max(by, iy)
                xx2 = min(bx + bw, ix + iw)
                yy2 = min(by + bh, iy + ih)
                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)
                inter_area = inter_w * inter_h
                union_area = (bw * bh) + (iw * ih) - inter_area
                iou = inter_area / max(1.0, float(union_area))

                # 中心位移距离
                cdist = float(np.hypot(bcx - icx, bcy - icy))
                min_dim = min(bw, bh)

                if iou > iou_thresh:
                    if cdist > max(14.0, min_dim * 0.28):
                        # 两怪紧挨着重叠走动 (双怪同屏)：适度衰减置信度但保留
                        decayed_score = iscore * (1.0 - 0.45 * iou)
                        if decayed_score >= min_score:
                            survived.append((ix, iy, iw, ih, iname, decayed_score))
                    else:
                        # 同一怪物的重复密集假阳性框：强衰减
                        decayed_score = iscore * (1.0 - iou)
                        if decayed_score >= min_score:
                            survived.append((ix, iy, iw, ih, iname, decayed_score))
                else:
                    survived.append(item)

            cands = survived
            cands.sort(key=lambda c: c[5], reverse=True)

        return picked


    def _update_tracker(
        self,
        detections: List[Tuple[int, int, int, int, str, float]],
        pcx: int,
        pcy: int,
        current_ts: Optional[float] = None,
        predict_tracks: bool = True,
        observation_ts: Optional[float] = None,
        full_scan_completed: bool = True,
    ) -> List[MonsterTarget]:
        """
        多目标追踪更新器 (MOT Association):
        1. 预测所有活跃 Track 的新坐标 (基于真实时间 dt 的惯性外推)
        2. 基于中心近邻与尺度进行贪心双向关联
        3. 未匹配的已存 Track 允许保留 tracker_buffer_time_sec (默认 0.22s)，抗受击闪白与伤害数字遮挡
        4. 新检测到的怪创建新 Track
        """
        if current_ts is None:
            current_ts = time.perf_counter()
        detected_at = current_ts if observation_ts is None else float(observation_ts)
        dt_sec = max(0.001, min(0.10, current_ts - self.last_frame_timestamp))
        self.last_frame_timestamp = current_ts

        # 1. 预测
        for trk in self.tracked_monsters:
            if predict_tracks:
                trk.predict(dt_sec)
            # 缓冲必须按“最后一次真正模板命中”算真实时间，不能受物理
            # 预测 dt 上限影响；否则低帧率或卡顿时表现会和滑杆毫秒数不一致。
            trk.time_since_update_sec = max(0.0, current_ts - trk.last_update_ts)

        matched_tracks = set()
        matched_dets = set()
        preexisting_track_count = len(self.tracked_monsters)

        # 2. 匹配关联 (贪心近邻匹配)
        if self.tracked_monsters and detections:
            cost_matrix = []
            track_species_ids = [mob_id_of(trk) for trk in self.tracked_monsters]
            detection_species_ids = [mob_id_from_name(det[4]) for det in detections]
            for t_idx, trk in enumerate(self.tracked_monsters):
                for d_idx, det in enumerate(detections):
                    dx, dy, dw, dh, dname, dscore = det
                    track_species = track_species_ids[t_idx]
                    detected_species = detection_species_ids[d_idx]
                    if (track_species is not None and detected_species is not None
                            and track_species != detected_species):
                        continue
                    dcx, dcy = dx + dw / 2.0, dy + dh / 2.0
                    dist = float(np.hypot(trk.cx - dcx, trk.cy - dcy))
                    cost_matrix.append((dist, t_idx, d_idx))

            cost_matrix.sort(key=lambda x: x[0])
            for dist, t_idx, d_idx in cost_matrix:
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                # 最大关联距离阈值 (自适应怪物体型)
                max_match_dist = max(55.0, self.tracked_monsters[t_idx].w * 1.1)
                if dist <= max_match_dist:
                    dx, dy, dw, dh, dname, dscore = detections[d_idx]
                    selected_track = self.tracked_monsters[t_idx]
                    if mob_id_of(selected_track) is not None and dname == "mob_hp":
                        dname = selected_track.name
                    selected_track.update(
                        (dx, dy, dw, dh), dscore, dname, detected_at
                    )
                    # 异步扫描完成后才合并到当前追踪帧，视觉存活时间应从
                    # 合并时刻计算，不能被较早的扫描时间倒退。
                    selected_track.last_track_ts = current_ts
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

        # 一次“漏检”只对应一次已经完成并被消费的完整模板扫描。60Hz
        # 光流帧会频繁用空 detections 推进轨迹，绝不能把它们算作漏检。
        if full_scan_completed:
            for t_idx in range(preexisting_track_count):
                if t_idx not in matched_tracks:
                    self.tracked_monsters[t_idx].consecutive_full_scan_misses += 1

        # 3. 未匹配的新检测 -> 创建新追踪目标
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                dx, dy, dw, dh, dname, dscore = det
                new_track = TrackedMob(
                    track_id=self.next_track_id,
                    name=dname,
                    bbox=(dx, dy, dw, dh),
                    cx=float(dx + dw / 2),
                    cy=float(dy + dh / 2),
                    w=float(dw),
                    h=float(dh),
                    score=dscore,
                    last_update_ts=detected_at,
                    last_track_ts=current_ts,
                    attack_observation_hard_timeout_sec=(
                        getattr(
                            self,
                            "attack_observation_hard_timeout_sec",
                            ATTACK_OBSERVATION_MAX_AGE_SEC,
                        )
                    ),
                    attack_observation_hard_timeout_enabled=(
                        getattr(
                            self,
                            "attack_observation_hard_timeout_enabled",
                            True,
                        )
                    ),
                )
                self.next_track_id += 1
                self.tracked_monsters.append(new_track)

        # 4. 清理超时丢失目标 (严格基于真实物理时间 tracker_buffer_time_sec 未观测到才认为死亡)
        alive_tracks = []
        for trk in self.tracked_monsters:
            if self.tracker_buffer_time_sec <= 0.0:
                # 0ms 的语义是关闭缓冲：仅允许“本次检测刚更新”的轨迹，
                # 不能因调用时间差仅有几毫秒而残留到下一帧。
                if abs(float(trk.last_update_ts) - float(current_ts)) <= 1e-6:
                    alive_tracks.append(trk)
            else:
                # 模板连续漏检时，只要相邻帧光流仍稳定跟随，就允许视觉框
                # 继续存在；攻击模块使用“两次完整漏检 + 硬上限”过滤。
                visual_age = max(0.0, current_ts - trk.last_track_ts)
                template_age = trk.time_since_update_sec
                # 连续光流不能无限延寿：防幽灵框安全上限内必须重新得到一次真实模板
                # 命中，否则按死亡/离屏处理，避免留下长期幽灵框。
                if getattr(self, "ghost_box_safety_timeout_enabled", True):
                    safety_timeout = getattr(self, "ghost_box_safety_timeout_sec", 0.75)
                    max_template_gap = max(self.tracker_buffer_time_sec, safety_timeout)
                    template_gap_ok = (template_age <= max_template_gap)
                else:
                    template_gap_ok = True

                if visual_age <= self.tracker_buffer_time_sec and template_gap_ok:
                    alive_tracks.append(trk)
        self.tracked_monsters = alive_tracks


        # 5. 格式化输出为 MonsterTarget
        targets = []
        for trk in self.tracked_monsters:
            mcx = int(trk.cx)
            mcy = int(trk.cy)
            dist = float(np.hypot(mcx - pcx, mcy - pcy))
            rel_dir = "right" if mcx >= pcx else "left"

            # 通用名称格式化
            clean_name = trk.name.replace("mob_", "").replace("_clean", "").replace("_right", "").replace("_left", "").replace(".png", "")
            if "_" in clean_name and not clean_name.replace("_", "").isdigit():
                clean_name = " ".join([w.capitalize() for w in clean_name.split("_")])
            else:
                clean_name = clean_name.capitalize()

            targets.append(MonsterTarget(
                name=clean_name,
                mob_id=mob_id_of(trk),
                bbox=trk.bbox,
                center=(mcx, mcy),
                score=trk.score,
                distance=dist,
                relative_direction=rel_dir,
                track_id=trk.track_id,
                last_update_ts=trk.last_update_ts,
                last_track_ts=trk.last_track_ts,
                consecutive_full_scan_misses=trk.consecutive_full_scan_misses,
                attack_observation_hard_timeout_sec=(
                    trk.attack_observation_hard_timeout_sec
                ),
                attack_observation_hard_timeout_enabled=(
                    getattr(trk, "attack_observation_hard_timeout_enabled", True)
                ),
                is_fresh_for_attack=is_fresh_attack_observation(trk, current_ts),
            ))

        targets.sort(key=lambda t: t.distance)
        return targets


    def calculate_attack_box(self, player_feet: Tuple[int, int], facing: str, frame_w: int = 1920) -> Tuple[int, int, int, int]:
        primary = skills_from_config(self.attack_config)[0] if getattr(self, "attack_config", None) else None
        if primary is not None and primary.get("wz_rect") is not None:
            return skill_attack_box(primary, player_feet, facing, frame_w)
        pfx, pfy = player_feet
        # 自适应分辨率缩放
        res_scale = max(0.4, float(frame_w) / 1920.0)
        reach_x = int(self.attack_reach_x * res_scale)
        reach_y_up = int(self.attack_reach_y_up * res_scale)
        reach_y_down = int(self.attack_reach_y_down * res_scale)
        behind_x = int(self.behind_reach_x * res_scale)
        ay = pfy - reach_y_up
        ah = reach_y_up + reach_y_down

        if self.attack_two_way:
            ax = pfx - reach_x
            aw = reach_x * 2
        elif facing == "right":
            ax = pfx - behind_x
            aw = reach_x + behind_x
        else:
            ax = pfx - reach_x
            aw = reach_x + behind_x

        return (ax, ay, aw, ah)

    def calculate_rear_attack_box(self, player_feet: Tuple[int, int], facing: str, frame_w: int = 1920) -> Tuple[int, int, int, int]:
        """单向技能的身后候选区：与前向射程等长，纵向范围完全一致。"""
        primary = skills_from_config(self.attack_config)[0] if getattr(self, "attack_config", None) else None
        if primary is not None and primary.get("wz_rect") is not None:
            return skill_attack_box(primary, player_feet,
                                    "left" if facing == "right" else "right", frame_w)
        pfx, pfy = player_feet
        res_scale = max(0.4, float(frame_w) / 1920.0)
        reach_x = int(self.attack_reach_x * res_scale)
        reach_y_up = int(self.attack_reach_y_up * res_scale)
        reach_y_down = int(self.attack_reach_y_down * res_scale)
        ay = pfy - reach_y_up
        ah = reach_y_up + reach_y_down
        ax = pfx - reach_x if facing == "right" else pfx
        return (ax, ay, reach_x, ah)

    def calculate_skirmish_boxes(
        self, player_feet: Tuple[int, int], facing: str, frame_w: int = 1920
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """返回紧贴前/后攻击范围外沿的游击带，不包含原攻击范围。"""
        primary = skills_from_config(self.attack_config)[0] if getattr(self, "attack_config", None) else None
        if primary is not None and primary.get("wz_rect") is not None:
            from src.engine.attack_skills import skirmish_boxes as skill_skirmish_boxes
            return skill_skirmish_boxes(primary, player_feet, facing, frame_w,
                                        self.skirmish_range_x)
        pfx, pfy = player_feet
        res_scale = max(0.4, float(frame_w) / 1920.0)
        reach_x = int(self.attack_reach_x * res_scale)
        extra_x = int(self.skirmish_range_x * res_scale)
        reach_y_up = int(self.attack_reach_y_up * res_scale)
        reach_y_down = int(self.attack_reach_y_down * res_scale)
        ay = pfy - reach_y_up
        ah = reach_y_up + reach_y_down
        if facing == "right":
            front = (pfx + reach_x, ay, extra_x, ah)
            rear = (pfx - reach_x - extra_x, ay, extra_x, ah)
        else:
            front = (pfx - reach_x - extra_x, ay, extra_x, ah)
            rear = (pfx + reach_x, ay, extra_x, ah)
        return front, rear


    def process(
        self,
        frame: np.ndarray,
        manual_facing: Optional[str] = None,
        reuse_player: bool = False,
        monster_detection_batch: Optional[MonsterDetectionBatch] = None,
        async_monster_matching: bool = False,
        full_scan_only: bool = False,
    ) -> MainViewResult:
        t0 = time.perf_counter()
        res = MainViewResult(timestamp=t0)

        # ``reuse_player`` skips detect_player on intermediate frames, so the
        # per-frame feature evidence must also be cleared at this entry point.
        self.last_feature_bbox = None

        if frame is None or frame.size == 0:
            return res

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray_frame.shape[:2]
        self.last_frame_width = int(fw)

        # 怪物模板匹配比角色定位更需要每帧刷新。角色定位最多每约 67ms
        # 重取一次；中间帧复用最近位置，避免名字牌模板匹配吞掉 1/3 算力。
        if reuse_player and self.last_player_pos is not None:
            found = True
            p_center = self.last_player_pos
            p_bbox = self.last_player_bbox
        else:
            found, p_center, p_bbox = self.detect_player(frame)
        res.player_found = found

        res.player_pos = p_center
        res.player_bbox = p_bbox

        if p_center is not None:
            if manual_facing in ("left", "right"):
                res.facing_direction = manual_facing
                # Manual/retained direction and feature visibility are separate:
                # refresh the current-frame feature box without allowing the
                # template result to change left/right state.
                _, feature_score, _ = self._detect_facing_direction(
                    gray_frame, p_center, update_direction=False
                )
                res.facing_confidence = feature_score
                res.is_facing_locked = False
            else:
                facing, conf, is_locked = self._detect_facing_direction(gray_frame, p_center)
                res.facing_direction = facing
                res.facing_confidence = conf
                res.is_facing_locked = is_locked

            res.attack_box = self.calculate_attack_box(p_center, res.facing_direction, frame_w=fw)
            attack_config = getattr(self, "attack_config", None)
            attack_skills = skills_from_config(attack_config) if attack_config is not None else []
            if attack_config is not None:
                res.skill_attack_boxes = {
                    skill["id"]: skill_attack_box(skill, p_center, res.facing_direction, fw)
                    for skill in attack_skills
                }
            if not self.attack_two_way:
                res.rear_attack_box = self.calculate_rear_attack_box(
                    p_center, res.facing_direction, frame_w=fw
                )
            if self.skirmish_range_x > 0:
                res.front_skirmish_box, res.rear_skirmish_box = self.calculate_skirmish_boxes(
                    p_center, res.facing_direction, frame_w=fw
                )

        if async_monster_matching:
            monsters = self.track_monsters_from_async_batch(
                gray_frame, p_center, monster_detection_batch
            )
        else:
            # 保留同步入口供离线工具、测试与兼容调用方使用；GUI运行时使用
            # 独立完整扫描线程，不再让重模板匹配阻塞60Hz轻量追踪。
            monsters = self.detect_monsters_classic(
                frame,
                p_center,
                gray_frame,
                force_full_scan=full_scan_only,
                use_optical_flow=not full_scan_only,
            )
        if res.attack_box:
            ax, ay, aw, ah = res.attack_box
            attack_snapshot_at = time.perf_counter()
            best_candidate = None
            for m in monsters:
                attack_config = getattr(self, "attack_config", None)
                if attack_config is not None:
                    in_range = any(
                        target_in_box(res.skill_attack_boxes[skill["id"]], m)
                        for skill in eligible_skills(attack_config, m.mob_id, attack_skills)
                    )
                else:
                    in_range = target_in_box(res.attack_box, m)
                m.is_in_attack_range = in_range
                m.is_fresh_for_attack = is_fresh_attack_observation(
                    m, attack_snapshot_at
                )
                if in_range and m.is_fresh_for_attack:
                    # 优先选择当前已经锁定的目标，保持红框锁定连贯性
                    if self._locked_target_id is not None and m.track_id == self._locked_target_id:
                        best_candidate = m
                    elif best_candidate is None:
                        best_candidate = m

            # 锁定目标防抖保持 (Hold Latch): 怪物在框内时保持稳定红色，杜绝红橙闪烁
            if best_candidate is not None:
                res.locked_target = best_candidate
                self._locked_target_id = best_candidate.track_id
                self._locked_target_until = attack_snapshot_at + 0.25
            elif self._locked_target_id is not None and attack_snapshot_at < self._locked_target_until:
                # 仍处于防抖保持期内，尝试找回仍存活的被锁目标
                for m in monsters:
                    if m.track_id == self._locked_target_id and getattr(m, 'is_in_attack_range', False):
                        res.locked_target = m
                        break
            else:
                self._locked_target_id = None
                self._locked_target_until = 0.0
            if res.locked_target is not None and attack_config is not None:
                selected_skill = choose_skill_for_target(
                    attack_config, res.locked_target, p_center,
                    res.facing_direction, fw, monsters, attack_skills,
                )
                res.locked_skill_id = selected_skill["id"] if selected_skill else None

        res.monsters = monsters
        if p_center:
            self.last_player_pos = p_center
            self.last_player_bbox = p_bbox
            self.last_player_time = t0
            res.feature_bbox = self.last_feature_bbox
        else:
            self.last_feature_bbox = None
            res.feature_bbox = None
        res.proc_time_ms = (time.perf_counter() - t0) * 1000.0
        return res


    def render_debug(self, frame: np.ndarray, result: MainViewResult, fps: float = 0.0) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        canvas = frame.copy()
        rendered_locked_target = result.locked_target
        if rendered_locked_target is not None and not is_fresh_attack_observation(rendered_locked_target):
            rendered_locked_target = None
        if rendered_locked_target is None and result.monsters:
            # 兜底保障：只要本帧有任何怪物在攻击范围内，即时呈现红框锁定
            for m in result.monsters:
                if getattr(m, "is_in_attack_range", False) and is_fresh_attack_observation(m):
                    rendered_locked_target = m
                    break

        # 1. 单向技能先绘制淡橙色身后候选框。它只表示停留阶段允许
        # 延迟回身的区域，不会在平台移动途中直接触发攻击。
        if result.rear_attack_box:
            rax, ray, raw, rah = result.rear_attack_box
            rear_overlay = canvas.copy()
            rear_color = (0, 165, 255)
            cv2.rectangle(
                rear_overlay, (rax, ray), (rax + raw, ray + rah), rear_color, -1
            )
            cv2.addWeighted(rear_overlay, 0.10, canvas, 0.90, 0, canvas)
            cv2.rectangle(
                canvas, (rax, ray), (rax + raw, ray + rah), (70, 175, 235), 1
            )
            cv2.putText(
                canvas, "REAR TURN BOX", (rax + 6, ray + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (70, 175, 235), 1,
            )

        # 游击范围只画攻击盒外侧的扩展带，避免用户误以为整个大框都
        # 会直接攻击。青色为前向，灰蓝色为后向低优先级游击带。
        for sk_box, color, label in (
            (result.front_skirmish_box, (255, 200, 60), "SKIRMISH F"),
            (result.rear_skirmish_box, (190, 150, 90), "SKIRMISH R"),
        ):
            if not sk_box or sk_box[2] <= 0:
                continue
            sx, sy, sw, sh = sk_box
            sk_overlay = canvas.copy()
            cv2.rectangle(sk_overlay, (sx, sy), (sx + sw, sy + sh), color, -1)
            cv2.addWeighted(sk_overlay, 0.07, canvas, 0.93, 0, canvas)
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, 1)
            cv2.putText(
                canvas, label, (sx + 4, sy + sh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
            )

        # 2. 绘制当前朝向的实际攻击判定盒
        if result.attack_box:
            ax, ay, aw, ah = result.attack_box
            overlay = canvas.copy()
            primary_red = bool(
                rendered_locked_target
                and (
                    getattr(self, "attack_config", None) is None
                    or (
                        result.locked_skill_id == "primary"
                        and target_in_box(result.attack_box, rendered_locked_target)
                    )
                )
            )
            box_color = (0, 0, 255) if primary_red else (0, 165, 255)
            cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), box_color, -1)
            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
            cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), box_color, 2)
            cv2.putText(canvas, f"ATTACK BOX [{result.facing_direction.upper()}]", (ax + 6, ay - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        for skill_id, box in result.skill_attack_boxes.items():
            if skill_id == "primary":
                continue
            sx, sy, sw, sh = box
            skill_red = bool(
                rendered_locked_target
                and target_in_box(box, rendered_locked_target)
                and result.locked_skill_id == skill_id
            ) if getattr(self, "attack_config", None) is not None else False
            skill_color = (0, 0, 255) if skill_red else (255, 90, 220)
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), skill_color, 2 if skill_red else 1)
            cv2.putText(canvas, skill_id.upper(), (sx + 4, max(12, sy - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, skill_color, 1)

        # 2. 绘制角色本体与朝向箭头
        if result.player_found and result.player_pos and result.player_bbox:
            pcx, pcy = result.player_pos
            bx, by, bw, bh = result.player_bbox
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.circle(canvas, (pcx, pcy), 5, (0, 0, 255), -1)
            cv2.putText(canvas, f"PLAYER [{result.facing_direction.upper()}]", (bx, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

            arrow_dx = 32 if result.facing_direction == "right" else -32
            cv2.arrowedLine(canvas, (pcx, pcy), (pcx + arrow_dx, pcy), (0, 255, 255), 2, tipLength=0.35)

            # 绘制匹配到的人物服饰/帽子/特征框 (蓝色)
            if getattr(result, "feature_bbox", None):
                fx, fy, fw, fh = result.feature_bbox
                # 纯蓝框 (BGR: (255, 0, 0))
                cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                cv2.putText(canvas, f"FEATURE [{result.facing_direction.upper()}]", (fx, max(12, fy - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 0, 0), 1)

        # 3. 绘制怪物与距离连线
        for m in result.monsters:
            mx, my, mw, mh = m.bbox
            mcx, mcy = m.center
            is_locked = (
                rendered_locked_target is not None and (
                    rendered_locked_target == m
                    or (
                        getattr(rendered_locked_target, "track_id", None) is not None
                        and getattr(rendered_locked_target, "track_id", None) == getattr(m, "track_id", None)
                    )
                )
            )
            color = (0, 0, 255) if is_locked else (0, 255, 0)
            tag = f"LOCKED! [{m.name}]" if is_locked else f"[{m.name}] ({m.score:.2f})"

            cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), color, 2)
            cv2.circle(canvas, (mcx, mcy), 4, color, -1)
            cv2.putText(canvas, tag, (mx, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

            if result.player_pos:
                pcx, pcy = result.player_pos
                cv2.line(canvas, (pcx, pcy), (mcx, mcy), (180, 180, 180), 1, cv2.LINE_AA)

        # 4. 浮动半透明 HUD (严格避开左上角 0~280px 小地图区域，确保地图名 100% 可视且不遮挡 OCR)
        h, w, _ = canvas.shape
        hud_w = 460
        hud_h = 36
        hud_x = max(290, (w - hud_w) // 2)
        hud_y = 8

        # 半透明深色圆角胶囊背景
        hud_overlay = canvas.copy()
        cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 24), -1)
        cv2.addWeighted(hud_overlay, 0.70, canvas, 0.30, 0, canvas)
        cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 65, 75), 1)

        status_text = "READY TO ATTACK!" if rendered_locked_target else "SEARCHING TARGETS..."
        status_color = (0, 0, 255) if rendered_locked_target else (0, 255, 128)
        cv2.putText(canvas, f"STATUS: {status_text}", (hud_x + 12, hud_y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, status_color, 1)

        info_str = f"Mobs: {len(result.monsters)} | {result.proc_time_ms:.1f}ms | FPS: {fps:.1f}"
        cv2.putText(canvas, info_str, (hud_x + hud_w - 200, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return canvas

    def release(self):
        """安全释放检测器占用的资源"""
        if hasattr(self, "executor") and self.executor:
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass
