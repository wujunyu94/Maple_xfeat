"""F6 断线检测与两类 v83 客户端的自动重连状态机。"""

from __future__ import annotations

import base64
import ctypes
import os
import threading
import time
import zlib
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


_crypt32 = ctypes.windll.crypt32
_kernel32 = ctypes.windll.kernel32
_crypt32.CryptProtectData.argtypes = [
    ctypes.POINTER(_DataBlob), wintypes.LPCWSTR, ctypes.POINTER(_DataBlob),
    ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
]
_crypt32.CryptProtectData.restype = wintypes.BOOL
_crypt32.CryptUnprotectData.argtypes = [
    ctypes.POINTER(_DataBlob), ctypes.c_void_p, ctypes.POINTER(_DataBlob),
    ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_DataBlob),
]
_crypt32.CryptUnprotectData.restype = wintypes.BOOL
_kernel32.LocalFree.argtypes = [ctypes.c_void_p]
_kernel32.LocalFree.restype = ctypes.c_void_p


def protect_secret(value: str) -> str:
    """用当前 Windows 用户的 DPAPI 加密敏感配置。"""
    raw = str(value).encode("utf-8")
    if not raw:
        return ""
    source_buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
    target = _DataBlob()
    if not _crypt32.CryptProtectData(
        ctypes.byref(source), "Maple reconnect", None, None, None, 0,
        ctypes.byref(target),
    ):
        raise ctypes.WinError()
    try:
        encrypted = ctypes.string_at(target.pbData, target.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        _kernel32.LocalFree(ctypes.cast(target.pbData, ctypes.c_void_p))


def unprotect_secret(value: str) -> str:
    """解密 DPAPI 密文；跨 Windows 用户或损坏时返回空字符串。"""
    if not value:
        return ""
    try:
        raw = base64.b64decode(str(value), validate=True)
        source_buffer = ctypes.create_string_buffer(raw)
        source = _DataBlob(len(raw), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
        target = _DataBlob()
        if not _crypt32.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
        ):
            return ""
        try:
            return ctypes.string_at(target.pbData, target.cbData).decode("utf-8")
        finally:
            _kernel32.LocalFree(ctypes.cast(target.pbData, ctypes.c_void_p))
    except Exception:
        return ""


@dataclass(frozen=True)
class VisualMatch:
    state: str
    profile: str
    score: float
    click_pos: Optional[Tuple[float, float]] = None


class ReconnectVision:
    """以用户提供的登录截图为参考，识别当前登录流程状态。"""

    _SIZE = (384, 216)
    _VISIBLE_LOGIN_EDGES_B64 = (
        "eNrtVNEOgCAIvP//6WtJOCQBVw+tzatVenLEyQQaiDvYYD4HDMsQI+EkkiGwAJ7xE40+jZZCVvanvPWS24wnvIu8lEURLvNVtQrHvqh87h8rAxILbUnxHtBnojA6pbwdGY+cPru0lsa0PdL/J/L6N1/wK/3zY37v/3uez4+XSh6F/OftdQAXBlW5"
    )
    _SPECS = {
        "official_disconnect": ("1_1.png", "official", 43, (0.55, 0.50, 0.86, 0.87)),
        "official_login": ("1_2.png", "official", 43, (0.47, 0.28, 0.72, 0.70)),
        "official_server": ("1_3.png", "official", 43, (0.29, 0.14, 0.74, 0.44)),
        "official_channel": ("1_4.png", "official", 43, (0.33, 0.39, 0.71, 0.76)),
        "official_character": ("1_5.png", "official", 43, (0.32, 0.24, 0.82, 0.73)),
        "classic_login": ("2_1.png", "classic", 36, (0.44, 0.34, 0.75, 0.70)),
        "classic_server": ("2_2.png", "classic", 36, (0.31, 0.15, 0.75, 0.45)),
        "classic_channel": ("2_3.png", "classic", 36, (0.32, 0.35, 0.73, 0.77)),
        "classic_character": ("2_4.png", "classic", 36, (0.22, 0.22, 0.76, 0.72)),
    }
    # 登录页有云层、角色动画和负载条等动态内容。用户提供的 1.mp4 表明，
    # 动态角色页相对静态参考图的边缘相关度稳定在 0.30 左右；若所有页面共用
    # 0.42，会导致流程停在角色选择页。这里仍对结构稳定的页面使用高阈值，
    # 只为已通过视频回放验证的动态页设置较低、且互相仍有充分间隔的阈值。
    _STATE_THRESHOLDS = {
        "official_server": 0.34,
        "official_character": 0.26,
        "classic_server": 0.34,
        "classic_character": 0.26,
    }
    _DEFAULT_THRESHOLD = 0.42

    def __init__(self, reference_dir: str):
        self.reference_dir = os.path.abspath(reference_dir)
        self._references: Dict[str, Tuple[str, np.ndarray, Tuple[float, float, float, float]]] = {}
        self._error_template: Optional[np.ndarray] = None
        self._disconnect_popup_template: Optional[np.ndarray] = None
        self._classic_password_error_template: Optional[np.ndarray] = None
        self._visible_login_template: Optional[np.ndarray] = None
        self._load()

    @property
    def ready(self) -> bool:
        return bool(self._references)

    @staticmethod
    def _edges(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 60, 150)

    @staticmethod
    def _roi(image: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi
        return image[
            max(0, int(y1 * h)):min(h, int(y2 * h)),
            max(0, int(x1 * w)):min(w, int(x2 * w)),
        ]

    def _load(self) -> None:
        width, height = self._SIZE
        for state, (filename, profile, title_height, roi) in self._SPECS.items():
            image = cv2.imread(os.path.join(self.reference_dir, filename), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] <= title_height:
                continue
            content = image[title_height:, :]
            normalized = cv2.resize(content, (width, height), interpolation=cv2.INTER_AREA)
            edges = self._edges(normalized)
            self._references[state] = (profile, self._roi(edges, roi), roi)
            if state == "official_disconnect":
                # 弹窗可能位于右下或中央；只取图标和错误文字并全图搜索。
                self._disconnect_popup_template = edges[127:151, 226:299].copy()

        error_image = cv2.imread(
            os.path.join(self.reference_dir, "1_ERROR1.png"), cv2.IMREAD_COLOR
        )
        if error_image is not None:
            # 只取“当前登录游戏的账号会登出游戏”文字区域。整张羊皮纸和
            # 确定按钮也存在于普通断线弹窗，用它们匹配会把 1_1 误认为
            # 账号占用错误，错误触发五分钟等待。
            text_crop = error_image[50:135, 120:465]
            error_w = max(24, int(round(width * text_crop.shape[1] / 1922.0)))
            error_h = max(12, int(round(height * text_crop.shape[0] / 1083.0)))
            resized = cv2.resize(text_crop, (error_w, error_h), interpolation=cv2.INTER_AREA)
            self._error_template = self._edges(resized)

        password_error_image = cv2.imread(
            os.path.join(self.reference_dir, "2_ERROR_PASSWORD.png"), cv2.IMREAD_COLOR
        )
        if password_error_image is not None:
            # 用户实机弹窗中的“密码错误！请确认！”文字区域。先归一化
            # 整张客户区，再裁模板，与实时画面使用同一缩放路径。
            normalized = cv2.resize(
                password_error_image, (width, height), interpolation=cv2.INTER_AREA
            )
            self._classic_password_error_template = self._edges(normalized)[69:84, 164:215]
        # Only the 56x32 edge mask is retained; no account text or screenshot is stored.
        self._visible_login_template = np.frombuffer(
            zlib.decompress(base64.b64decode(self._VISIBLE_LOGIN_EDGES_B64)),
            dtype=np.uint8,
        ).reshape(56, 32)

    def classify(self, frame: Optional[np.ndarray], profile_mode: str = "auto") -> Optional[VisualMatch]:
        if frame is None or not self._references:
            return None
        try:
            normalized = cv2.resize(frame, self._SIZE, interpolation=cv2.INTER_AREA)
            edges = self._edges(normalized)
        except Exception:
            return None

        mode = str(profile_mode or "auto").strip().lower()
        if mode in ("官方", "official_web"):
            mode = "official"
        elif mode in ("单机", "classic_local"):
            mode = "classic"
        elif mode not in ("official", "classic"):
            mode = "auto"

        if mode in ("auto", "official") and self._error_template is not None:
            template = self._error_template
            if edges.shape[0] >= template.shape[0] and edges.shape[1] >= template.shape[1]:
                result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
                score = float(cv2.minMaxLoc(result)[1])
                if score >= 0.62:
                    return VisualMatch("official_account_online_error", "official", score)

        if mode in ("auto", "official") and self._disconnect_popup_template is not None:
            template = self._disconnect_popup_template
            if edges.shape[0] >= template.shape[0] and edges.shape[1] >= template.shape[1]:
                result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv2.minMaxLoc(result)
                if score >= 0.48:
                    # 模板左上角到“确定”中心的偏移取自 1_1 原图。
                    click_pos = ((x + 49.0) / self._SIZE[0], (y + 35.0) / self._SIZE[1])
                    return VisualMatch("official_disconnect", "official", float(score), click_pos)

        if mode in ("auto", "classic") and self._classic_password_error_template is not None:
            # 限定中央弹窗文字的搜索区域；普通登录页在此模板上的实测
            # 最大相关度为 0.23，错误弹窗为 1.00。
            center = edges[52:120, 120:260]
            template = self._classic_password_error_template
            if center.shape[0] >= template.shape[0] and center.shape[1] >= template.shape[1]:
                score = float(cv2.minMaxLoc(
                    cv2.matchTemplate(center, template, cv2.TM_CCOEFF_NORMED)
                )[1])
                if score >= 0.72:
                    return VisualMatch("classic_password_error", "classic", score)

        if mode in ("auto", "official") and self._visible_login_template is not None:
            result = cv2.matchTemplate(
                edges, self._visible_login_template, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, (x, y) = cv2.minMaxLoc(result)
            if score >= 0.65:
                click_pos = ((x + 52.0) / self._SIZE[0], (y + 59.0) / self._SIZE[1])
                return VisualMatch("official_login", "official", float(score), click_pos)

        best: Optional[VisualMatch] = None
        for state, (profile, reference, roi) in self._references.items():
            if mode != "auto" and profile != mode:
                continue
            current = self._roi(edges, roi)
            if current.shape != reference.shape or current.size == 0:
                continue
            score = float(cv2.matchTemplate(current, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
            if best is None or score > best.score:
                best = VisualMatch(state, profile, score)
        if best is None:
            return None
        threshold = self._STATE_THRESHOLDS.get(best.state, self._DEFAULT_THRESHOLD)
        return best if best.score >= threshold else None


class ReconnectController:
    """独立于战斗 FSM 的重连控制器；只在 F6 运行期间监测。"""

    def __init__(
        self,
        *,
        input_driver: Any,
        frame_getter: Callable[[], Optional[np.ndarray]],
        settings_getter: Callable[[], Dict[str, Any]],
        is_f6_running: Callable[[], bool],
        emergency_stop: Callable[[], Any],
        prepare_game_entry: Optional[Callable[[], None]],
        is_game_ready: Callable[[], bool],
        request_resume: Callable[[], None],
        status_callback: Callable[[str, bool], None],
        log_callback: Callable[[str], None],
        reference_dir: str,
        visible_frame_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        set_visible_capture: Optional[Callable[[bool], None]] = None,
        incident_callback: Optional[Callable[[], None]] = None,
    ):
        self.driver = input_driver
        self.frame_getter = frame_getter
        self.visible_frame_getter = visible_frame_getter
        self.set_visible_capture = set_visible_capture
        self.incident_callback = incident_callback
        self._using_visible_capture = False
        self.settings_getter = settings_getter
        self.is_f6_running = is_f6_running
        self.emergency_stop = emergency_stop
        self.prepare_game_entry = prepare_game_entry
        self.is_game_ready = is_game_ready
        self.request_resume = request_resume
        self.status_callback = status_callback
        self.log = log_callback
        self.vision = ReconnectVision(reference_dir)

        self.active = False
        self.disconnected = False
        self._start_requested = False
        self._start_requested_at = 0.0
        self.profile: Optional[str] = None
        self.visual_state: Optional[str] = None
        self._visual_hits = 0
        self._last_candidate: Optional[str] = None
        self._action_stage = 0
        self._next_action_at = 0.0
        self._wait_until = 0.0
        self._waiting_error_seen = False
        self._input_quiet = True
        self._password_error_paused = False
        self._password_credential_token = ""
        self._login_submit_count = 0
        self._login_submitted_at = 0.0
        self._character_start_at: Optional[float] = None
        self._game_ready_hits = 0
        self._resume_requested = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_status = ""
        self._last_wait_log_at = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="ReconnectFSM", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._set_visible_capture_mode(False)

    def cancel(self, reason: str = "用户取消") -> bool:
        with self._lock:
            if not self.active and not self._start_requested:
                return False
            self.active = False
            self._start_requested = False
            self._resume_requested = False
            self.visual_state = None
            self._character_start_at = None
            self._game_ready_hits = 0
            self._wait_until = 0.0
            self._password_error_paused = False
            self._login_submit_count = 0
            self._login_submitted_at = 0.0
        self.driver.release_all_keys()
        self._set_visible_capture_mode(False)
        self._set_status(f"重连已取消：{reason}", False)
        self.log(f"🛑 [自动重连] {reason}，不会自动恢复 F6")
        return True

    @property
    def starting(self) -> bool:
        return self._start_requested

    def request_start_from_f6(self) -> bool:
        """登录/掉线页按 F6 时先接管重连，不运行地图路线预检。"""
        settings = self._settings()
        if not settings.get("reconnect_enabled", False) or not self.vision.ready:
            return False
        _, match = self._observe(settings)
        if match is None:
            return False
        with self._lock:
            if self.active:
                return True
            self.disconnected = True
            self._start_requested = True
            self._start_requested_at = time.monotonic()
        self.driver.release_all_keys()
        self._last_candidate = match.state
        self._visual_hits = 1
        self.log(f"🔌 [F6 重连启动] 识别到 {match.state}（{match.score:.3f}），跳过地图路线预检")
        self._set_status("重连中：准备接管输入", True)
        return True

    def _set_status(self, text: str, active: bool = True) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        try:
            self.status_callback(text, active)
        except Exception:
            pass

    def _settings(self) -> Dict[str, Any]:
        try:
            return dict(self.settings_getter() or {})
        except Exception:
            return {}

    def _set_visible_capture_mode(self, enabled: bool) -> None:
        if self._using_visible_capture == bool(enabled):
            return
        self._using_visible_capture = bool(enabled)
        if self.set_visible_capture is not None:
            self.set_visible_capture(bool(enabled))
        if enabled:
            self.log("[Reconnect] Switching to foreground login capture")

    def _observe(self, settings: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[VisualMatch]]:
        profile_mode = settings.get("reconnect_client_profile", "auto")
        frame = self.frame_getter()
        match = self.vision.classify(frame, profile_mode)
        if match is None and self.visible_frame_getter is not None:
            visible = self.visible_frame_getter()
            visible_match = self.vision.classify(visible, profile_mode)
            if visible_match is not None:
                self._set_visible_capture_mode(True)
                return visible, visible_match
        return frame, match

    def _activate(self, match: VisualMatch) -> None:
        was_f6_running = bool(self.is_f6_running())
        try:
            self.driver.release_all_keys()
        finally:
            quiet = self.emergency_stop()
            self.driver.release_all_keys()
        with self._lock:
            self.active = True
            self.disconnected = True
            self._start_requested = False
            self.profile = match.profile
            self.visual_state = match.state
            self._action_stage = 0
            self._next_action_at = 0.0
            self._wait_until = 0.0
            self._waiting_error_seen = False
            self._input_quiet = quiet is not False
            self._password_error_paused = False
            self._password_credential_token = ""
            self._login_submit_count = 0
            self._login_submitted_at = 0.0
            self._character_start_at = None
            self._game_ready_hits = 0
            self._resume_requested = False
        self.log(
            f"🔌 [掉线确认] 连续识别到 {match.state}（{match.score:.3f}）；"
            "已释放全部按键并停止战斗、巡逻与移动状态机"
        )
        if was_f6_running and self.incident_callback is not None:
            try:
                self.incident_callback()
            except Exception:
                pass
        self._set_status("重连中：已接管输入", True)

    def _stable_match(self, match: Optional[VisualMatch]) -> Optional[VisualMatch]:
        candidate = match.state if match is not None else None
        if candidate == self._last_candidate:
            self._visual_hits += 1
        else:
            self._last_candidate = candidate
            self._visual_hits = 1
        return match if match is not None and self._visual_hits >= 2 else None

    @staticmethod
    def _bounded_index(settings: Dict[str, Any], key: str, total_key: str, default: int) -> Tuple[int, int]:
        try:
            total = max(1, min(200, int(settings.get(total_key, default))))
        except (TypeError, ValueError):
            total = default
        try:
            index = max(1, min(total, int(settings.get(key, 1))))
        except (TypeError, ValueError):
            index = 1
        return index, total

    def _click_normalized(self, frame: np.ndarray, nx: float, ny: float) -> bool:
        h, w = frame.shape[:2]
        return bool(self.driver.click_client(int(nx * w), int(ny * h)))

    def _select_server(self, frame: np.ndarray, settings: Dict[str, Any]) -> None:
        index, total = self._bounded_index(
            settings, "reconnect_server_index", "reconnect_server_total", 5
        )
        if self.profile == "official":
            # 官方怀旧服横向 5 个世界；按实际总数均匀落在横板区域。
            left, right, y = 0.365, 0.676, 0.215
        else:
            # 经典客户端的世界牌更窄，最多可横排多个世界。
            left, right, y = 0.381, 0.681, 0.296
        ratio = 0.0 if total <= 1 else (index - 1) / (total - 1)
        self._click_normalized(frame, left + (right - left) * ratio, y)
        self.log(f"🔌 [自动重连] 选择区服 {index}/{total}")

    def _select_channel(self, frame: np.ndarray, settings: Dict[str, Any]) -> None:
        index, total = self._bounded_index(
            settings, "reconnect_channel_index", "reconnect_channel_total", 20
        )
        page = (index - 1) // 20
        local = (index - 1) % 20
        row, col = divmod(local, 4)
        if self.profile == "official":
            xs = (0.409, 0.478, 0.546, 0.615)
            ys = (0.505, 0.548, 0.590, 0.633, 0.668)
            enter = (0.617, 0.443)
            wheel_at = (0.655, 0.600)
        else:
            xs = (0.394, 0.468, 0.542, 0.616)
            ys = (0.483, 0.528, 0.573, 0.618, 0.663)
            enter = (0.632, 0.722)
            wheel_at = (0.600, 0.580)
        if self._action_stage == 0:
            if page:
                h, w = frame.shape[:2]
                self.driver.mouse_wheel_client(
                    int(wheel_at[0] * w), int(wheel_at[1] * h), -page * 5
                )
                time.sleep(0.15)
            self._click_normalized(frame, xs[col], ys[row])
            self._action_stage = 1
            self._next_action_at = time.monotonic() + 0.55
            self.log(f"🔌 [自动重连] 选择频道 {index}/{total}")
        else:
            self._click_normalized(frame, *enter)
            self._action_stage = 0
            self._next_action_at = time.monotonic() + 4.0

    def _select_character(self, frame: np.ndarray, settings: Dict[str, Any]) -> None:
        index, total = self._bounded_index(
            settings, "reconnect_character_index", "reconnect_character_total", 3
        )
        if self.profile == "official":
            slots, y, start = (0.427, 0.519, 0.611), 0.607, (0.746, 0.365)
        else:
            slots, y, start = (0.382, 0.480, 0.577), 0.567, (0.674, 0.307)
        # 两套客户端都固定显示三个槽位。不能按“人物总数”拉伸坐标：只有
        # 两个人物时，第二个人物仍在中间槽，而不是最右槽。
        slot_index = max(0, min(len(slots) - 1, index - 1))
        if self._action_stage == 0:
            self._click_normalized(frame, slots[slot_index], y)
            self._action_stage = 1
            self._next_action_at = time.monotonic() + 0.45
            self.log(f"🔌 [自动重连] 选择人物 {index}/{total}")
        else:
            if self.prepare_game_entry is not None:
                try:
                    self.prepare_game_entry()
                except Exception as exc:
                    self.log(f"⚠️ [自动重连] 清理旧地图定位失败：{exc}")
            self._click_normalized(frame, *start)
            self._action_stage = 0
            self._next_action_at = time.monotonic() + 6.0
            # 1.mp4 中点击“开始游戏”后会先停留在人物页，随后黑屏，再进入
            # 地图。这个时间戳不能因为人物页仍可见或进入黑屏而被清空。
            self._character_start_at = time.monotonic()
            self._set_status("重连中：等待进入地图", True)

    def _handle_account_online_error(self, settings: Dict[str, Any]) -> None:
        now = time.monotonic()
        if not self._waiting_error_seen:
            self.driver.press_key("enter", duration_ms=60)
            try:
                wait_sec = max(
                    0.0, min(3600.0, float(settings.get("reconnect_account_online_wait_sec", 300.0)))
                )
            except (TypeError, ValueError):
                wait_sec = 300.0
            self._wait_until = now + wait_sec
            self._waiting_error_seen = True
            self.log(
                f"⏳ [自动重连] 检测到账号仍在线提示，已确认弹窗；"
                f"暂停 {wait_sec:.0f} 秒后再继续"
            )
        remaining = max(0.0, self._wait_until - now)
        self._set_status(f"重连等待：{remaining:.0f}秒", True)

    def _pause_password_error(self, detail: str) -> None:
        self.driver.release_all_keys()
        if not self._password_error_paused:
            self.log(f"🛑 [自动重连密码错误] {detail}；停止重复提交，等待修改重连设置中的密码")
        self._password_error_paused = True
        self._set_status("重连暂停：密码错误，请在重连设置中更新密码", True)

    def _resume_after_password_update(
        self, settings: Dict[str, Any], match: Optional[VisualMatch]
    ) -> bool:
        """错误状态下只接受用户保存的新密码，不继续重发旧密码。"""
        if not self._password_error_paused:
            return False
        credential = str(settings.get("reconnect_password_protected", "") or "")
        if not credential or credential == self._password_credential_token:
            self._set_status("重连暂停：密码错误，请在重连设置中更新密码", True)
            return True
        if not unprotect_secret(credential):
            self._set_status("重连暂停：新密码无法解密，请重新保存", True)
            return True
        if match is None:
            self._set_status("重连中：新密码已保存，等待确认登录画面", True)
            return True
        self._password_error_paused = False
        self._password_credential_token = credential
        self._login_submit_count = 0
        self._login_submitted_at = 0.0
        if match.state == "classic_password_error":
            self.driver.press_key("enter", duration_ms=60)
        self._next_action_at = time.monotonic() + 0.6
        self.log("🔑 [自动重连] 检测到密码设置已更新，确认旧弹窗并重新登录")
        return True

    def _act(self, match: VisualMatch, frame: np.ndarray, settings: Dict[str, Any]) -> None:
        now = time.monotonic()
        if match.state == "classic_password_error":
            self._pause_password_error("识别到客户端密码错误弹窗")
            return
        if match.state == "official_account_online_error":
            self._handle_account_online_error(settings)
            return
        if self._wait_until > now:
            remaining = self._wait_until - now
            self._set_status(f"重连等待：{remaining:.0f}秒", True)
            if now - self._last_wait_log_at >= 30.0:
                self.log(f"⏳ [自动重连] 账号占用保护剩余 {remaining:.0f} 秒")
                self._last_wait_log_at = now
            return
        if self._waiting_error_seen:
            self._waiting_error_seen = False
            self._wait_until = 0.0
            self.log("▶️ [自动重连] 账号占用等待结束，恢复登录流程")
        if now < self._next_action_at:
            return

        state = match.state
        self._set_status(f"重连中：{state}", True)
        if state == "official_disconnect":
            self._click_normalized(frame, *(match.click_pos or (0.716, 0.751)))
            self._next_action_at = now + 2.0
        elif state == "official_login":
            self._click_normalized(frame, *(match.click_pos or (0.597, 0.517)))
            self._next_action_at = now + 4.0
        elif state == "classic_login":
            if self._login_submitted_at > 0.0:
                if now - self._login_submitted_at < 8.0:
                    return
                if self._login_submit_count >= 2:
                    self._pause_password_error("两次提交后仍停留在登录页")
                    return
            password = unprotect_secret(settings.get("reconnect_password_protected", ""))
            if not password:
                self._set_status("重连暂停：经典客户端密码未设置", True)
                self._next_action_at = now + 5.0
                return
            # 点在密码框右侧：即使客户端忽略 End，退格也能从已有密码末尾清空。
            if not self._click_normalized(frame, 0.635, 0.463):
                self._set_status("重连输入异常：密码框未聚焦", True)
                self._next_action_at = now + 1.5
                return
            time.sleep(0.08)
            if not self.driver.replace_text(password):
                self._set_status("重连输入异常：密码未完整送达", True)
                self._next_action_at = now + 1.5
                return
            time.sleep(0.08)
            if not self._click_normalized(frame, 0.689, 0.425):
                self._set_status("重连输入异常：连接按钮未点击", True)
                self._next_action_at = now + 1.5
                return
            self._password_credential_token = str(
                settings.get("reconnect_password_protected", "") or ""
            )
            self._login_submit_count += 1
            self._login_submitted_at = time.monotonic()
            self.log(f"🔑 [自动重连] 已清空密码框并提交登录（第{self._login_submit_count}次）")
            self._next_action_at = now + 1.0
        elif state.endswith("_server"):
            self._login_submitted_at = 0.0
            self._login_submit_count = 0
            self._select_server(frame, settings)
            self._next_action_at = now + 4.0
        elif state.endswith("_channel"):
            self._select_channel(frame, settings)
        elif state.endswith("_character"):
            self._select_character(frame, settings)

    def _finish_if_ready(self, match: Optional[VisualMatch]) -> None:
        now = time.monotonic()
        if self._character_start_at is None:
            return
        # 仍处于任一登录流程页面时绝不能恢复；无匹配（包括视频中的黑屏）
        # 只代表正在切图，最终还必须由地图定位链路确认黄点和世界坐标已恢复。
        if match is not None:
            self._game_ready_hits = 0
            return
        if now - self._character_start_at < 5.0 or not self.is_game_ready():
            self._game_ready_hits = 0
            return
        self._game_ready_hits += 1
        if self._game_ready_hits < 3:
            return
        with self._lock:
            self.active = False
            self.disconnected = False
            self._start_requested = False
            self._resume_requested = True
            self.visual_state = None
            self._character_start_at = None
            self._game_ready_hits = 0
        self.log("✅ [自动重连] 已重新进入地图并恢复定位，准备恢复断线前的 F6")
        self._set_status("重连成功：正在恢复F6", False)
        self._set_visible_capture_mode(False)
        self.request_resume()

    def _run_once(self) -> None:
        settings = self._settings()
        if not bool(settings.get("reconnect_enabled", False)):
            self._set_visible_capture_mode(False)
            if not self.active:
                self.disconnected = False
                self._start_requested = False
                self._last_candidate = None
                self._visual_hits = 0
            return
        # F6 停止时仍低频识别登录页并暂停怪物扫描；只有 F6 才会执行登录动作。
        frame, match = self._observe(settings)
        if frame is None:
            return
        stable = self._stable_match(match)

        if not self.active:
            if stable is not None and (self.is_f6_running() or self._start_requested):
                self.disconnected = True
                self._activate(stable)
            elif stable is not None and not self.disconnected:
                self.disconnected = True
                self.driver.release_all_keys()
                self._input_quiet = self.emergency_stop() is not False
                self.driver.release_all_keys()
                self.log(f"🔌 [掉线待命] 检测到 {stable.state}（{stable.score:.3f}），已停怪物识别；按 F6 开始重连")
                self._set_status("检测到掉线／登录界面，按 F6 重连", False)
            elif stable is None and self._start_requested:
                if time.monotonic() - self._start_requested_at >= 5.0:
                    self._start_requested = False
                    self.log("⚠️ [F6 重连启动] 5秒内未再次确认登录画面，未启动 F6")
                    self._set_status("重连待命：登录画面未确认", False)
            elif match is None and self.disconnected and self._visual_hits >= 2:
                self.disconnected = False
                self._set_visible_capture_mode(False)
                self._set_status("断线重连待命", False)
            return

        if not self._input_quiet:
            quiet = self.emergency_stop()
            self._input_quiet = quiet is not False
            if not self._input_quiet:
                self._set_status("重连中：等待原 F6 按键线程退出", True)
                return
            self.driver.release_all_keys()

        if self._resume_after_password_update(settings, stable):
            return

        if (
            self.profile == "classic"
            and self._login_submitted_at > 0.0
            and stable is None
            and time.monotonic() - self._login_submitted_at >= 20.0
        ):
            self._pause_password_error("提交后20秒未识别到下一登录页面，请检查客户端提示")
            return

        self._finish_if_ready(match)
        if not self.active or stable is None:
            return
        if stable.state != self.visual_state:
            self.visual_state = stable.state
            self.profile = stable.profile
            self._action_stage = 0
            self._next_action_at = 0.0
            self.log(
                f"🔎 [自动重连状态] {stable.state}（匹配={stable.score:.3f}）"
            )
        self._act(stable, frame, settings)

    def _run(self) -> None:
        while not self._stop_event.wait(
            0.25 if (self.active or self._start_requested or self.is_f6_running()) else 0.75
        ):
            try:
                self._run_once()
            except Exception as exc:
                # 登录页输入属于外部窗口操作；单次 SendInput/窗口焦点异常
                # 只能让本轮重试延后，绝不能终止整个重连线程。
                try:
                    self.driver.release_all_keys()
                except Exception:
                    pass
                self._next_action_at = time.monotonic() + 1.5
                self._set_status("重连输入异常：即将自动重试", True)
                self.log(
                    f"⚠️ [自动重连异常] {type(exc).__name__}: {exc}；"
                    "状态机仍在运行，1.5秒后重试当前页面"
                )
