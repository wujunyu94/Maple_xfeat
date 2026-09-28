"""
app.py - 游戏视觉监控与识别中枢 GUI
功能:
  - 自定义怪物增删改（iid 基于 mob_id 唯一，删除/更新稳定不错位）
  - 模糊搜索怪物名称（中文/英文），关联在线怪物资料
  - 鼠标悬浮怪物精灵图大图预览
  - 全键盘交互式按键录制（全覆盖）
  - 全自动 OCR 地图感知 + 本地优先永久存储
"""

import sys
import os
import json
import glob
import re
import time
import math
import threading
import queue
import ctypes
from ctypes import wintypes
import base64
import traceback
from dataclasses import replace
from io import BytesIO
from typing import Optional, Dict, List, Tuple, Any
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from src.gui.system_tray import SystemTray
from src.gui.log_snapshot import save_log_snapshot
import cv2
import numpy as np
import requests
import urllib.request

user32 = ctypes.windll.user32
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_F11 = 0x7A
VK_UP = 0x26

from src.core.window import WindowManager
from src.core.performance_timing import PerformanceTimingLog
from src.core.input_driver import InputDriver
from src.core.manual_input_recorder import ManualInputRecorder
from src.vision.capture import ScreenCapture
from src.vision.main_view_detector import (
    MainViewDetector,
    MainViewResult,
    MonsterDetectionBatch,
    MOB_ALIAS_MAP,
    CLASSIC_MOB_ID_MAP,
)
from src.vision.tracker import MinimapTracker
from src.vision.world_filter import WorldCoordinateKalman, InputAwareHorizontalKalman
from src.vision.horizontal_motion_model import HorizontalMotionModel
from src.vision.asset_downloader import AssetDownloader
from src.vision.map_resolver import MapResolver
from src.vision.wz_map_reader import WzMapReader
from src.vision.wz_skill_reader import load_skill_hitbox, search_skill_names
from src.vision.ladder_aligner import LadderAligner
from src.vision.status_bar_reader import (
    ExperienceRateTracker,
    StatusBarReader,
    StatusBarReading,
)
from src.vision.potion_stock_reader import (
    PotionStockMonitor, PotionStockReader, supports_stock_key,
)
from src.vision.death_dialog_detector import DeathDialogDetector
from src.gui.mob_template_editor import MobTemplateEditor
from src.gui.window_layout import (
    fit_image_to_work_area,
    fit_window_to_work_area,
    get_work_area,
)
from src.minigame.bridge import MiniGameBridge
from src.minigame.result_dialog import ResultDialogConfirmer, ResultDialogDetector
from src.minigame.session_recorder import MiniGameSessionRecorder
from src.minigame.video_test import MiniGameVideoTest
from src.engine.motion_controller import MotionController
from src.engine.ladder_grab_test_runner import LadderGrabTestConfig, LadderGrabTestRunner
from src.engine.random_path_test_runner import RandomPathTestRunner
from src.engine.waypoint_manager import WaypointManager
from src.engine.combat_fsm import CombatFSM, BotState
from src.engine.death_recovery import DeathRecoveryController
from src.engine.attack_skills import skills_from_config
from src.engine.pet_feeder import PetFeedTimer
from src.engine.reconnect_fsm import (
    ReconnectController,
    protect_secret,
    unprotect_secret,
)
from src.engine.platform_graph import PlatformGraphBuilder, PlatformGraph, PlatformNode, PlatformEdge
from src.engine.world_patrol import (
    CROSS_MAP_PORTAL_TYPES,
    WorldPatrolController,
    WorldPatrolStop,
    WorldRoutePlanner,
    parse_world_patrol_stops,
)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.json")


# ─────────────────────────────────────────────────────────────────────────────
#  PlatformTopologyDialog：当前地图平台与梯绳拓扑全景图弹窗
# ─────────────────────────────────────────────────────────────────────────────
class PlatformTopologyDialog:
    def __init__(self, parent: tk.Tk, get_graph_fn, get_player_pos_fn,
                 get_patrol_platforms_fn, get_active_path_fn=None,
                 on_merge_platforms_changed=None, on_calibrate_x=None, on_calibrate_y=None,
                 on_delete_calibration=None, initial_merged: bool = True):
        self.parent = parent
        self.get_graph_fn = get_graph_fn
        self.get_player_pos_fn = get_player_pos_fn
        self.get_patrol_platforms_fn = get_patrol_platforms_fn
        self.get_active_path_fn = get_active_path_fn or (lambda: [])
        self.on_merge_platforms_changed = on_merge_platforms_changed
        self.on_calibrate_x = on_calibrate_x
        self.on_calibrate_y = on_calibrate_y
        self.on_delete_calibration = on_delete_calibration
        self.is_open = True
        self.current_zoom = 0.85  # 默认 85% 高清特大显示，彻底告别微型缩略图
        self.is_auto_fit = False
        self.base_bgr_img: Optional[np.ndarray] = None
        self.last_map_id: Optional[int] = None
        self.last_patrol: Optional[List[int]] = None
        self.last_active_path = None
        self._render_generation = 0
        self.show_platform_edges = tk.BooleanVar(value=False)
        self.show_merged_platforms = tk.BooleanVar(value=initial_merged)
        # 仅影响 Canvas 视口；不重绘底图、不改动拓扑世界坐标。
        self.lock_view_to_player = tk.BooleanVar(value=False)

        self.scale_factor = 0.85
        self.offset_x = 0
        self.offset_y = 0

        self.top = tk.Toplevel(parent)
        self.top.title("🗺️ 当前地图平台与梯绳拓扑全景图 (实时角色高亮)")
        fit_window_to_work_area(self.top, (1180, 750), (800, 500), parent=parent)
        self.top.configure(bg="#16181d")

        # 顶部工具栏
        tb = tk.Frame(self.top, bg="#1a1c23", height=42)
        tb.pack(fill=tk.X, side=tk.TOP, padx=8, pady=6)
        tb2 = tk.Frame(self.top, bg="#1a1c23", height=32)
        tb2.pack(fill=tk.X, side=tk.TOP, padx=8, pady=(0, 4))
        tb3 = tk.Frame(self.top, bg="#1a1c23", height=32)
        tb3.pack(fill=tk.X, side=tk.TOP, padx=8, pady=(0, 4))

        self.lbl_info = tk.Label(tb, text="正在加载地图拓扑数据...", font=("Segoe UI", 10, "bold"),
                                 fg="#00e5ff", bg="#1a1c23")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        self.lbl_player_status = tk.Label(tb, text="📍 角色定位: 探测中...", font=("Segoe UI", 9),
                                          fg="#69f0ae", bg="#1a1c23")
        self.lbl_player_status.pack(side=tk.LEFT, padx=10)

        tk.Checkbutton(
            tb2, text="显示平台连线", variable=self.show_platform_edges,
            command=lambda: self.refresh_view(force_rebuild=True),
            font=("Segoe UI", 9), fg="#e0e0e0", bg="#1a1c23",
            activeforeground="#ffffff", activebackground="#1a1c23",
            selectcolor="#2a2a32", relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=8)

        tk.Checkbutton(
            tb2, text="合并短平台", variable=self.show_merged_platforms,
            command=self._toggle_merged_platforms,
            font=("Segoe UI", 9), fg="#e0e0e0", bg="#1a1c23",
            activeforeground="#ffffff", activebackground="#1a1c23",
            selectcolor="#2a2a32", relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=8)

        tk.Checkbutton(
            tb2, text="视角锁定玩家", variable=self.lock_view_to_player,
            command=self._toggle_player_view_lock,
            font=("Segoe UI", 9), fg="#e0e0e0", bg="#1a1c23",
            activeforeground="#ffffff", activebackground="#1a1c23",
            selectcolor="#2a2a32", relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=8)

        # 常用操作统一放在第二行，避免第一行地图状态信息过于拥挤。
        tk.Button(tb3, text="聚焦角色", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#2e7d32",
                  relief=tk.FLAT, padx=8, pady=2, command=self.center_on_player).pack(side=tk.LEFT, padx=4)

        tk.Button(tb3, text="标定 X 轴", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#6a1b9a",
                  relief=tk.FLAT, padx=8, pady=2, command=self._open_x_calibration).pack(side=tk.LEFT, padx=3)

        tk.Button(tb3, text="标定 Y 轴", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#ad6b00",
                  relief=tk.FLAT, padx=8, pady=2, command=self._open_y_calibration).pack(side=tk.LEFT, padx=3)

        tk.Button(tb3, text="删除标定", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#b71c1c",
                  relief=tk.FLAT, padx=8, pady=2, command=self._confirm_delete_calibration).pack(side=tk.LEFT, padx=3)

        tk.Button(tb3, text="刷新", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#0277bd",
                  relief=tk.FLAT, padx=8, pady=2, command=lambda: self.refresh_view(force_rebuild=True)).pack(side=tk.LEFT, padx=3)

        tk.Button(tb, text="100% 原图", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=6, pady=2, command=lambda: self.set_scale(1.0)).pack(side=tk.RIGHT, padx=2)

        tk.Button(tb, text="75%", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=6, pady=2, command=lambda: self.set_scale(0.75)).pack(side=tk.RIGHT, padx=2)

        tk.Button(tb, text="50%", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=6, pady=2, command=lambda: self.set_scale(0.50)).pack(side=tk.RIGHT, padx=2)

        tk.Button(tb, text="🌐 适应全图", font=("Segoe UI", 8), fg="#b0bec5", bg="#2a2a32",
                  relief=tk.FLAT, padx=6, pady=2, command=self.fit_to_window).pack(side=tk.RIGHT, padx=2)

        tk.Button(tb, text="➖ 缩小", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=6, pady=2, command=self.zoom_out).pack(side=tk.RIGHT, padx=2)

        tk.Button(tb, text="➕ 放大", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=6, pady=2, command=self.zoom_in).pack(side=tk.RIGHT, padx=2)

        self.lbl_zoom = tk.Label(tb, text="缩放: 85%", font=("Consolas", 8, "bold"), fg="#00e5ff", bg="#1a1c23")
        self.lbl_zoom.pack(side=tk.RIGHT, padx=6)

        # 图像展示区 (带水平与垂直滚动条)
        self.container = tk.Frame(self.top, bg="#16181d")
        self.container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(self.container, bg="#121418", highlightthickness=0)
        self.h_sb = ttk.Scrollbar(self.container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.v_sb = ttk.Scrollbar(self.container, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=self.h_sb.set, yscrollcommand=self.v_sb.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_sb.grid(row=0, column=1, sticky="ns")
        self.h_sb.grid(row=1, column=0, sticky="ew")

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.img_tk: Optional[ImageTk.PhotoImage] = None

        # 动态 Canvas 矢量图元 ID
        self.base_image_item = None
        self.player_glow_item = None
        self.player_dot_item = None
        self.player_tag_bg_item = None
        self.player_tag_text_item = None
        self.topology_label_items = []

        # 绑定鼠标拖拽平移 (按住左键/右键/中键任意拖动)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<ButtonPress-3>", self._on_drag_start)
        self.canvas.bind("<B3-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_drag_end)

        # 绑定鼠标滚轮缩放与窗口尺寸变化
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.container.bind("<Configure>", self._on_container_resized)

        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        self.top.update_idletasks()
        self.refresh_view(force_rebuild=True)
        # 初始自动聚焦角色位置
        self.top.after(300, self.center_on_player)
        self._schedule_auto_refresh()

    def _on_drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_drag_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_drag_end(self, event):
        self.canvas.config(cursor="")

    def on_close(self):
        self.is_open = False
        self.top.destroy()

    def _toggle_merged_platforms(self):
        if self.on_merge_platforms_changed:
            self.on_merge_platforms_changed(bool(self.show_merged_platforms.get()))

    def _toggle_player_view_lock(self):
        """启用时立即居中，随后由 100ms UI 刷新持续跟随 YOU。"""
        if self.lock_view_to_player.get():
            self.top.after_idle(self.center_on_player)

    def _open_x_calibration(self):
        """弹出绳梯编号输入框，采集当前黄点并标定 X 轴卷轴偏移。"""
        if not self.on_calibrate_x:
            messagebox.showinfo("提示", "X 轴标定功能尚未连接。", parent=self.top)
            return

        win = tk.Toplevel(self.top)
        win.title("📐 X 轴卷轴标定")
        fit_window_to_work_area(win, (360, 180), (320, 150), parent=self.top)
        win.resizable(True, True)
        win.configure(bg="#1a1c23")
        win.transient(self.top)
        win.grab_set()

        tk.Label(win, text="请输入当前角色所在的绳梯编号", fg="#e0e0e0", bg="#1a1c23",
                 font=("Segoe UI", 10, "bold")).pack(pady=(16, 6))
        entry = tk.Entry(win, width=18, justify=tk.CENTER, font=("Consolas", 12))
        entry.pack()
        entry.focus_set()

        buttons = tk.Frame(win, bg="#1a1c23")
        buttons.pack(pady=14)

        def confirm():
            raw = entry.get().strip().lstrip("#")
            try:
                rope_id = int(raw)
                if rope_id <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("输入无效", "请输入正整数绳梯编号。", parent=win)
                return
            ok, msg = self.on_calibrate_x(rope_id)
            if ok:
                win.grab_release()
                win.destroy()
                messagebox.showinfo("标定完成", msg, parent=self.top)
            else:
                messagebox.showwarning("标定失败", msg, parent=win)

        tk.Button(buttons, text="确认", width=10, command=confirm,
                  bg="#2e7d32", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="取消", width=10, command=lambda: (win.grab_release(), win.destroy()),
                  bg="#424242", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        win.bind("<Return>", lambda _e: confirm())
        win.bind("<Escape>", lambda _e: (win.grab_release(), win.destroy()))

    def _open_y_calibration(self):
        """弹出平台编号输入框，采集当前黄点并标定 Y 轴卷轴偏移。"""
        if not self.on_calibrate_y:
            messagebox.showinfo("提示", "Y 轴标定功能尚未连接。", parent=self.top)
            return

        win = tk.Toplevel(self.top)
        win.title("📐 Y 轴卷轴标定")
        fit_window_to_work_area(win, (420, 210), (360, 170), parent=self.top)
        win.resizable(True, True)
        win.configure(bg="#1a1c23")
        win.transient(self.top)
        win.grab_set()

        tk.Label(win, text="请输入当前角色站立的平台编号", fg="#e0e0e0", bg="#1a1c23",
                 font=("Segoe UI", 10, "bold")).pack(pady=(16, 4))
        tk.Label(win, text="请保证平台水平，非斜坡。", fg="#ffb74d", bg="#1a1c23",
                 font=("Segoe UI", 9)).pack(pady=(0, 7))
        entry = tk.Entry(win, width=18, justify=tk.CENTER, font=("Consolas", 12))
        entry.pack()
        entry.focus_set()

        buttons = tk.Frame(win, bg="#1a1c23")
        buttons.pack(pady=14)

        def confirm():
            raw = entry.get().strip().lstrip("#Pp")
            try:
                platform_id = int(raw)
                if platform_id <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("输入无效", "请输入正整数平台编号。", parent=win)
                return
            ok, msg = self.on_calibrate_y(platform_id)
            if ok:
                win.grab_release()
                win.destroy()
                messagebox.showinfo("标定完成", msg, parent=self.top)
            else:
                messagebox.showwarning("标定失败", msg, parent=win)

        tk.Button(buttons, text="确认", width=10, command=confirm,
                  bg="#2e7d32", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="取消", width=10, command=lambda: (win.grab_release(), win.destroy()),
                  bg="#424242", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        win.bind("<Return>", lambda _e: confirm())
        win.bind("<Escape>", lambda _e: (win.grab_release(), win.destroy()))

    def _confirm_delete_calibration(self):
        """仅删除当前地图的本地 X/Y 标定样本。"""
        if not self.on_delete_calibration:
            messagebox.showinfo("提示", "删除标定功能尚未连接。", parent=self.top)
            return
        graph = self.get_graph_fn()
        if graph is None:
            messagebox.showwarning("提示", "当前地图拓扑尚未载入。", parent=self.top)
            return
        confirmed = messagebox.askokcancel(
            "确认删除标定",
            f"将删除当前地图的全部 X/Y 标定数据：\n\n"
            f"地图：{graph.map_name or '未知地图'}\n"
            f"MapID：{graph.map_id}\n\n"
            "只会删除该 MapID 对应的本地标定数据，其他地图不受影响。",
            parent=self.top,
        )
        if not confirmed:
            return
        ok, msg = self.on_delete_calibration()
        if ok:
            messagebox.showinfo("已删除", msg, parent=self.top)
        else:
            messagebox.showwarning("删除失败", msg, parent=self.top)

    def _schedule_auto_refresh(self):
        if self.is_open:
            self._update_player_marker()
            if self.lock_view_to_player.get():
                self.center_on_player()
            self.top.after(100, self._schedule_auto_refresh)

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _on_container_resized(self, event=None):
        if self.is_auto_fit and self.base_bgr_img is not None:
            self._render_scaled_canvas()

    def zoom_in(self):
        self.is_auto_fit = False
        self.current_zoom = min(2.5, self.current_zoom * 1.2)
        self.lbl_zoom.config(text=f"缩放: {int(self.current_zoom*100)}%")
        self._render_scaled_canvas()

    def zoom_out(self):
        self.is_auto_fit = False
        self.current_zoom = max(0.20, self.current_zoom * 0.8)
        self.lbl_zoom.config(text=f"缩放: {int(self.current_zoom*100)}%")
        self._render_scaled_canvas()

    def set_scale(self, scale: float):
        self.is_auto_fit = False
        self.current_zoom = scale
        self.lbl_zoom.config(text=f"缩放: {int(self.current_zoom*100)}%")
        self._render_scaled_canvas()

    def fit_to_window(self):
        self.is_auto_fit = True
        self.lbl_zoom.config(text="缩放: 自适应")
        self._render_scaled_canvas()

    def center_on_player(self):
        """视口平滑滚动并自动居中在当前角色所在位置"""
        graph: Optional[PlatformGraph] = self.get_graph_fn()
        player_pos = self.get_player_pos_fn()
        if not graph or not player_pos or self.base_bgr_img is None:
            return

        ih, iw = self.base_bgr_img.shape[:2]
        ix, iy = graph.world_to_render_pixel(player_pos[0], player_pos[1], iw, ih)
        cx = int(ix * self.scale_factor + self.offset_x)
        cy = int(iy * self.scale_factor + self.offset_y)

        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        dw = max(10, int(iw * self.scale_factor))
        dh = max(10, int(ih * self.scale_factor))

        total_w = max(cw, dw + self.offset_x * 2)
        total_h = max(ch, dh + self.offset_y * 2)

        vx = (cx - cw / 2.0) / float(total_w)
        vy = (cy - ch / 2.0) / float(total_h)
        self.canvas.xview_moveto(float(np.clip(vx, 0.0, 1.0)))
        self.canvas.yview_moveto(float(np.clip(vy, 0.0, 1.0)))

    def refresh_view(self, force_rebuild: bool = False):
        graph: Optional[PlatformGraph] = self.get_graph_fn()
        if not graph or not graph.nodes:
            self.lbl_info.config(text="⚠️ 当前地图暂无物理平台数据，请先载入地图！", fg="#ff9800")
            return

        patrol_platforms = self.get_patrol_platforms_fn()
        active_path = self.get_active_path_fn() or []
        map_changed = (self.last_map_id != graph.map_id)
        patrol_changed = (self.last_patrol != patrol_platforms)
        path_signature = [(e.from_id, e.to_id, e.action, e.trigger_x) for e in active_path]
        path_changed = (self.last_active_path != path_signature)

        self.lbl_info.config(text=f"🗺️ MapID: {graph.map_id} | 共 {len(graph.nodes)} 个平台, {sum(len(e) for e in graph.edges.values())} 条通道")

        if force_rebuild or map_changed or patrol_changed or path_changed or self.base_bgr_img is None:
            self.last_map_id = graph.map_id
            self.last_patrol = list(patrol_platforms) if patrol_platforms else []
            self.last_active_path = path_signature

            # 第一阶段只渲染并立即显示底图，让地图切换不必等待所有拓扑
            # 曲线绘制完成；第二阶段在后台补齐平台连线与巡航高亮。
            self._render_generation += 1
            render_generation = self._render_generation
            bgr = graph.render_topology_image(
                player_pos=None, active_path=None, highlight_platforms=None,
                title=f"{graph.map_name} - 平台与梯绳动作拓扑图",
                show_platform_edges=False, draw_labels=False,
            )
            if bgr is not None:
                self.base_bgr_img = bgr
                self._render_scaled_canvas()
                self.lbl_info.config(text=f"🗺️ MapID: {graph.map_id} | 底图已显示，拓扑线绘制中…")

                def render_overlay():
                    try:
                        overlay = graph.render_topology_image(
                            player_pos=None,
                            active_path=active_path,
                            highlight_platforms=patrol_platforms,
                            title=f"{graph.map_name} - 平台与梯绳动作拓扑图",
                            show_platform_edges=self.show_platform_edges.get(),
                            draw_labels=False,
                        )
                    except Exception:
                        overlay = None

                    def apply_overlay():
                        if not self.is_open or render_generation != self._render_generation:
                            return
                        if self.get_graph_fn() is not graph or overlay is None:
                            return
                        self.base_bgr_img = overlay
                        self._render_scaled_canvas()

                    if self.is_open:
                        self.top.after(0, apply_overlay)

                threading.Thread(target=render_overlay, daemon=True).start()

        self._update_player_marker()

    def _render_scaled_canvas(self):
        if self.base_bgr_img is None or self.base_bgr_img.size == 0:
            return

        ih, iw = self.base_bgr_img.shape[:2]
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())

        if self.is_auto_fit:
            sc = min((cw - 16) / max(1, iw), (ch - 16) / max(1, ih), 1.0)
            self.current_zoom = sc
            self.lbl_zoom.config(text=f"自适应 ({int(sc*100)}%)")
        else:
            sc = self.current_zoom

        dw = max(10, int(iw * sc))
        dh = max(10, int(ih * sc))
        self.scale_factor = sc

        # 快速插值缩放
        resized_bgr = cv2.resize(self.base_bgr_img, (dw, dh), interpolation=cv2.INTER_AREA if sc < 0.8 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        self.offset_x = max(0, (cw - dw) // 2)
        self.offset_y = max(0, (ch - dh) // 2)

        self.base_image_item = self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.img_tk)
        self.canvas.config(scrollregion=(0, 0, max(cw, dw + self.offset_x), max(ch, dh + self.offset_y)))

        # 平台/梯绳标签使用独立 Canvas 矢量图元，和 YOU 一样按当前
        # 缩放比例重新计算字号与位置，避免底图文字经过插值后大小失真。
        for item in getattr(self, "topology_label_items", []):
            self.canvas.delete(item)
        self.topology_label_items = []
        graph = self.get_graph_fn()
        if graph is not None:
            def add_scaled_label(text, wx, wy, color, font_px, border):
                ix, iy = graph.world_to_render_pixel(wx, wy, iw, ih)
                cx = int(ix * sc + self.offset_x)
                cy = int(iy * sc + self.offset_y)
                text_id = self.canvas.create_text(
                    cx, cy, text=text, fill=color,
                    font=("Segoe UI", max(6, int(round(font_px * sc))), "bold"),
                    anchor=tk.CENTER,
                )
                bbox = self.canvas.bbox(text_id)
                if bbox:
                    pad = max(2, int(round(4 * sc)))
                    bg_id = self.canvas.create_rectangle(
                        bbox[0] - pad, bbox[1] - pad,
                        bbox[2] + pad, bbox[3] + pad,
                        fill="#14161c", outline=border, width=max(1, int(round(sc))),
                    )
                    self.canvas.tag_raise(text_id, bg_id)
                    self.topology_label_items.extend([bg_id, text_id])
                else:
                    self.topology_label_items.append(text_id)

            for node in graph.nodes.values():
                add_scaled_label(
                    f"P{node.id}", node.center_x, node.center_y,
                    "#00e676" if node.id in (self.get_patrol_platforms_fn() or []) else "#ffff00",
                    8.0, "#00e676" if node.id in (self.get_patrol_platforms_fn() or []) else "#dddddd",
                )
            for lr in graph.ladder_ropes.values():
                add_scaled_label(
                    f"{lr.label} X={lr.x}", lr.x, (lr.y1 + lr.y2) / 2.0,
                    "#00d9ff" if lr.is_ladder else "#ffb000",
                    7.0, "#00aacc" if lr.is_ladder else "#cc7700",
                )

        # 创建动态角色矢量标记图元 (初始隐藏)
        self.player_glow_item = self.canvas.create_oval(0, 0, 0, 0, outline="#00e676", width=3, state="hidden")
        self.player_dot_item = self.canvas.create_oval(0, 0, 0, 0, fill="#00e5ff", outline="#ffffff", width=2, state="hidden")
        self.player_tag_bg_item = self.canvas.create_rectangle(0, 0, 0, 0, fill="#16181d", outline="#00e676", width=1, state="hidden")
        tag_font_px = max(6, int(round(9 * self.scale_factor)))
        self.player_tag_text_item = self.canvas.create_text(
            0, 0, text="", fill="#00e676",
            font=("Segoe UI", tag_font_px, "bold"), state="hidden"
        )

        self._update_player_marker()

    def _update_player_marker(self):
        graph: Optional[PlatformGraph] = self.get_graph_fn()
        if not graph or self.base_bgr_img is None:
            return

        player_pos = self.get_player_pos_fn()

        # 1. 更新顶部状态文字
        curr_p, curr_lr, is_climbing = graph.find_player_location(player_pos[0], player_pos[1]) if player_pos else (None, None, False)
        if is_climbing and curr_lr:
            p_text = f"📍 角色当前状态: 🧗 攀爬中 [{curr_lr.kind_name} #{curr_lr.id}] (X={curr_lr.x}, Y: {curr_lr.y1}~{curr_lr.y2})"
            tag_str = f"YOU (攀爬 {curr_lr.label})"
            tag_color = "#00e5ff"
        elif curr_p:
            lr_hint = f" [紧邻{curr_lr.kind_name}#{curr_lr.id}]" if curr_lr else ""
            surface_y = int(round(curr_p.surface_y_at(player_pos[0]))) if player_pos else curr_p.center_y
            p_text = f"📍 角色当前所在: 平台 #{curr_p.id}{lr_hint} [X={curr_p.center_x}, Y={surface_y}]"
            tag_str = f"YOU (P{curr_p.id})"
            tag_color = "#00e676"
        elif curr_lr:
            p_text = f"📍 角色当前位置: 靠近{curr_lr.kind_name} #{curr_lr.id} [世界坐标: ({player_pos[0]}, {player_pos[1]})]"
            tag_str = f"YOU ({curr_lr.label})"
            tag_color = "#ffb74d"
        elif player_pos:
            p_text = f"📍 角色当前位置: 世界坐标 (X={player_pos[0]}, Y={player_pos[1]})"
            tag_str = f"YOU ({player_pos[0]}, {player_pos[1]})"
            tag_color = "#ffb74d"
        else:
            p_text = "📍 角色当前位置: 未定位"
            tag_str = ""
            tag_color = "#90a4ae"

        self.lbl_player_status.config(text=p_text)

        # 2. 毫秒级原生 Canvas 矢量更新角色位置
        if player_pos and self.player_dot_item is not None:
            ih, iw = self.base_bgr_img.shape[:2]
            ix, iy = graph.world_to_render_pixel(player_pos[0], player_pos[1], iw, ih)
            cx = int(ix * self.scale_factor + self.offset_x)
            cy = int(iy * self.scale_factor + self.offset_y)

            # 光晕与圆点
            self.canvas.coords(self.player_glow_item, cx - 12, cy - 12, cx + 12, cy + 12)
            self.canvas.itemconfigure(self.player_glow_item, outline=tag_color, state="normal")

            self.canvas.coords(self.player_dot_item, cx - 5, cy - 5, cx + 5, cy + 5)
            self.canvas.itemconfigure(self.player_dot_item, state="normal")

            # 角色头顶标签
            # YOU 是 Canvas 叠加层，不属于底图，必须显式跟随当前拓扑图
            # 缩放，否则缩放地图时 YOU 会保持固定字号。
            tag_font_px = max(6, int(round(9 * self.scale_factor)))
            self.canvas.itemconfigure(
                self.player_tag_text_item,
                font=("Segoe UI", tag_font_px, "bold"),
            )
            self.canvas.itemconfigure(self.player_tag_text_item, text=tag_str, fill=tag_color, state="normal")
            bbox = self.canvas.bbox(self.player_tag_text_item)
            if bbox:
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                tx = cx
                ty = cy - 20
                self.canvas.coords(self.player_tag_text_item, tx, ty)
                self.canvas.coords(self.player_tag_bg_item, tx - tw//2 - 4, ty - th//2 - 2, tx + tw//2 + 4, ty + th//2 + 2)
                self.canvas.itemconfigure(self.player_tag_bg_item, outline=tag_color, state="normal")
        else:
            if self.player_glow_item:
                self.canvas.itemconfigure(self.player_glow_item, state="hidden")
                self.canvas.itemconfigure(self.player_dot_item, state="hidden")
                self.canvas.itemconfigure(self.player_tag_bg_item, state="hidden")
                self.canvas.itemconfigure(self.player_tag_text_item, state="hidden")




# ─────────────────────────────────────────────────────────────────────────────
#  MinimapRadarDialog：小地图雷达与角色黄点实时定位调试器 (F8 快捷呼出)
# ─────────────────────────────────────────────────────────────────────────────
class MinimapRadarDialog:
    """
    小地图雷达与角色黄点实时定位调试器:
    1. 高清放大展示左上角小地图区域 (支持 1.5x / 2.0x / 3.0x 放大)
    2. 实时用高对比度黄色大光圈圆环 + 准星十字圈出角色定位
    3. 清晰显示小地图像素坐标 (px, py)、归一化比例 (norm_x, norm_y)、物理世界坐标 (wx, wy) 与当前匹配平台 P#
    4. 支持单帧抓取与 200ms 实时雷达连续监听
    """
    def __init__(self, parent: tk.Tk, get_frame_fn, get_tracker_fn, get_graph_fn,
                 get_motion_fn=None, get_kalman_fn=None, get_raw_world_pos_fn=None,
                 get_tracker_result_fn=None, on_manual_box_callback=None):
        self.parent = parent
        self.get_frame_fn = get_frame_fn
        self.get_tracker_fn = get_tracker_fn
        self.get_graph_fn = get_graph_fn
        self.get_motion_fn = get_motion_fn or (lambda: None)
        self.get_kalman_fn = get_kalman_fn or (lambda: None)
        # F8 只读后台 60Hz 坐标线程的最终原始测量，绝不再次调用 Canvas
        # 匹配器。否则 F8、HUD、拓扑图会并发改写同一张图的卷轴 offset。
        self.get_raw_world_pos_fn = get_raw_world_pos_fn or (lambda: None)
        # 与后台 60Hz raw_tracker 共用同一份测量结果，避免 F8 再用另一
        # 个 Tracker 复检而与导航坐标产生分歧。
        self.get_tracker_result_fn = get_tracker_result_fn or (lambda: None)
        self.on_manual_box_callback = on_manual_box_callback
        self.is_open = True
        self.auto_refresh = tk.BooleanVar(value=True)
        self.zoom_scale = 1.5
        self._json_canvas_map_id = None
        self._json_canvas_size = None

        self.top = tk.Toplevel(parent)
        self.top.title("🧭 小地图雷达与角色黄点定位调试器 [F8]")
        fit_window_to_work_area(self.top, (640, 860), (520, 560), parent=parent)
        self.top.configure(bg="#16181d")

        # 顶部工具栏
        tb = tk.Frame(self.top, bg="#1a1c23")
        tb.pack(fill=tk.X, side=tk.TOP, padx=8, pady=6)

        self.lbl_radar_status = tk.Label(
            tb, text="● 雷达状态: 正在监听...", font=("Segoe UI", 9, "bold"),
            fg="#00e5ff", bg="#1a1c23"
        )
        self.lbl_radar_status.pack(fill=tk.X, padx=6, pady=(2, 4))

        tb_controls = tk.Frame(tb, bg="#1a1c23")
        tb_controls.pack(fill=tk.X)

        # 刷新与连续刷新开关
        tk.Button(tb_controls, text="🔄 单帧抓取", font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#0277bd",
                  relief=tk.FLAT, padx=6, pady=2, command=self.refresh_single_frame).pack(side=tk.RIGHT, padx=3)

        tk.Checkbutton(
            tb_controls, text="连续雷达", variable=self.auto_refresh,
            font=("Segoe UI", 8), fg="#b0bec5", bg="#1a1c23", selectcolor="#121418",
            activebackground="#1a1c23", activeforeground="#00e5ff"
        ).pack(side=tk.RIGHT, padx=4)

        # 📐 手动精确框选小地图范围按钮 (1.5x 放大拉框)
        tk.Button(tb_controls, text="📐 手动框选小地图", font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#00838f",
                  relief=tk.FLAT, padx=6, pady=2, command=self._on_manual_crop_clicked).pack(side=tk.RIGHT, padx=3)

        # 缩放倍率切换
        for z in [3.0, 2.0, 1.5, 1.0]:
            tk.Button(
                tb_controls, text=f"{z}x", font=("Consolas", 8), fg="#e0e0e0", bg="#2a2a32",
                relief=tk.FLAT, padx=4, pady=2, command=lambda sc=z: self.set_zoom(sc)
            ).pack(side=tk.RIGHT, padx=1)

        # 图像展示区 (Canvas)
        self.canvas_frame = tk.Frame(self.top, bg="#121418", bd=1, relief=tk.SOLID)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d0e12", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.img_tk: Optional[ImageTk.PhotoImage] = None

        # 底部大号高清遥测数据显示卡片
        info_card = tk.LabelFrame(
            self.top, text="📊 实时小地图与物理坐标遥测 (Telemetry)",
            font=("Segoe UI", 9, "bold"), fg="#00e5ff", bg="#202026", padx=8, pady=6
        )
        info_card.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 8))

        # 网格排布 4 个核心数据指标胶囊
        g = tk.Frame(info_card, bg="#202026")
        g.pack(fill=tk.X)
        g.columnconfigure(0, weight=1)
        g.columnconfigure(1, weight=1)

        # 1. 小地图像素坐标
        c1 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c1.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        tk.Label(c1, text="📍 小地图局部像素 (Pixel):", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_pixel = tk.Label(c1, text="X: -- px, Y: -- px", font=("Consolas", 10, "bold"), fg="#ffff00", bg="#16181d")
        self.val_pixel.pack(anchor=tk.W)

        # 2. 归一化坐标
        c2 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c2.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        tk.Label(c2, text="📐 归一化相对比例 (Norm 0~1):", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_norm = tk.Label(c2, text="nx: --, ny: --", font=("Consolas", 10, "bold"), fg="#00e5ff", bg="#16181d")
        self.val_norm.pack(anchor=tk.W)

        # 3. 游戏世界物理坐标
        c3 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c3.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        tk.Label(c3, text="🗺️ 世界物理坐标 (World Pos):", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_world = tk.Label(c3, text="X: --, Y: --", font=("Consolas", 10, "bold"), fg="#69f0ae", bg="#16181d")
        self.val_world.pack(anchor=tk.W)

        # 4. 当前匹配承重平台
        c4 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c4.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        tk.Label(c4, text="🛤️ 匹配承重平台 (Platform):", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_platform = tk.Label(c4, text="未匹配到实体平台", font=("Consolas", 10, "bold"), fg="#ffb74d", bg="#16181d")
        self.val_platform.pack(anchor=tk.W)

        # 5. 两套连续世界坐标预测并行显示。Kalman 目前是影子模型，
        # 不参与 F6 控制，便于在同一批观测上做无风险 A/B 对比。
        c5 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c5.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        tk.Label(c5, text="🧭 原水平预测模型 (并行对照):", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_motion = tk.Label(c5, text="direction=-- | predicted X=-- | v=-- | blocked=--",
                                   font=("Consolas", 9, "bold"), fg="#ce93d8", bg="#16181d")
        self.val_motion.pack(anchor=tk.W)

        c5k = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c5k.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        tk.Label(c5k, text="📈 输入感知 Kalman (现用控制/拓扑):", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_kalman = tk.Label(
            c5k, text="direction=-- | X=-- | v=-- | σ=--\n观测区间=-- | innovation=--",
            font=("Consolas", 9, "bold"), fg="#80cbc4", bg="#16181d",
        )
        self.val_kalman.pack(anchor=tk.W)

        # 6/7. 小地图图像识别尺寸与 JSON 定义尺寸
        c6 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c6.grid(row=4, column=0, padx=2, pady=2, sticky="ew")
        tk.Label(c6, text="🖼️ 识别到的小地图像素范围:", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_detected_map_size = tk.Label(c6, text="-- x -- px", font=("Consolas", 10, "bold"), fg="#ffca28", bg="#16181d")
        self.val_detected_map_size.pack(anchor=tk.W)

        c7 = tk.Frame(g, bg="#16181d", padx=6, pady=4)
        c7.grid(row=4, column=1, padx=2, pady=2, sticky="ew")
        tk.Label(c7, text="🧾 JSON 小地图实际尺寸:", font=("Segoe UI", 8), fg="#90a4ae", bg="#16181d").pack(anchor=tk.W)
        self.val_json_map_size = tk.Label(c7, text="-- x -- px", font=("Consolas", 10, "bold"), fg="#80cbc4", bg="#16181d")
        self.val_json_map_size.pack(anchor=tk.W)

        self.top.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_single_frame()
        self._schedule_loop()

    def on_close(self):
        self.is_open = False
        self.top.destroy()

    def set_zoom(self, zoom: float):
        self.zoom_scale = zoom
        self.refresh_single_frame()

    def _schedule_loop(self):
        if self.is_open:
            if self.auto_refresh.get():
                self.refresh_single_frame()
            self.top.after(100, self._schedule_loop)

    def refresh_single_frame(self):
        frame = self.get_frame_fn()
        if frame is None or frame.size == 0:
            return

        tracker: MinimapTracker = self.get_tracker_fn()
        graph: Optional[PlatformGraph] = self.get_graph_fn()

        # 优先采用后台刚刚对捕获帧完成的原始测量。后台刚启动、尚未产生
        # 结果时，才由 F8 自己做一次临时检测以保持调试窗口可用。
        tr = self.get_tracker_result_fn()
        if tr is None:
            tr = tracker.detect(frame)

        # 显示当前识别框尺寸与地图 JSON 的 miniMap 尺寸，便于核对
        # 截屏裁剪范围和世界坐标换算比例是否一致。
        if tr.inner_box:
            _, _, detected_w, detected_h = tr.inner_box
            self.val_detected_map_size.config(text=f"{detected_w} x {detected_h} px")
        else:
            self.val_detected_map_size.config(text="未识别")
        if graph is not None:
            # miniMap.width/height 是大地图物理尺寸；这里要显示的是
            # JSON 中 miniMap.canvas 图片解码后的实际像素尺寸。
            if self._json_canvas_map_id != graph.map_id:
                self._json_canvas_map_id = graph.map_id
                self._json_canvas_size = None
                try:
                    canvas_b64 = (graph.minimap_meta or {}).get("canvas")
                    if canvas_b64:
                        if "," in canvas_b64 and canvas_b64.startswith("data:"):
                            canvas_b64 = canvas_b64.split(",", 1)[1]
                        canvas_img = Image.open(BytesIO(base64.b64decode(canvas_b64)))
                        self._json_canvas_size = canvas_img.size
                except Exception:
                    self._json_canvas_size = None
            if self._json_canvas_size:
                json_w, json_h = self._json_canvas_size
                self.val_json_map_size.config(text=f"{json_w} x {json_h} px")
            else:
                self.val_json_map_size.config(text="JSON canvas 未提供")
        else:
            self.val_json_map_size.config(text="等待地图拓扑载入")

        # 裁剪小地图搜索区域 (动态自适应宽幅地图，如地铁/废都等，彻底告别右侧和底部截断)
        fh, fw = frame.shape[:2]
        if tr.inner_box:
            ix, iy, iw, ih = tr.inner_box
            sw = min(fw, max(420, ix + iw + 35))
            sh = min(fh, max(320, iy + ih + 35))
        else:
            sw = min(fw, 420)
            sh = min(fh, 320)
        sx, sy = 0, 0
        roi = frame[sy:sh, sx:sw].copy()

        # 绘制检测框与黄色大圆环圈出
        if tr.inner_box:
            ix, iy, iw, ih = tr.inner_box
            rel_ix = ix - sx
            rel_iy = iy - sy
            # 绘制小地图内框绿色边界
            cv2.rectangle(roi, (rel_ix, rel_iy), (rel_ix + iw, rel_iy + ih), (0, 255, 128), 2)

            if tr.is_detected and tr.pixel_pos:
                cx, cy = tr.pixel_pos
                px_screen = rel_ix + cx
                py_screen = rel_iy + cy

                # 用醒目的黄色大光圈圆环 (Yellow Circle Ring) + 准星圈出角色
                cv2.circle(roi, (px_screen, py_screen), 14, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(roi, (px_screen, py_screen), 7, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(roi, (px_screen, py_screen), 2, (0, 0, 255), -1, cv2.LINE_AA)

                # 十字准星线
                cv2.line(roi, (px_screen - 18, py_screen), (px_screen + 18, py_screen), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(roi, (px_screen, py_screen - 18), (px_screen, py_screen + 18), (0, 255, 255), 1, cv2.LINE_AA)

                # 在光圈头顶打出小地图局部坐标文字
                coord_badge = f"({cx}, {cy})"
                cv2.putText(roi, coord_badge, (px_screen - 20, max(12, py_screen - 18)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 放大展示
        sc = self.zoom_scale
        dw = int(roi.shape[1] * sc)
        dh = int(roi.shape[0] * sc)
        resized_roi = cv2.resize(roi, (dw, dh), interpolation=cv2.INTER_NEAREST if sc >= 2.0 else cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        self.img_tk = ImageTk.PhotoImage(img_pil)

        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        offset_x = max(0, (cw - dw) // 2)
        offset_y = max(0, (ch - dh) // 2)

        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.img_tk)

        # 更新遥测数值面板
        if tr.is_detected and tr.pixel_pos and tr.norm_pos:
            px, py = tr.subpixel_pos or tr.pixel_pos
            nx, ny = tr.norm_pos
            self.val_pixel.config(text=f"X={px:.2f} px, Y={py:.2f} px", fg="#ffff00")
            self.val_norm.config(text=f"nx: {nx:.4f}, ny: {ny:.4f}", fg="#00e5ff")
            self.lbl_radar_status.config(text="● 正在追踪 (已精准圈出角色)", fg="#00e676")

            if graph:
                crop_gray = None
                if tr.inner_box:
                    bx, by, bw, bh = tr.inner_box
                    sub = frame[by:by+bh, bx:bx+bw]
                    if sub.size > 0:
                        crop_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if len(sub.shape) == 3 else sub
                raw_cached = self.get_raw_world_pos_fn()
                if raw_cached is not None:
                    wx, wy = raw_cached
                else:
                    # 后台线程尚未产出第一帧时才允许临时换算一次。
                    wx, wy = graph.minimap_norm_to_world(nx, ny, crop_gray_frame=crop_gray)
                # F8 只展示后台 60Hz 的原始测量，不能反向写入导航的运动模型
                # 或卷轴匹配状态。
                self.val_world.config(
                    text=f"X: {wx:.2f}, Y: {wy:.2f}\n图ID: {graph.map_id}",
                    fg="#69f0ae"
                )
                # 坐标文本保留黄点原始换算值；但承重层/梯绳状态必须
                # 与拓扑图使用同一份连续 X 预测值，否则量化黄点在绳子
                # 中轴两侧跳动时会错误显示“未匹配到承重层”。
                match_x = wx
                try:
                    kalman = self.get_kalman_fn()
                    predicted_x = kalman.predict() if kalman is not None else None
                    if predicted_x is not None:
                        match_x = int(round(predicted_x))
                except Exception:
                    pass
                p, lr, is_c = graph.find_player_location(match_x, wy)
                if is_c and lr:
                    self.val_platform.config(text=f"攀爬中 [{lr.kind_name} #{lr.id}]", fg="#00e5ff")
                elif p:
                    surface_y = int(round(p.surface_y_at(match_x)))
                    self.val_platform.config(text=f"平台 #{p.id} (承重层 Y={surface_y})", fg="#69f0ae")
                elif lr:
                    self.val_platform.config(text=f"靠近 {lr.kind_name} #{lr.id}", fg="#ffb74d")
                else:
                    self.val_platform.config(text="未匹配到承重层 (悬空/空中)", fg="#ffb74d")
            else:
                self.val_world.config(text="等待地图拓扑载入...", fg="#90a4ae")
                self.val_platform.config(text="--", fg="#90a4ae")
        else:
            self.val_pixel.config(text="X: --, Y: --", fg="#ff5252")
            self.val_norm.config(text="未检测到角色光点", fg="#ff5252")
            self.val_world.config(text="--", fg="#90a4ae")
            self.val_platform.config(text="--", fg="#90a4ae")
            self.lbl_radar_status.config(text="⚠️ 搜索中: 未检测到有效光点", fg="#ff5252")

        # 原始小地图值与预测值并列展示，避免 F8/拓扑图数据源混淆。
        try:
            motion = self.get_motion_fn()
            if motion is not None:
                predicted = motion.predict()
                active_direction = motion.direction
                inferred = False
                if active_direction == 0 and time.perf_counter() < getattr(motion, "inferred_motion_until", 0.0):
                    active_direction = getattr(motion, "inferred_direction", 0)
                    inferred = active_direction != 0
                direction_name = {1: "right", -1: "left", 0: "idle"}.get(active_direction, "?")
                if inferred:
                    direction_name += " (inferred)"
                self.val_motion.config(
                    text=(f"direction={direction_name} | predicted X="
                          f"{('--' if predicted is None else f'{predicted:.2f}')} | "
                          f"v={motion.vx:.1f} | blocked={motion.blocked}")
                )
        except Exception:
            pass

        try:
            kalman = self.get_kalman_fn()
            if kalman is not None:
                state = kalman.snapshot()
                direction_name = {1: "right", -1: "left", 0: "idle"}.get(
                    state["direction"], "?"
                )
                kx = state["x"]
                sigma = max(0.0, state["position_variance"]) ** 0.5
                reanchor = " | REANCHOR" if state["reanchored"] else ""
                interval_text = "--"
                if state["interval_low"] is not None:
                    interval_text = (
                        f"[{state['interval_low']:.1f}, {state['interval_high']:.1f}]"
                    )
                self.val_kalman.config(
                    text=(
                        f"direction={direction_name} | X="
                        f"{('--' if kx is None else f'{kx:.2f}')} | "
                        f"v={state['vx']:.1f} | σ={sigma:.1f}\n"
                        f"观测区间={interval_text} | "
                        f"innovation={state['innovation']:+.1f}{reanchor}"
                    )
                )
        except Exception:
            pass

    def _on_manual_crop_clicked(self):
        """点击手动高倍率框选小地图范围"""
        frame = self.get_frame_fn()
        if frame is None or frame.size == 0:
            messagebox.showwarning("提示", "未获取到游戏画面，请确保游戏窗口已运行且已捕获！", parent=self.top)
            return

        tracker = self.get_tracker_fn()
        current_box = getattr(tracker, "cached_inner_box", None)
        MinimapCropCalibrationDialog(self.top, frame, current_box, self._on_crop_confirmed)

    def _on_crop_confirmed(self, box: Optional[Tuple[int, int, int, int]]):
        """用户确认手动框选或恢复自动探测"""
        if self.on_manual_box_callback:
            self.on_manual_box_callback(box)
        else:
            tracker = self.get_tracker_fn()
            if tracker and hasattr(tracker, "set_manual_inner_box"):
                tracker.set_manual_inner_box(box)

        self.refresh_single_frame()
        if box:
            messagebox.showinfo("标定成功", f"🎉 小地图内画布范围已手动锁定为：\n[x={box[0]}, y={box[1]}, 宽={box[2]}, 高={box[3]} px]\n系统已立即同步！", parent=self.top)
        else:
            messagebox.showinfo("恢复成功", "已恢复小地图全自动算法探测模式！", parent=self.top)


# ─────────────────────────────────────────────────────────────────────────────
#  MonsterTooltip：怪物列表鼠标悬浮图片弹窗预览
# ─────────────────────────────────────────────────────────────────────────────
class MonsterTooltip:
    def __init__(self, tree_widget: ttk.Treeview, get_image_callback):
        self.tree = tree_widget
        self.get_image_callback = get_image_callback
        self.tip_window: Optional[tk.Toplevel] = None
        self.current_item: Optional[str] = None
        self.tree.bind("<Motion>", self._on_motion)
        self.tree.bind("<Leave>", self._on_leave)

    def _on_motion(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            self.hide()
            return
        if item_id == self.current_item and self.tip_window:
            return
        self.current_item = item_id
        values = self.tree.item(item_id, "values")
        if not values or len(values) < 2:
            self.hide()
            return
        # 主界面的本图怪物表首列是启用勾选框；Mob ID/名称位于第 2/3 列。
        # 编辑怪物弹窗仍使用旧的三列 Treeview，不经过本 Tooltip 实例。
        if len(values) >= 4:
            mob_id, mob_name = str(values[1]), str(values[2])
        else:
            mob_id, mob_name = str(values[0]), str(values[1])
        img_tk = self.get_image_callback(mob_id, mob_name)
        if img_tk:
            self.show(event.x_root + 20, event.y_root + 10, mob_id, mob_name, img_tk)
        else:
            self.hide()

    def show(self, x, y, mob_id, mob_name, img_tk):
        self.hide()
        self.tip_window = tk.Toplevel(self.tree)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        self.tip_window.attributes("-topmost", True)
        frame = tk.Frame(self.tip_window, bg="#1b1b22", bd=0,
                         highlightbackground="#00e5ff", highlightthickness=1)
        frame.pack()
        tk.Label(frame, text=f"👾 {mob_name}  (ID: {mob_id})",
                 font=("Segoe UI", 9, "bold"), fg="#00e5ff", bg="#1b1b22").pack(padx=12, pady=(6, 2))
        ic = tk.Frame(frame, bg="#121216", padx=8, pady=8)
        ic.pack(padx=10, pady=4)
        lbl = tk.Label(ic, image=img_tk, bg="#121216")
        lbl.image = img_tk
        lbl.pack()
        tk.Label(frame, text="GMS v83 官方动作精灵图",
                 font=("Segoe UI", 8), fg="#90a4ae", bg="#1b1b22").pack(padx=10, pady=(0, 6))

    def hide(self):
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None
        self.current_item = None

    def _on_leave(self, event):
        self.hide()


# ─────────────────────────────────────────────────────────────────────────────
#  ManualCropPlayerDialog：鼠标拖拽手动框选角色名牌
# ─────────────────────────────────────────────────────────────────────────────
class ManualCropPlayerDialog(tk.Toplevel):
    def __init__(self, parent, frame_bgr, on_crop_callback, title: Optional[str] = None, tip_text: Optional[str] = None):
        super().__init__(parent)
        self.title(title or "✂️ 鼠标拖拽框选名字牌 / 角色特征 (按住左键拉框，松开确认)")
        self.configure(bg="#1a1a20")
        self.transient(parent)
        self.grab_set()

        self.frame_bgr = frame_bgr
        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        self.on_crop_callback = on_crop_callback

        # 画布不再写死 1024px；按父窗口所在显示器的实际工作区与原图
        # 宽高比计算，给标题、底部操作栏和窗口边框预留空间。
        self.disp_w, self.disp_h, self.scale = fit_image_to_work_area(
            parent, self.orig_w, self.orig_h, reserved_width=48, reserved_height=205
        )
        fit_window_to_work_area(
            self,
            (self.disp_w + 36, self.disp_h + 185),
            (min(640, self.disp_w + 36), min(460, self.disp_h + 185)),
            parent=parent,
        )

        # 缩放图像用于显示
        rgb_disp = cv2.cvtColor(cv2.resize(frame_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(rgb_disp))

        # 顶部操作提示条
        top_bar = tk.Frame(self, bg="#202026", pady=6, padx=12)
        top_bar.pack(fill=tk.X)
        tip = tip_text or "🎯 提示：请在下方画面中【按住鼠标左键并拖动】框选出您的名字牌（建议仅包含名字文本或名牌条），松开后点击确认！"
        tk.Label(top_bar, text=tip,
                 font=("Segoe UI", 9, "bold"), fg="#00e5ff", bg="#202026",
                 justify=tk.LEFT, wraplength=max(320, self.disp_w - 24)).pack(fill=tk.X)

        # 画布区域
        canvas_frame = tk.Frame(self, bg="#101014")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.canvas = tk.Canvas(canvas_frame, width=self.disp_w, height=self.disp_h, bg="#000000", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(anchor=tk.CENTER)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

        # 底部控制条与选区预览
        bot_bar = tk.Frame(self, bg="#202026", pady=8, padx=12)
        bot_bar.pack(fill=tk.X)

        info_row = tk.Frame(bot_bar, bg="#202026")
        info_row.pack(fill=tk.X)
        action_row = tk.Frame(bot_bar, bg="#202026")
        action_row.pack(fill=tk.X, pady=(6, 0))

        self.preview_lbl = tk.Label(info_row, text="未选区", bg="#2a2a32", fg="#b0bec5", font=("Consolas", 9), width=18, height=2)
        self.preview_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.status_lbl = tk.Label(info_row, text="请在游戏画面中拖拽拉框...", font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026", anchor=tk.W, justify=tk.LEFT)
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_cancel = tk.Button(action_row, text="❌ 取消 (Esc)", font=("Segoe UI", 9), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=14, pady=4, command=self.destroy)
        btn_cancel.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_confirm = tk.Button(action_row, text="💾 确认应用此选区 (Enter)", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#2e7d32", relief=tk.FLAT, padx=16, pady=4, state=tk.DISABLED, command=self._confirm_crop)
        self.btn_confirm.pack(side=tk.RIGHT)

        # 鼠标交互变量
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.selected_crop_bgr = None

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Return>", lambda e: self._confirm_crop())
        self.bind("<Escape>", lambda e: self.destroy())

    def _on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.btn_confirm.config(state=tk.DISABLED, bg="#424242")

    def _on_mouse_move(self, event):
        if self.start_x is None:
            return
        cur_x, cur_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="#00ff00", width=2, dash=(4, 2))

    def _on_mouse_up(self, event):
        if self.start_x is None:
            return
        end_x, end_y = event.x, event.y
        x1 = max(0, min(self.start_x, end_x))
        x2 = min(self.disp_w, max(self.start_x, end_x))
        y1 = max(0, min(self.start_y, end_y))
        y2 = min(self.disp_h, max(self.start_y, end_y))

        # 映射回原始分辨率
        orig_x1 = int(x1 / self.scale)
        orig_x2 = int(x2 / self.scale)
        orig_y1 = int(y1 / self.scale)
        orig_y2 = int(y2 / self.scale)

        w = orig_x2 - orig_x1
        h = orig_y2 - orig_y1

        if w >= 15 and h >= 8:

            self.selected_crop_bgr = self.frame_bgr[orig_y1:orig_y2, orig_x1:orig_x2].copy()
            # 更新预览
            prev_w = min(140, max(20, w))
            prev_h = min(40, max(12, h))
            prev_rgb = cv2.cvtColor(cv2.resize(self.selected_crop_bgr, (prev_w, prev_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
            self.preview_tk = ImageTk.PhotoImage(image=Image.fromarray(prev_rgb))
            self.preview_lbl.config(image=self.preview_tk, text="")
            
            is_nametag = (h <= 28 and (w / float(max(1, h))) >= 2.8)
            mode_desc = "📌【脚底细名牌模式】(自动居中角色身材 48x56px)" if is_nametag else "🧍【直选角色身体模式】(1:1 精确贴合当前框)"
            self.status_lbl.config(text=f"✅ 已选中: {w}x{h} px | {mode_desc}", fg="#69f0ae")
            self.btn_confirm.config(state=tk.NORMAL, bg="#2e7d32")

        else:
            self.status_lbl.config(text="⚠️ 选区太小，请重新拉框框选", fg="#ff5252")
            self.btn_confirm.config(state=tk.DISABLED, bg="#424242")

    def _confirm_crop(self):
        if self.selected_crop_bgr is not None:
            self.on_crop_callback(self.selected_crop_bgr)
            self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  TwoStagePlayerCalibrationDialog：两段式角色标定（全身整体 + 核心特征 + 朝向）
# ─────────────────────────────────────────────────────────────────────────────
class TwoStagePlayerCalibrationDialog(tk.Toplevel):
    def __init__(self, parent, frame_bgr: np.ndarray, on_complete_callback):
        super().__init__(parent)
        self.title("👑 两段式角色标定 (第 1 步：框选全身整体)")
        self.configure(bg="#1a1a20")
        self.transient(parent)
        self.grab_set()

        self.frame_bgr = frame_bgr
        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        self.on_complete_callback = on_complete_callback

        self.stage = 1  # 1: 全身矩形框选, 2: 核心特征多边形圈选
        self.whole_bgr = None
        self.feature_bgr = None
        self.feature_box_in_whole = None
        self.feature_mask = None
        self.feature_polygon_in_crop = None

        # 依据当前显示器工作区自适应，避免固定宽度使 4:3 画面纵向溢出。
        self.disp_w, self.disp_h, self.scale = fit_image_to_work_area(
            parent, self.orig_w, self.orig_h, reserved_width=48, reserved_height=220
        )
        fit_window_to_work_area(
            self,
            (self.disp_w + 36, self.disp_h + 200),
            (min(680, self.disp_w + 36), min(480, self.disp_h + 200)),
            parent=parent,
        )

        # 顶部提示条
        self.top_bar = tk.Frame(self, bg="#202026", pady=6, padx=12)
        self.top_bar.pack(fill=tk.X)
        self.lbl_step_tip = tk.Label(
            self.top_bar,
            text="🎯 步骤 1/2：【鼠标拖拽拉框】框选角色全身（从帽顶/头顶直到脚底站立线），选好后点击【下一步】",
            font=("Segoe UI", 9, "bold"), fg="#ffd54f", bg="#202026",
            justify=tk.LEFT, wraplength=max(320, self.disp_w - 24),
        )
        self.lbl_step_tip.pack(fill=tk.X)

        # 画布区域
        self.canvas_frame = tk.Frame(self, bg="#101014")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        rgb_disp = cv2.cvtColor(cv2.resize(frame_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        self.tk_img_stage1 = ImageTk.PhotoImage(image=Image.fromarray(rgb_disp))

        self.canvas = tk.Canvas(self.canvas_frame, width=self.disp_w, height=self.disp_h, bg="#000000", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(anchor=tk.CENTER)
        self.canvas_img_id = self.canvas.create_image(0, 0, image=self.tk_img_stage1, anchor=tk.NW)

        # 底部控制条
        self.bot_bar = tk.Frame(self, bg="#202026", pady=8, padx=12)
        self.bot_bar.pack(fill=tk.X)

        self.info_row = tk.Frame(self.bot_bar, bg="#202026")
        self.info_row.pack(fill=tk.X)
        self.action_row = tk.Frame(self.bot_bar, bg="#202026")
        self.action_row.pack(fill=tk.X, pady=(6, 0))

        self.preview_lbl = tk.Label(self.info_row, text="未选区", bg="#2a2a32", fg="#b0bec5", font=("Consolas", 9), width=16, height=2)
        self.preview_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.status_lbl = tk.Label(self.info_row, text="请在画面中框选角色全身...", font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026", anchor=tk.W, justify=tk.LEFT)
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 按钮区
        self.btn_cancel = tk.Button(self.action_row, text="❌ 取消 (Esc)", font=("Segoe UI", 9), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=12, pady=4, command=self.destroy)
        self.btn_cancel.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_next = tk.Button(self.action_row, text="下一步：多边形圈选核心特征 ➡️", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=14, pady=4, state=tk.DISABLED, command=self._go_to_stage_2)
        self.btn_next.pack(side=tk.RIGHT)

        # 阶段 2 朝向选择按钮
        self.facing_btn_frame = tk.Frame(self.action_row, bg="#202026")
        self.btn_face_left = tk.Button(self.facing_btn_frame, text="⬅️ 当前人物朝左 (保存)", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=12, pady=4, state=tk.DISABLED, command=lambda: self._finish_calibration("left"))
        self.btn_face_left.pack(side=tk.LEFT, padx=4)
        self.btn_face_right = tk.Button(self.facing_btn_frame, text="➡️ 当前人物朝右 (保存)", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=12, pady=4, state=tk.DISABLED, command=lambda: self._finish_calibration("right"))
        self.btn_face_right.pack(side=tk.LEFT, padx=4)

        # 鼠标交互变量
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.feature_polygon_display = []
        self.feature_polygon_closed = False
        self.selected_whole_crop = None
        self.selected_feat_crop = None

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Button-3>", self._finalize_feature_polygon)
        self.bind("<Return>", self._finalize_feature_polygon)
        self.bind("<BackSpace>", self._undo_feature_polygon_point)
        self.bind("<Escape>", lambda e: self.destroy())

    def _on_mouse_down(self, event):
        if self.stage == 2:
            if self.feature_polygon_closed:
                self.feature_polygon_display = []
                self.feature_polygon_closed = False
                self.selected_feat_crop = None
                self.feature_mask = None
                self.feature_polygon_in_crop = None
                self.btn_face_left.config(state=tk.DISABLED, bg="#424242")
                self.btn_face_right.config(state=tk.DISABLED, bg="#424242")
            x = max(0, min(self.disp_stage2_w - 1, int(event.x)))
            y = max(0, min(self.disp_stage2_h - 1, int(event.y)))
            if not self.feature_polygon_display or (
                abs(x - self.feature_polygon_display[-1][0])
                + abs(y - self.feature_polygon_display[-1][1]) >= 3
            ):
                self.feature_polygon_display.append((x, y))
            self._redraw_feature_polygon()
            self.status_lbl.config(
                text=(
                    f"多边形已添加 {len(self.feature_polygon_display)} 个顶点；"
                    "继续左键添加，右键或 Enter 闭合，Backspace 撤销"
                ),
                fg="#ffd54f",
            )
            return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        if self.stage == 1:
            self.btn_next.config(state=tk.DISABLED, bg="#424242")
        else:
            self.btn_face_left.config(state=tk.DISABLED, bg="#424242")
            self.btn_face_right.config(state=tk.DISABLED, bg="#424242")

    def _on_mouse_move(self, event):
        if self.stage == 2:
            return
        if self.start_x is None:
            return
        cur_x, cur_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        outline_col = "#00e5ff" if self.stage == 1 else "#ffea00"
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline=outline_col, width=2, dash=(4, 2))

    def _on_mouse_up(self, event):
        if self.stage == 2:
            return
        if self.start_x is None:
            return
        end_x, end_y = event.x, event.y
        if self.stage == 1:
            x1 = max(0, min(self.start_x, end_x))
            x2 = min(self.disp_w, max(self.start_x, end_x))
            y1 = max(0, min(self.start_y, end_y))
            y2 = min(self.disp_h, max(self.start_y, end_y))

            orig_x1 = int(x1 / self.scale)
            orig_x2 = int(x2 / self.scale)
            orig_y1 = int(y1 / self.scale)
            orig_y2 = int(y2 / self.scale)

            w = orig_x2 - orig_x1
            h = orig_y2 - orig_y1

            if w >= 15 and h >= 20:
                self.selected_whole_crop = self.frame_bgr[orig_y1:orig_y2, orig_x1:orig_x2].copy()
                prev_rgb = cv2.cvtColor(cv2.resize(self.selected_whole_crop, (50, int(50 * h / max(1, w))), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
                self.preview_tk = ImageTk.PhotoImage(image=Image.fromarray(prev_rgb))
                self.preview_lbl.config(image=self.preview_tk, text="")
                self.status_lbl.config(text=f"✅ 已框选全身: {w}x{h} px (脚底即框底)。请点击【下一步】继续", fg="#69f0ae")
                self.btn_next.config(state=tk.NORMAL, bg="#2e7d32")
            else:
                self.status_lbl.config(text="⚠️ 全身选区太小，请重新拉框", fg="#ff5252")
                self.btn_next.config(state=tk.DISABLED, bg="#424242")

    def _on_canvas_motion(self, event):
        if self.stage != 2 or self.feature_polygon_closed or not self.feature_polygon_display:
            return
        self._redraw_feature_polygon((event.x, event.y))

    def _redraw_feature_polygon(self, preview_point=None):
        self.canvas.delete("feature_polygon")
        pts = list(self.feature_polygon_display)
        if preview_point is not None and pts:
            px = max(0, min(self.disp_stage2_w - 1, int(preview_point[0])))
            py = max(0, min(self.disp_stage2_h - 1, int(preview_point[1])))
            pts.append((px, py))
        if len(pts) >= 2:
            flat = [coord for point in pts for coord in point]
            self.canvas.create_line(
                *flat,
                fill="#ffea00",
                width=2,
                dash=() if self.feature_polygon_closed else (4, 2),
                tags="feature_polygon",
            )
        if self.feature_polygon_closed and len(self.feature_polygon_display) >= 3:
            first = self.feature_polygon_display[0]
            last = self.feature_polygon_display[-1]
            self.canvas.create_line(
                last[0], last[1], first[0], first[1],
                fill="#ffea00", width=2, tags="feature_polygon",
            )
        for x, y in self.feature_polygon_display:
            self.canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3,
                fill="#ffea00", outline="#202026", tags="feature_polygon",
            )

    def _undo_feature_polygon_point(self, _event=None):
        if self.stage != 2:
            return
        if self.feature_polygon_closed:
            self.feature_polygon_closed = False
            self.selected_feat_crop = None
            self.feature_mask = None
            self.feature_polygon_in_crop = None
        elif self.feature_polygon_display:
            self.feature_polygon_display.pop()
        self.btn_face_left.config(state=tk.DISABLED, bg="#424242")
        self.btn_face_right.config(state=tk.DISABLED, bg="#424242")
        self._redraw_feature_polygon()

    def _finalize_feature_polygon(self, _event=None):
        if self.stage != 2:
            return
        if len(self.feature_polygon_display) < 3:
            self.status_lbl.config(
                text="⚠️ 多边形至少需要3个顶点；请继续左键添加",
                fg="#ff5252",
            )
            return "break"

        char_h, char_w = self.whole_bgr.shape[:2]
        orig_points = []
        for dx, dy in self.feature_polygon_display:
            ox = max(0, min(char_w - 1, int(round(dx / self.scale_stage2))))
            oy = max(0, min(char_h - 1, int(round(dy / self.scale_stage2))))
            orig_points.append((ox, oy))
        points_np = np.asarray(orig_points, dtype=np.int32)
        fx, fy, fw, fh = cv2.boundingRect(points_np)
        if fw < 8 or fh < 8:
            self.status_lbl.config(text="⚠️ 多边形范围太小，请重新标定", fg="#ff5252")
            return "break"

        crop = self.whole_bgr[fy:fy + fh, fx:fx + fw].copy()
        local_points = points_np - np.asarray([fx, fy], dtype=np.int32)
        mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.fillPoly(mask, [local_points], 255)
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < 80:
            self.status_lbl.config(
                text=f"⚠️ 多边形有效区域太小（{mask_pixels}px），请完整圈选帽子、头发或服饰",
                fg="#ff5252",
            )
            return "break"

        self.selected_feat_crop = crop
        self.feature_mask = mask
        self.feature_box_in_whole = (int(fx), int(fy), int(fw), int(fh))
        self.feature_polygon_in_crop = [
            [int(p[0]), int(p[1])] for p in local_points.tolist()
        ]
        self.feature_polygon_closed = True
        self._redraw_feature_polygon()

        preview = crop.copy()
        preview[mask == 0] = (40, 40, 40)
        preview_w = 90
        preview_h = max(35, int(round(preview_w * fh / max(1, fw))))
        prev_rgb = cv2.cvtColor(
            cv2.resize(preview, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2RGB,
        )
        self.preview_tk2 = ImageTk.PhotoImage(image=Image.fromarray(prev_rgb))
        self.preview_lbl.config(image=self.preview_tk2, text="")
        coverage = mask_pixels / float(max(1, fw * fh)) * 100.0
        self.status_lbl.config(
            text=(
                f"✅ 多边形特征: {fw}x{fh}px，{len(local_points)}个顶点，"
                f"有效像素{coverage:.0f}%；请选择当前角色朝向保存"
            ),
            fg="#ffd54f",
        )
        self.btn_face_left.config(state=tk.NORMAL, bg="#00838f")
        self.btn_face_right.config(state=tk.NORMAL, bg="#2e7d32")
        return "break"

    def _go_to_stage_2(self):
        if self.selected_whole_crop is None:
            return
        self.stage = 2
        self.whole_bgr = self.selected_whole_crop
        char_h, char_w = self.whole_bgr.shape[:2]

        self.title("👑 两段式角色标定 (第 2 步：多边形核心特征 & 选择朝向)")
        self.lbl_step_tip.config(
            text="🎩 步骤 2/2：左键逐点圈选核心特征；右键/Enter闭合，Backspace撤销。只保留多边形内部像素参与识别。",
            fg="#00e5ff"
        )
        self.btn_next.pack_forget()
        self.facing_btn_frame.pack(side=tk.RIGHT)

        max_box = 480
        self.scale_stage2 = min(max_box / float(max(1, char_w)), max_box / float(max(1, char_h)))
        self.scale_stage2 = max(2.5, min(6.0, self.scale_stage2))
        self.disp_stage2_w = int(char_w * self.scale_stage2)
        self.disp_stage2_h = int(char_h * self.scale_stage2)

        zoom_rgb = cv2.cvtColor(cv2.resize(self.whole_bgr, (self.disp_stage2_w, self.disp_stage2_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
        self.tk_img_stage2 = ImageTk.PhotoImage(image=Image.fromarray(zoom_rgb))

        self.canvas.config(width=self.disp_stage2_w, height=self.disp_stage2_h)
        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(0, 0, image=self.tk_img_stage2, anchor=tk.NW)
        self.feature_polygon_display = []
        self.feature_polygon_closed = False
        self.status_lbl.config(text="请左键逐点圈选帽子、头发或服饰特征...", fg="#e0e0e0")

    def _finish_calibration(self, facing: str):
        if (
            self.whole_bgr is not None
            and self.selected_feat_crop is not None
            and self.feature_box_in_whole is not None
            and self.feature_mask is not None
        ):
            self.on_complete_callback(
                self.whole_bgr,
                self.selected_feat_crop,
                self.feature_box_in_whole,
                facing,
                self.feature_mask,
                self.feature_polygon_in_crop,
            )
            self.destroy()




# ─────────────────────────────────────────────────────────────────────────────
#  OcrRoiCalibrationDialog：地图名称 OCR 识别范围标定（拖拽拉框）
# ─────────────────────────────────────────────────────────────────────────────
class OcrRoiCalibrationDialog(tk.Toplevel):
    def __init__(
        self,
        parent,
        frame_bgr: np.ndarray,
        current_roi: Optional[Dict[str, int]],
        on_complete_callback,
        *,
        title_text: str = "📐 地图 OCR 识别范围标定 (鼠标拖拽拉框)",
        tip_text: str = (
            "🎯 请在小地图左上方【鼠标拖拽拉框】框选地图名称区域"
            "（建议仅框选小地图内部地图名，避开外部NPC如出租车），然后点击【确认保存范围】"
        ),
        purpose_text: str = "OCR",
        default_roi: Optional[Dict[str, int]] = None,
        min_size: Tuple[int, int] = (20, 10),
    ):
        super().__init__(parent)
        self.title(title_text)
        self.configure(bg="#1a1a20")
        self.transient(parent)
        self.grab_set()

        self.frame_bgr = frame_bgr
        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        self.on_complete_callback = on_complete_callback
        self.purpose_text = purpose_text
        self.default_roi = (
            default_roi.copy()
            if default_roi is not None
            else ({"x": 20, "y": 18, "w": 280, "h": 57} if purpose_text == "OCR" else None)
        )
        self.min_width = max(1, int(min_size[0]))
        self.min_height = max(1, int(min_size[1]))

        # 根据当前显示器工作区计算，不再使用固定的 1024px 视口。
        self.disp_w, self.disp_h, self.scale = fit_image_to_work_area(
            parent, self.orig_w, self.orig_h, reserved_width=48, reserved_height=225
        )
        fit_window_to_work_area(
            self,
            (self.disp_w + 36, self.disp_h + 205),
            (min(680, self.disp_w + 36), min(480, self.disp_h + 205)),
            parent=parent,
        )

        # 选区数据 (原图坐标)
        self.selected_roi: Optional[Dict[str, int]] = current_roi.copy() if current_roi else None

        # 顶部提示条
        self.top_bar = tk.Frame(self, bg="#202026", pady=6, padx=12)
        self.top_bar.pack(fill=tk.X)
        self.lbl_step_tip = tk.Label(
            self.top_bar,
            text=tip_text,
            font=("Segoe UI", 9, "bold"), fg="#00e5ff", bg="#202026",
            justify=tk.LEFT, wraplength=max(320, self.disp_w - 24),
        )
        self.lbl_step_tip.pack(fill=tk.X)

        # 画布区域
        self.canvas_frame = tk.Frame(self, bg="#101014")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        rgb_disp = cv2.cvtColor(cv2.resize(frame_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(rgb_disp))

        self.canvas = tk.Canvas(self.canvas_frame, width=self.disp_w, height=self.disp_h, bg="#000000", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(anchor=tk.CENTER)
        self.canvas_img_id = self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

        # 底部控制条
        self.bot_bar = tk.Frame(self, bg="#202026", pady=8, padx=12)
        self.bot_bar.pack(fill=tk.X)

        info_row = tk.Frame(self.bot_bar, bg="#202026")
        info_row.pack(fill=tk.X)
        action_row = tk.Frame(self.bot_bar, bg="#202026")
        action_row.pack(fill=tk.X, pady=(6, 0))

        self.preview_lbl = tk.Label(info_row, text="未选区", bg="#2a2a32", fg="#b0bec5", font=("Consolas", 9), width=20, height=2)
        self.preview_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.status_lbl = tk.Label(info_row, text=f"请拖拽框选{purpose_text}区域...", font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026", anchor=tk.W, justify=tk.LEFT)
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 按钮区
        self.btn_cancel = tk.Button(action_row, text="❌ 取消 (Esc)", font=("Segoe UI", 9), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=12, pady=4, command=self.destroy)
        self.btn_cancel.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_save = tk.Button(action_row, text="💾 确认保存范围", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=14, pady=4, state=tk.DISABLED, command=self._confirm_save)
        self.btn_save.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_reset = tk.Button(action_row, text="🔄 恢复默认范围", font=("Segoe UI", 9), fg="#ffffff", bg="#37474f", relief=tk.FLAT, padx=10, pady=4, command=self._reset_default)
        if self.default_roi is not None:
            self.btn_reset.pack(side=tk.RIGHT, padx=(6, 0))

        # 鼠标交互变量
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Escape>", lambda e: self.destroy())

        # 如果已有 ROI，在画布上画出来并显示预览
        if self.selected_roi:
            self._display_existing_roi(self.selected_roi)

    def _display_existing_roi(self, roi: Dict[str, int]):
        rx = roi.get("x", 20)
        ry = roi.get("y", 18)
        rw = roi.get("w", 260)
        rh = roi.get("h", 55)
        disp_x1 = int(rx * self.scale)
        disp_y1 = int(ry * self.scale)
        disp_x2 = int((rx + rw) * self.scale)
        disp_y2 = int((ry + rh) * self.scale)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline="#00e676", width=2, dash=(4, 2))
        self._update_preview(rx, ry, rw, rh)
        self.status_lbl.config(text=f"当前{self.purpose_text}范围: [x={rx}, y={ry}, w={rw}, h={rh}] (可重新拖拽修改)", fg="#69f0ae")
        self.btn_save.config(state=tk.NORMAL, bg="#2e7d32")

    def _on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.btn_save.config(state=tk.DISABLED, bg="#424242")

    def _on_mouse_move(self, event):
        if self.start_x is None:
            return
        cur_x, cur_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="#00e5ff", width=2, dash=(4, 2))

    def _on_mouse_up(self, event):
        if self.start_x is None:
            return
        end_x, end_y = event.x, event.y

        x1 = max(0, min(self.start_x, end_x))
        x2 = min(self.disp_w, max(self.start_x, end_x))
        y1 = max(0, min(self.start_y, end_y))
        y2 = min(self.disp_h, max(self.start_y, end_y))

        orig_x1 = int(x1 / self.scale)
        orig_x2 = int(x2 / self.scale)
        orig_y1 = int(y1 / self.scale)
        orig_y2 = int(y2 / self.scale)

        w = orig_x2 - orig_x1
        h = orig_y2 - orig_y1

        if w >= self.min_width and h >= self.min_height:
            self.selected_roi = {"x": orig_x1, "y": orig_y1, "w": w, "h": h}
            self._update_preview(orig_x1, orig_y1, w, h)
            self.status_lbl.config(text=f"✅ 已框选{self.purpose_text}范围: [x={orig_x1}, y={orig_y1}, w={w}, h={h}]。请点击【确认保存范围】", fg="#69f0ae")
            self.btn_save.config(state=tk.NORMAL, bg="#2e7d32")
        else:
            self.status_lbl.config(text=f"⚠️ 选区太小，请重新拉框框选{self.purpose_text}区域", fg="#ff5252")
            self.btn_save.config(state=tk.DISABLED, bg="#424242")

    def _update_preview(self, x, y, w, h):
        crop = self.frame_bgr[y:y+h, x:x+w]
        if crop.size > 0:
            prev_w = 120
            prev_h = max(15, int(prev_w * h / max(1, w)))
            prev_rgb = cv2.cvtColor(cv2.resize(crop, (prev_w, prev_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            self.preview_tk = ImageTk.PhotoImage(image=Image.fromarray(prev_rgb))
            self.preview_lbl.config(image=self.preview_tk, text="")

    def _reset_default(self):
        if self.default_roi is None:
            return
        default_roi = self.default_roi.copy()
        self.selected_roi = default_roi
        self._display_existing_roi(default_roi)
        self.status_lbl.config(text="已恢复默认范围: [x=20, y=18, w=280, h=57]。点击【确认保存范围】生效", fg="#ffd54f")

    def _confirm_save(self):
        if self.selected_roi and self.on_complete_callback:
            self.on_complete_callback(self.selected_roi)
            self.destroy()



# ─────────────────────────────────────────────────────────────────────────────
#  MinimapCropCalibrationDialog：高倍率放大手动框选小地图内画布范围
# ─────────────────────────────────────────────────────────────────────────────
class MinimapCropCalibrationDialog(tk.Toplevel):
    """
    针对小地图区域的放大框选弹窗（使用 1.5x 最近邻像素级拉框）。
    用户可以精确框选小地图深色物理画布边缘，彻底消除任何细微边框误差。
    """
    def __init__(self, parent, frame_bgr: np.ndarray, current_box: Optional[Tuple[int, int, int, int]], on_confirm_callback):
        super().__init__(parent)
        self.title("📐 手动精确框选小地图范围 (高倍率放大标定)")
        dialog_w, _dialog_h = fit_window_to_work_area(
            self, (1060, 780), (680, 500), parent=parent
        )
        self.configure(bg="#1a1a20")
        self.transient(parent)
        self.grab_set()

        self.full_frame = frame_bgr
        fh, fw = frame_bgr.shape[:2]
        self.on_confirm_callback = on_confirm_callback

        # 截取左上角小地图区域 (原图坐标系)
        # sw, sh 覆盖所有主流小地图（宽幅地铁/神木小地图）
        self.crop_w = min(fw, max(420, int(fw * 0.40)))
        self.crop_h = min(fh, max(360, int(fh * 0.40)))
        self.sub_bgr = frame_bgr[:self.crop_h, :self.crop_w].copy()

        # 1.5x 在保留像素辨识度的同时显著减少滚动距离和画布占用。
        self.zoom = 1.5
        self.disp_w = int(self.crop_w * self.zoom)
        self.disp_h = int(self.crop_h * self.zoom)

        # 选区数据 (原图坐标 x, y, w, h)
        self.selected_box: Optional[Tuple[int, int, int, int]] = current_box

        # 顶部提示条
        top_bar = tk.Frame(self, bg="#202026", pady=6, padx=12)
        top_bar.pack(fill=tk.X)
        tk.Label(
            top_bar,
            text="🎯 【1.5x 像素级放大模式】请在下方小地图深色物理画布上【鼠标拖拽拉框】，精确贴合内框边缘，然后点击【保存当前范围】",
            font=("Segoe UI", 9, "bold"), fg="#ffd54f", bg="#202026",
            justify=tk.LEFT, wraplength=max(320, dialog_w - 40),
        ).pack(fill=tk.X)

        # 画布带滚动条区域
        canvas_container = tk.Frame(self, bg="#101014")
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.canvas = tk.Canvas(canvas_container, bg="#0a0a0e", cursor="crosshair", highlightthickness=0)
        hbar = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = tk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 生成 1.5x 最近邻插值放大图 (最近邻保留清晰的真实像素格，无模糊渐变)
        sub_resized = cv2.resize(self.sub_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        rgb_disp = cv2.cvtColor(sub_resized, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(rgb_disp))

        self.canvas_img_id = self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
        self.canvas.config(scrollregion=(0, 0, self.disp_w, self.disp_h))

        # 底部控制条
        bot_bar = tk.Frame(self, bg="#202026", pady=8, padx=12)
        bot_bar.pack(fill=tk.X)

        preview_row = tk.Frame(bot_bar, bg="#202026")
        preview_row.pack(fill=tk.X)
        action_row = tk.Frame(bot_bar, bg="#202026")
        action_row.pack(fill=tk.X, pady=(6, 0))

        # 🔍 准星 10x 像素网格超微放大镜面板
        mag_card = tk.LabelFrame(preview_row, text="🔍 准星像素放大镜 (10x 像素级微距网格)", font=("Segoe UI", 8, "bold"), fg="#00e5ff", bg="#1a1c23", padx=6, pady=4)
        mag_card.pack(side=tk.LEFT, padx=(0, 10))

        self.mag_lbl = tk.Label(mag_card, bg="#000000", bd=1, relief=tk.SOLID)
        self.mag_lbl.pack(side=tk.LEFT, padx=(0, 6))

        self.mag_info_lbl = tk.Label(mag_card, text="准星位置: X=-- Y=--\n移动鼠标或拖拽框选\n即可查看单像素微距", justify=tk.LEFT, font=("Consolas", 8), fg="#ffd54f", bg="#1a1c23", width=18)
        self.mag_info_lbl.pack(side=tk.LEFT)

        # 选区缩略图预览
        self.preview_lbl = tk.Label(preview_row, text="选区缩略图", bg="#2a2a32", fg="#b0bec5", font=("Consolas", 8), width=16, height=2)
        self.preview_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.status_lbl = tk.Label(preview_row, text="请在小地图深色物理画布上拉框...", font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026", anchor=tk.W, justify=tk.LEFT)
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 按钮区
        tk.Button(action_row, text="❌ 取消 (Esc)", font=("Segoe UI", 9), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=12, pady=4, command=self.destroy).pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_save = tk.Button(action_row, text="💾 保存当前范围", font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#424242", relief=tk.FLAT, padx=14, pady=4, state=tk.DISABLED, command=self._confirm_save)
        self.btn_save.pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(action_row, text="🔄 恢复自动探测", font=("Segoe UI", 9), fg="#ffffff", bg="#37474f", relief=tk.FLAT, padx=10, pady=4, command=self._reset_to_auto).pack(side=tk.RIGHT, padx=(6, 0))

        # 鼠标交互变量
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.mag_tk = None

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<Motion>", self._on_mouse_hover)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Escape>", lambda e: self.destroy())

        # 初始刷新放大镜 (定位在左上角附近)
        self._update_magnifier(10, 75)

        # 如果已有框选或默认识别结果，绘制出来
        if self.selected_box:
            self._display_box(self.selected_box)

    def _display_box(self, box: Tuple[int, int, int, int]):
        bx, by, bw, bh = box
        disp_x1 = int(bx * self.zoom)
        disp_y1 = int(by * self.zoom)
        disp_x2 = int((bx + bw) * self.zoom)
        disp_y2 = int((by + bh) * self.zoom)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(disp_x1, disp_y1, disp_x2, disp_y2, outline="#00e676", width=2, dash=(4, 2))
        self._update_preview(bx, by, bw, bh)
        self._update_magnifier(bx, by)
        self.status_lbl.config(text=f"当前选区: [x={bx}, y={by}, 宽={bw}, 高={bh} px] (可重新拉框调整)", fg="#69f0ae")
        self.btn_save.config(state=tk.NORMAL, bg="#2e7d32")

    def _update_magnifier(self, orig_x: int, orig_y: int):
        """
        高倍率微距放大镜：以原图像素 (orig_x, orig_y) 为准星中心，
        抓取 17x17 原生像素窗口，以 10x 放大成 170x170 图像，
        并在其上绘制单个像素的微距网格线与红绿双色十字准心，
        保证用户 100% 清晰看清每一个像素的颜色与边框交界。
        """
        fh, fw = self.full_frame.shape[:2]
        orig_x = max(0, min(fw - 1, int(orig_x)))
        orig_y = max(0, min(fh - 1, int(orig_y)))

        radius = 8  # 8+1+8 = 17 像素宽度
        x1 = orig_x - radius
        x2 = orig_x + radius + 1
        y1 = orig_y - radius
        y2 = orig_y + radius + 1

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - fw)
        pad_bottom = max(0, y2 - fh)

        sx1 = max(0, x1)
        sx2 = min(fw, x2)
        sy1 = max(0, y1)
        sy2 = min(fh, y2)

        crop = self.full_frame[sy1:sy2, sx1:sx2].copy()
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(18, 18, 22))

        pix_size = 10
        mag_w = 17 * pix_size
        mag_h = 17 * pix_size
        big = cv2.resize(crop, (mag_w, mag_h), interpolation=cv2.INTER_NEAREST)

        # 绘制单个像素的分割网格线 (低明度暗灰，不遮挡像素原色)
        for i in range(1, 17):
            big[:, i * pix_size] = (45, 45, 52)
            big[i * pix_size, :] = (45, 45, 52)

        # 准星中心像素 (索引 8，坐标范围 [80..89])
        cx_s = radius * pix_size
        cx_e = cx_s + pix_size
        cy_s = radius * pix_size
        cy_e = cy_s + pix_size

        # 用高对比度准星线瞄准中心像素
        cv2.rectangle(big, (cx_s, cy_s), (cx_e - 1, cy_e - 1), (0, 255, 255), 1)
        # 上下左右延伸的十字瞄准线
        cv2.line(big, (cx_s + pix_size // 2, 0), (cx_s + pix_size // 2, cy_s - 2), (0, 255, 255), 1)
        cv2.line(big, (cx_s + pix_size // 2, cy_e + 2), (cx_s + pix_size // 2, mag_h - 1), (0, 255, 255), 1)
        cv2.line(big, (0, cy_s + pix_size // 2), (cx_s - 2, cy_s + pix_size // 2), (0, 255, 255), 1)
        cv2.line(big, (cx_e + 2, cy_s + pix_size // 2), (mag_w - 1, cy_s + pix_size // 2), (0, 255, 255), 1)

        # 读取准星中心像素的 BGR 颜色
        center_bgr = self.full_frame[orig_y, orig_x]
        b, g, r = int(center_bgr[0]), int(center_bgr[1]), int(center_bgr[2])
        is_dark = (r < 85 and g < 85 and b < 85)
        color_desc = "深色地图" if is_dark else "浅色边框"

        rgb_big = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
        self.mag_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb_big))
        self.mag_lbl.config(image=self.mag_tk)

        self.mag_info_lbl.config(
            text=f"准星像素: ({orig_x}, {orig_y})\n"
                 f"RGB: ({r}, {g}, {b})\n"
                 f"属性: 【{color_desc}】"
        )

    def _on_mouse_hover(self, event):
        """鼠标悬停时更新放大镜视口"""
        if self.start_x is None:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            orig_x = int(round(cur_x / self.zoom))
            orig_y = int(round(cur_y / self.zoom))
            self._update_magnifier(orig_x, orig_y)

    def _on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        orig_x = int(round(self.start_x / self.zoom))
        orig_y = int(round(self.start_y / self.zoom))
        self._update_magnifier(orig_x, orig_y)

        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.btn_save.config(state=tk.DISABLED, bg="#424242")

    def _on_mouse_move(self, event):
        if self.start_x is None:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # 实时根据当前拉框终点准星更新放大镜
        orig_x = int(round(cur_x / self.zoom))
        orig_y = int(round(cur_y / self.zoom))
        self._update_magnifier(orig_x, orig_y)

        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="#00e5ff", width=2, dash=(4, 2))

    def _on_mouse_up(self, event):
        if self.start_x is None:
            return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        orig_x = int(round(end_x / self.zoom))
        orig_y = int(round(end_y / self.zoom))
        self._update_magnifier(orig_x, orig_y)

        x1 = max(0, min(self.start_x, end_x))
        x2 = min(self.disp_w, max(self.start_x, end_x))
        y1 = max(0, min(self.start_y, end_y))
        y2 = min(self.disp_h, max(self.start_y, end_y))

        orig_x1 = int(round(x1 / self.zoom))
        orig_x2 = int(round(x2 / self.zoom))
        orig_y1 = int(round(y1 / self.zoom))
        orig_y2 = int(round(y2 / self.zoom))

        w = orig_x2 - orig_x1
        h = orig_y2 - orig_y1

        if w >= 20 and h >= 20:
            self.selected_box = (orig_x1, orig_y1, w, h)
            self._update_preview(orig_x1, orig_y1, w, h)
            self.status_lbl.config(text=f"✅ 已框选小地图: [x={orig_x1}, y={orig_y1}, 尺寸={w}x{h} px]。点击【保存当前范围】生效", fg="#69f0ae")
            self.btn_save.config(state=tk.NORMAL, bg="#2e7d32")
        else:
            self.status_lbl.config(text="⚠️ 选区尺寸过小，请重新拉框", fg="#ff5252")
            self.btn_save.config(state=tk.DISABLED, bg="#424242")

        self.start_x = None
        self.start_y = None

    def _update_preview(self, x, y, w, h):
        crop = self.sub_bgr[y:y+h, x:x+w]
        if crop.size > 0:
            prev_w = 90
            prev_h = max(20, int(prev_w * h / max(1, w)))
            prev_rgb = cv2.cvtColor(cv2.resize(crop, (prev_w, prev_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            self.preview_tk = ImageTk.PhotoImage(image=Image.fromarray(prev_rgb))
            self.preview_lbl.config(image=self.preview_tk, text="")

    def _reset_to_auto(self):
        self.selected_box = None
        if self.on_confirm_callback:
            self.on_confirm_callback(None)
        self.destroy()

    def _confirm_save(self):
        if self.selected_box and self.on_confirm_callback:
            self.on_confirm_callback(self.selected_box)
            self.destroy()



# ─────────────────────────────────────────────────────────────────────────────
#  EditMonsterDialog：自定义怪物增删改 + 模糊搜索
# ─────────────────────────────────────────────────────────────────────────────
class EditMonsterDialog:
    """
    怪物自定义编辑弹窗。
    设计原则:
      - 每次增/删/改操作立即调用 on_save_callback，主界面实时更新
      - iid 使用 mob_id 字符串，不依赖列表下标，删除不会错位
      - 本地离线库优先搜索，联网作为异步补充（不阻塞，不超时卡死）
    """
    IO_SEARCH_URL = "https://maplestory.io/api/GMS/83/mob?overrideVersion=1&page=0&count=20&search="

    # ── 本地离线怪物库（覆盖经典 GMS v83 全量怪物，无需网络） ─────────────────
    LOCAL_MOB_DB = [
        (100100, "Snail (蜗牛)"),
        (100101, "Blue Snail (蓝蜗牛)"),
        (100102, "Red Snail (红蜗牛)"),
        (110100, "Shroom (蘑菇头)"),
        (120100, "Orange Mushroom (橙蘑菇)"),
        (130100, "Stump (树桩)"),
        (130101, "Dark Stump (暗树桩)"),
        (210100, "Slime (史莱姆)"),
        (210101, "Green Slime (绿史莱姆)"),
        (1110100, "Green Mushroom (绿蘑菇)"),
        (1110101, "Horny Mushroom (角蘑菇)"),
        (1120100, "Zombie Mushroom (僵尸蘑菇)"),
        (1140100, "Ghost Stump (幽灵树桩)"),
        (1140130, "Smirking Ghost Stump (坏笑树桩)"),
        (1210100, "Pig (猪)"),
        (1210101, "Ribbon Pig (蝴蝶结猪)"),
        (2110200, "Horny Mushroom B (角蘑菇B)"),
        (2130100, "Curse Eye (诅咒眼)"),
        (2130103, "Zombie (僵尸)"),
        (2230100, "Evil Eye (独眼兽)"),
        (2230101, "Evil Eye 2 (独眼兽2)"),
        (2230102, "Cold Eye (冷眼)"),
        (2230110, "Wooden Mask (木面具)"),
        (3210100, "Zombie Lupin (僵尸猴)"),
        (3210101, "Lupin / Angel Monkey (天使猴)"),
        (3230100, "Ligator (鳄鱼)"),
        (3230101, "Croco (大鳄鱼)"),
        (4130100, "Curse Eye B (诅咒独眼B)"),
        (5130103, "Ice Drake (冰龙)"),
        (5130104, "Dark Drake (暗龙)"),
        (6230300, "Balrog (巴尔洛克)"),
        (8150100, "King Slime (史莱姆王)"),
        (9001000, "Clang (铁甲怪)"),
        (9300018, "Jr. Balrog (小巴尔洛克)"),
    ]

    def __init__(self, parent, map_id: int, map_name: str, current_mobs: List[Dict], on_save_callback, get_image_callback=None):
        self.top = tk.Toplevel(parent)
        self.top.title(f"自定义怪物种类 - 【{map_name}】")
        fit_window_to_work_area(self.top, (640, 600), (540, 440), parent=parent)
        self.top.configure(bg="#1a1a20")
        self.top.transient(parent)
        self.top.grab_set()

        self.map_id = map_id
        self.map_name = map_name
        self.mobs: List[Dict] = [dict(m) for m in current_mobs]
        self.on_save_callback = on_save_callback
        self.get_image_callback = get_image_callback
        self._search_timer: Optional[str] = None

        self._init_ui()

    def _init_ui(self):
        hdr = tk.Frame(self.top, bg="#22222a", padx=14, pady=10)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"地图: {self.map_name}  (Map ID: {self.map_id})",
                 font=("Segoe UI", 11, "bold"), fg="#00e5ff", bg="#22222a").pack(anchor=tk.W)
        tk.Label(hdr, text="鼠标悬浮可预览怪物图片；删除 / 应用 / 添加 操作立即保存并在主界面生效。",
                 font=("Segoe UI", 8), fg="#b0bec5", bg="#22222a").pack(anchor=tk.W, pady=(2, 0))

        content = tk.Frame(self.top, bg="#1a1a20", padx=12, pady=8)
        content.pack(fill=tk.BOTH, expand=True)

        # 当前怪物列表
        list_lf = tk.LabelFrame(content, text="当前怪物列表 (鼠标悬浮预览)", font=("Segoe UI", 9, "bold"),
                                 fg="#ffb74d", bg="#202026", padx=6, pady=6)
        list_lf.pack(fill=tk.BOTH, expand=True)

        cols = ("id", "name", "keywords")
        self.tree = ttk.Treeview(list_lf, columns=cols, show="headings", height=5)
        self.tree.heading("id", text="Mob ID")
        self.tree.heading("name", text="怪物名称")
        self.tree.heading("keywords", text="特征关键词")
        self.tree.column("id", width=75, anchor=tk.CENTER)
        self.tree.column("name", width=170, anchor=tk.W)
        self.tree.column("keywords", width=230, anchor=tk.W)
        sb = ttk.Scrollbar(list_lf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        if self.get_image_callback:
            self.dialog_tree_tooltip = MonsterTooltip(self.tree, self.get_image_callback)
        self._refresh_tree()

        # 操作按钮
        btns = tk.Frame(content, bg="#1a1a20")
        btns.pack(fill=tk.X, pady=(3, 6))
        tk.Button(btns, text="删除选中（立即生效）", font=("Segoe UI", 9),
                  fg="#ffffff", bg="#b71c1c", relief=tk.FLAT, padx=10, pady=3,
                  command=self._on_delete).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="应用修改（立即生效）", font=("Segoe UI", 9),
                  fg="#ffffff", bg="#1565c0", relief=tk.FLAT, padx=10, pady=3,
                  command=self._on_update).pack(side=tk.LEFT)

        # 搜索区
        search_lf = tk.LabelFrame(content,
                                   text="模糊搜索怪物（鼠标悬浮预览，点击填入）",
                                   font=("Segoe UI", 9, "bold"),
                                   fg="#69f0ae", bg="#202026", padx=8, pady=6)
        search_lf.pack(fill=tk.X, pady=(0, 4))

        sr = tk.Frame(search_lf, bg="#202026")
        sr.pack(fill=tk.X, pady=2)
        tk.Label(sr, text="搜索怪物名:", font=("Segoe UI", 9), fg="#e0e0e0",
                 bg="#202026", width=10, anchor="w").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search_changed)
        tk.Entry(sr, textvariable=self.search_var, bg="#2a2a32",
                 fg="#ffffff", insertbackground="#ffffff", width=30).pack(side=tk.LEFT, padx=4)
        self.search_status = tk.Label(sr, text="", font=("Segoe UI", 8), fg="#90a4ae", bg="#202026")
        self.search_status.pack(side=tk.LEFT, padx=6)

        res_cols = ("r_id", "r_name")
        self.res_tree = ttk.Treeview(search_lf, columns=res_cols, show="headings", height=4)
        self.res_tree.heading("r_id", text="Mob ID")
        self.res_tree.heading("r_name", text="怪物名称（悬浮预览 / 点击填入）")
        self.res_tree.column("r_id", width=80, anchor=tk.CENTER)
        self.res_tree.column("r_name", width=320, anchor=tk.W)
        self.res_tree.pack(fill=tk.X, pady=(4, 0))
        self.res_tree.bind("<<TreeviewSelect>>", self._on_search_select)
        if self.get_image_callback:
            self.search_res_tooltip = MonsterTooltip(self.res_tree, self.get_image_callback)

        # 编辑输入区
        edit_lf = tk.LabelFrame(content, text="属性编辑区", font=("Segoe UI", 9, "bold"),
                                  fg="#b39ddb", bg="#202026", padx=8, pady=6)
        edit_lf.pack(fill=tk.X, pady=(4, 0))

        r1 = tk.Frame(edit_lf, bg="#202026")
        r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="Mob ID:", font=("Segoe UI", 9), fg="#e0e0e0",
                 bg="#202026", width=9, anchor="w").pack(side=tk.LEFT)
        self.ent_id = tk.Entry(r1, bg="#2a2a32", fg="#ffffff", insertbackground="#ffffff", width=14)
        self.ent_id.pack(side=tk.LEFT, padx=4)
        tk.Label(r1, text="名称:", font=("Segoe UI", 9), fg="#e0e0e0",
                 bg="#202026", width=6, anchor="w").pack(side=tk.LEFT, padx=(10, 0))
        self.ent_name = tk.Entry(r1, bg="#2a2a32", fg="#ffffff", insertbackground="#ffffff", width=24)
        self.ent_name.pack(side=tk.LEFT, padx=4)

        r2 = tk.Frame(edit_lf, bg="#202026")
        r2.pack(fill=tk.X, pady=2)
        tk.Label(r2, text="特征词:", font=("Segoe UI", 9), fg="#e0e0e0",
                 bg="#202026", width=9, anchor="w").pack(side=tk.LEFT)
        self.ent_kw = tk.Entry(r2, bg="#2a2a32", fg="#ffffff", insertbackground="#ffffff", width=44)
        self.ent_kw.pack(side=tk.LEFT, padx=4)
        tk.Label(r2, text="(逗号分隔)", font=("Segoe UI", 8), fg="#90a4ae", bg="#202026").pack(side=tk.LEFT)

        tk.Button(edit_lf, text="添加为新怪物（立即生效）", font=("Segoe UI", 9, "bold"),
                  fg="#ffffff", bg="#00695c", relief=tk.FLAT, padx=10, pady=3,
                  command=self._on_add).pack(anchor=tk.W, pady=(4, 2))

        # 底部
        btn_bar = tk.Frame(self.top, bg="#1a1a20", padx=12, pady=8)
        btn_bar.pack(fill=tk.X)
        tk.Label(btn_bar, text="增删改操作立即保存，可直接关闭。",
                 font=("Segoe UI", 8, "italic"), fg="#78909c", bg="#1a1a20").pack(side=tk.LEFT)
        tk.Button(btn_bar, text="完成关闭",
                  font=("Segoe UI", 10, "bold"), fg="#ffffff", bg="#2e7d32",
                  relief=tk.FLAT, padx=16, pady=5,
                  command=self.top.destroy).pack(side=tk.RIGHT)

    def _refresh_tree(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        for m in self.mobs:
            mid = str(m.get("id", ""))
            mname = m.get("name", "")
            kws_raw = m.get("keywords", [])
            kws_str = ", ".join(kws_raw) if kws_raw else m.get("file_prefix", "")
            iid = mid
            attempt = 0
            while self.tree.exists(iid):
                attempt += 1
                iid = f"{mid}_{attempt}"
            self.tree.insert("", tk.END, iid=iid, values=(mid, mname, kws_str))

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        if not vals:
            return
        self.ent_id.delete(0, tk.END);   self.ent_id.insert(0, str(vals[0]))
        self.ent_name.delete(0, tk.END); self.ent_name.insert(0, str(vals[1]))
        self.ent_kw.delete(0, tk.END);   self.ent_kw.insert(0, str(vals[2]))

    def _on_update(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先在上方列表中选择要修改的怪物！", parent=self.top)
            return
        old_mid = str(self.tree.item(sel[0], "values")[0])
        mname = self.ent_name.get().strip()
        mid_str = self.ent_id.get().strip()
        kw_str = self.ent_kw.get().strip()
        if not mname:
            messagebox.showwarning("提示", "怪物名称不能为空！", parent=self.top)
            return
        try:
            mid = int(mid_str) if mid_str else int(old_mid)
        except Exception:
            mid = 0
        kws = [k.strip() for k in kw_str.split(",") if k.strip()] or [mname.lower()]
        for m in self.mobs:
            if str(m.get("id", "")) == old_mid:
                m["id"] = mid; m["name"] = mname
                m["keywords"] = kws; m["file_prefix"] = kws[0]
                break
        self._refresh_tree()
        self.on_save_callback(self.map_id, self.map_name, self.mobs)

    def _on_delete(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先在上方列表中选中要删除的怪物！", parent=self.top)
            return
        mid_str = str(self.tree.item(sel[0], "values")[0])
        name_str = str(self.tree.item(sel[0], "values")[1])
        if not messagebox.askyesno("确认", f"确定删除【{name_str}】(ID:{mid_str})？", parent=self.top):
            return
        self.mobs = [m for m in self.mobs if str(m.get("id", "")) != mid_str]
        self._refresh_tree()
        # 立即推送到主界面并持久化
        self.on_save_callback(self.map_id, self.map_name, self.mobs)

    def _on_add(self):
        mname = self.ent_name.get().strip()
        mid_str = self.ent_id.get().strip()
        kw_str = self.ent_kw.get().strip()
        if not mname:
            messagebox.showwarning("提示", "请输入怪物名称！", parent=self.top)
            return
        try:
            mid = int(mid_str) if mid_str else int(time.time() * 1000) % 10000000
        except Exception:
            mid = int(time.time() * 1000) % 10000000
        kws = [k.strip() for k in kw_str.split(",") if k.strip()] or [mname.lower()]
        self.mobs.append({"id": mid, "name": mname, "file_prefix": kws[0], "keywords": kws, "ready": True})
        self._refresh_tree()
        # 立即推送到主界面
        self.on_save_callback(self.map_id, self.map_name, self.mobs)

    # 本地离线怪物库（GMS v83 经典全量，中英文均可搜索，无网络依赖）
    LOCAL_MOB_DB = [
        (100100, "Snail (蜗牛)"),        (100101, "Blue Snail (蓝蜗牛)"),
        (100102, "Red Snail (红蜗牛)"),  (110100, "Shroom (蘑菇头)"),
        (120100, "Orange Mushroom (橙蘑菇)"), (130100, "Stump (树桩)"),
        (130101, "Dark Stump (暗树桩)"), (210100, "Slime (史莱姆)"),
        (210101, "Green Slime (绿史莱姆)"), (1110100, "Green Mushroom (绿蘑菇)"),
        (1110101, "Horny Mushroom (角蘑菇)"), (1120100, "Zombie Mushroom (僵尸蘑菇)"),
        (1140100, "Ghost Stump (幽灵树桩)"), (1140130, "Smirking Ghost Stump (坏笑树桩)"),
        (1210100, "Pig (猪)"),           (1210101, "Ribbon Pig (蝴蝶结猪)"),
        (2110200, "Horny Mushroom B (角蘑菇B)"), (2130100, "Curse Eye (诅咒眼)"),
        (2130103, "Zombie (僵尸)"),      (2230100, "Evil Eye (独眼兽)"),
        (2230101, "Evil Eye 2 (独眼兽2)"),(2230102, "Cold Eye (冷眼)"),
        (2230110, "Wooden Mask (木面具)"),(3210100, "Zombie Lupin (僵尸猴)"),
        (3210101, "Lupin / Angel Monkey (天使猴/猴子)"), (3230100, "Ligator (鳄鱼)"),
        (3230101, "Croco (大鳄鱼)"),    (4130100, "Curse Eye B (诅咒独眼B)"),
        (5130103, "Ice Drake (冰龙)"),   (5130104, "Dark Drake (暗龙)"),
        (6230300, "Balrog (巴尔洛克)"),  (8150100, "King Slime (史莱姆王)"),
        (9001000, "Clang (铁甲怪)"),     (9300018, "Jr. Balrog (小巴尔洛克)"),
    ]

    def _on_search_changed(self, *args):
        if self._search_timer:
            self.top.after_cancel(self._search_timer)
        self._search_timer = self.top.after(400, self._do_search)

    def _do_search(self):
        query = self.search_var.get().strip()
        if not query:
            for it in self.res_tree.get_children():
                self.res_tree.delete(it)
            self.search_status.config(text="")
            return

        for it in self.res_tree.get_children():
            self.res_tree.delete(it)

        # 1. 本地离线搜索（瞬时，无需网络）
        q = query.lower()
        local_hits = [(rid, label) for rid, label in self.LOCAL_MOB_DB
                      if q in label.lower() or q in str(rid)]
        for rid, label in local_hits[:20]:
            iid = f"sr_{rid}"
            if not self.res_tree.exists(iid):
                self.res_tree.insert("", tk.END, iid=iid, values=(str(rid), label))

        if local_hits:
            self.search_status.config(text=f"本地 {len(local_hits)} 条，联网补充中...")
        else:
            self.search_status.config(text="本地无结果，联网搜索中...")

        # 2. 异步联网补充（超时 10s，带浏览器 Headers 防屏蔽，不阻塞 UI）
        def fetch_net():
            try:
                url = self.IO_SEARCH_URL + requests.utils.quote(query)
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://maplestory.io/",
                }
                resp = requests.get(url, timeout=10, headers=headers)
                net = resp.json() if resp.status_code == 200 else []
            except Exception:
                net = []

            def merge():
                for r in net[:20]:
                    rid = str(r.get("id", ""))
                    rname = r.get("name", "Unknown")
                    iid = f"sr_{rid}"
                    if not self.res_tree.exists(iid):
                        self.res_tree.insert("", tk.END, iid=iid, values=(rid, rname))
                total = len(self.res_tree.get_children())
                self.search_status.config(
                    text=f"共 {total} 条（本地+网络）" if total else "无结果，请换关键词")
            try:
                self.top.after(0, merge)
            except Exception:
                pass

        threading.Thread(target=fetch_net, daemon=True).start()

    def _on_search_select(self, event=None):
        sel = self.res_tree.selection()
        if not sel:
            return
        values = self.res_tree.item(sel[0], "values")
        if not values:
            return
        rid, rname = str(values[0]), str(values[1])
        self.ent_id.delete(0, tk.END);   self.ent_id.insert(0, rid)
        self.ent_name.delete(0, tk.END); self.ent_name.insert(0, rname)
        try:
            mid_int = int(rid)
            kws = CLASSIC_MOB_ID_MAP.get(mid_int, [rname.lower().replace(" ", "_").split("(")[0].strip()])
        except Exception:
            kws = [rname.lower().replace(" ", "_")]
        self.ent_kw.delete(0, tk.END)
        self.ent_kw.insert(0, ", ".join(kws))




# ─────────────────────────────────────────────────────────────────────────────
#  MapleBotGUI：主 GUI
# ─────────────────────────────────────────────────────────────────────────────
class MapleBotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        # Tkinter 只能由主线程访问。后台控制/识别线程的日志必须先进入
        # 队列；直接从工作线程调用 root.after() 仍可能在 Tcl 层同步等待，
        # 并与跨图状态锁形成锁反转，表现为整个 Python 窗口“未响应”。
        self._log_ui_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._reconnect_ui_queue: "queue.Queue[Tuple[str, str, bool]]" = queue.Queue()
        self._session_incident_titles = {
            "disconnect": "掉线",
            "minigame": "小游戏",
            "unexpected_town": "意外回城",
        }
        self._session_incident_counts = {key: 0 for key in self._session_incident_titles}
        self._tray_events: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._system_tray = SystemTray(self._tray_events)
        self._tray_hide_pending = False
        self.root.title("⚡ 游戏视觉中枢 - 怪物监控与识别")
        
        # 依据当前显示器的 rcWork（排除任务栏）计算，而不是用全屏尺寸
        # 减一个固定常数。小屏时最低尺寸也会一并限幅，避免窗口跑出屏幕。
        _, _, work_w, work_h = get_work_area(self.root)
        self._compact_main_layout = work_w < 1180
        win_w = min(1860, max(980, int(work_w * 0.95)))
        win_h = min(1040, max(560, int(work_h * 0.94)))
        fit_window_to_work_area(
            self.root, (win_w, win_h), (900, 540),
            width_fraction=0.97, height_fraction=0.95,
        )
        self.root.configure(bg="#121214")

        self.config = self._load_config()
        self._minigame_handoff_queue: "queue.Queue[bool]" = queue.Queue()
        self._minigame_replay_record_queue: "queue.Queue[bool]" = queue.Queue()
        self._minigame_snapshot_queue: "queue.Queue[Any]" = queue.Queue(maxsize=8)
        self._minigame_owns_input = False
        self._minigame_resume_f6 = False
        self._recognition_pause_event = threading.Event()
        self._minigame_overlay_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._minigame_overlay_window = None
        self._minigame_overlay_label = None
        self._minigame_overlay_photo = None
        self._minigame_overlay_hwnd = 0
        # 传送门附近按上后的 OCR 加速请求由输入线程产生、地图哨兵线程消费。
        # 使用一个常驻哨兵和 Event 唤醒，避免每次按键另起 OCR 线程造成并发
        # 识别、CPU 峰值以及地图状态互相覆盖。
        self._portal_ocr_lock = threading.Lock()
        self._portal_ocr_pending: Optional[Dict[str, Any]] = None
        self._portal_ocr_wakeup = threading.Event()
        PlatformGraph.ladder_bottom_y_tolerance_px = float(
            self.config.get("ladder_bottom_y_tolerance_px", 35.0)
        )
        self.window_mgr = WindowManager(
            target_window_title=self.config.get("window_title", "冒险岛怀旧服")
        )
        self.hwnd = self.window_mgr.find_game_window()
        self.input_driver = InputDriver(
            target_hwnd=self.hwnd,
            input_mode=self.config.get("input_mode", "background")
        )
        self.input_driver.delivery_audit_enabled = bool(
            self.config.get("input_delivery_audit_enabled", True)
        )
        self.manual_input_recorder = ManualInputRecorder(
            self.hwnd,
            os.path.join(os.path.dirname(CONFIG_PATH), "logs", "manual_input_recording.json"),
        )
        self.horizontal_motion = HorizontalMotionModel(
            speed_percent=self.config.get("movement_speed_percent", 103.0),
            push_accel=1500.0,
            drag_accel=900.0,
        )
        # 输入感知 Kalman 是导航/拓扑的主水平坐标模型；传统模型继续
        # 接收相同按键与 raw 黄点，仅作为并行对照和快速回退依据。
        self.horizontal_kalman = InputAwareHorizontalKalman(
            speed_percent=self.config.get("movement_speed_percent", 103.0),
            push_accel=1500.0,
            drag_accel=900.0,
        )
        self.input_driver.key_event_callback = self._on_motion_key_event
        # 当前 windows-capture 版本在部分旧版 DirectX 客户端上即使调用
        # start_free_threaded() 也会长期占用 GIL，使 Tk 主线程在 UI 完成
        # 初始化前永久“未响应”。默认使用稳定的异步 GDI；WGC 代码保留为
        # 显式实验选项，便于以后升级依赖后直接重新启用。
        self.capture = ScreenCapture(
            self.window_mgr,
            use_async_thread=True,
            prefer_wgc=bool(self.config.get("prefer_wgc_capture", False)),
        )
        self.minigame_bridge = MiniGameBridge(self._queue_minigame_handoff)
        self.minigame_result_confirmer = ResultDialogConfirmer(
            ResultDialogDetector(), self.input_driver, self.log
        )
        self.minigame_session_recorder = MiniGameSessionRecorder(
            os.path.join(os.path.dirname(CONFIG_PATH), "logs", "minigame_sessions")
        )
        self.minigame_bridge.set_mouse_control_enabled(
            bool(self.config.get("lie_detector_mouse_control_enabled", False))
            and bool(self.config.get("lie_detector_auto_solve_enabled", False))
        )
        self.minigame_video_test = MiniGameVideoTest(status_callback=self.log)
        self.capture.set_frame_overlay_provider(self._composite_minigame_test_frame)
        self.detector = MainViewDetector(
            attack_reach_x=self.config.get("attack_reach_x", 260),
            attack_reach_y=self.config.get("attack_reach_y", 140),
            attack_reach_y_up=self.config.get("attack_reach_y_up", self.config.get("attack_reach_y", 140)),
            attack_reach_y_down=self.config.get("attack_reach_y_down", self.config.get("attack_reach_y", 140)),
            behind_reach_x=self.config.get("behind_reach_x", 40),
            skirmish_range_x=self.config.get("skirmish_range_x", 0),
            attack_two_way=bool(self.config.get("attack_two_way", False)),
            monster_threshold=self.config.get("monster_threshold", 0.52),
            scale_factor=0.35,
            tracker_buffer_time_sec=self.config.get("tracker_buffer_ms", 120) / 1000.0,
            monster_template_scale=self.config.get("monster_template_scale", 1.0),
            monster_compute_device=self.config.get("monster_compute_device", "auto"),
            monster_hp_bar_compute_device=self.config.get(
                "monster_hp_bar_compute_device", "auto"
            ),
            monster_coarse_scale=self.config.get("monster_coarse_scale", 0.6),
            monster_redetect_interval=(
                1
                if bool(self.config.get("monster_full_scan_every_frame", False))
                else self.config.get("monster_redetect_interval", 4)
            ),
            player_feature_threshold=self.config.get("player_feature_entry_threshold", 0.58),
            attack_observation_hard_timeout_sec=(
                self.config.get("monster_attack_hard_timeout_ms", 500.0) / 1000.0
            ),
            attack_observation_hard_timeout_enabled=bool(
                self.config.get("monster_attack_hard_timeout_enabled", True)
            ),
            ghost_box_safety_timeout_sec=(
                self.config.get("ghost_box_safety_timeout_ms", 750.0) / 1000.0
            ),
            ghost_box_safety_timeout_enabled=bool(
                self.config.get("ghost_box_safety_timeout_enabled", True)
            ),
        )
        self.detector.attack_config = self.config
        self.detector.enable_two_stage_feature = bool(self.config.get("enable_two_stage_feature", True))
        self.detector.enable_monster_detection = bool(self.config.get("enable_monster_detection", True))
        self.detector.enable_monster_hp_bar_detection = bool(self.config.get("enable_monster_hp_bar_detection", True))
        self.detector.set_exclusion_regions(self.config.get("recognition_exclusion_regions", []))
        self.tracker = MinimapTracker()
        # 后台控制坐标必须逐帧从当前小地图 HSV 连通域取黄点。尤其同图
        # 传送点不会触发 MapID 切换，若保留模板/上一帧锁定，传送后会把
        # 旧区域的黄色装饰误当角色，造成世界坐标固定在传送前位置。
        self.raw_tracker = MinimapTracker(enable_template_tracking=False)
        # F8 仅用于展示“当前帧里黄点实际在哪”。它不应和后台导航
        # 共用模板、历史轨迹或最后一次命中位置，否则会偶发显示旧坐标。
        self.radar_tracker = MinimapTracker(enable_template_tracking=False)
        yellow_sizes = self._parse_yellow_candidate_sizes(
            self.config.get("yellow_dot_candidate_sizes")
        )
        for _tracker in (self.tracker, self.raw_tracker, self.radar_tracker):
            _tracker.set_yellow_candidate_sizes(yellow_sizes)
        self._raw_world_lock = threading.Lock()
        self._minimap_world_per_pixel = (1.0, 1.0)
        self._raw_tracker_result_lock = threading.Lock()
        self._latest_raw_tracker_result = None
        # 换图性能埋点：区分拓扑/WZ 背景、小地图内框、黄点世界坐标
        # 三个阶段。pending 只覆盖本次 MapID 切换，不受同图 OCR 轮询影响。
        self._minimap_ready_lock = threading.Lock()
        self._minimap_ready_pending = None
        self._coordinate_trace_lock = threading.Lock()
        self._runtime_log_lock = threading.Lock()
        self._runtime_log_fp = None
        self._coordinate_trace_path = os.path.join(
            os.path.dirname(CONFIG_PATH), "logs", "coordinate_trace.csv"
        )
        self._yellow_identity_debug = bool(self.config.get("yellow_identity_debug", False))
        self._coordinate_trace_enabled = self._yellow_identity_debug
        self._yellow_identity_debug_last_save = 0.0
        self._yellow_identity_debug_seq = 0
        self._yellow_identity_debug_dir = os.path.join(
            os.path.dirname(CONFIG_PATH), "test_reports",
            f"yellow_identity_internal_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        if self._yellow_identity_debug:
            os.makedirs(self._yellow_identity_debug_dir, exist_ok=True)
        self._coordinate_trace_fp = None
        self._model_debug_lock = threading.Lock()
        self._model_debug_fp = None
        try:
            os.makedirs(os.path.dirname(self._coordinate_trace_path), exist_ok=True)
            # 每次启动独立记录，避免上次运行的坐标样本干扰复盘。
            self._coordinate_trace_fp = open(
                self._coordinate_trace_path, "w", encoding="utf-8", buffering=1
            )
            self._coordinate_trace_fp.write(
                "timestamp,monotonic_s,px,py,norm_x,norm_y,raw_world_x,raw_world_y,"
                "snapped_world_x,snapped_world_y,inner_x,inner_y,inner_w,inner_h,"
                "offset_x,offset_y,match_score,match_reason,candidate_count\n"
            )
            self._model_debug_fp = open(
                os.path.join(CONFIG_PATH.rsplit(os.sep, 1)[0], "logs", "topology_world_debug.csv"),
                "w", encoding="utf-8", buffering=1
            )
            self._model_debug_fp.write(
                "timestamp,raw_world_x,raw_world_y,predicted_x,kalman_x,"
                "topology_x,topology_y,direction,velocity,blocked,kalman_direction,"
                "kalman_velocity,kalman_position_variance,kalman_innovation,"
                "kalman_gain_x,kalman_reanchored,source\n"
            )
            self._runtime_log_fp = open(
                os.path.join(CONFIG_PATH.rsplit(os.sep, 1)[0], "logs", "runtime_actions.log"),
                "w", encoding="utf-8", buffering=1
            )
        except OSError:
            self._coordinate_trace_fp = None
            self._model_debug_fp = None
            self._runtime_log_fp = None
        self._performance_timing = None
        try:
            self._performance_timing = PerformanceTimingLog(
                os.path.join(os.path.dirname(CONFIG_PATH), "logs", "performance_timing.log")
            )
            self.capture.set_timing_callback(self._performance_timing.record_capture)
        except OSError:
            pass
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.downloader = AssetDownloader(region="gms", version="83")
        self.wz_map_reader = WzMapReader(
            os.path.join(project_root, "Map"),
            iv_hex="4D23C72B",
            parser_root=os.path.join(project_root, "wz_python_tool"),
        )
        self.map_resolver = MapResolver(
            downloader=self.downloader,
            ocr_roi=self.config.get("map_ocr_roi"),
            map_data_reader=self.wz_map_reader,
        )
        self.map_resolver.ocr_status_callback = self.log
        if self._performance_timing is not None:
            self.map_resolver.ocr_timing_callback = self._performance_timing.record_ocr
        self.ladder_aligner = LadderAligner()
        self.motion = MotionController(
            self.input_driver,
            jump_key=self.config.get("jump_key", "alt"),
            ladder_aligner=self.ladder_aligner,
            log_callback=self.log,
            motion_model=self.horizontal_kalman,
            raw_position_getter=lambda: self.current_player_raw_world_pos,
            takeoff_gate_tolerance_getter=lambda: float(
                self.config.get("takeoff_gate_tolerance_px", 20.0)
            ),
            config=self.config,
        )
        self.waypoint_mgr = WaypointManager()
        
        # 线程安全巡逻平台列表缓存
        saved_patrol = self.config.get("patrol_platforms", "25, 26")
        self.cached_patrol_platforms: List[int] = self._parse_patrol_string(saved_patrol)
        self.cached_patrol_dwell_range = self._parse_patrol_dwell_range(
            self.config.get("patrol_dwell_min_sec", 1.0),
            self.config.get("patrol_dwell_max_sec", 3.0),
        )
        self.cached_single_patrol_positions = self._parse_single_patrol_positions(
            self.config.get("single_patrol_positions", "30%, 60%")
        )
        self.cached_patrol_position_overrides = self._saved_position_overrides(
            self.config.get("patrol_position_overrides", {})
        )
        self.cached_patrol_position_random = self._parse_patrol_position_random(
            self.config.get("patrol_position_random_percent", 0.0)
        )
        self.cached_world_patrol_stops = []
        saved_world_maps = self.config.get("cross_map_patrol_maps")
        if isinstance(saved_world_maps, list) and saved_world_maps:
            try:
                for item in saved_world_maps:
                    map_id = int(item.get("map_id"))
                    platforms = tuple(self._parse_patrol_string(item.get("platforms", "")))
                    positions = tuple(self._parse_single_patrol_positions(item.get("positions", "")))
                    if not platforms:
                        continue
                    self.cached_world_patrol_stops.append(
                        WorldPatrolStop.from_saved_spec(
                            item, platforms, positions, self.cached_patrol_dwell_range,
                        )
                    )
            except (AttributeError, TypeError, ValueError):
                self.cached_world_patrol_stops = []
        if not self.cached_world_patrol_stops:
            try:
                self.cached_world_patrol_stops = parse_world_patrol_stops(
                    self.config.get("cross_map_patrol_route", "")
                )
            except ValueError:
                self.cached_world_patrol_stops = []
        self.world_route_planner = WorldRoutePlanner(
            os.path.join(project_root, "data", "maps"),
            merge_short_platforms=bool(self.config.get("merge_short_platforms", True)),
            map_loader=self.wz_map_reader.load_map,
            map_exists=self.wz_map_reader.has_map,
            map_ids=self.wz_map_reader.list_map_ids,
        )
        self.world_patrol_controller: Optional[WorldPatrolController] = None
        self.current_player_world_pos: Optional[Tuple[int, int]] = None
        self.world_kalman = WorldCoordinateKalman()

        self.combat_fsm = CombatFSM(
            detector=self.detector,
            input_driver=self.input_driver,
            motion_controller=self.motion,
            waypoint_manager=self.waypoint_mgr,
            config=self.config,
            log_callback=self.log,
            get_platform_graph=lambda: self.platform_graph,
            get_patrol_platforms=self._get_effective_patrol_platforms,
            get_current_platform=lambda: self.current_player_platform,
            get_is_climbing=lambda: self.current_player_is_climbing,
            get_player_world_pos=self._get_live_world_pos,
            get_player_raw_world_pos=self._get_live_raw_world_pos,
            get_current_frame=lambda: self.capture.capture_frame(copy=False) if self.capture else None,
            on_single_step_complete=lambda: self._set_coordinate_trace_enabled(False),
            reset_motion_prediction=self._reset_motion_prediction,
            get_enable_run_jump=lambda: bool(self.config.get("enable_run_jump_grab", True)),
            get_enable_run_jump_fallback=lambda: bool(
                self.config.get("run_jump_fallback_to_static_grab", False)
            ),
            get_run_jump_failure_limit=lambda: int(
                self.config.get("run_jump_failure_limit", 2) or 2
            ),
            get_enable_monster_detection=lambda: bool(
                self._has_current_map_monsters()
                and (
                    self.config.get("enable_monster_detection", True)
                    or self.config.get("enable_monster_hp_bar_detection", True)
                )
            ),
            get_patrol_dwell_range=self._get_effective_patrol_dwell_range,
            get_single_patrol_positions=self._get_effective_patrol_positions,
            get_platform_patrol_positions=self._get_effective_platform_patrol_positions,
            get_patrol_position_random=self._get_patrol_position_random,
            get_patrol_arrival_tolerance=lambda: float(
                self.config.get("patrol_arrival_tolerance_px", 20.0)
            ),
            get_rest_settings=self._get_effective_rest_settings,
            get_global_rest_due=lambda: bool(
                self.world_patrol_controller
                and self.world_patrol_controller.rest_deadline_due()
            ),
            cross_map_tick=self._tick_world_patrol,
            patrol_target_completed_callback=self._on_world_patrol_target_completed,
            rest_task_completed_callback=self._on_global_rest_completed,
        )
        # 独立于自动挂机的可重复跳抓压力测试；仅在用户从测试弹窗显式启动时运行。
        self.ladder_grab_test_runner = LadderGrabTestRunner(
            driver=self.input_driver,
            motion=self.motion,
            motion_model=self.horizontal_kalman,
            get_graph=lambda: self.platform_graph,
            get_raw_position=self._get_live_raw_world_pos,
            get_world_position=self._get_live_world_pos,
            capture_frame=lambda: self.capture.capture_frame(copy=True) if self.capture else None,
            log=self.log,
            status_callback=self._on_ladder_grab_test_status,
            get_pixel_world_scale=lambda: self._minimap_world_per_pixel,
            get_intra_map_portal_enabled=lambda: not bool(
                self.config.get("disable_intra_map_portals", False)
            ),
            f6_edge_executor=self.combat_fsm.execute_patrol_edge_for_test,
            begin_f6_test_session=self.combat_fsm.begin_external_navigation_test,
            end_f6_test_session=self.combat_fsm.end_external_navigation_test,
        )
        self._ladder_test_dialog = None
        self.random_path_test_runner = RandomPathTestRunner(
            driver=self.input_driver,
            motion=self.motion,
            get_graph=lambda: self.platform_graph,
            get_platform=lambda: self.current_player_platform,
            get_world_position=self._get_live_world_pos,
            get_raw_position=self._get_live_raw_world_pos,
            get_is_climbing=lambda: self.current_player_is_climbing,
            capture_frame=lambda: self.capture.capture_frame(copy=True) if self.capture else None,
            get_run_jump_enabled=lambda: bool(
                self.config.get("enable_run_jump_grab", True)
            ),
            get_intra_map_portal_enabled=lambda: not bool(
                self.config.get("disable_intra_map_portals", False)
            ),
            f6_edge_executor=self.combat_fsm.execute_patrol_edge_for_test,
            begin_f6_test_session=self.combat_fsm.begin_external_navigation_test,
            end_f6_test_session=self.combat_fsm.end_external_navigation_test,
            log=self.log,
            status_callback=self._on_random_path_test_status,
        )
        self._random_path_test_btn = None

        self.is_running = True
        self.stop_event = threading.Event()
        self.current_fps = 0.0
        self._monster_fps_lock = threading.Lock()
        self._monster_detection_fps: Optional[float] = None
        self._monster_fps_count = 0
        self._monster_fps_timer = time.perf_counter()
        self._monster_full_scan_fps: Optional[float] = None
        self._monster_full_scan_gap_ms: Optional[float] = None
        self._monster_full_scan_cost_ms: Optional[float] = None
        self._monster_full_scan_count = 0
        self._monster_full_scan_timer = time.perf_counter()
        self._monster_batch_lock = threading.Lock()
        self._monster_latest_batch: Optional[MonsterDetectionBatch] = None
        self.latest_render_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.cached_mob_thumbnails: Dict[str, ImageTk.PhotoImage] = {}
        self.status_bar_reader = StatusBarReader()
        self.death_status_reader = StatusBarReader()
        self.death_dialog_detector = DeathDialogDetector()
        self.potion_stock_reader = PotionStockReader()
        self.potion_stock_monitor = PotionStockMonitor()
        self._potion_stock_eval_lock = threading.Lock()
        self._pending_potion_bindings: Dict[str, Tuple[str, int]] = {}
        self._pending_potion_roi: Optional[Dict[str, Any]] = None
        self._potion_apply_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.status_exp_tracker = ExperienceRateTracker()
        self._status_reading_lock = threading.Lock()
        self._latest_status_reading: Optional[StatusBarReading] = None
        self._status_exact_reading: Optional[StatusBarReading] = None
        self._status_ocr_valid_until = 0.0
        self._status_hpmp_ocr_validated = False
        self._status_last_ui_refresh = 0.0
        self._death_recovery_status_text = ""
        self._death_recovery_status_active = False
        self.recording_target: Optional[str] = None
        self.active_rec_btn: Optional[tk.Button] = None
        self.current_map_info: Optional[Dict] = None
        # 后台 OCR/手动同步线程不能直接操作 Tk；统一把地图同步结果交给
        # GUI 主线程消费，避免偶发的 after 调用失败被工作线程静默吞掉。
        self._map_ui_update_queue: "queue.Queue[Dict]" = queue.Queue()
        self.merge_short_platforms = bool(self.config.get("merge_short_platforms", True))
        self._current_graph_map_data: Optional[Dict] = None
        self._platform_graph_cache: Dict[Tuple[str, bool], PlatformGraph] = {}
        # 当前进程内按 MapID 保留已读取的原始地图 JSON，返回旧地图时无需重复读盘。
        self._map_data_cache: Dict[str, Dict] = {}
        # 手动 MapID 选择后，OCR 哨兵不得把用户选择覆盖回相邻地图。
        # 点击“立即 OCR 刷新”时显式解除此锁。
        self._manual_map_override_id: Optional[int] = None

        # 平台巡航与拓扑引擎
        self.platform_graph: Optional[PlatformGraph] = None
        self.topology_dialog: Optional[PlatformTopologyDialog] = None
        self.minimap_radar_dialog: Optional[MinimapRadarDialog] = None
        self.current_player_platform: Optional[PlatformNode] = None
        self.current_player_is_climbing: bool = False
        self.current_player_raw_world_pos: Optional[Tuple[int, int]] = None
        self._last_logged_platform_id: Optional[int] = None
        self.world_patrol_controller = WorldPatrolController(
            planner=self.world_route_planner,
            motion=self.motion,
            driver=self.input_driver,
            stop_event=self.combat_fsm.stop_event,
            map_id_getter=self._get_current_map_id,
            graph_getter=lambda: self.platform_graph,
            platform_getter=lambda: self.current_player_platform,
            position_getter=self._get_live_world_pos,
            allow_run_jump_getter=lambda: bool(
                self.config.get("enable_run_jump_grab", True)
            ),
            allow_intra_map_portal_getter=lambda: not bool(
                self.config.get("disable_intra_map_portals", False)
            ),
            prepare_transition=self._prepare_world_map_transition,
            log_callback=self.log,
            recovery_completed_callback=self._on_world_patrol_recovery_completed,
            recovery_state_callback=lambda active, detail: self._reconnect_ui_queue.put(
                ("world_recovery", str(detail), bool(active))
            ),
            arrival_reset_callback=self._reset_tracking_after_world_transition,
            arrival_visual_getter=self._get_fresh_arrival_player_observation,
            visual_frame_size_getter=self._get_current_game_frame_size,
            arm_known_portal_ocr=self._arm_known_script_portal_ocr,
            begin_rest_callback=lambda settings: self.combat_fsm.request_forced_rest(settings),
            unexpected_map_callback=lambda: self._reconnect_ui_queue.put(
                ("incident", "unexpected_town", True)
            ),
        )
        self.world_patrol_controller.configure(
            bool(self.config.get("cross_map_patrol_enabled", False)),
            self.cached_world_patrol_stops,
        )

        self._reconnect_status_text = (
            "断线重连待命"
            if self.config.get("reconnect_enabled", False)
            else "断线重连已关闭"
        )
        self._reconnect_status_active = False
        self.reconnect_controller = ReconnectController(
            input_driver=self.input_driver,
            frame_getter=lambda: self.capture.capture_frame(
                copy=True, include_overlay=False,
            ) if self.capture else None,
            visible_frame_getter=lambda: self.capture.capture_visible_client()
            if self.capture else None,
            set_visible_capture=lambda enabled: self.capture.set_foreground_login_capture(enabled)
            if self.capture else None,
            incident_callback=lambda: self._reconnect_ui_queue.put(
                ("incident", "disconnect", True)
            ),
            settings_getter=lambda: self.config,
            is_f6_running=lambda: bool(self.combat_fsm.is_running),
            emergency_stop=self._stop_for_reconnect,
            prepare_game_entry=self._prepare_reconnect_game_entry,
            is_game_ready=self._reconnect_game_ready,
            request_resume=lambda: self._reconnect_ui_queue.put(("resume", "", False)),
            status_callback=lambda text, active: self._reconnect_ui_queue.put(
                ("status", str(text), bool(active))
            ),
            log_callback=self.log,
            reference_dir=os.path.join(project_root, "disconnectVideo"),
        )
        self.death_recovery_controller = DeathRecoveryController(
            detector=self.death_dialog_detector,
            frame_getter=lambda: self.capture.capture_frame(
                copy=True, include_overlay=False,
            ) if self.capture else None,
            map_id_getter=self._get_current_map_id,
            map_ready=self._death_town_map_ready,
            expected_town_getter=self._get_death_return_town,
            hp_zero_getter=self._death_hp_state,
            stock_getter=self._read_death_hp_stock,
            threshold_getter=lambda: int(self.config.get("death_hp_stock_threshold", 0)),
            f6_running=lambda: bool(self.combat_fsm.is_running),
            can_monitor=lambda: not self._minigame_owns_input,
            reconnecting=lambda: bool(self.reconnect_controller.active),
            interrupt=self._interrupt_for_death,
            click_confirm=self._click_death_confirm,
            wake_map_ocr=self._portal_ocr_wakeup.set,
            resume=self._resume_after_death,
            halt=self._halt_after_death,
            status=lambda active, detail: self._reconnect_ui_queue.put(
                ("death_recovery", str(detail), bool(active))
            ),
            log=self.log,
            stop_event=self.stop_event,
        )

        self._init_ui()
        self.root.after(40, self._drain_log_ui_updates)
        self.root.after(40, self._drain_map_ui_updates)
        self.root.after(80, self._drain_reconnect_ui_updates)
        self.root.after(40, self._drain_minigame_handoff_queue)
        self.root.after(33, self._drain_minigame_overlay_queue)
        self.root.after(100, self._drain_tray_events)
        self._start_background_workers()
        self.reconnect_controller.start()
        self._start_global_hotkey_listener()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 开发专项：在同一次管理员启动中，等待地图、黄点与拓扑全部就绪后，
        # 自动执行指定平台与绳梯的 raw 落台 Y 容差标定。
        startup_calibration = os.environ.pop(
            "MXD_START_TOP_EXIT_CALIBRATION", ""
        ).strip()
        if startup_calibration:
            self._startup_top_exit_calibration_deadline = time.perf_counter() + 90.0
            self.root.after(
                1000,
                lambda spec=startup_calibration: self._try_start_top_exit_calibration(spec),
            )

        # 启动时不预加载任何默认地图；等待 OCR 或用户手动应用地图。

    def _try_start_top_exit_calibration(self, spec: str) -> None:
        """解析 ``起点平台:绳号:轮数`` 并在初始化完成后自动启动专项测试。"""
        try:
            start_text, ladder_text, rounds_text = (
                spec.split(":") + ["5", "5"]
            )[:3]
            start_platform_id = int(start_text)
            ladder_id = int(ladder_text)
            rounds = max(1, min(20, int(rounds_text)))
        except (TypeError, ValueError):
            self.log(f"⛔ [绳顶容差标定] 启动参数无效：{spec!r}")
            return
        graph = getattr(self, "platform_graph", None)
        raw = self._get_live_raw_world_pos()
        platform = getattr(self, "current_player_platform", None)
        if (
            graph is None
            or raw is None
            or platform is None
            or int(getattr(platform, "id", -1)) != start_platform_id
        ):
            if time.perf_counter() >= self._startup_top_exit_calibration_deadline:
                self.log(
                    f"⛔ [绳顶容差标定] 90秒内未等到P{start_platform_id}、"
                    f"黄点与拓扑同时就绪，测试未启动"
                )
                return
            self.root.after(
                500, lambda spec=spec: self._try_start_top_exit_calibration(spec)
            )
            return
        cfg = LadderGrabTestConfig(
            start_platform_id=start_platform_id,
            ladder_id=ladder_id,
            start_x=float(raw[0]),
            rounds=rounds,
            closed_loop_walk=True,
            run_jump_grab=bool(self.config.get("enable_run_jump_grab", True)),
            calibrate_raw_landing_tolerance=True,
            landing_sample_seconds=0.8,
        )
        ok, message = self.ladder_grab_test_runner.start(cfg)
        self.log(
            f"{'🧪' if ok else '⛔'} [绳顶容差标定] "
            f"P{start_platform_id}->绳#{ladder_id}，{rounds}轮：{message}"
        )

    # ── 配置 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_yellow_candidate_sizes(value):
        """解析 ``4x4, 5x5, 6x6`` 形式的黄点候选尺寸配置。"""
        if not isinstance(value, (list, tuple, str)):
            return list(MinimapTracker.DEFAULT_YELLOW_CANDIDATE_SIZES)
        items = value.split(",") if isinstance(value, str) else value
        parsed = []
        for item in items:
            if isinstance(item, str):
                text = item.strip().lower().replace("×", "x")
                parts = text.split("x")
            else:
                parts = item
            try:
                if len(parts) != 2:
                    continue
                w, h = int(parts[0]), int(parts[1])
                if 2 <= w <= 12 and 2 <= h <= 12:
                    parsed.append((w, h))
            except (TypeError, ValueError):
                continue
        return parsed or list(MinimapTracker.DEFAULT_YELLOW_CANDIDATE_SIZES)

    def _on_motion_key_event(self, key: str, is_down: bool, timestamp: float) -> None:
        """记录所有实际输入，并用水平输入更新运动模型。"""
        runtime_fp = getattr(self, "_runtime_log_fp", None)
        if runtime_fp is not None:
            try:
                phase = getattr(
                    getattr(getattr(self, "combat_fsm", None), "platform_patrol", None),
                    "phase", None,
                )
                phase_text = getattr(phase, "value", "none")
                source = getattr(self.input_driver, "last_event_source", "unknown")
                thread_name = threading.current_thread().name
                now_wall = time.time()
                wall_time = time.strftime("%H:%M:%S", time.localtime(now_wall))
                millis = int((now_wall % 1.0) * 1000.0)
                delivery = getattr(
                    self.input_driver, "last_delivery_result", None
                )
                audit_text = (
                    json.dumps(delivery, ensure_ascii=False, separators=(",", ":"))
                    if delivery is not None else "null"
                )
                with self._runtime_log_lock:
                    runtime_fp.write(
                        f"[{wall_time}.{millis:03d}] ⌨️ [按键审计] "
                        f"{'DOWN' if is_down else 'UP'} {key} "
                        f"phase={phase_text} thread={thread_name} source={source} "
                        f"delivery={audit_text}\n"
                    )
            except Exception:
                pass
        if str(key).lower() == "up" and is_down:
            source = getattr(self.input_driver, "last_event_source", "F6自动输入")
            # 瞬移、爬梯等也会按 UP；只有真正执行穿门动作才能用已知
            # 门目标快速切图。其他 UP 可能恰好处于传送门附近。
            if "walk_through_portal" in str(source):
                self._request_portal_ocr_acceleration(f"自动输入:{source}")
        if key == "left":
            self.horizontal_motion.set_direction(-1 if is_down else 0, timestamp)
            self.horizontal_kalman.set_direction(-1 if is_down else 0, timestamp)
            self._append_model_debug("", "", self.horizontal_motion.predict(), self.current_player_world_pos, source="input_left")
        elif key == "right":
            self.horizontal_motion.set_direction(1 if is_down else 0, timestamp)
            self.horizontal_kalman.set_direction(1 if is_down else 0, timestamp)
            self._append_model_debug("", "", self.horizontal_motion.predict(), self.current_player_world_pos, source="input_right")

    def _composite_minigame_test_frame(self, frame: np.ndarray, frame_seq: int) -> np.ndarray:
        if not bool(self.config.get("lie_detector_test_enabled", False)):
            return frame
        player = getattr(self, "minigame_video_test", None)
        return player.composite(frame, frame_seq) if player is not None else frame

    def _start_minigame_video_test_session(self) -> None:
        if not bool(self.config.get("lie_detector_test_enabled", False)):
            return
        video_path = str(self.config.get("lie_detector_test_video", "")).strip()
        delay_sec = self.config.get("lie_detector_test_delay_sec", 30.0)
        try:
            delay_sec = max(0.0, float(delay_sec))
        except (TypeError, ValueError):
            delay_sec = 30.0
        self.minigame_video_test.start_f6_session(video_path, delay_sec)

    def _stop_minigame_video_test_session(self, reason: str) -> None:
        player = getattr(self, "minigame_video_test", None)
        if player is not None:
            player.stop_f6_session(reason)

    def _queue_minigame_handoff(self, owns_input: bool) -> None:
        if owns_input:
            self._recognition_pause_event.set()
            self._portal_ocr_wakeup.set()
        else:
            self._recognition_pause_event.clear()
        # Drop stale combat HUD data as recognition pauses for the mini-game.
        with self.lock:
            self.latest_result = MainViewResult(timestamp=time.perf_counter())
            self.latest_ladder_cols = []
        try:
            self._minigame_handoff_queue.put_nowait(bool(owns_input))
        except Exception:
            pass

    def _publish_minigame_overlay(self, frame, hwnd: int, point=None) -> None:
        if not bool(self.config.get("lie_detector_show_overlay_enabled", False)):
            return
        item = (frame.copy(), int(hwnd or 0), point) if frame is not None and frame.size else (None, 0, None)
        try:
            while True:
                self._minigame_overlay_queue.get_nowait()
        except queue.Empty:
            pass
        if item is not None:
            try:
                self._minigame_overlay_queue.put_nowait(item)
            except queue.Full:
                pass

    def _hide_minigame_overlay(self) -> None:
        window = self._minigame_overlay_window
        if window is not None:
            try:
                window.withdraw()
            except tk.TclError:
                self._minigame_overlay_window = None
                self._minigame_overlay_label = None
                self._minigame_overlay_photo = None

    def _drain_minigame_overlay_queue(self) -> None:
        if not self.stop_event.is_set():
            latest = None
            try:
                while True:
                    latest = self._minigame_overlay_queue.get_nowait()
            except queue.Empty:
                pass
            if not bool(self.config.get("lie_detector_show_overlay_enabled", False)):
                self._hide_minigame_overlay()
            elif latest is not None:
                frame, hwnd, point = latest
                if frame is None:
                    self._hide_minigame_overlay()
                else:
                    try:
                        self._show_minigame_overlay(frame, hwnd, point)
                    except Exception as exc:
                        self.log(f"⚠️ [测谎小游戏] 画面叠加窗更新失败：{exc}")
            self.root.after(33, self._drain_minigame_overlay_queue)

    def _show_minigame_overlay(self, frame, hwnd: int, point=None) -> None:
        hwnd_ptr = ctypes.c_void_p(hwnd) if hwnd else None
        if not hwnd_ptr or not user32.IsWindow(hwnd_ptr) or user32.IsIconic(hwnd_ptr):
            self._hide_minigame_overlay()
            return
        rect = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if (
            not user32.GetClientRect(hwnd_ptr, ctypes.byref(rect))
            or not user32.ClientToScreen(hwnd_ptr, ctypes.byref(origin))
        ):
            self._hide_minigame_overlay()
            return
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            self._hide_minigame_overlay()
            return

        shown = frame.copy()
        if point is not None:
            px = int(round(point[0] * shown.shape[1] / width))
            py = int(round(point[1] * shown.shape[0] / height))
            px = max(0, min(shown.shape[1] - 1, px))
            py = max(0, min(shown.shape[0] - 1, py))
            cv2.drawMarker(shown, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.circle(shown, (px, py), 9, (0, 255, 255), 2)
        if shown.shape[1] != width or shown.shape[0] != height:
            shown = cv2.resize(shown, (width, height), interpolation=cv2.INTER_AREA)
        if self._minigame_overlay_window is None or not self._minigame_overlay_window.winfo_exists():
            overlay = tk.Toplevel(self.root)
            overlay.withdraw()
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.configure(bg="black")
            label = tk.Label(overlay, bd=0, bg="black")
            label.pack(fill=tk.BOTH, expand=True)
            self._minigame_overlay_window = overlay
            self._minigame_overlay_label = label
            overlay.update_idletasks()
            overlay_hwnd = int(overlay.winfo_id())
            overlay_ptr = ctypes.c_void_p(overlay_hwnd)
            get_window_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_window_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_window_long.restype = ctypes.c_ssize_t
            set_window_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            set_window_long.restype = ctypes.c_ssize_t
            user32.SetLayeredWindowAttributes.argtypes = [
                wintypes.HWND, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD
            ]
            ex_style = get_window_long(overlay_ptr, -20)
            set_window_long(
                overlay_ptr,
                -20,
                ex_style | 0x00080000 | 0x00000020 | 0x08000000,
            )
            user32.SetLayeredWindowAttributes(overlay_ptr, 0, 255, 0x00000002)
            self._minigame_overlay_hwnd = overlay_hwnd
        overlay = self._minigame_overlay_window
        overlay.geometry(f"{width}x{height}+{origin.x}+{origin.y}")
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        self._minigame_overlay_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._minigame_overlay_label.configure(image=self._minigame_overlay_photo)
        overlay.deiconify()
        user32.SetWindowPos.argtypes = [
            wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        ]
        user32.SetWindowPos(
            ctypes.c_void_p(self._minigame_overlay_hwnd), ctypes.c_void_p(-1),
            origin.x, origin.y, width, height,
            0x0010 | 0x0040 | 0x0200,
        )

    def _save_minigame_state_snapshot(self, frame: np.ndarray, state: str) -> None:
        folder = os.path.join(os.path.dirname(CONFIG_PATH), "logs", "minigame_snapshots")
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        path = os.path.join(folder, f"{stamp}_{state}.png")
        if cv2.imwrite(path, frame):
            self.log(f"📸 [测谎小游戏] 已保存 {state} 状态截图：{path}")
        else:
            self.log(f"⚠️ [测谎小游戏] 保存 {state} 状态截图失败：{path}")

    @staticmethod
    def _parse_minigame_replay_hotkey(hotkey: str) -> list[int]:
        names = {
            "ALT": 0x12, "CTRL": 0x11, "CONTROL": 0x11, "SHIFT": 0x10,
            "WIN": 0x5B, "ESC": 0x1B, "ESCAPE": 0x1B, "SPACE": 0x20,
            "ENTER": 0x0D, "TAB": 0x09, "BACKSPACE": 0x08,
        }
        keys = []
        for token in str(hotkey or "").replace(" ", "").upper().split("+"):
            if not token:
                continue
            if token in names:
                keys.append(names[token])
            elif len(token) == 1 and ("A" <= token <= "Z" or "0" <= token <= "9"):
                keys.append(ord(token))
            elif token.startswith("F") and token[1:].isdigit() and 1 <= int(token[1:]) <= 24:
                keys.append(0x70 + int(token[1:]) - 1)
            else:
                return []
        return keys

    def _trigger_minigame_replay_record(self, hotkey: str) -> None:
        time.sleep(2.0)
        keys = self._parse_minigame_replay_hotkey(hotkey)
        if not keys:
            self.log(f"⚠️ [测谎小游戏] 即时回放按键无效：{hotkey}")
            return
        try:
            user32 = ctypes.windll.user32
            for vk in keys:
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
            time.sleep(0.06)
            for vk in reversed(keys):
                user32.keybd_event(vk, 0, 0x0002, 0)
                time.sleep(0.02)
            self.log(f"📹 [测谎小游戏] 已触发即时回放：{hotkey}")
        except Exception as exc:
            self.log(f"⚠️ [测谎小游戏] 即时回放触发失败：{exc}")

    def _drain_minigame_handoff_queue(self) -> None:
        requested = None
        try:
            while True:
                requested = self._minigame_handoff_queue.get_nowait()
        except queue.Empty:
            pass

        if requested is True and not self._minigame_owns_input:
            self._minigame_owns_input = True
            self._increment_session_incident("minigame")
            self._minigame_resume_f6 = bool(self.combat_fsm.is_running)
            try:
                minigame_phase = self.minigame_bridge.state_machine.state.value
            except Exception:
                minigame_phase = "UNKNOWN"
            if self._minigame_resume_f6:
                self.world_patrol_controller.stop()
                self.combat_fsm.stop()
                self._sync_bot_ui_state()
            if bool(self.config.get("lie_detector_mouse_control_enabled", False)):
                self.log(f"🎯 [测谎小游戏] 检测状态={minigame_phase}，鼠标接管；F6 自动操作暂停。")
            else:
                self.log(f"🎯 [测谎小游戏] 检测状态={minigame_phase}，鼠标接管关闭；F6 自动操作暂停。")
        elif requested is False and self._minigame_owns_input:
            self._minigame_owns_input = False
            should_resume = self._minigame_resume_f6
            self._minigame_resume_f6 = False
            self.log("✅ [测谎小游戏] 已结束，鼠标控制已释放。")
            if should_resume and bool(self.config.get("lie_detector_auto_solve_enabled", False)):
                self.toggle_autobot()
                self.log("▶️ [测谎小游戏] 恢复 F6 自动操作。")
            self._sync_bot_ui_state()

        replay_requested = False
        try:
            while True:
                self._minigame_replay_record_queue.get_nowait()
                replay_requested = True
        except queue.Empty:
            pass
        if (
            replay_requested
            and bool(self.config.get("lie_detector_auto_solve_enabled", False))
            and bool(self.config.get("lie_detector_replay_record_enabled", False))
        ):
            replay_hotkey = str(self.config.get("lie_detector_replay_hotkey", "Alt+F10"))
            threading.Thread(
                target=self._trigger_minigame_replay_record,
                args=(replay_hotkey,),
                daemon=True,
                name="MiniGameReplayRecord",
            ).start()

        self._refresh_minigame_state_display()
        if not self.stop_event.is_set():
            self.root.after(40, self._drain_minigame_handoff_queue)

    def _toggle_lie_detector_auto_solve(self) -> None:
        enabled = bool(self.lie_detector_auto_solve_var.get())
        self.config["lie_detector_auto_solve_enabled"] = enabled
        if enabled:
            self.lie_detector_subsettings_frame.pack(
                fill=tk.X, pady=(1, 0), after=self.lie_detector_auto_solve_checkbutton
            )
        else:
            self.config["lie_detector_mouse_control_enabled"] = False
            self.config["lie_detector_show_overlay_enabled"] = False
            self.config["lie_detector_test_enabled"] = False
            self.config["lie_detector_replay_record_enabled"] = False
            if hasattr(self, "lie_detector_replay_record_var"):
                self.lie_detector_replay_record_var.set(False)
            self.config["lie_detector_save_transition_snapshots"] = False
            if hasattr(self, "lie_detector_save_transition_snapshots_var"):
                self.lie_detector_save_transition_snapshots_var.set(False)
            self.lie_detector_mouse_control_var.set(False)
            self.lie_detector_show_overlay_var.set(False)
            self.lie_detector_test_enabled_var.set(False)
            self.minigame_bridge.set_mouse_control_enabled(False)
            self.minigame_bridge.reset()
            self._stop_minigame_video_test_session("识别测谎已关闭")
            self._hide_minigame_overlay()
            self.lie_detector_subsettings_frame.pack_forget()
            self.lie_detector_release_key_combo.configure(state="disabled")
        self._save_config()
        self.log("🧩 [识别测谎] " + ("已开启" if enabled else "已关闭；鼠标接管和叠加窗口已关闭"))

    def _refresh_minigame_state_display(self) -> None:
        if not hasattr(self, "minigame_state_lbl"):
            return
        if not bool(self.config.get("lie_detector_auto_solve_enabled", False)):
            state = "closed"
        else:
            try:
                state = self.minigame_bridge.state_machine.state.value
            except Exception:
                state = "IDLE"
        colors = {
            "closed": ("#90a4ae", "#303038"),
            "IDLE": ("#80cbc4", "#173633"),
            "COUNTDOWN": ("#ffcc80", "#4b3515"),
            "ACTIVE": ("#ff8a80", "#4b1f1f"),
            "COOLDOWN": ("#b0bec5", "#303038"),
        }
        fg, bg = colors.get(state, ("#ffffff", "#303038"))
        self.minigame_state_lbl.configure(text=state, fg=fg, bg=bg)
        patrol_label = getattr(self, "lbl_top_patrol_target", None)
        if patrol_label is not None:
            patrol_label.configure(text=self._top_patrol_target_text())

    def _increment_session_incident(self, key: str) -> None:
        if key not in self._session_incident_counts:
            return
        self._session_incident_counts[key] += 1
        label = getattr(self, "_session_incident_labels", {}).get(key)
        if label is not None:
            label.configure(
                text=f"{self._session_incident_titles[key]} {self._session_incident_counts[key]}"
            )

    def _top_patrol_target_text(self) -> str:
        """标题栏显示正在执行的巡逻目标或唯一休息平台。"""
        combat = getattr(self, "combat_fsm", None)
        controller = getattr(self, "world_patrol_controller", None)
        if combat is not None:
            rest_phase = getattr(combat, "_rest_phase", "idle")
            if rest_phase in ("travel", "settling", "resting"):
                rest_id = getattr(combat, "_rest_platform_id", None)
                if rest_id is not None:
                    prefix = {
                        "travel": "前往休息", "settling": "休息准备", "resting": "休息",
                    }[rest_phase]
                    return f"{prefix} P{rest_id}"
            if rest_phase == "returning":
                target = getattr(combat, "_rest_return_platform_id", None)
                if target is not None:
                    return f"休息返程 P{target}"
        if controller is not None and getattr(controller, "running", False):
            if getattr(controller, "_rest_stage", "idle") in (
                "pending", "to_rest_map", "resting", "rest_complete",
            ):
                index = getattr(controller, "_rest_stop_index", None)
                stops = getattr(controller, "stops", ())
                if index is not None and 0 <= index < len(stops):
                    stage = controller._rest_stage
                    return f"{'休息' if stage == 'resting' else '前往休息'} P{stops[index].rest_platform_id}"
            if getattr(controller, "_rest_stage", "idle") in (
                "returning_map", "returning_platform",
            ):
                resume = getattr(controller, "_rest_resume", None) or {}
                if resume.get("platform_id") is not None:
                    return f"休息返程 P{resume['platform_id']}"
        if combat is not None and getattr(combat, "is_running", False):
            target = getattr(getattr(combat, "platform_patrol", None), "target_id", None)
            if target is None:
                targets = combat._active_patrol_targets()
                target = targets[0] if targets else None
            if target is not None:
                phase = getattr(getattr(combat, "platform_patrol", None), "phase", None)
                phase_name = getattr(phase, "value", "")
                prefix = {
                    "positioning": "调整站位", "dwelling": "停留",
                }.get(phase_name, "前往")
                return f"{prefix} P{target}"
        return "巡逻未启动"

    def _mouse_release_hotkey_vk(self) -> int:
        key_name = str(self.config.get("lie_detector_mouse_release_hotkey", "F12")).upper()
        if key_name in ("F1", "F2", "F3", "F4", "F5", "F12"):
            return 0x70 + int(key_name[1:]) - 1
        return 0x7B

    def _handle_mouse_release_hotkey(self) -> None:
        self.config["lie_detector_mouse_control_enabled"] = False
        if hasattr(self, "lie_detector_mouse_control_var"):
            self.lie_detector_mouse_control_var.set(False)
        if hasattr(self, "lie_detector_release_key_combo"):
            self.lie_detector_release_key_combo.configure(state="disabled")
        self._save_config()
        self.log("🛑 [鼠标急停] 程序已立即失去测谎小游戏鼠标控制权限；可在 UI 中重新开启。")

    def _load_config(self) -> Dict:
        default = {
            "lie_detector_auto_solve_enabled": False,
            "lie_detector_replay_record_enabled": False,
            "lie_detector_replay_hotkey": "Alt+F10",
            "lie_detector_save_transition_snapshots": False,
            "lie_detector_mouse_control_enabled": False,
            "lie_detector_mouse_release_hotkey": "F12",
            "lie_detector_show_overlay_enabled": False,
            "lie_detector_test_enabled": False,
            "lie_detector_test_video": "",
            "lie_detector_test_delay_sec": 30.0,
            "attack_reach_x": 260, "attack_reach_y": 140,
            "attack_reach_y_up": 140, "attack_reach_y_down": 140,
            "behind_reach_x": 40, "skirmish_range_x": 0, "attack_two_way": False,
            "attack_area": False,
            "monster_threshold": 0.52,
            "monster_template_scale": 1.0,
            "monster_compute_device": "auto",
            "monster_hp_bar_compute_device": "auto",
            "monster_coarse_scale": 0.6,
            "monster_redetect_interval": 4,
            "monster_full_scan_interval_ms": 55,
            "monster_full_scan_every_frame": False,
            "monster_attack_hard_timeout_enabled": True,
            "monster_attack_hard_timeout_ms": 500.0,
            "ghost_box_safety_timeout_enabled": True,
            "ghost_box_safety_timeout_ms": 750.0,
            "recognition_exclusion_regions": [],
            "yellow_dot_candidate_sizes": ["4x4", "4x5", "5x4", "5x5", "6x5", "5x6", "6x6"],
            "tracker_buffer_ms": 120,
            "player_feature_entry_threshold": 0.58,
            "input_mode": "background",
            "input_delivery_audit_enabled": True,
            "attack_key": "ctrl", "attack_vk": 0x11,
            "extra_attack_skills": [],
            "monster_skill_rules": {},
            "attack_only_mode": False,
            "jump_key": "alt",    "jump_vk": 0x12,
            "pick_key": "z",      "pick_vk": 0x5A,
            "chair_key": "end",   "chair_vk": 35,
            "dialog_key": "y",    "dialog_vk": 0x59,
            "pet_feed_key": "", "pet_feed_vk": 0,
            "enable_auto_pet_feed": False,
            "pet_feed_interval_sec": 300.0,
            "hp_potion_key": "home", "hp_potion_vk": 0x24,
            "mp_potion_key": "pageup", "mp_potion_vk": 0x21,
            "enable_auto_potion": False,
            "hp_potion_threshold_percent": 50.0,
            "mp_potion_threshold_percent": 30.0,
            "potion_cooldown_ms": 800.0,
            "death_hp_stock_threshold": 0,
            "status_bar_roi": None,
            "potion_bar_roi": None,
            "teleport_key": "shift", "teleport_vk": 16,
            "enable_teleport": False,
            "teleport_cd_min_ms": 300.0,
            "teleport_cd_max_ms": 500.0,
            "teleport_distance_px": 150.0,
            "rest_platform_id": None,
            "rest_duration_min_sec": 30.0,
            "rest_duration_max_sec": 60.0,
            "rest_interval_min_sec": 30.0,
            "rest_interval_max_sec": 40.0,
            "patrol_platforms": "4, 5, 6"
            ,"cross_map_patrol_enabled": False
            ,"cross_map_patrol_route": ""
            ,"patrol_dwell_min_sec": 1.0
            ,"patrol_dwell_max_sec": 3.0
            ,"single_patrol_positions": "30%, 60%"
            ,"patrol_position_overrides": {}
            ,"patrol_position_random_percent": 0.0
            ,"patrol_arrival_tolerance_px": 20.0
            ,"rear_attack_turn_delay_sec": 2.0
            ,"patrol_combat_dwell_extension_sec": 2.0
            ,"patrol_failure_replan_enabled": False
            ,"skirmish_platform_guard_px": 50.0
            ,"attack_edge_guard_px": 50.0
            ,"climb_top_hold_sec": 1.0
            ,"jump_source_raw_y_tolerance_px": 65.0
            ,"top_exit_raw_y_tolerance_px": 36.0
            ,"use_horizontal_motion_prediction": True
            ,"takeoff_gate_tolerance_px": 20.0
            ,"portal_ocr_trigger_range_px": 200.0
            ,"merge_short_platforms": True
            ,"enable_run_jump_grab": True
            ,"disable_intra_map_portals": False
            ,"run_jump_fallback_to_static_grab": False
            ,"run_jump_failure_limit": 2
            ,"down_jump_pre_neutral_ms": 50.0
            ,"down_jump_down_prep_ms": 109.0
            ,"down_jump_jump_hold_ms": 124.0
            ,"down_jump_retry_enabled": False
            ,"down_jump_retry_jump_hold_ms": 100.0
            ,"down_jump_adaptive_full_retry_enabled": True
            ,"down_jump_post_down_hold_ms": 1.0
            ,"down_jump_post_neutral_ms": 100.0
            ,"down_jump_wait_land_ms": 450.0
            ,"enable_monster_detection": True
            ,"enable_monster_hp_bar_detection": True
            ,"movement_speed_percent": 103.0
            ,"window_title": "冒险岛怀旧服"
            ,"prefer_wgc_capture": False
            ,"inplace_dwell_safe_margin_px": 50.0
            ,"align_pulse_min_ms": 12.0
            ,"align_pulse_max_ms": 35.0
            ,"reconnect_enabled": False
            ,"reconnect_client_profile": "auto"
            ,"reconnect_server_index": 1
            ,"reconnect_server_total": 5
            ,"reconnect_channel_index": 1
            ,"reconnect_channel_total": 20
            ,"reconnect_character_index": 1
            ,"reconnect_character_total": 3
            ,"reconnect_account_online_wait_sec": 300.0
            ,"reconnect_password_protected": ""
        }
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    default.update(loaded)
                    # 从单一旧 Reach Y 无损迁移为上/下两个范围。
                    legacy_y = int(default.get("attack_reach_y", 140))
                    if "attack_reach_y_up" not in loaded:
                        default["attack_reach_y_up"] = legacy_y
                    if "attack_reach_y_down" not in loaded:
                        default["attack_reach_y_down"] = legacy_y
            except Exception:
                pass
        return default

    def _save_config(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[配置保存失败] {e}")

    # ── UI ───────────────────────────────────────────────────────────────────
    def _open_window_selector(self):
        """让用户从当前可见窗口中手动绑定游戏客户端。"""
        windows = self.window_mgr.list_visible_windows()
        if not windows:
            messagebox.showwarning("选择游戏窗口", "没有找到尺寸足够大的可见窗口。", parent=self.root)
            return

        win = tk.Toplevel(self.root)
        win.title("选择游戏窗口")
        fit_window_to_work_area(win, (760, 420), (520, 300), parent=self.root)
        win.transient(self.root)
        win.grab_set()
        tk.Label(
            win, text="请选择要捕获和发送按键的游戏客户端窗口：",
            font=("Segoe UI", 10, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 6))

        list_frame = tk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        lb = tk.Listbox(list_frame, font=("Consolas", 10), yscrollcommand=sb.set,
                        selectmode=tk.SINGLE, exportselection=False)
        sb.config(command=lb.yview)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def display_name(value, fallback):
            return re.sub("maplestory", "游戏客户端", str(value or fallback), flags=re.I)

        for item in windows:
            lb.insert(
                tk.END,
                f"{display_name(item['title'], '未命名窗口')}  |  "
                f"{display_name(item['process'], '未知进程')}  |  "
                f"{item['width']}x{item['height']}",
            )

        current_hwnd = self.window_mgr.hwnd
        for idx, item in enumerate(windows):
            if item["hwnd"] == current_hwnd:
                lb.selection_set(idx)
                lb.see(idx)
                break

        btns = tk.Frame(win)
        btns.pack(fill=tk.X, padx=12, pady=(4, 12))

        def choose():
            selected = lb.curselection()
            if not selected:
                messagebox.showwarning("选择游戏窗口", "请先选择一个窗口。", parent=win)
                return
            item = windows[selected[0]]
            if not self.window_mgr.select_window(item["hwnd"], item["title"]):
                messagebox.showerror("选择失败", "该窗口已关闭或句柄无效。", parent=win)
                return
            self.input_driver.target_hwnd = item["hwnd"]
            self.config["window_title"] = item["title"]
            self._save_config()
            # 先释放 grab 并关掉窗口。任何 WGC/PrintWindow 的重绑都不允许
            # 在 Tk 按钮回调中运行，否则个别 DX 客户端会让整个 GUI 假死。
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

            self.log(
                f"[窗口绑定] 已选择：{display_name(item['title'], '未命名窗口')} "
                f"({display_name(item['process'], '未知进程')})，正在后台切换捕获…"
            )

            def rebind_in_background():
                ok = bool(self.capture and self.capture.rebind_window(item["hwnd"]))
                def report():
                    if ok:
                        self.hwnd = int(item["hwnd"])
                        self.minigame_bridge.reset()
                        # 新客户端可能采用不同的客户区比例、小地图边框或 DPI。
                        # 不能沿用旧客户端缓存的内框/黄点模板，否则 F8 会在旧
                        # 区域内持续搜索并误报“未检测到角色”。窗口重绑本身
                        # 是明确的捕获源切换，因此在此一次性重置，而不是在
                        # 普通黄点漏检时反复重置画布缓存。
                        for tracker in (
                            getattr(self, "tracker", None),
                            getattr(self, "raw_tracker", None),
                            getattr(self, "radar_tracker", None),
                        ):
                            if tracker is not None and hasattr(tracker, "reset_map_calibration"):
                                tracker.reset_map_calibration()
                        self.current_player_world_pos = None
                        self.current_player_raw_world_pos = None
                        self.current_player_platform = None
                        self.current_player_ladder = None
                        with self._raw_tracker_result_lock:
                            self._latest_raw_tracker_result = None
                        self.horizontal_motion.reset()
                        self.horizontal_kalman.reset()
                        self._sync_tracker_minimap_canvas_size(self.platform_graph)
                        self.log("[窗口绑定] 捕获已切换，已重置小地图黄点缓存并同步当前地图画布。")
                    else:
                        self.log("[窗口绑定] ⚠️ 捕获切换失败，窗口可能已关闭。")
                try:
                    self.root.after(0, report)
                except Exception:
                    pass

            threading.Thread(target=rebind_in_background, daemon=True).start()

        tk.Button(btns, text="确定", command=choose, width=12,
                  bg="#0277bd", fg="white", relief=tk.FLAT).pack(side=tk.RIGHT, padx=4)
        tk.Button(btns, text="取消", command=win.destroy, width=12,
                  relief=tk.FLAT).pack(side=tk.RIGHT, padx=4)
        lb.bind("<Double-Button-1>", lambda _e: choose())

    def _minimize_to_tray(self):
        if self._tray_hide_pending:
            return
        self._tray_hide_pending = True
        # Keep the window visible until Windows confirms that the icon exists.
        self._system_tray.show()

    def _drain_tray_events(self):
        try:
            while True:
                action, detail = self._tray_events.get_nowait()
                if action == "ready" and self._tray_hide_pending:
                    self._tray_hide_pending = False
                    self.root.withdraw()
                    self.log("🗔 [系统托盘] 主窗口已隐藏；双击托盘图标可恢复")
                elif action == "error":
                    self._tray_hide_pending = False
                    self.log(f"⚠️ [系统托盘] 创建托盘图标失败，窗口保持显示：{detail}")
                elif action == "restore":
                    self._tray_hide_pending = False
                    self.root.deiconify()
                    self.root.lift()
                elif action == "exit":
                    self.on_close()
                    return
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.root.after(100, self._drain_tray_events)

    def _init_ui(self):
        # 顶部标题栏
        top_bar = tk.Frame(self.root, bg="#1a1a1e", height=46)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        # 右侧操作先占位，窄屏时长标题也不会把托盘按钮挤出视口。
        tk.Button(
            top_bar, text="最小化到托盘", command=self._minimize_to_tray,
            font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#39424c",
            activeforeground="#ffffff", activebackground="#52616e",
            relief=tk.FLAT, padx=8, pady=3,
        ).pack(side=tk.RIGHT, padx=(4, 8))
        tk.Button(
            top_bar, text="选择游戏窗口", command=self._open_window_selector,
            font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#0277bd",
            activeforeground="#ffffff", activebackground="#01579b",
            relief=tk.FLAT, padx=8, pady=3,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        self.fps_lbl = tk.Label(top_bar, text="FPS: 0.0", font=("Consolas", 10), fg="#00e676", bg="#1a1a1e")
        self.fps_lbl.pack(side=tk.RIGHT, padx=8)
        tk.Label(top_bar, text="👁️ 游戏视觉中枢",
                 font=("Segoe UI", 12, "bold"), fg="#00e5ff", bg="#1a1a1e").pack(side=tk.LEFT, padx=16, pady=8)
        self.minigame_state_lbl = tk.Label(
            top_bar,
            text="closed",
            font=("Consolas", 11, "bold"),
            fg="#90a4ae",
            bg="#303038",
            padx=12,
            pady=4,
        )
        self.minigame_state_lbl.pack(side=tk.LEFT, padx=(14, 4))
        self.lbl_top_patrol_target = tk.Label(
            top_bar, text="巡逻未启动", font=("Segoe UI", 10, "bold"),
            fg="#80deea", bg="#1a1a1e", padx=5, pady=4,
        )
        self.lbl_top_patrol_target.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(top_bar, text="● Auto Sentinel",
                 font=("Segoe UI", 9, "bold"), fg="#00e676", bg="#1e382b", padx=10, pady=3).pack(side=tk.LEFT, padx=10)

        # Always-visible counters reset only when the program starts.
        incident_bar = tk.Frame(self.root, bg="#25252d")
        incident_bar.pack(fill=tk.X, side=tk.TOP, padx=8, pady=(0, 2))
        tk.Label(
            incident_bar, text="本次启动事件", font=("Segoe UI", 9, "bold"),
            fg="#f5f5f5", bg="#25252d", padx=10, pady=3,
        ).pack(side=tk.LEFT)
        self._session_incident_labels = {}
        for key, color in (
            ("disconnect", "#ffb74d"),
            ("minigame", "#ce93d8"),
            ("unexpected_town", "#80deea"),
        ):
            label = tk.Label(
                incident_bar,
                text=f"{self._session_incident_titles[key]} 0",
                font=("Segoe UI", 9, "bold"),
                fg=color, bg="#25252d", padx=9, pady=3,
            )
            label.pack(side=tk.LEFT, padx=(3, 8))
            self._session_incident_labels[key] = label

        # 主内容
        self.paned = tk.PanedWindow(
            self.root,
            orient=(tk.VERTICAL if self._compact_main_layout else tk.HORIZONTAL),
            bg="#121214", bd=0, sashwidth=5,
        )
        self.paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左：视口 (紧凑贴合画面，居中对齐，底部常驻终端日志)
        left = tk.Frame(self.paned, bg="#18181c")
        # 左侧视口弹性容纳，最小宽度放宽至 560px，自适应中低分辨率与缩放
        left.configure(width=820)
        left.pack_propagate(False)
        self.paned.add(
            left,
            minsize=(250 if self._compact_main_layout else 560),
            stretch=("always" if self._compact_main_layout else "never"),
        )
        
        vp_hdr = tk.Frame(left, bg="#18181c")
        vp_hdr.pack(fill=tk.X, side=tk.TOP, padx=10, pady=(6, 2))
        self.video_visible_var = tk.BooleanVar(value=True)
        self._viewport_render_enabled = True
        # 复选框必须先 pack 到右侧以预留固定空间。旧版先放入长标题，
        # 在最小窗口宽度或高DPI下标题会吃满整行，导致该开关被裁出视口。
        self.viewport_toggle = tk.Checkbutton(
            vp_hdr, text="显示视口", variable=self.video_visible_var,
            command=self._toggle_video_viewport,
            font=("Segoe UI", 9, "bold"), fg="#80deea", bg="#18181c",
            activeforeground="#ffffff", activebackground="#18181c",
            selectcolor="#2a2a32", padx=4,
        )
        self.viewport_toggle.pack(side=tk.RIGHT, padx=(8, 0))
        tk.Label(
            vp_hdr,
            text="📺 实时视口  ·  绿怪物 / 橙攻击 / 蓝黄角色",
            font=("Segoe UI", 9, "bold"),
            fg="#cccccc", bg="#18181c", anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 左侧底部常驻运行日志 (无需放大窗口或滚动，永远 100% 直观可见)
        self._build_log_card(left)

        # 视口画面容器 (背景色与主界面融合，等比居中自适应)
        self.video_container = tk.Frame(left, bg="#121214")
        self.video_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        # 视口渲染尺寸必须跟随容器。旧版后台固定输出 800x450，在默认
        # 窄窗口（左栏约 520~560px）中 Label 会居中溢出，造成左右画面
        # 被父容器硬裁掉。这里只更新预览目标尺寸，不影响捕获/识别坐标。
        self.video_container.bind("<Configure>", self._on_video_container_configure)
        
        self.video_canvas = tk.Label(self.video_container, bg="#121214", bd=0, highlightthickness=0)
        self.video_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # 右：面板（完全自适应宽度的流畅滚动区，杜绝文字截断）
        # 右侧设置区承担主窗口的横向伸缩空间，放宽最小尺寸避免右侧被挤出屏幕。
        right_c = tk.Frame(self.paned, bg="#18181c", width=900)
        self.paned.add(
            right_c,
            minsize=(250 if self._compact_main_layout else 580),
            stretch="always",
        )
        canvas_r = tk.Canvas(right_c, bg="#18181c", highlightthickness=0)
        sb_r = ttk.Scrollbar(right_c, orient="vertical", command=canvas_r.yview)
        self.sf = tk.Frame(canvas_r, bg="#18181c")
        
        self.sf_window_id = canvas_r.create_window((0, 0), window=self.sf, anchor="nw")
        
        def _on_sf_configure(e):
            canvas_r.configure(scrollregion=canvas_r.bbox("all"))

        def _on_canvas_configure(e):
            # 内部面板宽度动态跟随右侧 Canvas，内容 100% 展开不截断
            canvas_r.itemconfig(self.sf_window_id, width=e.width)

        self.sf.bind("<Configure>", _on_sf_configure)
        canvas_r.bind("<Configure>", _on_canvas_configure)

        # 全局鼠标滚轮顺滑滑动：只要鼠标置于右侧面板区域内，即可直接滚动查看所有设置
        def _on_panel_mousewheel(event):
            if canvas_r.winfo_exists():
                canvas_r.yview_scroll(int(-1 * (event.delta / 120) * 3), "units")

        def _on_panel_enter(_event):
            canvas_r.bind_all("<MouseWheel>", _on_panel_mousewheel)

        def _on_panel_leave(_event):
            canvas_r.unbind_all("<MouseWheel>")

        right_c.bind("<Enter>", _on_panel_enter)
        right_c.bind("<Leave>", _on_panel_leave)

        canvas_r.configure(yscrollcommand=sb_r.set)
        canvas_r.pack(side="left", fill="both", expand=True)
        sb_r.pack(side="right", fill="y")

        # 右侧控制区双列排列，避免常用设置全部堆到纵向滚动区底部。
        self.panel_columns = tk.Frame(self.sf, bg="#18181c")
        self.panel_columns.pack(fill=tk.BOTH, expand=True, padx=2)
        self.panel_columns.columnconfigure(0, weight=1, uniform="control_col")
        self.panel_columns.columnconfigure(1, weight=1, uniform="control_col")
        left_col = tk.Frame(self.panel_columns, bg="#18181c")
        right_col = tk.Frame(self.panel_columns, bg="#18181c")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        right_col.grid(row=0, column=1, sticky="nsew", padx=(3, 0))

        self._panel_parent = left_col
        self._build_autobot_card()
        self._build_platform_patrol_card()
        self._build_player_calibration_card()
        self._panel_parent = right_col
        self._build_mob_card()
        self._build_map_card()
        self._build_status_card()
        self._build_key_card()
        self._build_range_card()
        self._panel_parent = self.sf

        # 启动后自动合理分配视口与控制面板的比例
        self.root.after(80, self._adjust_initial_layout)

    def _adjust_initial_layout(self):
        try:
            total_w = self.root.winfo_width()
            if total_w > 1650:
                sash_x = 820
            elif total_w > 1200:
                sash_x = int(total_w * 0.48)
            else:
                sash_x = max(520, int(total_w * 0.44))
            self.paned.sash_place(0, sash_x, 0)
        except Exception:
            pass

    # ── 平台循环巡航与拓扑管理卡片 ───────────────────────────────────────────
    def _build_platform_patrol_card(self):
        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf), text="🛤️ 路径设置",
                              font=("Segoe UI", 10, "bold"), fg="#00e5ff", bg="#202026", padx=8, pady=6)
        card.pack(fill=tk.X, padx=8, pady=3)
        self.path_settings_card = card

        # 1. 角色当前所处平台实时状态 (特大号高亮醒目胶囊)
        r_status = tk.Frame(card, bg="#16181d", bd=1, relief=tk.SOLID)
        r_status.pack(fill=tk.X, pady=(0, 4), padx=1)

        self.lbl_curr_platform_status = tk.Label(
            r_status, text="📍 角色当前所在: 正在定位...",
            font=("Segoe UI", 10, "bold"), fg="#00e676", bg="#16181d", anchor=tk.W, padx=8, pady=5
        )
        self.lbl_curr_platform_status.pack(fill=tk.X)

        self.path_map_cards = []
        self.var_cross_map_patrol = tk.BooleanVar(
            value=bool(self.config.get("cross_map_patrol_enabled", False))
        )
        self.var_disable_intra_map_portals = tk.BooleanVar(
            value=bool(self.config.get("disable_intra_map_portals", False))
        )

        self.path_maps_container = tk.Frame(card, bg="#202026")
        self.path_maps_container.pack(fill=tk.X)
        initial_specs = self._initial_path_map_specs()
        for index, spec in enumerate(initial_specs):
            self._create_path_map_card(spec, is_current=(index == 0), expanded=(index == 0))

        r_add_map = tk.Frame(card, bg="#202026")
        r_add_map.pack(fill=tk.X, pady=(3, 1))
        tk.Button(
            r_add_map, text="＋ 添加多地图巡逻", font=("Segoe UI", 9, "bold"),
            fg="#ffffff", bg="#202026", activebackground="#2a2a32",
            activeforeground="#80deea", relief=tk.FLAT, anchor=tk.W,
            command=self._add_path_map_card,
        ).pack(side=tk.LEFT)

        self.lbl_cross_map_patrol_status = tk.Label(
            card,
            text="添加两张及以上地图后自动启用跨地图巡逻（普通门与已确认脚本门）",
            font=("Segoe UI", 8), fg="#90a4ae", bg="#202026", anchor=tk.W,
        )
        self.lbl_cross_map_patrol_status.pack(fill=tk.X, pady=(0, 2))

        r_test = tk.Frame(card, bg="#202026")
        r_test.pack(fill=tk.X, pady=(1, 2))
        tk.Checkbutton(
            r_test,
            text="禁用地图内传送点（不影响跨地图出口）",
            variable=self.var_disable_intra_map_portals,
            font=("Segoe UI", 8), fg="#ffcc80", bg="#202026",
            activeforeground="#ffe0b2", activebackground="#202026",
            selectcolor="#303038", anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._random_path_test_btn = tk.Button(
            r_test,
            text="🎲 随机全图行走测试 [F11]",
            font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#00695c",
            activeforeground="#ffffff", activebackground="#00897b",
            relief=tk.FLAT, padx=7, pady=2,
            command=self._toggle_random_path_test,
        )
        self._random_path_test_btn.pack(side=tk.RIGHT)

        r_actions = tk.Frame(card, bg="#202026")
        r_actions.pack(fill=tk.X, pady=(3, 3))
        self._path_apply_button = tk.Button(
            r_actions, text="应用", font=("Segoe UI", 9, "bold"),
            fg="#ffffff", bg="#2e7d32", relief=tk.FLAT, padx=8, pady=3,
            command=self._apply_path_settings,
        )
        self._path_apply_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(
            r_actions,
            text="⏭️ 单步执行",
            font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#6a1b9a",
            relief=tk.FLAT, padx=8, pady=4,
            command=self._run_single_navigation_step,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        btn_topo = tk.Button(
            r_actions, text="🗺️ 世界地图",
            font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#1565c0",
            relief=tk.FLAT, padx=8, pady=3, width=20,
            command=self.open_topology_dialog
        )
        btn_topo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        r_files = tk.Frame(card, bg="#202026")
        r_files.pack(fill=tk.X, pady=(1, 3))
        tk.Button(
            r_files, text="📥 导入路径设置", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#455a64", relief=tk.FLAT,
            command=self._import_path_settings,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(
            r_files, text="📤 导出已应用设置", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#455a64", relief=tk.FLAT,
            command=self._export_path_settings,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

    def _initial_path_map_specs(self) -> List[Dict[str, Any]]:
        """把新版逐地图配置或旧路线字符串转换成 UI 卡片草稿。"""
        default = {
            "map_id": "",
            "platforms": str(self.config.get("patrol_platforms", "4, 5, 6")),
            "dwell_min_sec": self.config.get("patrol_dwell_min_sec", 1.0),
            "dwell_max_sec": self.config.get("patrol_dwell_max_sec", 3.0),
            "positions": str(self.config.get("single_patrol_positions", "30%, 60%")),
            "position_overrides": self.config.get("patrol_position_overrides", {}),
            "position_random_percent": self.config.get("patrol_position_random_percent", 0.0),
            "rest_platform_id": self.config.get("rest_platform_id", None),
            "rest_duration_min_sec": self.config.get("rest_duration_min_sec", 30.0),
            "rest_duration_max_sec": self.config.get("rest_duration_max_sec", 60.0),
            "rest_interval_min_sec": self.config.get("rest_interval_min_sec", 30.0),
            "rest_interval_max_sec": self.config.get("rest_interval_max_sec", 40.0),
        }
        saved = self.config.get("cross_map_patrol_maps")
        if isinstance(saved, list) and saved:
            specs = []
            for item in saved:
                if not isinstance(item, dict):
                    continue
                merged = dict(default)
                merged["position_overrides"] = {}
                merged.update(item)
                specs.append(merged)
            if specs:
                return specs

        if bool(self.config.get("cross_map_patrol_enabled", False)):
            try:
                old_stops = parse_world_patrol_stops(
                    self.config.get("cross_map_patrol_route", "")
                )
            except ValueError:
                old_stops = []
            if old_stops:
                return [
                    {
                        **default,
                        "map_id": str(stop.map_id),
                        "platforms": ", ".join(str(pid) for pid in stop.platforms),
                    }
                    for stop in old_stops
                ]
        return [default]

    def _create_path_map_card(
        self, spec: Dict[str, Any], *, is_current: bool, expanded: bool
    ) -> Dict[str, Any]:
        shell = tk.Frame(self.path_maps_container, bg="#202026")
        shell.pack(fill=tk.X, pady=(1, 2))
        header = tk.Frame(shell, bg="#202026")
        header.pack(fill=tk.X)
        toggle = tk.Button(
            header, text="▼" if expanded else "▶", width=2,
            font=("Segoe UI Symbol", 9, "bold"), fg="#e0e0e0", bg="#202026",
            activeforeground="#80deea", activebackground="#202026",
            relief=tk.FLAT, padx=0, pady=0,
        )
        toggle.pack(side=tk.LEFT)
        title = tk.Label(
            header, text="", font=("Segoe UI", 9, "bold"),
            fg="#ffffff", bg="#202026", anchor=tk.W, cursor="hand2",
        )
        title.pack(side=tk.LEFT, fill=tk.X, expand=True)

        body = tk.Frame(shell, bg="#202026", padx=4)
        if expanded:
            body.pack(fill=tk.X, padx=(16, 0), pady=(1, 2))

        map_id_var = tk.StringVar(value=str(spec.get("map_id", "") or ""))
        card_data = {
            "shell": shell,
            "header": header,
            "toggle": toggle,
            "title": title,
            "body": body,
            "map_id_var": map_id_var,
            "is_current": bool(is_current),
            "expanded": bool(expanded),
            "override_rows": [],
        }
        self.path_map_cards.append(card_data)
        toggle.config(command=lambda c=card_data: self._toggle_path_map_card(c))
        title.bind("<Button-1>", lambda _event, c=card_data: self._toggle_path_map_card(c))

        if not is_current:
            tk.Button(
                header, text="删除", font=("Segoe UI", 8, "bold"),
                fg="#ffffff", bg="#b71c1c", activebackground="#d32f2f",
                relief=tk.FLAT, padx=7, pady=1,
                command=lambda c=card_data: self._delete_path_map_card(c),
            ).pack(side=tk.RIGHT, padx=(4, 0))
            r_map = tk.Frame(body, bg="#202026")
            r_map.pack(fill=tk.X, pady=(1, 2))
            tk.Label(
                r_map, text="🗺️ 地图ID:", font=("Segoe UI", 8, "bold"),
                fg="#e0e0e0", bg="#202026",
            ).pack(side=tk.LEFT, padx=(0, 4))
            map_entry = tk.Entry(
                r_map, textvariable=map_id_var, width=14, font=("Consolas", 9, "bold"),
                bg="#2a2a32", fg="#80deea", insertbackground="white",
            )
            map_entry.pack(side=tk.LEFT)
            card_data["map_entry"] = map_entry

        if is_current:
            topology = tk.Label(
                body, text="📊 地图拓扑: 等待载入地图数据...",
                font=("Consolas", 8), fg="#b0bec5", bg="#202026", anchor=tk.W,
            )
            topology.pack(fill=tk.X, pady=(0, 3))
            self.lbl_topology_summary = topology

        r_input = tk.Frame(body, bg="#202026")
        r_input.pack(fill=tk.X, pady=2)
        tk.Label(
            r_input, text="🎯 循环长平台:", font=("Segoe UI", 9, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 4))
        platforms_entry = tk.Entry(
            r_input, font=("Consolas", 9, "bold"), bg="#2a2a32", fg="#00e5ff",
            insertbackground="white", width=14,
        )
        platforms_entry.pack(side=tk.LEFT, padx=(0, 4))
        platforms_entry.insert(0, str(spec.get("platforms", "")))
        card_data["platforms_entry"] = platforms_entry
        tk.Button(
            r_input, text="➕ 插入当前平台", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#4a148c", relief=tk.FLAT, padx=6, pady=2,
            command=lambda c=card_data: self._insert_current_platform_to_patrol(c),
        ).pack(side=tk.LEFT, padx=(0, 4))

        r_dwell = tk.Frame(body, bg="#202026")
        r_dwell.pack(fill=tk.X, pady=(3, 1))
        tk.Label(
            r_dwell, text="⏱️ 平台停留:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 4))
        dwell_min_entry = tk.Entry(
            r_dwell, width=5, font=("Consolas", 8),
            bg="#2a2a32", fg="#ffd54f", insertbackground="white",
        )
        dwell_min_entry.pack(side=tk.LEFT)
        dwell_min_entry.insert(0, str(spec.get("dwell_min_sec", 1.0)))
        tk.Label(
            r_dwell, text="～", font=("Segoe UI", 8), fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT, padx=2)
        dwell_max_entry = tk.Entry(
            r_dwell, width=5, font=("Consolas", 8),
            bg="#2a2a32", fg="#ffd54f", insertbackground="white",
        )
        dwell_max_entry.pack(side=tk.LEFT)
        dwell_max_entry.insert(0, str(spec.get("dwell_max_sec", 3.0)))
        tk.Label(
            r_dwell, text="秒   默认站位:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(3, 4))
        positions_entry = tk.Entry(
            r_dwell, width=14, font=("Consolas", 8),
            bg="#2a2a32", fg="#80deea", insertbackground="white",
        )
        positions_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        positions_entry.insert(0, str(spec.get("positions", "30%, 60%")))

        r_random = tk.Frame(body, bg="#202026")
        r_random.pack(fill=tk.X, pady=(3, 1))
        tk.Label(
            r_random, text="🎲 站位随机数:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 4))
        random_entry = tk.Entry(
            r_random, width=8, font=("Consolas", 8),
            bg="#2a2a32", fg="#80deea", insertbackground="white",
        )
        random_entry.pack(side=tk.LEFT)
        random_value = spec.get("position_random_percent", 0.0)
        random_entry.insert(0, f"{float(random_value):g}%")
        tk.Label(
            r_random, text="（如站位50%±10%：在40%～60%随机）",
            font=("Segoe UI", 8), fg="#b0bec5", bg="#202026", anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(6, 0))

        r_overrides = tk.Frame(body, bg="#202026")
        r_overrides.pack(fill=tk.X, pady=(3, 1))
        tk.Label(
            r_overrides, text="📍 单平台站位:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            r_overrides, text="＋ 单独设置平台", font=("Segoe UI", 8),
            fg="#ffffff", bg="#37474f", relief=tk.FLAT,
            command=lambda c=card_data: self._add_platform_position_override(c),
        ).pack(side=tk.LEFT)
        tk.Label(
            r_overrides, text="未设置的平台沿用上方默认站位",
            font=("Segoe UI", 8), fg="#78909c", bg="#202026",
        ).pack(side=tk.LEFT, padx=(7, 0))
        override_container = tk.Frame(body, bg="#202026")
        override_container.pack(fill=tk.X, padx=(18, 0))
        card_data["override_container"] = override_container
        for pid, values in (spec.get("position_overrides") or {}).items():
            self._add_platform_position_override(card_data, pid, values)

        # 休息点设置（分行清晰展示，避免横向空间拥挤被截断）
        r_rest1 = tk.Frame(body, bg="#202026")
        r_rest1.pack(fill=tk.X, pady=(4, 1))
        tk.Label(
            r_rest1, text="🪑 休息点设置:", font=("Segoe UI", 8, "bold"),
            fg="#00e5ff", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            r_rest1, text="平台:", font=("Segoe UI", 8),
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 2))
        rest_platform_entry = tk.Entry(
            r_rest1, width=5, font=("Consolas", 8, "bold"),
            bg="#2a2a32", fg="#00e5ff", insertbackground="white", justify="center",
        )
        rest_platform_entry.pack(side=tk.LEFT, padx=(0, 6))
        raw_rest_pid = spec.get("rest_platform_id")
        if raw_rest_pid is not None and str(raw_rest_pid).strip() != "":
            rest_platform_entry.insert(0, str(raw_rest_pid))
        tk.Label(
            r_rest1, text="（留空表示不开启休息）",
            font=("Segoe UI", 8), fg="#78909c", bg="#202026",
        ).pack(side=tk.LEFT)

        r_rest2 = tk.Frame(body, bg="#202026")
        r_rest2.pack(fill=tk.X, pady=(2, 1))
        tk.Label(
            r_rest2, text="⏱️ 休息时长:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 2))
        rest_duration_min_entry = tk.Entry(
            r_rest2, width=4, font=("Consolas", 8),
            bg="#2a2a32", fg="#ffd54f", insertbackground="white", justify="center",
        )
        rest_duration_min_entry.pack(side=tk.LEFT)
        rest_duration_min_entry.insert(0, str(spec.get("rest_duration_min_sec", 30.0)))
        tk.Label(r_rest2, text="～", font=("Segoe UI", 8), fg="#b0bec5", bg="#202026").pack(side=tk.LEFT, padx=1)
        rest_duration_max_entry = tk.Entry(
            r_rest2, width=4, font=("Consolas", 8),
            bg="#2a2a32", fg="#ffd54f", insertbackground="white", justify="center",
        )
        rest_duration_max_entry.pack(side=tk.LEFT)
        rest_duration_max_entry.insert(0, str(spec.get("rest_duration_max_sec", 60.0)))
        tk.Label(r_rest2, text="秒", font=("Segoe UI", 8), fg="#b0bec5", bg="#202026").pack(side=tk.LEFT, padx=(1, 6))

        tk.Label(
            r_rest2, text="⏳ 巡逻间隔:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026",
        ).pack(side=tk.LEFT, padx=(0, 2))
        rest_interval_min_entry = tk.Entry(
            r_rest2, width=4, font=("Consolas", 8),
            bg="#2a2a32", fg="#ffd54f", insertbackground="white", justify="center",
        )
        rest_interval_min_entry.pack(side=tk.LEFT)
        rest_interval_min_entry.insert(0, str(spec.get("rest_interval_min_sec", 30.0)))
        tk.Label(r_rest2, text="～", font=("Segoe UI", 8), fg="#b0bec5", bg="#202026").pack(side=tk.LEFT, padx=1)
        rest_interval_max_entry = tk.Entry(
            r_rest2, width=4, font=("Consolas", 8),
            bg="#2a2a32", fg="#ffd54f", insertbackground="white", justify="center",
        )
        rest_interval_max_entry.pack(side=tk.LEFT)
        rest_interval_max_entry.insert(0, str(spec.get("rest_interval_max_sec", 40.0)))
        tk.Label(r_rest2, text="秒", font=("Segoe UI", 8), fg="#b0bec5", bg="#202026").pack(side=tk.LEFT, padx=(1, 2))

        r_rest_tip = tk.Frame(body, bg="#202026")
        r_rest_tip.pack(fill=tk.X, pady=(2, 2))
        tk.Label(
            r_rest_tip,
            text="💡 到休息平台中点站稳5秒后按椅子键，休息结束后自动返回",
            font=("Segoe UI", 8), fg="#78909c", bg="#202026", anchor=tk.W, justify=tk.LEFT,
        ).pack(fill=tk.X)

        card_data.update({
            "platforms_entry": platforms_entry,
            "dwell_min_entry": dwell_min_entry,
            "dwell_max_entry": dwell_max_entry,
            "positions_entry": positions_entry,
            "random_entry": random_entry,
            "rest_platform_entry": rest_platform_entry,
            "rest_duration_min_entry": rest_duration_min_entry,
            "rest_duration_max_entry": rest_duration_max_entry,
            "rest_interval_min_entry": rest_interval_min_entry,
            "rest_interval_max_entry": rest_interval_max_entry,
        })
        if is_current:
            # 保留旧属性名，供现有单图预览、热键与测试代码继续使用。
            self.ent_patrol_platforms = platforms_entry
            self.ent_patrol_dwell_min = dwell_min_entry
            self.ent_patrol_dwell_max = dwell_max_entry
            self.ent_single_patrol_positions = positions_entry
            self.ent_patrol_position_random = random_entry
            self.ent_rest_platform = rest_platform_entry
            self.ent_rest_duration_min = rest_duration_min_entry
            self.ent_rest_duration_max = rest_duration_max_entry
            self.ent_rest_interval_min = rest_interval_min_entry
            self.ent_rest_interval_max = rest_interval_max_entry
        map_id_var.trace_add("write", lambda *_args: self._refresh_path_map_card_titles())
        self._refresh_path_map_card_titles()
        return card_data

    def _toggle_path_map_card(self, card_data: Dict[str, Any]) -> None:
        expanded = not bool(card_data.get("expanded"))
        self._set_path_map_card_expanded(card_data, expanded)

    @staticmethod
    def _format_position_values(values: Any) -> str:
        if isinstance(values, (list, tuple)):
            return ", ".join(f"{float(value) * 100:g}%" for value in values)
        return str(values or "")

    def _add_platform_position_override(
        self, card_data: Dict[str, Any], pid: Any = None, values: Any = ""
    ) -> None:
        platforms = self._parse_patrol_string(card_data["platforms_entry"].get())
        selected = {row["platform_var"].get() for row in card_data["override_rows"]}
        choices = [f"P{value}" for value in platforms]
        if pid is None:
            pid = next((value for value in choices if value not in selected), "")
            if not pid:
                messagebox.showinfo(
                    "单平台站位", "请先在循环长平台中添加一个尚未单独设置的平台。",
                    parent=self.root,
                )
                return
        else:
            pid = f"P{str(pid).lstrip('Pp')}"
            if pid not in choices:
                choices.append(pid)
        row_frame = tk.Frame(card_data["override_container"], bg="#202026")
        row_frame.pack(fill=tk.X, pady=1)
        platform_var = tk.StringVar(value=pid)
        platform_box = ttk.Combobox(
            row_frame, textvariable=platform_var, width=7, state="readonly",
        )
        platform_box.pack(side=tk.LEFT, padx=(0, 5))
        platform_box.configure(
            postcommand=lambda c=card_data, box=platform_box: box.configure(
                values=[f"P{value}" for value in self._parse_patrol_string(
                    c["platforms_entry"].get()
                )]
            )
        )
        platform_box.configure(values=choices)
        tk.Label(
            row_frame, text="站位:", font=("Segoe UI", 8),
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)
        value_entry = tk.Entry(
            row_frame, font=("Consolas", 8), bg="#2a2a32", fg="#80deea",
            insertbackground="white", width=22,
        )
        value_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 5))
        value_entry.insert(0, self._format_position_values(values))
        row = {"frame": row_frame, "platform_var": platform_var,
               "positions_entry": value_entry}
        card_data["override_rows"].append(row)
        tk.Button(
            row_frame, text="删除", font=("Segoe UI", 8), fg="#ef9a9a",
            bg="#202026", relief=tk.FLAT,
            command=lambda c=card_data, r=row: self._delete_platform_position_override(c, r),
        ).pack(side=tk.LEFT)

    @staticmethod
    def _delete_platform_position_override(card_data: Dict[str, Any], row: Dict[str, Any]) -> None:
        row["frame"].destroy()
        card_data["override_rows"].remove(row)

    def _set_path_map_card_expanded(
        self, card_data: Dict[str, Any], expanded: bool
    ) -> None:
        card_data["expanded"] = bool(expanded)
        card_data["toggle"].config(text="▼" if expanded else "▶")
        if expanded:
            card_data["body"].pack(fill=tk.X, padx=(16, 0), pady=(1, 2))
        else:
            card_data["body"].pack_forget()

    def _path_map_name(self, map_id: str) -> str:
        text = str(map_id or "").strip()
        current = getattr(self, "current_map_info", None) or {}
        if text and str(current.get("map_id", "")) == text:
            return str(current.get("chinese_name") or current.get("name") or "地图")
        if text.isdigit():
            try:
                for item in self.map_resolver.load_local_maps():
                    if str(item.get("map_id", "")) == text:
                        return str(item.get("name_cn") or item.get("name_en") or "地图")
            except Exception:
                pass
        return ""

    def _refresh_path_map_card_titles(self) -> None:
        cards = getattr(self, "path_map_cards", [])
        for index, card_data in enumerate(cards):
            map_id = card_data["map_id_var"].get().strip()
            name = self._path_map_name(map_id)
            prefix = "当前地图" if index == 0 else f"第{index + 1}张地图"
            suffix = f"：{name}[{map_id}]" if map_id else "：等待地图ID"
            card_data["title"].config(text=prefix + suffix)

    def _swap_path_map_card_values(self, first: Dict[str, Any], second: Dict[str, Any]) -> None:
        """只交换卡片草稿值，保留首卡不可删除和后续卡删除按钮的结构。"""
        first_map = first["map_id_var"].get()
        second_map = second["map_id_var"].get()
        entry_keys = (
            "platforms_entry", "dwell_min_entry", "dwell_max_entry",
            "positions_entry", "random_entry",
            "rest_platform_entry", "rest_duration_min_entry", "rest_duration_max_entry",
            "rest_interval_min_entry", "rest_interval_max_entry",
        )
        first_values = [first[key].get() for key in entry_keys]
        second_values = [second[key].get() for key in entry_keys]
        first_overrides = [
            (row["platform_var"].get(), row["positions_entry"].get())
            for row in first["override_rows"]
        ]
        second_overrides = [
            (row["platform_var"].get(), row["positions_entry"].get())
            for row in second["override_rows"]
        ]
        first["map_id_var"].set(second_map)
        second["map_id_var"].set(first_map)
        for key, value in zip(entry_keys, second_values):
            first[key].delete(0, tk.END)
            first[key].insert(0, value)
        for key, value in zip(entry_keys, first_values):
            second[key].delete(0, tk.END)
            second[key].insert(0, value)
        for card_data, overrides in ((first, second_overrides), (second, first_overrides)):
            for row in list(card_data["override_rows"]):
                self._delete_platform_position_override(card_data, row)
            for pid, values in overrides:
                self._add_platform_position_override(card_data, pid, values)

    def _sync_current_path_map_card(self, map_id: Any) -> None:
        """编辑态让首卡始终对应当前地图；跨图运行中不随切图重排。"""
        cards = getattr(self, "path_map_cards", [])
        if not cards or getattr(getattr(self, "combat_fsm", None), "is_running", False):
            return
        text = str(map_id or "").strip()
        if not text.isdigit():
            return
        first_text = cards[0]["map_id_var"].get().strip()
        if not first_text or len(cards) == 1:
            cards[0]["map_id_var"].set(text)
        elif first_text != text:
            match = next(
                (card for card in cards[1:] if card["map_id_var"].get().strip() == text),
                None,
            )
            if match is not None:
                self._swap_path_map_card_values(cards[0], match)
        self._refresh_path_map_card_titles()

    def _add_path_map_card(self) -> None:
        cards = getattr(self, "path_map_cards", [])
        if cards:
            self._set_path_map_card_expanded(cards[0], False)
            source = cards[-1]
            spec = {
                "map_id": "",
                "platforms": "",
                "dwell_min_sec": source["dwell_min_entry"].get().strip() or "1",
                "dwell_max_sec": source["dwell_max_entry"].get().strip() or "3",
                "positions": source["positions_entry"].get().strip() or "30%, 60%",
                "position_overrides": {},
                "position_random_percent": self._parse_patrol_position_random(
                    source["random_entry"].get()
                ) * 100.0,
                "rest_platform_id": None,
                "rest_duration_min_sec": source["rest_duration_min_entry"].get().strip() or "30.0",
                "rest_duration_max_sec": source["rest_duration_max_entry"].get().strip() or "60.0",
                "rest_interval_min_sec": source["rest_interval_min_entry"].get().strip() or "30.0",
                "rest_interval_max_sec": source["rest_interval_max_entry"].get().strip() or "40.0",
            }
        else:
            spec = self._initial_path_map_specs()[0]
        self._create_path_map_card(spec, is_current=False, expanded=True)
        self.var_cross_map_patrol.set(True)
        self._refresh_path_map_card_titles()

    def _delete_path_map_card(self, card_data: Dict[str, Any]) -> None:
        cards = getattr(self, "path_map_cards", [])
        if card_data not in cards or card_data.get("is_current"):
            return
        card_data["shell"].destroy()
        cards.remove(card_data)
        self.var_cross_map_patrol.set(len(cards) > 1)
        self._refresh_path_map_card_titles()

    def _parse_patrol_string(self, raw: str) -> List[int]:
        """解析平台逗号分隔字符串为整型列表"""
        res = []
        if not raw:
            return res
        for part in str(raw).replace("，", ",").replace(" ", ",").split(","):
            part = part.strip().lstrip("Pp")
            if part.isdigit():
                res.append(int(part))
        return res

    @staticmethod
    def _parse_patrol_dwell_range(
        minimum, maximum, fallback: Tuple[float, float] = (1.0, 3.0)
    ) -> Tuple[float, float]:
        try:
            low, high = float(minimum), float(maximum)
        except (TypeError, ValueError):
            return fallback
        low = max(0.0, min(600.0, low))
        high = max(0.0, min(600.0, high))
        return (min(low, high), max(low, high))

    @staticmethod
    def _parse_single_patrol_positions(raw) -> List[float]:
        """解析“30%, 60%”或“(30 60)”为平台宽度比例。"""
        text = str(raw or "").replace("（", " ").replace("）", " ")
        text = text.replace("(", " ").replace(")", " ")
        for separator in ("，", ",", ";", "；", "/", "|"):
            text = text.replace(separator, " ")
        parsed: List[float] = []
        for token in text.split():
            had_percent = "%" in token or "％" in token
            token = token.replace("%", "").replace("％", "").strip()
            try:
                value = float(token)
            except (TypeError, ValueError):
                continue
            fraction = value / 100.0 if had_percent or abs(value) > 1.0 else value
            fraction = max(0.0, min(1.0, fraction))
            if all(abs(fraction - existing) > 1e-6 for existing in parsed):
                parsed.append(fraction)
        return parsed or [0.30, 0.60]

    @staticmethod
    def _parse_single_patrol_positions_strict(raw) -> List[float]:
        """应用按钮使用的严格解析；无有效数字时不偷偷回退默认值。"""
        text = str(raw or "").replace("（", " ").replace("）", " ")
        text = text.replace("(", " ").replace(")", " ")
        for separator in ("，", ",", ";", "；", "/", "|"):
            text = text.replace(separator, " ")
        parsed: List[float] = []
        for token in text.split():
            had_percent = "%" in token or "％" in token
            token = token.replace("%", "").replace("％", "").strip()
            try:
                value = float(token)
            except (TypeError, ValueError):
                return []
            fraction = value / 100.0 if had_percent or abs(value) > 1.0 else value
            if not 0.0 <= fraction <= 1.0:
                return []
            if all(abs(fraction - existing) > 1e-6 for existing in parsed):
                parsed.append(fraction)
        return parsed

    @staticmethod
    def _parse_patrol_position_random(raw) -> float:
        text = str(raw if raw is not None else "0").strip()
        text = text.replace("%", "").replace("％", "").strip()
        try:
            value = float(text)
        except (TypeError, ValueError):
            return 0.0
        # config 与输入框统一存“百分数数值”：0.5 表示0.5%，10表示10%。
        fraction = value / 100.0
        return max(0.0, min(1.0, fraction))

    def _restore_path_settings_inputs(self):
        """校验失败后恢复上一次已生效的整组路径参数。"""
        for card_data in list(getattr(self, "path_map_cards", [])):
            card_data["shell"].destroy()
        self.path_map_cards = []
        specs = self._initial_path_map_specs()
        for index, spec in enumerate(specs):
            self._create_path_map_card(
                spec, is_current=(index == 0), expanded=(index == 0)
            )
        self.var_cross_map_patrol.set(
            len(self.path_map_cards) > 1
        )
        self.var_disable_intra_map_portals.set(
            bool(self.config.get("disable_intra_map_portals", False))
        )
        graph = getattr(self, "platform_graph", None)
        if graph is not None and getattr(graph, "nodes", None):
            total_edges = sum(len(edges) for edges in graph.edges.values())
            self.lbl_topology_summary.config(
                text=f"📊 地图拓扑: 已解析 {len(graph.nodes)} 个平台 | {total_edges} 条动作通道",
                fg="#69f0ae",
            )

    def _export_path_settings(self) -> None:
        """导出已应用的路径配置；草稿必须先点“应用”。"""
        specs = self.config.get("cross_map_patrol_maps")
        if not isinstance(specs, list) or not specs:
            messagebox.showinfo(
                "导出路径设置", "请先点击“应用”，再导出已生效的路径设置。",
                parent=self.root,
            )
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="导出路径设置", defaultextension=".json",
            filetypes=[("路径设置 JSON", "*.json")],
        )
        if not path:
            return
        document = {
            "schema": "patrol-route-settings", "version": 1,
            "maps": specs,
            "disable_intra_map_portals": bool(
                self.config.get("disable_intra_map_portals", False)
            ),
        }
        try:
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except OSError as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.root)
            return
        self.log(f"📤 [路径设置导出] {len(specs)}张地图 → {path}")

    def _import_path_settings(self) -> None:
        """导入草稿后复用“应用”的拓扑和参数校验；失败会回退。"""
        if getattr(getattr(self, "combat_fsm", None), "is_running", False):
            messagebox.showwarning(
                "导入路径设置", "请先按 F6 停止巡逻，再导入路径设置。",
                parent=self.root,
            )
            return
        runner = getattr(self, "random_path_test_runner", None)
        if runner is not None and runner.active:
            messagebox.showwarning(
                "导入路径设置", "请先停止随机全图行走测试，再导入路径设置。",
                parent=self.root,
            )
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="导入路径设置",
            filetypes=[("路径设置 JSON", "*.json")],
        )
        if not path:
            return
        try:
            if os.path.getsize(path) > 262144:
                raise ValueError("文件超过 256KB，不是有效的路径设置。")
            with open(path, "r", encoding="utf-8-sig") as stream:
                document = json.load(stream)
            if (not isinstance(document, dict)
                    or document.get("schema") != "patrol-route-settings"
                    or document.get("version") != 1):
                raise ValueError("文件格式或版本不受支持。")
            specs = document.get("maps")
            if not isinstance(specs, list) or not 1 <= len(specs) <= 32:
                raise ValueError("地图列表必须包含 1～32 张地图。")
            for index, spec in enumerate(specs, 1):
                if not isinstance(spec, dict):
                    raise ValueError(f"第{index}张地图的数据格式无效。")
                overrides = spec.get("position_overrides", {})
                if not isinstance(overrides, dict) or len(overrides) > 128:
                    raise ValueError(f"第{index}张地图的单平台站位格式无效。")
                float(spec.get("position_random_percent", 0.0))
            disable_portals = document.get("disable_intra_map_portals", False)
            if not isinstance(disable_portals, bool):
                raise ValueError("地图内传送点开关格式无效。")
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            messagebox.showerror("导入失败", str(exc), parent=self.root)
            return
        try:
            for card_data in list(self.path_map_cards):
                card_data["shell"].destroy()
            self.path_map_cards = []
            for index, spec in enumerate(specs):
                self._create_path_map_card(
                    spec, is_current=(index == 0), expanded=(index == 0)
                )
            self.var_cross_map_patrol.set(len(specs) > 1)
            self.var_disable_intra_map_portals.set(disable_portals)
        except Exception as exc:
            self._restore_path_settings_inputs()
            messagebox.showerror("导入失败", f"无法载入路径设置：{exc}", parent=self.root)
            return
        self.log(f"📥 [路径设置导入] {len(specs)}张地图，开始校验并应用：{path}")
        self._apply_path_settings()

    def _apply_path_settings(self, validated: Optional[Tuple[str, List[str]]] = None):
        """先校验整组路径草稿，全部可行后再原子提交。"""
        if validated is not None:
            button = getattr(self, "_path_apply_button", None)
            if button is not None:
                button.configure(state=tk.NORMAL, text="应用")
        if (
            getattr(self, "random_path_test_runner", None) is not None
            and self.random_path_test_runner.active
        ):
            messagebox.showwarning(
                "测试正在运行",
                "请先停止随机全图行走测试，再修改并应用路径设置。",
                parent=self.root,
            )
            return

        def reject(message: str):
            messagebox.showerror("路径设置不可用", message, parent=self.root)
            self._restore_path_settings_inputs()
            self.log(f"❌ [路径设置回退] {message}")

        cards = list(getattr(self, "path_map_cards", []))
        if not cards:
            reject("至少需要保留“当前地图”。")
            return

        current_map_id = self._get_current_map_id()
        cross_map_enabled = len(cards) > 1
        disable_intra_map_portals = bool(
            self.var_disable_intra_map_portals.get()
        )
        world_stops: List[WorldPatrolStop] = []
        saved_specs: List[Dict[str, Any]] = []
        for index, card_data in enumerate(cards):
            label = "当前地图" if index == 0 else f"第{index + 1}张地图"
            map_text = card_data["map_id_var"].get().strip()
            if index == 0 and not map_text and current_map_id is not None:
                map_text = str(current_map_id)
                card_data["map_id_var"].set(map_text)
            if cross_map_enabled and not map_text.isdigit():
                reject(f"{label}缺少有效的纯数字 MapID。")
                return

            patrol_text = card_data["platforms_entry"].get().strip()
            patrol = self._parse_patrol_string(patrol_text)
            if not patrol:
                reject(f"{label}的循环长平台不能为空。")
                return

            try:
                low = float(card_data["dwell_min_entry"].get().strip())
                high = float(card_data["dwell_max_entry"].get().strip())
                if not (0.0 <= low <= high <= 600.0):
                    raise ValueError
            except (TypeError, ValueError):
                reject(f"{label}的平台停留必须满足 0 ≤ 最小值 ≤ 最大值 ≤ 600 秒。")
                return

            positions_text = card_data["positions_entry"].get().strip()
            positions = self._parse_single_patrol_positions_strict(positions_text)
            if not positions:
                reject(f"{label}的平台站位格式无效，请填写例如 50% 或 30%, 60%。")
                return

            random_text = card_data["random_entry"].get().strip()
            try:
                stripped = random_text.replace("%", "").replace("％", "").strip()
                raw_random = float(stripped)
                random_fraction = raw_random / 100.0
                if not 0.0 <= random_fraction <= 1.0:
                    raise ValueError
            except (TypeError, ValueError):
                reject(f"{label}的站位随机数必须是 0%～100% 之间的百分比。")
                return

            invalid_ranges = [
                value for value in positions
                if value - random_fraction < 0.0 or value + random_fraction > 1.0
            ]
            if invalid_ranges:
                bad = ", ".join(f"{value * 100:g}%" for value in invalid_ranges)
                reject(
                    f"{label}的站位 {bad} 加减 {random_fraction * 100:g}% "
                    "会越出平台，必须完整落在 0%～100% 内。"
                )
                return

            position_overrides: Dict[str, List[float]] = {}
            for row in card_data["override_rows"]:
                platform_text = row["platform_var"].get().strip()
                if not platform_text.upper().startswith("P") or not platform_text[1:].isdigit():
                    reject(f"{label}的单平台站位缺少有效平台编号。")
                    return
                platform_id = int(platform_text[1:])
                if platform_id not in patrol:
                    reject(f"{label}的 P{platform_id} 不在循环长平台中。")
                    return
                key = str(platform_id)
                if key in position_overrides:
                    reject(f"{label}的 P{platform_id} 被重复设置，请只保留一行。")
                    return
                override_values = self._parse_single_patrol_positions_strict(
                    row["positions_entry"].get().strip()
                )
                if not override_values:
                    reject(f"{label}的 P{platform_id} 站位无效，请填写例如 30%, 50%, 70%。")
                    return
                if any(value - random_fraction < 0.0 or value + random_fraction > 1.0
                       for value in override_values):
                    reject(f"{label}的 P{platform_id} 站位加减随机数会越出平台 0%～100%。")
                    return
                position_overrides[key] = override_values

            # 休息点设置解析 (如果休息平台为空，则没有休息任务)
            rest_platform_raw = card_data["rest_platform_entry"].get().strip()
            rest_pid = None
            if rest_platform_raw:
                cleaned = rest_platform_raw.lower().lstrip("p").strip()
                if not cleaned.isdigit():
                    reject(f"{label}的休息平台格式无效，请输入有效平台编号（如 7 或 P7）或留空。")
                    return
                rest_pid = int(cleaned)

            try:
                rest_d_min = float(card_data["rest_duration_min_entry"].get().strip())
                rest_d_max = float(card_data["rest_duration_max_entry"].get().strip())
                if not (0.0 <= rest_d_min <= rest_d_max <= 86400.0):
                    raise ValueError
            except (TypeError, ValueError):
                reject(f"{label}的休息时间范围必须满足 0 ≤ 最小值 ≤ 最大值 秒。")
                return

            try:
                rest_i_min = float(card_data["rest_interval_min_entry"].get().strip())
                rest_i_max = float(card_data["rest_interval_max_entry"].get().strip())
                if not (1.0 <= rest_i_min <= rest_i_max <= 86400.0):
                    raise ValueError
            except (TypeError, ValueError):
                reject(f"{label}的休息间隔范围必须满足 1 ≤ 最小值 ≤ 最大值 秒。")
                return

            spec = {
                "map_id": int(map_text) if map_text.isdigit() else None,
                "platforms": patrol_text,
                "dwell_min_sec": low,
                "dwell_max_sec": high,
                "positions": positions_text,
                "position_overrides": position_overrides,
                "position_random_percent": random_fraction * 100.0,
                "rest_platform_id": rest_pid,
                "rest_duration_min_sec": rest_d_min,
                "rest_duration_max_sec": rest_d_max,
                "rest_interval_min_sec": rest_i_min,
                "rest_interval_max_sec": rest_i_max,
            }
            saved_specs.append(spec)
            if map_text.isdigit():
                world_stops.append(WorldPatrolStop(
                    int(map_text), tuple(patrol), low, high,
                    tuple(positions), random_fraction,
                    rest_platform_id=rest_pid,
                    rest_duration_min_sec=rest_d_min,
                    rest_duration_max_sec=rest_d_max,
                    rest_interval_min_sec=rest_i_min,
                    rest_interval_max_sec=rest_i_max,
                    position_overrides=tuple(sorted(
                        (int(pid), tuple(values))
                        for pid, values in position_overrides.items()
                    )),
                ))

        if cross_map_enabled:
            map_ids = [stop.map_id for stop in world_stops]
            if len(set(map_ids)) != len(map_ids):
                reject("多地图巡逻中的 MapID 不能重复。")
                return
            rest_maps = [stop.map_id for stop in world_stops if stop.rest_platform_id is not None]
            if len(rest_maps) > 1:
                reject(
                    "多地图巡逻同时只能设置一个休息点；"
                    f"当前地图 {', '.join(map(str, rest_maps))} 都设置了休息平台。"
                )
                return
            new_signature = json.dumps(saved_specs, ensure_ascii=False, sort_keys=True)
            old_signature = json.dumps(
                self.config.get("cross_map_patrol_maps", []),
                ensure_ascii=False, sort_keys=True,
            )
            if (
                getattr(getattr(self, "combat_fsm", None), "is_running", False)
                and new_signature != old_signature
            ):
                reject("修改多地图路线或逐地图参数前，请先按F6停止当前挂机。")
                return
            merge_platforms = bool(self.merge_short_platforms)
            allow_run_jump = bool(self.config.get("enable_run_jump_grab", True))
            validation_signature = json.dumps({
                "maps": saved_specs,
                "merge": merge_platforms,
                "run_jump": allow_run_jump,
                "disable_intra_map_portals": disable_intra_map_portals,
            }, ensure_ascii=False, sort_keys=True)
            if validated is None:
                button = getattr(self, "_path_apply_button", None)
                if button is not None:
                    button.configure(state=tk.DISABLED, text="校验中…")
                self.log("🌐 [跨地图路径校验] 后台检查平台与传送门，界面可继续操作")

                def validate_in_background():
                    try:
                        self.world_route_planner.set_merge_short_platforms(merge_platforms)
                        errors = self.world_route_planner.validate_stops(
                            world_stops,
                            allow_run_jump=allow_run_jump,
                            allow_intra_map_portal=not disable_intra_map_portals,
                        )
                    except Exception as exc:
                        errors = [f"校验异常：{exc}"]
                    try:
                        self.root.after(0, lambda: self._apply_path_settings(
                            (validation_signature, errors)
                        ))
                    except RuntimeError:
                        pass  # 窗口已关闭

                threading.Thread(
                    target=validate_in_background, daemon=True,
                    name="WorldRouteValidation",
                ).start()
                return
            if validated[0] != validation_signature:
                self.log("⚠️ [跨地图路径校验] 校验期间输入已变更；未应用旧结果，请重新点击应用")
                return
            world_errors = validated[1]
            if world_errors:
                reject("跨地图路线不可用：\n" + "\n".join(world_errors))
                return
        else:
            first = saved_specs[0]
            patrol = self._parse_patrol_string(first["platforms"])
            graph = self.platform_graph
            if graph is None or not getattr(graph, "nodes", None):
                reject("当前地图拓扑尚未加载，无法检查循环平台可达性。")
                return
            missing = [platform_id for platform_id in patrol if graph.get_node(platform_id) is None]
            if missing:
                reject(f"当前地图不存在长平台：{', '.join('P' + str(value) for value in missing)}。")
                return
            unreachable = []
            if len(patrol) > 1:
                allow_run_jump = bool(self.config.get("enable_run_jump_grab", True))
                for source, destination in zip(patrol, patrol[1:] + [patrol[0]]):
                    if not graph.find_path(
                        source,
                        destination,
                        allow_run_jump=allow_run_jump,
                        allow_portal=not disable_intra_map_portals,
                    ):
                        unreachable.append(f"P{source}→P{destination}")
            if unreachable:
                reject("以下循环段不可达：" + "、".join(unreachable) + "。")
                return

        # 到这里才提交，校验阶段不会污染缓存、config 或正在执行的 FSM。
        first = saved_specs[0]
        first_patrol = self._parse_patrol_string(first["platforms"])
        first_positions = self._parse_single_patrol_positions_strict(first["positions"])
        self.cached_patrol_platforms = list(first_patrol)
        self.cached_patrol_dwell_range = (
            float(first["dwell_min_sec"]), float(first["dwell_max_sec"])
        )
        self.cached_single_patrol_positions = list(first_positions)
        self.cached_patrol_position_overrides = self._saved_position_overrides(
            first.get("position_overrides", {})
        )
        self.cached_patrol_position_random = float(first["position_random_percent"]) / 100.0
        self.config["patrol_platforms"] = first["platforms"]
        self.config["patrol_dwell_min_sec"] = first["dwell_min_sec"]
        self.config["patrol_dwell_max_sec"] = first["dwell_max_sec"]
        self.config["single_patrol_positions"] = first["positions"]
        self.config["patrol_position_overrides"] = first.get("position_overrides", {})
        self.config["patrol_position_random_percent"] = first["position_random_percent"]
        self.config["rest_platform_id"] = first.get("rest_platform_id")
        self.config["rest_duration_min_sec"] = float(first.get("rest_duration_min_sec", 30.0))
        self.config["rest_duration_max_sec"] = float(first.get("rest_duration_max_sec", 60.0))
        self.config["rest_interval_min_sec"] = float(first.get("rest_interval_min_sec", 30.0))
        self.config["rest_interval_max_sec"] = float(first.get("rest_interval_max_sec", 40.0))
        self.config["cross_map_patrol_enabled"] = cross_map_enabled
        self.config["cross_map_patrol_maps"] = saved_specs
        self.config["disable_intra_map_portals"] = disable_intra_map_portals
        cross_map_text = ";".join(
            f"{stop.map_id}:{','.join('p' + str(pid) for pid in stop.platforms)}"
            for stop in world_stops
        )
        self.config["cross_map_patrol_route"] = cross_map_text
        self.cached_world_patrol_stops = list(world_stops)
        if self.world_patrol_controller is not None:
            self.world_patrol_controller.configure(cross_map_enabled, world_stops)
        self._save_config()
        if cross_map_enabled:
            world_summary = " → ".join(
                f"{stop.map_id}:{','.join('P' + str(pid) for pid in stop.platforms)}"
                f"[{stop.dwell_min_sec:g}～{stop.dwell_max_sec:g}s]"
                for stop in world_stops
            )
            self.log(
                f"✅ [跨地图路径已应用] {world_summary}；使用普通门与已确认脚本门；"
                "各地图使用独立平台、停留、站位与随机参数；"
                f"图内传送点={'禁用' if disable_intra_map_portals else '启用'}"
            )
        else:
            self.log(
                f"✅ [路径设置已应用] 平台={first_patrol}，"
                f"停留={first['dwell_min_sec']:g}～{first['dwell_max_sec']:g}s，"
                f"站位={first['positions']}，随机=±{first['position_random_percent']:g}%，"
                f"图内传送点={'禁用' if disable_intra_map_portals else '启用'}"
            )
        self.var_cross_map_patrol.set(cross_map_enabled)
        self._refresh_path_map_card_titles()
        self._preview_patrol_route()

    def _on_patrol_behavior_changed(self, event=None):
        try:
            low_text = self.ent_patrol_dwell_min.get().strip()
            high_text = self.ent_patrol_dwell_max.get().strip()
            # 编辑中的空字符串不覆盖上一份有效配置。
            if not low_text or not high_text:
                return
            dwell_range = self._parse_patrol_dwell_range(
                low_text, high_text, self.cached_patrol_dwell_range
            )
            positions_text = self.ent_single_patrol_positions.get().strip()
            positions = self._parse_single_patrol_positions(positions_text)
            self.cached_patrol_dwell_range = dwell_range
            self.cached_single_patrol_positions = positions
            self.config["patrol_dwell_min_sec"] = dwell_range[0]
            self.config["patrol_dwell_max_sec"] = dwell_range[1]
            self.config["single_patrol_positions"] = positions_text or "30%, 60%"
            self._save_config()
        except Exception:
            pass

    def _get_patrol_dwell_range(self) -> Tuple[float, float]:
        return tuple(getattr(self, "cached_patrol_dwell_range", (1.0, 3.0)))

    def _get_single_patrol_positions(self) -> List[float]:
        return list(getattr(self, "cached_single_patrol_positions", (0.30, 0.60)))

    @staticmethod
    def _saved_position_overrides(raw: Any) -> Dict[int, Tuple[float, ...]]:
        if not isinstance(raw, dict):
            return {}
        result = {}
        for pid, values in raw.items():
            if not isinstance(values, (list, tuple)):
                continue
            try:
                parsed = tuple(float(value) for value in values)
                if parsed and all(0.0 <= value <= 1.0 for value in parsed):
                    result[int(pid)] = parsed
            except (TypeError, ValueError):
                continue
        return result

    def _get_effective_platform_patrol_positions(self) -> Dict[int, Tuple[float, ...]]:
        controller = getattr(self, "world_patrol_controller", None)
        if controller is not None and controller.running:
            return controller.current_position_overrides()
        if bool(self.config.get("cross_map_patrol_enabled", False)):
            map_id = self._get_current_map_id()
            for stop in getattr(self, "cached_world_patrol_stops", ()):
                if stop.map_id == map_id:
                    return dict(stop.position_overrides)
        return dict(getattr(self, "cached_patrol_position_overrides", {}))

    def _get_current_map_id(self) -> Optional[int]:
        value = (getattr(self, "current_map_info", None) or {}).get("map_id")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _find_nearby_cross_map_portal(self) -> Optional[Tuple[Dict[str, Any], float]]:
        """返回触发范围内最近的 type=1/2 跨地图传送门及世界距离。"""
        try:
            trigger_range = float(self.config.get("portal_ocr_trigger_range_px", 200.0))
        except (TypeError, ValueError):
            trigger_range = 200.0
        if trigger_range < 0.0:
            return None

        world_pos = getattr(self, "current_player_world_pos", None)
        if world_pos is None:
            world_pos = getattr(self, "current_player_raw_world_pos", None)
        if world_pos is None:
            return None

        current_map_id = self._get_current_map_id()
        graph = getattr(self, "platform_graph", None)
        portals = list(getattr(graph, "portals", None) or [])
        if not portals:
            map_data = getattr(self, "_current_graph_map_data", None)
            if not map_data:
                map_data = (getattr(self, "current_map_info", None) or {}).get("raw_map_data")
            portals = list((map_data or {}).get("portals", []) or [])

        px, py = float(world_pos[0]), float(world_pos[1])
        nearest = None
        nearest_distance = float("inf")
        for portal in portals:
            if not isinstance(portal, dict):
                continue
            try:
                portal_type = int(portal.get("type", portal.get("pt", -1)))
                target_map_id = int(portal.get("toMap", portal.get("tm", 999999999)))
                portal_x = float(portal.get("x"))
                portal_y = float(portal.get("y"))
            except (TypeError, ValueError):
                continue
            # 排除出生点、同图门和没有有效目的地图的脚本占位门。
            if portal_type not in CROSS_MAP_PORTAL_TYPES or target_map_id == 999999999:
                continue
            if current_map_id is not None and target_map_id == current_map_id:
                continue
            distance = math.hypot(px - portal_x, py - portal_y)
            if distance <= trigger_range and distance < nearest_distance:
                nearest = portal
                nearest_distance = distance
        if nearest is None:
            return None
        return nearest, nearest_distance

    def _request_portal_ocr_acceleration(self, source: str) -> bool:
        """若人物在跨地图传送门附近，则唤醒并保持地图 OCR 加速态。"""
        match = self._find_nearby_cross_map_portal()
        if match is None:
            return False
        portal, distance = match
        now = time.perf_counter()
        target_map_id = int(portal.get("toMap", portal.get("tm", 999999999)))
        current_map_id = self._get_current_map_id()
        portal_name = str(portal.get("portalName", portal.get("pn", "?")) or "?")
        trusted_entry = "walk_through_portal" in str(source)

        with self._portal_ocr_lock:
            pending = self._portal_ocr_pending
            # 自动 SendInput 也可能被 GetAsyncKeyState 短暂看到；同一按键边沿
            # 会走到这里两次。同一传送门已有未完成请求时只负责再次唤醒，
            # 不重置起始时间，也不产生重复日志。
            if (
                pending is not None
                and int(pending.get("target_map_id", -1)) == target_map_id
            ):
                if trusted_entry:
                    pending["trusted_entry"] = True
                self._portal_ocr_wakeup.set()
                return True
            self._portal_ocr_pending = {
                "triggered_at": now,
                "source": str(source),
                "source_map_id": current_map_id,
                "target_map_id": target_map_id,
                "portal_name": portal_name,
                "distance": distance,
                "trusted_entry": trusted_entry,
            }

        # 这是明确的进门动作。即使是玩家手动按上，也必须解除手动地图锁，
        # 否则 OCR 正确识别新地图后仍会被旧 MapID 拒绝。
        self._manual_map_override_id = None
        self._portal_ocr_wakeup.set()
        self.log(
            f"🔎 [传送门OCR加速] {source}按下UP，门={portal_name} → {target_map_id}，"
            f"距离={distance:.1f}px；立即连续识别直到地图切换成功"
        )
        return True

    def _get_portal_ocr_pending(self) -> Optional[Dict[str, Any]]:
        with self._portal_ocr_lock:
            pending = self._portal_ocr_pending
            if pending is not None and (
                time.perf_counter() - float(pending.get("triggered_at", 0.0)) > 12.0
            ):
                self._portal_ocr_pending = None
                self._portal_ocr_wakeup.clear()
                pending = None
            return dict(pending) if pending is not None else None

    @staticmethod
    def _known_portal_target_after_title_change(
        pending: Optional[Dict[str, Any]], title_changed: bool
    ) -> Optional[int]:
        """已知普通门目标且标题确已变化时，允许跳过地图名 OCR。"""
        if not title_changed or not pending or not pending.get("trusted_entry"):
            return None
        try:
            target_map_id = int(pending.get("target_map_id", 999999999))
        except (TypeError, ValueError):
            return None
        return target_map_id if target_map_id != 999999999 else None

    def _finish_portal_ocr_acceleration(self, recognized_map_id: int) -> None:
        with self._portal_ocr_lock:
            pending = self._portal_ocr_pending
            if pending is None:
                return
            self._portal_ocr_pending = None
        self._portal_ocr_wakeup.clear()
        elapsed = max(0.0, time.perf_counter() - float(pending.get("triggered_at", 0.0)))
        self.log(
            f"✅ [传送门OCR加速] 已识别地图{recognized_map_id}（{elapsed:.2f}s），"
            "恢复标题变化触发/每8秒兜底的默认时序"
        )

    def _get_effective_patrol_platforms(self) -> List[int]:
        fallback = self._get_parsed_patrol_platforms()
        controller = getattr(self, "world_patrol_controller", None)
        if controller is not None and controller.running:
            return controller.current_patrol(fallback)
        if bool(self.config.get("cross_map_patrol_enabled", False)):
            current_map = self._get_current_map_id()
            for stop in getattr(self, "cached_world_patrol_stops", ()):
                if stop.map_id == current_map:
                    return list(stop.platforms)
        return fallback

    def _get_effective_patrol_dwell_range(self) -> Tuple[float, float]:
        fallback = self._get_patrol_dwell_range()
        controller = getattr(self, "world_patrol_controller", None)
        return controller.current_dwell_range(fallback) if controller is not None else fallback

    def _get_effective_patrol_positions(self) -> List[float]:
        fallback = self._get_single_patrol_positions()
        controller = getattr(self, "world_patrol_controller", None)
        return controller.current_positions(fallback) if controller is not None else fallback

    def _tick_world_patrol(self) -> bool:
        controller = getattr(self, "world_patrol_controller", None)
        return bool(controller.tick()) if controller is not None else False

    def _on_world_patrol_target_completed(self, platform_id: int) -> None:
        controller = getattr(self, "world_patrol_controller", None)
        if controller is not None:
            controller.on_target_completed(platform_id)

    def _on_global_rest_completed(self, platform_id: int) -> None:
        controller = getattr(self, "world_patrol_controller", None)
        if controller is not None:
            controller.on_rest_completed(platform_id)

    def _on_world_patrol_recovery_completed(self) -> None:
        """异常换图返程完成后丢弃旧地图遗留的边与失败计数。"""
        if getattr(self, "combat_fsm", None) is not None:
            self.combat_fsm.reset_navigation_failures()

    def _prepare_world_map_transition(self, expected_map_id: int) -> None:
        # 跨图运行期间必须解除用户手动 MapID 锁，否则 OCR 会正确识别出
        # 新地图却被哨兵丢弃，控制器最终只能得到“进门超时”。
        if self._manual_map_override_id is not None:
            self.log(
                f"🔓 [跨图地图识别] 解除手动MapID锁定，等待进入{expected_map_id}"
            )
        self._manual_map_override_id = None

    def _reset_tracking_after_world_transition(self) -> None:
        """入口光圈阶段的误识别不得延续到新图战斗判定。"""
        detector = getattr(self, "detector", None)
        if detector is not None:
            detector.reset_player_tracking(clear_monsters=True)
        with self._monster_batch_lock:
            self._monster_latest_batch = None
        with self.lock:
            self.latest_result = MainViewResult(timestamp=time.perf_counter())

    def _get_fresh_arrival_player_observation(
        self,
    ) -> Optional[Tuple[float, float, float]]:
        detector = getattr(self, "detector", None)
        # 售票处没有小地图；常规全屏人物框与名字牌兜底交替命中时，
        # 坐标源切换会被下楼闭环判成无法恢复的 X 突跳。此图始终优先
        # 使用原尺寸名字牌，只有它暂时不可见时才尝试常规人物框。
        if self._get_current_map_id() == 103000100:
            nametag = self._get_ticket_booth_nametag_observation(detector)
            if nametag is not None:
                return nametag
        if detector is not None and detector.last_player_pos is not None:
            observed_at = float(getattr(detector, "last_player_observed_time", 0.0) or 0.0)
            if observed_at > 0.0 and (time.perf_counter() - observed_at) <= 0.25:
                x, y = detector.last_player_pos
                return float(x), float(y), observed_at
        return None

    def _get_ticket_booth_nametag_observation(
        self, detector
    ) -> Optional[Tuple[float, float, float]]:
        now = time.perf_counter()
        cached = getattr(self, "_booth_nametag_observation", None)
        last_attempt = float(getattr(self, "_booth_nametag_last_attempt", 0.0))
        if now - last_attempt < 0.08:
            return cached if cached is not None and now - cached[2] <= 0.25 else None
        self._booth_nametag_last_attempt = now
        self._booth_nametag_observation = None
        graph = getattr(self, "platform_graph", None)
        template = (
            getattr(detector, "player_templates_gray", {}).get("nametag")
            if detector is not None else None
        )
        if graph is None or int(getattr(graph, "map_id", -1)) != 103000100 or template is None:
            return None
        frame = self.capture.capture_frame(copy=False, include_overlay=False) if self.capture else None
        if frame is None:
            return None
        frame_seq = getattr(self.capture, "frame_seq", None)
        if frame_seq is not None:
            previous_seq = getattr(self, "_booth_nametag_frame_seq", None)
            if previous_seq == frame_seq:
                return cached if cached is not None and now - cached[2] <= 0.25 else None
            self._booth_nametag_frame_seq = frame_seq
        fh, fw = frame.shape[:2]
        bounds = graph.vr_bounds or {}
        world_w = int(bounds.get("width", 0) or 0)
        world_h = int(bounds.get("height", 0) or 0)
        if world_w <= 0 or world_h <= 0 or fw < world_w or fh < world_h:
            return None
        left = max(0, int(round((fw - world_w) / 2.0)))
        top = max(0, int(round((fh - world_h) / 2.0)))
        th, tw = template.shape[:2]
        # 人物走到地图左缘时，名字牌会有一半伸进两侧黑边。
        # 搜索区按模板大小外扩，否则会在售票处下楼梯途中失明。
        crop_left = max(0, left - tw)
        crop_top = max(0, top - th)
        roi = frame[
            crop_top:min(fh, top + world_h + th),
            crop_left:min(fw, left + world_w + tw),
        ]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, best_score, _, (mx, my) = cv2.minMaxLoc(scores)
        # 入口光圈实时改变名牌背后的像素；实机静止样本的原尺寸分数
        # 在 0.564~0.577 间波动。由次高候选差距与 FSM 三帧稳定
        # 共同把关，不能用 0.60 在此处漏掉真人。
        if best_score < 0.55:
            return None
        # 同一名牌附近的相邻峰不算第二候选；若别处也有相似 UI，
        # 差距不足 0.12 就拒绝定位，避免朝错人走。
        scores[max(0, my - th):my + th, max(0, mx - tw):mx + tw] = -1.0
        _, second_score, _, _ = cv2.minMaxLoc(scores)
        if best_score - second_score < 0.12:
            return None
        feet_offset = 0
        for name, _bgr, _gray, _off_x, off_y in getattr(detector, "player_parts", ()):
            if name == "nametag_full":
                feet_offset = int(off_y)
                break
        observed = (
            float(crop_left + mx + tw / 2.0),
            float(crop_top + my + feet_offset),
            time.perf_counter(),
        )
        self._booth_nametag_observation = observed
        last_log = float(getattr(self, "_booth_nametag_last_log", 0.0))
        if now - last_log >= 5.0:
            self._booth_nametag_last_log = now
            self.log(
                f"🔎 [售票处人物兜底] 名字牌={best_score:.3f}，"
                f"次高={second_score:.3f}，画面脚底=({observed[0]:.1f},{observed[1]:.1f})"
            )
        return observed

    def _get_current_game_frame_size(self) -> Optional[Tuple[int, int]]:
        frame = self.capture.capture_frame(copy=False) if self.capture else None
        return (int(frame.shape[1]), int(frame.shape[0])) if frame is not None else None

    def _arm_known_script_portal_ocr(self, edge) -> None:
        """脚本门原始 toMap 为占位值；用已确认的落点加速切图。"""
        with self._portal_ocr_lock:
            self._portal_ocr_pending = {
                "triggered_at": time.perf_counter(),
                "source": "F6脚本门",
                "source_map_id": int(edge.source_map_id),
                "target_map_id": int(edge.target_map_id),
                "portal_name": str(edge.portal_name),
                "distance": 0.0,
                "trusted_entry": True,
            }
        self._manual_map_override_id = None
        self._portal_ocr_wakeup.set()
        self.log(
            f"🔎 [脚本门OCR加速] {edge.source_map_id}:{edge.portal_name}"
            f" → {edge.target_map_id}，等待标题区域切换"
        )

    def _get_patrol_position_random(self) -> float:
        fallback = float(getattr(self, "cached_patrol_position_random", 0.0))
        controller = getattr(self, "world_patrol_controller", None)
        return (
            controller.current_position_random(fallback)
            if controller is not None else fallback
        )

    def _get_single_rest_settings(self) -> Optional[Dict[str, Any]]:
        pid = self.config.get("rest_platform_id")
        if pid is None or str(pid).strip() == "":
            return None
        try:
            platform_id = int(pid)
        except (TypeError, ValueError):
            return None
        return {
            "map_id": self._get_current_map_id(),
            "platform_id": platform_id,
            "duration_min_sec": float(self.config.get("rest_duration_min_sec", 30.0)),
            "duration_max_sec": float(self.config.get("rest_duration_max_sec", 60.0)),
            "interval_min_sec": float(self.config.get("rest_interval_min_sec", 30.0)),
            "interval_max_sec": float(self.config.get("rest_interval_max_sec", 40.0)),
        }

    def _get_effective_rest_settings(self) -> Optional[Dict[str, Any]]:
        fallback = self._get_single_rest_settings()
        controller = getattr(self, "world_patrol_controller", None)
        if controller is not None and controller.running:
            return controller.current_rest_settings(None)
        if bool(self.config.get("cross_map_patrol_enabled", False)):
            # 多地图休息由全局 F6 计时器独占；旧的逐地图本地计时器
            # 即使在过渡/重连阶段也不能重新启动。
            return None
        return fallback

    def _run_single_navigation_step(self):
        """由 UI 触发一次、且仅一次的拓扑跨层动作。"""
        if not getattr(self, "combat_fsm", None):
            self.log("⚠️ [单步跨层] 控制器尚未就绪。")
            return
        self._set_coordinate_trace_enabled(True)
        if not self.combat_fsm.run_single_navigation_step():
            self._set_coordinate_trace_enabled(False)

    def _set_coordinate_trace_enabled(self, enabled: bool):
        self._coordinate_trace_enabled = bool(enabled)

    def _get_live_world_pos(self) -> Optional[Tuple[int, int]]:
        """高精度实时获取角色物理世界坐标 (经过 WZ Canvas 视口特征匹配 + 平台物理吸附，消除卷轴漂移)"""
        if self.current_player_world_pos is not None:
            return self.current_player_world_pos
        # 若主线程未更新，尝试主动捕获一帧感知
        try:
            if self.platform_graph and self.capture:
                frame = self.capture.capture_frame(copy=False)
                if frame is not None:
                    # 与后台唯一坐标源一致，不能让旧 HUD tracker 再次参与
                    # Canvas 卷轴匹配并覆盖 raw_tracker 的结果。
                    tr = self.raw_tracker.detect(frame)
                    if tr.is_detected and tr.norm_pos:
                        crop_gray = None
                        if tr.inner_box:
                            bx, by, bw, bh = tr.inner_box
                            sub = frame[by:by+bh, bx:bx+bw]
                            if sub.size > 0:
                                crop_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if len(sub.shape) == 3 else sub
                        raw_wx, raw_wy = self.platform_graph.minimap_norm_to_world(*tr.norm_pos, crop_gray_frame=crop_gray)
                        self._update_motion_measurement_step(self.platform_graph, tr.inner_box)
                        observation_t = time.perf_counter()
                        self.horizontal_motion.correct_measurement(raw_wx, timestamp=observation_t)
                        kalman_x = self.horizontal_kalman.correct_measurement(
                            raw_wx, timestamp=observation_t
                        )
                        preferred_platform_id = getattr(
                            self.current_player_platform, "id", None
                        )
                        snapped_wx, snapped_wy = self.platform_graph.get_snapped_player_world_pos(
                            int(round(kalman_x)),
                            raw_wy,
                            preferred_platform_id=preferred_platform_id,
                        )
                        with self._raw_world_lock:
                            self.current_player_raw_world_pos = (raw_wx, raw_wy)
                        self.current_player_world_pos = (snapped_wx, snapped_wy)
                        return self.current_player_world_pos
        except Exception:
            pass
        return self.current_player_world_pos

    def _update_motion_measurement_step(self, graph, inner_box) -> None:
        """同步当前地图 1px 黄点对应的世界坐标距离。

        必须与 ``PlatformGraph.minimap_norm_to_world`` 使用同一比例：
        ``miniMap.width / 完整WZ Canvas宽度``。滚动小地图的可视框只是一扇
        窗口，不能拿它代替完整 Canvas，否则量化区间会随视口宽度被放大。
        """
        try:
            box_w = float(inner_box[2]) if inner_box is not None else 0.0
            box_h = float(inner_box[3]) if inner_box is not None else 0.0
            map_w = float((getattr(graph, "minimap_meta", None) or {}).get("width", 0))
            map_h = float((getattr(graph, "minimap_meta", None) or {}).get("height", 0))
            canvas_size = graph.get_minimap_canvas_size()
            canvas_w = float(canvas_size[0]) if canvas_size else box_w
            canvas_h = float(canvas_size[1]) if canvas_size else box_h
            if canvas_w > 0.0 and map_w > 0.0:
                scale_x = map_w / canvas_w
                self.horizontal_motion.set_measurement_step(scale_x)
                self.horizontal_kalman.set_measurement_step(scale_x)
                scale_y = (
                    map_h / canvas_h
                    if canvas_h > 0.0 and map_h > 0.0
                    else scale_x
                )
                self._minimap_world_per_pixel = (scale_x, scale_y)
        except Exception:
            pass

    def _append_model_debug(self, raw_x, raw_y, predicted_x, topology_pos, source="unknown"):
        fp = getattr(self, "_model_debug_fp", None)
        if fp is None:
            return
        try:
            topo_x, topo_y = topology_pos if topology_pos else ("", "")
            kalman_state = self.horizontal_kalman.snapshot()
            kalman_x = "" if kalman_state["x"] is None else kalman_state["x"]
            with self._model_debug_lock:
                fp.write(
                    f"{time.time():.3f},{raw_x},{raw_y},{predicted_x},{kalman_x},"
                    f"{topo_x},{topo_y},"
                    f"{getattr(self.horizontal_motion, 'direction', '')},"
                    f"{getattr(self.horizontal_motion, 'vx', '')},"
                    f"{getattr(self.horizontal_motion, 'blocked', '')},"
                    f"{kalman_state['direction']},{kalman_state['vx']},"
                    f"{kalman_state['position_variance']},{kalman_state['innovation']},"
                    f"{kalman_state['gain_x']},{kalman_state['reanchored']},{source}\n"
                )
        except Exception:
            pass

    def _get_live_raw_world_pos(self) -> Optional[Tuple[int, int]]:
        with self._raw_world_lock:
            return self.current_player_raw_world_pos

    def _reset_motion_prediction(self) -> None:
        """跳抓失败后清除速度，但保留 Kalman 已估计出的格内 X。"""
        try:
            with self._raw_world_lock:
                raw = self.current_player_raw_world_pos
            if raw is not None:
                self.horizontal_motion.explicit_reanchor(raw[0], reset_velocity=True)
            self.horizontal_kalman.reset_velocity()
        except Exception:
            pass

    def _append_coordinate_trace(
        self,
        track_res,
        raw_pos: Tuple[int, int],
        snapped_pos: Tuple[int, int],
        conversion_debug: Optional[Dict[str, Any]] = None,
    ):
        """每张新截图写入小地图黄点与世界坐标换算链，供卷轴定位复盘。"""
        fp = getattr(self, "_coordinate_trace_fp", None)
        if fp is None or not getattr(self, "_coordinate_trace_enabled", False):
            return
        try:
            pixel = track_res.pixel_pos or (None, None)
            norm = track_res.norm_pos or (None, None)
            inner = track_res.inner_box or (None, None, None, None)
            debug = conversion_debug or {}
            with self._coordinate_trace_lock:
                fp.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{time.perf_counter():.6f},"
                    f"{pixel[0]},{pixel[1]},{norm[0]},{norm[1]},"
                    f"{raw_pos[0]},{raw_pos[1]},{snapped_pos[0]},{snapped_pos[1]},"
                    f"{inner[0]},{inner[1]},{inner[2]},{inner[3]},"
                    f"{debug.get('offset_x')},{debug.get('offset_y')},"
                    f"{debug.get('match_score')},{debug.get('match_reason')},"
                    f"{debug.get('candidate_count')}\n"
                )
        except Exception:
            pass

    def _on_patrol_platforms_changed(self, event=None):
        try:
            val = self.ent_patrol_platforms.get().strip()
            self.cached_patrol_platforms = self._parse_patrol_string(val)
            self.config["patrol_platforms"] = val
            self._save_config()
        except Exception:
            pass

    def _get_parsed_patrol_platforms(self) -> List[int]:
        """线程安全读取解析后的平台列表"""
        if hasattr(self, "cached_patrol_platforms") and self.cached_patrol_platforms:
            return list(self.cached_patrol_platforms)
        raw = self.config.get("patrol_platforms", "25, 26")
        return self._parse_patrol_string(raw)

    def _insert_current_platform_to_patrol(self, card_data: Optional[Dict[str, Any]] = None):
        """把当前长平台加入输入草稿；点击应用后才正式生效。"""
        if not self.current_player_platform:
            messagebox.showinfo("提示", "未检测到角色当前所在的平台，请确保角色处于游戏画面内！", parent=self.root)
            return
        if card_data is None:
            cards = getattr(self, "path_map_cards", [])
            card_data = cards[0] if cards else None
        if card_data is None:
            return
        live_map_id = self._get_current_map_id()
        card_map_text = card_data["map_id_var"].get().strip()
        if (
            not card_data.get("is_current")
            and card_map_text.isdigit()
            and live_map_id is not None
            and int(card_map_text) != int(live_map_id)
        ):
            messagebox.showinfo(
                "地图不一致",
                f"角色当前在 MapID {live_map_id}，不能把当前平台插入 MapID {card_map_text}。",
                parent=self.root,
            )
            return
        curr_id = self.current_player_platform.id
        entry = card_data["platforms_entry"]
        existing = self._parse_patrol_string(entry.get().strip())
        if curr_id not in existing:
            existing.append(curr_id)
        new_text = ", ".join(str(i) for i in existing)
        entry.delete(0, tk.END)
        entry.insert(0, new_text)
        card_index = self.path_map_cards.index(card_data)
        card_label = "当前地图" if card_index == 0 else f"第{card_index + 1}张地图"
        self.log(
            f"➕ [路径设置草稿] {card_label}已加入长平台 #{curr_id}: [{new_text}]；"
            "点击“应用”后生效"
        )

    def _preview_patrol_route(self):
        """路径动作序列自动求解（同步联动日志与全景拓扑图）。"""
        if not self.platform_graph or not self.platform_graph.nodes:
            self.log("⚠️ [路线预览] 尚未加载当前地图拓扑数据，请等待或手动选图！")
            return

        platforms = self._get_effective_patrol_platforms()
        if not platforms:
            self.log("💡 [路线预览] 请输入至少一个长平台编号！")
            return

        if len(platforms) == 1:
            positions = self._get_effective_platform_patrol_positions().get(
                platforms[0], self._get_effective_patrol_positions()
            )
            random_fraction = self._get_patrol_position_random()
            low, high = self._get_effective_patrol_dwell_range()
            station_text = " -> ".join(f"{value * 100:.0f}%" for value in positions)
            self.current_route_edges = []
            self.log(
                f"📋 [单平台巡逻] P{platforms[0]}，站位={station_text}，"
                f"停留={low:.2f}～{high:.2f}s，站位随机=±{random_fraction * 100:g}%"
            )
            return

        self.log(f"📋 [路线求解] 开始规划平台序列: [{' -> '.join(str(p) for p in platforms)} -> {platforms[0]}]")
        overrides = self._get_effective_platform_patrol_positions()
        if overrides:
            summary = "; ".join(
                f"P{pid}=" + ",".join(f"{value * 100:g}%" for value in values)
                for pid, values in sorted(overrides.items())
            )
            self.log(f"📍 [平台专属站位] {summary}；其余平台沿用默认站位")

        pairs = list(zip(platforms, platforms[1:] + [platforms[0]]))
        all_edges = []
        allow_run_jump = bool(self.config.get("enable_run_jump_grab", True))
        allow_portal = not bool(self.config.get("disable_intra_map_portals", False))
        for src, dst in pairs:
            path = self.platform_graph.find_path(
                src,
                dst,
                allow_run_jump=allow_run_jump,
                allow_portal=allow_portal,
            )
            if path:
                all_edges.extend(path)
                self.log(f"  ✔ [P{src} -> P{dst}] 求解成功: {len(path)} 步 ({path[0].action if path else ''})")
            else:
                self.log(f"  ⚠️ [P{src} -> P{dst}] 未找到自动直达动作边！")

        self.current_route_edges = all_edges

        # 联动更新拓扑全景图
        if self.topology_dialog and self.topology_dialog.is_open:
            self.topology_dialog.refresh_view()
        else:
            self.log("✅ [路线规划就绪] 求解完成！点击【🗺️ 世界地图】查看高亮路径。")

    def open_topology_dialog(self):
        """弹出平台与梯绳拓扑全景大图"""
        if not self.platform_graph or not self.platform_graph.nodes:
            messagebox.showinfo("提示", "请等待地图拓扑解析完成！", parent=self.root)
            return

        def get_graph():
            return self.platform_graph

        def get_player_pos():
            # 拓扑图是显示端，绝不能自行检测黄点或调用卷轴匹配：这会与
            # raw_tracker_worker 竞争同一张 Canvas 的可变卷轴偏移，并可能在
            # 同图传送后把旧 tracker 的模板结果写回全局 YOU 坐标。
            # 唯一坐标源是后台 60Hz raw_tracker_worker；这里仅作线程安全读取。
            with self._raw_world_lock:
                pos = self.current_player_world_pos
            if pos is not None:
                return pos
            if self.detector and self.detector.last_player_pos:
                return self.detector.last_player_pos
            return None

        def get_patrol():
            return self._get_effective_patrol_platforms()

        def get_active_path():
            return getattr(self, "current_route_edges", [])

        if self.topology_dialog is None or not self.topology_dialog.is_open:
            self.topology_dialog = PlatformTopologyDialog(
                self.root, get_graph, get_player_pos, get_patrol, get_active_path,
                on_merge_platforms_changed=self._on_merge_platforms_changed,
                on_calibrate_x=self._calibrate_x_from_rope,
                on_calibrate_y=self._calibrate_y_from_platform,
                on_delete_calibration=self._delete_current_map_calibrations,
                initial_merged=self.merge_short_platforms,
            )
        else:
            self.topology_dialog.top.lift()
            self.topology_dialog.refresh_view()

    def _calibrate_x_from_rope(self, rope_id: int) -> Tuple[bool, str]:
        """使用当前黄点与 JSON 绳梯 X 坐标，追加一条 X 轴卷轴标定样本。"""
        graph = self.platform_graph
        rope = graph.get_ladder_rope(int(rope_id)) if graph else None
        if graph is None or rope is None:
            return False, f"当前地图中找不到绳梯 #{rope_id}。"

        try:
            frame = self.capture.capture_frame(copy=False) if self.capture else None
            if frame is None:
                return False, "暂时无法读取游戏画面。"
            tr = self.tracker.detect(frame)
            if not tr.is_detected or not tr.norm_pos or not tr.inner_box:
                return False, "未检测到当前角色黄点，请确保角色正在该绳梯上。"

            bx, by, bw, bh = tr.inner_box
            sub = frame[by:by+bh, bx:bx+bw]
            if sub.size == 0:
                return False, "小地图视口读取失败。"
            crop_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if len(sub.shape) == 3 else sub
            conversion_debug: Dict[str, Any] = {}
            graph.minimap_norm_to_world(
                *tr.norm_pos, crop_gray_frame=crop_gray, debug_out=conversion_debug
            )

            canvas_w = float(conversion_debug.get("canvas_w", 0) or 0)
            world_w = float((graph.minimap_meta or {}).get("width", 0) or 0)
            center_x = float((graph.minimap_meta or {}).get("centerX", 0) or 0)
            if canvas_w <= 0 or world_w <= 0:
                return False, "当前地图没有有效的 Canvas 宽度数据。"

            # Tracker 的 norm_pos 按屏幕内框归一化，先恢复黄点的原始视口像素。
            pixel_x = float(tr.norm_pos[0]) * float(bw)
            world_scale_x = world_w / canvas_w
            expected_offset = (float(rope.x) + center_x) / world_scale_x - pixel_x
            measured_offset = float(conversion_debug.get("offset_x", 0.0) or 0.0)
            sample_delta = expected_offset - measured_offset
            current_delta = graph.add_x_calibration_sample(sample_delta)

            # 标定改变了坐标基准，立即重锚定水平运动模型；否则它会继续
            # 沿用标定前的 predicted X，直到重启程序才表现出修正结果。
            corrected_raw = graph.minimap_norm_to_world(
                *tr.norm_pos, crop_gray_frame=crop_gray
            )
            self.horizontal_motion.explicit_reanchor(corrected_raw[0], reset_velocity=True)
            self.horizontal_kalman.explicit_reanchor(corrected_raw[0], reset_velocity=True)
            self.current_player_raw_world_pos = corrected_raw
            self.current_player_world_pos = graph.get_snapped_player_world_pos(*corrected_raw)

            self.log(
                f"📐 [X轴标定] 绳梯#{rope_id} X={rope.x}，样本修正={sample_delta:.2f}px，"
                f"当前中位数修正={current_delta:.2f}px"
            )
            return True, (
                f"已记录绳梯 #{rope_id}（世界 X={rope.x}）。\n"
                f"本次修正：{sample_delta:.2f} Canvas px\n"
                f"当前累计中位数修正：{current_delta:.2f} Canvas px"
            )
        except Exception as exc:
            return False, f"标定过程中发生错误：{exc}"

    def _calibrate_y_from_platform(self, platform_id: int) -> Tuple[bool, str]:
        """使用当前黄点与水平平台承重面，追加一条 Y 轴标定样本。"""
        graph = self.platform_graph
        platform = graph.get_node(int(platform_id)) if graph else None
        if graph is None or platform is None:
            return False, f"当前地图中找不到平台 #{platform_id}。"
        try:
            frame = self.capture.capture_frame(copy=False) if self.capture else None
            if frame is None:
                return False, "暂时无法读取游戏画面。"
            tr = self.tracker.detect(frame)
            if not tr.is_detected or not tr.norm_pos or not tr.inner_box:
                return False, "未检测到当前角色黄点，请确保角色正站在该平台上。"

            bx, by, bw, bh = tr.inner_box
            sub = frame[by:by+bh, bx:bx+bw]
            if sub.size == 0:
                return False, "小地图视口读取失败。"
            crop_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if len(sub.shape) == 3 else sub
            conversion_debug: Dict[str, Any] = {}
            graph.minimap_norm_to_world(
                *tr.norm_pos, crop_gray_frame=crop_gray, debug_out=conversion_debug
            )

            canvas_h = float(conversion_debug.get("canvas_h", 0) or 0)
            world_h = float((graph.minimap_meta or {}).get("height", 0) or 0)
            center_y = float((graph.minimap_meta or {}).get("centerY", 0) or 0)
            if canvas_h <= 0 or world_h <= 0:
                return False, "当前地图没有有效的 Canvas 高度数据。"

            # 小地图黄点对应角色重心；站立时它在承重平台表面约 45px 上方。
            target_player_y = float(platform.y - 45)
            pixel_y = float(tr.norm_pos[1]) * float(bh)
            world_scale_y = world_h / canvas_h
            expected_offset = (target_player_y + center_y) / world_scale_y - pixel_y
            measured_offset = float(conversion_debug.get("offset_y", 0.0) or 0.0)
            sample_delta = expected_offset - measured_offset
            current_delta = graph.add_y_calibration_sample(sample_delta)
            corrected_raw = graph.minimap_norm_to_world(
                *tr.norm_pos, crop_gray_frame=crop_gray
            )
            self.horizontal_motion.explicit_reanchor(corrected_raw[0], reset_velocity=True)
            self.horizontal_kalman.explicit_reanchor(corrected_raw[0], reset_velocity=True)
            self.current_player_raw_world_pos = corrected_raw
            self.current_player_world_pos = graph.get_snapped_player_world_pos(*corrected_raw)
            self.log(
                f"📐 [Y轴标定] 平台#{platform_id} 承重Y={platform.y}，样本修正={sample_delta:.2f}px，"
                f"当前中位数修正={current_delta:.2f}px"
            )
            return True, (
                f"已记录水平平台 #{platform_id}（承重 Y={platform.y}）。\n"
                f"本次修正：{sample_delta:.2f} Canvas px\n"
                f"当前累计中位数修正：{current_delta:.2f} Canvas px"
            )
        except Exception as exc:
            return False, f"标定过程中发生错误：{exc}"

    def _delete_current_map_calibrations(self) -> Tuple[bool, str]:
        """清除当前内存标定，并兼容删除旧版本遗留的持久化数据。"""
        graph = self.platform_graph
        if graph is None:
            return False, "当前地图拓扑尚未载入。"
        cal_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "assets", "minimap_x_calibrations.json"
        )
        try:
            saved: Dict[str, Any] = {}
            if os.path.exists(cal_path):
                with open(cal_path, "r", encoding="utf-8") as fp:
                    loaded = json.load(fp)
                if isinstance(loaded, dict):
                    saved = loaded
            removed = saved.pop(str(graph.map_id), None)
            # 即使文件中没有数据，也必须清空当前图已加载到内存的修正。
            graph.x_calibration_samples_px = []
            graph.y_calibration_samples_px = []
            if os.path.exists(cal_path) or removed is not None:
                with open(cal_path, "w", encoding="utf-8") as fp:
                    json.dump(saved, fp, ensure_ascii=False, indent=2)
            self.log(
                f"🗑️ [标定删除] 已清除 MapID={graph.map_id} 的本次 X/Y 标定"
                "及旧版遗留数据"
            )
            return True, (
                f"已清除当前地图（{graph.map_name or graph.map_id}，"
                f"MapID={graph.map_id}）的本次标定。"
            )
        except Exception as exc:
            return False, f"无法删除当前地图标定：{exc}"

    def _topology_cache_path(
        self,
        map_id: str,
        merge_short_platforms: bool,
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
    ) -> str:
        """返回当前地图/拓扑模式对应的持久化拓扑缓存路径。"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_dir = os.path.join(project_root, "assets", "topology_cache")
        mode = "merged" if merge_short_platforms else "raw"
        tp_tag = f"_tp{int(teleport_distance_px)}" if enable_teleport else ""
        return os.path.join(cache_dir, f"{map_id}_{mode}{tp_tag}.json")

    def _topology_cache_signature(self, map_id: str) -> Dict[str, Optional[int]]:
        """取得缓存依赖文件的签名，源文件变化后自动重建拓扑。"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        source_path = os.path.join(project_root, "data", "maps", f"{map_id}.json")

        def mtime_ns(path: str) -> Optional[int]:
            try:
                return int(os.stat(path).st_mtime_ns)
            except OSError:
                return None

        signature = {
            "map_json_mtime_ns": mtime_ns(source_path),
            # 拓扑边的生成规则会随代码更新；仅比较 WZ/JSON 时间戳
            # 会把旧的“无路径”图永久从磁盘恢复出来。
            "graph_builder_mtime_ns": mtime_ns(
                os.path.join(project_root, "src", "engine", "platform_graph.py")
            ),
        }
        reader = getattr(self, "wz_map_reader", None)
        if reader is not None and str(map_id).isdigit():
            signature.update(reader.source_signature(int(map_id)))
        return signature

    def _load_persisted_topology_cache(
        self,
        map_id: str,
        merge_short_platforms: bool,
        signature: Dict[str, Optional[int]],
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
    ) -> Optional[PlatformGraph]:
        """读取并校验落盘拓扑缓存；损坏或过期时返回 None。"""
        path = self._topology_cache_path(map_id, merge_short_platforms, enable_teleport, teleport_distance_px)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if (
                payload.get("cache_version") != 3
                or str(payload.get("map_id")) != str(map_id)
                or bool(payload.get("merge_short_platforms")) != bool(merge_short_platforms)
                or bool(payload.get("enable_teleport", False)) != bool(enable_teleport)
                or payload.get("source_signature") != signature
            ):
                return None
            return PlatformGraph.from_cache_dict(payload.get("graph", {}))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            return None

    def _save_persisted_topology_cache(
        self,
        map_id: str,
        merge_short_platforms: bool,
        signature: Dict[str, Optional[int]],
        graph: PlatformGraph,
        enable_teleport: bool = False,
        teleport_distance_px: float = 150.0,
    ) -> None:
        """将拓扑解析结果安全写入 JSON，供下次启动直接恢复。"""
        path = self._topology_cache_path(map_id, merge_short_platforms, enable_teleport, teleport_distance_px)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp_path = f"{path}.tmp"
            payload = {
                # v3 invalidates graphs when edge-building code changes.
                "cache_version": 3,
                "map_id": str(map_id),
                "merge_short_platforms": bool(merge_short_platforms),
                "enable_teleport": bool(enable_teleport),
                "teleport_distance_px": float(teleport_distance_px),
                "source_signature": signature,
                "graph": graph.to_cache_dict(),
            }
            with open(temp_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
        except (OSError, TypeError, ValueError) as exc:
            self.log(f"⚠️ [拓扑缓存] 保存失败：{exc}")

    def _on_merge_platforms_changed(self, enabled: bool):
        """切换拓扑短平台合并模式，并同步重建导航图。"""
        self.merge_short_platforms = bool(enabled)
        if getattr(self, "world_route_planner", None) is not None:
            self.world_route_planner.set_merge_short_platforms(enabled)
        self.config["merge_short_platforms"] = bool(enabled)
        self._save_config()
        map_data = self._current_graph_map_data
        if not map_data:
            self.log("⚠️ [拓扑设置] 当前地图原始数据尚未准备好，请稍后重试")
            return
        target_id = str((self.current_map_info or {}).get("map_id", ""))
        cache_key = (target_id, bool(enabled))
        cached_graph = self._platform_graph_cache.get(cache_key)
        if cached_graph is not None:
            self.platform_graph = cached_graph
            self._sync_tracker_minimap_canvas_size(cached_graph)
            self.merge_short_platforms = bool(enabled)
            self._preview_patrol_route()
            if self.topology_dialog and self.topology_dialog.is_open:
                self.topology_dialog.refresh_view(force_rebuild=True)
            self.log(f"⚡ [拓扑设置] 已从缓存切换：{'合并短平台' if enabled else '保留短平台'}")
            return
        self.platform_graph = None
        if self.topology_dialog and self.topology_dialog.is_open:
            self.topology_dialog.base_bgr_img = None
            self.topology_dialog.last_map_id = None
            self.topology_dialog.lbl_info.config(text="⏳ 正在重建平台拓扑...", fg="#ffb74d")

        def worker():
            try:
                graph = PlatformGraphBuilder.build_from_map_dict(
                    map_data, merge_short_platforms=self.merge_short_platforms
                )
                if str(graph.map_id) != target_id:
                    return
                self._platform_graph_cache[cache_key] = graph
                self.platform_graph = graph
                self._sync_tracker_minimap_canvas_size(graph)
                self.root.after(0, self._preview_patrol_route)
                if self.topology_dialog and self.topology_dialog.is_open:
                    self.root.after(0, lambda: self.topology_dialog.refresh_view(force_rebuild=True))
                mode = "合并短平台" if self.merge_short_platforms else "保留短平台"
                self.log(f"✅ [拓扑设置] 已切换为：{mode}（平台数 {len(graph.nodes)}）")
            except Exception as exc:
                self.log(f"❌ [拓扑设置] 重建失败：{exc}")
        threading.Thread(target=worker, daemon=True).start()

    def _refresh_platform_graph_teleport(self):
        """当瞬移开关或距离变更时，刷新当前地图的平台拓扑图边。"""
        map_data = getattr(self, "_current_graph_map_data", None)
        target_id = str((self.current_map_info or {}).get("map_id", ""))
        if not map_data or not target_id:
            return
        enabled = bool(self.config.get("enable_teleport", False))
        tp_dist = float(self.config.get("teleport_distance_px", 150.0))
        cache_key = (target_id, bool(self.merge_short_platforms), enabled, tp_dist)
        cached_graph = self._platform_graph_cache.get(cache_key)
        if cached_graph is not None:
            self.platform_graph = cached_graph
            self._sync_tracker_minimap_canvas_size(cached_graph)
            self._preview_patrol_route()
            if self.topology_dialog and self.topology_dialog.is_open:
                self.topology_dialog.refresh_view(force_rebuild=True)
            self.log(f"⚡ [瞬移拓扑] 已应用缓存拓扑（瞬移={'开启' if enabled else '关闭'}，距离={tp_dist:g}px）")
            return

        def worker():
            try:
                graph = PlatformGraphBuilder.build_from_map_dict(
                    map_data,
                    merge_short_platforms=self.merge_short_platforms,
                    enable_teleport=enabled,
                    teleport_distance_px=tp_dist,
                )
                if str(graph.map_id) != target_id:
                    return
                self._platform_graph_cache[cache_key] = graph
                self.platform_graph = graph
                self._sync_tracker_minimap_canvas_size(graph)
                self.root.after(0, self._preview_patrol_route)
                if self.topology_dialog and self.topology_dialog.is_open:
                    self.root.after(0, lambda: self.topology_dialog.refresh_view(force_rebuild=True))
                self.log(f"✅ [瞬移拓扑] 拓扑图已更新（瞬移={'开启' if enabled else '关闭'}，距离={tp_dist:g}px）")
            except Exception as exc:
                self.log(f"⚠️ [瞬移拓扑] 更新拓扑失败：{exc}")
        threading.Thread(target=worker, daemon=True).start()


    def _sync_tracker_minimap_canvas_size(self, graph):
        """把 WZ Canvas 尺寸和背景同步给所有黄点检测入口。"""
        if graph is None:
            return
        try:
            canvas_size = graph.get_minimap_canvas_size()
        except Exception:
            canvas_size = None
        try:
            canvas_gray = graph.get_minimap_canvas_gray()
        except Exception:
            canvas_gray = None
        try:
            canvas_variants = graph.get_minimap_canvas_gray_variants()
        except Exception:
            canvas_variants = ()
        for tracker in (
            getattr(self, "tracker", None),
            getattr(self, "raw_tracker", None),
            getattr(self, "radar_tracker", None),
        ):
            if tracker is None:
                continue
            if hasattr(tracker, "set_expected_canvas_image"):
                tracker.set_expected_canvas_image(
                    canvas_gray, identity=graph.map_id,
                    image_variants=canvas_variants,
                )
            elif hasattr(tracker, "set_expected_canvas_size"):
                tracker.set_expected_canvas_size(canvas_size)

    def open_minimap_radar_dialog(self):
        """弹出左上角小地图雷达与角色黄点实时定位调试器 [F8]"""
        def get_frame():
            return self.capture.capture_frame() if self.capture else None

        def get_tracker():
            return self.radar_tracker

        def get_graph():
            return self.platform_graph

        def get_motion():
            return self.horizontal_motion

        def get_kalman():
            return self.horizontal_kalman

        def get_raw_tracker_result():
            with self._raw_tracker_result_lock:
                return self._latest_raw_tracker_result

        def on_manual_minimap_box(box):
            graph = getattr(self, "platform_graph", None)
            if graph is not None:
                # 任何框选变化都会改变框内坐标零点；原标定可能包含旧框
                # 的偏差，必须立即作废，不能继续叠加到新框。
                graph.x_calibration_samples_px = []
                graph.y_calibration_samples_px = []
            for tr in (
                getattr(self, "tracker", None),
                getattr(self, "raw_tracker", None),
                getattr(self, "radar_tracker", None),
            ):
                if tr and hasattr(tr, "set_manual_inner_box"):
                    tr.set_manual_inner_box(box)
            if box:
                self.log(f"📐 [小地图标定] 已手动锁定小地图内画布: [x={box[0]}, y={box[1]}, 宽={box[2]}, 高={box[3]}]")
            else:
                self.log("📐 [小地图标定] 已恢复小地图全自动算法探测模式。")
            self.log("♻️ [坐标标定] 小地图框已变化，本次 X/Y 标定已清空")

        if self.minimap_radar_dialog is None or not self.minimap_radar_dialog.is_open:
            self.minimap_radar_dialog = MinimapRadarDialog(
                self.root, get_frame, get_tracker, get_graph, get_motion, get_kalman,
                self._get_live_raw_world_pos, get_raw_tracker_result,
                on_manual_box_callback=on_manual_minimap_box
            )
        else:
            self.minimap_radar_dialog.top.lift()
            self.minimap_radar_dialog.refresh_single_frame()

    def open_ladder_grab_test_dialog(self):
        """配置并启动独立的“单次走位 + 跳抓”压力测试。"""
        if self._ladder_test_dialog is not None:
            try:
                if self._ladder_test_dialog.winfo_exists():
                    self._ladder_test_dialog.lift()
                    return
            except Exception:
                pass

        graph = self.platform_graph
        if graph is None or not graph.nodes:
            messagebox.showwarning("跳抓测试", "当前地图拓扑尚未加载完成，请稍后再试。", parent=self.root)
            return

        current = self._get_live_raw_world_pos()
        current_platform = self.current_player_platform
        default_platform = getattr(current_platform, "id", None)
        if default_platform is None:
            default_platform = min(graph.nodes) if graph.nodes else 1
        platform_node = graph.get_node(default_platform)
        default_x = float(current[0]) if current else float((platform_node.x_min + platform_node.x_max) / 2)
        ladder_ids = [
            lr.id for lr in graph.ladder_ropes.values()
            if lr.bottom_platform_id == default_platform
        ]
        default_ladder = ladder_ids[0] if ladder_ids else 1

        win = tk.Toplevel(self.root)
        self._ladder_test_dialog = win
        win.title("🧪 绳梯跳抓稳定性测试")
        fit_window_to_work_area(win, (620, 680), (520, 480), parent=self.root)
        win.configure(bg="#1b1d23")
        win.transient(self.root)

        test_footer = tk.Frame(win, bg="#1b1d23", padx=18, pady=(6, 12))
        test_footer.pack(side=tk.BOTTOM, fill=tk.X)
        test_viewport = tk.Frame(win, bg="#1b1d23")
        test_viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        test_canvas = tk.Canvas(test_viewport, bg="#1b1d23", highlightthickness=0)
        test_scrollbar = ttk.Scrollbar(
            test_viewport, orient=tk.VERTICAL, command=test_canvas.yview
        )
        body = tk.Frame(test_canvas, bg="#1b1d23", padx=18, pady=14)
        test_body_window = test_canvas.create_window((0, 0), window=body, anchor="nw")
        test_canvas.configure(yscrollcommand=test_scrollbar.set)
        test_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        test_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        body.bind(
            "<Configure>",
            lambda _event: test_canvas.configure(scrollregion=test_canvas.bbox("all")),
        )
        test_canvas.bind(
            "<Configure>",
            lambda event: test_canvas.itemconfigure(test_body_window, width=event.width),
        )
        win.bind(
            "<MouseWheel>",
            lambda event: test_canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        tk.Label(
            body, text=f"当前地图：MapID {graph.map_id}", font=("Segoe UI", 11, "bold"),
            fg="#00e5ff", bg="#1b1d23", anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 8))
        tk.Label(
            body,
             text="每轮：F6 共享单边执行器跳抓爬顶 → F6 拓扑最短路径返回 → 回起始 X。\n"
                  "走位、起跳、抓绳、爬顶、下跳及跨台均复用 F6 逻辑；F10 可模拟受击后的坐标偏移。",
            font=("Segoe UI", 9), fg="#cfd8dc", bg="#1b1d23", justify=tk.LEFT, anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 12))

        fields = tk.Frame(body, bg="#1b1d23")
        fields.pack(fill=tk.X)
        fields.columnconfigure(1, weight=1)
        start_platform_var = tk.StringVar(value=str(default_platform))
        ladder_var = tk.StringVar(value=str(default_ladder))
        start_x_var = tk.StringVar(value=f"{default_x:.0f}")
        rounds_var = tk.StringVar(value="10")
        disturbance_x_min_var = tk.StringVar(value="10")
        disturbance_x_max_var = tk.StringVar(value="10")
        disturbance_y_min_var = tk.StringVar(value="10")
        disturbance_y_max_var = tk.StringVar(value="10")
        disturbance_duration_var = tk.StringVar(value="1.5")
        closed_loop_var = tk.BooleanVar(value=True)
        run_jump_grab_var = tk.BooleanVar(value=bool(self.config.get("enable_run_jump_grab", True)))
        for row, (label, var, hint) in enumerate((
            ("起始平台", start_platform_var, "例如 1"),
            ("目标绳梯号", ladder_var, "底端必须属于起始平台"),
            ("起始 X", start_x_var, "例如 -400"),
            ("测试轮数", rounds_var, "默认 10，范围 1–50"),
            ("X 扰动最小误差(px)", disturbance_x_min_var, "黄点 X 像素误差，默认 10"),
            ("X 扰动最大误差(px)", disturbance_x_max_var, "黄点 X 像素误差，默认 10"),
            ("Y 扰动最小误差(px)", disturbance_y_min_var, "黄点 Y 像素误差，默认 10"),
            ("Y 扰动最大误差(px)", disturbance_y_max_var, "黄点 Y 像素误差，默认 10"),
            ("扰动持续时间(s)", disturbance_duration_var, "每次 F10 持续时间，默认 1.5，范围 0.05–30"),
        )):
            tk.Label(fields, text=label + "：", font=("Segoe UI", 9), fg="#e0e0e0", bg="#1b1d23").grid(
                row=row, column=0, sticky=tk.W, pady=4
            )
            entry = tk.Entry(fields, textvariable=var, font=("Consolas", 10), bg="#111318", fg="#00e5ff",
                             insertbackground="#ffffff", relief=tk.FLAT)
            entry.grid(row=row, column=1, sticky=tk.EW, padx=(8, 8), pady=4, ipady=3)
            tk.Label(fields, text=hint, font=("Segoe UI", 8), fg="#90a4ae", bg="#1b1d23").grid(
                row=row, column=2, sticky=tk.W, pady=4
            )

        tk.Checkbutton(
            body,
            text="F6 共享闭环：使用正常巡逻的实时重规划与动作执行器（固定开启）",
            variable=closed_loop_var,
            font=("Segoe UI", 9), fg="#ffcc80", bg="#1b1d23",
            activeforeground="#ffe0b2", activebackground="#1b1d23",
            selectcolor="#1b1d23", anchor=tk.W, state=tk.DISABLED,
        ).pack(fill=tk.X, pady=(8, 2))
        tk.Checkbutton(
            body,
            text="跑跳抓绳/梯：按绳梯底端高度反推起跳准备点与空中保持方向时间",
            variable=run_jump_grab_var,
            font=("Segoe UI", 9), fg="#81d4fa", bg="#1b1d23",
            activeforeground="#b3e5fc", activebackground="#1b1d23",
            selectcolor="#1b1d23", anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 2))

        status_var = tk.StringVar(value="● 待启动")
        tk.Label(test_footer, textvariable=status_var, font=("Segoe UI", 9, "bold"),
                 fg="#80cbc4", bg="#15171c", anchor=tk.W, padx=8, pady=6).pack(fill=tk.X, pady=(0, 8))

        buttons = tk.Frame(test_footer, bg="#1b1d23")
        buttons.pack(fill=tk.X)

        def start_test():
            if self.combat_fsm.is_running:
                messagebox.showwarning("跳抓测试", "请先停止自动挂机，再启动独立跳抓测试。", parent=win)
                return
            try:
                cfg = LadderGrabTestConfig(
                    start_platform_id=int(start_platform_var.get().strip()),
                    ladder_id=int(ladder_var.get().strip()),
                    start_x=float(start_x_var.get().strip()),
                    rounds=max(1, min(50, int(rounds_var.get().strip()))),
                    closed_loop_walk=True,
                    run_jump_grab=bool(run_jump_grab_var.get()),
                    disturbance_x_min_px=max(0.0, float(disturbance_x_min_var.get().strip())),
                    disturbance_x_max_px=max(0.0, float(disturbance_x_max_var.get().strip())),
                    disturbance_y_min_px=max(0.0, float(disturbance_y_min_var.get().strip())),
                    disturbance_y_max_px=max(0.0, float(disturbance_y_max_var.get().strip())),
                    disturbance_duration_sec=float(disturbance_duration_var.get().strip()),
                )
            except ValueError:
                messagebox.showwarning(
                    "参数错误",
                    "请填写有效的平台号、绳梯号、起始 X、轮数、X/Y 扰动误差及持续时间。",
                    parent=win,
                )
                return
            ok, msg = self.ladder_grab_test_runner.start(cfg)
            if not ok:
                messagebox.showwarning("无法启动测试", msg, parent=win)
                return
            btn_start.config(state=tk.DISABLED)
            btn_stop.config(state=tk.NORMAL)
            btn_disturbance.config(state=tk.NORMAL)
            status_var.set("● 测试运行中…")
            mode = "F6共享闭环"
            grab_mode = "跑跳抓取" if cfg.run_jump_grab else "原地跳抓"
            self.log(
                f"🧪 [跳抓测试] 已启动：P{cfg.start_platform_id} -> 绳梯#{cfg.ladder_id}，"
                f"共 {cfg.rounds} 轮，走位={mode}，抓取={grab_mode}"
                f"，黄点 X 扰动={cfg.disturbance_x_min_px:g}~{cfg.disturbance_x_max_px:g}px"
                f"，Y 扰动={cfg.disturbance_y_min_px:g}~{cfg.disturbance_y_max_px:g}px"
                f"，持续={cfg.disturbance_duration_sec:g}s"
            )

        def stop_test():
            self.ladder_grab_test_runner.stop()
            status_var.set("● 正在停止测试…")
            btn_stop.config(state=tk.DISABLED)
            btn_disturbance.config(state=tk.DISABLED)

        def inject_disturbance():
            if self.ladder_grab_test_runner.inject_position_disturbance():
                status_var.set(
                    f"● 已注入随机黄点扰动（持续 "
                    f"{self.ladder_grab_test_runner.disturbance_duration_sec:g} 秒）"
                )
            else:
                status_var.set("● 测试未运行，无法注入扰动")

        btn_start = tk.Button(buttons, text="▶ 开始稳定性测试", command=start_test, font=("Segoe UI", 9, "bold"),
                              fg="#ffffff", bg="#00796b", relief=tk.FLAT, padx=12, pady=5)
        btn_start.pack(side=tk.LEFT)
        btn_stop = tk.Button(buttons, text="■ 停止测试", command=stop_test, font=("Segoe UI", 9, "bold"),
                              fg="#ffffff", bg="#b71c1c", relief=tk.FLAT, padx=12, pady=5, state=tk.DISABLED)
        btn_stop.pack(side=tk.LEFT, padx=8)
        btn_disturbance = tk.Button(
            buttons, text="⚡ 注入随机黄点扰动 [F10]", command=inject_disturbance,
            font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#8e5a00",
            activeforeground="#ffffff", activebackground="#ad7100",
            relief=tk.FLAT, padx=10, pady=5, state=tk.DISABLED,
        )
        btn_disturbance.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(buttons, text="关闭", command=win.destroy, font=("Segoe UI", 9),
                  fg="#e0e0e0", bg="#42464f", relief=tk.FLAT, padx=12, pady=5).pack(side=tk.RIGHT)

        # 测试窗口获得焦点时，也可用 F10 触发同一个扰动入口。
        win.bind("<F10>", lambda _event: inject_disturbance())

        # 回调需由后台测试线程安全地更新 Tk 控件。
        self._ladder_test_status_var = status_var
        self._ladder_test_start_btn = btn_start
        self._ladder_test_stop_btn = btn_stop
        self._ladder_test_disturbance_btn = btn_disturbance

        def on_close():
            if self.ladder_grab_test_runner.active:
                self.ladder_grab_test_runner.stop()
            self._ladder_test_dialog = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)

    def _on_ladder_grab_test_status(self, text: str, active: bool) -> None:
        """测试线程的状态回传，统一投递到 Tk 主线程。"""
        def update():
            try:
                if self._ladder_test_dialog is None or not self._ladder_test_dialog.winfo_exists():
                    return
                self._ladder_test_status_var.set("● " + text)
                self._ladder_test_start_btn.config(state=tk.DISABLED if active else tk.NORMAL)
                self._ladder_test_stop_btn.config(state=tk.NORMAL if active else tk.DISABLED)
                if getattr(self, "_ladder_test_disturbance_btn", None) is not None:
                    self._ladder_test_disturbance_btn.config(
                        state=tk.NORMAL if active else tk.DISABLED
                    )
            except Exception:
                pass
        try:
            self.root.after(0, update)
        except Exception:
            pass

    def _build_autobot_card(self):
        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf), text="⚡ 自动挂机与巡逻控制 (F6 启停 | F7 航点录制)",
                              font=("Segoe UI", 10, "bold"), fg="#00e676", bg="#202026", padx=8, pady=6)
        card.pack(fill=tk.X, padx=8, pady=3)

        # 启动/停止按钮与状态指示
        r1 = tk.Frame(card, bg="#202026")
        r1.pack(fill=tk.X, pady=2)
        
        self.btn_autobot = tk.Button(r1, text="🚀 启动自动挂机 [F6]", font=("Segoe UI", 9, "bold"),
                                     fg="#ffffff", bg="#2e7d32", relief=tk.RAISED, padx=10, pady=4,
                                     command=self.toggle_autobot)
        self.btn_autobot.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.lbl_bot_state = tk.Label(r1, text="● 状态: 停止", font=("Segoe UI", 9, "bold"),
                                      fg="#90a4ae", bg="#1a1a20", padx=8, pady=4)
        self.lbl_bot_state.pack(side=tk.RIGHT)

        # 航点控制行
        r2 = tk.Frame(card, bg="#202026")
        r2.pack(fill=tk.X, pady=(3, 1))

        self.btn_rec_route = tk.Button(r2, text="⏺️ 录制路线 [F7]", font=("Segoe UI", 8, "bold"),
                                       fg="#ffffff", bg="#4527a0", relief=tk.FLAT, padx=6, pady=2,
                                       command=self.toggle_waypoint_recording)
        self.btn_rec_route.pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(r2, text="➕ 下跳点", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=4, pady=2,
                  command=lambda: self._insert_custom_action("DOWN_JUMP")).pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(r2, text="💾 保存", font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
                  relief=tk.FLAT, padx=4, pady=2,
                  command=self._save_current_route).pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(r2, text="📂 载入", font=("Segoe UI", 8), fg="#00e5ff", bg="#2a2a32",
                  relief=tk.FLAT, padx=4, pady=2,
                  command=self._load_current_route).pack(side=tk.RIGHT)

        tk.Button(
            r2, text="🧪 跳抓10次测试", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#00695c", relief=tk.FLAT, padx=6, pady=2,
            command=self.open_ladder_grab_test_dialog,
        ).pack(side=tk.RIGHT, padx=(0, 4))

        # 跑跳抓取是针对远距离/平台外侧梯绳的可选策略；关闭后仍保留普通原地跳抓。
        self.enable_run_jump_var = tk.BooleanVar(
            value=bool(self.config.get("enable_run_jump_grab", True))
        )
        def _toggle_run_jump_grab():
            self.config["enable_run_jump_grab"] = bool(self.enable_run_jump_var.get())
            self._save_config()
            self.log(
                "🏃 [跑跳抓取] "
                + ("已启用：所有可跑的跳抓绳/梯优先连续跑跳（失败后直跳由特殊参数控制）" if self.enable_run_jump_var.get()
                   else "已关闭：全部改用普通原地跳抓")
            )
            # 开关改变后立即用同一条件重算路线预览和拓扑高亮。
            if getattr(self, "platform_graph", None) is not None:
                self._preview_patrol_route()
        tk.Checkbutton(
            card,
            text="启用跑跳抓绳/梯（失败后直跳抓绳由特殊参数控制）",
            variable=self.enable_run_jump_var,
            command=_toggle_run_jump_grab,
            font=("Segoe UI", 8),
            fg="#ffcc80",
            bg="#202026",
            activeforeground="#ffe0b2",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 0))

        self.lie_detector_auto_solve_var = tk.BooleanVar(
            value=bool(self.config.get("lie_detector_auto_solve_enabled", False))
        )
        self.lie_detector_auto_solve_checkbutton = tk.Checkbutton(
            card,
            text="识别测谎",
            variable=self.lie_detector_auto_solve_var,
            command=self._toggle_lie_detector_auto_solve,
            font=("Segoe UI", 9, "bold"),
            fg="#80cbc4",
            bg="#202026",
            activeforeground="#b2dfdb",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        )
        self.lie_detector_auto_solve_checkbutton.pack(fill=tk.X, pady=(2, 0))

        self.lie_detector_subsettings_frame = tk.Frame(card, bg="#202026")
        if self.lie_detector_auto_solve_var.get():
            self.lie_detector_subsettings_frame.pack(fill=tk.X, pady=(1, 0))

        mouse_control_row = tk.Frame(self.lie_detector_subsettings_frame, bg="#202026")
        mouse_control_row.pack(fill=tk.X, pady=(2, 0))
        self.lie_detector_mouse_control_var = tk.BooleanVar(
            value=bool(self.config.get("lie_detector_mouse_control_enabled", False))
        )
        self.lie_detector_mouse_control_var.set(
            bool(self.lie_detector_auto_solve_var.get() and self.lie_detector_mouse_control_var.get())
        )
        self.config["lie_detector_mouse_control_enabled"] = self.lie_detector_mouse_control_var.get()
        self.minigame_bridge.set_mouse_control_enabled(self.lie_detector_mouse_control_var.get())

        def _toggle_lie_detector_mouse_control():
            enabled = bool(self.lie_detector_mouse_control_var.get())
            self.config["lie_detector_mouse_control_enabled"] = enabled
            self.minigame_bridge.set_mouse_control_enabled(enabled)
            self.lie_detector_release_key_combo.configure(
                state=("readonly" if enabled else "disabled")
            )
            self._save_config()
            self.log("🖱️ [测谎小游戏] 鼠标接管" + ("已开启" if enabled else "已关闭"))

        tk.Checkbutton(
            mouse_control_row,
            text="鼠标接管",
            variable=self.lie_detector_mouse_control_var,
            command=_toggle_lie_detector_mouse_control,
            font=("Segoe UI", 8),
            fg="#b0bec5",
            bg="#202026",
            activeforeground="#ffffff",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        release_key_var = tk.StringVar(
            value=str(self.config.get("lie_detector_mouse_release_hotkey", "F12")).upper()
        )
        release_key_var.set(
            release_key_var.get()
            if release_key_var.get() in ("F1", "F2", "F3", "F4", "F5", "F12")
            else "F12"
        )
        self.config["lie_detector_mouse_release_hotkey"] = release_key_var.get()

        def _save_mouse_release_hotkey(_event=None):
            key_name = release_key_var.get().upper()
            if key_name not in ("F1", "F2", "F3", "F4", "F5", "F12"):
                key_name = "F12"
                release_key_var.set(key_name)
            self.config["lie_detector_mouse_release_hotkey"] = key_name
            self._save_config()

        tk.Label(
            mouse_control_row,
            text="关闭接管键",
            font=("Segoe UI", 8),
            fg="#b0bec5",
            bg="#202026",
        ).pack(side=tk.LEFT, padx=(10, 4))
        release_key_combo = ttk.Combobox(
            mouse_control_row,
            textvariable=release_key_var,
            values=("F1", "F2", "F3", "F4", "F5", "F12"),
            width=5,
            state="readonly",
            font=("Segoe UI", 8),
        )
        release_key_combo.pack(side=tk.LEFT)
        release_key_combo.bind("<<ComboboxSelected>>", _save_mouse_release_hotkey)
        self.lie_detector_release_key_combo = release_key_combo
        release_key_combo.configure(
            state=("readonly" if self.lie_detector_mouse_control_var.get() else "disabled")
        )

        test_video_row = tk.Frame(self.lie_detector_subsettings_frame, bg="#202026")
        test_video_row.pack(fill=tk.X, pady=(4, 0))
        self.lie_detector_test_video_var = tk.StringVar(
            value=str(self.config.get("lie_detector_test_video", ""))
        )
        video_entry = tk.Entry(
            test_video_row,
            textvariable=self.lie_detector_test_video_var,
            font=("Segoe UI", 8),
            bg="#17171c",
            fg="#eeeeee",
            insertbackground="#ffffff",
        )
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        def _save_test_video_path(path: str):
            self.lie_detector_test_video_var.set(path)
            self.config["lie_detector_test_video"] = path
            self._save_config()
            if self.minigame_video_test.session_active:
                self._stop_minigame_video_test_session("测试视频已更换")
            if self.combat_fsm.is_running and bool(self.config.get("lie_detector_test_enabled", False)):
                self._start_minigame_video_test_session()

        def _choose_test_video():
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidates = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "testVideo"),
                os.path.join(project_root, "testVideo"),
            ]
            initial_dir = next((p for p in candidates if os.path.isdir(p)), project_root)
            path = filedialog.askopenfilename(
                parent=self.root,
                title="选择测谎小游戏测试视频",
                initialdir=initial_dir,
                filetypes=[
                    ("视频文件", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("所有文件", "*.*"),
                ],
            )
            if path:
                _save_test_video_path(path)

        tk.Button(
            test_video_row,
            text="选择视频…",
            command=_choose_test_video,
            font=("Segoe UI", 8),
            fg="#ffffff",
            bg="#37474f",
            relief=tk.FLAT,
            padx=7,
        ).pack(side=tk.RIGHT)

        delay_row = tk.Frame(self.lie_detector_subsettings_frame, bg="#202026")
        delay_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(
            delay_row,
            text="F6 启动后出现时间（秒）",
            font=("Segoe UI", 8),
            fg="#b0bec5",
            bg="#202026",
        ).pack(side=tk.LEFT)
        self.lie_detector_test_delay_var = tk.StringVar(
            value=str(self.config.get("lie_detector_test_delay_sec", 30.0))
        )

        def _save_test_delay(_event=None):
            try:
                delay = max(0.0, float(self.lie_detector_test_delay_var.get()))
            except (TypeError, ValueError):
                delay = 30.0
                self.lie_detector_test_delay_var.set("30")
            self.config["lie_detector_test_delay_sec"] = delay
            self._save_config()

        delay_spinbox = tk.Spinbox(
            delay_row,
            from_=0.0,
            to=3600.0,
            increment=0.5,
            textvariable=self.lie_detector_test_delay_var,
            width=8,
            font=("Segoe UI", 8),
            command=_save_test_delay,
        )
        delay_spinbox.pack(side=tk.LEFT, padx=(8, 0))
        delay_spinbox.bind("<FocusOut>", _save_test_delay)

        self.lie_detector_test_enabled_var = tk.BooleanVar(
            value=bool(self.config.get("lie_detector_test_enabled", False))
        )
        if not self.lie_detector_auto_solve_var.get():
            self.lie_detector_test_enabled_var.set(False)
            self.config["lie_detector_test_enabled"] = False

        def _toggle_lie_detector_test():
            enabled = bool(self.lie_detector_test_enabled_var.get())
            video_path = str(self.lie_detector_test_video_var.get()).strip()
            if enabled and not os.path.isfile(video_path):
                self.lie_detector_test_enabled_var.set(False)
                messagebox.showwarning(
                    "小游戏测试",
                    "请先从 testVideo 目录选择有效的视频文件。",
                    parent=self.root,
                )
                return
            if enabled and not self.lie_detector_auto_solve_var.get():
                self.lie_detector_auto_solve_var.set(True)
                self._toggle_lie_detector_auto_solve()
            self.config["lie_detector_test_enabled"] = enabled
            self.config["lie_detector_test_video"] = video_path
            _save_test_delay()
            self._save_config()
            if enabled:
                if self.combat_fsm.is_running:
                    self._start_minigame_video_test_session()
                else:
                    self.log("🎞️ [小游戏测试] 已启用；下次启动 F6 时开始计时。")
            else:
                self._stop_minigame_video_test_session("测试开关关闭")
                self.log("🎞️ [小游戏测试] 已关闭画面叠加。")
        tk.Checkbutton(
            self.lie_detector_subsettings_frame,
            text="小游戏测试回放（自动启用检测；F6运行时按设定延时叠加视频）",
            variable=self.lie_detector_test_enabled_var,
            command=_toggle_lie_detector_test,
            font=("Segoe UI", 8),
            fg="#ffcc80",
            bg="#202026",
            activeforeground="#ffe0b2",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 0))

        self.lie_detector_show_overlay_var = tk.BooleanVar(
            value=bool(self.config.get("lie_detector_show_overlay_enabled", False))
        )
        self.lie_detector_show_overlay_var.set(
            bool(self.lie_detector_auto_solve_var.get() and self.lie_detector_show_overlay_var.get())
        )
        self.config["lie_detector_show_overlay_enabled"] = self.lie_detector_show_overlay_var.get()

        def _toggle_lie_detector_show_overlay():
            enabled = bool(self.lie_detector_show_overlay_var.get())
            self.config["lie_detector_show_overlay_enabled"] = enabled
            self._save_config()
            if not enabled:
                self._hide_minigame_overlay()
            self.log("🪟 [测谎小游戏] 叠加窗口" + ("已开启（鼠标穿透）" if enabled else "已关闭"))
        self.lie_detector_overlay_checkbutton = tk.Checkbutton(
            self.lie_detector_subsettings_frame,
            text="叠加窗口",
            variable=self.lie_detector_show_overlay_var,
            command=_toggle_lie_detector_show_overlay,
            font=("Segoe UI", 8),
            fg="#80deea",
            bg="#202026",
            activeforeground="#b2ebf2",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        )
        self.lie_detector_overlay_checkbutton.pack(fill=tk.X, pady=(2, 0))


        self.lie_detector_replay_record_var = tk.BooleanVar(
            value=bool(self.config.get("lie_detector_replay_record_enabled", False))
            and bool(self.lie_detector_auto_solve_var.get())
        )
        self.config["lie_detector_replay_record_enabled"] = self.lie_detector_replay_record_var.get()

        def _toggle_lie_detector_replay_record():
            enabled = bool(self.lie_detector_replay_record_var.get())
            self.config["lie_detector_replay_record_enabled"] = enabled
            self._save_config()
            self.log("📹 [测谎小游戏] 游戏结束后自动录像" + ("已开启" if enabled else "已关闭"))

        self.lie_detector_replay_record_checkbutton = tk.Checkbutton(
            self.lie_detector_subsettings_frame,
            text="小游戏结束后自动录像",
            variable=self.lie_detector_replay_record_var,
            command=_toggle_lie_detector_replay_record,
            font=("Segoe UI", 8),
            fg="#ffcc80",
            bg="#202026",
            activeforeground="#ffe0b2",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        )
        self.lie_detector_replay_record_checkbutton.pack(fill=tk.X, pady=(2, 0))

        replay_hotkey_row = tk.Frame(self.lie_detector_subsettings_frame, bg="#202026")
        replay_hotkey_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(
            replay_hotkey_row,
            text="即时回放按键（N卡默认 Alt+F10）",
            font=("Segoe UI", 8),
            fg="#b0bec5",
            bg="#202026",
        ).pack(side=tk.LEFT)
        replay_hotkey_var = tk.StringVar(
            value=str(self.config.get("lie_detector_replay_hotkey", "Alt+F10"))
        )
        replay_hotkey_entry = tk.Entry(
            replay_hotkey_row,
            textvariable=replay_hotkey_var,
            width=12,
            font=("Segoe UI", 8),
            bg="#17171c",
            fg="#eeeeee",
            insertbackground="#ffffff",
        )
        replay_hotkey_entry.pack(side=tk.LEFT, padx=(8, 4))

        def _save_lie_detector_replay_hotkey(_event=None):
            hotkey = replay_hotkey_var.get().strip() or "Alt+F10"
            if not self._parse_minigame_replay_hotkey(hotkey):
                messagebox.showwarning(
                    "即时回放按键",
                    "按键格式无效。示例：Alt+F10、Ctrl+F10 或 F12。",
                    parent=self.root,
                )
                replay_hotkey_var.set(str(self.config.get("lie_detector_replay_hotkey", "Alt+F10")))
                return
            replay_hotkey_var.set(hotkey)
            self.config["lie_detector_replay_hotkey"] = hotkey
            self._save_config()

        tk.Button(
            replay_hotkey_row,
            text="保存",
            command=_save_lie_detector_replay_hotkey,
            font=("Segoe UI", 8),
            fg="#ffffff",
            bg="#37474f",
            relief=tk.FLAT,
            padx=7,
        ).pack(side=tk.LEFT)
        replay_hotkey_entry.bind("<FocusOut>", _save_lie_detector_replay_hotkey)


        self.lie_detector_save_transition_snapshots_var = tk.BooleanVar(
            value=bool(self.config.get("lie_detector_save_transition_snapshots", False))
            and bool(self.lie_detector_auto_solve_var.get())
        )
        self.config["lie_detector_save_transition_snapshots"] = (
            self.lie_detector_save_transition_snapshots_var.get()
        )

        def _toggle_lie_detector_save_transition_snapshots():
            enabled = bool(self.lie_detector_save_transition_snapshots_var.get())
            self.config["lie_detector_save_transition_snapshots"] = enabled
            self._save_config()
            self.log(
                "📸 [测谎小游戏] 状态切换截图" + ("已开启（logs/minigame_snapshots）" if enabled else "已关闭")
            )

        self.lie_detector_save_transition_snapshots_checkbutton = tk.Checkbutton(
            self.lie_detector_subsettings_frame,
            text="切换到 COUNTDOWN / ACTIVE 时保存截图",
            variable=self.lie_detector_save_transition_snapshots_var,
            command=_toggle_lie_detector_save_transition_snapshots,
            font=("Segoe UI", 8),
            fg="#80deea",
            bg="#202026",
            activeforeground="#b2ebf2",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        )
        self.lie_detector_save_transition_snapshots_checkbutton.pack(fill=tk.X, pady=(2, 0))

        self.enable_monster_detection_var = tk.BooleanVar(
            value=bool(self.config.get("enable_monster_detection", True))
        )
        def _toggle_monster_detection():
            enabled = bool(self.enable_monster_detection_var.get())
            self.config["enable_monster_detection"] = enabled
            if hasattr(self, "detector") and self.detector:
                self.detector.enable_monster_detection = enabled
            hp_on = bool(getattr(self, "enable_monster_hp_bar_detection_var", None) and self.enable_monster_hp_bar_detection_var.get())
            if not enabled and not hp_on and self.detector is not None:
                if hasattr(self.detector, "tracked_monsters"):
                    self.detector.tracked_monsters = []
                with self.lock:
                    if self.latest_result is not None:
                        self.latest_result.monsters = []
                        self.latest_result.locked_target = None
            self._save_config()
            self.log("👾 [怪物本体识别] " + ("已启用" if enabled else "已关闭" + ("（当前仅保留血条识别模式）" if hp_on else "：仅保留角色定位与平台导航")))

        self.enable_monster_hp_bar_detection_var = tk.BooleanVar(
            value=bool(self.config.get("enable_monster_hp_bar_detection", True))
        )
        def _toggle_monster_hp_bar_detection():
            enabled = bool(self.enable_monster_hp_bar_detection_var.get())
            self.config["enable_monster_hp_bar_detection"] = enabled
            self._save_config()
            if hasattr(self, "detector") and self.detector:
                self.detector.enable_monster_hp_bar_detection = enabled
            self.log("🩸 [怪物血条识别] " + ("已启用（怪物受遮挡时通过血条反推怪物位置）" if enabled else "已关闭"))

        r_mob_cbs = tk.Frame(card, bg="#202026")
        r_mob_cbs.pack(fill=tk.X, pady=(1, 0))

        tk.Checkbutton(
            r_mob_cbs,
            text="启用怪物识别/索敌",
            variable=self.enable_monster_detection_var,
            command=_toggle_monster_detection,
            font=("Segoe UI", 8),
            fg="#ce93d8",
            bg="#202026",
            activeforeground="#f3c4ff",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        tk.Checkbutton(
            r_mob_cbs,
            text="启用血条识别 (遮挡补充)",
            variable=self.enable_monster_hp_bar_detection_var,
            command=_toggle_monster_hp_bar_detection,
            font=("Segoe UI", 8),
            fg="#ef9a9a",
            bg="#202026",
            activeforeground="#ffcdd2",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(6, 0))

        hp_device_labels = {"自动": "auto", "GPU": "cuda", "CPU": "cpu"}
        hp_device_reverse = {value: label for label, value in hp_device_labels.items()}
        self.monster_hp_bar_device_var = tk.StringVar(
            value=hp_device_reverse.get(
                str(self.config.get("monster_hp_bar_compute_device", "auto")).lower(),
                "自动",
            )
        )
        hp_device_combo = ttk.Combobox(
            r_mob_cbs,
            textvariable=self.monster_hp_bar_device_var,
            values=tuple(hp_device_labels),
            width=4,
            state="readonly",
            font=("Segoe UI", 8),
        )
        hp_device_combo.pack(side=tk.LEFT, padx=(5, 0))

        def _on_hp_bar_device_changed(_event=None):
            requested = hp_device_labels.get(
                self.monster_hp_bar_device_var.get(), "auto"
            )
            self.config["monster_hp_bar_compute_device"] = requested
            if getattr(self, "detector", None) is not None:
                self.detector.set_monster_hp_bar_compute_device(requested)
            self._save_config()
            self.log(
                "🩸 [怪物血条后端] "
                + {
                    "auto": "自动选择GPU，异常时回退CPU",
                    "cuda": "优先使用GPU，异常时回退CPU",
                    "cpu": "固定使用保留的CPU算法",
                }[requested]
            )

        hp_device_combo.bind("<<ComboboxSelected>>", _on_hp_bar_device_changed)

        self.attack_only_mode_var = tk.BooleanVar(
            value=bool(self.config.get("attack_only_mode", False))
        )
        def _toggle_attack_only_mode():
            enabled = bool(self.attack_only_mode_var.get())
            self.config["attack_only_mode"] = enabled
            self._save_config()
            if hasattr(self, "combat_fsm") and self.combat_fsm:
                self.combat_fsm.attack_only_mode = enabled
            self._sync_bot_ui_state()
            self.log("⚔️ [挂机模式] " + ("已切换为【仅攻击键介入模式】（移动由玩家完全控制，怪物进入攻击范围自动按攻击）"
                                        if enabled else "已切换为【全自动巡逻挂机模式】"))
        tk.Checkbutton(
            card,
            text="仅攻击键介入模式（玩家自主移动，仅自动打怪/测试用）",
            variable=self.attack_only_mode_var,
            command=_toggle_attack_only_mode,
            font=("Segoe UI", 8, "bold"),
            fg="#ff80ab",
            bg="#202026",
            activeforeground="#ff4081",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(1, 0))

        # 参数设置与路线状态整合行
        r3 = tk.Frame(card, bg="#202026")
        r3.pack(fill=tk.X, pady=(2, 1))
        tk.Button(
            r3, text="⚙ 特殊参数", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#5d4037", relief=tk.FLAT, padx=7, pady=2,
            command=self.open_special_params_dialog,
        ).pack(side=tk.LEFT)
        tk.Button(
            r3, text="⏱️ 下跳时序", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#00695c", relief=tk.FLAT, padx=7, pady=2,
            command=self.open_down_jump_timing_dialog,
        ).pack(side=tk.LEFT, padx=(5, 0))
        tk.Button(
            r3, text="🔌 断线重连", font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#1565c0", relief=tk.FLAT, padx=7, pady=2,
            command=self.open_reconnect_settings_dialog,
        ).pack(side=tk.LEFT, padx=(5, 0))

        self.lbl_route_status = tk.Label(r3, text="路线: 自适应同层游走",
                                         font=("Consolas", 8), fg="#90a4ae", bg="#202026", anchor=tk.E)
        self.lbl_route_status.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        self.lbl_reconnect_status = tk.Label(
            card,
            text=("🔌 " + self._reconnect_status_text),
            font=("Segoe UI", 8),
            fg=("#64b5f6" if self.config.get("reconnect_enabled", False) else "#78909c"),
            bg="#202026",
            anchor=tk.W,
        )
        self.lbl_reconnect_status.pack(fill=tk.X, pady=(2, 0))

    def open_reconnect_settings_dialog(self):
        """配置两套客户端共用的自动重连目标。"""
        top = tk.Toplevel(self.root)
        top.title("断线自动重连设置")
        fit_window_to_work_area(top, (650, 500), (540, 400), parent=self.root)
        top.transient(self.root)
        top.grab_set()
        top.configure(bg="#202026")

        reconnect_footer = tk.Frame(top, bg="#202026", padx=16, pady=10)
        reconnect_footer.pack(side=tk.BOTTOM, fill=tk.X)
        reconnect_viewport = tk.Frame(top, bg="#202026")
        reconnect_viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        reconnect_canvas = tk.Canvas(reconnect_viewport, bg="#202026", highlightthickness=0)
        reconnect_scrollbar = ttk.Scrollbar(
            reconnect_viewport, orient=tk.VERTICAL, command=reconnect_canvas.yview
        )
        body = tk.Frame(reconnect_canvas, bg="#202026", padx=16, pady=14)
        reconnect_body_window = reconnect_canvas.create_window((0, 0), window=body, anchor="nw")
        reconnect_canvas.configure(yscrollcommand=reconnect_scrollbar.set)
        reconnect_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        reconnect_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        body.bind(
            "<Configure>",
            lambda _event: reconnect_canvas.configure(scrollregion=reconnect_canvas.bbox("all")),
        )
        reconnect_canvas.bind(
            "<Configure>",
            lambda event: reconnect_canvas.itemconfigure(reconnect_body_window, width=event.width),
        )
        top.bind(
            "<MouseWheel>",
            lambda event: reconnect_canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        enabled_var = tk.BooleanVar(value=bool(self.config.get("reconnect_enabled", False)))
        tk.Checkbutton(
            body,
            text="F6 运行期间启用断线自动重连",
            variable=enabled_var,
            font=("Segoe UI", 10, "bold"),
            fg="#64b5f6", bg="#202026", selectcolor="#303038",
            activeforeground="#90caf9", activebackground="#202026",
        ).pack(anchor=tk.W)

        profile_labels = {
            "自动识别两种客户端": "auto",
            "怀旧服客户端（账号已记忆）": "official",
            "经典/单机版（需要密码）": "classic",
        }
        reverse_profile = {value: label for label, value in profile_labels.items()}

        def add_row(label, variable, *, width=10, widget="entry", values=()):
            row = tk.Frame(body, bg="#202026")
            row.pack(fill=tk.X, pady=(10, 0))
            tk.Label(row, text=label, width=17, anchor=tk.W,
                     fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
            if widget == "combo":
                control = ttk.Combobox(
                    row, textvariable=variable, values=values,
                    state="readonly", width=width,
                )
            else:
                control = tk.Entry(row, textvariable=variable, width=width, justify=tk.CENTER)
            control.pack(side=tk.LEFT, padx=(8, 6))
            return row, control

        profile_var = tk.StringVar(
            value=reverse_profile.get(
                str(self.config.get("reconnect_client_profile", "auto")),
                "自动识别两种客户端",
            )
        )
        add_row("客户端类型", profile_var, width=31, widget="combo", values=tuple(profile_labels))

        server_index_var = tk.StringVar(value=str(self.config.get("reconnect_server_index", 1)))
        server_total_var = tk.StringVar(value=str(self.config.get("reconnect_server_total", 5)))
        row, _ = add_row("区服/世界序号", server_index_var)
        tk.Label(row, text="/ 总数", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=server_total_var, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=6)

        channel_index_var = tk.StringVar(value=str(self.config.get("reconnect_channel_index", 1)))
        channel_total_var = tk.StringVar(value=str(self.config.get("reconnect_channel_total", 20)))
        row, _ = add_row("频道序号", channel_index_var)
        tk.Label(row, text="/ 总数", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=channel_total_var, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=6)

        character_index_var = tk.StringVar(value=str(self.config.get("reconnect_character_index", 1)))
        character_total_var = tk.StringVar(value=str(self.config.get("reconnect_character_total", 3)))
        row, _ = add_row("人物序号（左/中/右）", character_index_var)
        tk.Label(row, text="/ 总数", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=character_total_var, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=6)

        password_var = tk.StringVar()
        password_row = tk.Frame(body, bg="#202026")
        password_row.pack(fill=tk.X, pady=(10, 0))
        tk.Label(password_row, text="经典客户端密码", width=17, anchor=tk.W,
                 fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(password_row, textvariable=password_var, width=24, show="●").pack(side=tk.LEFT, padx=(8, 6))
        has_password = bool(unprotect_secret(self.config.get("reconnect_password_protected", "")))
        password_hint = tk.Label(
            password_row,
            text=("已安全保存；留空保持" if has_password else "尚未设置"),
            fg=("#81c784" if has_password else "#ef9a9a"), bg="#202026",
        )
        password_hint.pack(side=tk.LEFT)
        clear_password_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            body, text="清除已保存密码", variable=clear_password_var,
            fg="#b0bec5", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
        ).pack(anchor=tk.W, padx=(132, 0), pady=(2, 0))

        ready = bool(getattr(self, "reconnect_controller", None) and self.reconnect_controller.vision.ready)
        tk.Label(
            body,
            text=(
                "状态识别模板已就绪。检测到登录界面后会先释放全部按键并停止战斗/巡逻；"
                "进入地图且重新定位成功后恢复原 F6。"
                if ready else
                "未找到 disconnectVideo 中的参考截图，自动重连不会启动。"
            ),
            justify=tk.LEFT, wraplength=600,
            fg=("#80cbc4" if ready else "#ef9a9a"), bg="#202026",
        ).pack(fill=tk.X, pady=(16, 0))

        buttons = reconnect_footer

        def parse_pair(index_var, total_var, label, max_total=200):
            try:
                index = int(index_var.get().strip())
                total = int(total_var.get().strip())
                if not (1 <= index <= total <= max_total):
                    raise ValueError
                return index, total
            except Exception:
                raise ValueError(
                    f"{label}必须满足 1 ≤ 选择序号 ≤ 总数 ≤ {max_total}。"
                )

        def save():
            try:
                server_index, server_total = parse_pair(server_index_var, server_total_var, "区服")
                channel_index, channel_total = parse_pair(channel_index_var, channel_total_var, "频道")
                character_index, character_total = parse_pair(
                    character_index_var, character_total_var, "人物", max_total=3
                )
            except ValueError as exc:
                messagebox.showwarning("参数无效", str(exc), parent=top)
                return
            profile = profile_labels.get(profile_var.get(), "auto")
            new_password = password_var.get()
            if clear_password_var.get():
                protected_password = ""
            elif new_password:
                try:
                    protected_password = protect_secret(new_password)
                except Exception as exc:
                    messagebox.showerror("密码保存失败", f"Windows DPAPI 加密失败：{exc}", parent=top)
                    return
            else:
                protected_password = self.config.get("reconnect_password_protected", "")
            if enabled_var.get() and profile == "classic" and not protected_password:
                messagebox.showwarning(
                    "缺少密码", "经典/单机版自动重连必须先填写密码。", parent=top
                )
                return

            self.config.update({
                "reconnect_enabled": bool(enabled_var.get()),
                "reconnect_client_profile": profile,
                "reconnect_server_index": server_index,
                "reconnect_server_total": server_total,
                "reconnect_channel_index": channel_index,
                "reconnect_channel_total": channel_total,
                "reconnect_character_index": character_index,
                "reconnect_character_total": character_total,
                "reconnect_password_protected": protected_password,
            })
            self._save_config()
            self.log(
                f"🔌 [重连设置] {'启用' if enabled_var.get() else '关闭'}，"
                f"客户端={profile}，区服={server_index}/{server_total}，"
                f"频道={channel_index}/{channel_total}，人物={character_index}/{character_total}"
            )
            if not enabled_var.get() and self.reconnect_controller.active:
                self.reconnect_controller.cancel("设置中关闭自动重连")
            self._reconnect_status_text = "断线重连待命" if enabled_var.get() else "断线重连已关闭"
            self._reconnect_status_active = False
            self._sync_reconnect_status_label()
            top.grab_release()
            top.destroy()

        tk.Button(buttons, text="取消", width=10, command=top.destroy).pack(side=tk.RIGHT, padx=6)
        tk.Button(buttons, text="应用", width=10, command=save,
                  bg="#1565c0", fg="#ffffff").pack(side=tk.RIGHT)

    def open_special_params_dialog(self):
        """编辑不常用的识别参数。"""
        top = tk.Toplevel(self.root)
        top.title("特殊参数")
        fit_window_to_work_area(top, (900, 900), (600, 520), parent=self.root)
        top.resizable(True, True)
        top.transient(self.root)
        top.grab_set()
        top.configure(bg="#202026")

        # 参数持续增加，不能再依赖放大固定窗口。底部动作栏常驻，中间
        # 表单统一滚动；在 768p 与高 DPI 下仍能访问每一项参数。
        footer = tk.Frame(top, bg="#202026", padx=14, pady=10)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        viewport = tk.Frame(top, bg="#202026")
        viewport.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        params_canvas = tk.Canvas(viewport, bg="#202026", highlightthickness=0)
        params_scrollbar = ttk.Scrollbar(
            viewport, orient=tk.VERTICAL, command=params_canvas.yview
        )
        body = tk.Frame(params_canvas, bg="#202026", padx=14, pady=12)
        body_window = params_canvas.create_window((0, 0), window=body, anchor="nw")
        params_canvas.configure(yscrollcommand=params_scrollbar.set)
        params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        body.bind(
            "<Configure>",
            lambda _event: params_canvas.configure(scrollregion=params_canvas.bbox("all")),
        )
        params_canvas.bind(
            "<Configure>",
            lambda event: params_canvas.itemconfigure(body_window, width=event.width),
        )
        top.bind(
            "<MouseWheel>",
            lambda event: params_canvas.yview_scroll(int(-event.delta / 120), "units"),
        )
        row = tk.Frame(body, bg="#202026")
        row.pack(fill=tk.X)
        tk.Label(row, text="怪物模板匹配比例", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        scale_var = tk.StringVar(value=f"{float(self.config.get('monster_template_scale', 1.0)):g}")
        entry = tk.Entry(row, textvariable=scale_var, width=10, justify=tk.CENTER)
        entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(row, text="北斗/经典原版为1.0", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        player_row = tk.Frame(body, bg="#202026")
        player_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(player_row, text="人物识别基础准入门限", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        player_th_var = tk.StringVar(value=f"{float(self.config.get('player_feature_entry_threshold', 0.58)):g}")
        player_th_entry = tk.Entry(player_row, textvariable=player_th_var, width=10, justify=tk.CENTER)
        player_th_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(player_row, text="默认0.58 (核验/宽松/激活框/救援等门限同步联动)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        size_row = tk.Frame(body, bg="#202026")
        size_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(size_row, text="黄点候选像素尺寸", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        sizes = self.config.get("yellow_dot_candidate_sizes")
        if isinstance(sizes, (list, tuple)):
            sizes_text = ", ".join(
                f"{item[0]}x{item[1]}" if isinstance(item, (list, tuple)) and len(item) == 2
                else str(item) for item in sizes
            )
        else:
            sizes_text = str(sizes or "4x4, 4x5, 5x4, 5x5, 6x5, 5x6, 6x6")
        sizes_var = tk.StringVar(value=sizes_text)
        sizes_entry = tk.Entry(size_row, textvariable=sizes_var, width=34)
        sizes_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(size_row, text="例如：4x4, 5x5, 6x6", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        region_row = tk.Frame(body, bg="#202026")
        region_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(region_row, text="底部屏蔽高度(Y)", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        default_bottom_h = 71
        if self.config.get("bottom_exclusion_height") is not None:
            default_bottom_h = int(self.config.get("bottom_exclusion_height"))
        else:
            regs = self.config.get("recognition_exclusion_regions", [])
            for r in regs:
                if r.get("h"):
                    default_bottom_h = int(r["h"])
                    break
        bottom_h_var = tk.StringVar(value=str(default_bottom_h))
        bottom_h_entry = tk.Entry(region_row, textvariable=bottom_h_var, width=10, justify=tk.CENTER)
        bottom_h_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(region_row, text="像素 (横向全宽，保证识别区域为单一矩形不降速)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        arrival_row = tk.Frame(body, bg="#202026")
        arrival_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(arrival_row, text="巡逻到站范围", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        arrival_tolerance_var = tk.StringVar(
            value=f"{float(self.config.get('patrol_arrival_tolerance_px', 20.0)):g}"
        )
        arrival_tolerance_entry = tk.Entry(
            arrival_row, textvariable=arrival_tolerance_var, width=10, justify=tk.CENTER
        )
        arrival_tolerance_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            arrival_row,
            text="px（目标点±此值；应用后运行中立即生效，短平台自动收窄）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        motion_prediction_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            body,
            text="使用输入感知 Kalman X（当前主坐标源）",
            variable=motion_prediction_var,
            fg="#ffffff", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
            state=tk.DISABLED,
        ).pack(anchor=tk.W, pady=(12, 0))
        tk.Label(
            body,
            text="固定启用；raw 黄点坐标仅用于外力突跳和源平台安全检查",
            fg="#b0bec5", bg="#202026",
        ).pack(anchor=tk.W)

        rear_delay_row = tk.Frame(body, bg="#202026")
        rear_delay_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(rear_delay_row, text="回身攻击等待时间", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        rear_delay_var = tk.StringVar(
            value=f"{float(self.config.get('rear_attack_turn_delay_sec', 2.0)):g}"
        )
        rear_delay_entry = tk.Entry(
            rear_delay_row, textvariable=rear_delay_var, width=10, justify=tk.CENTER
        )
        rear_delay_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            rear_delay_row,
            text="秒（仅单向技能、目标平台停留阶段；默认2秒）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        combat_dwell_row = tk.Frame(body, bg="#202026")
        combat_dwell_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(combat_dwell_row, text="攻击后停留延长", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        combat_dwell_var = tk.StringVar(
            value=f"{float(self.config.get('patrol_combat_dwell_extension_sec', 2.0)):g}"
        )
        combat_dwell_entry = tk.Entry(
            combat_dwell_row, textvariable=combat_dwell_var, width=10, justify=tk.CENTER
        )
        combat_dwell_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            combat_dwell_row,
            text="秒（截止前n秒有攻击或正反向攻击框有怪时续期；默认2秒）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        monster_attack_timeout_row = tk.Frame(body, bg="#202026")
        monster_attack_timeout_row.pack(fill=tk.X, pady=(12, 0))
        monster_attack_timeout_enabled_var = tk.BooleanVar(
            value=bool(self.config.get("monster_attack_hard_timeout_enabled", True))
        )
        tk.Checkbutton(
            monster_attack_timeout_row,
            text="怪物攻击判定硬上限",
            variable=monster_attack_timeout_enabled_var,
            fg="#ffffff", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
        ).pack(side=tk.LEFT)
        monster_attack_timeout_var = tk.StringVar(
            value=(
                f"{float(self.config.get('monster_attack_hard_timeout_ms', 500.0)):g}"
            )
        )
        monster_attack_timeout_entry = tk.Entry(
            monster_attack_timeout_row,
            textvariable=monster_attack_timeout_var,
            width=10,
            justify=tk.CENTER,
        )
        monster_attack_timeout_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            monster_attack_timeout_row,
            text="ms（不勾选即取消此判定；勾选时超限必定取消红框，默认500ms）",
            fg="#b0bec5",
            bg="#202026",
        ).pack(side=tk.LEFT)

        ghost_box_timeout_row = tk.Frame(body, bg="#202026")
        ghost_box_timeout_row.pack(fill=tk.X, pady=(12, 0))
        ghost_box_timeout_enabled_var = tk.BooleanVar(
            value=bool(self.config.get("ghost_box_safety_timeout_enabled", True))
        )
        tk.Checkbutton(
            ghost_box_timeout_row,
            text="防幽灵框的安全上限",
            variable=ghost_box_timeout_enabled_var,
            fg="#ffffff", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
        ).pack(side=tk.LEFT)
        ghost_box_timeout_var = tk.StringVar(
            value=(
                f"{float(self.config.get('ghost_box_safety_timeout_ms', 750.0)):g}"
            )
        )
        ghost_box_timeout_entry = tk.Entry(
            ghost_box_timeout_row,
            textvariable=ghost_box_timeout_var,
            width=10,
            justify=tk.CENTER,
        )
        ghost_box_timeout_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            ghost_box_timeout_row,
            text="ms（光流无模板强行注销上限；不勾选即不限制，默认750ms）",
            fg="#b0bec5",
            bg="#202026",
        ).pack(side=tk.LEFT)

        failure_replan_row = tk.Frame(body, bg="#202026")
        failure_replan_row.pack(fill=tk.X, pady=(12, 0))
        failure_replan_var = tk.BooleanVar(
            value=bool(self.config.get("patrol_failure_replan_enabled", False))
        )
        tk.Checkbutton(
            failure_replan_row,
            text="多次平台到达失败后的重规划",
            variable=failure_replan_var,
            fg="#ffffff", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
        ).pack(side=tk.LEFT)
        tk.Label(
            failure_replan_row,
            text="（改走备用路线；默认关闭，关闭时不记录失败次数）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT, padx=(8, 0))

        run_jump_fallback_var = tk.BooleanVar(
            value=bool(self.config.get("run_jump_fallback_to_static_grab", False))
        )
        tk.Checkbutton(
            body,
            text="跑跳抓绳失败后直跳抓绳",
            variable=run_jump_fallback_var,
            fg="#ffffff", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
        ).pack(anchor=tk.W, pady=(12, 0))
        tk.Label(
            body,
            text="默认关闭；开启后跑跳失败达到指定次数时改用绳梯下方原地跳抓",
            fg="#b0bec5", bg="#202026",
        ).pack(anchor=tk.W)

        run_jump_failure_row = tk.Frame(body, bg="#202026")
        run_jump_failure_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            run_jump_failure_row, text="跑跳抓绳失败次数", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        run_jump_failure_var = tk.StringVar(
            value=str(int(self.config.get("run_jump_failure_limit", 2) or 2))
        )
        run_jump_failure_entry = tk.Entry(
            run_jump_failure_row, textvariable=run_jump_failure_var,
            width=10, justify=tk.CENTER,
        )
        run_jump_failure_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            run_jump_failure_row, text="次（默认2次）", fg="#b0bec5", bg="#202026"
        ).pack(side=tk.LEFT)

        skirmish_guard_row = tk.Frame(body, bg="#202026")
        skirmish_guard_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(skirmish_guard_row, text="游击平台保护范围", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        skirmish_guard_var = tk.StringVar(
            value=f"{float(self.config.get('skirmish_platform_guard_px', 50.0)):g}"
        )
        skirmish_guard_entry = tk.Entry(
            skirmish_guard_row, textvariable=skirmish_guard_var, width=10, justify=tk.CENTER
        )
        skirmish_guard_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            skirmish_guard_row,
            text="px（游击时与当前平台左右边缘的最小距离；默认50px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        attack_edge_row = tk.Frame(body, bg="#202026")
        attack_edge_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            attack_edge_row, text="攻击前边缘保护范围", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        attack_edge_var = tk.StringVar(
            value=f"{float(self.config.get('attack_edge_guard_px', 50.0)):g}"
        )
        tk.Entry(
            attack_edge_row, textvariable=attack_edge_var, width=10, justify=tk.CENTER
        ).pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            attack_edge_row,
            text="px（红框已有怪时先内撤；独立于游击范围，默认50px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        climb_hold_row = tk.Frame(body, bg="#202026")
        climb_hold_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(climb_hold_row, text="绳顶闭环恢复基准", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        climb_hold_var = tk.StringVar(
            value=f"{float(self.config.get('climb_top_hold_sec', 1.0)):g}"
        )
        climb_hold_entry = tk.Entry(
            climb_hold_row, textvariable=climb_hold_var, width=10, justify=tk.CENTER
        )
        climb_hold_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            climb_hold_row,
            text="秒（用于计算自适应恢复预算；检测到真实落台会立即结束）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        source_raw_y_row = tk.Frame(body, bg="#202026")
        source_raw_y_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            source_raw_y_row, text="源平台 raw Y 容差", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        source_raw_y_var = tk.StringVar(
            value=f"{float(self.config.get('jump_source_raw_y_tolerance_px', 65.0)):g}"
        )
        source_raw_y_entry = tk.Entry(
            source_raw_y_row, textvariable=source_raw_y_var, width=10, justify=tk.CENTER
        )
        source_raw_y_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            source_raw_y_row,
            text="px（起跳前 raw Y 与斜坡当前位置高度的允许误差；默认65px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        top_exit_raw_y_row = tk.Frame(body, bg="#202026")
        top_exit_raw_y_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            top_exit_raw_y_row, text="绳顶落台 raw Y 容差", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        top_exit_raw_y_var = tk.StringVar(
            value=f"{float(self.config.get('top_exit_raw_y_tolerance_px', 36.0)):g}"
        )
        top_exit_raw_y_entry = tk.Entry(
            top_exit_raw_y_row,
            textvariable=top_exit_raw_y_var,
            width=10,
            justify=tk.CENTER,
        )
        top_exit_raw_y_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            top_exit_raw_y_row,
            text="px（脱离绳轴后 raw Y 与目标平台站立线的允许误差；实测默认36px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        rope_bottom_y_row = tk.Frame(body, bg="#202026")
        rope_bottom_y_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(rope_bottom_y_row, text="绳底吸附 Y 容差", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        rope_bottom_y_var = tk.StringVar(
            value=f"{float(self.config.get('ladder_bottom_y_tolerance_px', 35.0)):g}"
        )
        rope_bottom_y_entry = tk.Entry(rope_bottom_y_row, textvariable=rope_bottom_y_var, width=10, justify=tk.CENTER)
        rope_bottom_y_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(rope_bottom_y_row, text="px（绳梯最下端吸附范围；默认35px）", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        takeoff_gate_row = tk.Frame(body, bg="#202026")
        takeoff_gate_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            takeoff_gate_row, text="连续起跳线容差", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        takeoff_gate_var = tk.StringVar(
            value=f"{float(self.config.get('takeoff_gate_tolerance_px', 20.0)):g}"
        )
        takeoff_gate_entry = tk.Entry(
            takeoff_gate_row, textvariable=takeoff_gate_var, width=10, justify=tk.CENTER
        )
        takeoff_gate_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            takeoff_gate_row,
            text="px（连续坐标距计划起跳线的提前触发范围；默认20px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        portal_ocr_range_row = tk.Frame(body, bg="#202026")
        portal_ocr_range_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            portal_ocr_range_row, text="传送点触发OCR范围", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        portal_ocr_range_var = tk.StringVar(
            value=f"{float(self.config.get('portal_ocr_trigger_range_px', 200.0)):g}"
        )
        portal_ocr_range_entry = tk.Entry(
            portal_ocr_range_row,
            textvariable=portal_ocr_range_var,
            width=10,
            justify=tk.CENTER,
        )
        portal_ocr_range_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            portal_ocr_range_row,
            text="px（传送门附近按UP后立即连续识别地图；默认200px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        inplace_dwell_row = tk.Frame(body, bg="#202026")
        inplace_dwell_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            inplace_dwell_row, text="就地停留边缘保护", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        inplace_dwell_var = tk.StringVar(
            value=f"{float(self.config.get('inplace_dwell_safe_margin_px', 50.0)):g}"
        )
        inplace_dwell_entry = tk.Entry(
            inplace_dwell_row, textvariable=inplace_dwell_var, width=10, justify=tk.CENTER
        )
        inplace_dwell_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            inplace_dwell_row,
            text="px（走位被怪打断/击退时，距悬崖边缘大于此值直接就地停留；默认50px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        align_pulse_row = tk.Frame(body, bg="#202026")
        align_pulse_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            align_pulse_row, text="走位微调脉冲范围", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        align_pulse_min_var = tk.StringVar(
            value=f"{float(self.config.get('align_pulse_min_ms', 12.0)):g}"
        )
        align_pulse_min_entry = tk.Entry(
            align_pulse_row, textvariable=align_pulse_min_var, width=6, justify=tk.CENTER
        )
        align_pulse_min_entry.pack(side=tk.LEFT, padx=(12, 4))
        tk.Label(align_pulse_row, text="~", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        align_pulse_max_var = tk.StringVar(
            value=f"{float(self.config.get('align_pulse_max_ms', 35.0)):g}"
        )
        align_pulse_max_entry = tk.Entry(
            align_pulse_row, textvariable=align_pulse_max_var, width=6, justify=tk.CENTER
        )
        align_pulse_max_entry.pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(
            align_pulse_row,
            text="ms（PID对齐/梯绳对齐脉冲时长；小地图1px≈16px时可适当增大上限；默认12~35ms）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        teleport_dist_row = tk.Frame(body, bg="#202026")
        teleport_dist_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            teleport_dist_row, text="法师瞬移距离", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        teleport_dist_var = tk.StringVar(
            value=f"{float(self.config.get('teleport_distance_px', 150.0)):g}"
        )
        teleport_dist_entry = tk.Entry(
            teleport_dist_row, textvariable=teleport_dist_var, width=10, justify=tk.CENTER
        )
        teleport_dist_entry.pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            teleport_dist_row,
            text="px（法师单次瞬移水平/垂直跨度；默认150px）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        reconnect_wait_row = tk.Frame(body, bg="#202026")
        reconnect_wait_row.pack(fill=tk.X, pady=(12, 0))
        tk.Label(
            reconnect_wait_row, text="重连账号占用等待", fg="#ffffff", bg="#202026"
        ).pack(side=tk.LEFT)
        reconnect_wait_var = tk.StringVar(
            value=f"{float(self.config.get('reconnect_account_online_wait_sec', 300.0)):g}"
        )
        tk.Entry(
            reconnect_wait_row, textvariable=reconnect_wait_var,
            width=10, justify=tk.CENTER,
        ).pack(side=tk.LEFT, padx=(12, 8))
        tk.Label(
            reconnect_wait_row,
            text="秒（仅出现账号仍在线弹窗时等待；默认300秒）",
            fg="#b0bec5", bg="#202026",
        ).pack(side=tk.LEFT)

        down_jump_entry_row = tk.Frame(body, bg="#202026")
        down_jump_entry_row.pack(fill=tk.X, pady=(14, 0))
        tk.Button(
            down_jump_entry_row,
            text="⏱️ 下跳时序设置...",
            bg="#00695c",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
            command=lambda: self.open_down_jump_timing_dialog(parent=top),
        ).pack(side=tk.LEFT)
        tk.Label(
            down_jump_entry_row,
            text="配置 6 个阶段时序 (前中性/预压/重叠/补跳/保持/落地) 及一键置前测试",
            fg="#80cbc4",
            bg="#202026",
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT, padx=(10, 0))

        btns = footer
        def apply_value():
            try:
                value = float(scale_var.get().strip())
                if not 0.1 <= value <= 3.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning("参数无效", "模板匹配比例请输入 0.1 到 3.0 之间的数字。", parent=top)
                return
            try:
                p_th = float(player_th_var.get().strip())
                if not 0.1 <= p_th <= 1.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning("参数无效", "人物识别基础准入门限请输入 0.1 到 1.0 之间的数字。", parent=top)
                return
            parsed_sizes = self._parse_yellow_candidate_sizes(sizes_var.get())
            if not parsed_sizes:
                messagebox.showwarning("参数无效", "黄点尺寸请输入类似 4x4, 5x5 的格式。", parent=top)
                return
            try:
                bottom_h = int(bottom_h_var.get().strip())
                if not 0 <= bottom_h <= 600:
                    raise ValueError
            except Exception:
                messagebox.showwarning("参数无效", "底部屏蔽高度请输入 0 到 600 之间的整数像素。", parent=top)
                return
            try:
                arrival_tolerance = float(arrival_tolerance_var.get().strip())
                if not 1.0 <= arrival_tolerance <= 100.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "巡逻到站范围请输入 1 到 100 之间的像素值。", parent=top
                )
                return
            try:
                rear_attack_delay = float(rear_delay_var.get().strip())
                if not 0.0 <= rear_attack_delay <= 30.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "回身攻击等待时间请输入 0 到 30 秒之间的数字。", parent=top
                )
                return
            try:
                combat_dwell_extension = float(combat_dwell_var.get().strip())
                if not 0.0 <= combat_dwell_extension <= 60.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "攻击后停留延长请输入 0 到 60 秒之间的数字。", parent=top
                )
                return
            try:
                monster_attack_hard_timeout_ms = float(
                    monster_attack_timeout_var.get().strip()
                )
                if not 20.0 <= monster_attack_hard_timeout_ms <= 5000.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效",
                    "怪物攻击判定硬上限请输入 20 到 5000 之间的毫秒值。",
                    parent=top,
                )
                return
            try:
                ghost_box_safety_timeout_ms = float(
                    ghost_box_timeout_var.get().strip()
                )
                if not 20.0 <= ghost_box_safety_timeout_ms <= 5000.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效",
                    "防幽灵框的安全上限请输入 20 到 5000 之间的毫秒值。",
                    parent=top,
                )
                return
            try:
                run_jump_failure_limit = int(run_jump_failure_var.get().strip())
                if not 1 <= run_jump_failure_limit <= 20:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "跑跳抓绳失败次数请输入 1 到 20 之间的整数。", parent=top
                )
                return
            try:
                skirmish_guard = float(skirmish_guard_var.get().strip())
                if not 1.0 <= skirmish_guard <= 300.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "游击平台保护范围请输入 1 到 300 之间的像素值。", parent=top
                )
                return
            try:
                attack_edge_guard = float(attack_edge_var.get().strip())
                if not 1.0 <= attack_edge_guard <= 300.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "攻击前边缘保护范围请输入 1 到 300 之间的像素值。",
                    parent=top,
                )
                return
            try:
                climb_top_hold = float(climb_hold_var.get().strip())
                if not 0.0 <= climb_top_hold <= 5.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "登顶脱绳保持时间请输入 0 到 5 秒之间的数字。", parent=top
                )
                return
            try:
                source_raw_y_tolerance = float(source_raw_y_var.get().strip())
                if not 1.0 <= source_raw_y_tolerance <= 200.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "源平台 raw Y 容差请输入 1 到 200 之间的像素值。", parent=top
                )
                return
            try:
                top_exit_raw_y_tolerance = float(top_exit_raw_y_var.get().strip())
                if not 1.0 <= top_exit_raw_y_tolerance <= 200.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效",
                    "绳顶落台 raw Y 容差请输入 1 到 200 之间的像素值。",
                    parent=top,
                )
                return
            try:
                rope_bottom_y_tolerance = float(rope_bottom_y_var.get().strip())
                if not -200.0 <= rope_bottom_y_tolerance <= 200.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "绳底吸附 Y 容差请输入 -200 到 200 之间的像素值。", parent=top
                )
                return
            try:
                takeoff_gate_tolerance = float(takeoff_gate_var.get().strip())
                if not 1.0 <= takeoff_gate_tolerance <= 100.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "连续起跳线容差请输入 1 到 100 之间的像素值。", parent=top
                )
                return
            try:
                portal_ocr_range = float(portal_ocr_range_var.get().strip())
                if not 0.0 <= portal_ocr_range <= 2000.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "传送点触发OCR范围请输入 0 到 2000 之间的像素值。", parent=top
                )
                return
            try:
                inplace_dwell_safe_margin = float(inplace_dwell_var.get().strip())
                if not 0.0 <= inplace_dwell_safe_margin <= 300.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "就地停留边缘保护请输入 0 到 300 之间的像素值。", parent=top
                )
                return
            try:
                align_pulse_min = float(align_pulse_min_var.get().strip())
                align_pulse_max = float(align_pulse_max_var.get().strip())
                if not (1.0 <= align_pulse_min <= align_pulse_max <= 200.0):
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "走位微调脉冲范围请输入 1 到 200 之间的毫秒值，且下限不得高于上限。", parent=top
                )
                return
            try:
                teleport_distance = float(teleport_dist_var.get().strip())
                if teleport_distance <= 0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "法师瞬移距离请输入大于 0 的像素值。", parent=top
                )
                return
            try:
                reconnect_wait_sec = float(reconnect_wait_var.get().strip())
                if not 0.0 <= reconnect_wait_sec <= 3600.0:
                    raise ValueError
            except Exception:
                messagebox.showwarning(
                    "参数无效", "重连账号占用等待请输入 0 到 3600 秒之间的数字。", parent=top
                )
                return

            self.config["monster_template_scale"] = value
            self.config["player_feature_entry_threshold"] = p_th
            self.config["yellow_dot_candidate_sizes"] = [f"{w}x{h}" for w, h in parsed_sizes]
            self.config["bottom_exclusion_height"] = bottom_h
            self.config["patrol_arrival_tolerance_px"] = arrival_tolerance
            self.config["use_horizontal_motion_prediction"] = True
            self.config["rear_attack_turn_delay_sec"] = rear_attack_delay
            self.config["patrol_combat_dwell_extension_sec"] = combat_dwell_extension
            monster_attack_hard_timeout_enabled = bool(
                monster_attack_timeout_enabled_var.get()
            )
            ghost_box_safety_timeout_enabled = bool(
                ghost_box_timeout_enabled_var.get()
            )
            self.config["monster_attack_hard_timeout_enabled"] = (
                monster_attack_hard_timeout_enabled
            )
            self.config["monster_attack_hard_timeout_ms"] = (
                monster_attack_hard_timeout_ms
            )
            self.config["ghost_box_safety_timeout_enabled"] = (
                ghost_box_safety_timeout_enabled
            )
            self.config["ghost_box_safety_timeout_ms"] = (
                ghost_box_safety_timeout_ms
            )
            self.config["patrol_failure_replan_enabled"] = bool(failure_replan_var.get())
            self.config["run_jump_fallback_to_static_grab"] = bool(
                run_jump_fallback_var.get()
            )
            self.config["run_jump_failure_limit"] = run_jump_failure_limit
            self.config["skirmish_platform_guard_px"] = skirmish_guard
            self.config["attack_edge_guard_px"] = attack_edge_guard
            self.config["climb_top_hold_sec"] = climb_top_hold
            self.config["jump_source_raw_y_tolerance_px"] = source_raw_y_tolerance
            self.config["top_exit_raw_y_tolerance_px"] = top_exit_raw_y_tolerance
            self.config["ladder_bottom_y_tolerance_px"] = rope_bottom_y_tolerance
            PlatformGraph.ladder_bottom_y_tolerance_px = rope_bottom_y_tolerance
            self.config["takeoff_gate_tolerance_px"] = takeoff_gate_tolerance
            self.config["portal_ocr_trigger_range_px"] = portal_ocr_range
            self.config["inplace_dwell_safe_margin_px"] = inplace_dwell_safe_margin
            self.config["align_pulse_min_ms"] = align_pulse_min
            self.config["align_pulse_max_ms"] = align_pulse_max
            self.config["teleport_distance_px"] = teleport_distance
            self.config["reconnect_account_online_wait_sec"] = reconnect_wait_sec
            if hasattr(self, "motion") and self.motion is not None:
                self.motion.config["inplace_dwell_safe_margin_px"] = inplace_dwell_safe_margin
                self.motion.config["align_pulse_min_ms"] = align_pulse_min
                self.motion.config["align_pulse_max_ms"] = align_pulse_max
                self.motion.teleport_distance_px = teleport_distance
                self.motion.config["teleport_distance_px"] = teleport_distance

            # 根据当前帧尺寸生成横贯全屏的单一 Y 轴屏蔽区域
            fw = 1280
            fh = 720
            if self.capture and hasattr(self.capture, "_latest_frame") and self.capture._latest_frame is not None:
                fh, fw = self.capture._latest_frame.shape[:2]

            if bottom_h > 0:
                new_regions = [{
                    "x": 0,
                    "y": max(0, fh - bottom_h),
                    "w": fw,
                    "h": bottom_h,
                    "monster": True,
                    "player": True
                }]
            else:
                new_regions = []
            self.config["recognition_exclusion_regions"] = new_regions

            self._save_config()
            self._refresh_platform_graph_teleport()
            if self.detector is not None:
                self.detector.set_monster_template_scale(value)
                self.detector.set_player_feature_threshold(p_th)
                self.detector.set_exclusion_regions(new_regions)
                self.detector.set_attack_observation_hard_timeout_ms(
                    monster_attack_hard_timeout_ms,
                    enabled=monster_attack_hard_timeout_enabled,
                )
                self.detector.set_ghost_box_safety_timeout_ms(
                    ghost_box_safety_timeout_ms,
                    enabled=ghost_box_safety_timeout_enabled,
                )
                self.detector.load_dynamic_monster_templates(self._get_enabled_mobs())
            for _tracker in (self.tracker, self.raw_tracker, self.radar_tracker):
                _tracker.set_yellow_candidate_sizes(parsed_sizes)
            self.log(
                f"[特殊参数] 模板比例={value:g}x, 人物基础门限={p_th:g}, "
                f"底部Y屏蔽={bottom_h}px, 巡逻到站范围=±{arrival_tolerance:g}px, "
                f"回身攻击等待={rear_attack_delay:g}s, "
                f"攻击后停留延长={combat_dwell_extension:g}s, "
                f"怪物攻击判定硬上限={'开启(' + str(monster_attack_hard_timeout_ms) + 'ms)' if monster_attack_hard_timeout_enabled else '关闭'}, "
                f"防幽灵框安全上限={'开启(' + str(ghost_box_safety_timeout_ms) + 'ms)' if ghost_box_safety_timeout_enabled else '关闭'}, "
                f"失败后备用路线={'开启' if failure_replan_var.get() else '关闭'}, "
                f"跑跳失败后直跳={'开启' if run_jump_fallback_var.get() else '关闭'}，"
                f"失败次数={run_jump_failure_limit}次, "
                f"游击平台保护={skirmish_guard:g}px, "
                f"攻击前边缘保护={attack_edge_guard:g}px, "
                f"登顶脱绳保持={climb_top_hold:g}s, "
                f"源平台raw Y容差={source_raw_y_tolerance:g}px, "
                f"绳顶落台raw Y容差={top_exit_raw_y_tolerance:g}px, "
                f"连续起跳线容差=±{takeoff_gate_tolerance:g}px, "
                f"传送点触发OCR范围={portal_ocr_range:g}px, "
                f"就地停留边缘保护={inplace_dwell_safe_margin:g}px, "
                f"走位微调脉冲={align_pulse_min:g}~{align_pulse_max:g}ms, "
                f"法师瞬移距离={teleport_distance:g}px, "
                f"重连账号占用等待={reconnect_wait_sec:g}s"
            )
            top.grab_release()
            top.destroy()

        tk.Button(
            btns,
            text="⏱️ 下跳时序...",
            command=lambda: self.open_down_jump_timing_dialog(parent=top),
            bg="#00695c",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
        ).pack(side=tk.LEFT)
        tk.Button(btns, text="取消", command=lambda: (top.grab_release(), top.destroy()),
                  width=10).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(btns, text="应用", command=apply_value, width=10,
                  bg="#2e7d32", fg="#ffffff", relief=tk.FLAT).pack(side=tk.RIGHT)
        entry.focus_set()

    def open_down_jump_timing_dialog(self, parent=None):
        """平台下跳 6 阶段按键时序精细配置窗口 (支持毫秒微调、防趴下预设与一键置前测试)"""
        top = tk.Toplevel(parent or self.root)
        top.title("⏱️ 下跳按键时序设置 (Down-Jump Timing)")
        fit_window_to_work_area(top, (740, 840), (600, 520), parent=parent or self.root)
        top.resizable(True, True)
        top.transient(parent or self.root)
        top.grab_set()
        top.configure(bg="#18181c")

        # 变量定义并初始化当前配置
        v_pre_neutral = tk.StringVar(value=f"{float(self.config.get('down_jump_pre_neutral_ms', 50.0)):g}")
        v_down_prep = tk.StringVar(value=f"{float(self.config.get('down_jump_down_prep_ms', 109.0)):g}")
        v_jump_hold = tk.StringVar(value=f"{float(self.config.get('down_jump_jump_hold_ms', 124.0)):g}")
        v_retry_enabled = tk.BooleanVar(value=bool(self.config.get('down_jump_retry_enabled', False)))
        v_retry_jump_hold = tk.StringVar(value=f"{float(self.config.get('down_jump_retry_jump_hold_ms', 100.0)):g}")
        v_adaptive_full_retry = tk.BooleanVar(
            value=bool(self.config.get('down_jump_adaptive_full_retry_enabled', True))
        )
        v_post_down_hold = tk.StringVar(value=f"{float(self.config.get('down_jump_post_down_hold_ms', 1.0)):g}")
        v_post_neutral = tk.StringVar(value=f"{float(self.config.get('down_jump_post_neutral_ms', 100.0)):g}")
        v_wait_land = tk.StringVar(value=f"{float(self.config.get('down_jump_wait_land_ms', 450.0)):g}")

        # 容器
        container = tk.Frame(top, bg="#18181c", padx=14, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        # 顶部提示卡片
        header = tk.Frame(container, bg="#202026", padx=12, pady=10, relief=tk.GROOVE, bd=1)
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            header,
            text="⏱️ 平台下跳 6 阶段按键时序精细控制",
            font=("Segoe UI", 11, "bold"),
            fg="#4fc3f7",
            bg="#202026",
            anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Label(
            header,
            text="下跳动作由 DOWN + JUMP 组合按键触发。怀旧服若 DOWN 预压时间过长角色会直接趴下导致下跳失败；\n"
                 "若 JUMP 重叠保持过短可能漏触发；若未确认下落时补按跳跃可提高成功率。各阶段均可以毫秒(ms)为单位微调。\n"
                 "点击下方【▶️ 测试当前时序下跳】可立即将游戏窗口置前并执行一次完整下跳测试。",
            font=("Segoe UI", 8),
            fg="#90a4ae",
            bg="#202026",
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(4, 0))

        # 中间滚动区域
        canvas_frame = tk.Frame(container, bg="#18181c")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg="#18181c", highlightthickness=0)
        sb = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_content = tk.Frame(canvas, bg="#18181c")

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_win = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        top.bind("<MouseWheel>", _on_mousewheel)

        # 卡片生成函数
        def make_stage_frame(parent_widget, title_text, badge_color="#0288d1"):
            frame = tk.Frame(parent_widget, bg="#202026", padx=12, pady=8, relief=tk.GROOVE, bd=1)
            frame.pack(fill=tk.X, pady=(0, 8))
            title_bar = tk.Frame(frame, bg="#202026")
            title_bar.pack(fill=tk.X)
            tk.Label(
                title_bar,
                text=title_text,
                font=("Segoe UI", 9, "bold"),
                fg=badge_color,
                bg="#202026",
            ).pack(side=tk.LEFT)
            return frame

        # 阶段 1
        s1 = make_stage_frame(scroll_content, "阶段 1：起跳前中性缓冲时间 (ms)", "#4fc3f7")
        s1_row = tk.Frame(s1, bg="#202026")
        s1_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(s1_row, text="起跳前中性缓冲: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s1_row, textvariable=v_pre_neutral, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s1_row, text="ms (默认 50ms。确保所有按键松开及窗口焦点响应的静默期)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        # 阶段 2
        s2 = make_stage_frame(scroll_content, "阶段 2：DOWN 键预压保持时间 (ms)", "#ffb74d")
        s2_row = tk.Frame(s2, bg="#202026")
        s2_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(s2_row, text="DOWN 预压时长: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s2_row, textvariable=v_down_prep, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s2_row, text="ms (默认 109ms：17次手动录制的中位数)", fg="#ffcc80", bg="#202026").pack(side=tk.LEFT)

        # 阶段 3
        s3 = make_stage_frame(scroll_content, "阶段 3：Jump 键按下重叠时间 (ms)", "#81c784")
        s3_row = tk.Frame(s3, bg="#202026")
        s3_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(s3_row, text="Jump 键重叠时长: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s3_row, textvariable=v_jump_hold, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s3_row, text="ms (默认 124ms：手动录制的组合键真实重叠中位数)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        # 阶段 4
        s4 = make_stage_frame(scroll_content, "阶段 4：未检测到下落时的补按跳跃", "#ba68c8")
        s4_row1 = tk.Frame(s4, bg="#202026")
        s4_row1.pack(fill=tk.X, pady=(2, 0))
        tk.Checkbutton(
            s4_row1,
            text="启用首跳未下落补按跳跃",
            variable=v_retry_enabled,
            fg="#ffffff",
            bg="#202026",
            selectcolor="#303038",
            activeforeground="#ffffff",
            activebackground="#202026",
        ).pack(side=tk.LEFT)
        tk.Label(s4_row1, text="(首跳后若 Y 坐标未确认下落，且 DOWN 仍按住时触发补按)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT, padx=(8, 0))
        s4_row2 = tk.Frame(s4, bg="#202026")
        s4_row2.pack(fill=tk.X, pady=(4, 0))
        tk.Label(s4_row2, text="补按 Jump 保持: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s4_row2, textvariable=v_retry_jump_hold, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s4_row2, text="ms (默认 100ms。补按跳跃键持续保持的时长)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)
        s4_row3 = tk.Frame(s4, bg="#202026")
        s4_row3.pack(fill=tk.X, pady=(6, 0))
        tk.Checkbutton(
            s4_row3,
            text="启用 raw Y 未下落时完整组合键自适应重试（推荐）",
            variable=v_adaptive_full_retry,
            fg="#ffffff",
            bg="#202026",
            selectcolor="#303038",
            activeforeground="#ffffff",
            activebackground="#202026",
        ).pack(side=tk.LEFT)

        # 阶段 5
        s5 = make_stage_frame(scroll_content, "阶段 5：Jump 松开后 DOWN 保持与释放等待 (ms)", "#e57373")
        s5_row1 = tk.Frame(s5, bg="#202026")
        s5_row1.pack(fill=tk.X, pady=(2, 0))
        tk.Label(s5_row1, text="DOWN 保持最长等待: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s5_row1, textvariable=v_post_down_hold, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s5_row1, text="ms (默认 1ms：手动录制中两个键近似同时松开)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)
        s5_row2 = tk.Frame(s5, bg="#202026")
        s5_row2.pack(fill=tk.X, pady=(4, 0))
        tk.Label(s5_row2, text="DOWN 松开后中性等待: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s5_row2, textvariable=v_post_neutral, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s5_row2, text="ms (默认 100ms。释放 DOWN 键后所有键保持松开的静默缓冲)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        # 阶段 6
        s6 = make_stage_frame(scroll_content, "阶段 6：重力下落落地等待时间 (ms)", "#4db6ac")
        s6_row = tk.Frame(s6, bg="#202026")
        s6_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(s6_row, text="下落着陆等待: ", fg="#ffffff", bg="#202026").pack(side=tk.LEFT)
        tk.Entry(s6_row, textvariable=v_wait_land, width=10, justify=tk.CENTER).pack(side=tk.LEFT, padx=(4, 8))
        tk.Label(s6_row, text="ms (默认 450ms。下跳指令完成后等待重力落到下层平台的冷却时间)", fg="#b0bec5", bg="#202026").pack(side=tk.LEFT)

        # 手动实机按键录制：用低级键盘钩子捕获物理扫描码，随后按相同
        # 边沿与时间间隔回放。该测试独立于上面的参数化下跳状态机。
        recording_card = make_stage_frame(
            scroll_content,
            "实机对照：录制并原样回放手动按键",
            "#ffd54f",
        )
        recording_status = tk.StringVar(
            value=(
                "已有录制，可直接回放"
                if os.path.isfile(self.manual_input_recorder.save_path)
                else "尚未录制"
            )
        )
        tk.Label(
            recording_card,
            text=(
                "只记录游戏处于前台时的真实键盘事件；程序注入事件会被排除。"
                "停止后保存扫描码、按下/松开顺序及毫秒间隔。"
            ),
            fg="#b0bec5",
            bg="#202026",
            justify=tk.LEFT,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(4, 4))
        recording_buttons = tk.Frame(recording_card, bg="#202026")
        recording_buttons.pack(fill=tk.X)
        btn_record_manual = tk.Button(
            recording_buttons,
            text="⏺ 录制手动按键",
            bg="#ad1457",
            fg="#ffffff",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=9,
            pady=3,
        )
        btn_record_manual.pack(side=tk.LEFT)
        btn_replay_manual = tk.Button(
            recording_buttons,
            text="▶ 原样复现录制",
            bg="#6a1b9a",
            fg="#ffffff",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=9,
            pady=3,
            state=(
                tk.NORMAL
                if os.path.isfile(self.manual_input_recorder.save_path)
                else tk.DISABLED
            ),
        )
        btn_replay_manual.pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(
            recording_card,
            textvariable=recording_status,
            fg="#80cbc4",
            bg="#202026",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=650,
        ).pack(fill=tk.X, pady=(5, 0))

        def _manual_recording_busy() -> bool:
            combat = getattr(self, "combat_fsm", None)
            ladder_test = getattr(self, "ladder_grab_test_runner", None)
            random_test = getattr(self, "random_path_test_runner", None)
            return bool(
                (combat is not None and combat.is_running)
                or (ladder_test is not None and ladder_test.active)
                or (random_test is not None and random_test.active)
            )

        def stop_manual_recording() -> List[Dict]:
            events = self.manual_input_recorder.stop(save=True)
            btn_record_manual.configure(text="⏺ 录制手动按键", bg="#ad1457")
            if events:
                summary = self.manual_input_recorder.timing_summary(events)
                recording_status.set(summary)
                btn_replay_manual.configure(state=tk.NORMAL)
                edge_text = " ".join(
                    f"{str(e.get('key')).upper()}{'↓' if e.get('edge') == 'down' else '↑'}"
                    f"@{float(e.get('offset_ms', 0.0)):.1f}ms"
                    for e in events
                )
                self.log(f"⏹️ [手动按键录制完成] {summary} | {edge_text}")
                self.log(
                    f"💾 [手动按键录制] 已保存至 {self.manual_input_recorder.save_path}"
                )
            else:
                recording_status.set("未记录到游戏前台的物理键盘事件")
                self.log("⚠️ [手动按键录制] 没有捕获到游戏前台的物理按键")
            return events

        def toggle_manual_recording():
            if self.manual_input_recorder.is_recording:
                stop_manual_recording()
                return
            if _manual_recording_busy():
                messagebox.showwarning(
                    "无法录制",
                    "请先停止 F6、绳梯稳定性测试和随机全图测试，再录制手动输入。",
                    parent=top,
                )
                return
            self.motion.stop()
            self.manual_input_recorder.target_hwnd = int(self.hwnd or 0)
            ok, message = self.manual_input_recorder.start()
            if not ok:
                recording_status.set(message)
                self.log(f"❌ [手动按键录制] {message}")
                messagebox.showerror("录制启动失败", message, parent=top)
                return
            btn_record_manual.configure(text="⏹ 停止并保存", bg="#c62828")
            btn_replay_manual.configure(state=tk.DISABLED)
            recording_status.set("录制中：切到游戏手动下跳，完成后点击“停止并保存”")
            self.log(
                "⏺️ [手动按键录制] 已启动；仅捕获游戏前台物理键盘，"
                "忽略所有 injected 输入"
            )

        def replay_manual_recording():
            if self.manual_input_recorder.is_recording:
                messagebox.showwarning("仍在录制", "请先停止并保存录制。", parent=top)
                return
            if _manual_recording_busy():
                messagebox.showwarning(
                    "无法回放",
                    "请先停止 F6、绳梯稳定性测试和随机全图测试。",
                    parent=top,
                )
                return
            events = self.manual_input_recorder.load()
            if not events:
                messagebox.showwarning("没有录制", "请先录制一次手动下跳。", parent=top)
                return
            btn_record_manual.configure(state=tk.DISABLED)
            btn_replay_manual.configure(state=tk.DISABLED, text="⏳ 正在复现...")
            recording_status.set("即将置前游戏，并按原扫描码与原时间间隔回放")

            def worker():
                before_pos = self.current_player_raw_world_pos
                try:
                    self.log(
                        "▶️ [手动按键原样回放] 释放残留键、稳定游戏焦点后开始；"
                        f"事件数={len(events)}，起始raw={before_pos}"
                    )
                    report = self.manual_input_recorder.replay(self.input_driver, events)
                    time.sleep(0.60)
                    after_pos = self.current_player_raw_world_pos
                    report["before_raw_pos"] = before_pos
                    report["after_raw_pos"] = after_pos
                    report_path = os.path.join(
                        os.path.dirname(CONFIG_PATH),
                        "logs",
                        "manual_input_replay_report.json",
                    )
                    with open(report_path + ".tmp", "w", encoding="utf-8") as stream:
                        json.dump(report, stream, ensure_ascii=False, indent=2)
                    os.replace(report_path + ".tmp", report_path)
                    self.log(
                        "✅ [手动按键原样回放] 完成："
                        f"投递失败={report['transport_failures']}，"
                        f"最大调度误差={report['max_abs_drift_ms']:.2f}ms，"
                        f"raw={before_pos}->{after_pos}，报告={report_path}"
                    )
                    status_text = (
                        f"回放完成：最大误差{report['max_abs_drift_ms']:.2f}ms，"
                        f"raw {before_pos} → {after_pos}"
                    )
                except Exception as exc:
                    status_text = f"回放失败：{exc}"
                    self.log(f"❌ [手动按键原样回放] {exc}")
                finally:
                    def restore_buttons():
                        if not top.winfo_exists():
                            return
                        recording_status.set(status_text)
                        btn_record_manual.configure(state=tk.NORMAL)
                        btn_replay_manual.configure(
                            state=tk.NORMAL,
                            text="▶ 原样复现录制",
                        )
                    try:
                        top.after(0, restore_buttons)
                    except Exception:
                        pass

            threading.Thread(
                target=worker,
                name="ManualInputReplay",
                daemon=True,
            ).start()

        btn_record_manual.configure(command=toggle_manual_recording)
        btn_replay_manual.configure(command=replay_manual_recording)

        # 预设推荐栏
        preset_frame = tk.Frame(scroll_content, bg="#202026", padx=12, pady=8, relief=tk.GROOVE, bd=1)
        preset_frame.pack(fill=tk.X, pady=(2, 8))
        tk.Label(preset_frame, text="⚡ 时序预设：", fg="#ffffff", bg="#202026", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        def apply_preset_anti_crouch():
            v_pre_neutral.set("30")
            v_down_prep.set("20")
            v_jump_hold.set("80")
            v_retry_enabled.set(False)
            v_retry_jump_hold.set("80")
            v_adaptive_full_retry.set(True)
            v_post_down_hold.set("120")
            v_post_neutral.set("80")
            v_wait_land.set("400")
            self.log("⏱️ [下跳时序] 已载入推荐预设：⚡ 快速防趴下时序 (DOWN预压20ms, JUMP重叠80ms, 禁用补按)")

        def apply_preset_default():
            v_pre_neutral.set("50")
            v_down_prep.set("109")
            v_jump_hold.set("124")
            v_retry_enabled.set(False)
            v_retry_jump_hold.set("100")
            v_adaptive_full_retry.set(True)
            v_post_down_hold.set("1")
            v_post_neutral.set("100")
            v_wait_land.set("450")
            self.log(
                "⏱️ [下跳时序] 已载入手动录制中位数默认时序："
                "DOWN预压109ms，组合重叠124ms，近似同时松键1ms"
            )

        tk.Button(
            preset_frame,
            text="⚡ 快速防趴下推荐时序",
            bg="#e65100",
            fg="#ffffff",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=2,
            command=apply_preset_anti_crouch,
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            preset_frame,
            text="🔄 恢复录制中位数默认时序",
            bg="#37474f",
            fg="#ffffff",
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            padx=8,
            pady=2,
            command=apply_preset_default,
        ).pack(side=tk.LEFT, padx=6)

        # 解析与校验函数
        def parse_values() -> Optional[Dict[str, Any]]:
            try:
                pre_neutral = float(v_pre_neutral.get().strip())
                if not 0.0 <= pre_neutral <= 2000.0:
                    raise ValueError("起跳前中性缓冲必须在 0 到 2000 ms 之间")
                down_prep = float(v_down_prep.get().strip())
                if not 0.0 <= down_prep <= 2000.0:
                    raise ValueError("DOWN预压时长必须在 0 到 2000 ms 之间")
                jump_hold = float(v_jump_hold.get().strip())
                if not 10.0 <= jump_hold <= 2000.0:
                    raise ValueError("Jump重叠时长必须在 10 到 2000 ms 之间")
                retry_jump_hold = float(v_retry_jump_hold.get().strip())
                if not 10.0 <= retry_jump_hold <= 2000.0:
                    raise ValueError("补按Jump保持必须在 10 到 2000 ms 之间")
                post_down_hold = float(v_post_down_hold.get().strip())
                if not 0.0 <= post_down_hold <= 2000.0:
                    raise ValueError("DOWN保持等待必须在 0 到 2000 ms 之间")
                post_neutral = float(v_post_neutral.get().strip())
                if not 0.0 <= post_neutral <= 2000.0:
                    raise ValueError("DOWN松开后中性等待必须在 0 到 2000 ms 之间")
                wait_land = float(v_wait_land.get().strip())
                if not 0.0 <= wait_land <= 5000.0:
                    raise ValueError("下落着陆等待必须在 0 到 5000 ms 之间")
                return {
                    "down_jump_pre_neutral_ms": pre_neutral,
                    "down_jump_down_prep_ms": down_prep,
                    "down_jump_jump_hold_ms": jump_hold,
                    "down_jump_retry_enabled": bool(v_retry_enabled.get()),
                    "down_jump_retry_jump_hold_ms": retry_jump_hold,
                    "down_jump_adaptive_full_retry_enabled": bool(
                        v_adaptive_full_retry.get()
                    ),
                    "down_jump_post_down_hold_ms": post_down_hold,
                    "down_jump_post_neutral_ms": post_neutral,
                    "down_jump_wait_land_ms": wait_land,
                }
            except Exception as e:
                messagebox.showwarning("参数无效", str(e), parent=top)
                return None

        # 底部操作按钮栏
        bottom_bar = tk.Frame(container, bg="#18181c")
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        # 按照用户要求增加测试按钮：按下后让游戏窗口置前，接着执行一次当前设置时序的下跳流程
        btn_test = tk.Button(
            bottom_bar,
            text="▶️ 测试当前时序下跳",
            bg="#0277bd",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=4,
        )
        btn_test.pack(side=tk.LEFT)

        def run_test():
            if self.manual_input_recorder.is_recording:
                messagebox.showwarning(
                    "仍在录制",
                    "请先停止并保存手动按键录制。",
                    parent=top,
                )
                return
            timings = parse_values()
            if timings is None:
                return

            btn_test.configure(state=tk.DISABLED, text="⏳ 正在前台执行测试...")

            def _worker():
                try:
                    self.log("⏱️ [下跳测试] 正在将游戏窗口置前并激活输入焦点...")
                    focused = self.input_driver.ensure_focus()
                    if not focused:
                        self.log("⚠️ [下跳测试] 窗口焦点置前可能未完全成功，尝试继续发送下跳序列...")
                    time.sleep(0.15)
                    self.log(
                        f"⏱️ [下跳测试] 开始执行下跳时序: DOWN预压={timings['down_jump_down_prep_ms']}ms, "
                        f"JUMP重叠={timings['down_jump_jump_hold_ms']}ms, 补按={'开启' if timings['down_jump_retry_enabled'] else '关闭'}..."
                    )
                    confirmed = self.motion.down_jump(timings_override=timings)
                    self.log(
                        f"{'✅' if confirmed else '⚠️'} [下跳测试] "
                        f"raw Y下落确认={'成功' if confirmed else '失败'}"
                    )
                except Exception as ex:
                    self.log(f"❌ [下跳测试异常] {ex}")
                finally:
                    if top.winfo_exists():
                        top.after(0, lambda: btn_test.configure(state=tk.NORMAL, text="▶️ 测试当前时序下跳"))

            threading.Thread(target=_worker, daemon=True).start()

        btn_test.configure(command=run_test)

        def close_dialog():
            if self.manual_input_recorder.is_recording:
                stop_manual_recording()
            try:
                top.unbind("<MouseWheel>")
            except Exception:
                pass
            try:
                top.grab_release()
            except Exception:
                pass
            top.destroy()
            if parent is not None and parent.winfo_exists():
                try:
                    parent.grab_set()
                except Exception:
                    pass

        def save_and_close():
            timings = parse_values()
            if timings is None:
                return

            self.config.update(timings)
            self.motion.config = self.config
            self._save_config()

            self.log(
                f"[下跳时序已保存] 前中性={timings['down_jump_pre_neutral_ms']}ms, "
                f"DOWN预压={timings['down_jump_down_prep_ms']}ms, "
                f"Jump重叠={timings['down_jump_jump_hold_ms']}ms, "
                f"补按={'开启' if timings['down_jump_retry_enabled'] else '关闭'}({timings['down_jump_retry_jump_hold_ms']}ms), "
                f"完整组合自适应重试={'开启' if timings['down_jump_adaptive_full_retry_enabled'] else '关闭'}, "
                f"保持={timings['down_jump_post_down_hold_ms']}ms, "
                f"后中性={timings['down_jump_post_neutral_ms']}ms, "
                f"落地等待={timings['down_jump_wait_land_ms']}ms"
            )
            close_dialog()

        top.protocol("WM_DELETE_WINDOW", close_dialog)

        tk.Button(
            bottom_bar,
            text="取消",
            command=close_dialog,
            width=10,
            bg="#424242",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
        ).pack(side=tk.RIGHT, padx=(8, 0))

        tk.Button(
            bottom_bar,
            text="确定",
            command=save_and_close,
            width=10,
            bg="#2e7d32",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=4,
        ).pack(side=tk.RIGHT)

    def open_recognition_exclusion_dialog(self, parent=None):
        """框选并管理不参与人物/怪物识别的屏蔽区域。"""
        top = tk.Toplevel(parent or self.root)
        top.title("识别屏蔽区域")
        dialog_w, dialog_h = fit_window_to_work_area(
            top, (940, 820), (640, 520), parent=parent or self.root
        )
        top.transient(parent or self.root)
        top.configure(bg="#202026")

        regions = [dict(x) for x in self.config.get("recognition_exclusion_regions", [])]
        frame = self.capture.capture_frame(copy=True) if self.capture else None
        if frame is None:
            messagebox.showwarning("无法框选", "当前还没有可用的游戏画面。", parent=top)
            top.destroy()
            return
        fh, fw = frame.shape[:2]
        # 为类型选项、区域列表和底部按钮预留固定空间；截图画布只消费
        # 剩余区域，避免窗口被限幅后画布把操作按钮挤出客户区。
        max_canvas_w = max(320.0, float(dialog_w - 40))
        max_canvas_h = max(180.0, float(dialog_h - 285))
        scale = min(1.0, max_canvas_w / max(1, fw), max_canvas_h / max(1, fh))
        dw, dh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA) if scale != 1.0 else rgb
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        canvas = tk.Canvas(top, width=dw, height=dh, bg="#111111", highlightthickness=1,
                           highlightbackground="#607d8b")
        canvas.pack(padx=10, pady=(10, 4))
        canvas.configure(cursor="arrow")
        canvas.create_image(0, 0, image=tk_img, anchor=tk.NW)
        canvas._exclusion_image = tk_img
        selecting = {"active": False, "x": 0, "y": 0, "rect": None}
        monster_var = tk.BooleanVar(value=True)
        player_var = tk.BooleanVar(value=True)

        def redraw():
            canvas.delete("region")
            for i, r in enumerate(regions, 1):
                x1, y1 = r["x"] * scale, r["y"] * scale
                x2, y2 = (r["x"] + r["w"]) * scale, (r["y"] + r["h"]) * scale
                tags = []
                if r.get("monster"): tags.append("怪物")
                if r.get("player"): tags.append("人物")
                canvas.create_rectangle(x1, y1, x2, y2, outline="#ff5252", width=2, tags="region")
                canvas.create_text(x1 + 4, y1 + 4, text=f"#{i} {'/'.join(tags)}",
                                  anchor=tk.NW, fill="#ffeb3b", tags="region")

        def press(event):
            if not selecting["active"]:
                return
            # 只在按下瞬间记录起点；拖动时不能覆盖起点，否则矩形会始终退化成一个点。
            selecting.update(x=canvas.canvasx(event.x), y=canvas.canvasy(event.y))
            if selecting["rect"]:
                canvas.delete(selecting["rect"])
            selecting["rect"] = canvas.create_rectangle(selecting["x"], selecting["y"],
                                                         selecting["x"], selecting["y"],
                                                         outline="#00e5ff", width=2)

        def drag(event):
            if not selecting["active"] or selecting["rect"] is None:
                return
            # 使用固定起点更新当前预览框，保证按住左键拖动时能看到选区。
            canvas.coords(selecting["rect"], selecting["x"], selecting["y"],
                          canvas.canvasx(event.x), canvas.canvasy(event.y))

        def release(event):
            if not selecting["active"]:
                return
            x1, x2 = sorted((selecting["x"], canvas.canvasx(event.x)))
            y1, y2 = sorted((selecting["y"], canvas.canvasy(event.y)))
            if selecting["rect"]:
                canvas.delete(selecting["rect"])
                selecting["rect"] = None
            if x2 - x1 < 5 or y2 - y1 < 5:
                return
            if not monster_var.get() and not player_var.get():
                messagebox.showwarning("未选择类型", "至少勾选怪物识别或人物识别。", parent=top)
                return
            region_x = max(0, int(round(x1 / scale)))
            region_y = max(0, int(round(y1 / scale)))
            region_w = max(1, int(round((x2 - x1) / scale)))
            region_h = max(1, int(round((y2 - y1) / scale)))
            # 不允许保存超出截图边界的起点；宽高也限制在当前帧尺寸内。
            region_w = min(region_w, max(1, fw - region_x))
            region_h = min(region_h, max(1, fh - region_y))
            regions.append({"x": region_x, "y": region_y,
                            "w": region_w, "h": region_h,
                            "monster": bool(monster_var.get()), "player": bool(player_var.get())})
            selecting["active"] = False
            select_btn.config(text="开始框选")
            canvas.configure(cursor="arrow")
            redraw()

        canvas.bind("<Button-1>", press)
        canvas.bind("<B1-Motion>", drag)
        canvas.bind("<ButtonRelease-1>", release)
        canvas.bind("<Enter>", lambda _event: canvas.focus_set())

        controls = tk.Frame(top, bg="#202026")
        controls.pack(fill=tk.X, padx=10, pady=4)
        monster_cb = tk.Checkbutton(controls, text="屏蔽怪物识别", variable=monster_var,
                                    fg="#ffb74d", bg="#202026", selectcolor="#202026")
        monster_cb.pack(side=tk.LEFT)
        player_cb = tk.Checkbutton(controls, text="屏蔽人物识别", variable=player_var,
                                   fg="#ce93d8", bg="#202026", selectcolor="#202026")
        player_cb.pack(side=tk.LEFT, padx=12)
        select_btn = tk.Button(controls, text="开始框选", bg="#0277bd", fg="#ffffff", relief=tk.FLAT)
        select_btn.pack(side=tk.LEFT, padx=12)
        listbox = tk.Listbox(top, height=5, bg="#121214", fg="#e0f7fa", selectmode=tk.SINGLE)
        listbox.pack(fill=tk.X, padx=10, pady=4)

        def refresh_list():
            listbox.delete(0, tk.END)
            for i, r in enumerate(regions, 1):
                kinds = "/".join(k for k, label in (("monster", "怪物"), ("player", "人物")) if r.get(k))
                listbox.insert(tk.END, f"#{i}  x={r['x']} y={r['y']} w={r['w']} h={r['h']}  [{kinds}]")
            redraw()

        def toggle_select():
            selecting["active"] = not selecting["active"]
            select_btn.config(text="取消框选" if selecting["active"] else "开始框选")
            canvas.configure(cursor="crosshair" if selecting["active"] else "arrow")

        def delete_selected():
            sel = listbox.curselection()
            if sel:
                regions.pop(sel[0])
                refresh_list()

        select_btn.config(command=toggle_select)
        btns = tk.Frame(top, bg="#202026")
        btns.pack(fill=tk.X, padx=10, pady=(4, 10))
        tk.Button(btns, text="删除选中", command=delete_selected, bg="#6d4c41", fg="#ffffff").pack(side=tk.LEFT)
        tk.Button(btns, text="清空全部", command=lambda: (regions.clear(), refresh_list()), bg="#8e2424", fg="#ffffff").pack(side=tk.LEFT, padx=6)

        def apply_regions():
            # 复制一份可序列化的快照，避免 Tk 回调或后续列表操作影响已保存配置。
            saved_regions = []
            for r in regions:
                region_x = max(0, int(r.get("x", 0)))
                region_y = max(0, int(r.get("y", 0)))
                region_w = max(1, int(r.get("w", 1)))
                region_h = max(1, int(r.get("h", 1)))
                saved_regions.append({
                    "x": region_x, "y": region_y,
                    "w": region_w, "h": region_h,
                    "monster": bool(r.get("monster", False)),
                    "player": bool(r.get("player", False)),
                })
            self.config["recognition_exclusion_regions"] = saved_regions
            self._save_config()
            if self.detector is not None:
                self.detector.set_exclusion_regions(saved_regions)
            # 保存后立即回读校验，便于定位“界面显示有选区但 JSON 为空”的问题。
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    persisted = json.load(f).get("recognition_exclusion_regions", [])
                self.log(f"[特殊参数] 已应用 {len(saved_regions)} 个识别屏蔽区域，已写入 {CONFIG_PATH}（回读 {len(persisted)} 个）")
            except Exception as exc:
                self.log(f"⚠️ [特殊参数] 屏蔽区域保存校验失败: {exc}")
            top.destroy()

        tk.Button(btns, text="取消", command=top.destroy, width=10).pack(side=tk.RIGHT, padx=6)
        tk.Button(btns, text="应用", command=apply_regions, width=10, bg="#2e7d32", fg="#ffffff").pack(side=tk.RIGHT)
        refresh_list()

    def _clear_reconnect_monster_state(self):
        """Drop stale in-game targets while a login screen owns the capture."""
        with self._monster_batch_lock:
            self._monster_latest_batch = None
        with self._monster_fps_lock:
            self._monster_detection_fps = None
            self._monster_full_scan_fps = None
            self._monster_full_scan_gap_ms = None
            self._monster_full_scan_cost_ms = None
            self._monster_fps_count = 0
            self._monster_fps_timer = time.perf_counter()
        with self.lock:
            self.latest_result = MainViewResult(timestamp=time.perf_counter())
            self.latest_ladder_cols = []

    def _stop_for_reconnect(self):
        """断线确认后的原子急停；此方法不访问 Tk，可由重连线程调用。"""
        self._clear_reconnect_monster_state()
        try:
            self.input_driver.release_all_keys()
        except Exception:
            pass
        try:
            self.world_patrol_controller.stop()
        except Exception:
            pass
        try:
            self.combat_fsm.stop()
        except Exception:
            pass
        # stop() 只设置 Event；旧 F6 线程可能仍在当前动作中发送松键。
        # 密码框输入必须等它彻底退出，否则 Ctrl+A 会被旧线程打断。
        worker = getattr(self.combat_fsm, "thread", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        try:
            self.motion.stop()
        except Exception:
            pass
        try:
            self.input_driver.release_all_keys()
        except Exception:
            pass
        # 清除断线前的定位快照，防止人物选择后把旧坐标误当作“已进地图”。
        self.current_player_world_pos = None
        self.current_player_raw_world_pos = None
        self.current_player_platform = None
        self.current_player_is_climbing = False
        quiet = worker is None or not worker.is_alive()
        if not quiet:
            self.log("⚠️ [自动重连] F6 工作线程尚未退出，暂缓登录按键")
        return quiet

    def _get_death_return_town(self, source_map: Optional[int]) -> Optional[int]:
        if source_map is None:
            return None
        try:
            value = int(self.world_route_planner.load_map_data(source_map).get(
                "returnMap", 999999999,
            ))
            return value if 0 <= value < 999999999 else None
        except Exception:
            return None

    def _death_hp_state(self, frame) -> Optional[bool]:
        roi = self.config.get("status_bar_roi")
        if frame is None or not roi:
            return None
        reading = self.death_status_reader.read(
            frame, roi, run_ocr=True, ocr_hp_mp=True,
        )
        if not reading.roi_valid or not reading.layout_confident:
            return None
        if reading.hp_bar_percent is None:
            return None
        if reading.hp_bar_percent > 0.5:
            return False
        if reading.hp_current is None or reading.ocr_confidence < 0.70:
            return None
        return reading.hp_current == 0

    def _death_town_map_ready(self, map_id: int, expected_town: Optional[int]) -> bool:
        graph = getattr(self, "platform_graph", None)
        platform = getattr(self, "current_player_platform", None)
        try:
            is_town = bool(self.world_route_planner.load_map_data(map_id).get("isTown"))
        except Exception:
            is_town = False
        return bool(
            (map_id == expected_town or is_town)
            and
            self._get_current_map_id() == int(map_id)
            and graph is not None
            and int(getattr(graph, "map_id", -1)) == int(map_id)
            and platform is not None
            and graph.get_node(int(platform.id)) is not None
            and self.current_player_world_pos is not None
        )

    def _read_death_hp_stock(self, first_frame) -> Optional[int]:
        """Verify the HP-potion quantity twice after combat input has stopped."""
        roi = self.config.get("potion_bar_roi")
        hp_key = str(self.config.get("hp_potion_key", ""))
        mp_key = str(self.config.get("mp_potion_key", ""))
        if not roi or not supports_stock_key(hp_key):
            return None
        counts = []
        for index in range(2):
            if index:
                time.sleep(0.25)
                frame = self.capture.capture_frame(copy=True, include_overlay=False)
            else:
                frame = first_frame
            with self._potion_stock_eval_lock:
                reading = self.potion_stock_reader.read(frame, hp_key, mp_key, roi)
            counts.append(reading.hp_count if reading.layout_valid else None)
        return counts[0] if counts[0] is not None and counts[0] == counts[1] else None

    def _interrupt_for_death(self) -> bool:
        """Stop combat and keys but retain the world patrol's pre-death target."""
        self.combat_fsm.stop()
        self.motion.stop()
        self.input_driver.release_all_keys()
        self._stop_minigame_video_test_session("死亡弹窗急停")
        worker = getattr(self.combat_fsm, "thread", None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        quiet = worker is None or not worker.is_alive()
        if quiet:
            self.input_driver.release_all_keys()
        return quiet

    def _click_death_confirm(self) -> Optional[bool]:
        """Click only a freshly located Confirm button in a death modal."""
        frame = self.capture.capture_frame(copy=True, include_overlay=False) if self.capture else None
        if frame is None:
            return False
        point = self.death_dialog_detector.find_confirm_button(frame)
        if point is None:
            # A manual click can dismiss the popup while inventory is being
            # verified.  This is not a failed click; wait for the town map.
            return None if not self.death_dialog_detector.detect(frame) else False
        clicked = bool(self.input_driver.click_client(*point))
        if clicked:
            self.log(f"🖱️ [死亡确认点击] 客户区 X={point[0]} Y={point[1]}")
        return clicked

    def _halt_after_death(self) -> None:
        self.combat_fsm.stop_event.set()
        self.world_patrol_controller.stop()
        self.combat_fsm.stop()
        self.motion.stop()
        self.input_driver.release_all_keys()

    def _resume_after_death(self, town_map: int, death_map: Optional[int]) -> bool:
        if (
            self.stop_event.is_set() or self.reconnect_controller.active
            or not self.death_recovery_controller.active
        ):
            return False
        if self.combat_fsm.attack_only_mode:
            self.log("⚠️ [死亡返程] 仅攻击介入没有平台路线，禁止在主城盲目恢复攻击")
            return False
        # CombatFSM.stop() sets this shared cancellation event.  The existing
        # world-route executor needs it cleared before planning the return.
        self.combat_fsm.stop_event.clear()
        if not self.world_patrol_controller.begin_death_recovery(town_map, death_map):
            self.combat_fsm.stop_event.set()
            return False
        if not self.death_recovery_controller.active or self.reconnect_controller.active:
            self._halt_after_death()
            return False
        self.combat_fsm.start()
        self._start_minigame_video_test_session()
        self._reconnect_ui_queue.put(("death_recovery", "", False))
        return True

    def _toggle_random_path_test(self) -> None:
        """Start/stop the whole-map acceptance test from the path settings card."""
        runner = getattr(self, "random_path_test_runner", None)
        if runner is None:
            return
        if runner.active:
            runner.stop()
            return
        if self.combat_fsm.is_running:
            messagebox.showwarning(
                "无法开始测试",
                "请先按 F6 停止自动挂机，再开始随机全图行走测试。",
                parent=self.root,
            )
            return
        if self.ladder_grab_test_runner.active:
            messagebox.showwarning(
                "无法开始测试",
                "请先停止绳梯跳抓稳定性测试。",
                parent=self.root,
            )
            return
        draft_disabled = bool(self.var_disable_intra_map_portals.get())
        saved_disabled = bool(self.config.get("disable_intra_map_portals", False))
        if draft_disabled != saved_disabled:
            messagebox.showinfo(
                "请先应用路径设置",
                "“禁用地图内传送点”尚未应用。请先点击“应用”，再开始测试。",
                parent=self.root,
            )
            return
        ok, message = runner.start()
        if not ok:
            self.log(f"⛔ [随机路径测试未启动] {message}")
            messagebox.showwarning("随机全图行走测试", message, parent=self.root)
            return
        self.log(
            "🎲 [随机路径测试] 已启动；测试复用F6单边动作执行器，"
            "运行期间再次点击按钮可停止。"
        )
        self._on_random_path_test_status("正在生成覆盖路径…", True)

    def _on_random_path_test_status(self, text: str, active: bool) -> None:
        """Marshal random-path test status updates onto Tk's main thread."""
        def update() -> None:
            button = getattr(self, "_random_path_test_btn", None)
            if button is None or not button.winfo_exists():
                return
            button.config(
                text=(
                    "⏹ 停止随机行走测试 [F11]"
                    if active else "🎲 随机全图行走测试 [F11]"
                ),
                bg=("#b71c1c" if active else "#00695c"),
            )
            if text:
                self.lbl_cross_map_patrol_status.config(
                    text=text,
                    fg=("#80cbc4" if active else "#90a4ae"),
                )
        try:
            self.root.after(0, update)
        except Exception:
            pass

    def _prepare_reconnect_game_entry(self):
        """人物页确认前丢弃旧图快照，要求新地图重新完成 OCR 与拓扑定位。"""
        self.current_player_world_pos = None
        self.current_player_raw_world_pos = None
        self.current_player_platform = None
        self.current_player_is_climbing = False
        self.current_map_info = None
        self.platform_graph = None
        # MapResolver 对同 MapID 默认短路；重连后必须允许同一张图重新同步。
        self.map_resolver.current_map_id = None
        self.map_resolver.current_map_info = None
        self._portal_ocr_wakeup.set()

    def _reconnect_game_ready(self) -> bool:
        return bool(
            self.current_map_info
            and self.platform_graph is not None
            and self.current_player_raw_world_pos is not None
            and self.current_player_world_pos is not None
            and (self.current_player_platform is not None or self.current_player_is_climbing)
        )

    def _sync_reconnect_status_label(self):
        label = getattr(self, "lbl_reconnect_status", None)
        if label is None:
            return
        try:
            enabled = bool(self.config.get("reconnect_enabled", False))
            controller = getattr(self, "reconnect_controller", None)
            active = bool(controller and (controller.active or controller.starting or controller.disconnected))
            color = "#ffb74d" if active else ("#64b5f6" if enabled else "#78909c")
            label.config(text="🔌 " + self._reconnect_status_text, fg=color)
        except Exception:
            pass

    def _drain_reconnect_ui_updates(self):
        try:
            while True:
                kind, text, active = self._reconnect_ui_queue.get_nowait()
                if kind == "resume":
                    self._resume_after_reconnect()
                elif kind == "status":
                    self._reconnect_status_text = text
                    self._reconnect_status_active = bool(active)
                    if self.reconnect_controller.disconnected:
                        self._clear_reconnect_monster_state()
                    self._sync_reconnect_status_label()
                    self._sync_bot_ui_state()
                elif kind == "incident":
                    self._increment_session_incident(text)
                elif kind == "world_recovery":
                    self._sync_bot_ui_state()
                elif kind == "death_recovery":
                    self._death_recovery_status_text = text
                    self._death_recovery_status_active = bool(active)
                    self._sync_bot_ui_state()
        except queue.Empty:
            pass
        except Exception as exc:
            self.log(f"⚠️ [自动重连UI] {exc}")
        if not self.stop_event.is_set():
            self.root.after(80, self._drain_reconnect_ui_updates)

    def _resume_after_reconnect(self):
        if self.stop_event.is_set() or self.combat_fsm.is_running:
            return
        if not self.config.get("reconnect_enabled", False):
            self.log("ℹ️ [自动重连] 进入地图后发现重连开关已关闭，不恢复 F6")
            return
        self.log("▶️ [自动重连] 恢复断线前的 F6 配置")
        self.toggle_autobot()

    def _sync_bot_ui_state(self):
        try:
            if hasattr(self, "btn_autobot") and self.btn_autobot.winfo_exists():
                attack_only = bool(getattr(self, "attack_only_mode_var", None) and self.attack_only_mode_var.get())
                reconnecting = bool(
                    getattr(self, "reconnect_controller", None)
                    and (self.reconnect_controller.active or self.reconnect_controller.starting)
                )
                returning_home = bool(
                    getattr(self, "world_patrol_controller", None)
                    and self.world_patrol_controller.recovery_ui_active
                )
                if self._minigame_owns_input:
                    self.btn_autobot.config(text="等待小游戏结束[F6]", bg="#6a1b9a")
                    self.lbl_bot_state.config(text="● 小游戏中", fg="#ce93d8")
                elif self._death_recovery_status_active:
                    self.btn_autobot.config(text="🛑 取消死亡恢复 [F6]", bg="#b71c1c")
                    self.lbl_bot_state.config(
                        text="● " + (self._death_recovery_status_text or "死亡恢复中"),
                        fg="#ffab91",
                    )
                elif reconnecting:
                    self.btn_autobot.config(text="🛑 取消自动重连 [F6]", bg="#ef6c00")
                    self.lbl_bot_state.config(text="● 重连中", fg="#ffb74d")
                elif returning_home:
                    self.btn_autobot.config(
                        text="🛑 取消异常回城恢复 [F6]", bg="#ef6c00"
                    )
                    self.lbl_bot_state.config(text="● 返回原地图中", fg="#ffb74d")
                elif getattr(self, "_cross_map_start_pending", False):
                    self.btn_autobot.config(text="🛑 取消跨图路线校验 [F6]", bg="#ef6c00")
                    self.lbl_bot_state.config(text="● 跨图路线校验中", fg="#ffb74d")
                elif self.combat_fsm.is_running:
                    if getattr(self.combat_fsm, "attack_only_mode", False):
                        self.btn_autobot.config(text="🛑 停止攻击介入 [F6]", bg="#c62828")
                        self.lbl_bot_state.config(text="● 仅攻击介入", fg="#ff4081")
                    else:
                        self.btn_autobot.config(text="🛑 停止挂机 [F6]", bg="#c62828")
                        self.lbl_bot_state.config(text="● 巡逻中", fg="#ffb74d")
                else:
                    halted_text = (
                        self._death_recovery_status_text
                        if self.death_recovery_controller.input_suppressed else ""
                    )
                    if attack_only:
                        self.btn_autobot.config(text="⚔️ 启动仅攻击介入 [F6]", bg="#ad1457")
                        self.lbl_bot_state.config(
                            text="● " + halted_text if halted_text else "● 状态: 停止",
                            fg="#ffab91" if halted_text else "#90a4ae",
                        )
                    else:
                        self.btn_autobot.config(text="🚀 启动自动挂机 [F6]", bg="#2e7d32")
                        self.lbl_bot_state.config(
                            text="● " + halted_text if halted_text else "● 状态: 停止",
                            fg="#ffab91" if halted_text else "#90a4ae",
                        )
        except Exception:
            pass

    def toggle_autobot(self):
        if getattr(self, "death_recovery_controller", None) and self.death_recovery_controller.active:
            self.death_recovery_controller.cancel()
            self._halt_after_death()
            self._sync_bot_ui_state()
            self.log("🛑 [死亡恢复] 用户按 F6 取消复活后的自动返程")
            return
        if self._minigame_owns_input:
            if self.combat_fsm.is_running:
                self.world_patrol_controller.stop()
                self.combat_fsm.stop()
                self._minigame_resume_f6 = False
                self._stop_minigame_video_test_session("用户停止F6")
                self._sync_bot_ui_state()
                self.log("🛑 [测谎小游戏] F6 已手动停止，不会在小游戏结束后自动恢复。")
            elif self._minigame_resume_f6:
                self._minigame_resume_f6 = False
                self._stop_minigame_video_test_session("用户取消F6恢复")
                self._sync_bot_ui_state()
                self.log("ℹ️ [测谎小游戏] 已取消小游戏结束后的 F6 自动恢复。")
            else:
                self.log("ℹ️ [测谎小游戏] 正在控制鼠标，F6 暂不可启动。")
            return
        if (
            getattr(self, "reconnect_controller", None)
            and (self.reconnect_controller.active or self.reconnect_controller.starting)
        ):
            self.reconnect_controller.cancel("用户按下 F6")
            self._sync_bot_ui_state()
            return
        if getattr(self, "_cross_map_start_pending", False):
            cancel_event = getattr(self, "_cross_map_start_cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            self._cross_map_start_token = getattr(self, "_cross_map_start_token", 0) + 1
            self._cross_map_start_pending = False
            self._cross_map_start_cancel_event = None
            self._sync_bot_ui_state()
            self.log("🛑 [跨地图启动校验] 已取消，F6 未启动")
            return
        if not self.combat_fsm.is_running:
            if self.reconnect_controller.request_start_from_f6():
                self._sync_bot_ui_state()
                return
            if self.reconnect_controller.disconnected:
                self.log("⚠️ [F6 重连启动] 登录画面尚未重新确认，未启动地图巡逻")
                self._sync_bot_ui_state()
                return
            f6_pressed_at = time.perf_counter()
            if self.random_path_test_runner.active:
                self.log("⚠️ [F6 未启动] 随机全图行走测试正在复用F6导航执行器")
                messagebox.showwarning(
                    "无法启动 F6",
                    "请先停止随机全图行走测试，再启动自动挂机。",
                    parent=self.root,
                )
                return
            if self.ladder_grab_test_runner.active:
                self.log("⚠️ [F6 未启动] 绳梯跳抓验收正在复用 F6 导航执行器")
                messagebox.showwarning(
                    "无法启动 F6",
                    "请先停止绳梯跳抓稳定性测试，再启动自动挂机。",
                    parent=self.root,
                )
                return
            attack_only = bool(getattr(self, "attack_only_mode_var", None) and self.attack_only_mode_var.get())
            if attack_only:
                # 仅攻击键介入模式：移动由玩家完全自行控制，跳过路径与跨地图校验
                self.world_patrol_controller.configure(False, ())
                self.combat_fsm.attack_only_mode = True
                self.death_recovery_controller.arm()
                self.combat_fsm.start()
                self._start_minigame_video_test_session()
                self._sync_bot_ui_state()
                self.log("⚔️ [F6 热键] 启动【仅攻击键介入模式】（移动由玩家完全控制，怪物进入攻击范围自动攻击）！")
                return

            self.combat_fsm.attack_only_mode = False
            self.log(
                "🛡️ [换图战斗门禁] arrival_clear-v1 已加载："
                "新图先离开入口、清轨并连续识别人3次，再恢复攻击"
            )
            cross_map_enabled = bool(self.config.get("cross_map_patrol_enabled", False))
            if cross_map_enabled:
                stops = list(getattr(self, "cached_world_patrol_stops", ()))
                self._start_cross_map_patrol_async(stops, f6_started_at=f6_pressed_at)
                return
            else:
                self.world_patrol_controller.configure(False, ())
                # F6 启动前先校验整条循环路线；执行阶段再发现断边会导致
                # 角色在中间平台反复重规划，因此这里直接阻止启动。
                patrol = self._get_parsed_patrol_platforms()
                if len(patrol) >= 2:
                    graph = self.platform_graph
                    if graph is None or not graph.nodes:
                        self.log("⚠️ [路线校验] 当前地图拓扑尚未就绪，阻止启动自动挂机")
                        messagebox.showwarning(
                            "路线不可用",
                            "当前地图拓扑尚未加载完成，暂时无法验证循环平台是否连通。\n请等待地图同步完成后再按 F6。",
                            parent=self.root,
                        )
                        return

                    allow_run_jump = bool(self.config.get("enable_run_jump_grab", True))
                    allow_portal = not bool(
                        self.config.get("disable_intra_map_portals", False)
                    )
                    missing_nodes = [p for p in patrol if graph.get_node(p) is None]
                    unreachable = []
                    if not missing_nodes:
                        for src, dst in zip(patrol, patrol[1:] + [patrol[0]]):
                            if src == dst:
                                continue
                            path = graph.find_path(
                                src,
                                dst,
                                allow_run_jump=allow_run_jump,
                                allow_portal=allow_portal,
                            )
                            if not path:
                                unreachable.append((src, dst))

                    if missing_nodes or unreachable:
                        details = []
                        if missing_nodes:
                            details.append("不存在平台: " + ", ".join(f"P{x}" for x in missing_nodes))
                        if unreachable:
                            details.append("不可达区段: " + ", ".join(f"P{a} → P{b}" for a, b in unreachable))
                        mode = "启用跑跳抓取" if allow_run_jump else "关闭跑跳抓取"
                        self.log(f"⚠️ [路线校验] {mode} 下路线不可达，阻止 F6 启动：" + "；".join(details))
                        messagebox.showwarning(
                            "目标平台不可达",
                            "当前循环平台路线存在不可达目标，自动挂机未启动。\n"
                            + "\n".join(details)
                            + f"\n当前模式：{mode}\n请修改平台列表或调整跑跳抓取开关。",
                            parent=self.root,
                        )
                        return

                # 单地图 F6 也保存地图级恢复目标。异常死亡回城、脚本传送
                # 或其它非预期换图后，跨图控制器会暂停战斗并沿 type=1/2
                # 普通门返回这里；不要求用户把城镇加入循环列表。
                home_map_id = self._get_current_map_id()
                home_platforms = tuple(patrol)
                if not home_platforms and self.current_player_platform is not None:
                    home_platforms = (int(self.current_player_platform.id),)
                if home_map_id is not None and home_platforms:
                    self.world_patrol_controller.arm_home_recovery(
                        WorldPatrolStop(int(home_map_id), home_platforms)
                    )
                    self.log(
                        f"🏠 [异常回城恢复] 已记录 F6 家地图 MapID {home_map_id}，"
                        "异常换图后将自动规划返回"
                    )

            self.death_recovery_controller.arm()
            self.combat_fsm.start(f6_started_at=f6_pressed_at)
            self._start_minigame_video_test_session()
            self._sync_bot_ui_state()
            mode_text = "跨地图巡逻" if cross_map_enabled else "自动挂机与索敌"
            self.log(f"🚀 [F6 热键] 启动{mode_text}！")
        else:
            was_attack_only = getattr(self.combat_fsm, "attack_only_mode", False)
            self.world_patrol_controller.stop()
            self.combat_fsm.stop()
            self._stop_minigame_video_test_session("用户停止F6")
            self._sync_bot_ui_state()
            if was_attack_only:
                self.log("🛑 [F6 热键] 停止仅攻击键介入。")
            else:
                self.log("🛑 [F6 热键] 停止自动挂机。")

    def _start_cross_map_patrol_async(self, stops, f6_started_at=None):
        """在后台校验跨图路径；地图搜索不能阻塞 Tk 事件循环。"""
        if f6_started_at is None:
            f6_started_at = time.perf_counter()
        current_map = self._get_current_map_id()
        if current_map is None:
            self.log("⚠️ [跨地图启动失败] 当前地图尚未定位")
            messagebox.showwarning("跨地图巡逻未启动", "当前地图尚未定位", parent=self.root)
            return
        stops = tuple(stops)
        token = getattr(self, "_cross_map_start_token", 0) + 1
        cancel_event = threading.Event()
        self._cross_map_start_token = token
        self._cross_map_start_cancel_event = cancel_event
        self._cross_map_start_pending = True
        self._sync_bot_ui_state()
        self.log(f"🌐 [跨地图启动校验] 后台检查 MapID {current_map} 到首个目标图的路线")
        allow_run_jump = bool(self.config.get("enable_run_jump_grab", True))
        allow_intra_map_portal = not bool(self.config.get("disable_intra_map_portals", False))

        def worker():
            started = time.perf_counter()
            errors = []
            route = None
            try:
                errors = self.world_route_planner.validate_stops(
                    stops,
                    allow_run_jump=allow_run_jump,
                    allow_intra_map_portal=allow_intra_map_portal,
                )
                if not errors and not cancel_event.is_set() and int(current_map) not in {
                    stop.map_id for stop in stops
                }:
                    route = self.world_route_planner.find_map_route(
                        int(current_map), stops[0].map_id, cancel_event=cancel_event
                    )
            except Exception as exc:
                errors = [f"跨图路线校验异常：{exc}"]
            if cancel_event.is_set():
                return
            elapsed = time.perf_counter() - started
            try:
                self.root.after(
                    0,
                    lambda: self._finish_cross_map_patrol_start(
                        token, int(current_map), stops, errors, route, elapsed,
                        allow_run_jump, allow_intra_map_portal, f6_started_at,
                    ),
                )
            except (RuntimeError, tk.TclError):
                pass

        threading.Thread(target=worker, name="CrossMapF6Preflight", daemon=True).start()

    def _finish_cross_map_patrol_start(
        self, token, current_map, stops, errors, route, elapsed,
        allow_run_jump, allow_intra_map_portal, f6_started_at,
    ):
        """只接受最新一次、且仍对应当前地图和设置的预检结果。"""
        if token != getattr(self, "_cross_map_start_token", 0):
            return
        self._cross_map_start_pending = False
        self._cross_map_start_cancel_event = None
        self._sync_bot_ui_state()
        if (
            self.combat_fsm.is_running
            or not self.config.get("cross_map_patrol_enabled", False)
            or tuple(getattr(self, "cached_world_patrol_stops", ())) != stops
            or self._get_current_map_id() != current_map
            or bool(self.config.get("enable_run_jump_grab", True)) != allow_run_jump
            or (not bool(self.config.get("disable_intra_map_portals", False)))
            != allow_intra_map_portal
        ):
            self.log("ℹ️ [跨地图启动校验] 地图或巡逻设置已变化，丢弃过期路线")
            return
        self.log(f"⏱️ [跨地图启动校验] 耗时 {elapsed:.2f}s，界面未等待搜索")
        if errors:
            details = "\n".join(errors)
            self.log(f"⚠️ [跨地图路线校验] 阻止F6启动：{'；'.join(errors)}")
            messagebox.showwarning(
                "跨地图路线不可用",
                details + "\n请在“路径设置”中修正后点击“应用”。",
                parent=self.root,
            )
            return
        if current_map not in {stop.map_id for stop in stops} and not route:
            message = f"当前地图 {current_map} 无法到达首个目标地图 {stops[0].map_id}"
            self.log(f"⚠️ [跨地图启动失败] {message}")
            messagebox.showwarning("跨地图巡逻未启动", message, parent=self.root)
            return
        self.world_patrol_controller.configure(True, stops)
        ok, message = self.world_patrol_controller.start(
            initial_route=route, f6_started_at=f6_started_at,
        )
        if not ok:
            self.log(f"⚠️ [跨地图启动失败] {message}")
            messagebox.showwarning("跨地图巡逻未启动", message, parent=self.root)
            return
        self.log(f"🌐 [跨地图巡逻启动] {message}")
        self.death_recovery_controller.arm()
        self.combat_fsm.start(f6_started_at=f6_started_at)
        self._start_minigame_video_test_session()
        self._sync_bot_ui_state()
        self.log("🚀 [F6 热键] 启动跨地图巡逻！")

    def toggle_waypoint_recording(self):
        if not self.waypoint_mgr.is_recording:
            map_id = self.current_map_info.get("map_id", 0) if self.current_map_info else 0
            self.waypoint_mgr.start_recording(map_id)
            self.btn_rec_route.config(text="⏹️ 停止录制 [F7]", bg="#ad1457")
            self.lbl_route_status.config(text="路线状态: 正在录制中 (跑动一圈即可)...", fg="#ff4081")
            self.log("⏺️ [F7 热键] 开始录制当前地图巡逻路线！请在游戏内正常跑动一圈...")
        else:
            cnt = self.waypoint_mgr.stop_recording(auto_save=True)
            self.btn_rec_route.config(text="⏺️ 录制路线 [F7]", bg="#4527a0")
            self.lbl_route_status.config(text=f"路线状态: 录制完成，共 {cnt} 个航点 (已保存)", fg="#69f0ae")
            self.log(f"💾 [F7 热键] 录制完成并保存！共记录 {cnt} 个巡逻航点。")

    def _insert_custom_action(self, action: str):
        if self.detector and self.detector.last_player_pos:
            self.waypoint_mgr.add_manual_action(action, self.detector.last_player_pos)
            self.log(f"➕ [插入动作] 在当前位置插入动作航点: 【{action}】")
            self.lbl_route_status.config(text=f"路线状态: 已添加动作【{action}】，共 {len(self.waypoint_mgr.waypoints)} 点", fg="#00e5ff")
        else:
            messagebox.showinfo("提示", "请先在游戏内移动并定位角色坐标！")

    def _save_current_route(self):
        if not self.waypoint_mgr.waypoints:
            messagebox.showinfo("提示", "当前没有录制任何路线航点！")
            return
        map_id = self.current_map_info.get("map_id", 0) if self.current_map_info else 0
        ok = self.waypoint_mgr.save_route(map_id)
        if ok:
            self.log(f"💾 [路线保存] 成功保存地图【{map_id}】路线（共 {len(self.waypoint_mgr.waypoints)} 点）！")
            messagebox.showinfo("成功", f"成功保存路线！共包含 {len(self.waypoint_mgr.waypoints)} 个航点。")

    def _load_current_route(self):
        map_id = self.current_map_info.get("map_id", 0) if self.current_map_info else 0
        ok = self.waypoint_mgr.load_route(map_id)
        if ok:
            self.lbl_route_status.config(text=f"路线状态: 已载入本地路线 ({len(self.waypoint_mgr.waypoints)} 个航点)", fg="#69f0ae")
            self.log(f"📂 [路线载入] 成功从本地载入地图【{map_id}】路线（共 {len(self.waypoint_mgr.waypoints)} 点）！")
            messagebox.showinfo("成功", f"成功载入路线！共包含 {len(self.waypoint_mgr.waypoints)} 个航点。")
        else:
            messagebox.showwarning("提示", f"未找到地图 ID 【{map_id}】的本地路线文件，请先点击【录制路线】！")

    def _build_mob_card(self):

        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf),
                              text="👾 本图怪物列表 (可自定义 / 悬浮看图)",
                              font=("Segoe UI", 10, "bold"), fg="#ffb74d", bg="#202026", padx=8, pady=6)
        card.pack(fill=tk.X, padx=8, pady=3)

        # 第一列是模板启用开关。Treeview 没有原生 checkbox，因此用
        # ☑/☐ 绘制成可点击的勾选列；地图刷新时从 local_maps.json 恢复。
        cols = ("enabled", "id", "name", "tpl_count", "skill", "edit")
        self.mob_tree = ttk.Treeview(card, columns=cols, show="headings", height=4)
        self.mob_tree.heading("enabled", text="启用")
        self.mob_tree.heading("id", text="Mob ID")
        self.mob_tree.heading("name", text="怪物名称")
        self.mob_tree.heading("tpl_count", text="模板数量")
        self.mob_tree.heading("skill", text="攻击技能")
        self.mob_tree.heading("edit", text="操作")
        self.mob_tree.column("enabled", width=36, anchor=tk.CENTER, stretch=False)
        self.mob_tree.column("id", width=68, anchor=tk.CENTER)
        self.mob_tree.column("name", width=118, anchor=tk.W)
        self.mob_tree.column("tpl_count", width=78, anchor=tk.CENTER)
        self.mob_tree.column("skill", width=90, anchor=tk.CENTER)
        self.mob_tree.column("edit", width=45, anchor=tk.CENTER, stretch=False)
        self.mob_tree.pack(fill=tk.X, pady=2)
        self.mob_enabled_by_id: Dict[str, bool] = {}
        self.mob_rows_by_id: Dict[str, Dict] = {}
        self.mob_tree.bind("<Button-1>", self._on_mob_tree_click, add="+")
        self.mob_tooltip = MonsterTooltip(self.mob_tree, self._get_mob_preview_image)

        tk.Button(card, text="✏️ 自定义修改当前怪物列表（增 / 删 / 改 / 搜索）",
                  font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#0277bd",
                  relief=tk.FLAT, padx=8, pady=3,
                  command=self.open_edit_monster_dialog).pack(fill=tk.X, pady=(2, 0))

    def _toggle_video_viewport(self):
        if not hasattr(self, "video_canvas"):
            return
        if self.video_visible_var.get():
            self._viewport_render_enabled = True
            self.video_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        else:
            self._viewport_render_enabled = False
            self.video_canvas.place_forget()

    def _on_video_container_configure(self, event) -> None:
        """让预览完整等比装入容器，避免固定 800px 图像在窄栏中被裁边。"""
        width = max(100, int(event.width) - 4)
        height = max(100, int(event.height) - 4)
        self._viewport_target_size = (width, height)

    def _on_mob_tree_click(self, event):
        """处理怪物启用开关及每行模板编辑入口。"""
        if self.mob_tree.identify_region(event.x, event.y) != "cell":
            return
        item_id = self.mob_tree.identify_row(event.y)
        if not item_id:
            return
        values = self.mob_tree.item(item_id, "values")
        if len(values) < 2:
            return
        column = self.mob_tree.identify_column(event.x)
        mob_id = str(values[1])
        if column == "#6":
            mob_name = str(values[2]) if len(values) > 2 else ""
            self.root.after(0, lambda: self._open_mob_template_editor(mob_id, mob_name))
            return "break"
        if column == "#5":
            self.root.after(0, lambda: self._open_mob_skill_editor(mob_id))
            return "break"
        if column != "#1":
            return
        self.mob_enabled_by_id[mob_id] = not self.mob_enabled_by_id.get(mob_id, True)
        enabled = self.mob_enabled_by_id[mob_id]
        self.mob_tree.set(item_id, "enabled", "☑" if enabled else "☐")
        map_id = (self.current_map_info or {}).get("map_id")
        if map_id is not None:
            disabled = [key for key, on in self.mob_enabled_by_id.items() if not on]
            if self.current_map_info is not None:
                self.current_map_info["disabled_mob_ids"] = [int(x) for x in disabled]
            self.map_resolver.save_local_map_disabled_mobs(
                map_id,
                disabled,
                (self.current_map_info or {}).get("chinese_name", ""),
            )
        selected = [m for key, m in self.mob_rows_by_id.items()
                    if self.mob_enabled_by_id.get(key, True)]
        self.detector.load_dynamic_monster_templates(selected)
        self.log(f"[怪物模板] Mob {mob_id} {'已启用' if enabled else '已停用'}")

    def _populate_mob_tree(self, mobs: List[Dict], disabled_ids=None) -> None:
        """刷新本图怪物列表，并恢复 local_maps.json 中已关闭的 Mob。"""
        disabled = {str(x) for x in (disabled_ids or [])}
        self.mob_enabled_by_id = {
            str(m.get("id")): str(m.get("id")) not in disabled for m in mobs
        }
        self.mob_rows_by_id = {str(m.get("id")): m for m in mobs}
        for it in self.mob_tree.get_children():
            self.mob_tree.delete(it)
        for m in mobs:
            mob_id = str(m.get("id"))
            cnt = self._count_mob_templates(m.get("id"), m.get("name"))
            cnt_str = f"{cnt} 张 (✔)" if cnt > 0 else "0 张 (待下载)"
            enabled = self.mob_enabled_by_id.get(mob_id, True)
            self.mob_tree.insert(
                "", tk.END,
                values=("☑" if enabled else "☐", mob_id, m.get("name"), cnt_str,
                        self._mob_skill_label(mob_id), "编辑"),
            )

    def _mob_skill_label(self, mob_id: str) -> str:
        rules = self.config.get("monster_skill_rules", {}) or {}
        allowed = rules.get(str(mob_id)) if isinstance(rules, dict) else None
        if allowed is None:
            return "主攻击（默认）"
        names = {skill["id"]: skill["name"] for skill in skills_from_config(self.config)}
        return " / ".join(names[skill_id] for skill_id in allowed if skill_id in names) or "未设置"

    def _open_mob_skill_editor(self, mob_id: str) -> None:
        """Set allowed attack skills for one real Mob ID, not a temporary track ID."""
        win = tk.Toplevel(self.root)
        win.title(f"Mob {mob_id} 攻击技能")
        fit_window_to_work_area(win, (360, 320), (300, 240), parent=self.root)
        win.transient(self.root)
        win.grab_set()
        tk.Label(win, text=f"Mob ID {mob_id} 可使用的技能（至少一个）",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 5))
        rules = self.config.get("monster_skill_rules", {}) or {}
        selected = rules.get(str(mob_id), ["primary"])
        variables = {}
        for skill in skills_from_config(self.config):
            var = tk.BooleanVar(value=skill["id"] in selected)
            variables[skill["id"]] = var
            tk.Checkbutton(win, text=f"{skill['name']}  [{skill['key'].upper()}]",
                           variable=var).pack(anchor="w", padx=18)
        tk.Label(win, text="未单独指定的怪物始终只用主攻击。",
                 fg="#666666").pack(anchor="w", padx=12, pady=8)

        def save():
            chosen = [skill_id for skill_id, var in variables.items() if var.get()]
            if not chosen:
                messagebox.showerror("技能设置", "至少选择一个技能；如需完全忽略该怪物，请关闭其识别。", parent=win)
                return
            updated = dict(self.config.get("monster_skill_rules", {}) or {})
            if chosen == ["primary"]:
                updated.pop(str(mob_id), None)
            else:
                updated[str(mob_id)] = chosen
            self.config["monster_skill_rules"] = updated
            self._save_config()
            for item in self.mob_tree.get_children():
                if str(self.mob_tree.set(item, "id")) == str(mob_id):
                    self.mob_tree.set(item, "skill", self._mob_skill_label(mob_id))
                    break
            self.log(f"⚔️ [怪物技能规则] Mob {mob_id}: {', '.join(chosen)}")
            win.destroy()

        tk.Button(win, text="保存", command=save, bg="#2e7d32", fg="white").pack(
            fill=tk.X, padx=12, pady=(4, 12))

    def _open_mob_template_editor(self, mob_id: str, mob_name: str) -> None:
        """打开单个怪物的模板对查看/删减/特征框编辑器。"""
        MobTemplateEditor(
            self.root,
            mob_id=mob_id,
            mob_name=mob_name,
            template_root=getattr(self.detector, "multi_scale_template_root", ""),
            on_changed=lambda action: self._on_mob_templates_edited(mob_id, action),
        )

    def _on_mob_templates_edited(self, mob_id: str, action: str) -> None:
        """模板落盘后刷新数量、缩略图，并让当前检测器立即采用新模板。"""
        prefix = f"{mob_id}_"
        for key in list(self.cached_mob_thumbnails):
            if key.startswith(prefix):
                self.cached_mob_thumbnails.pop(key, None)
        for item_id in self.mob_tree.get_children():
            values = self.mob_tree.item(item_id, "values")
            if len(values) >= 2 and str(values[1]) == str(mob_id):
                row = self.mob_rows_by_id.get(str(mob_id), {})
                count = self._count_mob_templates(mob_id, str(row.get("name", "")))
                self.mob_tree.set(item_id, "tpl_count", f"{count} 张 (✔)" if count else "0 张 (待下载)")
                break
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.detector.load_dynamic_monster_templates(self._get_enabled_mobs())
        finally:
            self.root.config(cursor="")
        self.log(f"[怪物模板编辑] Mob {mob_id} 已{action}模板，当前检测器已热重载")

    def _get_enabled_mobs(self) -> List[Dict]:
        return [m for key, m in self.mob_rows_by_id.items()
                if self.mob_enabled_by_id.get(key, True)]

    def _has_current_map_monsters(self) -> bool:
        """本图怪物列表是所有怪物识别来源的总开关。"""
        return bool(getattr(self, "mob_rows_by_id", {}))

    def _monster_pipeline_state(self) -> Tuple[bool, bool]:
        """返回 ``(本图列表非空, 当前存在可运行的怪物识别来源)``。"""
        has_map_monsters = self._has_current_map_monsters()
        detector = getattr(self, "detector", None)
        monster_enabled = bool(self.config.get("enable_monster_detection", True))
        hp_bar_enabled = bool(self.config.get("enable_monster_hp_bar_detection", True))
        has_source = bool(
            has_map_monsters
            and (
                (monster_enabled and detector and detector.monster_cached_scaled_templates)
                or (
                    hp_bar_enabled
                    and detector
                    and getattr(detector, "enable_monster_hp_bar_detection", True)
                )
            )
        )
        return has_map_monsters, has_source

    def _build_map_card(self):
        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf), text="🗺️ 地图管理",
                              font=("Segoe UI", 10, "bold"), fg="#00e5ff", bg="#202026", padx=6, pady=4)
        card.pack(fill=tk.X, padx=8, pady=1)
        
        row_name = tk.Frame(card, bg="#202026")
        row_name.pack(fill=tk.X, pady=1)
        tk.Label(row_name, text="当前地图:", font=("Segoe UI", 9, "bold"),
                 fg="#ffff00", bg="#202026").pack(side=tk.LEFT, padx=(0, 5))
        self.ent_map_name = tk.Entry(row_name, font=("Segoe UI", 9, "bold"),
                                     bg="#2a2a32", fg="#00e676", insertbackground="white")
        self.ent_map_name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.ent_map_name.insert(0, "正在全自动探测中...")

        tk.Label(row_name, text="MapID:", font=("Consolas", 8),
                 fg="#b0bec5", bg="#202026").pack(side=tk.LEFT, padx=(0, 5))
        self.ent_map_id = tk.Entry(row_name, font=("Consolas", 9), width=11,
                                   bg="#2a2a32", fg="#00e5ff", insertbackground="white")
        self.ent_map_id.pack(side=tk.LEFT)
        # MapID 初始留空，避免启动时误显示默认地图。
        
        # 常用操作使用等宽小按钮横向排列，避免地图卡片占据三整行高度。
        actions = tk.Frame(card, bg="#202026")
        actions.pack(fill=tk.X, pady=(3, 1))
        for column in range(4):
            actions.columnconfigure(column, weight=1, uniform="map_action")
        saved_roi = self.config.get("map_ocr_roi")
        action_specs = (
            ("↻ OCR", "#0277bd", self.on_force_ocr),
            ("✓ 应用", "#2e7d32", self._on_apply_map_changes),
            ("▣ OCR范围✓" if saved_roi else "▣ OCR范围", "#00838f", self._on_calibrate_ocr_roi_clicked),
            ("⌖ 雷达 F8", "#e65100", self.open_minimap_radar_dialog),
        )
        for column, (text, colour, command) in enumerate(action_specs):
            button = tk.Button(
                actions, text=text, font=("Segoe UI", 8, "bold"),
                fg="#ffffff", bg=colour, relief=tk.FLAT, bd=0,
                highlightthickness=0, padx=3, pady=0,
                command=command,
            )
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 2, 0))
            if column == 2:
                self.btn_map_ocr_roi = button

    def _build_player_calibration_card(self):
        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf), text="👤 角色自适应标定 (支持任意名牌/时装)",
                              font=("Segoe UI", 10, "bold"), fg="#e040fb", bg="#202026", padx=8, pady=6)
        card.pack(fill=tk.X, padx=8, pady=3)

        # 👑 两段式高精度标定（全身整体 + 核心特征 + 纯视觉朝向）
        r_two_stage = tk.Frame(card, bg="#202026")
        r_two_stage.pack(fill=tk.X, pady=(2, 4))
        btn_two_stage = tk.Button(r_two_stage, text="👑 特征标定",
                                  font=("Segoe UI", 9, "bold"), fg="#ffffff", bg="#4a148c",
                                  relief=tk.FLAT, padx=6, pady=4,
                                  command=self._on_two_stage_crop_clicked)
        btn_two_stage.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.var_enable_two_stage = tk.BooleanVar(value=bool(self.config.get("enable_two_stage_feature", True)))
        self.chk_enable_two_stage = tk.Checkbutton(
            r_two_stage,
            text="启用特征识别",
            variable=self.var_enable_two_stage,
            font=("Segoe UI", 9, "bold"),
            fg="#ffd54f",
            bg="#202026",
            selectcolor="#2b2b36",
            activebackground="#202026",
            activeforeground="#ffe082",
            relief=tk.FLAT,
            command=self._on_toggle_two_stage_feature
        )
        self.chk_enable_two_stage.pack(side=tk.RIGHT, padx=(2, 0))

        two_stage_text = "两段式标定: 未标定 (推荐点击上方紫按钮)"
        if self.detector and getattr(self.detector, "two_stage_profile", None):
            prof = self.detector.two_stage_profile
            cw, ch = prof.get("char_w", 0), prof.get("char_h", 0)
            fw, fh = prof.get("feature_w", 0), prof.get("feature_h", 0)
            two_stage_text = f"两段式标定: ✅ 全身 {cw}x{ch} | 特征 {fw}x{fh} | 纯视觉朝向"

        self.lbl_two_stage_status = tk.Label(card, text=two_stage_text,
                                             font=("Consolas", 8, "bold"), fg="#ffd54f", bg="#202026", anchor=tk.W)
        self.lbl_two_stage_status.pack(fill=tk.X, pady=(1, 4))

        # 传统辅助标定行
        r_calib = tk.Frame(card, bg="#202026")
        r_calib.pack(fill=tk.X, pady=2)
        btn_manual = tk.Button(r_calib, text="✂️ 标定名牌 (可选)",
                               font=("Segoe UI", 8), fg="#b0bec5", bg="#2a2a32",
                               relief=tk.FLAT, padx=6, pady=3,
                               command=self._on_manual_crop_clicked)
        btn_manual.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_auto = tk.Button(r_calib, text="⚡ 自动提取名牌",
                             font=("Segoe UI", 8), fg="#b0bec5", bg="#2a2a32",
                             relief=tk.FLAT, padx=6, pady=3,
                             command=self._on_calibrate_player_clicked)
        btn_auto.pack(side=tk.RIGHT)

        self.lbl_calib_status = tk.Label(card, text="名牌状态: 已加载标准抗遮挡名牌",
                                         font=("Consolas", 8), fg="#69f0ae", bg="#202026", anchor=tk.W)
        self.lbl_calib_status.pack(fill=tk.X, pady=(2, 0))

    def _on_manual_crop_clicked(self):
        """点击手动框选标定角色"""
        frame = self.capture.capture_frame()
        if frame is not None and frame.size > 0:
            ManualCropPlayerDialog(self.root, frame, self._on_player_cropped_confirmed)
        else:
            messagebox.showwarning("提示", "未获取到游戏画面，请确保游戏已运行且已捕获！")

    def _on_player_cropped_confirmed(self, crop_bgr: np.ndarray):
        """用户框选完成回调"""
        if self.detector is not None:
            ok = self.detector.set_manual_cropped_player(crop_bgr)
            if ok:
                h, w = crop_bgr.shape[:2]
                self.lbl_calib_status.config(text=f"✅ 手动标定成功! 尺寸: {w}x{h} px", fg="#69f0ae")
                self.log(f"✅ [手动标定] 成功保存框选角色名牌 ({w}x{h} px)，已热重载多部件抗遮挡特征！")
                messagebox.showinfo("标定成功", f"成功应用框选的名牌特征 ({w}x{h} px)！\n系统已完成特征热重载。")
            else:
                self.lbl_calib_status.config(text="⚠️ 选区无效，请重新框选", fg="#ff5252")
                self.log("⚠️ [手动标定] 选区尺寸无效，请重新框选。")

    def _on_calibrate_player_clicked(self):
        """点击快速自动标定角色"""
        frame = self.capture.capture_frame()
        if frame is not None and self.detector is not None:
            ok = self.detector.calibrate_player_from_frame(frame)
            if ok:
                self.lbl_calib_status.config(text="✅ 自动标定成功! 已热重载", fg="#69f0ae")
                self.log("✅ [自动标定] 成功从当前画面自动提取名字牌与勋章，已热重载！")
                messagebox.showinfo("标定成功", "成功从当前画面自动提取到角色名字牌与勋章！\n系统已完成特征热重载。")
            else:
                self.lbl_calib_status.config(text="⚠️ 自动未找到名牌，请使用上方【手动框选】", fg="#ff5252")
                self.log("⚠️ [自动标定] 未能自动识别到标准天蓝名牌，建议直接点击【✂️ 手动框选标定】。")
                messagebox.showwarning("标定提示", "未能自动识别到标准天蓝色名字牌。\n建议直接点击【✂️ 手动框选标定当前角色】拉框框选！")
        else:
            messagebox.showwarning("提示", "未获取到游戏画面，请确保游戏已运行且已捕获！")

    def _on_calibrate_ocr_roi_clicked(self):
        """点击标定地图 OCR 识别范围（鼠标拖拽拉框）"""
        frame = self.capture.capture_frame()
        if frame is not None and frame.size > 0:
            current_roi = self.config.get("map_ocr_roi")
            OcrRoiCalibrationDialog(self.root, frame, current_roi, self._on_ocr_roi_confirmed)
        else:
            messagebox.showwarning("提示", "未获取到游戏画面，请确保游戏已运行且已捕获！")

    def _on_ocr_roi_confirmed(self, roi: Dict[str, int]):
        """OCR 识别范围框选保存回调"""
        self.config["map_ocr_roi"] = roi
        self._save_config()
        if hasattr(self, "map_resolver") and self.map_resolver is not None:
            self.map_resolver.set_ocr_roi(roi)

        rx, ry, rw, rh = roi.get("x", 0), roi.get("y", 0), roi.get("w", 0), roi.get("h", 0)
        if hasattr(self, "btn_map_ocr_roi"):
            self.btn_map_ocr_roi.config(text="▣ OCR范围✓")

        self.log(f"✅ [OCR范围] 成功保存小地图标题识别范围: [x={rx}, y={ry}, w={rw}, h={rh}]")
        messagebox.showinfo(
            "OCR范围标定成功",
            f"🎉 小地图 OCR 识别范围已成功保存！\n\n"
            f"• 起点: ({rx}, {ry})\n"
            f"• 尺寸: {rw} x {rh} 像素\n\n"
            f"系统已立即热更新 OCR 识别器与后台哨兵线程，有效避开小地图外部 NPC 干扰！"
        )

    def _on_manual_crop_hat_clicked(self):
        """点击手动框选标定服饰/魔法帽特征 (双重保险)"""
        frame = self.capture.capture_frame()
        if frame is not None and frame.size > 0:
            ManualCropPlayerDialog(
                self.root, frame, self._on_hat_cropped_confirmed,
                title="🎩 鼠标拖拽框选服饰 / 魔法帽特征 (双重防误判保险)",
                tip_text="🎯 提示：请框选角色的头部服饰、魔法帽、发型或上衣特征，松开后点击确认！"
            )
        else:
            messagebox.showwarning("提示", "未获取到游戏画面，请确保游戏已运行且已捕获！")

    def _on_hat_cropped_confirmed(self, crop_bgr: np.ndarray):
        """服饰框选完成回调"""
        if self.detector is not None:
            ok = self.detector.set_manual_cropped_hat(crop_bgr)
            if ok:
                h, w = crop_bgr.shape[:2]
                self.lbl_hat_status.config(text=f"服饰保险: ✅ 已加载 ({w}x{h} px)", fg="#80deea")
                self.log(f"✅ [服饰标定] 成功保存服饰/魔法帽特征 ({w}x{h} px)，已激活双重防误判与遮挡挽救保险！")
                messagebox.showinfo("标定成功", f"成功应用服饰/魔法帽特征 ({w}x{h} px)！\n系统已开启双重保险核验与转向镜像自适应。")
            else:
                self.lbl_hat_status.config(text="⚠️ 选区无效，请重新框选", fg="#ff5252")
                self.log("⚠️ [服饰标定] 选区尺寸无效，请重新框选。")

    def _on_clear_hat_clicked(self):
        """清除服饰特征"""
        if self.detector is not None:
            self.detector.clear_costume_template()
            self.lbl_hat_status.config(text="服饰保险: 未标定 (仅名牌定位)", fg="#80deea")
            self.log("ℹ️ [服饰标定] 已清除服饰特征，恢复纯名牌定位模式。")
            messagebox.showinfo("清除成功", "已清除服饰/魔法帽特征，恢复为纯名字牌定位模式。")

    def _on_two_stage_crop_clicked(self):
        """点击两段式角色标定（全身整体 + 核心特征 + 朝向选择）"""
        frame = self.capture.capture_frame()
        if frame is not None and frame.size > 0:
            TwoStagePlayerCalibrationDialog(self.root, frame, self._on_two_stage_calibration_confirmed)
        else:
            messagebox.showwarning("提示", "未获取到游戏画面，请确保游戏已运行且已捕获！")

    def _on_two_stage_calibration_confirmed(
        self,
        whole_bgr,
        feat_bgr,
        feat_box,
        facing,
        feature_mask=None,
        feature_polygon=None,
    ):
        """两段式标定完成回调"""
        if self.detector is not None:
            ok = self.detector.save_two_stage_calibration(
                whole_bgr,
                feat_bgr,
                feat_box,
                facing,
                feature_mask=feature_mask,
                feature_polygon=feature_polygon,
            )
            if ok:
                cw, ch = whole_bgr.shape[1], whole_bgr.shape[0]
                fw, fh = feat_bgr.shape[1], feat_bgr.shape[0]
                facing_name = "朝右" if facing == "right" else "朝左"
                self.lbl_two_stage_status.config(
                    text=f"两段式标定: ✅ 全身 {cw}x{ch} | 多边形特征 {fw}x{fh} | 标定时{facing_name}",
                    fg="#ffd54f"
                )
                self.lbl_calib_status.config(text=f"名牌状态: 双模并行 (全身包围盒 {cw}x{ch} px)", fg="#69f0ae")
                self.log(f"✅ [两段式标定] 成功保存角色全身 ({cw}x{ch}) 与多边形特征 ({fw}x{fh}, {facing_name})，纯视觉朝向已生效！")
                messagebox.showinfo(
                    "两段式标定成功",
                    f"🎉 角色全身与核心特征标定成功！\n\n"
                    f"• 全身包围盒: {cw} x {ch} px (脚底严格对齐框底)\n"
                    f"• 核心特征外接框: {fw} x {fh} px（仅多边形内部参与匹配）\n"
                    f"• 标定朝向: {facing_name} (系统已自动生成反向镜像)\n\n"
                    f"💡 纯视觉朝向判定已生效，即使没有名牌亦可精准锚定脚底！"
                )
            else:
                messagebox.showerror("标定失败", "保存两段式标定数据失败，请重试！")

    def _on_toggle_two_stage_feature(self):
        """切换是否启用特征识别辅助（取消勾选即仅名牌匹配）"""
        enabled = bool(self.var_enable_two_stage.get())
        self.config["enable_two_stage_feature"] = enabled
        self._save_config()
        if self.detector is not None:
            self.detector.enable_two_stage_feature = enabled
        state_str = "已开启 (全身+特征辅助与朝向)" if enabled else "已关闭 (仅使用纯名牌定位)"
        self.log(f"⚙️ [角色标定] 特征识别辅助: {state_str}")

    # ── 状态栏视觉读取与自动补给 ────────────────────────────────────────────
    def _build_status_card(self):
        card = tk.LabelFrame(
            getattr(self, "_panel_parent", self.sf),
            text="📊 状态栏",
            font=("Segoe UI", 10, "bold"),
            fg="#ffd54f",
            bg="#202026",
            padx=6,
            pady=2,
        )
        card.pack(fill=tk.X, padx=8, pady=1)

        style = ttk.Style(self.root)
        style.configure(
            "StatusHp.Horizontal.TProgressbar",
            troughcolor="#303038", background="#ff1744", bordercolor="#303038",
            lightcolor="#ff1744", darkcolor="#d50000",
        )
        style.configure(
            "StatusMp.Horizontal.TProgressbar",
            troughcolor="#303038", background="#2979ff", bordercolor="#303038",
            lightcolor="#40c4ff", darkcolor="#0d47a1",
        )

        resource_row = tk.Frame(card, bg="#202026")
        resource_row.pack(fill=tk.X, pady=0)

        def make_resource_cell(title, colour):
            cell = tk.Frame(resource_row, bg="#202026")
            cell.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(
                cell, text=f"{title}: -- / -- (--%)", anchor=tk.W,
                font=("Consolas", 7, "bold"), fg=colour, bg="#202026",
            )
            lbl.pack(fill=tk.X, padx=(1, 3))
            return lbl

        self.lbl_status_hp = make_resource_cell("HP", "#ff5252")
        self.lbl_status_mp = make_resource_cell("MP", "#40c4ff")
        self.pb_status_hp = None
        self.pb_status_mp = None

        exp_row = tk.Frame(card, bg="#202026")
        exp_row.pack(fill=tk.X, pady=0)
        self.lbl_status_exp = tk.Label(
            exp_row, text="EXP: -- (--%)", anchor=tk.W,
            font=("Consolas", 8, "bold"), fg="#ffd54f", bg="#202026",
        )
        self.lbl_status_exp.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 4))
        self.lbl_status_exp_hour = tk.Label(
            exp_row, text="每小时: --", anchor=tk.E,
            font=("Consolas", 8, "bold"), fg="#69f0ae", bg="#202026",
        )
        self.lbl_status_exp_hour.pack(side=tk.RIGHT)

        self.lbl_status_potion_stock = tk.Label(
            card, text="血药: --  |  蓝药: --", anchor=tk.W,
            font=("Consolas", 8, "bold"), fg="#b0bec5", bg="#202026",
        )
        self.lbl_status_potion_stock.pack(fill=tk.X, padx=2)

        saved_roi = self.config.get("status_bar_roi")
        self.status_auto_potion_var = tk.BooleanVar(
            value=bool(self.config.get("enable_auto_potion", False))
        )

        def on_auto_toggle():
            enabled = bool(self.status_auto_potion_var.get())
            if enabled and not self.config.get("status_bar_roi"):
                self.status_auto_potion_var.set(False)
                messagebox.showwarning(
                    "尚未框选状态栏",
                    "请先框选完整的 HP、MP、EXP 状态栏，再开启自动回血蓝。",
                    parent=self.root,
                )
                return
            if enabled and (
                not self.config.get("potion_bar_roi")
                or self._pending_potion_bindings
                or self._pending_potion_roi is not None
            ):
                self.status_auto_potion_var.set(False)
                messagebox.showwarning(
                    "药水快捷栏未确认",
                    "请先框选整块药水快捷栏、设置回血/回蓝键，并点击应用确认两种数量。",
                    parent=self.root,
                )
                return
            self.config["enable_auto_potion"] = enabled
            self._save_config()
            self.log(f"🧪 [自动补给] {'已开启' if enabled else '已关闭'}")

        control_row = tk.Frame(card, bg="#202026")
        control_row.pack(fill=tk.X, pady=(1, 0))
        tk.Checkbutton(
            control_row, text="自动补给", variable=self.status_auto_potion_var,
            command=on_auto_toggle, font=("Segoe UI", 8, "bold"),
            fg="#ffffff", bg="#202026", selectcolor="#303038",
            activeforeground="#ffffff", activebackground="#202026",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.btn_status_roi = tk.Button(
            control_row, text="▣ 框选", font=("Segoe UI", 8, "bold"),
            bg="#1565c0", fg="#ffffff", activebackground="#1976d2",
            activeforeground="#ffffff", relief=tk.FLAT, bd=0, highlightthickness=0,
            padx=4, pady=0,
            command=self._open_status_bar_roi_dialog,
        )
        self.btn_status_roi.pack(side=tk.LEFT, padx=(0, 4))

        details = tk.Frame(card, bg="#202026")
        self.status_details_frame = details

        def toggle_status_details():
            if details.winfo_manager():
                details.pack_forget()
                self.btn_status_details.configure(text="补给设置 ▸")
            else:
                details.pack(fill=tk.X, pady=(2, 0), after=control_row)
                self.btn_status_details.configure(text="补给设置 ▾")

        self.btn_status_details = tk.Button(
            control_row, text="补给设置 ▸", font=("Segoe UI", 8),
            bg="#37474f", fg="#ffffff", activebackground="#455a64",
            activeforeground="#ffffff", relief=tk.FLAT, bd=0, highlightthickness=0,
            padx=4, pady=0,
            command=toggle_status_details,
        )
        self.btn_status_details.pack(side=tk.LEFT, padx=(0, 5))
        self.lbl_status_roi = tk.Label(
            control_row,
            text="已框选" if saved_roi else "未框选",
            font=("Segoe UI", 7),
            fg="#69f0ae" if saved_roi else "#ffb74d",
            bg="#202026", anchor=tk.E,
        )
        self.lbl_status_roi.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        threshold_row = tk.Frame(details, bg="#202026")
        threshold_row.pack(fill=tk.X, pady=1)
        self.status_hp_threshold_var = tk.StringVar(
            value=f"{float(self.config.get('hp_potion_threshold_percent', 50.0)):g}"
        )
        self.status_mp_threshold_var = tk.StringVar(
            value=f"{float(self.config.get('mp_potion_threshold_percent', 30.0)):g}"
        )
        self.status_potion_cooldown_var = tk.StringVar(
            value=f"{float(self.config.get('potion_cooldown_ms', 800.0)):g}"
        )
        for label, variable, suffix, width in (
            ("HP≤", self.status_hp_threshold_var, "%", 5),
            ("MP≤", self.status_mp_threshold_var, "%", 5),
            ("冷却", self.status_potion_cooldown_var, "ms", 7),
        ):
            tk.Label(threshold_row, text=label, font=("Segoe UI", 8), fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT, padx=(2, 1))
            tk.Entry(threshold_row, textvariable=variable, width=width, justify=tk.CENTER, font=("Consolas", 8)).pack(side=tk.LEFT)
            tk.Label(threshold_row, text=suffix, font=("Segoe UI", 8), fg="#90a4ae", bg="#202026").pack(side=tk.LEFT, padx=(1, 5))

        death_row = tk.Frame(details, bg="#202026")
        death_row.pack(fill=tk.X, pady=(1, 1))
        tk.Label(
            death_row, text="死亡后血药≤", font=("Segoe UI", 8),
            fg="#ffcc80", bg="#202026",
        ).pack(side=tk.LEFT, padx=(2, 1))
        self.status_death_hp_stock_threshold_var = tk.StringVar(
            value=str(int(self.config.get("death_hp_stock_threshold", 0)))
        )
        tk.Entry(
            death_row, textvariable=self.status_death_hp_stock_threshold_var,
            width=5, justify=tk.CENTER, font=("Consolas", 8),
        ).pack(side=tk.LEFT)
        tk.Label(
            death_row, text="瓶：复活回城后停机；高于此值自动返程",
            font=("Segoe UI", 8), fg="#90a4ae", bg="#202026",
        ).pack(side=tk.LEFT, padx=(2, 0))

        key_row = tk.Frame(details, bg="#202026")
        key_row.pack(fill=tk.X, pady=(2, 1))
        for label, cfg_key, attr, fallback in (
            ("回血:", "hp_potion_key", "btn_rec_hp_potion", "HOME"),
            ("回蓝:", "mp_potion_key", "btn_rec_mp_potion", "PAGEUP"),
        ):
            group = tk.Frame(key_row, bg="#202026")
            group.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(group, text=label, font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)
            current = str(self.config.get(cfg_key, fallback)).upper()
            button = tk.Button(
                group, text=f"⌨️ {current}", font=("Segoe UI", 8, "bold"),
                fg="#00e5ff", bg="#2a2a32", relief=tk.RAISED, padx=4, pady=1,
            )
            button.configure(command=lambda k=cfg_key, b=button: self._start_key_recording(k, b))
            button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
            setattr(self, attr, button)

        potion_roi_row = tk.Frame(details, bg="#202026")
        potion_roi_row.pack(fill=tk.X, pady=(2, 1))
        tk.Button(
            potion_roi_row, text="▣ 框选药水快捷栏（整块两排）",
            command=self._open_potion_bar_roi_dialog,
            font=("Segoe UI", 8, "bold"), bg="#1565c0", fg="#ffffff",
            activebackground="#1976d2", relief=tk.FLAT, padx=5, pady=1,
        ).pack(side=tk.LEFT, padx=(1, 5))
        self.lbl_potion_bar_roi = tk.Label(
            potion_roi_row,
            text="已框选" if self.config.get("potion_bar_roi") else "未框选",
            font=("Segoe UI", 8), fg="#69f0ae" if self.config.get("potion_bar_roi") else "#ffb74d",
            bg="#202026", anchor=tk.W,
        )
        self.lbl_potion_bar_roi.pack(side=tk.LEFT, fill=tk.X, expand=True)

        action_row = tk.Frame(details, bg="#202026")
        action_row.pack(fill=tk.X, pady=(3, 0))
        self.btn_apply_status_settings = tk.Button(
            action_row, text="应用状态栏设置", command=self._apply_status_bar_settings,
            font=("Segoe UI", 8, "bold"), bg="#2e7d32", fg="#ffffff",
            activebackground="#388e3c", relief=tk.FLAT, padx=8, pady=2,
        )
        self.btn_apply_status_settings.pack(side=tk.LEFT, padx=(1, 5))
        self.lbl_potion_stock_apply = tk.Label(
            action_row, text="血药 -- / 蓝药 --", font=("Consolas", 8, "bold"),
            fg="#b0bec5", bg="#202026", anchor=tk.W,
        )
        self.lbl_potion_stock_apply.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(
            action_row, text="重置经验统计", command=self._reset_status_exp_rate,
            font=("Segoe UI", 8), bg="#37474f", fg="#ffffff",
            activebackground="#455a64", relief=tk.FLAT, padx=8, pady=2,
        ).pack(side=tk.LEFT)

    def _apply_status_bar_settings(self):
        try:
            hp_threshold = float(self.status_hp_threshold_var.get().strip())
            mp_threshold = float(self.status_mp_threshold_var.get().strip())
            cooldown_ms = float(self.status_potion_cooldown_var.get().strip())
            death_hp_threshold = int(self.status_death_hp_stock_threshold_var.get().strip())
            if not 1.0 <= hp_threshold <= 99.0:
                raise ValueError("回血阈值必须在 1% 到 99% 之间")
            if not 1.0 <= mp_threshold <= 99.0:
                raise ValueError("回蓝阈值必须在 1% 到 99% 之间")
            if not 100.0 <= cooldown_ms <= 10_000.0:
                raise ValueError("补给冷却必须在 100ms 到 10000ms 之间")
            if not 0 <= death_hp_threshold <= 9999:
                raise ValueError("死亡后血药停机阈值必须在 0 到 9999 瓶之间")
        except ValueError as exc:
            messagebox.showwarning("状态栏参数无效", str(exc), parent=self.root)
            return
        roi = self._pending_potion_roi or self.config.get("potion_bar_roi")
        if not roi:
            messagebox.showwarning(
                "药水快捷栏未框选",
                "请先框选包含全部两排按键的药水快捷栏，再点击应用。",
                parent=self.root,
            )
            return
        hp_binding = self._pending_potion_bindings.get("hp_potion_key", (
            str(self.config.get("hp_potion_key", "")),
            int(self.config.get("hp_potion_vk", 0) or 0),
        ))
        mp_binding = self._pending_potion_bindings.get("mp_potion_key", (
            str(self.config.get("mp_potion_key", "")),
            int(self.config.get("mp_potion_vk", 0) or 0),
        ))
        hp_key, mp_key = hp_binding[0], mp_binding[0]
        if hp_key == mp_key or not supports_stock_key(hp_key) or not supports_stock_key(mp_key):
            messagebox.showwarning(
                "药水快捷键无效",
                "回血和回蓝必须使用两个不同的、快捷栏可见的按键。",
                parent=self.root,
            )
            return
        if self.capture is None or self.capture.capture_frame(copy=True, include_overlay=False) is None:
            messagebox.showwarning("无法验证药量", "当前没有可用的游戏画面。", parent=self.root)
            return
        pending = {
            "hp_threshold": hp_threshold, "mp_threshold": mp_threshold,
            "cooldown_ms": cooldown_ms, "death_hp_threshold": death_hp_threshold,
            "hp_binding": hp_binding,
            "mp_binding": mp_binding, "roi": dict(roi),
            "enable_auto_potion": bool(self.status_auto_potion_var.get()),
        }
        self.btn_apply_status_settings.configure(state=tk.DISABLED, text="识别中…")

        def validate():
            try:
                readings = []
                for index in range(2):
                    if index:
                        time.sleep(0.25)
                    frame = self.capture.capture_frame(copy=True, include_overlay=False)
                    with self._potion_stock_eval_lock:
                        readings.append(self.potion_stock_reader.read(
                            frame, hp_key, mp_key, pending["roi"],
                        ))
                self._potion_apply_queue.put({"settings": pending, "readings": readings})
            except Exception as exc:
                self._potion_apply_queue.put({"settings": pending, "error": str(exc)})

        threading.Thread(target=validate, daemon=True, name="potion-stock-apply").start()

    def _handle_potion_apply_result(self, result: Dict[str, Any]) -> None:
        self.btn_apply_status_settings.configure(state=tk.NORMAL, text="应用状态栏设置")
        pending = result["settings"]
        hp_binding, mp_binding = pending["hp_binding"], pending["mp_binding"]
        current_hp = self._pending_potion_bindings.get("hp_potion_key", (
            str(self.config.get("hp_potion_key", "")),
            int(self.config.get("hp_potion_vk", 0) or 0),
        ))
        current_mp = self._pending_potion_bindings.get("mp_potion_key", (
            str(self.config.get("mp_potion_key", "")),
            int(self.config.get("mp_potion_vk", 0) or 0),
        ))
        if (hp_binding, mp_binding, pending["roi"]) != (
            current_hp, current_mp,
            self._pending_potion_roi or self.config.get("potion_bar_roi"),
        ):
            messagebox.showwarning(
                "设置已变化", "验证过程中键位或框选范围发生变化，请重新点击应用。",
                parent=self.root,
            )
            return
        readings = result.get("readings") or []
        valid = (
            len(readings) == 2
            and all(item.layout_valid and item.hp_count is not None
                    and item.mp_count is not None for item in readings)
            and readings[0].hp_count == readings[1].hp_count
            and readings[0].mp_count == readings[1].mp_count
        )
        if not valid:
            if result.get("error"):
                detail = result["error"]
            elif readings:
                missing = []
                if any(item.hp_count is None for item in readings):
                    missing.append(f"回血键 {hp_binding[0].upper()}")
                if any(item.mp_count is None for item in readings):
                    missing.append(f"回蓝键 {mp_binding[0].upper()}")
                detail = "、".join(missing) + ("的键位或数字无法识别" if missing else "两次药量读数不一致")
            else:
                detail = "没有取得游戏画面"
            messagebox.showwarning(
                "药水数量验证失败",
                f"{detail}。请确认框选包含整块两排快捷栏、键位设置正确，再重试；本次未应用。",
                parent=self.root,
            )
            self.log(f"⚠️ [药水快捷栏应用失败] {detail}；设置未保存")
            return

        self.config["hp_potion_threshold_percent"] = pending["hp_threshold"]
        self.config["mp_potion_threshold_percent"] = pending["mp_threshold"]
        self.config["potion_cooldown_ms"] = pending["cooldown_ms"]
        self.config["death_hp_stock_threshold"] = pending["death_hp_threshold"]
        self.config["hp_potion_key"], self.config["hp_potion_vk"] = hp_binding
        self.config["mp_potion_key"], self.config["mp_potion_vk"] = mp_binding
        self.config["potion_bar_roi"] = pending["roi"]
        self.config["enable_auto_potion"] = pending["enable_auto_potion"]
        self._pending_potion_bindings.clear()
        self._pending_potion_roi = None
        self.lbl_potion_bar_roi.configure(text="已框选·已应用", fg="#69f0ae")
        self._save_config()
        self.log(
            f"✅ [药水快捷栏已应用] 血药 {hp_binding[0].upper()}="
            f"{readings[-1].hp_count}，蓝药 {mp_binding[0].upper()}="
            f"{readings[-1].mp_count}；HP≤{pending['hp_threshold']:g}% "
            f"MP≤{pending['mp_threshold']:g}%"
        )

    def _open_potion_bar_roi_dialog(self):
        frame = self.capture.capture_frame(copy=True, include_overlay=False) if self.capture else None
        if frame is None:
            messagebox.showwarning("无法框选", "当前还没有可用的游戏画面。", parent=self.root)
            return
        roi = self._pending_potion_roi or self.config.get("potion_bar_roi")
        resolved = StatusBarReader.resolve_roi(frame.shape, roi)
        if resolved is not None:
            x, y, w, h = resolved
            roi = {"x": x, "y": y, "w": w, "h": h}
        OcrRoiCalibrationDialog(
            self.root, frame, roi, self._on_potion_bar_roi_confirmed,
            title_text="🧪 药水快捷栏区域框选",
            tip_text=(
                "🎯 请框选游戏右下角完整的两排快捷栏：包含键位文字、物品图标和下方数量。"
                "可以留少量边缘，但不要只框单个按键。"
            ),
            purpose_text="两排药水快捷栏", default_roi=None,
            min_size=(145, 65),
        )

    def _on_potion_bar_roi_confirmed(self, roi: Dict[str, int]):
        frame = self.capture.capture_frame(copy=False, include_overlay=False) if self.capture else None
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        saved = dict(roi)
        saved.update(
            frame_w=fw, frame_h=fh,
            nx=float(roi["x"]) / max(1, fw),
            ny=float(roi["y"]) / max(1, fh),
            nw=float(roi["w"]) / max(1, fw),
            nh=float(roi["h"]) / max(1, fh),
        )
        self._pending_potion_roi = saved
        with self._status_reading_lock:
            self.potion_stock_monitor = PotionStockMonitor()
        if self.config.get("enable_auto_potion", False):
            self.config["enable_auto_potion"] = False
            self.status_auto_potion_var.set(False)
            self._save_config()
        self.lbl_potion_bar_roi.configure(text="整块已框选·待应用", fg="#ffb74d")
        self.log(
            f"🧪 [药水快捷栏待应用] x={roi['x']} y={roi['y']} "
            f"w={roi['w']} h={roi['h']}；请核对实时数量并点击应用"
        )

    def _open_status_bar_roi_dialog(self):
        frame = self.capture.capture_frame(copy=True) if self.capture else None
        if frame is None:
            messagebox.showwarning("无法框选", "当前还没有可用的游戏画面。", parent=self.root)
            return
        current_roi = self.config.get("status_bar_roi")
        resolved = StatusBarReader.resolve_roi(frame.shape, current_roi)
        if resolved is not None:
            rx, ry, rw, rh = resolved
            current_roi = {"x": rx, "y": ry, "w": rw, "h": rh}
        OcrRoiCalibrationDialog(
            self.root,
            frame,
            current_roi,
            self._on_status_bar_roi_confirmed,
            title_text="📊 状态栏视觉范围标定",
            tip_text=(
                "🎯 请完整框选游戏底部的 HP、MP、EXP 三段状态栏；"
                "保留文字与下方三条色条，然后点击【确认保存范围】"
            ),
            purpose_text="状态栏",
            default_roi=None,
            min_size=(90, 18),
        )

    def _on_status_bar_roi_confirmed(self, roi: Dict[str, int]):
        frame = self.capture.capture_frame(copy=False) if self.capture else None
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        saved = dict(roi)
        saved.update(
            frame_w=fw,
            frame_h=fh,
            nx=float(roi["x"]) / max(1, fw),
            ny=float(roi["y"]) / max(1, fh),
            nw=float(roi["w"]) / max(1, fw),
            nh=float(roi["h"]) / max(1, fh),
        )
        self.config["status_bar_roi"] = saved
        self.config["enable_auto_potion"] = False
        self.status_auto_potion_var.set(False)
        self._status_ocr_valid_until = 0.0
        self._status_hpmp_ocr_validated = False
        with self._status_reading_lock:
            self._latest_status_reading = None
            self._status_exact_reading = None
            self.status_exp_tracker.reset()
        self._save_config()
        self.lbl_status_roi.configure(
            text="已框选·待校验", fg="#69f0ae"
        )
        self.log(
            f"📊 [状态栏框选] x={roi['x']}, y={roi['y']}, "
            f"w={roi['w']}, h={roi['h']}；自动补给已安全关闭，请确认读数后再开启。"
        )

    def _reset_status_exp_rate(self):
        with self._status_reading_lock:
            self.status_exp_tracker.reset()
        self.log("📈 [经验统计] 已从当前视觉读数重新开始计时。")

    @staticmethod
    def _format_status_number(value: Optional[float]) -> str:
        if value is None:
            return "--"
        return f"{int(round(value)):,}"

    def _refresh_status_card(self):
        if not hasattr(self, "lbl_status_hp"):
            return
        while True:
            try:
                apply_result = self._potion_apply_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_potion_apply_result(apply_result)
        now = time.perf_counter()
        if now - self._status_last_ui_refresh < 0.10:
            return
        self._status_last_ui_refresh = now
        with self._status_reading_lock:
            reading = self._latest_status_reading
            exact = self._status_exact_reading
            exp_per_hour = self.status_exp_tracker.per_hour(now)
            exp_gain = self.status_exp_tracker.total_gain
            hp_key = str(self.config.get("hp_potion_key", ""))
            mp_key = str(self.config.get("mp_potion_key", ""))
            hp_stock = self.potion_stock_monitor.confirmed("hp", hp_key, now)
            mp_stock = self.potion_stock_monitor.confirmed("mp", mp_key, now)
        stock_text = (
            f"血药({hp_key.upper()}): {hp_stock if hp_stock is not None else '--'}"
            f"  |  蓝药({mp_key.upper()}): {mp_stock if mp_stock is not None else '--'}"
        )
        self.lbl_status_potion_stock.configure(
            text=stock_text,
            fg=("#ff5252" if hp_stock == 0 else
                "#69f0ae" if hp_stock is not None and mp_stock is not None else "#b0bec5"),
        )
        preview_hp_key = self._pending_potion_bindings.get("hp_potion_key", (hp_key, 0))[0]
        preview_mp_key = self._pending_potion_bindings.get("mp_potion_key", (mp_key, 0))[0]
        with self._status_reading_lock:
            preview_hp = self.potion_stock_monitor.confirmed("hp", preview_hp_key, now)
            preview_mp = self.potion_stock_monitor.confirmed("mp", preview_mp_key, now)
        self.lbl_potion_stock_apply.configure(
            text=(f"血药 {preview_hp if preview_hp is not None else '--'} / "
                  f"蓝药 {preview_mp if preview_mp is not None else '--'}"),
            fg=("#ff5252" if preview_hp == 0 else
                "#69f0ae" if preview_hp is not None and preview_mp is not None
                else "#b0bec5"),
        )
        reading_is_fresh = reading is not None and now - reading.timestamp <= 1.5
        # OCR 偶尔会因游戏特效或 CPU 忙而连续漏掉几帧。最后一次完整读出的
        # HP/MP/EXP 是可信状态，显示层应保持它，不能把短暂“未观测”渲染为
        # “数值不存在”。自动补给仍由 _status_ocr_valid_until 单独控制时效。
        if not reading_is_fresh and exact is None:
            self.lbl_status_hp.configure(text="HP: -- / -- (--%)")
            self.lbl_status_mp.configure(text="MP: -- / -- (--%)")
            if self.pb_status_hp is not None:
                self.pb_status_hp["value"] = 0.0
            if self.pb_status_mp is not None:
                self.pb_status_mp["value"] = 0.0
            return

        trusted_exact = exact
        hp_pct = reading.hp_bar_percent if reading_is_fresh else None
        mp_pct = reading.mp_bar_percent if reading_is_fresh else None
        if trusted_exact is not None:
            # 新鲜颜色条优先；OCR精确值只在颜色暂时不可用时兜底。
            if hp_pct is None and trusted_exact.hp_percent is not None:
                hp_pct = trusted_exact.hp_percent
            if mp_pct is None and trusted_exact.mp_percent is not None:
                mp_pct = trusted_exact.mp_percent
        hp_estimate = (
            int(round(float(trusted_exact.hp_max) * hp_pct / 100.0))
            if trusted_exact is not None and trusted_exact.hp_max and hp_pct is not None
            else None
        )
        mp_estimate = (
            int(round(float(trusted_exact.mp_max) * mp_pct / 100.0))
            if trusted_exact is not None and trusted_exact.mp_max and mp_pct is not None
            else None
        )
        hp_text = (
            f"HP: ≈{self._format_status_number(hp_estimate)} / "
            f"{self._format_status_number(trusted_exact.hp_max)}"
            if trusted_exact is not None else "HP: -- / --"
        )
        mp_text = (
            f"MP: ≈{self._format_status_number(mp_estimate)} / "
            f"{self._format_status_number(trusted_exact.mp_max)}"
            if trusted_exact is not None else "MP: -- / --"
        )
        self.lbl_status_hp.configure(text=f"{hp_text} ({hp_pct:.1f}%)" if hp_pct is not None else f"{hp_text} (--%)")
        self.lbl_status_mp.configure(text=f"{mp_text} ({mp_pct:.1f}%)" if mp_pct is not None else f"{mp_text} (--%)")
        if self.pb_status_hp is not None:
            self.pb_status_hp["value"] = hp_pct if hp_pct is not None else 0.0
        if self.pb_status_mp is not None:
            self.pb_status_mp["value"] = mp_pct if mp_pct is not None else 0.0
        if trusted_exact is not None and trusted_exact.exp_current is not None:
            exp_pct_text = f"{trusted_exact.exp_percent:.2f}%" if trusted_exact.exp_percent is not None else "--%"
            self.lbl_status_exp.configure(
                text=f"EXP: {self._format_status_number(trusted_exact.exp_current)} ({exp_pct_text})"
            )
        elif trusted_exact is None:
            self.lbl_status_exp.configure(text="EXP: -- (--%)")
        rate_text = self._format_status_number(exp_per_hour)
        self.lbl_status_exp_hour.configure(
            text=f"每小时: {rate_text}  (+{self._format_status_number(exp_gain)})"
        )
        if now <= self._status_ocr_valid_until:
            self.lbl_status_roi.configure(text="读取正常", fg="#69f0ae")
        elif trusted_exact is not None:
            self.lbl_status_roi.configure(text="保持上次值", fg="#ffb74d")
        else:
            self.lbl_status_roi.configure(text="等待OCR", fg="#ffb74d")

    def _build_key_card(self):
        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf), text="⌨️ 快捷按键与输入模式",
                              font=("Segoe UI", 10, "bold"), fg="#d1c4e9", bg="#202026", padx=6, pady=2)
        card.pack(fill=tk.X, padx=8, pady=1)
        self.root.bind("<Key>", self._on_physical_key_pressed)

        # ── 输入模式下拉选择 ──
        r_mode = tk.Frame(card, bg="#202026")
        r_mode.pack(fill=tk.X, pady=(0, 2))

        tk.Label(
            r_mode, text="模式:", font=("Segoe UI", 8, "bold"),
            fg="#e0e0e0", bg="#202026"
        ).pack(side=tk.LEFT, padx=(2, 4))

        current_mode = self.config.get("input_mode", "background")
        initial_val = "后台 (PostMessage)" if current_mode == "background" else "前台 (SendInput)"

        self.combo_input_mode = ttk.Combobox(
            r_mode,
            values=["后台 (PostMessage)", "前台 (SendInput)"],
            state="readonly",
            width=15,
            font=("Segoe UI", 8),
        )
        self.combo_input_mode.set(initial_val)
        self.combo_input_mode.pack(side=tk.LEFT, padx=(0, 6))

        self.lbl_mode_desc = tk.Label(
            r_mode,
            text="免遮挡" if current_mode == "background" else "需置顶",
            font=("Segoe UI", 7),
            fg="#80cbc4" if current_mode == "background" else "#ffb74d",
            bg="#202026"
        )
        self.lbl_mode_desc.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _on_mode_selected(event=None):
            selected = self.combo_input_mode.get()
            if "后台" in selected or "PostMessage" in selected:
                mode_key = "background"
                desc_text = "免遮挡"
                desc_color = "#80cbc4"
                log_text = "⚙️ [输入模式] 已切换为: 后台 (PostMessage) —— 游戏可在后台/被遮挡运行，不抢占键盘鼠标"
            else:
                mode_key = "foreground"
                desc_text = "需置顶"
                desc_color = "#ffb74d"
                log_text = "⚙️ [输入模式] 已切换为: 前台 (SendInput) —— 每次输入将激活窗口到最前端"

            self.config["input_mode"] = mode_key
            if hasattr(self, "input_driver") and self.input_driver:
                self.input_driver.set_mode(mode_key)
            self.lbl_mode_desc.config(text=desc_text, fg=desc_color)
            self._save_config()
            self.log(log_text)

        self.combo_input_mode.bind("<<ComboboxSelected>>", _on_mode_selected)

        # ── 按键录制第 1 行：常规核心键 (攻击, 跳跃, 捡物) ──
        r_keys1 = tk.Frame(card, bg="#202026")
        r_keys1.pack(fill=tk.X, pady=1)

        for label, cfg_key, attr in [
            ("攻击:", "attack_key", "btn_rec_atk"),
            ("跳跃:", "jump_key",   "btn_rec_jmp"),
            ("捡物:", "pick_key",   "btn_rec_pck"),
        ]:
            f = tk.Frame(r_keys1, bg="#202026")
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(f, text=label, font=("Segoe UI", 8), fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)
            cur_val = str(self.config.get(cfg_key, "END")).upper()
            btn = tk.Button(f, text=f"⌨️ {cur_val}",
                            font=("Segoe UI", 8, "bold"), fg="#00e5ff", bg="#2a2a32",
                            relief=tk.FLAT, bd=0, highlightthickness=0, padx=3, pady=0)
            btn.config(command=lambda k=cfg_key, b=btn: self._start_key_recording(k, b))
            btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
            setattr(self, attr, btn)

        tk.Button(
            card, text="⚔️ 管理攻击技能（新增 / 编辑独立攻击盒）",
            command=self._open_attack_skills_dialog,
            font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#6a1b9a",
            relief=tk.FLAT, bd=0, padx=4, pady=1,
        ).pack(fill=tk.X, pady=(2, 1))

        # ── 按键录制第 2 行：动作/对话键 ──
        r_keys2 = tk.Frame(card, bg="#202026")
        r_keys2.pack(fill=tk.X, pady=1)

        for label, cfg_key, attr in [
            ("椅子:", "chair_key",    "btn_rec_chr"),
            ("瞬移:", "teleport_key", "btn_rec_tp"),
            ("对话:", "dialog_key", "btn_rec_dialog"),
        ]:
            f = tk.Frame(r_keys2, bg="#202026")
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(f, text=label, font=("Segoe UI", 8), fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)
            default_key = "SHIFT" if cfg_key == "teleport_key" else ("Y" if cfg_key == "dialog_key" else "END")
            cur_val = str(self.config.get(cfg_key, default_key)).upper()
            btn = tk.Button(f, text=f"⌨️ {cur_val}",
                            font=("Segoe UI", 8, "bold"), fg="#00e5ff", bg="#2a2a32",
                            relief=tk.FLAT, bd=0, highlightthickness=0, padx=3, pady=0)
            btn.config(command=lambda k=cfg_key, b=btn: self._start_key_recording(k, b))
            btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
            setattr(self, attr, btn)

        r_pet_feed = tk.Frame(card, bg="#202026")
        r_pet_feed.pack(fill=tk.X, pady=1)
        self.enable_auto_pet_feed_var = tk.BooleanVar(
            value=bool(self.config.get("enable_auto_pet_feed", False)))
        tk.Checkbutton(
            r_pet_feed, text="启用宠物喂食", variable=self.enable_auto_pet_feed_var,
            command=self._toggle_auto_pet_feed, font=("Segoe UI", 8),
            fg="#e0e0e0", bg="#202026", selectcolor="#2a2a32",
            activebackground="#202026", activeforeground="#e0e0e0",
            highlightthickness=0, bd=0, padx=2, pady=0,
        ).pack(side=tk.LEFT, padx=(0, 3))
        pet_key = str(self.config.get("pet_feed_key", "") or "").upper()
        self.btn_rec_pet_feed = tk.Button(
            r_pet_feed, text=f"⌨️ {pet_key or '未设置'}",
            font=("Segoe UI", 8, "bold"), fg="#00e5ff", bg="#2a2a32",
            relief=tk.FLAT, bd=0, highlightthickness=0, padx=3, pady=0,
        )
        self.btn_rec_pet_feed.config(command=lambda: self._start_key_recording(
            "pet_feed_key", self.btn_rec_pet_feed))
        self.btn_rec_pet_feed.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(r_pet_feed, text="自动间隔:", font=("Segoe UI", 8),
                 fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)
        self.pet_feed_interval_var = tk.StringVar(
            value=f"{float(self.config.get('pet_feed_interval_sec', 300.0)):g}")
        pet_feed_interval_entry = tk.Entry(
            r_pet_feed, textvariable=self.pet_feed_interval_var, width=6,
            font=("Segoe UI", 8), fg="#e0e0e0", bg="#2a2a32",
            insertbackground="#e0e0e0", relief=tk.FLAT,
        )
        pet_feed_interval_entry.pack(side=tk.LEFT, padx=(3, 2))
        pet_feed_interval_entry.bind("<Return>", lambda _event: self._apply_pet_feed_interval())
        tk.Label(r_pet_feed, text="秒", font=("Segoe UI", 8),
                 fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)
        tk.Button(r_pet_feed, text="应用", command=self._apply_pet_feed_interval,
                  font=("Segoe UI", 8), fg="#ffffff", bg="#3949ab",
                  relief=tk.FLAT, bd=0, padx=4, pady=0).pack(side=tk.LEFT, padx=(6, 0))

        # ── 法师瞬移设置第 3 行 ──
        tp_header = tk.Frame(card, bg="#202026")
        tp_header.pack(fill=tk.X, pady=(2, 0))
        r_tp = tk.Frame(card, bg="#202026")
        self.key_teleport_details_frame = r_tp

        self.enable_teleport_var = tk.BooleanVar(
            value=bool(self.config.get("enable_teleport", False))
        )

        def _on_toggle_teleport():
            enabled = bool(self.enable_teleport_var.get())
            self.config["enable_teleport"] = enabled
            if hasattr(self, "motion") and self.motion:
                self.motion.enable_teleport = enabled
            self._save_config()
            self.log(f"⚡ [法师瞬移] {'已开启' if enabled else '已关闭'}（移动/垂直寻路）")
            self._refresh_platform_graph_teleport()

        cb_tp = tk.Checkbutton(
            tp_header,
            text="启用法师瞬移",
            variable=self.enable_teleport_var,
            font=("Segoe UI", 8, "bold"),
            fg="#e1bee7",
            bg="#202026",
            selectcolor="#303038",
            activeforeground="#ba68c8",
            activebackground="#202026",
            command=_on_toggle_teleport,
        )
        cb_tp.pack(side=tk.LEFT, padx=(0, 4))

        self.lbl_tp_cd_summary = tk.Label(
            tp_header,
            text=(f"CD {float(self.config.get('teleport_cd_min_ms', 300.0)):g}~"
                  f"{float(self.config.get('teleport_cd_max_ms', 500.0)):g}ms"),
            font=("Consolas", 7), fg="#b39ddb", bg="#202026",
        )
        self.lbl_tp_cd_summary.pack(side=tk.LEFT, padx=(0, 4))

        def _toggle_tp_details():
            if r_tp.winfo_manager():
                r_tp.pack_forget()
                self.btn_tp_details.configure(text="瞬移设置 ▸")
            else:
                r_tp.pack(fill=tk.X, pady=(1, 0), after=tp_header)
                self.btn_tp_details.configure(text="瞬移设置 ▾")

        self.btn_tp_details = tk.Button(
            tp_header, text="瞬移设置 ▸", command=_toggle_tp_details,
            font=("Segoe UI", 7), fg="#ffffff", bg="#4527a0",
            activeforeground="#ffffff", activebackground="#512da8",
            relief=tk.FLAT, bd=0, highlightthickness=0, padx=4, pady=0,
        )
        self.btn_tp_details.pack(side=tk.RIGHT)

        tk.Label(
            r_tp, text="CD(ms):", font=("Segoe UI", 9),
            fg="#e0e0e0", bg="#202026"
        ).pack(side=tk.LEFT, padx=(2, 1))

        self.tp_cd_min_var = tk.StringVar(
            value=str(int(float(self.config.get("teleport_cd_min_ms", 300.0))))
        )
        self.tp_cd_max_var = tk.StringVar(
            value=str(int(float(self.config.get("teleport_cd_max_ms", 500.0))))
        )

        def _on_tp_cd_changed(event=None):
            try:
                min_v = float(self.tp_cd_min_var.get().strip())
                max_v = float(self.tp_cd_max_var.get().strip())
                if min_v <= 0 or max_v <= 0:
                    return
                if min_v > max_v:
                    min_v, max_v = max_v, min_v
                self.config["teleport_cd_min_ms"] = min_v
                self.config["teleport_cd_max_ms"] = max_v
                if hasattr(self, "motion") and self.motion:
                    self.motion.teleport_cd_min_ms = min_v
                    self.motion.teleport_cd_max_ms = max_v
                self.lbl_tp_cd_summary.config(text=f"CD {min_v:g}~{max_v:g}ms")
                self._save_config()
            except Exception:
                pass

        e_min = tk.Entry(r_tp, textvariable=self.tp_cd_min_var, width=5, justify=tk.CENTER,
                         font=("Consolas", 9), bg="#2a2a32", fg="#00e5ff", insertbackground="#ffffff")
        e_min.pack(side=tk.LEFT, padx=1)
        e_min.bind("<FocusOut>", _on_tp_cd_changed)
        e_min.bind("<Return>", _on_tp_cd_changed)

        tk.Label(r_tp, text="~", font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)

        e_max = tk.Entry(r_tp, textvariable=self.tp_cd_max_var, width=5, justify=tk.CENTER,
                         font=("Consolas", 9), bg="#2a2a32", fg="#00e5ff", insertbackground="#ffffff")
        e_max.pack(side=tk.LEFT, padx=1)
        e_max.bind("<FocusOut>", _on_tp_cd_changed)
        e_max.bind("<Return>", _on_tp_cd_changed)

        tk.Label(
            r_tp, text="(随机)", font=("Segoe UI", 8),
            fg="#90a4ae", bg="#202026"
        ).pack(side=tk.LEFT, padx=(2, 0))

    def _open_attack_skills_dialog(self) -> None:
        """Edit extra attack skills; the legacy controls remain skill 1."""
        win = tk.Toplevel(self.root)
        win.title("攻击技能管理")
        fit_window_to_work_area(win, (590, 370), (480, 300), parent=self.root)
        win.transient(self.root)
        win.grab_set()
        tk.Label(
            win, text="主攻击沿用当前按键和攻击判定盒；新增技能有独立按键和射程。",
            font=("Segoe UI", 9, "bold"), anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(10, 5))
        tree = ttk.Treeview(win, columns=("name", "key", "reach", "way"), show="headings", height=8)
        for col, title, width in (("name", "技能", 170), ("key", "按键", 70),
                                  ("reach", "前/上/下/后 px", 190), ("way", "方向", 70)):
            tree.heading(col, text=title)
            tree.column(col, width=width, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        def refresh():
            tree.delete(*tree.get_children())
            for skill in skills_from_config(self.config)[1:]:
                tree.insert("", tk.END, iid=skill["id"], values=(
                    skill["name"] + (" [本地]" if skill.get("wz_rect") else ""),
                    skill["key"].upper(),
                    f"{int(skill['reach_x'])}/{int(skill['reach_y_up'])}/"
                    f"{int(skill['reach_y_down'])}/{int(skill['behind_x'])}",
                    "双向" if skill["two_way"] else "单向",
                ))

        def selected():
            item = tree.selection()
            if not item:
                messagebox.showinfo("攻击技能", "请先选择一个技能。", parent=win)
                return None
            return item[0]

        def remove():
            skill_id = selected()
            if skill_id is None:
                return
            extras = [dict(item) for item in self.config.get("extra_attack_skills", [])
                      if item.get("id") != skill_id]
            self.config["extra_attack_skills"] = extras
            rules = {}
            for mob_id, allowed in (self.config.get("monster_skill_rules", {}) or {}).items():
                retained = [item for item in allowed if item != skill_id]
                if retained and retained != ["primary"]:
                    rules[mob_id] = retained
            self.config["monster_skill_rules"] = rules
            self._save_config()
            self._refresh_mob_skill_labels()
            refresh()

        buttons = tk.Frame(win)
        buttons.pack(fill=tk.X, padx=12, pady=(3, 10))
        tk.Button(buttons, text="＋ 添加技能", command=lambda: self._edit_attack_skill(win, None, refresh),
                  bg="#6a1b9a", fg="white").pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(buttons, text="编辑选中", command=lambda: (
            self._edit_attack_skill(win, skill_id, refresh) if (skill_id := selected()) else None
        )).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons, text="删除选中", command=remove, bg="#b71c1c", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(buttons, text="关闭", command=win.destroy).pack(side=tk.RIGHT)
        refresh()

    def _refresh_mob_skill_labels(self) -> None:
        if not hasattr(self, "mob_tree"):
            return
        for item in self.mob_tree.get_children():
            mob_id = str(self.mob_tree.set(item, "id"))
            self.mob_tree.set(item, "skill", self._mob_skill_label(mob_id))

    def _prompt_wz_skill_hitbox(self, parent, on_import, existing=None) -> None:
        """Import an explicit rank hitbox; never infer one from skill art."""
        existing = existing or {}
        prompt = tk.Toplevel(parent)
        prompt.title("从本地技能 IMG 读取命中范围")
        fit_window_to_work_area(prompt, (500, 390), (430, 330), parent=parent)
        prompt.transient(parent)
        prompt.grab_set()
        tk.Label(prompt, text="技能名模糊搜索（也可直接输入 ID）", anchor="w").pack(
            fill=tk.X, padx=12, pady=(10, 3))
        search_var = tk.StringVar(value=str(existing.get("skill_id", "")))
        search_entry = tk.Entry(prompt, textvariable=search_var)
        search_entry.pack(fill=tk.X, padx=12)
        search_entry.focus_set()
        results_frame = tk.Frame(prompt)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 0))
        results = tk.Listbox(results_frame, height=7, exportselection=False,
                             font=("Segoe UI", 9))
        results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(results_frame, command=results.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results.configure(yscrollcommand=scrollbar.set)
        choices = []
        search_status = tk.StringVar(value="输入名称后选择结果；同名技能以 ID 区分")
        tk.Label(prompt, textvariable=search_status, anchor="w", fg="#557085").pack(
            fill=tk.X, padx=12, pady=(2, 4))

        def run_search():
            pending[0] = None
            results.delete(0, tk.END)
            choices.clear()
            if not search_var.get().strip():
                search_status.set("输入名称后选择结果；同名技能以 ID 区分")
                return
            try:
                choices.extend(search_skill_names(search_var.get()))
            except Exception as exc:
                search_status.set(f"本地技能名称读取失败：{exc}")
                return
            for item in choices:
                results.insert(tk.END, f"{item.name}    [{item.skill_id}]  {item.book}")
            search_status.set(f"找到 {len(choices)} 项；选择后自动填入技能 ID" if choices else
                              "没有匹配的本地技能；可直接输入 ID")

        pending = [None]

        def schedule_search(*_):
            if pending[0] is not None:
                prompt.after_cancel(pending[0])
            pending[0] = prompt.after(180, run_search)

        search_var.trace_add("write", schedule_search)

        def choose_result(_event=None):
            selected = results.curselection()
            if selected and selected[0] < len(choices):
                item = choices[selected[0]]
                skill_var.set(item.skill_id)
                search_status.set(f"已选：{item.name} [{item.skill_id}]；请确认技能等级")

        results.bind("<<ListboxSelect>>", choose_result)
        if search_var.get():
            schedule_search()

        tk.Label(prompt, text="当前技能等级（不是职业等级）", anchor="w").pack(
            fill=tk.X, padx=12, pady=(6, 2))
        row = tk.Frame(prompt)
        row.pack(fill=tk.X, padx=12)
        tk.Label(row, text="技能 ID").pack(side=tk.LEFT)
        skill_var = tk.StringVar(value=str(existing.get("skill_id", "")))
        tk.Entry(row, textvariable=skill_var, width=14).pack(side=tk.LEFT, padx=(4, 12))
        tk.Label(row, text="等级").pack(side=tk.LEFT)
        level_var = tk.StringVar(value=str(existing.get("level", 1)))
        tk.Entry(row, textvariable=level_var, width=5).pack(side=tk.LEFT, padx=4)
        tk.Label(prompt, text="只采用该等级的 lt/rb；原点是人物脚底。没有矩形则不改原设置。",
                 wraplength=470, fg="#777777", anchor="w").pack(fill=tk.X, padx=12, pady=6)

        def close_prompt():
            if pending[0] is not None:
                prompt.after_cancel(pending[0])
            prompt.destroy()

        prompt.protocol("WM_DELETE_WINDOW", close_prompt)

        def run_import():
            try:
                hitbox = load_skill_hitbox(skill_var.get(), level_var.get())
            except Exception as exc:
                messagebox.showerror("读取技能范围失败", str(exc), parent=prompt)
                return
            rect = {"skill_id": hitbox.skill_id, "level": hitbox.level,
                    "lt": list(hitbox.lt), "rb": list(hitbox.rb),
                    "range_bonus": hitbox.range_bonus}
            on_import(rect, hitbox.mob_count)
            self.log(f"⚔️ [本地技能范围] {hitbox.skill_id} Lv{hitbox.level} "
                     f"lt={hitbox.lt} rb={hitbox.rb}，原点=脚底")
            close_prompt()

        tk.Button(prompt, text="读取并应用", command=run_import, bg="#1565c0", fg="white").pack(
            fill=tk.X, padx=12, pady=(3, 10))

    def _apply_primary_wz_hitbox(self, rect, mob_count) -> None:
        """Keep old manual settings as a reversible fallback."""
        if "primary_wz_rect" not in self.config:
            self.config["primary_manual_before_wz"] = {
                key: self.config.get(key) for key in (
                    "attack_reach_x", "attack_reach_y_up", "attack_reach_y_down",
                    "behind_reach_x", "attack_two_way", "attack_area")
            }
        self.config["primary_wz_rect"] = rect
        two_way = rect["lt"][0] < 0 < rect["rb"][0]
        self.config["attack_two_way"] = two_way
        if mob_count is not None:
            self.config["attack_area"] = mob_count > 1
        self._setting_primary_wz = True
        try:
            # Legacy reach is also consulted by pursuit. Keep its estimate in
            # sync while the exact WZ rectangle drives actual hit testing.
            estimates = (
                ("attack_reach_x", "rx_scale", min(800, max(abs(rect["lt"][0]),
                                                         abs(rect["rb"][0])) + rect.get("range_bonus", 0))),
                ("attack_reach_y_up", "ryu_scale", min(400, max(0, -rect["lt"][1]))),
                ("attack_reach_y_down", "ryd_scale", min(400, max(0, rect["rb"][1]))),
                ("behind_reach_x", "rb_scale", min(800, max(0, rect["rb"][0]))),
            )
            for cfg_key, scale_name, estimate in estimates:
                self.config[cfg_key] = estimate
                if hasattr(self, scale_name):
                    getattr(self, scale_name).set(estimate)
            if hasattr(self, "attack_two_way_var"):
                self.attack_two_way_var.set(two_way)
            if hasattr(self, "attack_area_var"):
                self.attack_area_var.set(bool(self.config["attack_area"]))
            if self.detector is not None:
                self.detector.attack_two_way = two_way
            if hasattr(self, "combat_fsm"):
                self.combat_fsm.attack_two_way = two_way
        finally:
            self._setting_primary_wz = False
        self._save_config()
        self._refresh_attack_range_summary()

    def _clear_primary_wz_hitbox(self) -> None:
        if self.config.pop("primary_wz_rect", None) is not None:
            backup = self.config.pop("primary_manual_before_wz", {})
            self._setting_primary_wz = True
            try:
                for key, scale_name in (("attack_reach_x", "rx_scale"),
                                        ("attack_reach_y_up", "ryu_scale"),
                                        ("attack_reach_y_down", "ryd_scale"),
                                        ("behind_reach_x", "rb_scale")):
                    value = backup.get(key)
                    if value is not None:
                        self.config[key] = value
                        if hasattr(self, scale_name):
                            getattr(self, scale_name).set(int(value))
                for key, var_name in (("attack_two_way", "attack_two_way_var"),
                                      ("attack_area", "attack_area_var")):
                    if backup.get(key) is not None:
                        self.config[key] = bool(backup[key])
                        if hasattr(self, var_name):
                            getattr(self, var_name).set(bool(backup[key]))
                if self.detector is not None:
                    self.detector.attack_two_way = bool(self.config.get("attack_two_way"))
                if hasattr(self, "combat_fsm"):
                    self.combat_fsm.attack_two_way = bool(self.config.get("attack_two_way"))
            finally:
                self._setting_primary_wz = False
            self._save_config()
            self._refresh_attack_range_summary()
            self.log("⚔️ [本地技能范围] 主攻击已恢复为原有手动范围")

    def _edit_attack_skill(self, parent: tk.Toplevel, skill_id: Optional[str], refresh) -> None:
        extras = [dict(item) for item in self.config.get("extra_attack_skills", [])]
        existing = next((item for item in extras if item.get("id") == skill_id), None)
        if skill_id is not None and existing is None:
            return
        win = tk.Toplevel(parent)
        win.title("编辑攻击技能" if existing else "添加攻击技能")
        fit_window_to_work_area(win, (460, 520), (390, 440), parent=parent)
        win.transient(parent)
        win.grab_set()
        fields = (
            ("名称", "name", "新技能"),
            ("按键", "key", ""),
            ("前向射程 px", "reach_x", "260"),
            ("上方范围 px", "reach_y_up", "140"),
            ("下方范围 px", "reach_y_down", "140"),
            ("身后范围 px", "behind_x", "0"),
        )
        variables = {}
        grid = tk.Frame(win)
        grid.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        for row, (label, key, default) in enumerate(fields):
            tk.Label(grid, text=label, anchor="w").grid(row=row, column=0, sticky="ew", pady=4)
            variable = tk.StringVar(value=str((existing or {}).get(key, default)))
            variables[key] = variable
            tk.Entry(grid, textvariable=variable, width=22).grid(row=row, column=1, sticky="ew", pady=4)
        vk_var = tk.StringVar(value=str((existing or {}).get("vk", 0)))
        two_way = tk.BooleanVar(value=bool((existing or {}).get("two_way", False)))
        area = tk.BooleanVar(value=bool((existing or {}).get("area", False)))
        imported_wz = [(existing or {}).get("wz_rect")]
        wz_status = tk.StringVar(value=(
            f"本地范围：{imported_wz[0]['skill_id']} Lv{imported_wz[0]['level']} "
            f"{imported_wz[0]['lt']} → {imported_wz[0]['rb']}（脚底原点）"
            if isinstance(imported_wz[0], dict) else "手动范围"))
        tk.Checkbutton(grid, text="双向技能（左右对称）", variable=two_way).grid(
            row=len(fields), column=0, columnspan=2, sticky="w", pady=6)
        tk.Checkbutton(grid, text="群体技能（多目标时优先）", variable=area).grid(
            row=len(fields) + 1, column=0, columnspan=2, sticky="w", pady=3)
        key_button = tk.Button(grid, text="录制按键")
        key_button.grid(row=1, column=2, padx=5)

        def record_key():
            key_button.configure(text="按任意键…")
            win.focus_set()

            def captured(event):
                aliases = {"control_l": "ctrl", "control_r": "ctrl", "shift_l": "shift",
                           "shift_r": "shift", "alt_l": "alt", "alt_r": "alt",
                           "return": "enter", "prior": "pageup", "next": "pagedown"}
                variables["key"].set(aliases.get(event.keysym.lower(), event.keysym.lower()))
                vk_var.set(str(int(event.keycode)))
                key_button.configure(text="录制按键")
                win.unbind("<Key>")
                return "break"

            win.bind("<Key>", captured)

        key_button.configure(command=record_key)
        grid.columnconfigure(1, weight=1)
        wz_row = tk.Frame(win)
        wz_row.pack(fill=tk.X, padx=12, pady=3)
        tk.Button(wz_row, text="从本地技能读取范围", command=lambda: self._prompt_wz_skill_hitbox(
            win, import_wz, imported_wz[0])).pack(side=tk.LEFT)
        tk.Label(win, textvariable=wz_status, wraplength=420, anchor="w",
                 justify=tk.LEFT, fg="#236d80").pack(fill=tk.X, padx=12)

        def manual_changed(*_):
            if imported_wz[0] is not None and not importing[0]:
                imported_wz[0] = None
                wz_status.set("手动范围（已修改数值）")

        importing = [False]
        for key in ("reach_x", "reach_y_up", "reach_y_down", "behind_x"):
            variables[key].trace_add("write", manual_changed)

        def import_wz(rect, mob_count):
            importing[0] = True
            try:
                lt, rb = rect["lt"], rect["rb"]
                if variables["name"].get().strip() in ("", "新技能"):
                    try:
                        matches = search_skill_names(rect["skill_id"], limit=1)
                    except (OSError, ValueError):
                        matches = []
                    if matches:
                        variables["name"].set(matches[0].name)
                for key, value in (("reach_x", max(abs(lt[0]), abs(rb[0]))),
                                   ("reach_y_up", max(0, -lt[1])),
                                   ("reach_y_down", max(0, rb[1])),
                                   ("behind_x", max(0, -lt[0]))):
                    variables[key].set(str(value))
                two_way.set(lt[0] < 0 < rb[0])
                if mob_count is not None and mob_count > 1:
                    area.set(True)
                imported_wz[0] = rect
                wz_status.set(f"本地范围：{rect['skill_id']} Lv{rect['level']} "
                              f"{lt} → {rb}（脚底原点）")
            finally:
                importing[0] = False
        tk.Label(win, text="按键请点“录制按键”；新怪物默认仍使用主攻击，需在怪物列表单独指定。",
                 wraplength=380, fg="#666666").pack(fill=tk.X, padx=12)

        def save():
            name = variables["name"].get().strip()
            key = variables["key"].get().strip().lower()
            try:
                vk = int(vk_var.get())
                reach = {field: int(variables[field].get()) for field in
                         ("reach_x", "reach_y_up", "reach_y_down", "behind_x")}
            except ValueError:
                messagebox.showerror("技能设置", "按键尚未录制，或射程不是整数。", parent=win)
                return
            if not name or not key or not 1 <= vk <= 255 or any(
                not 0 <= value <= (4000 if imported_wz[0] else 800) for value in reach.values()
            ):
                messagebox.showerror("技能设置", "名称、录制按键和 0–800px 范围均须有效。", parent=win)
                return
            if any(item["key"].lower() == key for item in skills_from_config(self.config)
                   if item["id"] != skill_id):
                messagebox.showerror("技能设置", "按键已被其他攻击技能使用。", parent=win)
                return
            if skill_id is None:
                used = {item.get("id") for item in extras}
                number = 2
                while f"skill_{number}" in used:
                    number += 1
                new_id = f"skill_{number}"
            else:
                new_id = skill_id
            new_skill = {"id": new_id, "name": name, "key": key, "vk": vk,
                         **reach, "two_way": bool(two_way.get()), "area": bool(area.get())}
            if imported_wz[0] is not None:
                new_skill["wz_rect"] = imported_wz[0]
            self.config["extra_attack_skills"] = [
                new_skill if item.get("id") == new_id else item for item in extras
            ] if existing else extras + [new_skill]
            self._save_config()
            self._refresh_mob_skill_labels()
            self.log(f"⚔️ [攻击技能] {new_id} {name} [{key.upper()}] 射程={reach}")
            refresh()
            win.destroy()

        tk.Button(win, text="保存技能", command=save, bg="#2e7d32", fg="white").pack(
            fill=tk.X, padx=12, pady=(6, 12))

    def _build_range_card(self):
        self._setting_primary_wz = True  # Scale.set during construction is not a user edit.
        card = tk.LabelFrame(getattr(self, "_panel_parent", self.sf), text="🎯 攻击判定盒",
                              font=("Segoe UI", 10, "bold"), fg="#ff9100", bg="#202026", padx=6, pady=2)
        card.pack(fill=tk.X, padx=8, pady=1)

        range_header = tk.Frame(card, bg="#202026")
        range_header.pack(fill=tk.X)
        self.lbl_attack_range_summary = tk.Label(
            range_header, text="", font=("Consolas", 7, "bold"),
            fg="#ffcc80", bg="#202026", anchor=tk.W,
        )
        self.lbl_attack_range_summary.pack(side=tk.LEFT, fill=tk.X, expand=True)

        range_details = tk.Frame(card, bg="#202026")
        self.attack_range_details_frame = range_details

        def _toggle_range_details():
            if range_details.winfo_manager():
                range_details.pack_forget()
                self.btn_attack_range_details.configure(text="详细设置 ▸")
            else:
                range_details.pack(fill=tk.X, pady=(2, 0), after=range_header)
                self.btn_attack_range_details.configure(text="详细设置 ▾")

        self.btn_attack_range_details = tk.Button(
            range_header, text="详细设置 ▸", command=_toggle_range_details,
            font=("Segoe UI", 7), fg="#ffffff", bg="#e65100",
            activeforeground="#ffffff", activebackground="#ef6c00",
            relief=tk.FLAT, bd=0, highlightthickness=0, padx=4, pady=0,
        )
        self.btn_attack_range_details.pack(side=tk.RIGHT, padx=(4, 0))
        wz_buttons = tk.Frame(range_details, bg="#202026")
        wz_buttons.pack(fill=tk.X, padx=4, pady=(0, 3))
        tk.Button(
            wz_buttons, text="从本地技能读取范围", font=("Segoe UI", 8),
            command=lambda: self._prompt_wz_skill_hitbox(
                self.root, self._apply_primary_wz_hitbox,
                self.config.get("primary_wz_rect")),
        ).pack(side=tk.LEFT)
        tk.Button(wz_buttons, text="恢复手动范围", font=("Segoe UI", 8),
                  command=self._clear_primary_wz_hitbox).pack(side=tk.LEFT, padx=4)
        self._refresh_attack_range_summary()

        self.monster_full_scan_every_frame_var = tk.BooleanVar(
            value=bool(self.config.get("monster_full_scan_every_frame", False))
        )

        def _toggle_monster_scan_mode():
            full_speed = bool(self.monster_full_scan_every_frame_var.get())
            self.config["monster_full_scan_every_frame"] = full_speed
            if self.detector is not None:
                self.detector.monster_redetect_interval = (
                    1
                    if full_speed
                    else max(1, int(self.config.get("monster_redetect_interval", 4)))
                )
                self.detector.clear_monster_tracks()
            with self._monster_batch_lock:
                self._monster_latest_batch = None
            with self.lock:
                if self.latest_result is not None:
                    self.latest_result.monsters = []
                    self.latest_result.locked_target = None
            with self._monster_fps_lock:
                self._monster_detection_fps = 0.0
                self._monster_full_scan_fps = 0.0
                self._monster_full_scan_gap_ms = None
                self._monster_full_scan_cost_ms = None
                self._monster_fps_count = 0
                self._monster_fps_timer = time.perf_counter()
            self._save_config()
            self.log(
                "👾 [怪物识别模式] "
                + (
                    "全速完整模板扫描：每张新捕获帧直接完整匹配，轻量追踪流水线已停用"
                    if full_speed
                    else "完整模板扫描+轻量追踪：55ms实扫与60Hz光流追踪并行"
                )
            )

        tk.Checkbutton(
            range_details,
            text="全速完整模板扫描（关闭时：完整模板扫描 + 轻量追踪）",
            variable=self.monster_full_scan_every_frame_var,
            command=_toggle_monster_scan_mode,
            font=("Segoe UI", 8, "bold"),
            fg="#80cbc4",
            bg="#202026",
            activeforeground="#b2dfdb",
            activebackground="#202026",
            selectcolor="#202026",
            anchor=tk.W,
        ).pack(fill=tk.X, padx=4, pady=(0, 4))

        # 2 列网格排布 (减少垂直占用，防止屏幕截断)
        sliders = [
            ("置信度 Threshold", "monster_threshold", 45, 95, 100, "#69f0ae", "_on_thresh_changed", "thresh", 0, 0),
            ("追踪缓冲 Buffer", "tracker_buffer_ms", 0, 400, 1, "#00e5ff", "_on_tracker_buffer_changed", "buf", 0, 1),
            ("前向射程 Reach X",     "attack_reach_x",    50, 800, 1, "#ff9100", "_on_reach_x_changed", "rx", 1, 0),
            ("上方范围 Reach Y+",    "attack_reach_y_up", 0, 400, 1, "#ff9100", "_on_reach_y_up_changed", "ryu", 1, 1),
            ("下方范围 Reach Y−",    "attack_reach_y_down", 0, 400, 1, "#ff9100", "_on_reach_y_down_changed", "ryd", 2, 0),
            ("游击范围 Skirmish",    "skirmish_range_x", 0, 600, 1, "#4dd0e1", "_on_skirmish_range_changed", "skirmish", 3, 0),
        ]

        grid_frame = tk.Frame(range_details, bg="#202026")
        grid_frame.pack(fill=tk.X)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        for label, cfg_key, lo, hi, divisor, color, handler, pfx, r, c in sliders:
            cell = tk.Frame(grid_frame, bg="#202026", padx=4, pady=1)
            cell.grid(row=r, column=c, sticky="ew")

            raw = self.config.get(cfg_key, (lo + hi) // 2)
            init_val = int(raw * divisor) if divisor == 100 else int(raw)
            if divisor == 100:
                init_txt = f"{raw:.2f}"
            elif "ms" in cfg_key:
                init_txt = f"{int(raw)} ms"
            else:
                init_txt = f"{int(raw)} px"

            h_row = tk.Frame(cell, bg="#202026")
            h_row.pack(fill=tk.X)
            tk.Label(h_row, text=label, font=("Segoe UI", 8), fg="#e0e0e0", bg="#202026", anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(h_row, text=init_txt, font=("Consolas", 8, "bold"), fg=color, bg="#202026", width=6)
            val_lbl.pack(side=tk.RIGHT)
            setattr(self, f"{pfx}_val_lbl", val_lbl)

            sc = tk.Scale(cell, from_=lo, to=hi, orient=tk.HORIZONTAL,
                          bg="#2a2a32", fg="#ffffff", highlightthickness=0,
                          troughcolor="#151518", activebackground=color,
                          showvalue=0, command=getattr(self, handler))
            sc.set(init_val)
            sc.pack(fill=tk.X, pady=(0, 2))
            setattr(self, f"{pfx}_scale", sc)

        # 双向技能不再受角色朝向和 Behind X 限制，攻击盒改为左右对称。
        self.attack_two_way_var = tk.BooleanVar(value=bool(self.config.get("attack_two_way", False)))
        def _toggle_attack_two_way():
            if not self._setting_primary_wz:
                self.config.pop("primary_wz_rect", None)
                self.config.pop("primary_manual_before_wz", None)
            enabled = bool(self.attack_two_way_var.get())
            self.detector.attack_two_way = enabled
            if hasattr(self, "combat_fsm"):
                self.combat_fsm.attack_two_way = enabled
            self.config["attack_two_way"] = enabled
            self._save_config()
            self._refresh_attack_range_summary()
        tk.Checkbutton(
            grid_frame,
            text="攻击技能双向（左右对称）",
            variable=self.attack_two_way_var,
            command=_toggle_attack_two_way,
            font=("Segoe UI", 8), fg="#ffcc80", bg="#202026",
            activeforeground="#ffe0b2", activebackground="#202026",
            selectcolor="#202026", anchor=tk.W,
        ).grid(row=2, column=1, sticky="w", padx=4, pady=1)

        self.attack_area_var = tk.BooleanVar(value=bool(self.config.get("attack_area", False)))
        def _toggle_attack_area():
            self.config["attack_area"] = bool(self.attack_area_var.get())
            self._save_config()
        tk.Checkbutton(
            grid_frame, text="主攻击为群体技能（多目标优先）",
            variable=self.attack_area_var, command=_toggle_attack_area,
            font=("Segoe UI", 8), fg="#ffcc80", bg="#202026",
            activeforeground="#ffe0b2", activebackground="#202026",
            selectcolor="#202026", anchor=tk.W,
        ).grid(row=3, column=1, sticky="w", padx=4, pady=1)

        # 身后近身 Behind X 单独单行
        cell_b = tk.Frame(range_details, bg="#202026", padx=4, pady=1)
        cell_b.pack(fill=tk.X)
        h_row_b = tk.Frame(cell_b, bg="#202026")
        h_row_b.pack(fill=tk.X)
        tk.Label(h_row_b, text="身后近身 Behind X", font=("Segoe UI", 8), fg="#e0e0e0", bg="#202026", anchor="w").pack(side=tk.LEFT)
        raw_b = self.config.get("behind_reach_x", 40)
        self.rb_val_lbl = tk.Label(h_row_b, text=f"{int(raw_b)} px", font=("Consolas", 8, "bold"), fg="#ff9100", bg="#202026", width=6)
        self.rb_val_lbl.pack(side=tk.RIGHT)

        self.rb_scale = tk.Scale(cell_b, from_=0, to=800, orient=tk.HORIZONTAL,
                                 bg="#2a2a32", fg="#ffffff", highlightthickness=0,
                                 troughcolor="#151518", activebackground="#ff9100",
                                 showvalue=0, command=self._on_behind_x_changed)
        self.rb_scale.set(int(raw_b))
        self.rb_scale.pack(fill=tk.X, pady=(0, 2))
        self._setting_primary_wz = False

        # 人物面板移速：统一供运动预测、刹停和直爬预按距离换算使用。
        speed_row = tk.Frame(range_details, bg="#202026", padx=4, pady=4)
        speed_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(speed_row, text="人物移速", font=("Segoe UI", 9), fg="#e0e0e0", bg="#202026").pack(side=tk.LEFT)
        tk.Label(speed_row, text="%", font=("Consolas", 9, "bold"), fg="#00e5ff", bg="#202026").pack(side=tk.RIGHT, padx=(4, 0))
        self.movement_speed_var = tk.StringVar(value=f"{float(self.config.get('movement_speed_percent', 103.0)):g}")
        self.ent_movement_speed = tk.Entry(
            speed_row, textvariable=self.movement_speed_var, width=7,
            font=("Consolas", 10, "bold"), fg="#00e5ff", bg="#121418",
            insertbackground="#ffffff", relief=tk.FLAT, justify=tk.CENTER,
        )
        self.ent_movement_speed.pack(side=tk.RIGHT)
        self.ent_movement_speed.bind("<Return>", lambda _e: self._apply_movement_speed())
        self.ent_movement_speed.bind("<FocusOut>", lambda _e: self._apply_movement_speed(silent=True))
        tk.Button(
            speed_row, text="应用", command=self._apply_movement_speed,
            font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#0277bd",
            relief=tk.FLAT, padx=8, pady=1,
        ).pack(side=tk.RIGHT, padx=(0, 6))



    def _build_log_card(self, parent):
        card = tk.Frame(parent, bg="#18181c", padx=6, pady=4)
        card.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(0, 6))
        header = tk.Frame(card, bg="#18181c")
        header.pack(fill=tk.X, pady=(0, 2))
        tk.Label(
            header, text="📜 运行日志 (实时终端)",
            font=("Segoe UI", 9, "bold"), fg="#90a4ae", bg="#18181c",
        ).pack(side=tk.LEFT)
        self.btn_save_current_logs = tk.Button(
            header, text="💾 保存当前日志", command=self._save_current_logs,
            font=("Segoe UI", 8, "bold"), fg="#ffffff", bg="#37474f",
            relief=tk.FLAT, padx=7, pady=1,
        )
        self.btn_save_current_logs.pack(side=tk.RIGHT)
        self.log_text = tk.Text(card, height=4, bg="#121214", fg="#cfd8dc",
                                font=("Consolas", 9), relief=tk.FLAT)
        self.log_text.pack(fill=tk.X)
        self.log("[系统就绪] 支持怪物模糊搜索 + 多平台循环巡航 + 平台拓扑图实时感知！")

    def _save_current_logs(self):
        """Snapshot diagnostics in a worker so large logs never freeze Tk."""
        self.btn_save_current_logs.config(state=tk.DISABLED)
        self.log("💾 [日志快照] 正在保存当前运行日志…")
        results = queue.Queue(maxsize=1)

        def worker():
            try:
                log_dir = os.path.join(os.path.dirname(CONFIG_PATH), "logs")
                archive = save_log_snapshot(log_dir)
                results.put((True, str(archive)))
            except Exception as exc:
                results.put((False, str(exc)))

        def poll_result():
            try:
                success, detail = results.get_nowait()
            except queue.Empty:
                if self.root.winfo_exists():
                    self.root.after(100, poll_result)
                return
            if self.btn_save_current_logs.winfo_exists():
                self.btn_save_current_logs.config(state=tk.NORMAL)
            if success:
                self.log(f"✅ [日志快照] 已保存：{detail}")
            else:
                self.log(f"⚠️ [日志快照] 保存失败：{detail}")

        threading.Thread(target=worker, name="LogSnapshot", daemon=True).start()
        self.root.after(100, poll_result)

    # ── 怪物自定义编辑 ───────────────────────────────────────────────────────
    def open_edit_monster_dialog(self):
        if not self.current_map_info:
            messagebox.showinfo("提示", "请等待地图识别完成或先在下拉框手动选择地图！", parent=self.root)
            return
        map_id   = self.current_map_info.get("map_id", 0)
        map_name = self.current_map_info.get("chinese_name", "当前地图")
        mobs     = self.current_map_info.get("mobs", [])
        EditMonsterDialog(self.root, map_id, map_name, mobs, self._on_save_custom_mobs, get_image_callback=self._get_mob_preview_image)

    def _on_save_custom_mobs(self, map_id: int, map_name: str, new_mobs: List[Dict]):
        # 本图怪物列表与地图表共用 local_maps.json；这样重新 OCR/重启后
        # 仍会从同一张地图记录恢复 mob_ids，而不是落到旧的旁路配置文件。
        ok = self.map_resolver.save_local_map_mobs(map_id, map_name, new_mobs)
        if ok:
            if self.current_map_info:
                self.current_map_info["mobs"] = new_mobs
                self.current_map_info["is_custom"] = False
            self._populate_mob_tree(
                new_mobs,
                (self.current_map_info or {}).get("disabled_mob_ids", []),
            )
            # populate 后必须只加载仍勾选的怪物；直接传 new_mobs 会让
            # disabled_mob_ids 在 UI 看似未选中、检测器却实际继续识别。
            self.detector.load_dynamic_monster_templates(self._get_enabled_mobs())
            self.log(f"[本图怪物保存] 地图【{map_name}】已更新 {len(new_mobs)} 种怪物，已写入 local_maps.json，下次进图自动恢复！")

    def _count_mob_templates(self, mob_id: Any, mob_name: str) -> int:
        """统计指定怪物在模板库中的有效活怪特征图数量"""
        # 新多尺度库严格按 Mob ID 分目录；其中 global 全身图与 mob
        # 特征图都属于有效配对素材，树形列表应据此显示真实数量。
        if mob_id is not None:
            multi_dir = os.path.join(
                getattr(self.detector, "multi_scale_template_root", ""), str(mob_id)
            )
            if os.path.isdir(multi_dir):
                return len(glob.glob(os.path.join(multi_dir, "*.png")))

        tpl_dir = self.detector.template_dir
        if not os.path.exists(tpl_dir):
            return 0
        
        keywords = set()
        if mob_id:
            keywords.add(f"mob_{mob_id}")
            keywords.add(str(mob_id))
            try:
                mid_int = int(mob_id)
                if mid_int in CLASSIC_MOB_ID_MAP:
                    keywords.update(CLASSIC_MOB_ID_MAP[mid_int])
            except Exception:
                pass
        if mob_name:
            clean = mob_name.replace("(", " ").replace(")", " ").replace("/", " ").lower()
            for part in clean.split():
                if len(part.strip()) >= 2:
                    keywords.add(part.strip())
            for k, aliases in MOB_ALIAS_MAP.items():
                if k in clean or any(a in clean for a in aliases):
                    keywords.update(aliases)

        count = 0
        for f in glob.glob(os.path.join(tpl_dir, "*.png")):
            base = os.path.basename(f).lower()
            if any(x in base for x in ["player", "nametag", "medal", "facing", "die", "dead"]):
                continue
            if any(k in base for k in keywords):
                count += 1
        return count

    # ── 怪物悬浮图 ───────────────────────────────────────────────────────────
    def _get_mob_preview_image(self, mob_id_str, mob_name_str) -> Optional[ImageTk.PhotoImage]:
        key = f"{mob_id_str}_{mob_name_str}"
        if key in self.cached_mob_thumbnails:
            return self.cached_mob_thumbnails[key]
        tpl_dir = self.detector.template_dir
        target = None
        
        try:
            mid = int(mob_id_str)
            multi_dir = os.path.join(
                getattr(self.detector, "multi_scale_template_root", ""), str(mid)
            )
            multi_files = glob.glob(os.path.join(multi_dir, "*.png")) if os.path.isdir(multi_dir) else []
            # 预览优先完整怪物图，便于用户确认该 ID 的模板目录。
            if multi_files:
                target = next(
                    (p for p in multi_files if "mob_global" in os.path.basename(p).lower()),
                    multi_files[0],
                )
            m_direct = glob.glob(os.path.join(tpl_dir, f"mob_{mid}*.png")) or \
                       glob.glob(os.path.join(tpl_dir, f"*{mid}*.png"))
            if not target and m_direct:
                target = m_direct[0]
            if not target:
                for kw in CLASSIC_MOB_ID_MAP.get(mid, []):
                    matches = glob.glob(os.path.join(tpl_dir, f"*{kw}*right*.png")) or \
                              glob.glob(os.path.join(tpl_dir, f"*{kw}*.png"))
                    if matches:
                        target = matches[0]
                        break
        except Exception:
            pass

        if not target:
            clean_name = mob_name_str.split("(")[0].split("/")[0].strip().lower().replace(" ", "_")
            if clean_name:
                m = glob.glob(os.path.join(tpl_dir, f"*{clean_name}*.png"))
                if m:
                    target = m[0]

        if not target:
            try:
                mid = int(mob_id_str)
                res = self.downloader.download_and_save_mob_template(mid, custom_name=f"mob_{mid}")
                if res and res[0] and os.path.exists(res[0]):
                    target = res[0]
            except Exception:
                pass

        if target and os.path.exists(target):
            try:
                img = Image.open(target)
                img.thumbnail((110, 110), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                self.cached_mob_thumbnails[key] = tk_img
                return tk_img
            except Exception:
                pass
        return None

    # ── 按键录制 ─────────────────────────────────────────────────────────────
    def _start_key_recording(self, cfg_key: str, btn: tk.Button):
        self.recording_target = cfg_key
        self.active_rec_btn = btn
        btn.config(text="🔴 请按下任意键...", bg="#d32f2f", fg="#ffffff")
        self.log(f"[录制等待] 请按下想绑定为【{cfg_key}】的键...")
        self.root.focus_set()

    def _apply_pet_feed_interval(self):
        try:
            interval = float(self.pet_feed_interval_var.get().strip())
            if not math.isfinite(interval) or not 1.0 <= interval <= 86400.0:
                raise ValueError("interval outside supported range")
        except (TypeError, ValueError):
            self.pet_feed_interval_var.set(
                f"{float(self.config.get('pet_feed_interval_sec', 300.0)):g}")
            messagebox.showwarning("宠物喂食间隔", "请输入 1～86400 秒之间的数值。", parent=self.root)
            return
        self.config["pet_feed_interval_sec"] = interval
        self._save_config()
        if not self.config.get("enable_auto_pet_feed", False):
            timing = "启用自动喂食后才开始计时"
        elif self.combat_fsm.is_running:
            timing = "当前 F6 从现在重新计时"
        else:
            timing = "下次 F6 启动后开始计时"
        self.log(f"🐾 [宠物喂食] 自动间隔设为 {interval:g} 秒；{timing}")

    def _toggle_auto_pet_feed(self):
        enabled = bool(self.enable_auto_pet_feed_var.get())
        if enabled and not str(self.config.get("pet_feed_key", "") or "").strip():
            self.enable_auto_pet_feed_var.set(False)
            messagebox.showwarning("宠物喂食", "请先设置宠物喂食按键。", parent=self.root)
            return
        self.config["enable_auto_pet_feed"] = enabled
        self._save_config()
        if enabled:
            timing = (
                "当前 F6 将从现在等待完整间隔"
                if self.combat_fsm.is_running else "下次 F6 启动后开始计时"
            )
            self.log(f"🐾 [宠物喂食] 已启用；{timing}")
        else:
            self.log("🐾 [宠物喂食] 已关闭")

    def _on_physical_key_pressed(self, event):
        if not self.recording_target:
            return
        keysym = event.keysym.lower()
        keycode = event.keycode
        name_map = {
            "control_l": "ctrl", "control_r": "ctrl",
            "shift_l": "shift",  "shift_r": "shift",
            "alt_l": "alt",      "alt_r": "alt",
            "return": "enter",   "prior": "pageup", "next": "pagedown",
        }
        final = name_map.get(keysym, keysym)
        target = self.recording_target
        if target == "pet_feed_key" and any(
            str(skill["key"]).lower() == final for skill in skills_from_config(self.config)
        ):
            if self.active_rec_btn:
                old = str(self.config.get("pet_feed_key", "") or "").upper()
                self.active_rec_btn.config(text=f"⌨️ {old or '未设置'}",
                                           bg="#2a2a32", fg="#00e5ff")
            self.recording_target = None
            self.active_rec_btn = None
            messagebox.showwarning("宠物喂食按键", "喂食键不能与任何攻击技能键相同。", parent=self.root)
            return
        vk_key = (
            f"{target[:-4]}_vk"
            if target.endswith("_key")
            else f"{target}_vk"
        )
        if target in ("hp_potion_key", "mp_potion_key"):
            # Potion key changes are staged until the visible slot and its
            # quantity have passed Apply validation; F6 keeps old bindings.
            self._pending_potion_bindings[target] = (final, keycode)
            with self._status_reading_lock:
                self.potion_stock_monitor = PotionStockMonitor()
            if self.config.get("enable_auto_potion", False):
                self.config["enable_auto_potion"] = False
                self.status_auto_potion_var.set(False)
                self._save_config()
            self.lbl_potion_bar_roi.configure(text="键位待应用", fg="#ffb74d")
        else:
            self.config[target] = final
            self.config[vk_key] = keycode
            self._save_config()
        if self.active_rec_btn:
            self.active_rec_btn.config(text=f"⌨️ {final.upper()}", bg="#2a2a32", fg="#00e5ff")
        self.log(f"[录制成功] 【{target}】-> [{final.upper()}]  VK=0x{keycode:02X}"
                 + ("（待应用验证库存）" if target in ("hp_potion_key", "mp_potion_key") else ""))
        self.recording_target = None
        self.active_rec_btn = None

    # ── 滑动条回调 ───────────────────────────────────────────────────────────
    def _refresh_attack_range_summary(self):
        label = getattr(self, "lbl_attack_range_summary", None)
        if label is None:
            return
        wz = self.config.get("primary_wz_rect")
        if isinstance(wz, dict):
            label.config(text=f"本地 {wz.get('skill_id')} Lv{wz.get('level')} "
                              f"lt{wz.get('lt')} rb{wz.get('rb')}（脚底）")
            return
        prefix = "双向 · " if bool(self.config.get("attack_two_way", False)) else ""
        label.config(
            text=(
                f"{prefix}前{int(self.config.get('attack_reach_x', 260))} "
                f"上{int(self.config.get('attack_reach_y_up', 140))} "
                f"下{int(self.config.get('attack_reach_y_down', 140))} "
                f"后{int(self.config.get('behind_reach_x', 40))} "
                f"游击{int(self.config.get('skirmish_range_x', 0))}px"
            )
        )

    def _on_thresh_changed(self, val):
        v = int(val) / 100.0
        self.thresh_val_lbl.config(text=f"{v:.2f}")
        self.detector.monster_threshold = v
        self.config["monster_threshold"] = v
        self._save_config()

    def _on_tracker_buffer_changed(self, val):
        ms = int(val)
        self.buf_val_lbl.config(text=f"{ms} ms")
        self.detector.tracker_buffer_time_sec = ms / 1000.0
        self.config["tracker_buffer_ms"] = ms
        self._save_config()

    def _on_reach_x_changed(self, val):
        if not getattr(self, "_setting_primary_wz", False):
            self.config.pop("primary_wz_rect", None)
            self.config.pop("primary_manual_before_wz", None)
        v = int(val)
        self.rx_val_lbl.config(text=f"{v} px")
        self.detector.attack_reach_x = v
        if hasattr(self, "combat_fsm"):
            self.combat_fsm.attack_reach_x = v
        self.config["attack_reach_x"] = v
        self._save_config()
        self._refresh_attack_range_summary()

    def _on_reach_y_up_changed(self, val):
        if not getattr(self, "_setting_primary_wz", False):
            self.config.pop("primary_wz_rect", None)
            self.config.pop("primary_manual_before_wz", None)
        v = int(val)
        self.ryu_val_lbl.config(text=f"{v} px")
        self.detector.attack_reach_y_up = v
        self.detector.attack_reach_y = v  # 旧字段兼容
        if hasattr(self, "combat_fsm"):
            self.combat_fsm.attack_reach_y_up = v
        self.config["attack_reach_y_up"] = v
        self.config["attack_reach_y"] = v
        self._save_config()
        self._refresh_attack_range_summary()

    def _on_reach_y_down_changed(self, val):
        if not getattr(self, "_setting_primary_wz", False):
            self.config.pop("primary_wz_rect", None)
            self.config.pop("primary_manual_before_wz", None)
        v = int(val)
        self.ryd_val_lbl.config(text=f"{v} px")
        self.detector.attack_reach_y_down = v
        if hasattr(self, "combat_fsm"):
            self.combat_fsm.attack_reach_y_down = v
        self.config["attack_reach_y_down"] = v
        self._save_config()
        self._refresh_attack_range_summary()

    def _on_skirmish_range_changed(self, val):
        v = int(val)
        self.skirmish_val_lbl.config(text=f"{v} px")
        self.detector.skirmish_range_x = v
        self.config["skirmish_range_x"] = v
        self._save_config()
        self._refresh_attack_range_summary()

    # ── 视频流 HUD 实时渲染 (最高 60 FPS 无锁渲染) ───────────────────────────
    def _refresh_platform_status_label(self):
        """用实时坐标线程的结果刷新主界面平台状态（Tk 主线程调用）。"""
        if not hasattr(self, "lbl_curr_platform_status"):
            return
        world_pos = getattr(self, "current_player_world_pos", None)
        platform = getattr(self, "current_player_platform", None)
        ladder = getattr(self, "current_player_ladder", None)
        climbing = bool(getattr(self, "current_player_is_climbing", False))
        if climbing and ladder is not None and world_pos is not None:
            self.lbl_curr_platform_status.config(
                text=(f"📍 角色当前状态: 🧗 攀爬中 [{ladder.kind_name} #{ladder.id}] "
                      f"(X={ladder.x}, Y:{ladder.y1}~{ladder.y2}) "
                      f"[世界坐标: ({world_pos[0]}, {world_pos[1]})]"),
                fg="#ffd54f",
            )
        elif platform is not None and world_pos is not None:
            hint = f" [可攀爬{ladder.kind_name}#{ladder.id}]" if ladder else ""
            p_prefix = "长平台" if getattr(self, "merge_short_platforms", False) else "平台"
            self.lbl_curr_platform_status.config(
                text=(f"📍 🟢 {p_prefix} #{platform.id}{hint} "
                      f"[世界坐标: ({world_pos[0]}, {world_pos[1]})] "
                      f"(宽{platform.length}px)"),
                fg="#00e676",
            )
        elif ladder is not None and world_pos is not None:
            self.lbl_curr_platform_status.config(
                text=(f"📍 角色当前位置: 🧗 靠近{ladder.kind_name} #{ladder.id} "
                      f"[世界坐标: ({world_pos[0]}, {world_pos[1]})]"),
                fg="#ffd54f",
            )
        if hasattr(self, "lbl_cross_map_patrol_status"):
            controller = getattr(self, "world_patrol_controller", None)
            if controller is not None:
                enabled = bool(
                    self.config.get("cross_map_patrol_enabled", False)
                    or controller.recovery_armed
                    or controller.recovery_active
                )
                self.lbl_cross_map_patrol_status.config(
                    text="🌐 " + controller.status_text(),
                    fg="#80deea" if enabled else "#90a4ae",
                )

    def _refresh_ui_video(self):
        if not self.is_running:
            return
        # 高帧率抓取最新原生游戏帧并叠加实时 HUD
        # ScreenCapture already publishes the latest frame; render_debug
        # creates its own canvas copy for overlays, so avoid a second full
        # frame copy on the UI thread.
        frame = self.capture.capture_frame(copy=False)
        if frame is not None:
            with self.lock:
                res = self.latest_result
                ladder_cols = list(self.latest_ladder_cols) if hasattr(self, "latest_ladder_cols") else []
            if self._recognition_pause_event.is_set():
                res = None
                ladder_cols = []
            fps = self.capture.fps
            if res is not None:
                dbg = self.detector.render_debug(frame, res, fps=fps)
                
                # 动态录制足迹航点
                if getattr(self, "waypoint_mgr", None) and self.waypoint_mgr.is_recording and res.player_pos is not None:
                    ok = self.waypoint_mgr.record_step(res.player_pos, action="WALK")
                    if ok:
                        self.lbl_route_status.config(
                            text=f"路线状态: 正在录制中 (已采点 {len(self.waypoint_mgr.waypoints)} 个)...",
                            fg="#00e5ff"
                        )
            else:
                dbg = frame

            # 实时视口纯几何叠加梯子与绳索识别边框及 X 坐标 (耗时 < 0.1ms)
            if hasattr(self, "ladder_aligner") and self.ladder_aligner and ladder_cols:
                p_screen = res.player_pos if (res and res.player_pos) else None
                dbg = self.ladder_aligner.draw_ladder_rope_overlay(dbg, columns=ladder_cols, player_pos=p_screen)

    def _on_behind_x_changed(self, val):
        if not getattr(self, "_setting_primary_wz", False):
            self.config.pop("primary_wz_rect", None)
            self.config.pop("primary_manual_before_wz", None)
        v = int(val)
        self.rb_val_lbl.config(text=f"{v} px")
        self.detector.behind_reach_x = v
        if hasattr(self, "combat_fsm"):
            self.combat_fsm.behind_reach_x = v
        self.config["behind_reach_x"] = v
        self._save_config()
        self._refresh_attack_range_summary()

    def _apply_movement_speed(self, silent: bool = False):
        """应用人物面板移速，避免输入框失焦时因半截内容干扰运行。"""
        try:
            value = float(self.movement_speed_var.get().strip())
            if not 1.0 <= value <= 200.0:
                raise ValueError
        except (TypeError, ValueError):
            if not silent:
                self.log("⚠️ [人物移速] 请输入 1～200 的百分比数值，例如 103")
            return
        applied = self.horizontal_motion.set_speed_percent(value)
        self.horizontal_kalman.set_speed_percent(applied)
        self.config["movement_speed_percent"] = applied
        self.movement_speed_var.set(f"{applied:g}")
        self._save_config()
        if not silent:
            self.log(f"⚙️ [人物移速] 已应用 {applied:g}%（水平速度上限 {125.0 * applied / 100.0:.2f}px/s）")

    # ── 地图相关 ─────────────────────────────────────────────────────────────
    def _queue_map_ui_update(self, res: Dict) -> None:
        """从任意工作线程安全投递地图同步结果。"""
        if isinstance(res, dict):
            self._map_ui_update_queue.put(res)

    def _drain_map_ui_updates(self) -> None:
        """仅由 Tk 主线程执行的地图 UI 更新入口。"""
        try:
            # 同一轮 OCR 可能有多个结果；保留队列里最后一项，避免旧结果
            # 紧随新结果覆盖界面。
            latest = None
            pending_name = None
            while True:
                try:
                    item = self._map_ui_update_queue.get_nowait()
                    if "_ocr_unmatched_name" in item:
                        pending_name = item["_ocr_unmatched_name"]
                    else:
                        latest = item
                except queue.Empty:
                    break
            if latest is not None:
                self._update_map_ui(latest)
            elif pending_name and self.current_map_info is None and hasattr(self, "ent_map_name"):
                # 输入框仍保留原始 OCR 名，方便用户直接填写 MapID 后“应用”；
                # 橙色文字与日志提示未匹配，不能把说明文字存进地图表。
                if self.ent_map_name.get() != pending_name:
                    self.ent_map_name.delete(0, tk.END)
                    self.ent_map_name.insert(0, pending_name)
                self.ent_map_name.config(fg="#ffb74d")
        except Exception as exc:
            # 不再静默吞掉 UI 更新异常，控制台必须能看到真实原因。
            print(f"[地图 UI 更新异常] {exc}")
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(40, self._drain_map_ui_updates)
            except tk.TclError:
                pass

    def _on_manual_map_id_entered(self, override_id: Optional[str] = None):
        raw = override_id if override_id is not None else self.ent_map_id.get().strip()
        if not raw or not raw.isdigit():
            self.log(f"[输入错误] 请输入纯数字 MapID（如 100040101）！")
            return
        mid = int(raw)
        # 启动时通过 override_id 注入默认值不属于用户手动锁定，
        # 首次地图识别仍以 OCR 为准。
        if override_id is None:
            self._manual_map_override_id = mid
        if override_id is None:
            self.log(f"[手动输入 MapID] 正在同步地图 ID【{mid}】及其怪物...")
        def worker():
            res = self.map_resolver.auto_detect_and_sync_map(None, manual_map_id=mid)
            if res:
                self._queue_map_ui_update(res)
            else:
                self.log(f"[失败] 未能获取到 MapID【{mid}】的信息！")
        threading.Thread(target=worker, daemon=True).start()

    def _on_apply_map_changes(self):
        """确认并应用地图名/MapID输入框中的修改。"""
        map_name = self.ent_map_name.get().strip()
        map_id = self.ent_map_id.get().strip()
        if not map_name and not map_id:
            messagebox.showwarning("输入错误", "地图名和 MapID 不能同时为空！", parent=self.root)
            return
        if map_id and not map_id.isdigit():
            messagebox.showwarning("输入错误", "MapID 必须是纯数字。", parent=self.root)
            return
        if not messagebox.askyesno(
            "确认应用地图修改",
            f"确定应用以下地图设置吗？\n\n地图名：{map_name or '（未填写）'}\nMapID：{map_id or '（按地图名解析）'}",
            parent=self.root,
        ):
            return
        if map_id:
            if map_name:
                saved = self.map_resolver.update_local_map_id(map_name, int(map_id))
                self.log(f"[本地地图表] 已保存【{map_name}】 -> {map_id}" if saved
                         else f"[本地地图表] 未找到【{map_name}】对应记录，未回写 MapID")
            self._on_manual_map_id_entered(map_id)
            # 只有确认“应用修改”后才锁定用户指定的 MapID。
            self._manual_map_override_id = int(map_id)
        else:
            def worker():
                res = self.map_resolver.auto_detect_and_sync_map(None, manual_map_name=map_name)
                if res:
                    self._queue_map_ui_update(res)
                else:
                    self.log(f"[失败] 本地/内置地图表未找到【{map_name}】")
            threading.Thread(target=worker, daemon=True).start()

    def _on_manual_map_selected(self, event=None):
        sel = self.combo_map.get()
        self.log(f"[手动选图] 切换至【{sel}】...")
        def worker():
            res = self.map_resolver.auto_detect_and_sync_map(None, manual_map_name=sel)
            if res:
                self._manual_map_override_id = int(res.get("map_id")) if str(res.get("map_id", "")).isdigit() else None
                self._queue_map_ui_update(res)
        threading.Thread(target=worker, daemon=True).start()


    def on_force_ocr(self):
        self.log("正在立即执行 OCR 探测...")
        self._manual_map_override_id = None
        frame = self.capture.capture_frame()
        if frame is None:
            self.log("[错误] 无法获取游戏画面！")
            return
        def worker():
            res = self.map_resolver.auto_detect_and_sync_map(frame, force_sync=True)
            if res:
                self._queue_map_ui_update(res)
            else:
                self.log(f"[OCR] 地图未变化: 【{self.map_resolver.current_map_name}】")
        threading.Thread(target=worker, daemon=True).start()

    def _update_map_ui(self, res: Dict):
        m_name = res.get("chinese_name", "未知地图")
        m_id   = res.get("map_id", "Unknown")
        gms    = res.get("gms_name", "")
        mobs   = res.get("mobs", [])
        is_custom = res.get("is_custom", False)

        # 地图名称 / ID 是 OCR 识别完成后最先应当反馈的 UI 状态，不能
        # 被后续的运动模型重置、模板加载或异步拓扑构建阻塞；否则控制台
        # 已确认识别到新图，输入框却会一直停在“正在全自动探测中…”。
        if hasattr(self, "ent_map_name"):
            self.ent_map_name.delete(0, tk.END)
            self.ent_map_name.insert(0, str(m_name))
            self.ent_map_name.config(fg="#00e676")
        if hasattr(self, "ent_map_id"):
            self.ent_map_id.delete(0, tk.END)
            self.ent_map_id.insert(0, str(m_id))

        # 地图哨兵会周期性重复识别同一张地图。只有 ID 真正变化时才
        # 清空世界坐标/运动模型；否则会把连续预测坐标每隔一轮重置，
        # 导致拓扑图 YOU 标记看起来始终不动。
        previous_map_id = None
        try:
            previous_map_id = str((getattr(self, "current_map_info", None) or {}).get("map_id"))
        except Exception:
            pass
        map_changed = previous_map_id not in (None, "None", "") and previous_map_id != str(m_id)
        first_map_sync = getattr(self, "current_map_info", None) is None

        # 强制 OCR/其它同步线程也可能抢先识别并排队新地图；只要 UI 已经
        # 确认 MapID 改变，就视为传送门加速识别完成，避免 pending 永久保留。
        if map_changed and self._get_portal_ocr_pending() is not None:
            try:
                self._finish_portal_ocr_acceleration(int(m_id))
            except (TypeError, ValueError):
                pass

        self.current_map_info = res
        self._sync_current_path_map_card(m_id)
        # 地图名称已切换而新拓扑仍在后台构建时，绝不能继续使用上一张
        # 地图的 miniMap 参数换算当前黄点；否则会出现“黄点正确、世界
        # 坐标完全不可能”的混合状态。
        if map_changed or first_map_sync:
            self.platform_graph = None
            self._current_graph_map_data = None
            self._graph_target_map_id = str(m_id)
            # 拓扑弹窗不能继续保留上一张地图的静态底图。
            # 新图在后台构建期间显示“正在加载”，构建完成后再强制重绘。
            if self.topology_dialog and self.topology_dialog.is_open:
                self.topology_dialog.base_bgr_img = None
                self.topology_dialog.last_map_id = None
                self.topology_dialog.last_active_path = None
                self.topology_dialog.last_patrol = None
                self.topology_dialog.lbl_info.config(
                    text=f"⏳ 正在加载地图拓扑：{m_name} (MapID: {m_id})",
                    fg="#ffb74d",
                )
            # 坐标缓存也必须随地图一起失效；否则拓扑图会继续显示上一张
            # 地图的 YOU 坐标，而 F8 已经根据新地图实时换算出正确位置。
            self.current_player_world_pos = None
            self.current_player_raw_world_pos = None
            self.current_player_platform = None
            self.current_player_ladder = None
            self.world_kalman.reset()
            self.horizontal_motion.reset()
            self.horizontal_kalman.reset()
            if getattr(self, "combat_fsm", None) is not None:
                self.combat_fsm.reset_navigation_failures()
            for tracker in (
                getattr(self, "tracker", None),
                getattr(self, "raw_tracker", None),
                getattr(self, "radar_tracker", None),
            ):
                if tracker is not None and hasattr(tracker, "reset_map_calibration"):
                    tracker.reset_map_calibration()
        self._populate_mob_tree(mobs, res.get("disabled_mob_ids", []))
        enabled_mobs = self._get_enabled_mobs()
        self.detector.load_dynamic_monster_templates(enabled_mobs)
        self.log(
            f"[怪物启用状态] enabled={[int(m.get('id')) for m in enabled_mobs]}，"
            f"disabled={[int(x) for x in (res.get('disabled_mob_ids', []) or [])]}"
        )

        # 异步构建当前地图平台与梯绳拓扑图 (本地离线优先，0 延迟秒读)
        def async_build_graph():
            try:
                map_dict = res.get("raw_map_data")
                map_data_source = res.get("map_data_source") or (
                    "wz_img" if isinstance(map_dict, dict) and map_dict.get("_source") == "wz_img" else None
                )
                enabled_tp = bool(self.config.get("enable_teleport", False))
                tp_dist = float(self.config.get("teleport_distance_px", 150.0))
                cache_key = (str(m_id), bool(self.merge_short_platforms), enabled_tp, tp_dist)
                cached_graph = self._platform_graph_cache.get(cache_key)
                cache_source = "内存" if cached_graph is not None else ""

                # 同一进程内返回已经解析过的地图时，直接复用 PlatformGraph，
                # 不再重复读取 JSON 和执行平台/梯绳拓扑解析。
                if cached_graph is not None:
                    g = cached_graph
                    map_dict = self._map_data_cache.get(str(m_id), map_dict)
                    if isinstance(map_dict, dict) and map_dict.get("_source") == "wz_img":
                        map_data_source = "wz_img"
                else:
                    if not map_dict:
                        map_dict = self._map_data_cache.get(str(m_id))
                        if isinstance(map_dict, dict) and map_dict.get("_source") == "wz_img":
                            map_data_source = "wz_img"

                if cached_graph is None and not map_dict and str(m_id).isdigit():
                    mid_int = int(m_id)
                    maps_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "maps")
                    os.makedirs(maps_dir, exist_ok=True)
                    local_json = os.path.join(maps_dir, f"{mid_int}.json")

                    # 1. 优先读取本地 Map.wz IMG。
                    try:
                        map_dict = self.wz_map_reader.load_map(mid_int)
                        map_data_source = "wz_img"
                    except FileNotFoundError:
                        map_dict = None
                    except Exception as img_exc:
                        self.log(f"⚠️ [地图 IMG] MapID={mid_int} 解析失败，改用后备 JSON：{img_exc}")
                        map_dict = None

                    # 2. IMG 不可用时读取旧版本地 JSON。
                    if not map_dict and os.path.exists(local_json):
                        try:
                            with open(local_json, "r", encoding="utf-8") as f:
                                map_dict = json.load(f)
                            map_data_source = "local_json"
                        except Exception:
                            map_dict = None

                    # 3. 两种本地来源都不存在时才访问在线后备接口。
                    if not map_dict:
                        url = f"https://maplestory.io/api/gms/83/map/{mid_int}"
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=8) as r:
                            map_dict = json.loads(r.read().decode('utf-8'))
                        map_data_source = "online_api"
                        with open(local_json, "w", encoding="utf-8") as f:
                            json.dump(map_dict, f, ensure_ascii=False, indent=2)

                cache_signature = self._topology_cache_signature(str(m_id))
                if cached_graph is None:
                    cached_graph = self._load_persisted_topology_cache(
                        str(m_id), bool(self.merge_short_platforms), cache_signature,
                        enable_teleport=enabled_tp, teleport_distance_px=tp_dist,
                    )
                    if cached_graph is not None:
                        cache_source = "文件"
                if map_dict or cached_graph is not None:
                    if map_dict:
                        self._map_data_cache[str(m_id)] = map_dict
                    if cached_graph is None:
                        g = PlatformGraphBuilder.build_from_map_dict(
                            map_dict,
                            merge_short_platforms=self.merge_short_platforms,
                            enable_teleport=enabled_tp,
                            teleport_distance_px=tp_dist,
                        )
                        self._save_persisted_topology_cache(
                            str(m_id), bool(self.merge_short_platforms), cache_signature, g,
                            enable_teleport=enabled_tp, teleport_distance_px=tp_dist,
                        )
                    else:
                        g = cached_graph
                    # OCR 可能在构图期间又识别到另一张地图。丢弃晚到的旧
                    # 构图结果，防止它覆盖当前地图的拓扑与世界坐标参数。
                    if getattr(self, "_graph_target_map_id", None) != str(m_id):
                        return
                    # 内存缓存的拓扑可以复用静态平台/绳梯与按 MapID 保存的
                    # Canvas 标定，但不能复用上次离开该图时的卷轴视口偏移。
                    # 当前小地图框与当前卷轴位置均由新画面重新检测。
                    if hasattr(g, "reset_minimap_runtime_alignment"):
                        g.reset_minimap_runtime_alignment()
                    self.platform_graph = g
                    self._sync_tracker_minimap_canvas_size(g)
                    graph_ready_at = time.perf_counter()
                    with self._minimap_ready_lock:
                        pending = self._minimap_ready_pending
                        if pending is not None and pending.get("map_id") == str(m_id):
                            pending["graph_ready_at"] = graph_ready_at
                            graph_elapsed = graph_ready_at - pending["sync_started_at"]
                        else:
                            graph_elapsed = None
                    if graph_elapsed is not None:
                        self.log(
                            f"⏱️ [小地图准备] MapID={m_id} 拓扑/WZ背景就绪="
                            f"{graph_elapsed:.3f}s"
                        )
                    self._current_graph_map_data = map_dict
                    self._platform_graph_cache[cache_key] = g
                    cache_note = f"（{cache_source}缓存）" if cache_source else ""
                    source_label = {
                        "wz_img": "WZ IMG",
                        "local_json": "本地 JSON 后备",
                        "online_api": "在线地图后备",
                    }.get(map_data_source, "地图数据")
                    def update_g_ui():
                        total_p = len(g.nodes)
                        total_e = sum(len(e) for e in g.edges.values())
                        if hasattr(self, "lbl_topology_summary"):
                            self.lbl_topology_summary.config(
                                text=(f"📊 地图拓扑: 已解析 {total_p} 个平台 | {total_e} 条动作通道 "
                                      f"(✔ {source_label}) {cache_note}"),
                                fg="#69f0ae"
                            )
                        self._preview_patrol_route()
                        if self.topology_dialog and self.topology_dialog.is_open:
                            self.topology_dialog.refresh_view(force_rebuild=True)
                    self.root.after(0, update_g_ui)
                else:
                    def update_fail():
                        if hasattr(self, "lbl_topology_summary"):
                            self.lbl_topology_summary.config(
                                text="📊 地图拓扑: ⚠️ 数据加载异常，请点击【🔄 立即刷新】重试",
                                fg="#ff5252"
                            )
                    self.root.after(0, update_fail)
            except Exception as exc:
                # Python 会在离开 except 块时主动清除异常变量。Tk 的
                # after 回调稍后才执行，不能在闭包中继续直接引用 exc/e，
                # 否则网络超时之后还会追加一次 NameError。
                error_detail = str(exc).strip() or type(exc).__name__
                is_timeout = "timed out" in error_detail.lower()
                if is_timeout:
                    status_text = "📊 地图拓扑: ⚠️ 本地 IMG/JSON 均不可用，在线读取超时"
                    log_text = (
                        f"[拓扑加载超时] MapID={m_id}：本地 IMG/JSON 均不可用，"
                        f"在线接口未响应（{error_detail}）"
                    )
                else:
                    status_text = f"📊 地图拓扑: ⚠️ 加载失败 ({error_detail})"
                    log_text = f"[拓扑构建异常] MapID={m_id}：{error_detail}"
                print(log_text)
                self.log(log_text)

                def update_err(message=status_text):
                    if hasattr(self, "lbl_topology_summary"):
                        self.lbl_topology_summary.config(
                            text=message,
                            fg="#ff9100"
                        )
                try:
                    self.root.after(0, update_err)
                except tk.TclError:
                    pass

        if map_changed or first_map_sync:
            # 怪物模板/UI 数据已同步完成，从这里开始单独计量拓扑、WZ
            # 背景及小地图框选；与紧随其后的 [地图同步] 日志时间基本一致。
            with self._minimap_ready_lock:
                self._minimap_ready_pending = {
                    "map_id": str(m_id),
                    "map_name": str(m_name),
                    "sync_started_at": time.perf_counter(),
                    "graph_ready_at": None,
                    "box_logged": False,
                    "position_logged": False,
                }
        threading.Thread(target=async_build_graph, daemon=True).start()

        if is_custom:
            self.log(f"[本地加载] 【{m_name}】从本地读取 {len(mobs)} 种自定义怪物（已跳过网络）！")
        else:
            self.log(f"[地图同步] 【{m_name}】(ID: {m_id})，载入 {len(mobs)} 种怪物！")

    def _drain_log_ui_updates(self) -> None:
        """仅由 Tk 主线程批量提交日志，后台线程绝不触碰 Tk。"""
        try:
            lines = []
            # 单次批量写入，避免高频按键日志产生大量 Tk insert 调用。
            for _ in range(1000):
                try:
                    timestamp, message = self._log_ui_queue.get_nowait()
                except queue.Empty:
                    break
                lines.append(f"[{timestamp}] {message}\n")
            if (
                lines
                and hasattr(self, "log_text")
                and self.log_text.winfo_exists()
            ):
                self.log_text.insert(tk.END, "".join(lines))
                self.log_text.see(tk.END)
        except tk.TclError:
            return
        except Exception as exc:
            print(f"[日志 UI 更新异常] {exc}")
        finally:
            try:
                if self.root.winfo_exists():
                    self.root.after(40, self._drain_log_ui_updates)
            except tk.TclError:
                pass

    # ── 日志：文件同步写入，UI 仅通过主线程队列刷新 ─────────────────────────
    def log(self, msg: str):
        msg = re.sub(r"maplestory\.io", "在线资源接口", str(msg), flags=re.I)
        msg = re.sub("maplestory", "游戏客户端", str(msg), flags=re.I)
        now = time.time()
        msec = int((now - int(now)) * 1000)
        t = time.strftime("%H:%M:%S", time.localtime(now)) + f".{msec:03d}"
        runtime_fp = getattr(self, "_runtime_log_fp", None)
        if runtime_fp is not None:
            try:
                with self._runtime_log_lock:
                    runtime_fp.write(f"[{t}] {msg}\n")
            except Exception:
                pass
        try:
            self._log_ui_queue.put_nowait((t, str(msg)))
        except Exception:
            print(f"[{t}] {msg}")

    # ── 全局热键 (带 0.3s 防抖与即时急停) ──────────────────────────────────────
    def _start_global_hotkey_listener(self):
        def worker():
            states = {}
            last_press_time = {
                VK_F6: 0.0,
                VK_F7: 0.0,
                VK_F9: 0.0,
                VK_F10: 0.0,
                VK_F11: 0.0,
            }
            # 方向键状态同时覆盖：
            # 1) 本程序通过 SendInput 发出的按键；
            # 2) 用户在游戏中实际按下的按键。
            # 后者必须在与游戏同等权限（通常为管理员）下轮询，
            # 否则 GetAsyncKeyState 会一直读到 0。
            last_motion_direction = None

            while not self.stop_event.is_set():
                now = time.time()

                # 将真实方向键按下/松开事件送入连续水平运动模型。
                # 使用 active_keys 合并自动输入，避免自动控制时物理轮询
                # 短暂读不到 SendInput 状态而把速度错误清零。
                try:
                    # 这里只监听游戏实际使用的左右方向键；A/D 不参与
                    # 世界坐标预测，避免其它窗口或快捷键误触发移动模型。
                    physical_left = (
                        (user32.GetAsyncKeyState(0x25) & 0x8000) != 0
                    )
                    physical_right = (
                        (user32.GetAsyncKeyState(0x27) & 0x8000) != 0
                    )
                    active = getattr(self.input_driver, "active_keys", set())
                    left_down = physical_left or ("left" in active)
                    right_down = physical_right or ("right" in active)
                    if left_down and not right_down:
                        motion_direction = -1
                    elif right_down and not left_down:
                        motion_direction = 1
                    else:
                        motion_direction = 0
                    if motion_direction != last_motion_direction:
                        motion_timestamp = time.perf_counter()
                        self.horizontal_motion.set_direction(
                            motion_direction, motion_timestamp
                        )
                        self.horizontal_kalman.set_direction(
                            motion_direction, motion_timestamp
                        )
                        self._append_model_debug(
                            "", "", self.horizontal_motion.predict(),
                            self.current_player_world_pos,
                            source="physical_key_poll"
                        )
                        last_motion_direction = motion_direction

                    # 玩家手动 UP 不经过 InputDriver。SendInput 也可能被
                    # GetAsyncKeyState 看见，须排除程序正在保持的 UP。
                    physical_up = (
                        (user32.GetAsyncKeyState(VK_UP) & 0x8000) != 0
                    )
                    if (
                        physical_up
                        and not states.get(VK_UP, False)
                        and "up" not in getattr(self.input_driver, "active_keys", set())
                    ):
                        self._request_portal_ocr_acceleration("玩家手动")
                    states[VK_UP] = physical_up
                except Exception:
                    pass

                # F6: 启停自动挂机 (即时执行，不被 UI 阻塞延迟)
                down_f6 = (user32.GetAsyncKeyState(VK_F6) & 0x8000) != 0
                if down_f6:
                    if not states.get(VK_F6) and (now - last_press_time[VK_F6]) > 0.30:
                        last_press_time[VK_F6] = now
                        if self._minigame_owns_input:
                            if self.combat_fsm.is_running:
                                self.world_patrol_controller.stop()
                                self.combat_fsm.stop()
                                self._minigame_resume_f6 = False
                                self._stop_minigame_video_test_session("用户停止F6")
                                self.log("🛑 [测谎小游戏] F6 已手动停止，不会在小游戏结束后自动恢复。")
                            elif self._minigame_resume_f6:
                                self._minigame_resume_f6 = False
                                self._stop_minigame_video_test_session("用户取消F6恢复")
                                self.log("ℹ️ [测谎小游戏] 已取消小游戏结束后的 F6 自动恢复。")
                            else:
                                self.log("ℹ️ [测谎小游戏] 正在控制鼠标，F6 暂不可启动。")
                        elif self.combat_fsm.is_running:
                            was_attack_only = getattr(self.combat_fsm, "attack_only_mode", False)
                            self.world_patrol_controller.stop()
                            self.combat_fsm.stop()
                            self._stop_minigame_video_test_session("用户停止F6")
                            if was_attack_only:
                                self.log("🛑 [F6 热键] 停止仅攻击键介入。")
                            else:
                                self.log("🛑 [F6 热键] 停止自动挂机。")
                        else:
                            # 启动必须与界面按钮共用同一入口；这里会完成跨图
                            # 路线校验、WorldPatrolController.start()，并允许
                            # 人物处于绳梯上时先由 CombatFSM 完成登顶定位。
                            self.root.after(0, self.toggle_autobot)
                        self.root.after(0, self._sync_bot_ui_state)
                states[VK_F6] = down_f6

                # F7: 启停路线录制
                down_f7 = (user32.GetAsyncKeyState(VK_F7) & 0x8000) != 0
                if down_f7:
                    if not states.get(VK_F7) and (now - last_press_time[VK_F7]) > 0.30:
                        last_press_time[VK_F7] = now
                        self.root.after(0, self.toggle_waypoint_recording)
                states[VK_F7] = down_f7

                # F8: 弹出小地图雷达与黄点定位调试器
                down_f8 = (user32.GetAsyncKeyState(VK_F8) & 0x8000) != 0
                if down_f8:
                    if not states.get(VK_F8) and (now - last_press_time.get(VK_F8, 0)) > 0.30:
                        last_press_time[VK_F8] = now
                        self.root.after(0, self.open_minimap_radar_dialog)
                states[VK_F8] = down_f8

                # 急停键可在 UI 中选择；默认 F12，避开 F6-F11 程序热键。
                release_vk = self._mouse_release_hotkey_vk()
                release_down = (user32.GetAsyncKeyState(release_vk) & 0x8000) != 0
                release_enabled = bool(
                    self.config.get("lie_detector_auto_solve_enabled", False)
                    and self.config.get("lie_detector_mouse_control_enabled", False)
                )
                if release_enabled and release_down and not states.get(release_vk, False):
                    self.minigame_bridge.set_mouse_control_enabled(False)
                    self.root.after(0, self._handle_mouse_release_hotkey)
                states[release_vk] = release_down

                # F9: 立即 OCR 感知
                down_f9 = (user32.GetAsyncKeyState(VK_F9) & 0x8000) != 0
                if down_f9:
                    if not states.get(VK_F9) and (now - last_press_time[VK_F9]) > 0.30:
                        last_press_time[VK_F9] = now
                        self.root.after(0, self.on_force_ocr)
                states[VK_F9] = down_f9

                # F10：稳定性测试期间注入一次随机黄点坐标扰动。
                down_f10 = (user32.GetAsyncKeyState(VK_F10) & 0x8000) != 0
                if down_f10:
                    if not states.get(VK_F10) and (now - last_press_time[VK_F10]) > 0.30:
                        last_press_time[VK_F10] = now
                        if self.ladder_grab_test_runner.active:
                            self.root.after(
                                0, self.ladder_grab_test_runner.inject_position_disturbance
                            )
                states[VK_F10] = down_f10

                # F11：启停随机全图行走验收。与界面按钮共用入口，所有
                # F6/绳梯测试互斥、管理员输入与定位检查保持一致。
                down_f11 = (user32.GetAsyncKeyState(VK_F11) & 0x8000) != 0
                if down_f11:
                    if (
                        not states.get(VK_F11)
                        and (now - last_press_time[VK_F11]) > 0.30
                    ):
                        last_press_time[VK_F11] = now
                        self.root.after(0, self._toggle_random_path_test)
                states[VK_F11] = down_f11

                time.sleep(0.03)
        threading.Thread(target=worker, daemon=True).start()

    # ── 后台工作线程 ─────────────────────────────────────────────────────────
    def _monster_fps_text(self) -> str:
        """返回60Hz轻量追踪与独立完整模板扫描的实际吞吐。"""
        with self._monster_fps_lock:
            tracking_fps = self._monster_detection_fps
            scan_fps = self._monster_full_scan_fps
            gap_ms = self._monster_full_scan_gap_ms
            cost_ms = self._monster_full_scan_cost_ms
        if tracking_fps is None:
            return "NULL"
        if bool(self.config.get("monster_full_scan_every_frame", False)):
            return f"全速实扫 {tracking_fps:.1f} FPS"
        scan_text = "NULL" if scan_fps is None else f"{scan_fps:.1f} FPS"
        return f"跟踪 {tracking_fps:.1f} FPS | 实扫 {scan_text}"

    def _start_background_workers(self):
        # 独立视口渲染缓冲：后台完成 HUD/缩放/RGB 转换，Tk 线程只提交图片。
        self._viewport_lock = threading.Lock()
        self._viewport_rgb = None
        self._viewport_seq = 0
        # 初始尺寸取当前容器；后续由 <Configure> 实时更新。后台只读取这
        # 个普通元组，不访问 Tk，因此不会引入跨线程 Tk 调用。
        container_w = self.video_container.winfo_width() if hasattr(self, "video_container") else 0
        container_h = self.video_container.winfo_height() if hasattr(self, "video_container") else 0
        self._viewport_target_size = (
            max(100, int(container_w) - 4) if container_w > 10 else 800,
            max(100, int(container_h) - 4) if container_h > 10 else 450,
        )

        def minigame_snapshot_writer():
            while not self.stop_event.is_set() or not self._minigame_snapshot_queue.empty():
                try:
                    frame, state = self._minigame_snapshot_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._save_minigame_state_snapshot(frame, state)
                except Exception as exc:
                    self.log(f"⚠️ [测谎小游戏] 保存状态截图失败：{exc}")
                finally:
                    self._minigame_snapshot_queue.task_done()

        threading.Thread(
            target=minigame_snapshot_writer,
            daemon=True,
            name="MiniGameSnapshotWriter",
        ).start()

        def minigame_worker():
            last_frame_seq = -1
            last_state = None
            was_enabled = False
            last_error_log_at = 0.0
            last_archive_error_at = 0.0

            def archive_session():
                nonlocal last_archive_error_at
                try:
                    archived = self.minigame_session_recorder.finish()
                    if archived:
                        self.log(f"💾 [测谎小游戏] 识别轨迹已归档：{archived[0]}、{archived[1]}")
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_archive_error_at >= 2.0:
                        self.log(f"⚠️ [测谎小游戏] 轨迹归档失败，将重试：{exc}")
                        last_archive_error_at = now

            while not self.stop_event.is_set():
                enabled = bool(
                    self.config.get("lie_detector_auto_solve_enabled", False)
                    or self.config.get("lie_detector_test_enabled", False)
                )
                if not enabled:
                    if self.minigame_session_recorder.current_session is not None:
                        archive_session()
                    if was_enabled:
                        self.minigame_bridge.reset()
                        self.minigame_result_confirmer.reset()
                    self._publish_minigame_overlay(None, 0)
                    was_enabled = False
                    last_frame_seq = -1
                    last_state = None
                    time.sleep(0.1)
                    continue
                was_enabled = True
                frame = self.capture.capture_frame(copy=True)
                if frame is None:
                    time.sleep(0.01)
                    continue
                frame_seq = int(getattr(self.capture, "frame_seq", 0))
                if frame_seq == last_frame_seq:
                    time.sleep(0.002)
                    continue
                last_frame_seq = frame_seq
                try:
                    auto_solve = bool(self.config.get("lie_detector_auto_solve_enabled", False))
                    result = self.minigame_bridge.process(
                        frame, int(self.hwnd or 0), hold_result_confirmation=auto_solve
                    )
                    if result is not None:
                        sample_time = time.perf_counter()
                        if result.just_entered_active:
                            self.minigame_session_recorder.start(
                                re.sub(
                                    "maplestory", "游戏客户端",
                                    str(self.config.get("window_title", "")),
                                    flags=re.I,
                                ),
                                getattr(result.track_result, "dialog_roi", None),
                                result.target_shape,
                                now=sample_time,
                            )
                        state_value = getattr(getattr(result, "state", None), "value", "")
                        if (
                            self.minigame_session_recorder.current_session is not None
                            and (state_value == "ACTIVE" or result.just_exited_active)
                        ):
                            self.minigame_session_recorder.record(
                                result.track_result, now=sample_time
                            )
                    if result is not None and result.just_exited_active:
                        archive_session()
                        self._minigame_replay_record_queue.put(True)
                        if auto_solve:
                            self.minigame_result_confirmer.begin()
                    elif (
                        result is not None
                        and self.minigame_session_recorder.current_session is not None
                        and state_value in ("COOLDOWN", "IDLE")
                    ):
                        # A short aborted round or a transient disk error must not
                        # keep accumulating standby frames indefinitely.
                        archive_session()
                    if self.minigame_bridge.awaiting_result_confirmation:
                        try:
                            dialog_vk = int(self.config.get("dialog_vk", 0x59))
                        except (TypeError, ValueError):
                            dialog_vk = None
                        # 录像测试可把视频叠加到识别帧；确认弹框只能基于
                        # 真正游戏客户区截图，绝不能对回放画面发送点击。
                        confirm_frame = self.capture.capture_frame(
                            copy=True, include_overlay=False
                        )
                        if confirm_frame is not None and self.minigame_result_confirmer.update(
                            confirm_frame,
                            dialog_key=str(self.config.get("dialog_key", "y")),
                            dialog_vk=dialog_vk,
                        ):
                            self.minigame_bridge.finish_result_confirmation()
                    test_visible = self.minigame_video_test.video_visible
                    state = getattr(getattr(result, "state", None), "value", "")
                    if state and state != last_state:
                        last_state = state
                        if (
                            state in ("COUNTDOWN", "ACTIVE")
                            and bool(self.config.get("lie_detector_save_transition_snapshots", False))
                        ):
                            try:
                                self._minigame_snapshot_queue.put_nowait((frame.copy(), state))
                            except queue.Full:
                                self.log("⚠️ [测谎小游戏] 状态截图队列已满，跳过本次截图。")
                    if test_visible or state in ("COUNTDOWN", "ACTIVE"):
                        track = getattr(result, "track_result", None)
                        point = None
                        if track is not None and getattr(track, "initialized", False):
                            point = (float(track.x), float(track.y))
                        self._publish_minigame_overlay(frame, int(self.hwnd or 0), point)
                    else:
                        self._publish_minigame_overlay(None, 0)
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_error_log_at >= 2.0:
                        self.log(f"⚠️ [测谎小游戏] 帧处理失败：{exc}")
                        last_error_log_at = now
                    time.sleep(0.05)

        threading.Thread(target=minigame_worker, daemon=True, name="MiniGameWorker").start()

        # 1. 独立 raw 世界坐标线程：只做小地图黄点检测与世界坐标换算。
        # ScreenCapture 本身已有独立抓帧线程；此线程按新帧消费，最高 60Hz，
        # 不依赖 F8/拓扑图/UI 的显示刷新周期。
        def raw_tracker_worker():
            last_frame_seq = -1
            while not self.stop_event.is_set():
                if self._recognition_pause_event.is_set():
                    time.sleep(0.03)
                    continue
                graph = self.platform_graph
                frame = self.capture.capture_frame(copy=False)
                if graph is None or frame is None:
                    time.sleep(0.01)
                    continue
                # WGC/GDI 都可能复用同一 ndarray 缓冲区，因此不能以 id(frame)
                # 判断新帧；必须使用 capture 的递增 frame_seq。
                frame_seq = getattr(self.capture, "frame_seq", 0)
                if frame_seq == last_frame_seq:
                    time.sleep(0.002)
                    continue
                last_frame_seq = frame_seq
                try:
                    track_res = self.raw_tracker.detect(frame)
                    readiness_logs = []
                    now_ready = time.perf_counter()
                    with self._minimap_ready_lock:
                        pending = self._minimap_ready_pending
                        graph_map_id = str(getattr(graph, "map_id", ""))
                        if pending is not None and pending.get("map_id") == graph_map_id:
                            sync_at = float(pending["sync_started_at"])
                            graph_at = pending.get("graph_ready_at")
                            if track_res.inner_box and not pending["box_logged"]:
                                pending["box_logged"] = True
                                bx, by, bw, bh = track_res.inner_box
                                graph_part = (
                                    f"，WZ背景后={now_ready - float(graph_at):.3f}s"
                                    if graph_at is not None else ""
                                )
                                readiness_logs.append(
                                    f"⏱️ [小地图框选就绪] MapID={graph_map_id} "
                                    f"范围=({bx},{by},{bw},{bh})，地图同步后="
                                    f"{now_ready - sync_at:.3f}s{graph_part}"
                                )
                            if (
                                track_res.is_detected
                                and track_res.norm_pos
                                and not pending["position_logged"]
                            ):
                                pending["position_logged"] = True
                                readiness_logs.append(
                                    f"⏱️ [黄点定位就绪] MapID={graph_map_id}，地图同步后="
                                    f"{now_ready - sync_at:.3f}s"
                                )
                            if pending["box_logged"] and pending["position_logged"]:
                                self._minimap_ready_pending = None
                    for readiness_log in readiness_logs:
                        self.log(readiness_log)
                    # F8/HUD 读取这一份后台原始测量，而不是各自重新检测。
                    # TrackerResult 在 detect 返回后不再原地修改，可安全共享。
                    with self._raw_tracker_result_lock:
                        self._latest_raw_tracker_result = track_res
                    # 黄点身份实机诊断必须复用本线程已经取得的唯一捕获帧。
                    # 禁止另起进程高频 PrintWindow；只按低频率保存小地图副本。
                    debug_now = time.perf_counter()
                    if (
                        self._yellow_identity_debug
                        and track_res.inner_box
                        and debug_now - self._yellow_identity_debug_last_save >= 0.18
                    ):
                        bx, by, bw, bh = track_res.inner_box
                        debug_mini = frame[by:by + bh, bx:bx + bw].copy()
                        if debug_mini.size > 0:
                            if track_res.subpixel_pos:
                                dx = int(round(track_res.subpixel_pos[0]))
                                dy = int(round(track_res.subpixel_pos[1]))
                                cv2.circle(debug_mini, (dx, dy), 7, (255, 0, 255), 1)
                                cv2.drawMarker(
                                    debug_mini, (dx, dy), (255, 255, 255),
                                    cv2.MARKER_CROSS, 7, 1,
                                )
                            else:
                                cv2.putText(
                                    debug_mini, "MISS", (3, 13),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (0, 0, 255), 1, cv2.LINE_AA,
                                )
                            debug_path = os.path.join(
                                self._yellow_identity_debug_dir,
                                f"frame_{self._yellow_identity_debug_seq:04d}.png",
                            )
                            cv2.imwrite(debug_path, debug_mini)
                            self._yellow_identity_debug_seq += 1
                            self._yellow_identity_debug_last_save = debug_now
                    if track_res.is_detected and track_res.norm_pos:
                        nx, ny = track_res.norm_pos
                        crop_gray = None
                        if track_res.inner_box:
                            bx, by, bw, bh = track_res.inner_box
                            sub = frame[by:by+bh, bx:bx+bw]
                            if sub.size > 0:
                                crop_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if len(sub.shape) == 3 else sub
                        conversion_debug = {}
                        raw_pos = graph.minimap_norm_to_world(
                            nx, ny, crop_gray_frame=crop_gray, debug_out=conversion_debug
                        )
                        self._update_motion_measurement_step(graph, track_res.inner_box)
                        observation_t = time.perf_counter()
                        # 两套模型始终吸收同一时刻、同一份 raw 黄点世界 X。
                        # 配置开关仅决定现阶段导航选用旧模型还是原始 X；
                        # Kalman 保持纯影子输出，尚不影响人物控制。
                        motion_predicted_x = self.horizontal_motion.correct_measurement(
                            raw_pos[0], timestamp=observation_t
                        )
                        kalman_predicted_x = self.horizontal_kalman.correct_measurement(
                            raw_pos[0], timestamp=observation_t
                        )
                        # 卡尔曼区间模型是当前唯一导航/拓扑 X 源。传统模型
                        # 仍并行运行并写入调试日志，只用于对照和快速回退。
                        predicted_x = kalman_predicted_x
                        preferred_platform_id = getattr(
                            self.current_player_platform, "id", None
                        )
                        snapped_pos = graph.get_snapped_player_world_pos(
                            int(round(predicted_x)),
                            raw_pos[1],
                            preferred_platform_id=preferred_platform_id,
                        )
                        # 地图切换期间，丢弃旧 graph 完成的晚到结果。
                        if self.platform_graph is graph:
                            with self._raw_world_lock:
                                self.current_player_raw_world_pos = raw_pos
                            patrol = getattr(getattr(self, "combat_fsm", None), "platform_patrol", None)
                            if patrol is not None:
                                patrol.observe_raw_motion(raw_pos, timestamp=observation_t)
                            self.current_player_world_pos = snapped_pos
                            p_curr, lr_curr, is_climbing = graph.find_player_location(
                                *snapped_pos,
                                preferred_platform_id=preferred_platform_id,
                            )
                            self.current_player_platform = p_curr
                            self.current_player_is_climbing = bool(is_climbing)
                            self.current_player_ladder = lr_curr
                            self._append_model_debug(
                                raw_pos[0], raw_pos[1], predicted_x,
                                snapped_pos, source="raw_tracker_60hz"
                            )
                            self._append_coordinate_trace(track_res, raw_pos, snapped_pos, conversion_debug)
                except Exception:
                    pass
                # 捕获帧率高于 60 时，也只以 60Hz 更新控制坐标。
                time.sleep(1.0 / 60.0)

        # 2a. 60Hz轻量目标追踪线程：只做人名定位、朝向、光流与轨迹合并。
        # 完整怪物模板匹配由下方独立线程执行，不能再阻塞本循环。
        def detector_worker():
            # 角色名字牌定位升级为 30Hz (每 1/30 秒测量)，配合短路早退 (0.3ms)，
            # 在极低 CPU 负载下实现极致平滑的角色跟踪与准星贴合。
            player_measure_interval = 1.0 / 30.0
            last_player_measure_at = 0.0
            last_full_speed_frame_seq = -1
            last_full_speed_started_at: Optional[float] = None
            last_full_speed_stats_log_at = 0.0
            while not self.stop_event.is_set():
                if self._recognition_pause_event.is_set() or self.reconnect_controller.disconnected:
                    time.sleep(0.03)
                    continue
                cycle_started_at = time.perf_counter()
                # 预览与F6运行期间均按60Hz消费捕获帧。
                target_monster_hz = 60.0
                full_speed_scan = bool(
                    self.config.get("monster_full_scan_every_frame", False)
                )
                # 检查怪物列表：若当前未启用任何怪物/血条检测，彻底停止怪物检测
                monster_enabled = bool(self.config.get("enable_monster_detection", True))
                hp_bar_enabled = bool(self.config.get("enable_monster_hp_bar_detection", True))
                has_map_monsters, has_mobs = self._monster_pipeline_state()
                if not monster_enabled and not hp_bar_enabled:
                    with self._monster_fps_lock:
                        self._monster_detection_fps = None
                        self._monster_full_scan_fps = None
                        self._monster_full_scan_gap_ms = None
                        self._monster_full_scan_cost_ms = None
                        self._monster_fps_count = 0
                        self._monster_fps_timer = time.perf_counter()
                
                frame = self.capture.capture_frame(copy=False)
                if frame is None:
                    time.sleep(0.02)
                    continue

                frame_seq = int(getattr(self.capture, "frame_seq", 0))
                if full_speed_scan and frame_seq == last_full_speed_frame_seq:
                    # 全速模式严格按“每张新捕获帧一次完整扫描”，不对同一
                    # 张截图重复烧算力；捕获器本身最高60FPS。
                    time.sleep(0.001)
                    continue
                if full_speed_scan:
                    last_full_speed_frame_seq = frame_seq
                else:
                    last_full_speed_frame_seq = -1
                    last_full_speed_started_at = None

                if not has_mobs:
                    with self._monster_fps_lock:
                        if not has_map_monsters:
                            # 本图怪物列表为空：无论模板/血条开关如何，整条
                            # 怪物流水线都视为未启用，而不是“开启但 0 FPS”。
                            self._monster_detection_fps = None
                            self._monster_full_scan_fps = None
                            self._monster_full_scan_gap_ms = None
                            self._monster_full_scan_cost_ms = None
                            self._monster_fps_count = 0
                            self._monster_fps_timer = time.perf_counter()
                        elif monster_enabled:
                            # 列表非空、开关打开，但尚无可匹配模板。
                            self._monster_detection_fps = 0.0
                    if self.detector is not None:
                        self.detector.clear_monster_tracks()
                    with self._monster_batch_lock:
                        self._monster_latest_batch = None
                    # 仅执行微秒级角色自身定位与朝向识别，完全不消耗怪物匹配算力
                    p_found, p_pos, p_box = self.detector.detect_player(frame)
                    facing_dir = "right"
                    facing_conf = 1.0
                    facing_locked = False
                    if p_found and p_pos is not None:
                        gray_sub = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                        facing_dir, facing_conf, facing_locked = self.detector._detect_facing_direction(gray_sub, p_pos)
                    res = MainViewResult(
                        player_found=p_found,
                        player_pos=p_pos,
                        player_bbox=p_box,
                        feature_bbox=self.detector.last_feature_bbox if p_found else None,
                        facing_direction=facing_dir,
                        facing_confidence=facing_conf,
                        is_facing_locked=facing_locked,
                        monsters=[],
                        locked_target=None
                    )
                    if self.reconnect_controller.disconnected:
                        continue
                    with self.lock:
                        self.latest_result = res
                    time.sleep(1.0 / 30.0)
                    continue

                # 列表中有怪物时才执行轻量追踪；角色定位按30Hz重测，
                # 怪物框在其余帧由光流保持60Hz位置更新。
                now_before_process = time.perf_counter()
                reuse_player = (
                    self.detector.last_player_pos is not None
                    and (now_before_process - last_player_measure_at) < player_measure_interval
                )
                if full_speed_scan:
                    self.detector.monster_redetect_interval = 1
                    res = self.detector.process(
                        frame,
                        reuse_player=reuse_player,
                        async_monster_matching=False,
                        full_scan_only=True,
                    )
                else:
                    self.detector.monster_redetect_interval = max(
                        1, int(self.config.get("monster_redetect_interval", 4))
                    )
                    with self._monster_batch_lock:
                        latest_batch = self._monster_latest_batch
                    res = self.detector.process(
                        frame,
                        reuse_player=reuse_player,
                        monster_detection_batch=latest_batch,
                        async_monster_matching=True,
                    )
                if self.reconnect_controller.disconnected:
                    continue
                if not reuse_player:
                    last_player_measure_at = now_before_process
                now_detect = time.perf_counter()
                if full_speed_scan and self._performance_timing is not None:
                    self._performance_timing.record_scan(
                        "full_speed_process", 0.0, 0.0, 0.0,
                        (now_detect - cycle_started_at) * 1000.0,
                        (now_detect - cycle_started_at) * 1000.0,
                        None if last_full_speed_started_at is None else
                        (cycle_started_at - last_full_speed_started_at) * 1000.0,
                    )
                with self._monster_fps_lock:
                    self._monster_fps_count += 1
                    elapsed = now_detect - self._monster_fps_timer
                    if elapsed >= 0.5:
                        self._monster_detection_fps = self._monster_fps_count / elapsed
                        if full_speed_scan:
                            self._monster_full_scan_fps = self._monster_detection_fps
                        self._monster_fps_count = 0
                        self._monster_fps_timer = now_detect
                    if full_speed_scan:
                        if last_full_speed_started_at is not None:
                            self._monster_full_scan_gap_ms = max(
                                0.0,
                                (cycle_started_at - last_full_speed_started_at) * 1000.0,
                            )
                        self._monster_full_scan_cost_ms = max(
                            0.0, (now_detect - cycle_started_at) * 1000.0
                        )
                        last_full_speed_started_at = cycle_started_at
                with self.lock:
                    self.latest_result = res
                if full_speed_scan and now_detect - last_full_speed_stats_log_at >= 5.0:
                    last_full_speed_stats_log_at = now_detect
                    with self._monster_fps_lock:
                        full_speed_fps = self._monster_detection_fps or 0.0
                        full_speed_gap_ms = self._monster_full_scan_gap_ms or 0.0
                        full_speed_cost_ms = self._monster_full_scan_cost_ms or 0.0
                    locked_track = getattr(res.locked_target, "track_id", None)
                    self.log(
                        "📈 [怪物全速完整扫描] "
                        f"实扫={full_speed_fps:.1f}FPS，"
                        f"帧间隔={full_speed_gap_ms:.1f}ms，"
                        f"耗时={full_speed_cost_ms:.1f}ms，"
                        f"血条={getattr(self.detector, 'last_hp_bar_backend', 'cpu')} "
                        f"{getattr(self.detector, 'last_hp_bar_cost_ms', 0.0):.1f}ms，"
                        f"原始候选={getattr(self.detector, 'last_full_scan_candidate_count', 0)}，"
                        f"有效轨迹={len(res.monsters)}，锁定={locked_track}"
                    )
                # 60FPS旧版节奏：计算未超预算时补足16.7ms；超预算时
                # 立即处理下一张捕获帧，不再额外等待3ms。
                remaining = (1.0 / target_monster_hz) - (time.perf_counter() - cycle_started_at)
                if remaining > 0.0:
                    time.sleep(remaining)

        # 2b. 独立完整模板扫描线程：按单调时钟每55ms取一次“当时最新帧”。
        # 没有任务队列，因此扫描慢时不会累积旧帧；一次完成后直接取最新帧。
        def monster_full_scan_worker():
            try:
                configured_scan_ms = float(
                    self.config.get("monster_full_scan_interval_ms", 55.0)
                )
            except (TypeError, ValueError):
                configured_scan_ms = 55.0
            # 35ms以下会徒增CPU占用；上限锁在65ms，为调度抖动预留
            # 至少5ms，正常机器上保持启动间隔不超过约70ms。
            target_period_sec = max(0.035, min(0.065, configured_scan_ms / 1000.0))
            normal_gap_limit_sec = 0.070
            pipeline_started_at = time.perf_counter()
            next_scan_at = pipeline_started_at
            last_scan_started_at: Optional[float] = None
            last_frame_seq = -1
            last_stats_log_at = pipeline_started_at
            last_overrun_log_at = 0.0
            last_error_log_at = 0.0
            schedule_ready_at: Optional[float] = None
            schedule_lag_sec = 0.0

            while not self.stop_event.is_set():
                if self._recognition_pause_event.is_set() or self.reconnect_controller.disconnected:
                    schedule_ready_at = None
                    time.sleep(0.03)
                    continue
                monster_enabled = bool(self.config.get("enable_monster_detection", True))
                hp_bar_enabled = bool(self.config.get("enable_monster_hp_bar_detection", True))
                full_speed_scan = bool(
                    self.config.get("monster_full_scan_every_frame", False)
                )
                if full_speed_scan:
                    # 全速模式由60Hz检测线程逐张新捕获帧同步完整匹配；这里
                    # 必须暂停，避免两套完整扫描同时争用CPU/GPU与模板锁。
                    with self._monster_batch_lock:
                        self._monster_latest_batch = None
                    next_scan_at = time.perf_counter()
                    last_scan_started_at = None
                    last_frame_seq = -1
                    schedule_ready_at = None
                    time.sleep(0.02)
                    continue
                has_map_monsters, has_mobs = self._monster_pipeline_state()
                if not has_mobs:
                    with self._monster_fps_lock:
                        self._monster_full_scan_fps = (
                            None
                            if (not has_map_monsters or (not monster_enabled and not hp_bar_enabled))
                            else 0.0
                        )
                        self._monster_full_scan_gap_ms = None
                        self._monster_full_scan_cost_ms = None
                        self._monster_full_scan_count = 0
                        self._monster_full_scan_timer = time.perf_counter()
                    with self._monster_batch_lock:
                        self._monster_latest_batch = None
                    next_scan_at = time.perf_counter()
                    last_scan_started_at = None
                    last_frame_seq = -1
                    schedule_ready_at = None
                    time.sleep(0.05)
                    continue

                now = time.perf_counter()
                if now < next_scan_at:
                    time.sleep(min(0.005, next_scan_at - now))
                    continue
                if schedule_ready_at is None:
                    schedule_ready_at = now
                    schedule_lag_sec = max(0.0, now - next_scan_at)

                # 捕获器发布后只会替换 _latest_frame，不会原地覆盖旧 ndarray；
                # 持有引用即可固定本轮输入，无需每 55ms 再复制整张画面。
                frame = self.capture.capture_frame(copy=False)
                frame_seq = int(getattr(self.capture, "frame_seq", 0))
                if frame is None or frame_seq == last_frame_seq:
                    time.sleep(0.002)
                    continue
                last_frame_seq = frame_seq

                scan_started_at = time.perf_counter()
                frame_wait_sec = max(0.0, scan_started_at - schedule_ready_at)
                schedule_ready_at = None
                gap_sec = (
                    0.0 if last_scan_started_at is None
                    else max(0.0, scan_started_at - last_scan_started_at)
                )
                last_scan_started_at = scan_started_at
                try:
                    batch = self.detector.run_full_monster_detection(
                        frame,
                        frame_seq=frame_seq,
                        captured_at=scan_started_at,
                    )
                except Exception as exc:
                    failed_at = time.perf_counter()
                    if self._performance_timing is not None:
                        self._performance_timing.record_scan(
                            "error", schedule_lag_sec * 1000.0,
                            frame_wait_sec * 1000.0, 0.0,
                            (failed_at - scan_started_at) * 1000.0,
                            (failed_at - scan_started_at) * 1000.0,
                            None if gap_sec <= 0.0 else gap_sec * 1000.0,
                        )
                    if failed_at - last_error_log_at >= 2.0:
                        last_error_log_at = failed_at
                        self.log(f"⚠️ [怪物完整重检异常] {type(exc).__name__}: {exc}")
                    next_scan_at = max(
                        scan_started_at + target_period_sec,
                        failed_at,
                    )
                    continue
                if self._performance_timing is not None:
                    lock_wait_ms = batch.backend_lock_wait_sec * 1000.0
                    total_ms = batch.duration_sec * 1000.0
                    self._performance_timing.record_scan(
                        "light_mode", schedule_lag_sec * 1000.0,
                        frame_wait_sec * 1000.0, lock_wait_ms,
                        max(0.0, total_ms - lock_wait_ms), total_ms,
                        None if gap_sec <= 0.0 else gap_sec * 1000.0,
                        template_ms=batch.template_match_sec * 1000.0,
                        hp_bar_ms=batch.hp_bar_sec * 1000.0,
                        candidate_filter_ms=batch.candidate_filter_sec * 1000.0,
                    )
                # 扫描过程中可能刚好切换到无怪地图；此时晚到结果、FPS
                # 统计和流水线日志都必须丢弃，不能让用户看到一次幽灵扫描。
                if self.reconnect_controller.disconnected or not self._has_current_map_monsters():
                    with self._monster_batch_lock:
                        self._monster_latest_batch = None
                    with self._monster_fps_lock:
                        self._monster_full_scan_fps = None
                        self._monster_full_scan_gap_ms = None
                        self._monster_full_scan_cost_ms = None
                        self._monster_full_scan_count = 0
                        self._monster_full_scan_timer = time.perf_counter()
                    next_scan_at = time.perf_counter()
                    last_scan_started_at = None
                    last_frame_seq = -1
                    continue
                if bool(self.config.get("monster_full_scan_every_frame", False)):
                    # 模式可能在本轮重匹配过程中切换；晚到批次不能重新
                    # 填回latest槽，否则切回轻量模式时会消费旧模式结果。
                    next_scan_at = time.perf_counter()
                    continue
                with self._monster_batch_lock:
                    # 单槽latest-wins：轻量线程只会看到最近完成的一批，旧批次
                    # 不排队，也不会在CPU追不上时逐帧积累延迟。
                    self._monster_latest_batch = batch

                completed_at = time.perf_counter()
                with self._monster_fps_lock:
                    self._monster_full_scan_count += 1
                    scan_elapsed = completed_at - self._monster_full_scan_timer
                    if scan_elapsed >= 0.5:
                        self._monster_full_scan_fps = (
                            self._monster_full_scan_count / scan_elapsed
                        )
                        self._monster_full_scan_count = 0
                        self._monster_full_scan_timer = completed_at
                    if gap_sec > 0.0:
                        self._monster_full_scan_gap_ms = gap_sec * 1000.0
                    self._monster_full_scan_cost_ms = batch.duration_sec * 1000.0

                if (
                    gap_sec > normal_gap_limit_sec
                    and completed_at - last_overrun_log_at >= 2.0
                ):
                    last_overrun_log_at = completed_at
                    self.log(
                        "⚠️ [怪物完整重检超时] "
                        f"启动间隔={gap_sec * 1000.0:.1f}ms > 70ms，"
                        f"本次耗时={batch.duration_sec * 1000.0:.1f}ms；"
                        "轻量追踪仍保持独立运行"
                    )
                if completed_at - last_stats_log_at >= 5.0:
                    last_stats_log_at = completed_at
                    with self._monster_fps_lock:
                        tracking_fps = self._monster_detection_fps
                        full_fps = self._monster_full_scan_fps
                    self.log(
                        "📈 [怪物识别流水线] "
                        f"轻量追踪={0.0 if tracking_fps is None else tracking_fps:.1f}FPS，"
                        f"完整重检={0.0 if full_fps is None else full_fps:.1f}FPS，"
                        f"启动间隔={gap_sec * 1000.0:.1f}ms，"
                        f"耗时={batch.duration_sec * 1000.0:.1f}ms，"
                        f"血条={getattr(self.detector, 'last_hp_bar_backend', 'cpu')} "
                        f"{getattr(self.detector, 'last_hp_bar_cost_ms', 0.0):.1f}ms，"
                        f"候选={len(batch.detections)}"
                    )

                # 以开始时间定频。若本次已经超预算，下一轮不补跑旧帧，
                # 而是立即取捕获器的最新帧继续扫描。
                next_scan_at = max(scan_started_at + target_period_sec, completed_at)

        # 2. 地图哨兵线程（定时检查小地图 OCR，自动同步地图与怪物）
        def map_sentinel():
            self._portal_ocr_wakeup.wait(1.0)
            self._portal_ocr_wakeup.clear()
            # OCR 是相对昂贵的推理；稳定地图时低频兜底，地图标题区域
            # 发生明显变化时立即唤醒。指纹只缩放灰度图，成本远低于 OCR。
            title_reference = None
            last_ocr_at = 0.0
            stable_ocr_interval = 8.0
            last_accelerated_error_log_at = 0.0
            last_map_error_log_at = 0.0
            last_unmatched_name = None
            startup_failures = 0
            # 标题变化可能先出现加载画面或低清首帧。变化后的短窗口内
            # 以 250ms 连续做 3x OCR，而不是失败一次后等待 8 秒兜底。
            switch_review_until = 0.0
            while not self.stop_event.is_set():
                if self._recognition_pause_event.is_set():
                    time.sleep(0.03)
                    continue
                try:
                    frame = self.capture.capture_frame(copy=False)
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        roi = getattr(self.map_resolver, "ocr_roi", None)
                        if roi and isinstance(roi, dict):
                            rx = max(0, int(roi.get("x", 20)))
                            ry = max(0, int(roi.get("y", 18)))
                            rw = max(10, int(roi.get("w", 260)))
                            rh = max(10, int(roi.get("h", 55)))
                            title = frame[ry:min(fh, ry + rh), rx:min(fw, rx + rw)]
                        else:
                            title = frame[18:min(75, fh), 20:min(300, fw)]
                        title_sig = None
                        if title.size > 0:
                            gray = cv2.cvtColor(title, cv2.COLOR_BGR2GRAY)
                            title_sig = cv2.resize(gray, (96, 28), interpolation=cv2.INTER_AREA)

                        now = time.perf_counter()
                        portal_ocr_pending = self._get_portal_ocr_pending()
                        portal_ocr_accelerated = portal_ocr_pending is not None
                        title_changed = False
                        if title_sig is not None and title_reference is not None:
                            title_changed = float(cv2.absdiff(title_sig, title_reference).mean()) >= 8.0
                        if title_changed and self.map_resolver.current_map_id is not None:
                            switch_review_until = max(switch_review_until, now + 4.0)
                        fast_portal_target = self._known_portal_target_after_title_change(
                            portal_ocr_pending, title_changed
                        )
                        switch_review_active = (
                            now < switch_review_until
                        )
                        # 普通门目标 MapID 已由 WZ 明确给出。按 UP 后只做
                        # 低成本标题指纹轮询，不在旧标题仍显示时连续运行 3x
                        # OCR；标题变化即证明客户端开始切图，直接同步目标图。
                        if fast_portal_target is not None:
                            title_reference = title_sig
                            last_ocr_at = now
                            portal_started_at = float(
                                portal_ocr_pending.get("triggered_at", now)
                            )
                            visual_switch_sec = max(0.0, now - portal_started_at)
                            self.log(
                                f"⚡ [传送门快速切图] 标题区域已变化，"
                                f"切图画面出现耗时={visual_switch_sec:.2f}s；"
                                f"直接同步已知目标 MapID {fast_portal_target}，跳过3x OCR"
                            )
                            res = self.map_resolver.auto_detect_and_sync_map(
                                None,
                                manual_map_id=fast_portal_target,
                            )
                        else:
                            # 已知目标门在标题尚未变化时不跑 OCR；否则一次
                            # 耗时较长的旧图 OCR 会反过来推迟下一次指纹采样。
                            known_target_waiting = bool(
                                portal_ocr_pending is not None
                                and portal_ocr_pending.get("trusted_entry")
                                and portal_ocr_pending.get("target_map_id") is not None
                            )
                            should_ocr = (
                                title_sig is not None
                                and not known_target_waiting
                                and (
                                    switch_review_active
                                    or (
                                        portal_ocr_accelerated
                                        and not known_target_waiting
                                        and (now - last_ocr_at) >= 0.5
                                    )
                                    or title_reference is None
                                    or title_changed
                                    or (now - last_ocr_at) >= (
                                        min(8.0, 1.5 * 2 ** min(max(startup_failures - 1, 0), 3))
                                        if self.current_map_info is None and startup_failures
                                        else stable_ocr_interval
                                    )
                                )
                            )
                            res = None
                        if fast_portal_target is not None or should_ocr:
                            # 先更新指纹，避免同一变化在 OCR 失败时每个轮询周期
                            # 都重复占满 CPU；8 秒后的兜底会再次尝试。
                            if fast_portal_target is None:
                                title_reference = title_sig
                                last_ocr_at = now
                                expected_map_id = None
                                if portal_ocr_pending is not None and portal_ocr_pending.get("trusted_entry"):
                                    expected_map_id = portal_ocr_pending.get("target_map_id")
                                res = self.map_resolver.auto_detect_and_sync_map(
                                    frame,
                                    high_precision_ocr=switch_review_active,
                                    confirm_map_change=switch_review_active,
                                    expected_map_id=expected_map_id,
                                )
                            if res:
                                startup_failures = 0
                                last_unmatched_name = None
                                switch_review_until = 0.0
                                locked_id = getattr(self, "_manual_map_override_id", None)
                                if locked_id is not None and int(res.get("map_id", -1)) != int(locked_id):
                                    self.log(
                                        f"⏸️ [地图锁定] 当前手动 MapID={locked_id}，忽略 OCR 结果 {res.get('map_id')}"
                                    )
                                else:
                                    self._queue_map_ui_update(res)
                                    # 请求可能恰好在本轮 OCR 执行期间到达；成功后
                                    # 必须读取最新 pending，不能只依赖调用前快照，
                                    # 否则 resolver 已切到新图后后续都会返回“同图”。
                                    completed_pending = self._get_portal_ocr_pending()
                                    if completed_pending is not None:
                                        recognized_map_id = int(res.get("map_id", -1))
                                        source_map_id = completed_pending.get("source_map_id")
                                        # auto_detect 在同图时本来就返回 None；这里再做
                                        # 一层保护，避免异常的旧图结果提前退出加速态。
                                        target_map_id = completed_pending.get("target_map_id")
                                        target_matches = (
                                            target_map_id is None
                                            or recognized_map_id == int(target_map_id)
                                        )
                                        if (
                                            target_matches
                                            and (
                                                source_map_id is None
                                                or recognized_map_id != int(source_map_id)
                                            )
                                        ):
                                            self._finish_portal_ocr_acceleration(recognized_map_id)
                            elif self.current_map_info is None:
                                startup_failures += 1
                                unmatched_name = getattr(
                                    self.map_resolver, "last_unmatched_ocr_name", None
                                )
                                if unmatched_name and self._manual_map_override_id is None:
                                    if unmatched_name != last_unmatched_name:
                                        last_unmatched_name = unmatched_name
                                        self._queue_map_ui_update({
                                            "_ocr_unmatched_name": unmatched_name
                                        })
                                        self.log(
                                            f"⚠️ [地图OCR未匹配] {unmatched_name!r} 未命中本地地图表；"
                                            "已尝试3x精识别，继续自动重试"
                                        )
                except Exception as exc:
                    now = time.perf_counter()
                    if self._get_portal_ocr_pending() is not None:
                        if (now - last_accelerated_error_log_at) >= 2.0:
                            last_accelerated_error_log_at = now
                            self.log(f"⚠️ [传送门OCR加速] 本轮识别异常，将继续重试：{exc}")
                    elif (now - last_map_error_log_at) >= 3.0:
                        last_map_error_log_at = now
                        self.log(f"⚠️ [地图自动探测] 本轮识别/同步异常，将继续重试：{exc}")
                # 默认仍按原来的 1.5 秒检查标题指纹；加速态以 250ms
                # 唤醒间隔连续重试。OCR 本身串行执行，不会叠加工作线程。
                review_is_active = time.perf_counter() < switch_review_until
                wait_timeout = (
                    0.25
                    if (
                        self._get_portal_ocr_pending() is not None
                        or review_is_active
                    )
                    else 1.5
                )
                self._portal_ocr_wakeup.wait(wait_timeout)
                self._portal_ocr_wakeup.clear()

        def viewport_render_worker():
            last_seq = -1
            next_render_at = 0.0
            while not self.stop_event.is_set():
                # 关闭视口时停止截图读取、HUD 绘制、裁剪、缩放和颜色转换；
                # raw_tracker、detector 与怪物识别线程仍独立运行。
                if not getattr(self, "_viewport_render_enabled", True):
                    time.sleep(0.05)
                    continue
                now_render = time.perf_counter()
                if now_render < next_render_at:
                    time.sleep(min(0.005, next_render_at - now_render))
                    continue
                frame = self.capture.capture_frame(copy=False)
                seq = getattr(self.capture, "frame_seq", 0)
                if frame is None or seq == last_seq:
                    time.sleep(0.002)
                    continue
                last_seq = seq
                next_render_at = now_render + (1.0 / 30.0)
                try:
                    with self.lock:
                        res = self.latest_result
                        ladder_cols = list(self.latest_ladder_cols) if hasattr(self, "latest_ladder_cols") else []
                    if self._recognition_pause_event.is_set() or self.reconnect_controller.disconnected:
                        res = None
                        ladder_cols = []
                    try:
                        dbg = self.detector.render_debug(frame, res, fps=self.capture.fps) if res is not None else frame
                    except Exception as e:
                        dbg = frame
                        if not hasattr(self, "_last_render_err_t") or time.perf_counter() - self._last_render_err_t > 3.0:
                            self._last_render_err_t = time.perf_counter()
                            print(f"[ViewportRender] render_debug error: {e}")
                            traceback.print_exc()
                    if hasattr(self, "ladder_aligner") and self.ladder_aligner and ladder_cols:
                        p_screen = res.player_pos if (res and res.player_pos) else None
                        dbg = self.ladder_aligner.draw_ladder_rope_overlay(
                            dbg, columns=ladder_cols, player_pos=p_screen
                        )

                    # 缓存视口黑边裁剪区域，避免每帧执行昂贵的全矩阵 np.any / np.mean 运算
                    if not hasattr(self, "_viewport_crop_box") or getattr(self, "_viewport_crop_frame_cnt", 0) >= 60:
                        self._viewport_crop_frame_cnt = 0
                        visible = np.any(dbg > 10, axis=2)
                        row_active = np.flatnonzero(np.mean(visible, axis=1) > 0.02)
                        col_active = np.flatnonzero(np.mean(visible, axis=0) > 0.02)
                        if row_active.size and col_active.size:
                            y1, y2 = int(row_active[0]), int(row_active[-1] + 1)
                            x1, x2 = int(col_active[0]), int(col_active[-1] + 1)
                            if (dbg.shape[0] - (y2 - y1) >= 24) or (dbg.shape[1] - (x2 - x1) >= 24):
                                self._viewport_crop_box = (y1, y2, x1, x2)
                            else:
                                self._viewport_crop_box = None
                        else:
                            self._viewport_crop_box = None
                    else:
                        self._viewport_crop_frame_cnt = getattr(self, "_viewport_crop_frame_cnt", 0) + 1

                    crop_box = getattr(self, "_viewport_crop_box", None)
                    if crop_box is not None:
                        cy1, cy2, cx1, cx2 = crop_box
                        if cy2 <= dbg.shape[0] and cx2 <= dbg.shape[1]:
                            dbg = dbg[cy1:cy2, cx1:cx2]

                    tw, th = self._viewport_target_size
                    h, w = dbg.shape[:2]
                    # 完整等比装入当前容器，不做 cover 裁切。这里只影响预览，
                    # 不影响原始捕获帧、坐标换算或视觉检测。
                    display_w = max(100, tw)
                    display_h = max(100, th)
                    sc = min(display_w / w, display_h / h)
                    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
                    interpolation = cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR
                    resized = cv2.resize(dbg, (nw, nh), interpolation=interpolation)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    with self._viewport_lock:
                        self._viewport_rgb = rgb
                        self._viewport_seq += 1
                except Exception:
                    pass

        def status_bar_worker():
            """10Hz colour HP/MP + 2Hz EXP OCR; HP/MP OCR仅初检与低频复核。"""
            next_ocr_at = 0.0
            next_hpmp_ocr_at = 0.0
            last_frame_seq = -1
            low_counts = {"hp": 0, "mp": 0}
            last_press_at = {"hp": 0.0, "mp": 0.0}
            logged_valid = False
            last_error_log = 0.0
            last_stock_block_log = {"hp": 0.0, "mp": 0.0}
            while not self.stop_event.is_set():
                if self._recognition_pause_event.is_set():
                    time.sleep(0.03)
                    continue
                roi = self.config.get("status_bar_roi")
                if not roi:
                    low_counts = {"hp": 0, "mp": 0}
                    time.sleep(0.20)
                    continue
                frame = self.capture.capture_frame(copy=False)
                frame_seq = getattr(self.capture, "frame_seq", 0)
                if frame is None or frame_seq == last_frame_seq:
                    time.sleep(0.01)
                    continue
                last_frame_seq = frame_seq
                now = time.perf_counter()
                run_ocr = now >= next_ocr_at
                run_hpmp_ocr = run_ocr and (
                    not self._status_hpmp_ocr_validated
                    or now >= next_hpmp_ocr_at
                )
                try:
                    reading = self.status_bar_reader.read(
                        frame,
                        roi,
                        run_ocr=run_ocr,
                        ocr_hp_mp=run_hpmp_ocr,
                    )
                    now = time.perf_counter()
                    if run_ocr:
                        # 从本轮完成时刻起计算下次间隔。旧逻辑按开始时刻
                        # 计算，推理超过 0.5s 后就会永久追赶、连续满载。
                        next_ocr_at = now + 0.50
                    if not reading.roi_valid:
                        time.sleep(0.10)
                        continue

                    if (
                        run_hpmp_ocr
                        and reading.ocr_confidence >= 0.70
                        and reading.hp_max
                        and reading.mp_max
                    ):
                        self._status_hpmp_ocr_validated = True
                        next_hpmp_ocr_at = now + 10.0
                        with self._status_reading_lock:
                            self._status_exact_reading = reading
                            if reading.exp_current is not None and reading.exp_percent is not None:
                                self.status_exp_tracker.update(
                                    reading.exp_current, reading.exp_percent, reading.timestamp
                                )
                        if not logged_valid:
                            logged_valid = True
                            self.log(
                                "✅ [状态栏视觉] OCR校验通过："
                                f"HP={reading.hp_current}/{reading.hp_max}, "
                                f"MP={reading.mp_current}/{reading.mp_max}, "
                                f"EXP={reading.exp_current}({reading.exp_percent}%)"
                            )
                    elif (
                        run_ocr
                        and not run_hpmp_ocr
                        and reading.exp_current is not None
                        and reading.exp_percent is not None
                    ):
                        # 只更新EXP字段；HP/MP最大值来自最近一次低频完整校验。
                        with self._status_reading_lock:
                            old_exact = self._status_exact_reading
                            if old_exact is not None:
                                self._status_exact_reading = replace(
                                    old_exact,
                                    timestamp=reading.timestamp,
                                    exp_current=reading.exp_current,
                                    exp_percent=reading.exp_percent,
                                    ocr_confidence=reading.ocr_confidence,
                                    ocr_lines=reading.ocr_lines,
                                )
                            self.status_exp_tracker.update(
                                reading.exp_current,
                                reading.exp_percent,
                                reading.timestamp,
                            )
                    # 初次OCR确认ROI后，持续由三条真实边框确认画面仍是状态栏。
                    # 颜色布局消失2秒即停止自动补给，断线/切登录界面不会把
                    # 黑画面误认为0%血蓝。
                    if (
                        self._status_hpmp_ocr_validated
                        and reading.layout_confident
                    ):
                        self._status_ocr_valid_until = now + 2.0
                    with self._status_reading_lock:
                        self._latest_status_reading = reading
                        exact = self._status_exact_reading

                    if (
                        not bool(self.config.get("enable_auto_potion", False))
                        or self.death_recovery_controller.input_suppressed
                    ):
                        low_counts = {"hp": 0, "mp": 0}
                        time.sleep(0.09)
                        continue
                    # A wrong ROI often looks like two empty bars.  Requiring a
                    # recent successful HP+MP OCR prevents that from generating
                    # any input, while short-lived skill effects remain tolerated.
                    if now > self._status_ocr_valid_until:
                        low_counts = {"hp": 0, "mp": 0}
                        time.sleep(0.09)
                        continue

                    exact_hp = exact.hp_percent if exact is not None else None
                    exact_mp = exact.mp_percent if exact is not None else None

                    def trusted_percent(bar_value, exact_value):
                        # 自动补给直接使用10Hz颜色条；OCR只承担首次ROI安全
                        # 校验和颜色暂失时的兜底，不再用旧文字值覆盖实时颜色。
                        return bar_value if bar_value is not None else exact_value

                    percentages = {
                        "hp": trusted_percent(reading.hp_bar_percent, exact_hp),
                        "mp": trusted_percent(reading.mp_bar_percent, exact_mp),
                    }
                    thresholds = {
                        "hp": float(self.config.get("hp_potion_threshold_percent", 50.0)),
                        "mp": float(self.config.get("mp_potion_threshold_percent", 30.0)),
                    }
                    cooldown_sec = max(
                        0.10, float(self.config.get("potion_cooldown_ms", 800.0)) / 1000.0
                    )
                    for kind in ("hp", "mp"):
                        percent = percentages[kind]
                        if percent is not None and percent <= thresholds[kind]:
                            low_counts[kind] += 1
                        else:
                            low_counts[kind] = 0
                        if low_counts[kind] < 2 or now - last_press_at[kind] < cooldown_sec:
                            continue
                        key_name = str(self.config.get(f"{kind}_potion_key", "")).strip()
                        if not key_name:
                            continue
                        with self._status_reading_lock:
                            stock = self.potion_stock_monitor.confirmed(
                                kind, key_name, now,
                            )
                        if stock is None or stock <= 0:
                            if now - last_stock_block_log[kind] >= 10.0:
                                last_stock_block_log[kind] = now
                                self.log(
                                    f"⛔ [自动{'回血' if kind == 'hp' else '回蓝'}库存保护] "
                                    f"{key_name.upper()} 数量"
                                    f"{'未知' if stock is None else '为0'}，不发送补药键"
                                )
                            continue
                        try:
                            vk_value = int(self.config.get(f"{kind}_potion_vk", 0) or 0) or None
                        except (TypeError, ValueError):
                            vk_value = None
                        self.input_driver.press_key(key_name, duration_ms=60, vk_code=vk_value)
                        with self._status_reading_lock:
                            self.potion_stock_monitor.note_use(kind)
                        last_press_at[kind] = time.perf_counter()
                        low_counts[kind] = 0
                        self.log(
                            f"🧪 [自动{'回血' if kind == 'hp' else '回蓝'}] "
                            f"{percent:.1f}% ≤ {thresholds[kind]:g}%，按下 {key_name.upper()}，"
                            f"冷却 {cooldown_sec * 1000.0:g}ms"
                        )
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_error_log >= 10.0:
                        last_error_log = now
                        self.log(f"⚠️ [状态栏视觉异常] {type(exc).__name__}: {exc}")
                time.sleep(0.09)

        def potion_stock_worker():
            """低频独立读取快捷栏；两帧同值后才供 UI 与补药保护使用。"""
            last_reported = {"hp": None, "mp": None}
            last_error_log = 0.0
            while not self.stop_event.is_set():
                if self._recognition_pause_event.is_set():
                    time.sleep(0.10)
                    continue
                try:
                    frame = self.capture.capture_frame(copy=True, include_overlay=False)
                    hp_key = self._pending_potion_bindings.get(
                        "hp_potion_key", (str(self.config.get("hp_potion_key", "")), 0),
                    )[0]
                    mp_key = self._pending_potion_bindings.get(
                        "mp_potion_key", (str(self.config.get("mp_potion_key", "")), 0),
                    )[0]
                    roi = self._pending_potion_roi or self.config.get("potion_bar_roi")
                    if frame is not None and roi and (
                        supports_stock_key(hp_key) or supports_stock_key(mp_key)
                    ):
                        with self._potion_stock_eval_lock:
                            reading = self.potion_stock_reader.read(frame, hp_key, mp_key, roi)
                        if (hp_key, mp_key, roi) != (
                            self._pending_potion_bindings.get(
                                "hp_potion_key", (str(self.config.get("hp_potion_key", "")), 0),
                            )[0],
                            self._pending_potion_bindings.get(
                                "mp_potion_key", (str(self.config.get("mp_potion_key", "")), 0),
                            )[0],
                            self._pending_potion_roi or self.config.get("potion_bar_roi"),
                        ):
                            continue
                        with self._status_reading_lock:
                            self.potion_stock_monitor.update(reading)
                            counts = {
                                "hp": self.potion_stock_monitor.confirmed("hp", hp_key),
                                "mp": self.potion_stock_monitor.confirmed("mp", mp_key),
                            }
                        for kind, count in counts.items():
                            if count is not None and count != last_reported[kind]:
                                last_reported[kind] = count
                                self.log(
                                    f"🧪 [药品库存] {'血药' if kind == 'hp' else '蓝药'} "
                                    f"{(hp_key if kind == 'hp' else mp_key).upper()}={count}"
                                )
                            elif count is None:
                                last_reported[kind] = None
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_error_log >= 10.0:
                        last_error_log = now
                        self.log(f"⚠️ [药品库存识别异常] {type(exc).__name__}: {exc}")
                time.sleep(0.75)

        def pet_feed_worker():
            timer = PetFeedTimer()
            observed_session = None
            observed_key = ""
            observed_enabled = False
            effective_started_at = 0.0
            last_error_log_at = 0.0
            while not self.stop_event.is_set():
                try:
                    fsm = self.combat_fsm
                    if not fsm.is_running:
                        timer.reset()
                        observed_session = None
                        observed_key = ""
                        observed_enabled = False
                    else:
                        now = time.perf_counter()
                        session_id = fsm.f6_session_id
                        key = str(self.config.get("pet_feed_key", "") or "").strip().lower()
                        enabled = bool(self.config.get("enable_auto_pet_feed", False))
                        if session_id != observed_session:
                            timer.reset()
                            observed_session = session_id
                            observed_key = key
                            observed_enabled = enabled
                            effective_started_at = fsm.f6_started_at
                        elif key != observed_key or enabled != observed_enabled:
                            # Enabling or rebinding during F6 starts a fresh
                            # full interval; disabling also cancels a due feed.
                            timer.reset()
                            observed_key = key
                            observed_enabled = enabled
                            effective_started_at = now
                        if enabled and key and timer.due(
                            session_id=session_id,
                            started_at=effective_started_at,
                            interval_sec=float(self.config.get("pet_feed_interval_sec", 300.0)),
                            now=now,
                        ):
                            vk = int(self.config.get("pet_feed_vk", 0) or 0) or None
                            if (bool(self.config.get("enable_auto_pet_feed", False))
                                    and fsm.feed_pet_safely(key, vk_code=vk, session_id=session_id)):
                                self.log(f"🐾 [宠物喂食] 已暂停攻击键并按下 {key.upper()}")
                except Exception as exc:
                    now = time.perf_counter()
                    if now - last_error_log_at >= 10.0:
                        last_error_log_at = now
                        self.log(f"⚠️ [宠物喂食异常] {type(exc).__name__}: {exc}")
                self.stop_event.wait(0.2)

        self.latest_result = None
        threading.Thread(target=raw_tracker_worker, daemon=True).start()
        threading.Thread(target=detector_worker, daemon=True).start()
        threading.Thread(target=monster_full_scan_worker, daemon=True).start()
        threading.Thread(target=map_sentinel, daemon=True).start()
        threading.Thread(target=status_bar_worker, daemon=True, name="status-bar-reader").start()
        threading.Thread(target=potion_stock_worker, daemon=True, name="potion-stock-reader").start()
        threading.Thread(target=pet_feed_worker, daemon=True, name="pet-feed-timer").start()
        threading.Thread(
            target=self.death_recovery_controller.run,
            daemon=True, name="death-recovery",
        ).start()
        threading.Thread(target=viewport_render_worker, daemon=True).start()
        self._viewport_thread_mode = True
        self._refresh_ui_video()

    def _refresh_ui_video(self):
        if not self.is_running:
            return
        if getattr(self, "_viewport_thread_mode", False):
            self._refresh_status_card()
            self._refresh_platform_status_label()
            if not getattr(self, "_viewport_render_enabled", True):
                self.fps_lbl.config(
                    text=(f"捕获: {self.capture.fps:.1f} FPS | 视口: 关闭"
                          f" | 怪物识别: {self._monster_fps_text()}")
                )
                self.root.after(100, self._refresh_ui_video)
                return
            with self._viewport_lock:
                seq = self._viewport_seq
                rgb = self._viewport_rgb
            if rgb is not None and seq != getattr(self, "_last_viewport_seq", -1):
                self._last_viewport_seq = seq
                tk_img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self.video_canvas.config(width=rgb.shape[1], height=rgb.shape[0], image=tk_img)
                self.video_canvas.image = tk_img
                if not hasattr(self, "_ui_fps_count"):
                    self._ui_fps_count = 0
                    self._ui_fps_timer = time.perf_counter()
                    self._ui_fps = 0.0
                self._ui_fps_count += 1
                now_ui = time.perf_counter()
                if now_ui - self._ui_fps_timer >= 0.5:
                    self._ui_fps = self._ui_fps_count / (now_ui - self._ui_fps_timer)
                    self._ui_fps_count = 0
                    self._ui_fps_timer = now_ui
                self.fps_lbl.config(
                    text=(f"捕获: {self.capture.fps:.1f} FPS | 视口: {self._ui_fps:.1f} FPS"
                          f" | 怪物识别: {self._monster_fps_text()}")
                )
            # 后台只生产 30FPS 视口；Tk 以 40Hz 取最新帧已足够流畅，
            # 不再用 100Hz 空轮询重复刷新状态标签。
            self.root.after(25, self._refresh_ui_video)
            return
        # 高帧率抓取最新原生游戏帧并叠加实时 HUD
        frame = self.capture.capture_frame(copy=False)
        if frame is not None:
            current_seq = getattr(self.capture, "frame_seq", 0)
            if current_seq == getattr(self, "_last_ui_frame_seq", -1):
                self.root.after(10, self._refresh_ui_video)
                return
            self._last_ui_frame_seq = current_seq
            with self.lock:
                res = self.latest_result
            if self._recognition_pause_event.is_set() or self.reconnect_controller.disconnected:
                res = None
            fps = self.capture.fps
            if res is not None:
                dbg = self.detector.render_debug(frame, res, fps=fps)
                
                # 动态录制足迹航点
                if getattr(self, "waypoint_mgr", None) and self.waypoint_mgr.is_recording and res.player_pos is not None:
                    ok = self.waypoint_mgr.record_step(res.player_pos, action="WALK")
                    if ok:
                        self.lbl_route_status.config(
                            text=f"路线状态: 正在录制中 (已采点 {len(self.waypoint_mgr.waypoints)} 个)...",
                            fg="#00e5ff"
                        )
            else:
                dbg = frame

            # 实时视口按需高亮当前正在对齐与攀爬的目标梯绳 (抓稳前持续保持，抓取成功后自动关闭，耗时 < 0.01ms)
            if hasattr(self, "ladder_aligner") and self.ladder_aligner:
                dbg = self.ladder_aligner.draw_active_target_overlay(dbg)

            # 动态更新挂机状态指示徽标
            if getattr(self, "combat_fsm", None) and self.combat_fsm.is_running:
                st = self.combat_fsm.state.value
                color = "#00e676" if ("攻击" in st or "靠近" in st) else "#ffb74d"
                self.lbl_bot_state.config(text=f"● {st}", fg=color)
            elif hasattr(self, "lbl_bot_state"):
                self.lbl_bot_state.config(text="● 状态: 停止", fg="#90a4ae")

            # 实时更新角色所在平台状态栏 (每 3 帧更新一次，避免每帧高开销计算)
            if not hasattr(self, "_tracker_frame_div"):
                self._tracker_frame_div = 0
            self._tracker_frame_div = (self._tracker_frame_div + 1) % 3

            if self._tracker_frame_div == 0:
                # 世界坐标、平台和绳梯状态只由 raw_tracker_worker 写入。
                # 视口渲染线程不得再用另一套 tracker 重做 Canvas 匹配，
                # 否则会与 F8/拓扑图竞争卷轴 offset，导致同一黄点得到
                # 两组世界坐标。
                self._refresh_platform_status_label()

            h, w = dbg.shape[:2]
            tw = max(100, self.video_container.winfo_width())
            th = max(100, self.video_container.winfo_height())
            if tw > 50 and th > 50:
                sc = min(tw / w, th / h)
                nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
                rgb = cv2.cvtColor(cv2.resize(dbg, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                tk_img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
                self.video_canvas.config(width=nw, height=nh, image=tk_img)
                self.video_canvas.image = tk_img
            
            # 统计并更新 UI 实时视口 FPS
            if not hasattr(self, "_ui_fps_count"):
                self._ui_fps_count = 0
                self._ui_fps_timer = time.perf_counter()
                self._ui_fps = 0.0
            self._ui_fps_count += 1
            now_ui = time.perf_counter()
            if now_ui - self._ui_fps_timer >= 0.5:
                self._ui_fps = self._ui_fps_count / (now_ui - self._ui_fps_timer)
                self._ui_fps_count = 0
                self._ui_fps_timer = now_ui

            self.fps_lbl.config(
                text=(f"捕获: {fps:.1f} FPS | 视口: {self._ui_fps:.1f} FPS"
                      f" | 怪物识别: {self._monster_fps_text()}")
            )
        self.root.after(10, self._refresh_ui_video)

    def on_close(self):
        self.is_running = False
        self.stop_event.set()
        if hasattr(self, "map_resolver"):
            try:
                self.map_resolver.close_ocr()
            except Exception:
                pass
        self._tray_hide_pending = False
        self._system_tray.close()
        self._hide_minigame_overlay()
        self._stop_minigame_video_test_session("程序关闭")
        if hasattr(self, "manual_input_recorder"):
            try:
                self.manual_input_recorder.stop(save=True)
            except Exception:
                pass
        if hasattr(self, "random_path_test_runner"):
            try:
                self.random_path_test_runner.stop()
            except Exception:
                pass
        if hasattr(self, "reconnect_controller"):
            try:
                self.reconnect_controller.shutdown()
            except Exception:
                pass
        if hasattr(self, "_portal_ocr_wakeup"):
            self._portal_ocr_wakeup.set()
        if hasattr(self, "world_patrol_controller") and self.world_patrol_controller:
            try:
                self.world_patrol_controller.stop()
            except Exception:
                pass
        if hasattr(self, "combat_fsm"):
            try:
                self.combat_fsm.stop()
            except Exception:
                pass
        if hasattr(self, "motion"):
            try:
                self.motion.stop()
            except Exception:
                pass
        if hasattr(self, "mob_tooltip"):
            try:
                self.mob_tooltip.hide()
            except Exception:
                pass
        if hasattr(self, "input_driver"):
            try:
                self.input_driver.release_all_keys()
            except Exception:
                pass
        if hasattr(self, "detector") and hasattr(self.detector, "release"):
            try:
                self.detector.release()
            except Exception:
                pass
        trace_fp = getattr(self, "_coordinate_trace_fp", None)
        if trace_fp is not None:
            try:
                trace_fp.close()
            except Exception:
                pass
        runtime_fp = getattr(self, "_runtime_log_fp", None)
        if runtime_fp is not None:
            try:
                runtime_fp.close()
            except Exception:
                pass
        if hasattr(self, "capture"):
            try:
                self.capture.release()
            except Exception:
                pass
        perf_timing = getattr(self, "_performance_timing", None)
        if perf_timing is not None:
            try:
                perf_timing.close()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass
