"""Tk UI; capture and localization run in an independent worker."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
import psutil
from PIL import Image, ImageTk

from .atlas import ROOT, load_atlas
from .display import annotate, overview, add_ladder_overlay
from .pipeline import Pipeline
from .player import PlayerAnchor
from .run import live_capture
from .preview_capture import PreviewCapture
from .map_preview import MapPreview
from .trial_runner import TrialRunner, launch_service


class App:
    def __init__(self, map_id, scale, backend="sift-cpu", autostart=True):
        self.root = tk.Tk()
        self.root.title("无小地图定位实验 v4 · 引导标定 / 场景定位 / 世界地图")
        self.root.geometry(f"{min(1400, self.root.winfo_screenwidth()-60)}x{min(850, self.root.winfo_screenheight()-80)}")
        self.root.configure(bg="#15202b")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.SimpleQueue()
        self.frame_lock = threading.Lock()
        self.pending_frame = None
        self.pending_map = None
        self.preview_capture = None
        self.preview_sequence = -1
        self.preview_times = deque(maxlen=120)
        self.preview_frame = None
        self.last_preview_stats = 0
        self.preview_metrics = {}
        self.commands = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.packet = None
        self.calibration_stage = None
        self.calibration_dialog = None
        self.selection_start = None
        self.selection_box = None
        self.pending_roi = None
        self.photos = {}
        self.trial = TrialRunner()
        self.trial_pending = False
        self.trial_keep_preview = True
        self.keep_preview = tk.BooleanVar(value=True)
        self.map_id = tk.StringVar(value=str(map_id))
        self.scale = tk.StringVar(value=str(scale))
        self.body_offset = tk.StringVar(value="45")
        self.backend = tk.StringVar(value=backend)
        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="地图 ID").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.map_id, width=13).pack(side="left", padx=6)
        ttk.Label(toolbar, text="画面倍率").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.scale, width=6).pack(side="left", padx=6)
        self.start_button = ttk.Button(toolbar, text="开始定位", command=self.start)
        self.start_button.pack(side="left", padx=5)
        ttk.Button(toolbar, text="停止", command=self.stop.set).pack(side="left", padx=5)
        ttk.Button(toolbar, text="重新定位", command=lambda: self.commands.put("reset")).pack(side="left", padx=5)
        ttk.Button(toolbar, text="③ 名牌与脚底", command=self.begin_calibration).pack(side="left", padx=5)
        ttk.Button(toolbar, text="取消标定", command=self.cancel_calibration).pack(side="left", padx=5)
        ttk.Button(toolbar, text="保存诊断", command=self.snapshot).pack(side="left", padx=5)
        ttk.Button(toolbar, text="忽略区域", command=self.edit_ignore_regions).pack(side="left", padx=5)
        ttk.Label(toolbar, text="脚底→身体 Y 偏移").pack(side="left", padx=4)
        ttk.Entry(toolbar, textvariable=self.body_offset, width=5).pack(side="left")
        options = ttk.Frame(self.root, padding=(10, 2))
        options.pack(fill="x")
        ttk.Label(options, text="地标后端（停止后切换）").pack(side="left")
        ttk.Combobox(options, textvariable=self.backend, values=("sift-cpu", "xfeat-cpu", "xfeat-cuda"),
                     state="readonly", width=15).pack(side="left", padx=5)
        ttk.Label(options, text="L=梯子 / R=绳索 · 橙色虚线=IMG 投影 · 绿色=近 200ms 画面验证 · XFeat 为实验选项").pack(side="left", padx=8)
        guide = ttk.LabelFrame(self.root, text="首次使用 · 按顺序操作", padding=8)
        guide.pack(fill="x", padx=10, pady=4)
        self.guide = tk.StringVar(value="① 填地图 ID、选择后端并开始定位 → ② 全身/核心特征/朝向 → ③ 名牌与脚底（可选）→ ④ 检查两幅图")
        ttk.Label(guide, textvariable=self.guide).pack(side="left", fill="x", expand=True)
        ttk.Button(guide, text="② 特征标定（主程流程）", command=self.calibrate_features).pack(side="left")
        ttk.Button(guide, text="打开最新测试结果", command=self.open_results).pack(side="left", padx=5)
        trial_bar = ttk.LabelFrame(self.root, text="⑤ 完整路线测试 · 固定地图 101000000 / XFeat GPU / 一轮", padding=5)
        trial_bar.pack(fill="x", padx=10)
        trial_controls = ttk.Frame(trial_bar)
        trial_controls.pack(fill='x')
        ttk.Button(trial_controls, text="启动按键服务（管理员）", command=self.start_input_service).pack(side="left", padx=4)
        self.trial_button = ttk.Button(trial_controls, text="完整测试一轮", command=self.start_trial)
        self.trial_button.pack(side="left", padx=4)
        self.trial_stop_button = ttk.Button(trial_controls, text="停止测试并生成结果", command=self.stop_trial, state="disabled")
        self.trial_stop_button.pack(side="left", padx=4)
        ttk.Button(trial_controls, text="本次结果/日志", command=self.open_trial_output).pack(side="left", padx=4)
        self.keep_preview_check = ttk.Checkbutton(trial_controls, text="测试时保持定位预览", variable=self.keep_preview)
        self.keep_preview_check.pack(side='left', padx=8)
        self.trial_status = tk.StringVar(value="P1 → 全部37条绳梯 → P87；自动保存录像与失败索引。F12 急停。")
        ttk.Label(trial_bar, textvariable=self.trial_status).pack(fill='x', padx=4, pady=(4,0))
        self.status = tk.StringVar(value="准备读取地图资源…")
        ttk.Label(self.root, textvariable=self.status, padding=8, font=("Microsoft YaHei UI", 11)).pack(fill="x")
        self.details = tk.StringVar(value="黄色矩形：摄像机视野；红点：角色；绿点：匹配到的静态地标。")
        ttk.Label(self.root, textvariable=self.details, padding=5).pack(fill="x")
        self.reference = tk.StringVar(value="黄点坐标：等待独立计算；不会用于修正无小地图定位")
        self.difference = tk.StringVar(value="坐标偏差：等待两路同帧结果")
        self.performance = tk.StringVar(value="CPU 定位 · 后台全局匹配 + 前台地标光流 · 原画独立采样最高 60 Hz，定位输出最高 10 Hz")
        self.navigation = tk.StringVar(value="绳梯：等待定位。这里只显示几何参考，不执行抓梯或脱困动作。")
        for variable in (self.reference, self.difference, self.performance, self.navigation):
            ttk.Label(self.root, textvariable=variable, padding=4).pack(fill="x")
        panes = ttk.Panedwindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=12, pady=6)
        scene = ttk.LabelFrame(panes, text="④ 当前场景 · XFeat / SIFT 地标 + 光流跟踪输出")
        world = ttk.LabelFrame(panes, text="④ 整个地图 · 视野区域与人物预测")
        panes.add(scene, weight=3)
        panes.add(world, weight=2)
        self.scene_tabs = ttk.Notebook(scene)
        self.scene_tabs.pack(fill="both", expand=True)
        raw_tab, matched_tab = ttk.Frame(self.scene_tabs), ttk.Frame(self.scene_tabs)
        self.scene_tabs.add(raw_tab, text="实时原画（独立刷新）")
        self.scene_tabs.add(matched_tab, text="定位输出（同帧地标）")
        self.scene_tabs.select(matched_tab)
        self.match_canvas = tk.Canvas(matched_tab, bg="#0c1219", highlightthickness=0)
        self.match_canvas.pack(fill="both", expand=True)
        self.preview_status = tk.StringVar(value="预览与识别独立运行；定位结果使用其原始帧，不叠加到较新的画面。")
        ttk.Label(self.root, textvariable=self.preview_status, padding=4).pack(fill="x")
        self.canvas = tk.Canvas(raw_tab, bg="#0c1219", height=470, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.map_canvas = tk.Canvas(world, bg="#0c1219", width=400, height=470, highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True)
        self.map_view = MapPreview(self.map_canvas)
        ttk.Label(self.root, text="黄色框：当前视野估计；红点：视觉脚底；橙色虚框/点：当前时刻的显示预测（最多250ms）；橙圈：不确定范围；青色十字：独立黄点。",
                  padding=5).pack(fill="x")
        if autostart:
            self.root.after(150, self.start)
        self.root.after(16, self.poll)
        self.root.after(500, self.poll_trial)

    def emit(self, kind, payload):
        if kind == "frame":
            with self.frame_lock:
                self.pending_frame = payload
        else:
            self.events.put((kind, payload))

    def start(self):
        if self.trial_pending or (self.trial.running and not self.trial_keep_preview):
            return
        if self.worker and self.worker.is_alive():
            return
        try:
            map_id, scale = int(self.map_id.get()), float(self.scale.get())
            body_offset = float(self.body_offset.get())
            if not .25 <= scale <= 4:
                raise ValueError("画面倍率需在 0.25 至 4 之间")
            if not np.isfinite(body_offset):
                raise ValueError("参考点偏移必须为有限数值")
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.cancel_calibration()
        self.packet = None
        self.preview_frame = None
        self.preview_sequence = -1
        self.preview_times.clear()
        with self.frame_lock:
            self.pending_frame = None
            self.pending_map = None
        self.map_view.clear()
        self.canvas.delete("all")
        self.map_canvas.delete("all")
        self.stop.clear()
        self.start_button.configure(state="disabled")
        self.worker = threading.Thread(target=self.work, args=(map_id, scale, body_offset, self.backend.get()), daemon=True)
        self.worker.start()

    def work(self, map_id, scale, body_offset, backend):
        capture = None
        pipeline = None
        try:
            from .check_environment import require
            self.emit('status', '正在检查 Python 依赖、PyTorch 和所选后端…')
            environment = require(backend)
            self.emit('status', f"环境检查通过：{backend} / PyTorch {environment.get('torch','无需')} / {environment.get('gpu','CPU')}")
            atlas = load_atlas(map_id, progress=lambda s: self.emit("status", s))
            self.emit("status", "正在提取世界场景地标…")
            pipeline = Pipeline(atlas, scale, body_offset_y=body_offset, backend=backend, asynchronous_reference=True)
            model = pipeline.model
            map_background = overview(atlas, (1, 1), SimpleNamespace(camera=None), scale, size=(700, 850))
            self.emit("map_setup", (map_background, atlas.origin, atlas.bgr.shape))
            capture = PreviewCapture(live_capture, self.stop).start()
            self.preview_capture = capture
            capture_sequence = -1
            process = psutil.Process()
            process.cpu_percent()
            times = deque(maxlen=120)
            last_render = last_stats = 0.0
            runtime = dict(fps=0, cpu_percent=0, cpu_machine_percent=0, rss_mb=0)
            while not self.stop.is_set():
                tick = time.perf_counter()
                while not self.commands.empty():
                    command = self.commands.get_nowait()
                    if command == "reset":
                        pipeline.reset()
                    elif command == "anchor":
                        pipeline.reload_anchor()
                    elif command == 'regions':
                        pipeline.reset()
                        pipeline.reload_anchor()
                sequence, captured = capture.get(capture_sequence, timeout=.05)
                if capture.error:
                    raise capture.error
                if sequence == capture_sequence:
                    continue
                capture_sequence = sequence
                if captured is None:
                    pipeline.reset()
                    self.emit("unavailable", "未取得游戏画面")
                    self.stop.wait(.1)
                    continue
                frame, frame_received, capture_ms = captured
                record = pipeline.update(frame, frame_received)
                record["timings_ms"]["capture"] = capture_ms
                completed = time.perf_counter()
                times.append(completed)
                if completed-last_stats >= 1:
                    cpu = process.cpu_percent()
                    runtime = dict(fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0,
                                   cpu_percent=cpu, cpu_machine_percent=cpu/(psutil.cpu_count() or 1),
                                   rss_mb=process.memory_info().rss/1024**2)
                    last_stats = completed
                record["runtime"] = runtime
                with self.frame_lock:
                    self.pending_map = (record, frame.shape, scale)
                # Diagnostic overlays run at 10Hz; raw preview is independent.
                if completed-last_render >= .10:
                    player = record["player"]
                    rendered = add_ladder_overlay(annotate(frame, model, pipeline.pose, player), record["ladder"], record["ladders"])
                    world = overview(atlas, frame.shape, pipeline.pose, scale, player, (700, 850), record["yellow"], background=map_background)
                    self.emit("frame", (frame, rendered, world, record))
                    last_render = completed
                self.stop.wait(max(0, 1/60-(time.perf_counter()-tick)))
        except Exception as exc:
            self.emit("error", f"{type(exc).__name__}: {exc}")
        finally:
            if capture:
                capture.close()
                self.preview_capture = None
            if pipeline:
                pipeline.close()
            self.emit("done", None)

    def show_image(self, canvas, image, key):
        w, h = max(10, canvas.winfo_width()), max(10, canvas.winfo_height())
        ratio = min(w/image.shape[1], h/image.shape[0])
        resized = cv2.resize(image, (max(1, round(image.shape[1]*ratio)), max(1, round(image.shape[0]*ratio))), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.photos[key] = ImageTk.PhotoImage(pil)
        items = canvas.find_withtag("bitmap")
        if items:
            canvas.itemconfigure(items[0], image=self.photos[key])
        else:
            canvas.create_image(0, 0, image=self.photos[key], anchor="nw", tags="bitmap")
        if key == "frame":
            self.display_ratio = ratio

    def poll(self):
        tick = time.perf_counter()
        capture = self.preview_capture
        if capture and not self.stop.is_set() and self.calibration_stage is None:
            sequence, latest = capture.get()
            if latest is not None and sequence != self.preview_sequence:
                self.preview_sequence = sequence
                self.preview_frame = latest[0]
                if self.scene_tabs.index(self.scene_tabs.select()) == 0:
                    self.show_image(self.canvas, latest[0], "frame")
                    self.preview_times.append(tick)
                if tick-self.last_preview_stats >= .25:
                    times = self.preview_times
                    fps = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0
                    age = (tick-self.packet[3]['timestamp'])*1000 if self.packet else None
                    label = f"{age:.0f}ms" if age is not None else "等待结果"
                    self.preview_metrics = dict(display_fps=fps, capture_age_ms=(tick-latest[1])*1000,
                                                localization_age_ms=age,
                                                visible_tab=self.scene_tabs.index(self.scene_tabs.select()))
                    self.preview_status.set(f"原画预览 {fps:.1f} FPS（原画页可见时统计） | 当前帧年龄 {(tick-latest[1])*1000:.0f}ms | 定位结果年龄 {label}")
                    self.last_preview_stats = tick
        with self.frame_lock:
            pending, self.pending_frame = self.pending_frame, None
            map_result, self.pending_map = self.pending_map, None
        messages = []
        if pending is not None:
            messages.append(("frame", pending))
        for _ in range(16):
            try:
                messages.append(self.events.get_nowait())
            except queue.Empty:
                break
        try:
            for kind, payload in messages:
                if kind == "frame" and self.calibration_stage is None:
                    self.packet = payload
                    _, rendered, world, record = payload
                    if self.scene_tabs.index(self.scene_tabs.select()) == 1:
                        self.show_image(self.match_canvas, rendered, "matches")
                    # World canvas is drawn independently below.
                    state = {"LOCKED": "已锁定", "COASTING": "短时光流推算", "LOST": "定位丢失", "AMBIGUOUS": "重复地形，位置不唯一"}[record["status"]]
                    camera = record["camera"]
                    location = f"视野左上角世界坐标 ({camera[0]:.1f}, {camera[1]:.1f})" if camera else "当前无可靠世界坐标"
                    self.status.set(f"{state}  |  {location}  |  地标 {record['inliers']}  / 次候选 {record['runner_up']}  |  {record['elapsed_ms']:.0f} ms")
                    pos = record.get("player_world")
                    if pos:
                        p = record["player_state"]
                        method = "短时推算，不能当作识别命中" if p["status"] == "PREDICTED" else "主程序名牌＋特征识别"
                        self.details.set(f"视觉脚底 ({pos[0]:.1f}, {pos[1]:.1f}) · {method} · 观测年龄 {p['observation_age']*1000:.0f} ms · 估计不确定范围 ±{p['uncertainty_px']:.1f} px")
                    else:
                        reason = record.get('player_state', {}).get('reason', '')
                        explanation = {
                            'Camera not localized': '地图定位尚未建立或已丢失',
                            'No verified player observation': '尚未取得可靠人物识别',
                            'Prediction expired or velocity not established': '人物观测已过期或尚未建立运动速度',
                        }.get(reason, reason or '等待可靠观测')
                        self.details.set(f"人物坐标暂不可用：{explanation}。")
                    yellow = record["yellow"]
                    if yellow["detected"]:
                        raw, snapped = yellow["raw_world"], yellow["snapped_world"]
                        self.reference.set(f"独立黄点原始世界坐标 ({raw[0]:.1f}, {raw[1]:.1f})  |  原程序几何吸附 ({snapped[0]:.1f}, {snapped[1]:.1f}) | 对照帧年龄 {yellow.get('observation_age',0)*1000:.0f}ms")
                    else:
                        self.reference.set("独立黄点：未检出；无小地图定位仍独立运行")
                    comp = record["comparison"]
                    if comp:
                        dx, dy = comp["visual_body_minus_yellow_raw"]
                        fx, fy = comp["feet_minus_yellow_raw"]
                        label = "黄点采样帧的预测对照" if comp["kind"] == "prediction_comparison" else "黄点采样帧的同帧测量对照"
                        self.difference.set(f"{label}：视觉身体参考 − 黄点 ΔX={dx:+.1f}, ΔY={dy:+.1f}, 距离={comp['distance']:.1f} px  |  未换参考点 Δ=({fx:+.1f}, {fy:+.1f})")
                    else:
                        self.difference.set("坐标偏差：等待两路有效结果")
                    timing, runtime = record["timings_ms"], record["runtime"]
                    self.performance.set(f"{record['backend']} / 定位 {runtime['fps']:.1f} Hz / CPU 整机 {runtime['cpu_machine_percent']:.1f}% / 内存 {runtime['rss_mb']:.0f} MB | 捕获 {timing['capture']:.1f} / 镜头 {timing['camera']:.1f} / 人物 {timing['player']:.1f} / 黄点 {timing['yellow']:.1f} / 绳梯 {timing['ladders']:.1f} ms | 后台匹配 {timing['background_anchor']:.0f} ms")
                    count = len(record["ladders"])
                    verified = sum(n["status"] != "projected" for n in record["ladders"])
                    prefix = f"视野内 {count} 条绳梯，画面验证 {verified} 条。"
                    ladder = record["ladder"]
                    if ladder:
                        self.navigation.set(prefix+f"附近绳梯 X={ladder['ladder_world'][0]}，横向距离 {ladder['dx_world']:+.1f} 世界像素 / {ladder['dx_screen']:+.1f} 屏幕像素 · 尚未验证抓稳")
                    else:
                        self.navigation.set(prefix+"人物未识别也可投影绳梯；自动抓梯/脱困尚未接入控制。")
                elif kind == "map_setup":
                    self.map_view.setup(*payload)
                elif kind in ("status", "error", "unavailable"):
                    self.status.set(payload)
                    if kind in ("error", "unavailable"):
                        self.packet = None
                        map_result = None
                        self.map_view.clear()
                        self.map_view.size = None
                        self.map_canvas.delete("all")
                        self.canvas.delete("all")
                        self.details.set("当前没有有效定位结果")
                        self.reference.set("独立黄点：当前无有效输入")
                        self.difference.set("坐标偏差：无有效对照")
                        self.navigation.set("绳梯：无有效定位")
                elif kind == "done":
                    self.start_button.configure(state="normal")
                    if self.stop.is_set():
                        self.status.set("已停止（画面为停止前快照）")
        except queue.Empty:
            pass
        if map_result is not None:
            self.map_view.update(*map_result)
        if not self.stop.is_set():
            self.map_view.draw(time.perf_counter())
        self.root.after(16, self.poll)

    def begin_calibration(self):
        if self.packet is None:
            messagebox.showinfo("标定", "请先开始定位，等游戏画面出现后再标定。")
            return
        self.cancel_calibration()
        from .nameplate_dialog import NameplateDialog
        frame = self.preview_frame if self.preview_frame is not None else self.packet[0]
        self.calibration_dialog = NameplateDialog(self.root, frame, self.save_nameplate)

    def edit_ignore_regions(self):
        frame = self.preview_frame if self.preview_frame is not None else self.packet[0] if self.packet is not None else None
        if frame is None:
            messagebox.showinfo('忽略区域','请先开始定位，取得游戏画面。')
            return
        from .ignore_dialog import IgnoreRegionsDialog
        from .ignore_regions import save
        self.cancel_calibration()
        def apply(regions):
            save(regions)
            self.commands.put('regions')
            self.status.set(f'已保存 {len(regions)} 个忽略区域；未设置的区域不会自动遮罩。')
        self.calibration_dialog = IgnoreRegionsDialog(self.root,frame,apply)

    def cancel_calibration(self):
        self.calibration_stage = None
        self.selection_start = None
        self.pending_roi = None
        self.canvas.delete("selection")
        self.canvas.delete("feet")
        dialog = self.calibration_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        self.calibration_dialog = None

    def save_nameplate(self, frame, roi, feet):
        from .calibration import calibration_detector
        detector = calibration_detector()
        x, y, w, h = roi
        try:
            if not detector.set_manual_cropped_player(frame[y:y+h, x:x+w]):
                raise ValueError("名牌选区无效，请重新框选")
        finally:
            detector.executor.shutdown(wait=False, cancel_futures=True)
        PlayerAnchor().save(frame, roi, feet)
        self.commands.put("anchor")
        self.canvas.delete("selection")
        self.status.set("角色名牌与脚底标定已保存")
        self.guide.set("③ 名牌与脚底已保存 → ④ 检查人物点是否贴合脚底、地图黄色视野框是否正确")

    def calibrate_features(self):
        if self.packet is None:
            messagebox.showinfo("标定", "请先开始定位，等游戏画面出现；让人物站稳再标定。")
            return
        from src.gui.app import TwoStagePlayerCalibrationDialog
        from .calibration import calibration_detector
        frame = (self.preview_frame if self.preview_frame is not None else self.packet[0]).copy()

        def save(whole, feature, box, facing, feature_mask=None, feature_polygon=None):
            detector = calibration_detector()
            try:
                if not detector.save_two_stage_calibration(whole, feature, box, facing,
                        feature_mask=feature_mask, feature_polygon=feature_polygon):
                    raise ValueError("特征选区无效，请重试")
                self.commands.put("anchor")
                self.guide.set("② 特征标定已保存 → ③ 可选标定名牌与脚底 → ④ 检查视觉输出")
                messagebox.showinfo("标定完成", "已保存实验程序角色特征，并自动生成反向模板。")
            except Exception as exc:
                messagebox.showerror("标定失败", str(exc))
            finally:
                detector.executor.shutdown(wait=False, cancel_futures=True)
        self.cancel_calibration()
        TwoStagePlayerCalibrationDialog(self.root, frame, save)

    def open_results(self):
        import webbrowser
        pages = list((ROOT / 'no_minimap_lab' / 'output').glob('*/RESULTS.html'))
        if pages:
            webbrowser.open(max(pages, key=lambda p:p.stat().st_mtime).as_uri())
        else:
            messagebox.showinfo("测试结果", "尚无结果，请先完成一次测试。")

    def start_input_service(self):
        try:
            launch_service()
            self.trial_status.set("按键服务启动请求已发出；完成 UAC 后点击“完整测试一轮”。")
        except Exception as exc:
            messagebox.showerror("按键服务", str(exc))

    def start_trial(self):
        if self.trial.running or self.trial_pending:
            return
        if self.map_id.get().strip() != '101000000' or self.scale.get().strip() not in ('1', '1.0'):
            messagebox.showinfo("完整测试", "这条完整路线固定使用地图101000000、画面倍率1.0，请先设置一致。")
            return
        from .trial_runner import service_ready
        if not service_ready():
            messagebox.showinfo("完整测试", "请先点击“启动按键服务（管理员）”，完成 UAC 后再开始。")
            return
        self.trial_pending = True
        self.trial_keep_preview = self.keep_preview.get()
        self.keep_preview_check.configure(state='disabled')
        self.trial_button.configure(state='disabled')
        self.start_button.configure(state='disabled')
        if self.trial_keep_preview:
            self.trial_status.set("正在启动完整测试，保持定位预览…")
        else:
            self.stop.set()
            self.trial_status.set("正在暂停定位预览，避免两套识别争用资源…")
        self.root.after(100, self.finish_trial_start)

    def finish_trial_start(self):
        if not self.trial_pending:
            return
        if not self.trial_keep_preview and self.worker and self.worker.is_alive():
            self.root.after(100, self.finish_trial_start)
            return
        try:
            self.trial.start()
            self.trial_stop_button.configure(state='normal')
            self.trial_status.set("完整测试已启动；"+('保持定位预览。' if self.trial_keep_preview else '预览暂停。')+"结果完成后自动打开。")
        except Exception as exc:
            self.trial_button.configure(state='normal')
            self.start_button.configure(state='normal')
            self.keep_preview_check.configure(state='normal')
            messagebox.showerror("无法开始测试", str(exc))
        finally:
            self.trial_pending = False

    def stop_trial(self):
        self.trial.stop()
        self.trial_status.set("已请求停止，正在释放按键并生成结果，请等待…")

    def poll_trial(self):
        if self.trial.running:
            preview_running = self.worker is not None and self.worker.is_alive() and not self.stop.is_set()
            self.start_button.configure(state='normal' if self.trial_keep_preview and not preview_running else 'disabled')
            self.trial_status.set(('测试中（定位预览运行中） · ' if preview_running else '测试中（预览未运行） · ')+self.trial.progress())
        elif self.trial.process is not None:
            code = self.trial.process.returncode
            self.trial_status.set("测试完成，已生成结果。" if code == 0 else "测试已停止或失败，请查看本次结果/日志。")
            self.trial.process = None
            self.trial_button.configure(state='normal')
            self.trial_stop_button.configure(state='disabled')
            self.start_button.configure(state='normal')
            self.keep_preview_check.configure(state='normal')
        self.root.after(500, self.poll_trial)

    def open_trial_output(self):
        if self.trial.output is None:
            messagebox.showinfo("本次结果", "尚未从 GUI 启动测试。历史报告可点击“打开最新测试结果”。")
            return
        import os
        import webbrowser
        report = self.trial.output/'RESULTS.html'
        if report.exists():
            webbrowser.open(report.as_uri())
        else:
            os.startfile(str(self.trial.output))

    def snapshot(self):
        if self.packet is None:
            return
        path = ROOT / "no_minimap_lab" / "output" / time.strftime("%Y%m%d_%H%M%S")
        path.mkdir(parents=True, exist_ok=True)
        for name, frame in zip(("raw", "matches", "world"), self.packet[:3]):
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path / f"{name}.png")
        (path / "pose.json").write_text(json.dumps(dict(self.packet[3], preview=self.preview_metrics), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.set(f"诊断已保存：{path}")

    def close(self):
        self.trial_pending = False
        self.trial.stop()
        self.stop.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
