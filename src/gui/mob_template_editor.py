"""怪物模板查看、删减与局部特征框编辑器。"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageDraw, ImageTk

from src.gui.window_layout import fit_window_to_work_area, get_work_area


@dataclass
class MobTemplatePair:
    """同一动作/朝向的全身图与局部特征图。"""

    key: str
    global_path: Optional[str]
    feature_path: Optional[str]

    @property
    def label(self) -> str:
        text = self.key
        if text.startswith("mob_"):
            text = text[4:]
        names = {
            "right": "站立 / 右",
            "left": "站立 / 左",
            "move_right": "移动 / 右",
            "move_left": "移动 / 左",
            "hit1_right": "受击 / 右",
            "hit1_left": "受击 / 左",
            "jump_right": "跳跃 / 右",
            "jump_left": "跳跃 / 左",
        }
        return names.get(text, text.replace("_", " / "))


def _normalized_template_key(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    return (
        stem.replace("mob_global_", "mob_", 1)
        .replace("global_", "", 1)
        .replace("_global", "")
        .replace("global", "")
    )


def list_mob_template_pairs(template_dir: str) -> List[MobTemplatePair]:
    """按匹配器采用的命名规则枚举模板配对，也保留孤立文件供删除。"""
    files = sorted(glob.glob(os.path.join(template_dir, "*.png")))
    pairs: Dict[str, Dict[str, str]] = {}
    for path in files:
        key = _normalized_template_key(path)
        slot = pairs.setdefault(key, {})
        kind = "global" if "global" in os.path.basename(path).lower() else "feature"
        slot[kind] = path
    return [
        MobTemplatePair(key, item.get("global"), item.get("feature"))
        for key, item in sorted(pairs.items())
    ]


def locate_feature_box(global_path: str, feature_path: str) -> Optional[Tuple[int, int, int, int, float]]:
    """定位局部特征在全身图中的像素框，规则与运行时匹配器一致。"""
    global_img = cv2.imread(global_path, cv2.IMREAD_UNCHANGED)
    feature_img = cv2.imread(feature_path, cv2.IMREAD_UNCHANGED)
    if global_img is None or feature_img is None:
        return None
    global_bgr = global_img[:, :, :3] if global_img.ndim == 3 else global_img
    feature_bgr = feature_img[:, :, :3] if feature_img.ndim == 3 else feature_img
    gh, gw = global_bgr.shape[:2]
    fh, fw = feature_bgr.shape[:2]
    if fw > gw or fh > gh or fw < 1 or fh < 1:
        return None
    result = cv2.matchTemplate(global_bgr, feature_bgr, cv2.TM_CCOEFF_NORMED)
    _, score, _, position = cv2.minMaxLoc(result)
    return int(position[0]), int(position[1]), int(fw), int(fh), float(score)


def _checkerboard(image: Image.Image, cell: int = 8) -> Image.Image:
    rgba = image.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (55, 57, 64, 255))
    draw = ImageDraw.Draw(bg)
    width, height = rgba.size
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(77, 80, 89, 255))
    bg.alpha_composite(rgba)
    return bg.convert("RGB")


class MobTemplateEditor(tk.Toplevel):
    """对单个 Mob ID 的模板对进行可视化管理。"""

    def __init__(
        self,
        parent,
        mob_id: str,
        mob_name: str,
        template_root: str,
        on_changed: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        self.mob_id = str(mob_id)
        self.mob_name = str(mob_name or "")
        self.template_dir = os.path.join(template_root, self.mob_id)
        self.on_changed = on_changed
        self.pairs: List[MobTemplatePair] = []
        self.current_pair: Optional[MobTemplatePair] = None
        self.original_image: Optional[Image.Image] = None
        self.selection_box: Optional[Tuple[int, int, int, int]] = None
        self.existing_box: Optional[Tuple[int, int, int, int]] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self._main_photo = None
        self._crop_photo = None
        self._magnifier_photo = None

        self.title(f"怪物模板编辑 - {self.mob_name} ({self.mob_id})")
        # 使用父窗口所在显示器的工作区，排除任务栏并对最小尺寸限幅。
        # 右侧主预览本身带双向滚动条，缩小窗口时优先压缩该区域。
        fit_window_to_work_area(
            self, (1280, 920), (760, 560), parent=parent,
            width_fraction=0.94, height_fraction=0.90,
        )
        self.configure(bg="#17181d")
        self.transient(parent)

        self.zoom_var = tk.IntVar(value=6)
        self.status_var = tk.StringVar(value="请选择左侧模板")
        self.crop_status_var = tk.StringVar(value="拖动鼠标重新框选局部特征")
        self._build_ui()
        self._refresh_pairs()

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#202228", padx=12, pady=9)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=f"👾 {self.mob_name}  Mob ID: {self.mob_id}",
            font=("Segoe UI", 12, "bold"), fg="#ffca66", bg="#202228",
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text="在右侧像素预览中按住左键拖框；保存后立即替换该动作的局部特征图",
            font=("Segoe UI", 9), fg="#aeb6c2", bg="#202228",
        ).pack(side=tk.RIGHT)

        _, _, work_width, _ = get_work_area(self)
        compact_layout = work_width < 1100
        body = tk.PanedWindow(
            self,
            orient=(tk.VERTICAL if compact_layout else tk.HORIZONTAL),
            sashwidth=5,
            bg="#111217",
        )
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(body, bg="#202228", width=350)
        body.add(left, minsize=(180 if compact_layout else 310))
        self.tree = ttk.Treeview(
            left, columns=("action", "global", "feature", "size"),
            show="headings", height=18,
        )
        self.tree.heading("action", text="动作 / 朝向")
        self.tree.heading("global", text="全身")
        self.tree.heading("feature", text="特征")
        self.tree.heading("size", text="特征尺寸")
        self.tree.column("action", width=125, anchor=tk.W)
        self.tree.column("global", width=45, anchor=tk.CENTER, stretch=False)
        self.tree.column("feature", width=45, anchor=tk.CENTER, stretch=False)
        self.tree.column("size", width=75, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=7, pady=7)
        self.tree.bind("<<TreeviewSelect>>", self._on_pair_selected)

        left_buttons = tk.Frame(left, bg="#202228")
        left_buttons.pack(fill=tk.X, padx=7, pady=(0, 7))
        self.delete_button = tk.Button(
            left_buttons, text="🗑 删除所选模板对", command=self._delete_current_pair,
            bg="#b71c1c", fg="white", activebackground="#d32f2f",
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self.delete_button.pack(fill=tk.X)
        tk.Label(
            left,
            text="删除会同时移除该动作/朝向的全身图与局部特征图，\n不会影响同一怪物的其他模板。",
            justify=tk.LEFT, font=("Segoe UI", 8), fg="#ef9a9a", bg="#202228",
        ).pack(anchor=tk.W, padx=8, pady=(0, 8))

        right = tk.Frame(body, bg="#202228")
        body.add(right, minsize=(340 if compact_layout else 560))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1, minsize=180)
        tools = tk.Frame(right, bg="#202228")
        tools.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        tk.Label(tools, text="像素放大:", fg="#d8dee9", bg="#202228").pack(side=tk.LEFT)
        zoom_box = ttk.Combobox(
            tools, textvariable=self.zoom_var, values=(2, 3, 4, 5, 6, 8, 10, 12, 16),
            width=4, state="readonly",
        )
        zoom_box.pack(side=tk.LEFT, padx=(5, 3))
        tk.Label(tools, text="倍（最近邻，不平滑像素）", fg="#90a4ae", bg="#202228").pack(side=tk.LEFT)
        zoom_box.bind("<<ComboboxSelected>>", lambda _e: self._render_main_preview())
        self.save_button = tk.Button(
            tools, text="💾 保存新特征框", command=self._save_feature_crop,
            bg="#2e7d32", fg="white", activebackground="#388e3c",
            relief=tk.FLAT, padx=12, state=tk.DISABLED,
        )
        self.save_button.pack(side=tk.RIGHT)

        preview_frame = tk.Frame(right, bg="#101116")
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.canvas = tk.Canvas(
            preview_frame, bg="#101116", highlightthickness=0,
            xscrollincrement=1, yscrollincrement=1,
        )
        sx = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        sy = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Motion>", self._on_mouse_hover)

        footer = tk.Frame(right, bg="#202228")
        # grid 第 2 行始终按实际请求高度保留；窗口变小时只压缩上方
        # 可滚动主预览，不再把实时放大镜裁掉一半。
        footer.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))
        info = tk.Frame(footer, bg="#202228")
        info.pack(fill=tk.X)
        tk.Label(info, textvariable=self.status_var, justify=tk.LEFT,
                 font=("Consolas", 9), fg="#80deea", bg="#202228").pack(anchor=tk.W)
        tk.Label(info, textvariable=self.crop_status_var, justify=tk.LEFT,
                 font=("Segoe UI", 9), fg="#ffd54f", bg="#202228").pack(anchor=tk.W, pady=(4, 0))

        preview_row = tk.Frame(footer, bg="#202228")
        preview_row.pack(fill=tk.X, pady=(6, 0))
        mag_box = tk.LabelFrame(
            preview_row, text="鼠标实时像素放大镜（10×）", font=("Segoe UI", 8),
            fg="#00e5ff", bg="#181a20", padx=4, pady=4,
        )
        mag_box.pack(side=tk.LEFT)
        self.magnifier_canvas = tk.Canvas(
            mag_box, width=170, height=170, bg="#101116", highlightthickness=0
        )
        self.magnifier_canvas.pack(side=tk.LEFT)
        self.magnifier_info_var = tk.StringVar(value="移动鼠标查看原始像素")
        tk.Label(
            mag_box, textvariable=self.magnifier_info_var, justify=tk.LEFT,
            font=("Consolas", 8), fg="#ffd54f", bg="#181a20", width=17,
        ).pack(side=tk.LEFT, padx=(6, 2))

        crop_box = tk.LabelFrame(
            preview_row, text="框选区域实时预览", font=("Segoe UI", 8),
            fg="#b0bec5", bg="#181a20", padx=4, pady=4,
        )
        crop_box.pack(side=tk.RIGHT, padx=(8, 0))
        self.crop_canvas = tk.Canvas(crop_box, width=220, height=145, bg="#101116", highlightthickness=0)
        self.crop_canvas.pack()

    def _refresh_pairs(self, select_key: Optional[str] = None) -> None:
        self.pairs = list_mob_template_pairs(self.template_dir)
        self.tree.delete(*self.tree.get_children())
        selected_item = None
        for index, pair in enumerate(self.pairs):
            size = "—"
            if pair.feature_path:
                try:
                    with Image.open(pair.feature_path) as image:
                        size = f"{image.width}×{image.height}"
                except OSError:
                    size = "损坏"
            item = self.tree.insert(
                "", tk.END, iid=str(index),
                values=(pair.label, "✔" if pair.global_path else "—", "✔" if pair.feature_path else "—", size),
            )
            if select_key == pair.key:
                selected_item = item
        if selected_item is None and self.pairs:
            selected_item = "0"
        if selected_item is not None:
            self.tree.selection_set(selected_item)
            self.tree.focus(selected_item)
            self.tree.see(selected_item)
            self._load_pair(self.pairs[int(selected_item)])
        else:
            self._clear_preview("此怪物当前没有可编辑模板")

    def _on_pair_selected(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if 0 <= index < len(self.pairs):
            self._load_pair(self.pairs[index])

    def _load_pair(self, pair: MobTemplatePair) -> None:
        self.current_pair = pair
        self.selection_box = None
        self.existing_box = None
        self.drag_start = None
        self.delete_button.config(state=tk.NORMAL)
        if not pair.global_path:
            self.original_image = None
            self.save_button.config(state=tk.DISABLED)
            self._clear_preview("此条只有局部特征图，没有可供重新框选的全身图")
            return
        try:
            self.original_image = Image.open(pair.global_path).convert("RGBA")
        except OSError as exc:
            self.original_image = None
            self.save_button.config(state=tk.DISABLED)
            self._clear_preview(f"全身模板读取失败：{exc}")
            return
        match_score = None
        if pair.feature_path:
            match = locate_feature_box(pair.global_path, pair.feature_path)
            if match:
                x, y, w, h, match_score = match
                self.existing_box = (x, y, x + w, y + h)
        self.save_button.config(state=tk.DISABLED)
        score_text = f"，定位分数 {match_score:.3f}" if match_score is not None else ""
        self.status_var.set(
            f"{pair.label} | 全身 {self.original_image.width}×{self.original_image.height}px"
            f" | {'已有局部特征' if pair.feature_path else '尚无局部特征'}{score_text}"
        )
        self.crop_status_var.set("绿色框为当前特征；拖动鼠标产生黄色新框")
        self._render_main_preview()
        self._render_crop_preview(self.existing_box)
        initial_box = self.existing_box or (0, 0, self.original_image.width, self.original_image.height)
        self._render_magnifier(
            (initial_box[0] + initial_box[2]) // 2,
            (initial_box[1] + initial_box[3]) // 2,
        )

    def _clear_preview(self, status: str) -> None:
        self.canvas.delete("all")
        self.crop_canvas.delete("all")
        self.status_var.set(status)
        self.crop_status_var.set("")
        self._main_photo = None
        self._crop_photo = None
        self._magnifier_photo = None
        if hasattr(self, "magnifier_canvas"):
            self.magnifier_canvas.delete("all")
        self.delete_button.config(state=tk.NORMAL if self.current_pair else tk.DISABLED)

    def _render_main_preview(self) -> None:
        if self.original_image is None:
            return
        zoom = max(1, int(self.zoom_var.get()))
        display = _checkerboard(self.original_image)
        display = display.resize(
            (display.width * zoom, display.height * zoom), Image.Resampling.NEAREST
        )
        self._main_photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._main_photo, anchor=tk.NW, tags="sprite")
        self.canvas.configure(scrollregion=(0, 0, display.width, display.height))
        if self.existing_box:
            self._draw_box(self.existing_box, "#00e676", "当前")
        if self.selection_box:
            self._draw_box(self.selection_box, "#ffea00", "新选")

    def _draw_box(self, box: Tuple[int, int, int, int], color: str, label: str) -> None:
        zoom = max(1, int(self.zoom_var.get()))
        x1, y1, x2, y2 = box
        coords = (x1 * zoom, y1 * zoom, x2 * zoom, y2 * zoom)
        self.canvas.create_rectangle(*coords, outline=color, width=2, tags="overlay")
        self.canvas.create_text(
            coords[0] + 3, coords[1] + 3, text=label, fill=color,
            font=("Segoe UI", 8, "bold"), anchor=tk.NW, tags="overlay",
        )

    def _canvas_to_pixel(self, event) -> Tuple[int, int]:
        zoom = max(1, int(self.zoom_var.get()))
        x = int(self.canvas.canvasx(event.x) // zoom)
        y = int(self.canvas.canvasy(event.y) // zoom)
        if self.original_image:
            x = min(max(0, x), self.original_image.width)
            y = min(max(0, y), self.original_image.height)
        return x, y

    def _on_drag_start(self, event) -> None:
        if self.original_image is None:
            return
        self.drag_start = self._canvas_to_pixel(event)
        self.selection_box = (*self.drag_start, *self.drag_start)
        self._render_magnifier(*self._magnifier_pixel(event))
        self._render_main_preview()

    def _on_drag_motion(self, event) -> None:
        if not self.drag_start or self.original_image is None:
            return
        x, y = self._canvas_to_pixel(event)
        x0, y0 = self.drag_start
        self.selection_box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        width = self.selection_box[2] - self.selection_box[0]
        height = self.selection_box[3] - self.selection_box[1]
        self.crop_status_var.set(f"正在框选：{width}×{height}px（松开鼠标后可保存）")
        self._render_crop_preview(self.selection_box)
        self._render_magnifier(*self._magnifier_pixel(event))
        self._render_main_preview()

    def _on_drag_end(self, event) -> None:
        if not self.drag_start or self.original_image is None:
            return
        self._on_drag_motion(event)
        self.drag_start = None
        if not self.selection_box:
            return
        x1, y1, x2, y2 = self.selection_box
        if x2 - x1 < 4 or y2 - y1 < 4:
            self.selection_box = None
            self.save_button.config(state=tk.DISABLED)
            self.crop_status_var.set("框选区域至少需要 4×4 个原始像素，请重新拖动")
            self._render_main_preview()
            return
        self.save_button.config(state=tk.NORMAL)
        self.crop_status_var.set(
            f"新特征框：x={x1}, y={y1}, w={x2 - x1}, h={y2 - y1}；确认后点击“保存新特征框”"
        )
        self._render_crop_preview(self.selection_box)

    def _magnifier_pixel(self, event) -> Tuple[int, int]:
        """取得放大镜中心像素；与拖框边界坐标分开，最大值为尺寸减一。"""
        x, y = self._canvas_to_pixel(event)
        if self.original_image:
            x = min(max(0, x), self.original_image.width - 1)
            y = min(max(0, y), self.original_image.height - 1)
        return x, y

    def _on_mouse_hover(self, event) -> None:
        if self.original_image is None:
            return
        self._render_magnifier(*self._magnifier_pixel(event))

    def _render_magnifier(self, center_x: int, center_y: int) -> None:
        """实时显示鼠标附近 17×17 个原始像素，并绘制网格及中心准星。"""
        if self.original_image is None:
            return
        center_x = min(max(0, int(center_x)), self.original_image.width - 1)
        center_y = min(max(0, int(center_y)), self.original_image.height - 1)
        radius = 8
        source = Image.new("RGBA", (17, 17), (18, 18, 22, 255))
        x1, y1 = center_x - radius, center_y - radius
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2 = min(self.original_image.width, center_x + radius + 1)
        sy2 = min(self.original_image.height, center_y + radius + 1)
        if sx2 > sx1 and sy2 > sy1:
            source.paste(
                self.original_image.crop((sx1, sy1, sx2, sy2)),
                (sx1 - x1, sy1 - y1),
            )
        pixel = self.original_image.getpixel((center_x, center_y))
        display = _checkerboard(source, cell=1).resize((170, 170), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(display)
        for index in range(1, 17):
            pos = index * 10
            draw.line((pos, 0, pos, 169), fill=(52, 54, 62), width=1)
            draw.line((0, pos, 169, pos), fill=(52, 54, 62), width=1)
        draw.rectangle((80, 80, 89, 89), outline=(255, 234, 0), width=2)
        draw.line((85, 0, 85, 77), fill=(255, 234, 0), width=1)
        draw.line((85, 92, 85, 169), fill=(255, 234, 0), width=1)
        draw.line((0, 85, 77, 85), fill=(255, 234, 0), width=1)
        draw.line((92, 85, 169, 85), fill=(255, 234, 0), width=1)
        self._magnifier_photo = ImageTk.PhotoImage(display)
        self.magnifier_canvas.delete("all")
        self.magnifier_canvas.create_image(0, 0, image=self._magnifier_photo, anchor=tk.NW)
        self.magnifier_info_var.set(
            f"像素: ({center_x}, {center_y})\n"
            f"RGBA: {tuple(int(value) for value in pixel)}\n"
            "黄框为鼠标所指像素"
        )

    def _render_crop_preview(self, box: Optional[Tuple[int, int, int, int]]) -> None:
        self.crop_canvas.delete("all")
        if self.original_image is None or box is None:
            self._crop_photo = None
            return
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return
        crop = _checkerboard(self.original_image.crop(box), cell=4)
        scale = max(1, min(16, 210 // max(1, crop.width), 135 // max(1, crop.height)))
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
        self._crop_photo = ImageTk.PhotoImage(crop)
        self.crop_canvas.create_image(110, 72, image=self._crop_photo, anchor=tk.CENTER)
        self.crop_canvas.create_text(
            4, 4, text=f"{x2 - x1}×{y2 - y1}px  放大 {scale}×",
            fill="#ffd54f", font=("Consolas", 8), anchor=tk.NW,
        )

    def _save_feature_crop(self) -> None:
        if not self.current_pair or self.original_image is None or not self.selection_box:
            return
        x1, y1, x2, y2 = self.selection_box
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        feature_path = self.current_pair.feature_path
        if not feature_path:
            if not self.current_pair.global_path:
                return
            filename = os.path.basename(self.current_pair.global_path).replace("mob_global", "mob", 1)
            feature_path = os.path.join(self.template_dir, filename)
        temp_path = feature_path + ".tmp.png"
        try:
            os.makedirs(self.template_dir, exist_ok=True)
            self.original_image.crop((x1, y1, x2, y2)).save(temp_path, format="PNG")
            os.replace(temp_path, feature_path)
        except OSError as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            messagebox.showerror("保存失败", f"无法写入局部特征图：\n{exc}", parent=self)
            return
        key = self.current_pair.key
        self._notify_changed("保存")
        self._refresh_pairs(select_key=key)
        messagebox.showinfo(
            "已保存",
            f"已更新 {self.current_pair.label} 的局部特征图：\n"
            f"{x2 - x1}×{y2 - y1}px，运行中的怪物检测器已重新载入。",
            parent=self,
        )

    def _delete_current_pair(self) -> None:
        pair = self.current_pair
        if not pair:
            return
        paths = [path for path in (pair.global_path, pair.feature_path) if path and os.path.exists(path)]
        if not paths:
            return
        if not messagebox.askyesno(
            "确认删除模板",
            f"确定删除“{pair.label}”的 {len(paths)} 张模板图吗？\n\n"
            "该操作会减少此怪物可用于识别的动作/朝向。",
            icon=messagebox.WARNING,
            parent=self,
        ):
            return
        errors = []
        for path in paths:
            try:
                os.remove(path)
            except OSError as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")
        if errors:
            messagebox.showerror("删除不完整", "\n".join(errors), parent=self)
        self.current_pair = None
        self.original_image = None
        self.selection_box = None
        self.existing_box = None
        self._notify_changed("删除")
        self._refresh_pairs()

    def _notify_changed(self, action: str) -> None:
        if self.on_changed:
            try:
                self.on_changed(action)
            except Exception as exc:
                messagebox.showwarning(
                    "模板已修改",
                    f"文件修改成功，但运行时重新载入失败：\n{exc}",
                    parent=self,
                )
