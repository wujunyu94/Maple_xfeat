"""Render static WZ tiles/objects in world coordinates, without miniMap data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
sys.path.insert(0, str(ROOT / "wz_python_tool"))
from wzpy.crypto import WzKey
from wzpy.wz_image import WzImage
from wzpy.canvas import decode_canvas
from .wz_adapter import install
install()


def value(node, key, default=None):
    child = node.child(key) if node is not None else None
    return child.value if child is not None else default


def children(node):
    return node.children() if node is not None else []


class Atlas:
    def __init__(self, rgba, meta):
        self.rgba = rgba
        self.meta = meta
        self.origin = np.array(meta["origin"], dtype=np.float32)
        self.bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        self.mask = np.uint8(rgba[:, :, 3] > 240) * 255


class Builder:
    def __init__(self, map_root=ROOT / "Map", iv="4D23C72B"):
        self.map_root = Path(map_root).resolve()
        self.key = WzKey(bytes.fromhex(iv))
        self.iv = iv
        self.roots = {}
        self.sprites = {}
        self.sources = {}
        self.warnings = []

    def read(self, path):
        path = Path(path).resolve()
        if path not in self.roots:
            im = WzImage.from_bytes(path.read_bytes(), key=self.key, name=path.name)
            self.roots[path] = im.parse()
            st = path.stat()
            self.sources[str(path.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
            self.warnings.extend(f"{path.name}: {w}" for w in im.parse_warnings)
        return self.roots[path]

    def resolve(self, node, root, depth=0):
        if node is None or depth > 16:
            raise ValueError("missing/cyclic sprite reference")
        if node.type_name == "UOL":
            return self.resolve(node.parent.get(node.value), root, depth + 1)
        if node.type_name != "Canvas":
            return self.resolve(node.child("0"), root, depth + 1)
        link = value(node, "_inlink")
        if link:
            return self.resolve(root.get(link), root, depth + 1)
        link = value(node, "_outlink")
        if link:
            relative, inside = link.split(".img/", 1)
            parts = relative.replace("\\", "/").split("/")
            if parts[0] in ("Map", "Map.wz"):
                parts = parts[1:]
            other = self.read(self.map_root.joinpath(*parts).with_suffix(".img"))
            return self.resolve(other.get(inside), other, depth + 1)
        return node

    def sprite(self, path, key):
        cache_key = (str(path), key)
        if cache_key not in self.sprites:
            root = self.read(path)
            original = root.get(key)
            node = self.resolve(original, root)
            # Extracted assets use the GMS pixel encryption key.
            image = decode_canvas(node).convert("RGBA")
            origin = value(original, "origin", value(node, "origin", (0, 0)))
            self.sprites[cache_key] = (image, origin, value(node, "z", 0))
        return self.sprites[cache_key]

    def build(self, map_id):
        name = f"{int(map_id):09d}.img"
        paths = [self.map_root / "Map" / f"Map{name[0]}" / name,
                 self.map_root / f"Map{name[0]}" / name, self.map_root / name]
        root = self.read(next((p for p in paths if p.exists()), paths[0]))
        linked = value(root.child("info"), "link")
        if linked is not None:
            raise ValueError(f"Map {map_id} links to {linked}; load the linked map ID explicitly")
        draws = []
        for layer_id in range(8):
            layer = root.child(str(layer_id))
            if layer is None:
                continue
            tile_set = value(layer.child("info"), "tS", "")
            for kind in ("obj", "tile"):
                for item in children(layer.child(kind)):
                    if kind == "tile":
                        path = self.map_root / "Tile" / f"{tile_set}.img"
                        key = f"{value(item, 'u')}/{value(item, 'no')}"
                    else:
                        path = self.map_root / "Obj" / f"{value(item, 'oS')}.img"
                        key = "/".join(str(value(item, k)) for k in ("l0", "l1", "l2"))
                    try:
                        sprite, (ox, oy), z = self.sprite(path, key)
                        if value(item, "f", 0):
                            sprite = ImageOps.mirror(sprite)
                            ox = sprite.width - ox
                        x, y = int(value(item, "x", 0) - ox), int(value(item, "y", 0) - oy)
                        draws.append((layer_id, int(value(item, "z", z)), len(draws), x, y, sprite))
                    except Exception as exc:
                        self.warnings.append(f"{path.name}/{key}: {exc}")
        if not draws:
            raise ValueError("No renderable static objects/tiles in this map")
        left = min(d[3] for d in draws)
        top = min(d[4] for d in draws)
        right = max(d[3] + d[5].width for d in draws)
        bottom = max(d[4] + d[5].height for d in draws)
        if (right-left) * (bottom-top) > 80_000_000:
            raise ValueError("Map exceeds the experiment's 80 megapixel memory limit")
        canvas = Image.new("RGBA", (right-left, bottom-top))
        for _, _, _, x, y, sprite in sorted(draws, key=lambda d: d[:3]):
            canvas.alpha_composite(sprite, (x-left, y-top))
        # Geometry is used only for visualization, never as a minimap-derived pose.
        footholds = []
        def visit(node):
            if node is not None and node.child("x1") is not None:
                footholds.append([value(node, k) for k in ("x1", "y1", "x2", "y2")])
            else:
                for child in children(node):
                    visit(child)
        visit(root.child("foothold"))
        ladders = [[value(n, k) for k in ("x", "y1", "y2")] for n in children(root.child("ladderRope"))]
        ladder_nodes = [dict(id=str(n.name), kind="ladder" if value(n, "l", 1) else "rope",
                             x=value(n, "x"), y1=value(n, "y1"), y2=value(n, "y2"),
                             upper_exit=bool(value(n, "uf", 0)), page=value(n, "page"))
                        for n in children(root.child("ladderRope"))]
        meta = dict(version=3, map_id=int(map_id), origin=[left, top], sources=self.sources,
                    iv=self.iv, objects=len(draws),
                    warnings=self.warnings, footholds=footholds, ladders=ladders, ladder_nodes=ladder_nodes)
        return Atlas(np.asarray(canvas), meta)


def load_atlas(map_id, rebuild=False, progress=print):
    CACHE.mkdir(exist_ok=True)
    png, metadata = CACHE / f"{map_id}.png", CACHE / f"{map_id}.json"
    if not rebuild and png.exists() and metadata.exists():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        valid = meta.get("version") == 3
        for path, signature in meta.get("sources", {}).items():
            p = ROOT / path
            valid = valid and p.exists() and [p.stat().st_size, p.stat().st_mtime_ns] == signature
        if valid:
            progress(f"地图 {map_id}：读取静态场景缓存")
            return Atlas(np.array(Image.open(png).convert("RGBA")), meta)
    progress(f"地图 {map_id}：正在解码 Tile / Obj 并拼合世界场景…")
    atlas = Builder().build(map_id)
    Image.fromarray(atlas.rgba).save(png)
    metadata.write_text(json.dumps(atlas.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"静态物件 {atlas.meta['objects']}，跳过/警告 {len(atlas.meta['warnings'])}")
    return atlas
