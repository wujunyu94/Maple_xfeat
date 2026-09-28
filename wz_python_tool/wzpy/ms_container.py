"""Reading ``data/Packs/*.ms`` as MapleStory **WZ Mob/Skill trees**.

These ``.ms`` files (see ``docs/ms_format_findings.md``) are *not* a SNOW2
``.ms`` archive. They are a custom container holding a
**WZ-encoded property tree** (the standard MapleStory WZ string cipher —
``cipher[i] ^ (0xAA + i)`` — plus WZ property tags ``0x09``/``0x73``…) describing
mobs, their animations, frames and parts, with two kinds of leaf art:

* **Canvas mobs** — per-frame 1×1 placeholder canvases with ``_outlink`` /
  ``UOL`` references like ``Mob/_Canvas/<id>.img/<anim>/<frame>`` (exactly the
  ``_outlink`` model the Character packs use).
* **Spine mobs** — embedded Spine 2.1.27 binary skeletons (parsed by
  :mod:`wzpy.ms_spine`).

This module recovers the full *logical structure* of every file by scanning the
WZ-encoded strings (fast: it searches for the encoded category prefix rather
than walking every byte) and stitches it together with the Spine skeletons, so
the whole container is viewable: every mob id, its animations, frame indices and
part names, plus skeleton bones/animations.

Not covered (documented as open in the findings): the per-img indirect-offset
base and the canvas *pixel* compression (an unidentified non-zlib codec), so raw
bitmap decoding is out of scope here — this exposes structure + skeletons.

Usage::

    from wzpy.ms_container import MsContainer
    c = MsContainer.open("data/Packs/Mob_00000.ms")
    print(c.summary())
    for mid in sorted(c.mobs)[:5]:
        print(mid, {a: sorted(f) for a, f in c.mobs[mid].items()})
    for sk in c.skeletons:            # Spine mobs
        print(sk.hash, sk.animations)
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .ms_spine import MsSpineContainer, SkeletonData

# WZ ASCII string cipher mask byte for index i is (0xAA + i) & 0xFF.
# Any "<Category.../>_Canvas/<img>.img/<subpath>" reference (mob frames, boss
# patterns, skill icons, …). The img stem groups the tree; the subpath is the
# animation/structure path beneath it.
_PATH_RE = re.compile(r"^(.*?)/_Canvas/([^/]+)\.img/(.+)$")


def _enc_prefix(s: str) -> bytes:
    """Encode the first chars of a WZ ASCII string (index 0..) for fast search."""
    return bytes((ord(c) ^ ((0xAA + i) & 0xFF)) & 0xFF for i, c in enumerate(s))


def _decode_ascii_at(data, content_off: int, sign_off: int) -> Optional[str]:
    """Decode a WZ ASCII string whose signed length byte is at ``sign_off`` and
    whose payload starts at ``content_off``. Returns None if not a plausible
    printable string."""
    sb = data[sign_off]
    if not (0x80 <= sb <= 0xFF):
        return None
    n = 256 - sb
    if content_off + n > len(data):
        return None
    out = bytearray(n)
    for i in range(n):
        out[i] = data[content_off + i] ^ ((0xAA + i) & 0xFF)
    if any(c < 32 or c >= 127 for c in out):
        return None
    return out.decode("ascii")


from .wz_image import WzImage  # noqa: E402


class _PackImage(WzImage):
    """A synthetic ``.img`` whose property tree is built on first parse from a
    set of ``<anim>/<frame>/<part>`` subpaths (canvas mobs). Leaves are string
    placeholders — the canvas pixels aren't decodable (unidentified codec)."""

    def __init__(self, name: str, parent, subpaths: Set[str]):
        super().__init__(name=name, parent=parent, offset=0, size=0, wz_file=None)
        self._subpaths = subpaths

    def parse(self):
        if self._parsed and self._root is not None:
            return self._root
        from .properties import WzSubProperty, WzStringProperty
        root = WzSubProperty(self.name)
        for sub in sorted(self._subpaths):
            segs = [s for s in sub.split("/") if s]
            node = root
            for seg in segs[:-1]:
                child = node.child(seg)
                if not isinstance(child, WzSubProperty):
                    child = WzSubProperty(seg)
                    node.add(child)
                node = child
            if segs and node.child(segs[-1]) is None:
                node.add(WzStringProperty(segs[-1], "canvas"))
        self._root = root
        self._parsed = True
        return root


@dataclass
class MsContainer:
    path: str
    category: str = ""                       # "Mob", "Skill", ...
    imgs: Dict[str, Set[str]] = field(default_factory=dict)   # img stem -> subpaths
    skeletons: List[SkeletonData] = field(default_factory=list)
    path_count: int = 0
    region: str = "BMS"                       # WZ string cipher is the BMS 0xAA+i mask
    version: Optional[int] = None             # n/a for these packs; kept for API parity
    _spine: Optional[MsSpineContainer] = None
    _root: object = None                      # lazily-built synthetic WzDirectory tree
    _skill_imgs: Optional[dict] = None        # lazily-parsed stat trees (Skill_*)

    # mob id -> {animation -> {frame}} (the simple <id>.img/<anim>/<frame> view,
    # a convenience projection of ``imgs`` for plain numeric-id mobs).
    @property
    def mobs(self) -> Dict[str, Dict[str, Set[str]]]:
        out: Dict[str, Dict[str, Set[str]]] = {}
        for img, subs in self.imgs.items():
            if not img.isdigit():
                continue
            anims: Dict[str, Set[str]] = {}
            for sub in subs:
                parts = sub.split("/")
                if len(parts) >= 2:
                    anims.setdefault(parts[0], set()).add(parts[1])
            if anims:
                out[img] = anims
        return out

    # ── lifecycle ───────────────────────────────────────────────────────
    @classmethod
    def open(cls, path: str) -> "MsContainer":
        c = cls(path=path)
        stem = os.path.basename(path)
        c.category = stem.split("_")[0] if "_" in stem else os.path.splitext(stem)[0]
        with open(path, "rb") as f:
            data = f.read()
        c._scan_paths(data)
        c._spine = MsSpineContainer.open(path)
        c.skeletons = [e.skeleton for e in c._spine.entries if e.skeleton]
        return c

    def close(self) -> None:
        if self._spine is not None:
            self._spine.close()
            self._spine = None

    def __enter__(self) -> "MsContainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── property/stat trees (Skill_*) ────────────────────────────────────
    def skill_imgs(self) -> Dict[str, object]:
        """Parsed per-job skill stat trees ``{stem: WzSubProperty}`` recovered
        from the body (``cooltime``, ``lt``/``rb``, ``mpCon``, ``damage``,
        ``level/<n>/…``, …). Empty for non-``Skill`` packs. Parsed once and
        cached; see :mod:`wzpy.ms_wz`."""
        if self._skill_imgs is None:
            self._skill_imgs = {}
            if self.category == "Skill":
                from .ms_wz import parse_skill_imgs
                try:
                    with open(self.path, "rb") as f:
                        self._skill_imgs = parse_skill_imgs(f.read())
                except Exception:
                    self._skill_imgs = {}
        return self._skill_imgs

    # ── synthetic WZ tree (for the web tree browser) ─────────────────────
    @property
    def root(self):
        """A synthetic :class:`~wzpy.wz_file.WzDirectory` tree mirroring the
        recovered structure, so the web browser (which walks
        ``WzDirectory`` → ``WzImage`` → ``WzProperty``) can navigate it:

            <Category>/_Canvas/<img>.img/<anim>/<frame>/<part>   (canvas mobs)
            _Spine/<NNN>_<hash>.img/{info,bones,slots,animations}

        Built lazily and cached; per-img property trees are themselves built
        on first parse so listing thousands of imgs stays cheap. Canvas leaves
        carry no pixels (the canvas codec is unidentified — see
        ``docs/ms_format_findings.md``); this is a structure view."""
        if self._root is None:
            self._root = self._build_tree()
        return self._root

    def _build_tree(self):
        from .wz_file import WzDirectory
        from .wz_image import WzImage
        from .properties import (
            WzSubProperty, WzStringProperty, WzIntProperty, WzNullProperty,
        )

        root = WzDirectory(name="")
        cat = WzDirectory(self.category or "Pack", parent=root)
        root.subdirs[cat.name] = cat
        canvas_dir = WzDirectory("_Canvas", parent=cat)
        cat.subdirs["_Canvas"] = canvas_dir
        for stem, subs in self.imgs.items():
            img = _PackImage(stem + ".img", canvas_dir, subs)
            canvas_dir.images[img.name] = img

        # Stat/property trees (Skill_*): one navigable <stem>.img per job group,
        # sitting beside _Canvas exactly like a real Skill.wz directory.
        for stem, tree in self.skill_imgs().items():
            wi = WzImage(name=stem + ".img", parent=cat, offset=0, size=0,
                         wz_file=None)
            wi._root = tree
            wi._parsed = True
            cat.images[wi.name] = wi

        if self.skeletons:
            spine = WzDirectory("_Spine", parent=root)
            root.subdirs["_Spine"] = spine
            for i, sk in enumerate(self.skeletons):
                safe = re.sub(r"[^A-Za-z0-9]", "", sk.hash or "skel")[:16] or "skel"
                name = f"{i:03d}_{safe}.img"
                img = WzImage(name=name, parent=spine, offset=0, size=0, wz_file=None)
                r = WzSubProperty(name)
                info = WzSubProperty("info"); r.add(info)
                info.add(WzStringProperty("version", sk.version or ""))
                info.add(WzIntProperty("width", int(sk.width)))
                info.add(WzIntProperty("height", int(sk.height)))
                info.add(WzStringProperty("hash", sk.hash or ""))
                bones = WzSubProperty("bones"); r.add(bones)
                for b in sk.bones:
                    bones.add(WzNullProperty(b))
                slots = WzSubProperty("slots"); r.add(slots)
                for sn, att in sk.slots:
                    slots.add(WzStringProperty(sn, att or ""))
                anims = WzSubProperty("animations"); r.add(anims)
                for a in sk.animations:
                    anims.add(WzNullProperty(a))
                img._root = r
                img._parsed = True
                spine.images[img.name] = img
        return root

    # ── scanning ────────────────────────────────────────────────────────
    def _scan_paths(self, data: bytes) -> None:
        """Reconstruct ``img -> {subpath}`` from the WZ ``_Canvas`` UOL/_outlink
        paths. Searches for the encoded ``<Category>/`` prefix so it only has to
        decode at real path sites (fast even on 100 MB files)."""
        imgs: Dict[str, Set[str]] = defaultdict(set)
        seen = 0
        # The category in a path may differ from the file stem; search every
        # prefix we plausibly expect.
        prefixes = {self.category, "Mob", "Skill", "Npc", "Reactor", "Effect"}
        offsets: Set[int] = set()
        for pfx in prefixes:
            enc = _enc_prefix(pfx + "/")
            for m in re.finditer(re.escape(enc), data):
                offsets.add(m.start())
        for content_off in offsets:
            s = _decode_ascii_at(data, content_off, content_off - 1)
            if not s:
                continue
            seen += 1
            pm = _PATH_RE.match(s)
            if pm:
                imgs[pm.group(2)].add(pm.group(3))
        self.path_count = seen
        self.imgs = {k: v for k, v in imgs.items()}

    # ── reporting ───────────────────────────────────────────────────────
    def summary(self) -> str:
        n_imgs = len(self.imgs)
        n_spine = len(self.skeletons)
        total_paths = sum(len(s) for s in self.imgs.values())
        # top-level segment under each img (animation / structure label)
        top_labels = sorted({sub.split("/")[0]
                             for subs in self.imgs.values() for sub in subs})
        lines = [
            f"{os.path.basename(self.path)}  ({self.category})",
            f"  imgs (_Canvas): {n_imgs}",
            f"  spine mobs    : {n_spine}",
            f"  uol paths     : {self.path_count}",
            f"  canvas refs   : {total_paths}",
            f"  top segments  : {', '.join(top_labels[:20])}"
            + (" …" if len(top_labels) > 20 else ""),
        ]
        if self.skeletons:
            sk = self.skeletons[0]
            lines.append(
                f"  e.g. skeleton : {sk.width:.0f}x{sk.height:.0f}, "
                f"{len(sk.bones)} bones, anims {sk.animations[:6]}"
            )
        return "\n".join(lines)
