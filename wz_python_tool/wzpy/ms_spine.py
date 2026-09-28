"""Reading ``data/Packs/*.ms`` — Spine skeleton asset containers.

These ``.ms`` files are **not** the MapleLib/Elem8100 Snow2 ``.ms`` archive
(that SNOW2 format is a different container from these files). They are an
uncompressed,
unencrypted asset container holding **Spine 2.1.x binary skeletons** (plus
texture/atlas data) for a Spine-animation-based MapleStory client. See
``docs/ms_format_findings.md`` for how this was established.

Container layout (reverse-engineered from ``Mob_*.ms`` / ``Skill_*.ms``):

* Each skeleton entry is framed by a fixed 17-byte signature
  ``56 c3 ce 11 bf 01 00 aa 00 55 59 5a 12 01 00 01 00`` followed by the
  skeleton byte length (u32 little-endian, stored twice), ``01 00`` and
  ``08 00 00 00``; the Spine binary skeleton then begins immediately.
* The Spine skeleton is standard **Spine 2.1.x binary** (big-endian floats,
  ``[len+1]``-prefixed UTF-8 strings, optimize-positive varints) — the format
  read by ``spine-runtimes-2.1.25/spine-csharp/src/SkeletonBinary.cs`` (bundled
  with MapleLib). :func:`read_skeleton` is a faithful Python port of
  ``ReadSkeletonData``.
* Texture/atlas blobs occupy the space between skeletons; decoding them is out
  of scope here (this module recovers the skeleton structure and lets you slice
  out the raw skeleton bytes for an external Spine runtime).

Usage::

    from wzpy.ms_spine import MsSpineContainer
    c = MsSpineContainer.open("data/Packs/Mob_00000.ms")
    print(len(c.entries), "skeletons")
    sk = c.entries[0].skeleton
    print(sk.version, sk.width, sk.height, sk.bones[:5])
    print([a for a in sk.animations])          # animation names
    raw = c.raw_skeleton_bytes(c.entries[0])   # feed to a Spine 2.1 runtime
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── container framing ───────────────────────────────────────────────────
# 17-byte signature that precedes the skeleton-length field of every entry.
_ENTRY_SIG = bytes.fromhex("56c3ce11bf0100aa0055595a12010001 00".replace(" ", ""))
# Offset from the signature start to the first Spine skeleton byte:
#   17 (sig) + 4 (len) + 4 (len repeat) + 2 (01 00) + 4 (08 00 00 00)
_SKELETON_OFFSET = 17 + 4 + 4 + 2 + 4
# The u32 skeleton length sits right after the 17-byte signature.
_LEN_OFFSET = 17

# ── Spine 2.1 binary constants (SkeletonBinary.cs) ──────────────────────
_ATTACH_REGION, _ATTACH_BOUNDINGBOX, _ATTACH_MESH, _ATTACH_SKINNEDMESH = 0, 1, 2, 3
_TL_SCALE, _TL_ROTATE, _TL_TRANSLATE, _TL_ATTACHMENT, _TL_COLOR, _TL_FLIPX, _TL_FLIPY = range(7)
_CURVE_STEPPED, _CURVE_BEZIER = 1, 2


# ── parsed model ────────────────────────────────────────────────────────
@dataclass
class SkeletonData:
    hash: Optional[str]
    version: Optional[str]
    width: float
    height: float
    images_path: Optional[str]
    bones: List[str] = field(default_factory=list)
    slots: List[Tuple[str, Optional[str]]] = field(default_factory=list)  # (name, attachment)
    skin_names: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    animations: List[str] = field(default_factory=list)


@dataclass
class MsSpineEntry:
    offset: int          # file offset of the entry signature
    skel_offset: int     # file offset of the first Spine skeleton byte
    size: int            # declared skeleton byte length
    skeleton: Optional[SkeletonData] = None
    error: Optional[str] = None


# ── Spine binary primitive reader ───────────────────────────────────────
class _Reader:
    __slots__ = ("b", "o")

    def __init__(self, b: bytes, o: int = 0):
        self.b = b
        self.o = o

    def byte(self) -> int:
        v = self.b[self.o]
        self.o += 1
        return v

    def boolean(self) -> bool:
        return self.byte() != 0

    def sbyte(self) -> int:
        v = self.byte()
        return v - 256 if v >= 128 else v

    def f(self) -> float:
        v = struct.unpack_from(">f", self.b, self.o)[0]
        self.o += 4
        return v

    def i32(self) -> int:
        v = struct.unpack_from(">i", self.b, self.o)[0]
        self.o += 4
        return v

    def vint(self, optimize_positive: bool = True) -> int:
        b = self.byte()
        r = b & 0x7F
        if b & 0x80:
            b = self.byte()
            r |= (b & 0x7F) << 7
            if b & 0x80:
                b = self.byte()
                r |= (b & 0x7F) << 14
                if b & 0x80:
                    b = self.byte()
                    r |= (b & 0x7F) << 21
                    if b & 0x80:
                        b = self.byte()
                        r |= (b & 0x7F) << 28
        r &= 0xFFFFFFFF
        if optimize_positive:
            return r
        return (r >> 1) ^ -(r & 1)

    def string(self) -> Optional[str]:
        n = self.vint()
        if n == 0:
            return None
        if n == 1:
            return ""
        s = self.b[self.o:self.o + n - 1].decode("utf-8", "replace")
        self.o += n - 1
        return s

    def float_array(self) -> int:
        n = self.vint()
        self.o += 4 * n
        return n

    def short_array(self) -> int:
        n = self.vint()
        self.o += 2 * n
        return n

    def int_array(self) -> int:
        n = self.vint()
        for _ in range(n):
            self.vint()
        return n

    def curve(self, frame_index: int, frame_count: int) -> None:
        """ReadCurve — only on non-final frames (caller guards frame index)."""
        if frame_index >= frame_count - 1:
            return
        t = self.byte()
        if t == _CURVE_BEZIER:
            self.o += 16  # 4 floats


# ── skeleton parse (port of SkeletonBinary.ReadSkeletonData) ────────────
def read_skeleton(r: _Reader) -> SkeletonData:
    hash_ = r.string()
    version = r.string()
    width = r.f()
    height = r.f()
    nonessential = r.boolean()
    images_path = r.string() if nonessential else None

    sk = SkeletonData(hash=hash_, version=version, width=width,
                      height=height, images_path=images_path)

    # Bones.
    n = r.vint()
    for i in range(n):
        sk.bones.append(r.string())
        r.vint()                       # parent index (+1)
        r.o += 4 * 6                   # x, y, scaleX, scaleY, rotation, length
        r.o += 4                       # flipX, flipY, inheritScale, inheritRotation
        if nonessential:
            r.i32()                    # bone color

    # IK constraints.
    n = r.vint()
    ik_count = n
    for _ in range(n):
        r.string()
        for _ in range(r.vint()):
            r.vint()                   # bone indices
        r.vint()                       # target
        r.f()                          # mix
        r.byte()                       # bend direction (sbyte)

    # Slots.
    n = r.vint()
    for _ in range(n):
        name = r.string()
        r.vint()                       # bone index
        r.i32()                        # color
        attachment = r.string()
        r.boolean()                    # additive blending
        sk.slots.append((name, attachment))
    slot_count = n

    # Default skin + named skins.
    _read_skin(r, nonessential)
    n = r.vint()
    for _ in range(n):
        sk.skin_names.append(r.string())
        _read_skin(r, nonessential)

    # Events.
    n = r.vint()
    for _ in range(n):
        sk.events.append(r.string())
        r.vint(False)                  # int
        r.f()                          # float
        r.string()                     # string

    # Animations.
    n = r.vint()
    for _ in range(n):
        name = r.string()
        sk.animations.append(name)
        _read_animation(r, slot_count, ik_count)

    return sk


def _read_skin(r: _Reader, nonessential: bool) -> None:
    slot_count = r.vint()
    if slot_count == 0:
        return
    for _ in range(slot_count):
        r.vint()                       # slot index
        for _ in range(r.vint()):
            name = r.string()
            _read_attachment(r, name, nonessential)


def _read_attachment(r: _Reader, attachment_name: Optional[str],
                     nonessential: bool) -> None:
    name = r.string()
    if name is None:
        name = attachment_name
    atype = r.byte()
    if atype == _ATTACH_REGION:
        r.string()                     # path
        r.o += 4 * 7                   # x,y,scaleX,scaleY,rotation,width,height
        r.i32()                        # color
    elif atype == _ATTACH_BOUNDINGBOX:
        r.float_array()                # vertices
    elif atype == _ATTACH_MESH:
        r.string()                     # path
        r.float_array()                # region UVs
        r.short_array()                # triangles
        r.float_array()                # vertices
        r.i32()                        # color
        r.vint()                       # hull length
        if nonessential:
            r.int_array()              # edges
            r.o += 8                   # width, height
    elif atype == _ATTACH_SKINNEDMESH:
        r.string()                     # path
        r.float_array()                # region UVs
        r.short_array()                # triangles
        vertex_count = r.vint()
        i = 0
        while i < vertex_count:
            bone_count = int(r.f())
            nn = i + bone_count * 4
            while i < nn:
                r.o += 16              # boneIndex + 3 weights (4 floats)
                i += 4
            i += 1
        r.i32()                        # color
        r.vint()                       # hull length
        if nonessential:
            r.int_array()              # edges
            r.o += 8                   # width, height
    # else: unknown attachment type -> nothing (matches C# default null)


def _read_animation(r: _Reader, slot_count: int, ik_count: int) -> None:
    # Slot timelines.
    for _ in range(r.vint()):
        r.vint()                       # slot index
        for _ in range(r.vint()):
            tl = r.byte()
            fc = r.vint()
            if tl == _TL_COLOR:
                for fi in range(fc):
                    r.f(); r.i32()
                    r.curve(fi, fc)
            elif tl == _TL_ATTACHMENT:
                for _ in range(fc):
                    r.f(); r.string()

    # Bone timelines.
    for _ in range(r.vint()):
        r.vint()                       # bone index
        for _ in range(r.vint()):
            tl = r.byte()
            fc = r.vint()
            if tl == _TL_ROTATE:
                for fi in range(fc):
                    r.f(); r.f()
                    r.curve(fi, fc)
            elif tl in (_TL_TRANSLATE, _TL_SCALE):
                for fi in range(fc):
                    r.f(); r.f(); r.f()
                    r.curve(fi, fc)
            elif tl in (_TL_FLIPX, _TL_FLIPY):
                for _ in range(fc):
                    r.f(); r.boolean()

    # IK timelines.
    for _ in range(r.vint()):
        r.vint()                       # ik index
        fc = r.vint()
        for fi in range(fc):
            r.f(); r.f(); r.byte()     # time, mix, bend (sbyte)
            r.curve(fi, fc)

    # FFD timelines.
    for _ in range(r.vint()):
        r.vint()                       # skin index
        for _ in range(r.vint()):
            r.vint()                   # slot index
            for _ in range(r.vint()):
                r.string()             # attachment name
                fc = r.vint()
                for fi in range(fc):
                    r.f()              # time
                    end = r.vint()
                    if end != 0:
                        r.vint()       # start
                        r.o += 4 * end  # vertices
                    r.curve(fi, fc)

    # Draw order timeline.
    draw_order_count = r.vint()
    for _ in range(draw_order_count):
        offset_count = r.vint()
        for _ in range(offset_count):
            r.vint()                   # slot index
            r.vint()                   # offset
        r.f()                          # time

    # Event timeline.
    event_count = r.vint()
    for _ in range(event_count):
        r.f()                          # time
        r.vint()                       # event index
        r.vint(False)                  # int
        r.f()                          # float
        if r.boolean():
            r.string()                 # string (if present)


# ── container ───────────────────────────────────────────────────────────
class MsSpineContainer:
    """A parsed ``Mob_*.ms`` / ``Skill_*.ms`` Spine-skeleton container.

    :attr:`entries` is the list of :class:`MsSpineEntry` (one per Spine
    skeleton found). Each entry's ``skeleton`` is the parsed
    :class:`SkeletonData` (or ``error`` is set if that skeleton failed to
    parse, which never aborts the whole container).
    """

    def __init__(self, path: str):
        self.path = path
        self.data = b""           # mmap (bytes-like) once opened
        self.entries: List[MsSpineEntry] = []
        self._fp = None
        self._mmap = None

    @classmethod
    def open(cls, path: str, *, parse: bool = True) -> "MsSpineContainer":
        import mmap
        c = cls(path)
        c._fp = open(path, "rb")
        c._mmap = mmap.mmap(c._fp.fileno(), 0, access=mmap.ACCESS_READ)
        c.data = c._mmap
        c._scan(parse=parse)
        return c

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        self.data = b""

    def __enter__(self) -> "MsSpineContainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _scan(self, *, parse: bool) -> None:
        data = self.data
        for m in re.finditer(re.escape(_ENTRY_SIG), data):
            sig = m.start()
            size = int.from_bytes(data[sig + _LEN_OFFSET:sig + _LEN_OFFSET + 4], "little")
            skel_off = sig + _SKELETON_OFFSET
            entry = MsSpineEntry(offset=sig, skel_offset=skel_off, size=size)
            if parse:
                try:
                    entry.skeleton = read_skeleton(_Reader(data, skel_off))
                except Exception as e:  # never let one bad skeleton kill the scan
                    entry.error = f"{type(e).__name__}: {e}"
            self.entries.append(entry)

    def raw_skeleton_bytes(self, entry: MsSpineEntry) -> bytes:
        """The raw Spine 2.1 binary skeleton bytes for ``entry`` (``entry.size``
        bytes from ``entry.skel_offset``) — ready to feed to a Spine 2.1
        runtime's ``SkeletonBinary``."""
        return self.data[entry.skel_offset:entry.skel_offset + entry.size]


def is_ms_spine_path(path: str) -> bool:
    """True if ``path`` is a Spine-skeleton ``.ms`` container (contains at least
    one entry signature). Uses an mmap scan so it doesn't load the whole file."""
    import os
    import mmap
    if not (os.path.isfile(path) and path.lower().endswith(".ms")):
        return False
    try:
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm.find(_ENTRY_SIG) != -1
    except (OSError, ValueError):
        return False
