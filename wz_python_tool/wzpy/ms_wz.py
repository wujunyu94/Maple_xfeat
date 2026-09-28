"""Parsing the WZ **property tree** inside ``data/Packs/*.ms`` Pack containers.

Companion to :mod:`wzpy.ms_container` (which recovers the ``_Canvas`` graphics
*structure*) and :mod:`wzpy.ms_spine` (Spine skeletons). This module recovers the
**stat/property data** — for ``Skill_*.ms`` that means the per-skill blocks
(``cooltime``, ``lt``/``rb`` attack range, ``mpCon``, ``damage``, ``mobCount``,
``attackCount``, ``maxLevel``, ``level/<n>/…``, ``common/…``) as a navigable
:class:`~wzpy.properties.WzSubProperty` tree, one per job-group img.

How it works (see ``docs/ms_format_findings.md`` for the full reverse engineering):

* The container's header/TOC is encrypted (unidentified cipher) and is **not**
  decoded here. Instead we parse the **body**, which is a standard MapleStory WZ
  property serialization (same tags as :mod:`wzpy.properties`) with the BMS
  zero-key string cipher (``byte ^ (0xAA + i)``).
* **String blocks** use three markers:
    - ``0x00``/``0x73`` inline → zero-key string at the current position;
    - ``0x1b`` → a *global* type-name pool living in the encrypted header. We
      never decrypt it; the small offset uniquely identifies the extended type
      (``1``→Property, ``10``→Vector2D, ``27``→Convex2D, ``44``→Canvas, plus
      :data:`_GLOBAL_TYPE_BY_OFFSET`).
    - ``0x01`` → a *per-img* back-reference into the zero-key body: the target is
      ``img_base + offset``. ``img_base`` (= the WZ ``img.Offset``) is recovered
      per img by :func:`_solve_base` (most refs resolve to a real, recurring
      property name only at the true base).
* Skills are located by their inline 7-/8-digit id followed by a tag-9 byte, and
  grouped into imgs by ``skillId[:-4]`` (e.g. ``5201001`` → img ``520``). The
  first skill of each img has its *name* in the encrypted header, so it is the
  one entry that may be missed; every other skill parses in full.

Canvas pixels are not decoded (the codec is unidentified — same gap as
``ms_container``); canvas leaves carry their ``_outlink``/``_inlink`` refs but no
bitmap.
"""

from __future__ import annotations

import re
import struct
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .properties import (
    WzCanvasProperty,
    WzConvexProperty,
    WzDoubleProperty,
    WzFloatProperty,
    WzIntProperty,
    WzLongProperty,
    WzNullProperty,
    WzProperty,
    WzShortProperty,
    WzStringProperty,
    WzSubProperty,
    WzUolProperty,
    WzVectorProperty,
)

# ── global extended-type pool (offsets into the encrypted header) ────────────
# The .ms body stores extended-property type names once, in the encrypted
# header, referenced by a tiny constant offset via the 0x1b marker. We never
# decrypt the header; instead the offsets that recur with huge, structurally
# consistent counts across every file are mapped directly (verified by
# structural classification — see docs/ms_format_findings.md):
#   1 → Property, 44 → Canvas, 70 → Shape2D#Vector2D.
# Other offsets are file-specific (the pool layout differs per file), so any
# unmapped offset is dispatched by STRUCTURE at parse time (_dispatch_unknown):
# this keeps the parser correct without a per-file pool dump.
_GLOBAL_TYPE_BY_OFFSET = {
    1: "Property",
    44: "Canvas",
    70: "Shape2D#Vector2D",
}

# Distinctive, recurring skill property names used to anchor the per-img base.
# Short/ambiguous names (lt, rb, x, y, z, hs) are deliberately excluded — they
# produce too many coincidental byte matches to anchor on.
_VOCAB = [
    "common", "level", "info", "action", "icon", "effect", "damage", "cooltime",
    "mpCon", "hpCon", "mastery", "attackCount", "mobCount", "bulletCount",
    "bulletConsume", "range", "maxLevel", "masterLevel", "invisible", "weapon",
    "skillList", "finalAttack", "randomHitOrigin", "useZ", "summon", "affected",
    "PVPcommon", "PVPdamage", "reqLev", "combatOrders", "hyper", "elemAttr",
    "criticaldamage", "fixdamage", "rangeAttack", "areaAttack", "subProp",
    "pairskill", "psdSkill", "notRemoved", "disable", "subTime", "dotTime",
    "dotInterval", "dotType",
]


def _enc(s: str) -> bytes:
    """Zero-key WZ ascii encoding of ``s`` (for byte-pattern search)."""
    return bytes((ord(c) ^ ((0xAA + i) & 0xFF)) & 0xFF for i, c in enumerate(s))


class _BodyParser:
    """Walks the zero-key WZ property grammar over the raw ``.ms`` bytes.

    Stateless w.r.t. resolution: string blocks return either a decoded ``str``
    (inline) or a deferred ``('REF', offset)`` / ``('TYPE', name)`` tuple, and
    every 0x01 offset is recorded in :attr:`refs`. A second pass resolves the
    deferred refs against the solved ``img_base``.
    """

    def __init__(self, data: bytes, type_cache: Optional[dict] = None):
        self.d = data
        self.refs: List[Tuple[int, int]] = []   # (marker_pos, offset) for 0x01
        self.unknown_type_offsets: Counter = Counter()  # 0x1b offsets not mapped
        # offset -> resolved kind ("vector"/"uol"/"sub"/"opaque"), shared across
        # every skill in a file so each file-specific type offset is classified
        # by structure only once (file_00008 has many; avoids per-node retrials).
        self._type_cache = type_cache if type_cache is not None else {}

    # ── primitives ──────────────────────────────────────────────────────
    def _u32(self, o: int) -> int:
        return int.from_bytes(self.d[o:o + 4], "little")

    def _cint(self, o: int) -> Tuple[int, int]:
        b = self.d[o]
        sb = b - 256 if b >= 128 else b
        if sb == -128:
            return int.from_bytes(self.d[o + 1:o + 5], "little", signed=True), o + 5
        return sb, o + 1

    def _clong(self, o: int) -> Tuple[int, int]:
        b = self.d[o]
        sb = b - 256 if b >= 128 else b
        if sb == -128:
            return int.from_bytes(self.d[o + 1:o + 9], "little", signed=True), o + 9
        return sb, o + 1

    def _inline(self, o: int) -> Tuple[str, int]:
        """Decode the inline string whose sign byte is at ``o``."""
        sb = self.d[o]
        s = sb - 256 if sb >= 128 else sb
        if s == 0:
            return "", o + 1
        if s < 0:
            n = -s
            c0 = o + 1
            if s == -128:
                n = self._u32(o + 1)
                c0 = o + 5
            out = bytes((self.d[c0 + i] ^ ((0xAA + i) & 0xFF)) & 0xFF for i in range(n))
            return out.decode("latin-1"), c0 + n
        # unicode (rare for these trees)
        n = s
        c0 = o + 1
        if s == 127:
            n = self._u32(o + 1)
            c0 = o + 5
        return self.d[c0:c0 + n * 2].decode("utf-16-le", "replace"), c0 + n * 2

    def _sblock(self, o: int):
        """String block at ``o`` → (value, next_offset).

        value is ``str`` (inline), ``('REF', off)`` (0x01 body ref) or
        ``('TYPE', name)`` (0x1b global type)."""
        m = self.d[o]
        if m in (0x00, 0x73):
            return self._inline(o + 1)
        if m == 0x01:
            off = self._u32(o + 1)
            self.refs.append((o, off))
            return ("REF", off), o + 5
        if m == 0x1b:
            off = self._u32(o + 1)
            if off not in _GLOBAL_TYPE_BY_OFFSET:
                self.unknown_type_offsets[off] += 1
            return ("TYPE", _GLOBAL_TYPE_BY_OFFSET.get(off, f"?{off}")), o + 5
        raise ValueError(f"bad string-block marker {m:#x} @ {o}")

    # ── grammar ─────────────────────────────────────────────────────────
    def proplist(self, o: int) -> Tuple[list, int]:
        cnt, o = self._cint(o)
        if cnt < 0 or cnt > 200000:
            raise ValueError(f"implausible property count {cnt} @ {o}")
        out = []
        for _ in range(cnt):
            name, o = self._sblock(o)
            tag = self.d[o]
            o += 1
            val, o = self.value(o, tag)
            out.append((name, tag, val))
        return out, o

    def value(self, o: int, tag: int):
        if tag == 0:
            return ("null", None), o
        if tag in (2, 11):
            return ("short", int.from_bytes(self.d[o:o + 2], "little", signed=True)), o + 2
        if tag in (3, 19):
            v, o = self._cint(o)
            return ("int", v), o
        if tag == 20:
            v, o = self._clong(o)
            return ("long", v), o
        if tag == 4:
            if self.d[o] == 0x80:
                return ("float", struct.unpack_from("<f", self.d, o + 1)[0]), o + 5
            return ("float", 0.0), o + 1
        if tag == 5:
            return ("double", struct.unpack_from("<d", self.d, o)[0]), o + 8
        if tag == 8:
            v, o = self._sblock(o)
            return ("string", v), o
        if tag == 9:
            bsize = self._u32(o)
            o2 = o + 4
            end = o2 + bsize
            etype, o2 = self._sblock(o2)
            tname = etype[1] if isinstance(etype, tuple) else etype
            v = self.extended(o2, tname, end)
            return v, end
        raise ValueError(f"bad property tag {tag} @ {o - 1}")

    def extended(self, o: int, tname: str, end: int):
        if tname == "Property":
            o += 2  # reserved
            lst, o = self.proplist(o)
            return ("sub", lst)
        if tname == "Canvas":
            o += 1  # reserved
            has = self.d[o]
            o += 1
            children = []
            if has == 1:
                o += 2
                children, o = self.proplist(o)
            return ("canvas", children)
        if tname == "Shape2D#Vector2D":
            x, o = self._cint(o)
            y, o = self._cint(o)
            return ("vector", (x, y))
        if tname == "UOL":
            o += 1
            v, o = self._sblock(o)
            return ("uol", v)
        if tname == "Shape2D#Convex2D":
            cnt, o = self._cint(o)
            pts = []
            for _ in range(cnt):
                _nm, o = self._sblock(o)
                tag = self.d[o]
                o += 1
                val, o = self.value(o, tag)
                if val[0] == "vector":
                    pts.append(val[1])
            return ("convex", pts)
        # Unknown (file-specific) ext-type offset — dispatch by structure, but
        # cache the classification per offset so it's only trialed once a file.
        off = int(tname[1:]) if tname.startswith("?") else -1
        cached = self._type_cache.get(off)
        if cached is not None:
            return self._parse_as(cached, o, end)
        kind, val = self._dispatch_unknown(o, end)
        if off >= 0:
            self._type_cache[off] = kind
        return val

    def _parse_as(self, kind: str, o: int, end: int):
        """Parse an extended block of a previously-classified ``kind``."""
        if kind == "vector":
            x, o = self._cint(o)
            y, _o = self._cint(o)
            return ("vector", (x, y))
        if kind == "uol":
            v, _o = self._sblock(o + 1)
            return ("uol", v)
        if kind == "sub":
            try:
                lst, _o = self.proplist(o + 2)
                return ("sub", lst)
            except Exception:
                return ("opaque", None)
        return ("opaque", None)

    def _sblock_len(self, o: int) -> Optional[int]:
        """End offset of the string block at ``o`` without side effects."""
        m = self.d[o]
        if m in (0x00, 0x73):
            sb = self.d[o + 1]
            s = sb - 256 if sb >= 128 else sb
            if s == 0:
                return o + 2
            if s < 0:
                n = -s
                c0 = o + 2
                if s == -128:
                    n = self._u32(o + 2)
                    c0 = o + 6
                return c0 + n
            n = s
            c0 = o + 2
            if s == 127:
                n = self._u32(o + 2)
                c0 = o + 6
            return c0 + n * 2
        if m in (0x01, 0x1b):
            return o + 5
        return None

    def _dispatch_unknown(self, o: int, end: int):
        """Classify an extended block whose type offset isn't in the stable map,
        by which standard WZ type consumes exactly ``end - o`` bytes. Tries the
        tightly-framed types first (Vector, UOL) then Property; falls back to
        opaque (safely skipped via the block's declared size). Returns
        ``(kind, value)`` so the caller can cache ``kind`` for the offset."""
        # Vector2D: exactly two compressed ints.
        try:
            x, o1 = self._cint(o)
            y, o2 = self._cint(o1)
            if o2 == end:
                return "vector", ("vector", (x, y))
        except Exception:
            pass
        # UOL: reserved byte + string block.
        try:
            n = self._sblock_len(o + 1)
            if n == end:
                v, _ = self._sblock(o + 1)
                return "uol", ("uol", v)
        except Exception:
            pass
        # Property: reserved(2) + property list that lands exactly on the end.
        save = len(self.refs)
        try:
            lst, o2 = self.proplist(o + 2)
            if o2 == end:
                return "sub", ("sub", lst)
        except Exception:
            pass
        del self.refs[save:]   # roll back refs collected during a failed trial
        return "opaque", ("opaque", None)


# ── per-img base recovery ───────────────────────────────────────────────────
def _vocab_positions(data: bytes) -> List[Tuple[int, str]]:
    """Sorted [(sign_offset, name)] for every inline occurrence of a vocab name."""
    hits: List[Tuple[int, str]] = []
    for name in _VOCAB:
        pat = bytes([(256 - len(name)) & 0xFF]) + _enc(name)
        i = data.find(pat)
        while i != -1:
            hits.append((i, name))
            i = data.find(pat, i + 1)
    hits.sort()
    return hits


def _solve_base(refs: List[int], vocab_pos: List[Tuple[int, str]],
                lo: int, hi: int, base_max: int) -> Optional[int]:
    """Recover img_base: the base maximizing the count of *distinct* vocab names
    that the img's 0x01 refs resolve to. Diversity (not raw count) is the
    discriminator — a wrong base piles many refs onto one or two strings, the
    true base spreads them across the real property vocabulary.
    """
    import bisect
    offs = vocab_pos[
        bisect.bisect_left(vocab_pos, (lo, "")):bisect.bisect_right(vocab_pos, (hi, "\xff"))
    ]
    # Distinct ref offsets only: a name referenced from N skills appears as N
    # identical offsets, all yielding the same base vote — collapsing them turns
    # an O(refs×vocab) scan into O(distinct_offsets×vocab) (≫10× on big imgs).
    uniq = set(refs)
    if not offs or not uniq:
        return None
    distinct: Dict[int, set] = defaultdict(set)
    for O in uniq:
        for (S, name) in offs:
            B = S - O
            if 0 < B < base_max:
                distinct[B].add(name)
    if not distinct:
        return None
    return max(distinct, key=lambda k: len(distinct[k]))


# ── public entry ────────────────────────────────────────────────────────────
def _scan_skill_anchors(data: bytes) -> List[Tuple[int, str]]:
    """[(sign_offset, skill_id)] for every inline 7-/8-digit id followed by a
    tag-9 byte (= the start of a skill's extended Property)."""
    res: List[Tuple[int, str]] = []
    for length, sign in ((7, 0xF9), (8, 0xF8)):
        sb = bytes([sign])
        i = data.find(sb)
        while i != -1:
            c0 = i + 1
            if c0 + length < len(data):
                digits = bytearray(length)
                ok = True
                for k in range(length):
                    ch = data[c0 + k] ^ ((0xAA + k) & 0xFF)
                    if not (48 <= ch <= 57):
                        ok = False
                        break
                    digits[k] = ch
                if ok and data[c0 + length] == 0x09:
                    res.append((i, digits.decode("ascii")))
            i = data.find(sb, i + 1)
    return res


def parse_skill_imgs(data: bytes) -> Dict[str, WzSubProperty]:
    """Parse every job-group skill img in a ``Skill_*.ms`` body.

    Returns ``{img_stem: WzSubProperty}`` where the sub-property's children are
    the skill nodes (``WzSubProperty`` named by skill id), each holding the
    fully-resolved stat tree. Unresolvable refs (names stored in the encrypted
    header — mostly icon/graphics labels) are kept as ``_ref_<offset>``.
    """
    anchors = _scan_skill_anchors(data)
    vocab_pos = _vocab_positions(data)
    type_cache: dict = {}   # file-wide offset -> structural kind (perf)

    # Phase 1: parse each inline-id skill subtree; keep (pos, sid, tree, refs).
    records: Dict[str, list] = defaultdict(list)   # stem -> [(pos, sid, tree, refs)]
    for (S, sid) in anchors:
        if len(sid) < 5:
            continue
        p = _BodyParser(data, type_cache)
        try:
            _, name_end = p._inline(S)
            if data[name_end] != 0x09:
                continue
            val, _end = p.value(name_end + 1, 0x09)
        except Exception:
            continue
        if val[0] != "sub":
            continue
        records[sid[:-4]].append((S, sid, val[1], [o for (_pos, o) in p.refs]))

    # Phase 2: an img is a *contiguous* file region, but the same id can recur
    # far away as a cross-reference (e.g. inside another job's skillList). Split
    # each stem's anchors into proximity clusters and solve a base PER cluster —
    # mixing a stray cross-ref's refs into the solve corrupts the base (and so
    # every 0x01 name in the img fails to resolve). Phase 2.5 then recovers
    # deduped (0x01-named) skills within each cluster's span.
    out: Dict[str, WzSubProperty] = {}
    GAP = 1_000_000
    for stem, recs in records.items():
        recs.sort(key=lambda r: r[0])
        clusters: List[list] = [[recs[0]]]
        for r in recs[1:]:
            if r[0] - clusters[-1][-1][0] > GAP:
                clusters.append([])
            clusters[-1].append(r)

        # sid -> (tree, base, score); keep the richest version across clusters.
        skills: Dict[str, tuple] = {}

        def consider(sid: str, tree: list, base: Optional[int]) -> None:
            score = len(tree) if tree else 0
            prev = skills.get(sid)
            if prev is None or score > prev[2]:
                skills[sid] = (tree, base, score)

        for cl in clusters:
            positions = [r[0] for r in cl]
            refs = [o for r in cl for o in r[3]]
            base = _solve_base(refs, vocab_pos, min(positions) - 4000,
                               max(positions) + 120000, base_max=min(positions))
            for (_S, sid, tree, _r) in cl:
                consider(sid, tree, base)
            if base is None:
                continue
            seen = {sid for (_S, sid, _t, _r) in cl}
            lo = max(0, min(positions) - 4000)
            hi = min(len(data), max(positions) + 120000)
            i = data.find(b"\x01", lo)
            while i != -1 and i < hi:
                if i + 6 <= len(data) and data[i + 5] == 0x09:
                    off = int.from_bytes(data[i + 1:i + 5], "little")
                    rid = _decode_inline_at(data, base + off)
                    if rid and rid.isdigit() and 5 <= len(rid) <= 8 \
                            and rid[:-4] == stem and rid not in seen:
                        seen.add(rid)
                        p = _BodyParser(data, type_cache)
                        try:
                            val, _ = p.value(i + 6, 0x09)
                            if val[0] == "sub" and val[1]:
                                consider(rid, val[1], base)
                        except Exception:
                            pass
                i = data.find(b"\x01", i + 1)

        # Phase 3: build resolved trees, dropping empty stubs.
        img = WzSubProperty(stem + ".img")
        skill_container = WzSubProperty("skill")
        img.add(skill_container)
        for sid in sorted(skills):
            tree, base, _score = skills[sid]
            if not tree:
                continue
            node = _build_tree(tree, sid, data, base)
            _infer_block_names(node)
            skill_container.add(node)
        if skill_container.has_children():
            out[stem] = img
    return out


_FORMULA = re.compile(r"[+*]|d\(")


def _is_stat_block(node: WzSubProperty) -> bool:
    """A stat block (common/level-leaf): has an lt+rb range, or a level-formula
    string value (e.g. '195+3*x', '10+d(x/2)')."""
    names = {c.name for c in node.children()}
    if "lt" in names and "rb" in names:
        return True
    for c in node.children():
        v = getattr(c, "value", None)
        if isinstance(v, str) and "x" in v and _FORMULA.search(v):
            return True
    return False


def _has_numeric_subprops(node: WzSubProperty) -> bool:
    n = sum(1 for c in node.children()
            if c.name.isdigit() and isinstance(c, WzSubProperty))
    return n >= 2


def _infer_block_names(skill: WzSubProperty) -> None:
    """Best-effort labelling of blocks whose names live in the (un-cracked)
    encrypted per-img name table, so the tree is navigable. Only renames
    ``_ref_<n>`` children — never overrides a file-sourced name — and only when
    the structure is unambiguous:

      * a block with lt/rb or a level-formula  → ``common`` / ``PVPcommon``
      * a block with numeric ``0/1/2…`` sub-levels → ``level`` / ``PVPlevel``
      * canvases right after ``icon``          → ``iconMouseOver`` / ``iconDisabled``

    Field names *inside* an encrypted table (which ref is mpCon vs damage) stay
    ``_ref_<n>``; lt/rb/cooltime and all values are already correct.
    """
    items = list(skill._children.items())
    taken = {n for n, _ in items if not n.startswith("_ref_")}
    stat = ["common", "PVPcommon"]
    lvl = ["level", "PVPlevel"]
    icon_v = ["iconMouseOver", "iconDisabled"]
    si = li = ii = 0
    icon_seen = False
    rebuilt: Dict[str, WzProperty] = {}
    for name, node in items:
        new = name
        if name == "icon":
            icon_seen = True
        elif (icon_seen and ii < len(icon_v) and name.startswith("_ref_")
              and isinstance(node, WzCanvasProperty)):
            new, ii = icon_v[ii], ii + 1
        elif (name.startswith("_ref_") and isinstance(node, WzSubProperty)
              and not isinstance(node, WzCanvasProperty)):
            if _has_numeric_subprops(node) and li < len(lvl):
                new, li = lvl[li], li + 1
            elif _is_stat_block(node) and si < len(stat):
                new, si = stat[si], si + 1
        if new != name and new in taken:
            new = name      # never clobber an existing sibling
        taken.add(new)
        node.name = new
        rebuilt[new] = node
    skill._children = rebuilt


def _resolve_name(name, data: bytes, base: Optional[int]) -> str:
    if isinstance(name, str):
        return name
    if isinstance(name, tuple) and name and name[0] == "REF":
        off = name[1]
        if base is not None:
            t = _decode_inline_at(data, base + off)
            if t is not None:
                return t
        return f"_ref_{off}"
    return str(name)


def _decode_inline_at(data: bytes, sign_off: int) -> Optional[str]:
    if sign_off < 0 or sign_off >= len(data):
        return None
    sb = data[sign_off]
    if not (0x80 <= sb <= 0xFF):
        return None
    n = 256 - sb
    c0 = sign_off + 1
    if n == 0 or c0 + n > len(data):
        return None
    out = bytearray(n)
    for i in range(n):
        ch = data[c0 + i] ^ ((0xAA + i) & 0xFF)
        if ch < 32 or ch >= 127:
            return None
        out[i] = ch
    return out.decode("ascii")


def _build_tree(children: list, name: str, data: bytes,
                base: Optional[int]) -> WzSubProperty:
    """Convert a raw parsed child list into resolved WzProperty nodes."""
    node = WzSubProperty(name)
    for (cname, tag, val) in children:
        rn = _resolve_name(cname, data, base)
        kind = val[0]
        if kind == "sub":
            node.add(_build_tree(val[1], rn, data, base))
        elif kind == "canvas":
            cv = WzCanvasProperty(rn)
            for ch in _build_tree(val[1], "_", data, base).children():
                cv.add(ch)
            node.add(cv)
        elif kind == "vector":
            node.add(WzVectorProperty(rn, val[1][0], val[1][1]))
        elif kind == "uol":
            node.add(WzUolProperty(rn, _resolve_value_str(val[1], data, base)))
        elif kind == "convex":
            conv = WzConvexProperty(rn)
            for (x, y) in val[1]:
                conv.points.append(WzVectorProperty("", x, y))
            node.add(conv)
        elif kind == "string":
            node.add(WzStringProperty(rn, _resolve_value_str(val[1], data, base)))
        elif kind == "int":
            node.add(WzIntProperty(rn, val[1]))
        elif kind == "short":
            node.add(WzShortProperty(rn, val[1]))
        elif kind == "long":
            node.add(WzLongProperty(rn, val[1]))
        elif kind == "float":
            node.add(WzFloatProperty(rn, val[1]))
        elif kind == "double":
            node.add(WzDoubleProperty(rn, val[1]))
        elif kind == "null":
            node.add(WzNullProperty(rn))
        else:  # opaque / unknown
            node.add(WzNullProperty(rn))
    return node


def _resolve_value_str(v, data: bytes, base: Optional[int]) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, tuple) and v and v[0] == "REF":
        if base is not None:
            t = _decode_inline_at(data, base + v[1])
            if t is not None:
                return t
        return f"_ref_{v[1]}"
    return str(v)
