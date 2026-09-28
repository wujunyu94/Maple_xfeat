"""IMG property tree.

Property type tags follow ``MapleLib/WzLib/WzProperties``:

  * 0  → Null
  * 2/11 → UShort  (16-bit signed in C# but the wire is u16)
  * 3/19 → Compressed Int
  * 4   → Float
  * 5   → Double
  * 8   → String
  * 9   → Extended (Property/Canvas/Vector/Convex/Sound/UOL)
  * 20  → Compressed Long  (post-BB)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .reader import WzBinaryReader
    from .wz_image import WzImage


class WzProperty:
    """Base class for all IMG property nodes."""

    type_name: str = "Property"

    def __init__(self, name: str, parent: Optional["WzProperty"] = None):
        self.name = name
        self.parent: Optional["WzProperty"] = parent

    @property
    def value(self) -> Any:
        return None

    def child(self, name: str) -> Optional["WzProperty"]:
        return None

    def children(self) -> List["WzProperty"]:
        return []

    def get(self, path: str) -> Optional["WzProperty"]:
        node: Optional["WzProperty"] = self
        for part in path.split("/"):
            if part == "" or part == ".":
                continue
            if part == "..":
                node = node.parent if node else None
                continue
            if node is None:
                return None
            node = node.child(part)
        return node


# ── leaf scalar types ─────────────────────────────────────────────────
class _Scalar(WzProperty):
    type_name = "Scalar"

    def __init__(self, name: str, value: Any, parent: Optional[WzProperty] = None):
        super().__init__(name, parent)
        self._value = value

    @property
    def value(self) -> Any:
        return self._value


class WzNullProperty(_Scalar):
    type_name = "Null"

    def __init__(self, name: str, parent: Optional[WzProperty] = None):
        super().__init__(name, None, parent)


class WzShortProperty(_Scalar):
    type_name = "Short"


class WzIntProperty(_Scalar):
    type_name = "Int"


class WzLongProperty(_Scalar):
    type_name = "Long"


class WzFloatProperty(_Scalar):
    type_name = "Float"


class WzDoubleProperty(_Scalar):
    type_name = "Double"


class WzStringProperty(_Scalar):
    type_name = "String"

    def __init__(self, name: str, value: str, parent: Optional[WzProperty] = None):
        super().__init__(name, value, parent)
        # Set by the parser via :meth:`WzBinaryReader.read_string_block_with_location`
        # so the editor can re-encrypt a same-length new value into the
        # exact byte slot the original occupies.
        self._payload_offset: Optional[int] = None
        self._payload_length: Optional[int] = None
        self._encoding: Optional[str] = None  # "ascii" | "unicode"
        self._indirected: bool = False


class WzVectorProperty(_Scalar):
    type_name = "Vector"

    def __init__(self, name: str, x: int, y: int, parent: Optional[WzProperty] = None):
        super().__init__(name, (x, y), parent)
        self.x = x
        self.y = y


class WzUolProperty(_Scalar):
    """A symlink to another property in the same .img (or across)."""
    type_name = "UOL"


class WzConvexProperty(WzProperty):
    type_name = "Convex"

    def __init__(self, name: str, parent: Optional[WzProperty] = None):
        super().__init__(name, parent)
        self.points: List[WzVectorProperty] = []

    @property
    def value(self) -> List[Tuple[int, int]]:
        return [(p.x, p.y) for p in self.points]

    def children(self) -> List[WzProperty]:
        return list(self.points)


# ── containers ────────────────────────────────────────────────────────
class WzSubProperty(WzProperty):
    type_name = "SubProperty"

    def __init__(self, name: str, parent: Optional[WzProperty] = None):
        super().__init__(name, parent)
        self._children: Dict[str, WzProperty] = {}

    def add(self, prop: WzProperty) -> None:
        prop.parent = self
        self._children[prop.name] = prop

    def child(self, name: str) -> Optional[WzProperty]:
        return self._children.get(name)

    def children(self) -> List[WzProperty]:
        return list(self._children.values())

    def has_children(self) -> bool:
        """O(1) emptiness check; avoids allocating the children list."""
        return bool(self._children)

    def child_count(self) -> int:
        return len(self._children)


class WzCanvasProperty(WzSubProperty):
    type_name = "Canvas"

    def __init__(self, name: str, parent: Optional[WzProperty] = None):
        super().__init__(name, parent)
        # Lazy-loaded image data — actual decoding lives in canvas.py.
        self.width = 0
        self.height = 0
        self.format = 0      # base pixel format
        self.format2 = 0     # secondary (mostly 0)
        self._png_offset = 0
        self._png_length = 0
        self._png_data: Optional[bytes] = None  # raw compressed payload
        self._wz_image: Optional["WzImage"] = None

    def has_pixels(self) -> bool:
        return self._png_length > 0

    @property
    def value(self) -> Tuple[int, int, int]:
        return (self.width, self.height, self.format + self.format2)


class WzSoundProperty(WzProperty):
    type_name = "Sound"

    def __init__(self, name: str, parent: Optional[WzProperty] = None):
        super().__init__(name, parent)
        self.length_ms = 0
        self.header: bytes = b""
        self._data_offset = 0
        self._data_length = 0
        self._wz_image: Optional["WzImage"] = None
        # cached when fetched
        self._data: Optional[bytes] = None

    @property
    def value(self) -> int:
        return self._data_length


# ── parser entry point ───────────────────────────────────────────────
def parse_property_list(
    reader: "WzBinaryReader",
    base_offset: int,
    parent: WzProperty,
    wz_image: "WzImage",
) -> List[WzProperty]:
    """Parse the children of a SubProperty/Canvas container.

    If the file is truncated mid-property (HaRepacker's "save image"
    feature has been observed to cut exports off at the original .img's
    in-WZ offset), we stop cleanly at the first EOFError and mark the
    image so the caller can surface a warning. The properties already
    parsed are returned in full."""
    try:
        count = reader.read_compressed_int()
    except EOFError:
        if wz_image is not None:
            wz_image.truncated = True
        return []
    items: List[WzProperty] = []
    for _ in range(count):
        try:
            name = reader.read_string_block(base_offset)
            prop = _parse_extended_or_basic(reader, base_offset, name, parent, wz_image)
        except EOFError:
            # File ended mid-property — partial parse, mark and stop.
            if wz_image is not None:
                wz_image.truncated = True
            break
        except ValueError as exc:
            # Unknown marker / unknown tag / similar structural oddity in
            # this property. Stop cleanly with what we have — the rest of
            # the image's tree is unreachable from here without risking
            # garbage bytes being interpreted as more properties. Real
            # WZ archives sometimes carry novel tag types we haven't
            # learned yet; degrading gracefully beats a 500 on the
            # whole subtree.
            if wz_image is not None:
                wz_image.truncated = True
                wz_image.parse_warnings.append(
                    f"stopped parsing {parent.name or '?'}: {exc}"
                )
            break
        items.append(prop)
    return items


def parse_property_list_filtered(
    reader: "WzBinaryReader",
    base_offset: int,
    parent: WzProperty,
    wz_image: "WzImage",
    *,
    only: frozenset,
) -> List[WzProperty]:
    """Parse children but recurse only into tag-9 (extended) entries
    whose name is in ``only``; skip the body of every other tag-9
    entry via its 4-byte ``block_size`` prefix.

    Lets ``_read_cash_flag`` land at ``info/cash`` without walking the
    pose / frame / canvas subtrees that dominate a Weapon img's parse
    cost — for those imgs we read ``count`` + a few ``string_block`` +
    ``seek(end_pos)`` per top-level child instead of recursively
    decoding 50+ canvas-metadata properties. Same EOFError / ValueError
    truncation handling as :func:`parse_property_list`.

    Tag-2 / 3 / 4 / 5 / 8 / 11 / 19 / 20 (basic scalars / strings) and
    tag 0 (Null) still parse normally — they're tiny and we may want
    them (e.g. ``info`` itself is tag-9, but its children are mostly
    these). Only tag-9 children get the name-gated skip.
    """
    try:
        count = reader.read_compressed_int()
    except EOFError:
        if wz_image is not None:
            wz_image.truncated = True
        return []
    items: List[WzProperty] = []
    for _ in range(count):
        try:
            name = reader.read_string_block(base_offset)
            tag = reader.read_byte()
            if tag == 9:
                block_size = reader.read_u32()
                end_pos = reader.position + block_size
                if name in only:
                    ext_type = reader.read_string_block(base_offset)
                    prop = _parse_extended(
                        reader, base_offset, name, ext_type,
                        parent, wz_image, end_pos,
                    )
                    if reader.position < end_pos:
                        reader.seek(end_pos)
                    items.append(prop)
                else:
                    # Skip the entire extended body (ext_type + payload).
                    reader.seek(end_pos)
            else:
                # Basic/scalar tag: cheap to parse, always include so
                # callers asking only for tag-9 names still see siblings
                # like ``cash`` (tag 3) when they're at the same depth.
                prop = _decode_basic_tag(
                    reader, base_offset, name, tag, parent, wz_image,
                )
                items.append(prop)
        except EOFError:
            if wz_image is not None:
                wz_image.truncated = True
            break
        except ValueError as exc:
            if wz_image is not None:
                wz_image.truncated = True
                wz_image.parse_warnings.append(
                    f"stopped parsing {parent.name or '?'}: {exc}"
                )
            break
    return items


def _decode_basic_tag(
    reader: "WzBinaryReader",
    base_offset: int,
    name: str,
    tag: int,
    parent: WzProperty,
    wz_image: "WzImage",
) -> WzProperty:
    """Tag-byte already consumed — dispatch the non-extended (non-tag-9)
    cases of ``_parse_extended_or_basic``. Used by the filtered parser
    so we don't have to read the tag twice for tag-9 children."""
    if tag == 0:
        return WzNullProperty(name, parent)
    if tag in (2, 11):
        v_off = reader.position
        prop = WzShortProperty(name, reader.read_i16(), parent)
        prop._value_offset, prop._value_length = v_off, 2
        return prop
    if tag in (3, 19):
        v_off = reader.position
        prop = WzIntProperty(name, reader.read_compressed_int(), parent)
        prop._value_offset, prop._value_length = v_off, reader.position - v_off
        return prop
    if tag == 20:
        v_off = reader.position
        prop = WzLongProperty(name, reader.read_compressed_long(), parent)
        prop._value_offset, prop._value_length = v_off, reader.position - v_off
        return prop
    if tag == 4:
        v_off = reader.position
        sub = reader.read_byte()
        if sub == 0x80:
            prop = WzFloatProperty(name, reader.read_f32(), parent)
        else:
            prop = WzFloatProperty(name, 0.0, parent)
        prop._value_offset, prop._value_length = v_off, reader.position - v_off
        return prop
    if tag == 5:
        v_off = reader.position
        prop = WzDoubleProperty(name, reader.read_f64(), parent)
        prop._value_offset, prop._value_length = v_off, 8
        return prop
    if tag == 8:
        text, p_off, p_len, enc, indirected = reader.read_string_block_with_location(base_offset)
        prop = WzStringProperty(name, text, parent)
        prop._payload_offset = p_off
        prop._payload_length = p_len
        prop._encoding = enc
        prop._indirected = indirected
        return prop
    raise ValueError(f"unknown property tag {tag} for '{name}' at 0x{reader.position - 1:X}")


def _parse_extended_or_basic(
    reader: "WzBinaryReader",
    base_offset: int,
    name: str,
    parent: WzProperty,
    wz_image: "WzImage",
) -> WzProperty:
    tag = reader.read_byte()
    if tag == 0:
        return WzNullProperty(name, parent)
    if tag in (2, 11):
        # Short: always 2 bytes after the tag — straightforward to patch.
        v_off = reader.position
        prop = WzShortProperty(name, reader.read_i16(), parent)
        prop._value_offset, prop._value_length = v_off, 2
        return prop
    if tag in (3, 19):
        # Int: compressed format — 1 byte if value fits in i8, else 5 bytes
        # (0x80 sentinel + 4-byte i32). Save the byte range so a later
        # write knows whether the new value still fits.
        v_off = reader.position
        prop = WzIntProperty(name, reader.read_compressed_int(), parent)
        prop._value_offset, prop._value_length = v_off, reader.position - v_off
        return prop
    if tag == 20:
        v_off = reader.position
        prop = WzLongProperty(name, reader.read_compressed_long(), parent)
        prop._value_offset, prop._value_length = v_off, reader.position - v_off
        return prop
    if tag == 4:
        # Float: 1-byte sub-tag (0x00 = exactly 0.0, 0x80 = 4 floats follow).
        # ``_value_offset`` always points at the sub-tag itself so writes
        # cover the full encoded form.
        v_off = reader.position
        sub = reader.read_byte()
        if sub == 0x80:
            prop = WzFloatProperty(name, reader.read_f32(), parent)
        else:
            prop = WzFloatProperty(name, 0.0, parent)
        prop._value_offset, prop._value_length = v_off, reader.position - v_off
        return prop
    if tag == 5:
        v_off = reader.position
        prop = WzDoubleProperty(name, reader.read_f64(), parent)
        prop._value_offset, prop._value_length = v_off, 8
        return prop
    if tag == 8:
        # Strings: capture where the encrypted payload bytes physically
        # live (so the editor can patch them in place) plus the encoding
        # so we can re-encrypt the new value identically.
        text, p_off, p_len, enc, indirected = reader.read_string_block_with_location(base_offset)
        prop = WzStringProperty(name, text, parent)
        prop._payload_offset = p_off
        prop._payload_length = p_len
        prop._encoding = enc
        prop._indirected = indirected
        return prop
    if tag == 9:
        block_size = reader.read_u32()
        end_pos = reader.position + block_size
        ext_type = reader.read_string_block(base_offset)
        prop = _parse_extended(reader, base_offset, name, ext_type, parent, wz_image, end_pos)
        # Defensive: skip any trailing payload (Canvas reads up to a known PNG end).
        # ``seek`` past EOF is silent; the next read elsewhere is what raises,
        # which our outer try/except in ``parse_property_list`` then catches.
        if reader.position < end_pos:
            reader.seek(end_pos)
        return prop
    raise ValueError(f"unknown property tag {tag} for '{name}' at 0x{reader.position - 1:X}")


def _parse_extended(
    reader: "WzBinaryReader",
    base_offset: int,
    name: str,
    ext_type: str,
    parent: WzProperty,
    wz_image: "WzImage",
    end_pos: int,
) -> WzProperty:
    if ext_type in ("Property", "Shape2D#Convex2D"):
        if ext_type == "Property":
            sub = WzSubProperty(name, parent)
            reader.skip(2)  # reserved
            for child in parse_property_list(reader, base_offset, sub, wz_image):
                sub.add(child)
            return sub
        # Convex
        convex = WzConvexProperty(name, parent)
        count = reader.read_compressed_int()
        for _ in range(count):
            sub_name = reader.read_string_block(base_offset)
            child = _parse_extended_or_basic(reader, base_offset, sub_name, convex, wz_image)
            if isinstance(child, WzVectorProperty):
                convex.points.append(child)
        return convex

    if ext_type == "Canvas":
        canvas = WzCanvasProperty(name, parent)
        canvas._wz_image = wz_image
        reader.skip(1)  # reserved
        has_children = reader.read_byte()
        if has_children == 1:
            reader.skip(2)
            for child in parse_property_list(reader, base_offset, canvas, wz_image):
                canvas.add(child)
        canvas.width = reader.read_compressed_int()
        canvas.height = reader.read_compressed_int()
        canvas.format = reader.read_compressed_int()
        canvas.format2 = reader.read_byte()
        reader.skip(4)  # reserved
        reader.read_i32()  # declared length field — see note below
        reader.skip(1)  # filler byte before PNG payload
        canvas._png_offset = reader.position
        # The declared length (``field - 1`` in MapleLib) is unreliable across
        # WZ versions; it's been off-by-one in newer 64-bit clients. The
        # extended-property block boundary is authoritative and zlib stops on
        # its own EOF marker, so reading "too many" bytes is harmless.
        canvas._png_length = max(0, end_pos - reader.position)
        reader.seek(end_pos)
        return canvas

    if ext_type == "Shape2D#Vector2D":
        x = reader.read_compressed_int()
        y = reader.read_compressed_int()
        return WzVectorProperty(name, x, y, parent)

    if ext_type in ("Sound_DX8", "Sound"):
        sound = WzSoundProperty(name, parent)
        sound._wz_image = wz_image
        reader.skip(1)  # reserved
        sound._data_length = reader.read_compressed_int()
        sound.length_ms = reader.read_compressed_int()
        header_len = 0x52  # WAVEFORMATEX header is variable; known constant in WZ
        # actually the WZ format stores: 0x51 fixed bytes + variable extension.
        # Read what's there before the audio payload begins.
        header_start = reader.position
        # The header is followed directly by the audio data of length _data_length.
        header_bytes_avail = end_pos - header_start - sound._data_length
        if header_bytes_avail > 0:
            sound.header = reader.read(header_bytes_avail)
        sound._data_offset = reader.position
        reader.skip(sound._data_length)
        return sound

    if ext_type == "UOL":
        reader.skip(1)  # reserved
        return WzUolProperty(name, reader.read_string_block(base_offset), parent)

    # Unknown extended type — treat as opaque container so parsing can continue.
    return WzSubProperty(name, parent)
