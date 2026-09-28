"""Encoders that produce the exact byte layout the parser consumes.

Used for in-place byte patching: an edit is accepted only when the new
value's encoded form has the same length as the original, so subsequent
properties don't shift in the file. For variable-length encodings
(compressed int/long, two-form float, strings) the caller must compare
``len(encode_*(...))`` against the property's recorded ``_value_length``.

We intentionally do not implement a full WZ writer here. That would
require regenerating directory offsets, version-hashed encoded offsets,
string-table indirection, etc. — a much larger surface area.
"""

from __future__ import annotations

import struct


def encode_compressed_int(value: int) -> bytes:
    """Encode a 32-bit signed value using WZ's compressed-int format.

    Mirrors :meth:`WzBinaryReader.read_compressed_int`: 1 byte if the
    value fits in ``[-127, 127]``, otherwise 5 bytes (``0x80`` sentinel
    + little-endian i32). The full int range ``[-2**31, 2**31 - 1]`` is
    accepted on the wide path.
    """
    if -127 <= value <= 127:
        return bytes([value & 0xFF])
    return b"\x80" + struct.pack("<i", value)


def encode_compressed_long(value: int) -> bytes:
    """64-bit counterpart of :func:`encode_compressed_int`. 1 byte for
    values in ``[-127, 127]`` else 9 bytes (``0x80`` + little-endian i64)."""
    if -127 <= value <= 127:
        return bytes([value & 0xFF])
    return b"\x80" + struct.pack("<q", value)


def encode_short(value: int) -> bytes:
    """Two-byte little-endian i16. The reader uses ``read_i16`` here."""
    return struct.pack("<h", value)


def encode_float(value: float) -> bytes:
    """Encode the bytes that follow a Float (tag 4) property's name.

    Two-form encoding: a single ``0x00`` byte represents exactly ``0.0``
    (no payload); otherwise ``0x80`` followed by an IEEE-754 little-
    endian f32.
    """
    if value == 0.0:
        return b"\x00"
    return b"\x80" + struct.pack("<f", value)


def encode_double(value: float) -> bytes:
    """Eight-byte little-endian f64. Always the same length."""
    return struct.pack("<d", value)


# ── string payload (re-)encryption ──────────────────────────────────────
# These mirror :meth:`WzBinaryReader._decode_ascii` / ``_decode_unicode``:
# the cipher is plain XOR with a precomputed ``mask ^ keystream`` table,
# so encryption and decryption are the same operation.
#
# The caller is expected to fetch the reader's already-built combined
# table — the same bytes the reader XORed against during decode — so
# that the new ciphertext interleaves cleanly with everything else
# already on disk. ``WzBinaryReader._ensure_ascii_combined(n)`` /
# ``_ensure_unicode_combined(n)`` return that table.

def encode_ascii_payload(plaintext: str, combined: bytes) -> bytes:
    """Encrypt a CP1252 payload of ``len(plaintext)`` bytes."""
    raw = plaintext.encode("cp1252")
    n = len(raw)
    if n > len(combined):
        raise ValueError(
            f"keystream table is {len(combined)} bytes, need {n}"
        )
    if n == 0:
        return b""
    # Same one-shot XOR trick the reader uses (reader.py: _decode_ascii):
    # one C-level operation regardless of length.
    xored = int.from_bytes(raw, "big") ^ int.from_bytes(combined[:n], "big")
    return xored.to_bytes(n, "big")


def encode_unicode_payload(plaintext: str, combined: bytes) -> bytes:
    """Encrypt a UTF-16-LE payload of ``2 * len(plaintext)`` bytes."""
    raw = plaintext.encode("utf-16-le")
    n = len(raw)
    if n > len(combined):
        raise ValueError(
            f"keystream table is {len(combined)} bytes, need {n}"
        )
    if n == 0:
        return b""
    xored = int.from_bytes(raw, "big") ^ int.from_bytes(combined[:n], "big")
    return xored.to_bytes(n, "big")


def encoded_string_payload_size(plaintext: str, encoding: str) -> int:
    """How many bytes a re-encrypted payload would occupy. Used by the
    server to budget-check before patching."""
    if encoding == "ascii":
        return len(plaintext.encode("cp1252"))
    if encoding == "unicode":
        return len(plaintext.encode("utf-16-le"))
    raise ValueError(f"unknown encoding: {encoding!r}")


def re_encrypt_string(reader, plaintext: str, encoding: str) -> bytes:
    """Convenience wrapper: pull the right combined-keystream table off
    ``reader`` and produce the encrypted payload bytes.

    Note: WZ's ASCII string encoding uses a position-only keystream
    (``mask = 0xAA + i, key = aes_keystream[i]``), so the bytes we
    write here are valid at *any* file offset — there's no per-offset
    salt to worry about.
    """
    if encoding == "ascii":
        n = len(plaintext.encode("cp1252"))
        if n == 0:
            return b""
        table = reader._ensure_ascii_combined(n)
        return encode_ascii_payload(plaintext, table)
    if encoding == "unicode":
        char_count = len(plaintext)
        if char_count == 0:
            return b""
        table = reader._ensure_unicode_combined(char_count)
        return encode_unicode_payload(plaintext, table)
    raise ValueError(f"unknown encoding: {encoding!r}")


# ── full WZ writer (used by save_as) ───────────────────────────────────
import struct as _struct
import zlib as _zlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .crypto import WZ_OFFSET_CONSTANT


def encode_offset(position: int, target: int, fstart: int, version_hash: int) -> int:
    """Inverse of :meth:`WzBinaryReader.read_offset`.

    Returns the u32 the parser will see at ``position`` such that
    decoding it produces ``target``. The math is the read pipeline
    re-arranged to solve for ``encrypted_offset``.
    """
    cur = position & 0xFFFFFFFF
    cur = (cur - fstart) ^ 0xFFFFFFFF
    cur = (cur * version_hash) & 0xFFFFFFFF
    cur = (cur - WZ_OFFSET_CONSTANT) & 0xFFFFFFFF
    rot = cur & 0x1F
    cur = ((cur << rot) | (cur >> (32 - rot))) & 0xFFFFFFFF
    target_pre = (target - fstart * 2) & 0xFFFFFFFF
    return cur ^ target_pre


# ── inline string-block encoder ─────────────────────────────────────────
# A string-block is the on-disk container for a string property's value
# (and for property names, image names, type tags inside extended blocks,
# etc.). We always emit the inline form on write — never the indirected
# form — so writes don't need a string table.
#
# Layout for a non-empty inline string-block:
#
#   marker (1)          0x00 for property bodies, 0x73 for image-header
#                       type names — but the parser accepts either
#                       interchangeably, so we use 0x00 everywhere.
#   sign  (1, signed)   < 0 → ASCII length, > 0 → Unicode (UTF-16LE) char count.
#                       If |sign| == 127, an i32 length follows.
#   [length i32]        only when |sign| == 127
#   payload (n)         encrypted bytes (n = |sign| if |sign| < 127 else the i32)
#
# Empty strings are encoded as a single sign byte 0x00 (the parser then
# returns "" without consuming a payload). The marker byte is still required.

def encode_string(reader: Any, s: str, prefer_ascii: bool = True) -> bytes:
    """Encode a bare string (sign byte + encrypted payload). The marker
    byte is *not* included — see :func:`encode_string_block` for that.

    A string is encoded as ASCII when ``prefer_ascii`` and every
    character fits in CP1252; otherwise Unicode UTF-16-LE.
    """
    if s == "":
        return bytes([0])
    use_ascii = prefer_ascii
    if use_ascii:
        try:
            s.encode("cp1252")
        except UnicodeEncodeError:
            use_ascii = False
    if use_ascii:
        cipher = re_encrypt_string(reader, s, "ascii")
        n = len(cipher)
        if n < 127:
            return _struct.pack("<b", -n) + cipher
        return _struct.pack("<bi", -127, n) + cipher
    cipher = re_encrypt_string(reader, s, "unicode")
    n = len(cipher) // 2  # char count
    if n < 127:
        return _struct.pack("<b", n) + cipher
    return _struct.pack("<bi", 127, n) + cipher


def encode_string_block(reader: Any, s: str, prefer_ascii: bool = True) -> bytes:
    """Marker (0x00) + inline string. Used for property names and any
    string property value where we always inline rather than indirect.
    """
    return bytes([0x00]) + encode_string(reader, s, prefer_ascii=prefer_ascii)


def encode_image_type_string(reader: Any, type_name: str) -> bytes:
    """Variant of :func:`encode_string_block` that uses marker ``0x73``,
    matching the ``WzImage`` "Property" header and extended-property
    type-tag layout that other readers expect.
    """
    return bytes([0x73]) + encode_string(reader, type_name, prefer_ascii=True)


# ── property body encoder ──────────────────────────────────────────────
# Walks a property tree and produces the bytes that go after the property's
# tag byte. Names live OUTSIDE this function (the parent's property-list
# emits them via encode_string_block) so this returns just the body.

# Tag values for non-extended properties.
_TAG = {
    "Null": 0,
    "Short": 2,      # tag 11 also reads as Short on parse; we always emit 2.
    "Int": 3,
    "Long": 20,
    "Float": 4,
    "Double": 5,
    "String": 8,
    "Extended": 9,
}


def _encode_property_body(prop: Any, reader: Any) -> bytes:
    """Body bytes only — caller writes the tag byte. For tag-9 (extended)
    types, the body is the size-prefixed extended block."""
    from .properties import (
        WzCanvasProperty, WzConvexProperty, WzDoubleProperty,
        WzFloatProperty, WzIntProperty, WzLongProperty, WzNullProperty,
        WzShortProperty, WzSoundProperty, WzStringProperty, WzSubProperty,
        WzUolProperty, WzVectorProperty,
    )
    if isinstance(prop, WzNullProperty):
        return b""
    if isinstance(prop, WzShortProperty):
        return encode_short(int(prop.value))
    if isinstance(prop, WzIntProperty):
        return encode_compressed_int(int(prop.value))
    if isinstance(prop, WzLongProperty):
        return encode_compressed_long(int(prop.value))
    if isinstance(prop, WzFloatProperty):
        return encode_float(float(prop.value))
    if isinstance(prop, WzDoubleProperty):
        return encode_double(float(prop.value))
    if isinstance(prop, WzStringProperty):
        return encode_string_block(reader, str(prop.value))
    # Extended types — produce the inner block first, then prepend the
    # u32 size of (ext_type_block + inner) so the caller can write
    # tag(9) + size + ext_type_block + inner.
    #
    # Canvas/Sound/UOL are checked BEFORE WzSubProperty because Canvas
    # inherits from it (children "origin", "delay", etc. live under the
    # canvas) and would otherwise match the Property branch first.
    if isinstance(prop, WzCanvasProperty):
        return _encode_canvas_extended_block(prop, reader)
    if isinstance(prop, WzVectorProperty):
        ext_type = "Shape2D#Vector2D"
        inner = encode_compressed_int(int(prop.x)) + encode_compressed_int(int(prop.y))
    elif isinstance(prop, WzConvexProperty):
        ext_type = "Shape2D#Convex2D"
        inner = encode_compressed_int(len(prop.points))
        for v in prop.points:
            # Each child is a Vector property without a named header — the
            # parser reads <name string-block> + tag + body, so we emit a
            # nameless entry to mirror what came in.
            inner += encode_string_block(reader, v.name)
            inner += bytes([_TAG["Extended"]])
            inner += _encode_property_body(v, reader)
    elif isinstance(prop, WzSoundProperty):
        ext_type = "Sound_DX8"
        inner = b"\x00"
        inner += encode_compressed_int(prop._data_length)
        inner += encode_compressed_int(prop.length_ms)
        inner += prop.header
        inner += _read_sound_payload(prop)
    elif isinstance(prop, WzUolProperty):
        ext_type = "UOL"
        inner = b"\x00" + encode_string_block(reader, str(prop.value))
    elif isinstance(prop, WzSubProperty):
        ext_type = "Property"
        inner = b"\x00\x00"  # reserved
        inner += _encode_property_list(prop.children(), reader)
    else:
        raise ValueError(f"don't know how to serialize {type(prop).__name__}")
    type_block = encode_image_type_string(reader, ext_type)
    block = type_block + inner
    return _struct.pack("<I", len(block)) + block


def _encode_canvas_extended_block(canvas: Any, reader: Any) -> bytes:
    """Canvas extended block (tag 9). Mirrors the parser layout exactly:
    ``size(u32) + ext_type("Canvas") + reserved(1) + has_children(1) +
    [reserved(2) + child-list]? + width(int) + height(int) + format(int) +
    format2(byte) + reserved(4) + len_field(i32) + filler(1) + payload``.
    """
    from .properties import WzSubProperty
    type_block = encode_image_type_string(reader, "Canvas")

    children = canvas.children()
    body = bytearray()
    body += b"\x00"                 # reserved
    if children:
        body += b"\x01"             # has_children
        body += b"\x00\x00"         # reserved
        body += _encode_property_list(children, reader)
    else:
        body += b"\x00"             # no children

    body += encode_compressed_int(int(canvas.width))
    body += encode_compressed_int(int(canvas.height))
    body += encode_compressed_int(int(canvas.format))
    body += bytes([int(canvas.format2) & 0xFF])
    body += b"\x00\x00\x00\x00"     # reserved

    payload = _get_canvas_payload_bytes(canvas)
    # Reader stores ``len = field - 1``; we emit ``field = payload_len + 1``.
    body += _struct.pack("<i", len(payload) + 1)
    body += b"\x00"                 # filler
    body += payload

    block = type_block + bytes(body)
    return _struct.pack("<I", len(block)) + block


def _get_canvas_payload_bytes(canvas: Any) -> bytes:
    """Return the bytes that go in the canvas's PNG slot. If the user
    staged a new image via ``encode_canvas_payload`` it lives on
    ``canvas._png_data``; otherwise fall back to reading the original
    bytes from the source file."""
    from .canvas import _read_canvas_bytes
    if canvas._png_data is not None:
        return canvas._png_data
    return _read_canvas_bytes(canvas)


def _read_sound_payload(sound: Any) -> bytes:
    """Pull the audio bytes for the on-disk write.

    For sounds parsed from a source WZ we re-read the original bytes
    via mmap (fast, byte-identical round-trip). For sounds *added*
    by the user, the audio bytes were stashed on ``sound._data`` at
    construction time — use those.
    """
    if getattr(sound, "_data", None) is not None:
        return sound._data
    if sound._wz_image is None:
        return b""
    r = sound._wz_image.wz_file.reader
    keep = r.position
    r.seek(sound._data_offset)
    data = r.read(sound._data_length)
    r.seek(keep)
    return data


def _tag_for(prop: Any) -> int:
    from .properties import (
        WzCanvasProperty, WzConvexProperty, WzSoundProperty, WzSubProperty,
        WzUolProperty, WzVectorProperty,
    )
    if isinstance(prop, (WzCanvasProperty, WzConvexProperty, WzSoundProperty,
                          WzSubProperty, WzUolProperty, WzVectorProperty)):
        return _TAG["Extended"]
    return _TAG.get(prop.type_name, _TAG["Extended"])


def _encode_property_list(props: Iterable[Any], reader: Any) -> bytes:
    """Compressed-int count + per-property ``<name><tag><body>`` records."""
    items = list(props)
    out = bytearray(encode_compressed_int(len(items)))
    for prop in items:
        out += encode_string_block(reader, prop.name)
        out += bytes([_tag_for(prop)])
        out += _encode_property_body(prop, reader)
    return bytes(out)


# ── image / directory / wz file ───────────────────────────────────────

def encode_image_body(image: Any, reader: Any) -> bytes:
    """Serialize a ``WzImage`` body (the bytes that live at the image's
    file offset and span ``image.size`` bytes)."""
    image.parse()
    out = bytearray(encode_image_type_string(reader, "Property"))
    out += b"\x00\x00"  # reserved bytes after the type tag
    out += _encode_property_list(image.root.children(), reader)
    return bytes(out)


def _entry_overhead_size(name: str, reader: Any) -> int:
    """Size of a directory entry header (kind byte + bare-name string).
    Directory entries store names with :func:`encode_string` (no marker
    byte); only property names inside images get the marker via
    :func:`encode_string_block`."""
    return 1 + len(encode_string(reader, name))


def _entry_trailing_size(image_size: int, checksum: int = 0) -> int:
    """The compressed-int size, compressed-int checksum, and u32
    encrypted-offset trailing each directory entry."""
    return (
        len(encode_compressed_int(image_size))
        + len(encode_compressed_int(checksum))
        + 4  # u32 encrypted offset
    )


# We plan positions in two passes. The plan is just dicts keyed by
# id(node). Compact and easy to debug.
class _Plan:
    __slots__ = ("entry_size", "image_body", "subdir_body", "img_position",
                 "subdir_position")

    def __init__(self):
        self.entry_size: Dict[int, int] = {}        # id(image|dir) -> size of its directory entry triple
        self.image_body: Dict[int, bytes] = {}       # id(image) -> serialized image body
        self.subdir_body: Dict[int, bytes] = {}      # id(directory) -> serialized directory body (children only)
        self.img_position: Dict[int, int] = {}       # id(image) -> file position of its body
        self.subdir_position: Dict[int, int] = {}    # id(directory) -> file position of its body


def _serialize_directory_body_skeleton(
    directory: Any, reader: Any, plan: _Plan,
) -> bytes:
    """Serialize the directory's entries with placeholder zeros for
    the encrypted offsets — used during sizing. We'll come back and
    rewrite the offsets once positions are known.
    """
    items: List[Tuple[str, Any, int]] = []
    # Subdirectories first, then images. Match the order the parser
    # produces (insertion order). kind=3 for subdir, kind=4 for image.
    for name, sub in directory.subdirs.items():
        items.append((name, sub, 3))
    for name, img in directory.images.items():
        items.append((name, img, 4))

    body = bytearray(encode_compressed_int(len(items)))
    for name, child, kind in items:
        body += bytes([kind])
        # Directory entries use the bare-string form (no marker byte);
        # only property names inside images get the marker.
        body += encode_string(reader, name)
        if kind == 3:
            child_size = len(plan.subdir_body[id(child)])
        else:
            child_size = len(plan.image_body[id(child)])
        body += encode_compressed_int(child_size)
        body += encode_compressed_int(0)  # checksum (parser ignores it)
        body += b"\x00\x00\x00\x00"        # encrypted-offset placeholder
    return bytes(body)


def _patch_directory_offsets(
    body: bytes, directory: Any, dir_position: int,
    reader: Any, plan: _Plan, fstart: int, version_hash: int,
) -> bytes:
    """Walk the freshly serialized directory body and rewrite each
    placeholder encrypted-offset with the real value, now that all
    image/subdir positions are known."""
    out = bytearray(body)
    cursor = 0  # offset within the directory body
    cursor += len(encode_compressed_int(_entry_count(directory)))

    items: List[Tuple[str, Any, int]] = []
    for name, sub in directory.subdirs.items():
        items.append((name, sub, 3))
    for name, img in directory.images.items():
        items.append((name, img, 4))

    for name, child, kind in items:
        cursor += 1  # kind byte
        cursor += len(encode_string(reader, name))
        if kind == 3:
            child_size = len(plan.subdir_body[id(child)])
            target = plan.subdir_position[id(child)]
        else:
            child_size = len(plan.image_body[id(child)])
            target = plan.img_position[id(child)]
        cursor += len(encode_compressed_int(child_size))
        cursor += len(encode_compressed_int(0))
        encrypted = encode_offset(
            dir_position + cursor, target, fstart, version_hash,
        )
        out[cursor:cursor + 4] = _struct.pack("<I", encrypted)
        cursor += 4
    return bytes(out)


def _entry_count(directory: Any) -> int:
    return len(directory.subdirs) + len(directory.images)


def _copy_image_verbatim(image: Any) -> bytes:
    """Read the image's bytes from the source mmap exactly as they
    are, with no parse/serialize round-trip. Used as the fallback for
    unedited images when our encoder can't handle some corner of an
    image's content — preserves arbitrary on-disk content losslessly."""
    if image._wz_file is None:
        raise RuntimeError(f"image {image.name!r} has no source file")
    r = image._wz_file.reader
    keep = r.position
    r.seek(image.offset)
    data = r.read(image.size)
    r.seek(keep)
    return data


def encode_wz_file(
    wz: Any,
    *,
    is_image_dirty=None,
    image_failures: Optional[List[str]] = None,
) -> bytes:
    """Re-serialize a parsed ``WzFile`` into the bytes of a fresh
    legacy 32-bit WZ archive.

    Re-uses the original copyright, ``fstart``, version hash, and
    region IV so the output decodes with the same parameters as the
    input. Subdirectories follow the file's directory tree exactly;
    images are written in the order the WZ parser would walk them
    (depth-first, subdirectories before images at each level — matches
    ``WzDirectory.walk_images`` insertion order).

    Per-image safety net: if an image is not flagged as dirty by the
    optional ``is_image_dirty(image_path) -> bool`` callback, the
    encoder copies its original on-disk bytes verbatim instead of
    re-serializing. This sidesteps any encoder bugs for the 99% of
    images that weren't touched by the user. If a dirty image's
    serialization raises, the exception propagates (we won't silently
    lose the user's edit). For non-dirty images that fail anyway, we
    fall back to verbatim and append the path to ``image_failures``.
    """
    from .wz_file import WzDirectory, WzFile
    from .wz_image import WzImage

    if not isinstance(wz, WzFile):
        raise TypeError("encode_wz_file expects a WzFile")
    if wz.header is None:
        raise RuntimeError("WzFile not loaded — call .open() first")

    reader = wz.reader
    fstart = wz.header.fstart
    version_hash = reader.version_hash
    copyright = wz.header.copyright.encode("latin-1", errors="replace")

    # ── pass 1: serialize each image body and each directory body
    #            (with placeholder zeros for encrypted offsets) ──────

    plan = _Plan()

    def _serialize_one(img: Any, image_path: str) -> bytes:
        dirty = bool(is_image_dirty(image_path)) if is_image_dirty else True
        if not dirty:
            # Fast + safe: copy the original bytes. No parse needed.
            try:
                return _copy_image_verbatim(img)
            except Exception:
                # If the source mmap is gone for some reason, fall
                # through and try to re-serialize from the parsed tree.
                pass
        try:
            return encode_image_body(img, reader)
        except Exception as exc:
            if dirty:
                # Don't hide an error on an image the user edited — the
                # caller needs to know their edit can't be persisted.
                raise RuntimeError(
                    f"failed to serialize edited image {image_path!r}: {exc}"
                ) from exc
            # Non-dirty image: fall back to verbatim and report.
            if image_failures is not None:
                image_failures.append(f"{image_path}: {exc}")
            return _copy_image_verbatim(img)

    def walk_subdir(directory: WzDirectory, dir_path: str = "") -> None:
        for name, sub in directory.subdirs.items():
            walk_subdir(sub, f"{dir_path}/{name}" if dir_path else name)
        for name, img in directory.images.items():
            img_path = f"{dir_path}/{name}" if dir_path else name
            plan.image_body[id(img)] = _serialize_one(img, img_path)
        plan.subdir_body[id(directory)] = _serialize_directory_body_skeleton(
            directory, reader, plan,
        )

    walk_subdir(wz.root)

    # ── pass 2: assign positions ────────────────────────────────────
    # Body section starts at fstart + 2 (the u16 encrypted_version).
    body_start = fstart + 2
    plan.subdir_position[id(wz.root)] = body_start

    cursor = body_start + len(plan.subdir_body[id(wz.root)])

    # All subdirectories laid out depth-first AFTER root's body.
    def assign_subdirs(directory: WzDirectory) -> None:
        nonlocal cursor
        for sub in directory.subdirs.values():
            plan.subdir_position[id(sub)] = cursor
            cursor += len(plan.subdir_body[id(sub)])
            assign_subdirs(sub)

    assign_subdirs(wz.root)

    # All image bodies laid out after the entire directory tree.
    def assign_images(directory: WzDirectory) -> None:
        nonlocal cursor
        for sub in directory.subdirs.values():
            assign_images(sub)
        for img in directory.images.values():
            plan.img_position[id(img)] = cursor
            cursor += len(plan.image_body[id(img)])

    assign_images(wz.root)

    total_body_size = cursor

    # ── pass 3: rewrite directory bodies with real encrypted offsets
    real_subdir_body: Dict[int, bytes] = {}
    def patch_dir(directory: WzDirectory) -> None:
        real_subdir_body[id(directory)] = _patch_directory_offsets(
            plan.subdir_body[id(directory)],
            directory,
            plan.subdir_position[id(directory)],
            reader, plan, fstart, version_hash,
        )
        for sub in directory.subdirs.values():
            patch_dir(sub)

    patch_dir(wz.root)

    # ── pass 4: stream out the bytes ────────────────────────────────
    encrypted_version = derive_version_check(version_hash)

    out = bytearray()
    out += b"PKG1"
    out += _struct.pack("<Q", total_body_size)  # file_size: bytes after the
                                                # header (some readers want
                                                # the full file size; others
                                                # just want the body. We
                                                # store the body size — the
                                                # parser only uses it as
                                                # informational metadata.)
    out += _struct.pack("<I", fstart)
    out += copyright
    out += b"\x00"
    # Pad to fstart in case the copyright doesn't reach.
    if len(out) < fstart:
        out += b"\x00" * (fstart - len(out))
    elif len(out) > fstart:
        raise RuntimeError(
            f"header bytes ({len(out)}) overflowed fstart ({fstart}); "
            f"copyright is too long")

    out += _struct.pack("<H", encrypted_version)
    out += real_subdir_body[id(wz.root)]

    # Subdirectories (DFS) follow root's body
    def emit_subdirs(directory: WzDirectory) -> None:
        for sub in directory.subdirs.values():
            out.extend(real_subdir_body[id(sub)])
            emit_subdirs(sub)
    emit_subdirs(wz.root)

    # Image bodies (DFS, subdirs first then images at each level)
    def emit_images(directory: WzDirectory) -> None:
        for sub in directory.subdirs.values():
            emit_images(sub)
        for img in directory.images.values():
            out.extend(plan.image_body[id(img)])
    emit_images(wz.root)

    return bytes(out)


# ── re-exports ──────────────────────────────────────────────────────────
# ``derive_version_check`` is used by encode_wz_file and re-exported here
# so callers don't need to know it lives in crypto.
from .crypto import derive_version_check  # noqa: E402  (must come after typing)
