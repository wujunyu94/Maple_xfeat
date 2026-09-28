"""Binary reader with WZ-specific helpers.

Implements the operations from ``MapleLib/WzLib/Util/WzBinaryReader.cs``:
compressed integers, encrypted ASCII/Unicode strings, and encrypted offsets.
"""

from __future__ import annotations

import io
import struct
from typing import BinaryIO, Dict, Optional, Tuple

from .crypto import WZ_OFFSET_CONSTANT, WzKey


class WzBinaryReader:
    """A small wrapper around a seekable binary stream.

    All multi-byte integers are little-endian, matching the WZ on-disk format.
    """

    def __init__(self, stream: BinaryIO, key: WzKey, header_fstart: int = 0):
        self._stream = stream
        self.key = key
        self.header_fstart = header_fstart
        self.version_hash = 0  # set after detection
        # Cache of fully-decoded strings keyed by absolute file offset. IMG
        # files reuse the same property names ("0", "delay", "origin", ...)
        # over and over via offset-indirection — caching collapses thousands
        # of XOR loops into one per unique string.
        self._string_cache: dict = {}
        # Precomputed ``mask ^ keystream`` table for ASCII decryption.
        # Built lazily because we don't know the longest string up front.
        self._ascii_combined: Optional[bytes] = None
        self._unicode_combined: Optional[bytes] = None
        # Bookkeeping for the editor: how many properties reach each
        # encrypted string payload via indirection (marker 0x01/0x1B).
        # ``shared_count(off) > 1`` means editing that string will affect
        # every property pointing at it.
        self._string_indirections: Dict[int, int] = {}

    # ── basic positioning ──────────────────────────────────────────────
    @property
    def position(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._stream.seek(offset, whence)

    def skip(self, n: int) -> None:
        self._stream.seek(n, io.SEEK_CUR)

    def read(self, n: int) -> bytes:
        data = self._stream.read(n)
        if len(data) != n:
            raise EOFError(f"wanted {n} bytes, got {len(data)}")
        return data

    # ── primitives ────────────────────────────────────────────────────
    def read_byte(self) -> int:
        return self.read(1)[0]

    def read_sbyte(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def read_i16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def read_i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def read_f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def read_f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    # ── compressed integers (used widely inside .img) ─────────────────
    def read_compressed_int(self) -> int:
        sb = self.read_sbyte()
        if sb == -128:
            return self.read_i32()
        return sb

    def read_compressed_long(self) -> int:
        sb = self.read_sbyte()
        if sb == -128:
            return self.read_i64()
        return sb

    # ── encrypted strings ─────────────────────────────────────────────
    def _ensure_ascii_combined(self, length: int) -> bytes:
        cached = self._ascii_combined
        if cached is not None and len(cached) >= length:
            return cached
        # Grow geometrically so we don't rebuild on every short string.
        target = max(length, 256, (len(cached) * 2) if cached else 256)
        self.key.ensure(target)
        out = bytearray(target)
        mask = 0xAA
        for i in range(target):
            out[i] = mask ^ self.key[i]
            mask = (mask + 1) & 0xFF
        self._ascii_combined = bytes(out)
        return self._ascii_combined

    def _ensure_unicode_combined(self, char_count: int) -> bytes:
        byte_len = char_count * 2
        cached = self._unicode_combined
        if cached is not None and len(cached) >= byte_len:
            return cached
        target_chars = max(char_count, 256, (len(cached) // 2 * 2) if cached else 256)
        target_bytes = target_chars * 2
        self.key.ensure(target_bytes)
        out = bytearray(target_bytes)
        mask = 0xAAAA
        for i in range(target_chars):
            key_part = (self.key[i * 2 + 1] << 8) | self.key[i * 2]
            combined = mask ^ key_part
            out[i * 2] = combined & 0xFF
            out[i * 2 + 1] = (combined >> 8) & 0xFF
            mask = (mask + 1) & 0xFFFF
        self._unicode_combined = bytes(out)
        return self._unicode_combined

    def _decode_ascii(self, length: int) -> str:
        raw = self.read(length)
        combined = self._ensure_ascii_combined(length)
        # int-XOR is a single C-level operation — orders of magnitude faster
        # than a Python loop for the typical IMG with ~10⁵ string reads.
        n = int.from_bytes(raw, "big") ^ int.from_bytes(combined[:length], "big")
        plain = n.to_bytes(length, "big")
        try:
            return plain.decode("cp1252")
        except UnicodeDecodeError:
            return plain.decode("latin-1", errors="replace")

    def _decode_unicode(self, length: int) -> str:
        byte_len = length * 2
        raw = self.read(byte_len)
        combined = self._ensure_unicode_combined(length)
        n = int.from_bytes(raw, "big") ^ int.from_bytes(combined[:byte_len], "big")
        plain = n.to_bytes(byte_len, "big")
        try:
            return plain.decode("utf-16-le")
        except UnicodeDecodeError:
            return plain.decode("utf-16-le", errors="replace")

    def read_string_with_length(self, sign: int) -> str:
        """``sign`` < 0 → ASCII, ``sign`` > 0 → Unicode (UTF-16LE).

        The length-extension sentinels are the signed-byte extremes:
        ``-128`` for ASCII and ``+127`` for Unicode. When the sign byte
        equals the sentinel, an i32 length follows; otherwise the
        absolute value of the sign byte IS the length. (Confusing
        previously-buggy form: ``length = -sign`` then ``length == 127``
        triggers for sign=-127, which is just a 127-char ASCII string,
        not the extension — and misses sign=-128, which IS the
        extension. This caused arbitrarily long ASCII descriptions to
        decode as 128 bytes of garbage.)
        """
        if sign < 0:
            if sign == -128:
                length = self.read_i32()
            else:
                length = -sign
            if length <= 0:
                return ""
            return self._decode_ascii(length)
        if sign == 127:
            length = self.read_i32()
        else:
            length = sign
        if length <= 0:
            return ""
        return self._decode_unicode(length)

    def read_string(self) -> str:
        sign = self.read_sbyte()
        if sign == 0:
            return ""
        return self.read_string_with_length(sign)

    def read_string_at(self, offset: int) -> str:
        cached = self._string_cache.get(offset)
        if cached is not None:
            return cached
        keep = self.position
        self.seek(offset)
        s = self.read_string()
        self.seek(keep)
        self._string_cache[offset] = s
        return s

    def read_string_block(self, base_offset: int) -> str:
        """Read either an inline string (``0x00``/``0x73``) or an indirected
        offset (``0x01``/``0x1B``). Used inside .img where the same string may
        appear repeatedly."""
        marker_pos = self.position
        marker = self.read_byte()
        if marker in (0x00, 0x73):
            return self.read_string()
        if marker in (0x01, 0x1B):
            offset = self.read_u32()
            return self.read_string_at(base_offset + offset)
        raise ValueError(
            f"unknown string-block marker 0x{marker:02X} at offset 0x{marker_pos:X}"
        )

    # ── location-aware string reads (used by the editor) ──────────────
    def _measure_string_at(self, offset: int) -> Tuple[str, int, int, str]:
        """Decode the inline string at ``offset`` and return
        ``(text, payload_offset, payload_byte_count, encoding)``.

        ``payload_offset`` is the absolute file position of the encrypted
        payload bytes (after the sign byte and any length-extension i32);
        ``payload_byte_count`` is the byte count for ``patch_bytes`` to
        rewrite; ``encoding`` is ``"ascii"`` or ``"unicode"``.

        Length-extension sentinels are the signed-byte extremes
        (``-128`` for ASCII, ``+127`` for Unicode); see
        :meth:`read_string_with_length` for why.
        """
        keep = self.position
        try:
            self.seek(offset)
            sign = self.read_sbyte()
            if sign == 0:
                # Empty string — no payload.
                return "", self.position, 0, "ascii"
            if sign < 0:
                if sign == -128:
                    length = self.read_i32()
                else:
                    length = -sign
                payload_off = self.position
                if length <= 0:
                    return "", payload_off, 0, "ascii"
                text = self._decode_ascii(length)
                return text, payload_off, length, "ascii"
            if sign == 127:
                length = self.read_i32()
            else:
                length = sign
            payload_off = self.position
            if length <= 0:
                return "", payload_off, 0, "unicode"
            text = self._decode_unicode(length)
            return text, payload_off, length * 2, "unicode"
        finally:
            self.seek(keep)

    def read_string_block_with_location(
        self, base_offset: int
    ) -> Tuple[str, int, int, str, bool]:
        """Variant of :meth:`read_string_block` that also returns where the
        encrypted payload bytes physically live, plus the encoding and a
        flag telling whether this site reaches the payload via offset
        indirection (so the caller knows other properties may share it).

        Returns ``(text, payload_offset, payload_byte_count, encoding, indirected)``.
        """
        marker_pos = self.position
        marker = self.read_byte()
        if marker in (0x00, 0x73):
            # Inline form. Snapshot the start of the inline body, measure
            # it (the helper save/restores position), then advance past
            # the whole form via the regular read.
            inline_start = self.position
            text, p_off, p_len, enc = self._measure_string_at(inline_start)
            _ = self.read_string()
            return text, p_off, p_len, enc, False
        if marker in (0x01, 0x1B):
            offset = self.read_u32()
            actual = base_offset + offset
            text, p_off, p_len, enc = self._measure_string_at(actual)
            # Record this indirection so the editor can surface a shared-
            # write warning ("editing this string will change N other
            # properties that reference the same offset").
            self._string_indirections[p_off] = self._string_indirections.get(p_off, 0) + 1
            return text, p_off, p_len, enc, True
        raise ValueError(
            f"unknown string-block marker 0x{marker:02X} at offset 0x{marker_pos:X}"
        )

    def shared_count(self, payload_offset: int) -> int:
        """How many indirection sites currently point at ``payload_offset``."""
        return self._string_indirections.get(payload_offset, 0)

    # ── encrypted offsets (directory entries) ─────────────────────────
    def read_offset(self) -> int:
        offset = self.position & 0xFFFFFFFF
        offset = (offset - self.header_fstart) ^ 0xFFFFFFFF
        offset = (offset * self.version_hash) & 0xFFFFFFFF
        offset = (offset - WZ_OFFSET_CONSTANT) & 0xFFFFFFFF
        rot = offset & 0x1F
        offset = ((offset << rot) | (offset >> (32 - rot))) & 0xFFFFFFFF
        encrypted = self.read_u32()
        offset ^= encrypted
        offset = (offset + self.header_fstart * 2) & 0xFFFFFFFF
        return offset
