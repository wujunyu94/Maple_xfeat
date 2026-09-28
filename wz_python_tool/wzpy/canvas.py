"""Canvas (PNG) decoding.

WZ Canvas blobs are zlib-compressed pixel data in one of several formats.
The compressed bytes themselves may also be XOR-obfuscated (the "listWz"
format) when the first two bytes do not look like a zlib header — see
``WzPngProperty.cs`` in MapleLib.
"""

from __future__ import annotations

import io
import zlib
from typing import TYPE_CHECKING

from PIL import Image

from .crypto import WzKey

if TYPE_CHECKING:
    from .properties import WzCanvasProperty


# zlib stream first-byte values we accept as "non-listWz".
_ZLIB_HEADERS = {0x9C78, 0xDA78, 0x0178, 0x5E78}


def _read_canvas_bytes(canvas: "WzCanvasProperty") -> bytes:
    if canvas._png_data is not None:
        return canvas._png_data
    wz_image = canvas._wz_image
    if wz_image is None:
        raise RuntimeError("canvas not bound to a WzImage")
    wz_file = wz_image.wz_file
    # Same reasoning as WzImage.parse: the reader's cursor is shared
    # across every canvas / image in this WzFile, so two threads
    # racing on _read_canvas_bytes would seek and read each other's
    # bytes. Acquire the file's reader lock for the duration.
    with wz_file.reader_lock:
        if canvas._png_data is not None:
            return canvas._png_data
        r = wz_file.reader
        keep = r.position
        r.seek(canvas._png_offset)
        canvas._png_data = r.read(canvas._png_length)
        r.seek(keep)
        return canvas._png_data


def _try_listwz(raw: bytes, key: WzKey) -> bytes:
    """Dechunk a listWz blob and return the inner deflate payload bytes."""
    src = io.BytesIO(raw)
    out = bytearray()
    while True:
        chunk_len_bytes = src.read(4)
        if len(chunk_len_bytes) < 4:
            break
        chunk_len = int.from_bytes(chunk_len_bytes, "little")
        if chunk_len <= 0 or chunk_len > len(raw):
            raise ValueError("invalid listWz chunk length")
        chunk = src.read(chunk_len)
        if len(chunk) < chunk_len:
            raise ValueError("truncated listWz chunk")
        key.ensure(chunk_len)
        # bytes-level XOR is much faster than a Python comprehension.
        n = int.from_bytes(chunk, "big") ^ int.from_bytes(key.slice(chunk_len), "big")
        out += n.to_bytes(chunk_len, "big")
    return bytes(out)


def _zlib_lenient(data: bytes, wbits: int = zlib.MAX_WBITS) -> bytes:
    """``decompressobj`` + ``flush`` — handles trailing data gracefully."""
    d = zlib.decompressobj(wbits)
    out = d.decompress(data) + d.flush()
    return out


def _decompress(canvas: "WzCanvasProperty", key: WzKey) -> bytes:
    raw = _read_canvas_bytes(canvas)
    if len(raw) < 2:
        return b""

    attempts = (
        ("zlib", lambda: _zlib_lenient(raw)),
        ("raw-deflate-skip2", lambda: _zlib_lenient(raw[2:], wbits=-zlib.MAX_WBITS)),
        ("listwz-zlib", lambda: _zlib_lenient(_try_listwz(raw, key))),
        (
            "listwz-raw-deflate-skip2",
            lambda: _zlib_lenient(_try_listwz(raw, key)[2:], wbits=-zlib.MAX_WBITS),
        ),
    )

    errors = []
    for name, fn in attempts:
        try:
            result = fn()
            if result:
                return result
            errors.append(f"{name}: empty result")
        except Exception as exc:  # noqa: BLE001 — we want to try the next one
            errors.append(f"{name}: {exc}")

    head = raw[:16].hex()
    raise ValueError(
        f"no decoder succeeded (first 16 bytes: {head}); attempts: " + " | ".join(errors)
    )


def _decode_pixels(data: bytes, width: int, height: int, fmt: int) -> Image.Image:
    """Convert raw decompressed pixels to a PIL image."""
    if fmt == 1:
        # ARGB4444, 2 bytes per pixel
        out = bytearray(width * height * 4)
        for i in range(width * height):
            lo = data[i * 2]
            hi = data[i * 2 + 1]
            b = (lo & 0x0F) | ((lo & 0x0F) << 4)
            g = (lo & 0xF0) | ((lo & 0xF0) >> 4)
            r = (hi & 0x0F) | ((hi & 0x0F) << 4)
            a = (hi & 0xF0) | ((hi & 0xF0) >> 4)
            out[i * 4 + 0] = r
            out[i * 4 + 1] = g
            out[i * 4 + 2] = b
            out[i * 4 + 3] = a
        return Image.frombytes("RGBA", (width, height), bytes(out))
    if fmt == 2:
        # ARGB8888 stored as BGRA on disk
        return Image.frombytes("RGBA", (width, height), data, "raw", "BGRA")
    if fmt == 3:
        # downsampled (each 4x4 block stores one ARGB8888 pixel = 4 bytes)
        small_w = (width + 3) // 4
        small_h = (height + 3) // 4
        small = Image.frombytes(
            "RGBA", (small_w, small_h), data, "raw", "BGRA",
        )
        return small.resize((width, height), Image.NEAREST)
    if fmt == 257:
        # ARGB1555 — uncommon but documented
        out = bytearray(width * height * 4)
        for i in range(width * height):
            v = data[i * 2] | (data[i * 2 + 1] << 8)
            a = 0xFF if v & 0x8000 else 0x00
            r = ((v >> 10) & 0x1F) * 8
            g = ((v >> 5) & 0x1F) * 8
            b = (v & 0x1F) * 8
            out[i * 4:i * 4 + 4] = bytes([r, g, b, a])
        return Image.frombytes("RGBA", (width, height), bytes(out))
    if fmt == 513:
        # RGB565
        out = bytearray(width * height * 4)
        for i in range(width * height):
            v = data[i * 2] | (data[i * 2 + 1] << 8)
            r = ((v >> 11) & 0x1F) * 8
            g = ((v >> 5) & 0x3F) * 4
            b = (v & 0x1F) * 8
            out[i * 4:i * 4 + 4] = bytes([r, g, b, 0xFF])
        return Image.frombytes("RGBA", (width, height), bytes(out))
    if fmt == 517:
        # downsampled RGB565
        small_w = (width + 15) // 16
        small_h = (height + 15) // 16
        small_data = data[: small_w * small_h * 2]
        out = bytearray(small_w * small_h * 4)
        for i in range(small_w * small_h):
            v = small_data[i * 2] | (small_data[i * 2 + 1] << 8)
            r = ((v >> 11) & 0x1F) * 8
            g = ((v >> 5) & 0x3F) * 4
            b = (v & 0x1F) * 8
            out[i * 4:i * 4 + 4] = bytes([r, g, b, 0xFF])
        small = Image.frombytes("RGBA", (small_w, small_h), bytes(out))
        return small.resize((width, height), Image.NEAREST)
    if fmt == 1026:
        return _decode_dxt3(data, width, height)
    if fmt == 2050:
        return _decode_dxt5(data, width, height)
    if fmt == 4098:
        # BC7 / BPTC — newer 64-bit clients (see _decode_bc7).
        return _decode_bc7(data, width, height)
    raise ValueError(f"unsupported canvas format {fmt}")


# ── DXT (BC2/BC3) block decoders ─────────────────────────────────────
# Each block covers a 4×4 pixel tile. Format reference: Microsoft BCn docs
# and ``MapleLib/WzLib/WzProperties/WzPngProperty.cs``.
#
# Vectorized via NumPy. The pure-Python implementations were ~580 ms on
# a 1368×720 DXT5 (~62k blocks × 16 inner-loop iterations); NumPy moves
# everything into C-level array math so the same image decodes in
# tens of milliseconds.

import numpy as np


def _decode_dxt_color_palette(color_block: np.ndarray) -> np.ndarray:
    """Given an ``(N, 8)`` array of color blocks, return ``(N, 4, 3)``
    of uint8 RGB palette entries. Mirrors the BC2/BC3 4-color table
    (which is unconditional, unlike DXT1 where ``c0 <= c1`` swaps to
    a 3-color + 1-bit-alpha mode)."""
    c0 = color_block[:, 0].astype(np.int32) | (color_block[:, 1].astype(np.int32) << 8)
    c1 = color_block[:, 2].astype(np.int32) | (color_block[:, 3].astype(np.int32) << 8)

    def unpack(c):
        r = ((c >> 11) & 0x1F) * 255 // 31
        g = ((c >> 5) & 0x3F) * 255 // 63
        b = (c & 0x1F) * 255 // 31
        return r, g, b

    r0, g0, b0 = unpack(c0)
    r1, g1, b1 = unpack(c1)
    n = color_block.shape[0]
    pal = np.empty((n, 4, 3), dtype=np.uint8)
    pal[:, 0, 0] = r0; pal[:, 0, 1] = g0; pal[:, 0, 2] = b0
    pal[:, 1, 0] = r1; pal[:, 1, 1] = g1; pal[:, 1, 2] = b1
    pal[:, 2, 0] = (2 * r0 + r1 + 1) // 3
    pal[:, 2, 1] = (2 * g0 + g1 + 1) // 3
    pal[:, 2, 2] = (2 * b0 + b1 + 1) // 3
    pal[:, 3, 0] = (r0 + 2 * r1 + 1) // 3
    pal[:, 3, 1] = (g0 + 2 * g1 + 1) // 3
    pal[:, 3, 2] = (b0 + 2 * b1 + 1) // 3
    return pal


def _dxt_color_pixels(color_block: np.ndarray) -> np.ndarray:
    """``(N, 8)`` color blocks → ``(N, 16, 3)`` RGB pixels."""
    pal = _decode_dxt_color_palette(color_block)  # (N, 4, 3)
    bits = (
        color_block[:, 4].astype(np.uint32)
        | (color_block[:, 5].astype(np.uint32) << 8)
        | (color_block[:, 6].astype(np.uint32) << 16)
        | (color_block[:, 7].astype(np.uint32) << 24)
    )
    # 16 indices, 2 bits each, ordered (py, px) inside the 4×4 tile.
    shifts = (np.arange(16, dtype=np.uint32) * 2)
    indices = ((bits[:, None] >> shifts[None, :]) & 0x3).astype(np.intp)  # (N, 16)
    n = color_block.shape[0]
    rows = np.arange(n)[:, None]
    return pal[rows, indices]  # (N, 16, 3)


def _blocks_to_image(rgba_blocks: np.ndarray, width: int, height: int) -> Image.Image:
    """Reshape ``(N, 16, 4)`` block-major RGBA pixels back into a
    ``(height, width, 4)`` image, cropping to (width, height)."""
    bh = (height + 3) // 4
    bw = (width + 3) // 4
    grid = rgba_blocks.reshape(bh, bw, 4, 4, 4)              # (bh, bw, py, px, ch)
    img = grid.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    img = img[:height, :width]
    return Image.frombytes("RGBA", (width, height), img.tobytes())


def _decode_dxt3(data: bytes, width: int, height: int) -> Image.Image:
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    n = bh * bw
    blocks = np.frombuffer(data, dtype=np.uint8, count=n * 16).reshape(n, 16)

    # Color (8 bytes) → 16 RGB pixels per block
    rgb = _dxt_color_pixels(blocks[:, 8:16])  # (N, 16, 3)

    # Alpha (8 bytes) — 4 bits per pixel, low nibble of byte 2*py is the
    # left-most pixel of row py.
    alpha_bytes = blocks[:, 0:8]                                # (N, 8)
    # Each pair of bytes encodes a row of 4 pixels: bits low→high.
    rows = alpha_bytes.astype(np.uint16).reshape(n, 4, 2)
    row_packed = rows[..., 0] | (rows[..., 1] << 8)             # (N, 4)
    px_shifts = (np.arange(4, dtype=np.uint16) * 4)
    alpha4 = (row_packed[..., None] >> px_shifts[None, None, :]) & 0xF   # (N, 4, 4)
    alpha = ((alpha4 << 4) | alpha4).astype(np.uint8).reshape(n, 16)

    rgba = np.empty((n, 16, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return _blocks_to_image(rgba, width, height)


def _decode_dxt5(data: bytes, width: int, height: int) -> Image.Image:
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    n = bh * bw
    blocks = np.frombuffer(data, dtype=np.uint8, count=n * 16).reshape(n, 16)

    a0 = blocks[:, 0].astype(np.int32)
    a1 = blocks[:, 1].astype(np.int32)

    # Build the 8-entry alpha palette per block. The "8-color" mode
    # (a0 > a1) interpolates 6 intermediate values; the "6-color" mode
    # interpolates 4 + plus 0 and 255 sentinels.
    alpha_pal = np.empty((n, 8), dtype=np.int32)
    alpha_pal[:, 0] = a0
    alpha_pal[:, 1] = a1
    mask8 = a0 > a1                                              # (N,)
    for i in range(1, 7):
        alpha_pal[:, 1 + i] = np.where(
            mask8,
            ((7 - i) * a0 + i * a1 + 3) // 7,
            alpha_pal[:, 1 + i],     # ignored on this branch — overwritten below
        )
    for i in range(1, 5):
        v = ((5 - i) * a0 + i * a1 + 2) // 5
        alpha_pal[~mask8, 1 + i] = v[~mask8]
    alpha_pal[~mask8, 6] = 0
    alpha_pal[~mask8, 7] = 255
    alpha_pal_u8 = alpha_pal.astype(np.uint8)

    # 48-bit alpha index field as one uint64 per block.
    ai = np.zeros(n, dtype=np.uint64)
    for k in range(6):
        ai |= blocks[:, 2 + k].astype(np.uint64) << (np.uint64(8 * k))
    px_shifts = (np.arange(16, dtype=np.uint64) * np.uint64(3))
    alpha_idx = ((ai[:, None] >> px_shifts[None, :]) & np.uint64(0x7)).astype(np.intp)
    alpha = np.take_along_axis(alpha_pal_u8, alpha_idx, axis=1)  # (N, 16)

    rgb = _dxt_color_pixels(blocks[:, 8:16])

    rgba = np.empty((n, 16, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return _blocks_to_image(rgba, width, height)


# ── BC7 (BPTC) block decoder ──────────────────────────────────────────
# Canvas format 4098 (0x1002 = format1 2 + (format2 16 << 8)). MapleStory's
# newer 64-bit clients store some canvases as DXGI_FORMAT_BC7_UNORM (BPTC).
# Like DXT3/DXT5 each 16-byte block covers a 4×4 tile, but BC7 packs RGBA
# together under one of 8 modes (partitions, per-endpoint P-bits, channel
# rotation, two index sets). Reference: the Khronos Data Format Spec §BPTC
# and Microsoft's BC7 documentation; the partition / anchor tables below are
# the canonical ones shared by every BC7 implementation.

# Index-interpolation weights, keyed by index bit-width.
_BC7_WEIGHTS = {
    2: (0, 21, 43, 64),
    3: (0, 9, 18, 27, 37, 46, 55, 64),
    4: (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64),
}

# Per-mode parameters:
# (subsets, partition_bits, rotation_bits, idxsel_bits, color_bits,
#  alpha_bits, endpoint_pbits, shared_pbits, index_bits, index_bits2)
_BC7_MODES = (
    (3, 4, 0, 0, 4, 0, 1, 0, 3, 0),  # 0
    (2, 6, 0, 0, 6, 0, 0, 1, 3, 0),  # 1
    (3, 6, 0, 0, 5, 0, 0, 0, 2, 0),  # 2
    (2, 6, 0, 0, 7, 0, 1, 0, 2, 0),  # 3
    (1, 0, 2, 1, 5, 6, 0, 0, 2, 3),  # 4
    (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),  # 5
    (1, 0, 0, 0, 7, 7, 1, 0, 4, 0),  # 6
    (2, 6, 0, 0, 5, 5, 1, 0, 2, 0),  # 7
)

# 2-subset partition table: 64 partitions × 16 pixels (subset id 0/1).
_BC7_PART2_STR = (
    "0011001100110011", "0001000100010001", "0111011101110111", "0001001100110111",
    "0000000100010011", "0011011101111111", "0001001101111111", "0000000100110111",
    "0000000000010011", "0011011111111111", "0000000101111111", "0000000000010111",
    "0001011111111111", "0000000011111111", "0000111111111111", "0000000000001111",
    "0000100011101111", "0111000100000000", "0000000010001110", "0111001100010000",
    "0011000100000000", "0000100011001110", "0000000010001100", "0111001100110001",
    "0011000100010000", "0000100010001100", "0110011001100110", "0011011001101100",
    "0001011111101000", "0000111111110000", "0111000110001110", "0011100110011100",
    "0101010101010101", "0000111100001111", "0101101001011010", "0011001111001100",
    "0011110000111100", "0101010110101010", "0110100101101001", "0101101010100101",
    "0111001111001110", "0001001111001000", "0011001001001100", "0011101111011100",
    "0110100110010110", "0011110011000011", "0110011010011001", "0000011001100000",
    "0100111001000000", "0010011100100000", "0000001001110010", "0000010011100100",
    "0110110010010011", "0011011011001001", "0110001110011100", "0011100111000110",
    "0110110011001001", "0110001100111001", "0111111010000001", "0001100011100111",
    "0000111100110011", "0011001111110000", "0010001011101110", "0100010001110111",
)

# 3-subset partition table: 64 partitions × 16 pixels (subset id 0/1/2).
_BC7_PART3_STR = (
    "0011001102212222", "0001001122112221", "0000200122112211", "0222002200110111",
    "0000000011221122", "0011001100220022", "0022002211111111", "0011001122112211",
    "0000000011112222", "0000111111112222", "0000111122222222", "0012001200120012",
    "0112011201120112", "0122012201220122", "0011011211221222", "0011200122002220",
    "0001001101121122", "0111001120012200", "0000112211221122", "0022002200221111",
    "0111011102220222", "0001000122212221", "0000001101220122", "0000110022102210",
    "0122012200110000", "0012001211222222", "0110122112210110", "0000011012211221",
    "0022110211020022", "0110011020022222", "0011012201220011", "0000200022112221",
    "0000000211221222", "0222002200120011", "0011001200220222", "0120012001200120",
    "0000111122220000", "0120120120120120", "0120201212010120", "0011220011220011",
    "0011112222000011", "0101010122222222", "0000000021212121", "0022112200221122",
    "0022001100220011", "0220122102201221", "0101222222220101", "0000212121212121",
    "0101010101012222", "0222011102220111", "0002111200021112", "0000211221122112",
    "0222011101110222", "0002111211120002", "0110011001102222", "0000000021122112",
    "0110011022222222", "0022001100110022", "0022112211220022", "0000000000002112",
    "0002000100020001", "0222122202221222", "0101222222222222", "0111201122012220",
)

_BC7_PART2 = tuple(tuple(int(c) for c in s) for s in _BC7_PART2_STR)
_BC7_PART3 = tuple(tuple(int(c) for c in s) for s in _BC7_PART3_STR)

# Anchor (fixup) pixel index per subset. Subset 0's anchor is always pixel 0.
_BC7_ANCHOR2 = (
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
)
_BC7_ANCHOR3_2 = (
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3,
    3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15,
    3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
)
_BC7_ANCHOR3_3 = (
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8,
    15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8,
    15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
)

assert len(_BC7_PART2) == 64 and all(len(p) == 16 for p in _BC7_PART2)
assert len(_BC7_PART3) == 64 and all(len(p) == 16 for p in _BC7_PART3)


def _bc7_unquant(value: int, bits: int) -> int:
    """Expand a ``bits``-wide quantized channel to 8 bits by replication."""
    value <<= (8 - bits)
    value |= value >> bits
    return value & 0xFF


def _decode_bc7_block(block: bytes) -> list:
    """Decode one 16-byte BC7 block into 16 ``(r, g, b, a)`` tuples in
    pixel order (row-major within the 4×4 tile)."""
    v = int.from_bytes(block, "little")
    if v == 0:
        # Mode would be "8" (reserved); the spec leaves the result
        # undefined — emit transparent black, matching reference decoders.
        return [(0, 0, 0, 0)] * 16
    mode = (v & -v).bit_length() - 1  # trailing-zero count = unary mode
    if mode > 7:
        return [(0, 0, 0, 0)] * 16
    ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 = _BC7_MODES[mode]

    pos = mode + 1

    def take(n: int) -> int:
        nonlocal pos
        if n == 0:
            return 0
        r = (v >> pos) & ((1 << n) - 1)
        pos += n
        return r

    partition = take(pb)
    rotation = take(rb)
    idx_sel = take(isb)

    ne = ns * 2  # two endpoints per subset
    reds = [take(cb) for _ in range(ne)]
    greens = [take(cb) for _ in range(ne)]
    blues = [take(cb) for _ in range(ne)]
    alphas = [take(ab) for _ in range(ne)] if ab else None

    if epb:
        pbits = [take(1) for _ in range(ne)]
    elif spb:
        shared = [take(1) for _ in range(ns)]
        pbits = [shared[i // 2] for i in range(ne)]
    else:
        pbits = None

    # Reconstruct 8-bit RGBA endpoints (append P-bit, then bit-replicate).
    cbits = cb + (1 if pbits is not None else 0)
    abits = (ab + (1 if pbits is not None else 0)) if ab else 0
    endpoints = []
    for i in range(ne):
        r = reds[i]; g = greens[i]; b = blues[i]
        if pbits is not None:
            p = pbits[i]
            r = (r << 1) | p; g = (g << 1) | p; b = (b << 1) | p
        r = _bc7_unquant(r, cbits)
        g = _bc7_unquant(g, cbits)
        b = _bc7_unquant(b, cbits)
        if ab:
            a = alphas[i]
            if pbits is not None:
                a = (a << 1) | pbits[i]
            a = _bc7_unquant(a, abits)
        else:
            a = 255
        endpoints.append((r, g, b, a))

    # Partition → per-pixel subset id, plus that partition's anchor pixels.
    if ns == 1:
        pmap = (0,) * 16
        anchors = (0,)
    elif ns == 2:
        pmap = _BC7_PART2[partition]
        anchors = (0, _BC7_ANCHOR2[partition])
    else:
        pmap = _BC7_PART3[partition]
        anchors = (0, _BC7_ANCHOR3_2[partition], _BC7_ANCHOR3_3[partition])

    def read_indices(width: int) -> list:
        # The anchor pixel of each subset drops its high bit (stored with
        # width-1 bits); every other pixel uses the full width.
        out = [0] * 16
        for px in range(16):
            full = width if px != anchors[pmap[px]] else width - 1
            out[px] = take(full)
        return out

    idx1 = read_indices(ib)
    idx2 = read_indices(ib2) if ib2 else None

    if ib2 == 0:
        color_index = alpha_index = idx1
        color_w = alpha_w = ib
    elif mode == 4 and idx_sel == 1:
        color_index, color_w = idx2, ib2
        alpha_index, alpha_w = idx1, ib
    else:
        color_index, color_w = idx1, ib
        alpha_index, alpha_w = idx2, ib2

    cw = _BC7_WEIGHTS[color_w]
    aw = _BC7_WEIGHTS[alpha_w]

    out = []
    for px in range(16):
        s = pmap[px]
        e0 = endpoints[2 * s]; e1 = endpoints[2 * s + 1]
        wc = cw[color_index[px]]
        ic = 64 - wc
        r = (e0[0] * ic + e1[0] * wc + 32) >> 6
        g = (e0[1] * ic + e1[1] * wc + 32) >> 6
        b = (e0[2] * ic + e1[2] * wc + 32) >> 6
        wa = aw[alpha_index[px]]
        a = (e0[3] * (64 - wa) + e1[3] * wa + 32) >> 6
        if rotation == 1:
            r, a = a, r
        elif rotation == 2:
            g, a = a, g
        elif rotation == 3:
            b, a = a, b
        out.append((r, g, b, a))
    return out


def _decode_bc7(data: bytes, width: int, height: int) -> Image.Image:
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    # Block-major RGBA buffer (bh*4 rows × bw*4 cols), cropped at the end.
    full_w = bw * 4
    canvas = bytearray(full_w * bh * 4 * 4)
    bi = 0
    for by in range(bh):
        for bx in range(bw):
            block = data[bi * 16: bi * 16 + 16]
            bi += 1
            pixels = _decode_bc7_block(block)
            base_y = by * 4
            base_x = bx * 4
            k = 0
            for py in range(4):
                row = (base_y + py) * full_w + base_x
                off = row * 4
                for px in range(4):
                    r, g, b, a = pixels[k]
                    k += 1
                    canvas[off] = r
                    canvas[off + 1] = g
                    canvas[off + 2] = b
                    canvas[off + 3] = a
                    off += 4
    img = Image.frombytes("RGBA", (full_w, bh * 4), bytes(canvas))
    if (full_w, bh * 4) != (width, height):
        img = img.crop((0, 0, width, height))
    return img


def decode_canvas(canvas: "WzCanvasProperty", region: str = "GMS") -> Image.Image:
    """Decode the canvas's pixels into a PIL ``Image`` (RGBA).

    ``region`` is needed because listWz blocks XOR against the same regional
    key stream used for strings.
    """
    fmt = canvas.format + canvas.format2
    key = WzKey.for_region(region)
    raw = _decompress(canvas, key)
    return _decode_pixels(raw, canvas.width, canvas.height, fmt)


# ── HSV recolor (custom hair colour) ───────────────────────────────────────
#
# The character compositor recolors the *red* hair variant to an arbitrary
# custom colour by replacing each pixel's hue and scaling its saturation /
# value. Red is the base because it has full chroma to shift; black (the stock
# default) has none, so its hue can't be moved. Applied per-canvas before the
# z-ordered alpha composite, so it reaches every pose identically.


def apply_hsv_adjust(
    img: Image.Image, hue: float, sat: float, val: float,
) -> Image.Image:
    """Recolor every pixel of an RGBA hair image: set its hue to ``hue``
    (degrees, wrapped to ``[0, 360)``), scale its saturation by ``sat`` and
    value by ``val`` (each clamped to ``[0, 1]``). Alpha is preserved, so
    anti-aliased edges keep their coverage. Near-grey pixels (white highlights,
    black outlines) have ~zero chroma, so the hue swap leaves them neutral —
    only the coloured strands shift. Vectorized with numpy; returns a new RGBA
    ``Image``. Defaults (``sat == val == 1.0``, ``hue`` = the base's own hue)
    leave the image unchanged.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.asarray(img, dtype=np.float32)
    rgb = arr[..., :3] / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    # value = max; saturation = chroma / max (0 where max is ~0).
    s = np.where(mx <= 1e-6, 0.0, d / np.where(mx <= 1e-6, 1.0, mx))
    v = mx

    h = float(hue) % 360.0
    ns = np.clip(s * float(sat), 0.0, 1.0)
    nv = np.clip(v * float(val), 0.0, 1.0)

    # HSV -> RGB at the single target hue (same for every pixel).
    c = nv * ns
    hp = (h / 60.0) % 6.0
    x = c * (1.0 - abs((hp % 2.0) - 1.0))
    m = nv - c
    sector = int(hp)  # 0..5
    if sector == 0:
        r1, g1, b1 = c, x, np.zeros_like(c)
    elif sector == 1:
        r1, g1, b1 = x, c, np.zeros_like(c)
    elif sector == 2:
        r1, g1, b1 = np.zeros_like(c), c, x
    elif sector == 3:
        r1, g1, b1 = np.zeros_like(c), x, c
    elif sector == 4:
        r1, g1, b1 = x, np.zeros_like(c), c
    else:
        r1, g1, b1 = c, np.zeros_like(c), x

    out = np.empty_like(arr)
    out[..., 0] = np.clip(np.rint((r1 + m) * 255.0), 0, 255)
    out[..., 1] = np.clip(np.rint((g1 + m) * 255.0), 0, 255)
    out[..., 2] = np.clip(np.rint((b1 + m) * 255.0), 0, 255)
    out[..., 3] = arr[..., 3]  # alpha untouched
    return Image.fromarray(out.astype(np.uint8), "RGBA")


# ── pixel encoders (inverse of _decode_pixels) ─────────────────────────
# Used by the canvas-replacement save path. We support the same formats
# that the decoder supports for ARGB/RGB565 family. The block-compressed
# formats (DXT3/DXT5/BC7) are intentionally refused because re-encoding
# requires a full block compressor, which is out of scope for v1.

_BLOCK_COMPRESSED_FORMATS = (1026, 2050, 4098)


def _encode_pixels(image: Image.Image, fmt: int, width: int, height: int) -> bytes:
    """Pack ``image`` into the raw pixel-bytes representation that
    :func:`_decode_pixels` would produce in reverse. Returns the bytes
    that go into the zlib stream.
    """
    if fmt in _BLOCK_COMPRESSED_FORMATS:
        kind = "BC7" if fmt == 4098 else "DXT"
        raise ValueError(
            f"writing canvas format {fmt} ({kind}) is not supported; pick a "
            f"different sprite or convert the format externally"
        )
    if image.size != (width, height):
        # Caller may want to preserve resolution; resize bilinearly.
        image = image.resize((width, height), Image.LANCZOS)
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    if fmt == 1:
        # ARGB4444: 2 bytes per pixel. High nibble of each byte encodes
        # the high 4 bits of the channel; low nibble the low 4. The
        # decoder expands a 4-bit channel ``c`` to ``(c << 4) | c``, so
        # we encode the ROUNDED top 4 bits ``(c + 8) >> 4`` clamped.
        rgba = image.tobytes("raw", "RGBA")
        out = bytearray(width * height * 2)
        for i in range(width * height):
            r = rgba[i * 4 + 0] >> 4
            g = rgba[i * 4 + 1] >> 4
            b = rgba[i * 4 + 2] >> 4
            a = rgba[i * 4 + 3] >> 4
            # Decoder: lo = b | (b << 4) for low nibble, hi nibble holds g
            #          hi = r | (r << 4) for low nibble, hi nibble holds a
            out[i * 2 + 0] = b | (g << 4)
            out[i * 2 + 1] = r | (a << 4)
        return bytes(out)

    if fmt == 2:
        # ARGB8888 stored as BGRA on disk. PIL's "raw" output with band
        # order "BGRA" gives us exactly that.
        return image.tobytes("raw", "BGRA")

    if fmt == 3:
        # Down-sampled BGRA8888: 4×4 source blocks collapse to one BGRA
        # pixel each. The decoder unpacks via NEAREST upscale.
        small_w = (width + 3) // 4
        small_h = (height + 3) // 4
        small = image.resize((small_w, small_h), Image.LANCZOS)
        return small.tobytes("raw", "BGRA")

    if fmt == 257:
        # ARGB1555: 2 bytes per pixel. 1 alpha bit + 5 each for RGB.
        rgba = image.tobytes("raw", "RGBA")
        out = bytearray(width * height * 2)
        for i in range(width * height):
            r = rgba[i * 4 + 0] >> 3
            g = rgba[i * 4 + 1] >> 3
            b = rgba[i * 4 + 2] >> 3
            a_bit = 0x8000 if rgba[i * 4 + 3] >= 128 else 0x0000
            v = a_bit | (r << 10) | (g << 5) | b
            out[i * 2 + 0] = v & 0xFF
            out[i * 2 + 1] = (v >> 8) & 0xFF
        return bytes(out)

    if fmt == 513:
        # RGB565: 5/6/5 bits — alpha discarded.
        rgb = image.convert("RGB").tobytes("raw", "RGB")
        out = bytearray(width * height * 2)
        for i in range(width * height):
            r = rgb[i * 3 + 0] >> 3
            g = rgb[i * 3 + 1] >> 2
            b = rgb[i * 3 + 2] >> 3
            v = (r << 11) | (g << 5) | b
            out[i * 2 + 0] = v & 0xFF
            out[i * 2 + 1] = (v >> 8) & 0xFF
        return bytes(out)

    if fmt == 517:
        # Down-sampled RGB565: 16×16 source blocks → one RGB565 pixel.
        small_w = (width + 15) // 16
        small_h = (height + 15) // 16
        small = image.resize((small_w, small_h), Image.LANCZOS).convert("RGB")
        rgb = small.tobytes("raw", "RGB")
        out = bytearray(small_w * small_h * 2)
        for i in range(small_w * small_h):
            r = rgb[i * 3 + 0] >> 3
            g = rgb[i * 3 + 1] >> 2
            b = rgb[i * 3 + 2] >> 3
            v = (r << 11) | (g << 5) | b
            out[i * 2 + 0] = v & 0xFF
            out[i * 2 + 1] = (v >> 8) & 0xFF
        return bytes(out)

    raise ValueError(f"unsupported canvas format {fmt} for write")


def _encode_listwz(payload: bytes, key: WzKey, chunk_size: int = 4096) -> bytes:
    """Wrap ``payload`` (already-zlib bytes) as a listWz blob: chunks of
    ``chunk_size`` bytes XORed against the WZ keystream, each preceded by
    a u32 little-endian length. Mirrors :func:`_try_listwz`.
    """
    out = bytearray()
    for start in range(0, len(payload), chunk_size):
        chunk = payload[start:start + chunk_size]
        n = len(chunk)
        key.ensure(n)
        xored = (
            int.from_bytes(chunk, "big")
            ^ int.from_bytes(key.slice(n), "big")
        )
        out += n.to_bytes(4, "little")
        out += xored.to_bytes(n, "big")
    return bytes(out)


def encode_canvas_payload(
    image: Image.Image,
    fmt: int,
    width: int,
    height: int,
    *,
    key: WzKey,
    listwz: bool = False,
    zlib_level: int = 6,
) -> bytes:
    """Full encode pipeline: pixels → zlib → optional listWz.

    Returns the bytes that should land at ``WzCanvasProperty._png_offset``
    (the caller is responsible for length-budget validation against
    ``_png_length`` and zero-padding any unused tail).
    """
    raw = _encode_pixels(image, fmt, width, height)
    compressed = zlib.compress(raw, zlib_level)
    if listwz:
        return _encode_listwz(compressed, key)
    return compressed
