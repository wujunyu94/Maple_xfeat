"""Reading **V2** MapleStory ``.ms`` packs (ChaCha20-encrypted WZ image archives).

The ``.ms`` files shipped in ``data/Packs`` (``Skill_*.ms``, ``Mob_*.ms``) are
this **V2** (``version == 4``) container, encrypted with ChaCha20. (An older
**V1** (``version == 2``, SNOW 2.0) archive format also exists — WzComparerR2's
``Ms_File.cs`` — but no files here use it.) Format ported from WzComparerR2's
``Ms_FileV2.cs`` / ``Ms_ImageV2.cs`` (credits Kagamia / Elem8100):

  1. An unencrypted prefix: ``randByteCount`` random bytes (count from the
     lowercased filename), each **arithmetic-shifted right by 1**; a version
     byte XOR ``rand[0]`` (== 4); a 4-byte obfuscated salt length; then salt
     bytes recovered via ``((a|0x4B)<<1) - a - 75`` where ``a = rand[i] ^
     saltBytes[2i]``.
  2. A ChaCha20-encrypted 8-byte header (``hash`` i32, ``entryCount`` i32),
     keyed by ``filename+salt`` (XOR a fixed 32-byte obscure mask).
  3. After ``8 + padAmount`` bytes, a ChaCha20-encrypted entry table (keyed
     differently); each record carries the image name (``Skill/xxxxx.img``),
     size, aligned size, a block index, a 16-byte per-entry key and two extra
     V2 fields. The table reader **resets the block counter every 64 bytes**
     (see :class:`~wzpy.chacha20.ChaCha20BlockReader`).
  4. Page-aligned (1024) image data. Only each image's **first 1024 bytes** are
     ChaCha20-encrypted (per-image key + nonce + counter derived from the salt
     hash, image name and entry key); the remainder is plaintext WZ.

The decrypted payload of each entry is an ordinary legacy 32-bit ``.img`` body
parsed with the BMS cipher (zero IV) by :class:`wzpy.wz_image.WzImage`. This
recovers the **complete** skill stat trees — ``common/lt``+``rb`` (attack-range
box), ``cooltime``, ``damage``, ``mpCon``, per-level blocks — with their real
property names, which the header-less body scanner in :mod:`wzpy.ms_wz` could
only partially reconstruct.

:class:`MsPackageV2` stitches every ``.ms`` in a folder into one virtual
:class:`~wzpy.wz_file.WzDirectory` tree, the same read surface as
:class:`wzpy.wz_package.WzPackage`.
"""

from __future__ import annotations

import io
import mmap
import os
import re
import struct
import threading
from dataclasses import dataclass
from typing import List, Optional

from .chacha20 import ChaCha20BlockReader, chacha20_xor
from .crypto import WzKey
from .reader import WzBinaryReader
from .wz_file import WzDirectory
from .wz_image import WzImage


# ── format constants (Ms_FileV2.cs) ─────────────────────────────────────
_SUPPORTED_VERSION = 4
_CHACHA_KEY_LEN = 32
_BLOCK_ALIGNMENT = 1024
_PAGE_MASK = 0x3FF
_RAND_BYTE_MOD = 312
_RAND_BYTE_OFFSET = 30
_HEADER_PAD_MOD = 212
_HEADER_PAD_OFFSET = 64
_INITIAL_KEY_HASH = 0x811C9DC5
_KEY_HASH_MULTIPLIER = 0x1000193
_DOUBLE_ENCRYPT_INITIAL_BYTES = 1024
_MASK = 0xFFFFFFFF
_ZERO_NONCE = bytes(12)

# Fixed 32-byte mask XORed into every ChaCha20 key (Ms_FileV2.chacha20KeyObscure).
_CHACHA20_KEY_OBSCURE = bytes([
    0x7B, 0x2F, 0x35, 0x48, 0x43, 0x95, 0x02, 0xB9,
    0xAE, 0x91, 0xA6, 0xE1, 0xD8, 0xD6, 0x24, 0xB4,
    0x33, 0x10, 0x1D, 0x3D, 0xC1, 0xBB, 0xC6, 0xF4,
    0xA5, 0xFE, 0xB3, 0x69, 0x6B, 0x56, 0xE4, 0x75,
])


def _align_page(pos: int) -> int:
    return (pos + _PAGE_MASK) & ~_PAGE_MASK


# ── entry metadata ──────────────────────────────────────────────────────
@dataclass
class MsEntryV2:
    name: str          # e.g. "Skill/112.img"
    check_sum: int
    flags: int
    block_index: int   # block offset within the data region
    size: int          # decrypted (== on-disk) payload byte count
    size_aligned: int  # payload footprint rounded up to 1024
    unk1: int
    unk2: int
    entry_key: bytes   # 16-byte per-entry key
    unk3: int          # V2-only
    unk4: int          # V2-only
    start_pos: int = 0  # absolute file offset of the payload (filled in later)


# ── key derivation (Ms_FileV2.cs / Ms_ImageV2.cs) ───────────────────────
def _obscure(key: bytearray) -> bytes:
    for i in range(_CHACHA_KEY_LEN):
        key[i] ^= _CHACHA20_KEY_OBSCURE[i]
    return bytes(key)


def _derive_header_key(name_with_salt: str) -> bytes:
    n = len(name_with_salt)
    key = bytearray(_CHACHA_KEY_LEN)
    for i in range(_CHACHA_KEY_LEN):
        key[i] = (ord(name_with_salt[i % n]) + i) & 0xFF
    return _obscure(key)


def _derive_entry_key(name_with_salt: str) -> bytes:
    n = len(name_with_salt)
    key = bytearray(_CHACHA_KEY_LEN)
    for i in range(_CHACHA_KEY_LEN):
        ch = ord(name_with_salt[n - 1 - i % n])
        key[i] = (i + (i % 3 + 2) * ch) & 0xFF
    return _obscure(key)


def _derive_img_key_nonce_counter(salt: str, entry: MsEntryV2):
    """Per-image ChaCha20 (key, nonce, counter) for the first-1024-byte block."""
    key_hash = _INITIAL_KEY_HASH
    for ch in salt:
        key_hash = ((key_hash ^ ord(ch)) * _KEY_HASH_MULTIPLIER) & _MASK
    key_hash2 = key_hash >> 1
    key_hash3 = key_hash2 ^ 0x6C
    digits = [ord(c) - 48 for c in str(key_hash)]
    nd = len(digits)
    name = entry.name
    nn = len(name)
    ek = entry.entry_key
    ne = len(ek)

    img_key = bytearray(_CHACHA_KEY_LEN)
    for i in range(_CHACHA_KEY_LEN):
        img_key[i] = (
            i + ord(name[i % nn]) * (
                (digits[i % nd] % 2)
                + ek[(digits[(i + 2) % nd] + i) % ne]
                + ((digits[(i + 1) % nd] + i) % 5)
            )
        ) & 0xFF
    key = _obscure(img_key)

    khd = bytearray(struct.pack("<III", key_hash, key_hash2, key_hash3))
    a = b = 0
    c = 90
    d = 0
    for i in range(12):
        khd[i] = (khd[i] ^ (d + 11 * (i // 11) + (c ^ (i >> 2)) + (a ^ b))) & 0xFF
        d -= 1
        a += 8
        b += 17
        c += 43
    nonce = b"\x00\x00\x00\x00" + bytes(khd[0:8])
    counter = struct.unpack("<I", bytes(khd[8:12]))[0]
    return key, nonce, counter


# ── single V2 .ms file ──────────────────────────────────────────────────
class MsFileV2:
    """A single parsed V2 (ChaCha20) ``.ms`` archive.

    Reads header + entry table eagerly; image payloads are decrypted on demand
    via :meth:`read_entry_data`. ``region`` is fixed to ``BMS`` (zero IV) since
    ``.ms`` payloads are BMS-encrypted WZ images.
    """

    def __init__(self, path: str):
        self.path = path
        self.file_name = os.path.basename(path)
        self.salt: str = ""
        self.name_with_salt: str = ""
        self.entry_count: int = 0
        self.header_hash: int = 0
        self.data_start: int = 0
        self.entries: List[MsEntryV2] = []
        self._mmap: Optional[mmap.mmap] = None
        self._fp = None
        self._entry_start: int = 0
        self._lock = threading.RLock()

    # ── lifecycle ───────────────────────────────────────────────────────
    @classmethod
    def open(cls, path: str) -> "MsFileV2":
        ms = cls(path)
        ms._load()
        return ms

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "MsFileV2":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── header + entry parsing ──────────────────────────────────────────
    def _load(self) -> None:
        self._fp = open(self.path, "rb")
        self._mmap = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        self._read_header()
        self._read_entries()

    def _read_header(self) -> None:
        mm = self._mmap
        file_name = self.file_name.lower()

        rand_count = sum(ord(c) for c in file_name) % _RAND_BYTE_MOD + _RAND_BYTE_OFFSET
        pos = 0
        rand = bytearray(mm[pos:pos + rand_count])
        pos += rand_count
        # arithmetic shift right by 1 (signed byte)
        for i in range(len(rand)):
            v = rand[i]
            rand[i] = ((v - 256 if v >= 128 else v) >> 1) & 0xFF

        version = mm[pos] ^ rand[0]
        pos += 1
        if version != _SUPPORTED_VERSION:
            raise ValueError(
                f"{self.file_name}: decrypted .ms header version is {version}, "
                f"expected {_SUPPORTED_VERSION} (this is the V2/ChaCha20 reader; "
                f"a version-2 SNOW2 archive is a different, unsupported format)."
            )

        hashed_salt_len = int.from_bytes(mm[pos:pos + 4], "little", signed=True)
        pos += 4
        salt_len = (hashed_salt_len & 0xFF) ^ rand[0]
        if salt_len <= 0 or salt_len > 4096:
            raise ValueError(
                f"{self.file_name}: implausible salt length {salt_len} "
                "(not a V2 .ms file or wrong filename?)"
            )
        salt_bytes = mm[pos:pos + salt_len * 2]
        pos += salt_len * 2
        chars = []
        for i in range(salt_len):
            a = rand[i] ^ salt_bytes[i * 2]
            chars.append(chr((((a | 0x4B) << 1) - a - 75) & 0xFFFF))
        self.salt = "".join(chars)
        self.name_with_salt = file_name + self.salt

        header_start = pos
        header_key = _derive_header_key(self.name_with_salt)
        hdr = chacha20_xor(header_key, _ZERO_NONCE, 0,
                           bytes(mm[header_start:header_start + 8]))
        header_hash, entry_count = struct.unpack("<ii", hdr)
        if entry_count < 0 or entry_count > 1_000_000:
            raise ValueError(
                f"{self.file_name}: implausible entry count {entry_count} "
                "(wrong key / not a V2 .ms file?)"
            )
        self.header_hash = header_hash
        self.entry_count = entry_count

        pad = sum(ord(c) * 3 for c in file_name) % _HEADER_PAD_MOD + _HEADER_PAD_OFFSET
        self._entry_start = header_start + 8 + pad

    def _read_entries(self) -> None:
        if self.entry_count == 0:
            self.data_start = _align_page(self._entry_start)
            return
        mm = self._mmap
        entry_key = _derive_entry_key(self.name_with_salt)
        # The table is < entry_count * (56 + 2*name) bytes; names are short, so
        # 1 KB/entry is a generous upper bound. The slice is clamped to EOF.
        approx = self.entry_count * 1024 + 8192
        raw = bytes(mm[self._entry_start:self._entry_start + approx])
        rdr = ChaCha20BlockReader(raw, entry_key, _ZERO_NONCE)

        entries: List[MsEntryV2] = []
        for _ in range(self.entry_count):
            name = rdr.read_utf16_string()
            check_sum = rdr.read_i32()
            flags = rdr.read_i32()
            block_index = rdr.read_i32()
            size = rdr.read_i32()
            size_aligned = rdr.read_i32()
            unk1 = rdr.read_i32()
            unk2 = rdr.read_i32()
            ek = rdr.read_bytes(_CHACHA_KEY_LEN // 2)   # 16-byte per-entry key
            unk3 = rdr.read_i32()
            unk4 = rdr.read_i32()
            entries.append(MsEntryV2(
                name=name, check_sum=check_sum, flags=flags,
                block_index=block_index, size=size, size_aligned=size_aligned,
                unk1=unk1, unk2=unk2, entry_key=ek, unk3=unk3, unk4=unk4,
            ))

        # Data starts at the page boundary after the (64-byte-chunked) table.
        self.data_start = _align_page(self._entry_start + rdr.pos)
        for e in entries:
            e.start_pos = self.data_start + e.block_index * _BLOCK_ALIGNMENT
        self.entries = entries

    # ── payload decryption ──────────────────────────────────────────────
    def read_entry_data(self, entry: MsEntryV2) -> bytes:
        """Decrypt and return the raw ``.img`` bytes for ``entry``.

        Only the first 1024 bytes are ChaCha20-encrypted; the remainder is
        plaintext. Result length == ``entry.size``.
        """
        key, nonce, counter = _derive_img_key_nonce_counter(self.salt, entry)
        size = entry.size
        start = entry.start_pos
        n1 = min(size, _DOUBLE_ENCRYPT_INITIAL_BYTES)
        with self._lock:
            head_cipher = bytes(self._mmap[start:start + n1])
            tail = (bytes(self._mmap[start + _DOUBLE_ENCRYPT_INITIAL_BYTES:start + size])
                    if size > _DOUBLE_ENCRYPT_INITIAL_BYTES else b"")
        head = chacha20_xor(key, nonce, counter, head_cipher)
        if not tail:
            return head[:size]
        return head + tail


# ── shared pack plumbing ─────────────────────────────────────────────────
class _MsImageFile:
    """Per-image shim presenting the ``wz_file`` surface :class:`WzImage`
    needs (``reader`` + ``reader_lock``), decrypting the payload lazily on the
    first parse and caching the resulting reader."""

    __slots__ = ("_ms", "_entry", "_region", "_reader", "reader_lock")

    def __init__(self, ms: "MsFileV2", entry: MsEntryV2, region: str):
        self._ms = ms
        self._entry = entry
        self._region = region
        self._reader: Optional[WzBinaryReader] = None
        self.reader_lock = threading.RLock()

    @property
    def reader(self) -> WzBinaryReader:
        r = self._reader
        if r is not None:
            return r
        with self.reader_lock:
            if self._reader is None:
                data = self._ms.read_entry_data(self._entry)
                self._reader = WzBinaryReader(
                    io.BytesIO(data), WzKey.for_region(self._region),
                )
            return self._reader


def _list_ms_files(path: str) -> List[str]:
    """Return the ``.ms`` files at ``path`` (a single file or a folder),
    sorted by (category, numeric index) so a category's parts load in order."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    names = [n for n in os.listdir(path) if n.lower().endswith(".ms")]
    idx_re = re.compile(r"^(.*?)_(\d+)\.ms$", re.IGNORECASE)

    def sort_key(n: str):
        m = idx_re.match(n)
        if m:
            return (m.group(1).lower(), int(m.group(2)))
        return (n.lower(), -1)

    names.sort(key=sort_key)
    return [os.path.join(path, n) for n in names]


def is_ms_path(path: str) -> bool:
    """True if ``path`` is a ``.ms`` file or a folder containing ``.ms`` files
    (used by :func:`wzpy.wz_package.open_wz` to dispatch)."""
    if os.path.isfile(path):
        return path.lower().endswith(".ms")
    if os.path.isdir(path):
        try:
            return any(n.lower().endswith(".ms") for n in os.listdir(path))
        except OSError:
            return False
    return False


# ── multi-file pack ─────────────────────────────────────────────────────
class MsPackageV2:
    """A virtual WZ tree composed of the V2 ``.ms`` files in a folder.

    Drop-in compatible with :class:`wzpy.wz_package.WzPackage` for read-only
    use: exposes ``root``, ``get``, ``region``, ``version``, ``close`` and the
    context-manager protocol. Each entry's ``Category/name.img`` is placed at
    ``root / Category / name.img``.
    """

    def __init__(self, path: str, region: str = "BMS",
                 version: Optional[int] = None):
        self.path = path
        self.region = region
        self.version = version
        self.root = WzDirectory(name="")
        self._files: List[MsFileV2] = []

    @classmethod
    def open(cls, path: str, region: str = "BMS",
             version: Optional[int] = None) -> "MsPackageV2":
        """Open every V2 ``.ms`` file at ``path`` (a folder or a single file)."""
        pkg = cls(path=path, region=region, version=version)
        pkg._load()
        return pkg

    def close(self) -> None:
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = []

    def __enter__(self) -> "MsPackageV2":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get(self, path: str):
        return self.root.get(path)

    # ── load ────────────────────────────────────────────────────────────
    def _load(self) -> None:
        ms_paths = _list_ms_files(self.path)
        if not ms_paths:
            raise ValueError(f"no .ms files found at {self.path!r}")
        for p in ms_paths:
            ms = MsFileV2.open(p)
            self._files.append(ms)
            for entry in ms.entries:
                self._add_entry(ms, entry)

    def _add_entry(self, ms: MsFileV2, entry: MsEntryV2) -> None:
        parts = entry.name.replace("\\", "/").split("/")
        img_name = parts[-1]
        directory = self.root
        for seg in parts[:-1]:
            sub = directory.subdirs.get(seg)
            if sub is None:
                sub = WzDirectory(seg, parent=directory)
                directory.subdirs[seg] = sub
            directory = sub
        image = WzImage(
            name=img_name, parent=directory, offset=0, size=entry.size,
            wz_file=_MsImageFile(ms, entry, self.region),
        )
        directory.images[img_name] = image


# ── version detection ───────────────────────────────────────────────────
def detect_ms_version(path: str) -> Optional[int]:
    """Peek at a ``.ms`` file's version byte without decrypting the body.

    Returns ``4`` for a V2 (ChaCha20) archive, ``2`` for a V1 (SNOW2) archive,
    or ``None`` when the file doesn't look like either. Cheap — reads only the
    filename-derived random prefix plus one byte.
    """
    try:
        file_name = os.path.basename(path).lower()
        rand_count = sum(ord(c) for c in file_name) % _RAND_BYTE_MOD + _RAND_BYTE_OFFSET
        with open(path, "rb") as f:
            head = f.read(rand_count + 1)
        if len(head) < rand_count + 1:
            return None
        r0 = head[0]
        r0_shifted = ((r0 - 256 if r0 >= 128 else r0) >> 1) & 0xFF
        if (head[rand_count] ^ r0_shifted) == _SUPPORTED_VERSION:
            return 4
        # V1 has no leading version byte; its SNOW2 header decrypts to version 2.
        # Positively identifying V1 would need the (unsupported) SNOW2 path, so
        # only report a best-effort "maybe V1" here.
        return 2 if head[rand_count] != 0 else None
    except OSError:
        return None


def is_ms_v2(path: str) -> bool:
    """True if ``path`` is a V2 (ChaCha20, version 4) ``.ms`` archive."""
    return os.path.isfile(path) and detect_ms_version(path) == 4
