"""WzFile and WzDirectory parsing.

Parses the legacy 32-bit WZ container format (PKG1 header + nested directories
+ embedded .img files). 64-bit ``Data/`` layouts and ``.ms`` Snowcrypt packs
are out of scope for this minimal implementation but the directory model is
the same once unwrapped.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .crypto import WzKey, WZ_IV, compute_version_hash, derive_version_check
from .reader import WzBinaryReader
from .wz_image import WzImage


# ── header ─────────────────────────────────────────────────────────────
@dataclass
class WzHeader:
    ident: str
    fsize: int
    fstart: int
    copyright: str


# ── tree nodes ─────────────────────────────────────────────────────────
class WzNode:
    """Common base for directories and images.

    Held weakly in mind: every node knows its name, parent, and full path so
    the web UI can navigate without holding a separate index.
    """

    def __init__(self, name: str, parent: Optional["WzNode"] = None):
        self.name = name
        self.parent = parent

    @property
    def path(self) -> str:
        parts: List[str] = []
        node: Optional[WzNode] = self
        while node is not None and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))


class WzDirectory(WzNode):
    def __init__(self, name: str, parent: Optional[WzNode] = None):
        super().__init__(name, parent)
        self.subdirs: Dict[str, "WzDirectory"] = {}
        self.images: Dict[str, WzImage] = {}

    # ── lookup ──────────────────────────────────────────────────────
    def child(self, name: str) -> Optional[WzNode]:
        if name in self.subdirs:
            return self.subdirs[name]
        if name in self.images:
            return self.images[name]
        return None

    def get(self, path: str) -> Optional[WzNode]:
        if not path:
            return self
        node: Optional[WzNode] = self
        for part in path.split("/"):
            if part == "":
                continue
            if not isinstance(node, WzDirectory):
                return None
            node = node.child(part)
            if node is None:
                return None
        return node

    def walk_images(self, prefix: str = "") -> Iterator[Tuple[str, WzImage]]:
        """Yield ``(relative_path, image)`` for every ``.img`` in this
        directory and its subdirectories. Subdirs are walked first in
        insertion order (matching the order they appear in the WZ
        directory listing), then images at the current level."""
        for name, sub in self.subdirs.items():
            yield from sub.walk_images(f"{prefix}/{name}" if prefix else name)
        for name, img in self.images.items():
            yield (f"{prefix}/{name}" if prefix else name), img


# ── parser ─────────────────────────────────────────────────────────────
class WzFile:
    """A parsed WZ container.

    Use :py:meth:`open` to load a file; the directory tree is parsed eagerly
    (it's small) but image properties are parsed lazily on access.
    """

    def __init__(self, path: str, region: str = "GMS",
                 version: Optional[int] = None, writable: bool = False):
        self.path = path
        self.region = region
        self.version = version  # may be None until detected
        self.writable = writable
        self.header: Optional[WzHeader] = None
        self.root = WzDirectory(name="")
        self._fp = None
        self._reader: Optional[WzBinaryReader] = None
        # The underlying ``WzBinaryReader`` keeps a single position
        # cursor that's shared across every WzImage in this file (and
        # by canvas raw-byte reads). Werkzeug serves requests on
        # multiple threads, so without a lock two simultaneous parses
        # / canvas reads end up doing interleaved seeks on the same
        # mmap and one of them parses garbage. Acquire this lock for
        # any operation that moves the reader's cursor.
        self.reader_lock = threading.RLock()

    # ── lifecycle ───────────────────────────────────────────────────
    @classmethod
    def open(cls, path: str, region: str = "GMS",
             version: Optional[int] = None, writable: bool = False) -> "WzFile":
        wz = cls(path, region=region, version=version, writable=writable)
        wz._load()
        return wz

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            if self.writable:
                self._mmap.flush()
            self._mmap.close()
            self._mmap = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._reader = None

    def __enter__(self) -> "WzFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── reader access (used by lazy WzImage parsing) ────────────────
    @property
    def reader(self) -> WzBinaryReader:
        assert self._reader is not None
        return self._reader

    # ── writes (in-place value patches; see ``wzpy.writer``) ────────
    def patch_bytes(self, offset: int, data: bytes) -> None:
        """Overwrite ``len(data)`` bytes starting at file ``offset``.

        Requires ``writable=True`` at open time. Writes go through the
        memory map so subsequent reads through :pyattr:`reader` see the
        new bytes immediately. The caller is responsible for bounds and
        for keeping the encoded length unchanged — see
        :func:`wzpy.writer.encode_compressed_int` and friends, plus the
        ``_value_offset``/``_value_length`` recorded by the parser.
        """
        if not self.writable:
            raise RuntimeError(
                "WzFile was opened read-only; pass writable=True to enable patching"
            )
        end = offset + len(data)
        if offset < 0 or end > len(self._mmap):
            raise ValueError(
                f"patch range {offset}..{end} out of bounds (file is {len(self._mmap)} bytes)"
            )
        self._mmap[offset:end] = data

    def flush(self) -> None:
        """Flush any in-flight mmap writes to disk."""
        if getattr(self, "_mmap", None) is not None and self.writable:
            self._mmap.flush()

    # ── full archive re-serialization (variable-length edits) ────────
    def save_as(
        self,
        output_path: str,
        *,
        dirty_paths: Optional[set] = None,
        image_failures: Optional[List[str]] = None,
    ) -> int:
        """Re-serialize this WZ — including any edits made in memory —
        into a new file at ``output_path``. Returns the byte count
        written.

        Use this when an edit changed the byte length of a value (longer
        string, larger canvas, etc.) — those can't be patched in place
        because every downstream offset would shift. The output is a
        canonical legacy 32-bit WZ archive: same copyright, fstart,
        version hash, and region IV as the input, so the same
        ``WzFile.open(..., region=...)`` call that read the input also
        reads the output.

        Writes to ``output_path + '.tmp'`` first and then atomically
        ``os.replace``s — so a crash mid-write doesn't corrupt the
        target if it already exists.

        ``dirty_paths`` (optional): a set of property paths that have
        unsaved edits. Images containing any dirty path are
        re-serialized from the parsed tree; everything else is copied
        verbatim from the source mmap (faster + sidesteps any encoder
        gaps). When ``dirty_paths`` is omitted, every image is
        re-serialized — match the V1 behavior but slower and more
        sensitive to encoder bugs.

        ``image_failures`` (optional): if provided, populated with
        ``"<path>: <reason>"`` strings for non-dirty images that
        couldn't be re-serialized and were preserved by verbatim copy.
        """
        import os as _os
        from . import writer as _writer

        is_image_dirty = None
        if dirty_paths is not None:
            # Match by image-path prefix: an image whose name appears as
            # a path component in a dirty entry has at least one staged
            # edit underneath it.
            def is_image_dirty(image_path: str) -> bool:
                prefix = image_path + "/"
                for d in dirty_paths:
                    if d == image_path or d.startswith(prefix):
                        return True
                return False

        # Force-parse every dirty image so its in-memory tree reflects
        # staged edits. Non-dirty images are copied verbatim from the
        # source mmap, so they don't need parsing at all.
        if is_image_dirty is None:
            for _, img in self.root.walk_images():
                img.parse()
        else:
            for path, img in self.root.walk_images():
                if is_image_dirty(path):
                    img.parse()

        data = _writer.encode_wz_file(
            self,
            is_image_dirty=is_image_dirty,
            image_failures=image_failures,
        )

        tmp_path = output_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        _os.replace(tmp_path, output_path)
        return len(data)

    # ── parsing ─────────────────────────────────────────────────────
    def _load(self) -> None:
        import mmap
        self._fp = open(self.path, "r+b" if self.writable else "rb")
        # Memory-map the whole file so seeks are O(1). Critical for tree
        # browsing of multi-GB _Canvas files where each .img parse does
        # thousands of small seek+read calls. ACCESS_WRITE makes writes
        # go straight through to the file (mmap and disk stay coherent).
        access = mmap.ACCESS_WRITE if self.writable else mmap.ACCESS_READ
        self._mmap = mmap.mmap(self._fp.fileno(), 0, access=access)
        key = WzKey.for_region(self.region)
        r = WzBinaryReader(self._mmap, key)
        self._reader = r

        # header
        ident = r.read(4).decode("ascii", errors="replace")
        fsize = r.read_u64()
        fstart = r.read_u32()
        copyright_bytes = bytearray()
        while True:
            b = r.read_byte()
            if b == 0:
                break
            copyright_bytes.append(b)
            if len(copyright_bytes) > 1024:
                raise ValueError("copyright string too long; not a WZ file?")
        self.header = WzHeader(
            ident=ident,
            fsize=fsize,
            fstart=fstart,
            copyright=copyright_bytes.decode("latin-1", errors="replace"),
        )
        r.header_fstart = fstart

        # encrypted version
        encrypted_version = r.read_u16()
        body_start = r.position

        # detect version + region
        version, version_hash = self._detect_version(encrypted_version, body_start, key)
        self.version = version
        r.version_hash = version_hash

        # parse root directory
        r.seek(body_start)
        self._parse_directory(self.root)

    def _detect_version(
        self,
        encrypted_version: int,
        body_start: int,
        key: WzKey,
    ) -> Tuple[int, int]:
        # If a specific version was requested, just verify and use it.
        if self.version is not None:
            h = compute_version_hash(self.version)
            if derive_version_check(h) != encrypted_version:
                # don't fail hard — sometimes the user knows better
                pass
            return self.version, h

        # Search for any version whose check byte matches the header. For each
        # candidate, briefly try parsing a few directory entries and score by
        # how many strings decode to printable ASCII.
        candidates: List[Tuple[int, int]] = []
        for ver in range(1, 1000):
            h = compute_version_hash(ver)
            if derive_version_check(h) == encrypted_version:
                candidates.append((ver, h))

        if not candidates:
            raise ValueError("could not match WZ version (encrypted=0x%X)" % encrypted_version)

        best: Optional[Tuple[int, int, float]] = None
        for ver, h in candidates:
            score = self._score_version(body_start, h, key)
            if best is None or score > best[2]:
                best = (ver, h, score)
        assert best is not None
        return best[0], best[1]

    def _score_version(self, body_start: int, version_hash: int, key: WzKey) -> float:
        """Speculatively decode the root directory; score by printable ratio."""
        r = self._reader
        assert r is not None
        keep = r.position
        r.version_hash = version_hash
        r.seek(body_start)
        printable = 0
        total = 0
        try:
            count = r.read_compressed_int()
            count = min(count, 50)
            for _ in range(count):
                kind = r.read_byte()
                if kind == 1:
                    r.skip(10)
                    continue
                if kind == 2:
                    string_offset = r.read_i32()
                    name = r.read_string_at(body_start - 1 + string_offset)
                    new_kind = ord("?")
                    # placeholder: name decoded above
                elif kind in (3, 4):
                    name = r.read_string()
                else:
                    return -1.0
                r.read_compressed_int()  # size
                r.read_compressed_int()  # checksum
                r.read_offset()
                for ch in name:
                    total += 1
                    if 0x20 <= ord(ch) < 0x7F:
                        printable += 1
        except Exception:
            return -1.0
        finally:
            r.seek(keep)
        if total == 0:
            return 0.0
        return printable / total

    def _parse_directory(self, directory: WzDirectory) -> None:
        r = self._reader
        assert r is not None
        count = r.read_compressed_int()

        # Pass 1: collect entries (name, kind, size, offset). The entries don't
        # store nested data inline — they all point elsewhere in the file.
        entries: List[Tuple[str, int, int, int]] = []
        for _ in range(count):
            kind = r.read_byte()
            name = ""
            if kind == 1:
                # unknown, 10 bytes follow then continue
                r.skip(10)
                continue
            if kind == 2:
                string_offset = r.read_i32()
                # absolute name location: header.fstart + 1 + string_offset
                name_pos = (self.header.fstart + 1 + string_offset) & 0xFFFFFFFF
                name_kind_pos = name_pos
                # peek into the indirected entry header to get the real kind
                keep = r.position
                r.seek(name_kind_pos)
                real_kind = r.read_byte()
                name = r.read_string()
                r.seek(keep)
                kind = real_kind
            elif kind in (3, 4):
                name = r.read_string()
            else:
                # unknown — skip the rest of the directory rather than crashing
                continue

            size = r.read_compressed_int()
            checksum = r.read_compressed_int()
            offset = r.read_offset()
            entries.append((name, kind, size, offset))

        # Pass 2: recurse / register
        for name, kind, size, offset in entries:
            if kind == 3:
                sub = WzDirectory(name, parent=directory)
                directory.subdirs[name] = sub
                keep = r.position
                r.seek(offset)
                self._parse_directory(sub)
                r.seek(keep)
            elif kind == 4:
                img = WzImage(name=name, parent=directory, offset=offset, size=size, wz_file=self)
                directory.images[name] = img
