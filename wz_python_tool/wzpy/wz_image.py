"""WzImage — a single .img inside a WZ container.

The image header byte tells us whether the body is a SubProperty (the common
case) or some other container. The property tree is parsed lazily on first
access since some WZ files contain thousands of images.
"""

from __future__ import annotations

import contextlib
import io
import threading
from typing import TYPE_CHECKING, List, Optional

from .properties import (
    WzProperty,
    WzSubProperty,
    parse_property_list,
    parse_property_list_filtered,
)

if TYPE_CHECKING:
    from .crypto import WzKey
    from .wz_file import WzDirectory, WzFile


class _StandaloneWzFile:
    """Minimal stand-in for :class:`WzFile` used by :meth:`WzImage.from_bytes`.

    The .img parser only ever reaches for ``self._wz_file.reader`` while
    parsing, so a tiny shim suffices and we avoid pretending to be a full
    parsed WZ container. The lock mirrors the one on :class:`WzFile` so
    callers (parse, canvas reads) can lock uniformly without checking
    which kind of file backs the image.
    """

    __slots__ = ("reader", "reader_lock")

    def __init__(self, reader):
        self.reader = reader
        self.reader_lock = threading.RLock()


class WzImage:
    def __init__(
        self,
        name: str,
        parent: "WzDirectory",
        offset: int,
        size: int,
        wz_file: "WzFile",
    ):
        self.name = name
        self.parent = parent
        self.offset = offset
        self.size = size
        self._wz_file = wz_file
        self._parsed = False
        self._root: Optional[WzSubProperty] = None
        # Set by parse_property_list when it hits EOF mid-property — tells the
        # caller the parse returned partial results and the file is truncated.
        self.truncated: bool = False
        # Diagnostic messages from per-property parse failures that didn't
        # raise (e.g. unknown markers we now degrade gracefully). The image
        # is still usable; this just tells the UI a subtree is missing.
        self.parse_warnings: List[str] = []

    @classmethod
    def from_bytes(cls, data: bytes, *, key: "WzKey",
                   name: str = "image.img") -> "WzImage":
        """Create a standalone :class:`WzImage` from raw bytes.

        Useful for ``.img`` files extracted from a WZ container by tools
        like HaRepacker. Inside an ``.img`` the property-list offsets are
        local to the image's start (``base_offset = 0``) and the WZ-level
        ``version_hash``/``header_fstart`` are unused, so only the cipher
        ``key`` matters. Pick one with :meth:`WzKey.for_region`,
        :class:`StaticWzKey`, or :func:`detect_region_from_img`.
        """
        from .reader import WzBinaryReader
        reader = WzBinaryReader(io.BytesIO(data), key)
        return cls(name=name, parent=None, offset=0, size=len(data),
                   wz_file=_StandaloneWzFile(reader))

    @property
    def wz_file(self) -> "WzFile":
        return self._wz_file

    @property
    def path(self) -> str:
        if self.parent is None:
            return self.name
        parent_path = self.parent.path
        return f"{parent_path}/{self.name}" if parent_path else self.name

    # ── lazy parse ──────────────────────────────────────────────────
    def parse(self) -> WzSubProperty:
        # Fast path: parse already done, no lock required (write to
        # ``_parsed`` is ordered after ``_root`` below, so once we see
        # ``_parsed=True`` the root is fully populated).
        if self._parsed and self._root is not None:
            return self._root
        # Slow path: serialize on the underlying file's reader lock.
        # The ``WzBinaryReader`` cursor is shared across every WzImage
        # in the same WzFile, so two threads parsing different images
        # would otherwise interleave seeks and read each other's
        # bytes. The double-checked ``_parsed`` re-check inside the
        # lock keeps subsequent waiters from re-parsing.
        with self._wz_file.reader_lock:
            if self._parsed and self._root is not None:
                return self._root
            r = self._wz_file.reader
            r.seek(self.offset)
            tag = r.read_byte()
            # Identifier byte: 0x73 (Property/SubProperty) is by far the most
            # common. Read the type name to confirm.
            if tag != 0x73:
                # Some images don't follow this layout; fall back to empty.
                self._root = WzSubProperty(self.name)
                self._parsed = True
                return self._root
            type_name = r.read_string()
            r.skip(2)  # reserved (matches Property block)
            if type_name != "Property":
                self._root = WzSubProperty(self.name)
                self._parsed = True
                return self._root
            root = WzSubProperty(self.name)
            for child in parse_property_list(r, self.offset, root, self):
                root.add(child)
            self._root = root
            self._parsed = True
            return root

    def parse_partial(self, *, only) -> WzSubProperty:
        """Parse only the named top-level subtrees, skipping every
        other tag-9 child via its ``block_size`` prefix.

        Lets ``_read_cash_flag`` reach ``info/cash`` without walking
        the pose / frame / canvas-metadata tree that dominates parse
        cost on Weapon imgs (~5-10ms full vs. ~0.1-0.3ms partial).

        Critical invariant: this method **does not** populate
        ``self._root`` or set ``self._parsed``. The returned tree
        contains only the requested children; subsequent full
        ``parse()`` calls must still produce the complete tree, so
        leaving the cache flags untouched is essential.

        ``only`` is a frozenset of top-level child names to actually
        recurse into. Names not in the set are skipped via
        ``seek(end_pos)`` if tag-9 (extended), or fully decoded if
        they're cheap basic scalars at the same depth (tag 0/2/3/4/5/8/
        11/19/20). Re-entrant on the file's reader lock (RLock)."""
        with self._wz_file.reader_lock:
            r = self._wz_file.reader
            r.seek(self.offset)
            tag = r.read_byte()
            if tag != 0x73:
                return WzSubProperty(self.name)
            type_name = r.read_string()
            r.skip(2)
            if type_name != "Property":
                return WzSubProperty(self.name)
            root = WzSubProperty(self.name)
            for child in parse_property_list_filtered(
                r, self.offset, root, self, only=only,
            ):
                root.add(child)
            return root

    # ── access ──────────────────────────────────────────────────────
    @property
    def root(self) -> WzSubProperty:
        return self.parse()

    def children(self) -> List[WzProperty]:
        return self.root.children()

    def get(self, path: str) -> Optional[WzProperty]:
        return self.root.get(path)
