"""Hierarchical WZ folder loading.

Modern (64-bit) MapleStory ships data as a directory tree where each
subdir has a *structure* file (``Foo.wz`` containing only directory
entries) plus one or more numbered siblings (``Foo_000.wz``,
``Foo_001.wz``, ...) that hold the actual image entries:

    data/Character/
        Character.wz          # only WzDirectory entries (Cap, Coat, ...)
        Character_000.wz      # root-level .img entries
        Cap/
            Cap.wz            # only WzDirectory entries (_Canvas)
            Cap_000.wz        # the per-cap .img entries
            _Canvas/
                _Canvas.wz
                _Canvas_000.wz

Each individual ``.wz`` is a normal legacy 32-bit container — :class:`WzFile`
parses each one fine. This module stitches them into a single virtual
:class:`WzDirectory` tree by merging their roots and, for every subdir
entry seen, recursing into the matching on-disk folder.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from .properties import (
    WzCanvasProperty,
    WzProperty,
    WzStringProperty,
    WzSubProperty,
    WzUolProperty,
)
from .wz_file import WzDirectory, WzFile
from .wz_image import WzImage


# ``Foo_000.wz`` / ``Foo_12.wz`` — case-insensitive trailing index.
_INDEXED_RE = re.compile(r"_(\d+)\.wz$", re.IGNORECASE)


class WzPackage:
    """A virtual WZ tree composed of multiple hierarchical .wz files.

    Exposes the same surface that the server consumes from
    :class:`WzFile` (``root``, ``version``, ``region``, ``path``,
    ``close``, context-manager) so it can be a drop-in for read-only
    use. Writes (``save_as``, ``patch_bytes``) are not supported on a
    hierarchical pack — re-splitting an edited tree across structure
    files and indexed siblings is out of scope.
    """

    def __init__(self, path: str, region: str = "GMS",
                 version: Optional[int] = None):
        self.path = path
        self.region = region
        self.version = version
        self.root = WzDirectory(name="")
        self._files: List[WzFile] = []

    # ── lifecycle ─────────────────────────────────────────────────────
    @classmethod
    def open(cls, path: str, region: str = "GMS",
             version: Optional[int] = None,
             writable: bool = False) -> "WzPackage":
        """Open a hierarchical WZ pack rooted at ``path``.

        ``path`` may be either a folder (e.g. ``data/Character``) or
        the structure file inside such a folder
        (``data/Character/Character.wz``); the folder name / .wz stem
        is used as the base for ``<base>.wz`` / ``<base>_NNN.wz``
        matching.
        """
        folder, base_name = _resolve_root_folder(path)
        pkg = cls(path=path, region=region, version=version)
        pkg._load_into(pkg.root, folder, base_name, writable=writable)
        if pkg.version is None and pkg._files:
            pkg.version = pkg._files[0].version
        return pkg

    def close(self) -> None:
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = []

    def __enter__(self) -> "WzPackage":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── reader (compat best-effort) ────────────────────────────────────
    @property
    def reader(self):
        # No single reader — image-level code uses each WzImage's own
        # ``wz_file.reader``. Returning the first underlying file's
        # reader is a sensible best-effort for callers that only need
        # the cipher key.
        if not self._files:
            raise RuntimeError("package has no underlying WZ files")
        return self._files[0].reader

    # ── lookup ────────────────────────────────────────────────────────
    def get(self, path: str):
        return self.root.get(path)

    # ── load ──────────────────────────────────────────────────────────
    def _load_into(self, target: WzDirectory, folder: str, base_name: str,
                   *, writable: bool) -> None:
        """Open every ``base_name.wz`` / ``base_name_NNN.wz`` in
        ``folder``, merge their root directories into ``target``, then
        recurse into any subfolder whose name matches a subdir of
        ``target``.
        """
        wz_paths = _list_pack_files(folder, base_name)
        for wz_path in wz_paths:
            wz = WzFile.open(
                wz_path, region=self.region,
                version=self.version, writable=writable,
            )
            self._files.append(wz)
            # Pin the version once we detect it so subsequent files in
            # the same pack reuse it (avoids per-file rescans).
            if self.version is None:
                self.version = wz.version
            _merge_dir(target, wz.root)

        try:
            entries = os.listdir(folder)
        except OSError:
            return
        sub_lookup = {e.lower(): e for e in entries
                      if os.path.isdir(os.path.join(folder, e))}

        for sub_name in list(target.subdirs):
            on_disk = sub_lookup.get(sub_name.lower())
            if on_disk is None:
                continue
            sub_folder = os.path.join(folder, on_disk)
            sub_target = target.subdirs[sub_name]
            self._load_into(sub_target, sub_folder, on_disk, writable=writable)


# ── helpers ────────────────────────────────────────────────────────────
def _resolve_root_folder(path: str) -> Tuple[str, str]:
    """Normalize ``path`` to ``(folder, base_name)``.

    A ``.wz`` path resolves to ``(parent_dir, stem)``; a directory
    path resolves to ``(dir, dir_basename)`` (so ``Character/`` →
    base ``Character`` and structure file ``Character/Character.wz``).
    """
    if path.endswith(".wz") or os.path.isfile(path):
        folder = os.path.dirname(os.path.abspath(path)) or "."
        stem = os.path.basename(path)
        if stem.lower().endswith(".wz"):
            stem = stem[:-3]
        # If the user pointed at an indexed sibling (Foo_000.wz), strip
        # the index so the base matches the convention.
        m = _INDEXED_RE.search(os.path.basename(path))
        if m:
            stem = stem[: m.start()]
        return folder, stem
    folder = os.path.abspath(path).rstrip(os.sep)
    base = os.path.basename(folder)
    return folder, base


def _list_pack_files(folder: str, base_name: str) -> List[str]:
    """Return ``[base.wz, base_000.wz, base_001.wz, ...]`` (existing
    files only, in load order). Indexed files come after the structure
    file and are sorted by their numeric index for stability."""
    out: List[str] = []
    structure = os.path.join(folder, f"{base_name}.wz")
    if os.path.isfile(structure):
        out.append(structure)

    indexed: List[Tuple[int, str]] = []
    base_lower = base_name.lower() + "_"
    try:
        names = os.listdir(folder)
    except OSError:
        return out
    for name in names:
        lname = name.lower()
        if not lname.endswith(".wz"):
            continue
        if not lname.startswith(base_lower):
            continue
        m = _INDEXED_RE.search(name)
        if m is None:
            continue
        idx = int(m.group(1))
        indexed.append((idx, os.path.join(folder, name)))
    indexed.sort(key=lambda x: x[0])
    out.extend(p for _, p in indexed)
    return out


def is_hierarchical_pack(path: str) -> bool:
    """Heuristic: does ``path`` look like the entry point of a 64-bit
    hierarchical WZ pack?

    True when sibling ``<base>_NNN.wz`` files exist next to ``path``,
    or when ``path`` is itself a folder containing a ``<folder>.wz`` +
    ``<folder>_NNN.wz`` pair. Used by :func:`open_wz` to dispatch
    between :class:`WzFile` and :class:`WzPackage`.
    """
    if os.path.isdir(path):
        folder, base = _resolve_root_folder(path)
        return len(_list_pack_files(folder, base)) >= 1
    if not os.path.isfile(path):
        return False
    folder, base = _resolve_root_folder(path)
    # Structure file alone counts as legacy. Indexed siblings present →
    # hierarchical.
    has_indexed = False
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    base_lower = base.lower() + "_"
    for name in names:
        lname = name.lower()
        if lname.endswith(".wz") and lname.startswith(base_lower) \
                and _INDEXED_RE.search(name):
            has_indexed = True
            break
    return has_indexed


def _merge_dir(target: WzDirectory, source: WzDirectory) -> None:
    """Move ``source``'s images and subdir contents into ``target``.

    Subdirs are deep-merged so a placeholder ``Accessory`` from the
    structure file picks up actual content later. Images stay bound
    to their original :class:`WzFile`/reader; we just reseat the
    parent reference so the merged tree's path queries traverse the
    new parent chain.
    """
    for img_name, img in source.images.items():
        img.parent = target
        target.images[img_name] = img
    for sub_name, sub in source.subdirs.items():
        if sub_name in target.subdirs:
            _merge_dir(target.subdirs[sub_name], sub)
        else:
            sub.parent = target
            target.subdirs[sub_name] = sub


# ── link resolution (_outlink / _inlink) ──────────────────────────────
def resolve_canvas_link(prop: WzCanvasProperty, root: WzDirectory,
                        max_depth: int = 8) -> Optional[WzCanvasProperty]:
    """If ``prop`` is a placeholder canvas with an ``_outlink`` (or
    ``_inlink``) child, follow the chain to the canvas that holds the
    actual pixel data. Returns ``None`` when the link can't be resolved.

    The link target itself may be another linked placeholder — chains
    of length > 1 happen in some clients, so we walk up to ``max_depth``
    hops with cycle detection.

    ``_outlink`` is package-absolute (e.g.
    ``Character/Accessory/_Canvas/01010000.img/info/icon``); when the
    leading segment doesn't match a child of ``root`` we strip it once
    so links that include the legacy top-level WZ-category prefix
    (``Character/...``) still resolve under a hierarchical pack whose
    root contents are that category's direct subdirs.

    ``_inlink`` is image-relative — resolved against the property
    tree's root SubProperty (the ``.img`` itself).
    """
    seen: set = set()
    cur: WzProperty = prop
    for _ in range(max_depth):
        if id(cur) in seen:
            return None
        seen.add(id(cur))

        if isinstance(cur, WzUolProperty):
            cur = _follow_uol(cur)
            if cur is None:
                return None
            continue

        if not isinstance(cur, WzCanvasProperty):
            return None

        outlink = cur.child("_outlink")
        if isinstance(outlink, WzStringProperty) and outlink.value:
            target = _navigate_link(root, str(outlink.value))
            if target is not None:
                cur = target
                continue

        inlink = cur.child("_inlink")
        if isinstance(inlink, WzStringProperty) and inlink.value:
            img_root = _find_image_root(cur)
            if img_root is not None:
                target = img_root.get(str(inlink.value))
                if target is not None:
                    cur = target
                    continue

        return cur if cur.has_pixels() else None
    return None


def _navigate_link(root: WzDirectory, path: str) -> Optional[WzProperty]:
    """Walk a slash-separated path from ``root`` into directories,
    images, and property trees. Tries the path as-is first; if the
    first segment doesn't exist at root, retries with that segment
    stripped (handles ``_outlink`` strings that prefix the top-level
    WZ-category name)."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return None
    candidates = [parts]
    if parts[0] not in root.subdirs and parts[0] not in root.images:
        candidates.append(parts[1:])
    for ps in candidates:
        node = _walk_path(root, ps)
        if node is not None:
            return node
    return None


def _walk_path(root: WzDirectory, parts: List[str]):
    node = root
    i = 0
    while i < len(parts):
        part = parts[i]
        if isinstance(node, WzDirectory):
            if part in node.subdirs:
                node = node.subdirs[part]
            elif part in node.images:
                node = node.images[part]
                # Force the lazy parse so subsequent segments find the
                # in-image property tree.
                node.parse()
            else:
                return None
            i += 1
        elif isinstance(node, WzImage):
            return node.get("/".join(parts[i:]))
        elif isinstance(node, WzProperty):
            return node.get("/".join(parts[i:]))
        else:
            return None
    return node


def _find_image_root(prop: WzProperty) -> Optional[WzSubProperty]:
    cur: Optional[WzProperty] = prop
    while cur is not None:
        if cur.parent is None and isinstance(cur, WzSubProperty):
            return cur
        cur = cur.parent
    return None


def _follow_uol(uol: WzUolProperty) -> Optional[WzProperty]:
    """Walk a UOL chain (sibling-relative path) to its non-UOL target.
    Mirrors :func:`server.app._resolve_uol_target` so the link resolver
    can ride through UOL hops embedded between outlinks."""
    seen: set = set()
    cur: Optional[WzProperty] = uol
    for _ in range(16):
        if cur is None or not isinstance(cur, WzUolProperty):
            return cur
        if id(cur) in seen:
            return None
        seen.add(id(cur))
        target_str = cur.value
        if not target_str or cur.parent is None:
            return None
        cur = cur.parent.get(str(target_str))
    return None


def open_wz(path: str, region: str = "GMS",
            version: Optional[int] = None,
            writable: bool = False):
    """Factory: open ``path`` as either a legacy single-file
    :class:`WzFile` or a hierarchical :class:`WzPackage`, picking
    automatically based on what's on disk next to ``path``.

    A hierarchical pack is detected by sibling ``<base>_NNN.wz``
    files. Hierarchical packs ignore ``writable`` because re-splitting
    edits across multiple files isn't currently supported.

    A ``.ms`` file (the WZ-tree containers in ``data/Packs/*.ms`` — mob/skill
    canvases + embedded Spine skeletons) opens as
    :class:`wzpy.ms_container.MsContainer` — a structure inventory + skeletons,
    not a WZ tree handle (see ``docs/ms_format_findings.md``). To read the
    *decrypted* skill/mob image trees instead, use
    :class:`wzpy.ms_file_v2.MsPackageV2` directly.
    """
    from .ms_file_v2 import is_ms_path
    if is_ms_path(path):
        import os
        from .ms_container import MsContainer
        if os.path.isfile(path):
            return MsContainer.open(path)
        raise ValueError(
            f"{path!r} is a folder of .ms Pack files; open a single .ms with "
            "open_wz(), or use wzpy.ms_file_v2.MsPackageV2 for the whole folder."
        )
    if is_hierarchical_pack(path):
        return WzPackage.open(path, region=region, version=version,
                              writable=writable)
    return WzFile.open(path, region=region, version=version,
                       writable=writable)
