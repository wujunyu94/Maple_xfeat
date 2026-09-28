"""Look up human-readable names for Character.wz items via String.wz.

Each item in MapleStory's ``String.wz/Eqp.img/Eqp/<sub>/<id>/name``
holds the localized display name for the equip with that ID. Hair /
Face / Skin (= Body) live under sibling subtrees of the same
``Eqp`` group. The IDs in String.wz drop leading zeros — e.g.,
Character.wz's ``01007088`` becomes ``1007088`` — so the lookup
strips them at query time.

The lookup module is stand-alone so it can be reused outside the
Flask app (e.g., from a notebook). It owns the loaded String pack
and can ``close()`` it on shutdown.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from .properties import WzStringProperty, WzSubProperty
from .wz_file import WzDirectory, WzFile
from .wz_image import WzImage
from .wz_package import WzPackage


# Maps Character.wz categories (CharacterRenderer's ``category`` arg)
# to the matching subdirectory under ``String.wz/Eqp.img/Eqp``. The
# Accessory categories share one directory because String.wz doesn't
# split FaceAcc / Glass / Earring — they're all "Accessory" with
# different ID prefixes.
_CHAR_TO_STRING_CATEGORY: Dict[str, str] = {
    "Body":     "Skin",
    "Hair":     "Hair",
    "Face":     "Face",
    "Cap":      "Cap",
    "Coat":     "Coat",
    "Longcoat": "Longcoat",
    "Pants":    "Pants",
    "Shoes":    "Shoes",
    "Glove":    "Glove",
    "Cape":     "Cape",
    "Shield":   "Shield",
    "FaceAcc":  "Accessory",
    "Glass":    "Accessory",
    "Earring":  "Accessory",
    "Weapon":   "Weapon",
    # ``Head`` has no String entry — head canvases are tied to the
    # body's skin tone, not a named item.
}


class StringLookup:
    """Resolve equip IDs to display names via ``String.wz/Eqp.img``.

    Construct with an open :class:`WzFile` or :class:`WzPackage` whose
    root has an ``Eqp.img``. ``name(category, equip_id)`` returns the
    matching ``name`` string, or ``None`` when the lookup misses.

    The Eqp tree is parsed once on first access and the result is
    cached as ``{category: {id_int: name}}`` so repeated lookups are
    a dict hit.
    """

    def __init__(self, wz: Union[WzFile, WzPackage]):
        self._wz = wz
        self._cache: Dict[str, Dict[int, str]] = {}
        self._eqp_root: Optional[WzSubProperty] = None
        self._loaded = False

    @classmethod
    def open(cls, path: str, region: str = "auto",
             version: Optional[int] = None) -> Optional["StringLookup"]:
        """Open the String pack at ``path`` (folder or .wz). Returns
        ``None`` when the path doesn't look like a usable String
        pack, so callers can fall back to ID-only display."""
        from .wz_package import open_wz
        try:
            wz = open_wz(path, region=region, version=version)
        except Exception:
            return None
        # Sanity check: a real String pack ships ``Eqp.img`` at root.
        if not isinstance(wz.root.get("Eqp.img"), WzImage):
            try:
                wz.close()
            except Exception:
                pass
            return None
        return cls(wz)

    def close(self) -> None:
        try:
            self._wz.close()
        except Exception:
            pass

    @property
    def root(self):
        """Pack root — exposed so the multi-pack bundle view can mount
        the loaded String tree alongside Character and Effect."""
        return self._wz.root

    # ── lookup ──────────────────────────────────────────────────────
    def name(self, category: str, equip_id: str) -> Optional[str]:
        sub = _CHAR_TO_STRING_CATEGORY.get(category)
        if sub is None:
            return None
        # Strip leading zeros — Character.wz's ``01007088.img`` is
        # stored as ``1007088`` in String.wz.
        try:
            id_int = int(equip_id)
        except (TypeError, ValueError):
            return None
        return self._for_category(sub).get(id_int)

    # ── internals ───────────────────────────────────────────────────
    def _for_category(self, sub: str) -> Dict[int, str]:
        cached = self._cache.get(sub)
        if cached is not None:
            return cached
        cached = {}
        eqp_root = self._eqp()
        if eqp_root is not None:
            cat_node = eqp_root.get(sub)
            if isinstance(cat_node, WzSubProperty):
                for child in cat_node.children():
                    if not child.name.isdigit():
                        continue
                    name_node = child.get("name") if hasattr(child, "get") else None
                    if isinstance(name_node, WzStringProperty) and name_node.value:
                        cached[int(child.name)] = str(name_node.value)
        self._cache[sub] = cached
        return cached

    def _eqp(self) -> Optional[WzSubProperty]:
        if not self._loaded:
            img = self._wz.root.get("Eqp.img")
            if isinstance(img, WzImage):
                root = img.parse()
                self._eqp_root = root.get("Eqp") if isinstance(root, WzSubProperty) else None
            self._loaded = True
        return self._eqp_root
