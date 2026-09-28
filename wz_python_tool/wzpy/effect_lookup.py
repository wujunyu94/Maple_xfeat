"""Lookup for equipment-specific effect overlays in ``ItemEff.img``.

When the user equips an item with an authored effect (e.g. cap
``01004759`` with ember-flames around the head), MapleStory's client
loads the matching subtree from ``Effect.wz/ItemEff.img/<id>`` and
composites its canvases on top of the character. The IDs in
``ItemEff.img`` drop the leading zero — Character.wz's ``01004759``
becomes ``1004759`` — so the lookup strips them at query time.

The lookup module is stand-alone so it can be reused outside the
Flask app (e.g. from a notebook). It owns the loaded Effect pack
and can ``close()`` it on shutdown.
"""

from __future__ import annotations

from typing import Optional, Union

from .properties import WzSubProperty
from .wz_file import WzFile
from .wz_image import WzImage
from .wz_package import WzPackage, open_wz


class EffectLookup:
    """Resolve equip IDs to their ``effect`` subtree under
    ``ItemEff.img``.

    Construct with an open :class:`WzFile` or :class:`WzPackage` whose
    root has an ``ItemEff.img``. ``find(equip_id)`` returns the
    matching ``effect`` :class:`WzSubProperty`, or ``None`` when no
    effect is authored for that ID.
    """

    def __init__(self, wz: Union[WzFile, WzPackage]):
        self._wz = wz
        self._itemeff_root: Optional[WzSubProperty] = None
        self._loaded = False

    @classmethod
    def open(cls, path: str, region: str = "auto",
             version: Optional[int] = None) -> Optional["EffectLookup"]:
        """Open the Effect pack at ``path`` (folder or .wz). Returns
        ``None`` when the path doesn't look like a usable Effect
        pack, so callers can silently fall back to no overlay."""
        try:
            wz = open_wz(path, region=region, version=version)
        except Exception:
            return None
        # Sanity check: a real Effect pack ships ``ItemEff.img`` at root.
        if not isinstance(wz.root.get("ItemEff.img"), WzImage):
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
        """Pack root — needed by the renderer to resolve ``_outlink``
        canvas placeholders into the sibling ``_Canvas`` tree."""
        return self._wz.root

    @property
    def region(self) -> str:
        return getattr(self._wz, "region", "GMS")

    # ── lookup ──────────────────────────────────────────────────────
    def find(self, equip_id: str) -> Optional[WzSubProperty]:
        """Return the ``effect`` subtree for ``equip_id`` (8-digit
        zero-padded), or ``None``. The lookup strips leading zeros
        because ``ItemEff.img`` keys them as plain integers."""
        try:
            id_int = int(equip_id)
        except (TypeError, ValueError):
            return None
        root = self._itemeff()
        if root is None:
            return None
        node = root.get(str(id_int))
        if not isinstance(node, WzSubProperty):
            return None
        eff = node.get("effect")
        return eff if isinstance(eff, WzSubProperty) else None

    # ── internals ───────────────────────────────────────────────────
    def _itemeff(self) -> Optional[WzSubProperty]:
        if not self._loaded:
            img = self._wz.root.get("ItemEff.img")
            if isinstance(img, WzImage):
                self._itemeff_root = img.parse()
            self._loaded = True
        return self._itemeff_root
