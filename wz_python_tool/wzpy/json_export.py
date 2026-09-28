"""JSON serialization for WZ tree nodes.

Produces the same shape used by the web UI's ``/api/export/json`` endpoint
and by the standalone ``convert_img.py`` CLI. Canvas pixel data and Sound
audio bytes are not inlined — exporting them as base64 would balloon a
typical Mob/Map dump from kilobytes to gigabytes. Use the dedicated image
export (in the web UI) when you need the binaries.
"""

from __future__ import annotations

from typing import Any, Dict

from .properties import (
    WzCanvasProperty,
    WzConvexProperty,
    WzNullProperty,
    WzProperty,
    WzSoundProperty,
    WzSubProperty,
    WzUolProperty,
    WzVectorProperty,
)
from .wz_file import WzDirectory
from .wz_image import WzImage


def property_to_dict(prop: WzProperty) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": prop.name, "type": prop.type_name}
    if isinstance(prop, WzNullProperty):
        out["value"] = None
    elif isinstance(prop, WzVectorProperty):
        out["value"] = {"x": prop.x, "y": prop.y}
    elif isinstance(prop, WzCanvasProperty):
        out["width"], out["height"] = prop.width, prop.height
        out["format"] = prop.format + prop.format2
        if prop.has_children():
            out["children"] = [property_to_dict(c) for c in prop.children()]
    elif isinstance(prop, WzSoundProperty):
        out["length_ms"] = prop.length_ms
        out["bytes"] = prop.value
    elif isinstance(prop, WzConvexProperty):
        out["points"] = [{"x": p.x, "y": p.y} for p in prop.points]
    elif isinstance(prop, WzUolProperty):
        out["target"] = prop.value
    elif isinstance(prop, WzSubProperty):
        out["children"] = [property_to_dict(c) for c in prop.children()]
    else:
        try:
            out["value"] = prop.value
        except Exception:
            out["value"] = None
    return out


def node_to_dict(node: Any) -> Dict[str, Any]:
    if isinstance(node, WzDirectory):
        return {
            "name": node.name,
            "type": "Directory",
            "subdirs": {n: node_to_dict(d) for n, d in node.subdirs.items()},
            "images": {n: node_to_dict(i) for n, i in node.images.items()},
        }
    if isinstance(node, WzImage):
        node.parse()
        return {
            "name": node.name,
            "type": "Image",
            "children": [property_to_dict(c) for c in node.children()],
        }
    if isinstance(node, WzProperty):
        return property_to_dict(node)
    return {"name": getattr(node, "name", ""), "type": "Unknown"}
