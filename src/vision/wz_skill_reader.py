"""Read authoritative per-level attack rectangles from extracted Skill IMG files.

Only ``level/N/lt`` and ``rb`` are interpreted as a hit area. Animation
canvases and the scalar ``range`` are not sufficient to infer a hitbox.
Coordinates are relative to the game's character position (the foot anchor).
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import List, Optional, Tuple


DEFAULT_SKILL_ROOT = str(Path(__file__).resolve().parents[2] / "Data" / "Skill")
DEFAULT_STRING_ROOT = str(Path(__file__).resolve().parents[2] / "Data" / "String")
DEFAULT_PARSER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "wz_python_tool")
)
# These rectangles describe an impact area around the selected target, not
# a cast area around the player. Treating them as foot-anchored would trigger
# attacks at the wrong distance.
TARGET_ANCHORED_SKILL_IDS = frozenset({"3101005"})


@dataclass(frozen=True)
class SkillHitbox:
    skill_id: str
    level: int
    lt: Tuple[int, int]
    rb: Tuple[int, int]
    mob_count: Optional[int]
    range_bonus: int
    source_path: str


@dataclass(frozen=True)
class SkillName:
    skill_id: str
    name: str
    book: str = ""


@lru_cache(maxsize=4)
def _skill_name_index(path: str, mtime_ns: int, size: int,
                      parser_root: str, iv_hex: str) -> Tuple[SkillName, ...]:
    """Lazily parse the local String/Skill.img; cache by file signature."""
    if parser_root not in sys.path:
        sys.path.insert(0, parser_root)
    from wzpy.crypto import WzKey
    from wzpy.wz_image import WzImage

    with open(path, "rb") as stream:
        image = WzImage.from_bytes(stream.read(), key=WzKey(bytes.fromhex(iv_hex)),
                                   name=os.path.basename(path))
    root = image.parse()
    if getattr(image, "truncated", False):
        raise ValueError(f"技能名称 IMG 读取不完整：{path}")
    records = []
    for child in root.children():
        skill_id = str(child.name)
        if not re.fullmatch(r"\d{7,8}", skill_id):
            continue
        name_node = child.child("name")
        name = str(name_node.value).strip() if name_node is not None else ""
        if name:
            book_node = root.child(f"{int(skill_id) // 10000:03d}")
            book_name = book_node.child("bookName") if book_node is not None else None
            book = str(book_name.value).strip() if book_name is not None else ""
            records.append(SkillName(skill_id, name, book))
    return tuple(records)


def search_skill_names(query: str, *, limit: int = 30,
                       string_root: str = DEFAULT_STRING_ROOT,
                       skill_root: str = DEFAULT_SKILL_ROOT,
                       parser_root: str = DEFAULT_PARSER_ROOT,
                       iv_hex: str = "4D23C72B") -> List[SkillName]:
    """Search installed skill names by substring, partial name or close typo."""
    term = "".join(unicodedata.normalize("NFKC", str(query)).lower().split())
    if not term:
        return []
    path = os.path.join(string_root, "Skill.img")
    stat = os.stat(path)
    records = _skill_name_index(path, stat.st_mtime_ns, stat.st_size,
                                parser_root, iv_hex)
    ranked = []
    for record in records:
        filename = f"{int(record.skill_id) // 10000:03d}.img"
        if not os.path.isfile(os.path.join(skill_root, filename)):
            continue
        name = "".join(unicodedata.normalize("NFKC", record.name).lower().split())
        if term == name or term == record.skill_id:
            score = 100
        elif name.startswith(term) or record.skill_id.startswith(term):
            score = 90
        elif term in name or term in record.skill_id:
            score = 80
        elif all(token in name for token in term):
            score = 65
        else:
            similarity = SequenceMatcher(None, term, name).ratio()
            if similarity < 0.62:
                continue
            score = 40 + similarity * 20
        ranked.append((-score, len(name), len(record.skill_id), record.skill_id, record))
    ranked.sort()
    return [item[4] for item in ranked[:max(1, min(100, int(limit)))]]


def _point(value: object) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("技能矩形坐标格式无效")
    x, y = int(value[0]), int(value[1])
    if not -4000 <= x <= 4000 or not -4000 <= y <= 4000:
        raise ValueError("技能矩形坐标超出合理范围")
    return x, y


def load_skill_hitbox(skill_id: str, level: int, *, skill_root: str = DEFAULT_SKILL_ROOT,
                      parser_root: str = DEFAULT_PARSER_ROOT,
                      iv_hex: str = "4D23C72B") -> SkillHitbox:
    """Load an exact rank; fail closed when no explicit ``lt/rb`` exists."""
    skill_id = str(skill_id).strip()
    if not re.fullmatch(r"\d{7,8}", skill_id):
        raise ValueError("请输入 7–8 位技能 ID（新手技能需保留前导 0）")
    if skill_id in TARGET_ANCHORED_SKILL_IDS:
        raise ValueError(f"技能 {skill_id} 的 lt/rb 以命中目标为原点，不能作为人物攻击范围导入")
    level = int(level)
    if level < 1 or level > 100:
        raise ValueError("技能等级必须在 1–100 之间")
    filename = f"{int(skill_id) // 10000:03d}.img"
    source_path = os.path.join(skill_root, filename)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"本地技能文件不存在：{source_path}")
    if parser_root not in sys.path:
        sys.path.insert(0, parser_root)
    from wzpy.crypto import WzKey
    from wzpy.wz_image import WzImage

    with open(source_path, "rb") as stream:
        image = WzImage.from_bytes(stream.read(), key=WzKey(bytes.fromhex(iv_hex)), name=filename)
    root = image.parse_partial(only={"skill"})
    skill = root.child("skill")
    skill = skill.child(skill_id) if skill is not None else None
    if skill is None:
        raise ValueError(f"本地文件没有技能 {skill_id}")
    levels = skill.child("level")
    rank = levels.child(str(level)) if levels is not None else None
    if rank is None:
        raise ValueError(f"技能 {skill_id} 没有等级 {level} 的数据")
    lt_node, rb_node = rank.child("lt"), rank.child("rb")
    if lt_node is None or rb_node is None:
        raise ValueError(
            f"技能 {skill_id} 等级 {level} 没有明确的 lt/rb 命中矩形；"
            "不能用动画尺寸或 range 推测，请手动设置"
        )
    lt, rb = _point(lt_node.value), _point(rb_node.value)
    if lt[0] >= rb[0] or lt[1] >= rb[1]:
        raise ValueError("技能 lt/rb 矩形为空或方向错误")
    mob_node = rank.child("mobCount")
    mob_count = int(mob_node.value) if mob_node is not None else None
    range_node = rank.child("range")
    range_bonus = max(0, int(range_node.value)) if range_node is not None else 0
    if getattr(image, "truncated", False):
        raise ValueError(f"技能 IMG 读取不完整：{source_path}")
    return SkillHitbox(skill_id, level, lt, rb, mob_count, range_bonus, source_path)
