"""Shared attack-skill geometry and per-Mob-ID eligibility rules.

The legacy attack controls remain the primary skill. Additional skills are
opt-in per Mob ID, so adding a Heal key cannot silently change attacks on
unconfigured monsters.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


Box = Tuple[int, int, int, int]


def mob_id_from_name(name: str) -> Optional[str]:
    match = re.match(r"^(?:mob_)?(\d+)(?:_|$)", str(name))
    return match.group(1) if match else None


def mob_id_of(target: Any) -> Optional[str]:
    explicit = getattr(target, "mob_id", None)
    if explicit is not None and str(explicit).isdigit():
        return str(explicit)
    return mob_id_from_name(getattr(target, "name", ""))


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _wz_rect(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    try:
        lt = [int(v) for v in raw["lt"]]
        rb = [int(v) for v in raw["rb"]]
        if len(lt) == len(rb) == 2 and lt[0] < rb[0] and lt[1] < rb[1] and all(
            -4000 <= v <= 4000 for v in lt + rb
        ):
            return {"lt": lt, "rb": rb, "skill_id": str(raw.get("skill_id", "")),
                    "level": int(raw.get("level", 0)),
                    "range_bonus": max(0, min(4000, int(raw.get("range_bonus", 0))))}
    except (KeyError, TypeError, ValueError):
        pass
    return None


def skills_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return validated primary + extra skills in user-defined priority order."""
    primary = {
        "id": "primary",
        "name": "主攻击",
        "key": str(config.get("attack_key", "ctrl")).lower(),
        "vk": int(_number(config.get("attack_vk", 0x11), 0x11, 1, 255)),
        "reach_x": _number(config.get("attack_reach_x", 260), 260, 0, 800),
        "reach_y_up": _number(config.get("attack_reach_y_up", config.get("attack_reach_y", 140)), 140, 0, 400),
        "reach_y_down": _number(config.get("attack_reach_y_down", config.get("attack_reach_y", 140)), 140, 0, 400),
        "behind_x": _number(config.get("behind_reach_x", 40), 40, 0, 400),
        "two_way": bool(config.get("attack_two_way", False)),
        "area": bool(config.get("attack_area", False)),
        "wz_rect": _wz_rect(config.get("primary_wz_rect")),
    }
    result = [primary]
    seen = {"primary"}
    for raw in config.get("extra_attack_skills", []) or []:
        if not isinstance(raw, dict):
            continue
        skill_id = str(raw.get("id", "")).strip()
        key = str(raw.get("key", "")).strip().lower()
        if not re.fullmatch(r"skill_[1-9]\d*", skill_id) or skill_id in seen or not key:
            continue
        seen.add(skill_id)
        result.append({
            "id": skill_id,
            "name": str(raw.get("name", skill_id)).strip() or skill_id,
            "key": key,
            "vk": int(_number(raw.get("vk", 0), 0, 1, 255)),
            "reach_x": _number(raw.get("reach_x", 260), 260, 0, 800),
            "reach_y_up": _number(raw.get("reach_y_up", 140), 140, 0, 400),
            "reach_y_down": _number(raw.get("reach_y_down", 140), 140, 0, 400),
            "behind_x": _number(raw.get("behind_x", 0), 0, 0, 400),
            "two_way": bool(raw.get("two_way", False)),
            "area": bool(raw.get("area", False)),
            "wz_rect": _wz_rect(raw.get("wz_rect")),
        })
    return result


def eligible_skills(config: Dict[str, Any], mob_id: Optional[str],
                    skills: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if skills is None:
        skills = skills_from_config(config)
    rules = config.get("monster_skill_rules", {}) or {}
    if mob_id is None and rules:
        # A health-bar-only detection has no species identity. With restricted
        # skills configured, guessing primary could cast the wrong skill.
        return []
    rule = rules.get(str(mob_id)) if isinstance(rules, dict) and mob_id is not None else None
    if rule is None:
        return skills[:1]
    allowed = {str(item) for item in rule} if isinstance(rule, (list, tuple)) else set()
    return [skill for skill in skills if skill["id"] in allowed]


def attack_box(skill: Dict[str, Any], player: Tuple[float, float], facing: str, frame_w: int) -> Box:
    scale = max(0.4, float(frame_w) / 1920.0)
    px, py = player
    wz = skill.get("wz_rect")
    if wz is not None:
        lx, ty = wz["lt"]
        rx, by = wz["rb"]
        # WZ skill rectangles use the same foot-origin as last_player_pos.
        # The source interval is left-facing; mirror around the foot X for right.
        # Native WZ coordinates are game pixels, not percentages of viewport
        # width: a wider client shows more map without stretching the skill.
        wz_scale = 1.0
        bonus = wz.get("range_bonus", 0)
        if facing == "left":
            lx -= bonus
        else:
            lx, rx = -rx, -lx + bonus
        x1, x2 = round(px + lx * wz_scale), round(px + rx * wz_scale)
        y1, y2 = round(py + ty * wz_scale), round(py + by * wz_scale)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    reach = int(skill["reach_x"] * scale)
    behind = int(skill["behind_x"] * scale)
    up = int(skill["reach_y_up"] * scale)
    down = int(skill["reach_y_down"] * scale)
    if skill["two_way"]:
        left, width = int(px) - reach, reach * 2
    elif facing == "right":
        left, width = int(px) - behind, reach + behind
    else:
        left, width = int(px) - reach, reach + behind
    return left, int(py) - up, width, up + down


def rear_box(skill: Dict[str, Any], player: Tuple[float, float], facing: str, frame_w: int) -> Optional[Box]:
    if skill["two_way"]:
        return None
    if skill.get("wz_rect") is not None:
        return attack_box(skill, player, "left" if facing == "right" else "right", frame_w)
    scale = max(0.4, float(frame_w) / 1920.0)
    px, py = player
    reach = int(skill["reach_x"] * scale)
    up = int(skill["reach_y_up"] * scale)
    down = int(skill["reach_y_down"] * scale)
    left = int(px) - reach if facing == "right" else int(px)
    return left, int(py) - up, reach, up + down


def skirmish_boxes(skill: Dict[str, Any], player: Tuple[float, float], facing: str,
                   frame_w: int, extra_x: float) -> Tuple[Box, Box]:
    scale = max(0.4, float(frame_w) / 1920.0)
    px, py = player
    if skill.get("wz_rect") is not None:
        front = attack_box(skill, player, facing, frame_w)
        rear = attack_box(skill, player, "left" if facing == "right" else "right", frame_w)
        extra = max(0, int(extra_x * scale))
        front_edge = front[0] + front[2] if facing == "right" else front[0]
        rear_edge = rear[0] if facing == "right" else rear[0] + rear[2]
        return ((front_edge, front[1], extra, front[3]) if facing == "right"
                else (front_edge - extra, front[1], extra, front[3])), (
                (rear_edge - extra, rear[1], extra, rear[3]) if facing == "right"
                else (rear_edge, rear[1], extra, rear[3]))
    reach = int(skill["reach_x"] * scale)
    extra = max(0, int(extra_x * scale))
    up = int(skill["reach_y_up"] * scale)
    down = int(skill["reach_y_down"] * scale)
    top, height = int(py) - up, up + down
    if facing == "right":
        return (int(px) + reach, top, extra, height), (int(px) - reach - extra, top, extra, height)
    return (int(px) - reach - extra, top, extra, height), (int(px) + reach, top, extra, height)


def contains(box: Box, target: Any) -> bool:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    cx = float(getattr(target, "center_x", getattr(target, "cx", getattr(target, "center", (0, 0))[0])))
    cy = float(getattr(target, "center_y", getattr(target, "cy", getattr(target, "center", (0, 0))[1])))
    if x <= cx <= x + w and y <= cy <= y + h:
        return True
    bx, by, bw, bh = getattr(target, "bbox", (cx, cy, 0, 0))
    return bx < x + w and bx + bw > x and by < y + h and by + bh > y


def choose_skill_for_target(config: Dict[str, Any], target: Any,
                            player: Tuple[float, float], facing: str, frame_w: int,
                            monsters: Optional[List[Any]] = None,
                            skills: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Pick the same actual key for the HUD and both combat modes."""
    if skills is None:
        skills = skills_from_config(config)
    candidates = []
    for skill in eligible_skills(config, mob_id_of(target), skills):
        box = attack_box(skill, player, facing, frame_w)
        if not contains(box, target):
            continue
        hits = 1
        if skill["area"]:
            hits += sum(
                mob is not target
                and not getattr(mob, "is_ghost", False)
                and not getattr(mob, "is_dead", False)
                and skill in eligible_skills(config, mob_id_of(mob), skills)
                and contains(box, mob)
                for mob in (monsters or [])
            )
        candidates.append((hits if skill["area"] and hits >= 2 else 0, skill))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None
