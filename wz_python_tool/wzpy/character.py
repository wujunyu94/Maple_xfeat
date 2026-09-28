"""Character builder: assemble a static MapleStory character from Character.wz.

Given a list of equip IDs (body, head, hair, face, cap, coat, …) this module
walks each part image, resolves the per-canvas anchor points (``origin`` plus
``map/<navel|neck|hand|brow|handMove>``) and chains them so that named anchors
align across parts (head's ``neck`` ↔ body's ``neck``, weapon's ``hand`` ↔
arm's ``hand``, etc.). Parts are then sorted by the ``z`` slot referenced in
each canvas and composited into a single PNG.

The algorithm matches HaCreator/MapleNecrocer's CharacterAssembler:

  body draws so its navel sits at world (0, 0). For every other canvas we
  pick the highest-priority ``map/<anchor>`` it advertises and place the
  canvas so that anchor coincides with the same-named anchor on whichever
  part already populated it. Map values are *relative to ``origin``* so the
  world position of an anchor is ``top_left + origin + map[name]``.

The static composite uses ``stand1/0`` for body-frame parts and
``default``/``backDefault`` for hair/face/cap (so that hat covers, back-hair,
etc. all show up). Animation, expression cycling, dye, and HSL adjustments
from MapleNecrocer's full Avatar form are out of scope for v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .canvas import apply_hsv_adjust, decode_canvas
from .properties import (
    WzCanvasProperty,
    WzProperty,
    WzStringProperty,
    WzSubProperty,
    WzUolProperty,
    WzVectorProperty,
)
from .wz_file import WzDirectory, WzFile
from .wz_image import WzImage
from .wz_package import resolve_canvas_link


# ── category mapping ──────────────────────────────────────────────────────
# Mirrors MapleNecrocer's ``Equip.GetDir`` / ``GetPart`` for the parts the
# static builder exposes. Keys are the high 4 digits of the equip ID divided
# by 10000 (i.e., ``int(eid)//10000``).

_CATEGORY_BY_ID_PREFIX: Dict[int, str] = {
    0: "Body",
    1: "Head",
    2: "Face", 5: "Face",
    3: "Hair", 4: "Hair", 6: "Hair",
    100: "Cap",
    101: "FaceAcc",
    102: "Glass",
    103: "Earring",
    104: "Coat",
    105: "Longcoat",
    106: "Pants",
    107: "Shoes",
    108: "Glove",
    109: "Shield",
    110: "Cape",
    170: "Weapon",
}
# Weapon range: 121..160 (and 170, included above).
for _n in range(121, 161):
    _CATEGORY_BY_ID_PREFIX[_n] = "Weapon"


CATEGORIES: Tuple[str, ...] = (
    "Body", "Head", "Hair", "Face",
    "Cap", "Coat", "Longcoat", "Pants", "Shoes", "Glove",
    "Cape", "Shield", "FaceAcc", "Glass", "Earring", "Weapon",
)

# Categories for which ``list_parts`` reads ``info/cash`` so the UI can
# split the grid into Cash / Non-Cash sub-tabs. Skipped for Body / Head
# (always non-cash character bases) and Hair / Face (character looks,
# and parsing 16k Hair imgs just to set a flag is a 30+s first-load
# regression nobody wants if the UI isn't filtering on it anyway).
_CASH_FILTERED_CATEGORIES: frozenset = frozenset({
    "Cap", "Coat", "Longcoat", "Pants", "Shoes", "Glove",
    "Cape", "Shield", "FaceAcc", "Glass", "Earring", "Weapon",
})

CATEGORY_DIR: Dict[str, str] = {
    "Body": "",
    "Head": "",
    "Hair": "Hair",
    "Face": "Face",
    "Cap": "Cap",
    "Coat": "Coat",
    "Longcoat": "Longcoat",
    "Pants": "Pants",
    "Shoes": "Shoes",
    "Glove": "Glove",
    "Cape": "Cape",
    "Shield": "Shield",
    "FaceAcc": "Accessory",
    "Glass": "Accessory",
    "Earring": "Accessory",
    "Weapon": "Weapon",
}

# Default anchor when a canvas has no map/* point. Picked so that the
# canvas's ``origin`` point lands at the implicit anchor in world space —
# matches the convention HaCreator's CharacterAssembler uses when
# ``GetMapPoint`` returns ``Point.Zero``.
_DEFAULT_ANCHOR_BY_CATEGORY: Dict[str, str] = {
    "Body": "navel",
    "Head": "neck",
    "Hair": "brow",
    "Face": "brow",
    "Cap": "navel",
    "Coat": "navel",
    "Longcoat": "navel",
    "Pants": "navel",
    "Shoes": "navel",
    "Glove": "hand",
    "Cape": "navel",
    "Shield": "navel",
    "FaceAcc": "brow",
    "Glass": "brow",
    "Earring": "brow",
    "Weapon": "hand",
}

_ANCHOR_PRIORITY: Tuple[str, ...] = ("navel", "neck", "hand", "brow", "handMove")

# Anchors that ride on the breathing body's per-frame arm rather than the
# (still) head. In :meth:`CharacterRenderer.compose_animation`'s stabilized
# stand poses the head-family anchors (``neck`` / ``brow``) are frozen at
# frame 0 so head / hair / cap / face don't jitter, but the hand sits at the
# end of the arm canvas whose breathing IS baked into each frame. Held items
# (weapon / glove / shield) anchor on the hand, so the hand world anchor must
# track the body's current-frame arm — freezing it makes a weapon with a
# single shared canvas drift down-then-up relative to the moving grip. These
# anchors are therefore re-registered from the body's per-frame canvases even
# when the rest of the anchor map is frozen. ``navel`` stays frozen (it's the
# body's own reference point, and body-anchored equipment tracks it through
# the right-edge re-pin delta instead).
_DYNAMIC_ANIM_ANCHORS: frozenset = frozenset({"hand", "lHand"})


# Back-facing pose (ladder, rope) z overrides for the head-region
# ``back*`` cluster. The front-facing zmap puts backHair (162) below
# backHead (184) and below the body/coat cluster (backBody 185,
# backMailChest 225) — in FRONT view those slots represent hair /
# cap canvases drawn BEHIND the body silhouette (e.g. long hair
# trailing past the torso). For a back-facing pose the same canvases
# represent the back of the head and need to render ON TOP of the
# body / coat / head: hair covers head, cap covers hair, just like
# in front view (hair > head, cap > hair). The numeric values are
# picked to land above ``backMailChest`` (225, the topmost body-
# cluster slot) while preserving cap > hair > head ordering.
_BACK_FACING_Z_OVERRIDE: Dict[str, int] = {
    # Hair sub-cluster (above body / coat).
    "backHairBelowCapWide":   250,
    "backHairBelowCapNarrow": 251,
    "backHairBelowCap":       252,
    "backHairOverCape":       253,
    "backHair":               254,
    # Cap sub-cluster (above hair).
    "backCap":                260,
    "backCapOverHair":        261,
    "backCapAccessory":       262,
    "backAccessoryEar":       263,
    # Back-strapped shield + weapon. Only the slots the stock
    # ``Base.wz/zmap.img`` places ABOVE the whole head/hair/cap cluster
    # are overridden here — because that cluster itself is re-tuned to
    # 250-263 above, a slot that should out-rank it needs an explicit
    # value above 263 rather than its raw (sub-263) zmap index.
    # Canonically ``backShield`` (idx 70) and ``backWeaponOverShield``
    # (72) both sit above ``backHair`` (68) / ``backCap`` (67), so a
    # shield (and a weapon drawn over it) strapped to the back stays
    # visible over the back of the head while climbing.
    "backShield":             271,
    "backWeaponOverShield":   274,
    # NOTE: the remaining back weapon/shield slots are deliberately
    # NOT overridden — they fall through to their canonical zmap
    # index. The stock zmap places them at or BELOW the head/hair/cap
    # cluster, so lifting them (as an earlier "outermost from back
    # view" version did, ``backWeaponOverGlove`` -> 275 etc.) painted
    # them on top of the head, which is wrong:
    #   * ``backWeapon`` (34) / ``backShieldBelowBody`` (37) — below
    #     the back body entirely.
    #   * ``backWeaponOverGlove`` (46) — just above ``backBody`` (42)
    #     but below ``backMailChest`` (54) / ``backHead`` (57) /
    #     hair / cap. This is the climbing-claw case (e.g. 01472026's
    #     ladder/rope canvas): it peeks out from behind the body and
    #     is occluded by the head and hair instead of floating above
    #     the hair.
    #   * ``backWeaponOverHead`` (65) — above ``backHead`` (57) but
    #     still below the hair (68) / cap (67) that drape over it.
    # Cape slots used by back-view animations. From a back-facing
    # view the cape is on the side of the character we're looking at,
    # so cape canvases should render IN FRONT of the body cluster.
    # The MapleStory data uses many cape slot names — most ladder/
    # rope canvases use ``backCape`` (the canonical back-of-cape
    # slot), with ``capeBelowHair`` / ``capeBelowBody`` / ``cape`` /
    # ``capeOverBody`` / ``capeOverWepon`` showing up too. Slot
    # names roughly indicate z within the cape stack:
    #   * ``capeBelowBody`` — deepest cape layer
    #   * ``backCape`` / ``cape`` / ``capeBelowHair`` — main cloth
    #   * ``capeOverBody`` / ``capeOverWepon`` — over outfit
    #   * ``capeOverHead`` — straps / hood layer above hair cluster
    # Bump them above the body cluster (max ~99) and below the hair
    # cluster (250+) so back hair still drapes over the cape.
    # ``capeOverHead`` lands above the hair cluster.
    "capeBelowBody":          235,
    "backCape":               238,
    "cape":                   239,
    "capeBelowHair":          240,
    "capeOverBody":           244,
    "capeOverWepon":          246,
    "backHairOverCape":       248,  # hair-over-cape strands; above cape, below main hair
    "capeOverHead":           265,
    # ``weaponOverGlove`` shows up on ~500 cape ladder/rope canvases
    # (cape 01102292's main cloth uses it). The slot name reads as
    # weapon-related but the artist's intent for capes is "above
    # everything" — wing/cloak shapes that drape from the back and
    # need to be visible past hair / cap AND past any
    # back-strapped weapon/shield (the cape covers them from a
    # back-facing view). Place above the entire back cluster.
    "weaponOverGlove":        280,
}


# Canonical MapleStory z-order, embedded verbatim from
# ``Base.wz/Base/zmap.img``. The list is back-to-front: index 0 is
# drawn first (deepest), the last entry sits on top. Used when
# ``Base/zmap.img`` isn't reachable inside the Character pack
# (HaSuite-style exports frequently drop ``Base/``); the embedded
# copy keeps the renderer self-contained.
#
# The leading two-letter tokens (``Bd``, ``Hd``, …) are vslot
# aliases that Maple's zmap stores alongside the real z-slots; they
# never match any canvas's ``z`` string, so they sit harmlessly at
# the deepest indices.
#
# When a slot name appears that's not in this table, ``_z_index``
# falls back to a name-pattern heuristic (``back…`` → behind,
# ``…Below…`` → mid, ``…Over…`` → in front) so a previously-unseen
# variant still composites in roughly the right place instead of
# pinning to mid-stack.
_DEFAULT_ZMAP: Tuple[str, ...] = (
    "Bd", "Hd", "Hr", "Fc", "At", "Af", "Am", "Ae", "As", "Ay",
    "Cp", "Ri", "Gv", "Wp", "Si", "So", "Pn", "Ws", "Ma", "Wg",
    "Sr", "Tm", "Sd",
    "backTamingMobMid",
    "backMobEquipUnderSaddle",
    "backSaddle",
    "backMobEquipMid",
    "backTamingMobFront",
    "backMobEquipFront",
    "mobEquipRear",
    "tamingMobRear",
    "saddleRear",
    "characterEnd",
    "backWeaponEffectUnder",
    "backWeapon",
    "backWeaponEffectOver",
    "backHairBelowHead",
    "backShieldBelowBody",
    "backMailChestAccessory",
    "backCapAccessory",
    "backAccessoryFace",
    "backAccessoryEar",
    "backBody",
    "backGlove",
    "backGloveWrist",
    "backWeaponOverGloveEffectUnder",
    "backWeaponOverGlove",
    "backWeaponOverGloveEffectOver",
    "backMailChestBelowPants",
    "backPantsBelowShoes",
    "backShoesBelowPants",
    "backPants",
    "backShoes",
    "backPantsOverShoesBelowMailChest",
    "backMailChest",
    "backPantsOverMailChest",
    "backMailChestOverPants",
    "backHead",
    "backAccessoryFaceOverHead",
    "backAccessoryOverHead",
    "backCape",
    "backHairBelowCap",
    "backHairBelowCapNarrow",
    "backHairBelowCapWide",
    "backWeaponOverHeadEffectUnder",
    "backWeaponOverHead",
    "backWeaponOverHeadEffectOver",
    "backCap",
    "backHair",
    "backCapOverHair",
    "backShield",
    "backWeaponOverShieldEffectUnder",
    "backWeaponOverShield",
    "backWeaponOverShieldEffectOver",
    "backWing",
    "backHairOverCape",
    "weaponBelowBodyEffectUnder",
    "weaponBelowBody",
    "weaponBelowBodyEffectOver",
    "hairBelowBody",
    "capeBelowBody",
    "shieldBelowBody",
    "capAccessoryBelowBody",
    "gloveBelowBody",
    "gloveWristBelowBody",
    "body",
    "gloveOverBody",
    "mailChestBelowPants",
    "pantsBelowShoes",
    "shoes",
    "pants",
    "mailChestOverPants",
    "shoesOverPants",
    "pantsOverShoesBelowMailChest",
    "shoesTop",
    "mailChest",
    "pantsOverMailChest",
    "mailChestOverHighest",
    "gloveWristOverBody",
    "mailChestTop",
    "capeBelowWeapon",
    "weaponOverBodyEffectUnder",
    "weaponOverBody",
    "weaponOverBodyEffectOver",
    "armBelowHead",
    "mailArmBelowHead",
    "armBelowHeadOverMailChest",
    "gloveBelowHead",
    "mailArmBelowHeadOverMailChest",
    "gloveWristBelowHead",
    "weaponOverArmBelowHeadEffectUnder",
    "weaponOverArmBelowHead",
    "weaponOverArmBelowHeadEffectOver",
    "shield",
    "weaponEffectUnder",
    "weapon",
    "weaponEffectOver",
    "arm",
    "hand",
    "glove",
    "mailArm",
    "gloveWrist",
    "cape",
    "head",
    "hairShade",
    "accessoryFaceBelowFace",
    "accessoryEyeBelowFace",
    "face",
    "accessoryFaceOverFaceBelowCap",
    "capBelowAccessory",
    "accessoryEar",
    "capAccessoryBelowAccFace",
    "accessoryFace",
    "accessoryEyeShadow",
    "accessoryEye",
    "capeOverFace",
    "hair",
    "cap",
    "capAccessory",
    "accessoryEyeOverCap",
    "hairOverHead",
    "accessoryOverHair",
    "accessoryEarOverHair",
    "capOverHair",
    "weaponBelowArmEffectUnder",
    "weaponBelowArm",
    "weaponBelowArmEffectOver",
    "armOverHairBelowWeapon",
    "mailArmOverHairBelowWeapon",
    "armOverHair",
    "gloveBelowMailArm",
    "mailArmOverHair",
    "gloveWristBelowMailArm",
    "weaponOverArmEffectUnder",
    "weaponOverArm",
    "weaponOverArmEffectOver",
    "handBelowWeapon",
    "gloveBelowWeapon",
    "gloveWristBelowWeapon",
    "shieldOverHair",
    "weaponOverHandEffectUnder",
    "weaponOverHand",
    "weaponOverHandEffectOver",
    "handOverHair",
    "gloveOverHair",
    "gloveWristOverHair",
    "weaponOverGloveEffectUnder",
    "weaponOverGlove",
    "weaponOverGloveEffectOver",
    "capeOverHead",
    "weaponWristOverGloveEffectUnder",
    "weaponWristOverGlove",
    "weaponWristOverGloveEffectOver",
    "emotionOverBody",
    "characterStart",
    "backSaddleFront",
    "saddleMid",
    "tamingMobMid",
    "mobEquipUnderSaddle",
    "saddleFront",
    "mobEquipMid",
    "tamingMobFront",
    "mobEquipFront",
)

# Slots whose z is ABSOLUTE — independent of pose orientation. The
# back-facing branch in ``z_for`` normally pushes any non-``back*``
# slot to ``back_floor`` so that front-only canvases (face, hair,
# etc., with no back variant) sink behind the body silhouette. These
# slots opt out of that rule because they're explicit absolute
# positions: ``characterStart`` always renders in front of every
# other layer, ``characterEnd`` always renders behind every other
# layer, regardless of whether the character is being shown from
# the front or the back. The back-facing values are picked to
# beat the head/cap/shield/weapon overrides above (max ~275).
_BACK_FACING_ABSOLUTE_Z: Dict[str, int] = {
    "characterStart": 999,    # frontmost — above every back-facing override
    "characterEnd":   -999,   # deepest — below every other slot
}


# Conditional z-slot remap applied during compose. Each entry is
# ``(target_slot, vslot_token)``: the canvas's declared z-slot is
# rewritten to ``target_slot`` whenever the remap should kick in. The
# remap kicks in if EITHER:
#   1. The equipped cap (if any) doesn't hide any hair, leaving the
#      bangs visible (so ``*OverCap`` accessories drop behind them).
#   2. The equipped cap's vslot lists ``vslot_token``, meaning the
#      cap claims the accessory's slot — e.g., a hat with vslot
#      containing ``Ay`` covers the eye-accessory area, so an
#      ``accessoryEyeOverCap`` glass must render BEHIND that cap
#      regardless of what the slot name says (caps 01004141..48 vs
#      glass 01022032). ``vslot_token=None`` means "only the
#      doesn't-hide-hair arm applies" — used for the bare ``cap``
#      slot which is about bangs visibility, not accessory cover.
# With a hair-hiding cap on AND no matching vslot token, the remap
# is skipped — the canvases stay at their declared slot.
#
# ``capOverHair`` and ``capAccessory`` are deliberately left out:
# their slot names promise "above hair / above the cap canvas",
# which the user wants honored even when the cap doesn't hide hair
# (that's how 01002575 / 01002576 / 01002598 / 01002842 render in
# front of the bangs).
_OVER_CAP_REMAP: Dict[str, Tuple[str, Optional[str]]] = {
    "accessoryEyeOverCap": ("accessoryEye", "Ay"),
    "accessoryFaceOverCap": ("accessoryFace", "Af"),
    "accessoryFaceUpperOverCap": ("accessoryFace", "Af"),
    "cap": ("capBelowHairOverHead", None),
}


def _parse_vslot_tokens(vslot: Optional[str]) -> frozenset:
    """Split a cap's vslot string into its 2-character tokens.

    The vslot is a concatenation of two-char slot names like ``Cp``,
    ``H1``, ``Hd``, ``Af``, ``Ay``, ``Ae``. We don't need to validate
    the alphabet — anything length-2 is a token, and unknown tokens
    just don't match any of the rules that consume them.
    """
    if not vslot:
        return frozenset()
    return frozenset(vslot[i:i + 2] for i in range(0, len(vslot), 2))

# Per-category candidate frame paths to walk when collecting render leaves.
# We try each path in order, dedupe by canvas identity (UOLs into the same
# target only contribute once), and combine the results.
# NOTE: ``backDefault`` is intentionally excluded. It's the back-facing-pose
# variant Maple swaps in when the character is shown from behind (rope/
# ladder climbing), and contains canvases like ``backHair`` /
# ``backHairBelowCap`` whose only purpose is to fill in the back of the
# head when you can't see the face. Including them in our front-facing
# stand1 composite was rendering back-of-head hair behind the body and
# masquerading as a fallback for missing ``default/hairBelowBody``,
# which is wrong: hair styles without ``hairBelowBody`` simply don't
# show flowing hair behind the torso in front-facing poses.
# Curated set of action subtrees the static builder exposes. The first
# two are rest poses (one-handed / two-handed) — every weapon ships
# them and the breathing-anchor stabilization in
# :meth:`CharacterRenderer.compose_animation` is tuned for them. The
# rest are real-motion actions (locomotion, alert, attacks); their
# multi-frame canvases animate naturally and we do NOT freeze head
# anchors across frames for those (see the ``stabilize`` branch in
# ``compose_animation``).
SUPPORTED_POSES: Tuple[str, ...] = (
    "stand1", "stand2",
    "walk1", "walk2",
    "alert",
    "jump",
    "prone", "proneStab",
    "sit",
    "fly",
    "ladder", "rope",
    "heal",
    "swingO1", "swingO2", "swingO3",
    "swingT1", "swingT2", "swingT3",
    "stabO1", "stabO2",
    "stabT1", "stabT2",
    "shoot1", "shoot2", "shootF",
    "dead",
)
DEFAULT_POSE = "stand1"

# Subset used by ``_render_root``'s cash-weapon fallback (when the
# requested pose isn't authored in any numeric weapon-num child).
# Cash weapons always ship a rest pose so this tiny set is enough.
_WEAPON_FALLBACK_POSES: Tuple[str, ...] = ("stand1", "stand2")

# Poses that always remain selectable, even when the equipped weapon
# doesn't ship art for them. These are back-facing climbing poses
# (ladder / rope) — body / hair / cap art is what the user wants to
# inspect, and the weapon (held in front during normal poses) just
# isn't visible from behind. The renderer omits the weapon canvas
# rather than refusing the pose.
_WEAPON_OPTIONAL_POSES: frozenset = frozenset({"ladder", "rope"})

# Each Head image ships its ``head`` canvas alongside one or more ear
# variants under ``front/`` (e.g. ``humanEar``, ``lefEar``,
# ``highlefEar``). The real client picks one to render based on the
# character's race; we mirror that by filtering Head canvases to keep
# only ``head`` plus the canvas matching the selected ``ear_type``.
DEFAULT_EAR_TYPE = "humanEar"


# Caps occupy "visual slots" listed in ``info/vslot``: a concatenation
# of two-character tokens like ``Cp`` (the cap itself), ``H1``..``H6``
# (hair sub-slots), ``Hd``, ``Hs``, ``Hf``, ``Hb`` (more hair), and
# ``Af``/``Ay``/``As``/``Ae``/``Fc`` (face/eye/ear accessories).
#
# Hair-hiding decision is token-based — the original length-cutoff
# heuristic from MapleNecrocer's ``CapType`` derivation
# (MapleCharacter.cs:301) misses edge cases like 01002470's
# ``CpHdH1H2H3H4`` (length exactly 12, clearly meant to hide hair).
# The token rule below replaces it without changing the answer for
# any of the canonical vslots:
#
#   * ``H2`` token present                          → full helmet
#                                                     (every Hair canvas hidden, returns ``None``)
#   * ``H1`` and ``H3`` tokens present              → hide the four
#                                                     "front-hair" canvases
#   * ``H1`` token present (alone or with H4..H6)   → hide ``hairOverHead`` + ``backHair``
#   * otherwise                                     → no hair hidden
#
# A returned ``None`` means "every Hair canvas is hidden". An empty
# frozenset means "show everything".
_CAP_HIDE_PARTIAL: frozenset = frozenset(
    {"hairOverHead", "backHair", "hairBelowBody", "backHairBelowCap"}
)
_CAP_HIDE_TOP: frozenset = frozenset({"hairOverHead", "backHair"})


def _cap_hidden_hair_canvases(vslot: Optional[str]) -> Optional[frozenset]:
    """Resolve a cap's ``info/vslot`` string to the set of Hair canvas
    names it covers, or ``None`` if it covers every Hair canvas."""
    tokens = _parse_vslot_tokens(vslot)
    if "H2" in tokens:
        return None
    if "H1" in tokens and "H3" in tokens:
        return _CAP_HIDE_PARTIAL
    if "H1" in tokens:
        return _CAP_HIDE_TOP
    return frozenset()


def _frame_paths(category: str, pose: str, frame: int = 0) -> Tuple[str, ...]:
    """Return ordered candidate frame paths for collecting render leaves.

    Pose-aware: ``stand2`` swaps in the two-handed body / coat / weapon
    canvases. ``frame`` selects which sub-frame of the pose to render
    — Body's ``stand1`` ships ``0`` / ``1`` / ``2`` (the standby
    breathing animation), and most equipment imgs follow suit so the
    full character can be cycled 0→1→2→1 in the preview. Hair / cap
    / face / accessories ignore pose AND frame because their static
    canvases live under ``default`` (the per-pose ``stand*`` subtrees
    are just UOLs back to it)."""
    pf = f"{pose}/{frame}"
    if category == "Body":      return (pf,)
    if category == "Head":      return (pf, "front")
    if category == "Hair":      return (pf, "default")
    if category == "Face":      return ("default",)
    if category == "Cap":       return (pf, "default")
    if category == "Cape":      return (pf,)
    if category == "Coat":      return (pf,)
    if category == "Longcoat":  return (pf,)
    if category == "Pants":     return (pf,)
    if category == "Shoes":     return (pf,)
    if category == "Glove":     return (pf,)
    if category == "Shield":    return (pf,)
    if category == "Weapon":    return (pf,)
    if category in ("FaceAcc", "Glass", "Earring"): return ("default",)
    return (pf, "default")


# ── helpers ──────────────────────────────────────────────────────────────

def category_for_id(equip_id: str) -> Optional[str]:
    """Return the category for an equip ID like ``"01302000"``, or None."""
    if not equip_id or not equip_id.isdigit():
        return None
    return _CATEGORY_BY_ID_PREFIX.get(int(equip_id) // 10000)


def _resolve_uol(node: Optional[WzProperty]) -> Optional[WzProperty]:
    """Follow a UOL chain to its non-UOL target. Same idea as the
    server-side helper, but local so the renderer doesn't depend on Flask."""
    seen: set = set()
    cur: Optional[WzProperty] = node
    for _ in range(16):
        if cur is None or not isinstance(cur, WzUolProperty):
            return cur
        if id(cur) in seen:
            return None
        seen.add(id(cur))
        target_str = cur.value
        if not target_str or cur.parent is None:
            return None
        cur = cur.parent.get(target_str)
    return None


def _vec(prop: Optional[WzProperty]) -> Optional[Tuple[int, int]]:
    p = _resolve_uol(prop) if prop is not None else None
    if isinstance(p, WzVectorProperty):
        return (p.x, p.y)
    return None


def _string(prop: Optional[WzProperty]) -> Optional[str]:
    p = _resolve_uol(prop) if prop is not None else None
    if isinstance(p, WzStringProperty):
        return p.value
    return None


def _origin(canvas: WzCanvasProperty) -> Tuple[int, int]:
    return _vec(canvas.child("origin")) or (0, 0)


def _map_point(canvas: WzCanvasProperty, name: str) -> Optional[Tuple[int, int]]:
    map_node = canvas.child("map")
    if isinstance(map_node, WzSubProperty):
        return _vec(map_node.child(name))
    return None


def _map_anchors(canvas: WzCanvasProperty) -> Dict[str, Tuple[int, int]]:
    """All ``map/<name>`` vectors as a dict (UOL-resolved)."""
    out: Dict[str, Tuple[int, int]] = {}
    map_node = canvas.child("map")
    if isinstance(map_node, WzSubProperty):
        for c in map_node.children():
            v = _vec(c)
            if v is not None:
                out[c.name] = v
    return out


def _z_slot(canvas: WzCanvasProperty) -> Optional[str]:
    return _string(canvas.child("z"))


# ── per-part canvas collection ────────────────────────────────────────────

@dataclass
class _Placement:
    equip_id: str
    category: str
    name: str                          # leaf canvas name (e.g. "body", "arm")
    canvas: WzCanvasProperty           # owns the metadata: origin / map / z
    pixel_canvas: WzCanvasProperty     # owns the actual pixels (== canvas, except
                                       # for hierarchical _outlink placeholders)
    origin: Tuple[int, int]
    map_anchors: Dict[str, Tuple[int, int]]
    z_slot: Optional[str]
    top_left: Optional[Tuple[int, int]] = None
    # ItemEff overlays carry an integer z that doesn't map to a zmap
    # slot name. ``z_for`` reads this when ``z_slot`` is None and the
    # category is "Effect" — positive z lands AFTER all character
    # parts, negative z BEFORE the body.
    extra_z: Optional[int] = None
    # Anchor name actually used to compute ``top_left`` (e.g. ``brow``
    # for pos=1 ItemEff overlays). compose_animation's body-delta
    # translation step uses this to decide whether the placement is
    # head-anchored (skip translation, stays glued to the frozen
    # head) or body-anchored (track the body's per-frame motion).
    # Only Effect placements set this today — character placements
    # let ``_determine_anchor`` derive the anchor from the canvas's
    # ``map`` instead.
    anchor_override: Optional[str] = None
    # Pre-decoded RGBA Image. When set, ``_render_placements`` uses
    # it directly instead of calling ``decode_canvas(pixel_canvas)``
    # — needed for stabilized stand-pose ItemEff overlays where we
    # composite each frame's content into a uniform-size canvas to
    # eliminate per-frame bbox extent variation.
    decoded_override: Optional[Any] = None
    # Width/height the bbox calculation should use when the placement
    # carries a decoded_override whose dimensions differ from the
    # source pixel_canvas. Falls back to pixel_canvas.width/height
    # when None.
    width_override: Optional[int] = None
    height_override: Optional[int] = None


def _is_cash_weapon(equip_id: str) -> bool:
    """Cash weapons (170xxxx range) nest their actions one level deeper
    inside numeric weapon-num children like ``30`` / ``31`` / …"""
    return bool(equip_id) and equip_id.startswith("0170")


# Color-encoding rules for Hair / Face. Each tuple is
# ``(base_divisor, color_modulus)`` such that ``id // base_divisor``
# is shared across colors and ``(id // (base_divisor // 10)) % 10``
# yields the color index. For Hair the color sits in the ones digit
# (last). For Face the color sits in the hundreds digit, e.g.
# ``00022017``, ``00022117`` … ``00022817`` are the same style.
_HAIR_COLOR_INDEX = (10, 1)     # base = id // 10, color = id % 10
_FACE_COLOR_INDEX = (1000, 100)  # base = (id//1000)*100 + id%100, color = (id//100)%10


def _dedupe_by_color(
    parts: List[Dict[str, Any]],
    rule: Tuple[int, int],
) -> List[Dict[str, Any]]:
    """Collapse color-variant groups to a single canonical entry.

    ``rule`` is ``(base_divisor, color_unit)``:
      * ``base_divisor``: divide the numeric ID by this and keep the
        quotient — this is the "style" key that variants share.
        For Hair (``10``), the last digit is dropped. For Face
        (``1000``), the hundreds digit and below are dropped.
      * ``color_unit``: divide the ID by this and modulo 10 to read
        out the color digit (1 for Hair's last digit, 100 for
        Face's hundreds digit).

    The smallest-numbered variant in each group becomes the canonical
    ``id`` so the icon thumbnail and default render show the leftmost
    color in the swatch row (typically black). The full set of
    available color indices is returned under ``colors`` so the UI
    can disable missing colors.
    """
    base_div, color_unit = rule
    grouped: Dict[int, Dict[str, Any]] = {}
    for entry in parts:
        n = int(entry["id"])
        # Strip the color digit out of the ID so styles that differ
        # only in color collapse to the same key. ``base = n - color
        # * color_unit`` is robust whether the color sits in the
        # ones, hundreds, or any other position.
        color = (n // color_unit) % 10
        base = n - color * color_unit
        existing = grouped.get(base)
        if existing is None:
            new_entry = dict(entry)
            new_entry["colors"] = [color]
            grouped[base] = new_entry
        else:
            existing["colors"].append(color)
            if n < int(existing["id"]):
                existing["id"] = entry["id"]
                existing["icon_paths"] = entry["icon_paths"]
                existing["icon_path"] = entry["icon_path"]
    out = list(grouped.values())
    for e in out:
        e["colors"].sort()
    out.sort(key=lambda r: int(r["id"]))
    return out


_INFO_ONLY: frozenset = frozenset({"info"})


def _read_cash_flag(img: WzImage) -> bool:
    """True when the image's ``info/cash`` is a non-zero int.

    Body / Head don't ship an ``info`` subdir at all, so they always
    return False — which matches reality (the character bases aren't
    cash items). Other categories typically carry ``info/cash = 0``
    for the regular ID range and ``info/cash = 1`` for cash-shop
    variants; the pre-built ID prefix isn't a reliable signal because
    cash and non-cash IDs share the same 4-digit prefix in stock data.

    Uses ``WzImage.parse_partial`` so we read just the ``info``
    subtree — for a Weapon img that's a few small scalars instead of
    the full pose / frame / canvas-metadata walk (~30x faster). The
    full-parse cache is left untouched so ``compose`` / canvas
    requests still trigger and cache a complete parse on demand.
    """
    try:
        info = img.parse_partial(only=_INFO_ONLY).get("info")
    except Exception:
        return False
    if not isinstance(info, WzSubProperty):
        return False
    cash = info.get("cash")
    try:
        return bool(cash and getattr(cash, "value", 0))
    except Exception:
        return False


def _has_action(node: WzSubProperty, action: str) -> bool:
    """True if ``<node>/<action>/0`` resolves to a real SubProperty
    (handles UOLs that redirect e.g. ``41/stand2`` → ``../43/stand2``)."""
    n = node.get(action)
    n = _resolve_uol(n) if isinstance(n, WzUolProperty) else n
    if not isinstance(n, WzSubProperty):
        return False
    fr = n.get("0")
    fr = _resolve_uol(fr) if isinstance(fr, WzUolProperty) else fr
    return isinstance(fr, WzSubProperty)


def _pose_data_home(numeric_root: WzSubProperty, pose: str) -> Optional[WzSubProperty]:
    """Walk ``numeric_root.<pose>`` through any UOLs and return the
    weapon-num SubProperty that *physically* owns the pose data.

    Cash weapons (e.g. ``01702087``) commonly have multiple
    weapon-nums where only one ships the real ``stand2`` tree and the
    rest carry ``stand2`` as a UOL pointing at it (``../43/stand2``).
    Returning the UOL'd numeric child as the render root makes the
    later ``base.get("stand2/0")`` lookup fail (``WzProperty.get``
    doesn't follow intermediate UOLs), so we always resolve to the
    home that owns the canvas tree.
    """
    pose_node = numeric_root.child(pose)
    resolved = _resolve_uol(pose_node) if isinstance(pose_node, WzUolProperty) else pose_node
    if not isinstance(resolved, WzSubProperty):
        return None
    fr0 = resolved.get("0")
    fr0 = _resolve_uol(fr0) if isinstance(fr0, WzUolProperty) else fr0
    if not isinstance(fr0, WzSubProperty):
        return None
    home = resolved.parent
    if isinstance(home, WzSubProperty):
        return home
    return numeric_root


def _render_root(img: WzImage, category: str, equip_id: str, pose: str) -> WzProperty:
    """Return the SubProperty under which the per-action subtrees live.

    For ordinary parts that's the .img root (``stand1`` / ``walk1`` /
    ``default`` etc. live directly underneath). Cash weapons need an
    extra hop: the .img root holds numeric weapon-num children, and the
    pose's action tree lives inside *one* of them — but stand1 and
    stand2 may be owned by *different* numeric children (in
    ``01702087`` stand1 is in ``44``, stand2 in ``43``), and the rest
    of the children just UOL into them. We resolve the UOL chain so the
    render root is the actual data owner, not a UOL stub.
    """
    root = img.parse()
    if category == "Weapon" and _is_cash_weapon(equip_id):
        numeric = sorted(
            (c for c in root.children()
             if c.name.isdigit() and isinstance(c, WzSubProperty)),
            key=lambda c: int(c.name),
        )
        # First pass: smallest weapon-num whose <pose> data home exists.
        for c in numeric:
            home = _pose_data_home(c, pose)
            if home is not None:
                return home
        # Second pass: any rest pose (rare; weapon ships only one).
        # Limited to stand1/stand2 because cash weapons are weapons —
        # the action poses (walk1, swing*, stab*, …) are all UOLs back
        # to a rest pose, and falling back to those would just chase
        # a UOL chain that the first pass already covered.
        for p in _WEAPON_FALLBACK_POSES:
            for c in numeric:
                home = _pose_data_home(c, p)
                if home is not None:
                    return home
        # Last resort: smallest numeric child, even without action data.
        if numeric:
            return numeric[0]
    return root


def _collect_part_canvases(
    img: WzImage, category: str, equip_id: str, pose: str,
    pkg_root: Optional[WzDirectory] = None,
    ear_type: str = DEFAULT_EAR_TYPE,
    hide_hair: frozenset = frozenset(),
    frame: int = 0,
) -> List[Tuple[str, WzCanvasProperty, WzCanvasProperty]]:
    """Return ``(leaf_name, metadata_canvas, pixel_canvas)`` triples to render.

    Walks each frame-path candidate from the appropriate render root,
    follows UOLs, and deduplicates by canvas identity. For hierarchical
    packs the per-frame canvas is a 1×1 placeholder with ``_outlink``
    pointing into a ``_Canvas`` sibling — we resolve that link and
    return the linked canvas as ``pixel_canvas`` while keeping the
    placeholder as the metadata canvas (it owns ``origin`` / ``map`` /
    ``z``). When no link is present, both fields point at the same
    canvas. ``pkg_root`` is the WZ root used for absolute outlink
    navigation; when ``None`` (legacy single-file Character.wz), only
    ``_inlink`` resolves.

    For Head, ``front/`` siblings other than ``head`` are treated as ear
    variants and filtered to just the one matching ``ear_type`` —
    matching MapleNecrocer's per-frame visibility filter so we don't
    composite (e.g.) ``humanEar`` and ``lefEar`` on top of each other.

    Frame paths are tried in priority order and treated as
    EXCLUSIVE: the first path that produces any canvases wins and
    later paths are skipped. ``stand1/0`` typically UOLs to the
    canvases that live under ``default`` / ``front`` for Hair /
    Head / standard caps, so UOL resolution + the existing
    ``id()``-based dedup already collapses them to the same
    placements. The exclusive walk fixes caps that ship distinct
    placeholders in both paths — 01003843's ``default/default``
    (z=``capOverHairOverHair``, a static rest frame) and
    ``stand1/0/0`` (z=``capOverHair``, a real action frame) both
    rendered before, doubling the cap. Caps with only a
    ``default/default`` (no ``stand1/0`` body) still render
    because the empty primary path leaves ``out_before == len(out)``
    and the loop falls through to ``default``.

    Within a single path the z-slot dedup still fires as a safety
    net so any same-z duplicates inside one frame collapse cleanly."""
    base = _render_root(img, category, equip_id, pose)
    seen_ids: set = set()
    seen_zslots: set = set()
    out: List[Tuple[str, WzCanvasProperty, WzCanvasProperty]] = []
    for path in _frame_paths(category, pose, frame):
        node = base.get(path)
        node = _resolve_uol(node) if isinstance(node, WzUolProperty) else node
        if not isinstance(node, WzSubProperty):
            continue
        out_before = len(out)
        for child in node.children():
            # For Head, every sibling of ``head`` is treated as an ear
            # variant (matches MapleNecrocer's per-frame visibility
            # filter in ``MapleCharacter.cs:1218``). Filter is name-based
            # rather than path-based because ``stand1/0`` UOLs into
            # ``front`` — without this the UOL'd ears get collected
            # before we ever reach ``front`` and the dedupe table hides
            # them from the explicit ``front`` filter.
            if category == "Head" and child.name not in ("head", ear_type):
                continue
            # For Hair, drop canvases the equipped Cap covers (mirrors
            # MapleCharacter.cs:1189-1212 — when DressCap and ShowHair
            # are both true the cap's vslot decides which hair canvases
            # stay visible). The "hide every Hair" case (full helmet)
            # is short-circuited in ``compose`` so we don't need a
            # special sentinel here.
            if category == "Hair" and child.name in hide_hair:
                continue
            target = _resolve_uol(child) if isinstance(child, WzUolProperty) else child
            if not isinstance(target, WzCanvasProperty):
                continue
            if id(target) in seen_ids:
                continue
            # Skip a canvas whose z slot is already covered by an
            # earlier frame-path. ``z`` is read after the canvas-type
            # check (cheap — just a child lookup) and ``None`` z
            # slots are NOT deduped because the heuristic anchor
            # falls back to per-canvas placement and treating them
            # all as a single slot would drop legitimate layers.
            z_slot = _z_slot(target)
            if z_slot is not None and z_slot in seen_zslots:
                continue
            # Resolve _outlink/_inlink (a no-op when neither child is
            # present, returns the original canvas). For hierarchical
            # packs the placeholder is 1×1 and the link target carries
            # the real pixels; for legacy single-file Character.wz the
            # placeholder *is* the pixel canvas.
            pixels: WzCanvasProperty = target
            if (target.child("_outlink") is not None
                    or target.child("_inlink") is not None):
                root_for_link = pkg_root if pkg_root is not None \
                    else WzDirectory(name="")
                resolved = resolve_canvas_link(target, root_for_link)
                if isinstance(resolved, WzCanvasProperty):
                    pixels = resolved
            if not pixels.has_pixels():
                continue
            seen_ids.add(id(target))
            if z_slot is not None:
                seen_zslots.add(z_slot)
            out.append((child.name, target, pixels))
        if len(out) > out_before:
            # Primary path produced canvases — don't walk the
            # fallback path. See docstring.
            break
    return out


def _determine_anchor(canvas: WzCanvasProperty, category: str) -> str:
    """Pick which ``map/<name>`` anchor to use for placement.

    Highest-priority anchor present on the canvas wins; falls back to the
    category default (e.g. ``brow`` for face, ``hand`` for weapon) so a
    canvas with an empty ``map`` still gets a consistent reference."""
    anchors = _map_anchors(canvas)
    for name in _ANCHOR_PRIORITY:
        if name in anchors:
            return name
    return _DEFAULT_ANCHOR_BY_CATEGORY.get(category, "navel")


# ── renderer ─────────────────────────────────────────────────────────────

class CharacterRenderer:
    """Compose static MapleStory character frames from a Character.wz."""

    def __init__(
        self, wz: WzFile, region: str = "GMS",
        effects: Optional[Any] = None,
    ):
        self.wz = wz
        self.region = region
        self._zmap: Tuple[str, ...] = self._load_zmap()
        # Optional ``EffectLookup`` (wzpy.effect_lookup) for ItemEff
        # overlays. ``None`` disables overlay rendering, which is the
        # default when no Effect sibling is found next to Character.wz.
        self.effects = effects

    # ── tree traversal helpers ──────────────────────────────────────────
    def _load_zmap(self) -> Tuple[str, ...]:
        """Try to read ``Base/zmap.img`` from the Character pack itself
        (HaSuite-exported packs sometimes embed it); otherwise fall back
        to the embedded canonical :data:`_DEFAULT_ZMAP`."""
        node = self.wz.root.get("Base/zmap.img")
        if isinstance(node, WzImage):
            try:
                node.parse()
                names = tuple(c.name for c in node.children())
                if names:
                    # zmap.img is ordered front-to-back (lower index = on top
                    # in the Maple client). HaCreator's ZMapReference is
                    # stored back-to-front; flip so our index convention is
                    # uniform: lower = behind.
                    return tuple(reversed(names))
            except Exception:
                pass
        return _DEFAULT_ZMAP

    def _z_index(self, slot: Optional[str]) -> int:
        if not slot:
            return len(self._zmap)  # no slot → drawn in front
        try:
            return self._zmap.index(slot)
        except ValueError:
            # Unknown slot — use the name to guess a reasonable position
            # rather than pinning to mid-stack (which dropped back-hair
            # variants like ``backHairBelowCapWide`` in front of the
            # body). Multiplied indices keep us inside the integer range
            # the regular zmap uses, with deliberate spacing so future
            # additions can slot in without rebalancing.
            anchor = self._heuristic_anchor(slot)
            return anchor

    def _heuristic_anchor(self, slot: str) -> int:
        """Best-effort placement for a previously-unseen z-slot."""
        n = len(self._zmap)
        s = slot.lower()
        # Anything starting with ``back`` is a back-side variant; keep it
        # well behind the body.
        if s.startswith("back"):
            try:
                # If we have a non-back equivalent, sit just behind it.
                base = slot[4].lower() + slot[5:] if len(slot) > 4 else slot
                return max(1, self._zmap.index(base) - 5)
            except (ValueError, IndexError):
                pass
            # ``backXxxBelow/OverYyy`` variants the stock zmap omits
            # (e.g. ``backWeaponBelowGlove``) resolve against the
            # BACK-prefixed target so they stay down in the back
            # cluster. Without this they fell to ``body`` - 1 (~84,
            # mid front-stack) and, in a climbing pose, painted the
            # weapon on top of the back of the head — the same defect
            # the canonical-order ``backWeapon*`` slots avoid. Sitting
            # just behind ``backGlove`` keeps a "below glove" weapon
            # with the rest of the back-strapped gear, below the head.
            for sep, delta in (("Below", -1), ("Over", 1)):
                if sep in slot:
                    tail = slot.split(sep)[-1]
                    sib = "back" + tail[:1].upper() + tail[1:] if tail else ""
                    if sib in self._zmap:
                        return max(1, self._zmap.index(sib) + delta)
            # Generic back layer — between shadow and body.
            return max(1, self._zmap.index("body") - 1) if "body" in self._zmap else 1
        # Non-canonical cap*Below* slots — ``capBelowBody`` /
        # ``capBelowHead`` / ``capBelowHair`` aren't in Maple's stock
        # ``Base/zmap.img`` but a number of caps use them, and in
        # those caps the *Below* canvas frequently IS the main
        # visible hat shape (e.g. 01001036's mushroom dome on
        # ``capBelowBody`` while the front-side ``cap`` canvas is
        # just a thin white trim). Default these slots to land just
        # before the front ``cap`` slot so the main hat draws above
        # hair/head; the cap-back-piece detection in
        # ``_build_placements`` handles the secondary back-piece
        # case (demoting them behind the body when the front cap
        # canvas is the larger one).
        if s.startswith("cap") and "below" in s and "cap" in self._zmap:
            return max(1, self._zmap.index("cap") - 1)
        # ``…Below…`` → sits behind the named target.
        for separator in ("Below", "below"):
            if separator in slot:
                target = slot.split(separator)[-1]
                target = target[0].lower() + target[1:] if target else target
                if target in self._zmap:
                    return max(1, self._zmap.index(target) - 1)
        # ``…Over…`` → sits in front of the named target.
        for separator in ("Over", "over"):
            if separator in slot:
                target = slot.split(separator)[-1]
                target = target[0].lower() + target[1:] if target else target
                if target in self._zmap:
                    return min(n - 1, self._zmap.index(target) + 1)
        # Default: front-leaning (small, decorative slots like
        # ``emotionOverBody`` should be visible).
        return n - 1

    def _open_part(self, equip_id: str) -> Optional[WzImage]:
        cat = category_for_id(equip_id)
        if cat is None:
            return None
        sub = CATEGORY_DIR.get(cat, "")
        path = f"{sub}/{equip_id}.img" if sub else f"{equip_id}.img"
        node = self.wz.root.get(path)
        return node if isinstance(node, WzImage) else None

    # ── public API ──────────────────────────────────────────────────────
    def list_parts(self, category: str) -> List[Dict[str, Any]]:
        """Enumerate available equip IDs in the given category.

        Returns
        ``[{"id": "01002000", "icon_paths": ["Cap/01002000.img/info/icon"]}, …]``.
        Each ``icon_paths`` entry is a list of WZ-relative thumbnail paths
        the client should try in order; the first one with pixel data
        wins, the rest are fallbacks. Hair styles in particular don't all
        ship a ``default/hairOverHead`` canvas (some short hair only has
        ``default/hair``), so we hand the client both candidates and let
        it pick — keeps ``list_parts`` from having to parse 1500 imgs to
        probe which canvas actually exists.
        """
        sub = CATEGORY_DIR.get(category)
        if sub is None:
            return []
        if sub == "":
            target = self.wz.root
        else:
            target = self.wz.root.get(sub)
            if not isinstance(target, WzDirectory):
                return []

        results: List[Dict[str, Any]] = []
        for img_name, img in target.images.items():
            # Match the category by ID prefix (so we don't surface non-equip
            # imgs like ``info.img`` if the WZ ever contains them).
            stem = img_name.split(".")[0]
            if not stem.isdigit():
                continue
            cat = category_for_id(stem)
            if cat != category:
                continue

            entry: Dict[str, Any] = {"id": stem}
            # Pick a sensible thumbnail path. Body / Head have no info
            # subdir; use front/head or stand1/0/body. Hair / Face have
            # a default frame canvas. Everything else has info/icon.
            prefix = f"{sub}/{img_name}" if sub else img_name
            if category == "Body":
                paths = [f"{prefix}/stand1/0/body"]
            elif category == "Head":
                paths = [f"{prefix}/front/head"]
            elif category == "Hair":
                # Most hairs have hairOverHead; some short styles
                # (e.g., 00030030) only ship the bare ``hair`` canvas.
                paths = [
                    f"{prefix}/default/hairOverHead",
                    f"{prefix}/default/hair",
                ]
            elif category == "Face":
                paths = [f"{prefix}/default/face"]
            else:
                paths = [f"{prefix}/info/icon"]
            entry["icon_paths"] = paths
            # Back-compat: keep ``icon_path`` set to the first candidate
            # so older clients that expected a single string still work.
            entry["icon_path"] = paths[0]
            # Read ``info/cash`` only for categories the UI offers a
            # cash / non-cash sub-tab on — parsing the img on others
            # (Hair: 16k imgs, ~30s) is a wasted first-load regression
            # because the response field would never be consumed.
            if category in _CASH_FILTERED_CATEGORIES:
                entry["cash"] = _read_cash_flag(img)
            results.append(entry)

        # Natural sort by ID.
        results.sort(key=lambda r: int(r["id"]))

        # Hair / Face styles encode color in a specific digit position
        # (last digit for Hair, hundreds digit for Face). Most styles
        # ship the full color palette as consecutive IDs (Hair: 8
        # colors, Face: 9), so collapse the listing to one entry per
        # style, keep the smallest variant as ``id``, and surface the
        # available color indices so the client can build a color
        # picker that grays out colors the style doesn't ship.
        if category == "Hair":
            results = _dedupe_by_color(results, _HAIR_COLOR_INDEX)
        elif category == "Face":
            results = _dedupe_by_color(results, _FACE_COLOR_INDEX)
        return results

    # ── pose discovery ──────────────────────────────────────────────────
    def get_weapon_poses(self, equip_id: str) -> List[str]:
        """Return the action subtrees a weapon ships with — the subset
        of :data:`SUPPORTED_POSES` whose ``<pose>/0`` resolves to a
        real frame on this weapon. The UI uses this to filter the
        pose dropdown so the user only sees poses the equipped
        weapon actually has art for."""
        if category_for_id(equip_id) != "Weapon":
            return []
        img = self._open_part(equip_id)
        if img is None:
            return []
        root = img.parse()
        # Cash weapons: scan every numeric weapon-num. Non-cash: just
        # the .img root.
        if _is_cash_weapon(equip_id):
            roots = [c for c in root.children()
                     if c.name.isdigit() and isinstance(c, WzSubProperty)]
        else:
            roots = [root]
        seen: List[str] = []
        for p in SUPPORTED_POSES:
            for r in roots:
                if _has_action(r, p):
                    seen.append(p)
                    break
        return seen

    def pose_frame_delays(
        self, pose: str, body_id: str = "00002000",
    ) -> List[int]:
        """Return per-frame delays (ms) for the given pose, read from
        ``Body/<body_id>.img/<pose>/<n>/delay``. The list length is
        the number of authored frames; an empty list means the body
        doesn't ship this pose. Frames missing an explicit ``delay``
        default to 100 ms — same fallback the Maple client uses for
        single-frame actions like ``prone`` / ``sit`` that ship no
        delay because they're held until interrupted."""
        img = self._open_part(body_id)
        if img is None:
            return []
        node = img.parse().get(pose)
        node = _resolve_uol(node) if isinstance(node, WzUolProperty) else node
        if not isinstance(node, WzSubProperty):
            return []
        frames = sorted(
            (c for c in node.children() if c.name.isdigit()),
            key=lambda c: int(c.name),
        )
        out: List[int] = []
        for fr in frames:
            d = fr.get("delay") if isinstance(fr, WzSubProperty) else None
            d = _resolve_uol(d) if isinstance(d, WzUolProperty) else d
            v = getattr(d, "value", None) if d is not None else None
            try:
                ms = int(v) if v is not None else 0
            except (TypeError, ValueError):
                ms = 0
            out.append(ms if ms > 0 else 100)
        return out

    def effect_frame_delays(
        self, equip_id: str, pose: str,
    ) -> List[int]:
        """Per-frame delays (ms) for an equip's ItemEff overlay in the
        given pose. Returns ``[]`` when no effect is authored or the
        pose subtree falls back to ``default``-only with no delays.
        Frames missing an explicit ``delay`` default to 100 ms — same
        convention as :meth:`pose_frame_delays`. The renderer's
        timeline builder uses this to play the effect at its native
        rate alongside a body that may cycle slower."""
        if self.effects is None:
            return []
        try:
            eff_node = self.effects.find(equip_id)
        except Exception:
            return []
        if not isinstance(eff_node, WzSubProperty):
            return []
        pose_tree = eff_node.get(pose)
        pose_tree = _resolve_uol(pose_tree) if isinstance(pose_tree, WzUolProperty) else pose_tree
        if not isinstance(pose_tree, WzSubProperty):
            pose_tree = eff_node.get("default")
            pose_tree = _resolve_uol(pose_tree) if isinstance(pose_tree, WzUolProperty) else pose_tree
        if not isinstance(pose_tree, WzSubProperty):
            return []
        frames = sorted(
            (c for c in pose_tree.children() if c.name.isdigit()),
            key=lambda c: int(c.name),
        )
        out: List[int] = []
        for fr in frames:
            # Frame entries are usually canvases (which support .child)
            # but pose-tree entries can be UOLs into ``default``; either
            # way, try to read the linked frame's ``delay``.
            target = _resolve_uol(fr) if isinstance(fr, WzUolProperty) else fr
            d = target.child("delay") if hasattr(target, "child") else None
            d = _resolve_uol(d) if isinstance(d, WzUolProperty) else d
            v = getattr(d, "value", None) if d is not None else None
            try:
                ms = int(v) if v is not None else 0
            except (TypeError, ValueError):
                ms = 0
            out.append(ms if ms > 0 else 100)
        return out

    def _cap_hair_filter(
        self, equip_ids: List[str],
    ) -> Tuple[bool, frozenset, frozenset]:
        """Find the equipped Cap's ``info/vslot`` and derive the
        per-canvas filters compose needs.

        Returns ``(hide_hair_full, hide_hair_set, vslot_tokens)``:
        - ``hide_hair_full``: the cap covers every Hair canvas (full
          helmet), so ``compose`` should skip the Hair part entirely.
        - ``hide_hair_set``: the specific Hair canvases the cap covers
          when not a full helmet.
        - ``vslot_tokens``: every 2-character token in the cap's
          vslot string (``Cp``, ``H1``, ``Ay``, ``Af``, …). The
          accessory tokens (``Af``/``Ay``/``Ae``/…) drive the
          "cap covers face accessories" arm of the z-sort remap.
        """
        cap_id = next(
            (e for e in equip_ids if category_for_id(e) == "Cap"), None,
        )
        if cap_id is None:
            return (False, frozenset(), frozenset())
        img = self._open_part(cap_id)
        if img is None:
            return (False, frozenset(), frozenset())
        info = img.parse().get("info")
        vslot: Optional[str] = None
        if isinstance(info, WzSubProperty):
            v = info.get("vslot")
            if isinstance(v, WzStringProperty):
                vslot = v.value
        tokens = _parse_vslot_tokens(vslot)
        hidden = _cap_hidden_hair_canvases(vslot)
        if hidden is None:
            return (True, frozenset(), tokens)
        return (False, hidden, tokens)

    def get_ear_types(self, equip_id: str) -> List[str]:
        """Return the ear-canvas names a Head image ships with.

        Walks ``Head/<id>.img/front/`` and returns every child canvas
        name except ``head`` (which is the face/scalp). Stock GMS v83
        Heads ship a single ear, but custom Heads can carry multiple
        (e.g. ``humanEar``, ``lefEar``, ``highlefEar``) and the client
        picks one to render via ``compose(..., ear_type=...)``."""
        if category_for_id(equip_id) != "Head":
            return []
        img = self._open_part(equip_id)
        if img is None:
            return []
        front = img.parse().get("front")
        if not isinstance(front, WzSubProperty):
            return []
        out: List[str] = []
        for child in front.children():
            if child.name == "head":
                continue
            target = _resolve_uol(child) if isinstance(child, WzUolProperty) else child
            if isinstance(target, WzCanvasProperty):
                out.append(child.name)
        return out

    def detect_pose(self, equip_ids: List[str], requested: Optional[str] = None) -> str:
        """Pick a pose for a composite. Honors ``requested`` if the
        equipped weapon supports it; otherwise falls back to the
        weapon's first available pose, then to ``stand1``.

        Back-facing poses (ladder / rope) are an exception: even when
        the weapon doesn't ship those animations, the request is
        honored — the renderer simply omits the weapon canvas for
        that pose. Climbing animations are body / hair / cap art and
        the user expects them to remain selectable regardless of what
        weapon happens to be equipped."""
        weapon_id = next(
            (e for e in equip_ids if category_for_id(e) == "Weapon"),
            None,
        )
        if weapon_id is None:
            return requested if requested in SUPPORTED_POSES else DEFAULT_POSE
        if requested in _WEAPON_OPTIONAL_POSES:
            return requested
        poses = self.get_weapon_poses(weapon_id) or [DEFAULT_POSE]
        if requested in poses:
            return requested
        return poses[0]

    def compose(
        self, equip_ids: List[str], pose: Optional[str] = None,
        ear_type: str = DEFAULT_EAR_TYPE,
        flip: bool = False,
        frame: int = 0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        hair_hsv: Optional[Tuple[float, float, float]] = None,
    ) -> Image.Image:
        """Render the equipped parts as a single RGBA :class:`PIL.Image`.

        ``pose`` is one of ``"stand1"`` / ``"stand2"`` and drives which
        body / coat / weapon canvases get pulled. If ``None`` (or an
        unsupported value) we auto-detect from the equipped weapon.

        ``ear_type`` is the canvas name under ``Head/<id>.img/front/``
        to render alongside ``head`` (e.g. ``humanEar``, ``lefEar``,
        ``highlefEar``). If the Head image doesn't ship a matching
        canvas the ear simply doesn't render — call
        :meth:`get_ear_types` first to enumerate what's available.

        ``flip=True`` mirrors the final composite horizontally, which
        is how MapleStory renders a right-facing character — the
        bitmaps are authored facing left and flipped at draw time.

        ``bbox`` overrides the auto-computed content bounding box
        ``(min_x, min_y, max_x, max_y)`` in world coordinates. Used
        by :meth:`compose_animation` to render every frame at the
        same image size with the navel at the same image-space
        pixel — without this, hair / cap stay put while the body
        bitmap leans left and right between frames (the body's per-
        frame canvas position differs by a few pixels) so the body
        appears to wobble in the cycling preview."""
        pose = self.detect_pose(equip_ids, pose)
        hide_hair_full, hide_hair_set, cap_vslot_tokens = \
            self._cap_hair_filter(equip_ids)
        placements = self._build_placements(
            equip_ids, pose, ear_type,
            hide_hair_full, hide_hair_set, cap_vslot_tokens, frame,
        )
        if not placements:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        return self._render_placements(placements, flip=flip, bbox=bbox, hair_hsv=hair_hsv)

    def compose_animation_timeline(
        self, equip_ids: List[str], pose: Optional[str] = None,
        ear_type: str = DEFAULT_EAR_TYPE,
        flip: bool = False,
        max_total_ms: int = 6000,
        hair_hsv: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[List[Image.Image], List[int], List[Tuple[int, Dict[str, int]]]]:
        """Render a full animation cycle that respects each ItemEff
        overlay's authored frame rate independently of the body's.

        Returns ``(images, delays_ms, steps)`` where ``images[i]`` is
        the composite at the i-th playback step, ``delays_ms[i]`` is
        how long that step is on screen, and ``steps[i]`` is the
        ``(body_frame, {equip_id: effect_frame})`` snapshot used to
        render it (mostly useful for diagnostics / debugging the
        timeline).

        Algorithm:

          1. Compute the body's ordered playback sequence — for stand
             poses we emit ``[0, 1, 2, 1]`` so the breathing loop's
             return path passes through the neutral frame instead of
             hard-cutting from peak inhale back to neutral.
          2. Compute every effect's authored cycle (per-frame
             durations from ``ItemEff/.../delay``, default 100 ms).
          3. Build a timeline that places each transition (body
             change, effect change) on a shared time axis from 0 up
             to LCM(body_cycle, *effect_cycles) — capped at
             ``max_total_ms`` so a body cycle of 720 ms paired with a
             nearly-coprime effect cycle doesn't generate hundreds of
             frames.
          4. For each interval between consecutive transitions, pick
             the body frame and per-equip effect frame active at that
             moment, render the composite, and record the duration.
        """
        from math import gcd
        pose = self.detect_pose(equip_ids, pose)

        # Body cycle ordering. ``stand1`` / ``stand2`` use the
        # 0→1→2→1 breathing return so the loop doesn't snap; other
        # poses just play in order.
        body_id = next(
            (e for e in equip_ids if category_for_id(e) == "Body"),
            "00002000",
        )
        body_delays = self.pose_frame_delays(pose, body_id) or [100]
        n_body = len(body_delays)
        if pose in ("stand1", "stand2") and n_body == 3:
            body_seq = [0, 1, 2, 1]
        else:
            body_seq = list(range(n_body))
        body_cycle_ms = sum(body_delays[s] for s in body_seq)

        # Per-equip effect cycle.
        effect_cycles: Dict[str, List[int]] = {}
        for eid in equip_ids:
            delays = self.effect_frame_delays(eid, pose)
            if delays:
                effect_cycles[eid] = delays

        # If nothing has an effect cycle, fall back to the simpler
        # body-only path so we don't generate spurious extra frames.
        if not effect_cycles:
            imgs = self.compose_animation(
                equip_ids, pose=pose, ear_type=ear_type, flip=flip,
                hair_hsv=hair_hsv,
            )
            steps = [(s, {}) for s in body_seq]
            # Stretch / fold images to match body_seq (compose_animation
            # already returns one image per body frame; we map by index).
            timeline_imgs = [imgs[s] if s < len(imgs) else imgs[-1] for s in body_seq]
            timeline_delays = [body_delays[s] for s in body_seq]
            return timeline_imgs, timeline_delays, steps

        def _lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b) if a and b else max(a, b)

        total_ms = body_cycle_ms
        for delays in effect_cycles.values():
            total_ms = _lcm(total_ms, sum(delays))
        if max_total_ms and total_ms > max_total_ms:
            total_ms = max_total_ms

        # Body active-interval table: (start, end, body_frame) over
        # one body_cycle, repeated to fill total_ms.
        body_table: List[Tuple[int, int, int]] = []
        t = 0
        for s in body_seq:
            body_table.append((t, t + body_delays[s], s))
            t += body_delays[s]
        # Same for each effect — over its own cycle.
        effect_tables: Dict[str, List[Tuple[int, int, int]]] = {}
        for eid, delays in effect_cycles.items():
            tbl: List[Tuple[int, int, int]] = []
            t = 0
            for i, d in enumerate(delays):
                tbl.append((t, t + d, i))
                t += d
            effect_tables[eid] = tbl

        # Collect transition timestamps in [0, total_ms).
        ts: set = {0, total_ms}
        for entry in body_table:
            t = entry[0]
            while t < total_ms:
                ts.add(t)
                t += body_cycle_ms
        for eid, tbl in effect_tables.items():
            ec = sum(effect_cycles[eid])
            for entry in tbl:
                t = entry[0]
                while t < total_ms:
                    ts.add(t)
                    t += ec
        sorted_ts = sorted(ts)

        def _frame_at(table: List[Tuple[int, int, int]], cycle: int, when: int) -> int:
            wrapped = when % cycle if cycle else 0
            for s, e, f in table:
                if s <= wrapped < e:
                    return f
            return table[-1][2] if table else 0

        steps: List[Tuple[int, Dict[str, int]]] = []
        delays_out: List[int] = []
        per_frame_placements: List[List[_Placement]] = []
        hide_hair_full, hide_hair_set, cap_vslot_tokens = \
            self._cap_hair_filter(equip_ids)
        stabilize = pose in ("stand1", "stand2")
        frozen_anchors: Optional[Dict[str, Tuple[int, int]]] = None
        head_derived = frozenset({"brow", "neck", "handMove"})
        frame0_canvases: Dict[Tuple[str, str], Any] = {}
        body_anchor_0: Optional[Tuple[int, int]] = None

        def _body_anchor_of(pls: List[_Placement]) -> Optional[Tuple[int, int]]:
            for pl in pls:
                if pl.category == "Body" and pl.name == "body" \
                        and pl.top_left is not None:
                    return (pl.top_left[0] + pl.pixel_canvas.width,
                            pl.top_left[1])
            return None

        for i in range(len(sorted_ts) - 1):
            t = sorted_ts[i]
            duration = sorted_ts[i + 1] - t
            if duration <= 0:
                continue
            body_frame = _frame_at(body_table, body_cycle_ms, t)
            eff_overrides: Dict[str, int] = {}
            for eid, tbl in effect_tables.items():
                eff_overrides[eid] = _frame_at(
                    tbl, sum(effect_cycles[eid]), t,
                )
            placements, anchors = self._build_placements(
                equip_ids, pose, ear_type,
                hide_hair_full, hide_hair_set, cap_vslot_tokens, body_frame,
                frozen_anchors=frozen_anchors,
                return_anchors=True,
                stabilize_effects=stabilize,
                effect_frame_overrides=eff_overrides,
            )
            # Stand-pose stabilization (the breathing-anchor +
            # body-delta translation pass) applies the same way per
            # timeline step as it does per body-frame in the
            # non-timeline path.
            if stabilize:
                if frozen_anchors is None:
                    frozen_anchors = anchors
                    for pl in placements:
                        frame0_canvases[(pl.equip_id, pl.name)] = pl.pixel_canvas
                    body_anchor_0 = _body_anchor_of(placements)
                else:
                    body_anchor_now = _body_anchor_of(placements)
                    dx = dy = 0
                    if body_anchor_0 is not None and body_anchor_now is not None:
                        dx = body_anchor_0[0] - body_anchor_now[0]
                        dy = body_anchor_0[1] - body_anchor_now[1]
                    for pl in placements:
                        if pl.top_left is None:
                            continue
                        # ItemEff overlays are anchored off the
                        # frozen character anchors (navel / brow /
                        # neck / hand) — their top_left is already
                        # constant across frames, so the body-canvas
                        # delta would shift them away from the
                        # frozen anchor (e.g. effect 1103249 anchored
                        # to navel=(0,0) every frame would otherwise
                        # drift with the body's breathing).
                        if pl.category == "Effect":
                            continue
                        key = (pl.equip_id, pl.name)
                        f0_canvas = frame0_canvases.get(key)
                        anchor_name = pl.anchor_override or _determine_anchor(
                            pl.canvas, pl.category,
                        )
                        if f0_canvas is pl.pixel_canvas \
                                and anchor_name in head_derived:
                            continue
                        if anchor_name in head_derived:
                            continue
                        if dx or dy:
                            pl.top_left = (
                                pl.top_left[0] + dx,
                                pl.top_left[1] + dy,
                            )
            per_frame_placements.append(placements)
            steps.append((body_frame, eff_overrides))
            delays_out.append(duration)

        all_pls = [p for fr in per_frame_placements for p in fr if p.top_left is not None]
        if not all_pls:
            empty = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            return [empty for _ in delays_out], delays_out, steps

        def _w(p): return p.width_override if p.width_override is not None else p.pixel_canvas.width
        def _h(p): return p.height_override if p.height_override is not None else p.pixel_canvas.height
        bbox = (
            min(p.top_left[0] for p in all_pls),
            min(p.top_left[1] for p in all_pls),
            max(p.top_left[0] + _w(p) for p in all_pls),
            max(p.top_left[1] + _h(p) for p in all_pls),
        )
        images = [
            self._render_placements(pls, flip=flip, bbox=bbox, hair_hsv=hair_hsv)
            if pls else Image.new(
                "RGBA",
                (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
                (0, 0, 0, 0),
            )
            for pls in per_frame_placements
        ]
        return images, delays_out, steps

    def compose_animation(
        self, equip_ids: List[str], pose: Optional[str] = None,
        ear_type: str = DEFAULT_EAR_TYPE,
        flip: bool = False,
        frames: Optional[Tuple[int, ...]] = None,
        hair_hsv: Optional[Tuple[float, float, float]] = None,
    ) -> List[Image.Image]:
        """Compose multiple frames at consistent image dimensions.

        For ``stand1`` / ``stand2`` the body's per-frame canvas
        advertises slightly different anchor offsets — neck moves
        zig-zag ``(4,-11) → (3,-12) → (2,-11)`` for body 00002000,
        hand drifts by ~1px each frame — so equipment that anchors
        on neck (head / hair / cap / face) or hand (weapon) wobbles
        in lockstep with the breathing animation. To keep the
        animated preview from looking jittery, we render frame 0
        normally, capture its world-anchor map, and reuse those
        frozen anchors for the rest of the cycle. The body bitmap
        still gets its per-frame artwork (so the chest visibly
        breathes), only the *positions* of dependent parts stay
        locked. Then the union of every frame's bbox is taken so all
        returned images share canvas dimensions and the navel sits
        at the same image-space pixel.

        For action poses (``walk1``, ``swingO1``, ``jump``, …) the
        body's per-frame anchor changes ARE the motion (the leg
        swings forward, the head bobs with the stride) and freezing
        them would defeat the animation. Those poses skip the
        breathing stabilization and let every part follow the body's
        per-frame anchors naturally, so cap / hair / weapon track
        the head and hand the way the in-game client renders them.

        ``frames`` defaults to the actual frame count of the equipped
        body's pose subtree; pass an explicit tuple to render only a
        slice (e.g. ``(0,)`` for a single still).
        """
        pose = self.detect_pose(equip_ids, pose)
        hide_hair_full, hide_hair_set, cap_vslot_tokens = \
            self._cap_hair_filter(equip_ids)

        if frames is None:
            body_id = next(
                (e for e in equip_ids if category_for_id(e) == "Body"),
                "00002000",
            )
            n = len(self.pose_frame_delays(pose, body_id))
            frames = tuple(range(n)) if n > 0 else (0,)

        # Breathing stabilization is meaningful only for the
        # in-place rest poses. Other poses are real motion and need
        # natural per-frame anchors.
        stabilize = pose in ("stand1", "stand2")

        per_frame: List[List[_Placement]] = []
        frozen_anchors: Optional[Dict[str, Tuple[int, int]]] = None
        # Frame-0 placements drive two things across the rest of the
        # cycle:
        #
        # 1. ``frozen_anchors`` — head / hair / cap / face anchor on
        #    the values frame 0 registered, so they stay still.
        #
        # 2. Per-canvas TRANSLATION COMPENSATION. The body's per-frame
        #    bitmap is a different size each frame (21 → 22 → 23 px
        #    wide and 1px taller for stock body 00002000) and the
        #    navel sits at a different point within each bitmap. We
        #    pin the body bitmap by its RIGHT edge to frame 0's right
        #    edge so the chest expansion always grows leftward —
        #    that's the direction MapleStory's standby breathing
        #    actually goes (the character faces left so its chest /
        #    front IS the bitmap's left side), and a left-only
        #    expansion reads as "breathing in place" instead of
        #    "shifting right". Coat / longcoat / pants / glove ship
        #    per-frame bitmaps tuned to track the body, so we apply
        #    ``-delta`` to every placement whose pixel canvas isn't
        #    shared with frame 0. Hair / cap / face / earring etc.
        #    UOL into ``default`` and share pixel canvases with frame
        #    0, so they stay on their frozen anchor positions.
        frame0_canvases: Dict[Tuple[str, str], Any] = {}
        frame0_top_lefts: Dict[Tuple[str, str], Tuple[int, int]] = {}
        body_anchor_0: Optional[Tuple[int, int]] = None

        # Anchors that hang off the head — placements that resolve
        # to one of these should stay locked to frame 0's position
        # regardless of whether their pixel canvas changes per
        # frame, otherwise per-frame cap art (e.g., 01000090,
        # 01000108, 01000127) makes the cap drift relative to the
        # static head.
        head_derived = frozenset({"brow", "neck", "handMove"})

        def _body_anchor(pls: List[_Placement]) -> Optional[Tuple[int, int]]:
            """Use the body bitmap's RIGHT edge + TOP edge as the
            stable reference. Right edge stays put across frames
            (chest puffs leftward); vertical alignment stays at
            the bitmap top so the head / shoulders don't drift."""
            for pl in pls:
                if pl.category == "Body" and pl.name == "body" \
                        and pl.top_left is not None:
                    return (
                        pl.top_left[0] + pl.pixel_canvas.width,
                        pl.top_left[1],
                    )
            return None

        for f in frames:
            placements, anchors = self._build_placements(
                equip_ids, pose, ear_type,
                hide_hair_full, hide_hair_set, cap_vslot_tokens, f,
                frozen_anchors=frozen_anchors,
                return_anchors=True,
                stabilize_effects=stabilize,
            )
            if not stabilize:
                # Action poses: leave every frame's anchors alone so
                # the natural per-frame motion (stride, swing, head
                # bob) is preserved. We still pad to the union bbox
                # below so the cycling preview canvas size is stable.
                per_frame.append(placements)
                continue
            if frozen_anchors is None:
                frozen_anchors = anchors
                for pl in placements:
                    frame0_canvases[(pl.equip_id, pl.name)] = pl.pixel_canvas
                    if pl.top_left is not None:
                        frame0_top_lefts[(pl.equip_id, pl.name)] = pl.top_left
                body_anchor_0 = _body_anchor(placements)
            else:
                body_anchor_now = _body_anchor(placements)
                dx = dy = 0
                if body_anchor_0 is not None and body_anchor_now is not None:
                    dx = body_anchor_0[0] - body_anchor_now[0]
                    dy = body_anchor_0[1] - body_anchor_now[1]
                for pl in placements:
                    if pl.top_left is None:
                        continue
                    # ItemEff overlays are anchored off frozen
                    # character anchors (navel / brow / etc.) and
                    # their top_left is already constant across
                    # frames; the body-canvas delta would shift them
                    # off the frozen anchor (e.g. effect 1103249
                    # anchored to navel=(0,0) drifts with the body's
                    # breathing if we apply dx/dy here).
                    if pl.category == "Effect":
                        continue
                    key = (pl.equip_id, pl.name)
                    f0_canvas = frame0_canvases.get(key)
                    anchor_name = pl.anchor_override or _determine_anchor(
                        pl.canvas, pl.category,
                    )
                    # Same pixel canvas as frame 0 AND head-derived
                    # anchor — UOL'd static piece glued to the frozen
                    # head; leave it. Body-anchored placements with
                    # UOL'd canvases (e.g., shoes 01070000 sharing
                    # one bitmap across all 3 frames) still need the
                    # body-delta compensation below to track the
                    # animated body.
                    if f0_canvas is pl.pixel_canvas \
                            and anchor_name in head_derived:
                        continue
                    if anchor_name in head_derived:
                        # Head-attached placement (cap / face / hair
                        # variants whose canvases differ per frame).
                        # Leave the natural anchor placement alone:
                        # ``top_left = brow_world - origin`` already
                        # keeps the brow attached to the (frozen)
                        # head's brow. Any apparent "wobble" is the
                        # artist's encoded breathing — silhouette
                        # puffing around a fixed attachment point.
                        # Don't apply the body-delta translation
                        # because the cap doesn't track the body.
                        continue
                    # Body-attached per-frame placement — apply the
                    # body's right-edge translation compensation so
                    # it tracks the (now-stable) body.
                    if dx or dy:
                        pl.top_left = (
                            pl.top_left[0] + dx,
                            pl.top_left[1] + dy,
                        )
            per_frame.append(placements)

        all_pls = [p for fr in per_frame for p in fr if p.top_left is not None]
        if not all_pls:
            return [Image.new("RGBA", (1, 1), (0, 0, 0, 0)) for _ in frames]
        def _w(p): return p.width_override if p.width_override is not None else p.pixel_canvas.width
        def _h(p): return p.height_override if p.height_override is not None else p.pixel_canvas.height
        bbox = (
            min(p.top_left[0] for p in all_pls),
            min(p.top_left[1] for p in all_pls),
            max(p.top_left[0] + _w(p) for p in all_pls),
            max(p.top_left[1] + _h(p) for p in all_pls),
        )
        return [
            self._render_placements(pls, flip=flip, bbox=bbox, hair_hsv=hair_hsv)
            if pls else Image.new(
                "RGBA",
                (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
                (0, 0, 0, 0),
            )
            for pls in per_frame
        ]

    # Map ``ItemEff/effect/<pose>/pos`` (integer or string-int) to
    # the world_anchors key the effect canvas anchors against. pos=1
    # is far and away the most common (head-anchored cap effects);
    # pos=0 is body-center; pos=-1 falls through to body-center too.
    # pos=2 / pos=3 are rare and don't have a documented meaning here
    # — fall back to navel until we find counter-examples.
    _EFFECT_POS_ANCHOR: Dict[int, str] = {
        -1: "navel",
        0: "navel",
        1: "brow",
        2: "navel",
        3: "navel",
    }

    def _build_effect_placements(
        self,
        equip_ids: List[str],
        pose: str,
        frame: int,
        world_anchors: Dict[str, Tuple[int, int]],
        pin_origin_to_frame_0: bool = False,
        effect_frame_overrides: Optional[Dict[str, int]] = None,
    ) -> List[_Placement]:
        """Build :class:`_Placement` entries for every equipped item
        whose ID has an authored overlay under ``ItemEff.img``.

        Each ``effect`` subtree ships per-pose frame trees plus a
        ``default`` (used when the requested pose isn't authored).
        Most pose subtrees UOL into ``default`` so we just resolve
        the chain. The frame index is taken modulo the effect's own
        frame count — the effect's tempo isn't synced to the body's,
        so this is a best-effort cycling; effects authored to match
        the body's frame count will line up exactly.

        ``pin_origin_to_frame_0`` is set in stabilized stand poses so
        the placement uses frame 0's origin for ALL frames while
        still cycling the per-frame pixel canvas. The animation
        plays in place instead of sliding 1-2 px around the cap each
        frame (the natural per-frame origin variation was authored
        to compose with per-frame body motion that stand-pose
        stabilization deliberately suppresses).
        """
        out: List[_Placement] = []
        if self.effects is None:
            return out
        for eid in equip_ids:
            try:
                eff_node = self.effects.find(eid)
            except Exception:
                continue
            if not isinstance(eff_node, WzSubProperty):
                continue
            # Pick the pose subtree, falling back to ``default``. For
            # back-facing poses (ladder / rope) prefer ``backDefault``
            # when the effect ships one — it's the canonical
            # back-view variant (e.g. effect 1103248 UOLs both
            # ``ladder`` and ``rope`` straight to ``backDefault``).
            # Many effects don't author a back variant, so the
            # default fallback still applies.
            back_pose = pose in _WEAPON_OPTIONAL_POSES
            pose_tree = eff_node.get(pose)
            pose_tree = _resolve_uol(pose_tree) if isinstance(pose_tree, WzUolProperty) else pose_tree
            if not isinstance(pose_tree, WzSubProperty) and back_pose:
                pose_tree = eff_node.get("backDefault")
                pose_tree = _resolve_uol(pose_tree) if isinstance(pose_tree, WzUolProperty) else pose_tree
            if not isinstance(pose_tree, WzSubProperty):
                pose_tree = eff_node.get("default")
                pose_tree = _resolve_uol(pose_tree) if isinstance(pose_tree, WzUolProperty) else pose_tree
            if not isinstance(pose_tree, WzSubProperty):
                continue
            # Collect numeric frame children, ordered by index.
            frame_children = sorted(
                (c for c in pose_tree.children() if c.name.isdigit()),
                key=lambda c: int(c.name),
            )
            if not frame_children:
                continue
            # ``effect_frame_overrides`` lets the timeline builder pick
            # the per-equip effect frame independent of the body's
            # frame index — so an effect with a 100 ms cycle plays at
            # its own rate against a body cycling at 500 ms.
            if effect_frame_overrides is not None and eid in effect_frame_overrides:
                idx = effect_frame_overrides[eid] % len(frame_children)
            else:
                idx = frame % len(frame_children)
            target = frame_children[idx]
            # Pose-tree entries are usually UOLs into ``default``.
            if isinstance(target, WzUolProperty):
                target = _resolve_uol(target)
            if not isinstance(target, WzCanvasProperty):
                continue
            # Resolve the placeholder's _outlink to the real pixel canvas.
            try:
                pixel_canvas = resolve_canvas_link(target, self.effects.root)
            except Exception:
                pixel_canvas = target
            if pixel_canvas is None or not pixel_canvas.has_pixels():
                continue
            # Origin source: by default, the per-frame canvas's own
            # origin. In stabilized mode, frame 0's origin so the
            # canvas world position stays put across the cycle.
            origin_source = target
            if pin_origin_to_frame_0 and idx != 0:
                f0 = frame_children[0]
                if isinstance(f0, WzUolProperty):
                    f0 = _resolve_uol(f0)
                if isinstance(f0, WzCanvasProperty):
                    origin_source = f0
            # Pose-tree pos picks the world anchor:
            #   * pos=1            → ``brow`` (head-attached effect)
            #   * pos=0 or no pos  → ``navel`` plus a fixed (+8, +21)
            #     offset (right 8, down 21 px) so effects land at the
            #     same lower-body point the MapleStory client uses as
            #     a default attachment.
            # When the pose subtree (e.g. ``ladder``) doesn't author
            # its own ``pos``, cascade to ``effect/default`` so the
            # artist's default attachment carries through to the
            # back-pose subtrees that share the same canvas (e.g.
            # 1103187 sets pos=1 only on default but reuses the
            # canvas across default / ladder / rope).
            def _pos_of(tree: Any) -> Optional[int]:
                node = tree.get("pos") if isinstance(tree, WzSubProperty) else None
                node = _resolve_uol(node) if isinstance(node, WzUolProperty) else node
                if node is None:
                    return None
                try:
                    return int(getattr(node, "value", None))
                except (TypeError, ValueError):
                    return None
            pos_val = _pos_of(pose_tree)
            if pos_val is None and pose_tree.name != "default":
                default_tree = eff_node.get("default")
                default_tree = _resolve_uol(default_tree) \
                    if isinstance(default_tree, WzUolProperty) else default_tree
                if isinstance(default_tree, WzSubProperty):
                    pos_val = _pos_of(default_tree)
            anchor_name = self._EFFECT_POS_ANCHOR.get(pos_val, "navel")
            anchor_world = world_anchors.get(anchor_name) or world_anchors.get("navel") or (0, 0)
            if pos_val != 1:
                anchor_world = (anchor_world[0] + 8, anchor_world[1] + 21)
            # Stabilized mode: build a union-bbox composite that
            # places frame N's pixels at the same WORLD coordinates
            # the artist authored (origin_n inside the canvas). Every
            # cycled frame ends up with the SAME placement bbox so
            # the user sees animation play in place — no per-frame
            # canvas-size variation visible as drift.
            decoded_override = None
            width_override = height_override = None
            if pin_origin_to_frame_0 and len(frame_children) > 1:
                composite = self._build_stabilized_effect_composite(
                    frame_children, idx,
                )
                if composite is not None:
                    layer_image, comp_origin, comp_w, comp_h = composite
                    decoded_override = layer_image
                    width_override = comp_w
                    height_override = comp_h
                    top_left = (
                        anchor_world[0] - comp_origin[0],
                        anchor_world[1] - comp_origin[1],
                    )
                    origin = comp_origin
                else:
                    origin = _origin(origin_source)
                    top_left = (anchor_world[0] - origin[0], anchor_world[1] - origin[1])
            else:
                origin = _origin(origin_source)
                top_left = (anchor_world[0] - origin[0], anchor_world[1] - origin[1])
            # Z handling. The canonical authored layer offset lives at
            # ``effect/default/z`` (often mirrored at the top-level
            # ``effect/z`` and on each per-pose subtree). Per-frame
            # canvas ``z`` values are usually ``0`` and not meaningful
            # as a layer offset. Resolution priority:
            #   1. pose_tree.z         — per-pose override when authored
            #   2. effect/default/z    — canonical base
            #   3. effect/z            — top-level fallback
            #   4. per-frame canvas z  — last-resort
            def _try_int(node: Optional[WzProperty]) -> Optional[int]:
                node = _resolve_uol(node) if isinstance(node, WzUolProperty) else node
                if node is None:
                    return None
                try:
                    return int(getattr(node, "value", None))
                except (TypeError, ValueError):
                    return None
            z_int: Optional[int] = _try_int(pose_tree.get("z"))
            if z_int is None:
                default_tree = eff_node.get("default")
                default_tree = _resolve_uol(default_tree) \
                    if isinstance(default_tree, WzUolProperty) else default_tree
                if isinstance(default_tree, WzSubProperty):
                    z_int = _try_int(default_tree.get("z"))
            if z_int is None:
                z_int = _try_int(eff_node.get("z"))
            if z_int is None:
                z_int = _try_int(target.child("z"))

            out.append(_Placement(
                equip_id=eid, category="Effect",
                name=f"effect/{pose_tree.name}/{target.name}",
                canvas=target, pixel_canvas=pixel_canvas,
                origin=origin, map_anchors={},
                z_slot=None, top_left=top_left, extra_z=z_int,
                anchor_override=anchor_name,
                decoded_override=decoded_override,
                width_override=width_override,
                height_override=height_override,
            ))
        return out

    def _build_stabilized_effect_composite(
        self, frame_children: List[Any], idx: int,
    ) -> Optional[Tuple[Any, Tuple[int, int], int, int]]:
        """Composite frame ``idx``'s pixels into a uniform-size RGBA
        Image whose bbox covers EVERY frame in the cycle. Returns
        ``(image, composite_origin, width, height)`` or ``None`` if
        anything fails — the caller falls back to natural rendering.

        Each frame in the cycle ships its own canvas size + origin;
        the artist's intent is that the world position of the brow
        anchor (or whatever anchor the pose uses) lands at the same
        world point for all frames, with each frame's canvas
        cropped to its content bbox. To eliminate the per-frame bbox
        variation visible as drift in stabilized stand poses, we:

          1. Compute every frame's brow-relative bbox.
          2. Take the union — that's the composite size.
          3. Place frame ``idx``'s pixels at the offset in the
             composite that puts its anchor at the same composite
             pixel as every other frame.

        The returned ``composite_origin`` is the canvas-relative
        anchor pixel; ``top_left = anchor_world - composite_origin``
        positions the composite identically across every frame in
        the cycle.
        """
        from PIL import Image as _PILImage
        # Gather (origin, width, height) for every frame, resolving
        # UOLs the same way the main loop does.
        frame_meta: List[Tuple[Any, Tuple[int, int], int, int]] = []
        for fc in frame_children:
            tgt = _resolve_uol(fc) if isinstance(fc, WzUolProperty) else fc
            if not isinstance(tgt, WzCanvasProperty):
                return None
            try:
                pix = resolve_canvas_link(tgt, self.effects.root)
            except Exception:
                return None
            if pix is None or not pix.has_pixels():
                return None
            frame_meta.append((tgt, _origin(tgt), pix.width, pix.height))
        # Union bbox in anchor-relative coords. For each frame the
        # canvas occupies [-origin.x, -origin.x + width) horizontally
        # and [-origin.y, -origin.y + height) vertically (relative
        # to the anchor at world (0, 0)).
        min_x = min(-o[0] for _, o, _, _ in frame_meta)
        min_y = min(-o[1] for _, o, _, _ in frame_meta)
        max_x = max(-o[0] + w for _, o, w, _ in frame_meta)
        max_y = max(-o[1] + h for _, o, _, h in frame_meta)
        comp_w = max_x - min_x
        comp_h = max_y - min_y
        if comp_w <= 0 or comp_h <= 0:
            return None
        # Composite "origin" so that anchor_world - comp_origin
        # = world coords of composite's (0, 0). The composite's
        # anchor lands at composite pixel (-min_x, -min_y).
        comp_origin = (-min_x, -min_y)
        # Decode just the requested frame and paste at its
        # anchor-aligned offset within the composite.
        target_canvas, target_origin, _, _ = frame_meta[idx]
        try:
            target_pix = resolve_canvas_link(target_canvas, self.effects.root)
            layer = decode_canvas(target_pix, region=self.region)
        except Exception:
            return None
        if layer.mode != "RGBA":
            layer = layer.convert("RGBA")
        composite = _PILImage.new("RGBA", (comp_w, comp_h), (0, 0, 0, 0))
        # Frame N's top-left in composite coords:
        # world(top_left) = anchor + (-target_origin)
        # composite(top_left) = world(top_left) - composite(world top-left)
        #                     = (-target_origin) - (min_x, min_y)
        ox = -target_origin[0] - min_x
        oy = -target_origin[1] - min_y
        composite.alpha_composite(layer, dest=(ox, oy))
        return composite, comp_origin, comp_w, comp_h

    def _build_placements(
        self,
        equip_ids: List[str], pose: str, ear_type: str,
        hide_hair_full: bool, hide_hair_set: frozenset,
        cap_vslot_tokens: frozenset, frame: int,
        frozen_anchors: Optional[Dict[str, Tuple[int, int]]] = None,
        return_anchors: bool = False,
        body_frame: Optional[int] = None,
        stabilize_effects: bool = False,
        effect_frame_overrides: Optional[Dict[str, int]] = None,
    ):
        """Collect, anchor, and z-sort placements for a single frame.

        Shared between :meth:`compose` (one frame) and
        :meth:`compose_animation` (multiple frames at consistent
        bbox). When ``frozen_anchors`` is provided, the body /
        sub-body canvases position themselves normally (their own
        per-frame navel still goes to world (0, 0)) but DO NOT
        register their other map anchors (neck / hand / lHand etc.).
        Instead, the supplied ``frozen_anchors`` are used for every
        downstream placement — keeps head / hair / cap / face /
        weapon at fixed image positions across frames so only the
        body bitmap actually breathes on screen.

        ``body_frame`` overrides the frame index used for the Body
        category only — :meth:`compose_animation` pins it to 0 so
        the body bitmap stays fixed across the cycling preview while
        other equipment (coat sleeves, capes, weapons with their own
        per-frame data) still animate. Without that, the body's
        per-frame bitmap shifts the torso left/up/down each frame —
        breathing motion that's faithful to the WZ data but reads
        as a wobble in the small preview.

        Returns the placements list, or ``(placements, world_anchors)``
        when ``return_anchors=True``.
        """
        cap_covers_hair = hide_hair_full or bool(hide_hair_set)
        placements: List[_Placement] = []
        for eid in equip_ids:
            cat = category_for_id(eid)
            if cat is None:
                continue
            if cat == "Hair" and hide_hair_full:
                continue
            img = self._open_part(eid)
            if img is None:
                continue
            f_for_part = body_frame if (body_frame is not None and cat == "Body") else frame
            for leaf_name, canvas, pixel_canvas in _collect_part_canvases(
                img, cat, eid, pose, pkg_root=self.wz.root,
                ear_type=ear_type, hide_hair=hide_hair_set, frame=f_for_part,
            ):
                placements.append(_Placement(
                    equip_id=eid, category=cat, name=leaf_name,
                    canvas=canvas, pixel_canvas=pixel_canvas,
                    origin=_origin(canvas), map_anchors=_map_anchors(canvas),
                    z_slot=_z_slot(canvas),
                ))

        if not placements:
            return (placements, {}) if return_anchors else placements

        # Pre-compute cape presence for the ``hairBelowBody`` remap in
        # ``z_for``: long-hair canvases that drape past the body land
        # at zmap 40 (right above ``capeBelowBody`` at 21) by default,
        # which puts the back hair IN FRONT of a draping cape and
        # hides most of the cape under the hair. When a cape is also
        # in the equip list, demote those long-hair canvases below
        # the cape so the cape covers them.
        has_cape = any(p.category == "Cape" for p in placements)

        def order_key(pl: _Placement) -> Tuple[int, int]:
            if pl.category == "Body":
                return (0, 0 if pl.name == "body" else 1)
            if pl.category == "Head":
                return (1, 0 if pl.name == "head" else 1)
            return (2, 0)
        placements.sort(key=order_key)

        # When frozen_anchors is set, downstream parts read from those
        # values and the body's anchor contributions are suppressed
        # (its bitmap still positions itself via its own navel
        # offset → world(0,0), but other anchor wobbles don't leak
        # through). Without it, behaviour matches the original
        # single-frame compose path.
        world_anchors: Dict[str, Tuple[int, int]] = (
            dict(frozen_anchors) if frozen_anchors is not None else {}
        )
        body_anchored = False

        # In stabilized animation (``frozen_anchors`` set) the supplied
        # anchor map keeps head-family anchors pinned to frame 0, but the
        # hand-family anchors (:data:`_DYNAMIC_ANIM_ANCHORS`) must follow
        # the body's per-frame arm so held items stay glued to the moving
        # grip. We re-register those from the body's own canvases as we
        # position them, first-body-part-wins to mirror the unfrozen path
        # (``_register_anchors(overwrite=False)`` lets the arm define
        # ``hand`` before any later canvas can).
        dynamic_seen: set = set()

        def _refresh_dynamic_anchors(pl: _Placement) -> None:
            if frozen_anchors is None or pl.category != "Body" \
                    or pl.top_left is None:
                return
            ox, oy = pl.origin
            tx, ty = pl.top_left
            for name, vec in pl.map_anchors.items():
                if name in _DYNAMIC_ANIM_ANCHORS and name not in dynamic_seen:
                    world_anchors[name] = (tx + ox + vec[0], ty + oy + vec[1])
                    dynamic_seen.add(name)

        body_navel = (0, 0)
        for pl in placements:
            if pl.category == "Body" and pl.name == "body" and not body_anchored:
                navel = pl.map_anchors.get("navel", (0, 0))
                pl.top_left = (-pl.origin[0] - navel[0],
                               -pl.origin[1] - navel[1])
                body_navel = navel
                if frozen_anchors is None:
                    self._register_anchors(pl, world_anchors, overwrite=True)
                else:
                    _refresh_dynamic_anchors(pl)
                body_anchored = True
                continue

            anchor_name = _determine_anchor(pl.canvas, pl.category)
            map_pt = pl.map_anchors.get(anchor_name, (0, 0))
            anchor_world = world_anchors.get(anchor_name)
            if anchor_world is not None:
                pl.top_left = (
                    anchor_world[0] - pl.origin[0] - map_pt[0],
                    anchor_world[1] - pl.origin[1] - map_pt[1],
                )
            elif anchor_name == "handMove":
                # `handMove` is the off-hand's attach point in weapon / aim
                # stances (body `lHand` in `alert` / `heal`; z=`handBelowWeapon`).
                # No earlier part registers a `handMove` world anchor, so the
                # first such part is the *definer*: per the v83 client
                # (HeavenClient `BodyDrawInfo`/`Body` — `hand_position` is the
                # body lHand's own `handMove`, so its draw shift is 0) it pins to
                # the body's navel pivot, NOT the arm's `hand` grip. That lands it
                # on the body's far side; attaching to `hand` would cluster it on
                # the near hand. ``top_left = -origin - body_navel``. It then
                # registers `handMove` below, so an equipped glove sharing this
                # slot chains onto the same off-hand grip.
                pl.top_left = (-pl.origin[0] - body_navel[0],
                               -pl.origin[1] - body_navel[1])
            else:
                pl.top_left = (-pl.origin[0], -pl.origin[1])
            if frozen_anchors is None:
                self._register_anchors(pl, world_anchors, overwrite=False)
            else:
                _refresh_dynamic_anchors(pl)

        # ItemEff overlays: composite per-equip effects (e.g. cap
        # 01004759 ember-flames) on top of the character. Anchored
        # off the world_anchors map populated above; world_anchors
        # is final at this point because all character placements
        # have already registered their map points.
        #
        # ``frozen_anchors`` is only set in stand1 / stand2's
        # post-frame-0 calls, where head/hair/cap/face are pinned
        # to frame 0 to keep them from wobbling against the body's
        # breathing. Effect canvases cycle through per-frame
        # bitmaps (ember-flicker, ribbon-sway) AND ship per-frame
        # origin shifts of 1-2 px that compose NATURALLY with
        # per-frame anchors but DRIFT visibly when the anchor is
        # frozen — the user sees the canvas slide around the
        # static cap each frame. In stabilized mode we still cycle
        # the per-frame bitmaps (so the animation plays) but pin
        # the placement origin to frame 0 so the canvas's world
        # position stays glued to the cap.
        if self.effects is not None:
            placements.extend(self._build_effect_placements(
                equip_ids, pose, frame, world_anchors,
                pin_origin_to_frame_0=stabilize_effects,
                effect_frame_overrides=effect_frame_overrides,
            ))

        zmap_size = len(self._zmap)

        # Caps frequently ship a front canvas (z=``cap`` /
        # ``capOverHair`` / ``capAccessory``) plus a ``z=cap*Below*``
        # canvas. The default zmap promotes the cap*Below* slots
        # ABOVE hair/head because for some caps (e.g. 01001036
        # mushroom) the main visible hat lives on ``capBelowBody``
        # while the front canvas is just a thin trim. But for caps
        # where the cap*Below* is a back-accessory (a piece tucked
        # behind the head/body), the promoted placement renders it
        # on top of the face — e.g. 01006186's ``defaultAc`` red
        # back-of-head shape, or 01000111's ``default2`` dome that
        # belongs behind the body silhouette. Demote those back-
        # accessory placements to ``_z_index("body") - 1`` so the
        # body, head, and face all cover them.
        #
        # Slot-name based split:
        #   * ``capAccessoryBelowBody`` — the slot name says
        #     "accessory below body", canonical placement is behind
        #     body. Always demote when the cap also has a non-
        #     ``*Below*`` front canvas.
        #   * ``capBelowBody`` / ``capBelowHead`` / ``capBelowHair``
        #     — the back of the main cap; in many caps this IS the
        #     main visible hat shape. Demote only when the front
        #     canvas's bitmap area is at least the ``*Below*``
        #     canvas's area AND the ``*Below*`` doesn't extend
        #     above the front canvas's top (i.e. the front IS the
        #     main hat and the ``*Below*`` is a secondary back-
        #     piece). 01001036's mushroom and 01000114's witch-hat
        #     brim both have a larger ``*Below*`` than front and
        #     are correctly skipped by this guard.
        _FRONT_CAP_SLOTS = frozenset({"cap", "capOverHair", "capAccessory"})
        _ALWAYS_BACK_BELOW = frozenset({"capAccessoryBelowBody"})
        caps_by_id: Dict[str, List[_Placement]] = {}
        for pl in placements:
            if pl.category == "Cap":
                caps_by_id.setdefault(pl.equip_id, []).append(pl)
        cap_below_demote: set = set()
        for cap_pls in caps_by_id.values():
            front_candidates = [
                p for p in cap_pls
                if p.z_slot in _FRONT_CAP_SLOTS and p.top_left is not None
            ]
            if not front_candidates:
                continue
            front = max(
                front_candidates,
                key=lambda p: p.pixel_canvas.width * p.pixel_canvas.height,
            )
            front_area = front.pixel_canvas.width * front.pixel_canvas.height
            front_top_y = front.top_left[1]
            for p in cap_pls:
                if (p is front or p.top_left is None
                        or not p.z_slot
                        or not p.z_slot.startswith("cap")
                        or "Below" not in p.z_slot):
                    continue
                if p.z_slot in _ALWAYS_BACK_BELOW:
                    cap_below_demote.add(id(p))
                    continue
                p_area = p.pixel_canvas.width * p.pixel_canvas.height
                if front_area >= p_area and p.top_left[1] >= front_top_y:
                    cap_below_demote.add(id(p))

        def slot_z(pl: _Placement) -> int:
            """zmap index for a non-effect placement, after the
            pose/category-aware slot remaps below. Extracted so
            ``z_for`` can use it both directly and to seed
            ``parent_z`` for effect placements (whose z is
            authored RELATIVE to the equip they overlay)."""
            # Demote secondary cap*Below* canvases that the
            # back-piece detection above flagged: render them just
            # before ``body`` so the body, head, and face all cover
            # them.
            if id(pl) in cap_below_demote:
                return self._z_index("body") - 1
            slot = pl.z_slot
            if slot is None:
                return self._z_index(slot)
            # Stand1 is the one-handed rest pose: only the front arm
            # is visible and one-handed weapons should hang from the
            # hand with the arm covering the grip. Some weapons (e.g.
            # 01312058) are authored with z='weaponOverArm' — a slot
            # at the top of the zmap meant for stand2 polearms held
            # vertically — which floats the entire blade in front of
            # the body. Demote to 'weapon' so the arm sits over the
            # grip like every other one-handed weapon.
            if pose == "stand1" and pl.category == "Weapon" \
                    and slot == "weaponOverArm":
                slot = "weapon"
            # Poses where both the held weapon and the gripping
            # arm need to render IN FRONT of the head — i.e. the
            # arm + weapon cross the face area and the natural z
            # slots (``weapon`` / ``weaponBelowArm`` at indices
            # ~262/264, ``arm`` / ``armOverHair`` at ~267/268) sit
            # BEFORE ``head`` (305), so the head/face ends up
            # covering them.
            #
            # Covers: prone / proneStab (lying face-down with arm
            # extended forward); shoot1 / shoot2 / shootF (ranged
            # firing stance, weapon held out at face level);
            # swingT1 (2H side wind-up, blade can pass the face);
            # stabO1 / stabO2 (1H thrust at face level).
            #
            # Promote the weapon to ``weaponOverHand`` (351) and
            # the gripping arm to ``handOverHair`` (358), preserving
            # the natural arm-over-grip ordering (hand sits above
            # weapon). ``armOverHairBelowWeapon`` is left alone —
            # the slot's name explicitly says it should stay below
            # the weapon (used for the back arm in swingT1 frames
            # 0/1, paired with ``armOverHair`` for the front arm).
            elif pose in (
                "prone", "proneStab",
                "shoot1", "shoot2", "shootF",
                "swingT1",
                "stabO1", "stabO2",
            ):
                if pl.category == "Weapon" \
                        and slot in ("weapon", "weaponBelowArm"):
                    slot = "weaponOverHand"
                elif pl.category == "Body" and slot in (
                    "arm", "armOverHair",
                ):
                    slot = "handOverHair"
            # Two-handed swing / stab: the weapon is already
            # authored above the head (``weaponOverBody`` /
            # ``weaponOverGlove`` at 349-352), but the gripping
            # ``arm`` leaf is at ``arm`` / ``armBelowHead`` /
            # ``armBelowHeadOverMailChest`` — far below — so the
            # weapon covers the grip. Promote those plain arm
            # slots to ``handOverHair`` so both gripping hands
            # render above the blade, matching the in-game look.
            #
            # ``swingT3`` is the overhead wind-up, and frame 2's
            # recoil canvas is authored with ``armBelowHead``
            # specifically so the head covers the gripping hand as
            # it crosses the face — promoting it would render the
            # arm OVER the face. That frame already pairs with
            # ``weaponBelowBody`` so the natural layering puts
            # weapon < arm < head without any remap; we just skip
            # the promotion for ``armBelowHead`` in that pose.
            elif pose in (
                "swingT2", "stabT1", "stabT2",
            ) and pl.category == "Body" and slot in (
                "arm", "armBelowHead", "armBelowHeadOverMailChest",
            ):
                slot = "handOverHair"
            elif pose == "swingT3" and pl.category == "Body" and slot in (
                "arm", "armBelowHeadOverMailChest",
            ):
                slot = "handOverHair"
            # Stand2 = the 2H rest stance, weapon held vertically in
            # front of the chest with the shaft extending above the
            # gripping hand. Properly-authored 2H weapons use
            # ``weaponOverArm`` (zmap 129, above ``head`` at 104) so
            # the blade crosses in front of the head. Some weapons
            # (~312, mostly cash 0170* whose action 30 doubles as the
            # 1H sword mode) author ``weaponBelowArm`` (76) instead —
            # a holdover from the 1H idle where the weapon hangs at
            # the hip and never crosses the head. In stand2 that
            # leaves the blade tip stuck behind the head/hair.
            # Promote those to the correct slot.
            elif pose == "stand2" and pl.category == "Weapon" \
                    and slot == "weaponBelowArm":
                slot = "weaponOverArm"
            # Long hair (e.g. 00041940) authors the back-hanging
            # canvas with z='hairBelowBody' (zmap 40 — right before
            # ``body`` at 41 so the hair is BEHIND the body but ON
            # TOP OF every other back-cluster layer). When a cape is
            # also equipped (``capeBelowBody`` at 21), that back hair
            # ends up IN FRONT of the cape and hides most of it.
            # Demote to ``backHair`` (15) — deepest back-cluster slot
            # — so the cape drapes naturally over the hair from the
            # waist down. Front hair (``hair`` / ``hairOverHead``)
            # stays where authored; only the body-trailing back hair
            # moves.
            if has_cape and pl.category == "Hair" \
                    and slot == "hairBelowBody":
                slot = "backHair"
            rule = _OVER_CAP_REMAP.get(slot)
            if rule is None:
                return self._z_index(slot)
            target, token = rule
            if not cap_covers_hair or (
                token is not None and token in cap_vslot_tokens
            ):
                slot = target
            return self._z_index(slot)

        # ``ladder`` and ``rope`` are MapleStory's only back-facing
        # poses — the character is shown from behind so the visible
        # canvases come from the ``back*`` zmap cluster (backHair,
        # backHead, backBody, backMailChest, …). The body/coat
        # cluster is already authored in the correct relative order:
        # backMailChest sits above backBody for the same reason
        # mailChest sits above body in front view. The HEAD-region
        # cluster (backHair, backCap, backAccessoryEar) however is
        # authored DEEP in the front zmap because in front view those
        # canvases represent things drawn behind the body silhouette —
        # for back view the same canvases are the back of the head
        # and need to render ON TOP of the body / coat / head, with
        # cap > hair > head. ``_BACK_FACING_Z_OVERRIDE`` patches the
        # head-region slots to land above the body cluster while
        # leaving everything else at its natural zmap index. Any
        # front-only canvas that slipped through (e.g. ``face`` —
        # the Face img has no back variant) is pushed below the
        # back cluster so it can't poke through.
        back_facing = pose in ("ladder", "rope")
        back_floor = -1  # below backHair; any non-back slot lands here

        def effective_slot_z(pl: _Placement) -> int:
            """slot_z with the back-facing override / floor applied."""
            z = slot_z(pl)
            if not back_facing:
                return z
            slot = pl.z_slot
            absolute = _BACK_FACING_ABSOLUTE_Z.get(slot)
            if absolute is not None:
                return absolute
            override = _BACK_FACING_Z_OVERRIDE.get(slot)
            if override is not None:
                return override
            if not slot or not slot.startswith("back"):
                return back_floor
            return z

        # Effect overlays carry an integer z that's interpreted as
        # an ABSOLUTE layer offset relative to the whole character:
        # positive values render in front of every body / equip
        # canvas; negative values render behind everything. Within
        # each band the integer value drives ordering (z=2 sits
        # above z=1, z=-1 sits above z=-2). The base is chosen so
        # the effect clears every other body / equip slot:
        #   * front-facing: just above ``zmap_size`` (the natural
        #     "front-default" idx) so positives beat every front
        #     zmap slot.
        #   * back-facing: above the ``_BACK_FACING_Z_OVERRIDE``
        #     ceiling (head / cap / shield / weapon overrides go up
        #     to ~275) so positives beat the back-facing cluster
        #     too — otherwise an effect like ItemEff.img/1103248's
        #     ``backDefault`` z=2 ended up behind backHair (override
        #     254) on the ladder pose.
        # ``effect_back_base`` lands negative effect z below
        # ``characterEnd`` so the absolute "deepest" character slot
        # still draws on top of any background effect (cape /
        # backdrop sparkles authored with z=-1, -2, ...). In
        # front-facing the natural zmap order already does this
        # — characterEnd is at idx 1 and effect z=-1 → -1, deeper.
        # In back-facing the absolute characterEnd is -999, so the
        # effect base needs to sit below that.
        if back_facing and _BACK_FACING_Z_OVERRIDE:
            effect_front_base = max(_BACK_FACING_Z_OVERRIDE.values()) + 1
            effect_back_base = _BACK_FACING_ABSOLUTE_Z["characterEnd"] - 1
        else:
            effect_front_base = zmap_size
            effect_back_base = 0

        def z_for(pl: _Placement) -> int:
            if pl.category == "Effect":
                ez = pl.extra_z if pl.extra_z is not None else 0
                if ez >= 0:
                    return effect_front_base + ez
                return effect_back_base + ez
            return effective_slot_z(pl)

        placements.sort(key=z_for)
        return (placements, world_anchors) if return_anchors else placements

    def _render_placements(
        self, placements: List[_Placement], *, flip: bool = False,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        hair_hsv: Optional[Tuple[float, float, float]] = None,
    ) -> Image.Image:
        """Composite the supplied placements into a single image. When
        ``bbox`` is None, the canvas is the tight bounding box of
        ``placements``; otherwise it's the supplied
        ``(min_x, min_y, max_x, max_y)`` so multiple frames can
        share dimensions and a stable navel position."""
        def _w(pl: _Placement) -> int:
            return pl.width_override if pl.width_override is not None \
                else pl.pixel_canvas.width
        def _h(pl: _Placement) -> int:
            return pl.height_override if pl.height_override is not None \
                else pl.pixel_canvas.height
        if bbox is None:
            min_x = min(p.top_left[0] for p in placements if p.top_left is not None)
            min_y = min(p.top_left[1] for p in placements if p.top_left is not None)
            max_x = max(p.top_left[0] + _w(p)
                        for p in placements if p.top_left is not None)
            max_y = max(p.top_left[1] + _h(p)
                        for p in placements if p.top_left is not None)
        else:
            min_x, min_y, max_x, max_y = bbox
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        composite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for pl in placements:
            if pl.top_left is None:
                continue
            # Stabilized ItemEff placements ship a pre-decoded RGBA
            # composite (uniform-bbox across the cycle); use it
            # directly so per-frame canvas-size variation doesn't
            # leak into the bbox.
            if pl.decoded_override is not None:
                layer = pl.decoded_override
            else:
                try:
                    layer = decode_canvas(pl.pixel_canvas, region=self.region)
                except Exception:
                    continue
            if layer.mode != "RGBA":
                layer = layer.convert("RGBA")
            # Custom-hair recolor: HSV-recolor every Hair canvas (hair,
            # hairOverHead, hairShade, backHair, …) before compositing, so the
            # whole head recolors consistently and matches the live preview.
            if hair_hsv is not None and pl.category == "Hair":
                layer = apply_hsv_adjust(layer, *hair_hsv)
            composite.alpha_composite(
                layer,
                (pl.top_left[0] - min_x, pl.top_left[1] - min_y),
            )
        if flip:
            composite = composite.transpose(Image.FLIP_LEFT_RIGHT)
        return composite

    # ── internals ──────────────────────────────────────────────────────
    def _register_anchors(
        self,
        placement: _Placement,
        world_anchors: Dict[str, Tuple[int, int]],
        *,
        overwrite: bool,
    ) -> None:
        """Add this canvas's map points to the shared world-anchor dict.

        ``overwrite`` is True only for the canonical body canvas (it
        defines the reference frame). Sub-parts (arm, head, lHand…) only
        contribute anchors that haven't been seen yet — that's how arm
        provides ``hand`` for weapons and head provides ``brow`` for face
        without stomping body's navel."""
        if placement.top_left is None:
            return
        ox, oy = placement.origin
        tx, ty = placement.top_left
        for name, vec in placement.map_anchors.items():
            if not overwrite and name in world_anchors:
                continue
            world_anchors[name] = (tx + ox + vec[0], ty + oy + vec[1])
