"""Same-frame, independent visual and yellow-dot coordinate comparison."""
import time

from .async_localizer import AsyncLocalizer
from .localizer import Localizer
from .main_adapters import MainPlayerAnchor, YellowWorldReference
from .player_state import PlayerState, compare_coordinates, geometry_advice
from .ladder_overlay import LadderOverlay


class Pipeline:
    def __init__(self, atlas, scale=1.0, asynchronous=True, body_offset_y=45.0, backend="sift-cpu", asynchronous_reference=False):
        self.atlas = atlas
        self.scale = scale
        self.body_offset_y = body_offset_y
        self.model = (AsyncLocalizer if asynchronous else Localizer)(atlas, scale, backend=backend)
        self.ladder_overlay = LadderOverlay(atlas, scale)
        self.anchor = MainPlayerAnchor()
        self.yellow = YellowWorldReference(atlas.meta["map_id"])
        self.async_reference = None
        if asynchronous_reference:
            from .async_reference import AsyncReference
            self.async_reference = AsyncReference(self.yellow, body_offset_y)
        self.player_state = PlayerState()
        self.last_timestamp = None

        self.pose = None

    def reset(self):
        self.model.reset()
        self.anchor.reset()
        self.player_state.reset()
        self.ladder_overlay.reset()
        self.last_timestamp = None
        if self.async_reference:
            self.async_reference.reset()

    def reload_anchor(self):
        self.anchor.close()
        self.anchor = MainPlayerAnchor()
        self.player_state.reset()
        if self.async_reference:
            self.async_reference.reset()

    def close(self):
        if self.async_reference:
            self.async_reference.close()
        self.anchor.close()
        if isinstance(self.model, AsyncLocalizer):
            self.model.close()

    def update(self, frame, timestamp=None):
        start = time.perf_counter()
        now = start if timestamp is None else float(timestamp)
        if self.last_timestamp is not None and (now <= self.last_timestamp or now-self.last_timestamp > 1):
            self.reset()
        self.last_timestamp = now
        # Deliberately no yellow result is passed into either of these calls.
        self.pose = self.model.update(frame, now)
        after_camera = time.perf_counter()
        observation = self.anchor.detect(frame)
        after_player = time.perf_counter()
        player = self.player_state.update(observation, self.pose, now, self.scale)
        # This separate tracker sees the identical raw frame, never scene outputs.
        if self.async_reference:
            yellow, comparison = self.async_reference.update(frame, now, player)
        else:
            yellow = self.yellow.detect(frame)
            comparison = compare_coordinates(player, yellow, self.body_offset_y)
        advice = geometry_advice(self.atlas, self.pose, player, self.scale)
        ladder_start = time.perf_counter()
        ladders = self.ladder_overlay.update(frame, self.pose, now)
        record = dict(map_id=self.atlas.meta["map_id"], timestamp=now, **self.pose.to_dict(),
                      player=player if player.get("screen") is not None else None,
                      player_state=player, player_world=player.get("world"),
                      yellow=yellow, comparison=comparison, ladder=advice, ladders=ladders,
                      backend=self.model.backend)
        record["timings_ms"] = dict(camera=(after_camera-start)*1000,
                                     player=(after_player-after_camera)*1000,
                                     yellow=yellow["elapsed_ms"],
                                     ladders=(time.perf_counter()-ladder_start)*1000,
                                     pipeline=(time.perf_counter()-start)*1000,
                                     background_anchor=self.pose.anchor_ms)
        return record
