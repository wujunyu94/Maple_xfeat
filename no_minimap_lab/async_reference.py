"""Optional yellow comparison must never block visual localization publication."""
from concurrent.futures import ThreadPoolExecutor
from .player_state import compare_coordinates


class AsyncReference:
    def __init__(self, detector, body_offset_y):
        self.detector, self.body_offset_y = detector, body_offset_y
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='yellow-reference')
        self.pending = None
        self.generation = 0
        self.last = None
        self.error = None

    def reset(self):
        self.generation += 1
        self.last = None
        self.error = None

    def compute(self, frame, timestamp, player, generation):
        yellow = self.detector.detect(frame)
        comparison = compare_coordinates(player, yellow, self.body_offset_y)
        return generation, timestamp, yellow, comparison

    def update(self, frame, timestamp, player):
        if self.pending is not None and self.pending.done():
            try:
                generation, stamp, yellow, comparison = self.pending.result()
                if generation == self.generation:
                    self.last = stamp, yellow, comparison
                    self.error = None
            except Exception as exc:
                self.last = None
                self.error = f'{type(exc).__name__}: {exc}'
            self.pending = None
        if self.pending is None:
            self.pending = self.pool.submit(self.compute, frame.copy(), timestamp, dict(player), self.generation)
        if self.last is not None:
            stamp, yellow, comparison = self.last
            age = timestamp-stamp
            if 0 <= age <= .25:
                return dict(yellow, timestamp=stamp, observation_age=age), comparison
        return dict(detected=False, raw_world=None, snapped_world=None, elapsed_ms=0,
                    reason=self.error or 'Reference pending or older than 250ms', timestamp=None), None

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
