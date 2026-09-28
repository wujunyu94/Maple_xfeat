"""Bounded world-velocity extrapolation, never inferred from camera motion alone."""
import numpy as np


class PlayerState:
    def __init__(self, max_age=.25):
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.last_world = None
        self.last_observed = None
        self.velocity = np.zeros(2)
        self.velocity_ready = False
        self.pending = None
        self.last_meta = None

    def update(self, observation, pose, now, screen_scale=1.0):
        if pose.camera is None:
            self.reset()
            return dict(status="LOST", world=None, screen=None, reason="Camera not localized")
        camera = np.array(pose.camera)
        accepted = False
        reacquired = False
        if observation:
            point = camera+np.array(observation["screen"])/screen_scale
            if self.last_world is not None:
                dt = now-self.last_observed
                if dt <= 0 or dt > 1:
                    self.reset()
                else:
                    expected = self.last_world+self.velocity*dt
                    if np.linalg.norm(point-expected) > 35+900*dt:
                        if self.pending is not None and now-self.pending[1] <= .15 and np.linalg.norm(point-self.pending[0]) < 25:
                            self.reset()
                            reacquired = True
                        else:
                            self.pending = (point, now)
                            observation = None
            if observation:
                if self.last_world is not None:
                    dt = now-self.last_observed
                    if .008 <= dt <= self.max_age:
                        velocity = (point-self.last_world)/dt
                        speed = np.linalg.norm(velocity)
                        if speed <= 1200:
                            self.velocity = .55*velocity+.45*self.velocity if self.velocity_ready else velocity
                            self.velocity_ready = True
                self.last_world = point
                self.last_observed = now
                self.last_meta = observation
                self.pending = None
                accepted = True
        if self.last_world is None:
            return dict(status="LOST", world=None, screen=None, reason="No verified player observation")
        age = now-self.last_observed
        if not accepted and (age > self.max_age or not self.velocity_ready):
            if age > self.max_age:
                self.velocity_ready = False
                self.velocity[:] = 0
            return dict(status="LOST", world=None, screen=None, observation_age=age,
                        reason="Prediction expired or velocity not established")
        world = self.last_world if accepted else self.last_world+self.velocity*age
        uncertainty = 3.0 if accepted else 3+180*age+800*age*age
        screen = (world-camera)*screen_scale
        return dict(status=("REACQUIRED" if reacquired else "MEASURED") if accepted else "PREDICTED",
                    world=world.tolist(), screen=screen.tolist(), observation_age=age,
                    uncertainty_px=uncertainty, velocity=self.velocity.tolist(),
                    source=self.last_meta.get("source", "visual"), calibrated=self.last_meta.get("calibrated", False),
                    feature_bbox=self.last_meta.get("feature_bbox") if accepted else None,
                    bbox=self.last_meta.get("bbox") if accepted else None,
                    reason="Fresh name/feature observation" if accepted else "World velocity extrapolation; NOT a current observation")


def compare_coordinates(player, yellow, offset_y=45.0):
    """Difference on the same input frame; the minimap is a reference, not truth."""
    if player.get("world") is None or not yellow.get("detected"):
        return None
    feet = np.array(player["world"])
    raw = np.array(yellow["raw_world"])
    body = feet-[0, offset_y]
    delta = body-raw
    return dict(feet_minus_yellow_raw=(feet-raw).tolist(),
                visual_body_world=body.tolist(), visual_body_minus_yellow_raw=delta.tolist(),
                distance=float(np.linalg.norm(delta)), body_offset_y=offset_y,
                kind="prediction_comparison" if player["status"] == "PREDICTED" else "measurement_comparison")


def geometry_advice(atlas, pose, player, scale):
    if pose.camera is None or player.get("world") is None:
        return None
    px, py = player["world"]
    candidates = []
    for x, y1, y2 in atlas.meta.get("ladders", []):
        low, high = sorted((y1, y2))
        vertical = max(low-py, py-high, 0)
        if vertical <= 90 and abs(x-px) <= 250:
            candidates.append((abs(x-px)+vertical*2, x, low, high))
    if not candidates:
        return None
    _, x, y1, y2 = min(candidates)
    fresh = player["status"] in ("MEASURED", "REACQUIRED")
    return dict(ladder_world=[x, y1, y2], dx_world=x-px, dx_screen=(x-px)*scale,
                screen_line=[(x-pose.camera[0])*scale, (y1-pose.camera[1])*scale, (y2-pose.camera[1])*scale],
                predicted_player=not fresh, control_ready=False,
                reason="Projected WZ geometry only; requires local ladder verification and a control state machine")
