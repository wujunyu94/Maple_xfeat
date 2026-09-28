import unittest
from types import SimpleNamespace

from .movement_control import walk_intervals
from .visual_motion import VisualMotion, MotionKeys


class FeedbackTest(unittest.TestCase):
    def simulate(self, distance, lose=False, delayed=False):
        state = SimpleNamespace(t=1., x=0., v=0., direction=0)
        transitions, events = [], []
        captures = [(1., 0.)]
        def sleep(dt):
            old = state.v
            state.v = max(-125, min(125, old+state.direction*1500*dt)) if state.direction else (
                (1 if old>=0 else -1)*max(0, abs(old)-900*dt))
            state.x += (old+state.v)*.5*dt
            state.t += dt
            if state.t-captures[-1][0] >= .11:
                captures.append((state.t, state.x))
        def observation():
            if lose and state.t>1.3:
                return None
            if delayed:
                ready = [r for r in captures if r[0]<=state.t-.10]
                if not ready:
                    return None
                t, x = ready[-1]
                return dict(time=t, world=[x,0], platform=1)
            return dict(time=state.t, world=[state.x,0], platform=1)
        def set_keys(*keys):
            direction = 1 if 'right' in keys else -1 if 'left' in keys else 0
            if direction != state.direction:
                transitions.append((state.t, direction, state.x))
            state.direction = direction
        motion = VisualMotion()
        clock = SimpleNamespace(perf_counter=lambda:state.t, sleep=sleep)
        nav = SimpleNamespace(motion=motion, keys=MotionKeys(SimpleNamespace(set=set_keys), motion, clock.perf_counter),
            alive=lambda:state.t<15, event=lambda name, **data:events.append((name,data)),
            o=SimpleNamespace(get=observation))
        ok = walk_intervals(nav, [(distance-2,distance+2)], 10, True, None, clock)
        return ok, state, transitions, events

    def test_long_distance_is_one_hold_until_near_target(self):
        ok, state, transitions, _ = self.simulate(600)
        self.assertTrue(ok)
        self.assertLessEqual(abs(state.x-600), 2)
        self.assertGreater(transitions[1][0]-transitions[0][0], 4)
        self.assertGreater(transitions[1][2], 580)

    def test_smaller_error_shortens_hold(self):
        holds = []
        for distance in (6, 12, 25):
            ok, state, transitions, _ = self.simulate(distance)
            self.assertTrue(ok)
            holds.append(transitions[1][0]-transitions[0][0])
        self.assertLess(holds[0], holds[1])
        self.assertLess(holds[1], holds[2])

    def test_slow_feature_rescue_does_not_pulse_distant_travel(self):
        _, _, transitions, _ = self.simulate(600, delayed=True)
        self.assertGreater(transitions[1][0]-transitions[0][0], 4)
        self.assertGreater(transitions[1][2], 530)

    def test_observation_loss_releases_and_records_reason(self):
        ok, state, _, events = self.simulate(600, lose=True)
        self.assertFalse(ok)
        self.assertEqual(state.direction, 0)
        self.assertTrue(any(data.get('reason')=='missing_or_stale_ground_observation' for _,data in events))


if __name__ == '__main__':
    unittest.main()
