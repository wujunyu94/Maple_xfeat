import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

import cv2
import numpy as np

from .jump_policy import (covered_by_source,ground_speed,airborne_keys,
                          safe_drop_intervals,safe_vertical_interval,launch_projection)
from .visual_motion import VisualMotion,MotionKeys
from unittest.mock import patch
from .diagnostic_recorder import DiagnosticRecorder


class JumpFeedbackRegression(unittest.TestCase):
    def test_late_frame_does_not_authorize_takeoff_beyond_real_window(self):
        edge=SimpleNamespace(action='JUMP_RIGHT',takeoff_x=334.24,trigger_x=334,
                             takeoff_x_range=(328.096,334.24),trigger_x_range=None)
        self.assertEqual(launch_projection(edge,330.66,117.52,.018)[0],'late')
        self.assertEqual(launch_projection(edge,326.66,117.52,.018)[0],'ready')

    def test_vertical_edge_landing_rejected_when_safe_interior_is_too_narrow(self):
        a=SimpleNamespace(x_min=-1214,x_max=73,surface_y_at=lambda x:-2948)
        b=SimpleNamespace(x_min=-843,x_max=-206,surface_y_at=lambda x:-2997)
        e=SimpleNamespace(from_id=75,to_id=78,takeoff_x_range=(-220,-212),landing_x_range=(-220,-212))
        self.assertIsNone(safe_vertical_interval(SimpleNamespace(nodes={75:a,78:b}),e))

    def test_drop_zone_avoids_rope_and_intermediate_platform(self):
        def node(i,lo,hi,y):return SimpleNamespace(id=i,x_min=lo,x_max=hi,surface_y_at=lambda x:y)
        g=SimpleNamespace(nodes={1:node(1,0,200,0),2:node(2,0,200,100),3:node(3,130,155,50)},
                          ladder_ropes={1:SimpleNamespace(x=80,y1=-10,y2=110)})
        e=SimpleNamespace(from_id=1,to_id=2,trigger_x_range=(0,200))
        zones=safe_drop_intervals(g,e)
        self.assertTrue(any(lo<=30<=hi for lo,hi in zones))
        self.assertFalse(any(lo<=80<=hi or lo<=140<=hi for lo,hi in zones))

    def test_delayed_visual_kalman_does_not_reuse_prediction_as_measurement(self):
        m=VisualMotion();o=dict(time=1.,platform=1,world=[0.,0.])
        m.estimate(o,1.);m.command(1,1.02)
        first=m.estimate(o,1.04);second=m.estimate(o,1.04)
        self.assertAlmostEqual(first['x'],second['x'])
        self.assertGreater(first['vx'],0)
        self.assertIsNone(m.estimate(o,1.3))
        self.assertIsNone(m.estimate(dict(time=1.3,platform=None,world=[8,0]),1.31))

    def test_predictive_walk_has_no_fixed_near_target_pulse_train(self):
        from .coverage_trial import CoverageNavigator
        clock=SimpleNamespace(t=1.,x=0.,v=0.,direction=0,transitions=[])
        def sleep(dt):
            for _ in range(max(1,round(dt/.001))):
                step=dt/max(1,round(dt/.001))
                old=clock.v
                clock.v=max(-125,min(125,old+clock.direction*1500*step)) if clock.direction else (1 if old>=0 else -1)*max(0,abs(old)-900*step)
                clock.x+=(old+clock.v)*.5*step;clock.t+=step
        class Keys:
            def set(self,*keys):
                d=1 if 'right' in keys else -1 if 'left' in keys else 0
                if d!=clock.direction:clock.transitions.append(d)
                clock.direction=d
        nav=CoverageNavigator.__new__(CoverageNavigator)
        nav.motion=VisualMotion();nav.keys=MotionKeys(Keys(),nav.motion,lambda:clock.t)
        nav.o=SimpleNamespace(get=lambda:dict(time=clock.t,world=[clock.x,0],platform=1))
        nav.alive=lambda:clock.t<10
        nav.event=lambda *a,**k:None
        with patch('no_minimap_lab.coverage_trial.time',SimpleNamespace(perf_counter=lambda:clock.t,sleep=sleep)):
            self.assertTrue(nav.walk(100,tolerance=2))
        self.assertLessEqual(abs(clock.x-100),2)
        self.assertLessEqual(sum(d!=0 for d in clock.transitions),3)
        clock.transitions.clear()
        with patch('no_minimap_lab.coverage_trial.time',SimpleNamespace(perf_counter=lambda:clock.t,sleep=sleep)):
            self.assertTrue(nav.walk_intervals([(80,180)],source=1))
        self.assertFalse(any(clock.transitions))

    def test_overlap_landing_hits_higher_source_first(self):
        a=SimpleNamespace(x_min=-722,x_max=-268,surface_y_at=lambda x:-946.3)
        b=SimpleNamespace(x_min=-300,x_max=-107,surface_y_at=lambda x:-898.8)
        g=SimpleNamespace(nodes={23:a,24:b})
        self.assertTrue(covered_by_source(g,SimpleNamespace(from_id=23,to_id=24,landing_x=-287)))
        self.assertFalse(covered_by_source(g,SimpleNamespace(from_id=23,to_id=24,landing_x=-240)))

    def test_runup_uses_ground_velocity_and_rejects_missing_or_jumping_position(self):
        rows=[dict(time=i*.04,world=[i*5.,0.],platform=71) for i in range(5)]
        self.assertAlmostEqual(ground_speed(rows,71,0,.16),125)
        rows[2]['platform']=None
        self.assertIsNone(ground_speed(rows,71,0,.16))
        rows[2]['platform']=71;rows[2]['world'][0]=1000
        self.assertIsNone(ground_speed(rows,71,0,.16))

    def test_landing_and_overshoot_never_request_opposite_direction(self):
        self.assertEqual(airborne_keys('left',-525,-599,False),('left',))
        self.assertEqual(airborne_keys('left',-525,-599,True),())
        self.assertEqual(airborne_keys('left',-605,-599,False),())
        self.assertEqual(airborne_keys('right',250,239,False),())

    def test_video_frame_index_preserves_irregular_capture_times(self):
        folder=Path(__file__).resolve().parent/'output'/('recorder_test_'+uuid.uuid4().hex)
        recorder=DiagnosticRecorder(folder)
        frame=np.zeros((96,128,3),np.uint8)
        try:
            for stamp in (10.,10.04,10.15,10.7):
                recorder.record(frame,dict(time=stamp,observed_at=stamp+.01,status='TEST',platform=1,world=[0,0]))
        finally:
            recorder.close()
        rows=[json.loads(s) for s in (folder/'video_frames.jsonl').read_text().splitlines()]
        self.assertEqual([r['capture_time'] for r in rows],[10.,10.15,10.7])
        self.assertEqual([r['playback_seconds'] for r in rows],[0.,.1,.2])
        video=cv2.VideoCapture(str(folder/'replay.avi'))
        try:
            self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)),3)
            self.assertTrue(video.read()[0])
        finally:
            video.release()


if __name__=='__main__':
    unittest.main()
