"""Main-program input-aware Kalman driven exclusively by visual world X.

Measurements arrive late: advance the filter only along capture timestamps,
replay intervening input transitions, and project to now without modifying its
clock. Never feed a delayed capture into a filter already predicted to now.
"""
from collections import deque
from src.vision.world_filter import InputAwareHorizontalKalman


class VisualMotion:
    def __init__(self):
        self.model=InputAwareHorizontalKalman(measurement_noise_floor=1.)
        self.model.set_measurement_step(1.)  # Visual pixels, not minimap grid cells.
        self.commands=deque(maxlen=256)
        self.direction=0
        self.last_stamp=None
        self.platform=None

    def command(self,direction,timestamp):
        if direction!=self.direction:
            self.direction=direction
            self.commands.append((timestamp,direction))

    def estimate(self,observation,now,max_age=.18):
        if (not observation or not observation.get('world') or observation.get('platform') is None
                or not 0<=now-observation['time']<=min(.30,max_age)):
            self.last_stamp=None
            return None
        stamp=observation['time']
        if self.last_stamp is not None and stamp<self.last_stamp:
            return None
        if self.last_stamp is None or stamp-self.last_stamp>.25 or self.platform!=observation['platform']:
            self.model.reset()
            direction=next((d for t,d in reversed(self.commands) if t<=stamp),0)
            self.model.set_direction(direction,stamp)
            self.model.correct_measurement(observation['world'][0],stamp)
        elif stamp>self.last_stamp:
            for t,d in self.commands:
                if self.last_stamp<t<=stamp:
                    self.model.set_direction(d,t)
            self.model.correct_measurement(observation['world'][0],stamp)
        self.last_stamp=stamp
        self.platform=observation['platform']
        x,v=self.model.x,self.model.vx
        direction=self.model.direction
        last=stamp
        for t,d in self.commands:
            if stamp<t<=now:
                x,v=self.integrate(x,v,direction,t-last)
                last=t;direction=d
        x,v=self.integrate(x,v,direction,now-last)
        return dict(x=x,vx=v,raw_x=observation['world'][0],
                    stop_x=x+v*.025+(1 if v>=0 else -1)*v*v/(2*self.model.drag_accel))

    def integrate(self,x,v,direction,dt):
        while dt>1e-9:
            step=min(.01,dt)
            if direction:
                new=max(-self.model.v_max,min(self.model.v_max,v+direction*self.model.push_accel*step))
            else:
                new=(1 if v>=0 else -1)*max(0,abs(v)-self.model.drag_accel*step)
            x+=(v+new)*.5*step;v=new;dt-=step
        return x,v


class MotionKeys:
    def __init__(self,keys,motion,clock):
        self.keys=keys;self.motion=motion;self.clock=clock

    def set(self,*keys):
        self.keys.set(*keys)
        self.motion.command(1 if 'right' in keys else -1 if 'left' in keys else 0,self.clock())
