"""Five physical all-rope tours. Observations never consume minimap pixels."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2

from .navigation import Observer, Navigator
from .input_service import STATE, write_json
from .atlas import load_atlas
from .jump_policy import (covered_by_source, ground_speed, airborne_keys,
                          safe_drop_intervals,safe_vertical_interval,launch_projection)
from .visual_motion import VisualMotion,MotionKeys
from .ignore_regions import load as load_ignore_regions
from src.vision.wz_map_reader import WzMapReader
from src.engine.platform_graph import PlatformGraphBuilder, PlatformEdge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'no_minimap_lab/output/coverage_five'
_REPORT_OUTPUT = None
_OPEN_REPORT = True


def is_travel_portal(portal):
    # Spawn points (pt=0) and named script markers without a destination do
    # not transport the player. Keep real intra-map and inter-map entrances.
    return int(portal.get('tm', portal.get('toMap', 999999999))) != 999999999


def rope_takeoff(node, rope_x):
    if node.x_min+2<=rope_x<=node.x_max-2:
        return rope_x
    return max(node.x_min+8, min(node.x_max-8, rope_x))


def build_plan():
    load_atlas(101000000, progress=lambda _: None)
    g = PlatformGraphBuilder.build_from_map_dict(WzMapReader('Map').load_map(101000000), enable_teleport=False)
    for n in g.edges:
        g.edges[n] = [e for e in g.edges[n] if 'TELEPORT' not in e.action and e.action != 'PORTAL']
    current = 1
    steps = []
    for lid, rope in sorted(g.ladder_ropes.items()):
        source = rope.bottom_platform_id or rope.top_platform_id
        route = g.find_path(current, source, allow_portal=False) if current != source else []
        if current != source and not route:
            raise RuntimeError(f'No physical connector P{current}->P{source} for rope {lid}')
        edge = next((e for e in g.edges[source] if e.ladder_id == lid and 'UP' in e.action), None)
        if rope.bottom_platform_id and edge is None:
            raise RuntimeError(f'No ascent for rope {lid}')
        steps.append(dict(ladder=asdict(rope), approach=[asdict(e) for e in route],
                          traversal=asdict(edge) if edge else None,
                          mode='ascent' if edge else 'hanging_down_and_return'))
        current = rope.top_platform_id
    finish = g.find_path(current, 87, allow_portal=False)
    if current != 87 and not finish:
        raise RuntimeError('No physical finish')
    return dict(map_id=101000000, source=1, target=87, rounds=5,
                yellow_dot_used=False, steps=steps, finish=[asdict(e) for e in finish])


class CoverageNavigator(Navigator):
    """Counts actual issued actions, including failures, independently of coverage."""
    def __init__(self, *args):
        super().__init__(*args)
        self.serial = 0
        self.covered = set()
        self.motion=VisualMotion()
        self.keys=MotionKeys(self.keys,self.motion,time.perf_counter)
        self.pending_snapshot=None
        # Merged sloping platforms can invert average-height DOWN_JUMP edges.
        # Use the actual surface at the trigger, preserving the shared graph file.
        repairs=[]
        for source,edges in self.g.edges.items():
            keep=[]
            for e in edges:
                x=e.trigger_x
                if e.action=='DOWN_JUMP' and x is not None:
                    a,b=self.g.nodes[e.from_id],self.g.nodes[e.to_id]
                    if b.surface_y_at(x)<=a.surface_y_at(x)+10:
                        self.event('edge_geometry_rejected',edge=asdict(e))
                        if a.surface_y_at(x)>b.surface_y_at(x)+20:
                            repairs.append(PlatformEdge(e.to_id,e.from_id,'DOWN_JUMP',e.cost,
                                trigger_x=x,landing_x=x,target_y=round(a.surface_y_at(x)),
                                description='Corrected direction using local foothold heights'))
                        continue
                keep.append(e)
            self.g.edges[source]=keep
        for e in repairs:
            self.g.edges[e.from_id].append(e)

    def attempt(self, kind, edge=None, defer_snapshot=False, **extra):
        self.serial += 1
        data = dict(attempt=self.serial, kind=kind, **extra)
        if edge:
            data['edge'] = asdict(edge)
        data['observation'] = self.o.get()
        if self.o.frame is not None:
            if defer_snapshot:
                self.pending_snapshot=(self.serial,self.o.frame.copy())
            else:
                cv2.imwrite(str(self.output / f'action_{self.serial:04d}_before.jpg'), self.o.frame)
        self.event('action_attempt', **data)
        return self.serial

    def walk(self,x,timeout=25,tolerance=5):
        return self.walk_intervals([(x-tolerance,x+tolerance)],timeout=timeout,precise=tolerance<=5)

    def walk_intervals(self,intervals,timeout=25,precise=False,source=None):
        from .movement_control import walk_intervals
        return walk_intervals(self, intervals, timeout, precise, source, time)

    def result(self, number, success, **extra):
        self.keys.set()
        if self.o.frame is not None:
            cv2.imwrite(str(self.output / f'action_{number:04d}_after.jpg'), self.o.frame)
        self.event('action_result', attempt=number, success=bool(success), observation=self.o.get(), **extra)
        return bool(success)

    def wait(self, seconds, keys=()):
        # DOWN+ALT at a rope can leave the avatar attached without a platform.
        # Recover only from stationary, fresh visual evidence on a known axis.
        obs=self.o.get()
        if not keys and obs and obs['platform'] is None:
            history=[r for r in self.o.history if obs['time']-.8<=r['time']<=obs['time']]
            stationary=(len(history)>=8 and history[-1]['time']-history[0]['time']>.6 and
                        all(r['world'] and abs(r['world'][0]-obs['world'][0])<5 and
                            abs(r['world'][1]-obs['world'][1])<5 for r in history))
            rope=next((l for l in self.g.ladder_ropes.values() if abs(obs['world'][0]-l.x)<8 and
                       l.y1-10<=obs['world'][1]<=l.y2+40),None)
            if stationary and rope:
                self.event('attached_rope_recovery',ladder=rope.id,observation=obs)
                return super().wait(seconds,('up',))
        return super().wait(seconds,keys)

    def execute(self, edge):
        self.event('edge_start', edge=asdict(edge))
        action = edge.action
        x = edge.takeoff_x if edge.takeoff_x is not None else edge.trigger_x
        if x is None:
            x = self.g.nodes[edge.to_id].center_x
        horizontal_jump=action in ('JUMP_LEFT','JUMP_RIGHT')
        if horizontal_jump and covered_by_source(self.g,edge):
            self.event('jump_geometry_rejected',edge=asdict(edge),reason='landing covered by higher source foothold')
            return False
        if action=='DOWN_JUMP':
            intervals=safe_drop_intervals(self.g,edge)
            if not intervals:
                self.event('jump_geometry_rejected',edge=asdict(edge),reason='no safe drop interval')
                return False
            self.event('drop_interval',edge=asdict(edge),intervals=intervals)
        elif action=='JUMP_UP':
            interval=safe_vertical_interval(self.g,edge)
            if interval is None:
                self.event('jump_geometry_rejected',edge=asdict(edge),reason='landing interior too narrow after edge margin')
                return False
        if 'CLIMB' in action:
            x=rope_takeoff(self.g.nodes[edge.from_id],x)
        if horizontal_jump:aligned=self.prepare_runup(edge,x)
        elif action=='DOWN_JUMP':aligned=self.walk_intervals(intervals,source=edge.from_id)
        elif action=='JUMP_UP':aligned=self.walk_intervals([interval],source=edge.from_id)
        else:aligned=self.walk(x,tolerance=2 if 'CLIMB' in action else 5)
        if not aligned:
            self.event('alignment_failed', edge=asdict(edge))
            return False
        if not horizontal_jump:
            self.wait(.15)
        before = self.o.get()
        if not before:
            self.keys.set()
            return False
        if not action.startswith('WALK') and before['platform']!=edge.from_id:
            self.keys.set()
            self.event('source_changed_during_alignment',edge=asdict(edge),observation=before)
            return False
        if 'CLIMB' in action:
            rope = self.g.ladder_ropes[edge.ladder_id]
            if any(is_travel_portal(p) and abs(float(p.get('x', 1e9))-x)<28 and
                   abs(float(p.get('y', 1e9))-before['world'][1])<70 for p in self.g.portals):
                self.event('portal_proximity_rejected', ladder=rope.id)
                return False
            down = 'DOWN' in action
            jump = 'JUMP' in action or (not down and abs(before['world'][0]-rope.x)>5)
            number = self.attempt('grab', edge, ladder=rope.id, jump=jump, direction='down' if down else 'up')
            side_rope=not (self.g.nodes[edge.from_id].x_min<=rope.x<=self.g.nodes[edge.from_id].x_max)
            horizontal=() if not side_rope else ('right' if before['world'][0]<rope.x else 'left',)
            self.wait(.09, ('alt', 'up')+horizontal if jump else (('down',) if down else ('up',)))
            start = time.perf_counter()
            caught = False
            evidence = []
            while self.alive() and time.perf_counter()-start < max(10, rope.length/65+4):
                obs = self.o.get()
                if not obs:
                    self.keys.set()
                    time.sleep(.025)
                    continue
                px, py = obs['world']
                # Require movement along the rope past the maximum jump rise;
                # shorter ropes use stable arrival plus in-rope observations.
                if abs(px-rope.x)<=7 and rope.y1+12<py<rope.y2-8:
                    if not evidence or obs['time'] != evidence[-1]['time']:
                        evidence.append(dict(time=obs['time'], x=px, y=py))
                if not caught and len(evidence)>=4:
                    dy=evidence[-1]['y']-evidence[0]['y']
                    if (dy>65 if down else dy < -65) and evidence[-1]['time']-evidence[0]['time']>.65:
                        caught=True
                        self.event('grab_verified', attempt=number, ladder=rope.id, evidence=evidence[-4:])
                if self.o.stable(edge.to_id):
                    arrived = len(evidence)>=3
                    self.result(number, caught or arrived, reached_target=True, ladder=rope.id,
                                evidence_count=len(evidence))
                    if arrived or caught:
                        self.covered.add(rope.id)
                    return True
                if jump and side_rope and not caught and abs(px-rope.x)>4 and time.perf_counter()-start<.6:
                    self.keys.set('up','right' if rope.x>px else 'left')
                else:
                    self.keys.set('down' if down else 'up')
                time.sleep(.025)
            self.result(number, caught, reached_target=False, ladder=rope.id, evidence_count=len(evidence))
            return False
        if action == 'DOWN_JUMP' or action.startswith('JUMP'):
            if horizontal_jump:
                latest=self.o.get()
                final_projection=launch_projection(edge,latest['world'][0],self.launch_speed,
                    time.perf_counter()-latest['time']) if latest else None
                if (not latest or latest['platform']!=edge.from_id or final_projection[0]!='ready'):
                    self.keys.set()
                    self.event('runup_rejected',reason='launch window expired before keypress',observation=latest)
                    return False
            number = self.attempt('platform_jump', edge,defer_snapshot=horizontal_jump)
            if action == 'DOWN_JUMP':
                self.wait(.12, ('down',))
                self.wait(.12, ('down', 'alt'))
            else:
                direction='left' if 'LEFT' in action else 'right' if 'RIGHT' in action else None
                if horizontal_jump:
                    # Send the time-critical jump before disk encoding the before image.
                    self.keys.set('alt',direction)
                    if self.pending_snapshot is not None:
                        ident,frame=self.pending_snapshot;self.pending_snapshot=None
                        cv2.imwrite(str(self.output/f'action_{ident:04d}_before.jpg'),frame)
                self.wait(.09, ('alt', direction) if direction else ('alt',))
            start=time.perf_counter()
            contacted=False
            previous_y=None
            while self.alive() and time.perf_counter()-start<2.5:
                obs=self.o.get()
                if self.o.stable(edge.to_id):
                    return self.result(number, True)
                landing=edge.landing_x if edge.landing_x is not None else x
                if horizontal_jump and obs:
                    if obs['platform']==edge.to_id and previous_y is not None and obs['world'][1]>=previous_y-2:
                        if not contacted:
                            self.event('landing_contact',attempt=number,observation=obs)
                        contacted=True
                    previous_y=obs['world'][1]
                    self.keys.set(*airborne_keys(direction,obs['world'][0],landing,contacted))
                else:
                    self.keys.set()
                time.sleep(.025)
            return self.result(number, False)
        # WALK_*_DROP is performed by walking beyond the source edge.
        self.wait(.7)
        return self.o.stable(edge.to_id)

    def prepare_runup(self,edge,x):
        """Do not stop between the measured ground run and jump keypress."""
        sign=1 if edge.action=='JUMP_RIGHT' else -1
        direction='right' if sign==1 else 'left'
        source=self.g.nodes[edge.from_id]
        start_x=max(source.x_min+8,min(source.x_max-8,x-sign*55))
        if sign*(x-start_x)<35:
            self.event('runup_rejected',reason='insufficient ground distance',edge=asdict(edge))
            return False
        if not self.walk(start_x,tolerance=8):
            return False
        begun=time.perf_counter()
        while self.alive() and time.perf_counter()-begun<1.5:
            obs=self.o.get()
            if not obs or obs['platform']!=edge.from_id:
                self.keys.set()
                self.event('runup_rejected',reason='lost source platform or position',observation=obs)
                return False
            px=obs['world'][0]
            speed=ground_speed(list(self.o.history),edge.from_id,begun,obs['time'])
            # 125 world px/s is the current graph's walking model, not a
            # measured universal maximum. Log the launch speed for calibration.
            projection=launch_projection(edge,px,speed or 0,time.perf_counter()-obs['time'])
            if projection[0]!='approach':
                ready=(projection[0]=='ready' and time.perf_counter()-begun>=.22
                       and speed is not None and sign*speed>=112.5)
                self.event('runup_ready' if ready else 'runup_rejected',edge=asdict(edge),
                           measured_vx=speed,minimum_speed=112.5,observation=obs,
                           projected_launch_x=projection[1],launch_window=projection[2])
                if not ready:
                    self.keys.set()
                else:
                    self.launch_speed=speed
                return ready
            self.keys.set(direction)
            time.sleep(.01)
        self.keys.set()
        return False

    def hanging(self, rope):
        if not self.walk(rope.x):
            return False
        before=self.o.get()
        if not before or not self.o.stable(rope.top_platform_id):
            return False
        number=self.attempt('grab', ladder=rope.id, jump=False, direction='down', hanging=True)
        end=time.perf_counter()+max(12,rope.length/50)
        samples=[]
        depth=min(rope.y2-45,rope.y1+max(100,rope.length*.7))
        reached=False
        while self.alive() and time.perf_counter()<end:
            obs=self.o.get()
            if obs and abs(obs['world'][0]-rope.x)<8:
                if not samples or samples[-1]['time']!=obs['time']:
                    samples.append(dict(time=obs['time'],world=obs['world']))
                if obs['world'][1]>=depth:
                    reached=True
                    break
            self.keys.set(*(('down',) if obs else ()))
            time.sleep(.025)
        self.keys.set()
        # A hanging-rope success needs a direction reversal on its axis and a
        # stable return to the original top platform, not merely falling past it.
        if reached:
            end=time.perf_counter()+max(12,rope.length/50)
            while self.alive() and time.perf_counter()<end:
                if self.o.stable(rope.top_platform_id):
                    self.covered.add(rope.id)
                    return self.result(number, True, ladder=rope.id, reached_target=True,
                                       excursion_depth=depth, descent_samples=samples)
                self.keys.set(*(('up',) if self.o.get() else ()))
                time.sleep(.025)
        return self.result(number, False, ladder=rope.id, reached_target=False)

    def tour(self, plan, completed=(), started=None):
        if not completed and not self.o.stable(1,.35):
            return dict(success=False,reason='P1 start not stable')
        started=started if started is not None else time.perf_counter()
        self.covered.update(completed)
        self.event('tour_start',source=1,target=87,required_ladders=list(self.g.ladder_ropes),
                   continued=bool(completed),previously_completed=list(completed),original_start=started)
        for step in plan['steps']:
            rope=self.g.ladder_ropes[step['ladder']['id']]
            if rope.id in completed:
                continue
            source=rope.bottom_platform_id or rope.top_platform_id
            self.event('coverage_target', ladder=rope.id, source=source)
            done=False
            for retry in range(8):
                if not self.alive():
                    break
                approach=self.run(source)
                if not approach.get('success'):
                    break
                if step['traversal']:
                    edge=next(e for e in self.g.edges[source] if e.ladder_id==rope.id and 'UP' in e.action)
                    done=self.execute(edge) and rope.id in self.covered
                else:
                    done=self.hanging(rope)
                self.event('coverage_target_result',ladder=rope.id,success=done,retry=retry)
                if done:
                    break
                self.wait(.5)
            if not done:
                return dict(success=False,reason=f'Rope {rope.id} incomplete',covered=sorted(self.covered),seconds=time.perf_counter()-started)
        finish=self.run(87)
        result=dict(success=finish.get('success',False) and self.covered==set(self.g.ladder_ropes),
                    covered=sorted(self.covered),seconds=time.perf_counter()-started,source=1,target=87)
        self.event('tour_end',**result)
        return result


def _main():
    global OUT, _REPORT_OUTPUT, _OPEN_REPORT
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--backend',choices=['sift','xfeat'],default='sift')
    parser.add_argument('--resume-round',type=int)
    parser.add_argument('--output',type=Path,default=OUT,help='New result directory for a fresh test')
    parser.add_argument('--no-open-report',action='store_true',help='Generate results without opening browser')
    args=parser.parse_args()
    if args.rounds<1 or (args.resume_round is not None and not 1<=args.resume_round<=args.rounds):
        parser.error('rounds must be positive and resume-round must be within rounds')
    OUT=args.output.resolve()
    if not args.resume_round and not args.plan_only and OUT.exists() and any(OUT.glob('round_*')):
        parser.error('Existing round records: choose a new --output directory, or explicitly --resume-round')
    OUT.mkdir(parents=True,exist_ok=True)
    plan=build_plan()
    plan['rounds']=args.rounds
    (OUT/'route.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if args.plan_only:
        print(json.dumps(dict(ropes=len(plan['steps']),hanging=[s['ladder']['id'] for s in plan['steps'] if not s['traversal']])))
        return
    _REPORT_OUTPUT=OUT
    _OPEN_REPORT=not args.no_open_report
    cv2.setNumThreads(4)
    for index in range(args.resume_round or 1,args.rounds+1):
        folder=OUT/f'round_{index:02d}'
        completed=[]
        original_start=None
        if index==args.resume_round:
            parts=sorted(OUT.glob(f'round_{index:02d}*'))
            for part in parts:
                events=[json.loads(s) for s in (part/'events.jsonl').read_text(encoding='utf-8').splitlines()]
                completed.extend(r['ladder'] for r in events if r['event']=='coverage_target_result' and r['success'])
                starts=[r.get('original_start',r['time']) for r in events if r['event']=='tour_start']
                if starts:
                    original_start=min(starts+[original_start] if original_start is not None else starts)
            completed=sorted(set(completed))
            folder=OUT/f'round_{index:02d}_part{len(parts)+1:02d}'
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f'Refusing to overwrite {folder}')
        folder.mkdir(parents=True,exist_ok=True)
        files=('coverage_trial.py','navigation.py','localizer.py','async_localizer.py','main_adapters.py',
               'jump_policy.py','diagnostic_recorder.py','visual_motion.py',
               'movement_control.py','calibration.py','ignore_regions.py')
        write_json(folder/'manifest.json',dict(backend=args.backend,round=index,
            source_sha256={**{f:hashlib.sha256((ROOT/'no_minimap_lab'/f).read_bytes()).hexdigest() for f in files},
                           'src/vision/world_filter.py':hashlib.sha256((ROOT/'src/vision/world_filter.py').read_bytes()).hexdigest()},
            ignore_regions=load_ignore_regions(),mandatory_minimap_mask=False,teleport_allowed=False,portal_allowed=False))
        observer=Observer(args.backend,folder)
        nav=CoverageNavigator(observer,folder,2400)
        try:
            write_json(OUT/'status.json',dict(round=index,status='preparing',pid=__import__('os').getpid()))
            if not completed:
                prep=nav.run(1)
                if not prep.get('success') or not nav.walk(21):
                    raise RuntimeError(f'Cannot prepare round {index}: {prep}')
            else:
                nav.wait(2)
            nav.wait(.8)
            # Preparation remains in raw evidence but is excluded by tour_start.
            nav.covered.clear()
            write_json(OUT/'status.json',dict(round=index,status='running',pid=__import__('os').getpid()))
            result=nav.tour(plan,completed,original_start)
            result.update(backend=args.backend,yellow_dot_used=False,map_id=101000000)
            write_json(folder/'result.json',result)
            if not result['success']:
                raise RuntimeError(str(result))
        finally:
            nav.keys.set()
            nav.events.close()
            observer.close()
    write_json(OUT/'status.json',dict(status='completed',rounds=args.rounds))


def main():
    try:
        _main()
    finally:
        if _REPORT_OUTPUT is not None and any(_REPORT_OUTPUT.glob('round_*/events.jsonl')):
            try:
                from .results_page import publish
                publish(_REPORT_OUTPUT,_OPEN_REPORT)
            except Exception as exc:
                print(f'结果生成失败（原始记录已保留）：{exc}\n记录目录：{_REPORT_OUTPUT}',flush=True)


if __name__=='__main__':
    main()
