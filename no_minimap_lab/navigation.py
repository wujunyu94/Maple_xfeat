"""Live, minimap-free P1->P87 evaluation with a restricted input service."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from .atlas import load_atlas
from .async_localizer import AsyncLocalizer
from .localizer import Localizer
from .localizer import scene_mask
from .ignore_regions import load as load_ignore_regions
from .main_adapters import MainPlayerAnchor
from .run import live_capture
from .input_service import STATE, write_json
from .diagnostic_recorder import DiagnosticRecorder
from src.vision.wz_map_reader import WzMapReader
from src.engine.platform_graph import PlatformGraphBuilder


class Keys:
    def __init__(self):
        self.token = uuid.uuid4().hex
        self.n = 0

    def set(self, *keys):
        if not set(keys) <= {"left", "right", "up", "down", "alt", "tab"}:
            raise ValueError("Movement-only input")
        self.n += 1
        # Service rejects partial JSON; direct overwrite avoids Windows sharing
        # violations when its polling reader briefly holds the destination open.
        (STATE / "command.json").write_text(json.dumps(dict(sequence=f"{self.token}:{self.n}",
                   time=time.time(), ttl=.25, keys=list(keys))),encoding="utf-8")

    def pulse(self, keys, seconds):
        end = time.perf_counter()+seconds
        while time.perf_counter() < end:
            self.set(*keys)
            time.sleep(.025)
        self.set()


class Observer:
    def __init__(self, backend, output):
        self.output = output
        self.atlas = load_atlas(101000000)
        cls = Localizer if backend == "opencv-v1" else AsyncLocalizer
        self.model = cls(self.atlas, backend="xfeat-cuda" if backend == "xfeat" else "sift-cpu")
        self.anchor = MainPlayerAnchor(feature_rescue=True)
        self.graph = PlatformGraphBuilder.build_from_map_dict(WzMapReader("Map").load_map(101000000),
                                                              enable_teleport=False)
        # Remove all nonphysical shortcuts even if a future planner forgets its flag.
        for node in self.graph.edges:
            self.graph.edges[node] = [e for e in self.graph.edges[node]
                                     if "TELEPORT" not in e.action and e.action != "PORTAL"]
        self.capture = None
        self.stop = threading.Event()
        self.latest = None
        self.frame = None
        self.history = deque(maxlen=120)
        self.error = None
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        recorder = DiagnosticRecorder(self.output)
        log = (self.output / "observations.jsonl").open("w", encoding="utf-8", buffering=1)
        try:
            self.capture = live_capture()
            regions = load_ignore_regions()
            while not self.stop.is_set():
                tick = time.perf_counter()
                frame = self.capture.capture_frame()
                if frame is None:
                    self.latest = None
                    self.stop.wait(.03)
                    continue
                current_regions = load_ignore_regions()
                if current_regions != regions:
                    self.model.reset()
                    self.anchor.reset()
                    regions = current_regions
                # Apply only configured exclusions; preserve raw replay evidence.
                vision_frame = frame.copy()
                vision_frame[scene_mask(frame.shape) == 0] = 0
                pose = self.model.update(vision_frame, tick)
                player = self.anchor.detect(vision_frame)
                world = None
                platform = None
                if pose.camera is not None and player:
                    world = np.array(pose.camera)+player["screen"]
                    candidates = [(abs(n.surface_y_at(world[0])-world[1]), n.id)
                                  for n in self.graph.nodes.values()
                                  if n.x_min-3 <= world[0] <= n.x_max+3]
                    if candidates:
                        dist, ident = min(candidates)
                        if dist <= 9:
                            platform = ident
                packet = dict(time=tick, observed_at=time.perf_counter(), camera=pose.camera,
                    status=pose.status, world=world.tolist() if world is not None else None,
                    platform=platform, player=player, latency_ms=(time.perf_counter()-tick)*1000)
                self.latest = packet
                self.frame = frame
                self.history.append(packet)
                log.write(json.dumps(packet)+"\n")
                recorder.record(frame, packet)
                self.stop.wait(max(0,.025-(time.perf_counter()-tick)))
        except Exception as exc:
            self.error = repr(exc)
        finally:
            log.close()
            recorder.close()
            if self.capture:
                self.capture.release()

    def get(self):
        value = self.latest
        if value and value["world"] and time.perf_counter()-value["time"] < .3:
            return value
        return None

    def stable(self, ident, duration=.20):
        now = time.perf_counter()
        samples = [r for r in list(self.history) if now-r["time"] <= duration+.35]
        return len(samples)>=3 and now-samples[-1]["time"]<.3 and samples[-1]["time"]-samples[0]["time"]>=duration and all(
            r["platform"] == ident for r in samples)

    def close(self):
        self.stop.set()
        self.thread.join(5)
        self.anchor.close()
        if isinstance(self.model, AsyncLocalizer):
            self.model.close()


class Navigator:
    def __init__(self, observer, output, budget):
        self.o = observer
        self.g = observer.graph
        self.keys = Keys()
        self.output = output
        self.deadline = time.perf_counter()+budget
        self.events = (output / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self.attempts = 0
        self.catches = 0
        self.penalties = {}

    def event(self, event, **data):
        row = dict(time=time.perf_counter(),event=event,**data)
        self.events.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(json.dumps(row,ensure_ascii=True),flush=True)
        write_json(self.output / "progress.json", row)

    def alive(self):
        if self.o.error:
            raise RuntimeError(self.o.error)
        return time.perf_counter()<self.deadline and not (STATE / "stop_navigation").exists()

    def wait(self, seconds, keys=()):
        end = time.perf_counter()+seconds
        while self.alive() and time.perf_counter()<end:
            self.keys.set(*(keys if self.o.get() else ()))
            time.sleep(.025)
        self.keys.set()

    def walk(self, x, timeout=25, tolerance=5):
        from .visual_motion import VisualMotion, MotionKeys
        from .movement_control import walk_intervals
        if not hasattr(self, "motion"):
            self.motion = VisualMotion()
            self.keys = MotionKeys(self.keys, self.motion, time.perf_counter)
        return walk_intervals(self, [(x-tolerance, x+tolerance)], timeout,
                              tolerance <= 5, None, time)

    def execute(self, edge):
        self.event("edge_start", edge=asdict(edge))
        action = edge.action
        x = edge.takeoff_x if edge.takeoff_x is not None else edge.trigger_x
        if x is None:
            x=self.g.nodes[edge.to_id].center_x
        if not self.walk(x):
            return False
        self.wait(.15)
        if "CLIMB" in action:
            ladder=self.g.ladder_ropes[edge.ladder_id]
            before=self.o.get()
            if before is None:
                return False
            # Avoid pressing UP near portal interaction hitboxes.
            if any(abs(float(p.get("x",1e9))-x)<28 and
                   abs(float(p.get("y",1e9))-before["world"][1])<70 for p in self.g.portals):
                self.event("portal_proximity_rejected", ladder=edge.ladder_id)
                return False
            is_jump="JUMP" in action
            if is_jump:
                self.attempts+=1
                self.wait(.09,("alt","up"))
                self.event("grab_attempt",number=self.attempts,ladder=edge.ladder_id)
            start=time.perf_counter()
            caught=False
            while self.alive() and time.perf_counter()-start<max(8,ladder.length/65+3):
                obs=self.o.get()
                if obs is None:
                    self.keys.set()
                    time.sleep(.025)
                    continue
                px,py=obs["world"]
                if self.o.stable(edge.to_id):
                    self.keys.set()
                    return True
                if not caught and abs(px-ladder.x)<7 and before["world"][1]-py>65 and time.perf_counter()-start>.65:
                    caught=True
                    if is_jump:
                        self.catches+=1
                    self.event("grab_confirmed",ladder=edge.ladder_id,jump=is_jump)
                self.keys.set("down" if "DOWN" in action else "up")
                time.sleep(.025)
            self.keys.set()
            return False
        if action == "DOWN_JUMP":
            self.wait(.12,("down",))
            self.wait(.12,("down","alt"))
            self.wait(.8)
        elif action.startswith("JUMP"):
            direction="left" if "LEFT" in action else "right" if "RIGHT" in action else None
            self.wait(.09,tuple(["alt"]+([direction] if direction else [])))
            start=time.perf_counter()
            landing=edge.landing_x if edge.landing_x is not None else x
            while self.alive() and time.perf_counter()-start<1.4:
                obs=self.o.get()
                if self.o.stable(edge.to_id):
                    self.keys.set()
                    return True
                if obs and direction and abs(landing-obs["world"][0])>5:
                    self.keys.set("right" if landing>obs["world"][0] else "left")
                else:
                    self.keys.set()
                time.sleep(.025)
        else:
            self.event("unsupported_action", action=action)
        self.keys.set()
        return self.o.stable(edge.to_id)

    def run(self,target):
        start=time.perf_counter()
        while self.alive():
            obs=self.o.get()
            if obs and obs["platform"] and self.o.stable(obs["platform"]):
                break
            self.wait(.1)
        obs=self.o.get()
        if not obs or not obs["platform"]:
            return dict(success=False,reason="No stable visual starting platform")
        source=obs["platform"]
        self.event("start",backend=self.o.model.backend,source=source,target=target,world=obs["world"])
        start=time.perf_counter()
        while self.alive():
            obs=self.o.get()
            if obs is None or obs["platform"] is None:
                self.wait(.15)
                continue
            if self.o.stable(target,.35):
                return dict(success=True,seconds=time.perf_counter()-start,source=source,target=target,
                            jump_grab_attempts=self.attempts,jump_grab_successes=self.catches)
            if obs["platform"] == target:
                self.wait(.1)
                continue
            key=lambda e:(e.from_id,e.to_id,e.action)
            route=self.g.find_path(obs["platform"],target,allow_portal=False,
                edge_penalty_fn=lambda e:self.penalties.get(key(e),0))
            if not route:
                return dict(success=False,reason="No allowed route",platform=obs["platform"])
            edge=route[0]
            ok=self.execute(edge)
            self.event("edge_end",success=ok,edge=[edge.from_id,edge.to_id,edge.action],observation=self.o.get())
            if not ok:
                self.penalties[key(edge)]=self.penalties.get(key(edge),0)+5
                self.wait(.35)
        return dict(success=False,reason="Time budget or external stop",seconds=time.perf_counter()-start,
                    jump_grab_attempts=self.attempts,jump_grab_successes=self.catches)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=("opencv-v1","sift","xfeat"),default="sift")
    p.add_argument("--target",type=int,default=87)
    p.add_argument("--budget",type=float,default=600)
    p.add_argument("--observe",action="store_true")
    p.add_argument("--position",type=float,help="After reaching target, walk to this X; preparation only")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4)
    observer=Observer(args.backend,output)
    nav=Navigator(observer,output,args.budget)
    try:
        if args.observe:
            nav.wait(args.budget)
            result=dict(observation=observer.get())
        else:
            result=nav.run(args.target)
            if result.get("success") and args.position is not None:
                result["positioned"] = nav.walk(args.position)
        result.update(backend=args.backend,map_id=101000000,yellow_dot_used=False,
                      teleport_allowed=False,portals_allowed=False)
        write_json(output / "result.json", result)
        print(json.dumps(result),flush=True)
    finally:
        nav.keys.set()
        nav.events.close()
        observer.close()


if __name__ == "__main__":
    main()
