"""Recompute results from recorded observations and action events."""
import argparse
import json
from pathlib import Path
import numpy as np
from .atlas import ROOT
from src.vision.wz_map_reader import WzMapReader
from src.engine.platform_graph import PlatformGraphBuilder


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(folder):
    folder=Path(folder)
    result=json.loads((folder / "result.json").read_text(encoding="utf-8"))
    observations=rows(folder / "observations.jsonl")
    events=rows(folder / "events.jsonl")
    graph=PlatformGraphBuilder.build_from_map_dict(WzMapReader(str(ROOT / "Map")).load_map(101000000),
                                                  enable_teleport=False)
    start=next((e for e in events if e["event"]=="start"),None)
    verified=[]
    if start:
        goal=graph.nodes[87]
        for r in observations:
            world=r.get("world")
            if world and r["time"]>=start["time"]:
                x,y=world
                if goal.x_min-3<=x<=goal.x_max+3 and abs(goal.surface_y_at(x)-y)<=9:
                    verified.append(r)
                else:
                    verified=[]
            else:
                verified=[]
    endpoint_ok=len(verified)>=3 and verified[-1]["time"]-verified[0]["time"]>=.35
    start_ok=bool(start and start["source"]==1 and abs(start["world"][0]-21)<=8 and
                  abs(graph.nodes[1].surface_y_at(start["world"][0])-start["world"][1])<=9)
    attempts=[e for e in events if e["event"]=="grab_attempt"]
    confirmed=[e for e in events if e["event"]=="grab_confirmed" and e.get("jump",True)]
    rope_attempts=[e for e in attempts if not graph.ladder_ropes[e["ladder"]].is_ladder]
    rope_confirmed=[e for e in confirmed if not graph.ladder_ropes[e["ladder"]].is_ladder]
    forbidden=[e for e in events if e["event"]=="edge_start" and
               (e["edge"]["action"]=="PORTAL" or "TELEPORT" in e["edge"]["action"])]
    jumps=[]
    for a,b in zip(observations,observations[1:]):
        if a["world"] and b["world"]:
            dt=b["time"]-a["time"]
            dx,dy=np.asarray(b["world"])-a["world"]
            if 0<dt<=.3 and (abs(dx)>60+300*dt or abs(dy)>90+700*dt):
                jumps.append(dict(time=b["time"],dx=float(dx),dy=float(dy)))
    report=dict(backend=result["backend"],reported_success=result.get("success",False),
        verified_start=start_ok,verified_p87=endpoint_ok,forbidden_edges=len(forbidden),
        seconds=result.get("seconds"),jump_grabs=dict(attempts=len(attempts),confirmed=len(confirmed)),
        rope_grabs=dict(attempts=len(rope_attempts),confirmed=len(rope_confirmed)),
        observation_count=len(observations),
        abrupt_coordinate_changes=jumps,
        valid_world_fraction=sum(r["world"] is not None for r in observations)/max(1,len(observations)),
        latency_ms_p50=float(np.median([r["latency_ms"] for r in observations])) if observations else None)
    report["qualifies"]=bool(result.get("success") and start_ok and endpoint_ok and not forbidden)
    (folder / "audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folders",nargs="+")
    args=p.parse_args()
    print(json.dumps([audit(f) for f in args.folders],indent=2))
