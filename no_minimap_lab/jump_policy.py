"""Pure geometry and control rules for diagnostic platform jumps."""
from statistics import median
import math


def covered_by_source(graph, edge):
    """A descent onto overlapping lower footholds lands on the source first."""
    x=edge.landing_x
    if x is None:
        return False
    source,target=graph.nodes[edge.from_id],graph.nodes[edge.to_id]
    return (source.x_min<=x<=source.x_max and
            source.surface_y_at(x)<target.surface_y_at(x)-9)


def ground_speed(history, source, since, now):
    samples=[r for r in history if max(since,now-.20)<=r['time']<=now]
    if len(samples)<3 or samples[-1]['time']-samples[0]['time']<.10:
        return None
    if any(r['platform']!=source or not r['world'] for r in samples):
        return None
    speeds=[]
    for a,b in zip(samples,samples[1:]):
        dt=b['time']-a['time']
        if dt>0:
            v=(b['world'][0]-a['world'][0])/dt
            if abs(v)>350:  # Discontinuous localization cannot authorize takeoff.
                return None
            speeds.append(v)
    return median(speeds) if speeds else None


def airborne_keys(direction, x, landing, contacted):
    """Never reverse to chase a point, especially after target contact."""
    if contacted or direction is None:
        return ()
    sign=1 if direction=='right' else -1
    return (direction,) if sign*(landing-x)>5 else ()


def surface(node,x):
    if not node.x_min<=x<=node.x_max:
        return None
    lines=getattr(node,'raw_lines',[])
    if lines and not any(min(l['x1'],l['x2'])<=x<=max(l['x1'],l['x2']) and l['x1']!=l['x2'] for l in lines):
        return None
    return node.surface_y_at(x)


def safe_drop_intervals(graph,edge,margin=12):
    a,b=graph.nodes[edge.from_id],graph.nodes[edge.to_id]
    bounds=edge.trigger_x_range or (max(a.x_min,b.x_min),min(a.x_max,b.x_max))
    lo=max(bounds[0],a.x_min+margin,b.x_min+margin)
    hi=min(bounds[1],a.x_max-margin,b.x_max-margin)
    good=[]
    for x in range(math.ceil(lo),math.floor(hi)+1):
        ay,by=surface(a,x),surface(b,x)
        if ay is None or by is None or by<=ay+15:continue
        if any(abs(x-l.x)<24 and l.y1<=by+30 and l.y2>=ay-30 for l in graph.ladder_ropes.values()):continue
        if any(n.id not in (a.id,b.id) and (ny:=surface(n,x)) is not None and ay+10<ny<by-9 for n in graph.nodes.values()):continue
        good.append(x)
    intervals=[]
    for x in good:
        if intervals and x==intervals[-1][1]+1:intervals[-1][1]=x
        else:intervals.append([x,x])
    return [(lo,hi) for lo,hi in intervals if hi-lo>=6]


def safe_vertical_interval(graph,edge,margin=14):
    a,b=graph.nodes[edge.from_id],graph.nodes[edge.to_id]
    bounds=edge.takeoff_x_range or edge.trigger_x_range
    if not bounds:return None
    landing=edge.landing_x_range or bounds
    lo=max(bounds[0],landing[0],a.x_min+4,b.x_min+margin)
    hi=min(bounds[1],landing[1],a.x_max-4,b.x_max-margin)
    # Do not replace a physically narrow landing with repeated tiny alignment.
    if hi-lo<6:return None
    if any(surface(a,x) is None or surface(b,x) is None for x in (lo,(lo+hi)/2,hi)):return None
    return lo,hi


def launch_projection(edge,x,speed,age,input_delay=.025):
    sign=1 if edge.action=='JUMP_RIGHT' else -1
    nominal=edge.takeoff_x if edge.takeoff_x is not None else edge.trigger_x
    lo,hi=edge.takeoff_x_range or edge.trigger_x_range or (nominal-6,nominal+6)
    projected=x+speed*(max(0,age)+input_delay)
    # Use the early side of the physically allowed window; never its old ±6 extension.
    state='late' if (projected>hi if sign==1 else projected<lo) else 'ready' if lo<=projected<=hi else 'approach'
    return state,projected,(lo,hi)
