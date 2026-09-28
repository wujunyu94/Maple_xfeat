"""Shared editable screen exclusions, normalized to capture dimensions."""
import json
import math
import os
from pathlib import Path
import threading

PATH = Path(__file__).resolve().parent/'calibration/ignore_regions.json'
# No mandatory minimap exclusion. Defaults are editable too.
DEFAULTS = ((0.,0.,1.,.08),(0.,.86,1.,.14))
_lock = threading.RLock()
_key = None
_regions = DEFAULTS


def validate(regions):
    result=[]
    for row in regions:
        if len(row)!=4:
            raise ValueError('忽略区域需要 x、y、宽、高四个值')
        x,y,w,h=map(float,row)
        if not all(math.isfinite(v) for v in (x,y,w,h)) or x<0 or y<0 or w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001:
            raise ValueError('忽略区域超出画面范围')
        result.append((x,y,w,h))
    return tuple(result)


def load():
    global _key, _regions
    with _lock:
        try:
            stat=PATH.stat();key=(str(PATH),stat.st_mtime_ns,stat.st_size)
        except FileNotFoundError:
            key=(str(PATH),None)
        if key!=_key:
            _regions=validate(json.loads(PATH.read_text(encoding='utf-8'))['regions']) if key[1] is not None else DEFAULTS
            _key=key
        return _regions


def save(regions):
    regions=validate(regions)
    PATH.parent.mkdir(parents=True,exist_ok=True)
    temp=PATH.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(version=1,regions=regions),indent=2),encoding='utf-8')
    os.replace(temp,PATH)


def rectangles(shape):
    h,w=shape[:2]
    return [(round(x*w),round(y*h),round((x+rw)*w)-round(x*w),round((y+rh)*h)-round(y*h)) for x,y,rw,rh in load()]
