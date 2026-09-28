"""Sequential real-game evaluations; no simulated scores or concurrent inputs."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"no_minimap_lab/output/final_suite"


def save(data):
    (OUT/"suite.json").write_text(json.dumps(data,indent=2),encoding="utf-8")


def run(backend,folder,target=87,position=None,budget=600):
    folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","no_minimap_lab.navigation","--backend",backend,
             "--target",str(target),"--budget",str(budget),"--output",str(folder)]
    if position is not None:
        command += ["--position",str(position)]
    files=("navigation.py","localizer.py","async_localizer.py","xfeat_backend.py","main_adapters.py")
    manifest=dict(backend=backend,map_id=101000000,target=target,position=position,
        minimap_pixels_zeroed_before_detection=True,teleport_allowed=False,portals_allowed=False,
        source_sha256={f:hashlib.sha256((ROOT/"no_minimap_lab"/f).read_bytes()).hexdigest() for f in files})
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    with (folder/"console.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save(dict(status="running",stage=folder.name,backend=backend,pid=process.pid,
                  parent_pid=__import__('os').getpid(),started=time.time()))
        print(f"START {folder.name} PID={process.pid}",flush=True)
        code=process.wait(timeout=budget+60)
    if code:
        raise RuntimeError(f"{folder.name} exited {code}; inspect console.log")
    result=json.loads((folder/"result.json").read_text())
    print(f"END {folder.name}: {json.dumps(result)}",flush=True)
    if not result.get("success") or (position is not None and not result.get("positioned")):
        raise RuntimeError(f"{folder.name} did not finish")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        for index,backend in enumerate(("opencv-v1","xfeat","sift")):
            from .audit_navigation import audit
            if args.resume and (OUT/backend/"result.json").exists() and audit(OUT/backend)["qualifies"]:
                print(f"VERIFIED existing {backend}",flush=True)
                continue
            if index:
                suffix=f"_{int(time.time())}" if args.resume else ""
                run("xfeat",OUT/f"prepare_{backend}{suffix}",target=1,position=21,budget=200)
            run(backend,OUT/backend)
            verified=audit(OUT/backend)
            if not verified["qualifies"]:
                raise RuntimeError(f"{backend}: endpoint audit did not pass")
        save(dict(status="completed",finished=time.time()))
    except Exception as exc:
        save(dict(status="needs_attention",error=repr(exc),time=time.time()))
        raise


if __name__ == "__main__":
    main()
