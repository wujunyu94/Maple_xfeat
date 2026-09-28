"""Compact, tolerant read-only status while JSON files are being written."""
import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parent/'output/coverage_five'


def read_rows(path):
    rows=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    state=read_rows(ROOT/'status.json')
    if not state:
        print('status write in progress')
        return
    state=state[-1]
    print('suite',state,'process_alive',psutil.pid_exists(state.get('pid',0)))
    for folder in sorted(ROOT.glob(f"round_{state.get('round',5):02d}*")):
        events=read_rows(folder/'events.jsonl')
        obs=read_rows(folder/'observations.jsonl')
        targets=[r['ladder'] for r in events if r['event']=='coverage_target_result' and r['success']]
        current=next((r['ladder'] for r in reversed(events) if r['event']=='coverage_target'),None)
        print(folder.name,'covered',targets,'current_target',current,'event',events[-1]['event'] if events else None)
        if obs:
            o=obs[-1]
            print('observation',{k:o[k] for k in ('time','status','world','platform')})


if __name__=='__main__':
    main()
