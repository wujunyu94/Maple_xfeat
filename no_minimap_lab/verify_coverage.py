"""Cross-check coverage claims against action results and raw visual samples."""
import json
from pathlib import Path
from .coverage_trial import OUT, build_plan
from .coverage_status import read_rows


def verify():
    required={s['ladder']['id'] for s in build_plan()['steps']}
    rounds=json.loads((OUT/'rounds.json').read_text(encoding='utf-8'))
    report={}
    for name,r in rounds.items():
        issues=[]
        covered=set()
        successful_jumps=0
        successful_grabs=0
        for part in r['parts']:
            folder=OUT/part
            events=read_rows(folder/'events.jsonl')
            observations=read_rows(folder/'observations.jsonl')
            start=next((e['time'] for e in events if e['event']=='tour_start'),float('inf'))
            events=[e for e in events if e['time']>=start]
            attempts={e['attempt']:e for e in events if e['event']=='action_attempt'}
            good_results=[]
            for e in events:
                if e['event']=='action_result' and e['success']:
                    a=attempts.get(e['attempt'])
                    if not a:
                        issues.append(f'{part}: result without attempt {e["attempt"]}')
                        continue
                    samples=[o for o in observations if a['time']<=o['time']<=e['time']
                             and o.get('observed_at',o['time'])<=e['time']]
                    if a['kind']=='platform_jump':
                        successful_jumps+=1
                        target=a['edge']['to_id']
                        # Match the controller's .20 s stability span within its
                        # .55 s history window. Slow detections can be 120 ms apart;
                        # a frame captured before the decision but processed after
                        # it was not yet available to that decision.
                        tail=[o for o in samples if e['time']-.55<=o['time']]
                        if (len(tail)<3 or tail[-1]['time']-tail[0]['time']<.20
                                or not all(o['world'] and o['platform']==target for o in tail)):
                            issues.append(f'{part}: jump {e["attempt"]} lacks stable target samples')
                    else:
                        successful_grabs+=1
                        lid=a['ladder']
                        verified=any(v['event']=='grab_verified' and v.get('attempt')==e['attempt'] for v in events)
                        if a.get('hanging'):
                            descent=e.get('descent_samples',[])
                            if len(descent)<3 or max(o['world'][1] for o in descent)<e.get('excursion_depth',float('inf')):
                                issues.append(f'{part}: hanging rope {lid} lacks descent evidence')
                        elif not verified and e.get('evidence_count',0)<3:
                            issues.append(f'{part}: grab {e["attempt"]} lacks rope-axis evidence')
                        if e.get('reached_target'):
                            good_results.append((lid,e['time']))
                elif e['event']=='coverage_target_result' and e['success']:
                    if not any(lid==e['ladder'] and t<=e['time'] for lid,t in good_results):
                        issues.append(f'{part}: target {e["ladder"]} not backed by a completed grab')
                    else:
                        covered.add(e['ladder'])
            forbidden=[e for e in events if e['event']=='edge_start' and
                       ('TELEPORT' in e['edge']['action'] or e['edge']['action']=='PORTAL')]
            if forbidden:
                issues.append(f'{part}: forbidden traversal')
        if successful_jumps!=sum(v['successes'] for v in r['jumps'].values()):
            issues.append('jump count mismatch')
        if successful_grabs!=sum(v['successes'] for v in r['grabs'].values()):
            issues.append('grab count mismatch')
        report[name]=dict(complete=bool(r['complete'] and covered==required and not issues),
                          observed_coverage=sorted(covered),issues=issues,
                          successful_grabs=successful_grabs,successful_jumps=successful_jumps)
    (OUT/'evidence_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    verify()
