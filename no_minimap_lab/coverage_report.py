"""Read-only action audit; retain all failed and interrupted attempts."""
import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .coverage_trial import OUT


def audit(folder):
    rows=[]
    for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    start=next((r['time'] for r in rows if r['event']=='tour_start'),None)
    if start is None:
        return dict(started=False)
    rows=[r for r in rows if r['time']>=start]
    results={r['attempt']:r for r in rows if r['event']=='action_result'}
    grabs=defaultdict(lambda:dict(attempts=0,successes=0,incomplete=0,response_seconds=[]))
    jumps=defaultdict(lambda:dict(attempts=0,successes=0,incomplete=0,response_seconds=[]))
    for row in rows:
        if row['event']!='action_attempt':
            continue
        result=results.get(row['attempt'])
        if row['kind']=='grab':
            entry=grabs[str(row['ladder'])]
        else:
            e=row['edge']
            entry=jumps[f"P{e['from_id']}->P{e['to_id']} {e['action']}"]
        entry['attempts']+=1
        entry['successes']+=int(bool(result and result['success']))
        entry['incomplete']+=int(result is None)
        if result:
            entry['response_seconds'].append(result['time']-row['time'])
    targets=[r for r in rows if r['event']=='coverage_target_result' and r['success']]
    covered=sorted(set(r['ladder'] for r in targets))
    end=next((r for r in reversed(rows) if r['event']=='tour_end'),None)
    observations=[]
    for line in (folder/'observations.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            obs=json.loads(line)
        except json.JSONDecodeError:  # Live writer may be in the middle of a line.
            continue
        if obs['time']>=start and (not end or obs['time']<=end['time']):
            observations.append(obs)
    timing={}
    if observations:
        timing=dict(frames=len(observations),world_missing=sum(o['world'] is None for o in observations),
                    latency_ms_p50_p95=np.percentile([o['latency_ms'] for o in observations],[50,95]).tolist())
    return dict(started=True,complete=bool(end and end['success'] and covered==list(range(1,38))),
                covered=covered,seconds=end['seconds'] if end else None,grabs=dict(grabs),jumps=dict(jumps),timing=timing)


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT,help='Existing test result directory')
    args=parser.parse_args()
    OUT=args.output.resolve()
    reports={p.name:audit(p) for p in sorted(OUT.glob('round_*')) if (p/'events.jsonl').exists()}
    (OUT/'metrics.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    rounds={}
    for name,report in reports.items():
        if not report.get('started'):
            continue
        key=name[:8]
        merged=rounds.setdefault(key,dict(covered=[],grabs={},jumps={},parts=[],seconds=None))
        merged['parts'].append(name)
        merged['covered']=sorted(set(merged['covered']+report['covered']))
        if report['seconds'] is not None:
            merged['seconds']=report['seconds']
        for kind in ('grabs','jumps'):
            for target,stats in report[kind].items():
                total=merged[kind].setdefault(target,dict(attempts=0,successes=0,incomplete=0,response_seconds=[]))
                for field in ('attempts','successes','incomplete'):
                    total[field]+=stats[field]
                total['response_seconds']+=stats['response_seconds']
    for key,r in rounds.items():
        last=OUT/r['parts'][-1]
        result=json.loads((last/'result.json').read_text(encoding='utf-8')) if (last/'result.json').exists() else {}
        obs=[json.loads(s) for s in (last/'observations.jsonl').read_text(encoding='utf-8').splitlines() if s.endswith('}')]
        endtime=obs[-1]['time'] if obs else 0
        tail=[o for o in obs if endtime-o['time']<=.5]
        terminal_ok=bool(len(tail)>=3 and tail[-1]['time']-tail[0]['time']>=.35 and all(o['platform']==87 and o['world'] for o in tail))
        first=OUT/r['parts'][0]
        events=[json.loads(s) for s in (first/'events.jsonl').read_text(encoding='utf-8').splitlines() if s.endswith('}')]
        begin=next((e for e in events if e['event']=='tour_start'),None)
        observations=[json.loads(s) for s in (first/'observations.jsonl').read_text(encoding='utf-8').splitlines() if s.endswith('}')]
        initial=[o for o in observations if begin and begin['time']-.5<=o['time']<=begin['time']]
        start_ok=bool(len(initial)>=3 and all(o['platform']==1 and o['world'] and abs(o['world'][0]-21)<=8 for o in initial))
        r.update(start_verified=start_ok,finish_verified=terminal_ok,
                 complete=bool(result.get('success') and r['covered']==list(range(1,38)) and terminal_ok and start_ok))
    (OUT/'rounds.json').write_text(json.dumps(rounds,indent=2),encoding='utf-8')
    for kind,filename in [('grabs','rope_rates.csv'),('jumps','platform_jump_rates.csv')]:
        targets=sorted(set(k for r in rounds.values() for k in r[kind]))
        if kind=='grabs':
            targets=[str(i) for i in range(1,38)]
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['target','round','successes','attempt_records','conservative_success_rate','incomplete',
                             'mean_action_seconds','resolved_success_rate'])
            for target in targets:
                total_s=total_n=total_i=0
                for name,r in rounds.items():
                    s=r[kind].get(target,dict(successes=0,attempts=0,incomplete=0,response_seconds=[]))
                    times=s['response_seconds']
                    writer.writerow([target,name,s['successes'],s['attempts'],
                                     s['successes']/s['attempts'] if s['attempts'] else '',s['incomplete'],
                                     sum(times)/len(times) if times else '',
                                     s['successes']/(s['attempts']-s['incomplete']) if s['attempts']>s['incomplete'] else ''])
                    total_s+=s['successes'];total_n+=s['attempts'];total_i+=s['incomplete']
                writer.writerow([target,'TOTAL',total_s,total_n,total_s/total_n if total_n else '',total_i,'',
                                 total_s/(total_n-total_i) if total_n>total_i else ''])
    legacy=OUT.name=='coverage_five'
    text=[f'# 全绳梯实测记录（{len(rounds)} 轮）','',
          '以下结果来自原始动作事件，所有失败及中断记录均保留。准备回到 P1 的动作不计入正式成绩。', '',
          '表中的成功率是成功数÷全部发起记录数，未完成判定按未成功计入，属于保守口径；CSV 另列 resolved_success_rate（成功数÷已有结果的动作数）。未完成不等于已证实抓取失败。', '',
          '动作耗时是发起至结果的时间，不是键盘输入到画面响应延迟；逐帧处理延迟另在 metrics.json 的 timing 中保存。', '',
          '| 轮次 | 全部 37 条覆盖并到 P87 | 已完成指定绳梯 | 总耗时（秒） |', '|---|---|---:|---:|']
    for name,r in rounds.items():
        text.append(f"| {name} | {r['complete']} | {len(r['covered'])}/37 | {r['seconds']} |")
    text+=['','| 轮次 | 抓取成功 / 发起 | 平台跳跃成功 / 发起 |','|---|---:|---:|']
    for name,r in rounds.items():
        counts=[f"{sum(s['successes'] for s in r[k].values())}/{sum(s['attempts'] for s in r[k].values())}" for k in ('grabs','jumps')]
        text.append(f'| {name} | {counts[0]} | {counts[1]} |')
    text+=['','[失败录像与时间索引](failure_clips/INDEX.md)','']
    if legacy:
        text+=['历史五轮中，第一轮包含控制器修复及后端切换，第五轮中断续跑；第五轮 14 号有一条未判定记录。','']
    for kind,title in [('grabs','每条绳梯抓取'),('jumps','平台到平台跳跃')]:
        text+=['',f'## {title}','', '| 目标 | 轮次 | 成功 / 尝试 | 成功率 | 未完成判定 |', '|---|---|---:|---:|---:|']
        for name,r in rounds.items():
            for key,s in r.get(kind,{}).items():
                text.append(f"| {key} | {name} | {s['successes']}/{s['attempts']} | {s['successes']/s['attempts']:.1%} | {s['incomplete']} |")
    (OUT/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    chart=Image.new('RGB',(850,1190),'white')
    draw=ImageDraw.Draw(chart)
    font=ImageFont.load_default()
    titlefont=ImageFont.load_default()
    draw.text((25,15),'All-rope coverage: successful grabs / attempts',font=titlefont,fill='#172337')
    draw.text((25,47),'Failures and interrupted attempts retained; blank = not tested yet',font=font,fill='#555555')
    labels=[f'round_{i:02d}' for i in range(1,6)]
    for i,label in enumerate(labels):
        draw.text((145+i*140,85),f'Round {i+1}',font=font,fill='#172337')
    for lid in range(1,38):
        y=116+(lid-1)*27
        draw.text((30,y+3),f'#{lid}',font=font,fill='#172337')
        for i,label in enumerate(labels):
            stat=rounds.get(label,{}).get('grabs',{}).get(str(lid))
            x=135+i*140
            if stat and stat['attempts']:
                rate=stat['successes']/stat['attempts']
                color='#ccebd8' if rate==1 else '#fff0b5' if rate>=.75 else '#f8c4be'
                content=f"{stat['successes']}/{stat['attempts']}"
                if stat['incomplete']:
                    color='#d9e5f5';content+='*'
            else:
                color='#eeeeee';content='-'
            draw.rectangle((x,y,x+127,y+24),fill=color)
            draw.text((x+45,y+2),content,font=font,fill='#172337')
    draw.text((25,1130),'Capture timing and action results are retained in JSONL logs.',font=font,fill='#555555')
    draw.text((25,1155),'* Includes an interrupted, unconfirmed record, not a proven failure.',font=font,fill='#555555')
    chart.save(OUT/'rope_success_rates.png')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
