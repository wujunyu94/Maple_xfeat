"""Generate a local result page with playable failure clips and open it."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
import webbrowser

from .failure_clips import rows


def render(output):
    data=json.loads((output/'rounds.json').read_text(encoding='utf-8'))
    h=html.escape
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>无黄点定位测试结果</title>',
           '<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182433}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#075fc5}video{width:100%;max-width:800px;background:#111}article{background:white;border:1px solid #ddd;padding:20px;margin:20px 0;border-radius:10px}small{color:#556}summary{cursor:pointer;padding:12px}code{overflow-wrap:anywhere}</style>',
           '<h1>无黄点定位测试结果</h1>',f'<p>记录目录：<code>{h(str(output))}</code></p>',
           '<p><a href="RESULTS.md">完整结果文档</a> · <a href="failure_clips/INDEX.md">失败录像索引</a> · <a href="rope_rates.csv">绳梯 CSV</a> · <a href="platform_jump_rates.csv">跳跃 CSV</a></p>',
           '<table><tr><th>轮次</th><th>完成复核</th><th>覆盖绳梯</th><th>耗时</th><th>抓取成功/发起</th><th>跳跃成功/发起</th></tr>']
    for name,r in data.items():
        counts=[f"{sum(s['successes'] for s in r[k].values())}/{sum(s['attempts'] for s in r[k].values())}" for k in ('grabs','jumps')]
        duration=f"{r['seconds']:.1f} 秒" if r['seconds'] is not None else '未完成'
        parts.append(f"<tr><td>{h(name)}</td><td>{'通过' if r['complete'] else '未完成或未通过'}</td><td>{len(r['covered'])}/37</td><td>{duration}</td><td>{counts[0]}</td><td>{counts[1]}</td></tr>")
    parts+=['</table><p><small>未判定记录保留在发起次数中；CSV 另有已判定口径。录像是诊断采样，播放时长不等于实际动作耗时。起跳前拒绝见索引，不计作已起跳失败。</small></p>','<h2>失败动作录像</h2>']
    failures=0
    for folder in sorted(output.glob('round_*')):
        events=rows(folder/'events.jsonl')
        start=next((e['time'] for e in events if e['event']=='tour_start'),float('inf'))
        attempts={e['attempt']:e for e in events if e['event']=='action_attempt'}
        for e in events:
            if e['event']!='action_result' or e['success'] or e['time']<start:continue
            a=attempts[e['attempt']];edge=a.get('edge',{})
            label=f"绳梯 #{a['ladder']}" if a['kind']=='grab' else f"P{edge['from_id']} → P{edge['to_id']} {edge['action']}"
            clip=f"failure_clips/{folder.name}_action_{a['attempt']:04d}.mp4"
            parts.append(f'<article><h3>{h(folder.name)} · 动作 {a["attempt"]} · {h(label)}</h3>')
            if (output/clip).exists():
                parts.append(f'<video controls preload="none" src="{h(clip)}"></video><p><a href="{h(clip)}">单独打开短片</a></p>')
            else:parts.append('<p>短片未生成，请查看原录像及失败索引。</p>')
            parts.append('</article>');failures+=1
    if not failures:parts.append('<p>没有已判定失败动作。请结合上表确认测试是否已完成。</p>')
    for kind,title in [('grabs','逐绳梯统计'),('jumps','逐平台连接统计')]:
        parts.append(f'<details><summary>{title}</summary><table><tr><th>轮次</th><th>目标</th><th>成功/发起</th><th>未判定</th></tr>')
        for name,r in data.items():
            for target,s in r[kind].items():
                parts.append(f"<tr><td>{h(name)}</td><td>{h(target)}</td><td>{s['successes']}/{s['attempts']}</td><td>{s['incomplete']}</td></tr>")
        parts.append('</table></details>')
    parts.append('</html>')
    page=output/'RESULTS.html';page.write_text('\n'.join(parts),encoding='utf-8')
    return page


def publish(output,open_browser=True):
    output=Path(output).resolve()
    root=Path(__file__).resolve().parents[1]
    print(f'正在生成统计与失败短片：{output}',flush=True)
    for module in ('coverage_report','failure_clips'):
        with (output/f'{module}_generation.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m',f'no_minimap_lab.{module}','--output',str(output)],
                           cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    page=render(output)
    print(f'结果页面：{page}',flush=True)
    if open_browser and not webbrowser.open(page.as_uri()):
        print('未能自动打开浏览器，请双击上述 RESULTS.html。',flush=True)
    return page


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-open',action='store_true');args=parser.parse_args()
    publish(args.output,not args.no_open)
