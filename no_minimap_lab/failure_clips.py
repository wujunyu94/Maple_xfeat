"""Export failed-action clips using captured video frame timestamps."""
import argparse
import bisect
import json
from pathlib import Path
import shutil
import subprocess
import cv2


def rows(path):
    result=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: result.append(json.loads(line))
            except ValueError: pass
    return result


def export(root):
    out=root/'failure_clips';out.mkdir(exist_ok=True)
    ffmpeg=shutil.which('ffmpeg')
    lines=['# 失败录像索引','','播放位置按实际视频帧时间索引换算。诊断录像约 10 FPS，播放时长不等于动作真实耗时。短片包含动作前后约 2 秒。准备回 P1 的动作不计入本表。','',
           '| 轮次 | 动作 | 失败项目 | 原录像位置 | 失败短片 |','|---|---:|---|---|---|']
    count=0
    for folder in sorted(root.glob('round_*')):
        video=folder/'replay.avi'
        if not video.exists(): continue
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        timestamps=[r['capture_time'] for r in rows(folder/'video_frames.jsonl')]
        if not timestamps:
            next_time=0
            for r in rows(folder/'observations.jsonl'):
                if r['time']>=next_time:
                    timestamps.append(r['time']);next_time=r['time']+.1
        if fps<=0 or len(timestamps)!=n:
            lines.append(f'\n{folder.name}：录像与帧索引不一致，未生成精确定位。\n');continue
        events=rows(folder/'events.jsonl')
        start=next((e['time'] for e in events if e['event']=='tour_start'),float('inf'))
        attempts={e['attempt']:e for e in events if e['event']=='action_attempt'}
        for e in events:
            if e['event']!='action_result' or e['success'] or e['time']<start: continue
            a=attempts[e['attempt']];edge=a.get('edge',{})
            label=f"绳梯 #{a['ladder']}" if a['kind']=='grab' else f"P{edge['from_id']}→P{edge['to_id']} {edge['action']}"
            sec=bisect.bisect_left(timestamps,a['time'])/fps
            stamp=f'{int(sec//60):02d}:{sec%60:04.1f}'
            link='未找到 ffmpeg，仅提供原录像定位'
            if ffmpeg:
                lo=bisect.bisect_left(timestamps,a['time']-2);hi=bisect.bisect_right(timestamps,e['time']+2)
                dest=out/f'{folder.name}_action_{a["attempt"]:04d}.mp4'
                subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-ss',str(lo/fps),'-i',str(video),
                    '-t',str((hi-lo)/fps),'-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(dest)],check=True)
                check=cv2.VideoCapture(str(dest));ok,_=check.read();check.release()
                if not ok: raise RuntimeError(f'Unreadable exported clip: {dest}')
                link=f'[播放]({dest.name})'
            lines.append(f'| {folder.name} | {a["attempt"]} | {label} | [{stamp}](../{folder.name}/replay.avi) | {link} |')
            count+=1
        rejected=[e for e in events if e['time']>=start and e['event'] in ('runup_rejected','jump_geometry_rejected','alignment_failed')]
        if rejected:
            lines+=['','## 起跳前拒绝或对齐失败（不计作已起跳失败）','','| 事件 | 原录像位置 |','|---|---|']
            for e in rejected:
                sec=bisect.bisect_left(timestamps,e['time'])/fps
                lines.append(f"| {e['event']}：{e.get('reason','')} | [{int(sec//60):02d}:{sec%60:04.1f}](../{folder.name}/replay.avi) |")
    lines+=['',f'已判定失败动作共 {count} 次。完整动作、未判定记录及统计见上级目录 RESULTS.md。']
    (out/'INDEX.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(failed_actions=count,index=str(out/'INDEX.md'),clips_exported=bool(ffmpeg))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output.resolve())
