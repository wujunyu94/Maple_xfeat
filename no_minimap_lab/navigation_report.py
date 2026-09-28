"""Build a scoreboard and world-coordinate trajectories from completed trials."""
import argparse
import json
from pathlib import Path

from .audit_navigation import audit, rows
from .atlas import ROOT
from src.vision.wz_map_reader import WzMapReader
from src.engine.platform_graph import PlatformGraphBuilder


def main():
    p=argparse.ArgumentParser()
    p.add_argument("folders",nargs="+")
    p.add_argument("--output",default="no_minimap_lab/output/navigation_results")
    args=p.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    reports=[dict(audit(f),folder=str(Path(f).resolve())) for f in args.folders]
    (out/"scores.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
    names={"opencv-v1":"第一版 OpenCV/SIFT（同步复现）","xfeat":"XFeat CUDA（修复后）","sift":"SIFT CPU（异步）"}
    text=["# 魔法密林 P1→P87 实测结果","", "地图 101000000；禁用瞬移、传送点；相同起点 X=21±8。用时包括重试、重规划和失锁等待。", "",
          "| 方案 | 完成并通过复核 | 总耗时 | 跳抓绳梯 | 其中跳抓绳索 |",
          "| --- | --- | ---: | ---: | ---: |"]
    for r in reports:
        grabs=r["jump_grabs"]; ropes=r["rope_grabs"]
        text.append(f"| {names[r['backend']]} | {'是' if r['qualifies'] else '否'} | {r.get('seconds',0):.2f} s | {grabs['confirmed']}/{grabs['attempts']} | {ropes['confirmed']}/{ropes['attempts']} |")
    text.extend(["", "每种方案只有一次最终完成样本，不能据此推断长期成功率或统计显著性。XFeat 两次调试试跑分别因地图候选丢失、宠物遮挡名牌终止，保留在 trial_xfeat_01 和 trial_xfeat_02；修复后从 P1 重新计时。",
        "", "第一版 OpenCV 本身采用 SIFT，因此第一组与第三组主要比较同步和异步架构。所有组共享路径图、控制器和恢复规则，实际失败可能触发不同绕路。",
        "", "小地图窗口在实机画面中仍可见；最终版送入镜头与人物识别前将该区域直接置零，导航不实例化黄点识别。起点及终点遮黑小地图的两后端回归通过。",
        "", "成绩采用事件的单调时钟，录像是约 10 FPS 的诊断采样，不能用视频长度替代成绩。每组 observations.jsonl、events.jsonl、manifest.json、audit.json 和 replay.avi 可复核。", ""])
    text.extend(["轨迹图表示算法输出的坐标，不是独立真值。同步 OpenCV 组出现两次约 1770 世界像素的反向坐标跳变（误匹配后恢复）；按键审计没有瞬移键。图用橙圈标出突跳，断开异常连线，原始数据保留且不扣除对应耗时。XFeat 和异步 SIFT 本轮未出现同一阈值下的突跳。", ""])
    (out/"RESULTS.md").write_text("\n".join(text),encoding="utf-8")
    from PIL import Image,ImageDraw,ImageFont
    graph=PlatformGraphBuilder.build_from_map_dict(WzMapReader(str(ROOT/"Map")).load_map(101000000),enable_teleport=False)
    chart=Image.new("RGB",(500*len(reports),1100),"white")
    draw=ImageDraw.Draw(chart)
    font=ImageFont.load_default()
    small=ImageFont.load_default()
    for column,report in enumerate(reports):
        left=column*500+60
        def xy(x,y):
            return (left+(x+2200)/3900*415,100+(y+4400)/4850*900)
        for level in (-4000,-3000,-2000,-1000,0):
            a,b=xy(-2200,level),xy(1700,level)
            draw.line([a,b],fill="#edf2f7")
            draw.text((left-8,a[1]),str(level),fill="#64748b",font=small,anchor="rm")
        for x in (-1000,0,1000):
            a,b=xy(x,-4400),xy(x,450)
            draw.line([a,b],fill="#edf2f7")
            draw.text((b[0],1010),str(x),fill="#64748b",font=small,anchor="mt")
        for n in graph.nodes.values():
            for line in n.raw_lines:
                draw.line([xy(line['x1'],line['y1']),xy(line['x2'],line['y2'])],fill="#cbd5e1")
        for rope in graph.ladder_ropes.values():
            draw.line([xy(rope.x,rope.y1),xy(rope.x,rope.y2)],fill="#d6bcfa")
        obs=rows(Path(report['folder'])/'observations.jsonl')
        previous=None
        previous_row=None
        jump_times={j["time"] for j in report["abrupt_coordinate_changes"]}
        for r in obs:
            point=xy(*r['world']) if r['world'] else None
            if point and previous:
                if r['time'] in jump_times:
                    for x,y in (previous,point):
                        draw.ellipse((x-5,y-5,x+5,y+5),outline="#f59e0b",width=2)
                else:
                    draw.line([previous,point],fill="#0284c7",width=2)
            previous=point
        for name,point,color in (("P1",xy(21,269),"#22c55e"),("P87",xy(-129,-4148),"#ef4444")):
            x,y=point
            draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
            draw.text((x+10,y),name,fill=color,font=small,anchor="lm")
        draw.text((column*500+250,30),report['backend'],fill="#0f172a",font=font,anchor="mm")
        draw.text((column*500+250,60),f"{report.get('seconds',0):.2f} s",fill="#0284c7",font=font,anchor="mm")
        draw.text((column*500+250,1050),"World X  /  World Y (up = negative)",fill="#64748b",font=small,anchor="mm")
        draw.text((column*500+250,1080),f"Abrupt coordinate changes: {len(jump_times)}",fill="#b45309",font=small,anchor="mm")
    chart.save(out/'trajectories.png')
    print(json.dumps(reports,indent=2))


if __name__ == "__main__":
    main()
