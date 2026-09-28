# 魔法密林无小地图 P1→P87 实机测试

地图 `101000000`，平台编号沿用主程序 `PlatformGraphBuilder` 合并结果（共 87 个）。人物沿用主程序保存的名牌、服饰特征及脚底标定。导航进程不创建 `MinimapTracker`，也不创建黄点坐标对照支路；屏幕左上角区域同时排除在镜头和人物识别之外。地图 IMG 的 foothold / ladderRope 提供平台和绳梯物理坐标。

三组共用控制器、路径规划规则与起点 P1 的 X=21（允许 ±8 世界像素），正式计时从取得稳定起点开始，到连续至少 350 ms 确认脚底位于 P87 表面结束。计时包含行走、跳跃、攀爬、失锁等待、失败重试与重规划；回到底部的准备过程单独记录，不计入成绩。时间成绩越低越好，跳抓成功率单独作为加分依据，不擅自设置用户未指定的分值权重。

| 名称 | 实际定位实现 |
| --- | --- |
| 第一版 OpenCV（同步复现） | SIFT 地标重定位 + 有限时长 LK 光流；全局匹配阻塞当前观察循环 |
| XFeat | PyTorch CUDA 学习特征 + 不透明地图像素校验；后台全局匹配 + 前台 LK |
| SIFT | 当前 CPU SIFT；后台全局匹配 + 前台 LK |

OpenCV 是库，SIFT 是算法；第一版本身使用 SIFT，因此第一组与第三组比较的是同步/异步架构，并非两个互不相关的特征算法。当前工作区没有第一版不可变发布包，第一组复现的是其同步架构。

路径规划剔除 `TELEPORT*` 和 `PORTAL` 边。独立管理员输入服务仅允许方向键、Alt 跳跃和 Tab；不允许配置中的 C 瞬移键。接近地图传送点的攀爬动作另行拒绝。服务只绑定启动时的游戏窗口，命令失联最多 250 ms 后松键，失去游戏焦点也松键；F12 终止服务。服务不接受任意代码或命令执行。

控制流程：视觉脚底匹配原始斜坡表面 → 规划下一动作 → 走到起跳/抓取位置 → 依据实时脚底闭环执行 → 稳定落台核验；失败则加大该动作代价并重规划。跳抓计数记录实际发出 Alt+Up 的尝试，单独记录抓稳和抵达目标平台。绳索和梯子分别统计。自动导航不会使用人物预测值作为新鲜控制坐标。

## 运行与证据

输入服务需一次管理员启动；后续观察与导航使用普通权限，共用这个服务。

```powershell
python -m no_minimap_lab.input_service
python -m no_minimap_lab.navigation --backend opencv-v1 --target 87 --budget 600 --output no_minimap_lab/output/trial_opencv_01
python -m no_minimap_lab.navigation --backend xfeat --target 87 --budget 600 --output no_minimap_lab/output/trial_xfeat_01
python -m no_minimap_lab.navigation --backend sift --target 87 --budget 600 --output no_minimap_lab/output/trial_sift_01
python -m no_minimap_lab.audit_navigation no_minimap_lab/output/trial_opencv_01
```

三个导航命令必须顺序运行，不能同时争用按键。每次新组开始前，需要独立返回 P1 并对齐起点；`--target 1 --position 21` 用于这一准备步骤。观察模式 `--observe` 不移动角色。

每组保存：`result.json`（成绩）、`audit.json`（从观察记录重新核对起终点）、`observations.jsonl`（采集时间、镜头/人物坐标、承重平台）、`events.jsonl`（每段路径与抓取事件）、`latest.jpg`（最新/终点画面）、`replay.avi`（约 10 FPS 诊断录像）。录像帧间隔受处理耗时影响，不用视频时长算成绩；成绩采用单调时钟。输入服务的 `output/navigation_control/keys.jsonl` 单独保存真实按键变化。

初步版本 `trial_opencv_01` 完成：149.73 秒，跳抓绳梯 6/6，其中绳索 4/4。随后统一加入人物高置信特征重找、直接抹黑小地图输入及更长的落台稳定观测窗口，三组最终比较使用 `output/final_suite`，初步版本不混入最终排名。

XFeat 调试过程保留两次未完成记录：`trial_xfeat_01` 在 P41 附近丢失地图候选，`trial_xfeat_02` 因宠物遮挡名牌而失去人物。修复包括更宽的候选搜索、分布充分的地图像素核验、连续两帧的唯一高分人物特征重找；另加入有时限的光流位置先验以提出局部地图搜索范围，仍须当前画面通过实际地图像素验证才刷新世界锚点。没有接入 SIFT 为 XFeat 的正式上行代算坐标。

最终三组均从 P1 到达 P87，起终点复核通过：

| 方案 | 总耗时 | 跳抓绳梯 | 其中绳索 |
| --- | ---: | ---: | ---: |
| 第一版 OpenCV/SIFT 同步复现 | 204.92 秒 | 6/6 | 4/4 |
| XFeat CUDA | 112.39 秒 | 5/5 | 3/3 |
| SIFT CPU 异步 | 114.82 秒 | 5/5 | 3/3 |

每组只有一次最终完成样本；2.42 秒的 XFeat / 异步 SIFT 差距不足以推断长期优劣。同步组有更多重试和绕路；所有恢复耗时均计入成绩，抓取次数随实际路线变化。完整结果在 [成绩表](output/navigation_results/RESULTS.md)、[轨迹图](output/navigation_results/trajectories.png) 和 `output/final_suite/<后端>/` 的原始记录中。

准备回程可切换定位后端帮助回到底部，但正式每组从 P1 重新开始，使用指定单一后端。38 项回归测试通过。管理员输入服务在三组完成后已停止，人物留在 P87。

实机截图中小地图仍可见；最终版在进入镜头和人物识别前，直接把左上角 30% 宽 × 38% 高像素置零，完全没有读取黄点。Tab 尝试没有将该客户端的小地图关闭，不能宣称已经物理关闭；原始画面仅用于人工录像复核。起终点遮黑回归和代码路径隔离验证了不依赖该区域。
