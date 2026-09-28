# Maple_xfeat

无小地图的游戏画面定位实验：根据场景静态纹理估计摄像机在地图中的位置，再用光流短时跟踪；同时尝试识别角色、对照小地图黄点坐标，并投影场景中的梯子/绳索。默认启动的是只读观察界面，不发送游戏按键。

## 效果

- 在画面中标出地图特征点、定位状态和角色估计位置。
- 显示场景总览及当前视野投影，并绘制可见梯子/绳索。
- 提供 SIFT CPU、XFeat CPU 和 XFeat CUDA 三种全局特征后端；SIFT 是默认后端，不要求安装 PyTorch。
- 可读取图片、视频或实时游戏窗口；实验输出包括逐帧 JSONL、汇总 JSON 和可视化 PNG。

定位是否成功取决于地图素材、场景遮挡、画面缩放和重复纹理。XFeat 不保证在每张地图/每一帧都能锁定。此仓库保留的是实验工具，不代表可靠的游戏自动导航器。

## 目录结构

```text
Maple_xfeat/
├── README.md
├── requirements.txt
├── config.json                 # 可编辑的默认识别参数，不含本机配置
├── no_minimap_lab/             # GUI、定位流水线、地图构建与实验工具
│   ├── cache/                  # 一个示例地图的静态图集和元数据
│   ├── run.py                  # 主入口
│   └── ...
├── src/                        # 定位器复用的主视口识别、地图图结构和窗口捕获模块
├── wz_python_tool/             # WZ 地图素材解码器
├── assets/templates/            # 空的用户模板目录；运行后可自行加入模板
├── tests/data/                  # 可选的本地演示输入
└── third_party/accelerated_features/  # XFeat 官方 Git submodule，含其权重与许可证
```

仓库包含一张预构建地图图集缓存，以便无需本地 WZ 素材也能查看其地图背景。要从新的地图 ID 构建图集，需自行准备合法取得的 `Map/` 与 `Mob/` WZ 资源放在仓库根目录；这些资源不随仓库分发。缓存元数据已移除生成机器路径。

## 环境与安装

推荐 Windows 10/11、Python 3.10 或 3.11。实时窗口采集依赖 Windows；离线处理图片/视频也建议在 Windows 环境使用完整项目依赖。

```powershell
git clone --recurse-submodules YOUR_REPOSITORY_URL Maple_xfeat
cd Maple_xfeat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若仓库已克隆但 XFeat 目录为空：

```powershell
git submodule update --init --recursive
python -m no_minimap_lab.install_xfeat
```

CPU SIFT 不需要 PyTorch。使用 XFeat 时，另外安装与你的 Python、CPU 或 CUDA 驱动匹配的 PyTorch；CUDA 版本请按 PyTorch 官方安装选择器给出的命令安装。

## 使用

### 打开只读实时定位界面

先打开游戏并进入目标地图，在仓库根目录执行：

```powershell
python -m no_minimap_lab.run --map-id 101000000 --backend sift-cpu
```

XFeat 版本：

```powershell
python -m no_minimap_lab.run --map-id 101000000 --backend xfeat-cpu
# 或
python -m no_minimap_lab.run --map-id 101000000 --backend xfeat-cuda
```

`--map-id` 是人工指定的地图 ID。首次使用其他地图时，程序会从 `Map/`、`Mob/` 构建图集并写入 `no_minimap_lab/cache/`；再次运行会复用缓存。按界面提示停止定位后，才能切换地图或特征后端。

### 图片和视频离线处理

```powershell
python -m no_minimap_lab.run --map-id 101000000 --image path\to\frame.png --output no_minimap_lab/output/demo
python -m no_minimap_lab.run --map-id 101000000 --video path\to\clip.mp4 --max-frames 300 --output no_minimap_lab/output/video
```

输出目录含 `poses.jsonl`（逐帧坐标和状态）、`summary.json`、`last_frame.png` 和 `world_overview.png`。输出和运行期缓存不会提交到 Git。

### 测量后端

```powershell
python -m no_minimap_lab.check_environment --backend sift-cpu
python -m no_minimap_lab.check_environment --backend xfeat-cpu
python -m no_minimap_lab.compare_backends
```

## XFeat 上游与许可证

XFeat 通过 Git submodule 引用官方仓库 [verlab/accelerated_features](https://github.com/verlab/accelerated_features)，固定在 `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`。XFeat 源码、预训练权重和 Apache-2.0 许可证均由 submodule 提供；此仓库不复制或改写上游文件。请保留上游许可证及版权声明。

## 文件清单

完整 Git 跟踪文件列表见 [FILE_TREE.md](FILE_TREE.md)。

## 来源与隐私

项目未写入开发者姓名、邮箱、用户目录或机器绝对路径。示例缓存元数据中的源文件签名已移除；个人校准图、模板、录屏、日志、游戏 WZ 资源和运行结果均不纳入版本控制。Git 提交使用仓库级通用 noreply 身份。
