# Maple_xfeat

无小地图的游戏画面定位实验：根据场景静态纹理估计摄像机在地图中的位置，再用光流短时跟踪；同时尝试识别角色，并投影场景中的梯子/绳索。默认启动的是只读观察界面，不发送游戏按键。

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
├── requirements.txt            # GUI、图像处理和地图解码基础依赖
├── requirements-xfeat.txt      # XFeat 的 PyTorch + tqdm 依赖
├── install_dependencies.bat    # Windows 一键安装脚本
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

仓库包含地图 `101000000` 的预构建图集缓存，因此可先用此地图试运行。其他地图需要自行准备合法取得的 WZ 地图资源；地图素材不随仓库分发。缓存元数据已移除生成机器路径。

## 环境与安装

推荐 Windows 10/11、Python 3.10 或 3.11。实时窗口采集依赖 Windows；离线处理图片/视频也建议在 Windows 环境使用完整项目依赖。首次安装可在仓库根目录双击 `install_dependencies.bat`，按提示选择后端：

- `1`：SIFT CPU，只安装基础依赖，不装 PyTorch。
- `2`：XFeat CPU，安装 CPU 版 PyTorch 和 XFeat 所需的 `tqdm`。
- `3`：XFeat CUDA，安装基础依赖及 PyTorch，然后检查 PyTorch 是否能访问 NVIDIA GPU。

脚本会创建 `.venv`、安装依赖并初始化 XFeat submodule。GPU 模式使用 PyPI 当前提供的 PyTorch wheel；若 CUDA 检查失败，请按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择适合 Windows、Python 和显卡驱动的 CUDA 构建，再在虚拟环境里重装 PyTorch。安装脚本不需要管理员权限；完整路线测试中的按键服务会另外请求管理员授权。

```powershell
git clone --recurse-submodules YOUR_REPOSITORY_URL Maple_xfeat
cd Maple_xfeat
.\install_dependencies.bat
```

也可以手动安装基础依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git submodule update --init --recursive
```

手动装 XFeat CPU 时先运行 `python -m pip install tqdm`，再运行 `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`。`requirements-xfeat.txt` 是使用默认 PyPI 安装 XFeat 的可选依赖清单。GPU 构建依赖本机显卡和驱动，优先使用上方一键脚本或 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)生成的安装命令。可运行 `python -m no_minimap_lab.install_xfeat` 检查 submodule 和权重是否齐全。

## 使用

### 打开只读实时定位界面

先打开游戏并进入目标地图，在仓库根目录执行。程序会打开 GUI 并自动开始定位：

```powershell
python -m no_minimap_lab.run --map-id 101000000 --backend sift-cpu
```

XFeat 版本：

```powershell
python -m no_minimap_lab.run --map-id 101000000 --backend xfeat-cpu
# 或
python -m no_minimap_lab.run --map-id 101000000 --backend xfeat-cuda
```

### GUI 操作

1. 游戏进入与地图 ID 对应的场景后启动程序。顶部的“地图 ID”应与当前游戏地图一致；“画面倍率”是游戏地图像素到截图像素的比例，默认 `1.0`，可设范围为 `0.25` 到 `4.0`。
2. 在“地标后端”选择 `sift-cpu`、`xfeat-cpu` 或 `xfeat-cuda`。XFeat 选项需要已安装 PyTorch；GPU 选项还要求 `torch.cuda.is_available()` 为真。
3. 点击“开始定位”。初次会检查依赖、生成或载入地图图集，然后开始只读捕获。点击“停止”后可改地图 ID 或后端，再点击开始。
4. 左侧“实时原画”和“定位输出”标签可切换原画及地标/梯绳叠加；右侧显示地图总览和视野位置。状态栏显示锁定/失锁、人物、黄点坐标和耗时。
5. 需要人物脚底坐标时，站稳后使用“③ 名牌与脚底”按提示框选名牌和脚底；“② 特征标定（主程流程）”用于录入人物外观特征。标定结果保存在本机 `no_minimap_lab/calibration/`，不会进 Git。
6. “忽略区域”可框出不参与匹配的界面区域；“重新定位”会重置当前跟踪；“保存诊断”保存当前帧和状态供排查。

定位 GUI 默认只读、不发游戏按键。下方“完整路线测试”是单独的自动控制实验，会启动按键服务并控制游戏，仅在明确需要时使用；该测试固定地图 `101000000` 和倍率 `1.0`，开始前要确认游戏状态。

`--map-id` 是人工指定的地图 ID。首次使用其他地图时，程序会从仓库根目录的 `Map/` 解码所需地图、Tile 和 Obj `.img` 文件，构建图集并写入 `no_minimap_lab/cache/`；再次运行会复用缓存。期望目录结构如下：

```text
Map/
├── Map/Map1/101000000.img       # 示例 ID 的地图 IMG；其他地图按其 ID 首位分组
├── Tile/<tileSet>.img           # 地图引用的 Tile IMG
└── Obj/<objectSet>.img          # 地图引用的 Obj IMG
```

取得与解码 WZ 资源后，把生成的 `Map/` 放在仓库根目录，与 `no_minimap_lab/` 同级。图集由它实际引用到的文件构建；缺少 Tile/Obj 文件会造成图块空缺或构建失败。运行时在 GUI 中输入正确 ID，然后点“开始定位”即可生成缓存，不需要先执行额外命令。已有缓存时，删除对应 ID 的 `.png` 和 `.json` 再次启动可让它从当前 `Map/` 重建。部分地图是链接地图，程序会提示其目标 ID；请改用提示的 ID。地图变化时先停止，再填入新 ID 并开始。

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
