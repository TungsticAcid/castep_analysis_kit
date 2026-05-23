# PDOS 态密度数据提取与绘图工具

解析 Materials Studio 导出的 `PDOS.xcd` 文件（XML 格式），绘制分轨道/自旋的态密度图，并提供 d 带中心、晶场劈裂、自旋劈裂等催化领域常用分析。

## 功能概览

- **文件解析** — 自动检测轨道类型（s/p/d/f/sum）和自旋极化（α/β），支持 MS 全部 5 种常见系列组合
- **三种绘图模式** — 自旋极化（α↑/β↓ 分侧）、轨道分别显示、总态密度（TDOS）
- **双模式运行** — GUI（tkinter）和 CLI（argparse），互不依赖
- **态密度分析** — d/p 带中心、带宽、晶场劈裂、自旋劈裂、占据态比例、积分面积、峰搜索
- **最近文件** — 自动记录最近打开的 10 个文件，跨会话保留
- **分析结果** — 同时输出到 GUI 文本框和命令行终端

## 依赖

| 依赖 | 用途 |
|------|------|
| Python ≥ 3.8 | — |
| numpy | 数值计算、梯形积分 |
| scipy | 峰搜索 (`signal.find_peaks`) |
| matplotlib | 绑图与渲染 |
| tkinter | GUI 界面（Python 标准库自带） |
| xml.etree.ElementTree | XML 解析（Python 标准库自带） |

```bash
pip install numpy scipy matplotlib
```

## 快速开始

### GUI 模式

```bash
# 直接启动
python pdos_plotter.py

# 启动并自动加载指定文件
python pdos_plotter.py -f "path/to/PDOS.xcd"
```

界面从上到下依次为：文件选择（含路径输入框、最近打开、浏览按钮）→ 轨道选择 → 绘图模式 → 能量范围 → 保存设置 → 分析报告 → 操作按钮。

### CLI 模式

```bash
# 自旋极化图
python pdos_plotter.py -f "PDOS.xcd" --spin --no-gui -o output.png

# 总态密度
python pdos_plotter.py -f "PDOS.xcd" --total --no-gui

# 轨道分别显示
python pdos_plotter.py -f "PDOS.xcd" --orbitals s,p,d --no-gui -o orbitals.png

# 绘图 + 分析
python pdos_plotter.py -f "PDOS.xcd" --spin --analyze --analysis-orbital d --no-gui

# 指定能量范围 & 弹窗预览
python pdos_plotter.py -f "PDOS.xcd" --spin --no-gui --emin -10 --emax 5 --show
```

## CLI 参数一览

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f`, `--file` | PDOS.xcd 文件路径（CLI 必填） | — |
| `-o`, `--output` | 输出图片路径 | `./pic/<文件名>_pdos.png` |
| `--no-gui` | 启用 CLI 模式，绑图后退出 | 关闭（启动 GUI） |
| `--spin` | 自旋极化绘图模式 | 关闭 |
| `--total` | 总态密度绘图模式 | 关闭 |
| `--orbitals` | 轨道列表（逗号分隔，如 `s,p,d`） | 自动检测 |
| `--analyze` | 执行态密度分析并打印报告 | 关闭 |
| `--analysis-orbital` | 分析的目标轨道 | `d` |
| `--analysis-spin` | 分析的自旋方向（`alpha`/`beta`） | 自动选择 |
| `--emin` | 能量范围下限 (eV) | 数据最小值 |
| `--emax` | 能量范围上限 (eV) | 数据最大值 |
| `--title` | 图表标题 | 文件名 |
| `--dpi` | 输出图片分辨率 | `300` |
| `--show` | 保存后弹窗预览图片 | 关闭 |

## GUI 界面说明

### 文件选择

- **路径输入框** — 可直接粘贴 `.xcd` 文件路径
- **最近打开 ▼** — 下拉菜单列出最近 10 个文件，点击即加载；底部可「清空最近记录」
- **浏览...** — 弹出系统文件对话框选择文件

### 绘图模式

| 模式 | 说明 |
|------|------|
| 自旋极化 | α 和 β 自旋在 y>0 同侧显示，α 实线填充、β 虚线斜线填充 |
| 轨道分别 | 各轨道（s/p/d）在 y>0 同侧用不同颜色绘制 |
| 总态密度 | 仅绘制 Sum 系列（费米能级 E=0 处虚线） |

### 能量范围

- **自动** — 使用文件中数据的完整能量范围
- **手动** — 自行输入上下限 (eV)，仅绘制指定区间

### 分析报告

点击「分析报告」按钮对当前选中轨道执行完整分析。右侧「分析选项」按钮可勾选需要的分析项，并设置积分面积的积分区间。

## 分析功能详解

分析结果按以下顺序输出：

1. **带中心（band center）** — ε = ∫E·DOS dE / ∫DOS dE，分全部态和占据态两种方式。自旋极化文件自动展示 α/β 双自旋带中心对比及 Δ(α−β)
2. **带宽（band width）** — DOS 加权标准差 σ、半高宽 FWHM、能量跨度
3. **劈裂分析** — 自旋劈裂 ΔE_spin（交换劈裂）和晶场劈裂 ΔE_cf（主峰间距）并列展示
4. **占据态比例** — 费米能级以下（E<0）的 DOS 积分占比
5. **积分面积** — 指定区间内的 DOS 积分，GUI 中可自定义上下限
6. **峰列表** — 按 DOS 强度降序排列的各峰能量与强度

### 分析选项

| 选项 | 说明 |
|------|------|
| 带中心 | 全部态 + 占据态 d/p 带中心，自旋极化时区分 α/β |
| 带宽 | 标准差 σ、FWHM、能量跨度 |
| 自旋劈裂 | α−β 带中心能量差（仅自旋极化文件有效） |
| 晶场劈裂 | 最高两峰的能级间距 ΔE |
| 占据态比例 | E<0 积分 / 全区间积分 |
| 积分面积 | 指定能量区间的 DOS 积分，可自定义上下限 |
| 峰列表 | scipy.find_peaks 搜索结果 |

## 文件格式

脚本自动识别以下 5 种 MS 导出的系列组合：

| 类型 | 系列名 | 系列数 |
|------|--------|--------|
| 自旋极化 + s/p/d 全轨道 | `s alpha`, `s beta`, `p alpha`, `p beta`, `d alpha`, `d beta`, `Sum alpha`, `Sum beta` | 8 |
| 自旋极化 + 仅 d 轨道 | `d alpha`, `d beta`, `Sum alpha`, `Sum beta` | 4 |
| 非自旋极化 + s/p/d | `s `, `p `, `d `, `Sum ` | 4 |
| 非自旋极化 + 仅总量 | `total ` | 1 |

> **注意**：Materials Studio 导出的 beta 自旋 DOS 数据为负值，脚本内部自动取绝对值处理。

## 项目结构

```
pdos_plotter/
├── pdos_plotter.py   # 主脚本（~2800 行，含 5 个类）
├── pic/              # 默认图片输出目录
└── README.md         # 本文件
```

### 代码架构

| 类 | 职责 |
|----|------|
| `PDOSParser` | XML 解析、自旋/轨道自动检测、数据筛选 |
| `PDOSPlotter` | matplotlib 绑图（3 种模式）、图片保存 |
| `PDOSAnalyzer` | 带中心/带宽/劈裂/占据态/峰面积/峰搜索分析 |
| `PDOSGUI` | tkinter GUI 界面、最近文件管理、分析选项对话框 |
| CLI 函数 | argparse 命令行解析与调度 |
