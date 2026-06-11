# PDOS Plotter —— 态密度数据提取与可视化工具

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

解析 Materials Studio `.xcd` 及 CASTEP `.castep_bin`/`.pdos_bin` 输出文件，绘制分轨道/自旋的态密度（PDOS）图，并提供 d 带中心、晶场劈裂、自旋劈裂、重叠面积等催化领域常用分析功能。

**作者**: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)

---

## 目录

- [功能概览](#功能概览)
- [依赖与安装](#依赖与安装)
- [快速开始](#快速开始)
- [GUI 界面说明](#gui-界面说明)
- [CLI 命令行参考](#cli-命令行参考)
- [交互式预览](#交互式预览)
- [分析功能详解](#分析功能详解)
- [支持的输入格式](#支持的输入格式)
- [项目架构](#项目架构)
- [License](#license)

---

## 功能概览

### 数据输入
| 格式 | 来源 | 支持程度 |
|------|------|----------|
| `.xcd` (XML) | Materials Studio PDOS 导出 | 完整（自旋极化、轨道分辨） |
| `.castep_bin` | CASTEP 主输出 (Fortran binary) | 元素级 DOS + 晶体结构 |
| `.pdos_bin` | CASTEP 投影权重 (Fortran binary) | m_l 分辨（px/py/pz/d_xy/...） |

### 绘图模式
- **自旋极化** — α/β 自旋对比显示，填充+斜线区分
- **轨道分辨** — s/p/d/f 轨道分色绘制，m_l 亚轨道独立着色
- **总态密度** — TDOS 汇总，费米能级参考线
- **多文件叠加** — 不同文件/轨道/自旋的 DOS 曲线叠放在同一张图上对比

### 分析功能
- d/p 带中心（全部态 + 占据态）、带宽（σ/FWHM/跨度）
- 晶场劈裂 ΔE_cf、自旋劈裂 ΔE_spin
- 占据态比例、指定区间积分面积
- 自动峰搜索（scipy）+ **手动识峰**（交互式点击）
- **轨道重叠面积**计算（两 DOS 曲线间的重叠指数）

### 图形界面
- tkinter 完整 GUI，含文件浏览、最近打开记录
- **晶体结构 3D 查看器**（球棍模型，支持点击选中原子）
- **轨道树**选择器（checkbox 勾选，支持聚合级别切换）
- **交互式预览**——十字光标、快捷键识峰、重叠分析、基线切换
- 自定义图表标题、能量范围、输出分辨率

---

## 依赖与安装

### 运行环境
- Python ≥ 3.8
- numpy, scipy, matplotlib

### 安装依赖

```bash
pip install numpy scipy matplotlib
```

tkinter 为 Python 标准库自带，无需额外安装。若 Linux 下缺失，可执行：

```bash
# Ubuntu/Debian
sudo apt install python3-tk
# CentOS/RHEL
sudo yum install python3-tkinter
```

### 使用方式

```bash
# 方式一：包模式启动（推荐）
python -m pdos_plotter

# 方式二：脚本直接启动
python main.py

# 方式三：加载指定文件
python -m pdos_plotter -f "path/to/PDOS.xcd"

# 方式四：CLI 模式（不启动 GUI）
python -m pdos_plotter -f "PDOS.xcd" --total --no-gui -o output.png
```

---

## 快速开始

### GUI 模式

```bash
# 直接启动 GUI
python -m pdos_plotter

# 启动并自动加载文件（支持同时加载多个）
python -m pdos_plotter -f "Ni_bulk.xcd" -f "Ni_slab.xcd"
```

操作流程：
1. 加载文件（拖入路径 / 浏览 / 最近打开）
2. 勾选轨道（轨道树 checkbox，CASTEP 模式下可切换聚合级别）
3. 选择绘图模式（轨道分离 / 自旋极化 / 总态密度 / 多文件叠加）
4. 设置能量范围（自动或手动）
5. 可选：输入自定义标题
6. 点击「预览」→ 交互式 matplotlib 窗口
7. 满意后点击「保存图片」

### CLI 模式

```bash
# ── XCD 文件 ──
# 自旋极化绘图
python -m pdos_plotter -f "PDOS.xcd" --spin --no-gui -o spin.png

# 总态密度
python -m pdos_plotter -f "PDOS.xcd" --total --no-gui

# 指定轨道的轨道分离图
python -m pdos_plotter -f "PDOS.xcd" --orbitals s,p,d --no-gui -o orbitals.png

# 绘图 + 分析报告
python -m pdos_plotter -f "PDOS.xcd" --spin --analyze --analysis-orbital d --no-gui

# 指定能量范围并弹窗显示
python -m pdos_plotter -f "PDOS.xcd" --spin --no-gui --emin -10 --emax 5 --show

# ── CASTEP 二进制文件 ──
# 元素级 DOS
python -m pdos_plotter --mode castep --castep-bin "Ni_DOS.castep_bin" --total --no-gui

# 完整 m_l 分辨
python -m pdos_plotter --mode castep \
    --castep-bin "Ni_DOS.castep_bin" \
    --pdos-bin "Ni_DOS.pdos_bin" \
    --group-by species_orbital \
    --orbitals d_xy,d_yz,d_z2 --no-gui

# 多文件叠加（CLI）
python -m pdos_plotter -f "sys1.xcd" -f "sys2.xcd" \
    --multi-overlay "0:s,p:sum:系统A;1:s,p:sum:系统B" --no-gui
```

---

## GUI 界面说明

### 整体布局

```
+-- 文件选择 -----------------------------------+
| [路径输入框________________________] [浏览]    |
| 已加载文件列表 (多选支持)        [晶体结构]    |
+-- 轨道选择 -----------------------------------+
| [轨道树: [x] Ni-px  [x] Ni-py  [ ] Ni-d_xy...]|
| 聚合级别: [物种+轨道 v]                        |
+-- 叠加系列构建 (多文件叠加模式) --------------+
| [系列列表________________________]            |
| [+ 添加] [x 移除] [清空] [^ 上移] [v 下移]    |
+-- 绘图模式 -----------------------------------+
| ( ) 轨道分离  ( ) 自旋极化  ( ) 总态密度  ( ) 叠加 |
+-- 自旋模式: [alpha v]                         |
+-- 能量范围: E_min [_____] E_max [_____]       |
+-- 输出设置 -----------------------------------+
| 标题: [________________________]              |
| 保存路径: [________________________] [浏览]   |
| DPI: [300]                                    |
+-- 操作按钮 -----------------------------------+
| [刷新] [预览] [保存图片] [分析]               |
+-----------------------------------------------+
```

### 轨道树（Orbital Tree）

- checkbox 勾选/取消轨道
- CASTEP 模式下支持聚合级别切换：总 DOS → 元素 → 元素+角动量 → 元素+轨道 → 原子+角动量 → 原子+轨道
- 右键快捷操作

### 晶体结构查看器

- 从 `.castep_bin` 自动提取晶格矢量和原子坐标
- matplotlib 3D 球棍模型渲染
- 点击原子高亮选中
- 自动处理多 CELL 块（取最后一个）

### 叠加系列构建

多文件叠加模式下，可逐个添加系列：
1. 选择源文件
2. 选择目标轨道和自旋聚合方式
3. 系列按列表顺序绘制，支持 **上移 / 下移** 调整次序
4. 颜色从调色板自动循环分配

---

## CLI 命令行参考

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f`, `--file` | 输入文件路径（可多次指定） | GUI 模式可选 |
| `-o`, `--output` | 输出图片路径 | `./pic/<文件名>_pdos.png` |
| `--no-gui` | CLI 模式，绑图后退出 | 关闭 |
| `--title` | 自定义图表标题 | 文件名 |
| `--dpi` | 输出图片 DPI | `300` |
| `--show` | 保存后弹窗显示 | 关闭 |

### 绘图模式

| 参数 | 说明 |
|------|------|
| `--spin` | 自旋极化图（α/β 对比） |
| `--total` | 总态密度（TDOS） |
| `--orbitals s,p,d` | 轨道分离模式，指定轨道列表（逗号分隔） |
| `--multi-overlay SPEC` | 多文件叠加模式 |

### CASTEP 模式参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode castep` | 切换至 CASTEP 二进制输入 | `xcd` |
| `--castep-bin PATH` | `.castep_bin` 文件路径 | 必填 |
| `--pdos-bin PATH` | `.pdos_bin` 文件路径 | 可选 |
| `--group-by` | 聚合级别 | `species_orbital` |
| `--sigma` | Gaussian 展宽 (eV) | `0.2` |
| `--n-points` | 能量网格点数 | `500` |

`--group-by` 可选值：
- `total` — 全体系总 DOS
- `species` — 按元素（Ni / C / H）
- `species_l` — 按元素+角动量（Ni-d / C-p）
- `species_orbital` — 按元素+m_l 轨道（Ni-d_xy / C-px）
- `atom_l` — 按原子+角动量（Ni1-d / Ni2-d）
- `atom_orbital` — 按原子+m_l 轨道（Ni1-d_xy）

### 分析参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--analyze` | 执行分析并打印报告 | 关闭 |
| `--analysis-orbital` | 目标轨道 | `d` |
| `--analysis-spin` | 自旋方向 (`alpha`/`beta`) | 自动 |
| `--emin` | 能量下限 (eV) | 数据范围 |
| `--emax` | 能量上限 (eV) | 数据范围 |

### 多文件叠加规格格式

```
文件索引:轨道列表:自旋聚合[:自定义标签]
```

示例：
```bash
# 文件0的s和p轨道(加和) vs 文件1的s轨道(加和)
--multi-overlay "0:s,p:sum:催化剂;1:s:sum:基底"
# 多个系列用分号分隔
--multi-overlay "0:s:sum;0:p:sum;1:s:sum"
```

---

## 交互式预览

在 GUI 中点击「预览」或 CLI 模式加 `--show` 后弹出的 matplotlib 窗口中，提供以下交互功能：

### 鼠标操作
| 操作 | 效果 |
|------|------|
| 移动鼠标 | 十字光标跟随，左上角实时显示 (E, DOS) 坐标 |

### 键盘快捷键

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| `p` | 识峰模式 | 点击两点定义峰区间，自动计算峰位置/高度/面积/FWHM |
| `b` | 基线切换 | 切换是否扣除线性基线（识峰模式下） |
| `n` | 系列切换 | 切换识峰目标轨道/系列 |
| `s` | 自旋切换 | 切换识峰自旋方向（alpha → beta → sum） |
| `d` | 显示模式 | 循环切换峰结果显示详略（全部/简要/仅位置/位置+面积+FWHM） |
| `1` | 选重叠 A | 指定重叠分析的第一个系列 |
| `2` | 选重叠 B | 指定重叠分析的第二个系列 |
| `t` | 计算重叠 | 点击两点定义积分区间，计算 A/B 两系列的重叠面积指数 |
| `c` | 清除 | 清除所有识峰标记和重叠分析结果 |

### 识峰模式

1. 按 `p` 进入识峰模式 → 标题变为红色提示
2. 点击两个点定义待分析峰的能量区间
3. 自动输出：峰位置、峰高、面积、质心、FWHM、拖尾因子
4. 按 `b` 切换基线扣除，按 `n`/`s` 切换目标，按 `d` 切换显示详略
5. 再次按 `p` 退出

### 重叠面积分析

1. 按 `n` 切换到系列 A → 按 `1` 标记
2. 按 `n` 切换到系列 B → 按 `2` 标记
3. 按 `t` → 点击两点定义积分区间
4. 自动输出重叠指数（两 DOS 曲线的归一化交叉积分）

重叠指数定义：
```
overlap = integral min(DOS_A(E), DOS_B(E)) dE / (integral DOS_A dE + integral DOS_B dE)
```
值域 [0, 0.5]，越大表示两 DOS 分布越相似。

---

## 分析功能详解

### 带中心 (Band Center)

ε = ∫E·DOS(E) dE / ∫DOS(E) dE，分别计算：
- **全部态带中心** — 全能量范围
- **占据态带中心** — 仅 E < E_fermi
- 自旋极化体系自动对比 α/β 双自旋带中心，展示 Δ(α−β)

### 带宽 (Band Width)

- **加权标准差 σ** — DOS 分布的统计展宽
- **半高宽 FWHM** — 峰高一半处的能量宽度
- **能量跨度** — DOS 非零区间的能量范围

### 劈裂分析

- **自旋劈裂 ΔE_spin** — α 和 β 占据态带中心之差（交换劈裂）
- **晶场劈裂 ΔE_cf** — 最高两个峰的能级间距

### 其他分析项

| 项目 | 说明 |
|------|------|
| 占据态比例 | E < E_fermi 区间积分 / 全区间积分 |
| 积分面积 | 自定义区间内的 DOS 梯形积分 |
| 峰列表 | scipy.find_peaks 自动检测，按强度降序排列 |
| 手动识峰 | 交互式点击选峰（见[交互式预览](#识峰模式)） |
| 重叠面积 | 两 DOS 曲线相似度量化（见[交互式预览](#重叠面积分析)） |

---

## 支持的输入格式

### Materials Studio `.xcd` (XML)

Materials Studio 导出的 PDOS 数据文件，自动识别以下系列组合：

| 类型 | 包含系列 | 系列数 |
|------|----------|--------|
| 自旋极化 + s/p/d 全轨道 | `s alpha`, `s beta`, `p alpha`, `p beta`, `d alpha`, `d beta`, `Sum alpha`, `Sum beta` | 8 |
| 自旋极化 + 仅 d 轨道 | `d alpha`, `d beta`, `Sum alpha`, `Sum beta` | 4 |
| 非自旋 + s/p/d | `s`, `p`, `d`, `Sum` | 4 |
| 非自旋 + 仅总量 | `total` | 1 |

> **注意**：Materials Studio 导出的 beta 自旋 DOS 数据为负值，程序内部自动取绝对值。

### CASTEP 二进制文件

| 文件 | 内容 |
|------|------|
| `*_DOS.castep_bin` | Fortran Big-Endian 顺序记录文件。含晶格矢量、原子坐标、物种信息（CELL 段）、元素级 PDOS、总 DOS |
| `*_DOS.pdos_bin` | m_l 分辨的投影权重矩阵。记录每个 k 点/能带的投影权重 |

单独提供 `.castep_bin` 即可获得元素级 DOS；同时提供 `.pdos_bin` 则可获得完整 m_l 轨道分辨。

---

## 项目架构

```
pdos_plotter/
├── __init__.py              # 包入口，公开 API 导出
├── __main__.py              # 支持 python -m pdos_plotter
├── main.py                  # 直接运行入口
├── cli.py                   # argparse CLI 参数解析与调度
├── constants.py             # 全局常量（颜色方案、轨道表、默认参数）
│
├── pdos_parser.py           # MS .xcd XML 解析器
├── pdos_plotter.py          # matplotlib 绑图引擎（4 种模式）
├── pdos_analyzer.py         # 态密度分析器（带中心/劈裂/峰搜索/重叠面积）
├── pdos_gui.py              # tkinter GUI 主界面
├── interactive_preview.py   # matplotlib 交互预览（识峰/重叠/十字光标）
│
├── binary_io.py             # Fortran 二进制记录 I/O（Big-Endian）
├── castep_bin_parser.py     # .castep_bin 解析器（元素级 DOS + 晶体结构）
├── pdos_calc.py             # CASTEP PDOS 计算器 + 适配器
├── crystal_viewer.py        # 晶体结构 3D 球棍模型查看器
├── orbital_tree.py          # tkinter 轨道树组件（checkbox 选择）
│
├── pic/                     # 默认图片输出目录
└── README.md
```

### 核心类关系

| 类 | 模块 | 职责 |
|----|------|------|
| `PDOSParser` | `pdos_parser.py` | MS .xcd XML 解析，自旋/轨道自动检测 |
| `PDOSPlotter` | `pdos_plotter.py` | matplotlib 绑图（spin/total/orbitals/multi_overlay） |
| `PDOSAnalyzer` | `pdos_analyzer.py` | 带中心/带宽/劈裂/占据态/峰搜索/重叠面积 |
| `PDOSGUI` | `pdos_gui.py` | tkinter GUI 主界面，文件管理，参数设置 |
| `InteractivePreview` | `interactive_preview.py` | matplotlib 交互预览（十字光标/识峰/重叠） |
| `CastepBinParser` | `castep_bin_parser.py` | .castep_bin 解析（Fortran Big-Endian 二进制） |
| `CastepPDOSCalculator` | `pdos_calc.py` | 从 .castep_bin + .pdos_bin 计算 m_l 分辨 PDOS |
| `CastepPDOSAdapter` | `pdos_calc.py` | CASTEP 数据 → PDOSPlotter 接口的适配器（鸭子类型） |
| `CrystalStructure` | `crystal_viewer.py` | 晶体结构数据容器 |
| `CrystalViewer` | `crystal_viewer.py` | matplotlib 3D 球棍模型渲染 |
| `OrbitalTree` | `orbital_tree.py` | tkinter Treeview 轨道 checkbox 选择器 |

### 设计模式

- **适配器模式**：`CastepPDOSAdapter` 将 CASTEP 特有的聚合/查询接口转换为 `PDOSPlotter` 期望的接口（`get_data()`, `available_orbitals`, `has_spin` 等），实现 CASTEP 数据与 MS 数据的统一绑图管线。
- **依赖注入**：`InteractivePreview` 通过构造函数接收 `PDOSAnalyzer` 实例，便于测试和解耦。
- **猴子补丁**：`pdos_plotter.py` 和 `interactive_preview.py` 使用 `FontProperties(fname=path)` 对 `ax.set_title` / `ax.text` 自动注入 CJK 字体属性，解决 Windows 下中文乱码问题。

---

## License

MIT License. 详见 [LICENSE](./LICENSE) 文件。
