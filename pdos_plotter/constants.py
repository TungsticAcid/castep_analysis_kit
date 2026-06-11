#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdos_plotter 全局常量定义
==========================

包含颜色方案、轨道映射表、图形默认参数等所有模块级常量。
集中管理，避免魔法数字散落各处。

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11 (重构 + CASTEP 支持)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# ============================================================
# 路径相关
# ============================================================

# 当前脚本所在目录的绝对路径
SCRIPT_DIR: Path = Path(__file__).resolve().parent
# 默认图片输出目录（脚本所在目录下的 pic 子目录）
DEFAULT_PIC_DIR: Path = SCRIPT_DIR / "pic"

# ============================================================
# 颜色方案
# ============================================================

# 轨道颜色映射表：用于不同轨道的曲线/填充颜色区分
#   s 轨道 → 红色，p 轨道 → 蓝色，d 轨道 → 绿色
#   f 轨道 → 紫色，sum/total → 黑色
ORBITAL_COLORS: Dict[str, str] = {
    "s":     "#E74C3C",  # 红色 - s 轨道
    "p":     "#3498DB",  # 蓝色 - p 轨道
    "d":     "#2ECC71",  # 绿色 - d 轨道
    "f":     "#9B59B6",  # 紫色 - f 轨道
    "sum":   "#1A1A1A",  # 黑色 - 总态密度
    "total": "#1A1A1A",  # 黑色 - 总态密度（备用键名）
}

# m_l 分辨子轨道颜色方案
# 设计原则:
#   - p 亚轨道 (px, py, pz): 蓝色系，从深到浅便于区分
#   - d 亚轨道 (d_xy, d_yz, d_z2, d_xz, d_x2-y2): 绿色系，从深到浅
#   - s 轨道和汇总轨道沿用 ORBITAL_COLORS 的配色
ML_ORBITAL_COLORS: Dict[str, str] = {
    # s 轨道（单一，不区分 m_l）
    "s":       "#E74C3C",  # 红色
    # p 亚轨道 —— 蓝色系（从深到浅）
    "px":      "#2980B9",  # 深蓝 (px)
    "py":      "#3498DB",  # 中蓝 (py)
    "pz":      "#85C1E9",  # 浅蓝 (pz)
    # d 亚轨道 —— 绿色系（按能量顺序从深到浅）
    "d_xy":    "#27AE60",  # 深绿 (d_xy)
    "d_yz":    "#2ECC71",  # 中绿 (d_yz)
    "d_z2":    "#1E8449",  # 暗绿 (d_z2, 最暗以突出其独特性)
    "d_xz":    "#58D68D",  # 浅绿 (d_xz)
    "d_x2-y2": "#82E0AA",  # 淡绿 (d_x2-y2)
    # f 亚轨道 —— 紫色系（预留，按需扩展）
    # 汇总轨道
    "sum":     "#1A1A1A",
    "total":   "#1A1A1A",
}


def get_orbital_color(label: str) -> str:
    """
    从轨道标签中智能提取对应颜色。

    解析策略（按优先级）:
      1. 精确匹配 ML_ORBITAL_COLORS 键（如 "d_xy", "px"）
      2. 精确匹配 ORBITAL_COLORS 键（如 "s", "p", "d", "f"）
      3. 从复合标签提取轨道名后缀匹配（"Ni-d_xy" → "d_xy"）
      4. 提取角量子数字符匹配（"Ni-p" → "p"）
      5. 回退到 COLOR_FALLBACK

    参数
    ----
    label : str
        裸标签（不含自旋后缀），如 "Ni-d_xy", "d_xy", "s", "Ni-p"。

    返回
    -------
    color : str
        十六进制颜色字符串。
    """
    ll: str = label.lower().strip()

    # 1) 精确匹配 m_l 轨道颜色
    if ll in ML_ORBITAL_COLORS:
        return ML_ORBITAL_COLORS[ll]

    # 2) 精确匹配角动量颜色
    if ll in ORBITAL_COLORS:
        return ORBITAL_COLORS[ll]

    # 3) 复合标签: "Ni-d_xy" → 提取 "d_xy" 匹配
    #    按长度降序排序，避免 "d_xy" 中包含的短键误匹配
    _ml_keys = [k for k in ML_ORBITAL_COLORS if k not in ("sum", "total")]
    for suffix in sorted(_ml_keys, key=len, reverse=True):
        if ll.endswith('-' + suffix) or ll.endswith('_' + suffix):
            return ML_ORBITAL_COLORS[suffix]

    # 4) 提取角量子数字符: "Ni-p" → "p"
    for sep in ('-', '_'):
        if sep in ll:
            token: str = ll.rsplit(sep, 1)[-1]
            if token and token[0] in ORBITAL_COLORS:
                return ORBITAL_COLORS[token[0]]
            break

    # 5) 回退
    return COLOR_FALLBACK


# 自旋颜色映射表：用于自旋极化图中区分 α 和 β 自旋
#   alpha (↑) → 红色，beta (↓) → 蓝色
SPIN_COLORS: Dict[str, str] = {
    "alpha": "#E74C3C",  # 红 - α 自旋（向上）
    "beta":  "#3498DB",  # 蓝 - β 自旋（向下）
}

# 多文件叠加模式调色板：用于区分来自不同文件/轨道的系列
# 颜色按顺序循环分配，确保各系列视觉上可区分
OVERLAY_PALETTE: List[str] = [
    "#E74C3C",  # 红
    "#3498DB",  # 蓝
    "#2ECC71",  # 绿
    "#9B59B6",  # 紫
    "#F39C12",  # 橙
    "#1ABC9C",  # 青
    "#E67E22",  # 深橙
    "#2980B9",  # 深蓝
    "#8E44AD",  # 深紫
    "#27AE60",  # 深绿
    "#C0392B",  # 暗红
    "#16A085",  # 深青
    "#D35400",  # 棕橙
    "#7F8C8D",  # 灰
    "#2C3E50",  # 深灰蓝
]

# 功能颜色（交互预览 / GUI 使用的标识色）
COLOR_PEAK_MODE = "#E74C3C"       # 识峰模式标题颜色
COLOR_OVERLAP_A = "#E74C3C"       # 重叠分析系列 A 标记色
COLOR_OVERLAP_B = "#3498DB"       # 重叠分析系列 B 标记色
COLOR_OVERLAP_RESULT = "#2ECC71"  # 重叠分析结果标识
COLOR_SERIES_INDICATOR = "#2980B9"  # 系列指示器文字色
COLOR_SERIES_INDICATOR_BG = "#EBF5FB"  # 系列指示器背景色
COLOR_PEAK_RESULT_BG = "#FFFDE7"    # 峰结果文本框背景
COLOR_OVERLAP_RESULT_BG = "#E8F5E9"  # 重叠结果文本框背景
COLOR_FALLBACK = "#333333"          # 后备颜色

# ============================================================
# CASTEP 轨道相关常量
# ============================================================

# CASTEP cubic harmonics 基组轨道名称（单原子内的局部顺序）
# 顺序: [s] [px] [py] [pz] [d_xy] [d_yz] [d_z2] [d_xz] [d_x2-y2]
ORBITAL_NAMES: List[str] = [
    's', 'px', 'py', 'pz',
    'd_xy', 'd_yz', 'd_z2', 'd_xz', 'd_x2-y2',
]

# 每个元素的轨道数量
SPECIES_ORBITAL_COUNTS: Dict[str, int] = {
    'Ni': 9,   # s, px, py, pz, d_xy, d_yz, d_z2, d_xz, d_x2-y2
    'C':  4,   # s, px, py, pz
    'H':  1,   # s
}

# 每个元素包含的具体轨道名称列表
SPECIES_ORBITALS: Dict[str, List[str]] = {
    'Ni': ORBITAL_NAMES,                          # 全部 9 个
    'C':  ['s', 'px', 'py', 'pz'],                # 前 4 个
    'H':  ['s'],                                   # 仅 s
}

# 元素索引 → 元素名称（对应 .pdos_bin 记录6 的值域 {1,2,3}）
SPECIES_INDEX_MAP: Dict[int, str] = {
    1: 'Ni',
    2: 'C',
    3: 'H',
}

# 轨道名 → 角量子数
ORBITAL_TO_L: Dict[str, int] = {
    's': 0,
    'px': 1, 'py': 1, 'pz': 1,
    'd_xy': 2, 'd_yz': 2, 'd_z2': 2, 'd_xz': 2, 'd_x2-y2': 2,
}

# 角量子数 → 字母名
L_TO_NAME: Dict[int, str] = {
    0: 's', 1: 'p', 2: 'd', 3: 'f',
}

# 轨道排序优先级（用于 available_orbitals 排序: s < p < d < f）
ORBITAL_SORT_ORDER: Dict[str, int] = {"s": 0, "p": 1, "d": 2, "f": 3}

# ============================================================
# 自旋映射（消除 CLI/GUI 中重复定义的字典）
# ============================================================

# 自旋标识符 → 希腊字母显示名
SPIN_KEY_TO_GREEK: Dict[str, str] = {
    "alpha": "α",
    "beta":  "β",
    "sum":   "α+β",
    "both":  "α/β",
}

# 自旋 CLI 简写 → 完整标识符
SPIN_SHORT_TO_KEY: Dict[str, str] = {
    "a": "alpha", "alpha": "alpha",
    "b": "beta",  "beta":  "beta",
    "s": "sum",   "sum":   "sum",
    "both": "both",
}

# ============================================================
# 图形默认参数
# ============================================================

# 默认图片尺寸（宽, 高）单位英寸
DEFAULT_FIGSIZE: tuple = (8, 6)

# 默认保存分辨率 DPI
DEFAULT_DPI: int = 300

# 默认 GUI 窗口尺寸
DEFAULT_WINDOW_SIZE: str = "680x800"

# 费米能级参考线样式
FERMI_LINE_STYLE: dict = {
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.8,
    "alpha": 0.7,
}

# 零基线样式
ZERO_LINE_STYLE: dict = {
    "color": "gray",
    "linestyle": "-",
    "linewidth": 0.5,
    "alpha": 0.5,
}

# ============================================================
# 分析参数默认值
# ============================================================

# 高斯 FWHM 与标准差换算系数（FWHM = 2√(2ln2)·σ ≈ 2.355σ）
GAUSSIAN_FWHM_FACTOR: float = 2.355

# DOS 噪声阈值（相对于最大值的比例）
NOISE_THRESHOLD_RATIO: float = 0.001

# 安全小量，防止除零
EPSILON: float = 1e-30

# 分析报告显示宽度（字符数）
REPORT_WIDTH: int = 44

# 默认峰搜索参数
DEFAULT_PEAK_MIN_HEIGHT: float = 0.0
DEFAULT_PEAK_MIN_DISTANCE: float = 0.1  # eV

# 默认积分区间（宽范围，覆盖几乎所有态）
DEFAULT_INTEGRAL_EMIN: float = -100.0
DEFAULT_INTEGRAL_EMAX: float = 100.0

# 最近文件最大数量
MAX_RECENT_FILES: int = 10

# 最近文件记录路径
RECENT_FILES_PATH: Path = Path.home() / ".pdos_plotter_recent.json"

# ============================================================
# 文件相关常量
# ============================================================

# 默认输出文件名
DEFAULT_OUTPUT_FILENAME: str = "pdos_plot.png"
DEFAULT_OVERLAY_FILENAME: str = "multi_overlay.png"

# 支持的文件扩展名
XCD_EXTENSION: str = ".xcd"
CASTEP_BIN_SUFFIX: str = "_DOS.castep_bin"
PDOS_BIN_SUFFIX: str = "_DOS.pdos_bin"

# 分析报告等宽字体
REPORT_FONT: str = "Consolas"
