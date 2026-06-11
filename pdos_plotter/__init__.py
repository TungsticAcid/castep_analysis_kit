#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdos_plotter —— PDOS 态密度数据提取与绘图工具包
==================================================

支持 Materials Studio .xcd (XML) 和 CASTEP .castep_bin/.pdos_bin 两种输入格式。

核心类:
  - PDOSParser:          MS .xcd 文件解析
  - PDOSPlotter:         态密度绘图
  - PDOSAnalyzer:        态密度分析（d带中心、峰搜索、重叠面积等）
  - CastepPDOSCalculator: CASTEP 二进制 PDOS 计算
  - CastepPDOSAdapter:    CASTEP → PDOSPlotter 桥接适配器
  - PDOSGUI:             tkinter 图形界面
  - InteractivePreview:  matplotlib 交互预览

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/06/11
"""
from __future__ import annotations

from .constants import (
    ORBITAL_COLORS,
    ML_ORBITAL_COLORS,
    get_orbital_color,
    SPIN_COLORS,
    OVERLAY_PALETTE,
    ORBITAL_NAMES,
    SPECIES_ORBITAL_COUNTS,
    SPECIES_ORBITALS,
    SPECIES_INDEX_MAP,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
    SCRIPT_DIR,
    DEFAULT_PIC_DIR,
)

from .pdos_parser import PDOSParser
from .pdos_plotter import PDOSPlotter
from .pdos_analyzer import PDOSAnalyzer
from .interactive_preview import InteractivePreview
from .pdos_gui import PDOSGUI, AnalysisDialog
from .binary_io import read_all_records
from .pdos_calc import CastepPDOSCalculator, CastepPDOSAdapter
from .castep_bin_parser import CastepBinParser
from .crystal_viewer import CrystalStructure, CrystalViewer
from .orbital_tree import OrbitalTree

__all__ = [
    # 核心类
    "PDOSParser",
    "PDOSPlotter",
    "PDOSAnalyzer",
    "InteractivePreview",
    "PDOSGUI",
    "AnalysisDialog",
    # CASTEP 支持
    "CastepPDOSCalculator",
    "CastepPDOSAdapter",
    "CastepBinParser",
    "CrystalStructure",
    "CrystalViewer",
    "OrbitalTree",
    "read_all_records",
    # 常量
    "ORBITAL_COLORS",
    "ML_ORBITAL_COLORS",
    "get_orbital_color",
    "SPIN_COLORS",
    "OVERLAY_PALETTE",
    "ORBITAL_NAMES",
    "SPECIES_ORBITAL_COUNTS",
    "SPECIES_ORBITALS",
    "SPECIES_INDEX_MAP",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    "SCRIPT_DIR",
    "DEFAULT_PIC_DIR",
]
