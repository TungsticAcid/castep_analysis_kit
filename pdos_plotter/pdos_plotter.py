#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDOS 态密度数据提取与绘图工具
================================

功能概述:
  - 解析 Materials Studio 导出的 PDOS.xcd 文件（XML 格式）
  - 自动检测文件中的轨道类型（s / p / d / f）和自旋极化（alpha / beta）
  - 支持三种绑图模式：
      1. 自旋极化模式（α↑ / β↓）
      2. 轨道分别显示模式
      3. 总态密度 (TDOS) 模式
  - 提供 GUI（tkinter）和 CLI（argparse）两种交互方式
  - GUI 运行时所有日志信息同步输出到命令行终端
  - 图片保存路径可自定义，默认保存到 ./pic/ 目录

文件格式说明:
  .xcd 文件是 Materials Studio 导出的 XML 格式图表数据文件，结构如下：
    <XCD Version="20.1" NumCharts="1">
      <CHART_2D>
        <DATA_2D NumSeries="N">
          <SERIES_2D UniqueID="0" Name="系列名称" NumPoints="M">
            <POINT_2D XY="能量,DOS值"/>
            ...
          </SERIES_2D>
        </DATA_2D>
      </CHART_2D>
    </XCD>

  系列命名模式（共5种常见情况）：
    - 自旋极化 + s/p/d 全轨道 (8系列): s alpha, s beta, p alpha, p beta, d alpha, d beta, Sum alpha, Sum beta
    - 自旋极化 + 仅d轨道   (4系列): d alpha, d beta, Sum alpha, Sum beta
    - 非自旋极化 + s/p/d   (4系列): s, p, d, Sum
    - 非自旋极化 + 仅总量   (1系列): total
    - 非自旋极化 + 总量    (1系列): total

用法示例:
  GUI 模式（默认）:
    python pdos_plotter.py
    python pdos_plotter.py -f "path/to/PDOS.xcd"

  CLI 总态密度:
    python pdos_plotter.py -f "PDOS.xcd" --total --no-gui -o output.png

  CLI 自旋极化:
    python pdos_plotter.py -f "PDOS.xcd" --spin --orbitals s,p,d --no-gui

  CLI 轨道分别显示:
    python pdos_plotter.py -f "PDOS.xcd" --orbitals s,p,d --no-gui -o output.png

依赖:
  - Python 标准库: xml.etree.ElementTree, tkinter, argparse, os, sys, pathlib
  - 第三方库: matplotlib, numpy

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
"""

from __future__ import annotations

# ---------- 标准库导入 ----------
import os
import json
import sys

# ---- Windows 终端中文编码修复 ----
# 在 Windows 系统上将 stdout/stderr 的输出编码统一设置为 UTF-8，
# 解决中文日志在命令行中显示为乱码的问题。
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------- 第三方库导入 ----------
import matplotlib
# 显式指定 matplotlib 后端为 TkAgg，确保与 tkinter GUI 兼容
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 全局常量定义
# ============================================================

# --- 路径相关 ---
# 当前脚本所在目录的绝对路径
SCRIPT_DIR: Path = Path(__file__).resolve().parent
# 默认图片输出目录（脚本所在目录下的 pic 子目录）
DEFAULT_PIC_DIR: Path = SCRIPT_DIR / "pic"

# --- 颜色方案 ---
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

# 自旋颜色映射表：用于自旋极化图中区分 α 和 β 自旋
#   alpha (↑) → 红色，beta (↓) → 蓝色
SPIN_COLORS: Dict[str, str] = {
    "alpha": "#E74C3C",  # 红 - α 自旋（向上）
    "beta":  "#3498DB",  # 蓝 - β 自旋（向下）
}

# --- matplotlib 全局配置 ---
# 设置中文字体回退方案，确保中文标题/标签正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
# 解决负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# PDOSParser 类
# ============================================================
class PDOSParser:
    """
    Materials Studio .xcd 格式 PDOS 文件解析器。

    核心功能:
      1. 使用 xml.etree.ElementTree 解析 XML 结构
      2. 提取每条 SERIES_2D 中的能量-DOS 数据点对
      3. 自动检测文件是否包含自旋极化（通过检查系列名中是否含 "alpha"/"beta"）
      4. 自动识别可用的轨道类型（s, p, d, f, sum）

    使用方式:
      parser = PDOSParser("path/to/PDOS.xcd")
      parser.parse()                        # 执行解析
      data = parser.get_data(orbitals=["s", "d"], spin="alpha")  # 提取指定数据
      total = parser.get_total_dos()        # 获取总态密度
    """

    def __init__(self, filepath: str) -> None:
        """
        初始化 PDOS 解析器。

        参数
        ----
        filepath : str
            .xcd 文件的完整路径（绝对路径或相对路径均可）。
        """
        # 文件路径和文件名
        self.filepath: str = filepath
        self.filename: str = os.path.basename(filepath)

        # raw_data 字典结构：
        #   { "系列名称": (energies_ndarray, dos_ndarray), ... }
        # 例如: {"s alpha": (array([-10, -9.9, ...]), array([0, 0.1, ...]))}
        self.raw_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # 自旋极化检测标志：True 表示文件中同时存在 alpha 和 beta 系列
        self.has_spin: bool = False

        # 可用轨道列表，如 ["s", "p", "d", "sum"]
        self.available_orbitals: List[str] = []
        # 可用自旋方向列表，如 ["alpha", "beta"]；非自旋极化时为空列表
        self.available_spins: List[str] = []

    # ----------------------------------------------------------
    # 主解析入口
    # ----------------------------------------------------------
    def parse(self) -> None:
        """
        执行完整的文件解析流程:
          1. 解析 XML 结构，遍历所有 SERIES_2D 节点
          2. 提取每个系列的名称和数据点
          3. 自动检测轨道和自旋结构
          4. 打印解析摘要到命令行
        """
        print(f"[INFO] 正在解析文件: {self.filepath}")

        # ---- 第1步：解析 XML 文件 ----
        # ET.parse 会自动处理文件编码
        tree = ET.parse(self.filepath)
        root = tree.getroot()  # <XCD> 根节点

        # ---- 第2步：定位 DATA_2D 节点 ----
        # XML 路径为: XCD → CHART_2D → DATA_2D
        chart = root.find("CHART_2D")
        if chart is None:
            raise ValueError(f"文件 '{self.filename}' 中未找到 CHART_2D 节点，请确认这是有效的 .xcd 文件。")

        data_2d = chart.find("DATA_2D")
        if data_2d is None:
            raise ValueError(f"文件 '{self.filename}' 中未找到 DATA_2D 节点，请确认这是有效的 .xcd 文件。")

        # NumSeries 属性表示该文件包含多少条数据系列
        num_series = int(data_2d.get("NumSeries", "0"))
        print(f"[INFO] 检测到 {num_series} 条数据系列")

        # ---- 第3步：遍历每条 SERIES_2D 并提取数据 ----
        for series in data_2d.findall("SERIES_2D"):
            # 系列名称
            name: str = series.get("Name", "").strip()
            if not name:
                print("[WARN] 发现无名系列，已跳过。")
                continue

            # 收集该系列下所有 POINT_2D 的 XY 属性值
            points: List[str] = []
            for pt in series.findall("POINT_2D"):
                xy_str = pt.get("XY", "")
                if xy_str:
                    points.append(xy_str)

            # 将 XY 字符串列表解析为 numpy 数组
            energies, dos = self._parse_points(points)
            # 存入 raw_data 字典
            self.raw_data[name] = (energies, dos)
            print(f"[INFO]   - 系列 \"{name}\": {len(points)} 个数据点, 能量范围 [{energies.min():.4f}, {energies.max():.4f}]")

        # ---- 第4步：自动检测文件结构和轨道类型 ----
        self._detect_structure()

        # ---- 第5步：打印摘要信息 ----
        self._print_summary()

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------
    def _parse_points(self, points: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 POINT_2D 的 XY 字符串列表解析为能量和 DOS 的 numpy 数组。

        XY 格式: "能量值,DOS值"（如 "-0.3941401221613795,0"）
        注意: 能量以 eV 为单位，DOS 以 states/eV 为单位。

        参数
        ----
        points : list of str
            "能量,DOS" 格式的字符串列表。

        返回
        ----
        (energies, dos_values) : (ndarray, ndarray)
            分别包含所有能量值和对应 DOS 值的浮点数组。
        """
        energies = []
        dos_values = []
        for xy_str in points:
            # 以逗号分割（注意：DOS值可能为负，但能量通常不会）
            parts = xy_str.split(",")
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[0]))
                    dos_values.append(float(parts[1]))
                except ValueError:
                    print(f"[WARN] 无法解析数据点: '{xy_str}'，已跳过。")
        return np.array(energies), np.array(dos_values)

    def _detect_structure(self) -> None:
        """
        自动检测文件的轨道类型和自旋极化存在性。

        检测逻辑:
          1. 遍历所有系列名称，查找含 "alpha" 和 "beta" 关键词的系列
          2. 如果同时存在 alpha 和 beta 系列 → has_spin = True
          3. 从 alpha 系列名中提取轨道标识符（strip 掉 "alpha" 后剩余的字符串）

        例如:
          "s alpha"   → 轨道 "s"
          "d beta"    → 轨道 "d"
          "Sum alpha" → 轨道 "sum"
        """
        series_names = list(self.raw_data.keys())

        # --- 检测自旋极化 ---
        # 分别筛选名称中包含 "alpha" 和 "beta" 的系列
        alpha_names = [n for n in series_names if "alpha" in n.lower()]
        beta_names = [n for n in series_names if "beta" in n.lower()]

        # 只有当两种自旋的系列都存在时，才判定为自旋极化
        self.has_spin = len(alpha_names) > 0 and len(beta_names) > 0

        if self.has_spin:
            # 自旋极化存在 → 从 alpha 系列中提取轨道类型
            self.available_spins = ["alpha", "beta"]
            for name in alpha_names:
                # 将系列名中的 "alpha" 去掉，得到纯轨道名
                # 例如 "s alpha" → "s", "Sum alpha" → "sum"
                orbital = name.lower().replace("alpha", "").strip()
                # 统一 "sum" 标记
                if orbital in ("sum", "total"):
                    orbital = "sum"
                if orbital and orbital not in self.available_orbitals:
                    self.available_orbitals.append(orbital)
        else:
            # 非自旋极化 → 直接从所有系列名中提取轨道类型
            self.available_spins = []
            for name in series_names:
                orbital = name.lower().strip()
                # 规范化："sum" 和 "total" 都统一为 "sum"
                if orbital in ("sum", "total"):
                    orbital = "sum"
                if orbital and orbital not in self.available_orbitals:
                    self.available_orbitals.append(orbital)

        # --- 对轨道列表进行排序 ---
        # 按照 s → p → d → f → sum 的顺序排列，方便 GUI 显示
        order = {"s": 0, "p": 1, "d": 2, "f": 3}
        self.available_orbitals.sort(key=lambda x: order.get(x, 99))

    def _print_summary(self) -> None:
        """在命令行打印解析结果摘要，方便用户确认文件内容。"""
        spin_info = "有自旋极化 (α/β)" if self.has_spin else "无自旋极化"
        print(f"[INFO] ──────────────────────────────")
        print(f"[INFO] 解析完成: {spin_info}")
        print(f"[INFO] 可用轨道: {self.available_orbitals}")
        if self.has_spin:
            print(f"[INFO] 自旋方向: {self.available_spins}")
        print(f"[INFO] ──────────────────────────────")

    # ----------------------------------------------------------
    # 数据提取方法
    # ----------------------------------------------------------
    def get_data(
        self,
        orbitals: Optional[List[str]] = None,
        spin: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        根据指定的轨道和自旋条件筛选数据。

        筛选逻辑:
          1. 先按轨道匹配：检查系列名中是否包含目标轨道关键字
          2. 再按自旋匹配：如果指定了 spin，检查系列名中是否包含对应自旋关键字
          3. 标签生成：有自旋时自动附加 (α) 或 (β) 后缀

        参数
        ----
        orbitals : list of str, 可选
            要获取的轨道列表，如 ["s", "d"]。
            传入 None 表示获取所有可用轨道。
        spin : str, 可选
            自旋方向筛选，"alpha" 或 "beta"。
            传入 None 表示不按自旋筛选（非自旋极化或获取全部）。

        返回
        ----
        result : dict
            键为数据标签（如 "s (α)"），值为 (能量数组, DOS数组) 元组。
        """
        # 默认获取所有可用轨道
        if orbitals is None:
            orbitals = self.available_orbitals

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for name, (energies, dos) in self.raw_data.items():
            name_lower = name.lower().strip()

            # ---- 第1步：匹配轨道 ----
            orbital_match = None
            for orb in orbitals:
                if orb == "sum":
                    # "sum" 需要同时匹配 "sum" 和 "total"
                    if "sum" in name_lower or "total" in name_lower:
                        orbital_match = "sum"
                        break
                else:
                    # 将系列名按空格拆分，检查目标轨道是否在 token 列表中
                    # 例如 "s alpha" → ["s", "alpha"]，检查 "s" 是否在其中
                    # 这样做可以避免 "sum" 误匹配 "s"
                    tokens = name_lower.split()
                    if orb in tokens:
                        orbital_match = orb
                        break

            # 如果该系列不属于目标轨道，跳过
            if orbital_match is None:
                continue

            # ---- 第2步：匹配自旋 ----
            if self.has_spin and spin is not None:
                # 如果指定了自旋，但当前系列不含该自旋关键词，则跳过
                if spin.lower() not in name_lower:
                    continue

            # 如果文件有自旋极化但用户未指定自旋，默认只取 alpha（避免数据重复）
            if self.has_spin and spin is None:
                if "beta" in name_lower:
                    continue

            # ---- 第3步：构造显示标签 ----
            if self.has_spin:
                # 根据系列名中的自旋标识符附加后缀
                if "alpha" in name_lower:
                    label = f"{orbital_match} (α)"
                elif "beta" in name_lower:
                    label = f"{orbital_match} (β)"
                else:
                    label = orbital_match
            else:
                # 非自旋极化 → sum 统一显示为 "Sum"
                label = orbital_match if orbital_match != "sum" else "Sum"

            result[label] = (energies, dos)

        return result

    def get_total_dos(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取总态密度数据（Sum 或 total 系列）。

        遍历 raw_data 查找名称中含有 "sum" 或 "total" 的第一条系列。
        对于自旋极化文件，会同时返回 "Sum alpha" 系列（通常 alpha 和 beta
        的 Sum 应相同，因为 Sum 是各轨道的总和）。

        返回
        ----
        (energies, dos) : tuple of ndarray, 或 None
            总态密度的能量和 DOS 值。如果未找到则返回 None。
        """
        for name, (energies, dos) in self.raw_data.items():
            name_lower = name.lower().strip()
            # 优先匹配 "sum" 关键字
            if "sum" in name_lower or "total" in name_lower:
                return energies, dos
        return None

    def get_energy_range(self) -> Tuple[float, float]:
        """
        获取所有数据系列的能量范围（最小值和最大值）。

        返回
        ----
        (emin, emax) : (float, float)
            全部数据中的最小能量和最大能量值 (eV)。
        """
        e_min = float("inf")
        e_max = float("-inf")
        for energies, _ in self.raw_data.values():
            if len(energies) > 0:
                e_min = min(e_min, energies.min())
                e_max = max(e_max, energies.max())
        return e_min, e_max


# ============================================================
# PDOSPlotter 类 —— 态密度数据绑图
# ============================================================
class PDOSPlotter:
    """
    PDOS 数据绘图器。

    提供三种绘图模式:
      1. plot_spin_polarized() —— 自旋极化模式
         α 和 β 自旋都在 y>0 同侧显示，使用不同颜色和线型区分：
         α（↑）实线 + 填充，β（↓）虚线 + 斜线填充。
         Materials Studio 导出文件中的 beta 系列值已经是负数，
         绘图时自动取绝对值以在 y>0 方向显示。

      2. plot_orbitals() —— 轨道分别显示模式
         所有选中的轨道都在 y>0 半轴分别绘制，用颜色区分不同轨道。

      3. plot_total() —— 总态密度模式
         只绘制 Sum/Total 系列的态密度曲线。

    公用功能:
      - save_figure() 保存图片到指定路径
      - 自动添加费米能级 (E=0) 参考线
    """

    def __init__(self, parser: PDOSParser) -> None:
        """
        初始化绘图器。

        参数
        ----
        parser : PDOSParser
            已完成解析的 PDOSParser 实例，包含所有态密度数据。
        """
        self.parser = parser

    # ----------------------------------------------------------
    # 绘图方法一：自旋极化模式
    # ----------------------------------------------------------
    def plot_spin_polarized(
        self,
        orbitals: Optional[List[str]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制自旋极化态密度图。

        绑图约定:
          - α 和 β 自旋都在 y>0 方向同侧显示
          - α 自旋（↑）: 实线 + 实心填充
          - β 自旋（↓）: 虚线 + 斜线花纹填充
          - 每条轨道的 α 和 β 使用相同的颜色系（α 较深、β 较浅）
          - 注意：Materials Studio 导出的 beta DOS 值通常已经是负数，
            绘图时自动取绝对值处理

        参数
        ----
        orbitals : list of str, 可选
            要显示的轨道列表，如 ["s", "p", "d"]。
            默认使用所有非 sum 轨道。
        energy_range : (float, float), 可选
            能量范围 (emin, emax)，单位 eV。
            默认使用全部数据范围。
        title : str, 可选
            图表标题。默认使用文件名。
        figsize : (float, float)
            图片尺寸（宽, 高），单位英寸。

        返回
        ----
        fig, ax : matplotlib Figure 和 Axes 对象
            可用于后续的保存、显示或进一步修改。
        """
        # 默认使用所有非 sum 轨道
        if orbitals is None:
            orbitals = [o for o in self.parser.available_orbitals if o != "sum"]

        # 创建画布和坐标轴
        fig, ax = plt.subplots(figsize=figsize)

        # 对每个轨道在同一 y>0 侧分别绘制 α 和 β
        for orb in orbitals:
            base_color = ORBITAL_COLORS.get(orb, "#333333")

            # --- α 自旋（实线 + 实心填充） ---
            data_alpha = self.parser.get_data(orbitals=[orb], spin="alpha")
            for label, (e, dos) in data_alpha.items():
                # α 值通常为正，直接使用
                dos_alpha = np.abs(dos)  # 保险起见取绝对值
                ax.fill_between(e, 0, dos_alpha, alpha=0.3, color=base_color)
                ax.plot(e, dos_alpha, color=base_color, linewidth=1.2, label=label)

            # --- β 自旋（虚线 + 斜线填充） ---
            # Materials Studio 文件中 beta DOS 已经是负数，取绝对值后画在 y>0
            data_beta = self.parser.get_data(orbitals=[orb], spin="beta")
            for label, (e, dos) in data_beta.items():
                dos_beta = np.abs(dos)  # 取绝对值（MS 文件中 beta 是负数）
                # 使用稍浅的颜色或同色但虚线以区分 α/β
                beta_color = self._lighten_color(base_color, factor=0.6)
                # 斜线花纹填充 α 和 β 不重叠的区域
                ax.fill_between(
                    e, 0, dos_beta,
                    alpha=0.25, color=beta_color,
                    hatch="////", edgecolor=beta_color, linewidth=0,
                )
                # 虚线绘制 β 曲线
                ax.plot(e, dos_beta, color=beta_color, linewidth=1.0,
                       linestyle="--", dashes=(4, 2), label=label)

        # --- 参考线 ---
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # --- 样式 ---
        self._style_plot(ax, title, energy_range, xlabel="Energy (eV)", ylabel="PDOS (states/eV)")
        ax.set_ylim(bottom=0)

        return fig, ax

    @staticmethod
    def _lighten_color(hex_color: str, factor: float = 0.5) -> str:
        """
        将十六进制颜色变浅。

        通过 RGB 各通道向白色（255）方向插值实现颜色变浅效果，
        用于区分同一轨道的 α 和 β 自旋。

        参数
        ----
        hex_color : str
            原始颜色，格式如 "#E74C3C"。
        factor : float
            变浅系数，0~1 之间。值越大颜色越浅，1.0 = 白色。

        返回
        ----
        lightened : str
            变浅后的十六进制颜色字符串。
        """
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # 向 255 方向插值
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ----------------------------------------------------------
    # 绘图方法二：总态密度模式
    # ----------------------------------------------------------
    def plot_total(
        self,
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制总态密度 (TDOS) 图。

        优先使用 Sum/Total 系列；如果文件没有明确的 Sum 系列，
        会自动将所有轨道数据加和作为总态密度。

        参数
        ----
        energy_range : (float, float), 可选
            能量范围 (emin, emax)。
        title : str, 可选
            图表标题。
        figsize : (float, float)
            图片尺寸。

        返回
        ----
        fig, ax : matplotlib Figure 和 Axes 对象
        """
        fig, ax = plt.subplots(figsize=figsize)

        # --- 尝试获取 Sum/Total 系列 ---
        total_dos = self.parser.get_total_dos()
        if total_dos is not None:
            e, dos = total_dos
            # 填充区域
            ax.fill_between(e, 0, dos, alpha=0.35, color=ORBITAL_COLORS["sum"])
            # 曲线
            ax.plot(e, dos, color=ORBITAL_COLORS["sum"], linewidth=1.0, label="Total DOS")
        else:
            # --- 回退方案：手动加和所有非 sum 轨道 ---
            print("[WARN] 未找到 Total/Sum 系列，将对所有轨道数据进行加和。")
            all_data = self.parser.get_data(spin=None)  # 获取所有数据（不按自旋筛选）
            if all_data:
                ref_energies = None
                summed_dos = None
                for label, (e, dos) in all_data.items():
                    # 跳过自身就是 sum 的数据（避免重复加和）
                    if "sum" in label.lower():
                        continue
                    if ref_energies is None:
                        ref_energies = e
                        summed_dos = dos.copy()  # 使用 copy 避免修改原数据
                    elif len(e) == len(ref_energies):
                        summed_dos += dos
                if summed_dos is not None:
                    ax.fill_between(ref_energies, 0, summed_dos, alpha=0.35,
                                   color=ORBITAL_COLORS["sum"])
                    ax.plot(ref_energies, summed_dos, color=ORBITAL_COLORS["sum"],
                           linewidth=1.0, label="Total DOS (加和)")

        # --- 参考线 ---
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # --- 样式 ---
        self._style_plot(ax, title, energy_range, xlabel="Energy (eV)", ylabel="DOS (states/eV)")
        # y 轴从 0 开始
        ax.set_ylim(bottom=0)

        return fig, ax

    # ----------------------------------------------------------
    # 绘图方法三：轨道分别显示模式
    # ----------------------------------------------------------
    def plot_orbitals(
        self,
        orbitals: Optional[List[str]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (8, 6),
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制指定轨道的态密度图（非自旋极化/轨道分离模式）。

        所有轨道都在 y>0 方向显示，用不同颜色区分。

        参数
        ----
        orbitals : list of str, 可选
            要显示的轨道列表，默认使用所有非 sum 轨道。
        energy_range : (float, float), 可选
            能量范围 (emin, emax)。
        title : str, 可选
            图表标题。
        figsize : (float, float)
            图片尺寸。

        返回
        ----
        fig, ax : matplotlib Figure 和 Axes 对象
        """
        if orbitals is None:
            orbitals = [o for o in self.parser.available_orbitals if o != "sum"]

        fig, ax = plt.subplots(figsize=figsize)

        # 获取数据并绘图
        data = self.parser.get_data(orbitals=orbitals, spin=None)
        for label, (e, dos) in data.items():
            # 从标签中提取轨道名（处理 "s (α)" 这样的格式）
            orb = label.split()[0] if " " in label else label
            orb_key = orb.lower()
            # 获取该轨道对应的颜色
            color = ORBITAL_COLORS.get(orb_key, "#333333")
            # 绘制曲线
            ax.plot(e, dos, color=color, linewidth=1.2, label=label)
            # 半透明填充曲线下方区域
            ax.fill_between(e, 0, dos, alpha=0.2, color=color)

        # --- 参考线 ---
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # --- 样式 ---
        self._style_plot(ax, title, energy_range, xlabel="Energy (eV)", ylabel="PDOS (states/eV)")
        ax.set_ylim(bottom=0)

        return fig, ax

    # ----------------------------------------------------------
    # 公用：图表样式设置
    # ----------------------------------------------------------
    def _style_plot(
        self,
        ax: plt.Axes,
        title: Optional[str],
        energy_range: Optional[Tuple[float, float]],
        xlabel: str,
        ylabel: str,
    ) -> None:
        """
        统一设置图表的标题、轴标签、图例、网格等样式。

        参数
        ----
        ax : matplotlib Axes
            要设置样式的坐标轴对象。
        title : str or None
            图表标题。为 None 时使用文件名。
        energy_range : (float, float) or None
            能量显示范围。为 None 时不限制。
        xlabel : str
            x 轴标签文字。
        ylabel : str
            y 轴标签文字。
        """
        # 设置标题：用户指定 > 文件名
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            ax.set_title(self.parser.filename, fontsize=12)

        # 设置轴标签
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # 设置能量显示范围
        if energy_range is not None:
            ax.set_xlim(energy_range)

        # 图例：置于右上角，半透明背景
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

        # 刻度字号
        ax.tick_params(labelsize=10)

        # 网格线：浅色虚线
        ax.grid(True, alpha=0.3, linestyle="--")

        # 自动调整布局，避免标签被裁切
        fig = ax.figure
        if fig is not None:
            fig.tight_layout()

    # ----------------------------------------------------------
    # 公用：保存图片
    # ----------------------------------------------------------
    def save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """
        将图表保存为图片文件。

        支持所有 matplotlib 支持的格式: .png, .jpg, .pdf, .svg 等。
        自动创建不存在的父目录。

        参数
        ----
        fig : matplotlib Figure
            要保存的图表对象。
        save_path : str
            图片保存的完整路径（包括文件名和扩展名）。
        """
        # 确保目标目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        # 保存图片，dpi=300 保证高质量输出
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] 图片已保存至: {save_path}")


# ============================================================
# PDOSAnalyzer 类 —— 态密度数据分析
# ============================================================
class PDOSAnalyzer:
    """
    PDOS 数据分析器，提供 d带中心、峰搜索、积分等功能。

    使用 scipy.integrate 和 scipy.signal 进行数值积分和峰检测。

    分析功能:
      - calc_band_center()      计算 d带/p带中心（加权平均能量）
      - find_peaks()            搜索态密度峰位置
      - calc_peak_area()        计算指定能量区间内的积分面积
      - calc_occupancy()        计算费米能级以下的占据态电子数
    """

    def __init__(self, parser: PDOSParser) -> None:
        """
        初始化分析器。

        参数
        ----
        parser : PDOSParser
            已解析完成的 PDOSParser 实例。
        """
        self.parser = parser

    # ----------------------------------------------------------
    # d带中心 / p带中心 计算
    # ----------------------------------------------------------
    def calc_band_center(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        method: str = "all",
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        计算指定轨道的带中心（如 d带中心、p带中心）。

        公式: ε_band = ∫(E * DOS(E) dE) / ∫(DOS(E) dE)
        即在能量区间内，以 DOS 为权重的加权平均能量。

        支持两种计算方式:
          - "all" (全部态):      对 [emin, emax] 全区间积分，反映全部电子态
          - "occupied" (占据态): 仅对 E < 0（费米能级以下）的占据态积分

        催化领域通常使用占据态 d带中心作为描述符。

        参数
        ----
        orbital : str
            轨道名称，如 "d"、"p"、"s"。
        spin : str, 可选
            自旋方向筛选（"alpha" 或 "beta"）。
            为 None 时，对于自旋极化文件默认使用 alpha。
        emin : float, 可选
            积分能量下限 (eV)。默认使用全部数据范围的最小值。
        emax : float, 可选
            积分能量上限 (eV)。默认使用全部数据范围的最大值。
        method : str
            计算方式: "all"（全部态）或 "occupied"（占据态）。
        verbose : bool
            是否打印结果。在 full_report 中统一管理输出。

        返回
        ----
        result : dict
            {"band_center": float, "integral": float, "method": str}，
            band_center 单位为 eV，integral 为 DOS 积分总面积。
        """
        data = self.parser.get_data(orbitals=[orbital], spin=spin)
        if not data:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据，无法计算带中心。")
            return {"band_center": float("nan"), "integral": float("nan"), "method": method}

        # 取第一条匹配的数据
        label, (energies, dos_values) = list(data.items())[0]
        dos_abs = np.abs(dos_values)  # beta 数据可能是负数，取绝对值

        # 限制能量范围
        mask = np.ones(len(energies), dtype=bool)
        if emin is not None:
            mask &= (energies >= emin)
        if emax is not None:
            mask &= (energies <= emax)

        # 占据态模式：额外限制 E < 0（费米能级以下）
        if method == "occupied":
            mask &= (energies < 0.0)

        e_range = energies[mask]
        d_range = dos_abs[mask]

        if len(e_range) < 2:
            print(f"[WARN] 能量范围内数据点不足，无法计算{method}带中心。")
            return {"band_center": float("nan"), "integral": float("nan"), "method": method}

        # 梯形法数值积分
        numerator = np.trapz(e_range * d_range, e_range)   # ∫ E*DOS dE
        denominator = np.trapz(d_range, e_range)            # ∫ DOS dE

        if denominator == 0:
            print(f"[WARN] {method}态 DOS 积分为零，无法计算带中心。")
            return {"band_center": float("nan"), "integral": 0.0, "method": method}

        band_center = numerator / denominator
        method_name = "占据态" if method == "occupied" else "全部态"
        spin_tag = f" ({spin})" if spin else ""
        if verbose:
            print(f"[INFO] {orbital.upper()}带中心 ({method_name}){spin_tag}: {band_center:.4f} eV")
            print(f"[INFO]   DOS 积分面积 ({method_name}): {denominator:.4f}")

        return {"band_center": band_center, "integral": denominator, "method": method}

    # ----------------------------------------------------------
    # 峰位置搜索
    # ----------------------------------------------------------
    def find_peaks(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        min_height: float = 0.0,
        min_distance: float = 0.1,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        搜索态密度图中的峰位置。

        使用 scipy.signal.find_peaks 进行一维峰检测，
        可设置最小峰高和最小峰间距来过滤噪声。

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        min_height : float
            最小峰高阈值，低于此值的峰将被忽略。
        min_distance : float
            峰之间的最小能量间距 (eV)，距离太近的峰将被合并。
        emin : float, 可选
            搜索能量下限 (eV)。
        emax : float, 可选
            搜索能量上限 (eV)。
        verbose : bool
            是否在命令行打印峰列表。在 full_report 中会统一打印，
            单独调用时可设为 True 查看详情。

        返回
        ----
        peaks : list of dict
            按峰高降序排列的峰信息列表，每个元素包含:
            {"energy": float, "dos": float, "prominence": float}
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        data = self.parser.get_data(orbitals=[orbital], spin=spin)
        if not data:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return []

        label, (energies, dos_values) = list(data.items())[0]
        dos_abs = np.abs(dos_values)

        # 限制能量范围进行峰搜索
        mask = np.ones(len(energies), dtype=bool)
        if emin is not None:
            mask &= (energies >= emin)
        if emax is not None:
            mask &= (energies <= emax)

        e_range = energies[mask]
        d_range = dos_abs[mask]

        if len(e_range) < 3:
            return []

        # 估算 min_distance 对应的数据点间隔数
        # 计算能量网格的平均步长
        avg_step = np.mean(np.diff(e_range)) if len(e_range) > 1 else 0.01
        distance_points = max(1, int(min_distance / avg_step))

        # 调用 scipy 峰检测
        peak_indices, peak_props = scipy_find_peaks(
            d_range,
            height=min_height,
            distance=distance_points,
        )

        # 整理结果：按峰高降序排列
        peak_list = []
        for idx in peak_indices:
            peak_list.append({
                "energy": float(e_range[idx]),
                "dos": float(d_range[idx]),
                "prominence": float(peak_props.get("prominences", [0])[0]) if "prominences" in peak_props else 0.0,
            })

        peak_list.sort(key=lambda p: p["dos"], reverse=True)

        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 检测到 {len(peak_list)} 个峰:")
            for i, p in enumerate(peak_list):
                print(f"[INFO]   峰{i+1}: E = {p['energy']:.4f} eV, DOS = {p['dos']:.4f}")

        return peak_list

    # ----------------------------------------------------------
    # 晶场劈裂分析
    # ----------------------------------------------------------
    def calc_crystal_field_splitting(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        peaks: Optional[List[Dict]] = None,
        min_height: float = 0.0,
        min_distance: float = 0.1,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        计算晶场劈裂能（主峰之间的能量差）。

        通过寻找态密度中两个最显著的峰，计算它们之间的能量差。
        对于 d轨道，这近似对应晶体场劈裂（如八面体场中 t2g-eg 分裂）；
        对于成键-反键分析，两峰间距可反映配体场/杂化强度。

        工作原理:
          1. 若未提供 pre-computed peaks，则调用 find_peaks() 搜索
          2. 取最高的两个峰（按 DOS 强度排序）
          3. 计算能量差 ΔE = E_high - E_low
          4. 峰数 < 2 时返回 None

        注意:
          - 该分析基于总 d 带 PDOS，不是各个 d 轨道的单独投影。
            如果文件中有 dxy, dyz, dz2, dxz, dx2-y2 的单独系列，
            则可以通过它们各自的带中心差来更精确地计算晶场劈裂。
          - min_distance 默认 0.1 eV，避免把同一个宽峰的尖刺算成两个峰。

        参数
        ----
        orbital : str
            轨道名称，通常为 "d"。
        spin : str, 可选
            自旋方向。
        peaks : list of dict, 可选
            预先计算好的峰列表（避免重复调用 find_peaks）。
            传入 None 则内部自动搜索。
        min_height : float
            峰搜索的最小高度阈值。
        min_distance : float
            峰之间的最小间距 (eV)。
        emin : float, 可选
            搜索能量下限。
        emax : float, 可选
            搜索能量上限。

        返回
        ----
        result : dict
            {
                "splitting": float or None,   # 劈裂能 ΔE (eV)
                "peak_lower": dict or None,   # 低能峰 {energy, dos}
                "peak_upper": dict or None,   # 高能峰 {energy, dos}
                "num_peaks_found": int,       # 参与分析的峰总数
            }
        """
        # 若未提供预计算的峰列表，则自行搜索
        if peaks is None:
            peaks = self.find_peaks(
                orbital=orbital, spin=spin,
                min_height=min_height, min_distance=min_distance,
                emin=emin, emax=emax,
            )

        spin_tag = f" ({spin})" if spin else ""

        if len(peaks) < 2:
            if verbose:
                print(f"[INFO] {orbital.upper()}轨道{spin_tag} 晶场劈裂: 峰数量不足 ({len(peaks)})，"
                      f"无法计算劈裂能。")
            return {
                "splitting": None,
                "peak_lower": peaks[0] if peaks else None,
                "peak_upper": None,
                "num_peaks_found": len(peaks),
            }

        # 取最高的两个峰，按能量排序（低能 → 高能）
        top_two = sorted(peaks[:2], key=lambda p: p["energy"])
        peak_lower = top_two[0]  # 能量较低
        peak_upper = top_two[1]  # 能量较高

        # 劈裂能 = 高能峰能量 - 低能峰能量
        splitting = peak_upper["energy"] - peak_lower["energy"]

        if verbose:
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 晶场劈裂分析:")
            print(f"[INFO]   低能峰: E = {peak_lower['energy']:.4f} eV, DOS = {peak_lower['dos']:.4f}")
            print(f"[INFO]   高能峰: E = {peak_upper['energy']:.4f} eV, DOS = {peak_upper['dos']:.4f}")
            print(f"[INFO]   劈裂能 ΔE = {splitting:.4f} eV")

        return {
            "splitting": splitting,
            "peak_lower": peak_lower,
            "peak_upper": peak_upper,
            "num_peaks_found": len(peaks),
        }

    # ----------------------------------------------------------
    # d带宽度分析
    # ----------------------------------------------------------
    def calc_band_width(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        计算轨道带宽（以 DOS 为权重的标准差和半高宽 FWHM）。

        不使用简单的能量范围 E_max - E_min，而是计算 DOS 加权的统计分布宽度，
        更能反映态密度的实际展宽程度。

        计算指标:
          - std (标准差):  σ = sqrt(∫(E-ε_center)²·DOS dE / ∫DOS dE)
          - FWHM (半高宽): 2.355 × σ（假设高斯分布）
          - energy_span:    数据中 DOS > 0 的能量跨度

        参数
        ----
        orbital : str
            轨道名称，如 "d"、"p"、"s"。
        spin : str, 可选
            自旋方向。
        emin : float, 可选
            分析能量下限。
        emax : float, 可选
            分析能量上限。

        返回
        ----
        result : dict
            {"std": float, "fwhm": float, "energy_span": float}
            单位均为 eV。
        """
        data = self.parser.get_data(orbitals=[orbital], spin=spin)
        if not data:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return {"std": float("nan"), "fwhm": float("nan"), "energy_span": float("nan")}

        label, (energies, dos_values) = list(data.items())[0]
        dos_abs = np.abs(dos_values)

        # 限制能量范围
        mask = np.ones(len(energies), dtype=bool)
        if emin is not None:
            mask &= (energies >= emin)
        if emax is not None:
            mask &= (energies <= emax)

        e_range = energies[mask]
        d_range = dos_abs[mask]

        if len(e_range) < 2:
            return {"std": float("nan"), "fwhm": float("nan"), "energy_span": float("nan")}

        # ---- 加权标准差 ----
        # 先算加权平均值（带中心）
        total_dos = np.trapz(d_range, e_range)
        if total_dos == 0:
            return {"std": 0.0, "fwhm": 0.0, "energy_span": 0.0}

        band_center = np.trapz(e_range * d_range, e_range) / total_dos
        # 方差 = ∫(E-ε_d)²·DOS dE / ∫DOS dE
        variance = np.trapz((e_range - band_center) ** 2 * d_range, e_range) / total_dos
        std = float(np.sqrt(variance))
        fwhm = 2.355 * std  # 高斯半高宽近似

        # ---- 能量跨度（非零 DOS 范围） ----
        nonzero_mask = d_range > 0.001 * d_range.max()  # 忽略极小值噪声
        if nonzero_mask.any():
            energy_span = float(e_range[nonzero_mask].max() - e_range[nonzero_mask].min())
        else:
            energy_span = 0.0

        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 带宽:")
            print(f"[INFO]   标准差 σ = {std:.4f} eV")
            print(f"[INFO]   半高宽 FWHM ≈ {fwhm:.4f} eV")
            print(f"[INFO]   能量跨度 = {energy_span:.4f} eV")

        return {"std": std, "fwhm": fwhm, "energy_span": energy_span}

    # ----------------------------------------------------------
    # 自旋劈裂分析
    # ----------------------------------------------------------
    def calc_spin_splitting(
        self,
        orbital: str = "d",
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        method: str = "occupied",
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        计算自旋劈裂能（α 和 β 自旋带中心之间的能量差）。

        仅适用于自旋极化文件。分别计算 α 和 β 自旋的 d带中心，
        劈裂能 ΔE_spin = ε_d(α) - ε_d(β)，正值表示 α 带较 β 带更深。

        物理意义:
          - 交换劈裂 (exchange splitting) 反映了磁矩大小
          - 劈裂越大 → 局域磁矩越强
          - 结合晶场劈裂可分析自旋态和配位环境

        参数
        ----
        orbital : str
            轨道名称，通常为 "d"。
        emin : float, 可选
            积分能量下限 (eV)。
        emax : float, 可选
            积分能量上限 (eV)。
        method : str
            带中心计算方式: "all"（全部态）或 "occupied"（占据态）。

        返回
        ----
        result : dict
            {
                "spin_splitting": float or nan,  # 自旋劈裂能 (eV)
                "band_center_alpha": float,      # α 自旋带中心
                "band_center_beta": float,       # β 自旋带中心
            }
        """
        if not self.parser.has_spin:
            print("[INFO] 自旋劈裂分析: 非自旋极化文件，跳过。")
            return {
                "spin_splitting": float("nan"),
                "band_center_alpha": float("nan"),
                "band_center_beta": float("nan"),
            }

        # 分别计算 α 和 β 的带中心（静默模式，由 full_report 统一输出）
        bc_alpha = self.calc_band_center(
            orbital=orbital, spin="alpha", emin=emin, emax=emax, method=method,
            verbose=False,
        )
        bc_beta = self.calc_band_center(
            orbital=orbital, spin="beta", emin=emin, emax=emax, method=method,
            verbose=False,
        )

        e_alpha = bc_alpha["band_center"]
        e_beta = bc_beta["band_center"]

        if np.isnan(e_alpha) or np.isnan(e_beta):
            splitting = float("nan")
        else:
            splitting = e_alpha - e_beta  # α 带中心 − β 带中心

        if verbose:
            method_name = "占据态" if method == "occupied" else "全部态"
            print(f"[INFO] {orbital.upper()}轨道 自旋劈裂分析 ({method_name}):")
            print(f"[INFO]   α 带中心: {e_alpha:.4f} eV")
            print(f"[INFO]   β 带中心: {e_beta:.4f} eV")
            if not np.isnan(splitting):
                direction = "α 更深" if splitting < 0 else "β 更深"
                print(f"[INFO]   自旋劈裂能 ΔE_spin = {splitting:.4f} eV ({direction})")

        return {
            "spin_splitting": splitting,
            "band_center_alpha": e_alpha,
            "band_center_beta": e_beta,
        }

    # ----------------------------------------------------------
    # 峰面积计算（指定能量区间积分）
    # ----------------------------------------------------------
    def calc_peak_area(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        emin: float = -10.0,
        emax: float = 10.0,
        verbose: bool = True,
    ) -> float:
        """
        计算指定能量区间内态密度的积分面积。

        可用于量化某个峰的强度或特定能量范围内的电子态数量。

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        emin : float
            积分能量下限 (eV)。
        emax : float
            积分能量上限 (eV)。

        返回
        ----
        area : float
            积分面积（单位：states）。
        """
        data = self.parser.get_data(orbitals=[orbital], spin=spin)
        if not data:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return 0.0

        label, (energies, dos_values) = list(data.items())[0]
        dos_abs = np.abs(dos_values)

        # 限制能量范围
        mask = (energies >= emin) & (energies <= emax)
        e_range = energies[mask]
        d_range = dos_abs[mask]

        if len(e_range) < 2:
            return 0.0

        area = float(np.trapz(d_range, e_range))
        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 在 [{emin:.2f}, {emax:.2f}] eV 区间积分面积: {area:.4f}")
        return area

    # ----------------------------------------------------------
    # 费米能级以下占据态计算
    # ----------------------------------------------------------
    def calc_occupancy(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        计算费米能级（E=0）以下的占据态电子数。

        对能量 < 0 的 DOS 进行积分，反映该轨道在基态下的电子占据情况。

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。

        返回
        ----
        result : dict
            {"occupied": float, "total": float, "occupancy_ratio": float}
            occupied 为 E<0 的 DOS 积分，total 为全能量范围积分，
            occupancy_ratio 为占据比例。
        """
        data = self.parser.get_data(orbitals=[orbital], spin=spin)
        if not data:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return {"occupied": float("nan"), "total": float("nan"), "occupancy_ratio": float("nan")}

        label, (energies, dos_values) = list(data.items())[0]
        dos_abs = np.abs(dos_values)

        # E < 0 占据态
        mask_occ = energies < 0
        if mask_occ.sum() < 2:
            occupied = 0.0
        else:
            occupied = float(np.trapz(dos_abs[mask_occ], energies[mask_occ]))

        # 全范围
        total = float(np.trapz(dos_abs, energies))
        ratio = occupied / total if total > 0 else float("nan")

        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 占据态: {occupied:.4f} / {total:.4f} = {ratio:.2%}")
        return {"occupied": occupied, "total": total, "occupancy_ratio": ratio}

    # ----------------------------------------------------------
    # 综合分析报告
    # ----------------------------------------------------------
    def full_report(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        peak_emin: Optional[float] = None,
        peak_emax: Optional[float] = None,
        options: Optional[Dict[str, bool]] = None,
    ) -> Dict:
        """
        生成完整的轨道分析报告。

        包含，按输出顺序:
          1. 带中心 (全部态 + 占据态)   —— 最核心的描述符，自旋极化时区分 α/β
          2. 带宽 (标准差 / FWHM)       —— 态密度展宽
          3. 自旋劈裂 (α−β)            —— 交换劈裂/磁矩
          4. 晶场劈裂 (主峰间距)       —— 配位场效应
          5. 占据态比例 (E<0)          —— 电子占据
          6. 积分面积                   —— 总电子态数
          7. 峰列表                    —— 各峰详情（末尾，紧凑格式）

        参数
        ----
        orbital : str
            要分析的轨道名称。
        spin : str, 可选
            自旋方向。None 时若文件有自旋极化则默认用 alpha，
            同时自动展示 α/β 双自旋的带中心对比。
        emin : float, 可选
            分析能量下限（通用，用于带中心/带宽/劈裂/占据态）。
        emax : float, 可选
            分析能量上限（通用）。
        peak_emin : float, 可选
            积分面积的积分下限 (eV)，默认 -100。
        peak_emax : float, 可选
            积分面积的积分上限 (eV)，默认 100。
        options : dict, 可选
            控制各分析项的开关，如 {"band_center": True, "peaks": False, ...}。
            None 表示全部开启。支持的键:
              band_center, band_width, spin_splitting, crystal_field,
              occupancy, peak_area, peaks

        返回
        ----
        report : dict
            包含所有分析结果的字典。
        """
        if options is None:
            options = {}

        def _opt(key: str) -> bool:
            return options.get(key, True)

        W = 44  # 报告宽度
        title = f"  {orbital.upper()}轨道 态密度分析报告"
        
        def display_width(s):
            width = 0
            for c in s:
                width += 2 if ord(c) > 127 else 1
            return width
        
        title_width = display_width(title)
        padding_total = W - title_width
        padding_left = padding_total // 2
        
        # 构造不含右边界的内容部分
        content = f"║{' ' * padding_left}{title}"
        # 计算实际显示宽度（边界符在终端只占1个宽度，需要减1修正）
        actual_content_width = display_width(content) - 1
        # 右侧填充 = 总宽度 - 当前内容实际宽度 - 右边界(1)
        extra_padding = W - actual_content_width - 1
        line = f"{content}{' ' * extra_padding}║"
        
        print(f"\n╔{'═' * W}╗")
        print(line)
        print(f"╚{'═' * W}╝")

        report: Dict = {"orbital": orbital, "spin": spin}

        # ================================================================
        # 1. 带中心
        # ================================================================
        show_dual_spin = self.parser.has_spin and spin is None

        if _opt("band_center"):
            print(f"── 带中心 (band center) ──")

            # 全部态
            bc_all = self.calc_band_center(
                orbital=orbital, spin=spin, emin=emin, emax=emax, method="all",
                verbose=False,
            )
            report["band_center_all"] = bc_all["band_center"]
            report["band_center_integral_all"] = bc_all["integral"]

            # 占据态
            bc_occ = self.calc_band_center(
                orbital=orbital, spin=spin, emin=emin, emax=emax, method="occupied",
                verbose=False,
            )
            report["band_center_occupied"] = bc_occ["band_center"]
            report["band_center_integral_occupied"] = bc_occ["integral"]

            if show_dual_spin:
                # 分别计算 α 和 β 的带中心，并排展示
                bc_all_a = self.calc_band_center(
                    orbital=orbital, spin="alpha", emin=emin, emax=emax, method="all",
                    verbose=False,
                )
                bc_all_b = self.calc_band_center(
                    orbital=orbital, spin="beta", emin=emin, emax=emax, method="all",
                    verbose=False,
                )
                bc_occ_a = self.calc_band_center(
                    orbital=orbital, spin="alpha", emin=emin, emax=emax, method="occupied",
                    verbose=False,
                )
                bc_occ_b = self.calc_band_center(
                    orbital=orbital, spin="beta", emin=emin, emax=emax, method="occupied",
                    verbose=False,
                )
                d_all = bc_all_a["band_center"] - bc_all_b["band_center"]
                d_occ = bc_occ_a["band_center"] - bc_occ_b["band_center"]

                print(f"  全部态  ε = {bc_all['band_center']:+.4f} eV")
                print(f"         α: {bc_all_a['band_center']:+.4f} eV  "
                      f"β: {bc_all_b['band_center']:+.4f} eV  "
                      f"Δ(α−β) = {d_all:+.4f} eV")
                print(f"  占据态  ε = {bc_occ['band_center']:+.4f} eV")
                print(f"         α: {bc_occ_a['band_center']:+.4f} eV  "
                      f"β: {bc_occ_b['band_center']:+.4f} eV  "
                      f"Δ(α−β) = {d_occ:+.4f} eV")
            else:
                print(f"  全部态  ε = {bc_all['band_center']:+.4f} eV"
                      f"  [积分: {bc_all['integral']:.2f}]")
                print(f"  占据态  ε = {bc_occ['band_center']:+.4f} eV"
                      f"  [积分: {bc_occ['integral']:.2f}]")
        else:
            for k in ["band_center_all", "band_center_occupied",
                      "band_center_integral_all", "band_center_integral_occupied"]:
                report[k] = float("nan")

        # ================================================================
        # 2. 带宽
        # ================================================================
        if _opt("band_width"):
            bw = self.calc_band_width(
                orbital=orbital, spin=spin, emin=emin, emax=emax, verbose=False,
            )
            report["band_width_std"] = bw["std"]
            report["band_width_fwhm"] = bw["fwhm"]
            report["band_energy_span"] = bw["energy_span"]
            print(f"── 带宽 ──")
            print(f"  σ = {bw['std']:.4f} eV  │  FWHM = {bw['fwhm']:.4f} eV  │  "
                  f"跨度 = {bw['energy_span']:.4f} eV")
        else:
            for k in ["band_width_std", "band_width_fwhm", "band_energy_span"]:
                report[k] = float("nan")

        # ================================================================
        # 3 & 4. 劈裂分析
        # ================================================================
        peaks: List[Dict] = []
        need_peaks = _opt("crystal_field") or _opt("peaks")
        need_spin = _opt("spin_splitting") and self.parser.has_spin

        if need_spin or _opt("crystal_field"):
            print(f"── 劈裂分析 ──")

        # 3a. 自旋劈裂（静默计算，统一输出）
        if need_spin:
            ss = self.calc_spin_splitting(
                orbital=orbital, emin=emin, emax=emax, method="occupied",
                verbose=False,
            )
            report["spin_splitting"] = ss["spin_splitting"]
            report["spin_splitting_alpha"] = ss["band_center_alpha"]
            report["spin_splitting_beta"] = ss["band_center_beta"]

            spl = ss["spin_splitting"]
            if not np.isnan(spl):
                direction = "α 更深" if spl < 0 else "β 更深"
                print(f"  自旋劈裂 (占据态):  ΔE_spin = {spl:+.4f} eV  ({direction})")
            else:
                print(f"  自旋劈裂:  无法计算")
        else:
            for k in ["spin_splitting", "spin_splitting_alpha", "spin_splitting_beta"]:
                report[k] = float("nan")

        # 3b. 晶场劈裂（峰搜索复用，静默计算，统一输出）
        if need_peaks:
            peaks = self.find_peaks(
                orbital=orbital, spin=spin, emin=emin, emax=emax, verbose=False,
            )
        report["main_peak"] = peaks[0] if peaks else None
        report["all_peaks"] = peaks

        if _opt("crystal_field"):
            cfs = self.calc_crystal_field_splitting(
                orbital=orbital, spin=spin,
                peaks=peaks if peaks else None,
                emin=emin, emax=emax,
                verbose=False,
            )
            report["crystal_field_splitting"] = cfs["splitting"]
            report["cfs_peaks"] = (cfs["peak_lower"], cfs["peak_upper"])

            if cfs["splitting"] is not None:
                lo = cfs["peak_lower"]
                hi = cfs["peak_upper"]
                print(f"  晶场劈裂:           ΔE_cf   = {cfs['splitting']:.4f} eV  "
                      f"(峰@{lo['energy']:.3f} → 峰@{hi['energy']:.3f} eV)")
            else:
                if len(peaks) < 2:
                    print(f"  晶场劈裂:  峰数不足 ({len(peaks)})，无法计算")
        else:
            report["crystal_field_splitting"] = None
            report["cfs_peaks"] = (None, None)

        # ================================================================
        # 5. 占据态比例 —— E<0 的电子占据
        # ================================================================
        if _opt("occupancy"):
            occ = self.calc_occupancy(orbital=orbital, spin=spin, verbose=False)
            report["occupied"] = occ["occupied"]
            report["occupancy_ratio"] = occ["occupancy_ratio"]
            print(f"── 占据态 ──")
            print(f"  占据: {occ['occupied']:.2f} / {occ['total']:.2f} = {occ['occupancy_ratio']:.2%}")
        else:
            report["occupied"] = float("nan")
            report["occupancy_ratio"] = float("nan")

        # ================================================================
        # 6. 积分面积 —— 全区间总电子态数
        # ================================================================
        if _opt("peak_area"):
            _e_lo = peak_emin if peak_emin is not None else -100.0
            _e_hi = peak_emax if peak_emax is not None else 100.0
            total_area = self.calc_peak_area(
                orbital=orbital, spin=spin,
                emin=_e_lo, emax=_e_hi, verbose=False,
            )
            report["total_area"] = total_area
            print(f"── 积分面积 ──")
            print(f"  区间 [{_e_lo:.2f}, {_e_hi:.2f}] eV: {total_area:.2f}")
        else:
            report["total_area"] = float("nan")

        # ================================================================
        # 7. 峰列表
        # ================================================================
        if _opt("peaks") and peaks:
            print(f"── 峰列表 ──")
            for i, p in enumerate(peaks):
                print(f"  #{i+1:<2}  E = {p['energy']:+9.4f} eV   "
                      f"DOS = {p['dos']:10.2f}")

        print(f"╚{'═' * W}╝\n")
        return report


# ============================================================
# PDOSGUI 类 —— 基于 tkinter 的图形用户界面
# ============================================================
class PDOSGUI:
    """
    PDOS 绑图工具的 GUI 界面。

    界面布局（从上到下）:
      1. 文件选择区 —— 输入框 + 浏览按钮
      2. 轨道选择区 —— 根据文件内容动态生成的复选框
      3. 绑图模式区 —— 自旋极化 / 轨道分别 / 总态密度 三个单选按钮
      4. 能量范围区 —— 自动/手动切换 + 输入框
      5. 图片保存区 —— 保存目录选择 + 文件名输入
      6. 操作按钮 —— 预览 / 绘制并保存
      7. 状态栏 —— 显示当前操作状态

    所有操作日志会同步输出到命令行终端。
    """

    def __init__(self, initial_file: Optional[str] = None) -> None:
        """
        初始化 GUI 界面。

        参数
        ----
        initial_file : str, 可选
            启动时自动加载的文件路径（通常来自命令行 -f 参数）。
        """
        # --- 延迟导入 tkinter 组件（避免在 headless 环境下导入失败） ---
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        # 保存 tkinter 组件的引用，方便后续使用
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox

        # --- 核心数据对象 ---
        # 文件解析器实例
        self.parser: Optional[PDOSParser] = None
        # 绘图器实例
        self.plotter: Optional[PDOSPlotter] = None
        # 当前绘制的 Figure 对象引用
        self.fig: Optional[plt.Figure] = None

        # --- GUI 变量 ---
        # 轨道复选框变量字典: {"s": BooleanVar, "p": BooleanVar, ...}
        self.orbital_vars: Dict[str, "tk.BooleanVar"] = {}
        # 文件路径输入框变量
        self.file_path_var: "tk.StringVar"
        # 绑图模式选择变量: "spin" | "orbitals" | "total"
        self.plot_mode: "tk.StringVar"
        # 能量范围自动/手动开关
        self.energy_auto: "tk.BooleanVar"
        # 手动能量范围输入变量
        self.emin_var: "tk.StringVar"
        self.emax_var: "tk.StringVar"
        # 保存路径和文件名变量
        self.save_path_var: "tk.StringVar"
        self.save_name_var: "tk.StringVar"
        # 状态栏文字变量
        self.status_var: "tk.StringVar"
        # 能量输入框引用（用于启用/禁用切换）
        self.emin_entry: "tk.Entry"
        self.emax_entry: "tk.Entry"

        # --- 分析选项（控制 full_report 中各项分析的开关） ---
        self.analysis_options: Dict[str, bool] = {
            "band_center": True,
            "band_width": True,
            "spin_splitting": True,
            "crystal_field": True,
            "occupancy": True,
            "peak_area": True,
            "peaks": True,
        }
        # --- 积分面积专用积分区间（None 表示全区间） ---
        self.analysis_peak_emin: Optional[float] = None
        self.analysis_peak_emax: Optional[float] = None

        # --- 最近打开文件列表 ---
        self._recent_max: int = 10
        self._recent_file: str = os.path.join(
            os.path.expanduser("~"), ".pdos_plotter_recent.json"
        )
        self._recent_files: List[str] = self._load_recent_files()

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("PDOS 态密度绘图工具")
        self.root.geometry("650x720")
        self.root.resizable(True, True)

        # 构建界面
        self._build_ui()

        # 如果提供了初始文件，自动加载
        if initial_file and os.path.isfile(initial_file):
            self._load_file(initial_file)

        # 打印启动提示
        print("[INFO] GUI 界面已启动。所有操作日志将输出到此命令行窗口。")

    # ----------------------------------------------------------
    # 界面构建
    # ----------------------------------------------------------
    def _build_ui(self) -> None:
        """
        构建完整的 GUI 界面布局。

        布局采用垂直排列的 LabelFrame 分组，从上到下依次为：
          文件选择 → 轨道选择 → 绑图模式 → 能量范围 → 保存设置 → 按钮 → 状态栏
        """
        # 主容器框架
        main_frame = self.ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=self.tk.BOTH, expand=True)

        # 第1部分：文件选择区
        file_frame = self.ttk.LabelFrame(main_frame, text="文件选择", padding=5)
        file_frame.pack(fill=self.tk.X, pady=(0, 5))

        # 文件路径输入框
        self.file_path_var = self.tk.StringVar()
        file_entry = self.ttk.Entry(file_frame, textvariable=self.file_path_var)
        file_entry.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 5))

        # 最近打开按钮
        self.recent_btn = self.ttk.Button(
            file_frame, text="最近打开 ▼", command=self._show_recent_menu, width=12,
        )
        self.recent_btn.pack(side=self.tk.RIGHT, padx=(0, 3))

        # 浏览按钮
        browse_btn = self.ttk.Button(file_frame, text="浏览...", command=self._browse_file)
        browse_btn.pack(side=self.tk.RIGHT, padx=(0, 0))

        # 第2部分：轨道选择区
        self.orbital_frame = self.ttk.LabelFrame(main_frame, text="轨道选择", padding=5)
        self.orbital_frame.pack(fill=self.tk.X, pady=(0, 5))

        # 轨道复选框容器
        self.orbital_checkbox_frame = self.ttk.Frame(self.orbital_frame)
        self.orbital_checkbox_frame.pack(fill=self.tk.X)

        # 占位标签
        self.orbital_placeholder = self.ttk.Label(
            self.orbital_checkbox_frame,
            text="请先选择一个 PDOS.xcd 文件",
            foreground="gray",
        )
        self.orbital_placeholder.pack()

        # 第3部分：绑图模式选择区
        mode_frame = self.ttk.LabelFrame(main_frame, text="绘图模式", padding=5)
        mode_frame.pack(fill=self.tk.X, pady=(0, 5))

        # 模式变量
        self.plot_mode = self.tk.StringVar(value="spin")

        mode_inner = self.ttk.Frame(mode_frame)
        mode_inner.pack(fill=self.tk.X)

        # 三种模式单选按钮
        self.ttk.Radiobutton(
            mode_inner, text="自旋极化（α↑ / β↓）",
            variable=self.plot_mode, value="spin"
        ).pack(side=self.tk.LEFT, padx=(0, 15))

        self.ttk.Radiobutton(
            mode_inner, text="轨道分别显示",
            variable=self.plot_mode, value="orbitals"
        ).pack(side=self.tk.LEFT, padx=(0, 15))

        self.ttk.Radiobutton(
            mode_inner, text="总态密度 (TDOS)",
            variable=self.plot_mode, value="total"
        ).pack(side=self.tk.LEFT)

        # 第4部分：能量范围设置区
        energy_frame = self.ttk.LabelFrame(main_frame, text="能量范围 (eV)", padding=5)
        energy_frame.pack(fill=self.tk.X, pady=(0, 5))

        # 自动/手动切换
        self.energy_auto = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(
            energy_frame,
            text="自动（使用全部数据范围）",
            variable=self.energy_auto,
            command=self._toggle_energy_input,  # 切换输入框的启用/禁用状态
        ).pack(anchor=self.tk.W)

        # 手动输入范围
        range_frame = self.ttk.Frame(energy_frame)
        range_frame.pack(fill=self.tk.X, pady=(5, 0))

        self.ttk.Label(range_frame, text="E_min:").pack(side=self.tk.LEFT)
        self.emin_var = self.tk.StringVar(value="-10")
        self.emin_entry = self.ttk.Entry(range_frame, textvariable=self.emin_var, width=10, state=self.tk.DISABLED)
        self.emin_entry.pack(side=self.tk.LEFT, padx=(2, 15))

        self.ttk.Label(range_frame, text="E_max:").pack(side=self.tk.LEFT)
        self.emax_var = self.tk.StringVar(value="10")
        self.emax_entry = self.ttk.Entry(range_frame, textvariable=self.emax_var, width=10, state=self.tk.DISABLED)
        self.emax_entry.pack(side=self.tk.LEFT, padx=(2, 0))

        # 第5部分：图片保存设置区
        save_frame = self.ttk.LabelFrame(main_frame, text="图片保存", padding=5)
        save_frame.pack(fill=self.tk.X, pady=(0, 5))

        # 保存目录选择
        save_inner = self.ttk.Frame(save_frame)
        save_inner.pack(fill=self.tk.X)
        self.save_path_var = self.tk.StringVar(value=str(DEFAULT_PIC_DIR))
        save_entry = self.ttk.Entry(save_inner, textvariable=self.save_path_var)
        save_entry.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 5))
        save_browse_btn = self.ttk.Button(save_inner, text="选择目录", command=self._browse_save_dir)
        save_browse_btn.pack(side=self.tk.RIGHT)

        # 文件名输入
        name_frame = self.ttk.Frame(save_frame)
        name_frame.pack(fill=self.tk.X, pady=(5, 0))
        self.ttk.Label(name_frame, text="文件名:").pack(side=self.tk.LEFT)
        self.save_name_var = self.tk.StringVar(value="pdos_plot.png")
        name_entry = self.ttk.Entry(name_frame, textvariable=self.save_name_var, width=30)
        name_entry.pack(side=self.tk.LEFT, padx=(5, 10))
        # 支持格式提示
        self.ttk.Label(
            name_frame, text="（支持 .png / .jpg / .pdf / .svg）",
            foreground="gray"
        ).pack(side=self.tk.LEFT)

        # 第6部分：分析结果输出区
        analysis_frame = self.ttk.LabelFrame(main_frame, text="分析报告输出", padding=5)
        analysis_frame.pack(fill=self.tk.BOTH, expand=True, pady=(0, 5))

        # 带滚动条的文本框
        self.analysis_text = self.tk.Text(
            analysis_frame, height=6, wrap=self.tk.WORD,
            font=("Consolas", 9),
            state=self.tk.DISABLED,
        )
        analysis_scrollbar = self.ttk.Scrollbar(
            analysis_frame, orient=self.tk.VERTICAL, command=self.analysis_text.yview
        )
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        self.analysis_text.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        analysis_scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        # 第7部分：操作按钮区
        btn_frame = self.ttk.Frame(main_frame)
        btn_frame.pack(fill=self.tk.X, pady=(10, 0))

        # 绘制并保存按钮
        self.plot_btn = self.ttk.Button(
            btn_frame, text="绘制并保存", command=self._plot_and_save, width=18
        )
        self.plot_btn.pack(side=self.tk.RIGHT, padx=(5, 0))

        # 预览按钮
        self.preview_btn = self.ttk.Button(
            btn_frame, text="预览", command=self._preview, width=18
        )
        self.preview_btn.pack(side=self.tk.RIGHT, padx=(5, 0))

        # 分析报告按钮
        self.analyze_btn = self.ttk.Button(
            btn_frame, text="分析报告", command=self._run_analysis, width=14
        )
        self.analyze_btn.pack(side=self.tk.LEFT, padx=(0, 5))

        # 分析选项按钮
        self.options_btn = self.ttk.Button(
            btn_frame, text="分析选项", command=self._show_analysis_options, width=14
        )
        self.options_btn.pack(side=self.tk.LEFT)

        # 第8部分：状态栏
        self.status_var = self.tk.StringVar(value="就绪 - 请选择 PDOS.xcd 文件")
        status_bar = self.ttk.Label(
            self.root, textvariable=self.status_var,
            relief=self.tk.SUNKEN, anchor=self.tk.W, padding=(5, 2)
        )
        status_bar.pack(side=self.tk.BOTTOM, fill=self.tk.X)

    # ----------------------------------------------------------
    # 事件处理方法
    # ----------------------------------------------------------

    def _browse_file(self) -> None:
        """
        打开系统文件浏览器选择 .xcd 文件。

        使用 tkinter.filedialog.askopenfilename 弹出原生文件选择对话框，
        限定显示 .xcd 文件和所有文件两种类型。
        """
        path = self.filedialog.askopenfilename(
            title="选择 PDOS.xcd 文件",
            filetypes=[
                ("XCD 文件", "*.xcd"),
                ("所有文件", "*.*"),
            ],
        )
        # 用户可能取消选择（返回空字符串），需要判断
        if path:
            self._load_file(path)

    # ----------------------------------------------------------
    # 最近文件管理
    # ----------------------------------------------------------
    def _load_recent_files(self) -> List[str]:
        """从 JSON 文件读入最近打开的文件路径列表。"""
        try:
            if os.path.isfile(self._recent_file):
                with open(self._recent_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [
                            p for p in data
                            if os.path.isfile(p) and p.lower().endswith(".xcd")
                        ][:self._recent_max]
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _save_recent_files(self) -> None:
        """将最近文件列表持久化保存到 JSON 文件。"""
        try:
            with open(self._recent_file, "w", encoding="utf-8") as f:
                json.dump(self._recent_files, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _add_to_recent_files(self, path: str) -> None:
        """将文件路径添加到最近打开列表顶部，去重后保存。"""
        path = os.path.abspath(path)
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:self._recent_max]
        self._save_recent_files()

    def _show_recent_menu(self) -> None:
        """
        「最近打开 ▼」按钮回调：弹出下拉菜单列出最近打开的文件。
        点击菜单项直接加载对应文件。
        """
        menu = self.tk.Menu(self.root, tearoff=0)
        if not self._recent_files:
            menu.add_command(label="(无最近文件)", state=self.tk.DISABLED)
        else:
            for i, path in enumerate(self._recent_files):
                label = f"{i+1}. {os.path.basename(path)}  —  {path}"
                menu.add_command(
                    label=label,
                    command=lambda p=path: self._load_file(p),
                )
            menu.add_separator()
            menu.add_command(label="清空最近记录", command=self._clear_recent_files)
        menu.post(
            self.recent_btn.winfo_rootx(),
            self.recent_btn.winfo_rooty() + self.recent_btn.winfo_height(),
        )

    def _clear_recent_files(self) -> None:
        """清空最近文件列表。"""
        self._recent_files.clear()
        self._save_recent_files()
        print("[INFO] 最近文件列表已清空。")

    def _load_file(self, path: str) -> None:
        """
        加载并解析 PDOS.xcd 文件，更新界面轨道选择。

        流程:
          1. 更新文件路径输入框
          2. 创建 PDOSParser 实例并执行解析
          3. 创建 PDOSPlotter 实例
          4. 根据解析结果动态生成轨道复选框
          5. 根据是否有自旋极化调整模式按钮

        参数
        ----
        path : str
            xcd 文件的完整路径。
        """
        # 更新输入框显示
        self.file_path_var.set(path)
        self.status_var.set(f"正在解析: {os.path.basename(path)} ...")
        self.root.update_idletasks()  # 强制刷新界面

        try:
            # 创建解析器并解析文件
            self.parser = PDOSParser(path)
            self.parser.parse()

            # 创建绑图器（需要先有解析好的数据）
            self.plotter = PDOSPlotter(self.parser)

            # 更新 GUI 控件
            self._update_orbital_checkboxes()  # 重建轨道复选框
            self._update_mode_buttons()         # 调整模式按钮

            self.status_var.set(
                f"已加载: {os.path.basename(path)} — "
                f"{len(self.parser.raw_data)} 条数据系列"
            )
            print(f"[INFO] 文件加载成功: {path}")

            # 成功加载后加入最近文件列表
            self._add_to_recent_files(path)

        except Exception as e:
            self.messagebox.showerror("解析错误", f"无法解析文件:\n{e}")
            self.status_var.set(f"解析失败: {e}")
            print(f"[ERROR] 解析文件失败: {e}")

    def _update_orbital_checkboxes(self) -> None:
        """
        根据解析结果动态生成轨道选择复选框。

        操作:
          1. 清除旧复选框
          2. 根据 parser.available_orbitals 创建新的 BooleanVar
          3. 默认选中所有非 sum 轨道（常用场景）
          4. sum 轨道默认不选中
        """
        # 清除旧控件和变量
        for widget in self.orbital_checkbox_frame.winfo_children():
            widget.destroy()
        self.orbital_vars.clear()

        if self.parser is None:
            return

        orbitals = self.parser.available_orbitals
        if not orbitals:
            # 无轨道数据时的占位提示
            self.ttk.Label(
                self.orbital_checkbox_frame,
                text="未检测到任何轨道数据",
                foreground="gray"
            ).pack()
            return

        # 为每个轨道创建复选框
        for orb in orbitals:
            # sum 轨道默认不勾选（因为总态密度通常用单独的 TDOS 模式查看）
            var = self.tk.BooleanVar(value=(orb != "sum"))
            self.orbital_vars[orb] = var

            # 显示名称：普通轨道大写，sum 显示为 "Sum (总)"
            display_name = orb.upper() if orb != "sum" else "Sum (总)"
            cb = self.ttk.Checkbutton(
                self.orbital_checkbox_frame,
                text=display_name,
                variable=var,
            )
            cb.pack(side=self.tk.LEFT, padx=5)

    def _update_mode_buttons(self) -> None:
        """
        根据文件是否有自旋极化来调整默认绑图模式。

        - 有自旋极化 → 保持 "spin" 模式
        - 无自旋极化 → 自动切换到 "orbitals" 模式（spin 模式对非极化数据无意义）
        """
        if self.parser is None:
            return

        if not self.parser.has_spin:
            self.plot_mode.set("orbitals")
            print("[INFO] 检测到非自旋极化文件，已自动切换为「轨道分别显示」模式。")

    def _toggle_energy_input(self) -> None:
        """
        切换手动能量范围输入框的启用/禁用状态。

        当 "自动" 复选框勾选时 → 输入框禁用（灰色）
        当 "自动" 复选框取消时 → 输入框启用（可编辑）
        """
        state = self.tk.DISABLED if self.energy_auto.get() else self.tk.NORMAL
        self.emin_entry.configure(state=state)
        self.emax_entry.configure(state=state)

    def _browse_save_dir(self) -> None:
        """
        打开系统目录选择对话框，选择图片保存的目标文件夹。

        使用 tkinter.filedialog.askdirectory 弹出原生目录选择器。
        默认初始目录为当前 save_path_var 的值。
        """
        path = self.filedialog.askdirectory(
            title="选择图片保存目录",
            initialdir=self.save_path_var.get(),
        )
        if path:
            self.save_path_var.set(path)

    def _get_save_path(self) -> str:
        """
        获取完整的图片保存路径。

        拼接规则: 保存目录 + 文件名
        如果文件名为空，默认使用 "pdos_plot.png"。

        返回
        ----
        full_path : str
            完整的保存路径（目录 + 文件名 + 扩展名）。
        """
        directory = self.save_path_var.get()
        filename = self.save_name_var.get()
        if not filename.strip():
            filename = "pdos_plot.png"
        return os.path.join(directory, filename)

    def _get_selected_orbitals(self) -> List[str]:
        """
        获取用户在 GUI 中选中的轨道列表。

        遍历 self.orbital_vars 字典，找出所有 BooleanVar 为 True 的轨道。

        返回
        ----
        selected : list of str
            用户选中的轨道名列表，如 ["s", "p", "d"]。
        """
        return [orb for orb, var in self.orbital_vars.items() if var.get()]

    def _get_energy_range(self) -> Optional[Tuple[float, float]]:
        """
        获取用户设定的能量范围。

        如果 "自动" 模式 → 返回 None（使用数据全范围）
        如果 "手动" 模式 → 从输入框读取并转换为浮点数

        返回
        ----
        (emin, emax) : (float, float) or None
            手动设定的能量范围；None 表示自动。
        """
        if self.energy_auto.get():
            return None
        try:
            emin = float(self.emin_var.get())
            emax = float(self.emax_var.get())
            return (emin, emax)
        except ValueError:
            self.messagebox.showwarning("输入错误", "能量范围请输入有效的数字（如 -10, 10）。")
            return None

    # ----------------------------------------------------------
    # 核心绘图和保存操作
    # ----------------------------------------------------------

    def _do_plot(self) -> Optional[plt.Figure]:
        """
        执行绑图的核心逻辑。

        流程:
          1. 检查文件是否已加载
          2. 根据当前模式选择（spin / orbitals / total）调用对应的绘图方法
          3. 返回 Figure 对象以供保存或显示

        返回
        ----
        fig : matplotlib Figure or None
            绘图结果。如果发生错误或未加载文件则返回 None。
        """
        # 检查前置条件
        if self.parser is None or self.plotter is None:
            self.messagebox.showwarning("未加载文件", "请先选择一个 PDOS.xcd 文件。")
            return None

        mode = self.plot_mode.get()
        energy_range = self._get_energy_range()
        title = self.parser.filename

        # 关闭所有已有的 matplotlib 图表窗口，避免累积
        plt.close("all")

        try:
            if mode == "spin" and self.parser.has_spin:
                # ---- 自旋极化模式 ----
                orbitals = self._get_selected_orbitals()
                if not orbitals:
                    # 如果用户什么也没选，默认选所有非 sum 轨道
                    orbitals = [o for o in self.parser.available_orbitals if o != "sum"]
                fig, _ = self.plotter.plot_spin_polarized(
                    orbitals=orbitals, energy_range=energy_range, title=title
                )
                print(f"[INFO] 已绘制自旋极化图，轨道: {orbitals}")

            elif mode == "total":
                # ---- 总态密度模式 ----
                fig, _ = self.plotter.plot_total(
                    energy_range=energy_range, title=title
                )
                print("[INFO] 已绘制总态密度 (TDOS) 图")

            else:
                # ---- 轨道分别显示模式 ----
                orbitals = self._get_selected_orbitals()
                if not orbitals:
                    orbitals = [o for o in self.parser.available_orbitals if o != "sum"]
                fig, _ = self.plotter.plot_orbitals(
                    orbitals=orbitals, energy_range=energy_range, title=title
                )
                print(f"[INFO] 已绘制轨道分别显示图，轨道: {orbitals}")

            # 保存当前 Figure 引用
            self.fig = fig
            return fig

        except Exception as e:
            self.messagebox.showerror("绘图错误", f"绘图时出错:\n{e}")
            print(f"[ERROR] 绘图失败: {e}")
            return None

    def _preview(self) -> None:
        """
        「预览」按钮回调：绘制图表并弹出 matplotlib 交互式窗口显示。

        与 _plot_and_save 的区别是不保存文件。
        plt.show(block=False) 以非阻塞方式显示，GUI 可继续响应操作。
        """
        print("[INFO] 正在生成预览...")
        fig = self._do_plot()
        if fig is not None:
            plt.show(block=False)
            self.status_var.set("预览已打开 — 关闭预览窗口后可继续操作")
            print("[INFO] 预览窗口已打开。关闭窗口后可继续操作。")

    def _plot_and_save(self) -> None:
        """
        「绘制并保存」按钮回调：执行绑图并保存到指定路径。

        流程:
          1. 调用 _do_plot() 绑图
          2. 拼装保存路径
          3. 调用 plotter.save_figure() 保存
          4. 保存后以非阻塞方式显示图片
        """
        print("[INFO] 正在绘制并保存图片...")
        fig = self._do_plot()
        if fig is None:
            return

        save_path = self._get_save_path()
        try:
            self.plotter.save_figure(fig, save_path)
            self.status_var.set(f"已保存: {save_path}")
            # 保存后显示图片
            plt.show(block=False)
        except Exception as e:
            self.messagebox.showerror("保存错误", f"保存图片时出错:\n{e}")
            print(f"[ERROR] 保存失败: {e}")

    def _show_analysis_options(self) -> None:
        """
        「分析选项」按钮回调：弹出选项对话框，勾选需要执行的分析项目。

        可选项包括:
          - 带中心 (全部态 + 占据态)
          - 带宽 (标准差 / FWHM / 跨度)
          - 自旋劈裂 (α−β 带中心差)
          - 晶场劈裂 (主峰能量间距 ΔE)
          - 占据态比例 (E < 0)
          - 积分面积 (可自定义积分区间)
          - 峰列表 (各峰能量/DOS)
        """
        dialog = self.tk.Toplevel(self.root)
        dialog.title("分析选项")
        dialog.geometry("400x440")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # 标题
        self.ttk.Label(
            dialog, text="选择要执行的分析项目:",
            font=("", 10, "bold"),
        ).pack(pady=(10, 5), padx=15, anchor=self.tk.W)

        # 分析项列表与说明（与报告输出顺序一致）
        option_items = [
            ("band_center",    "带中心 (全部态 + 占据态)"),
            ("band_width",     "带宽 (标准差 / FWHM / 跨度)"),
            ("spin_splitting", "自旋劈裂 (α−β 带中心差)"),
            ("crystal_field",  "晶场劈裂 (主峰能量间距 ΔE)"),
            ("occupancy",      "占据态比例 (E < 0)"),
            ("peak_area",      "积分面积 (可自定义积分区间)"),
            ("peaks",          "峰列表 (各峰能量/DOS)"),
        ]

        # 为每个选项创建 BooleanVar（从当前设置初始化）
        option_vars: Dict[str, "tk.BooleanVar"] = {}
        for key, label_text in option_items:
            var = self.tk.BooleanVar(value=self.analysis_options.get(key, True))
            option_vars[key] = var
            cb = self.ttk.Checkbutton(dialog, text=label_text, variable=var)
            cb.pack(pady=2, padx=25, anchor=self.tk.W)

        # ---- 积分面积专用积分区间设置 ----
        range_frame = self.ttk.LabelFrame(dialog, text="积分面积区间设置 (eV)")
        range_frame.pack(pady=(10, 5), padx=25, fill=self.tk.X)

        row = self.ttk.Frame(range_frame)
        row.pack(pady=5, padx=10)
        self.ttk.Label(row, text="下限:").pack(side=self.tk.LEFT)
        peak_emin_var = self.tk.StringVar(
            value="" if self.analysis_peak_emin is None else str(self.analysis_peak_emin)
        )
        self.ttk.Entry(row, textvariable=peak_emin_var, width=10).pack(
            side=self.tk.LEFT, padx=(5, 15)
        )
        self.ttk.Label(row, text="上限:").pack(side=self.tk.LEFT)
        peak_emax_var = self.tk.StringVar(
            value="" if self.analysis_peak_emax is None else str(self.analysis_peak_emax)
        )
        self.ttk.Entry(row, textvariable=peak_emax_var, width=10).pack(
            side=self.tk.LEFT, padx=(5, 0)
        )
        self.ttk.Label(
            range_frame, text="留空 = 自动使用全区间",
            font=("", 8), foreground="gray",
        ).pack(anchor=self.tk.CENTER, pady=(0, 3))

        # 按钮行
        btn_frame = self.ttk.Frame(dialog)
        btn_frame.pack(pady=(15, 10))

        def _on_ok() -> None:
            """确定：保存选项和积分区间并关闭对话框。"""
            for key, var in option_vars.items():
                self.analysis_options[key] = var.get()
            # 保存积分区间
            try:
                emin_str = peak_emin_var.get().strip()
                self.analysis_peak_emin = float(emin_str) if emin_str else None
            except ValueError:
                self.analysis_peak_emin = None
            try:
                emax_str = peak_emax_var.get().strip()
                self.analysis_peak_emax = float(emax_str) if emax_str else None
            except ValueError:
                self.analysis_peak_emax = None
            print(f"[INFO] 分析选项已更新: {self.analysis_options}")
            if self.analysis_peak_emin is not None or self.analysis_peak_emax is not None:
                print(f"[INFO]   积分区间: [{self.analysis_peak_emin or '-∞'}, "
                      f"{self.analysis_peak_emax or '+∞'}] eV")
            dialog.destroy()

        def _select_all() -> None:
            """全选所有分析项。"""
            for var in option_vars.values():
                var.set(True)

        def _deselect_all() -> None:
            """取消全选。"""
            for var in option_vars.values():
                var.set(False)

        self.ttk.Button(dialog, text="全选", command=_select_all, width=8).pack(
            side=self.tk.LEFT, padx=(40, 10)
        )
        self.ttk.Button(dialog, text="取消全选", command=_deselect_all, width=10).pack(
            side=self.tk.LEFT
        )
        self.ttk.Button(dialog, text="确定", command=_on_ok, width=10).pack(
            side=self.tk.RIGHT, padx=(0, 40)
        )

    def _run_analysis(self) -> None:
        """
        「分析报告」按钮回调：执行态密度分析并显示在 GUI 文本框和命令行。

        分析内容包括:
          - d带/p带中心（加权平均能量）
          - 峰位置搜索
          - 费米能级以下占据态
          - 指定区间积分面积

        输出同时发送到:
          1. GUI 分析报告文本框（实时可见）
          2. 命令行终端（供后续查阅）
        """
        if self.parser is None:
            self.messagebox.showwarning("未加载文件", "请先选择一个 PDOS.xcd 文件。")
            return

        # 选择分析目标：使用用户当前选中的第一个轨道
        selected = self._get_selected_orbitals()
        if not selected:
            selected = [o for o in self.parser.available_orbitals if o != "sum"]
        if not selected:
            self.messagebox.showwarning("无可用轨道", "文件中没有可用于分析的轨道。")
            return

        orbital = selected[0]
        spin = "alpha" if self.parser.has_spin else None

        self.status_var.set(f"正在分析 {orbital.upper()} 轨道...")
        self.root.update_idletasks()

        # 使用 StringIO 捕获分析过程中的 print 输出，同时保留原始 stdout
        import io
        old_stdout = sys.stdout
        captured_output = io.StringIO()

        try:
            sys.stdout = captured_output  # 临时重定向到 StringIO

            analyzer = PDOSAnalyzer(self.parser)
            energy_range = self._get_energy_range()
            emin = energy_range[0] if energy_range else None
            emax = energy_range[1] if energy_range else None
            report = analyzer.full_report(
                orbital=orbital, spin=spin, emin=emin, emax=emax,
                peak_emin=self.analysis_peak_emin,
                peak_emax=self.analysis_peak_emax,
                options=self.analysis_options,
            )

            # 恢复 stdout 并获取捕获的文本
            sys.stdout = old_stdout
            captured_text = captured_output.getvalue()

            # ---- 输出到命令行 ----
            print(captured_text, end="")
            sys.stdout.flush()

            # ---- 输出到 GUI 文本框 ----
            self.analysis_text.configure(state=self.tk.NORMAL)
            self.analysis_text.delete("1.0", self.tk.END)
            self.analysis_text.insert(self.tk.END, captured_text)
            self.analysis_text.see(self.tk.END)  # 滚动到底部
            self.analysis_text.configure(state=self.tk.DISABLED)

            # ---- 弹窗摘要 ----
            bc_all = report["band_center_all"]
            bc_occ = report["band_center_occupied"]
            cfs = report["crystal_field_splitting"]
            spin_s = report["spin_splitting"]
            main_peak = report["main_peak"]
            occ_ratio = report["occupancy_ratio"]

            summary_lines = [f"轨道: {orbital.upper()}"]
            if not np.isnan(bc_all):
                summary_lines.append(f"全部态 d带中心:  {bc_all:.4f} eV")
            if not np.isnan(bc_occ):
                summary_lines.append(f"占据态 d带中心:  {bc_occ:.4f} eV")
            if cfs is not None and not np.isnan(cfs):
                summary_lines.append(f"晶场劈裂 ΔE:    {cfs:.4f} eV")
            if not np.isnan(spin_s):
                summary_lines.append(f"自旋劈裂 ΔE_s:  {spin_s:.4f} eV")
            if main_peak:
                summary_lines.append(
                    f"主峰: E = {main_peak['energy']:.4f} eV, DOS = {main_peak['dos']:.4f}"
                )
            if not np.isnan(occ_ratio):
                summary_lines.append(f"占据比例 (E<0): {occ_ratio:.2%}")

            self.messagebox.showinfo(
                f"{orbital.upper()}轨道分析报告",
                "\n".join(summary_lines) + "\n\n详细结果已在下方文本框和命令行中显示。",
            )
            bc_display = f"全部={bc_all:.4f}, 占据={bc_occ:.4f}" if not np.isnan(bc_all) else "N/A"
            self.status_var.set(f"分析完成: {orbital.upper()}带中心 [{bc_display}] eV")

        except Exception as e:
            sys.stdout = old_stdout  # 确保恢复 stdout
            self.messagebox.showerror("分析错误", f"分析时出错:\n{e}")
            print(f"[ERROR] 分析失败: {e}")

    # ----------------------------------------------------------
    # GUI 启动
    # ----------------------------------------------------------
    def run(self) -> None:
        """启动 tkinter 主事件循环。"""
        self.root.mainloop()


# ============================================================
# 命令行参数解析
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    CLI 模式下支持的完整参数列表，所有参数都有详细的帮助信息。

    返回
    ----
    parser : argparse.ArgumentParser
        配置完成的参数解析器。
    """
    parser = argparse.ArgumentParser(
        description="PDOS 态密度数据提取与绘图工具 —— 支持 GUI 和 CLI 两种模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  ── GUI 模式 ──
    python pdos_plotter.py
    python pdos_plotter.py -f "path/to/PDOS.xcd"

  ── CLI 总态密度 ──
    python pdos_plotter.py -f "PDOS.xcd" --total --no-gui -o total_dos.png

  ── CLI 自旋极化 ──
    python pdos_plotter.py -f "PDOS.xcd" --spin --orbitals s,p,d --no-gui

  ── CLI 轨道分别显示 ──
    python pdos_plotter.py -f "PDOS.xcd" --orbitals s,p,d --no-gui -o orbitals.png
        """,
    )

    # --- 文件输入 ---
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="PDOS.xcd 文件的完整路径。GUI 模式下可选，CLI 模式下必填。",
    )

    # --- 输出设置 ---
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出图片的保存路径（包括文件名和扩展名）。"
             "默认保存到 ./pic/<原文件名>_pdos.png。",
    )

    # --- 运行模式 ---
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="禁用 GUI，直接在命令行绑图并保存后退出。",
    )

    # --- 绑图模式选择 ---
    parser.add_argument(
        "--spin",
        action="store_true",
        default=False,
        help="使用自旋极化模式绘图（α 自旋向上、β 自旋向下）。",
    )
    parser.add_argument(
        "--total",
        action="store_true",
        default=False,
        help="绘制总态密度 (TDOS)。",
    )
    parser.add_argument(
        "--orbitals",
        type=str,
        default=None,
        help="要显示的轨道列表，逗号分隔（如 s,p,d）。"
             "仅在 --no-gui 模式下生效。示例: --orbitals s,p,d",
    )

    # --- 分析模式 ---
    parser.add_argument(
        "--analyze",
        action="store_true",
        default=False,
        help="执行态密度分析（d带中心、峰位置、占据态等），结果打印到命令行。"
             "可与绘图模式同时使用。",
    )
    parser.add_argument(
        "--analysis-orbital",
        type=str,
        default="d",
        help="分析的目标轨道（默认 d）。示例: --analysis-orbital p",
    )
    parser.add_argument(
        "--analysis-spin",
        type=str,
        default=None,
        choices=["alpha", "beta"],
        help="分析时筛选的自旋方向（alpha/beta），默认自动选择。",
    )

    # --- 能量范围 ---
    parser.add_argument(
        "--emin",
        type=float,
        default=None,
        help="能量范围下限 (eV)。不指定则使用数据最小值。",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=None,
        help="能量范围上限 (eV)。不指定则使用数据最大值。",
    )

    # --- 图表样式 ---
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="图表标题。默认使用文件名。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图片的 DPI 分辨率（默认 300）。",
    )

    # --- 显示选项 ---
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="保存后弹出窗口显示图片（仅在 --no-gui 模式下生效）。",
    )

    return parser


# ============================================================
# CLI 模式运行函数
# ============================================================
def run_cli(args: argparse.Namespace) -> None:
    """
    命令行模式入口：解析文件 → 绘图 → 保存 → 退出。

    根据 args 中的参数选择绑图模式和配置，无交互操作。

    参数
    ----
    args : argparse.Namespace
        命令行解析后的参数命名空间。
    """
    # ---- 验证文件路径 ----
    if not args.file:
        print("[ERROR] CLI 模式需要指定文件路径: -f <文件路径>")
        print("[INFO] 使用 --help 查看完整用法。")
        sys.exit(1)

    if not os.path.isfile(args.file):
        print(f"[ERROR] 文件不存在: {args.file}")
        sys.exit(1)

    # ---- 解析文件 ----
    pdos_parser = PDOSParser(args.file)
    pdos_parser.parse()
    plotter = PDOSPlotter(pdos_parser)

    # ---- 确定能量范围 ----
    energy_range = None
    if args.emin is not None or args.emax is not None:
        # 如果只指定了其中一个，另一个从数据范围自动获取
        emin = args.emin if args.emin is not None else pdos_parser.get_energy_range()[0]
        emax = args.emax if args.emax is not None else pdos_parser.get_energy_range()[1]
        energy_range = (emin, emax)
        print(f"[INFO] 能量范围: [{emin:.2f}, {emax:.2f}] eV")

    # ---- 确定输出路径 ----
    if args.output:
        save_path = args.output
    else:
        # 默认输出路径：./pic/<文件名>_pdos.png
        DEFAULT_PIC_DIR.mkdir(parents=True, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        save_path = str(DEFAULT_PIC_DIR / f"{base_name}_pdos.png")

    # ---- 解析轨道参数 ----
    orbitals_list = None
    if args.orbitals:
        orbitals_list = [o.strip().lower() for o in args.orbitals.split(",")]
        print(f"[INFO] 指定轨道: {orbitals_list}")

    # ---- 根据参数选择绘图模式 ----
    title = args.title

    if args.total:
        # 总态密度模式
        print("[INFO] 绑图模式: 总态密度 (TDOS)")
        fig, _ = plotter.plot_total(energy_range=energy_range, title=title)
    elif args.spin and pdos_parser.has_spin:
        # 自旋极化模式（需要文件本身包含自旋数据）
        print("[INFO] 绑图模式: 自旋极化 (α↑/β↓)")
        fig, _ = plotter.plot_spin_polarized(
            orbitals=orbitals_list, energy_range=energy_range, title=title
        )
    else:
        # 默认模式：轨道分别显示
        print("[INFO] 绑图模式: 轨道分别显示")
        fig, _ = plotter.plot_orbitals(
            orbitals=orbitals_list, energy_range=energy_range, title=title
        )

    # ---- 保存图片 ----
    plotter.save_figure(fig, save_path)
    print(f"[INFO] 完成! 图片已保存至: {save_path}")

    # ---- 分析模式（可选） ----
    if args.analyze:
        print()
        analyzer = PDOSAnalyzer(pdos_parser)
        analyzer.full_report(
            orbital=args.analysis_orbital,
            spin=args.analysis_spin,
            emin=args.emin,
            emax=args.emax,
        )

    # ---- 可选显示 ----
    if args.show:
        print("[INFO] 正在显示图片...")
        plt.show()


# ============================================================
# 程序主入口
# ============================================================
def main() -> None:
    """
    程序主入口函数。

    根据命令行参数决定启动模式:
      - 带有 --no-gui → CLI 命令行模式（绑图、保存、退出）
      - 不带 --no-gui → GUI 图形界面模式（交互式操作）
    """
    # 打印启动横幅
    print("=" * 60)
    print("  PDOS 态密度数据提取与绘图工具")
    print("  支持 Materials Studio .xcd 格式 (XML)")
    print("=" * 60)

    # 解析命令行参数
    args = build_argparser().parse_args()

    # 确保默认图片输出目录存在
    DEFAULT_PIC_DIR.mkdir(parents=True, exist_ok=True)

    if args.no_gui:
        # ── CLI 模式 ──
        run_cli(args)
    else:
        # ── GUI 模式 ──
        print("[INFO] 启动 GUI 图形界面模式...")
        print("[INFO] 所有操作日志将同步显示在此命令行窗口中。")
        print("[INFO] 提示: 使用 --no-gui 参数可进入命令行模式。")
        print()
        gui = PDOSGUI(initial_file=args.file)
        gui.run()


# 程序入口点
if __name__ == "__main__":
    main()
