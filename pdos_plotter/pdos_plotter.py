#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDOSPlotter —— 态密度数据绘图器
================================

提供四种绘图模式 + 统一的样式控制和系列绘制工具。

绘图模式:
  1. plot_spin_polarized() —— 自旋极化模式（α↑/β↓ 同侧显示）
  2. plot_total()           —— 总态密度 (TDOS) 模式
  3. plot_orbitals()        —— 轨道分别显示模式
  4. plot_multi_overlay()   —— 多文件叠加模式

改进（2026/06/11 重构）:
  - 抽取 _add_reference_lines() 消除 4 处参考线重复
  - 抽取 _plot_series() 统一 α/β 系列绘制
  - save_figure() 支持外部 dpi 参数

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11 (重构: 消除重复、提取工具方法)
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# ── 配置 matplotlib 中文字体支持 ──────────────────────────────
# 策略：直接用字体文件路径创建 FontProperties，绕过 rcParams 匹配问题
_CJK_FONT_PROP = None  # 全局 CJK 字体属性对象

def _init_cjk_font() -> None:
    """搜索可用的 CJK 字体文件并创建 FontProperties。"""
    global _CJK_FONT_PROP
    _CJK_SEARCH_PATHS = [
        Path("C:/Windows/Fonts/simhei.ttf"),      # 黑体（Windows 首选 .ttf）
        Path("C:/Windows/Fonts/msyh.ttf"),        # 微软雅黑（可能为 .ttf 版本）
        Path("C:/Windows/Fonts/simsun.ttc"),      # 宋体
        Path("C:/Windows/Fonts/simkai.ttf"),      # 楷体
        Path("C:/Windows/Fonts/STKAITI.TTF"),     # 华文楷体
    ]
    for _fp_path in _CJK_SEARCH_PATHS:
        if _fp_path.exists():
            try:
                _prop = FontProperties(fname=str(_fp_path))
                # 验证字体可用
                _prop.get_name()
                _CJK_FONT_PROP = _prop
                print(f"[INFO] CJK 字体已加载: {_fp_path.name} ({_fp_path})")
                return
            except Exception:
                continue
    # 回退：设置 rcParams 尽力而为
    _CJK_FONT_LIST = ["SimHei", "Microsoft YaHei", "SimSun", "KaiTi"]
    _existing = set(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["font.sans-serif"] = [f for f in _CJK_FONT_LIST if f not in _existing] + list(_existing)
    plt.rcParams["axes.unicode_minus"] = False
    print("[WARN] 未找到 CJK 字体文件，中文可能无法正常显示。")

_init_cjk_font()
# ──────────────────────────────────────────────────────────────

try:
    from .constants import (
        ORBITAL_COLORS,
        ML_ORBITAL_COLORS,
        get_orbital_color,
        OVERLAY_PALETTE,
        DEFAULT_FIGSIZE,
        DEFAULT_DPI,
        FERMI_LINE_STYLE,
        ZERO_LINE_STYLE,
    )
except ImportError:
    from constants import (
        ORBITAL_COLORS,
        ML_ORBITAL_COLORS,
        get_orbital_color,
        OVERLAY_PALETTE,
        DEFAULT_FIGSIZE,
        DEFAULT_DPI,
        FERMI_LINE_STYLE,
        ZERO_LINE_STYLE,
    )


# ============================================================
# PDOSPlotter 类
# ============================================================
class PDOSPlotter:
    """
    PDOS 数据绘图器。

    提供四种绘图模式:
      1. plot_spin_polarized() —— 自旋极化模式
         α 和 β 自旋都在 y>0 同侧显示，使用不同颜色和线型区分：
         α（↑）实线 + 填充，β（↓）虚线 + 斜线填充。

      2. plot_orbitals() —— 轨道分别显示模式
         所有选中的轨道都在 y>0 半轴分别绘制，用颜色区分不同轨道。

      3. plot_total() —— 总态密度模式
         只绘制 Sum/Total 系列的态密度曲线。

      4. plot_multi_overlay() —— 多文件叠加模式
         将来自不同文件的系列绘制在同一幅图上。

    公用功能:
      - save_figure() 保存图片到指定路径
      - 自动添加费米能级 (E=0) 参考线
    """

    def __init__(self, parser) -> None:
        """
        初始化绘图器。

        参数
        ----
        parser : PDOSParser 或 CastepPDOSAdapter
            已完成解析的数据源实例（duck-typing 接口）：
            需要提供 get_data(), get_summed_data(), get_total_dos(),
            get_energy_range(), has_spin, available_orbitals,
            available_spins, filename, label 等属性和方法。
        """
        self.parser = parser

    # ----------------------------------------------------------
    # 公用工具方法
    # ----------------------------------------------------------
    @staticmethod
    def _add_reference_lines(ax: plt.Axes) -> None:
        """
        添加费米能级参考线（E=0 竖虚线）和零基线（y=0 横实线）。

        消除原代码中 4 处完全重复的 axvline/axhline 调用。
        """
        ax.axvline(x=0, **FERMI_LINE_STYLE)
        ax.axhline(y=0, **ZERO_LINE_STYLE)

    def _plot_series(
        self,
        ax: plt.Axes,
        e: np.ndarray,
        dos: np.ndarray,
        color: str,
        *,
        is_beta: bool = False,
        label: str = "",
    ) -> None:
        """
        统一绘制单条态密度系列（α 或 β 自旋）。

        消除 plot_spin_polarized / plot_orbitals / plot_multi_overlay
        中 ~80 行重复的 fill_between + plot 模式。

        参数
        ----
        ax : matplotlib Axes
            目标坐标轴。
        e : ndarray
            能量数组。
        dos : ndarray
            DOS 值数组（已取绝对值）。
        color : str
            基色（十六进制）。
        is_beta : bool
            True → β 自旋（虚线 + 斜线填充 + 浅色）
            False → α 自旋（实线 + 实心填充）
        label : str
            图例标签。
        """
        if is_beta:
            beta_color = self._lighten_color(color, factor=0.5)
            ax.fill_between(
                e, 0, dos,
                alpha=0.18, color=beta_color,
                hatch="////", edgecolor=beta_color, linewidth=0,
            )
            ax.plot(e, dos, color=beta_color, linewidth=1.0,
                    linestyle="--", dashes=(4, 2), label=label)
        else:
            ax.fill_between(e, 0, dos, alpha=0.25, color=color)
            ax.plot(e, dos, color=color, linewidth=1.2, label=label)

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
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _default_orbitals(self) -> List[str]:
        """获取默认的非 sum 轨道列表。"""
        if hasattr(self.parser, 'non_sum_orbitals'):
            return self.parser.non_sum_orbitals
        return [o for o in self.parser.available_orbitals if o != "sum"]

    # ----------------------------------------------------------
    # 绘图方法一：自旋极化模式
    # ----------------------------------------------------------
    def plot_spin_polarized(
        self,
        orbitals: Optional[List[str]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制自旋极化态密度图。

        α 和 β 自旋都在 y>0 同侧显示：
        - α 自旋（↑）: 实线 + 实心填充
        - β 自旋（↓）: 虚线 + 斜线花纹填充

        参数
        ----
        orbitals : list of str, 可选
            要显示的轨道列表。默认使用所有非 sum 轨道。
        energy_range : (float, float), 可选
            能量范围 (emin, emax)，单位 eV。
        title : str, 可选
            图表标题。默认使用文件名。
        figsize : (float, float)
            图片尺寸（宽, 高），单位英寸。

        返回
        ----
        fig, ax : matplotlib Figure 和 Axes 对象
        """
        if orbitals is None:
            orbitals = self._default_orbitals()

        fig, ax = plt.subplots(figsize=figsize)

        for orb in orbitals:
            base_color = get_orbital_color(orb)

            # α 自旋：实线 + 实心填充
            data_alpha = self.parser.get_data(orbitals=[orb], spin="alpha")
            for label, (e, dos) in data_alpha.items():
                self._plot_series(ax, e, np.abs(dos), base_color, is_beta=False, label=label)

            # β 自旋：虚线 + 斜线填充
            data_beta = self.parser.get_data(orbitals=[orb], spin="beta")
            for label, (e, dos) in data_beta.items():
                self._plot_series(ax, e, np.abs(dos), base_color, is_beta=True, label=label)

        self._add_reference_lines(ax)
        self._style_plot(ax, title, energy_range,
                         xlabel="Energy (eV)", ylabel="PDOS (states/eV)")
        ax.set_ylim(bottom=0)
        return fig, ax

    # ----------------------------------------------------------
    # 绘图方法二：总态密度模式
    # ----------------------------------------------------------
    def plot_total(
        self,
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
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
        sum_color = get_orbital_color("sum")

        total_dos = self.parser.get_total_dos()
        if total_dos is not None:
            e, dos = total_dos
            dos = np.abs(dos)
            ax.fill_between(e, 0, dos, alpha=0.35, color=sum_color)
            ax.plot(e, dos, color=sum_color, linewidth=1.0, label="Total DOS")
        else:
            # 回退方案：手动加和所有非 sum 轨道
            print("[WARN] 未找到 Total/Sum 系列，将对所有轨道数据进行加和。")
            all_data = self.parser.get_data(spin=None)
            if all_data:
                ref_energies = None
                summed_dos = None
                for label, (e, dos) in all_data.items():
                    if "sum" in label.lower():
                        continue
                    if ref_energies is None:
                        ref_energies = e
                        summed_dos = dos.copy()
                    elif len(e) == len(ref_energies):
                        summed_dos += dos
                if summed_dos is not None:
                    ax.fill_between(ref_energies, 0, summed_dos, alpha=0.35, color=sum_color)
                    ax.plot(ref_energies, summed_dos, color=sum_color,
                            linewidth=1.0, label="Total DOS (加和)")

        self._add_reference_lines(ax)
        self._style_plot(ax, title, energy_range,
                         xlabel="Energy (eV)", ylabel="DOS (states/eV)")
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
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
        spin_mode: str = "alpha",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        绘制指定轨道的态密度图（轨道分离模式）。

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
        spin_mode : str
            自旋选择:
              "alpha" — 仅 α 自旋  "beta" — 仅 β 自旋
              "both"  — α/β 分别  "sum"  — α+β 求和

        返回
        ----
        fig, ax : matplotlib Figure 和 Axes 对象
        """
        if orbitals is None:
            orbitals = self._default_orbitals()

        fig, ax = plt.subplots(figsize=figsize)

        # 根据 spin_mode 获取数据
        if spin_mode == "sum" and self.parser.has_spin:
            data = self.parser.get_summed_data(orbitals=orbitals)
        elif spin_mode == "both" and self.parser.has_spin:
            data = self.parser.get_data(orbitals=orbitals, spin="both")
        elif self.parser.has_spin:
            data = self.parser.get_data(orbitals=orbitals, spin=spin_mode)
        else:
            data = self.parser.get_data(orbitals=orbitals, spin=None)

        for label, (e, dos) in data.items():
            dos = np.abs(dos)
            # 从标签中提取轨道名（处理 "s (α)"、"d_xy (alpha)" 等格式）
            orb = label.split()[0] if " " in label else label
            color = get_orbital_color(orb)

            # "both" 模式下 α/β 用不同线型
            is_beta = ("β" in label) or ("down" in label.lower())
            self._plot_series(ax, e, dos, color, is_beta=is_beta, label=label)

        self._add_reference_lines(ax)
        self._style_plot(ax, title, energy_range,
                         xlabel="Energy (eV)", ylabel="PDOS (states/eV)")
        ax.set_ylim(bottom=0)
        return fig, ax

    # ----------------------------------------------------------
    # 绘图方法四：多文件叠加模式
    # ----------------------------------------------------------
    def plot_multi_overlay(
        self,
        series_specs: List[Dict],
        energy_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        多文件叠加绘图：将来自不同文件的系列绘制在同一幅图上。

        每条系列通过 series_spec 字典独立指定数据来源、轨道和自旋聚合方式，
        支持跨文件、跨轨道、跨自旋的灵活组合。

        参数
        ----
        series_specs : list of dict
            每条系列的规格字典，包含以下键:
              - parser: 数据来源解析器实例（必填）
              - file_label (str): 来源文件标识（必填）
              - orbital (str): 轨道名称（必填）
              - spin_agg (str): 自旋聚合方式: "alpha"/"beta"/"both"/"sum"
              - label (str, 可选): 图例标签
        energy_range : (float, float), 可选
            能量范围 (emin, emax)。
        title : str, 可选
            图表标题。
        figsize : (float, float)
            图片尺寸（宽, 高），单位英寸。

        返回
        ----
        fig, ax : matplotlib Figure 和 Axes 对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        color_idx = 0

        for spec in series_specs:
            parser = spec["parser"]
            file_label: str = spec.get("file_label", parser.label)
            orbital: str = spec["orbital"]
            spin_agg: str = spec.get("spin_agg", "alpha")
            custom_label: Optional[str] = spec.get("label")

            # 根据自旋聚合方式获取数据
            if not parser.has_spin:
                data = parser.get_data(orbitals=[orbital], spin=None)
            elif spin_agg == "sum":
                data = parser.get_summed_data(orbitals=[orbital])
            elif spin_agg == "both":
                data = parser.get_data(orbitals=[orbital], spin="both")
            elif spin_agg in ("alpha", "beta"):
                data = parser.get_data(orbitals=[orbital], spin=spin_agg)
            else:
                print(f"[WARN] 未知的自旋聚合方式 '{spin_agg}'，默认使用 alpha。")
                data = parser.get_data(orbitals=[orbital], spin="alpha")

            if not data:
                print(f"[WARN] 未找到数据: {file_label} / {orbital} / {spin_agg}")
                continue

            # 颜色分配：同一 spec 中 α 和 β 共用基色
            base_color = OVERLAY_PALETTE[color_idx % len(OVERLAY_PALETTE)]
            color_idx += 1

            for data_label, (e, dos) in data.items():
                dos_abs = np.abs(dos)

                # 构造图例标签
                if custom_label is not None:
                    if spin_agg == "both":
                        if "β" in data_label:
                            legend_label = f"{custom_label} (β)"
                        else:
                            legend_label = f"{custom_label} (α)"
                    else:
                        legend_label = custom_label
                else:
                    legend_label = data_label
                    if file_label not in data_label:
                        legend_label = f"{file_label} {data_label}"

                # 判断是否 β 自旋
                is_beta = (parser.has_spin and spin_agg == "both"
                          and ("β" in data_label or "down" in data_label.lower()))
                self._plot_series(ax, e, dos_abs, base_color,
                                  is_beta=is_beta, label=legend_label)

        self._add_reference_lines(ax)
        self._style_plot(ax, title, energy_range,
                         xlabel="Energy (eV)", ylabel="PDOS (states/eV)")
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
        _fp = _CJK_FONT_PROP  # 使用全局 CJK 字体确保中文正常渲染
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", fontproperties=_fp)
        else:
            ax.set_title(self.parser.filename, fontsize=12, fontproperties=_fp)

        ax.set_xlabel(xlabel, fontsize=12, fontproperties=_fp)
        ax.set_ylabel(ylabel, fontsize=12, fontproperties=_fp)

        if energy_range is not None:
            ax.set_xlim(energy_range)

        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        fig = ax.figure
        if fig is not None:
            fig.tight_layout()

    # ----------------------------------------------------------
    # 公用：保存图片
    # ----------------------------------------------------------
    def save_figure(
        self,
        fig: plt.Figure,
        save_path: str,
        dpi: int = DEFAULT_DPI,
    ) -> None:
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
        dpi : int
            输出分辨率（默认 300）。
        """
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[INFO] 图片已保存至: {save_path}")
