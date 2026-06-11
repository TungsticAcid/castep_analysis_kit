#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
InteractivePreview —— matplotlib 预览窗口交互功能
===================================================

为绑图预览窗口添加十字光标、手动识峰、轨道重叠分析等交互功能。

改进（2026/06/11 重构）:
  - PDOSAnalyzer 通过依赖注入传入，而非内部直接实例化
  - 识峰/重叠分析相关的魔法数字提取为类常量
  - 使用 constants 模块中的颜色常量

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import warnings
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# ── 配置 matplotlib 中文字体支持 ──────────────────────────────
# 策略：搜索 CJK 字体文件，创建 FontProperties 供所有 text/set_title 使用
_CJK_FONT_PROP = None

def _setup_cjk_font() -> None:
    global _CJK_FONT_PROP
    _CJK_SEARCH_PATHS = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simkai.ttf"),
    ]
    for _fp_path in _CJK_SEARCH_PATHS:
        if _fp_path.exists():
            try:
                _prop = FontProperties(fname=str(_fp_path))
                _prop.get_name()
                _CJK_FONT_PROP = _prop
                print(f"[INFO] CJK 字体已加载: {_fp_path.name}")
                return
            except Exception:
                continue
    print("[WARN] 未找到 CJK 字体文件，中文可能无法正常显示。")

_setup_cjk_font()

# 抑制 TkAgg 后端因中文字形缺失产生的 UserWarning
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")
# ──────────────────────────────────────────────────────────────

try:
    from .constants import (
        COLOR_PEAK_MODE,
        COLOR_OVERLAP_A,
        COLOR_OVERLAP_B,
        COLOR_OVERLAP_RESULT,
        COLOR_SERIES_INDICATOR,
        COLOR_SERIES_INDICATOR_BG,
        COLOR_PEAK_RESULT_BG,
        COLOR_OVERLAP_RESULT_BG,
        COLOR_FALLBACK,
    )
except ImportError:
    from constants import (
        COLOR_PEAK_MODE,
        COLOR_OVERLAP_A,
        COLOR_OVERLAP_B,
        COLOR_OVERLAP_RESULT,
        COLOR_SERIES_INDICATOR,
        COLOR_SERIES_INDICATOR_BG,
        COLOR_PEAK_RESULT_BG,
        COLOR_OVERLAP_RESULT_BG,
        COLOR_FALLBACK,
    )


# ============================================================
# InteractivePreview 类
# ============================================================
class InteractivePreview:
    """
    为 matplotlib 预览窗口添加交互功能。

    功能:
      - 十字光标：跟随鼠标的横纵辅助线，左上角实时坐标显示
      - 手动识峰：按 p 键进入模式，点击两点定义峰区间
      - 峰值显示模式：按 d 键循环切换（全部/简要/仅位置/详细）
      - 多系列切换：按 n 键在多文件叠加的系列间切换
      - 基线切换：按 b 键切换是否扣除线性基线
      - 轨道重叠分析：按 1 选系列 A，按 2 选系列 B，按 t 进入点击模式
      - 清除结果：按 c 键清除识峰标记和重叠结果
      - 自旋切换：按 s 键切换识峰自旋方向
    """

    # 峰值显示模式定义
    PEAK_DISPLAY_MODES = [
        {"label": "全部", "fields": ["area", "center", "position", "height", "fwhm", "tailing"]},
        {"label": "简要", "fields": ["position", "area"]},
        {"label": "仅位置", "fields": ["position"]},
        {"label": "位置+面积+FWHM", "fields": ["position", "height", "area", "fwhm"]},
    ]

    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        parser,
        selected_orbitals: List[str],
        has_spin: bool,
        plot_mode: str,
        spin_mode: str = "alpha",
        overlay_specs: Optional[List[Dict]] = None,
        analyzer=None,  # 依赖注入
    ) -> None:
        """
        初始化交互预览。

        参数
        ----
        fig, ax : matplotlib Figure/Axes
            绘图对象。
        parser : PDOSParser 或 CastepPDOSAdapter
            数据源。
        selected_orbitals : list
            当前预览中选中的轨道列表。
        has_spin : bool
            文件是否有自旋极化。
        plot_mode : str
            绘图模式（"spin"/"orbitals"/"total"/"multi_overlay"）。
        spin_mode : str
            自旋选择: "alpha"/"beta"/"both"/"sum"。
        overlay_specs : list of dict, 可选
            多文件叠加模式的系列规格列表。
        analyzer : PDOSAnalyzer, 可选
            分析器实例（依赖注入）。None 时延迟创建。
        """
        self.fig = fig
        self.ax = ax
        self.parser = parser
        self.selected_orbitals = selected_orbitals
        self.has_spin = has_spin
        self.plot_mode = plot_mode
        self.spin_mode = spin_mode
        self.original_title: str = ax.get_title()
        self._analyzer = analyzer

        # 为 axes 打猴子补丁，自动为所有 set_title / text 添加 CJK 字体
        if _CJK_FONT_PROP:
            _orig_set_title = ax.set_title
            _orig_text = ax.text
            def _patched_set_title(*a, **kw):
                if 'fontproperties' not in kw:
                    kw['fontproperties'] = _CJK_FONT_PROP
                return _orig_set_title(*a, **kw)
            def _patched_text(*a, **kw):
                if 'fontproperties' not in kw:
                    kw['fontproperties'] = _CJK_FONT_PROP
                return _orig_text(*a, **kw)
            ax.set_title = _patched_set_title
            ax.text = _patched_text

        # 多文件叠加模式：系列切换支持
        self._overlay_specs: List[Dict] = overlay_specs or []
        self._overlay_series_idx: int = 0

        # 峰值显示模式
        self._peak_display_mode: int = 0

        # 轨道重叠分析状态
        self._overlap_mode: bool = False
        self._overlap_clicks: List[Tuple[float, float]] = []
        self._overlap_series_a: Optional[int] = None
        self._overlap_series_b: Optional[int] = None
        self._overlap_result_texts: List = []
        self._overlap_markers: List = []

        # 当前分析轨道（需在系列指示器更新前初始化）
        self._analysis_orbital: str = ""

        # 识峰自旋选择（需在系列指示器更新前初始化）
        if self.has_spin:
            self._peak_spins = ["alpha", "beta", "sum"]
            self._peak_spin: Optional[str] = "alpha"
        else:
            self._peak_spins = [None]
            self._peak_spin: Optional[str] = None
        self._peak_spin_idx: int = 0

        # 十字光标
        self.cross_h = ax.axhline(0, color="#666666", ls="--", lw=0.6, alpha=0)
        self.cross_v = ax.axvline(0, color="#666666", ls="--", lw=0.6, alpha=0)
        self.coord_text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )

        # 识峰状态
        self._peak_mode: bool = False
        self._peak_clicks: List[Tuple[float, float]] = []
        self._peak_markers: List = []
        self._peak_region_lines: List = []
        self._peak_result_texts: List = []
        self._subtract_baseline: bool = True

        # 快捷键提示
        self._hint_text = ax.text(
            0.98, 0.02, "", transform=ax.transAxes,
            fontsize=7, va="bottom", ha="right", color="#999999",
        )
        self._update_hint_text()

        # 连接事件（鼠标/点击）
        self._cid_motion = fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._cid_click = fig.canvas.mpl_connect("button_press_event", self._on_click)

        # 清除 matplotlib 工具栏注册的键盘快捷键回调（p=平移、s=保存等），
        # 避免与交互预览的识峰快捷键冲突。工具栏的鼠标操作不受影响。
        # 使用官方 mpl_disconnect API 逐个断开，避免破坏 CallbackRegistry 内部簿记。
        _cb_registry = fig.canvas.callbacks
        try:
            # callbacks 是 CallbackRegistry 内部存储回调的 defaultdict(list)
            _cb_slots = _cb_registry.callbacks.get('key_press_event', [])
            for _cid, *_ in list(_cb_slots):
                _cb_registry.disconnect(_cid)
            if _cb_slots:
                print(f"[INFO] 已清除 {len(_cb_slots)} 个工具栏键盘快捷键绑定")
        except Exception as _e:
            print(f"[WARN] 清除键盘回调失败: {_e}")
        self._cid_key = fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        print(f"[INFO] 键盘事件已注册 cid={self._cid_key}")

        print("[INFO] 交互预览已激活 — p=识峰  b=基线  d=显示模式  c=清除")
        print("[INFO]   n=切换系列  1/2=选重叠目标  t=点击计算重叠面积")

    @property
    def analyzer(self):
        """延迟创建 PDOSAnalyzer（依赖注入模式）。"""
        if self._analyzer is None:
            try:
                from .pdos_analyzer import PDOSAnalyzer
            except ImportError:
                from pdos_analyzer import PDOSAnalyzer
            self._analyzer = PDOSAnalyzer(self.parser)
        return self._analyzer

    # ----------------------------------------------------------
    # 十字光标
    # ----------------------------------------------------------
    def _on_mouse_move(self, event: mpl.backend_bases.MouseEvent) -> None:
        """鼠标移动：更新十字光标位置和坐标显示。"""
        if event.inaxes != self.ax:
            self.cross_h.set_alpha(0)
            self.cross_v.set_alpha(0)
            self.coord_text.set_text("")
            self.fig.canvas.draw_idle()
            return
        self.cross_h.set_ydata([event.ydata, event.ydata])
        self.cross_v.set_xdata([event.xdata, event.xdata])
        self.cross_h.set_alpha(0.4)
        self.cross_v.set_alpha(0.4)
        self.coord_text.set_text(f"E = {event.xdata:.4f} eV\nDOS = {event.ydata:.4f}")
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    # 键盘事件
    # ----------------------------------------------------------
    def _on_key_press(self, event: mpl.backend_bases.KeyEvent) -> None:
        """键盘事件处理: p=识峰 c=清除 b=基线 n=切换系列 s=切换自旋 d=显示模式 1/2=重叠 t=计算"""
        print(f"[DEBUG] 按键事件: key={event.key!r}")
        if event.key == "p":
            self._toggle_peak_mode()
        elif event.key == "c":
            self._clear_peak_results()
            self._clear_overlap_results()
        elif event.key == "b":
            self._toggle_baseline()
        elif event.key == "n":
            self._cycle_orbital()
        elif event.key == "s":
            self._cycle_spin()
        elif event.key == "d":
            self._cycle_peak_display()
        elif event.key == "1":
            self._select_overlap_a()
        elif event.key == "2":
            self._select_overlap_b()
        elif event.key == "t":
            self._toggle_overlap_mode()

    # ----------------------------------------------------------
    # 识峰模式
    # ----------------------------------------------------------
    def _toggle_peak_mode(self) -> None:
        """切换识峰模式 (p 键)。"""
        self._peak_mode = not self._peak_mode
        if self._peak_mode:
            self._peak_clicks = []
            self._clear_peak_markers()
            # 初始化识峰目标
            if self.plot_mode == "multi_overlay" and self._overlay_specs:
                self._overlay_series_idx = 0
                spec = self._overlay_specs[0]
                self._analysis_orbital = spec["orbital"]
                self._peak_spin = spec.get("spin_agg", "sum")
            else:
                self._overlay_series_idx = 0
                non_sum = [o for o in self.selected_orbitals if o != "sum"]
                self._analysis_orbital = non_sum[0] if non_sum else self.selected_orbitals[0]
            self.ax.set_title("识峰模式: 请点击两个点定义峰区间 (p=退出 b=切换基线)",
                             fontsize=12, color=COLOR_PEAK_MODE, fontweight="bold")
            self._update_hint_text()
        else:
            self._clear_peak_markers()
            self.ax.set_title(self.original_title, fontsize=14, color="black")
            self._update_hint_text()
        self.fig.canvas.draw_idle()

    def _toggle_baseline(self) -> None:
        """切换是否扣除线性基线 (b 键)。"""
        self._subtract_baseline = not self._subtract_baseline
        status = "扣除基线" if self._subtract_baseline else "不扣除基线（原始 DOS）"
        print(f"[INFO] 手动识峰: {status}")
        self._update_hint_text()
        if self._peak_result_texts and len(self._peak_clicks) == 2:
            self._clear_peak_result_texts()
            self._analyze_peak()

    def _cycle_orbital(self) -> None:
        """切换识峰目标轨道/系列 (n 键)。"""
        if self.plot_mode == "multi_overlay" and self._overlay_specs:
            if len(self._overlay_specs) <= 1:
                return
            self._overlay_series_idx = (self._overlay_series_idx + 1) % len(self._overlay_specs)
            spec = self._overlay_specs[self._overlay_series_idx]
            self._analysis_orbital = spec["orbital"]
            self._peak_spin = spec.get("spin_agg", "sum")
            print(f"[INFO] 识峰系列切换为: {spec.get('label', spec['file_label'] + ' ' + spec['orbital'])}")
        else:
            non_sum = [o for o in self.selected_orbitals if o != "sum"]
            if not non_sum:
                return
            idx = non_sum.index(self._analysis_orbital) if self._analysis_orbital in non_sum else -1
            self._analysis_orbital = non_sum[(idx + 1) % len(non_sum)]
            print(f"[INFO] 识峰轨道切换为: {self._analysis_orbital}")
        self._update_hint_text()
        if self._peak_result_texts and len(self._peak_clicks) == 2:
            self._clear_peak_result_texts()
            self._analyze_peak()

    def _cycle_spin(self) -> None:
        """切换识峰自旋选择 (s 键)。"""
        if self.plot_mode == "multi_overlay":
            return
        if not self._peak_spins:
            return
        self._peak_spin_idx = (self._peak_spin_idx + 1) % len(self._peak_spins)
        self._peak_spin = self._peak_spins[self._peak_spin_idx]
        spin_label = self._peak_spin if self._peak_spin else "default"
        print(f"[INFO] 识峰自旋切换为: {spin_label}")
        self._update_hint_text()
        if self._peak_result_texts and len(self._peak_clicks) == 2:
            self._clear_peak_result_texts()
            self._analyze_peak()

    def _cycle_peak_display(self) -> None:
        """切换峰值显示模式 (d 键)。"""
        self._peak_display_mode = (self._peak_display_mode + 1) % len(self.PEAK_DISPLAY_MODES)
        mode_label = self.PEAK_DISPLAY_MODES[self._peak_display_mode]["label"]
        print(f"[INFO] 峰值显示模式切换为: {mode_label}")
        self._update_hint_text()
        if self._peak_result_texts and len(self._peak_clicks) == 2:
            self._clear_peak_result_texts()
            self._analyze_peak()

    # ----------------------------------------------------------
    # 重叠分析
    # ----------------------------------------------------------
    def _select_overlap_a(self) -> None:
        """选择系列 A (1 键)。"""
        if self.plot_mode == "multi_overlay" and self._overlay_specs:
            self._overlap_series_a = self._overlay_series_idx
        else:
            self._overlap_series_a = None
        self._clear_overlap_results()
        label = self._current_series_label()
        self._draw_overlap_label("A", label, COLOR_OVERLAP_A)
        print(f"[INFO] 重叠分析 A = [{label}]")

    def _select_overlap_b(self) -> None:
        """选择系列 B (2 键)。"""
        if self.plot_mode == "multi_overlay" and self._overlay_specs:
            self._overlap_series_b = self._overlay_series_idx
        else:
            self._overlap_series_b = None
        label = self._current_series_label()
        self._draw_overlap_label("B", label, COLOR_OVERLAP_B)
        print(f"[INFO] 重叠分析 B = [{label}]")

    def _draw_overlap_label(self, tag: str, label: str, color: str) -> None:
        """在图上绘制重叠分析的 A/B 标记。"""
        y_pos = 0.80 if tag == "A" else 0.72
        marker = self.ax.text(
            0.02, y_pos, f"重叠{tag}: {label}", transform=self.ax.transAxes,
            fontsize=8, va="top", ha="left", fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.9),
        )
        self._overlap_markers.append(marker)
        self.fig.canvas.draw_idle()

    def _toggle_overlap_mode(self) -> None:
        """进入/退出重叠分析点击模式 (t 键)。"""
        if self._peak_mode:
            return
        if self._overlap_series_a is None and self._overlap_series_b is None:
            print("[INFO] 重叠分析: 请先按 1 选择系列 A。")
            return
        self._overlap_mode = not self._overlap_mode
        if self._overlap_mode:
            if self._overlap_series_b is None:
                self._overlap_series_b = self._overlap_series_a
            self._overlap_clicks = []
            self.ax.set_title("重叠分析: 点击两个点定义积分区间 (t=退出)",
                             fontsize=12, color=COLOR_OVERLAP_RESULT, fontweight="bold")
        else:
            self.ax.set_title(self.original_title, fontsize=14, color="black")
        self._update_hint_text()
        self.fig.canvas.draw_idle()

    def _execute_overlap(self) -> None:
        """执行重叠面积计算。"""
        if len(self._overlap_clicks) < 2:
            return
        emin_val = min(self._overlap_clicks[0][0], self._overlap_clicks[1][0])
        emax_val = max(self._overlap_clicks[0][0], self._overlap_clicks[1][0])

        pa = self._get_overlap_parser(self._overlap_series_a)
        pb = self._get_overlap_parser(self._overlap_series_b)
        orb_a = self._get_overlap_orbital(self._overlap_series_a)
        orb_b = self._get_overlap_orbital(self._overlap_series_b)
        spin_a = self._get_overlap_spin(self._overlap_series_a)
        spin_b = self._get_overlap_spin(self._overlap_series_b)

        result = self.analyzer.calc_overlap_area(
            orbital_a=orb_a, orbital_b=orb_b,
            spin_a=spin_a, spin_b=spin_b,
            parser_a=pa, parser_b=pb,
            emin=emin_val, emax=emax_val, normalize=True, verbose=False,
        )

        label_a = f"{orb_a}" + (f"({spin_a})" if spin_a else "")
        label_b = f"{orb_b}" + (f"({spin_b})" if spin_b else "")
        text = (f"重叠 [{label_a} vs {label_b}]\n"
                f"区间 [{emin_val:.3f}, {emax_val:.3f}] eV\n"
                f"指数 = {result.get('overlap', float('nan')):.4f}")
        print(f"[INFO] {text.replace(chr(10), ' | ')}")

        y_top = 0.55
        t = self.ax.text(
            0.02, y_top, text, transform=self.ax.transAxes,
            fontsize=8, va="top", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_OVERLAP_RESULT_BG,
                      edgecolor=COLOR_OVERLAP_RESULT, alpha=0.9),
        )
        self._overlap_result_texts.append(t)
        self._clear_overlap_markers()
        self.fig.canvas.draw_idle()

    def _get_overlap_parser(self, series_idx):
        if self.plot_mode == "multi_overlay" and self._overlay_specs and series_idx is not None:
            if 0 <= series_idx < len(self._overlay_specs):
                return self._overlay_specs[series_idx]["parser"]
        return self.parser

    def _get_overlap_orbital(self, series_idx) -> str:
        if self.plot_mode == "multi_overlay" and self._overlay_specs and series_idx is not None:
            if 0 <= series_idx < len(self._overlay_specs):
                return self._overlay_specs[series_idx]["orbital"]
        return self._analysis_orbital if self._analysis_orbital else self.selected_orbitals[0]

    def _get_overlap_spin(self, series_idx) -> Optional[str]:
        if self.plot_mode == "multi_overlay" and self._overlay_specs and series_idx is not None:
            if 0 <= series_idx < len(self._overlay_specs):
                return self._overlay_specs[series_idx].get("spin_agg", "sum")
        return self._peak_spin

    # ----------------------------------------------------------
    # 鼠标点击事件
    # ----------------------------------------------------------
    def _on_click(self, event: mpl.backend_bases.MouseEvent) -> None:
        """鼠标点击事件分发。"""
        if event.inaxes != self.ax or event.xdata is None:
            return

        if self._overlap_mode:
            self._overlap_clicks.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'o', color=COLOR_OVERLAP_RESULT,
                        markersize=8, markerfacecolor='none', markeredgewidth=2)
            self.fig.canvas.draw_idle()
            if len(self._overlap_clicks) == 2:
                self._execute_overlap()
                self._overlap_clicks = []
                self._overlap_mode = False
                self.ax.set_title(self.original_title, fontsize=14, color="black")
            return

        if self._peak_mode:
            self._peak_clicks.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'o', color=COLOR_PEAK_MODE,
                        markersize=8, markerfacecolor='none', markeredgewidth=2)
            self.fig.canvas.draw_idle()
            if len(self._peak_clicks) == 2:
                self._analyze_peak()
            return

    def _analyze_peak(self) -> None:
        """执行手动识峰分析。"""
        if len(self._peak_clicks) < 2:
            return
        (e1, y1), (e2, y2) = self._peak_clicks
        emin_val = min(e1, e2)
        emax_val = max(e1, e2)

        result = self.analyzer.calc_manual_peak(
            orbital=self._analysis_orbital,
            emin=emin_val, emax=emax_val,
            y1=y1, y2=y2,
            spin=self._peak_spin,
            subtract_baseline=self._subtract_baseline,
        )

        # 渲染结果
        self._render_peak_result(result, emin_val, emax_val)
        self._peak_clicks = []

    def _render_peak_result(self, result: Dict, emin_val: float, emax_val: float) -> None:
        """根据当前显示模式渲染峰值分析结果标签。"""
        mode = self.PEAK_DISPLAY_MODES[self._peak_display_mode]
        fields = mode["fields"]

        lines = [f"[{self._analysis_orbital.upper()}] {mode['label']}"]
        lines.append(f"区间 [{emin_val:.3f}, {emax_val:.3f}] eV")

        field_labels = {
            "position": lambda r: f"峰值: {r['peak_max_e']:.4f} eV",
            "height":   lambda r: f"峰高: {r['peak_max_dos']:.4f}",
            "area":     lambda r: f"面积: {r['peak_area']:.4f}",
            "center":   lambda r: f"质心: {r['peak_center']:.4f} eV",
            "fwhm":     lambda r: f"FWHM: {r['fwhm']:.4f} eV",
            "tailing":  lambda r: f"拖尾: {r['tailing_factor']:.4f}",
        }

        for f in fields:
            if f in field_labels:
                lines.append(field_labels[f](result))

        bl = "扣除" if self._subtract_baseline else "未扣除"
        lines.append(f"基线: {bl}")

        text = "\n".join(lines)
        y_pos = 0.50 - len(self._peak_result_texts) * 0.12
        t = self.ax.text(
            0.02, y_pos, text, transform=self.ax.transAxes,
            fontsize=8, va="top", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_PEAK_RESULT_BG,
                      edgecolor=COLOR_PEAK_MODE, alpha=0.9),
        )
        self._peak_result_texts.append(t)
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    # 清除方法
    # ----------------------------------------------------------
    def _clear_peak_markers(self) -> None:
        for m in self._peak_markers:
            m.remove()
        self._peak_markers.clear()
        for line in self._peak_region_lines:
            line.remove()
        self._peak_region_lines.clear()

    def _clear_peak_result_texts(self) -> None:
        for t in self._peak_result_texts:
            t.remove()
        self._peak_result_texts.clear()

    def _clear_peak_results(self) -> None:
        self._clear_peak_markers()
        self._clear_peak_result_texts()
        self._peak_clicks = []
        self.fig.canvas.draw_idle()

    def _clear_overlap_results(self) -> None:
        for t in self._overlap_result_texts:
            t.remove()
        self._overlap_result_texts.clear()
        self._clear_overlap_markers()

    def _clear_overlap_markers(self) -> None:
        for m in self._overlap_markers:
            m.remove()
        self._overlap_markers.clear()

    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # UI 更新
    # ----------------------------------------------------------
    def _current_series_label(self) -> str:
        """获取当前系列的可读标签。"""
        if self.plot_mode == "multi_overlay" and self._overlay_specs:
            spec = self._overlay_specs[self._overlay_series_idx]
            return spec.get("label", f"{spec['file_label']} {spec['orbital']}")
        spin_tag = f" ({self._peak_spin})" if self._peak_spin else ""
        return f"{self.parser.label} {self._analysis_orbital}{spin_tag}"

    def _update_hint_text(self) -> None:
        """更新快捷键提示文字。"""
        hints = ["p=识峰", "c=清除"]
        if self._peak_mode:
            hints = ["p=退出识峰", "b=基线", "s=自旋", "n=切换", "d=显示", "c=清除"]
        if self._overlap_mode:
            hints = ["t=退出重叠"]
        elif self._overlap_series_a is not None or self._overlap_series_b is not None:
            hints = ["1=A", "2=B", "t=计算重叠", "c=清除"]

        mode_info = ""
        if self._peak_mode:
            bl = "基线" if self._subtract_baseline else "原DOS"
            mode_info = f" | 识峰:{self._analysis_orbital} [{bl}]"

        self._hint_text.set_text(" ".join(hints) + mode_info)
        self.fig.canvas.draw_idle()
