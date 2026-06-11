#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDOSGUI —— tkinter 图形用户界面
================================

提供 PDOS 态密度数据提取与绘图的完整图形界面。

支持功能:
  - 单文件 / 多文件叠加模式
  - 轨道选择、自旋模式切换
  - 能量范围自定义
  - 图片预览与保存
  - 态密度分析（d带中心、峰搜索、晶场劈裂等）
  - 最近文件记录

改进（2026/06/11 重构）:
  - _build_ui() 拆分为 8 个子工厂方法
  - 分析选项对话框提取为独立内部类 AnalysisDialog
  - 统一多文件管理为 _insert_file_entry / _remove_file_entry / _reindex_file_list
  - 使用 constants 模块中的常量
  - 适配 CastepPDOSAdapter（duck-typing）

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11
"""
from __future__ import annotations

import os
import json
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np

try:
    from .constants import (
        DEFAULT_PIC_DIR,
        DEFAULT_WINDOW_SIZE,
        DEFAULT_DPI,
        DEFAULT_FIGSIZE,
        ORBITAL_COLORS,
        OVERLAY_PALETTE,
        SPIN_KEY_TO_GREEK,
        SPIN_SHORT_TO_KEY,
        MAX_RECENT_FILES,
        RECENT_FILES_PATH,
        XCD_EXTENSION,
        DEFAULT_OUTPUT_FILENAME,
        COLOR_FALLBACK,
    )
    from .pdos_parser import PDOSParser
    from .pdos_plotter import PDOSPlotter
    from .castep_bin_parser import CastepBinParser
except ImportError:
    from constants import (
        DEFAULT_PIC_DIR,
        DEFAULT_WINDOW_SIZE,
        DEFAULT_DPI,
        DEFAULT_FIGSIZE,
        ORBITAL_COLORS,
        OVERLAY_PALETTE,
        SPIN_KEY_TO_GREEK,
        SPIN_SHORT_TO_KEY,
        MAX_RECENT_FILES,
        RECENT_FILES_PATH,
        XCD_EXTENSION,
        DEFAULT_OUTPUT_FILENAME,
        COLOR_FALLBACK,
    )
    from pdos_parser import PDOSParser
    from pdos_plotter import PDOSPlotter
    from castep_bin_parser import CastepBinParser


# ============================================================
# AnalysisDialog —— 分析选项对话框
# ============================================================
class AnalysisDialog(tk.Toplevel):
    """
    分析选项对话框（独立类，从 PDOSGUI._show_analysis_options 提取）。

    允许用户选择要执行的分析项:
      - 带中心 (全部态 / 占据态)
      - 带宽 (σ / FWHM)
      - 晶场劈裂 / 自旋劈裂
      - 占据态比例
      - 积分面积
      - 峰列表
    """

    ANALYSIS_OPTIONS = [
        ("band_center",    "带中心 (全部态 + 占据态)"),
        ("band_width",     "带宽 (标准差 / FWHM)"),
        ("spin_splitting", "自旋劈裂 (α−β)"),
        ("crystal_field",  "晶场劈裂 (主峰间距)"),
        ("occupancy",      "占据态比例 (E<0)"),
        ("peak_area",      "积分面积"),
        ("peaks",          "峰列表"),
    ]

    def __init__(self, parent: tk.Widget, callback: Callable[[Dict[str, bool]], None]) -> None:
        """
        初始化分析选项对话框。

        参数
        ----
        parent : tk.Widget
            父窗口。
        callback : callable
            用户点击确认后的回调函数，接收 {option_key: bool} 字典。
        """
        super().__init__(parent)
        self.title("分析选项")
        self.geometry("350x360")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._callback = callback
        self._vars: Dict[str, tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """构建对话框 UI。"""
        main_frame = ttk.Frame(self, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="请选择要执行的分析项:",
                  font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))

        for key, label in self.ANALYSIS_OPTIONS:
            var = tk.BooleanVar(value=True)
            self._vars[key] = var
            ttk.Checkbutton(main_frame, text=label, variable=var).pack(anchor=tk.W, padx=10, pady=2)

        # 按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))

        ttk.Button(btn_frame, text="全选", command=self._select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="全不选", command=self._deselect_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="确认", command=self._on_ok).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT)

    def _select_all(self) -> None:
        for v in self._vars.values():
            v.set(True)

    def _deselect_all(self) -> None:
        for v in self._vars.values():
            v.set(False)

    def _on_ok(self) -> None:
        result = {key: var.get() for key, var in self._vars.items()}
        self._callback(result)
        self.destroy()


# ============================================================
# PDOSGUI 类
# ============================================================
class PDOSGUI:
    """
    PDOS 图形用户界面主窗口。

    使用方式:
        gui = PDOSGUI(initial_file="path/to/PDOS.xcd")
        gui.run()
    """

    def __init__(self, initial_file: Optional[str] = None) -> None:
        """初始化 GUI。"""
        self.root = tk.Tk()
        self.root.title("PDOS 态密度分析工具")
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        self.root.minsize(600, 500)

        # 数据状态
        self.loaded_files: List[Dict] = []          # [{"path", "parser", "label", "has_spin", "orbitals"}]
        self._file_counter: int = 0                  # 用于默认标签编号
        self._active_file_idx: int = 0               # 当前选中的文件索引
        self._overlay_series: List[Dict] = []        # 多文件叠加系列规格
        self._recent_files: List[str] = []

        # tkinter 变量
        self._var_plot_mode = tk.StringVar(value="orbitals")
        self._var_spin_mode = tk.StringVar(value="alpha")
        self._var_emin = tk.StringVar(value="")
        self._var_emax = tk.StringVar(value="")
        self._var_custom_energy = tk.BooleanVar(value=False)
        self._var_save_dir = tk.StringVar(value=str(DEFAULT_PIC_DIR))
        self._var_output_name = tk.StringVar(value=DEFAULT_OUTPUT_FILENAME)

        # 轨道选择 checkbox 变量
        self._orbital_vars: Dict[str, tk.BooleanVar] = {}

        # 图形状态
        self.current_fig: Optional["plt.Figure"] = None

        # 加载最近文件
        self._load_recent_files()

        # 构建 UI
        self._build_root()

        # 加载初始文件
        if initial_file and os.path.isfile(initial_file):
            self._add_multi_file_from_path(initial_file)

    # ================================================================
    # 根窗口构建
    # ================================================================
    def _build_root(self) -> None:
        """构建主窗口结构（主 PanedWindow + 菜单栏）。"""
        # 菜单栏
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开文件...", command=self._browse_file)
        file_menu.add_command(label="打开最近文件", command=self._show_recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)

        # 主面板: 左侧控制面板 + 右侧文件管理面板
        main_pw = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        main_pw.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板（可滚动）
        self._ctrl_canvas = tk.Canvas(main_pw, width=420)
        ctrl_scroll = ttk.Scrollbar(main_pw, orient=tk.VERTICAL, command=self._ctrl_canvas.yview)
        self._ctrl_frame = ttk.Frame(self._ctrl_canvas, padding=10)
        self._ctrl_frame.bind("<Configure>",
                              lambda e: self._ctrl_canvas.configure(scrollregion=self._ctrl_canvas.bbox("all")))
        self._ctrl_canvas.create_window((0, 0), window=self._ctrl_frame, anchor=tk.NW)
        self._ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)

        main_pw.add(self._ctrl_canvas)
        main_pw.add(ctrl_scroll)

        # 右侧多文件管理面板
        right_frame = ttk.Frame(main_pw, padding=10)
        main_pw.add(right_frame)

        # 构建各区域
        self._build_file_selector(right_frame)
        self._build_orbital_panel(self._ctrl_frame)
        self._build_overlay_builder(self._ctrl_frame)
        self._build_mode_selector(self._ctrl_frame)
        self._build_spin_selector(self._ctrl_frame)
        self._build_energy_range(self._ctrl_frame)
        self._build_castep_params(self._ctrl_frame)
        self._build_output_settings(self._ctrl_frame)
        self._build_action_buttons(self._ctrl_frame)

        # 绑定滚轮
        self._ctrl_canvas.bind_all("<MouseWheel>",
                                   lambda e: self._ctrl_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    # ================================================================
    # UI 子区域构建（从 _build_ui 拆分）
    # ================================================================

    def _build_file_selector(self, parent: ttk.Frame) -> None:
        """构建右侧多文件管理面板。"""
        ttk.Label(parent, text="加载的文件列表", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)

        # 文件列表
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self._file_listbox = tk.Listbox(list_frame, height=12, width=35, exportselection=False)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._file_listbox.yview)
        self._file_listbox.configure(yscrollcommand=scroll.set)
        self._file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_list_select)
        # 右键菜单：粘贴路径
        self._file_listbox.bind("<Button-3>", self._on_file_list_right_click)

        # 按钮行 1
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="+ 添加文件", command=self._browse_file).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(btn_frame, text="📂 最近", command=self._show_multi_recent_menu).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="✕ 移除", command=self._remove_multi_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="✎ 标签", command=self._edit_file_label).pack(side=tk.LEFT, padx=3)

        # 文件信息
        self._file_info_label = ttk.Label(parent, text="", font=("Microsoft YaHei", 8), foreground="gray")
        self._file_info_label.pack(anchor=tk.W, pady=(5, 0))

    def _build_orbital_panel(self, parent: ttk.Frame) -> None:
        """构建轨道选择面板（树形 + 按钮）。"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        hdr = ttk.Frame(parent)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="轨道选择", font=("Microsoft YaHei", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(hdr, text="全选", command=self._tree_select_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(hdr, text="全不选", command=self._tree_deselect_all).pack(side=tk.RIGHT, padx=2)
        ttk.Button(hdr, text="晶体查看", command=self._open_crystal_viewer).pack(side=tk.RIGHT, padx=2)

        # OrbitalTree 替换原来的扁平 checkbox 面板
        from .orbital_tree import OrbitalTree
        self._orbital_tree = OrbitalTree(parent)
        self._orbital_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        self._orbital_frame = self._orbital_tree  # 兼容旧代码

    def _build_overlay_builder(self, parent: ttk.Frame) -> None:
        """构建叠加系列构建区域。"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        hdr = ttk.Frame(parent)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="叠加系列构建", font=("Microsoft YaHei", 10, "bold")).pack(side=tk.LEFT)

        # 系列列表
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.X, pady=5)
        self._overlay_listbox = tk.Listbox(list_frame, height=5, exportselection=False)
        self._overlay_listbox.pack(fill=tk.X)

        # 按钮行
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="+ 添加", command=self._add_overlay_series).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(btn_frame, text="✕ 移除", command=self._remove_overlay_series).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="清空", command=self._clear_overlay_series).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="↑ 上移", command=self._move_overlay_up).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="↓ 下移", command=self._move_overlay_down).pack(side=tk.LEFT, padx=3)

    def _build_mode_selector(self, parent: ttk.Frame) -> None:
        """构建绑图模式选择区域。"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="绑图模式", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)

        mode_frame = ttk.Frame(parent)
        mode_frame.pack(fill=tk.X, pady=5)

        self._mode_btns = {}
        for mode_key, mode_label in [("orbitals", "轨道分离"), ("spin", "自旋极化"), ("total", "总态密度"),
                                      ("overlay", "多文件叠加")]:
            btn = ttk.Radiobutton(mode_frame, text=mode_label, variable=self._var_plot_mode,
                                  value=mode_key, command=self._on_plot_mode_changed)
            btn.pack(side=tk.LEFT, padx=3)
            self._mode_btns[mode_key] = btn

    def _build_spin_selector(self, parent: ttk.Frame) -> None:
        """构建自旋模式选择区域。"""
        self._spin_frame = ttk.Frame(parent)
        self._spin_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self._spin_frame, text="自旋模式:").pack(side=tk.LEFT)

        self._spin_combo = ttk.Combobox(self._spin_frame, textvariable=self._var_spin_mode,
                                         values=["alpha", "beta", "both", "sum"], state="readonly", width=8)
        self._spin_combo.pack(side=tk.LEFT, padx=5)
        self._spin_combo.bind("<<ComboboxSelected>>", self._on_spin_mode_changed)

    def _build_energy_range(self, parent: ttk.Frame) -> None:
        """构建能量范围输入区域。"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="能量范围 (eV)", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)

        range_frame = ttk.Frame(parent)
        range_frame.pack(fill=tk.X, pady=5)
        ttk.Label(range_frame, text="E_min:").pack(side=tk.LEFT)
        self._entry_emin = ttk.Entry(range_frame, textvariable=self._var_emin, width=8,
                                      state="disabled")
        self._entry_emin.pack(side=tk.LEFT, padx=(3, 10))
        ttk.Label(range_frame, text="E_max:").pack(side=tk.LEFT)
        self._entry_emax = ttk.Entry(range_frame, textvariable=self._var_emax, width=8,
                                      state="disabled")
        self._entry_emax.pack(side=tk.LEFT, padx=(3, 10))

        self._energy_toggle = ttk.Checkbutton(range_frame, text="自定义",
                                               variable=self._var_custom_energy,
                                               command=self._toggle_energy_input)
        self._energy_toggle.pack(side=tk.LEFT, padx=5)

    def _build_output_settings(self, parent: ttk.Frame) -> None:
        """构建输出设置区域。"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="输出设置", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)

        # 保存目录
        dir_frame = ttk.Frame(parent)
        dir_frame.pack(fill=tk.X, pady=3)
        ttk.Label(dir_frame, text="目录:").pack(side=tk.LEFT)
        ttk.Entry(dir_frame, textvariable=self._var_save_dir, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="浏览", command=self._browse_save_dir).pack(side=tk.LEFT)

        # 文件名
        name_frame = ttk.Frame(parent)
        name_frame.pack(fill=tk.X, pady=3)
        ttk.Label(name_frame, text="文件名:").pack(side=tk.LEFT)
        ttk.Entry(name_frame, textvariable=self._var_output_name, width=30).pack(side=tk.LEFT, padx=5)

        # 图表标题
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=3)
        ttk.Label(title_frame, text="标题:").pack(side=tk.LEFT)
        self._var_title = tk.StringVar(value="")
        self._title_entry = ttk.Entry(title_frame, textvariable=self._var_title, width=40)
        self._title_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="（留空则使用文件名）", foreground="gray",
                  font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)

    def _build_castep_params(self, parent: ttk.Frame) -> None:
        """构建 CASTEP 参数面板（展宽宽度 + 网格点数）。"""
        self._castep_frame = ttk.LabelFrame(parent, text="CASTEP 参数", padding=8)
        self._castep_frame.pack(fill=tk.X, pady=5)

        # σ 展宽
        sigma_frame = ttk.Frame(self._castep_frame)
        sigma_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sigma_frame, text="展宽 σ:", width=8).pack(side=tk.LEFT)
        self._var_sigma = tk.DoubleVar(value=0.2)
        ttk.Scale(sigma_frame, from_=0.05, to=1.0, variable=self._var_sigma,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(sigma_frame, textvariable=tk.StringVar(value="0.2"), width=4).pack(side=tk.LEFT)
        # 同步 label
        self._sigma_label = ttk.Label(sigma_frame, text="0.20", width=4)
        self._sigma_label.pack(side=tk.LEFT)
        self._var_sigma.trace_add("write", lambda _n, _i, _m: self._sigma_label.configure(
            text=f"{self._var_sigma.get():.2f}"))

        # 能量网格点数
        npts_frame = ttk.Frame(self._castep_frame)
        npts_frame.pack(fill=tk.X, pady=2)
        ttk.Label(npts_frame, text="网格点:", width=8).pack(side=tk.LEFT)
        self._var_npoints = tk.IntVar(value=500)
        ttk.Spinbox(npts_frame, from_=100, to=2000, increment=100,
                    textvariable=self._var_npoints, width=6).pack(side=tk.LEFT, padx=5)

    def _build_action_buttons(self, parent: ttk.Frame) -> None:
        """构建操作按钮区域。"""
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="预览", command=self._preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="保存图片", command=self._plot_and_save).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="分析", command=self._show_analysis_options).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=3)

    # ================================================================
    # 文件管理
    # ================================================================
    def _browse_file(self) -> None:
        """浏览并添加 XCD 或 CASTEP 文件。"""
        paths = filedialog.askopenfilenames(
            title="选择 PDOS 文件",
            filetypes=[
                ("所有支持格式", "*.xcd;*.castep_bin"),
                ("XCD 文件", "*.xcd"),
                ("CASTEP 文件", "*.castep_bin"),
                ("所有文件", "*.*"),
            ],
        )
        for p in paths:
            self._add_multi_file_from_path(p)

    def _add_multi_file(self) -> None:
        """兼容旧方法名。"""
        self._browse_file()

    def _add_multi_file_from_path(self, path: str) -> None:
        """从路径添加文件到列表（自动检测 XCD / CASTEP 格式）。"""
        path = os.path.abspath(path)
        # 去重
        if any(f["path"] == path for f in self.loaded_files):
            print(f"[INFO] 文件已加载: {os.path.basename(path)}")
            return

        ext = os.path.splitext(path)[1].lower()
        file_type = "xcd"
        ml_available = False
        adapter = None  # m_l 分辨适配器

        try:
            if ext in (".castep_bin",):
                file_type = "castep"
                parser = CastepBinParser(path)
                parser.parse()
                ml_available = parser._has_ml_resolved

                # 若 .pdos_bin 可用，立即计算 m_l 分辨 PDOS
                if ml_available and parser._ml_calc is not None:
                    try:
                        from .pdos_calc import CastepPDOSAdapter
                    except ImportError:
                        from pdos_calc import CastepPDOSAdapter
                    calc = parser._ml_calc
                    calc.compute_pdos(e_min=-15, e_max=10, n_points=500, sigma=0.2)
                    adapter = CastepPDOSAdapter(calc, group_by="species_orbital")
                    adapter.parse()
                    # 用适配器替换 parser（提供完整 m_l 轨道列表）
                    parser = adapter
                    print(f"[INFO] m_l 分辨 PDOS 已计算，可用轨道: {adapter.available_orbitals}")
            else:
                parser = PDOSParser(path)
                parser.parse()
        except Exception as e:
            messagebox.showerror("解析错误", f"无法解析文件:\n{path}\n\n{str(e)}")
            return

        self._file_counter += 1
        type_tag = "[CASTEP]" if file_type == "castep" else "[XCD]"
        ml_tag = " [m_l]" if ml_available else ""
        label = f"[{self._file_counter}] {type_tag}{ml_tag} {parser.label}"
        entry = {
            "path": path, "parser": parser, "label": label,
            "has_spin": parser.has_spin, "orbitals": parser.available_orbitals,
            "type": file_type, "ml_available": ml_available,
            "_adapter": adapter,  # 缓存已计算的适配器
        }
        self.loaded_files.append(entry)

        self._insert_file_entry(len(self.loaded_files) - 1, entry)
        self._add_to_recent_files(path)
        self._save_recent_files()

        # 自动选中第一个文件
        if len(self.loaded_files) == 1:
            self._file_listbox.selection_set(0)
            self._active_file_idx = 0
            self._update_orbital_checkboxes()
            self._update_mode_buttons()
        # 加载新文件时清空自定义标题
        if hasattr(self, '_var_title'):
            self._var_title.set("")

    def _insert_file_entry(self, index: int, entry: Dict) -> None:
        """在 Listbox 中插入一个文件条目。"""
        display = f"{entry['label']}"
        if entry["has_spin"]:
            display += "  [α/β]"
        if entry.get("ml_available"):
            display += "  [m_l]"
        display += f"  ({', '.join(entry['orbitals'][:5])}{'...' if len(entry['orbitals']) > 5 else ''})"
        self._file_listbox.insert(index, display)

    def _remove_multi_file(self) -> None:
        """移除选中的文件。"""
        sel = self._file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._remove_file_entry(idx)

    def _remove_file_entry(self, index: int) -> None:
        """移除指定索引的文件条目。"""
        if 0 <= index < len(self.loaded_files):
            self._file_listbox.delete(index)
            removed = self.loaded_files.pop(index)
            print(f"[INFO] 已移除: {removed['label']}")
            self._reindex_file_list()
            if self.loaded_files:
                new_idx = min(index, len(self.loaded_files) - 1)
                self._file_listbox.selection_set(new_idx)
                self._active_file_idx = new_idx
                self._update_orbital_checkboxes()
                self._update_mode_buttons()
            else:
                self._clear_orbital_checkboxes()

    def _reindex_file_list(self) -> None:
        """重新索引文件列表的显示（在移除后调用）。"""
        for i in range(self._file_listbox.size()):
            entry = self.loaded_files[i]
            display = f"{entry['label']}"
            if entry["has_spin"]:
                display += "  [α/β]"
            display += f"  ({', '.join(entry['orbitals'])})"
            self._file_listbox.delete(i)
            self._file_listbox.insert(i, display)

    def _edit_file_label(self) -> None:
        """编辑选中文件的标签。"""
        sel = self._file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]

        dialog = tk.Toplevel(self.root)
        dialog.title("编辑文件标签")
        dialog.geometry("350x120")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="新标签:").pack(pady=(10, 5))
        var = tk.StringVar(value=self.loaded_files[idx]["label"])
        entry = ttk.Entry(dialog, textvariable=var, width=40)
        entry.pack(padx=10, pady=5)
        entry.selection_range(0, tk.END)
        entry.focus_set()

        def _on_ok() -> None:
            new_label = var.get().strip()
            if new_label:
                self.loaded_files[idx]["label"] = new_label
                self.loaded_files[idx]["parser"].label = new_label
                self._reindex_file_list()
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="确认", command=_on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        dialog.bind("<Return>", lambda e: _on_ok())

    def _on_file_list_select(self, event: tk.Event) -> None:
        """文件列表选中事件。"""
        sel = self._file_listbox.curselection()
        if not sel:
            return
        self._active_file_idx = sel[0]
        self._refresh_orbital_panel_for_active_file()
        self._update_mode_buttons()
        entry = self.loaded_files[self._active_file_idx]
        self._file_info_label.config(
            text=f"路径: {entry['path']}\n自旋: {'有' if entry['has_spin'] else '无'}  |  "
                 f"轨道: {', '.join(entry['orbitals'])}")
        # 切换文件时自动将标题重置为文件名
        if hasattr(self, '_var_title'):
            self._var_title.set("")

    def _refresh_orbital_panel_for_active_file(self) -> None:
        """为当前选中的文件刷新轨道面板。"""
        if not self.loaded_files:
            return
        entry = self.loaded_files[self._active_file_idx]
        self._populate_orbital_tree(entry)

    def _populate_orbital_tree(self, entry: Dict) -> None:
        """根据 entry 的 orbital_map 填充树形轨道选择器（防重复）。"""
        if not hasattr(self, '_orbital_tree'):
            return
        # 防重复：相同 entry 不重复填充
        cached_key = f"{id(entry)}"
        if getattr(self, '_last_populated_key', '') == cached_key:
            return
        self._last_populated_key = cached_key

        parser = entry["parser"]
        # 尝试获取 orbital_map
        orbital_map = None
        species_list = None
        if hasattr(parser, 'orbital_map') and parser.orbital_map:
            orbital_map = parser.orbital_map
            species_list = parser.species if hasattr(parser, 'species') else []
        elif hasattr(parser, 'calc') and hasattr(parser.calc, 'orbital_map'):
            orbital_map = parser.calc.orbital_map
            species_list = parser.calc.species
        if orbital_map:
            self._orbital_tree.populate(orbital_map, species_list)

    def _clear_orbital_checkboxes(self) -> None:
        """清除所有轨道复选框（树模式下为兼容占位）。"""
        self._orbital_vars.clear()

    def _update_orbital_checkboxes_for_parser(self, parser) -> None:
        """根据解析器更新轨道复选框（兼容旧接口）。"""
        pass  # 树模式由 _populate_orbital_tree 处理

    def _update_orbital_checkboxes(self) -> None:
        """更新轨道复选框（兼容旧调用）。"""
        if self.loaded_files:
            entry = self.loaded_files[self._active_file_idx]
            self._populate_orbital_tree(entry)
        else:
            self._clear_orbital_checkboxes()

    def _tree_select_all(self) -> None:
        """全选所有轨道。"""
        if hasattr(self, '_orbital_tree'):
            self._orbital_tree.select_all()

    def _tree_deselect_all(self) -> None:
        """取消全选。"""
        if hasattr(self, '_orbital_tree'):
            self._orbital_tree.deselect_all()

    def _open_crystal_viewer(self) -> None:
        """打开晶体结构查看器。"""
        if not self.loaded_files:
            messagebox.showinfo("提示", "请先加载包含晶体结构的文件。")
            return
        entry = self.loaded_files[self._active_file_idx]
        crystal = None
        parser = entry["parser"]
        if hasattr(parser, 'calc') and parser.calc.crystal:
            crystal = parser.calc.crystal
        elif hasattr(parser, 'crystal') and parser.crystal:
            crystal = parser.crystal

        if crystal is None:
            messagebox.showinfo("提示", "当前文件不包含晶体结构数据。\n请加载 .castep_bin 文件。")
            return

        from .crystal_viewer import CrystalViewer
        viewer = CrystalViewer(crystal)
        viewer.show()

    # ================================================================
    # 叠加系列管理
    # ================================================================
    def _add_overlay_series(self) -> None:
        """添加叠加系列。"""
        if not self.loaded_files:
            messagebox.showinfo("提示", "请先加载文件。")
            return

        sel = self._file_listbox.curselection()
        file_idx = sel[0] if sel else 0
        parser = self.loaded_files[file_idx]["parser"]

        # 简单对话框：选择轨道和自旋
        dialog = tk.Toplevel(self.root)
        dialog.title("添加叠加系列")
        dialog.geometry("320x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="轨道:").pack(pady=(10, 3))
        orb_var = tk.StringVar(value=parser.non_sum_orbitals[0] if parser.non_sum_orbitals else "d")
        orb_combo = ttk.Combobox(dialog, textvariable=orb_var,
                                  values=parser.non_sum_orbitals, state="readonly")
        orb_combo.pack()

        ttk.Label(dialog, text="自旋聚合:").pack(pady=(10, 3))
        spin_var = tk.StringVar(value="sum")
        spin_choices = ["alpha", "beta", "sum", "both"] if parser.has_spin else ["none"]
        spin_combo = ttk.Combobox(dialog, textvariable=spin_var,
                                   values=spin_choices, state="readonly")
        spin_combo.pack()

        def _on_add() -> None:
            orb = orb_var.get()
            spin = spin_var.get() if parser.has_spin else "sum"
            greek = SPIN_KEY_TO_GREEK.get(spin, spin)
            label = f"{self.loaded_files[file_idx]['label']} {orb} ({greek})"
            spec = {"parser": parser, "file_label": self.loaded_files[file_idx]["label"],
                    "orbital": orb, "spin_agg": spin, "label": label}
            self._overlay_series.append(spec)
            self._refresh_overlay_listbox()
            dialog.destroy()

        ttk.Button(dialog, text="添加", command=_on_add).pack(pady=15)
        dialog.bind("<Return>", lambda e: _on_add())

    def _remove_overlay_series(self) -> None:
        """移除选中的叠加系列。"""
        sel = self._overlay_listbox.curselection()
        if sel:
            self._overlay_series.pop(sel[0])
            self._refresh_overlay_listbox()

    def _clear_overlay_series(self) -> None:
        """清空所有叠加系列。"""
        self._overlay_series.clear()
        self._refresh_overlay_listbox()

    def _move_overlay_up(self) -> None:
        """将选中系列上移一位。"""
        sel = self._overlay_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        self._overlay_series[idx], self._overlay_series[idx - 1] = \
            self._overlay_series[idx - 1], self._overlay_series[idx]
        self._refresh_overlay_listbox()
        self._overlay_listbox.selection_set(idx - 1)

    def _move_overlay_down(self) -> None:
        """将选中系列下移一位。"""
        sel = self._overlay_listbox.curselection()
        if not sel or sel[0] >= len(self._overlay_series) - 1:
            return
        idx = sel[0]
        self._overlay_series[idx], self._overlay_series[idx + 1] = \
            self._overlay_series[idx + 1], self._overlay_series[idx]
        self._refresh_overlay_listbox()
        self._overlay_listbox.selection_set(idx + 1)

    def _refresh_overlay_listbox(self) -> None:
        """刷新叠加系列列表显示。"""
        self._overlay_listbox.delete(0, tk.END)
        for s in self._overlay_series:
            self._overlay_listbox.insert(tk.END, s["label"])

    # ================================================================
    # 最近文件
    # ================================================================
    def _load_recent_files(self) -> None:
        """加载最近文件记录。"""
        try:
            if RECENT_FILES_PATH.exists():
                data = json.loads(RECENT_FILES_PATH.read_text(encoding="utf-8"))
                self._recent_files = data if isinstance(data, list) else []
        except Exception:
            self._recent_files = []

    def _save_recent_files(self) -> None:
        """保存最近文件记录。"""
        try:
            RECENT_FILES_PATH.write_text(
                json.dumps(self._recent_files[:MAX_RECENT_FILES], ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception:
            pass

    def _add_to_recent_files(self, path: str) -> None:
        """将文件路径添加到最近文件列表头部。"""
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        if len(self._recent_files) > MAX_RECENT_FILES:
            self._recent_files = self._recent_files[:MAX_RECENT_FILES]

    def _show_recent_menu(self) -> None:
        """显示最近文件子菜单。"""
        menu = tk.Menu(self.root, tearoff=0)
        if not self._recent_files:
            menu.add_command(label="(无最近文件)", state=tk.DISABLED)
        else:
            for path in self._recent_files:
                menu.add_command(label=os.path.basename(path),
                                 command=lambda p=path: self._add_multi_file_from_path(p))
            menu.add_separator()
            menu.add_command(label="清除最近文件", command=self._clear_recent_files)
        menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())

    def _show_multi_recent_menu(self) -> None:
        """在多文件面板中显示最近文件弹出菜单。"""
        menu = tk.Menu(self.root, tearoff=0)
        if not self._recent_files:
            menu.add_command(label="(无最近文件)", state=tk.DISABLED)
        else:
            for path in self._recent_files:
                menu.add_command(
                    label=os.path.basename(path),
                    command=lambda p=path: self._add_multi_file_from_path(p))
            menu.add_separator()
            menu.add_command(label="清除最近文件", command=self._clear_recent_files)
        # 弹出在鼠标位置附近
        try:
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
        except Exception:
            x, y = 100, 100
        menu.tk_popup(x, y)

    def _on_file_list_right_click(self, event: tk.Event) -> None:
        """文件列表右键菜单：支持粘贴路径。"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="粘贴路径并添加", command=self._paste_path_add)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _paste_path_add(self) -> None:
        """从剪贴板粘贴路径并添加文件。"""
        try:
            clip_text = self.root.clipboard_get()
        except Exception:
            return
        for line in clip_text.strip().split("\n"):
            path = line.strip().strip('"').strip("'")
            if path and os.path.isfile(path):
                self._add_multi_file_from_path(path)

    def _clear_recent_files(self) -> None:
        """清除最近文件记录。"""
        self._recent_files.clear()
        self._save_recent_files()

    # ================================================================
    # 模式切换
    # ================================================================
    def _on_plot_mode_changed(self, *_args) -> None:
        """绑图模式变更回调。"""
        mode = self._var_plot_mode.get()
        if mode == "overlay":
            if len(self.loaded_files) < 1:
                messagebox.showinfo("提示", "多文件叠加模式需要至少加载一个文件。")
                self._var_plot_mode.set("orbitals")
                return
        self._update_panel_visibility()

    def _update_panel_visibility(self) -> None:
        """根据当前绑图模式更新面板可见性。"""
        mode = self._var_plot_mode.get()
        if mode == "overlay":
            self._spin_frame.pack_forget()
        else:
            self._spin_frame.pack(fill=tk.X, pady=5, after=self._mode_btns.get("total", None))

    def _update_mode_buttons(self) -> None:
        """根据当前文件是否有自旋极化来更新模式按钮状态。"""
        if self.loaded_files:
            has_spin = self.loaded_files[self._active_file_idx]["has_spin"]
            state = tk.NORMAL if has_spin else tk.DISABLED
            self._mode_btns.get("spin", None) and self._mode_btns["spin"].configure(state=state)
            if not has_spin and self._var_plot_mode.get() == "spin":
                self._var_plot_mode.set("orbitals")

    def _on_spin_mode_changed(self, event: tk.Event = None) -> None:
        """自旋模式变更回调。"""
        pass  # 由 _do_plot 读取变量值

    # ================================================================
    # 能量范围 / 输出路径
    # ================================================================
    def _toggle_energy_input(self) -> None:
        """切换自定义能量范围输入框的启用状态。"""
        state = "normal" if self._var_custom_energy.get() else "disabled"
        if hasattr(self, '_entry_emin'):
            self._entry_emin.configure(state=state)
        if hasattr(self, '_entry_emax'):
            self._entry_emax.configure(state=state)

    def _browse_save_dir(self) -> None:
        """浏览保存目录。"""
        d = filedialog.askdirectory(title="选择图片保存目录")
        if d:
            self._var_save_dir.set(d)

    def _get_save_path(self) -> str:
        """获取完整的保存路径。"""
        d = self._var_save_dir.get()
        name = self._var_output_name.get()
        if not name.endswith(".png"):
            name += ".png"
        return os.path.join(d, name)

    def _get_selected_orbitals(self) -> List[str]:
        """获取用户选中的轨道列表（树模式返回扁平原子-轨道标签）。"""
        if hasattr(self, '_orbital_tree') and self._orbital_tree._orbital_map:
            return self._orbital_tree.get_selected_flat()
        # 回退到旧 checkbox 模式
        return [orb for orb, var in self._orbital_vars.items() if var.get()]

    def _get_energy_range(self) -> Optional[Tuple[float, float]]:
        """获取用户自定义的能量范围。"""
        if not self._var_custom_energy.get():
            return None
        try:
            emin = float(self._var_emin.get()) if self._var_emin.get() else None
            emax = float(self._var_emax.get()) if self._var_emax.get() else None
            if emin is not None and emax is not None:
                return (emin, emax)
        except ValueError:
            pass
        return None

    # ================================================================
    # 绘图调度
    # ================================================================
    def _read_custom_title(self) -> str:
        """读取用户在标题输入框中输入的自定义标题，为空返回 ''。"""
        try:
            if hasattr(self, '_title_entry'):
                val = self._title_entry.get()
                if val:
                    return val.strip()
        except Exception:
            pass
        try:
            if hasattr(self, '_var_title'):
                val = self._var_title.get()
                if val:
                    return val.strip()
        except Exception:
            pass
        return ""

    def _do_plot(self) -> Optional["plt.Figure"]:
        """
        根据当前 GUI 状态执行绘图，返回 matplotlib Figure。

        统一的绘图调度逻辑（XCD / CASTEP 均在此处理）。
        """
        mode = self._var_plot_mode.get()

        # --- 多文件叠加模式 ---
        if mode == "overlay":
            if not self._overlay_series:
                messagebox.showinfo("提示", "请先在叠加系列构建区域添加系列。")
                return None
            first_parser = self._overlay_series[0]["parser"]
            plotter = PDOSPlotter(first_parser)
            energy_range = self._get_energy_range()
            # 标题：优先用户自定义
            _custom = self._read_custom_title()
            title = _custom if _custom else "PDOS 多文件叠加"
            fig, _ = plotter.plot_multi_overlay(
                series_specs=self._overlay_series,
                energy_range=energy_range,
                title=title,
            )
            return fig

        if not self.loaded_files:
            messagebox.showinfo("提示", "请先加载文件。")
            return None

        entry = self.loaded_files[self._active_file_idx]
        parser = entry["parser"]
        plotter = PDOSPlotter(parser)
        orbitals = self._get_selected_orbitals()
        energy_range = self._get_energy_range()

        # --- CASTEP 模式：检查是否启用 m_l 分辨 ---
        if entry.get("type") == "castep" and entry.get("_adapter"):
            adapter = entry["_adapter"]
            sigma = self._var_sigma.get()
            n_points = self._var_npoints.get()
            emin, emax = (-15.0, 10.0)
            if energy_range:
                emin, emax = energy_range

            # 检查树选择器是否有原子级选择
            tree_selected = self._get_selected_orbitals() if hasattr(self, '_orbital_tree') else []
            if tree_selected:
                # 原子级选择：使用 atom_orbital 分组
                adapter.calc.compute_pdos(e_min=emin, e_max=emax, n_points=n_points, sigma=sigma)
                atom_adapter = type(adapter)(adapter.calc, group_by="atom_orbital")
                atom_adapter.parse()

                # 构建自定义数据集：仅包含选中的原子-轨道组合
                custom_data = {}
                for lbl in tree_selected:
                    # lbl 格式: "Ni_1-d_xy"
                    orb_key = lbl + "_up"
                    orb_key_down = lbl + "_down"
                    agg = atom_adapter._aggregated
                    if orb_key in agg:
                        custom_data[f"{lbl} (α)"] = (atom_adapter.calc.energy_grid.copy(), agg[orb_key].copy())
                    if orb_key_down in agg:
                        custom_data[f"{lbl} (β)"] = (atom_adapter.calc.energy_grid.copy(), agg[orb_key_down].copy())

                # 创建临时 adapter 包装自定义数据
                class _CustomAdapter:
                    # up/down → alpha/beta 映射（CASTEP 内部使用 up/down 后缀）
                    _SPIN_UI_MAP = {"up": "alpha", "down": "beta",
                                    "α": "alpha", "β": "beta"}

                    def __init__(self, edata, egrid, *_args):
                        self._data = edata
                        self.energy_grid = egrid
                        self.has_spin = True
                        self.filename = entry["label"]
                        self.available_orbitals = list(set(k.split(" ")[0] for k in edata.keys()))
                        self.available_spins = ["alpha", "beta"]

                    def get_data(self, orbitals=None, spin=None):
                        result = {}
                        for k, v in self._data.items():
                            orb_part = k.split(" (")[0]
                            spin_part = k.split("(")[1].rstrip(")")
                            # 将内部 spin 标签统一映射到 alpha/beta
                            spin_part = self._SPIN_UI_MAP.get(spin_part, spin_part)
                            if orbitals and orb_part not in orbitals:
                                continue
                            if spin and spin != "both" and spin != spin_part:
                                continue
                            result[k] = v
                        return result

                    def get_summed_data(self, orbitals=None):
                        return {}

                    def get_total_dos(self):
                        return None

                    def get_energy_range(self):
                        return (float(self.energy_grid[0]), float(self.energy_grid[-1]))

                    @property
                    def non_sum_orbitals(self):
                        return self.available_orbitals

                plotter = PDOSPlotter(_CustomAdapter(custom_data, adapter.calc.energy_grid, True, tree_selected))
                orbitals = tree_selected
            else:
                # 无树选择：使用默认分组
                adapter.calc.compute_pdos(e_min=emin, e_max=emax, n_points=n_points, sigma=sigma)
                adapter.parse()
                plotter = PDOSPlotter(adapter)
                if not orbitals:
                    orbitals = [o for o in adapter.available_orbitals if o != "sum"]
        else:
            if not orbitals and hasattr(parser, 'non_sum_orbitals'):
                orbitals = parser.non_sum_orbitals
            elif not orbitals:
                orbitals = [o for o in parser.available_orbitals if o != "sum"]

        spin_mode = self._var_spin_mode.get()

        # 图表标题：优先用户自定义，否则使用文件标签
        _custom = self._read_custom_title()
        _base = entry["label"]
        title = _custom if _custom else _base

        if mode == "total":
            fig, _ = plotter.plot_total(energy_range=energy_range,
                                         title=f"{title} TDOS")
        elif mode == "spin" and parser.has_spin:
            fig, _ = plotter.plot_spin_polarized(
                orbitals=orbitals, energy_range=energy_range,
                title=title,
            )
        else:  # "orbitals" (默认)
            fig, _ = plotter.plot_orbitals(
                orbitals=orbitals, energy_range=energy_range,
                title=title, spin_mode=spin_mode,
            )

        return fig

    def _preview(self) -> None:
        """预览绑图。"""
        import matplotlib.pyplot as plt
        fig = self._do_plot()
        if fig is None:
            return
        self.current_fig = fig
        self._attach_interactive_preview(fig)
        plt.show()

    def _plot_and_save(self) -> None:
        """绑图并保存到文件。"""
        fig = self._do_plot()
        if fig is None:
            return
        self.current_fig = fig

        save_path = self._get_save_path()
        plotter = PDOSPlotter(self.loaded_files[self._active_file_idx]["parser"] if self.loaded_files else None)
        plotter.save_figure(fig, save_path)

        # 同时显示交互预览
        self._attach_interactive_preview(fig)
        import matplotlib.pyplot as plt
        plt.show()

    def _attach_interactive_preview(self, fig: "plt.Figure") -> None:
        """为 matplotlib Figure 附加交互功能。"""
        try:
            from .interactive_preview import InteractivePreview
        except ImportError:
            from interactive_preview import InteractivePreview

        if not self.loaded_files:
            return

        parser = self.loaded_files[self._active_file_idx]["parser"]
        ax = fig.axes[0] if fig.axes else None
        if ax is None:
            return

        orbitals = self._get_selected_orbitals()
        if not orbitals:
            orbitals = parser.non_sum_orbitals

        overlay_specs = self._overlay_series if self._var_plot_mode.get() == "overlay" else None

        InteractivePreview(
            fig=fig, ax=ax, parser=parser,
            selected_orbitals=orbitals,
            has_spin=parser.has_spin,
            plot_mode=self._var_plot_mode.get(),
            spin_mode=self._var_spin_mode.get(),
            overlay_specs=overlay_specs,
        )

    # ================================================================
    # 分析功能
    # ================================================================
    def _show_analysis_options(self) -> None:
        """显示分析选项对话框。"""
        if not self.loaded_files:
            messagebox.showinfo("提示", "请先加载文件。")
            return

        def _callback(options: Dict[str, bool]) -> None:
            self._run_analysis(options)

        AnalysisDialog(self.root, _callback)

    def _run_analysis(self, options: Dict[str, bool]) -> None:
        """执行分析并生成报告。"""
        if not self.loaded_files:
            return

        try:
            from .pdos_analyzer import PDOSAnalyzer
        except ImportError:
            from pdos_analyzer import PDOSAnalyzer

        parser = self.loaded_files[self._active_file_idx]["parser"]
        analyzer = PDOSAnalyzer(parser)

        orbitals = self._get_selected_orbitals()
        if not orbitals:
            orbitals = parser.non_sum_orbitals
        target_orb = orbitals[0] if orbitals else "d"

        spin = self._var_spin_mode.get() if parser.has_spin else None

        energy_range = self._get_energy_range()
        emin = energy_range[0] if energy_range else None
        emax = energy_range[1] if energy_range else None

        # 打印分析报告到控制台
        print("\n" + "=" * 60)
        print(f"  分析目标: {parser.filename}")
        print(f"  轨道: {target_orb}" + (f"  自旋: {spin}" if spin else ""))
        print("=" * 60)

        analyzer.full_report(
            orbital=target_orb,
            spin=spin,
            emin=emin, emax=emax,
            options=options,
        )

    # ================================================================
    # 运行
    # ================================================================
    def run(self) -> None:
        """启动 GUI 主循环。"""
        self.root.mainloop()


# ============================================================
# 直接运行入口
# ============================================================
if __name__ == "__main__":
    import sys
    import os
    # 将当前目录从 sys.path 中移除（避免覆盖包导入），
    # 并将父目录添加到 sys.path，使 pdos_plotter 可作为包导入
    _curdir = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_curdir)
    # 移除当前目录，防止与包名冲突
    sys.path = [p for p in sys.path if os.path.abspath(p) != _curdir]
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    # 清除可能已被缓存的无效 pdos_plotter 条目
    for key in list(sys.modules.keys()):
        if key == 'pdos_plotter' or key.startswith('pdos_plotter.'):
            del sys.modules[key]

    from pdos_plotter.cli import main
    main()
