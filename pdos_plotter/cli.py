#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI 命令行模式入口
==================

提供 argparse 参数解析和命令行模式绑图调度。

支持两种输入模式:
  - xcd (默认):  从 Materials Studio .xcd 文件读取
  - castep:      从 CASTEP .castep_bin + .pdos_bin 计算 PDOS

改进（2026/06/11 重构）:
  - 新增 --mode castep 参数和 CASTEP 二进制模式支持
  - 使用统一的自旋/颜色常量

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .constants import (
        DEFAULT_PIC_DIR,
        DEFAULT_DPI,
        DEFAULT_FIGSIZE,
        SPIN_SHORT_TO_KEY,
        SPIN_KEY_TO_GREEK,
        DEFAULT_OUTPUT_FILENAME,
        DEFAULT_OVERLAY_FILENAME,
    )
    from .pdos_parser import PDOSParser
    from .pdos_plotter import PDOSPlotter
except ImportError:
    from constants import (
        DEFAULT_PIC_DIR,
        DEFAULT_DPI,
        DEFAULT_FIGSIZE,
        SPIN_SHORT_TO_KEY,
        SPIN_KEY_TO_GREEK,
        DEFAULT_OUTPUT_FILENAME,
        DEFAULT_OVERLAY_FILENAME,
    )
    from pdos_parser import PDOSParser
    from pdos_plotter import PDOSPlotter


# ============================================================
# argparse 构建
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    CLI 模式下支持的完整参数列表。

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
    python -m pdos_plotter
    python -m pdos_plotter -f "path/to/PDOS.xcd"

  ── CLI XCD 总态密度 ──
    python -m pdos_plotter -f "PDOS.xcd" --total --no-gui -o total_dos.png

  ── CLI XCD 自旋极化 ──
    python -m pdos_plotter -f "PDOS.xcd" --spin --orbitals s,p,d --no-gui

  ── CLI CASTEP d轨道 ──
    python -m pdos_plotter --mode castep --castep-bin "Ni_DOS.castep_bin" \\
        --pdos-bin "Ni_DOS.pdos_bin" --orbitals d_xy,dyz,d_z2 --no-gui
        """,
    )

    # --- 输入模式 ---
    parser.add_argument(
        "--mode", type=str, choices=["xcd", "castep"], default="xcd",
        help="输入模式: xcd (MS .xcd 文件, 默认) 或 castep (CASTEP 二进制)",
    )

    # --- 文件输入 ---
    parser.add_argument(
        "-f", "--file",
        type=str,
        action="append",
        default=None,
        help="PDOS.xcd 文件的完整路径。可多次指定以加载多个文件（多文件叠加模式）。"
             "XCD 模式下必填。",
    )

    # --- CASTEP 专用参数 ---
    castep_group = parser.add_argument_group("CASTEP 二进制输入 (--mode castep)")
    castep_group.add_argument(
        "--castep-bin", type=str, default=None,
        help="*_DOS.castep_bin 文件的完整路径（必需）",
    )
    castep_group.add_argument(
        "--pdos-bin", type=str, default=None,
        help="*_DOS.pdos_bin 文件的完整路径（可选，提供后启用完整 m_l 分辨）",
    )
    castep_group.add_argument(
        "--group-by", type=str, default="species_orbital",
        choices=["total", "species", "species_l", "species_orbital",
                 "atom_l", "atom_orbital"],
        help="PDOS 聚合级别（仅 castep 模式，默认 species_orbital）",
    )
    castep_group.add_argument(
        "--sigma", type=float, default=0.2,
        help="Gaussian 展宽宽度 eV（仅 castep 模式，默认 0.2）",
    )
    castep_group.add_argument(
        "--n-points", type=int, default=500,
        help="能量网格点数（仅 castep 模式，默认 500）",
    )

    # --- 输出设置 ---
    parser.add_argument(
        "-o", "--output",
        type=str, default=None,
        help="输出图片的保存路径。默认保存到 ./pic/<文件名>_pdos.png。",
    )

    # --- 运行模式 ---
    parser.add_argument(
        "--no-gui",
        action="store_true", default=False,
        help="禁用 GUI，直接在命令行绑图并保存后退出。",
    )

    # --- 绑图模式选择 ---
    parser.add_argument(
        "--spin",
        action="store_true", default=False,
        help="使用自旋极化模式绘图（α↑/β↓）。",
    )
    parser.add_argument(
        "--total",
        action="store_true", default=False,
        help="绘制总态密度 (TDOS)。",
    )
    parser.add_argument(
        "--orbitals",
        type=str, default=None,
        help="要显示的轨道列表，逗号分隔（如 s,p,d 或 d_xy,dyz,d_z2）。"
             "仅在 --no-gui 模式下生效。",
    )
    parser.add_argument(
        "--multi-overlay",
        type=str, default=None,
        help="多文件叠加绘图规格。格式: '文件索引:轨道列表:自旋聚合[:标签];...'"
             "（如 '0:s,p:s;1:s:s'）",
    )

    # --- 分析模式 ---
    parser.add_argument(
        "--analyze",
        action="store_true", default=False,
        help="执行态密度分析（d带中心、峰位置等），结果打印到命令行。",
    )
    parser.add_argument(
        "--analysis-orbital",
        type=str, default="d",
        help="分析的目标轨道（默认 d）。",
    )
    parser.add_argument(
        "--analysis-spin",
        type=str, default=None, choices=["alpha", "beta"],
        help="分析的自旋方向（alpha/beta），默认自动选择。",
    )

    # --- 能量范围 ---
    parser.add_argument(
        "--emin", type=float, default=None,
        help="能量范围下限 (eV)。",
    )
    parser.add_argument(
        "--emax", type=float, default=None,
        help="能量范围上限 (eV)。",
    )

    # --- 图表样式 ---
    parser.add_argument(
        "--title", type=str, default=None,
        help="图表标题。默认使用文件名。",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"输出图片的 DPI 分辨率（默认 {DEFAULT_DPI}）。",
    )

    # --- 显示选项 ---
    parser.add_argument(
        "--show",
        action="store_true", default=False,
        help="保存后弹出窗口显示图片（仅 --no-gui 模式）。",
    )

    return parser


# ============================================================
# CLI 模式运行函数
# ============================================================
def run_cli(args: argparse.Namespace) -> None:
    """
    命令行模式入口：解析文件 → 绘图 → 保存 → 退出。

    参数
    ----
    args : argparse.Namespace
        命令行解析后的参数命名空间。
    """
    # ---- 确保输出目录 ----
    DEFAULT_PIC_DIR.mkdir(parents=True, exist_ok=True)

    # ---- CASTEP 模式 ----
    if args.mode == "castep":
        _run_castep_mode(args)
        return

    # ---- XCD 模式 ----
    _run_xcd_mode(args)


def _run_castep_mode(args: argparse.Namespace) -> None:
    """CASTEP 二进制输入模式（支持仅有 .castep_bin 或带 .pdos_bin 的完整 m_l 分辨）。"""
    # 验证参数
    if not args.castep_bin:
        print("[ERROR] CASTEP 模式需要指定 --castep-bin。")
        sys.exit(1)
    if not os.path.isfile(args.castep_bin):
        print(f"[ERROR] .castep_bin 文件不存在: {args.castep_bin}")
        sys.exit(1)

    has_pdos_bin = args.pdos_bin and os.path.isfile(args.pdos_bin)
    if not has_pdos_bin and args.pdos_bin:
        print(f"[WARN] .pdos_bin 不存在: {args.pdos_bin}，将仅使用 .castep_bin 内嵌 DOS。")

    # 延迟导入
    try:
        from .castep_bin_parser import CastepBinParser
    except ImportError:
        from castep_bin_parser import CastepBinParser

    if has_pdos_bin:
        # 完整 m_l 分辨（需要 .pdos_bin）
        try:
            from .pdos_calc import CastepPDOSCalculator, CastepPDOSAdapter
        except ImportError:
            from pdos_calc import CastepPDOSCalculator, CastepPDOSAdapter

        calc = CastepPDOSCalculator()
        calc.load_from_files(args.castep_bin, args.pdos_bin)
        emin = args.emin if args.emin is not None else -15.0
        emax = args.emax if args.emax is not None else 10.0
        calc.compute_pdos(e_min=emin, e_max=emax, n_points=args.n_points, sigma=args.sigma)
        adapter = CastepPDOSAdapter(calc, group_by=args.group_by)
        adapter.parse()
        data_source = adapter
    else:
        # 仅 .castep_bin（元素分辨，非 m_l）
        parser = CastepBinParser(args.castep_bin)
        parser.parse()
        data_source = parser

    # 输出路径
    if args.output:
        save_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.castep_bin))[0]
        base_name = base_name.replace("_DOS", "")
        save_path = str(DEFAULT_PIC_DIR / f"{base_name}_pdos.png")

    # 解析轨道参数
    orbitals_list = None
    if args.orbitals:
        orbitals_list = [o.strip().lower() for o in args.orbitals.split(",")]
        print(f"[INFO] 指定轨道: {orbitals_list}")

    # 默认轨道
    default_orb = data_source.non_sum_orbitals if hasattr(data_source, 'non_sum_orbitals') else [o for o in data_source.available_orbitals if o != "sum"]

    # 绘图
    plotter = PDOSPlotter(data_source)
    title = args.title or "CASTEP PDOS"

    if args.total:
        fig, _ = plotter.plot_total(energy_range=_get_cli_energy_range(args, data_source), title=title)
    elif args.spin and data_source.has_spin:
        fig, _ = plotter.plot_spin_polarized(
            orbitals=orbitals_list or default_orb,
            energy_range=_get_cli_energy_range(args, data_source), title=title)
    else:
        if not orbitals_list:
            orbitals_list = default_orb
        fig, _ = plotter.plot_orbitals(
            orbitals=orbitals_list,
            energy_range=_get_cli_energy_range(args, data_source),
            title=title, spin_mode="both")

    # 保存
    plotter.save_figure(fig, save_path, dpi=args.dpi)
    print(f"[INFO] 完成! 图片已保存至: {save_path}")

    if args.show:
        plt.show()


def _run_xcd_mode(args: argparse.Namespace) -> None:
    """XCD XML 文件输入模式。"""
    if not args.file:
        print("[ERROR] XCD 模式需要指定文件路径: -f <文件路径>")
        sys.exit(1)

    file_paths: List[str] = args.file
    for fp in file_paths:
        if not os.path.isfile(fp):
            print(f"[ERROR] 文件不存在: {fp}")
            sys.exit(1)

    # ---- 多文件叠加模式 ----
    if args.multi_overlay:
        _run_multi_overlay(args, file_paths)
        return

    # ---- 单文件模式 ----
    file_path = file_paths[0]
    pdos_parser = PDOSParser(file_path)
    pdos_parser.parse()
    plotter = PDOSPlotter(pdos_parser)

    energy_range = _get_cli_energy_range(args, pdos_parser)

    # 输出路径
    if args.output:
        save_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = str(DEFAULT_PIC_DIR / f"{base_name}_pdos.png")

    # 轨道参数
    orbitals_list = None
    if args.orbitals:
        orbitals_list = [o.strip().lower() for o in args.orbitals.split(",")]
        print(f"[INFO] 指定轨道: {orbitals_list}")

    # 根据参数选择绘图模式
    title = args.title

    if args.total:
        print("[INFO] 绑图模式: 总态密度 (TDOS)")
        fig, _ = plotter.plot_total(energy_range=energy_range, title=title)
    elif args.spin and pdos_parser.has_spin:
        print("[INFO] 绑图模式: 自旋极化 (α↑/β↓)")
        fig, _ = plotter.plot_spin_polarized(
            orbitals=orbitals_list, energy_range=energy_range, title=title)
    else:
        print("[INFO] 绑图模式: 轨道分别显示")
        fig, _ = plotter.plot_orbitals(
            orbitals=orbitals_list, energy_range=energy_range, title=title)

    # 保存图片
    plotter.save_figure(fig, save_path, dpi=args.dpi)
    print(f"[INFO] 完成! 图片已保存至: {save_path}")

    # ---- 分析模式（可选） ----
    if args.analyze:
        print()
        try:
            from .pdos_analyzer import PDOSAnalyzer
        except ImportError:
            from pdos_analyzer import PDOSAnalyzer
        analyzer = PDOSAnalyzer(pdos_parser)
        analyzer.full_report(
            orbital=args.analysis_orbital,
            spin=args.analysis_spin,
            emin=args.emin, emax=args.emax,
        )

    if args.show:
        print("[INFO] 正在显示图片...")
        plt.show()


def _run_multi_overlay(args: argparse.Namespace, file_paths: List[str]) -> None:
    """多文件叠加模式。"""
    # 解析所有文件
    parsers: List[PDOSParser] = []
    for fp in file_paths:
        p = PDOSParser(fp)
        p.parse()
        parsers.append(p)

    # 解析叠加规格
    overlay_specs = _parse_overlay_arg(args.multi_overlay, parsers)
    if not overlay_specs:
        print("[ERROR] 未能解析任何有效的叠加系列规格。")
        sys.exit(1)

    # 确定能量范围
    energy_range = _get_cli_energy_range(args, parsers[0])

    # 输出路径
    if args.output:
        save_path = args.output
    else:
        save_path = str(DEFAULT_PIC_DIR / DEFAULT_OVERLAY_FILENAME)

    # 绘制
    print(f"[INFO] 绘图模式: 多文件叠加 ({len(overlay_specs)} 个系列)")
    plotter = PDOSPlotter(parsers[0])
    title = args.title or "PDOS 多文件叠加"
    fig, _ = plotter.plot_multi_overlay(
        series_specs=overlay_specs,
        energy_range=energy_range,
        title=title,
    )

    plotter.save_figure(fig, save_path, dpi=args.dpi)
    print(f"[INFO] 完成! 图片已保存至: {save_path}")

    if args.show:
        plt.show()


def _parse_overlay_arg(arg_str: str, parsers: List[PDOSParser]) -> List[Dict]:
    """
    解析 --multi-overlay 参数字符串为叠加系列规格列表。

    格式: "文件索引:轨道列表:自旋聚合[:标签];..."

    参数
    ----
    arg_str : str
        --multi-overlay 的参数值。
    parsers : list of PDOSParser
        按 -f 顺序排列的解析器列表。

    返回
    ----
    specs : list of dict
        可直接传入 plot_multi_overlay() 的系列规格列表。
    """
    specs: List[Dict] = []

    for part in arg_str.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split(":")]
        if len(fields) < 3:
            print(f"[WARN] 无法解析叠加规格: '{part}'，需要至少 3 个字段")
            continue

        try:
            file_idx = int(fields[0])
        except ValueError:
            print(f"[WARN] 文件索引无效: '{fields[0]}'")
            continue

        if file_idx < 0 or file_idx >= len(parsers):
            print(f"[WARN] 文件索引 {file_idx} 超出范围（共 {len(parsers)} 个）")
            continue

        parser = parsers[file_idx]
        orbitals_str = fields[1]
        spin_agg_raw = fields[2].lower()
        spin_agg = SPIN_SHORT_TO_KEY.get(spin_agg_raw)
        if spin_agg is None:
            print(f"[WARN] 未知的自旋聚合方式 '{spin_agg_raw}'，支持: a/alpha, b/beta, s/sum, both")
            continue

        custom_label = fields[3] if len(fields) > 3 else None
        orbitals = [o.strip().lower() for o in orbitals_str.split(",") if o.strip()]

        for orb in orbitals:
            label = custom_label if custom_label else (
                f"{parser.label} {orb} ({SPIN_KEY_TO_GREEK.get(spin_agg, spin_agg)})")
            specs.append({
                "parser": parser,
                "file_label": parser.label,
                "orbital": orb,
                "spin_agg": spin_agg,
                "label": label,
            })
            print(f"[INFO]   叠加系列: [{file_idx}] {parser.label} / {orb} / {spin_agg}")

    return specs


def _get_cli_energy_range(args: argparse.Namespace, data_source) -> Optional[Tuple[float, float]]:
    """
    从 CLI 参数和数据源确定能量范围。

    参数
    ----
    args : argparse.Namespace
        CLI 参数。
    data_source : PDOSParser 或 CastepPDOSAdapter
        数据源（需要 get_energy_range() 方法）。

    返回
    ----
    (emin, emax) : (float, float) 或 None
    """
    if args.emin is not None or args.emax is not None:
        auto_range = data_source.get_energy_range()
        emin = args.emin if args.emin is not None else auto_range[0]
        emax = args.emax if args.emax is not None else auto_range[1]
        return (emin, emax)
    return None


# ============================================================
# 程序主入口
# ============================================================
def main() -> None:
    """
    程序主入口函数。

    根据命令行参数决定启动模式:
      - 带有 --no-gui → CLI 命令行模式
      - 不带 --no-gui → GUI 图形界面模式
    """
    # 打印启动横幅
    print("=" * 60)
    print("  PDOS 态密度数据提取与绘图工具")
    print("  支持: Materials Studio .xcd + CASTEP .castep_bin/.pdos_bin")
    print("=" * 60)

    args = build_argparser().parse_args()

    if args.no_gui:
        # ── CLI 模式 ──
        run_cli(args)
    else:
        # ── GUI 模式 ──
        print("[INFO] 启动 GUI 图形界面模式...")
        print("[INFO] 提示: 使用 --no-gui 参数可进入命令行模式。")
        print()
        try:
            from .pdos_gui import PDOSGUI
        except ImportError:
            from pdos_gui import PDOSGUI
        initial = args.file[0] if args.file else None
        gui = PDOSGUI(initial_file=initial)
        gui.run()
