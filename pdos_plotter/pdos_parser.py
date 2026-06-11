#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDOSParser —— Materials Studio .xcd 格式 PDOS 文件解析器
==========================================================

解析 MS 导出的 XML 格式态密度文件，提取能量-DOS 数据，
自动检测自旋极化和轨道类型。

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11 (从 pdos_plotter.py 拆分，增加 non_sum_orbitals property)
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from .constants import ORBITAL_SORT_ORDER
except ImportError:
    from constants import ORBITAL_SORT_ORDER


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
        # 用户可自定义的标签，用于多文件叠加时标识数据来源
        # 默认取文件名主干（不含扩展名），如 "Ni (1 1 1)-CH4-top PDOS H"
        self.label: str = os.path.splitext(self.filename)[0]

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
    # 便捷属性
    # ----------------------------------------------------------
    @property
    def non_sum_orbitals(self) -> List[str]:
        """
        返回排除 "sum"/"total" 的可用轨道列表。

        消除代码中 7 处重复的 [o for o in ... if o != "sum"] 模式。
        """
        return [o for o in self.available_orbitals if o != "sum"]

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
            print(f"[INFO]   - 系列 \"{name}\": {len(points)} 个数据点, "
                  f"能量范围 [{energies.min():.4f}, {energies.max():.4f}]")

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
        self.available_orbitals.sort(key=lambda x: ORBITAL_SORT_ORDER.get(x, 99))

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

        # "sum" 自旋模式：直接返回 α+β 求和数据
        if self.has_spin and spin == "sum":
            return self.get_summed_data(orbitals=orbitals)

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for name, (energies, dos) in self.raw_data.items():
            name_lower = name.lower().strip()

            # ---- 第1步：匹配轨道 ----
            orbital_match: Optional[str] = None
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
            if self.has_spin:
                if spin == "both":
                    # 不过滤自旋，α 和 β 都保留
                    pass
                elif spin is not None:
                    # 指定了自旋方向，过滤不匹配的系列
                    if spin.lower() not in name_lower:
                        continue
                else:
                    # spin is None: 默认只取 alpha（向后兼容）
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

    def get_summed_data(
        self,
        orbitals: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        获取 α+β 自旋求和后的态密度数据。

        对每个指定轨道，找到对应的 α 和 β 系列，将 DOS 值相加。
        MS 文件中 beta DOS 为负值，求和前自动取绝对值。
        当 α 和 β 能量采样点不一致时，构建公共均匀能量网格并插值后再求和。

        参数
        ----
        orbitals : list of str, 可选
            要获取的轨道列表。默认使用所有非 sum 轨道。

        返回
        ----
        result : dict
            键为 "s (α+β)" 格式的标签，值为 (能量, 求和后的DOS) 元组。
            若某轨道只有 α 没有 β，则直接返回 α 的数据。
        """
        if orbitals is None:
            orbitals = self.non_sum_orbitals

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for orb in orbitals:
            alpha_key, beta_key = None, None
            for name in self.raw_data:
                nl = name.lower()
                if orb in nl.split():
                    if "alpha" in nl:
                        alpha_key = name
                    elif "beta" in nl:
                        beta_key = name

            energies_a, dos_a = (self.raw_data[alpha_key] if alpha_key
                                 else (np.array([]), np.array([])))
            energies_b, dos_b_neg = (self.raw_data[beta_key] if beta_key
                                      else (np.array([]), np.array([])))

            # 只有单一自旋：直接返回
            if len(energies_a) == 0 and len(energies_b) == 0:
                continue
            if len(energies_a) > 0 and len(energies_b) == 0:
                label = f"{orb} (α+β)"
                result[label] = (energies_a, np.abs(dos_a))
                continue
            if len(energies_b) > 0 and len(energies_a) == 0:
                label = f"{orb} (α+β)"
                result[label] = (energies_b, np.abs(dos_b_neg))
                continue

            # α 和 β 都存在，且能量网格一致 → 直接求和
            if len(energies_a) == len(energies_b) and np.allclose(energies_a, energies_b):
                label = f"{orb} (α+β)"
                result[label] = (energies_a, np.abs(dos_a) + np.abs(dos_b_neg))
                continue

            # 网格不一致 → 构建公共均匀能量网格
            # 公共范围：取 α 和 β 的交集，避免外推
            e_min = max(energies_a.min(), energies_b.min())
            e_max = min(energies_a.max(), energies_b.max())
            # 使用较密的分辨率（取两点格中步长较小的）
            step_a = (energies_a[-1] - energies_a[0]) / max(len(energies_a) - 1, 1)
            step_b = (energies_b[-1] - energies_b[0]) / max(len(energies_b) - 1, 1)
            step = min(step_a, step_b)
            n_points = max(int(np.ceil((e_max - e_min) / step)) + 1, 2)
            common_e = np.linspace(e_min, e_max, n_points)

            # 插值到公共网格后求和
            dos_a_interp = np.interp(common_e, energies_a, np.abs(dos_a))
            dos_b_interp = np.interp(common_e, energies_b, np.abs(dos_b_neg))

            label = f"{orb} (α+β)"
            result[label] = (common_e, dos_a_interp + dos_b_interp)

        return result
