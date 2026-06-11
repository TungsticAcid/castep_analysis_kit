#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CastepBinParser —— 直接读取 .castep_bin 内嵌 DOS 数据
======================================================

无需 .pdos_bin 文件，直接从 CASTEP 的 .castep_bin 提取已计算的:
  - 总态密度 (Total DOS)
  - 积分态密度 (Integrated DOS)
  - 元素分辨 PDOS（如 H, C, Ni）

数据来源: .castep_bin 的 G 段（4808 字节记录，601 个 float64）
  每条记录 = [占位, DOS(E1), 0, DOS(E2), 0, ..., DOS(E300), 0]
  300 个能量点，费米能级已对齐到 E=0

可选增强: 若同目录存在 *_DOS.pdos_bin，自动加载 m_l 分辨。

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/06/11
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .binary_io import read_all_records, try_decode_ascii, read_record_float64, read_record_int32
except ImportError:
    from binary_io import read_all_records, try_decode_ascii, read_record_float64, read_record_int32

try:
    from .constants import DEFAULT_PIC_DIR
except ImportError:
    from constants import DEFAULT_PIC_DIR


KPT_GROUP_SIZE = 16  # 每个 K 点的记录组大小
SPIN_COMPONENTS = 4  # up PDOS, up IDOS, down PDOS, down IDOS


class CastepBinParser:
    """
    从 .castep_bin 文件提取元素分辨 DOS，与 PDOSParser 接口兼容。

    使用方式:
        parser = CastepBinParser("path/to/file.castep_bin")
        parser.parse()
        data = parser.get_data(orbitals=["H", "C", "Ni"], spin="alpha")
        total = parser.get_total_dos()
    """

    def __init__(self, filepath: str) -> None:
        """
        初始化解析器。

        参数
        ----
        filepath : str
            .castep_bin 文件的完整路径。
        """
        self.filepath: str = filepath
        self.filename: str = os.path.basename(filepath)
        self.label: str = os.path.splitext(self.filename)[0]

        # 原始数据
        self.raw_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.energy_grid: Optional[np.ndarray] = None
        self.e_fermi: float = 0.0

        # 轨道和自旋信息
        self.has_spin: bool = True  # CASTEP DOS 总是自旋极化
        self.available_orbitals: List[str] = []
        self.available_spins: List[str] = ["alpha", "beta"]

        # K 点权重
        self.kpoint_weights: Optional[np.ndarray] = None

        # 是否有 m_l 分辨（来自 .pdos_bin）
        self._has_ml_resolved: bool = False
        self._ml_calc = None  # 延迟导入 CastepPDOSCalculator

    # ----------------------------------------------------------
    # 便捷属性
    # ----------------------------------------------------------
    @property
    def non_sum_orbitals(self) -> List[str]:
        """排除 "sum" 的可用轨道列表。"""
        return [o for o in self.available_orbitals if o != "sum"]

    # ----------------------------------------------------------
    # 主解析入口
    # ----------------------------------------------------------
    def parse(self) -> None:
        """
        解析 .castep_bin 文件，提取 DOS 数据。

        流程:
          1. 读取所有 Fortran 记录
          2. 筛选 4808 字节 DOS 记录
          3. 按 K 点分组（16 条/组）
          4. 构建能量网格
          5. 解析物种信息
          6. 提取各轨道 DOS 并 K 点平均
        """
        print(f"[INFO] 正在解析 .castep_bin: {self.filepath}")
        records = read_all_records(self.filepath)

        # --- 提取费米能级 ---
        self._extract_fermi(records)

        # --- 提取 K 点权重 ---
        self._extract_kpoint_weights(records)

        # --- 筛选 DOS 数据记录 ---
        dos_records = [r for r in records if len(r) == 4808]
        n_dos = len(dos_records)
        if n_dos == 0:
            raise ValueError("未在 .castep_bin 中找到 DOS 数据（4808 字节记录）。")

        n_kpts = n_dos // KPT_GROUP_SIZE
        print(f"[INFO] DOS 记录: {n_dos} 条 → {n_kpts} 个 K 点 × {KPT_GROUP_SIZE} 条/组")

        # --- 构建能量网格 ---
        self._build_energy_grid(records, dos_records)

        # --- 解析物种信息 ---
        species_names = self._extract_species_names(records)
        print(f"[INFO] 元素: {species_names}")

        # --- 提取每个轨道的 DOS ---
        # CASTEP 输出顺序: Total, 然后按 DOS 权重从大到小排列各元素
        # （通常与 SPECIES_SYMBOL 输入顺序相反，即 Ni, C, H 而非 H, C, Ni）
        orbital_names = ["Total"] + list(reversed(species_names))
        n_orbital_types = len(orbital_names)       # 应当 = 4（此文件格式）

        for i_orb, orb_name in enumerate(orbital_names):
            pdos_up_list = []
            pdos_down_list = []

            for ikpt in range(n_kpts):
                base = ikpt * KPT_GROUP_SIZE + i_orb * SPIN_COMPONENTS
                if base + 3 < n_dos:
                    pdos_up   = read_record_float64(dos_records[base])[1::2]     # 奇数索引=有效 DOS
                    pdos_down = read_record_float64(dos_records[base + 2])[1::2]
                    pdos_up_list.append(pdos_up)
                    pdos_down_list.append(pdos_down)

            if pdos_up_list:
                # K 点加权平均
                pdos_up_avg = self._average_over_kpts(pdos_up_list)
                pdos_down_avg = self._average_over_kpts(pdos_down_list)

                self.raw_data[f"{orb_name} alpha"] = (self.energy_grid.copy(), pdos_up_avg)
                self.raw_data[f"{orb_name} beta"] = (self.energy_grid.copy(), -pdos_down_avg)  # 负值保持 MS 兼容

                # 计算 Sum（α+β，beta取绝对值）
                sum_dos = pdos_up_avg + pdos_down_avg
                self.raw_data[f"{orb_name} Sum"] = (self.energy_grid.copy(), sum_dos)

        # --- 设置轨道列表 ---
        self.available_orbitals = species_names + ["sum"]

        # --- 尝试加载 .pdos_bin ---
        self._try_load_pdos_bin()

        self._print_summary()

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------
    def _extract_fermi(self, records: List[bytes]) -> None:
        """提取费米能级。"""
        for i, r in enumerate(records):
            text = try_decode_ascii(r)
            if text and text.strip() == 'E_FERMI' and i + 1 < len(records):
                self.e_fermi = read_record_float64(records[i + 1])[0]
                print(f"[INFO] 费米能级: {self.e_fermi:.6f} eV")
                return

    def _extract_kpoint_weights(self, records: List[bytes]) -> None:
        """提取 K 点权重。"""
        for i, r in enumerate(records):
            text = try_decode_ascii(r)
            if text and 'KPOINT_WEIGHTS' in text and i + 1 < len(records):
                try:
                    self.kpoint_weights = read_record_float64(records[i + 1])
                    print(f"[INFO] K 点权重: sum={self.kpoint_weights.sum():.4f}")
                    return
                except Exception:
                    pass

    def _build_energy_grid(self, records: List[bytes],
                           dos_records: List[bytes]) -> None:
        """
        从本征值范围自动构建能量网格。

        策略: 找到 G2 段的本征值记录，用全局 min/max 作为网格范围，
        网格点数 = DOS 记录的有效数据点数（300）。
        """
        # 从 3728 字节的本征值记录中获取能量范围
        eig_records = [r for r in records if len(r) == 3728]
        if eig_records:
            all_eigs = []
            for rec in eig_records:
                all_eigs.append(read_record_float64(rec))
            all_eigs = np.concatenate(all_eigs)
            e_min, e_max = all_eigs.min(), all_eigs.max()
        else:
            e_min, e_max = -30, 10  # 回退默认值

        # 300 个均匀能量点
        n_points = 300
        self.energy_grid = np.linspace(e_min, e_max, n_points)
        print(f"[INFO] 能量网格: [{e_min:.2f}, {e_max:.2f}] eV, {n_points} 点")

    def _extract_species_names(self, records: List[bytes]) -> List[str]:
        """从文件头解析元素符号列表。"""
        for i, r in enumerate(records):
            text = try_decode_ascii(r)
            if text and 'SPECIES_SYMBOL' in text and i + 1 < len(records):
                try:
                    raw = records[i + 1].decode('ascii', errors='replace').strip()
                    # 格式: "H       C       Ni"
                    names = raw.split()
                    return names
                except Exception:
                    pass
        # 回退
        return ["El1", "El2", "El3"]

    def _average_over_kpts(self, kpt_data: List[np.ndarray]) -> np.ndarray:
        """对各 K 点数据做加权平均。"""
        n_kpts = len(kpt_data)
        if self.kpoint_weights is not None and len(self.kpoint_weights) >= n_kpts:
            weights = self.kpoint_weights[:n_kpts]
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_kpts) / n_kpts
        return np.average(np.array(kpt_data), axis=0, weights=weights)

    def _try_load_pdos_bin(self) -> None:
        """尝试在同目录查找对应的 .pdos_bin 文件。"""
        base_str = str(Path(self.filepath))
        if base_str.endswith('_DOS.castep_bin'):
            pdos_path = Path(base_str.replace('_DOS.castep_bin', '_DOS.pdos_bin'))
        elif base_str.endswith('.castep_bin'):
            pdos_path = Path(base_str.replace('.castep_bin', '.pdos_bin'))
        else:
            pdos_path = Path(base_str + '.pdos_bin')

        if pdos_path.exists():
            print(f"[INFO] 发现 .pdos_bin: {pdos_path.name}，加载完整 m_l 分辨...")
            try:
                from .pdos_calc import CastepPDOSCalculator, CastepPDOSAdapter
            except ImportError:
                from pdos_calc import CastepPDOSCalculator, CastepPDOSAdapter
            # 使用 load_from_files 同时加载两个文件，确保 crystal/eigenvalues/weights 全部就位
            self._ml_calc = CastepPDOSCalculator()
            self._ml_calc.load_from_files(self.filepath, str(pdos_path))
            self._has_ml_resolved = True
            print("[INFO] m_l 分辨 PDOS 可用")

    def _print_summary(self) -> None:
        """打印解析结果摘要。"""
        ml_info = "，含 m_l 分辨" if self._has_ml_resolved else ""
        print(f"[INFO] ──────────────────────────────")
        print(f"[INFO] .castep_bin 解析完成{ml_info}")
        print(f"[INFO] 可用轨道: {self.available_orbitals}")
        print(f"[INFO] 自旋方向: {self.available_spins}")
        print(f"[INFO] ──────────────────────────────")

    # ----------------------------------------------------------
    # 数据提取方法（兼容 PDOSParser 接口）
    # ----------------------------------------------------------
    def get_data(
        self,
        orbitals: Optional[List[str]] = None,
        spin: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        根据轨道和自旋筛选数据。

        参数
        ----
        orbitals : list of str, 可选
            轨道列表，如 ["H", "Ni"]。
        spin : str, 可选
            "alpha" / "beta" / "both" / "sum"。

        返回
        ----
        result : dict
            键如 "H (α)"，值为 (energies, dos)。
        """
        if orbitals is None:
            orbitals = self.available_orbitals

        if self.has_spin and spin == "sum":
            return self.get_summed_data(orbitals=orbitals)

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for name, (energies, dos) in self.raw_data.items():
            name_lower = name.lower().strip()

            # 匹配轨道
            orbital_match = None
            for orb in orbitals:
                orb_lower = orb.lower()
                tokens = name_lower.split()
                if orb_lower in tokens:
                    orbital_match = orb
                    break

            if orbital_match is None:
                continue

            # 匹配自旋
            if spin == "both":
                pass  # 保留全部
            elif spin is not None and spin != "both":
                spin_map = {"alpha": "alpha", "beta": "beta", "up": "alpha", "down": "beta"}
                expected = spin_map.get(spin, spin)
                if expected not in name_lower:
                    continue
            else:
                # 默认只取 alpha
                if "beta" in name_lower:
                    continue

            # 构造显示标签
            if "alpha" in name_lower:
                label = f"{orbital_match} (α)"
            elif "beta" in name_lower:
                label = f"{orbital_match} (β)"
            elif "sum" in name_lower:
                label = f"{orbital_match} (α+β)"
            else:
                label = orbital_match

            result[label] = (energies.copy(), dos.copy())

        return result

    def get_total_dos(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """获取总态密度（α+β 求和）。"""
        if "Total Sum" in self.raw_data:
            e, dos = self.raw_data["Total Sum"]
            return e, np.abs(dos)
        # 回退：从 alpha 和 beta 求和
        if "Total alpha" in self.raw_data and "Total beta" in self.raw_data:
            e = self.raw_data["Total alpha"][0]
            dos = np.abs(self.raw_data["Total alpha"][1]) + np.abs(self.raw_data["Total beta"][1])
            return e, dos
        return None

    def get_energy_range(self) -> Tuple[float, float]:
        """获取能量范围。"""
        if self.energy_grid is not None:
            return (float(self.energy_grid[0]), float(self.energy_grid[-1]))
        return (-30.0, 10.0)

    def get_summed_data(
        self,
        orbitals: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """α+β 求和。"""
        if orbitals is None:
            orbitals = [o for o in self.available_orbitals if o != "sum"]

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for orb in orbitals:
            key_a = f"{orb} alpha"
            key_b = f"{orb} beta"
            if key_a in self.raw_data:
                e = self.raw_data[key_a][0]
                dos_a = np.abs(self.raw_data[key_a][1])
                dos_b = np.abs(self.raw_data.get(key_b, (None, np.zeros_like(dos_a)))[1])
                label = f"{orb} (α+β)"
                result[label] = (e.copy(), dos_a + dos_b)

        return result
