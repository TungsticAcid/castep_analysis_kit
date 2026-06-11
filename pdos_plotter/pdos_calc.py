#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CASTEP PDOS 计算器 —— 从 .castep_bin + .pdos_bin 计算 m_l 分辨态密度
======================================================================

功能概述:
  - CastepPDOSCalculator: 从 CASTEP 二进制输出加载原始数据，
    执行 Gaussian 展宽计算，得到 m_l 分辨的 PDOS
  - CastepPDOSAdapter: 桥接适配器，使计算结果兼容 PDOSPlotter 接口

算法:
  PDOS(E, orbital) = Σ_kpt w_kpt × Σ_band W(kpt,spin,band,orb) × S(E - ε)
  其中 S 为 Gaussian 展宽函数，W 为轨道投影权重矩阵。

输入文件:
  - *_DOS.castep_bin: 能带本征值 ε(k, spin, band)、费米能级、K 点权重
  - *_DOS.pdos_bin: 轨道投影权重 W(k, spin, band, orbital)，584 个 m_l 轨道

轨道映射（584 个轨道）:
  - Ni (原子 1-64) × 9 轨道 = 576 个: s, px, py, pz, d_xy, d_yz, d_z2, d_xz, d_x2-y2
  - C  (原子 65)   × 4 轨道 =   4 个: s, px, py, pz
  - H  (原子 66-69) × 1 轨道 =   4 个: s
  总计 584 个

参考: pdos_calculation_方案.md

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/06/11
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from .binary_io import (
        read_all_records,
        read_record_float64,
        read_record_float64_scalar,
        read_record_int32,
        try_decode_ascii,
        find_label_indices,
    )
    from .constants import (
        ORBITAL_NAMES,
        SPECIES_ORBITALS,
        SPECIES_INDEX_MAP,
        ORBITAL_TO_L,
        L_TO_NAME,
        ORBITAL_SORT_ORDER,
        DEFAULT_INTEGRAL_EMIN,
        DEFAULT_INTEGRAL_EMAX,
    )
except ImportError:
    from binary_io import (
        read_all_records,
        read_record_float64,
        read_record_float64_scalar,
        read_record_int32,
        try_decode_ascii,
        find_label_indices,
    )
    from constants import (
        ORBITAL_NAMES,
        SPECIES_ORBITALS,
        SPECIES_INDEX_MAP,
        ORBITAL_TO_L,
        L_TO_NAME,
        ORBITAL_SORT_ORDER,
        DEFAULT_INTEGRAL_EMIN,
        DEFAULT_INTEGRAL_EMAX,
    )


# ============================================================
# CastepPDOSCalculator —— 核心计算类
# ============================================================
class CastepPDOSCalculator:
    """
    从 CASTEP .castep_bin + .pdos_bin 计算 m_l 分辨的 PDOS。

    使用方式:
        calc = CastepPDOSCalculator()
        calc.load_from_files("path/to/DOS.castep_bin", "path/to/DOS.pdos_bin")
        result = calc.compute_pdos(e_min=-15, e_max=10, n_points=500, sigma=0.2)
        aggregated = calc.aggregate(group_by='species_orbital')
    """

    def __init__(self) -> None:
        """初始化计算器，所有数据数组初始为 None。"""
        # --- 原始输入数据 ---
        self.eigenvalues_up: Optional[NDArray] = None      # (nkpt, nband_castep) 自旋向上本征值
        self.eigenvalues_down: Optional[NDArray] = None    # (nkpt, nband_castep) 自旋向下
        self.pdos_weights: Optional[NDArray] = None        # (nkpt, nspin, nband_pdos, norb)
        self.kpoint_weights: Optional[NDArray] = None      # (nkpt,)
        self.e_fermi: float = 0.0
        self.orbital_map: List[dict] = []                  # 584 个轨道的完整映射表
        self.species: List[str] = []
        self.crystal = None  # CrystalStructure，由 _load_castep_bin 填充

        # --- 计算结果缓存 ---
        self.energy_grid: Optional[NDArray] = None         # (n_e,) 能量网格
        self.pdos_array: Optional[NDArray] = None          # (n_e, norb, nspin) PDOS 结果
        self._computed: bool = False

    # ================================================================
    # 步骤 1: 加载数据
    # ================================================================

    def load_from_files(self, castep_bin_path: str, pdos_bin_path: str) -> None:
        """
        从两个二进制文件加载所有原始数据。

        参数
        ----
        castep_bin_path : str
            *_DOS.castep_bin 文件的完整路径。
        pdos_bin_path : str
            *_DOS.pdos_bin 文件的完整路径。
        """
        self._load_castep_bin(castep_bin_path)
        self._load_pdos_bin(pdos_bin_path)
        print(f"[INFO] CASTEP 数据加载完成: "
              f"nkpt={self.eigenvalues_up.shape[0] if self.eigenvalues_up is not None else '?'}, "
              f"nspin=2, E_Fermi={self.e_fermi:.4f} eV")

    def _load_castep_bin(self, path: str) -> None:
        """
        解析 .castep_bin 文件，提取本征值、K 点权重和费米能级。

        文件结构（关键记录）:
          - SPECTRAL_KPOINT_WEIGHTS: K 点权重数组
          - E_FERMI: 费米能级值
          - 能带本征值: 每个 K 点 × 自旋的 466 条 float64 记录
        """
        records = read_all_records(path)

        # --- 提取费米能级 ---
        for i, rec in enumerate(records):
            text = try_decode_ascii(rec)
            if text and text.strip() == 'E_FERMI' and i + 1 < len(records):
                self.e_fermi = read_record_float64_scalar(records[i + 1])
                break
        print(f"[INFO] 费米能级: {self.e_fermi:.6f} eV")

        # --- 提取晶体结构 ---
        try:
            from .crystal_viewer import CrystalStructure
        except ImportError:
            from crystal_viewer import CrystalStructure
        self.crystal = CrystalStructure.from_castep_bin(records)
        print(f"[INFO] 晶体结构: {len(self.crystal.species)} 原子, 晶胞 {self.crystal.lattice[0,0]:.2f} x {self.crystal.lattice[1,1]:.2f} x {self.crystal.lattice[2,2]:.2f} A^3")

        # --- 提取 K 点权重 ---
        label_idx = find_label_indices(records)
        if 'SPECTRAL_KPOINT_WEIGHTS' in label_idx:
            wrec = records[label_idx['SPECTRAL_KPOINT_WEIGHTS'] + 1]
            self.kpoint_weights = read_record_float64(wrec)
            print(f"[INFO] K 点权重: {self.kpoint_weights} (sum={self.kpoint_weights.sum():.4f})")
        else:
            # 回退：假设均匀权重
            print("[WARN] 未找到 SPECTRAL_KPOINT_WEIGHTS，将使用均匀 K 点权重。")
            self.kpoint_weights = None

        # --- 提取本征值 ---
        # 找到最后一个 END_CELL_GLOBAL 之后的记录
        end_cg_indices = [i for i, r in enumerate(records)
                          if try_decode_ascii(r) == 'END_CELL_GLOBAL']
        start = end_cg_indices[-1] + 1 if end_cg_indices else 0

        eig_up_list, eig_down_list = [], []
        i = start
        # 跳过全局参数记录（4 字节整数、8 字节浮点、24 字节 K 点坐标）
        while i < len(records) and len(records[i]) in (4, 8, 24):
            i += 1

        kpt_count = 0
        nkpts_expected = 5  # 本项目固定 5 个 K 点
        while i < len(records) and kpt_count < nkpts_expected:
            rec_len = len(records[i])
            # 每个 K 点有 4 条 3728 字节记录: occ_up, eig_up, occ_down, eig_down
            if rec_len in (3728, 3736) and i + 3 < len(records):
                next_lens = [len(records[i+j]) for j in range(4)]
                if all(l in (3728, 3736) for l in next_lens):
                    # occ_up 的平均值 > occ_down（铁磁体系，上自旋占据更多）
                    eig_up_arr = read_record_float64(records[i + 1])
                    eig_down_arr = read_record_float64(records[i + 3])
                    eig_up_list.append(eig_up_arr)
                    eig_down_list.append(eig_down_arr)
                    i += 4
                    kpt_count += 1
                else:
                    i += 1
            elif rec_len == 24:
                # 跳过 K 点坐标记录（24 字节: float64[3]）
                i += 1
            else:
                i += 1

        nkpts_actual = len(eig_up_list)
        if nkpts_actual == 0:
            raise ValueError("无法从 .castep_bin 中提取本征值。请检查文件完整性。")
        # 使用实际读取的 K 点数（可能少于预期）
        nkpts = nkpts_actual

        self.eigenvalues_up = np.array(eig_up_list)     # (nkpt, nband)
        self.eigenvalues_down = np.array(eig_down_list)
        print(f"[INFO] 本征值提取完成: {nkpts} K 点 × {self.eigenvalues_up.shape[1]} 能带 × 2 自旋")

        # 若未读取到 K 点权重，使用均匀权重
        if self.kpoint_weights is None:
            self.kpoint_weights = np.ones(nkpts) / nkpts

    def _load_pdos_bin(self, path: str) -> None:
        """
        解析 .pdos_bin 文件，提取轨道映射表和投影权重矩阵。

        文件结构（关键记录）:
          记录 0:  float64 = 1.0 (文件格式版本)
          记录 1:  char[80] (时间戳)
          记录 2:  int32 = nkpts
          记录 3:  int32 = nspin
          记录 4:  int32 = norb
          记录 5:  int32 = nbands
          记录 6:  int32[norb] (元素索引)
          记录 7:  int32[norb] (原子索引)
          记录 8:  int32[norb] (角动量/自旋标记)
          记录 9+: K 点坐标 + 自旋标记 + PDOS 权重
        """
        records = read_all_records(path)

        # 解析头部
        nkpts  = int(read_record_float64_scalar(records[0]) if len(records[0]) == 8
                     else read_record_int32(np.frombuffer(records[2], dtype='>i4').tobytes())[0])
        # 由于头部各记录长度不同，使用可靠方式读取：
        # 记录 2 是 int32 标量（4 字节），记录 3、4、5 同理
        nkpts  = self._read_header_int32(records, 2)   # = 5
        nspin  = self._read_header_int32(records, 3)   # = 2
        norb   = self._read_header_int32(records, 4)   # = 584
        nbands = self._read_header_int32(records, 5)   # = 714

        print(f"[INFO] .pdos_bin 头部: nkpts={nkpts}, nspin={nspin}, norb={norb}, nbands={nbands}")

        # --- 轨道映射 ---
        species_idx = read_record_int32(records[6])  # (584,) 元素索引
        ion_idx     = read_record_int32(records[7])  # (584,) 原子序号
        # records[8]: 角动量/自旋标记，可选用

        # 自动检测元素索引 → 名称的映射（基于每原子轨道数）
        self._detect_species_mapping(species_idx, ion_idx)

        self.orbital_map = self._build_orbital_map(species_idx, ion_idx)
        # 按 Ni, C, H 顺序排序物种列表
        species_order = {'Ni': 0, 'C': 1, 'H': 2}
        unique_species = sorted(
            set(orb['species'] for orb in self.orbital_map),
            key=lambda s: species_order.get(s, 99)
        )
        self.species = unique_species

        # --- 提取 PDOS 权重矩阵 ---
        weights_all = np.zeros((nkpts, nspin, nbands, norb), dtype=np.float64)
        rec_idx = 9  # 权重数据从记录 9 开始

        for ikpt in range(nkpts):
            # 跳过 K 点头记录（一般为 28 字节: float64[3] + int32）
            rec_idx += 1
            for ispin in range(nspin):
                # 跳过 spin_marker (4 字节 int32) 和 count (4 字节 int32)
                rec_idx += 2
                for iband in range(nbands):
                    weights_all[ikpt, ispin, iband, :] = read_record_float64(records[rec_idx])
                    rec_idx += 1

        self.pdos_weights = weights_all  # (nkpts, nspin, nbands, norb)
        print(f"[INFO] PDOS 权重矩阵加载完成: {nkpts}×{nspin}×{nbands}×{norb}")

    @staticmethod
    def _read_header_int32(records: List[bytes], idx: int) -> int:
        """从 records[idx] 中读取 int32 标量值。"""
        return int(np.frombuffer(records[idx], dtype='>i4')[0])

    # ================================================================
    # 步骤 2a: 自动检测元素索引映射
    # ================================================================
    def _detect_species_mapping(
        self,
        species_idx_arr: np.ndarray,
        ion_idx_arr: np.ndarray,
    ) -> None:
        """
        自动检测元素索引 → 元素名称的映射。

        基于每原子轨道数进行推断：
          - 9 轨道/原子 → Ni (s + 3p + 5d)
          - 4 轨道/原子 → C  (s + 3p)
          - 1 轨道/原子 → H  (s)

        结果存储在实例变量 self._species_idx_to_name 中。
        """
        # 每原子轨道数 → 元素名
        norb_to_species = {9: 'Ni', 4: 'C', 1: 'H'}

        self._species_idx_to_name: dict = {}
        for sp_idx in np.unique(species_idx_arr):
            mask = species_idx_arr == sp_idx
            ions_of_species = ion_idx_arr[mask]
            unique_ions = np.unique(ions_of_species)
            # 每原子轨道数 = 该物种的总轨道数 / 原子数
            n_per_ion = len(ions_of_species) // len(unique_ions)
            name = norb_to_species.get(n_per_ion, f'El{sp_idx}')
            self._species_idx_to_name[int(sp_idx)] = name
            print(f"[INFO] 元素索引 {sp_idx} → {name} "
                  f"({len(unique_ions)} 原子 × {n_per_ion} 轨道 = {len(ions_of_species)} 总轨道)")

    # ================================================================
    # 步骤 2b: 构建轨道映射表
    # ================================================================

    def _build_orbital_map(
        self,
        species_idx_arr: np.ndarray,
        ion_idx_arr: np.ndarray,
    ) -> List[dict]:
        """
        根据记录6(元素索引)和记录7(原子索引)构建 584 轨道的完整映射表。

        利用轨道按原子连续排列的特性：
        每遇到新的 ion_index，重置该原子的局部轨道计数器，
        按 SPECIES_ORBITALS 定义的顺序分配轨道名。

        参数
        ----
        species_idx_arr : ndarray (int32)
            记录6，值 1=Ni, 2=C, 3=H。
        ion_idx_arr : ndarray (int32)
            记录7，值 1-64=Ni, 65=C, 66-69=H。

        返回
        ----
        orbital_map : list of dict
            每个元素包含:
              - global_index: 0-583
              - species: 'Ni'/'C'/'H'
              - ion_index: 1-69 (原子序号)
              - orbital_name: 's'/'px'/.../'d_x2-y2'
              - angular_momentum: 0/1/2
        """
        result: List[dict] = []
        prev_ion: Optional[int] = None
        prev_species: Optional[str] = None
        local_orb_counter: int = 0

        for i_global in range(len(species_idx_arr)):
            sp_idx = int(species_idx_arr[i_global])
            species = self._species_idx_to_name.get(sp_idx, '?')
            ion = int(ion_idx_arr[i_global])

            # 遇到新原子（不同元素或不同离子序号）→ 重置局部轨道计数器
            if ion != prev_ion or species != prev_species:
                local_orb_counter = 0
                prev_ion = ion
                prev_species = species
            else:
                local_orb_counter += 1

            # 获取该元素的轨道列表
            orb_list = SPECIES_ORBITALS.get(species, ['?'])
            # 防止越界（理论上不会发生）
            orb_name = orb_list[local_orb_counter % len(orb_list)]

            result.append({
                'global_index': i_global,
                'species': species,
                'ion_index': ion,
                'orbital_name': orb_name,
                'angular_momentum': ORBITAL_TO_L.get(orb_name, -1),
            })

        return result

    # ================================================================
    # 步骤 3: 计算 PDOS
    # ================================================================

    def compute_pdos(
        self,
        e_min: float = -15.0,
        e_max: float = 10.0,
        n_points: int = 500,
        sigma: float = 0.2,
        progress_callback=None,
    ) -> dict:
        """
        计算 m_l 分辨的 PDOS。

        核心算法:
          PDOS(E, orb) = Σ_kpt w_kpt × Σ_band W(kpt,spin,band,orb) × S(E - ε)
          使用矩阵乘法: PDOS = w_kpt × (smear @ weights)
          smear: (n_e, nband), weights: (nband, norb) → result: (n_e, norb)

        参数
        ----
        e_min, e_max : float
            能量范围 (eV, 相对于费米能级)。
        n_points : int
            能量网格点数。
        sigma : float
            Gaussian 展宽宽度 (eV)。
        progress_callback : callable, 可选
            进度回调函数。

        返回
        ----
        result : dict
            {'energy': ndarray, 'e_fermi': float, 'orbital_map': list,
             'pdos_raw': ndarray, 'species': list, 'total': dict}
        """
        if self.eigenvalues_up is None or self.pdos_weights is None:
            raise RuntimeError("请先调用 load_from_files() 加载数据。")

        # 1. 构建能量网格（相对于费米能级）
        self.energy_grid = np.linspace(e_min, e_max, n_points)

        # 2. 对齐本征值到费米能级
        eig_up   = self.eigenvalues_up - self.e_fermi     # (nkpt, nband_castep)
        eig_down = self.eigenvalues_down - self.e_fermi

        nkpt = eig_up.shape[0]
        nband_castep = eig_up.shape[1]
        _, _, nband_pdos, norb = self.pdos_weights.shape

        # .castep_bin 本征值数量可能 ≠ .pdos_bin 能带数，取交集
        nband_use = min(nband_castep, nband_pdos)
        if nband_castep != nband_pdos:
            print(f"[INFO] 能带数不一致: .castep_bin={nband_castep}, "
                  f".pdos_bin={nband_pdos}，使用 {nband_use} 条")

        # 3. 对每个 K 点、自旋：计算展宽矩阵 @ 权重矩阵
        pdos_raw = np.zeros((n_points, nkpt, 2, norb), dtype=np.float64)

        for ikpt in range(nkpt):
            wkpt = self.kpoint_weights[ikpt]

            # 自旋向上
            smear_up = self._gaussian_kernel(
                self.energy_grid, eig_up[ikpt, :nband_use], sigma
            )
            pdos_raw[:, ikpt, 0, :] = wkpt * (
                smear_up @ self.pdos_weights[ikpt, 0, :nband_use, :]
            )

            # 自旋向下
            smear_down = self._gaussian_kernel(
                self.energy_grid, eig_down[ikpt, :nband_use], sigma
            )
            pdos_raw[:, ikpt, 1, :] = wkpt * (
                smear_down @ self.pdos_weights[ikpt, 1, :nband_use, :]
            )

            if progress_callback:
                progress_callback(ikpt + 1, nkpt)

        # 4. 汇总 K 点: (n_e, nkpt, nspin, norb) → (n_e, norb, nspin)
        self.pdos_array = pdos_raw.sum(axis=1)           # (n_e, nspin, norb)
        self.pdos_array = self.pdos_array.transpose(0, 2, 1)  # (n_e, norb, nspin)
        self._computed = True

        # 5. 构建返回字典
        return self._build_result_dict()

    @staticmethod
    def _gaussian_kernel(
        e_grid: NDArray,
        eigenvalues: NDArray,
        sigma: float,
    ) -> NDArray:
        """
        Gaussian 展宽矩阵。

        S(E, ε) = exp(-0.5 * ((E-ε)/σ)²) / (σ × √(2π))

        参数
        ----
        e_grid : (n_e,) ndarray
            能量网格。
        eigenvalues : (n_band,) ndarray
            本征值。
        sigma : float
            展宽宽度 (eV)。

        返回
        ----
        smear : (n_e, n_band) ndarray
            展宽矩阵，每列是一个本征态在所有能量网格点的展宽值。
        """
        de = e_grid[:, np.newaxis] - eigenvalues[np.newaxis, :]  # (n_e, n_band)
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        return norm * np.exp(-0.5 * (de / sigma) ** 2)

    # ================================================================
    # 步骤 4: 轨道聚合
    # ================================================================

    def aggregate(self, group_by: str = 'species_orbital') -> dict:
        """
        将 584 轨道的 PDOS 聚合到用户指定的分组级别。

        参数
        ----
        group_by : str
            聚合级别，可选:
              - 'total':           总 DOS（全部轨道求和）
              - 'species':         按元素（Ni, C, H）
              - 'species_l':       按元素 + 角动量（Ni-s, Ni-p, Ni-d, ...）
              - 'species_orbital': 按元素 + 具体轨道（Ni-s, Ni-px, ..., Ni-d_x2-y2, ...）
              - 'atom_l':          按原子 + 角动量（Ni_1-d, Ni_2-d, ...）
              - 'atom_orbital':    按原子 + 具体轨道（Ni_1-d_xy, ...）
              - 'raw':             不聚合，保留 584 轨道

        返回
        ----
        result : dict
            键为标签（如 "Ni-d_xy_up"），值为 (n_e,) 的 float64 数组。
            每个标签有 _up 和 _down 两个自旋版本。
        """
        if not self._computed:
            raise RuntimeError("请先调用 compute_pdos()")

        result: Dict[str, np.ndarray] = {}

        # 总 DOS
        if group_by == 'total':
            result['total_up']   = self.pdos_array[:, :, 0].sum(axis=1)
            result['total_down'] = self.pdos_array[:, :, 1].sum(axis=1)
            return result

        # 构建聚合字典 {label: [global_indices]}
        agg_dict: Dict[str, List[int]] = {}

        for orb in self.orbital_map:
            gidx = orb['global_index']
            sp   = orb['species']
            orb_name = orb['orbital_name']
            l_val = orb['angular_momentum']
            ion  = orb['ion_index']

            if group_by == 'species':
                key = sp
            elif group_by == 'species_l':
                l_name = L_TO_NAME.get(l_val, f'l{l_val}')
                key = f'{sp}-{l_name}'
            elif group_by == 'species_orbital':
                key = f'{sp}-{orb_name}'
            elif group_by == 'atom_l':
                l_name = L_TO_NAME.get(l_val, f'l{l_val}')
                key = f'{sp}_{ion}-{l_name}'
            elif group_by == 'atom_orbital':
                key = f'{sp}_{ion}-{orb_name}'
            else:  # 'raw'
                key = f'orb_{gidx:03d}_{sp}_{ion}_{orb_name}'

            if key not in agg_dict:
                agg_dict[key] = []
            agg_dict[key].append(gidx)

        # 执行聚合：对每组 global_indices 求和
        for key, gidx_list in agg_dict.items():
            result[f'{key}_up']   = self.pdos_array[:, gidx_list, 0].sum(axis=1)
            result[f'{key}_down'] = self.pdos_array[:, gidx_list, 1].sum(axis=1)

        return result

    # ================================================================
    # 辅助方法
    # ================================================================

    def _build_result_dict(self) -> dict:
        """构建完整的结果字典。"""
        total_up   = self.pdos_array[:, :, 0].sum(axis=1)
        total_down = self.pdos_array[:, :, 1].sum(axis=1)

        return {
            'energy': self.energy_grid,
            'e_fermi': self.e_fermi,
            'orbital_map': self.orbital_map,
            'species': self.species,
            'pdos_raw': self.pdos_array,  # (n_e, norb, nspin)
            'total': {
                'up':   total_up,
                'down': total_down,
            },
        }


# ============================================================
# CastepPDOSAdapter —— 与 PDOSParser 兼容的桥接适配器
# ============================================================
class CastepPDOSAdapter:
    """
    桥接 CastepPDOSCalculator → PDOSPlotter 接口。

    实现与 PDOSParser 完全兼容的 duck-typing 接口，
    使 PDOSPlotter 无需任何修改即可使用 CASTEP 计算结果。

    使用方式:
        calc = CastepPDOSCalculator()
        calc.load_from_files("DOS.castep_bin", "DOS.pdos_bin")
        calc.compute_pdos()
        adapter = CastepPDOSAdapter(calc, group_by='species_orbital')
        plotter = PDOSPlotter(adapter)
        fig, ax = plotter.plot_orbitals(orbitals=['d_xy', 'd_yz', ...])
    """

    def __init__(
        self,
        calculator: CastepPDOSCalculator,
        group_by: str = 'species_orbital',
    ) -> None:
        """
        初始化适配器。

        参数
        ----
        calculator : CastepPDOSCalculator
            已完成 compute_pdos() 的计算器实例。
        group_by : str
            PDOS 聚合级别，参见 CastepPDOSCalculator.aggregate()。
        """
        self.calc = calculator
        self.group_by = group_by
        self._aggregated: Optional[dict] = None

        # --- 兼容 PDOSParser 的属性 ---
        self.filepath: str = ""
        self.filename: str = "CASTEP PDOS"
        self.label: str = "CASTEP"
        self.has_spin: bool = True  # CASTEP 总是输出自旋极化

        # 可用轨道列表（从聚合结果解析）
        self.available_orbitals: List[str] = []
        self.available_spins: List[str] = ['alpha', 'beta']

    def parse(self) -> None:
        """
        模拟 PDOSParser.parse() —— 执行聚合。

        必须在 compute_pdos() 之后调用。
        """
        if not self.calc._computed:
            raise RuntimeError("请先调用 CastepPDOSCalculator.compute_pdos()")
        self._aggregated = self.calc.aggregate(group_by=self.group_by)
        self._detect_orbitals()

    def get_data(
        self,
        orbitals: Optional[List[str]] = None,
        spin: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        返回兼容 PDOSParser.get_data() 的数据格式。

        参数
        ----
        orbitals : list of str, 可选
            要获取的轨道名列表，如 ['d_xy', 'd_yz']。
            None 表示所有可用轨道。
        spin : str, 可选
            'alpha' / 'beta' / 'both' / 'sum'。

        返回
        ----
        result : dict
            键如 "d_xy (alpha)", "d_xy (beta)"，值为 (energies, dos) 元组。
        """
        if self._aggregated is None:
            self.parse()

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for key, dos in self._aggregated.items():
            # 解析键: "Ni-d_xy_up" → orb_part="Ni-d_xy", spin_part="alpha"
            orb_part, spin_part = self._parse_key(key)
            if orb_part is None:
                continue

            # 过滤轨道：支持全名匹配（"Ni-d_xy"）和后缀匹配（"d_xy"）
            if orbitals:
                matched = False
                for o in orbitals:
                    if orb_part == o or orb_part.endswith('-' + o):
                        matched = True
                        break
                if not matched:
                    continue

            # 过滤自旋
            if spin is not None and spin != 'both':
                if spin == 'sum':
                    # sum 模式交给 get_summed_data 处理
                    continue
                if spin_part != spin:
                    continue

            if spin == 'both' or spin is None:
                label = f"{orb_part} ({spin_part})"
            else:
                label = f"{orb_part} ({spin_part})"

            result[label] = (self.calc.energy_grid.copy(), dos.copy())

        return result

    def get_summed_data(
        self,
        orbitals: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        α+β (up+down) 自旋求和。

        参数
        ----
        orbitals : list of str, 可选
            轨道列表。None 表示所有可用轨道。

        返回
        ----
        result : dict
            键如 "d_xy (α+β)"，值为 (energies, dos) 元组。
        """
        up_data   = self.get_data(orbitals=orbitals, spin='alpha')
        down_data = self.get_data(orbitals=orbitals, spin='beta')

        result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        for key in up_data:
            orb_name = key.split(' (')[0]
            dos_up = np.abs(up_data[key][1])
            down_key = f"{orb_name} (β)"
            dos_down = np.abs(down_data.get(down_key, (None, np.zeros_like(dos_up)))[1])
            label = f"{orb_name} (α+β)"
            result[label] = (self.calc.energy_grid.copy(), dos_up + dos_down)

        return result

    def get_total_dos(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        总 DOS = 全部轨道 + 全部自旋求和。

        返回
        ----
        (energies, dos) : (ndarray, ndarray)
        """
        total_up   = self.calc.pdos_array[:, :, 0].sum(axis=1)
        total_down = self.calc.pdos_array[:, :, 1].sum(axis=1)
        total = total_up + total_down
        return self.calc.energy_grid.copy(), total

    def get_energy_range(self) -> Tuple[float, float]:
        """
        获取能量网格的范围。

        返回
        ----
        (emin, emax) : (float, float)
        """
        eg = self.calc.energy_grid
        return (float(eg[0]), float(eg[-1]))

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------
    def _detect_orbitals(self) -> None:
        """自动检测可用轨道列表并排序。"""
        orb_set: set = set()
        for key in self._aggregated:
            orb_part, _ = self._parse_key(key)
            if orb_part:
                orb_set.add(orb_part)
        # 按 CASTEP 轨道顺序排序: s < p < d
        self.available_orbitals = sorted(
            orb_set,
            key=lambda x: (
                ORBITAL_SORT_ORDER.get(x[0], 99) if x else 99,
                x,
            )
        )

    @staticmethod
    def _parse_key(key: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析聚合键 "Ni-d_xy_up" → ("Ni-d_xy", "alpha")。

        UI 自旋约定为 "alpha"/"beta"，内部存储为 "_up"/"_down" 后缀，
        此方法完成映射。

        参数
        ----
        key : str
            聚合字典的键。

        返回
        ----
        (orb_part, spin_part) : (str or None, str or None)
        """
        if key.endswith('_up'):
            return key[:-3], 'alpha'
        elif key.endswith('_down'):
            return key[:-5], 'beta'
        return None, None
