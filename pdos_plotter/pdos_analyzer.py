#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDOSAnalyzer —— 态密度数据分析器
================================

提供 d带中心、峰搜索、晶场劈裂、自旋劈裂、带宽、占据态、
重叠面积等态密度分析功能。

改进（2026/06/11 重构）:
  - 抽取 _prepare_data() 消除 5 个方法中重复的前处理逻辑
  - 抽取 _get_first_series() 静态辅助函数
  - full_report() 分解为多个 _report_xxx() 私有方法

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/05/23
最后更新: 2026/06/11
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from .constants import (
        ORBITAL_COLORS,
        GAUSSIAN_FWHM_FACTOR,
        NOISE_THRESHOLD_RATIO,
        EPSILON,
        REPORT_WIDTH,
        DEFAULT_PEAK_MIN_HEIGHT,
        DEFAULT_PEAK_MIN_DISTANCE,
        L_TO_NAME,
        REPORT_FONT,
    )
except ImportError:
    from constants import (
        ORBITAL_COLORS,
        GAUSSIAN_FWHM_FACTOR,
        NOISE_THRESHOLD_RATIO,
        EPSILON,
        REPORT_WIDTH,
        DEFAULT_PEAK_MIN_HEIGHT,
        DEFAULT_PEAK_MIN_DISTANCE,
        L_TO_NAME,
        REPORT_FONT,
    )


# ============================================================
# 模块级工具函数
# ============================================================
def _get_first_series(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """
    从 get_data() 返回的字典中提取第一条系列。

    消除代码中 7 处重复的 list(data.items())[0] 模式。

    参数
    ----
    data : dict
        get_data() 的返回结果。

    返回
    ----
    (label, energies, dos) : (str, ndarray, ndarray) 或 None
    """
    if not data:
        return None
    key = next(iter(data))
    return key, data[key][0], data[key][1]


def _apply_energy_mask(
    energies: np.ndarray,
    emin: Optional[float] = None,
    emax: Optional[float] = None,
    extra_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    构建能量范围布尔掩码。

    参数
    ----
    energies : ndarray
        能量数组。
    emin : float, 可选
        能量下限。
    emax : float, 可选
        能量上限。
    extra_mask : ndarray, 可选
        额外的布尔掩码（如 E < 0 占据态条件），与范围掩码取 AND。

    返回
    ----
    mask : ndarray (bool)
    """
    mask = np.ones(len(energies), dtype=bool)
    if emin is not None:
        mask &= (energies >= emin)
    if emax is not None:
        mask &= (energies <= emax)
    if extra_mask is not None:
        mask &= extra_mask
    return mask


# ============================================================
# PDOSAnalyzer 类
# ============================================================
class PDOSAnalyzer:
    """
    PDOS 数据分析器，提供 d带中心、峰搜索、积分等功能。

    分析功能:
      - calc_band_center()      计算 d带/p带中心（加权平均能量）
      - find_peaks()            搜索态密度峰位置
      - calc_crystal_field_splitting()  晶场劈裂分析
      - calc_band_width()       带宽分析
      - calc_spin_splitting()   自旋劈裂分析
      - calc_peak_area()        指定能量区间积分
      - calc_occupancy()        费米能级以下占据态
      - calc_manual_peak()      手动选点识峰
      - calc_overlap_area()     轨道重叠面积
      - full_report()           综合分析报告
    """

    def __init__(self, parser) -> None:
        """
        初始化分析器。

        参数
        ----
        parser : PDOSParser 或 CastepPDOSAdapter
            已解析/计算完成的数据源实例。
        """
        self.parser = parser

    # ----------------------------------------------------------
    # 数据准备（消除重复）
    # ----------------------------------------------------------
    def _prepare_data(
        self,
        orbital: str,
        spin: Optional[str] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        extra_mask: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        统一的数据获取与前处理流水线。

        整合原先在 5 个方法中重复的:
          1. parser.get_data()
          2. 取第一条系列
          3. np.abs(dos)
          4. 能量范围掩码过滤
          5. 数据点数量检查

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        emin : float, 可选
            能量下限。
        emax : float, 可选
            能量上限。
        extra_mask : ndarray, 可选
            额外掩码（与范围掩码取 AND）。

        返回
        ----
        (e_range, d_range) : (ndarray, ndarray) 或 None
            掩码过滤后的能量和 DOS 数组。
        """
        data = self.parser.get_data(orbitals=[orbital], spin=spin)
        series = _get_first_series(data)
        if series is None:
            return None

        _, energies, dos_values = series
        dos_abs = np.abs(dos_values)

        mask = _apply_energy_mask(energies, emin, emax, extra_mask)
        e_range = energies[mask]
        d_range = dos_abs[mask]

        if len(e_range) < 2:
            return None

        return e_range, d_range

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

        支持两种计算方式:
          - "all" (全部态):      对 [emin, emax] 全区间积分
          - "occupied" (占据态): 仅对 E < 0 的占据态积分

        参数
        ----
        orbital : str
            轨道名称，如 "d"、"p"、"s"。
        spin : str, 可选
            自旋方向筛选。
        emin, emax : float, 可选
            积分能量上下限。
        method : str
            "all" 或 "occupied"。
        verbose : bool
            是否打印结果。

        返回
        ----
        result : dict
            {"band_center": float, "integral": float, "method": str}
        """
        extra = (energies < 0.0) if method == "occupied" else None
        prepared = self._prepare_data(orbital, spin, emin, emax, extra_mask=extra)
        if prepared is None:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据，无法计算带中心。")
            return {"band_center": float("nan"), "integral": float("nan"), "method": method}

        e_range, d_range = prepared
        numerator = np.trapz(e_range * d_range, e_range)
        denominator = np.trapz(d_range, e_range)

        if denominator == 0:
            if verbose:
                print(f"[WARN] {method}态 DOS 积分为零。")
            return {"band_center": float("nan"), "integral": 0.0, "method": method}

        band_center = numerator / denominator
        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            method_name = "占据态" if method == "occupied" else "全部态"
            print(f"[INFO] {orbital.upper()}带中心 ({method_name}){spin_tag}: {band_center:.4f} eV")
        return {"band_center": band_center, "integral": denominator, "method": method}

    # ----------------------------------------------------------
    # 峰位置搜索
    # ----------------------------------------------------------
    def find_peaks(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        min_height: float = DEFAULT_PEAK_MIN_HEIGHT,
        min_distance: float = DEFAULT_PEAK_MIN_DISTANCE,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        搜索态密度图中的峰位置。

        使用 scipy.signal.find_peaks 进行一维峰检测。

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        min_height : float
            最小峰高阈值。
        min_distance : float
            峰之间的最小能量间距 (eV)。
        emin, emax : float, 可选
            搜索能量范围。
        verbose : bool
            是否打印峰列表。

        返回
        ----
        peaks : list of dict
            按峰高降序排列，每项 {"energy": float, "dos": float, "prominence": float}。
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        prepared = self._prepare_data(orbital, spin, emin, emax)
        if prepared is None:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return []

        e_range, d_range = prepared
        if len(e_range) < 3:
            return []

        avg_step = np.mean(np.diff(e_range)) if len(e_range) > 1 else 0.01
        distance_points = max(1, int(min_distance / avg_step))

        peak_indices, peak_props = scipy_find_peaks(
            d_range, height=min_height, distance=distance_points,
        )

        peak_list = []
        for idx in peak_indices:
            peak_list.append({
                "energy": float(e_range[idx]),
                "dos": float(d_range[idx]),
                "prominence": float(
                    peak_props.get("prominences", [0])[0]
                    if "prominences" in peak_props else 0.0
                ),
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
        min_height: float = DEFAULT_PEAK_MIN_HEIGHT,
        min_distance: float = DEFAULT_PEAK_MIN_DISTANCE,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        计算晶场劈裂能（主峰之间的能量差）。

        取最高两个峰的间距 ΔE = E_high - E_low。

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        peaks : list of dict, 可选
            预先计算的峰列表。
        min_height, min_distance : float
            峰搜索参数（peaks 为 None 时使用）。
        emin, emax : float, 可选
            搜索能量范围。

        返回
        ----
        result : dict
            {"splitting": float or None, "peak_lower": dict, "peak_upper": dict,
             "num_peaks_found": int}
        """
        if peaks is None:
            peaks = self.find_peaks(
                orbital=orbital, spin=spin,
                min_height=min_height, min_distance=min_distance,
                emin=emin, emax=emax,
            )

        spin_tag = f" ({spin})" if spin else ""
        if len(peaks) < 2:
            if verbose:
                print(f"[INFO] {orbital.upper()}轨道{spin_tag} 晶场劈裂: 峰数不足 ({len(peaks)})")
            return {
                "splitting": None,
                "peak_lower": peaks[0] if peaks else None,
                "peak_upper": None,
                "num_peaks_found": len(peaks),
            }

        top_two = sorted(peaks[:2], key=lambda p: p["energy"])
        peak_lower, peak_upper = top_two[0], top_two[1]
        splitting = peak_upper["energy"] - peak_lower["energy"]

        if verbose:
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 晶场劈裂:")
            print(f"[INFO]   低能峰: E={peak_lower['energy']:.4f} eV  "
                  f"高能峰: E={peak_upper['energy']:.4f} eV  "
                  f"ΔE={splitting:.4f} eV")

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
        计算轨道带宽（以 DOS 为权重的标准差 σ 和半高宽 FWHM）。

        指标:
          - std (σ):   sqrt(∫(E-ε_center)²·DOS dE / ∫DOS dE)
          - FWHM:      2.355 × σ（高斯近似）
          - energy_span: DOS > 0 的能量跨度

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        emin, emax : float, 可选
            分析能量范围。

        返回
        ----
        result : dict
            {"std": float, "fwhm": float, "energy_span": float}
        """
        prepared = self._prepare_data(orbital, spin, emin, emax)
        if prepared is None:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return {"std": float("nan"), "fwhm": float("nan"), "energy_span": float("nan")}

        e_range, d_range = prepared
        total_dos = np.trapz(d_range, e_range)
        if total_dos == 0:
            return {"std": 0.0, "fwhm": 0.0, "energy_span": 0.0}

        band_center = np.trapz(e_range * d_range, e_range) / total_dos
        variance = np.trapz((e_range - band_center) ** 2 * d_range, e_range) / total_dos
        std = float(np.sqrt(variance))
        fwhm = GAUSSIAN_FWHM_FACTOR * std

        # 能量跨度
        nonzero_mask = d_range > NOISE_THRESHOLD_RATIO * d_range.max()
        energy_span = float(e_range[nonzero_mask].max() - e_range[nonzero_mask].min()) if nonzero_mask.any() else 0.0

        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 带宽: σ={std:.4f} eV, "
                  f"FWHM≈{fwhm:.4f} eV, 跨度={energy_span:.4f} eV")
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
        计算自旋劈裂能 ΔE_spin = ε_d(α) - ε_d(β)。

        参数
        ----
        orbital : str
            轨道名称。
        emin, emax : float, 可选
            积分能量范围。
        method : str
            "all" 或 "occupied"。
        verbose : bool
            是否打印结果。

        返回
        ----
        result : dict
            {"spin_splitting": float, "band_center_alpha": float, "band_center_beta": float}
        """
        if not self.parser.has_spin:
            print("[INFO] 自旋劈裂: 非自旋极化文件，跳过。")
            return {"spin_splitting": float("nan"),
                    "band_center_alpha": float("nan"), "band_center_beta": float("nan")}

        bc_alpha = self.calc_band_center(orbital=orbital, spin="alpha",
                                          emin=emin, emax=emax, method=method, verbose=False)
        bc_beta  = self.calc_band_center(orbital=orbital, spin="beta",
                                          emin=emin, emax=emax, method=method, verbose=False)

        e_alpha = bc_alpha["band_center"]
        e_beta  = bc_beta["band_center"]
        splitting = e_alpha - e_beta if not (np.isnan(e_alpha) or np.isnan(e_beta)) else float("nan")

        if verbose:
            method_name = "占据态" if method == "occupied" else "全部态"
            print(f"[INFO] {orbital.upper()}轨道 自旋劈裂 ({method_name}):")
            print(f"[INFO]   α: {e_alpha:.4f} eV  β: {e_beta:.4f} eV  ΔE={splitting:.4f} eV")
        return {"spin_splitting": splitting,
                "band_center_alpha": e_alpha, "band_center_beta": e_beta}

    # ----------------------------------------------------------
    # 峰面积 / 占据态计算
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

        参数
        ----
        orbital : str
            轨道名称。
        spin : str, 可选
            自旋方向。
        emin, emax : float
            积分能量上下限 (eV)。

        返回
        ----
        area : float
            积分面积（单位: states）。
        """
        prepared = self._prepare_data(orbital, spin, emin, emax)
        if prepared is None:
            print(f"[WARN] 未找到轨道 '{orbital}' 的数据。")
            return 0.0

        e_range, d_range = prepared
        area = float(np.trapz(d_range, e_range))
        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} [{emin:.2f}, {emax:.2f}] eV 积分: {area:.4f}")
        return area

    def calc_occupancy(
        self,
        orbital: str = "d",
        spin: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        计算费米能级以下的占据态电子数。

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
        """
        # 占据态
        prepared_occ = self._prepare_data(orbital, spin, extra_mask=None)
        if prepared_occ is None:
            return {"occupied": float("nan"), "total": float("nan"), "occupancy_ratio": float("nan")}

        e_range, d_range = prepared_occ
        total = float(np.trapz(d_range, e_range))

        mask_occ = e_range < 0
        occupied = float(np.trapz(d_range[mask_occ], e_range[mask_occ])) if mask_occ.sum() >= 2 else 0.0
        ratio = occupied / total if total > 0 else float("nan")

        if verbose:
            spin_tag = f" ({spin})" if spin else ""
            print(f"[INFO] {orbital.upper()}轨道{spin_tag} 占据态: {occupied:.4f} / {total:.4f} = {ratio:.2%}")
        return {"occupied": occupied, "total": total, "occupancy_ratio": ratio}

    # ----------------------------------------------------------
    # 手动识峰分析
    # ----------------------------------------------------------
    def calc_manual_peak(
        self,
        orbital: str,
        emin: float,
        emax: float,
        y1: float,
        y2: float,
        spin: Optional[str] = None,
        subtract_baseline: bool = True,
    ) -> Dict[str, float]:
        """
        对手动选定的两个端点区间进行峰分析。

        参数
        ----
        orbital : str
            目标轨道名。
        emin, emax : float
            区间左右端点能量。
        y1, y2 : float
            左右端点 DOS 值（用于构建基线）。
        spin : str, 可选
            自旋方向。
        subtract_baseline : bool
            是否扣除线性基线。

        返回
        ----
        result : dict
            peak_area, peak_center, peak_max_e, peak_max_dos,
            fwhm, tailing_factor, half_max 等。
        """
        prepared = self._prepare_data(orbital, spin)
        if prepared is None:
            return {"peak_area": float("nan"), "fwhm": float("nan"), "tailing_factor": float("nan")}

        energies, dos_values = prepared
        mask = (energies >= emin) & (energies <= emax)
        e_range = energies[mask]
        d_range = dos_values[mask].copy()

        if len(e_range) < 3:
            return {"peak_area": float("nan"), "fwhm": float("nan"), "tailing_factor": float("nan")}

        baseline = None
        if subtract_baseline:
            slope = (y2 - y1) / (emax - emin) if emax != emin else 0.0
            baseline = y1 + slope * (e_range - emin)
            d_range_sub = d_range - baseline
        else:
            d_range_sub = d_range

        peak_area = float(np.trapz(d_range_sub, e_range))

        total_dos = np.trapz(d_range_sub, e_range)
        peak_center = float(np.trapz(e_range * d_range_sub, e_range) / total_dos) if total_dos > 0 else float("nan")

        idx_max = int(np.argmax(d_range_sub))
        peak_max_e = float(e_range[idx_max])
        peak_max_dos = float(d_range_sub[idx_max])

        if peak_max_dos <= 0:
            return {"peak_area": peak_area, "peak_center": peak_center,
                    "peak_max_e": peak_max_e, "peak_max_dos": peak_max_dos,
                    "fwhm": float("nan"), "half_max": float("nan"),
                    "tailing_factor": float("nan"), "baseline": baseline}

        half_max = peak_max_dos / 2.0

        # 左半高穿越点
        e_left = None
        for i in range(idx_max, 0, -1):
            if d_range_sub[i] >= half_max >= d_range_sub[i - 1]:
                frac = (half_max - d_range_sub[i]) / (d_range_sub[i - 1] - d_range_sub[i] + EPSILON)
                e_left = e_range[i] + frac * (e_range[i - 1] - e_range[i])
                break

        # 右半高穿越点
        e_right = None
        for i in range(idx_max, len(e_range) - 1):
            if d_range_sub[i] >= half_max >= d_range_sub[i + 1]:
                frac = (half_max - d_range_sub[i]) / (d_range_sub[i + 1] - d_range_sub[i] + EPSILON)
                e_right = e_range[i] + frac * (e_range[i + 1] - e_range[i])
                break

        fwhm = e_right - e_left if (e_left is not None and e_right is not None) else float("nan")
        if e_left is not None:
            left_hw = peak_max_e - e_left
            right_hw = e_right - peak_max_e if e_right is not None else float("nan")
            tailing = right_hw / left_hw if left_hw > EPSILON else float("nan")
        else:
            tailing = float("nan")

        return {"peak_area": peak_area, "peak_center": peak_center,
                "peak_max_e": peak_max_e, "peak_max_dos": peak_max_dos,
                "fwhm": fwhm, "half_max": half_max, "tailing_factor": tailing,
                "baseline": baseline}

    # ----------------------------------------------------------
    # 轨道重叠面积计算
    # ----------------------------------------------------------
    def calc_overlap_area(
        self,
        orbital_a: str,
        orbital_b: str,
        spin_a: Optional[str] = None,
        spin_b: Optional[str] = None,
        parser_a=None,
        parser_b=None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        计算两个轨道态密度在指定能量区间内的重叠面积。

        overlap = ∫ min(DOS_A(E), DOS_B(E)) dE

        参数
        ----
        orbital_a, orbital_b : str
            两个轨道的名称。
        spin_a, spin_b : str, 可选
            两个轨道的自旋选择。
        parser_a, parser_b : PDOSParser, 可选
            数据源，默认使用 self.parser（支持跨文件比较）。
        emin, emax : float, 可选
            积分能量范围。
        normalize : bool
            True → 归一化重叠指数 S ∈ [0,1]。
        verbose : bool
            是否打印结果。

        返回
        ----
        result : dict
            {"overlap": float, "area_a": float, "area_b": float,
             "overlap_raw": float, "energy_range": (float, float)}
        """
        pa = parser_a if parser_a is not None else self.parser
        pb = parser_b if parser_b is not None else self.parser

        data_a = pa.get_data(orbitals=[orbital_a], spin=spin_a)
        data_b = pb.get_data(orbitals=[orbital_b], spin=spin_b)

        series_a = _get_first_series(data_a)
        series_b = _get_first_series(data_b)
        if series_a is None or series_b is None:
            print("[WARN] 无法获取两个系列的数据。")
            return {"overlap": float("nan"), "area_a": float("nan"),
                    "area_b": float("nan"), "overlap_raw": float("nan")}

        _, e_a, d_a_raw = series_a
        _, e_b, d_b_raw = series_b
        d_a = np.abs(d_a_raw)
        d_b = np.abs(d_b_raw)

        if emin is None:
            emin = max(e_a.min(), e_b.min())
        if emax is None:
            emax = min(e_a.max(), e_b.max())

        # 公共均匀能量网格
        step_a = (e_a[-1] - e_a[0]) / max(len(e_a) - 1, 1)
        step_b = (e_b[-1] - e_b[0]) / max(len(e_b) - 1, 1)
        step = min(step_a, step_b)
        n_points = max(int(np.ceil((emax - emin) / step)) + 1, 2)
        common_e = np.linspace(emin, emax, n_points)

        d_a_interp = np.interp(common_e, e_a, d_a)
        d_b_interp = np.interp(common_e, e_b, d_b)

        overlap_raw = float(np.trapz(np.minimum(d_a_interp, d_b_interp), common_e))
        area_a = float(np.trapz(d_a_interp, common_e))
        area_b = float(np.trapz(d_b_interp, common_e))

        overlap = overlap_raw / np.sqrt(area_a * area_b) if (normalize and area_a > 0 and area_b > 0) else overlap_raw

        if verbose:
            print(f"[INFO] 轨道重叠: {orbital_a} vs {orbital_b} "
                  f"[{emin:.4f}, {emax:.4f}] eV → overlap={overlap:.4f}")

        return {"overlap": overlap, "area_a": area_a, "area_b": area_b,
                "overlap_raw": overlap_raw, "energy_range": (emin, emax)}

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

        包含: 带中心、带宽、自旋劈裂、晶场劈裂、占据态、积分面积、峰列表。

        参数
        ----
        orbital : str
            要分析的轨道名称。
        spin : str, 可选
            自旋方向。
        emin, emax : float, 可选
            分析能量范围。
        peak_emin, peak_emax : float, 可选
            积分面积的能量范围。
        options : dict, 可选
            控制各分析项开关，如 {"band_center": True, "peaks": False}。
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

        # 打印报告标题
        self._print_report_header(orbital)

        report: Dict = {"orbital": orbital, "spin": spin}
        show_dual_spin = self.parser.has_spin and spin is None

        # 1. 带中心
        if _opt("band_center"):
            self._report_band_center(report, orbital, spin, emin, emax, show_dual_spin)
        else:
            for k in ["band_center_all", "band_center_occupied",
                      "band_center_integral_all", "band_center_integral_occupied"]:
                report[k] = float("nan")

        # 2. 带宽
        if _opt("band_width"):
            self._report_band_width(report, orbital, spin, emin, emax)
        else:
            for k in ["band_width_std", "band_width_fwhm", "band_energy_span"]:
                report[k] = float("nan")

        # 3. 劈裂分析
        need_peaks = _opt("crystal_field") or _opt("peaks")
        need_spin = _opt("spin_splitting") and self.parser.has_spin

        if need_spin or _opt("crystal_field"):
            print(f"── 劈裂分析 ──")

        # 3a. 自旋劈裂
        if need_spin:
            self._report_spin_splitting(report, orbital, emin, emax)
        else:
            for k in ["spin_splitting", "spin_splitting_alpha", "spin_splitting_beta"]:
                report[k] = float("nan")

        # 3b. 晶场劈裂
        peaks = self.find_peaks(orbital=orbital, spin=spin, emin=emin, emax=emax, verbose=False) if need_peaks else []
        report["main_peak"] = peaks[0] if peaks else None
        report["all_peaks"] = peaks

        if _opt("crystal_field"):
            self._report_crystal_field(report, orbital, spin, peaks)
        else:
            report["crystal_field_splitting"] = None
            report["cfs_peaks"] = (None, None)

        # 4. 占据态
        if _opt("occupancy"):
            self._report_occupancy(report, orbital, spin)
        else:
            report["occupied"] = float("nan")
            report["occupancy_ratio"] = float("nan")

        # 5. 积分面积
        if _opt("peak_area"):
            self._report_peak_area(report, orbital, spin, peak_emin, peak_emax)
        else:
            report["total_area"] = float("nan")

        # 6. 峰列表
        if _opt("peaks") and peaks:
            print(f"── 峰列表 ──")
            for i, p in enumerate(peaks):
                print(f"  #{i+1:<2}  E = {p['energy']:+9.4f} eV   DOS = {p['dos']:10.2f}")

        print(f"╚{'═' * REPORT_WIDTH}╝\n")
        return report

    # ----------------------------------------------------------
    # full_report 子方法
    # ----------------------------------------------------------
    @staticmethod
    def _print_report_header(orbital: str) -> None:
        """打印报告标题框。"""
        title = f"  {orbital.upper()}轨道 态密度分析报告"

        def _display_width(s: str) -> int:
            return sum(2 if ord(c) > 127 else 1 for c in s)

        title_width = _display_width(title)
        padding_total = REPORT_WIDTH - title_width
        padding_left = padding_total // 2
        content = f"║{' ' * padding_left}{title}"
        actual_w = _display_width(content) - 1
        extra = REPORT_WIDTH - actual_w - 1

        print(f"\n╔{'═' * REPORT_WIDTH}╗")
        print(f"{content}{' ' * extra}║")
        print(f"╚{'═' * REPORT_WIDTH}╝")

    def _report_band_center(self, report: Dict, orbital: str, spin: Optional[str],
                            emin: Optional[float], emax: Optional[float],
                            show_dual_spin: bool) -> None:
        """报告带中心。"""
        bc_all = self.calc_band_center(orbital=orbital, spin=spin, emin=emin, emax=emax,
                                        method="all", verbose=False)
        bc_occ = self.calc_band_center(orbital=orbital, spin=spin, emin=emin, emax=emax,
                                        method="occupied", verbose=False)
        report["band_center_all"] = bc_all["band_center"]
        report["band_center_integral_all"] = bc_all["integral"]
        report["band_center_occupied"] = bc_occ["band_center"]
        report["band_center_integral_occupied"] = bc_occ["integral"]

        print(f"── 带中心 (band center) ──")
        if show_dual_spin:
            bc_all_a = self.calc_band_center(orbital=orbital, spin="alpha", emin=emin, emax=emax,
                                              method="all", verbose=False)
            bc_all_b = self.calc_band_center(orbital=orbital, spin="beta", emin=emin, emax=emax,
                                              method="all", verbose=False)
            bc_occ_a = self.calc_band_center(orbital=orbital, spin="alpha", emin=emin, emax=emax,
                                              method="occupied", verbose=False)
            bc_occ_b = self.calc_band_center(orbital=orbital, spin="beta", emin=emin, emax=emax,
                                              method="occupied", verbose=False)
            d_all = bc_all_a["band_center"] - bc_all_b["band_center"]
            d_occ = bc_occ_a["band_center"] - bc_occ_b["band_center"]
            print(f"  全部态  ε = {bc_all['band_center']:+.4f} eV")
            print(f"         α: {bc_all_a['band_center']:+.4f} eV  "
                  f"β: {bc_all_b['band_center']:+.4f} eV  Δ(α−β)={d_all:+.4f} eV")
            print(f"  占据态  ε = {bc_occ['band_center']:+.4f} eV")
            print(f"         α: {bc_occ_a['band_center']:+.4f} eV  "
                  f"β: {bc_occ_b['band_center']:+.4f} eV  Δ(α−β)={d_occ:+.4f} eV")
        else:
            print(f"  全部态  ε = {bc_all['band_center']:+.4f} eV  [积分: {bc_all['integral']:.2f}]")
            print(f"  占据态  ε = {bc_occ['band_center']:+.4f} eV  [积分: {bc_occ['integral']:.2f}]")

    def _report_band_width(self, report: Dict, orbital: str, spin: Optional[str],
                           emin: Optional[float], emax: Optional[float]) -> None:
        """报告带宽。"""
        bw = self.calc_band_width(orbital=orbital, spin=spin, emin=emin, emax=emax, verbose=False)
        report["band_width_std"] = bw["std"]
        report["band_width_fwhm"] = bw["fwhm"]
        report["band_energy_span"] = bw["energy_span"]
        print(f"── 带宽 ──")
        print(f"  σ={bw['std']:.4f} eV  │  FWHM={bw['fwhm']:.4f} eV  │  跨度={bw['energy_span']:.4f} eV")

    def _report_spin_splitting(self, report: Dict, orbital: str,
                                emin: Optional[float], emax: Optional[float]) -> None:
        """报告自旋劈裂。"""
        ss = self.calc_spin_splitting(orbital=orbital, emin=emin, emax=emax, method="occupied", verbose=False)
        report["spin_splitting"] = ss["spin_splitting"]
        report["spin_splitting_alpha"] = ss["band_center_alpha"]
        report["spin_splitting_beta"] = ss["band_center_beta"]
        spl = ss["spin_splitting"]
        if not np.isnan(spl):
            direction = "α 更深" if spl < 0 else "β 更深"
            print(f"  自旋劈裂 (占据态): ΔE_spin = {spl:+.4f} eV ({direction})")
        else:
            print(f"  自旋劈裂: 无法计算")

    def _report_crystal_field(self, report: Dict, orbital: str, spin: Optional[str],
                               peaks: List[Dict]) -> None:
        """报告晶场劈裂。"""
        cfs = self.calc_crystal_field_splitting(orbital=orbital, spin=spin, peaks=peaks if peaks else None,
                                                  verbose=False)
        report["crystal_field_splitting"] = cfs["splitting"]
        report["cfs_peaks"] = (cfs["peak_lower"], cfs["peak_upper"])
        if cfs["splitting"] is not None:
            lo, hi = cfs["peak_lower"], cfs["peak_upper"]
            print(f"  晶场劈裂: ΔE_cf = {cfs['splitting']:.4f} eV "
                  f"(峰@{lo['energy']:.3f} → 峰@{hi['energy']:.3f} eV)")
        else:
            print(f"  晶场劈裂: 峰数不足 ({len(peaks)})，无法计算")

    def _report_occupancy(self, report: Dict, orbital: str, spin: Optional[str]) -> None:
        """报告占据态。"""
        occ = self.calc_occupancy(orbital=orbital, spin=spin, verbose=False)
        report["occupied"] = occ["occupied"]
        report["occupancy_ratio"] = occ["occupancy_ratio"]
        print(f"── 占据态 ──")
        print(f"  占据: {occ['occupied']:.2f} / {occ['total']:.2f} = {occ['occupancy_ratio']:.2%}")

    def _report_peak_area(self, report: Dict, orbital: str, spin: Optional[str],
                           peak_emin: Optional[float], peak_emax: Optional[float]) -> None:
        """报告积分面积。"""
        e_lo = peak_emin if peak_emin is not None else -100.0
        e_hi = peak_emax if peak_emax is not None else 100.0
        total_area = self.calc_peak_area(orbital=orbital, spin=spin, emin=e_lo, emax=e_hi, verbose=False)
        report["total_area"] = total_area
        print(f"── 积分面积 ──")
        print(f"  区间 [{e_lo:.2f}, {e_hi:.2f}] eV: {total_area:.2f}")
