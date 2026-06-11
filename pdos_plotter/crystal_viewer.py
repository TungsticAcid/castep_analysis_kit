#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
晶体结构查看器 —— 基于 matplotlib 3D 的轻量级晶体可视化
=========================================================

从 .castep_bin CELL 段提取晶格矢量、原子坐标、物种信息，
渲染交互式 3D 球棍模型，标注原子序号。

用法:
    from crystal_viewer import CrystalStructure, CrystalViewer
    struct = CrystalStructure(lattice, positions, species, labels)
    viewer = CrystalViewer(struct)
    viewer.show()

作者: Xin Jinglong
日期: 2026/06/11
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 元素配色
# ============================================================
ELEMENT_COLORS = {
    "H":  "#FFFFFF",  # 白色
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "C":  "#404040",  # 深灰
    "N":  "#3050F8",  # 蓝
    "O":  "#FF0D0D",  # 红
    "F":  "#90E050",
    "Ni": "#50D050",  # 绿
    "Cu": "#C88033",
    "Pd": "#006985",
    "Pt": "#D0D0E0",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Au": "#FFD123",
    "Fe": "#E06633",
    "Co": "#F08080",
    "default": "#808080",
}


# ============================================================
# CrystalStructure 数据类
# ============================================================
@dataclass
class CrystalStructure:
    """晶体结构数据容器。"""
    lattice: np.ndarray          # (3, 3) 晶格矢量 [Å]
    positions: np.ndarray        # (n_atoms, 3) 分数坐标
    species: List[str]           # 每个原子的元素符号 (n_atoms,)
    species_colors: List[str]    # 每个原子的显示颜色
    labels: List[str]            # 每个原子的显示标签
    species_list: List[str]      # 唯一元素列表
    atom_indices: np.ndarray     # 全局原子序号 (1-based)

    @classmethod
    def from_castep_bin(cls, records: List[bytes]) -> "CrystalStructure":
        """从 .castep_bin 记录构建晶体结构。"""
        try:
            from .binary_io import try_decode_ascii, read_record_float64
        except ImportError:
            from binary_io import try_decode_ascii, read_record_float64
        labels_map = {}
        for i, r in enumerate(records):
            t = try_decode_ascii(r)
            if t:
                labels_map[t.strip()] = i

        # 过滤 labels_map：若有多个 CELL 块，仅保留最后一个块内的标签，
        # 避免从不同 CELL 段混用晶格/坐标/物种数据。
        _cell_starts = [i for i, r in enumerate(records)
                        if try_decode_ascii(r) and "BEGIN_BLOCK_CELL" in try_decode_ascii(r)]
        if len(_cell_starts) > 1:
            _last_cell = _cell_starts[-1]
            # 查找最后一个 CELL 块之后的下一个 BEGIN_BLOCK 作为结束标记
            _end_markers = [i for i, r in enumerate(records)
                            if i > _last_cell
                            and try_decode_ascii(r)
                            and try_decode_ascii(r).startswith("BEGIN_BLOCK")]
            _cell_end = _end_markers[0] if _end_markers else len(records)
            _filtered = {}
            for _k, _v in labels_map.items():
                if _last_cell <= _v < _cell_end:
                    _filtered[_k] = _v
            # 保留 CELL 块之前的顶层标签（如 E_FERMI 等）
            for _k, _v in labels_map.items():
                if _v < _cell_starts[0]:
                    _filtered[_k] = _v
            labels_map = _filtered
            print(f"[INFO] 检测到 {len(_cell_starts)} 个 CELL 块，使用最后一个 "
                  f"(索引 #{_last_cell}–#{_cell_end})")

        # 晶格矢量
        lattice = read_record_float64(records[labels_map["CELL%REAL_LATTICE"] + 1])
        lattice = lattice[:9].reshape(3, 3)

        # 原子数（int32，非 float64）
        n_atoms_rec = records[labels_map["CELL%NUM_IONS"] + 1]
        n_atoms = int(np.frombuffer(n_atoms_rec, dtype=">i4")[0])

        # 原子坐标（分数坐标）
        pos_raw = read_record_float64(records[labels_map["CELL%IONIC_POSITIONS"] + 1])
        positions = pos_raw[:n_atoms * 3].reshape(n_atoms, 3)

        # 物种符号 —— 搜索所有包含 SPECIES_SYMBOL 的标签（兼容 CELL% 前缀有无的情况）
        species_text = ""
        for label_text, rec_idx in labels_map.items():
            if "SPECIES_SYMBOL" in label_text:
                rec = records[rec_idx + 1]
                try:
                    species_text = rec.decode("ascii", errors="replace").strip()
                    if species_text:
                        break
                except Exception:
                    pass
        species_names = species_text.split() if species_text else ["El1", "El2", "El3"]
        print(f"[INFO] 晶体结构物种: {species_names}")

        # 每个原子的物种索引 (int32, 1-based)
        spec_rec = records[labels_map["CELL%ION_PACK_SPECIES"] + 1]
        spec_idx = np.frombuffer(spec_rec, dtype=">i4")[:n_atoms]

        # 每个原子的同类序号 (int32, 1-based)
        pack_rec = records[labels_map["CELL%ION_PACK_INDEX"] + 1]
        pack_idx = np.frombuffer(pack_rec, dtype=">i4")[:n_atoms]

        # 验证物种索引范围
        unique_spec = np.unique(spec_idx)
        print(f"[INFO] 物种索引: {unique_spec.tolist()} (共 {len(species_names)} 种元素)")

        # 构建原子标签
        species_list = []
        labels = []
        species_colors = []
        for i in range(n_atoms):
            si = int(spec_idx[i])
            # 1-based → 0-based Python 索引，带边界检查
            if 1 <= si <= len(species_names):
                sp_name = species_names[si - 1]
            else:
                sp_name = "?"
            species_list.append(sp_name)
            labels.append(f"{sp_name}{int(pack_idx[i])}")
            species_colors.append(ELEMENT_COLORS.get(sp_name, ELEMENT_COLORS["default"]))

        return cls(
            lattice=lattice,
            positions=positions,
            species=species_list,
            species_colors=species_colors,
            labels=labels,
            species_list=list(dict.fromkeys(species_list)),
            atom_indices=np.arange(1, n_atoms + 1),
        )


# ============================================================
# CrystalViewer 类
# ============================================================
class CrystalViewer:
    """交互式 3D 晶体结构查看器。"""

    def __init__(self, structure: CrystalStructure, figsize: Tuple[float, float] = (8, 6)):
        self.struct = structure
        self.fig = plt.figure(figsize=figsize)
        self.ax: Axes3D = self.fig.add_subplot(111, projection="3d")
        self._selected_atoms: List[int] = []
        self._atom_artists: List = []
        self._label_artists: List = []

        self._build()

    def _build(self) -> None:
        """绘制晶体结构。"""
        ax = self.ax
        s = self.struct

        # --- 绘制晶胞框架 ---
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ]).T  # (3, 8)
        cart_vertices = vertices.T @ s.lattice  # (8, 3)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            ax.plot3D(
                [cart_vertices[i, 0], cart_vertices[j, 0]],
                [cart_vertices[i, 1], cart_vertices[j, 1]],
                [cart_vertices[i, 2], cart_vertices[j, 2]],
                color="gray", linewidth=0.5, alpha=0.6,
            )

        # --- 绘制原子 ---
        cart_positions = s.positions @ s.lattice  # (n_atoms, 3)
        self._atom_artists = []
        self._label_artists = []

        for i in range(len(s.species)):
            color = s.species_colors[i]
            edge_color = "black" if color in ("#FFFFFF", "#D9FFFF") else color

            # 原子球（缩小标记以避免 slab 模型中重叠）
            artist = ax.scatter(
                cart_positions[i, 0], cart_positions[i, 1], cart_positions[i, 2],
                s=60 if s.species[i] == "Ni" else 45 if s.species[i] == "C" else 35,
                c=color, edgecolors=edge_color, linewidths=0.5, alpha=0.9,
                picker=True,
            )
            self._atom_artists.append(artist)

            # 标签
            lbl = ax.text(
                cart_positions[i, 0], cart_positions[i, 1], cart_positions[i, 2],
                s.labels[i], fontsize=6, ha="center", va="bottom",
                color="blue", weight="bold",
            )
            self._label_artists.append(lbl)

        # --- 调试输出：前 5 个原子的笛卡尔坐标 ---
        n_show = min(5, len(s.species))
        print(f"[INFO] 前 {n_show} 个原子笛卡尔坐标 (Å):")
        for _i in range(n_show):
            print(f"  {s.labels[_i]:>8s}: [{cart_positions[_i, 0]:8.4f}, "
                  f"{cart_positions[_i, 1]:8.4f}, {cart_positions[_i, 2]:8.4f}]")

        # --- 图例 ---
        for sp in s.species_list:
            color = ELEMENT_COLORS.get(sp, ELEMENT_COLORS["default"])
            ax.scatter([], [], [], c=color, s=80, label=sp, edgecolors="gray", linewidths=0.5)
        ax.legend(loc="upper right", fontsize=8, title="Elements")

        # --- 自动缩放 ---
        max_range = np.max(cart_positions.max(axis=0) - cart_positions.min(axis=0))
        mid = cart_positions.mean(axis=0)
        ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
        ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
        ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

        # --- 样式 ---
        ax.set_xlabel("X (A)")
        ax.set_ylabel("Y (A)")
        ax.set_zlabel("Z (A)")
        ax.set_title("Crystal Structure (click atom to select)", fontsize=11)
        # 等比例轴
        ax.set_box_aspect([1, 1, 1])
        self.fig.tight_layout()

        # --- 点击事件 ---
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def _on_pick(self, event) -> None:
        """点击原子时高亮选中。"""
        ind = event.ind[0] if hasattr(event, "ind") and len(event.ind) > 0 else None
        if ind is None:
            return
        if ind in self._selected_atoms:
            self._selected_atoms.remove(ind)
        else:
            self._selected_atoms.append(ind)

        # 更新所有原子的外观
        s = self.struct
        for i in range(len(s.species)):
            if i in self._selected_atoms:
                # 高亮（与 _build 中的基础尺寸按比例放大）
                self._atom_artists[i].set_sizes([120])
                self._atom_artists[i].set_edgecolors(["red"])
                self._atom_artists[i].set_linewidths([2.0])
                self._label_artists[i].set_color("red")
                self._label_artists[i].set_fontweight("bold")
            else:
                color = s.species_colors[i]
                ec = "black" if color in ("#FFFFFF", "#D9FFFF") else color
                base_size = 60 if s.species[i] == "Ni" else 45 if s.species[i] == "C" else 35
                self._atom_artists[i].set_sizes([base_size])
                self._atom_artists[i].set_edgecolors([ec])
                self._atom_artists[i].set_linewidths([0.5])
                self._label_artists[i].set_color("blue")
                self._label_artists[i].set_fontweight("normal")

        selected_labels = [s.labels[i] for i in self._selected_atoms]
        self.ax.set_title(f"Selected: {', '.join(selected_labels)}", fontsize=11, color="red")
        self.fig.canvas.draw_idle()

    def get_selected_atoms(self) -> List[int]:
        """返回选中原子的全局索引列表 (1-based)。"""
        return [i + 1 for i in self._selected_atoms]

    def show(self) -> None:
        """显示晶体查看器窗口。"""
        plt.show()
