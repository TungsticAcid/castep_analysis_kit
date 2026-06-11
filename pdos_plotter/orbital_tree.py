#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OrbitalTree —— 可展开的层级轨道选择树
======================================

基于 ttk.Treeview，支持:
  - Element → Atom → Angular Momentum → Orbital 四级层级
  - 每行点击切换 ☑/☐ 状态，父节点自动汇总
  - 输出选中原子的轨道列表

用法:
    tree = OrbitalTree(parent, orbital_map, species_list)
    tree.pack(fill=tk.BOTH, expand=True)
    selected = tree.get_selected()  # {species: {ion: [orbitals]}}

作者: Xin Jinglong
日期: 2026/06/11
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional

try:
    from .constants import ORBITAL_TO_L, L_TO_NAME, SPECIES_ORBITALS, ORBITAL_NAMES
except ImportError:
    from constants import ORBITAL_TO_L, L_TO_NAME, SPECIES_ORBITALS, ORBITAL_NAMES


class OrbitalTree(ttk.Frame):
    """可展开的轨道选择树，每行带 checkbox 状态。"""

    CHECKED = "☑"
    UNCHECKED = "☐"
    PARTIAL = "☒"

    def __init__(self, parent, orbital_map: Optional[List[dict]] = None,
                 species_list: Optional[List[str]] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self._orbital_map = orbital_map or []
        self._species_list = species_list or []
        self._states: Dict[str, str] = {}  # item_iid → "checked"/"unchecked"/"partial"
        self._children: Dict[str, List[str]] = {}  # parent_iid → [child_iids]
        self._parent: Dict[str, str] = {}  # child_iid → parent_iid
        self._item_data: Dict[str, dict] = {}  # iid → {type, species, ion, l, orbital}

        self._build_ui()
        if orbital_map:
            self.populate(orbital_map, species_list)

    def _build_ui(self) -> None:
        """构建 Treeview 和滚动条。"""
        self.tree = ttk.Treeview(self, columns=(), show="tree", height=20)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.bind("<Button-1>", self._on_click)

    # ----------------------------------------------------------
    # 数据填充
    # ----------------------------------------------------------
    def populate(self, orbital_map: List[dict], species_list: List[str]) -> None:
        """根据轨道映射表填充树结构（幂等：相同数据不重复填充）。"""
        # 如果数据完全相同，跳过重建
        if (self._orbital_map is orbital_map and self._species_list == species_list
                and len(self._item_data) > 0):
            return
        self._orbital_map = orbital_map
        self._species_list = species_list
        self._clear()

        # 构建物种 → 离子 → {l → [orbitals]} 嵌套结构
        tree_data: Dict[str, Dict[int, Dict[int, List[str]]]] = {}
        for orb in orbital_map:
            sp = orb["species"]
            ion = orb["ion_index"]
            l_val = orb["angular_momentum"]
            orb_name = orb["orbital_name"]

            if sp not in tree_data:
                tree_data[sp] = {}
            if ion not in tree_data[sp]:
                tree_data[sp][ion] = {}
            if l_val not in tree_data[sp][ion]:
                tree_data[sp][ion][l_val] = []
            if orb_name not in tree_data[sp][ion][l_val]:
                tree_data[sp][ion][l_val].append(orb_name)

        # 按 species_list 顺序填充树
        for sp in species_list:
            if sp not in tree_data:
                continue
            sp_iid = f"s_{sp}"
            self.tree.insert("", tk.END, iid=sp_iid, text=f"{self.UNCHECKED} {sp}", open=True)
            self._states[sp_iid] = "unchecked"
            self._children[sp_iid] = []
            self._item_data[sp_iid] = {"type": "species", "species": sp}

            for ion in sorted(tree_data[sp].keys()):
                ion_iid = f"a_{sp}_{ion}"
                self.tree.insert(sp_iid, tk.END, iid=ion_iid,
                                 text=f"{self.UNCHECKED} {sp} #{ion}", open=False)
                self._states[ion_iid] = "unchecked"
                self._children[ion_iid] = []
                self._parent[ion_iid] = sp_iid
                self._children[sp_iid].append(ion_iid)
                self._item_data[ion_iid] = {"type": "atom", "species": sp, "ion": ion}

                for l_val in sorted(tree_data[sp][ion].keys()):
                    l_name = L_TO_NAME.get(l_val, f"l{l_val}")
                    l_iid = f"l_{sp}_{ion}_{l_val}"
                    self.tree.insert(ion_iid, tk.END, iid=l_iid,
                                     text=f"{self.UNCHECKED} {l_name}", open=False)
                    self._states[l_iid] = "unchecked"
                    self._children[l_iid] = []
                    self._parent[l_iid] = ion_iid
                    self._children[ion_iid].append(l_iid)
                    self._item_data[l_iid] = {"type": "l", "species": sp, "ion": ion, "l": l_val}

                    for orb_name in tree_data[sp][ion][l_val]:
                        orb_iid = f"o_{sp}_{ion}_{orb_name}"
                        self.tree.insert(l_iid, tk.END, iid=orb_iid,
                                         text=f"{self.UNCHECKED} {orb_name}", open=False)
                        self._states[orb_iid] = "unchecked"
                        self._parent[orb_iid] = l_iid
                        self._children[l_iid].append(orb_iid)
                        self._item_data[orb_iid] = {"type": "orbital", "species": sp, "ion": ion, "orbital": orb_name}

        # 展开前几个元素
        for sp in species_list[:2]:
            sp_iid = f"s_{sp}"
            if self.tree.exists(sp_iid):
                self.tree.item(sp_iid, open=True)

    def _clear(self) -> None:
        """清空树。"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._states.clear()
        self._children.clear()
        self._parent.clear()
        self._item_data.clear()

    # ----------------------------------------------------------
    # 交互
    # ----------------------------------------------------------
    def _on_click(self, event: tk.Event) -> None:
        """点击树节点切换 checkbox 状态。"""
        region = self.tree.identify_region(event.x, event.y)
        if region != "tree":
            return
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        # 检测是否点击在 checkbox 区域（前 30 像素）
        if event.x > 30:
            return
        self._toggle(iid)

    def _toggle(self, iid: str) -> None:
        """切换节点状态，向下传播到后代，向上重算祖先。"""
        current = self._states.get(iid, "unchecked")
        new_state = "checked" if current != "checked" else "unchecked"
        self._set_descendants(iid, new_state)
        # 自底向上全部重算（比 _update_ancestors 更鲁棒）
        self._recompute_all_parents()
        self._refresh_display()

    def _set_descendants(self, iid: str, state: str) -> None:
        """递归设置所有后代节点的状态。"""
        self._states[iid] = state
        for child in self._children.get(iid, []):
            self._set_descendants(child, state)

    def _update_ancestors(self, iid: str) -> None:
        """向上传播更新祖先节点状态。"""
        parent = self._parent.get(iid)
        if parent is None:
            return
        child_states = [self._states[c] for c in self._children.get(parent, [])]
        if all(s == "checked" for s in child_states):
            self._states[parent] = "checked"
        elif any(s in ("checked", "partial") for s in child_states):
            self._states[parent] = "partial"
        else:
            self._states[parent] = "unchecked"
        self._update_ancestors(parent)

    def _refresh_display(self) -> None:
        """刷新所有树节点的显示文字。"""
        for iid, state in self._states.items():
            if not self.tree.exists(iid):
                continue
            prefix = {"checked": self.CHECKED, "unchecked": self.UNCHECKED, "partial": self.PARTIAL}[state]
            text = self.tree.item(iid, "text")
            # 去掉旧前缀（第一个字符 + 空格），确保不累积
            ch0 = text[0] if text else ""
            if ch0 in (self.CHECKED, self.UNCHECKED, self.PARTIAL):
                rest = text[2:]  # 跳过 "☑ " 两个字符
            else:
                rest = text
            self.tree.item(iid, text=f"{prefix} {rest}")

    # ----------------------------------------------------------
    # 数据提取
    # ----------------------------------------------------------
    def get_selected(self) -> Dict[str, Dict[int, List[str]]]:
        """
        返回用户选中的轨道结构。

        返回格式: {species: {ion_index: [orbital_names]}}
        例如: {"Ni": {1: ["d_xy", "d_yz"], 3: ["s"]}, "C": {65: ["pz"]}}
        """
        result: Dict[str, Dict[int, List[str]]] = {}
        for iid, data in self._item_data.items():
            if data["type"] == "orbital" and self._states.get(iid) == "checked":
                sp = data["species"]
                ion = data["ion"]
                orb = data["orbital"]
                if sp not in result:
                    result[sp] = {}
                if ion not in result[sp]:
                    result[sp][ion] = []
                result[sp][ion].append(orb)
        return result

    def get_selected_flat(self) -> List[str]:
        """返回选中轨道的扁平标签列表（如 "Ni_1-d_xy"）。"""
        result = []
        for sp, ions in self.get_selected().items():
            for ion, orbs in ions.items():
                for orb in orbs:
                    result.append(f"{sp}_{ion}-{orb}")
        return result

    def _recompute_all_parents(self) -> None:
        """自底向上重新计算所有父节点状态。"""
        # 按深度排序（最深先处理），确保子节点状态已确定再处理父节点
        items_by_depth = sorted(self._item_data.keys(), key=lambda iid: -iid.count('_'))
        for iid in items_by_depth:
            children = self._children.get(iid, [])
            if not children:
                continue
            child_states = [self._states.get(c, "unchecked") for c in children]
            if all(s == "checked" for s in child_states):
                self._states[iid] = "checked"
            elif any(s in ("checked", "partial") for s in child_states):
                self._states[iid] = "partial"
            else:
                self._states[iid] = "unchecked"

    def select_all(self) -> None:
        """全选所有轨道。"""
        for iid, data in self._item_data.items():
            if data["type"] == "orbital":
                self._states[iid] = "checked"
        self._recompute_all_parents()
        self._refresh_display()

    def deselect_all(self) -> None:
        """取消全选。"""
        for iid in self._states:
            self._states[iid] = "unchecked"
        self._refresh_display()
