#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fortran Unformatted Sequential Big-Endian 二进制文件读取器
==========================================================

用于读取 CASTEP 产生的 .castep_bin 和 .pdos_bin 文件。

文件格式:
  - Fortran Unformatted Sequential 记录结构
  - 每条记录: [int32 BE 记录长度] [数据体] [int32 BE 记录长度]
  - 记录长度 = 数据体的字节数
  - 所有多字节数值均为 Big-Endian 字节序

用法:
    from binary_io import read_all_records, read_record_float64

    records = read_all_records("path/to/file.castep_bin")
    eig_values = read_record_float64(records[100])  # 解析为 >f8 数组

作者: Xin Jinglong (Institute of Theoretical Chemistry, Shandong University)
日期: 2026/06/11
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Optional

import numpy as np


# ============================================================
# 核心函数
# ============================================================

def read_all_records(filepath: Path | str) -> List[bytes]:
    """
    读取 Fortran Unformatted Sequential 文件中的所有记录。

    每条记录的结构为:
        [int32 BE: 记录体长度 N] [N 字节数据体] [int32 BE: 记录体长度 N]

    参数
    ----
    filepath : Path | str
        二进制文件的完整路径。

    返回
    ----
    records : list of bytes
        每条记录的数据体部分（不含头尾的长度标记）。
        若文件格式异常（如记录长度为负数或过大），则提前终止读取。

    异常
    ----
    FileNotFoundError
        当指定文件不存在时抛出。
    ValueError
        当文件为空或无法解析时抛出。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    data: bytes = filepath.read_bytes()
    if len(data) < 8:
        raise ValueError(f"文件过小（{len(data)} 字节），无法解析 Fortran 记录。")

    records: List[bytes] = []
    offset: int = 0
    # 允许的最大记录体长度（10 MB），超过则认为是格式错误
    max_reclen: int = 10_000_000

    while offset < len(data) - 8:
        # 读取头部记录长度（4 字节 Big-Endian int32）
        reclen: int = struct.unpack('>i', data[offset:offset + 4])[0]

        # 有效性检查
        if reclen < 0 or reclen > max_reclen:
            # 可能到达文件末尾填充区或格式损坏，终止读取
            break

        # 提取记录体
        body_end: int = offset + 4 + reclen
        if body_end > len(data):
            # 记录体超出文件范围，文件可能被截断
            break

        records.append(data[offset + 4:body_end])
        offset = body_end + 4  # 跳过尾部长度标记

    return records


# ============================================================
# 记录类型转换辅助函数
# ============================================================

def read_record_float64(record: bytes) -> np.ndarray:
    """
    将 record bytes 解释为 Big-Endian float64 (double precision) 的 numpy 数组。

    参数
    ----
    record : bytes
        单条记录的数据体。

    返回
    ----
    arr : np.ndarray (dtype=float64)
        一维浮点数组。
    """
    return np.frombuffer(record, dtype='>f8')


def read_record_int32(record: bytes) -> np.ndarray:
    """
    将 record bytes 解释为 Big-Endian int32 的 numpy 数组。

    参数
    ----
    record : bytes
        单条记录的数据体。

    返回
    ----
    arr : np.ndarray (dtype=int32)
        一维整数数组。
    """
    return np.frombuffer(record, dtype='>i4')


def read_record_float64_scalar(record: bytes) -> float:
    """
    将单值 float64 record 解析为 Python float。

    参数
    ----
    record : bytes
        单条记录的数据体（应为 8 字节）。

    返回
    ----
    val : float
        解析出的浮点值。
    """
    return float(np.frombuffer(record, dtype='>f8')[0])


def read_record_int32_scalar(record: bytes) -> int:
    """
    将单值 int32 record 解析为 Python int。

    参数
    ----
    record : bytes
        单条记录的数据体（应为 4 字节）。

    返回
    ----
    val : int
        解析出的整数值。
    """
    return int(np.frombuffer(record, dtype='>i4')[0])


# ============================================================
# 文本/标签识别
# ============================================================

def try_decode_ascii(record: bytes) -> Optional[str]:
    """
    尝试将 record bytes 解码为 ASCII 文本。

    用于识别 .castep_bin / .pdos_bin 中的标签记录
    （如 "E_FERMI", "BEGIN_CELL_GLOBAL", "SPECTRAL_KPOINTS" 等）。

    参数
    ----
    record : bytes
        单条记录的数据体。

    返回
    ----
    text : str or None
        解码成功时返回去除首尾空白的字符串，失败时返回 None。
    """
    try:
        return record.decode('ascii').strip()
    except (UnicodeDecodeError, ValueError):
        return None


def find_label_indices(records: List[bytes]) -> dict[str, int]:
    """
    在记录列表中搜索所有可解码为 ASCII 文本的记录，
    返回 {标签文本: 记录索引} 的字典。

    用于快速定位关键标签（如 SPECTRAL_KPOINTS, E_FERMI 等）。

    参数
    ----
    records : list of bytes
        read_all_records() 的返回结果。

    返回
    ----
    index_map : dict[str, int]
        键为标签文本（去除首尾空白），值为记录在列表中的索引。
        若同一标签出现多次，保留最后一次出现的索引。
    """
    idx: dict[str, int] = {}
    for i, rec in enumerate(records):
        text = try_decode_ascii(rec)
        if text:
            idx[text.strip()] = i
    return idx
