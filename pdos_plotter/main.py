#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDOS 态密度工具 —— 直接运行入口

支持:
  python main.py                   # GUI 模式
  python main.py --no-gui -f ...   # CLI 模式
  python main.py --mode castep ... # CASTEP 二进制模式
"""
import sys
import os

# 将当前目录从 sys.path 中移除（避免覆盖包导入），
# 并将父目录添加到 sys.path，使 pdos_plotter 可作为包导入
_curdir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_curdir)
sys.path = [p for p in sys.path if os.path.abspath(p) != _curdir]
if _parent not in sys.path:
    sys.path.insert(0, _parent)
# 清除可能已被缓存的无效 pdos_plotter 条目
for key in list(sys.modules.keys()):
    if key == 'pdos_plotter' or key.startswith('pdos_plotter.'):
        del sys.modules[key]

from pdos_plotter.cli import main

if __name__ == "__main__":
    main()
