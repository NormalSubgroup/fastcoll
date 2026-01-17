# md5fastcoll Python

基于 Marc Stevens Fast Collision Attack on MD5 的可复现实现，强调一步一验证。

## 目标

- 生成与 HashClash md5_fastcoll 一致的两块碰撞文件对
- 提供纯 Python 引擎，不需要 make
- 提供可验证的中间状态，便于调试与学习

## 依赖

- Python 3.11 或更高
- 纯 Python 引擎依赖
  - pip install -U numpy numba

## 引擎说明

fastcoll 子命令支持三种引擎

- python
  - 只使用 Python 与 pip 安装的依赖
  - 不使用 tools/md5_fastcoll_lib_wrap.cpp
  - 同一 IHV seed prefixfile 下输出与 HashClash md5_fastcoll 一致
  - 首次运行会触发 Numba 编译，后续会使用磁盘缓存
- ctypes
  - 通过 ctypes 调用共享库 md5_fastcoll_lib
  - build-native-lib 会使用 tools/md5_fastcoll_lib_wrap.cpp 并编译 HashClash 源码
  - 需要本机有 C++ 编译器
- native
  - 直接运行 HashClash 可执行文件 md5_fastcoll
  - build-native 会拉取 HashClash 源码并编译
  - 需要本机有 make 与 autotools 工具链

## 快速开始

- 校验核心 MD5 与 hashlib 一致
  - python -m md5fastcoll.cli verify-core
- 生成 fastcoll 兼容碰撞文件对
  - python -m md5fastcoll.cli fastcoll --engine python --seed 123 -q -o msg1.bin msg2.bin
- 带前缀生成
  - python -m md5fastcoll.cli fastcoll --engine python --prefixfile prefix.bin -q -o msg1.bin msg2.bin

## 搜索命令

- Block1 搜索
  - python -m md5fastcoll.cli search-block1 --restarts 50
- Block2 搜索
  - python -m md5fastcoll.cli search-block2 --restarts 500 --ihv 0xC4DA537C 0x1051DD8E 0x42867DB3 0x0D67B366
  - 该命令是概率搜索，restarts 50 常见会失败
- 固定随机种子复现
  - python -m md5fastcoll.cli search-block2 --seed 42 --restarts 500 --ihv 0xC4DA537C 0x1051DD8E 0x42867DB3 0x0D67B366

## 脚本

- 准确性检查
  - python scripts/accuracy.py
- 性能与吞吐基准
  - python scripts/perf.py --trials 20 --budget 4096
- 论文一致性检查
  - python scripts/test_paper_compliance.py --skip-search

## 说明

- fastcoll 前缀按 64 字节块零填充，与 HashClash md5_fastcoll 行为一致
- seed 为 0 时 HashClash 的 rng64 可能进入全零退化状态，md5fastcoll 会自动将 seed2 置为 0x12345678
- `tables/block1.txt` 与 `tables/block2.txt` 为论文附录 A 条件表的转录
- `I/J/K` 表示同一未知位，K 与 I 相同

## 来源

- Marc Stevens Fast Collision Attack on MD5
  - 本仓库包含 fastcoll.pdf
- HashClash 源码仓库
  - https://github.com/cr-marcstevens/hashclash
- Python fastcoll 引擎主要移植自 HashClash 的 src/md5fastcoll 目录
  - md5.cpp
  - block0.cpp
  - block1wang.cpp
  - block1stevens00.cpp
  - block1stevens01.cpp
  - block1stevens10.cpp
  - block1stevens11.cpp
- tools/md5_fastcoll_lib_wrap.cpp 用于 ctypes 引擎
  - build-native-lib 会编译该 wrapper 并链接 HashClash 源码以导出 md5fastcoll_find_blocks
