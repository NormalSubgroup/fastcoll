# md5fastcoll (Python)

基于 Marc Stevens《Fast Collision Attack on MD5》的可复现实现，强调“一步一验证”。

功能分层：

- 核心：MD5 压缩函数（论文中的 Q、T 记号）+ 可逆求 Wt。
- 条件：对 Q、T 逐位条件表达与验证（支持 '.', '0','1','^','!' 以及符号位 I/J/K）。
- 算法：实现论文算法 6-1、6-2（包含 Q9/Q10 子空间枚举与末端验证）。
- CLI / Scripts：校验、性能测试、论文一致性检查。

使用方法（入门）：

- 运行核心一致性验证：
  - python -m md5fastcoll.cli verify-core
- 演示一步步反推单块 W：
  - python -m md5fastcoll.cli demo-inverse
- Block1/Block2 搜索（算法 6-1/6-2）：
  - python -m md5fastcoll.cli search-block1 --restarts 50
  - python -m md5fastcoll.cli search-block2 --restarts 50 --ihv 0xC4DA537C 0x1051DD8E 0x42867DB3 0x0D67B366
- 两块管线（较慢）：
  - python -m md5fastcoll.cli search-collision --restarts 500
- 生成 fastcoll 兼容碰撞文件（支持前缀）：
  - （无需 make / 纯 Python；安装依赖即可跑到接近原生速度）：
    - pip install -U numpy numba
    - python -m md5fastcoll.cli fastcoll --engine python --seed 123 -q -o msg1.bin msg2.bin
  - python -m md5fastcoll.cli fastcoll --out msg1.bin msg2.bin
  - python -m md5fastcoll.cli fastcoll --prefixfile prefix.bin
  - （推荐，更快）构建并使用原生 `md5_fastcoll`：
    - python -m md5fastcoll.cli build-native
    - python -m md5fastcoll.cli fastcoll --engine native --prefixfile prefix.bin
  - （推荐，更快）构建 HashClash 共享库并通过 ctypes 引擎使用：
    - python -m md5fastcoll.cli build-native-lib
    - python -m md5fastcoll.cli fastcoll --engine ctypes --prefixfile prefix.bin
    - 可用 `MD5_FASTCOLL_LIB` 或 `fastcoll --engine ctypes --native-lib /path/to/md5_fastcoll_lib.{so|dylib}` 指定库位置

脚本：

- 准确性检查：
  - python scripts/accuracy.py
- 性能/吞吐基准：
  - python scripts/perf.py --trials 20 --budget 4096
- 论文一致性检查：
  - python scripts/test_paper_compliance.py --skip-search

**性能数据**
- 环境: Apple M1 Pro, Darwin 23.6.0 arm64, Python 3.14.2
- 命令: `python scripts/perf.py --trials 50 --budget 8192 --seed 123`
- 结果:
```text
block1_step1_3: trials=50 success=41 time=1.113s rate=44.91/s
block2_step1_3: trials=50 success=6 time=2.205s rate=22.67/s
block1_step4: budget=8192 ok=False time=0.511s rate=16025.54 it/s
```

注意：

- 搜索复杂度较高，通常需要提高 `--restarts` 或 `scripts/perf.py --budget` 以观察更深层的成功率变化。
- `tables/block1.txt` 与 `tables/block2.txt` 为论文附录 A 条件表的转录，`I/J/K` 表示同一未知位（K=I）。
- Block2 搜索要求 `IHV2[25]=1` 且 `IHV3[25]=0`，CLI 未提供 `--ihv` 时会自动用示例 IHV（满足 Table A-3 的 -2..0 条件）。
- `fastcoll` 子命令在有 `--prefixfile` 时按 64 字节块零填充前缀（与原始 fastcoll 行为一致）。
