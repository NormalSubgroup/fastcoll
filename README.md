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
  - python -m md5fastcoll.cli search-block2 --restarts 50 --ihv 0x12345678 0x87654321 0x02000000 0x00000000
- 两块管线（较慢）：
  - python -m md5fastcoll.cli search-collision --restarts 500

脚本：

- 准确性检查：
  - python scripts/accuracy.py
- 性能/吞吐基准：
  - python scripts/perf.py --trials 20 --budget 4096
- 论文一致性检查：
  - python scripts/test_paper_compliance.py --skip-search

注意：

- 搜索复杂度较高，通常需要提高 `--restarts` 或 `scripts/perf.py --budget` 以观察更深层的成功率变化。
- `tables/block1.txt` 与 `tables/block2.txt` 为论文附录 A 条件表的转录，`I/J/K` 表示同一未知位（K=I）。
