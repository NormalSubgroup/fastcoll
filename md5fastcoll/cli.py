from __future__ import annotations

import argparse
import hashlib
from typing import List

from .md5 import md5_hex
from .inverse import demo_inverse_once
from .algos import search_block1_once, search_block2_once, search_two_block_collision
from .core import MD5_IV


def cmd_verify_core(_: argparse.Namespace) -> int:
    vectors = [
        b"",
        b"a",
        b"abc",
        b"message digest",
        b"abcdefghijklmnopqrstuvwxyz",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        b"1234567890" * 8,
    ]
    ok_all = True
    for m in vectors:
        ours = md5_hex(m)
        ref = hashlib.md5(m).hexdigest()
        status = "OK" if ours == ref else "FAIL"
        print(f"MD5('{m[:20] + (b'...' if len(m) > 20 else b'')}') -> {status}")
        if ours != ref:
            print(f"  ours={ours}\n  ref ={ref}")
            ok_all = False
    print("verify-core:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def cmd_demo_inverse(_: argparse.Namespace) -> int:
    ok, stats = demo_inverse_once()
    print(f"demo-inverse: reverse W check -> {'PASS' if ok else 'FAIL'}; mismatches={stats['mismatches']}")
    return 0 if ok else 1


def cmd_search_block1(ns: argparse.Namespace) -> int:
    ihv = tuple(ns.ihv) if ns.ihv is not None else MD5_IV
    ok, res, stats = search_block1_once(ihv, max_restarts=ns.restarts)
    print("search-block1 once:", ok, stats)
    return 0


def cmd_search_block2(ns: argparse.Namespace) -> int:
    if ns.ihv is None:
        ihv = (0xC4DA537C, 0x1051DD8E, 0x42867DB3, 0x0D67B366)
        print("search-block2: using sample IHV with IHV2[25]=1, IHV3[25]=0")
    else:
        ihv = tuple(ns.ihv)
    from .verify import check_next_block_iv_conditions
    ok_iv, issues_iv = check_next_block_iv_conditions(ihv)
    if not ok_iv:
        print(f"search-block2: invalid IHV for block2, issues={issues_iv}")
        return 1
    from .conditions import minimal_block2_q_constraints
    qc = minimal_block2_q_constraints()
    base = 3
    Q = [0] * (base + 65)
    IV0, IV1, IV2, IV3 = ihv
    Q[base - 3] = IV0
    Q[base - 2] = IV3
    Q[base - 1] = IV2
    Q[base + 0] = IV1
    ok_iv_q, bad_iv_q = qc.check_all(Q, base=base, start_t=-2, end_t=1)
    if not ok_iv_q:
        print(f"search-block2: IHV violates Table A-3 (-2..0) conditions, bad={bad_iv_q}")
        return 1
    ok, res, stats = search_block2_once(ihv, max_restarts=ns.restarts)
    print("search-block2 once:", ok, stats)
    return 0 if ok else 1


def cmd_search_collision(ns: argparse.Namespace) -> int:
    ok, info = search_two_block_collision(max_restarts=ns.restarts, seed=ns.seed)
    print("search-collision:", ok)
    print("restarts:", info.get("restarts"))
    return 0 if ok else 1


def cmd_fastcoll(ns: argparse.Namespace) -> int:
    from pathlib import Path
    import os

    from .core import MD5_IV
    from .fastcoll import (
        FASTCOLL_DEFAULT_IHV_HEX,
        build_collision_messages,
        default_output_names,
        format_ihv_hex,
        ihv_after_prefix,
        parse_ihv_hex,
    )
    from .md5 import md5_hex
    from .native_fastcoll import find_md5_fastcoll_bin, run_md5_fastcoll

    prefix_padded = b""
    ihv = MD5_IV
    out1: Path
    out2: Path

    if ns.prefixfile:
        prefix_path = Path(ns.prefixfile)
        if not prefix_path.exists():
            print(f"fastcoll: prefix file not found: {prefix_path}")
            return 1
        ihv, prefix_padded = ihv_after_prefix(prefix_path.read_bytes(), MD5_IV)
        if ns.out is None:
            out1, out2 = default_output_names(prefix_path)
        else:
            out1, out2 = Path(ns.out[0]), Path(ns.out[1])
        if not ns.quiet:
            print(f"fastcoll: using prefix file {prefix_path} (padded to {len(prefix_padded)} bytes)")
    else:
        if ns.ihv:
            ihv = parse_ihv_hex(ns.ihv)
        elif ns.md5_iv:
            ihv = MD5_IV
        else:
            ihv = parse_ihv_hex(FASTCOLL_DEFAULT_IHV_HEX)
        if ns.out is None:
            out1, out2 = Path("msg1.bin"), Path("msg2.bin")
        else:
            out1, out2 = Path(ns.out[0]), Path(ns.out[1])

    if not ns.quiet:
        print(f"fastcoll: using IHV {format_ihv_hex(ihv)}")

    engine = ns.engine
    bin_path = find_md5_fastcoll_bin(ns.native_bin)

    lib_path = None
    if engine in ("auto", "ctypes") or ns.native_lib is not None:
        from .hashclash_lib import find_md5_fastcoll_lib

        lib_path = find_md5_fastcoll_lib(ns.native_lib)

    if engine == "auto":
        if bin_path is not None:
            engine = "native"
        elif lib_path is not None:
            engine = "ctypes"
        else:
            engine = "python"

    if engine == "native":
        if bin_path is None:
            print("fastcoll: native engine requested but no `md5_fastcoll` binary found")
            print("fastcoll: run `python -m md5fastcoll.cli build-native` or set `MD5_FASTCOLL_BIN`")
            return 1
        run_md5_fastcoll(
            bin_path,
            out1=out1,
            out2=out2,
            prefixfile=Path(ns.prefixfile) if ns.prefixfile else None,
            ihv_hex=format_ihv_hex(ihv),
            seed=ns.seed,
            quiet=ns.quiet,
        )
        # verify outputs
        h1 = md5_hex(out1.read_bytes())
        h2 = md5_hex(out2.read_bytes())
        if h1 != h2:
            print("fastcoll: native mismatch, computed hashes differ")
            print(f"  {out1}={h1}")
            print(f"  {out2}={h2}")
            return 1
        if not ns.quiet:
            print(f"fastcoll: wrote {out1} and {out2}")
            print(f"fastcoll: md5={h1}")
        return 0

    if engine == "ctypes":
        # HashClash md5fastcoll via ctypes shared library
        if lib_path is None:
            print("fastcoll: ctypes engine requested but no `md5_fastcoll_lib` shared library found")
            print("fastcoll: run `python -m md5fastcoll.cli build-native-lib` or set `MD5_FASTCOLL_LIB`")
            print("fastcoll: or pass `fastcoll --engine ctypes --native-lib /path/to/md5_fastcoll_lib.{so|dylib}`")
            return 1

        from .hashclash_lib import HashClashFastCollLib

        lib = HashClashFastCollLib(lib_path)
        b0, b1 = lib.find_collision_blocks(ihv, seed=ns.seed)
        msg1, msg2 = build_collision_messages(prefix_padded, b0, b1)
        h1 = md5_hex(msg1)
        h2 = md5_hex(msg2)
        if h1 != h2:
            print("fastcoll: internal mismatch, computed hashes differ (native-lib)")
            print(f"  msg1={h1}")
            print(f"  msg2={h2}")
            return 1

        out1.write_bytes(msg1)
        out2.write_bytes(msg2)
        if not ns.quiet:
            print(f"fastcoll: wrote {out1} and {out2}")
            print(f"fastcoll: md5={h1}")
        return 0

    # python engine (HashClash md5_fastcoll port, pure Python with optional NumPy/Numba)
    if ns.native_lib is not None and not ns.quiet:
        print("fastcoll: note: --native-lib is ignored when --engine python (use --engine ctypes)")

    from .numba_fastcoll import find_collision_blocks_numba, numba_available
    from .py_fastcoll import find_collision_blocks, find_collision_blocks_parallel

    jobs = int(ns.jobs or 0)
    if jobs < 0:
        print("fastcoll: --jobs must be >= 0")
        return 1
    if jobs == 0:
        jobs = 1 if ns.seed is not None else (os.cpu_count() or 1)

    if jobs <= 1:
        if numba_available():
            if not ns.quiet:
                print("fastcoll: using python+numba engine (JIT)")
            b0, b1 = find_collision_blocks_numba(ihv, seed=ns.seed)
        else:
            b0, b1 = find_collision_blocks(ihv, seed=ns.seed)
    else:
        b0, b1 = find_collision_blocks_parallel(ihv, seed=ns.seed, jobs=jobs)

    msg1, msg2 = build_collision_messages(prefix_padded, b0, b1)
    h1 = md5_hex(msg1)
    h2 = md5_hex(msg2)
    if h1 != h2:
        print("fastcoll: internal mismatch, computed hashes differ")
        print(f"  msg1={h1}")
        print(f"  msg2={h2}")
        return 1

    out1.write_bytes(msg1)
    out2.write_bytes(msg2)
    if not ns.quiet:
        print(f"fastcoll: wrote {out1} and {out2}")
        print(f"fastcoll: md5={h1}")
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="md5fastcoll")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("verify-core", help="验证核心 MD5 实现是否与 hashlib 一致")
    s1.set_defaults(func=cmd_verify_core)

    s2 = sub.add_parser("demo-inverse", help="演示逐步反推单块 Wt 的一致性")
    s2.set_defaults(func=cmd_demo_inverse)

    sT = sub.add_parser("check-t", help="检查论文第3节 T 限制（随机样本）")
    sT.add_argument("--samples", type=int, default=10)
    def _cmd_check_t(ns: argparse.Namespace) -> int:
        from .core import MD5_IV
        from .algos import run_block, random_block
        from .verify import check_T_restrictions_full
        ok_all = True
        fails = 0
        for i in range(ns.samples):
            res = run_block(MD5_IV, random_block())
            ok, bad = check_T_restrictions_full(res.trace)
            if not ok:
                fails += 1
        print(f"check-t: samples={ns.samples}, fails={fails}")
        return 0 if fails == 0 else 1
    sT.set_defaults(func=_cmd_check_t)

    s3 = sub.add_parser("search-block1", help="Block1 搜索（算法 6-1，真实实现）")
    s3.add_argument("--restarts", type=int, default=50)
    s3.add_argument("--ihv", nargs=4, type=lambda x: int(x, 0), default=None, help="4x 32-bit hex words")
    s3.set_defaults(func=cmd_search_block1)

    s4 = sub.add_parser("search-block2", help="Block2 搜索（算法 6-2，真实实现）")
    s4.add_argument("--restarts", type=int, default=50)
    s4.add_argument("--ihv", nargs=4, type=lambda x: int(x, 0), default=None, help="4x 32-bit hex words")
    s4.set_defaults(func=cmd_search_block2)

    # Full Stevens (6-1/6-2) commands
    sf1 = sub.add_parser("search-block1-full", help="严格匹配论文 6-1 的分步构造器（需完整条件表）")
    sf1.add_argument("--restarts", type=int, default=100)
    def _cmd_block1_full(ns: argparse.Namespace) -> int:
        from .stevens_full import Block1FullSearcher
        s = Block1FullSearcher()
        r = s.search(MD5_IV, max_restarts=ns.restarts)
        ok = r is not None
        print("search-block1-full:", ok)
        return 0 if ok else 1
    sf1.set_defaults(func=_cmd_block1_full)

    sf2 = sub.add_parser("search-collision-full", help="严格匹配论文 6-1/6-2 的两块管线（需完整条件表）")
    sf2.add_argument("--seed", type=int, default=None)
    sf2.add_argument("--restarts", type=int, default=1000)
    sf2.add_argument("--block1-restarts", type=int, default=50)
    sf2.add_argument("--block2-restarts", type=int, default=100)
    def _cmd_collision_full(ns: argparse.Namespace) -> int:
        from .stevens_full import search_collision_full
        r = search_collision_full(
            seed=ns.seed,
            max_restarts=ns.restarts,
            block1_restarts=ns.block1_restarts,
            block2_restarts=ns.block2_restarts,
        )
        ok = r is not None
        print("search-collision-full:", ok)
        if ok:
            (b1, b2) = r
            print("IHV after block1:", tuple(hex(x) for x in b1.ihv))
            print("IHV after block2:", tuple(hex(x) for x in b2.ihv))
        return 0 if ok else 1
    sf2.set_defaults(func=_cmd_collision_full)

    s5 = sub.add_parser("search-collision", help="两块碰撞搜索（算法 6-1/6-2）")
    s5.add_argument("--restarts", type=int, default=1000)
    s5.add_argument("--seed", type=int, default=None)
    s5.set_defaults(func=cmd_search_collision)

    # stevens mode
    s8 = sub.add_parser("stevens-block1", help="legacy：按论文 6-1 的构造流程（近似实现）")
    s8.add_argument("--max-outer", type=int, default=2000)
    def _cmd_stev_b1(ns: argparse.Namespace) -> int:
        from .stevens import construct_block1_stevens
        r = construct_block1_stevens(MD5_IV, max_outer=ns.max_outer)
        ok = r is not None
        print("stevens-block1:", ok)
        return 0 if ok else 1
    s8.set_defaults(func=_cmd_stev_b1)

    s9 = sub.add_parser("stevens-block2", help="legacy：按论文 6-2 的构造流程（近似实现）")
    s9.add_argument("--max-outer", type=int, default=5000)
    def _cmd_stev_b2(ns: argparse.Namespace) -> int:
        from .stevens import construct_block2_stevens
        # 用 MD5_IV 作为占位，实际应传入第一块后的 IHV
        r = construct_block2_stevens(MD5_IV, max_outer=ns.max_outer)
        ok = r is not None
        print("stevens-block2:", ok)
        return 0 if ok else 1
    s9.set_defaults(func=_cmd_stev_b2)

    s10 = sub.add_parser("stevens-collision", help="legacy：两块碰撞搜索（近似实现）")
    s10.add_argument("--seed", type=int, default=None)
    s10.add_argument("--max-outer", type=int, default=20000)
    def _cmd_stev_col(ns: argparse.Namespace) -> int:
        from .stevens import construct_two_block_collision_stevens
        r = construct_two_block_collision_stevens(seed=ns.seed, max_outer=ns.max_outer)
        ok = r is not None
        print("stevens-collision:", ok)
        return 0 if ok else 1
    s10.set_defaults(func=_cmd_stev_col)

    s11 = sub.add_parser("fastcoll", help="生成与 fastcoll 兼容的碰撞文件对")
    s11.add_argument("--prefixfile", "-p", type=str, default=None, help="前缀文件（按块零填充）")
    s11.add_argument("--ihv", type=str, default=None, help="32 hex chars IHV (little-endian)")
    s11.add_argument("--md5-iv", action="store_true", help="使用标准 MD5 IV 作为初始值")
    s11.add_argument("--out", "-o", nargs=2, default=None, metavar=("MSG1", "MSG2"))
    s11.add_argument("--seed", type=int, default=None)
    s11.add_argument("--engine", choices=["auto", "native", "ctypes", "python"], default="auto", help="碰撞生成引擎")
    s11.add_argument("--native-bin", type=str, default=None, help="本地 md5_fastcoll 可执行文件路径（或使用 MD5_FASTCOLL_BIN）")
    s11.add_argument("--native-lib", type=str, default=None, help="HashClash md5fastcoll 共享库路径（ctypes 引擎；或使用 MD5_FASTCOLL_LIB）")
    s11.add_argument("--jobs", type=int, default=0, help="python 引擎并行进程数（0=自动；指定 --seed 时默认 1）")
    s11.add_argument("--quiet", "-q", action="store_true")
    s11.set_defaults(func=cmd_fastcoll)

    sN = sub.add_parser("build-native", help="构建本地 md5_fastcoll（HashClash）用于加速")
    sN.add_argument("--out", type=str, default="tools/md5_fastcoll", help="输出二进制路径")
    sN.add_argument("--repo", type=str, default=None, help="HashClash git repo URL")
    sN.add_argument("--jobs", type=int, default=None, help="make -j 并发数")
    def _cmd_build_native(ns: argparse.Namespace) -> int:
        from pathlib import Path
        from .native_fastcoll import HASHCLASH_REPO_URL, build_md5_fastcoll
        repo = ns.repo or HASHCLASH_REPO_URL
        out = build_md5_fastcoll(Path(ns.out), repo_url=repo, jobs=ns.jobs)
        print(f"build-native: wrote {out}")
        return 0
    sN.set_defaults(func=_cmd_build_native)

    sNL = sub.add_parser("build-native-lib", help="构建 HashClash md5fastcoll 共享库（ctypes）用于 python 引擎加速")
    sNL.add_argument("--out", type=str, default=None, help="输出共享库路径（默认 tools/md5_fastcoll_lib.{so|dylib}）")
    sNL.add_argument("--repo", type=str, default=None, help="HashClash git repo URL")
    def _cmd_build_native_lib(ns: argparse.Namespace) -> int:
        from pathlib import Path
        from .hashclash_lib import build_md5_fastcoll_lib, _lib_suffix
        from .native_fastcoll import HASHCLASH_REPO_URL

        out = ns.out
        if out is None:
            out = str(Path("tools") / f"md5_fastcoll_lib{_lib_suffix()}")
        default_repo = "tools/hashclash-src" if Path("tools/hashclash-src").exists() else HASHCLASH_REPO_URL
        repo = ns.repo or default_repo
        built = build_md5_fastcoll_lib(Path(out), repo_url=repo)
        print(f"build-native-lib: wrote {built}")
        return 0
    sNL.set_defaults(func=_cmd_build_native_lib)

    sB = sub.add_parser("bench-fastcoll", help="基准测试：重复运行 fastcoll 并记录耗时")
    sB.add_argument("--mode", choices=["md5iv", "random", "recommended", "all"], default="md5iv")
    sB.add_argument("--runs", type=int, default=1, help="每种模式运行次数")
    sB.add_argument("--restarts", type=int, default=1000)
    sB.add_argument("--block1-restarts", type=int, default=50)
    sB.add_argument("--block2-restarts", type=int, default=100)
    sB.add_argument("--seed", type=int, default=None)
    sB.add_argument("--no-verify", action="store_true", help="跳过生成消息的 MD5 验证")
    sB.add_argument("--timings-out", type=str, default=None, help="将耗时追加写入文件")
    sB.add_argument("--quiet", "-q", action="store_true")
    def _cmd_bench_fastcoll(ns: argparse.Namespace) -> int:
        from pathlib import Path
        import random
        import time

        from .core import MD5_IV
        from .fastcoll import (
            build_collision_messages,
            format_ihv_hex,
            random_iv,
            recommended_iv,
        )
        from .md5 import md5_hex
        from .stevens_full import search_collision_full

        if ns.runs < 1:
            print("bench-fastcoll: --runs must be >= 1")
            return 1

        modes = ["md5iv", "recommended", "random"] if ns.mode == "all" else [ns.mode]
        rng = random.Random(ns.seed)

        def _timings_path(mode: str) -> Path | None:
            if ns.timings_out is None:
                return None
            template = ns.timings_out
            if "{mode}" in template:
                return Path(template.format(mode=mode))
            if ns.mode == "all":
                base = Path(template)
                suffix = base.suffix
                stem = base.stem
                return base.with_name(f"{stem}_{mode}{suffix}")
            return Path(template)

        for mode in modes:
            timing_path = _timings_path(mode)
            for run_idx in range(1, ns.runs + 1):
                if mode == "md5iv":
                    ihv = MD5_IV
                elif mode == "recommended":
                    ihv = recommended_iv(rng)
                else:
                    ihv = random_iv(rng)
                run_seed = rng.getrandbits(64)
                if not ns.quiet:
                    print(f"bench-fastcoll: mode={mode} run={run_idx}/{ns.runs} ihv={format_ihv_hex(ihv)}")

                start = time.perf_counter()
                res = search_collision_full(
                    seed=run_seed,
                    max_restarts=ns.restarts,
                    ihv=ihv,
                    block1_restarts=ns.block1_restarts,
                    block2_restarts=ns.block2_restarts,
                )
                elapsed = time.perf_counter() - start
                if res is None:
                    print(f"bench-fastcoll: mode={mode} run={run_idx} FAIL (increase --restarts)")
                    return 1
                if not ns.no_verify:
                    b1, b2 = res
                    msg1, msg2 = build_collision_messages(b"", b1.m_words, b2.m_words)
                    h1 = md5_hex(msg1)
                    h2 = md5_hex(msg2)
                    if h1 != h2:
                        print("bench-fastcoll: verification failed (hash mismatch)")
                        print(f"  msg1={h1}")
                        print(f"  msg2={h2}")
                        return 1
                if not ns.quiet:
                    print(f"bench-fastcoll: mode={mode} run={run_idx} time={elapsed:.3f}s")
                if timing_path is not None:
                    timing_path.parent.mkdir(parents=True, exist_ok=True)
                    with timing_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"{elapsed:.6f}\n")
        return 0
    sB.set_defaults(func=_cmd_bench_fastcoll)

    s6 = sub.add_parser("check-conds", help="使用 tables/block{1,2}.txt 检查 Qt 条件解析与匹配")
    s6.add_argument("--block", type=int, choices=[1,2], required=True)
    s6.add_argument("--samples", type=int, default=50)
    def _cmd_check_conds(ns: argparse.Namespace) -> int:
        from .core import MD5_IV
        from .algos import run_block, random_block
        from .conditions import minimal_block1_q_constraints, minimal_block2_q_constraints
        qc = minimal_block1_q_constraints() if ns.block == 1 else minimal_block2_q_constraints()
        if not qc.conds:
            print("no conditions loaded (place patterns in tables/block1.txt or tables/block2.txt)")
            return 1
        fails = 0
        for _ in range(ns.samples):
            res = run_block(MD5_IV, random_block())
            ok, bad = qc.check_all(res.trace["Q"])  # using base=3 default
            if not ok:
                fails += 1
        print(f"check-conds block{ns.block}: samples={ns.samples} fails={fails}")
        return 0 if fails < ns.samples else 1
    s6.set_defaults(func=_cmd_check_conds)

    s7 = sub.add_parser("export-conds-skeleton", help="导出条件表骨架到 tables/block{1,2}.txt")
    def _cmd_export(_: argparse.Namespace) -> int:
        import pathlib
        base = pathlib.Path("tables")
        base.mkdir(exist_ok=True)
        skel = "\n".join(f"{t}: \"{'.'*32}\"" for t in range(0, 65)) + "\n"
        for name in ["block1.txt", "block2.txt"]:
            p = base / name
            if not p.exists():
                p.write_text(skel, encoding="utf-8")
                print("wrote", p)
            else:
                print("exists", p)
        return 0
    s7.set_defaults(func=_cmd_export)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
