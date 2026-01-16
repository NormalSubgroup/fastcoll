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
    ihv = tuple(ns.ihv) if ns.ihv is not None else MD5_IV
    ok, res, stats = search_block2_once(ihv, max_restarts=ns.restarts)
    print("search-block2 once:", ok, stats)
    return 0


def cmd_search_collision(ns: argparse.Namespace) -> int:
    ok, info = search_two_block_collision(max_restarts=ns.restarts, seed=ns.seed)
    print("search-collision:", ok)
    print("restarts:", info.get("restarts"))
    return 0 if ok else 1


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
    def _cmd_collision_full(ns: argparse.Namespace) -> int:
        from .stevens_full import search_collision_full
        r = search_collision_full(seed=ns.seed, max_restarts=ns.restarts)
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
