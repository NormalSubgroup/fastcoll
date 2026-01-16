#!/usr/bin/env python3
"""
统计与分析 T 限制违规情况：
- 随机消息基线（baseline）
- 我们当前 Block1 搜索结果（stevens_full.Block1FullSearcher）
输出：每条 T 限制违规的频次与占比，并给出对应论文中应添加的 Qt 附加条件建议。
"""
import sys
import random
from collections import Counter, defaultdict

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from md5fastcoll.core import MD5_IV, compress_block
from md5fastcoll.verify import check_T_restrictions_full
from md5fastcoll.stevens_full import Block1FullSearcher

# 论文 3.x -> 对应建议的 Qt 条件（来源：论文第3节）
RECOMMENDED_Q_FIXES = {
    "T4[31]": "确保: Q4[4]=1, Q4[5]=1, Q5[4]=0, 以及 Q4[6]=Q5[6]=0",
    "T6[14]": "确保: Q6[30]=Q6[29]=Q6[28]=0 且 Q6[26]=0 (从而 R6[31]=0)",
    "T10[13]": "确保: Q11[29]=Q11[28]=0, Q10[29]=0, Q10[28]=1",
    "T11[8]": "确保: Q12[29]=0 (并结合上一步使 R11[30]=1)",
    "T14[30|31]": "确保: Q15[16]=0 (保证 R14[15]=1 或 R14[16]=1)",
    "T15[7|8|9]": "确保: Q16[30]=Q15[30] (三者其一为1)",
    "T15[25|26|27]": "确保: Q16[17]=Q15[17] (三者其一为0)",
    "T16[24|25]": "确保: Q17[30]=Q16[30] (二者其一为0)",
    "T19[29|30]": "确保: Q20[18]=Q19[18] (二者其一为1)",
    "T22[17]": "直接验证/筛选: T22[17]==0（论文建议直接检查而非强制Qt）",
    "T34[15]": "直接验证/筛选: T34[15]==0（论文建议直接检查而非强制Qt）",
}


def analyze_random_baseline(samples: int = 200):
    counts = Counter()
    for _ in range(samples):
        m = [random.getrandbits(32) for _ in range(16)]
        _, trace = compress_block(MD5_IV, m)
        ok, issues = check_T_restrictions_full(trace)
        if not ok:
            counts.update(issues.keys())
    return counts, samples


def analyze_block1_results(successes: int = 20, max_restarts_per_try: int = 200):
    counts = Counter()
    got = 0
    tries = 0
    rng = random.Random(1234)
    searcher = Block1FullSearcher(rng)

    while got < successes and tries < successes * 50:
        res = searcher.search(MD5_IV, max_restarts=max_restarts_per_try)
        tries += 1
        if not res:
            continue
        ok, issues = check_T_restrictions_full(res.trace)
        if not ok:
            counts.update(issues.keys())
        got += 1
    return counts, got, tries


def pretty_print_counts(title: str, counts: Counter, total: int):
    print("\n" + title)
    print("=" * len(title))
    if total == 0:
        print("无样本")
        return
    for k, c in counts.most_common():
        pct = 100.0 * c / total
        print(f"- {k:<14} : {c:4d} / {total:<4d}  ({pct:6.2f}%)  建议: {RECOMMENDED_Q_FIXES.get(k, '—')}")


def main():
    print("统计 T 限制违规分布 …")

    # 1) 随机消息基线
    base_counts, base_total = analyze_random_baseline(samples=200)
    pretty_print_counts("[随机消息 基线] 违规频次", base_counts, base_total)

    # 2) 我们的 Block1 搜索结果
    block1_counts, got, tries = analyze_block1_results(successes=15, max_restarts_per_try=100)
    pretty_print_counts(f"[Block1 搜索结果] 违规频次 (成功样本 {got} / 尝试 {tries})", block1_counts, got)

    # 3) 汇总建议
    print("\n修复建议(根据论文第3节)：")
    print("- 在 tables/block1.txt 中增加/恢复以下关键位条件，对应上面频次高的违规项：")
    for key in [k for k, _ in base_counts.most_common(5)]:
        print(f"  * {key}: {RECOMMENDED_Q_FIXES.get(key, '—')}")
    print("- 在构造流程中于 t=4,6,10,11,14,15,16,19 处优先满足这些位条件，以提高 T 限制满足概率。")
    print("- T22 和 T34 按论文建议直接在末端过滤（不以 Qt 强制），可在循环中加入早停。")

if __name__ == "__main__":
    raise SystemExit(main())
