#!/usr/bin/env python3
"""
调试分析脚本，深入分析算法失败的原因
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import random
from md5fastcoll.core import MD5_IV, compress_block, ft, wt_index
from md5fastcoll.conditions import minimal_block1_q_constraints, BitCond
from md5fastcoll.stevens_full import Block1FullSearcher, _choose_Q_with_constraints, _compute_W_from_Q, _set_m
from md5fastcoll.verify import check_T_restrictions_full

def analyze_condition_complexity():
    """分析条件的复杂度"""
    print("=" * 60)
    print("分析1: 条件复杂度分析")
    print("=" * 60)
    
    qc = minimal_block1_q_constraints()
    
    for t in range(1, 22):  # 关键步骤
        if t in qc.conds:
            pattern = qc.conds[t].pattern
            fixed_bits = pattern.count('0') + pattern.count('1')
            rel_bits = pattern.count('^') + pattern.count('!')
            free_bits = pattern.count('.')
            total_constraints = fixed_bits + rel_bits
            
            print(f"Q{t:2d}: 固定{fixed_bits:2d}位, 相关{rel_bits:2d}位, 自由{free_bits:2d}位, 总约束{total_constraints:2d}")
        else:
            print(f"Q{t:2d}: 无条件")
    
    # 计算理论成功概率
    total_fixed = sum(qc.conds[t].pattern.count('0') + qc.conds[t].pattern.count('1') 
                     for t in qc.conds)
    theoretical_prob = 2**(-total_fixed)
    print(f"\n理论成功概率: ~2^{-total_fixed:.1f} = {theoretical_prob:.2e}")

def test_q_sampling():
    """测试Q值采样"""
    print("=" * 60)
    print("分析2: Q值采样测试") 
    print("=" * 60)
    
    qc = minimal_block1_q_constraints()
    rng = random.Random(42)
    
    # 测试关键步骤的采样成功率
    for t in [4, 5, 6, 7, 8]:
        if t not in qc.conds:
            continue
            
        successes = 0
        attempts = 1000
        
        # 模拟前一个Q值
        prev_q = rng.getrandbits(32)
        
        for _ in range(attempts):
            Q = [None] * (3 + 65)
            Q[3 + t - 1] = prev_q
            q = _choose_Q_with_constraints(rng, Q, qc, t)
            if q is not None:
                successes += 1
        
        success_rate = successes / attempts
        print(f"Q{t} 采样成功率: {successes}/{attempts} = {success_rate:.3f}")

def analyze_w_computation():
    """分析W计算的正确性"""
    print("=" * 60)
    print("分析3: W值计算验证")
    print("=" * 60)
    
    # 创建一个简单的测试用例
    rng = random.Random(42)
    base = 3
    Q = [0] * (base + 65)
    
    # 初始化Q[-3..0]
    IV0, IV1, IV2, IV3 = MD5_IV
    Q[base - 3] = IV0
    Q[base - 2] = IV3  
    Q[base - 1] = IV2
    Q[base + 0] = IV1
    
    # 设置一些Q值用于测试
    for t in range(1, 6):
        Q[base + t] = rng.getrandbits(32)
    
    # 测试W计算
    for t in range(0, 5):
        try:
            wt = _compute_W_from_Q(Q, t)
            print(f"W{t} = {wt:08x}")
            
            # 验证：用计算得到的W重新计算是否一致
            if t > 0:  # 需要Qt+1才能计算
                # 手动验证计算
                from md5fastcoll.stevens_full import _RC, _AC
                from md5fastcoll.core import rr, ft
                Qt = Q[base + t]
                Qt1 = Q[base + t + 1] 
                Qtm1 = Q[base + t - 1]
                Qtm2 = Q[base + t - 2] 
                Qtm3 = Q[base + t - 3]
                
                Rt = (Qt1 - Qt) & 0xFFFFFFFF
                Tt = rr(Rt, _RC[t])
                expected_wt = (Tt - ft(t, Qt, Qtm1, Qtm2) - Qtm3 - _AC[t]) & 0xFFFFFFFF
                
                if wt != expected_wt:
                    print(f"  ❌ W{t} 计算错误: 得到{wt:08x}, 期望{expected_wt:08x}")
                else:
                    print(f"  ✓ W{t} 计算正确")
        except Exception as e:
            print(f"W{t} 计算失败: {e}")

def test_step_by_step_construction():
    """逐步测试构造过程"""
    print("=" * 60)
    print("分析4: 逐步构造测试")
    print("=" * 60)
    
    searcher = Block1FullSearcher(random.Random(42))
    
    # 初始化
    Q = searcher._init_Q(MD5_IV)
    print("✓ Q初始化完成")
    print(f"Q[-3..0]: {[hex(Q[i]) for i in range(0, 4)]}")
    
    # 步骤1: 选择Q1...Q16
    print("\n测试步骤1: 选择Q1...Q16")
    success = searcher.step1_choose_Qs(Q)
    print(f"步骤1结果: {'成功' if success else '失败'}")
    
    if success:
        for t in range(1, 17):
            if Q[3 + t] is not None:
                print(f"Q{t} = {Q[3+t]:08x}")
    
    # 步骤2: 计算消息字
    if success:
        print("\n测试步骤2: 计算消息字")
        m = [None] * 16
        success2 = searcher.step2_compute_m0_to_m15(Q, m)
        print(f"步骤2结果: {'成功' if success2 else '失败'}")
        
        computed_words = [i for i in range(16) if m[i] is not None]
        print(f"成功计算的消息字: {computed_words}")
    
    return success

def analyze_constraint_conflicts():
    """分析约束冲突"""
    print("=" * 60)
    print("分析5: 约束冲突分析")
    print("=" * 60)
    
    qc = minimal_block1_q_constraints()
    
    # 检查相邻Q值之间的约束关系
    conflicts = 0
    for t in range(2, 17):
        if t in qc.conds and (t-1) in qc.conds:
            curr_pattern = qc.conds[t].pattern
            prev_pattern = qc.conds[t-1].pattern
            
            # 检查^和!约束是否与前一个Q的固定位冲突
            for i in range(32):
                curr_bit = curr_pattern[i]
                prev_bit = prev_pattern[i]
                
                if curr_bit == '^' and prev_bit in '01':
                    # 当前位必须等于前一位，但前一位已固定
                    pass  # 这实际上是好事，减少了自由度
                elif curr_bit == '!' and prev_bit in '01':
                    # 当前位必须不等于前一位，但前一位已固定
                    pass  # 这也减少了一个自由度
                elif curr_bit in '01' and prev_bit in '01':
                    if curr_bit == '^' and curr_bit != prev_bit:
                        print(f"潜在冲突: Q{t}[{31-i}] = ^ 但 Q{t-1}[{31-i}] = {prev_bit}")
                        conflicts += 1
    
    print(f"发现 {conflicts} 个潜在约束冲突")

def main():
    """主分析函数"""
    analyze_condition_complexity()
    print()
    test_q_sampling()
    print()
    analyze_w_computation()
    print()
    test_step_by_step_construction()
    print()
    analyze_constraint_conflicts()

if __name__ == "__main__":
    main()
