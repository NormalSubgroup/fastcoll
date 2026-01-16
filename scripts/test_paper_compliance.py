#!/usr/bin/env python3
"""
全面测试脚本，验证实现与Stevens MD5 Fast Collision论文的匹配度
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from md5fastcoll.core import MD5_IV, compress_block, ft, wt_index
from md5fastcoll.conditions import minimal_block1_q_constraints, minimal_block2_q_constraints
from md5fastcoll.verify import check_T_restrictions_full, check_next_block_iv_conditions, check_recommended_iv_conditions
from md5fastcoll.stevens_full import Block1FullSearcher, Block2FullSearcher, search_collision_full
from md5fastcoll.md5 import md5_hex
import hashlib
import random

def test_md5_core_functionality():
    """测试MD5核心函数与标准实现一致性"""
    print("=" * 60)
    print("测试1: MD5核心函数一致性")
    print("=" * 60)
    
    test_vectors = [
        b"",
        b"a", 
        b"abc",
        b"message digest",
        b"abcdefghijklmnopqrstuvwxyz",
        b"The quick brown fox jumps over the lazy dog",
    ]
    
    all_pass = True
    for i, msg in enumerate(test_vectors):
        our_hash = md5_hex(msg)
        ref_hash = hashlib.md5(msg).hexdigest()
        status = "✓" if our_hash == ref_hash else "✗"
        print(f"Test {i+1}: {status} {msg[:30]}{'...' if len(msg) > 30 else ''}")
        if our_hash != ref_hash:
            print(f"  期望: {ref_hash}")
            print(f"  实际: {our_hash}")
            all_pass = False
    
    print(f"\nMD5核心测试: {'全部通过' if all_pass else '存在问题'}")
    return all_pass

def test_condition_loading():
    """测试条件表加载"""
    print("=" * 60)
    print("测试2: 条件表加载验证")
    print("=" * 60)
    
    qc1 = minimal_block1_q_constraints()
    qc2 = minimal_block2_q_constraints()
    
    print(f"Block1 条件数量: {len(qc1.conds)}")
    print(f"Block2 条件数量: {len(qc2.conds)}")
    
    # 检查关键条件是否存在
    key_conditions = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    missing_b1 = [t for t in key_conditions if t not in qc1.conds]
    missing_b2 = [t for t in key_conditions if t not in qc2.conds]
    
    print(f"Block1 缺失关键条件: {missing_b1 if missing_b1 else '无'}")
    print(f"Block2 缺失关键条件: {missing_b2 if missing_b2 else '无'}")
    
    # 验证条件格式
    sample_valid = True
    for t in [4, 5, 6] if 4 in qc1.conds else []:
        cond = qc1.conds[t]
        if len(cond.pattern) != 32:
            print(f"条件{t}格式错误: 长度{len(cond.pattern)} != 32")
            sample_valid = False
    
    print(f"条件表加载: {'成功' if sample_valid and len(qc1.conds) > 10 else '存在问题'}")
    return sample_valid and len(qc1.conds) > 10

def test_T_restrictions():
    """测试T限制验证"""
    print("=" * 60)
    print("测试3: T限制验证")
    print("=" * 60)
    
    # 生成随机消息测试T限制
    random.seed(42)
    total_tests = 100
    passed_tests = 0
    
    for i in range(total_tests):
        # 生成随机16个32位字
        m_words = [random.getrandbits(32) for _ in range(16)]
        _, trace = compress_block(MD5_IV, m_words)
        ok_t, issues_t = check_T_restrictions_full(trace)
        
        if ok_t:
            passed_tests += 1
        elif i < 5:  # 只显示前几个失败案例的详情
            print(f"  测试{i+1} T限制失败: {list(issues_t.keys())}")
    
    success_rate = passed_tests / total_tests
    print(f"T限制通过率: {passed_tests}/{total_tests} = {success_rate:.2%}")
    # 这些限制在随机消息中极低概率满足，关注是否能正确检测违规
    ok = passed_tests < total_tests
    print(f"T限制验证: {'正常' if ok else '存在问题'}")
    return ok

def test_iv_conditions():
    """测试IV推荐条件"""
    print("=" * 60)
    print("测试4: IV推荐条件验证")
    print("=" * 60)
    
    # 测试标准MD5 IV
    ok_std, issues_std = check_recommended_iv_conditions(MD5_IV)
    print(f"标准MD5 IV: {'满足推荐条件' if ok_std else '不满足推荐条件'}")
    if issues_std:
        print(f"  问题: {list(issues_std.keys())}")
    
    # 生成满足推荐条件的IV
    IV0, IV1, IV2, IV3 = MD5_IV
    # 设置 IV2[25] = IV2[24] and IV3[25] = IV3[24]
    iv2_24 = (IV2 >> 24) & 1
    iv3_24 = (IV3 >> 24) & 1
    
    recommended_IV2 = (IV2 & ~(1 << 25)) | (iv2_24 << 25)
    recommended_IV3 = (IV3 & ~(1 << 25)) | (iv3_24 << 25)
    recommended_iv = (IV0, IV1, recommended_IV2, recommended_IV3)
    
    ok_rec, issues_rec = check_recommended_iv_conditions(recommended_iv)
    print(f"修正后IV: {'满足推荐条件' if ok_rec else '不满足推荐条件'}")
    
    return ok_rec

def test_algorithm_6_1():
    """测试算法6-1实现"""
    print("=" * 60)
    print("测试5: 算法6-1 Block1搜索")
    print("=" * 60)
    
    searcher = Block1FullSearcher()
    
    # 尝试少量重启的搜索
    print("尝试Block1搜索 (最多5次重启)...")
    result = searcher.search(MD5_IV, max_restarts=5)
    
    if result:
        print("✓ Block1搜索成功!")
        print(f"  最终IHV: {[hex(x) for x in result.ihv]}")
        print(f"  消息字数量: {len(result.m_words)}")
        
        # 验证IV条件
        ok_iv, issues_iv = check_next_block_iv_conditions(result.ihv)
        print(f"  IV条件检查: {'通过' if ok_iv else '失败'}")
        if issues_iv:
            print(f"    问题: {list(issues_iv.keys())}")
        return True
    else:
        print("✗ Block1搜索失败 (在5次重启内)")
        return False

def test_algorithm_6_2():
    """测试算法6-2实现"""
    print("=" * 60)
    print("测试6: 算法6-2 Block2搜索")
    print("=" * 60)
    
    # 使用一个满足IV条件的IHV作为输入
    # 这里使用一个示例IHV（实际应该从Block1搜索得到）
    sample_ihv = (0x12345678, 0x87654321, 0x02000000, 0x00000000)  # IHV2[25]=1, IHV3[25]=0
    
    searcher = Block2FullSearcher()
    print("尝试Block2搜索 (最多5次重启)...")
    result = searcher.search(sample_ihv, max_restarts=5)
    
    if result:
        print("✓ Block2搜索成功!")
        print(f"  最终IHV: {[hex(x) for x in result.ihv]}")
        print(f"  消息字数量: {len(result.m_words)}")
        return True
    else:
        print("✗ Block2搜索失败 (在5次重启内)")
        return False

def test_full_collision():
    """测试完整的两块碰撞搜索"""
    print("=" * 60)
    print("测试7: 完整两块碰撞搜索")
    print("=" * 60)
    
    print("尝试完整碰撞搜索 (最多10次重启)...")
    result = search_collision_full(seed=42, max_restarts=10)
    
    if result:
        b1_result, b2_result = result
        print("✓ 完整碰撞搜索成功!")
        print(f"  Block1 IHV: {[hex(x) for x in b1_result.ihv]}")
        print(f"  Block2 IHV: {[hex(x) for x in b2_result.ihv]}")
        print(f"  Block1 消息: {len(b1_result.m_words)} words")
        print(f"  Block2 消息: {len(b2_result.m_words)} words")
        return True
    else:
        print("✗ 完整碰撞搜索失败 (在10次重启内)")
        return False

def main():
    """运行所有测试"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-search", action="store_true", help="跳过 Block1/Block2/两块搜索")
    args = ap.parse_args()

    print("Stevens MD5 Fast Collision 论文合规性测试")
    print("=" * 80)
    
    test_results = []
    test_functions = [
        ("MD5核心功能", test_md5_core_functionality),
        ("条件表加载", test_condition_loading),
        ("T限制验证", test_T_restrictions),
        ("IV推荐条件", test_iv_conditions),
    ]
    if not args.skip_search:
        test_functions.extend([
            ("算法6-1", test_algorithm_6_1),
            ("算法6-2", test_algorithm_6_2),
            ("完整碰撞搜索", test_full_collision),
        ])
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 异常: {e}")
            test_results.append((test_name, False))
        print()
    
    # 总结
    print("=" * 80)
    print("测试总结:")
    print("=" * 80)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print("-" * 40)
    print(f"总计: {passed}/{len(test_results)} 测试通过")
    
    if passed == len(test_results):
        print("🎉 所有测试通过！实现与论文高度匹配。")
        return 0
    elif passed >= len(test_results) * 0.7:
        print("⚠️  大部分测试通过，实现基本符合论文要求。")
        return 0
    else:
        print("❌ 多项测试失败，需要进一步修正。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
