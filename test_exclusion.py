#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试排除功能的简单脚本
"""

import os
from excluded_samples import EXCLUDED_SAMPLE_IDS, is_sample_excluded, get_excluded_count

def test_exclusion_logic():
    """
    测试排除逻辑
    """
    print("🧪 测试排除功能...")

    # 测试几个已知的排除样本
    test_cases = [
        ("0A5E2E5E-A5B4-41D2-A090-B41E38A68788", True),   # 应该被排除
        ("00B2919C-3EE9-4180-933D-FFD14B6D3C8B", True),   # 应该被排除
        ("not-in-list-sample-id", False),                  # 不应该被排除
        ("another-fake-id", False),                        # 不应该被排除
    ]

    for sample_id, expected in test_cases:
        result = is_sample_excluded(sample_id)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {sample_id}: 期望={expected}, 实际={result}")

    print(f"   📊 排除列表总数: {get_excluded_count()}")

def check_real_data():
    """
    检查真实数据中的排除情况
    """
    print("\n📁 检查实际数据中的排除情况...")

    RAW_DATA_DIR = "data_prepare/重标注后完整牙列lower_jaw"

    if not os.path.exists(RAW_DATA_DIR):
        print(f"   ❌ 数据目录不存在: {RAW_DATA_DIR}")
        return

    total_samples = 0
    excluded_samples = 0
    excluded_list = []

    for jaw_dir in ["lower_jaw1", "lower_jaw2", "lower_jaw3"]:
        jaw_path = os.path.join(RAW_DATA_DIR, jaw_dir)
        if not os.path.exists(jaw_path):
            continue

        print(f"   📂 检查 {jaw_dir}:")
        jaw_total = 0
        jaw_excluded = 0

        for instance_id in os.listdir(jaw_path):
            instance_path = os.path.join(jaw_path, instance_id)
            if os.path.isdir(instance_path):
                jaw_total += 1
                total_samples += 1

                if is_sample_excluded(instance_id):
                    jaw_excluded += 1
                    excluded_samples += 1
                    excluded_list.append(instance_id)

        print(f"      总样本: {jaw_total}, 排除: {jaw_excluded}, 有效: {jaw_total - jaw_excluded}")

    print(f"\n   📊 总体统计:")
    print(f"      总样本: {total_samples}")
    print(f"      排除样本: {excluded_samples}")
    print(f"      有效样本: {total_samples - excluded_samples}")
    print(f"      排除率: {excluded_samples/total_samples*100:.1f}%" if total_samples > 0 else "      排除率: 0%")

    # 显示前几个被排除的样本
    if excluded_list:
        print(f"\n   🗑️ 实际被排除的样本示例:")
        for i, sample_id in enumerate(excluded_list[:5]):
            print(f"      {i+1}. {sample_id}")
        if len(excluded_list) > 5:
            print(f"      ... 还有 {len(excluded_list) - 5} 个")

def main():
    """
    主测试函数
    """
    print("=" * 50)
    print("🧪 排除功能测试")
    print("=" * 50)

    test_exclusion_logic()
    check_real_data()

    print("\n" + "=" * 50)
    print("✅ 排除功能测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()