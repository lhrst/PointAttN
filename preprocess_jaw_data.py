#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointAttN 牙列数据预处理脚本 - 快速版本
去除复杂的网格操作，专注于快速处理
"""

import os
import json
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
from excluded_samples import EXCLUDED_SAMPLE_IDS, is_sample_excluded

# 配置参数
RAW_DATA_DIR = "data_prepare/重标注后完整牙列lower_jaw"
OUTPUT_DIR = "processed_jaw_data"
N_POINTS = 2048
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
N_PARTIAL_VIEWS = 8

# 性能优化配置
NUM_PROCESSES = min(mp.cpu_count(), 16)  # 使用多进程
CHUNK_SIZE = 10  # 每个进程处理的样本数

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_obj_fast(obj_path):
    """快速读取OBJ文件的顶点"""
    vertices = []
    try:
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(vertices) if vertices else None
    except Exception as e:
        print(f"Error reading {obj_path}: {e}")
        return None

def read_labels_fast(json_path):
    """快速读取标签"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        labels = []
        for label in label_data['labels']:
            if isinstance(label, (int, float)):
                labels.append(int(label))
            elif isinstance(label, str) and label.isdigit():
                labels.append(int(label))
            else:
                labels.append(0)

        return np.array(labels)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

def normalize_vertices(vertices):
    """快速归一化顶点"""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    scale = np.max(bbox_max - bbox_min) / 2
    return (vertices - center) / scale

def sample_points_fast(vertices, n_points):
    """快速点采样"""
    if len(vertices) >= n_points:
        indices = np.random.choice(len(vertices), n_points, replace=False)
    else:
        indices = np.random.choice(len(vertices), n_points, replace=True)
    return vertices[indices]

def create_partial_fast(vertices, strategy='z_split', **kwargs):
    """快速创建partial点云"""
    if strategy == 'z_split':
        mask_ratio = kwargs.get('mask_ratio', 0.5)
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        z_threshold = z_min + (z_max - z_min) * mask_ratio
        mask = vertices[:, 2] > z_threshold

        # 确保至少有一些点
        if np.sum(mask) < 100:
            mask = vertices[:, 2] > (z_min + (z_max - z_min) * 0.3)

        return vertices[mask]

    elif strategy == 'random_remove':
        keep_ratio = kwargs.get('keep_ratio', 0.7)
        n_keep = max(int(len(vertices) * keep_ratio), 100)
        indices = np.random.choice(len(vertices), n_keep, replace=False)
        return vertices[indices]

    else:
        # 默认随机采样
        n_keep = max(int(len(vertices) * 0.7), 100)
        indices = np.random.choice(len(vertices), n_keep, replace=False)
        return vertices[indices]

def save_points_simple(points, filepath):
    """简单快速的点云保存（文本格式）"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存为简单的文本格式（更快）
        np.savetxt(filepath.replace('.pcd', '.txt'), points, fmt='%.6f')

        # 尝试使用open3d保存PCD格式
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            o3d.io.write_point_cloud(filepath, pcd)
        except ImportError:
            # 如果open3d不可用，使用纯文本PCD格式
            with open(filepath, 'w') as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z\n")
                f.write("SIZE 4 4 4\n")
                f.write("TYPE F F F\n")
                f.write("COUNT 1 1 1\n")
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                for point in points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False

def process_single_sample_fast(args):
    """快速处理单个样本"""
    sample_path, instance_id, output_dir, split = args

    try:
        # 读取数据
        obj_path = os.path.join(sample_path, "lower.obj")
        json_path = os.path.join(sample_path, "modified_seg.json")

        vertices = read_obj_fast(obj_path)
        labels = read_labels_fast(json_path)

        if vertices is None or labels is None:
            return None

        # 快速归一化
        vertices = normalize_vertices(vertices)

        # 生成完整点云
        complete_points = sample_points_fast(vertices, N_POINTS)

        # 保存完整点云
        complete_dir = os.path.join(output_dir, split, "complete", "000")
        complete_path = os.path.join(complete_dir, f"{instance_id}.pcd")
        save_points_simple(complete_points, complete_path)

        # 生成partial点云
        partial_dir = os.path.join(output_dir, split, "partial", "000", instance_id)

        for i in range(N_PARTIAL_VIEWS):
            # 快速生成partial
            if random.random() < 0.5:
                partial_vertices = create_partial_fast(
                    vertices, 'z_split',
                    mask_ratio=random.uniform(0.3, 0.7)
                )
            else:
                partial_vertices = create_partial_fast(
                    vertices, 'random_remove',
                    keep_ratio=random.uniform(0.5, 0.8)
                )

            # 采样partial点云
            partial_points = sample_points_fast(partial_vertices, N_POINTS // 4)

            # 保存partial点云
            partial_path = os.path.join(partial_dir, f"{i:02d}.pcd")
            save_points_simple(partial_points, partial_path)

        return instance_id

    except Exception as e:
        print(f"Error processing {instance_id}: {e}")
        return None

def collect_all_samples():
    """收集所有样本信息"""
    samples = []
    excluded_count = 0
    total_found = 0

    print("📋 扫描数据目录...")

    for jaw_dir in ["lower_jaw1", "lower_jaw2", "lower_jaw3"]:
        jaw_path = os.path.join(RAW_DATA_DIR, jaw_dir)
        if not os.path.exists(jaw_path):
            continue

        jaw_samples = []
        for instance_id in os.listdir(jaw_path):
            instance_path = os.path.join(jaw_path, instance_id)
            if os.path.isdir(instance_path):
                obj_path = os.path.join(instance_path, "lower.obj")
                json_path = os.path.join(instance_path, "modified_seg.json")

                if os.path.exists(obj_path) and os.path.exists(json_path):
                    total_found += 1

                    if is_sample_excluded(instance_id):
                        excluded_count += 1
                        continue

                    jaw_samples.append((instance_path, instance_id))

        samples.extend(jaw_samples)
        print(f"   ✅ {jaw_dir}: {len(jaw_samples)} 有效样本")

    print(f"📊 总统计: {len(samples)} 有效样本 / {excluded_count} 排除 / {total_found} 总计")
    return samples

def create_dataset_metadata(output_dir, train_samples, test_samples):
    """创建数据集元数据文件"""
    dataset_json = [
        {
            "taxonomy_id": "000",
            "taxonomy_name": "LowerJaw",
            "test": test_samples,
            "train": train_samples,
            "val": []
        }
    ]

    with open(os.path.join(output_dir, "PCN.json"), 'w') as f:
        json.dump(dataset_json, f, indent=2)

    with open(os.path.join(output_dir, "category.txt"), 'w') as f:
        f.write("LowerJaw\n")

def main():
    """主处理函数 - 快速版本"""
    start_time = time.time()

    print("🚀 开始快速牙列数据预处理...")
    print(f"⚡ 使用 {NUM_PROCESSES} 个进程并行处理")

    # 创建输出目录
    ensure_dir(OUTPUT_DIR)
    for split in ['train', 'test']:
        ensure_dir(os.path.join(OUTPUT_DIR, split, 'complete', '000'))
        ensure_dir(os.path.join(OUTPUT_DIR, split, 'partial', '000'))

    # 收集样本
    all_samples = collect_all_samples()

    if len(all_samples) == 0:
        print("❌ 没有找到有效样本！")
        return

    # 分割数据
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:]

    print(f"📊 数据分割: {len(train_samples)} 训练 + {len(test_samples)} 测试")

    # 准备任务
    train_tasks = [(sample_path, instance_id, OUTPUT_DIR, "train")
                   for sample_path, instance_id in train_samples]
    test_tasks = [(sample_path, instance_id, OUTPUT_DIR, "test")
                  for sample_path, instance_id in test_samples]
    all_tasks = train_tasks + test_tasks

    print(f"⚡ 开始并行处理 {len(all_tasks)} 个样本...")

    # 多进程处理
    processed_train = []
    processed_test = []

    with mp.Pool(NUM_PROCESSES) as pool:
        with tqdm(total=len(all_tasks), desc="处理样本", unit="samples") as pbar:
            # 使用 imap_unordered 获得更好的性能
            results = pool.imap_unordered(process_single_sample_fast, all_tasks, chunksize=CHUNK_SIZE)

            for i, result in enumerate(results):
                if result is not None:
                    if i < len(train_tasks):
                        processed_train.append(result)
                    else:
                        processed_test.append(result)
                pbar.update(1)

    # 创建元数据
    print("📝 创建数据集元数据...")
    create_dataset_metadata(OUTPUT_DIR, processed_train, processed_test)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n🎉 快速预处理完成！")
    print(f"⏱️  总处理时间: {total_time:.2f} 秒")
    print(f"📊 处理速度: {len(all_samples)/total_time:.2f} 样本/秒")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"✅ 成功处理: {len(processed_train)} 训练 + {len(processed_test)} 测试")

    # 显示性能提升
    estimated_original_time = len(all_samples) * 5  # 假设原版本每样本5秒
    speedup = estimated_original_time / total_time
    print(f"🚀 预估加速比: {speedup:.1f}x")

if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    main()