#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointAttN 牙列数据预处理脚本 - GPU加速版本
优化了处理速度，充分利用GPU和并行处理
"""

import os
import json
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm
import random
import shutil
from pathlib import Path
import multiprocessing
from functools import partial
import torch
import time
from excluded_samples import EXCLUDED_SAMPLE_IDS, is_sample_excluded

# 配置参数
RAW_DATA_DIR = "data_prepare/重标注后完整牙列lower_jaw"
OUTPUT_DIR = "processed_jaw_data"
N_POINTS = 2048
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
N_PARTIAL_VIEWS = 8

# GPU加速配置
USE_GPU = torch.cuda.is_available()
BATCH_SIZE = 16  # 批量处理大小
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

print(f"🚀 GPU加速: {'启用' if USE_GPU else '禁用'} (设备: {DEVICE})")

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def gpu_sample_points(vertices, n_points, device=DEVICE):
    """
    使用GPU加速的点云采样
    """
    if not USE_GPU:
        # 回退到CPU采样
        if len(vertices) >= n_points:
            indices = np.random.choice(len(vertices), n_points, replace=False)
        else:
            indices = np.random.choice(len(vertices), n_points, replace=True)
        return vertices[indices]

    try:
        # 转换为GPU tensor
        vertices_gpu = torch.from_numpy(vertices.astype(np.float32)).to(device)

        # GPU上的随机采样
        n_vertices = vertices_gpu.shape[0]
        if n_vertices >= n_points:
            indices = torch.randperm(n_vertices, device=device)[:n_points]
        else:
            # 重复采样
            indices = torch.randint(0, n_vertices, (n_points,), device=device)

        sampled_points = vertices_gpu[indices]

        # 转回CPU numpy
        return sampled_points.cpu().numpy()

    except Exception as e:
        print(f"GPU采样失败，回退到CPU: {e}")
        # 回退到CPU采样
        if len(vertices) >= n_points:
            indices = np.random.choice(len(vertices), n_points, replace=False)
        else:
            indices = np.random.choice(len(vertices), n_points, replace=True)
        return vertices[indices]

def gpu_normalize_batch(vertices_list):
    """
    批量GPU归一化
    """
    if not USE_GPU or len(vertices_list) == 0:
        return [normalize_mesh_cpu(v) for v in vertices_list]

    try:
        # 计算全局边界框
        all_vertices = np.concatenate(vertices_list, axis=0)

        # 转到GPU
        vertices_gpu = torch.from_numpy(all_vertices.astype(np.float32)).to(DEVICE)

        # GPU上计算边界框
        bbox_min = torch.min(vertices_gpu, dim=0)[0]
        bbox_max = torch.max(vertices_gpu, dim=0)[0]

        center = (bbox_min + bbox_max) / 2
        scale = torch.max(bbox_max - bbox_min) / 2

        # 批量归一化
        normalized_list = []
        start_idx = 0

        for vertices in vertices_list:
            end_idx = start_idx + len(vertices)
            vertices_gpu_batch = torch.from_numpy(vertices.astype(np.float32)).to(DEVICE)
            normalized_gpu = (vertices_gpu_batch - center) / scale
            normalized_list.append(normalized_gpu.cpu().numpy())
            start_idx = end_idx

        return normalized_list

    except Exception as e:
        print(f"GPU批量归一化失败，回退到CPU: {e}")
        return [normalize_mesh_cpu(v) for v in vertices_list]

def normalize_mesh_cpu(vertices):
    """CPU版本的网格归一化"""
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    scale = np.max(bbox_max - bbox_min) / 2
    return (vertices - center) / scale

def read_mesh_and_labels(obj_path, json_path):
    """读取mesh和对应的标签"""
    try:
        mesh = trimesh.load(obj_path)
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

        return mesh, np.array(labels)
    except Exception as e:
        print(f"Error reading {obj_path} or {json_path}: {e}")
        return None, None

def get_partial_jaw_by_z_coordinate_fast(vertices, mask_ratio=0.5):
    """
    快速版本的Z坐标分割（只处理顶点，不处理面）
    """
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()
    z_threshold = z_min + (z_max - z_min) * mask_ratio

    # 保留Z坐标大于阈值的点
    mask = vertices[:, 2] > z_threshold
    if np.sum(mask) < 100:  # 确保至少有100个点
        mask = vertices[:, 2] > (z_min + (z_max - z_min) * 0.3)

    return vertices[mask]

def get_partial_jaw_by_random_removal(vertices, exclude_ratio=0.3):
    """
    随机移除部分点
    """
    n_keep = int(len(vertices) * (1 - exclude_ratio))
    n_keep = max(n_keep, 100)  # 至少保留100个点

    indices = np.random.choice(len(vertices), n_keep, replace=False)
    return vertices[indices]

def process_batch_samples(batch_args):
    """
    批量处理样本（优化版本）
    """
    batch_data, output_dir, split = batch_args
    processed_samples = []

    try:
        # 批量读取数据
        meshes_data = []
        valid_samples = []

        for sample_path, instance_id in batch_data:
            obj_path = os.path.join(sample_path, "lower.obj")
            json_path = os.path.join(sample_path, "modified_seg.json")

            mesh, labels = read_mesh_and_labels(obj_path, json_path)
            if mesh is not None and labels is not None:
                meshes_data.append((mesh, labels, instance_id, sample_path))
                valid_samples.append(instance_id)

        if not meshes_data:
            return []

        # 批量归一化
        vertices_list = [mesh.vertices for mesh, _, _, _ in meshes_data]
        normalized_vertices_list = gpu_normalize_batch(vertices_list)

        # 处理每个样本
        for i, ((mesh, labels, instance_id, sample_path), normalized_vertices) in enumerate(zip(meshes_data, normalized_vertices_list)):
            try:
                # 生成完整点云
                complete_points = gpu_sample_points(normalized_vertices, N_POINTS)

                # 保存完整点云
                complete_dir = os.path.join(output_dir, split, "complete", "000")
                ensure_dir(complete_dir)
                complete_path = os.path.join(complete_dir, f"{instance_id}.pcd")
                save_pointcloud_fast(complete_points, complete_path)

                # 生成多个partial点云
                partial_dir = os.path.join(output_dir, split, "partial", "000", instance_id)
                ensure_dir(partial_dir)

                for j in range(N_PARTIAL_VIEWS):
                    # 快速生成partial
                    if random.random() < 0.5:
                        partial_vertices = get_partial_jaw_by_z_coordinate_fast(
                            normalized_vertices, mask_ratio=random.uniform(0.3, 0.7)
                        )
                    else:
                        partial_vertices = get_partial_jaw_by_random_removal(
                            normalized_vertices, exclude_ratio=random.uniform(0.2, 0.5)
                        )

                    # 采样partial点云
                    partial_points = gpu_sample_points(partial_vertices, N_POINTS // 4)

                    # 保存partial点云
                    partial_path = os.path.join(partial_dir, f"{j:02d}.pcd")
                    save_pointcloud_fast(partial_points, partial_path)

                processed_samples.append(instance_id)

            except Exception as e:
                print(f"Error processing sample {instance_id}: {e}")
                continue

    except Exception as e:
        print(f"Error in batch processing: {e}")

    return processed_samples

def save_pointcloud_fast(points, filepath):
    """快速保存点云"""
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        o3d.io.write_point_cloud(filepath, pcd)
    except Exception as e:
        # 备用保存方法
        np.savetxt(filepath.replace('.pcd', '.txt'), points, fmt='%.6f')

def collect_all_samples():
    """收集所有样本信息，排除有问题的样本"""
    samples = []
    excluded_count = 0
    total_found = 0

    for jaw_dir in ["lower_jaw1", "lower_jaw2", "lower_jaw3"]:
        jaw_path = os.path.join(RAW_DATA_DIR, jaw_dir)
        if not os.path.exists(jaw_path):
            print(f"Warning: {jaw_path} does not exist")
            continue

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

                    samples.append((instance_path, instance_id))

    print(f"📊 样本统计:")
    print(f"   总发现样本: {total_found}")
    print(f"   排除样本: {excluded_count}")
    print(f"   有效样本: {len(samples)}")
    print(f"   排除率: {excluded_count/total_found*100:.1f}%" if total_found > 0 else "   排除率: 0%")

    return samples

def create_batches(samples, batch_size):
    """将样本分批"""
    batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        batches.append(batch)
    return batches

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

    print(f"Dataset metadata saved:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")

def main():
    """主处理函数 - GPU加速版本"""
    start_time = time.time()

    print("🚀 开始GPU加速牙列数据预处理...")

    # 创建输出目录
    ensure_dir(OUTPUT_DIR)

    # 收集所有样本
    print("📋 收集样本信息...")
    all_samples = collect_all_samples()

    if len(all_samples) == 0:
        print("No valid samples found!")
        return

    # 随机打乱并分割数据
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:]

    print(f"📊 数据分割: {len(train_samples)} train, {len(test_samples)} test")

    # 创建批次
    train_batches = create_batches(train_samples, BATCH_SIZE)
    test_batches = create_batches(test_samples, BATCH_SIZE)

    print(f"🔄 批处理配置: {len(train_batches)} train batches, {len(test_batches)} test batches")

    # 处理训练集
    print("⚡ 处理训练集...")
    processed_train = []

    with tqdm(total=len(train_batches), desc="Training batches") as pbar:
        for batch in train_batches:
            batch_result = process_batch_samples((batch, OUTPUT_DIR, "train"))
            processed_train.extend(batch_result)
            pbar.update(1)

    # 处理测试集
    print("⚡ 处理测试集...")
    processed_test = []

    with tqdm(total=len(test_batches), desc="Testing batches") as pbar:
        for batch in test_batches:
            batch_result = process_batch_samples((batch, OUTPUT_DIR, "test"))
            processed_test.extend(batch_result)
            pbar.update(1)

    # 创建数据集元数据
    print("📝 创建数据集元数据...")
    create_dataset_metadata(OUTPUT_DIR, processed_train, processed_test)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n🎉 GPU加速预处理完成！")
    print(f"⏱️  总处理时间: {total_time:.2f} 秒")
    print(f"📊 处理速度: {len(all_samples)/total_time:.2f} 样本/秒")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"✅ 成功处理: {len(processed_train)} train + {len(processed_test)} test = {len(processed_train) + len(processed_test)} 样本")

if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    if USE_GPU:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    main()