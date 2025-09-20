#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
牙列数据预处理脚本 for PointAttN
参考PoinTR的处理方式，为PointAttN模型准备数据

主要功能：
1. 处理data_prepare中的lower_jaw数据
2. 实现下半部分牙齿mask（移除下半部分让模型预测）
3. 分割train/test数据集
4. 生成PointAttN兼容的数据格式
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
from excluded_samples import EXCLUDED_SAMPLE_IDS, is_sample_excluded

# 配置参数
RAW_DATA_DIR = "data_prepare/重标注后完整牙列lower_jaw"
OUTPUT_DIR = "processed_jaw_data"
N_POINTS = 2048  # PointAttN使用的点数
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
N_PARTIAL_VIEWS = 8  # 每个complete对应8个partial

# 下颌牙齿FDI编号（用于确定下半部分）
# FDI编号中，下颌牙齿的编号
LOWER_JAW_TEETH = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)

def read_mesh_and_labels(obj_path, json_path):
    """
    读取mesh和对应的标签
    """
    try:
        # 读取mesh
        mesh = trimesh.load(obj_path)

        # 读取标签
        with open(json_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        # 处理标签，确保是数值型
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

def get_partial_jaw_by_z_coordinate(mesh, labels, mask_ratio=0.5):
    """
    根据Z坐标移除下半部分牙齿（模拟mask下半部分）
    mask_ratio: 需要移除的下半部分比例 (0.5表示移除下半50%)
    """
    vertices = mesh.vertices

    # 计算Z坐标的范围
    z_min = vertices[:, 2].min()
    z_max = vertices[:, 2].max()

    # 计算分割线：移除下半部分
    z_threshold = z_min + (z_max - z_min) * mask_ratio

    # 找出需要保留的面（至少有一个顶点在分割线以上）
    valid_faces = []
    for i, face in enumerate(mesh.faces):
        # 检查这个面的三个顶点是否至少有一个在分割线以上
        face_vertices = vertices[face]
        if np.any(face_vertices[:, 2] > z_threshold):
            valid_faces.append(i)

    if len(valid_faces) == 0:
        # 如果没有有效面，返回原mesh的一个小部分
        valid_faces = list(range(min(100, len(mesh.faces))))

    # 创建新的mesh
    try:
        partial_mesh = mesh.submesh([valid_faces], append=True)
        return partial_mesh
    except:
        # 如果submesh失败，创建一个简单的partial mesh
        return mesh

def get_partial_jaw_by_teeth_removal(mesh, labels, exclude_ratio=0.3):
    """
    通过随机移除一些下颌牙齿来创建partial jaw
    exclude_ratio: 要移除的牙齿比例
    """
    # 找出所有存在的牙齿标签
    unique_labels = np.unique(labels)
    existing_teeth = [label for label in unique_labels if label > 0]

    if len(existing_teeth) == 0:
        return mesh

    # 随机选择要移除的牙齿
    n_exclude = max(1, int(len(existing_teeth) * exclude_ratio))
    exclude_teeth = random.sample(existing_teeth, n_exclude)

    # 找出需要保留的面
    valid_faces = []
    for i, face in enumerate(mesh.faces):
        # 检查这个面的三个顶点对应的标签
        face_labels = labels[face]
        # 如果面的任一顶点属于要移除的牙齿，则移除这个面
        if not np.any(np.isin(face_labels, exclude_teeth)):
            valid_faces.append(i)

    if len(valid_faces) == 0:
        valid_faces = list(range(min(100, len(mesh.faces))))

    try:
        partial_mesh = mesh.submesh([valid_faces], append=True)
        return partial_mesh
    except:
        return mesh

def normalize_mesh(mesh):
    """
    归一化mesh到[-0.5, 0.5]^3
    """
    vertices = mesh.vertices

    # 计算边界框
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    # 计算中心和缩放比例
    center = (bbox_min + bbox_max) / 2
    scale = np.max(bbox_max - bbox_min)

    # 归一化
    mesh.vertices = (vertices - center) / scale

    return mesh

def mesh_to_pointcloud(mesh, n_points=N_POINTS):
    """
    将mesh转换为点云
    """
    try:
        # 使用trimesh采样
        if hasattr(mesh, 'sample'):
            points = mesh.sample(n_points)
        else:
            # 备选方法：均匀采样顶点
            vertices = mesh.vertices
            if len(vertices) >= n_points:
                indices = np.random.choice(len(vertices), n_points, replace=False)
            else:
                # 如果顶点数不够，进行重复采样
                indices = np.random.choice(len(vertices), n_points, replace=True)
            points = vertices[indices]

        return points
    except Exception as e:
        print(f"Error sampling mesh: {e}")
        # 返回随机点作为备选
        return np.random.randn(n_points, 3) * 0.1

def save_pointcloud(points, filepath):
    """
    保存点云为PCD格式
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)

def process_single_sample(args):
    """
    处理单个样本
    """
    sample_dir, instance_id, output_dir, split = args

    try:
        # 构建文件路径
        obj_path = os.path.join(sample_dir, "lower.obj")
        json_path = os.path.join(sample_dir, "modified_seg.json")

        # 检查文件是否存在
        if not os.path.exists(obj_path) or not os.path.exists(json_path):
            print(f"Missing files for {instance_id}")
            return None

        # 读取数据
        mesh, labels = read_mesh_and_labels(obj_path, json_path)
        if mesh is None or labels is None:
            return None

        # 归一化mesh
        mesh = normalize_mesh(mesh)

        # 生成完整点云
        complete_points = mesh_to_pointcloud(mesh, N_POINTS)

        # 保存完整点云
        complete_dir = os.path.join(output_dir, split, "complete", "000")  # 000是lower jaw的类别ID
        ensure_dir(complete_dir)
        complete_path = os.path.join(complete_dir, f"{instance_id}.pcd")
        save_pointcloud(complete_points, complete_path)

        # 生成多个partial点云
        partial_dir = os.path.join(output_dir, split, "partial", "000", instance_id)
        ensure_dir(partial_dir)

        for i in range(N_PARTIAL_VIEWS):
            # 随机选择partial生成方式
            if random.random() < 0.5:
                # 使用Z坐标移除下半部分
                partial_mesh = get_partial_jaw_by_z_coordinate(
                    mesh, labels, mask_ratio=random.uniform(0.3, 0.7)
                )
            else:
                # 使用牙齿移除方式
                partial_mesh = get_partial_jaw_by_teeth_removal(
                    mesh, labels, exclude_ratio=random.uniform(0.2, 0.5)
                )

            # 生成partial点云
            partial_points = mesh_to_pointcloud(partial_mesh, N_POINTS // 4)

            # 保存partial点云
            partial_path = os.path.join(partial_dir, f"{i:02d}.pcd")
            save_pointcloud(partial_points, partial_path)

        return instance_id

    except Exception as e:
        print(f"Error processing {instance_id}: {e}")
        return None

def collect_all_samples():
    """
    收集所有样本信息，排除有问题的样本
    """
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
                # 检查必要文件是否存在
                obj_path = os.path.join(instance_path, "lower.obj")
                json_path = os.path.join(instance_path, "modified_seg.json")

                if os.path.exists(obj_path) and os.path.exists(json_path):
                    total_found += 1

                    # 检查是否在排除列表中
                    if is_sample_excluded(instance_id):
                        excluded_count += 1
                        print(f"❌ 排除样本: {instance_id}")
                        continue

                    samples.append((instance_path, instance_id))

    print(f"📊 样本统计:")
    print(f"   总发现样本: {total_found}")
    print(f"   排除样本: {excluded_count}")
    print(f"   有效样本: {len(samples)}")
    print(f"   排除率: {excluded_count/total_found*100:.1f}%" if total_found > 0 else "   排除率: 0%")

    return samples

def create_dataset_metadata(output_dir, train_samples, test_samples):
    """
    创建数据集元数据文件
    """
    # 创建PCN格式的json文件
    dataset_json = [
        {
            "taxonomy_id": "000",
            "taxonomy_name": "LowerJaw",
            "test": test_samples,
            "train": train_samples,
            "val": []  # 我们只用train和test
        }
    ]

    with open(os.path.join(output_dir, "PCN.json"), 'w') as f:
        json.dump(dataset_json, f, indent=2)

    # 创建类别文件
    with open(os.path.join(output_dir, "category.txt"), 'w') as f:
        f.write("LowerJaw\n")

    print(f"Dataset metadata saved:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")

def main():
    """
    主处理函数
    """
    print("开始处理牙列数据...")

    # 创建输出目录
    ensure_dir(OUTPUT_DIR)

    # 收集所有样本
    print("收集样本信息...")
    all_samples = collect_all_samples()

    if len(all_samples) == 0:
        print("No valid samples found!")
        return

    # 随机打乱并分割数据
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:]

    print(f"数据分割: {len(train_samples)} train, {len(test_samples)} test")

    # 准备处理任务
    tasks = []

    # 训练集任务
    for sample_path, instance_id in train_samples:
        tasks.append((sample_path, instance_id, OUTPUT_DIR, "train"))

    # 测试集任务
    for sample_path, instance_id in test_samples:
        tasks.append((sample_path, instance_id, OUTPUT_DIR, "test"))

    # 并行处理
    print("开始处理样本...")
    num_processes = min(8, len(tasks))  # 使用8个进程或任务数

    processed_train = []
    processed_test = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks), desc="Processing samples") as pbar:
            results = pool.imap_unordered(process_single_sample, tasks)

            for i, result in enumerate(results):
                if result is not None:
                    # 确定这个样本属于哪个split
                    if i < len(train_samples):
                        processed_train.append(result)
                    else:
                        processed_test.append(result)
                pbar.update(1)

    # 创建数据集元数据
    print("创建数据集元数据...")
    create_dataset_metadata(OUTPUT_DIR, processed_train, processed_test)

    print(f"数据处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"成功处理: {len(processed_train)} train + {len(processed_test)} test = {len(processed_train) + len(processed_test)} samples")

    # 显示目录结构
    print(f"\n生成的目录结构:")
    print(f"{OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── complete/000/")
    print(f"  │   └── partial/000/")
    print(f"  ├── test/")
    print(f"  │   ├── complete/000/")
    print(f"  │   └── partial/000/")
    print(f"  ├── PCN.json")
    print(f"  └── category.txt")

if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    main()