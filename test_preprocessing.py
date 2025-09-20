#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据预处理脚本的快速版本
用于验证代码是否能正常运行
"""

import os
import json
import trimesh
import numpy as np
import open3d as o3d
from pathlib import Path
import random

def test_single_sample():
    """
    测试处理单个样本
    """
    # 找到第一个可用的样本进行测试
    RAW_DATA_DIR = "data_prepare/重标注后完整牙列lower_jaw"
    test_sample = None

    for jaw_dir in ["lower_jaw1", "lower_jaw2", "lower_jaw3"]:
        jaw_path = os.path.join(RAW_DATA_DIR, jaw_dir)
        if not os.path.exists(jaw_path):
            continue

        for instance_id in os.listdir(jaw_path):
            instance_path = os.path.join(jaw_path, instance_id)
            if os.path.isdir(instance_path):
                obj_path = os.path.join(instance_path, "lower.obj")
                json_path = os.path.join(instance_path, "modified_seg.json")

                if os.path.exists(obj_path) and os.path.exists(json_path):
                    test_sample = (instance_path, instance_id, obj_path, json_path)
                    break

        if test_sample:
            break

    if not test_sample:
        print("❌ 没有找到可用的测试样本")
        return False

    instance_path, instance_id, obj_path, json_path = test_sample
    print(f"✅ 找到测试样本: {instance_id}")

    try:
        # 测试读取mesh
        print("📖 测试读取mesh...")
        mesh = trimesh.load(obj_path)
        print(f"   Mesh顶点数: {len(mesh.vertices)}")
        print(f"   Mesh面数: {len(mesh.faces)}")

        # 测试读取标签
        print("📖 测试读取标签...")
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

        labels = np.array(labels)
        unique_labels = np.unique(labels)
        print(f"   标签数量: {len(labels)}")
        print(f"   唯一标签: {unique_labels}")

        # 测试归一化
        print("🔄 测试归一化...")
        vertices = mesh.vertices
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        scale = np.max(bbox_max - bbox_min)
        mesh.vertices = (vertices - center) / scale

        normalized_min = mesh.vertices.min(axis=0)
        normalized_max = mesh.vertices.max(axis=0)
        print(f"   归一化前范围: {bbox_min} to {bbox_max}")
        print(f"   归一化后范围: {normalized_min} to {normalized_max}")

        # 测试点云采样
        print("☁️ 测试点云采样...")
        try:
            points = mesh.sample(2048)
            print(f"   采样点数: {points.shape}")
        except Exception as e:
            print(f"   采样失败，使用顶点: {e}")
            vertices = mesh.vertices
            if len(vertices) >= 2048:
                indices = np.random.choice(len(vertices), 2048, replace=False)
            else:
                indices = np.random.choice(len(vertices), 2048, replace=True)
            points = vertices[indices]
            print(f"   使用顶点采样: {points.shape}")

        # 测试partial生成
        print("✂️ 测试partial生成...")

        # 方法1：Z坐标分割
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        z_threshold = z_min + (z_max - z_min) * 0.5

        valid_faces = []
        for i, face in enumerate(mesh.faces):
            face_vertices = mesh.vertices[face]
            if np.any(face_vertices[:, 2] > z_threshold):
                valid_faces.append(i)

        print(f"   Z坐标分割保留面数: {len(valid_faces)}/{len(mesh.faces)}")

        # 方法2：标签分割
        existing_teeth = [label for label in unique_labels if label > 0]
        if len(existing_teeth) > 0:
            n_exclude = max(1, int(len(existing_teeth) * 0.3))
            exclude_teeth = random.sample(existing_teeth, n_exclude)

            valid_faces_teeth = []
            for i, face in enumerate(mesh.faces):
                face_labels = labels[face]
                if not np.any(np.isin(face_labels, exclude_teeth)):
                    valid_faces_teeth.append(i)

            print(f"   牙齿分割保留面数: {len(valid_faces_teeth)}/{len(mesh.faces)}")
            print(f"   移除的牙齿: {exclude_teeth}")

        # 测试保存点云
        print("💾 测试保存点云...")
        test_output_dir = "test_output"
        Path(test_output_dir).mkdir(exist_ok=True)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        test_pcd_path = os.path.join(test_output_dir, "test_sample.pcd")
        success = o3d.io.write_point_cloud(test_pcd_path, pcd)

        if success:
            print(f"   ✅ 成功保存测试点云: {test_pcd_path}")
        else:
            print(f"   ❌ 保存点云失败")

        print("\n🎉 所有测试通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """
    检查依赖项
    """
    print("🔍 检查依赖项...")

    required_packages = {
        'trimesh': 'trimesh',
        'open3d': 'open3d',
        'numpy': 'numpy',
        'tqdm': 'tqdm'
    }

    missing_packages = []

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - 需要安装")
            missing_packages.append(package_name)

    if missing_packages:
        print(f"\n安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def check_data_structure():
    """
    检查数据结构
    """
    print("📁 检查数据结构...")

    RAW_DATA_DIR = "data_prepare/重标注后完整牙列lower_jaw"

    if not os.path.exists(RAW_DATA_DIR):
        print(f"   ❌ 数据目录不存在: {RAW_DATA_DIR}")
        return False

    print(f"   ✅ 数据目录存在: {RAW_DATA_DIR}")

    # 检查子目录
    jaw_dirs = ["lower_jaw1", "lower_jaw2", "lower_jaw3"]
    found_dirs = []
    total_samples = 0

    for jaw_dir in jaw_dirs:
        jaw_path = os.path.join(RAW_DATA_DIR, jaw_dir)
        if os.path.exists(jaw_path):
            found_dirs.append(jaw_dir)
            samples = [d for d in os.listdir(jaw_path) if os.path.isdir(os.path.join(jaw_path, d))]
            total_samples += len(samples)
            print(f"   ✅ {jaw_dir}: {len(samples)} 样本")
        else:
            print(f"   ⚠️ {jaw_dir}: 不存在")

    if len(found_dirs) == 0:
        print("   ❌ 没有找到任何jaw目录")
        return False

    print(f"   📊 总计: {total_samples} 样本 in {len(found_dirs)} 目录")
    return total_samples > 0

def main():
    """
    主测试函数
    """
    print("="*50)
    print("🧪 PointAttN 数据预处理测试")
    print("="*50)

    # 检查依赖项
    if not check_dependencies():
        return

    print()

    # 检查数据结构
    if not check_data_structure():
        return

    print()

    # 测试单个样本处理
    if not test_single_sample():
        return

    print()
    print("="*50)
    print("🎉 所有测试通过！可以运行完整的预处理脚本了。")
    print("运行命令: python preprocess_jaw_data.py")
    print("="*50)

if __name__ == "__main__":
    main()