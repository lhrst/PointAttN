# PointAttN 牙列数据预处理指南

## 概述

这个预处理脚本将处理 `data_prepare` 目录中的牙列数据，生成适用于 PointAttN 模型训练的数据集。主要功能包括：

1. **数据格式转换**: 将 OBJ 网格数据转换为点云格式
2. **下半部分遮挡**: 实现牙列下半部分的 mask，让模型预测被遮挡的部分
3. **数据分割**: 自动分割训练集和测试集
4. **标准化处理**: 归一化点云到 [-0.5, 0.5]³ 空间

## 文件说明

- `preprocess_jaw_data.py` - 主要的数据预处理脚本
- `test_preprocessing.py` - 测试脚本，用于验证环境和数据
- `README_数据预处理.md` - 本说明文档

## 环境要求

### Python 版本
- Python 3.7+

### 依赖包
```bash
pip install trimesh open3d numpy tqdm
```

或者如果你有 requirements.txt:
```bash
pip install -r requirements.txt
```

## 数据结构要求

确保你的数据目录结构如下：
```
data_prepare/
└── 重标注后完整牙列lower_jaw/
    ├── lower_jaw1/
    │   ├── 00A231F8-98B4-40FB-B420-5DBB989F35BA/
    │   │   ├── lower.obj
    │   │   ├── modified_seg.json
    │   │   └── ...
    │   └── ...
    ├── lower_jaw2/
    └── lower_jaw3/
```

每个样本目录应包含：
- `lower.obj` - 牙列的 3D 网格文件
- `modified_seg.json` - 顶点标签文件，包含牙齿分割信息

## 使用步骤

### 方法一：一键运行（推荐）

使用提供的自动化脚本：

```bash
./run_jaw_training.sh
```

这个脚本会自动执行：
1. 环境检查
2. 数据预处理测试
3. 完整数据预处理
4. 模型训练

### 方法二：分步执行

#### 1. 测试环境（推荐）

在运行完整预处理之前，先运行测试脚本：

```bash
python test_preprocessing.py
```

这将检查：
- ✅ 依赖包是否正确安装
- ✅ 数据目录结构是否正确
- ✅ 能否正确读取和处理样本数据
- ✅ 点云生成和保存功能

#### 2. 运行预处理

如果测试通过，运行主预处理脚本：

```bash
python preprocess_jaw_data.py
```

#### 3. 开始训练

使用专门的牙列训练脚本：

```bash
python train_jaw.py --config cfgs/PointAttN_Jaw.yaml
```

或者使用原始训练脚本：

```bash
python train.py --config cfgs/PointAttN_Jaw.yaml
```

### 3. 处理过程

脚本将执行以下步骤：

1. **数据收集**: 扫描所有 jaw 目录，收集有效样本
2. **数据分割**: 按 8:2 比例分割训练集和测试集
3. **并行处理**: 使用多进程处理样本（默认8个进程）
4. **点云生成**:
   - 生成完整牙列点云（2048点）
   - 生成8个不同的部分点云（512点）
5. **保存结果**: 保存为 PCD 格式，生成元数据文件

## 输出结果

处理完成后，将生成以下目录结构：

```
processed_jaw_data/
├── train/
│   ├── complete/
│   │   └── 000/          # 类别ID（Lower Jaw）
│   │       ├── instance1.pcd
│   │       ├── instance2.pcd
│   │       └── ...
│   └── partial/
│       └── 000/
│           ├── instance1/
│           │   ├── 00.pcd  # 第1个partial view
│           │   ├── 01.pcd  # 第2个partial view
│           │   └── ...     # 共8个
│           └── ...
├── test/
│   ├── complete/
│   └── partial/
├── PCN.json             # 数据集元数据
└── category.txt         # 类别信息
```

## Mask 策略说明

脚本实现了两种 mask 策略来生成部分牙列：

### 1. Z坐标分割 (推荐用于下半部分遮挡)
- 根据 Z 坐标移除下半部分牙齿
- `mask_ratio` 参数控制移除比例（0.3-0.7随机）
- 模拟真实的下半部分缺失场景

### 2. 随机牙齿移除
- 随机选择并移除特定牙齿
- `exclude_ratio` 参数控制移除的牙齿比例（0.2-0.5随机）
- 增加数据多样性

## 配置参数

可以在 `preprocess_jaw_data.py` 中调整以下参数：

```python
N_POINTS = 2048          # 完整点云的点数
TRAIN_RATIO = 0.8        # 训练集比例
TEST_RATIO = 0.2         # 测试集比例
N_PARTIAL_VIEWS = 8      # 每个complete对应的partial数量
```

## 与 PointAttN 集成

处理完成后，修改 PointAttN 的训练配置：

1. **数据路径配置**:
   ```python
   # 在训练脚本中
   args.pcnpath = "processed_jaw_data"
   ```

2. **数据集配置**:
   ```python
   # 使用 PCN_pcd 数据加载器
   dataset = PCN_pcd(args.pcnpath, prefix="train")
   dataset_test = PCN_pcd(args.pcnpath, prefix="test")
   ```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少并行进程数：修改 `num_processes = 4`
   - 减少点云密度：修改 `N_POINTS = 1024`

2. **文件读取错误**
   - 检查文件编码：确保 JSON 文件是 UTF-8 编码
   - 检查文件权限：确保有读取权限

3. **网格处理错误**
   - 检查 OBJ 文件完整性
   - 确保 trimesh 版本兼容

### 日志和调试

脚本会输出详细的处理日志：
- ✅ 成功处理的样本数量
- ❌ 失败的样本和错误信息
- 📊 最终统计信息

## 性能建议

- **硬件要求**: 建议 8GB+ 内存，多核 CPU
- **存储空间**: 根据样本数量，预留足够磁盘空间
- **处理时间**: 取决于样本数量，通常每个样本 5-10 秒

## 验证结果

处理完成后，可以：

1. **检查文件数量**:
   ```bash
   find processed_jaw_data -name "*.pcd" | wc -l
   ```

2. **验证点云**:
   ```python
   import open3d as o3d
   pcd = o3d.io.read_point_cloud("processed_jaw_data/train/complete/000/sample.pcd")
   print(f"Points: {len(pcd.points)}")
   ```

3. **可视化检查**:
   ```python
   import open3d as o3d
   pcd = o3d.io.read_point_cloud("processed_jaw_data/train/complete/000/sample.pcd")
   o3d.visualization.draw_geometries([pcd])
   ```

## 支持和反馈

如果遇到问题，请检查：
1. 是否按照环境要求安装了所有依赖
2. 数据目录结构是否正确
3. 是否有足够的磁盘空间和内存

预处理完成后，就可以开始训练 PointAttN 模型了！