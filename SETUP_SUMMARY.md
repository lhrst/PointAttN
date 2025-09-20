# PointAttN 牙列训练 - 完整设置总结

## 🎯 目标
训练PointAttN模型来预测牙列下半部分的缺失牙齿。

## 📁 创建的文件

### 数据预处理
- **`preprocess_jaw_data.py`** - 主要预处理脚本
- **`test_preprocessing.py`** - 环境和数据测试脚本
- **`excluded_samples.py`** - 排除样本列表（107个有问题的样本）
- **`test_exclusion.py`** - 排除功能测试脚本

### 配置文件
- **`cfgs/PointAttN_Jaw.yaml`** - 牙列数据专用配置文件

### 训练脚本
- **`train_jaw.py`** - 牙列数据专用训练脚本
- **`run_jaw_training.sh`** - 一键运行脚本

### 文档
- **`README_数据预处理.md`** - 详细使用说明
- **`SETUP_SUMMARY.md`** - 本文档

## 🚀 快速开始

### 一键运行（推荐）
```bash
./run_jaw_training.sh
```

### 分步运行
```bash
# 1. 测试排除功能
python test_exclusion.py

# 2. 测试环境
python test_preprocessing.py

# 3. 数据预处理
python preprocess_jaw_data.py

# 4. 开始训练
python train_jaw.py --config cfgs/PointAttN_Jaw.yaml
```

## ⚙️ 关键配置参数

### 数据预处理参数 (preprocess_jaw_data.py)
```python
N_POINTS = 2048          # 完整点云点数
TRAIN_RATIO = 0.8        # 训练集比例
TEST_RATIO = 0.2         # 测试集比例
N_PARTIAL_VIEWS = 8      # 每个complete对应的partial数量
```

### 训练参数 (cfgs/PointAttN_Jaw.yaml)
```yaml
batch_size: 16           # 批次大小
num_points: 2048         # 点云点数
nepoch: 300              # 训练轮数
lr: 0.0005              # 学习率
dataset: 'pcn'          # 数据集类型
pcnpath: ./processed_jaw_data  # 数据路径
```

## 📂 数据流程

### 输入数据结构
```
data_prepare/重标注后完整牙列lower_jaw/
├── lower_jaw1/
│   ├── instance_id/
│   │   ├── lower.obj
│   │   ├── modified_seg.json
│   │   └── ...
│   └── ...
├── lower_jaw2/
└── lower_jaw3/
```

### 输出数据结构
```
processed_jaw_data/
├── train/
│   ├── complete/000/
│   └── partial/000/
├── test/
│   ├── complete/000/
│   └── partial/000/
├── PCN.json
└── category.txt
```

## 🎯 Mask策略

### 1. Z坐标分割（主要用于下半部分遮挡）
- 根据Z坐标移除下半部分
- mask_ratio: 0.3-0.7 (随机)
- 模拟真实的下半部分缺失

### 2. 随机牙齿移除
- 随机选择并移除特定牙齿
- exclude_ratio: 0.2-0.5 (随机)
- 增加数据多样性

## 🔧 自定义配置

### 修改mask策略
在 `preprocess_jaw_data.py` 中：
```python
# 调整Z坐标分割比例
partial_mesh = get_partial_jaw_by_z_coordinate(
    mesh, labels, mask_ratio=0.5  # 改为你想要的比例
)

# 调整牙齿移除比例
partial_mesh = get_partial_jaw_by_teeth_removal(
    mesh, labels, exclude_ratio=0.3  # 改为你想要的比例
)
```

### 修改训练参数
在 `cfgs/PointAttN_Jaw.yaml` 中：
```yaml
# 调整批次大小（根据GPU内存）
batch_size: 8  # 或16, 32

# 调整学习率
lr: 0.001  # 或0.0001

# 调整训练轮数
nepoch: 500  # 或更多
```

## 📊 监控训练

### 查看训练日志
```bash
tail -f log/jaw_experiments/jaw_lower.log
```

### 查看保存的模型
```bash
ls log/jaw_experiments/
```

### 可视化结果
训练过程中会在 `log/jaw_experiments/` 目录下保存：
- 训练日志
- 模型检查点
- 最佳模型 (`best_model_jaw_lower.pth`)

## ⚠️ 注意事项

1. **内存需求**: 建议8GB+内存，GPU训练
2. **存储空间**: 根据数据量预留足够空间
3. **训练时间**: 取决于数据量和硬件，可能需要数小时到数天
4. **数据质量**: 确保OBJ文件和JSON标签文件完整且对应

## 🆘 故障排除

### 常见问题
1. **CUDA错误**: 检查GPU驱动和PyTorch版本
2. **内存不足**: 减小batch_size或num_workers
3. **数据加载错误**: 运行test_preprocessing.py检查
4. **训练不收敛**: 调整学习率或检查数据质量

### 获取帮助
- 查看详细文档: `README_数据预处理.md`
- 检查训练日志: `log/jaw_experiments/jaw_lower.log`
- 运行测试脚本: `python test_preprocessing.py`

## 🎉 完成！

如果一切设置正确，运行 `./run_jaw_training.sh` 就可以开始训练你的牙列补全模型了！
