# PointAttN - 牙列补全项目

基于PointAttN的点云补全模型，专门用于牙列下半部分缺失的预测和补全。

## 🚀 快速开始

### 完整设置（推荐）
```bash
# 一键安装所有依赖并测试环境
./setup.sh
```

### 一键训练
```bash
# 完整训练流程（预处理 + 训练）
./run_jaw_training.sh
```

### 分步运行
```bash
# 1. 测试环境
python test_preprocessing.py

# 2. 数据预处理
python preprocess_jaw_data.py

# 3. 开始训练
python train_jaw.py --config cfgs/PointAttN_Jaw.yaml
```

## 📁 项目结构

```
PointAttN/
├── setup.sh                 # 完整环境设置脚本
├── run_jaw_training.sh      # 一键训练脚本
├── preprocess_jaw_data.py   # 数据预处理脚本（快速版本）
├── train_jaw.py             # 牙列专用训练脚本
├── test_preprocessing.py     # 预处理测试脚本
├── test_exclusion.py        # 排除功能测试脚本
├── excluded_samples.py      # 排除样本列表
├── cfgs/
│   ├── PointAttN_Jaw.yaml  # 牙列训练配置
│   └── PointAttN.yaml      # 原始配置
├── models/                  # 模型定义
├── utils/                   # 工具函数
└── dataset.py              # 数据加载器
```

## 🔧 环境要求

### 自动安装（推荐）
```bash
./setup.sh  # 自动安装所有依赖并测试环境
```

### 手动安装
```bash
pip install -r requirements.txt

# 编译第三方模块
cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install

cd utils/mm3d_pn2
python setup.py build_ext --inplace
```

## 📊 数据预处理

预处理脚本会将牙列OBJ文件转换为点云格式，并生成训练所需的完整和部分点云数据。

**主要功能：**
- 自动排除有问题的样本（107个）
- Z坐标分割生成部分点云（模拟下半部分缺失）
- 随机牙齿移除增加数据多样性
- 8:2分割训练集和测试集

**输出结构：**
```
processed_jaw_data/
├── train/
│   ├── complete/000/        # 完整点云
│   └── partial/000/        # 部分点云
├── test/
│   ├── complete/000/
│   └── partial/000/
├── PCN.json                # 数据集元数据
└── category.txt            # 类别信息
```

## 🎯 训练配置

主要训练参数（cfgs/PointAttN_Jaw.yaml）：
- `batch_size: 16` - 批次大小
- `num_points: 2048` - 点云点数
- `nepoch: 300` - 训练轮数
- `lr: 0.0005` - 学习率
- `pcnpath: ./processed_jaw_data` - 数据路径

## 📈 性能优化

- **多进程处理**: 使用16个进程并行处理
- **快速采样**: 优化的点云采样算法
- **内存优化**: 批量处理和流式处理
- **GPU加速**: 支持CUDA加速（如果可用）

## 🧪 测试和验证

### 预处理测试
```bash
python test_preprocessing.py  # 测试环境和数据
python test_exclusion.py      # 测试排除功能
```

### 模型测试
```bash
python test_pcn.py -c PointAttN.yaml    # PCN数据集测试
python test_c3d.py -c PointAttN.yaml    # Completion3D测试
```

## 📚 原始论文

PointAttN: You Only Need Attention for Point Cloud Completion
- **会议**: AAAI 2024
- **性能**: Completion3D CD=6.63, PCN CD=6.86

## 🆘 故障排除

### open3d依赖冲突
如果遇到 `AttributeError: module 'typing_extensions' has no attribute 'TypeVar'` 错误：

```bash
# 手动修复
pip install --upgrade typing_extensions
pip uninstall open3d -y
pip install open3d==0.13.0
# 如果仍有问题，使用CPU版本
pip install open3d-cpu
```

### 其他常见问题
1. **CUDA错误**: 检查GPU驱动和PyTorch版本
2. **内存不足**: 减小batch_size或num_workers
3. **数据加载错误**: 运行test_preprocessing.py检查
4. **训练不收敛**: 调整学习率或检查数据质量

### 获取帮助
- 查看详细文档: README.md
- 检查训练日志: log/jaw_experiments/jaw_lower.log
- 运行测试脚本: python test_preprocessing.py

