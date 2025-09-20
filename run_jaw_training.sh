#!/bin/bash

# PointAttN 牙列训练一键运行脚本
# 这个脚本会依次执行数据预处理和模型训练

echo "========================================"
echo "PointAttN 牙列补全训练流程"
echo "========================================"

# 检查Python环境
echo "🔍 检查Python环境..."
if ! command -v python &> /dev/null; then
    echo "❌ Python未找到，请确保Python已安装"
    exit 1
fi

# 检查必要的包
echo "🔍 检查依赖包..."
python -c "import trimesh, open3d, numpy, torch, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少必要的依赖包，请运行："
    echo "pip install trimesh open3d numpy torch pyyaml tqdm"
    exit 1
fi

echo "✅ 环境检查通过"

# 步骤1: 测试预处理环境
echo ""
echo "📋 步骤1: 测试预处理环境..."
python test_preprocessing.py
if [ $? -ne 0 ]; then
    echo "❌ 预处理测试失败，请检查数据和环境"
    exit 1
fi

echo "✅ 预处理测试通过"

# 询问是否继续
echo ""
read -p "⏳ 预处理测试通过，是否继续数据预处理？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 用户取消"
    exit 1
fi

# 步骤2: 数据预处理
echo ""
echo "📋 步骤2: 运行数据预处理..."
echo "这可能需要几分钟到几小时，取决于数据量..."

python preprocess_jaw_data.py
if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi

echo "✅ 数据预处理完成"

# 检查预处理结果
if [ ! -d "processed_jaw_data" ]; then
    echo "❌ 预处理输出目录不存在"
    exit 1
fi

# 统计处理结果
echo ""
echo "📊 预处理统计:"
train_complete=$(find processed_jaw_data/train/complete -name "*.pcd" 2>/dev/null | wc -l)
train_partial=$(find processed_jaw_data/train/partial -name "*.pcd" 2>/dev/null | wc -l)
test_complete=$(find processed_jaw_data/test/complete -name "*.pcd" 2>/dev/null | wc -l)
test_partial=$(find processed_jaw_data/test/partial -name "*.pcd" 2>/dev/null | wc -l)

echo "   训练集完整点云: $train_complete"
echo "   训练集部分点云: $train_partial"
echo "   测试集完整点云: $test_complete"
echo "   测试集部分点云: $test_partial"

# 询问是否开始训练
echo ""
read -p "⏳ 数据预处理完成，是否开始模型训练？(y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "ℹ️  数据预处理已完成，可以稍后手动运行："
    echo "python train_jaw.py --config cfgs/PointAttN_Jaw.yaml"
    exit 0
fi

# 步骤3: 模型训练
echo ""
echo "📋 步骤3: 开始模型训练..."
echo "这将需要很长时间，建议在后台运行..."

# 创建日志目录
mkdir -p log/jaw_experiments

# 开始训练
echo "🚀 启动训练进程..."
python train_jaw.py --config cfgs/PointAttN_Jaw.yaml

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 训练流程完成！"
    echo "📁 检查以下目录查看结果："
    echo "   - processed_jaw_data/ (预处理数据)"
    echo "   - log/jaw_experiments/ (训练日志和模型)"
else
    echo ""
    echo "❌ 训练过程中出现错误"
    echo "📋 检查日志文件: log/jaw_experiments/jaw_lower.log"
fi

echo ""
echo "========================================"
echo "流程结束"
echo "========================================"