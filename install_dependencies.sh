#!/bin/bash

echo "🔧 安装 PointAttN 牙列数据预处理依赖包"
echo "============================================"

# 检查Python环境
echo "📋 检查Python环境..."
python --version
echo ""

# 安装基本依赖
echo "📦 安装基本依赖包..."
pip install trimesh open3d numpy tqdm pyyaml

# 安装训练相关依赖
echo "📦 安装训练依赖..."
pip install torch torchvision munch easydict transforms3d h5py

# 安装额外的几何处理依赖
echo "📦 安装额外依赖..."
pip install scipy matplotlib tensorpack

# 验证安装
echo ""
echo "✅ 验证安装..."
python -c "import trimesh; print('✅ trimesh 安装成功')"
python -c "import open3d; print('✅ open3d 安装成功')"
python -c "import numpy; print('✅ numpy 安装成功')"
python -c "import tqdm; print('✅ tqdm 安装成功')"
python -c "import yaml; print('✅ pyyaml 安装成功')"

echo ""
echo "🎉 依赖安装完成！"
echo "现在可以运行："
echo "python test_exclusion.py"
echo "python preprocess_jaw_data.py"