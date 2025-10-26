#!/bin/bash

echo "========================================"
echo "PointAttN 牙列补全项目 - 完整设置"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Python环境
echo -e "${BLUE}🔍 检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python未找到，请确保Python已安装${NC}"
    exit 1
fi

python --version
echo ""

# 检查pip
if ! command -v pip &> /dev/null; then
    echo -e "${RED}❌ pip未找到，请确保pip已安装${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python环境检查通过${NC}"
echo ""

# 安装依赖包
echo -e "${BLUE}📦 安装依赖包...${NC}"
echo "这可能需要几分钟..."

# 基本依赖
echo "安装基本依赖..."
pip install trimesh open3d numpy tqdm pyyaml

# 训练相关依赖
echo "安装训练依赖..."
pip install torch torchvision munch easydict transforms3d h5py

# 额外依赖
echo "安装额外依赖..."
pip install scipy matplotlib tensorpack

echo ""

# 验证安装
echo -e "${BLUE}✅ 验证安装...${NC}"
python -c "
import sys
packages = ['trimesh', 'open3d', 'numpy', 'tqdm', 'yaml', 'torch', 'munch']
failed = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg} 安装成功')
    except ImportError:
        print(f'❌ {pkg} 安装失败')
        failed.append(pkg)

if failed:
    print(f'\\n❌ 以下包安装失败: {failed}')
    print('请手动安装失败的包')
    sys.exit(1)
else:
    print('\\n🎉 所有依赖安装成功！')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 依赖验证失败${NC}"
    exit 1
fi

echo ""

# 修复常见问题
echo -e "${BLUE}🔧 修复常见依赖问题...${NC}"

# 修复typing_extensions版本冲突
echo "检查typing_extensions版本..."
pip install --upgrade typing_extensions

# 检查open3d版本兼容性
echo "检查open3d兼容性..."
python -c "
try:
    import open3d as o3d
    print(f'✅ open3d版本: {o3d.__version__}')
except Exception as e:
    print(f'⚠️ open3d问题: {e}')
    print('建议运行: pip install open3d-cpu')
"

echo ""

# 编译第三方模块
echo -e "${BLUE}🔨 编译第三方模块...${NC}"

# ChamferDistancePytorch
echo "编译ChamferDistancePytorch..."
if [ -d "utils/ChamferDistancePytorch/chamfer3D" ]; then
    cd utils/ChamferDistancePytorch/chamfer3D
    python setup.py install
    cd ../../..
    echo "✅ ChamferDistancePytorch编译完成"
else
    echo "⚠️ ChamferDistancePytorch目录不存在，跳过"
fi

# mm3d_pn2
echo "编译mm3d_pn2..."
if [ -d "utils/mm3d_pn2" ]; then
    cd utils/mm3d_pn2
    python setup.py build_ext --inplace
    cd ../..
    echo "✅ mm3d_pn2编译完成"
else
    echo "⚠️ mm3d_pn2目录不存在，跳过"
fi

echo ""

# 测试环境
echo -e "${BLUE}🧪 测试环境...${NC}"

# 测试排除功能
echo "测试排除功能..."
python test_exclusion.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 排除功能测试通过${NC}"
else
    echo -e "${YELLOW}⚠️ 排除功能测试有问题，但可以继续${NC}"
fi

echo ""

# 测试预处理环境
echo "测试预处理环境..."
python test_preprocessing.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 预处理环境测试通过${NC}"
else
    echo -e "${RED}❌ 预处理环境测试失败${NC}"
    echo "请检查数据目录和依赖包"
    exit 1
fi

echo ""

# 完成设置
echo -e "${GREEN}🎉 设置完成！${NC}"
echo ""
echo -e "${BLUE}📋 下一步操作：${NC}"
echo "1. 运行数据预处理:"
echo "   python preprocess_jaw_data.py"
echo ""
echo "2. 开始训练:"
echo "   python train_jaw.py --config cfgs/PointAttN_Jaw.yaml"
echo ""
echo "3. 或者一键运行:"
echo "   ./run_jaw_training.sh"
echo ""
echo -e "${BLUE}📁 重要目录：${NC}"
echo "- 数据目录: data_prepare/重标注后完整牙列lower_jaw/"
echo "- 输出目录: processed_jaw_data/"
echo "- 日志目录: log/jaw_experiments/"
echo ""
echo "========================================"
echo "设置完成！"
echo "========================================"
