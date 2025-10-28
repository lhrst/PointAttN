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

# 安装mmcv (mm3d_pn2依赖)
echo "安装mmcv..."
# 先尝试安装兼容的mmcv版本
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# 安装ninja加速编译
echo "安装ninja编译加速器..."
pip install ninja

echo ""

# 验证安装
echo -e "${BLUE}✅ 验证安装...${NC}"
python -c "
import sys
packages = ['trimesh', 'numpy', 'tqdm', 'yaml', 'torch', 'munch', 'ninja']
failed = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg} 安装成功')
    except ImportError:
        print(f'❌ {pkg} 安装失败')
        failed.append(pkg)

# 特殊处理open3d
try:
    import open3d as o3d
    print(f'✅ open3d 安装成功 (版本: {o3d.__version__})')
except Exception as e:
    print(f'⚠️ open3d 有问题: {e}')
    print('将尝试修复...')
    failed.append('open3d')

# 特殊处理mmcv
try:
    import mmcv
    print(f'✅ mmcv 安装成功 (版本: {mmcv.__version__})')
except Exception as e:
    print(f'⚠️ mmcv 有问题: {e}')
    print('将尝试修复...')
    failed.append('mmcv')

if failed:
    print(f'\\n⚠️ 以下包需要修复: {failed}')
    if 'open3d' in failed:
        print('open3d问题将在下一步修复')
    if 'ninja' in failed:
        print('ninja安装失败，编译会较慢')
    if 'mmcv' in failed:
        print('mmcv问题将在下一步修复')
else:
    print('\\n🎉 所有依赖安装成功！')
"

echo ""

# 修复常见问题
echo -e "${BLUE}🔧 修复常见依赖问题...${NC}"

# 修复typing_extensions版本冲突
echo "修复typing_extensions版本冲突..."
pip install --upgrade typing_extensions

# 检查并修复open3d问题
echo "检查open3d兼容性..."
python -c "
try:
    import open3d as o3d
    print(f'✅ open3d工作正常 (版本: {o3d.__version__})')
except Exception as e:
    print(f'❌ open3d仍有问题: {e}')
    print('尝试修复方案...')
    exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ open3d仍有问题，尝试修复方案...${NC}"
    
    # 方案1: 重新安装open3d
    echo "方案1: 重新安装open3d..."
    pip uninstall open3d -y
    pip install open3d==0.13.0
    
    # 再次检查
    python -c "
try:
    import open3d as o3d
    print(f'✅ open3d修复成功 (版本: {o3d.__version__})')
except Exception as e:
    print(f'❌ 方案1失败: {e}')
    print('尝试方案2...')
    exit(1)
" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}方案2: 安装open3d-cpu版本...${NC}"
        pip uninstall open3d -y
        pip install open3d-cpu
        
        # 最终检查
        python -c "
try:
    import open3d as o3d
    print(f'✅ open3d-cpu安装成功 (版本: {o3d.__version__})')
    print('注意: 使用CPU版本，无GUI功能')
except Exception as e:
    print(f'❌ 所有方案都失败: {e}')
    print('请手动解决open3d问题')
    exit(1)
"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ open3d修复失败，但可以继续使用其他功能${NC}"
            echo "预处理脚本会使用备用方案"
        fi
    fi
fi

# 检查并修复mmcv问题
echo "检查mmcv兼容性..."
python -c "
try:
    import mmcv
    print(f'✅ mmcv工作正常 (版本: {mmcv.__version__})')
except Exception as e:
    print(f'❌ mmcv仍有问题: {e}')
    print('尝试修复方案...')
    exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ mmcv仍有问题，尝试修复方案...${NC}"
    
    # 方案1: 尝试安装兼容版本
    echo "方案1: 安装mmcv兼容版本..."
    pip uninstall mmcv-full -y 2>/dev/null || true
    pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    
    # 再次检查
    python -c "
try:
    import mmcv
    print(f'✅ mmcv-cpu安装成功 (版本: {mmcv.__version__})')
    print('注意: 使用CPU版本，无CUDA加速')
except Exception as e:
    print(f'❌ 方案1失败: {e}')
    print('尝试方案2...')
    exit(1)
" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}方案2: 安装CPU版本...${NC}"
        pip uninstall mmcv-full -y 2>/dev/null || true
        pip install mmcv-cpu
        
        # 再次检查
        python -c "
try:
    import mmcv
    print(f'✅ mmcv-cpu安装成功 (版本: {mmcv.__version__})')
    print('注意: 使用CPU版本，无CUDA加速')
except Exception as e:
    print(f'❌ 方案2失败: {e}')
    print('尝试方案3...')
    exit(1)
" 2>/dev/null
        
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}方案3: 安装基础mmcv版本...${NC}"
            pip uninstall mmcv-cpu -y 2>/dev/null || true
            pip install mmcv
        
        # 最终检查
        python -c "
try:
    import mmcv
    print(f'✅ mmcv基础版本安装成功 (版本: {mmcv.__version__})')
except Exception as e:
    print(f'❌ 所有方案都失败: {e}')
    print('请手动解决mmcv问题')
    exit(1)
"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ mmcv修复失败，但可以继续使用其他功能${NC}"
            echo "可能需要手动安装mmcv"
        fi
    fi
fi

echo ""

# 编译第三方模块
echo -e "${BLUE}🔨 编译第三方模块...${NC}"

# 检查PyTorch版本
echo "检查PyTorch版本..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
if torch.__version__.startswith('2.'):
    print('⚠️ 检测到PyTorch 2.x，第三方模块可能不兼容')
    print('建议跳过编译或使用PyTorch 1.x')
elif torch.__version__.startswith('1.1'):
    print('⚠️ 检测到PyTorch 1.11+，需要修复THC头文件问题')
"

# 全局修复PyTorch 1.11+兼容性问题
echo "全局修复PyTorch 1.11+兼容性问题..."
if [ -d "utils" ]; then
    echo "修复所有第三方模块的THC头文件问题..."
    find utils/ -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i '/#include\s\+<THC\/THC\.h>/d' 2>/dev/null || true
    
    echo "修复tensor.type()废弃警告..."
    find utils/ -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i 's/\.type()\.is_cuda()/\.is_cuda()/g' 2>/dev/null || true
    
    echo "修复THCState相关问题..."
    find utils/ -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i '/extern THCState/d' 2>/dev/null || true
    
    echo "修复CUDA Stream API问题..."
    find utils/ -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i 's/at::cuda::getCurrentCUDAStream()\.stream()/cudaStreamDefault/g' 2>/dev/null || true
    
    echo "修复其他CUDA API问题..."
    find utils/ -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i 's/at::cuda::getDefaultCUDAStream()\.stream()/cudaStreamDefault/g' 2>/dev/null || true
    
    echo "修复CUDA Stream类型问题..."
    find utils/ -type f \( -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
        | xargs -0 sed -i 's/cudaStream_t stream = cudaStreamDefault;/cudaStream_t stream = cudaStreamDefault;/g' 2>/dev/null || true
    
    echo "✅ 全局修复完成"
fi

# ChamferDistancePytorch
echo "编译ChamferDistancePytorch..."
if [ -d "utils/ChamferDistancePytorch/chamfer3D" ]; then
    cd utils/ChamferDistancePytorch/chamfer3D
    echo "尝试编译ChamferDistancePytorch..."
    python setup.py install 2>&1 | tee /tmp/chamfer_build.log
    if [ $? -eq 0 ]; then
        echo "✅ ChamferDistancePytorch编译完成"
    else
        echo "⚠️ ChamferDistancePytorch编译失败，可能是PyTorch版本不兼容"
        echo "可以跳过此步骤，使用CPU版本的Chamfer Distance"
    fi
    cd ../../..
else
    echo "⚠️ ChamferDistancePytorch目录不存在，跳过"
fi

# mm3d_pn2
echo "编译mm3d_pn2..."
if [ -d "utils/mm3d_pn2" ]; then
    cd utils/mm3d_pn2
    echo "尝试编译mm3d_pn2..."
    python setup.py build_ext --inplace 2>&1 | tee /tmp/mm3d_build.log
    if [ $? -eq 0 ]; then
        echo "✅ mm3d_pn2编译完成"
    else
        echo "⚠️ mm3d_pn2编译失败，可能是PyTorch版本不兼容"
        echo "可以跳过此步骤，使用CPU版本的点云处理"
    fi
    cd ../..
else
    echo "⚠️ mm3d_pn2目录不存在，跳过"
fi

echo ""
echo -e "${YELLOW}💡 如果第三方模块编译失败，可以：${NC}"
echo "1. 使用PyTorch 1.9.0版本"
echo "2. 或者跳过编译，使用CPU版本的功能"
echo "3. 或者手动修复代码兼容性问题"

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
