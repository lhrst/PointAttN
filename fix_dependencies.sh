#!/bin/bash

echo "🔧 修复 PointAttN 依赖版本冲突"
echo "================================"

echo "📋 问题分析: typing_extensions版本过低导致dash冲突"
echo ""

echo "🔄 修复方案1: 更新typing_extensions..."
pip install --upgrade typing_extensions

echo ""
echo "🔄 修复方案2: 如果问题仍然存在，降级open3d版本..."
pip install "open3d==0.13.0"

echo ""
echo "🔄 修复方案3: 或者使用open3d-cpu版本（无GUI功能但更稳定）..."
echo "如果上述方案都不行，请运行："
echo "pip uninstall open3d"
echo "pip install open3d-cpu"

echo ""
echo "✅ 验证修复..."
python -c "
try:
    import typing_extensions
    print(f'✅ typing_extensions版本: {typing_extensions.__version__}')
except:
    print('❌ typing_extensions导入失败')

try:
    import open3d as o3d
    print(f'✅ open3d版本: {o3d.__version__}')
    print('✅ open3d导入成功')
except Exception as e:
    print(f'❌ open3d导入失败: {e}')
    print('建议使用: pip install open3d-cpu')
"

echo ""
echo "🎯 如果问题解决，现在可以运行:"
echo "python test_preprocessing.py"