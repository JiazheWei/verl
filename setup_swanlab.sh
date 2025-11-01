#!/bin/bash

# SwanLab环境设置和验证脚本

set -e

echo "🔧 SwanLab环境设置向导"
echo "========================"

# 1. 检查是否已安装SwanLab
echo "📦 检查SwanLab安装状态..."
if python -c "import swanlab" 2>/dev/null; then
    echo "✅ SwanLab已安装"
    python -c "import swanlab; print(f'SwanLab版本: {swanlab.__version__}')"
else
    echo "❌ SwanLab未安装，正在安装..."
    pip install swanlab
    echo "✅ SwanLab安装完成"
fi

# 2. 检查API Key设置
echo ""
echo "🔑 检查API Key配置..."
if [ -z "$SWANLAB_API_KEY" ]; then
    echo "⚠️  未设置SWANLAB_API_KEY环境变量"
    echo ""
    echo "请按照以下步骤设置API Key："
    echo "1. 访问 https://swanlab.cn/"
    echo "2. 注册/登录账号"
    echo "3. 在设置页面获取API Key"
    echo "4. 设置环境变量："
    echo "   export SWANLAB_API_KEY='your_api_key_here'"
    echo ""
    echo "或者在运行训练时设置："
    echo "   SWANLAB_API_KEY='your_key' bash start_poster_training_with_reward_server.sh"
    
    read -p "是否要现在输入API Key进行测试？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入您的SwanLab API Key: " api_key
        export SWANLAB_API_KEY="$api_key"
    else
        echo "⏭️  跳过API Key测试"
    fi
fi

if [ -n "$SWANLAB_API_KEY" ]; then
    echo "✅ API Key已设置: ${SWANLAB_API_KEY:0:8}..."
    
    # 3. 测试SwanLab连接
    echo ""
    echo "🧪 测试SwanLab连接..."
    
    # 创建测试脚本
    cat > /tmp/test_swanlab.py << 'EOF'
import swanlab
import os
import time

try:
    # 测试初始化
    swanlab.init(
        project="verl_test",
        experiment_name=f"connection_test_{int(time.time())}",
        mode=os.environ.get("SWANLAB_MODE", "cloud")
    )
    
    # 测试记录指标
    for i in range(5):
        swanlab.log({"test_metric": i * 0.1}, step=i)
    
    print("✅ SwanLab连接测试成功")
    swanlab.finish()
    
except Exception as e:
    print(f"❌ SwanLab连接测试失败: {e}")
    exit(1)
EOF

    python /tmp/test_swanlab.py
    rm /tmp/test_swanlab.py
    
else
    echo "⚠️  未设置API Key，跳过连接测试"
fi

# 4. 显示配置信息
echo ""
echo "📋 当前SwanLab配置："
echo "   API Key: ${SWANLAB_API_KEY:+已设置}"
echo "   日志目录: ${SWANLAB_LOG_DIR:-swanlog/poster_layout_grpo}"
echo "   运行模式: ${SWANLAB_MODE:-cloud}"

# 5. 创建日志目录
echo ""
echo "📁 创建日志目录..."
mkdir -p "${SWANLAB_LOG_DIR:-swanlog/poster_layout_grpo}"
echo "✅ 日志目录已创建: ${SWANLAB_LOG_DIR:-swanlog/poster_layout_grpo}"

echo ""
echo "🎉 SwanLab环境设置完成！"
echo ""
echo "接下来可以运行训练："
echo "   bash examples/grpo_trainer/start_poster_training_with_reward_server.sh"
echo ""
echo "监控面板访问："
echo "   在线模式: https://swanlab.cn/"
echo "   项目名称: poster_layout_grpo"
echo "   实验名称: qwen2_5_vl_7b_poster_8h20_optimized"
