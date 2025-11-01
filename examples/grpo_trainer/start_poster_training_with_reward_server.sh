#!/bin/bash
set -e

# 海报生成GRPO训练完整启动脚本
# 自动管理reward model服务和训练任务

REWARD_SERVER_PORT=8899
REWARD_SERVER_GPU=7
TRAINING_LOG="training.log"
REWARD_SERVER_LOG="reward_server.log"

# SwanLab监控配置 - 确保环境变量传递给子进程
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-"Z6PZIvJAcy6CiF5INwAZh"}
export SWANLAB_LOG_DIR="swanlog/poster_layout_grpo"
export SWANLAB_MODE="cloud"

echo "🚀 Starting GRPO Training with Dedicated Reward Model Server"
echo "=========================================================="
echo "📊 GPU配置:"
echo "   - GPU 0-6: PPO训练"
echo "   - GPU 7:   VisualQuality-R1 Reward Model服务"
echo "   - 端口:     $REWARD_SERVER_PORT"
echo "📈 监控配置:"
echo "   - SwanLab: ${SWANLAB_MODE}模式"
echo "   - 日志目录: $SWANLAB_LOG_DIR"
echo ""

# 检查GPU状态
echo "🔍 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits

# 清理之前的日志
rm -f $TRAINING_LOG $REWARD_SERVER_LOG

# 创建停止函数
cleanup() {
    echo ""
    echo "🛑 停止所有进程..."
    
    # 停止训练进程
    if [[ -n $TRAINING_PID ]]; then
        echo "   停止训练进程 (PID: $TRAINING_PID)"
        kill -TERM $TRAINING_PID 2>/dev/null || true
        wait $TRAINING_PID 2>/dev/null || true
    fi
    
    # 停止reward server进程
    if [[ -n $REWARD_SERVER_PID ]]; then
        echo "   停止Reward Server (PID: $REWARD_SERVER_PID)"
        kill -TERM $REWARD_SERVER_PID 2>/dev/null || true
        wait $REWARD_SERVER_PID 2>/dev/null || true
    fi
    
    echo "✅ 所有进程已停止"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 1. 启动Reward Model服务
echo "🔥 启动VisualQuality-R1 Reward Model服务..."
export CUDA_VISIBLE_DEVICES=$REWARD_SERVER_GPU
cd /opt/liblibai-models/user-workspace/jiazhewei/verl
python reward_model_server.py > $REWARD_SERVER_LOG 2>&1 &
REWARD_SERVER_PID=$!

echo "   服务PID: $REWARD_SERVER_PID"
echo "   日志文件: $REWARD_SERVER_LOG"

# 等待服务启动
echo "⏳ 等待Reward Model服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:$REWARD_SERVER_PORT/health > /dev/null 2>&1; then
        echo "✅ Reward Model服务已就绪"
        break
    fi
    
    if ! kill -0 $REWARD_SERVER_PID 2>/dev/null; then
        echo "❌ Reward Model服务启动失败"
        echo "📋 服务日志:"
        cat $REWARD_SERVER_LOG
        exit 1
    fi
    
    echo "   等待中... ($i/30)"
    sleep 2
done

# 检查服务状态
HEALTH_RESPONSE=$(curl -s http://localhost:$REWARD_SERVER_PORT/health || echo "failed")
if [[ "$HEALTH_RESPONSE" != *"healthy"* ]]; then
    echo "❌ Reward Model服务健康检查失败"
    echo "响应: $HEALTH_RESPONSE"
    cleanup
    exit 1
fi

echo "🎯 Reward Model服务健康检查通过"

# 2. 启动GRPO训练
echo ""
echo "🚀 启动GRPO训练任务..."
echo "   使用GPU: 0,1,2,3,4,5,6"
echo "   日志文件: $TRAINING_LOG"

bash examples/grpo_trainer/run_poster_layout_qwen2_5_vl_7b_optimized.sh > $TRAINING_LOG 2>&1 &
TRAINING_PID=$!

echo "   训练PID: $TRAINING_PID"

# 3. 实时监控
echo ""
echo "📊 开始监控训练进程..."
echo "   按 Ctrl+C 停止训练"
echo ""

# 显示初始状态
echo "=== 服务状态 ==="
curl -s http://localhost:$REWARD_SERVER_PORT/health | python -m json.tool 2>/dev/null || echo "无法获取服务状态"

# 监控训练进程
tail -f $TRAINING_LOG &
TAIL_PID=$!

# 等待训练完成或被中断
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# 停止日志监控
kill $TAIL_PID 2>/dev/null || true

echo ""
echo "🎯 训练任务完成，退出码: $TRAINING_EXIT_CODE"

# 显示最终统计
echo ""
echo "=== 最终统计 ==="
if [[ -f $REWARD_SERVER_LOG ]]; then
    echo "Reward Server 请求统计:"
    grep -c "POST /compute_reward" $REWARD_SERVER_LOG 2>/dev/null || echo "  无请求记录"
fi

if [[ -f $TRAINING_LOG ]]; then
    echo "训练日志大小: $(wc -l < $TRAINING_LOG) 行"
fi

# 清理
cleanup
