# 海报生成GRPO训练指南

## 🎯 方案概述

这个解决方案采用**独立Reward Model服务**的架构，完美解决了VERL框架中的GPU访问限制问题：

- **GPU 0-6**: PPO训练任务
- **GPU 7**: 独立的VisualQuality-R1 Reward Model服务
- **通信方式**: HTTP API (端口8899)

## 🚀 快速启动

### 一键启动（推荐）

```bash
cd /opt/liblibai-models/user-workspace/jiazhewei/verl
./examples/grpo_trainer/start_poster_training_with_reward_server.sh
```

这个脚本会自动：
1. 在GPU 7上启动VisualQuality-R1服务
2. 等待服务就绪
3. 启动7-GPU的GRPO训练
4. 实时监控日志
5. 优雅地停止所有进程

### 手动启动（调试用）

#### 1. 启动Reward Model服务

```bash
# 终端1：启动reward model服务
cd /opt/liblibai-models/user-workspace/jiazhewei/verl
export CUDA_VISIBLE_DEVICES=7
python reward_model_server.py
```

#### 2. 启动训练任务

```bash
# 终端2：启动训练
cd /opt/liblibai-models/user-workspace/jiazhewei/verl
./examples/grpo_trainer/run_poster_layout_qwen2_5_vl_7b_optimized.sh
```

## 📊 系统架构

```
┌─────────────────┐    ┌─────────────────┐
│   PPO Training  │    │  Reward Model   │
│   (GPU 0-6)     │    │   Service       │
│                 │    │   (GPU 7)       │
│  ┌─────────────┐│    │ ┌─────────────┐ │
│  │   Actor     ││    │ │VisualQuality││ │
│  │   Critic    ││    │ │    -R1      ││ │
│  │   Reference ││────┼─┤  FastAPI    ││ │
│  │             ││HTTP│ │             ││ │
│  └─────────────┘│    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘
```

## 🔧 配置说明

### GPU分配
- `trainer.n_gpus_per_node=7`: 使用7张GPU进行训练
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`: 训练任务只看到前7张GPU
- Reward服务独占GPU 7

### 内存优化
- `max_model_len=32768`: 支持长序列（多图像海报）
- `max_num_batched_tokens=65536`: 匹配chunked prefill要求
- `gpu_memory_utilization=0.75`: 75%显存利用率

### Reward配置
- `structure_weight=0.4`: 结构匹配权重
- `accuracy_weight=0.4`: 文本准确度权重
- `visual_weight=0.2`: 视觉质量权重

## 📋 服务监控

### 健康检查

```bash
curl http://localhost:8899/health
```

### 测试Reward计算

```bash
curl -X POST http://localhost:8899/compute_reward \
  -H "Content-Type: application/json" \
  -d '{
    "solution_str": "{\"layers\": [...]}",
    "ground_truth": {...},
    "extra_info": {"sample_id": "test"}
  }'
```

## 📁 日志文件

- `reward_server.log`: Reward模型服务日志
- `training.log`: GRPO训练日志
- 实时监控：`tail -f training.log`

## 🛠 故障排除

### 1. Reward服务启动失败

```bash
# 检查GPU 7状态
nvidia-smi
export CUDA_VISIBLE_DEVICES=7
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 训练连接失败

```bash
# 检查服务状态
curl http://localhost:8899/health
netstat -tlnp | grep 8899
```

### 3. 显存不足

```bash
# 降低配置参数
# 在训练脚本中调整：
# - max_num_seqs=16 (降低并行序列数)
# - gpu_memory_utilization=0.6 (降低显存利用率)
```

## 📈 性能指标

### 预期吞吐量
- **单GPU版本**: ~2-3 samples/sec
- **7-GPU版本**: ~15-20 samples/sec
- **Reward评估**: ~1-2 sec/sample

### 资源使用
- **训练GPU**: 7 × H20 × 75% ≈ 500GB显存
- **Reward GPU**: 1 × H20 × 70% ≈ 67GB显存
- **总算力**: 充分利用8张H20显卡

## ✅ 优势总结

1. **GPU隔离**: 完美解决Ray worker GPU访问限制
2. **高可用性**: HTTP服务提供稳定的reward计算
3. **易监控**: 独立日志和健康检查端点
4. **易扩展**: 可以轻松添加更多reward模型服务
5. **资源优化**: 充分利用8张H20显卡的计算能力

## 🎉 开始训练

现在你可以运行训练了：

```bash
cd /opt/liblibai-models/user-workspace/jiazhewei/verl
./examples/grpo_trainer/start_poster_training_with_reward_server.sh
```

训练会自动管理所有组件，你只需要观察日志即可！
