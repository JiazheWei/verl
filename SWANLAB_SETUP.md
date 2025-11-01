# SwanLab监控集成指南

## 概述
已为GRPO海报训练任务集成SwanLab监控，可实时跟踪训练指标和模型性能。

## 1. 安装SwanLab

```bash
pip install swanlab
```

## 2. 配置API Key

### 方法1：环境变量设置（推荐）
```bash
export SWANLAB_API_KEY="your_api_key_here"
```

### 方法2：在启动脚本中设置
```bash
# 在运行训练前设置
SWANLAB_API_KEY="your_api_key_here" bash examples/grpo_trainer/start_poster_training_with_reward_server.sh
```

## 3. 获取API Key

1. 访问 [SwanLab官网](https://swanlab.cn/)
2. 注册/登录账号
3. 在设置页面获取API Key

## 4. 监控指标

训练过程中将自动记录以下指标：

### 训练指标
- **Loss相关**：
  - `train/policy_loss`: 策略损失
  - `train/value_loss`: 价值函数损失  
  - `train/total_loss`: 总损失

- **GRPO算法指标**：
  - `train/advantage`: 优势值统计
  - `train/return`: 回报值
  - `train/reward`: 即时奖励

- **KL散度**：
  - `train/kl_divergence`: 新旧策略KL散度
  - `train/kl_penalty`: KL惩罚项

- **性能指标**：
  - `train/learning_rate`: 学习率
  - `train/grad_norm`: 梯度范数
  - `train/entropy`: 策略熵

### 验证指标
- **生成质量**：
  - `val/reward_score`: 验证集奖励得分
  - `val/generations`: 生成样本表格

### 系统指标
- **GPU使用率**: 各GPU的内存和计算利用率
- **训练时间**: 每步训练耗时

## 5. 访问监控面板

### 在线模式（默认）
- 训练开始后访问：https://swanlab.cn/
- 找到项目：`poster_layout_grpo`
- 实验名称：`qwen2_5_vl_7b_poster_8h20_optimized`

### 离线模式
如果网络不稳定，可以使用离线模式：

```bash
export SWANLAB_MODE="offline"
```

离线日志保存在：`swanlog/poster_layout_grpo/`

## 6. 高级配置

### 自定义日志目录
```bash
export SWANLAB_LOG_DIR="/path/to/your/logs"
```

### 项目和实验名称
在训练脚本中已配置：
- 项目名：`poster_layout_grpo`
- 实验名：`qwen2_5_vl_7b_poster_8h20_optimized`

## 7. 故障排除

### 常见问题

**问题1**: SwanLab连接失败
```bash
# 检查网络连接和API Key
swanlab auth
```

**问题2**: 指标不显示
- 确认已正确设置`trainer.logger='["console","swanlab"]'`
- 检查训练日志中是否有SwanLab相关错误

**问题3**: 离线模式使用
```bash
# 训练完成后上传离线日志
swanlab upload swanlog/poster_layout_grpo/
```

## 8. 监控要点

训练过程中重点关注：
1. **收敛性**: `train/total_loss`是否稳定下降
2. **策略稳定性**: `train/kl_divergence`不应过大
3. **奖励提升**: `train/reward`是否呈上升趋势
4. **生成质量**: 定期查看`val/generations`中的样本

## 9. 与其他监控工具对比

| 功能 | SwanLab | WandB | TensorBoard |
|------|---------|-------|-------------|
| 实时监控 | ✅ | ✅ | ✅ |
| 中文界面 | ✅ | ❌ | ❌ |
| 离线模式 | ✅ | ✅ | ✅ |
| 模型比较 | ✅ | ✅ | ✅ |
| 团队协作 | ✅ | ✅ | ❌ |

SwanLab特别适合中文用户，界面友好，功能完善。
