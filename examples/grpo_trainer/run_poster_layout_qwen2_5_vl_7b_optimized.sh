#!/bin/bash
set -x

# GRPO训练脚本 - 海报布局生成任务 (优化版)
# 充分利用8张H20显卡：7张训练 + 1张专门给reward model
# 总显存：8 x 96GB = 768GB

# 优化NCCL设置
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=0  # 启用P2P提高多GPU通信效率
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0  # 使用tree算法优化多GPU通信

# 设置CUDA可见设备（0-6用于训练，7专门给reward model）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ENGINE=${1:-vllm}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/opt/liblibai-models/user-workspace/jiazhewei/verl-data/poster-layout/train.parquet \
    data.val_files=/opt/liblibai-models/user-workspace/jiazhewei/verl-data/poster-layout/test.parquet \
    data.train_batch_size=84 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.image_key=images \
    data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=2 \
    actor_rollout_ref.model.path=/opt/liblibai-models/user-workspace/jiazhewei/checkpoints-secondepoch \
    actor_rollout_ref.model.tokenizer_path=/opt/liblibai-models/model-weights/Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=21 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=3 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='poster_layout_grpo' \
    trainer.experiment_name='qwen2_5_vl_7b_poster_8h20_optimized' \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    reward_model.enable=False \
    custom_reward_function.path=verl/utils/reward_score/visual_quality_poster_reward.py \
    custom_reward_function.name=compute_score $@
