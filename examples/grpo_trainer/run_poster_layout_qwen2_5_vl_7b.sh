#!/bin/bash
set -x

# GRPO训练脚本 - 海报布局生成任务
# 使用SFT好的Qwen2.5-VL-7B模型checkpoint进行强化学习训练

# 优化NCCL设置以减少共享内存使用
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1  
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=INFO

ENGINE=${1:-vllm}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/opt/liblibai-models/user-workspace/jiazhewei/verl-data/poster-layout/train.parquet \
    data.val_files=/opt/liblibai-models/user-workspace/jiazhewei/verl-data/poster-layout/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.image_key=images \
    data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=1 \
    actor_rollout_ref.model.path=/opt/liblibai-models/user-workspace/jiazhewei/checkpoints-secondepoch \
    actor_rollout_ref.model.tokenizer_path=/opt/liblibai-models/model-weights/Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='poster_layout_grpo' \
    trainer.experiment_name='qwen2_5_vl_7b_poster_visual_quality_reward' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    +trainer.reward_manager_name=poster_layout \
    +trainer.reward_manager_kwargs.structure_weight=0.4 \
    +trainer.reward_manager_kwargs.accuracy_weight=0.4 \
    +trainer.reward_manager_kwargs.visual_weight=0.2 \
    +trainer.reward_manager_kwargs.visual_quality_gpu_id=7 \
    +trainer.reward_manager_kwargs.jsonl_file_path=/opt/liblibai-models/user-workspace/jiazhewei/typo_master/psd_dataset_169000_merged_with_caption.jsonl $@
