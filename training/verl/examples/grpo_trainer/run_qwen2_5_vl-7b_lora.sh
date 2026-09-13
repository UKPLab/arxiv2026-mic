set -x
ENGINE=${1:-vllm}

# Use all 8 GPUs + ROCm compatibility
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_TRITON_FLASH_ATTN=0
export VERL_DISABLE_GC=1
export NCCL_CUMEM_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:False

# MIOpen: use writable user DB to avoid "readonly database" errors
export MIOPEN_USER_DB_PATH=/tmp/miopen_user_db_${USER}
mkdir -p $MIOPEN_USER_DB_PATH
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache_${USER}
mkdir -p $MIOPEN_CUSTOM_CACHE_DIR

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen2_5_vl_7b_lora_geo3k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
