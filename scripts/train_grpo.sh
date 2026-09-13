#!/usr/bin/env bash
# Launch MIC GRPO from a merged SFT checkpoint using the active Python environment.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/training/verl${PYTHONPATH:+:$PYTHONPATH}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/outputs/merged/qwen3vl4b}"
DATA_DIR="${GRPO_DATA_DIR:-$REPO_ROOT/training/data/grpo}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/outputs/grpo/qwen3vl4b}"
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/val.parquet" ]; then
  echo "Prepare GRPO data and merge the SFT checkpoint first. See README.md#training." >&2
  exit 2
fi
exec python -m verl.trainer.main_ppo \
  --config-path "$REPO_ROOT/configs/grpo" --config-name mic \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  reward.custom_reward_function.path="$REPO_ROOT/src/training/reward.py" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
  "$@"
