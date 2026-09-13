#!/usr/bin/env bash
# Export an explicitly selected verl FSDP actor checkpoint to Hugging Face format.
set -Eeuo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/export_grpo.sh CHECKPOINT/actor OUTPUT_DIR [merger options]" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/training/verl${PYTHONPATH:+:$PYTHONPATH}"
ACTOR_DIR="$1"
TARGET_DIR="$2"
shift 2
if [ ! -d "$ACTOR_DIR" ]; then
  echo "Actor checkpoint not found: $ACTOR_DIR" >&2
  exit 2
fi
if [ -e "$TARGET_DIR" ]; then
  echo "Choose a new output directory: $TARGET_DIR" >&2
  exit 2
fi
exec python -m verl.model_merger merge --backend fsdp \
  --local_dir "$ACTOR_DIR" --target_dir "$TARGET_DIR" --trust-remote-code "$@"
