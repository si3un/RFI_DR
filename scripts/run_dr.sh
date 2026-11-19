#!/usr/bin/env bash
# ============================================================
#  DR training launcher.
#  Exports the path environment variables (${ENV}), then runs configs/referable_dr/*.yml via scripts/run.sh.
#  (Nothing is hard-coded in the configs or code.)
#
#  Usage:
#    # all 21 (sequential, automatic DDP depending on the number of GPUs)
#    CUDA_VISIBLE_DEVICES=0 bash scripts/run_dr.sh
#    # a single specific config only
#    CUDA_VISIBLE_DEVICES=0 bash scripts/run_dr.sh configs/referable_dr/dinov2_vitl14_linear.yml
# ============================================================
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Path environment variables used by the configs. Set them here or export them before running.
export SAVE_ROOT="${SAVE_ROOT:?set SAVE_ROOT (see README)}"
export PRETRAINED_ROOT="${PRETRAINED_ROOT:?set PRETRAINED_ROOT (see README)}"
export DINOV3_REPO="${DINOV3_REPO:?set DINOV3_REPO (see README)}"
export DETECTION_CSV="${DETECTION_CSV:?set DETECTION_CSV (see README)}"
export EXTERNAL_CSV="${EXTERNAL_CSV:?set EXTERNAL_CSV (see README)}"

# With arguments, only those configs; otherwise all of configs/referable_dr/*.yml
if [ "$#" -ge 1 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("$ROOT_DIR"/configs/referable_dr/*.yml)   # any-DR sensitivity: pass configs/any_dr/*.yml
fi

for cfg in "${CONFIGS[@]}"; do
  echo "=========================================="
  echo "  CONFIG=$cfg"
  echo "=========================================="
  CONFIG="$cfg" bash "$ROOT_DIR/scripts/run.sh"
done
echo "Done."
