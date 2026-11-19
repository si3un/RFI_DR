#!/usr/bin/env bash
# ============================================================
#  Training launcher for one config (scripts/run_dr.sh calls it for every
#  configuration in configs/referable_dr/ or configs/any_dr/).
#  Nothing in the code here changes with the domain/backbone/ft.method;
#  everything is determined by the values in the yaml given by CONFIG.
# ------------------------------------------------------------
#  Picks the launch mode automatically based on the number of GPUs (CUDA_VISIBLE_DEVICES).
#   - multiple: DDP via torchrun (one process per GPU -> efficient)
#   - single  : plain python run
#
#  Examples)
#    # on a single GPU 0
#    CONFIG=configs/referable_dr/dinov2_vitl14_lora.yml bash scripts/run.sh
#    # on 4 GPUs 0-3 (DDP)
#    CONFIG=configs/referable_dr/dinov2_vitl14_lora.yml \
#      CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run.sh
#    # paths come from environment variables used in the config (${SAVE_ROOT}, ${DETECTION_CSV}, ...)
#    SAVE_ROOT=/path/to/runs CONFIG=configs/referable_dr/dinov2_vitl14_lora.yml bash scripts/run.sh
#    # resume interrupted training
#    CONFIG=configs/referable_dr/dinov2_vitl14_lora.yml RESUME=/path/to/ckpt_last.pth bash scripts/run.sh
#
#  * Note: under DDP, batch_size in the config is the size "per GPU".
#          (total effective batch = batch_size x number of GPUs)
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

if [ -z "${CONFIG:-}" ]; then
  echo "Specify the config path to run with the CONFIG environment variable. e.g.:" >&2
  echo "  CONFIG=configs/referable_dr/dinov2_vitl14_lora.yml bash scripts/run.sh" >&2
  exit 1
fi

# GPUs to use (default: a single GPU 0). e.g. CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Count GPUs from the number of commas
NUM_GPUS="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  RESUME_ARGS=(--resume "${RESUME}")
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "[multi-GPU / DDP] GPUs=${CUDA_VISIBLE_DEVICES} (nproc=${NUM_GPUS})"
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    "${ROOT_DIR}/trainers/classification_trainer.py" \
    --config-path "${CONFIG}" "${RESUME_ARGS[@]}"
else
  echo "[single-GPU] GPU=${CUDA_VISIBLE_DEVICES}"
  python "${ROOT_DIR}/trainers/classification_trainer.py" \
    --config-path "${CONFIG}" "${RESUME_ARGS[@]}"
fi
