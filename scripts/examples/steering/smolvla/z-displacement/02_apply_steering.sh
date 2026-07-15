#!/usr/bin/env bash
# Apply the z-displacement steering vector trained by
# 01_train_steering_vector.sh on the 5 tasks held out from training
# (task_id 5-9), and compare against the un-steered baseline you already
# extracted.
#
# Direction: "high_to_low" pushes activations toward the low-|delta_z|
# cluster, i.e. it should make the robot move LESS vertically (flatter
# trajectories). Swap for "shift_hidden_states_dimas_low_to_high"
# for MORE vertical movement (retrain first, see 02_apply_steering.sh in
# steering/speed/ for why).
set -euo pipefail

export MUJOCO_GL=egl  # headless rendering, required by LIBERO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_VLAS_ROOT="${XL_VLAS_ROOT:-$SCRIPT_DIR/../../../../..}"
EVAL_SCRIPT="$XL_VLAS_ROOT/src/xl_vlas/save_features.py"

POLICY_PATH="${POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
SUITE="${SUITE:-libero_object}"
N_EPISODES="${N_EPISODES:-10}"
LAYER="${LAYER:-30}"
ALPHA="${ALPHA:-0.5}"

VECS_DIR="${VECS_DIR:-./results/examples/vectors/z-displacement/SmolVLA}"
VECTOR="$VECS_DIR/FM_eef_height_steering_vecs_OT__${SUITE}_layer${LAYER}.pt"
OUTPUT_DIR="${OUTPUT_DIR:-./results/examples/steered/z-displacement/SmolVLA}"

if [[ ! -f "$VECTOR" ]]; then
    echo "ERROR: steering vector not found at $VECTOR — run 01_train_steering_vector.sh first."
    exit 1
fi

MODULES="[[\"model.vlm_with_expert.layer_hooks.1.${LAYER}\"]]"
HOOK='["shift_hidden_states_dimas_high_to_low"]'

echo "[Apply steering] z-displacement h2l | layer=${LAYER} alpha=${ALPHA} | ${SUITE} task_id=[5,6,7,8,9]"
python "$EVAL_SCRIPT" \
    --policy.path="$POLICY_PATH" \
    --env.type=libero \
    --env.task="$SUITE" \
    --env.task_id="[5,6,7,8,9]" \
    --env.episode_length=500 \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EPISODES" \
    --output_dir="$OUTPUT_DIR" \
    --modules_to_hook="$MODULES" \
    --hook_names="$HOOK" \
    --steering_alpha="$ALPHA" \
    --shift_vector_path="[\"$VECTOR\"]" \
    --token_idx="[0]"

echo "=== Done: steered rollouts -> ${OUTPUT_DIR} (see eval_info.json and videos/) ==="
