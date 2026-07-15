#!/usr/bin/env bash
# Apply the speed steering vector trained by 01_train_steering_vector.sh on
# the 5 tasks held out from training (task_id 5-9), and compare against
# the un-steered baseline you already extracted.
#
# Direction: "high_to_low" pushes activations toward the low-speed cluster,
# i.e. it should make the robot move SLOWER. Swap for
# "shift_hidden_states_dimas_low_to_high_pi05" to make it move FASTER
# (you'll need to retrain — 01_train_steering_vector.sh only saves one
# direction's vector by default since train-OT trains a single classifier;
# both directions are symmetric around the same OT coupling, see
# get_OT_flow_matching in scripts/features/pi05/speed.py).
set -euo pipefail

export MUJOCO_GL=egl  # headless rendering, required by LIBERO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_VLAS_ROOT="${XL_VLAS_ROOT:-$SCRIPT_DIR/../../../../..}"
EVAL_SCRIPT="$XL_VLAS_ROOT/src/xl_vlas/save_features.py"

POLICY_PATH="${POLICY_PATH:-lerobot/pi05-libero}"
SUITE="${SUITE:-libero_object}"
N_EPISODES="${N_EPISODES:-10}"
LAYER="${LAYER:-30}"
ALPHA="${ALPHA:-0.5}"

VECS_DIR="${VECS_DIR:-./results/examples/vectors/speed/pi05}"
VECTOR="$VECS_DIR/FM_steering_vecs_OT__${SUITE}_layer${LAYER}.pt"
OUTPUT_DIR="${OUTPUT_DIR:-./results/examples/steered/speed/pi05}"

if [[ ! -f "$VECTOR" ]]; then
    echo "ERROR: steering vector not found at $VECTOR — run 01_train_steering_vector.sh first."
    exit 1
fi

MODULES="[[\"model.paligemma_with_expert.gemma_expert.model.layers.${LAYER}.mlp_gated_residual\"]]"
HOOK='["shift_hidden_states_dimas_high_to_low_pi05"]'

echo "[Apply steering] speed h2l | layer=${LAYER} alpha=${ALPHA} | ${SUITE} task_id=[5,6,7,8,9]"
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
