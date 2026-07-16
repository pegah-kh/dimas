#!/usr/bin/env bash
# Extract per-denoising-step hidden states from the flow-matching / action-
# expert layers ("layer group 1") of a few un-steered PI-0.5 rollouts.
# This is the data consumed by steering/pi05/speed/01_train_steering_vector.sh
# and steering/pi05/z-displacement/01_train_steering_vector.sh.
set -euo pipefail

export MUJOCO_GL=egl  # headless rendering, required by LIBERO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_VLAS_ROOT="${XL_VLAS_ROOT:-$SCRIPT_DIR/../../../../..}"
EVAL_SCRIPT="$XL_VLAS_ROOT/src/xl_vlas/save_features.py"

POLICY_PATH="${POLICY_PATH:-lerobot/pi05-libero}"
SUITE="${SUITE:-libero_object}"
N_EPISODES="${N_EPISODES:-10}"
TASK_IDS="${TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}"  # all 10 tasks: 0-4 used to train
                                                # the steering vector, 5-9 held
                                                # out for the apply-steering step
LAYERS=(14 15 16 17)                   # late layers — gemma_300m action-expert only has 18 layers (0-17)

EXTRACTION_DIR="${EXTRACTION_DIR:-./results/examples/pi05/${SUITE}_fm}"

# modules_to_hook JSON array, e.g. [["model.paligemma_with_expert.gemma_expert.model.layers.14.mlp_gated_residual", ...]]
MODULE_LIST=$(printf '"model.paligemma_with_expert.gemma_expert.model.layers.%d.mlp_gated_residual",' "${LAYERS[@]}")
MODULES="[[${MODULE_LIST%,}]]"

HOOK='["save_input_hidden_states_given_token_idx"]'

echo "[Extract FM hidden states] ${SUITE} task_ids=${TASK_IDS}"
python "$EVAL_SCRIPT" \
    --policy.path="$POLICY_PATH" \
    --env.type=libero \
    --env.task="$SUITE" \
    --env.task_id="$TASK_IDS" \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EPISODES" \
    --output_dir="$EXTRACTION_DIR" \
    --modules_to_hook="$MODULES" \
    --hook_names="$HOOK" \
    --token_idx="[0]"

echo "=== Done: FM extraction for ${SUITE} -> ${EXTRACTION_DIR} ==="
