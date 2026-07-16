#!/usr/bin/env bash
# Extract per-denoising-step hidden states from the flow-matching / action-
# expert layers ("layer group 1") of a few un-steered SmolVLA rollouts.
# This is the data consumed by steering/smolvla/speed/01_train_steering_vector.sh
# and steering/smolvla/z-displacement/01_train_steering_vector.sh.
set -euo pipefail

export MUJOCO_GL=egl  # headless rendering, required by LIBERO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_VLAS_ROOT="${XL_VLAS_ROOT:-$SCRIPT_DIR/../../../../..}"
EVAL_SCRIPT="$XL_VLAS_ROOT/src/xl_vlas/save_features.py"

POLICY_PATH="${POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
SUITE="${SUITE:-libero_object}"
N_EPISODES="${N_EPISODES:-10}"
TASK_IDS="${TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}"  # all 10 tasks: 0-4 used to train
                                                # the steering vector, 5-9 held
                                                # out for the apply-steering step
LAYERS=(28 29 30 31)                   # late layers, where steering tends to work best

EXTRACTION_DIR="${EXTRACTION_DIR:-./results/examples/smolvla/${SUITE}_fm}"

# Build the modules_to_hook JSON array, e.g. [["model.vlm_with_expert.layer_hooks.1.28", ...]]
MODULES="["
for L in "${LAYERS[@]}"; do
    MODULES="${MODULES}\"model.vlm_with_expert.layer_hooks.1.${L}\","
done
MODULES="${MODULES%,}]"
MODULES="[${MODULES}]"

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
