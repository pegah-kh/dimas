#!/usr/bin/env bash
# Extract mean-pooled VLM (vision-language backbone) hidden states from a
# few un-steered SmolVLA rollouts. This is the "layer group 0" data used by
# scripts/features/smolvla/speed.py's/z_displacement.py's VLM-based training
# commands (train-regression-vlm-clf, train-diff-means-vlm-clf).
set -euo pipefail

export MUJOCO_GL=egl  # headless rendering, required by LIBERO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_VLAS_ROOT="${XL_VLAS_ROOT:-$SCRIPT_DIR/../../../../..}"
EVAL_SCRIPT="$XL_VLAS_ROOT/src/xl_vlas/save_features.py"

POLICY_PATH="${POLICY_PATH:-HuggingFaceVLA/smolvla_libero}"
SUITE="${SUITE:-libero_object}"
N_EPISODES="${N_EPISODES:-10}"
TASK_IDS=(0 1 2 3 4 5 6 7 8 9)  # all 10 tasks: 0-4 will later be used to train
                                 # the steering vector, 5-9 held out for eval
LAYERS=(28 29 30 31)            # late layers, where steering tends to work best

EXTRACTION_DIR="${EXTRACTION_DIR:-./results/examples/smolvla/${SUITE}}"

# Build the modules_to_hook JSON array, e.g. [["model.vlm_with_expert.layer_hooks.0.28", ...]]
MODULES="["
for L in "${LAYERS[@]}"; do
    MODULES="${MODULES}\"model.vlm_with_expert.layer_hooks.0.${L}\","
done
MODULES="${MODULES%,}]"
MODULES="[${MODULES}]"

HOOK='["save_input_hidden_states_mean"]'

for TASK_ID in "${TASK_IDS[@]}"; do
    OUT="$EXTRACTION_DIR/${SUITE}_${TASK_ID}_mean_vlm"

    if [[ -f "$OUT/videos/${SUITE}_${TASK_ID}/episode_output_data.pt" ]]; then
        echo "[Skip] task_id=${TASK_ID} already extracted"
        continue
    fi

    echo "[Extract VLM mean] ${SUITE} task_id=${TASK_ID}"
    python "$EVAL_SCRIPT" \
        --policy.path="$POLICY_PATH" \
        --env.type=libero \
        --env.task="$SUITE" \
        --env.task_id="[$TASK_ID]" \
        --eval.batch_size=1 \
        --eval.n_episodes="$N_EPISODES" \
        --output_dir="$OUT" \
        --modules_to_hook="$MODULES" \
        --hook_names="$HOOK"
done

echo "=== Done: VLM extraction for ${SUITE} -> ${EXTRACTION_DIR} ==="
