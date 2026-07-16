#!/usr/bin/env bash
# Train an Optimal-Transport steering vector for "speed" from the FM
# extraction produced by extraction/SmolVLA/FM/extract_baseline.sh.
#
# Only tasks 0-4 are passed here (--n-train-tasks 4, so task 4 is used as an
# internal held-out check on the SVM gate's accuracy). Tasks 5-9 are
# deliberately NOT passed at all, so they never enter the OT coupling itself
# — see fm_steering_generate_OT in scripts/features/smolvla/speed.py: as soon as any
# task ends up in its internal test_episodes split, it gets concatenated
# back into the data the OT transport plan is fit on. Keeping 5-9 out of
# --episodes entirely is the only way (short of patching that function) to
# guarantee they stay unseen for 02_apply_steering.sh's demo.
#
# The OT method is the flagship approach in this repo. Two other families of
# methods exist in scripts/features/smolvla/speed.py and are worth trying as
# alternatives once you're comfortable with this pipeline:
#   - train-regression-fm-clf   (Ridge regression + SVM gate, FM layers)
#   - train-regression-vlm-clf  (Ridge regression + SVM gate, VLM layers —
#                                 needs extraction/SmolVLA/VLM data instead)
#   - train-diff-means-vlm-clf  (difference of means, VLM layers)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XL_VLAS_ROOT="${XL_VLAS_ROOT:-$SCRIPT_DIR/../../../../..}"
SPEED_SCRIPT="$XL_VLAS_ROOT/scripts/features/smolvla/speed.py"

SUITE="${SUITE:-libero_object}"
N_EPISODES="${N_EPISODES:-10}"
LAYER="${LAYER:-30}"

EXTRACTION_DIR="${EXTRACTION_DIR:-./results/examples/smolvla/${SUITE}_fm}"
VECS_DIR="${VECS_DIR:-./results/examples/vectors/speed/SmolVLA}"
SUFFIX="_${SUITE}_layer${LAYER}"

mkdir -p "$VECS_DIR"

EPISODES="${SUITE}_0 ${SUITE}_1 ${SUITE}_2 ${SUITE}_3 ${SUITE}_4"

echo "[Train OT] speed | layer=${LAYER} | ${SUITE}"
python "$SPEED_SCRIPT" \
    --extraction-dir "$EXTRACTION_DIR" \
    --layer-nums "$LAYER" \
    --episodes $EPISODES \
    --output-dir "$VECS_DIR" \
    --suffix "$SUFFIX" \
    train-OT \
    --mode classifier \
    --kernel linear \
    --steps 0 1 2 3 4 5 6 7 8 9 \
    --low-q 0.25 --high-q 0.75 \
    --n-episodes-per-task "$N_EPISODES" \
    --n-train-tasks 4

echo "=== Done: vector saved under ${VECS_DIR}/FM_steering_vecs_OT_speed${SUFFIX}.pt ==="
