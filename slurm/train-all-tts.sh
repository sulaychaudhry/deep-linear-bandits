#!/bin/bash
# Submits all two-tower training jobs to Slurm

SBATCH_SCRIPT="slurm/train-tt.sbatch"
DLB_DIR="/dcs/23/u5567816/deep-linear-bandits"
LOG_DIR="${DLB_DIR}/slurm/logs"

mkdir -p $LOG_DIR # If doesn't exist

COMMON="--seed 117 --epochs 100"

NAMES=()
FLAGS=()

add() { NAMES+=("$1"); FLAGS+=("$2"); }

add default                       ""

# Different tower configurations
add narrower                      "--hidden-size 128  --hidden-size 64"
add wider                         "--hidden-size 1024 --hidden-size 512"
add deeper                        "--hidden-size 512  --hidden-size 256 --hidden-size 128"
add shallower                     "--hidden-size 512"

# Loss sampling techniques
add in-batch-neg                  "--negative-sampling in-batch"
add user-uniform-neg              "--negative-sampling score-weighted --score-sharpness 0"
add score-weight-neg              "--negative-sampling score-weighted"
add wr-banded-neg                 "--negative-sampling watch-ratio"
add pop-weight-neg                "--negative-sampling popularity"
add full-softmax-neg              "--negative-sampling full-softmax"    

# Ablations
add no-sidefeats                  "--no-side-features"
add no-relu                       "--no-relu"
add id-only-mf                    "--skip-towers --no-side-features"
add no-l2                         "--no-l2-norm"

# Weighted loss from YouTube paper
add weighted-loss                 "--weighted-loss"

# Output embedding dimensionality
add dim-16                        "--output-size 16"
add dim-64                        "--output-size 64"

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"

    # Skip if model already trained
    if [[ -f "${DLB_DIR}/tt-models/${name}/model.pt" ]]; then
        echo "Skipping tt-${name}: model already exists"
        continue
    fi

    # Skip if job already queued or running
    if squeue -u "$USER" -h -o "%j" | grep -qx "tt-${name}"; then
        echo "Skipping tt-${name}: job already in queue"
        continue
    fi

    flags="--save-name ${name} ${COMMON} ${FLAGS[$i]}"
    echo "Submitting tt-${name}: ${flags}"
    sbatch --job-name="tt-${name}" \
           --output="${LOG_DIR}/train-tt_${name}_%j.out" \
           --error="${LOG_DIR}/train-tt_${name}_%j.err" \
           --export=ALL,TT_FLAGS="${flags}" \
           "${SBATCH_SCRIPT}"
done

echo "All jobs submitted."