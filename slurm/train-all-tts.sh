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

# add ...                  "..."

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