#!/bin/bash
# Submits all simulate jobs to Slurm

SIM_SCRIPT="slurm/simulate.sbatch"
COL_SCRIPT="slurm/collate.sbatch"
DLB_DIR="/dcs/23/u5567816/deep-linear-bandits"
LOG_DIR="${DLB_DIR}/slurm/logs"

mkdir -p $LOG_DIR # If doesn't exist

PARALLEL_SEEDS=true
SEED_COUNT=100
COMMON="--seed 117 --rounds 10000 --seed-count ${SEED_COUNT}"

NAMES=()
FLAGS=()

add() { NAMES+=("$1"); FLAGS+=("$2"); }

# add ...               "..."

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"

    # Skip if model already trained
    if [[ -d "${DLB_DIR}/simulations/${name}" ]]; then
        echo "Skipping sim-${name}: simulation run already exists"
        continue
    fi

    # Skip if job already queued or running
    if squeue -u "$USER" -h -o "%j" | grep -qx "sim-${name}"; then
        echo "Skipping sim-${name}: job already in queue"
        continue
    fi

    flags="--save-name ${name} ${COMMON} ${FLAGS[$i]}"
    echo "Submitting ${SEED_COUNT} seed(s) for sim-${name}: ${flags}"
    if [ "$PARALLEL_SEEDS" = true ]; then
        JOB_ID=$( \
            sbatch --parsable \
                --job-name="sim-${name}" \
                --output="${LOG_DIR}/sim_${name}_%j.out" \
                --error="${LOG_DIR}/sim_${name}_%j.err" \
                --export=ALL,SIM_FLAGS="${flags}" \
                --array=0-$((SEED_COUNT-1)) \
                "${SIM_SCRIPT}" \
        )

        # Dispatch collate job too
        sbatch --job-name="col-${name}"  \
            --dependency=afterok:$JOB_ID \
            --output="${LOG_DIR}/col_${name}_%j.out" \
            --error="${LOG_DIR}/col_${name}_%j.err" \
            --export=ALL,SAVE_NAME="${name}" \
            "${COL_SCRIPT}"
    else
        sbatch --job-name="sim-${name}" \
           --output="${LOG_DIR}/sim_${name}_%j.out" \
           --error="${LOG_DIR}/sim_${name}_%j.err" \
           --export=ALL,SIM_FLAGS="${flags}" \
           "${SIM_SCRIPT}"
    fi
done

echo "All jobs submitted."