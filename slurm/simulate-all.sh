#!/bin/bash
# Submits all simulate jobs to Slurm

SIM_SCRIPT="slurm/simulate.sbatch"
COL_SCRIPT="slurm/collate.sbatch"
DLB_DIR="/dcs/23/u5567816/deep-linear-bandits"
LOG_DIR="${DLB_DIR}/slurm/logs"

mkdir -p $LOG_DIR # If doesn't exist

PARALLEL_SEEDS=true
SEED_COUNT=10
COMMON="--rounds 10000 --seed 117 --seed-count ${SEED_COUNT}"

NAMES=()
FLAGS=()

add() { NAMES+=("$1"); FLAGS+=("$2"); }

# Dispatch a simulation for all existing models
add default             "--model default"
add shallow             "--model shallow"
add deep                "--model deep"
add wide                "--model wide"
add narrow              "--model narrow"
add narrowest           "--model narrowest"

add no-sidefeats        "--model no-sidefeats"
add no-relu             "--model no-relu"
add mf-baseline         "--model mf-baseline"
add no-l2               "--model no-l2"

add dim-16              "--model dim-16"
add dim-64              "--model dim-64"

add temp-005            "--model temp-005"
add temp-01             "--model temp-01"

add in-batch-neg        "--model in-batch-neg"
add user-uniform-neg    "--model user-uniform-neg"
add score-weight-neg1   "--model score-weight-neg1"
add score-weight-neg2   "--model score-weight-neg2"
add wr-banded-neg1      "--model wr-banded-neg1"
add wr-banded-neg2      "--model wr-banded-neg2"

add neg-64              "--model neg-64"
add neg-256             "--model neg-256"
add neg-512             "--model neg-512"

add no-dropout          "--model no-dropout"
add dropout-04          "--model dropout-04"

add adamw               "--model adamw"

add lr-5em4             "--model lr-5em4"
add lr-2em3             "--model lr-2em3"

add wt-1                "--model wt-1"
add wt-3                "--model wt-3"

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
        JOB_ID = $( \
            sbatch --parsable \
            --job-name="sim-${name}" \
            --output="${LOG_DIR}/sim_${name}_%j.out" \
            --error="${LOG_DIR}/sim_${name}__%j.err" \
            --export=ALL,SIM_FLAGS="${flags}" \
            --array=0-$((SEED_COUNT-1)) \
            "${SIM_SCRIPT}" \
        )

        # Dispatch collate job too
        sbatch --job-name="col-${name}" --dependency=afterok:$JOB_ID --export=ALL,SAVE_NAME="${name}" ${COL_SCRIPT}
    else
        sbatch --job-name="sim-${name}" \
           --output="${LOG_DIR}/sim_${name}_%j.out" \
           --error="${LOG_DIR}/sim_${name}__%j.err" \
           --export=ALL,SIM_FLAGS="${flags}" \
           "${SIM_SCRIPT}"
    fi
done

echo "All jobs submitted."