#!/bin/bash
# Submits all simulate jobs to Slurm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIM_SCRIPT="${SCRIPT_DIR}/simulate.sbatch"
COL_SCRIPT="${SCRIPT_DIR}/collate.sbatch"
LOG_DIR="${DLB_DIR}/slurm/logs"

mkdir -p "$LOG_DIR" # If doesn't exist

PARALLEL_SEEDS=true
SEED_COUNT=100
COMMON="--seed 117 --rounds 10000 --seed-count ${SEED_COUNT}"

NAMES=()
FLAGS=()

add() { NAMES+=("$1"); FLAGS+=("$2"); }

# Dispatch binary reward tests, 10k rounds
add default                       "--model default --binary-reward"
add narrower                      "--model narrower --binary-reward"
add wider                         "--model wider --binary-reward"
add deeper                        "--model deeper --binary-reward"
add shallower                     "--model shallower --binary-reward"
add in-batch-neg                  "--model in-batch-neg --binary-reward"
add user-uniform-neg              "--model user-uniform-neg --binary-reward"
add score-weight-neg              "--model score-weight-neg --binary-reward"
add wr-banded-neg                 "--model wr-banded-neg --binary-reward"
add pop-weight-neg                "--model pop-weight-neg --binary-reward"
add full-softmax-neg              "--model full-softmax-neg --binary-reward"  
add no-sidefeats                  "--model no-sidefeats --binary-reward"
add no-relu                       "--model no-relu --binary-reward"
add id-only-mf                    "--model id-only-mf --binary-reward"
add no-l2                         "--model no-l2 --binary-reward"
add weighted-loss                 "--model weighted-loss --binary-reward"
add dim-16                        "--model dim-16 --binary-reward"
add dim-64                        "--model dim-64 --binary-reward"

# Dispatch continuous reward tests, 10k rounds
add default-cont                  "--model default --continuous-reward"
add narrower-cont                 "--model narrower --continuous-reward"
add wider-cont                    "--model wider --continuous-reward"
add deeper-cont                   "--model deeper --continuous-reward"
add shallower-cont                "--model shallower --continuous-reward"
add in-batch-neg-cont             "--model in-batch-neg --continuous-reward"
add user-uniform-neg-cont         "--model user-uniform-neg --continuous-reward"
add score-weight-neg-cont         "--model score-weight-neg --continuous-reward"
add wr-banded-neg-cont            "--model wr-banded-neg --continuous-reward"
add pop-weight-neg-cont           "--model pop-weight-neg --continuous-reward"
add full-softmax-neg-cont         "--model full-softmax-neg --continuous-reward"
add no-sidefeats-cont             "--model no-sidefeats --continuous-reward"
add no-relu-cont                  "--model no-relu --continuous-reward"
add id-only-mf-cont               "--model id-only-mf --continuous-reward"      
add no-l2-cont                    "--model no-l2 --continuous-reward"
add weighted-loss-cont            "--model weighted-loss --continuous-reward"
add dim-16-cont                   "--model dim-16 --continuous-reward"
add dim-64-cont                   "--model dim-64 --continuous-reward"

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
                --chdir="${DLB_DIR}" \
                --output="${LOG_DIR}/sim_${name}_%j.out" \
                --error="${LOG_DIR}/sim_${name}_%j.err" \
                --export=ALL,DLB_DIR="${DLB_DIR}",SIM_FLAGS="${flags}" \
                --array=0-$((SEED_COUNT-1)) \
                "${SIM_SCRIPT}" \
        )

        # Dispatch collate job too
        sbatch --job-name="col-${name}"  \
            --dependency=afterok:$JOB_ID \
            --chdir="${DLB_DIR}" \
            --output="${LOG_DIR}/col_${name}_%j.out" \
            --error="${LOG_DIR}/col_${name}_%j.err" \
            --export=ALL,DLB_DIR="${DLB_DIR}",SAVE_NAME="${name}" \
            "${COL_SCRIPT}"
    else
        sbatch --job-name="sim-${name}" \
           --chdir="${DLB_DIR}" \
           --output="${LOG_DIR}/sim_${name}_%j.out" \
           --error="${LOG_DIR}/sim_${name}_%j.err" \
           --export=ALL,DLB_DIR="${DLB_DIR}",SIM_FLAGS="${flags}" \
           "${SIM_SCRIPT}"
    fi
done

echo "All jobs submitted."