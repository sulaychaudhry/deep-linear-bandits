#!/bin/bash
# Submits all two-tower training jobs to Slurm

SBATCH_SCRIPT="slurm/train-tt.sbatch"
DLB_DIR="/dcs/23/u5567816/deep-linear-bandits"
LOG_DIR="${DLB_DIR}/slurm"

COMMON="--epochs 100 --seed 117"

NAMES=()
FLAGS=()

add() { NAMES+=("$1"); FLAGS+=("$2"); }

# Default config: (hidden sizes [256, 128], output dims 32)
add default             ""

# Different tower configurations
add narrow              "--hidden-size 128 --hidden-size 64"
add narrowest           "--hidden-size 64  --hidden-size 32"
add wide                "--hidden-size 512 --hidden-size 256"
add deep                "--hidden-size 256 --hidden-size 128 --hidden-size 64"
add shallow             "--hidden-size 256"

# Ablations
add no-sidefeats        "--no-side-features"
add no-relu             "--no-relu"
add mf-baseline         "--skip-towers --no-side-features"
add no-l2               "--no-l2-norm"

# Output embedding dimensionality (default is 32)
add dim-16              "--output-size 16"
add dim-64              "--output-size 64"

# Logit temperature (i.e. training sharpness; default is 0.07, lower=sharper)
add temp-005            "--logit-temp 0.05"
add temp-01             "--logit-temp 0.1"

# Different sampling techniques (default: global uniform)
add in-batch-neg        "--negative-sampling in-batch"
add user-uniform-neg    "--negative-sampling score-weighted --score-sharpness 0"
add score-weight-neg1   "--negative-sampling score-weighted --score-sharpness 1"
add score-weight-neg2   "--negative-sampling score-weighted --score-sharpness 3"
add wr-banded-neg1      "--negative-sampling watch-ratio --wr-band-ratio 0.15 0.25 0.25 0.2 0.15 --dropout 0.4"
add wr-banded-neg2      "--negative-sampling watch-ratio --wr-band-ratio 0.10 0.35 0.30 0.15 0.10 --dropout 0.4"

# Number of negatives if not in-batch (default is 256)
add neg-64              "--num-negatives 64"
add neg-512             "--num-negatives 512"

# Varying dropout (default is 0.2)
add no-dropout          "--dropout 0.0"
add dropout-04          "--dropout 0.4"

# Quick look at weight decay
add adamw               "--optimiser adamw"

# Varying learning rate (default is 1e-3)
add lr-5em4             "--lr 0.0005"
add lr-2em3             "--lr 0.002"

# Watch threshold (work focuses around 2.0, but interesting to see)
add wt-1                "--watch-threshold 1.0"
add wt-3                "--watch-threshold 3.0"

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
           --output="${LOG_DIR}/joboutput_%j.out" \
           --error="${LOG_DIR}/joboutput_%j.err" \
           --export=ALL,TT_FLAGS="${flags}" \
           "${SBATCH_SCRIPT}"
done

echo "All jobs submitted."