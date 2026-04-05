#!/bin/bash
# Submits all two-tower training jobs to Slurm

SBATCH_SCRIPT="slurm/train-tt.sbatch"
DLB_DIR="/dcs/23/u5567816/deep-linear-bandits"
LOG_DIR="${DLB_DIR}/slurm"

COMMON="--epochs 100 --seed 117"

NAMES=()
FLAGS=()

add() { NAMES+=("$1"); FLAGS+=("$2"); }

# Default (1 hidden layer, 128-wide) and different tower architectures
add default             ""
add main                "--hidden-size 256 --hidden-size 128"
add deep                "--hidden-size 256 --hidden-size 128 --hidden-size 64"

# Look at potential bottlenecks: strip down depth of architecture
add narrow-64-out-32    "--hidden-size 64 --output-size 32"
add narrow-48-out-32    "--hidden-size 48 --output-size 32"
add narrow-64-out-16    "--hidden-size 64 --output-size 16"
add narrow-48-out-16    "--hidden-size 48 --output-size 16"

# Look at how ID dimensions are affecting things:
add n48o16-id16         "--id-emb-dims 16 --hidden-size 48 --output-size 16"
add n48o16-itemftr32    "--item-cat-emb-dims 32 --hidden-size 48 --output-size 16"

# Try using score-weighted negative sampling
# (i.e. informative per-user hard negatives not just easy per-batch uniform ones)
add hard-negs-0         "--negative-sampling score-weighted --score-sharpness 0" # Per-user uniform negatives
add hard-negs-05        "--negative-sampling score-weighted --score-sharpness 0.5"
add hard-negs-1         "--negative-sampling score-weighted --score-sharpness 1"
add hard-negs-5         "--negative-sampling score-weighted --score-sharpness 5"
add hard-negs-10        "--negative-sampling score-weighted --score-sharpness 10"
add hard-negs-50        "--negative-sampling score-weighted --score-sharpness 10"
add hard-negs-100       "--negative-sampling score-weighted --score-sharpness 10"

# Look at how it performs on the deeper architectures too
add hn1-main            "--negative-sampling score-weighted --score-sharpness 1 --hidden-size 256 --hidden-size 128"
add hn5-main            "--negative-sampling score-weighted --score-sharpness 5 --hidden-size 256 --hidden-size 128"
add hn10-main           "--negative-sampling score-weighted --score-sharpness 10 --hidden-size 256 --hidden-size 128"
add hn1-deep            "--negative-sampling score-weighted --score-sharpness 1 --hidden-size 256 --hidden-size 128 --hidden-size 64"
add hn5-deep            "--negative-sampling score-weighted --score-sharpness 5 --hidden-size 256 --hidden-size 128 --hidden-size 64"
add hn10-deep           "--negative-sampling score-weighted --score-sharpness 10 --hidden-size 256 --hidden-size 128 --hidden-size 64"

# Try varying the learning rate to see if it can learn faster
add hn1-main-lr-2em3    "--negative-sampling score-weighted --score-sharpness 1 --hidden-size 256 --hidden-size 128 --lr 0.002"
add hn1-main-lr-3em3    "--negative-sampling score-weighted --score-sharpness 1 --hidden-size 256 --hidden-size 128 --lr 0.003"

# Side features ablation
add no-sidefeats        "--no-side-features"

# Linear towers (no non-linearity)
add no-relu             "--no-relu"

# MF baseline (id-emb-dims=64 to match default output dims without tower transformation)
add mf-baseline         "--skip-towers --no-side-features --id-emb-dims 64"

# Output embedding dimensionality (i.e. context vector size for bandits)
add dim-32              "--output-size 32"
add dim-128             "--output-size 128"

# Effects of L2 normalisation
add no-l2               "--no-l2-norm"

# Logit temperature (default is 0.07)
add temp-005            "--logit-temp 0.05"
add temp-01             "--logit-temp 0.1"

# Show effects of in-batch negative sampling
add in-batch-neg        "--negative-sampling in-batch"

# Show effects of weight decay
add adamw               "--optimiser adamw"

# Watch threshold (default is 2.0)
add wt-1                "--watch-threshold 1.0"
add wt-3                "--watch-threshold 3.0"

# Number of uniform negatives (default is 256)
add neg-64              "--num-negatives 64"
add neg-512             "--num-negatives 512"

# Dropout changes (default is 0.2)
add no-dropout          "--dropout 0.0"
add dropout-04          "--dropout 0.4"

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