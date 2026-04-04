#!/bin/bash
#
#SBATCH --partition=gecko
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

## Note that --output, --error, TT_FLAGS are set by `./train-all-tts.sh`

echo -e "Starting job at $(date)\n"
echo -e "Running:\n\t uv run dlb train-tt ${TT_FLAGS}"

uv run dlb train-tt ${TT_FLAGS}

echo "Job finished at $(date)"