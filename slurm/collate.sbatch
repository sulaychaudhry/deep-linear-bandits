#!/bin/bash
#
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --time=0-00:30:00

echo -e "Starting job at $(date)\n"
echo -e "Running:\n\t uv run dlb collate --save-name ${SAVE_NAME}"

uv run dlb collate --save-name ${SAVE_NAME}

echo "Job finished at $(date)"