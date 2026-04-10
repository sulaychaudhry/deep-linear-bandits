#!/bin/bash
#
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000
#SBATCH --time=2-00:00:00

echo -e "Starting job at $(date)\n"
echo -e "Running:\n\t uv run dlb simulate ${SIM_FLAGS}"

if [ "$PARALLEL_SEEDS" = true ]; then
    echo "Running seed-index=$SLURM_ARRAY_TASK_ID"
    uv run dlb simulate ${SIM_FLAGS} --seed-index=$SLURM_ARRAY_TASK_ID
else
    uv run dlb simulate ${SIM_FLAGS}
fi

echo "Job finished at $(date)"