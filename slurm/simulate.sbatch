#!/bin/bash
#
#SBATCH --partition=tiger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=2000
#SBATCH --time=2-00:00:00
#SBATCH --exclusive=user

echo -e "Starting job at $(date)\n"
echo -e "Running:\n\t uv run dlb simulate ${SIM_FLAGS}"

# If empty, just run normally, else use the task ID
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    uv run dlb simulate ${SIM_FLAGS}
else
    echo "Running seed-index=$SLURM_ARRAY_TASK_ID"
    uv run dlb simulate ${SIM_FLAGS} --seed-index=$SLURM_ARRAY_TASK_ID
fi

echo "Job finished at $(date)"