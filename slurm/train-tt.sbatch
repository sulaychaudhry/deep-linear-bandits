#!/bin/bash
#
#SBATCH --job-name=train-tt
#SBATCH --partition=gecko
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=/dcs/23/u5567816/deep-linear-bandits/slurm/joboutput_%j.out
#SBATCH --error=/dcs/23/u5567816/deep-linear-bandits/slurm/joboutput_%j.err

uv run dlb train-tt \
    --save-name test-1