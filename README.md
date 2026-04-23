# Deep Linear Bandits

## Overview

This project explores hybrid systems that mix two-tower models for representation learning (on a fixed update latency, batch-trained offline) with contextual linear bandit policies that enable these models to adapt online in real-time without the scalability issues of an end-to-end neural bandit.

Specifically, it inspects the frozen update window in which the two-tower must deploy its static (gradually staling) policy, missing out on capturing additional reward and the ability to explore its uncertainty about the underlying reward geometry.

This is motivated by observations that two-tower models are generally exploitation-heavy, and tend to fall into degenerative feedback loops that amplify their biases; contextual linear bandits provide a means of principled real-time exploration.

High-level steps:
1. Train and validate a two-tower architecture on KuaiRec-Big.
2. Run bandit simulations on KuaiRec-Small using the trained embeddings.
3. Compare reward, regret, and all beyond-accuracy metrics.

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/), if not already installed

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync necessary Python dependencies

```bash
# CPU build
uv sync --extra cpu

# GPU build (CUDA 12.8)
uv sync --extra gpu
```

3. Download the [KuaiRec dataset](https://kuairec.com/) (just `KuaiRec.zip`) and ensure data files are at `kuairec/data`

```bash
wget https://zenodo.org/records/18164998/files/KuaiRec.zip
unzip KuaiRec.zip
mv "KuaiRec 2.0"/ kuairec/
```

## Project Root

The CLI uses `DLB_DIR` as the project root. If this is not set, it will use your current working directory. As such, it is recommended that commands are all run from the project root `/.../deep-linear-bandits`.

If you want to run commands from anywhere, export `DLB_DIR` to your shell:
```bash
export DLB_DIR=/.../deep-linear-bandits
```

## Main Commands

You can use `uv run dlb --help` at any time to see all commands.

### 1) Training a two-tower model

```bash
uv run dlb train-tt --save-name default
```
Use `uv run dlb train-tt --help` to see all two-tower parameters, including defaults.  
Outputs are written to `tt-models/<model-name>/`.  

### 2) Running a simulation

```bash
uv run dlb simulate --save-name sim1 --model default
```
Use `uv run dlb simulate --help` to see all simulation parameters, including defaults.  
Outputs are written to `simulations/<sim-name>`.

## Example Local Run (GPU)
```bash
uv run dlb train-tt --save-name tt1 --hidden-size 512 --hidden-size 256 --output-size 32
uv run dlb simulate --save-name sim1 --model-name tt1 --seed 125 --rounds 100
```

## Additional Commands

### Collating simulation jobs (`uv run dlb collate`)

The `simulate` command by default only runs the simulation for one seed. However, it is possible to run the simulation for e.g. $10$ seeds:
```bash
uv run dlb simulate --save-name sim2 --model default --seed 117 --seed-count 10
```
In this case, the master seed $117$ will spawn $10$ child seeds and run $10$ simulations sequentially before collating the results to save a single set of plots and metrics, using the **mean $\pm$ std** of each policy across all $10$ simulations.

This is incredibly slow especially with a larger `--seed-count`, and so `uv run dlb simulate` supports splitting these simulations to be run as different jobs in parallel via `--seed-index` (0-indexed).
```bash
# On worker thread 1
uv run dlb simulate --save-name <sim-name> --model default --seed 117 --seed-count 100 --seed-index 0

...

# On worker thread 2
uv run dlb simulate --save-name <sim-name> --model default --seed 117 --seed-count 100 --seed-index 1

...

# On worker thread 100
uv run dlb simulate --save-name <sim-name> --model default --seed 117 --seed-count 100 --seed-index 99
```
The separate worker threads will each write their simulation results to `simulations/<sim-name>/seed_<seed-index>.npz`. After all jobs terminate and you have verified that all seed files have been written to the folder, you can collate the results to get the plots and metrics:
```bash
uv run dlb collate --save-name <sim-name>
```
Use `uv run dlb collate --help` for more information.  
Outputs are written to `simulations/<sim-name>`.

### Regenerating simulation plots from saved metrics

```bash
uv run dlb plot --save-name <sim-name>
```
Use `uv run dlb plot --help` for more information.  
Outputs overwrite `simulations/<sim-name>`.  

### Helper: rank all trained models
```bash
uv run dlb-helpers best-tt --sort-by recall --k 50
```
Use `uv run dlb-helpers best-tt --help` for more information.

## Slurm Workflow

Various scripts are available under `slurm/` for the use of the DCS Batch Compute System. In reality, you only need to modify these files:
- `slurm/train-all-tts.sh`
- `slurm/simulate-all.sh`

`COMMON` denotes common flags passed to all models.  
To add a model or simulation respectively (ready to be dispatched), simply add a new line of this format below the `add()` function:
```bash
# In `train-all-tts,sh`
add <model-name>           "<two-tower flags>"

# In `simulate-all.sh`
add <sim-name>             "<simulation flags>"
```
Note that `--save-name` does not need passing as a flag; this is handled automatically, and again you do not need to pass the `COMMON` flags.

For `slurm/simulate-all.sh`, note that the script is set up the parallelise the seeds and will collate them automatically for you; you simply need to change the `SEED_COUNT` variable. You may set `PARALLEL_SEEDS=false` if you would like, but there is no benefit to this.

Then for dispatching the two-tower jobs:
```bash
slurm/train-all-tts.sh
```

Or for dispatching the simulation jobs:
```bash
slurm/simulate-all.sh
```

All logs are written to `slurm/logs/`.

## Notes

- If you can't run the scripts in `slurm/`, try running `chmod u+x slurm/*`