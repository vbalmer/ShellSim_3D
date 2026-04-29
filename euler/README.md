# Running ShellSim_3D on Euler

This folder contains everything needed to run the sampling + training pipeline
on the ETH Euler HPC. Two SLURM jobs are submitted, the training job depends
on the sampling job (`--dependency=afterok`).

## Files

| File | Purpose |
|---|---|
| `requirements_euler.txt` | Python deps pinned to `stack/2024-06 + python_cuda/3.11.6`. |
| `setup_env.sh` | One-time: load modules, create `$HOME/venvs/shellsim`, install deps. |
| `activate.sh` | Sourced by every job to load modules + activate the venv. |
| `sample.sbatch` | SLURM job: runs `sampling/sampler_analytical_RC3D.py`. |
| `train.sbatch` | SLURM job: runs `training/train.py`. |
| `submit.sh` | Submits both jobs, chained via `--dependency=afterok`. |
| `smoke_sample.sbatch` | Smoke variant of `sample.sbatch` (tiny dataset, 30 min). |
| `smoke_train.sbatch` | Smoke variant of `train.sbatch` (tiny model, 3 epochs, 30 min). |
| `submit_smoke.sh` | Submits the smoke pipeline. Run this before `submit.sh`. |

## I/O conventions

The two scripts now read three environment variables (defaults shown):

| Variable | Default on Euler | What lives there |
|---|---|---|
| `DATA_DIR` | `$SCRATCH/ShellSim3D/data` | Sampled HDF5 files (large, multi-TB). |
| `MODEL_DIR` | `$HOME/ShellSim3D/models/current` | `*.pkl` configs + the trained `.pt` weights. |
| `LOGS_DIR` | `$HOME/ShellSim3D/models/logs` | Per-run archive folders (`v_0`, `v_1`, ...). |

If none are set, the scripts fall back to the original Windows path
`D:\VeraBalmer\ShellSim3D` so local runs on your remote desktop keep working.

`$SCRATCH` is the right place for the sampled data: it's the largest, fastest
filesystem on Euler, but **files older than ~15 days are auto-purged**. Once
training finishes you can `cp -r $MODEL_DIR /cluster/work/<group>/...` (or
similar) to keep the model long-term.

## First-time setup (do this once)

```bash
# On the Euler login node:
ssh <username>@euler.ethz.ch
cd $HOME

# Clone the repo (or rsync from your remote desktop):
git clone <your-repo-url> ShellSim_3D
cd ShellSim_3D

# Create the venv and install deps. ~5-10 min the first time.
bash euler/setup_env.sh

# One-time wandb auth (or set WANDB_MODE=offline in train.sbatch).
source $HOME/venvs/shellsim/bin/activate
wandb login        # paste API key from https://wandb.ai/authorize
```

## Smoke test (recommended -- run this first)

Before committing real compute to a sampling/training run, validate the whole
Euler pipeline end-to-end on a tiny dataset:

```bash
cd $HOME/ShellSim_3D
bash euler/submit_smoke.sh
```

What this exercises:

- modules + venv activate inside a SLURM job;
- `cupy` finds the GPU and the sampler runs end-to-end;
- HDF5 files land at `$SCRATCH/ShellSim3D/smoke/data/`;
- the training job sees the chained `--dependency=afterok` and starts;
- `torch` + `lightning` find the GPU, train.py reads the env vars, and a
  full set of model artefacts (`*.pt`, `inp.pkl`, `stats.pkl`,
  `test_data.pkl`) lands at `$HOME/ShellSim3D/smoke/models/`;
- a versioned snapshot is copied to `$HOME/ShellSim3D/smoke/logs/v_<n>/`.

Smoke defaults (override on the command line if useful):

| Var | Default | Meaning |
|---|---|---|
| `N_SAMPLES_3D` | `1e6` | passed to `get_constant_sampling_params` |
| `NUM_EPOCHS` | `3` | overrides `inp['num_epochs']` |
| `BATCH_SIZE` | `1024` | overrides `inp['batch_size']` |
| `HIDDEN_LAYERS` | `[256, 256, 256]` | overrides `inp['hidden_layers']` |
| `WANDB_MODE` | `offline` | so smoke runs don't pollute the dashboard |

Runtime: ~5-10 min per stage on any Euler GPU. Resources requested:
`--gpus=1 --gres=gpumem:8g --mem-per-cpu=4G --time=00:30:00`.

The smoke pipeline writes to `$SCRATCH/ShellSim3D/smoke/...` and
`$HOME/ShellSim3D/smoke/...`, **separate from the production paths**, so it
can never overwrite a real run.

If both jobs come back `COMPLETED` and the model artefacts exist, the
production submission below is good to go.

## Submitting a run

```bash
cd $HOME/ShellSim_3D

# Submits sample.sbatch + train.sbatch (the latter waits on the former).
bash euler/submit.sh

# Override I/O locations on the fly:
DATA_DIR=$SCRATCH/foo MODEL_DIR=$HOME/bar bash euler/submit.sh

# Skip sampling (re-use existing $DATA_DIR contents) and only train:
bash euler/submit.sh --train-only
```

Then:
```bash
squeue --me                              # see job state
tail -f euler/logs/sample-<id>.out       # follow sampling output
tail -f euler/logs/train-<id>.out        # follow training output
```

## Resource defaults

Both jobs request:

```
--gpus=1 --gres=gpumem:20g
--cpus-per-task=4 --mem-per-cpu=8G   (=> 32 GB RAM)
--time=24:00:00
```

Tweak in the `#SBATCH` headers of `sample.sbatch` / `train.sbatch` if needed.
For larger datasets bump `--mem-per-cpu`; for faster sampling raise
`--cpus-per-task` (CuPy itself uses the GPU, but the pyDOE/numpy stages are
multi-core).

## How big is "as much data as possible"?

`sampling/sampler_utils_RC3D.py:get_constant_sampling_params` sets
`n_samples_3D = 6e6`. Since you asked for the largest dataset Euler can
hold, you can raise that and re-run:

- `eps_g`, `sig_g`: float32, shape `(N, 6)` -> 24 N bytes each
- `D`: float32, shape `(N, 6, 6)` -> 144 N bytes
- Total per sample: **~192 bytes**. So 6e6 samples = ~1.15 GB.

`$SCRATCH` quotas on Euler are normally ~2.5 TB per user, which gives you
headroom to crank `n_samples_3D` up to roughly **1e10** (~1.9 TB) before
storage becomes the limit. GPU memory will become the bottleneck before that
though -- the script already batches stress simulation (`n_batches=6`); raise
that if you go above ~30M samples.

## Moving the trained model elsewhere later

The model lives at `$MODEL_DIR` (default `$HOME/ShellSim3D/models/current`).
Copy it wherever you need:

```bash
# Onto a group share for permanent storage:
cp -r $HOME/ShellSim3D/models/current /cluster/work/<group>/shellsim/v_<n>

# Down to your local machine via scp / rsync:
rsync -avh euler:ShellSim3D/models/current/ ./local_model/
```

## What was changed in the source tree

To make the scripts portable to Linux / Euler, four files in the repo were
patched (see the diffs in your VCS):

- `sampling/sampler_analytical_RC3D.py` -- reads `DATA_DIR` and `N_SAMPLES_3D`;
  plot path now resolved relative to `__file__` (was a Windows-only `cwd` path).
- `sampling/sampler_utils_RC3D.py` -- same plot-path fix.
- `training/train.py` -- reads `DATA_DIR`, `MODEL_DIR`, `LOGS_DIR`,
  `NUM_EPOCHS`, `BATCH_SIZE`, `HIDDEN_LAYERS`.
- `training/data_utils.py` -- `os.path.join` instead of literal `\\`.
- `training/train_utils.py` -- archive folder honours `LOGS_DIR`.
- `training/test_utils.py` -- default `plot_path` resolved relative to `__file__`.

The defaults are unchanged for Windows runs (no env vars set => original
`D:\VeraBalmer\ShellSim3D` is used), so your remote-desktop workflow is not
affected.
