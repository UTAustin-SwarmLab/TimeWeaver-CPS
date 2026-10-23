# Running the experiments

This page shows how to run each experiment in the codebase. The commands are
adapted from the CPS NeurIPS 2025 supplementary material. The examples use the
traffic dataset. For another dataset, change the config name, the
`<dataset>_dataset.log_dir` key, and the checkpoints.

Run everything from the repository root. Hydra changes the working directory at
run time, so every path must be **absolute**. Set these once per shell:

```bash
REPO=$PWD
BASE_DIR=$REPO/runs                      # where training writes checkpoints and MLflow logs
DATA=$REPO/data/traffic                  # $REPO/data/air_quality, $REPO/data/stocks, ...
DIFFUSION_CKPT=$REPO/logs/diffusion_model_logs/traffic_conditional/checkpoints/traffic_conditional.ckpt
GAN_CKPT=$REPO/logs/gan_logs/traffic_conditional.ckpt
CLTSP_CKPT=$REPO/logs/cltsp_logs/traffic_conditional.ckpt
SYNTHETIC_DIR=$REPO/logs/diffusion_model_logs/traffic_conditional/
```

See the [README](../README.md#data-and-checkpoints) for downloading the data and
checkpoints.

## Training

```bash
# Conditional diffusion model (Time Weaver)
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_timeweaver_diffusion.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA

# Unconditional diffusion model
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_timeweaver_diffusion.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA \
  csdi_timeseries_denoiser_v4_config.use_metadata=False

# Conditional GAN
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_timeweaver_gan.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA \
  wavegan_v1_config.use_metadata=True wavegan_v1_config.generator_config.blow_up_factor=10

# Unconditional GAN
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_timeweaver_gan.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA \
  wavegan_v1_config.use_metadata=False wavegan_v1_config.generator_config.blow_up_factor=30

# FTSD feature extractor
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_cltsp_metric.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA \
  training.use_timeseries_scl_loss=True training.use_condition_scl_loss=True training.use_clip_loss=False

# J-FTSD feature extractor
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_cltsp_metric.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA
```

Set `use_metadata` and `generator_config.blow_up_factor` to the values for
your dataset.

The diffusion model trains for 5000 epochs by default (`training.max_epochs`).
On one GPU, a traffic run takes about 5 hours; air quality takes about a minute
per epoch. The best checkpoint (lowest validation loss) is saved as
`$BASE_DIR/Time_Series_Diffusion_Training/<dataset>/<run_name>/<run_id>/checkpoints/best_model.ckpt`.

## Storing checkpoints

Copy each trained checkpoint to the location the generation and evaluation
commands expect. For example, a conditional traffic diffusion model goes to
`logs/diffusion_model_logs/traffic_conditional/checkpoints/traffic_conditional.ckpt`.
GAN and CLTSP checkpoints go to `logs/gan_logs/<dataset>.ckpt` and
`logs/cltsp_logs/<dataset>.ckpt`. The pretrained checkpoints download already
uses this layout.

## Generating synthetic data

```bash
GENERATE="poetry run python timeseries_synthesis/scripts/generate_constrained_synthetic_dataset.py \
  --config-name=traffic_conditional.yaml \
  synthesizer_wrapper_checkpoint_path=$DIFFUSION_CKPT \
  traffic_dataset.log_dir=$DATA \
  discriminator_params.discriminator_wrapper_checkpoint_path=$GAN_CKPT"

$GENERATE                                                                       # unconstrained generation
$GENERATE use_projection=True project_during_synthesis=True use_penalty_based_projection=True   # CPS
$GENERATE use_projection=True project_after_synthesis=True use_real_seed=True   # COP
$GENERATE use_projection=True project_after_synthesis=True                      # COP-FT (needs unconstrained generation first)
$GENERATE use_guidance=True                                                     # Diffusion-TS
$GENERATE use_diffts=True                                                       # Guided-DiffTime
$GENERATE use_projection=True project_during_synthesis=True use_strict_projection=True                  # PDM
$GENERATE use_projection=True project_during_synthesis=True use_strict_projection=True use_prodigy=True # PRODIGY
```

Each run writes `train/`, `val/`, and `test/` folders of `timeseries_<i>.npy`
files to its own folder under `logs/diffusion_model_logs/traffic_conditional/`.
Unconstrained samples go to `normal/`. CPS samples go to
`11_constraints/projection_during_synthesis_with_penalty_based_projection_using_exp/`.

For the stocks OHLC constraint, also export `STOCKS_DATASET_DIR=$REPO/data/stocks`.

## Evaluating generated samples

Each evaluation script takes the method flags from the previous section to
choose which samples to score:

```bash
EVAL_ARGS="--config-name=traffic_conditional_eval.yaml \
  similarity_checker_wrapper_checkpoint_path=$CLTSP_CKPT \
  synthetic_dataset_dir=$SYNTHETIC_DIR \
  traffic_dataset.log_dir=$DATA"

# Constraint violation
poetry run python timeseries_synthesis/scripts/compute_constraint_violation_rate.py $EVAL_ARGS <method flags>
# FTSD or J-FTSD
poetry run python timeseries_synthesis/scripts/generate_timeseries_and_condition_embeddings.py $EVAL_ARGS <method flags>
# DTW
poetry run python timeseries_synthesis/scripts/compute_dtw_metric.py $EVAL_ARGS <method flags>
```

J-FTSD is the Time Weaver metric. The script prints `The JFTSD score is : ...`
when `use_conditions: True`, and FTSD otherwise.

## Varying the number of constraints

Generate with `traffic1.yaml`, `traffic3.yaml`, ..., `traffic9.yaml` in place of
the generation config above. To evaluate, use `traffic_eval.yaml` and set
`traffic_dataset.equality_constraints_to_extract` to match.

## Models trained with constraints

```bash
# Train Loss-DiffTime, a diffusion model trained with constraints
poetry run python timeseries_synthesis/scripts/train_pl_model.py --config-name=config_timeweaver_constrained_diffusion.yaml \
  base_path=$BASE_DIR dataset_name=traffic traffic_dataset.log_dir=$DATA

CONSTRAINED_CKPT=$REPO/logs/loss_difftime_logs/traffic/checkpoints/traffic.ckpt

# Generate
poetry run python timeseries_synthesis/scripts/generate_constrained_synthetic_dataset.py --config-name=traffic_trained.yaml \
  synthesizer_wrapper_checkpoint_path=$CONSTRAINED_CKPT traffic_dataset.log_dir=$DATA

# FTSD. DTW and constraint violation take the same arguments
poetry run python timeseries_synthesis/scripts/generate_timeseries_and_condition_embeddings.py --config-name=traffic_trained_eval.yaml \
  similarity_checker_wrapper_checkpoint_path=$CLTSP_CKPT synthetic_dataset_dir=$SYNTHETIC_DIR traffic_dataset.log_dir=$DATA
```

## Inference latency

```bash
LATENCY="poetry run python timeseries_synthesis/scripts/compute_inference_time.py \
  synthesizer_wrapper_checkpoint_path=$DIFFUSION_CKPT traffic_dataset.log_dir=$DATA"

# CPS with 1, 3, 5, ... constraints
$LATENCY --config-name=traffic1.yaml use_projection=True project_during_synthesis=True use_penalty_based_projection=True
$LATENCY --config-name=traffic3.yaml use_projection=True project_during_synthesis=True use_penalty_based_projection=True

# All constraints, CPS and baselines
$LATENCY --config-name=traffic.yaml use_projection=True project_during_synthesis=True use_penalty_based_projection=True
$LATENCY --config-name=traffic.yaml use_projection=True project_after_synthesis=True \
  discriminator_params.discriminator_wrapper_checkpoint_path=$GAN_CKPT          # COP-FT
$LATENCY --config-name=traffic.yaml use_guidance=True                           # Guided-DiffTime
```

## Non-convex constraints

```bash
STOCKS=$REPO/data/stocks
poetry run python timeseries_synthesis/scripts/generate_constrained_synthetic_test_set.py --config-name=stocks_non_convex.yaml \
  synthesizer_wrapper_checkpoint_path=$REPO/logs/diffusion_model_logs/stocks/checkpoints/stocks.ckpt \
  discriminator_params.discriminator_wrapper_checkpoint_path=$REPO/logs/gan_logs/stocks.ckpt \
  stocks_dataset.log_dir=$STOCKS <method flags>

poetry run python timeseries_synthesis/scripts/generate_timeseries_and_condition_embeddings.py --config-name=stocks_non_convex_eval.yaml \
  similarity_checker_wrapper_checkpoint_path=$REPO/logs/cltsp_logs/stocks.ckpt \
  synthetic_dataset_dir=$REPO/logs/diffusion_model_logs/stocks/ stocks_dataset.log_dir=$STOCKS <method flags>
```

`compute_dtw_metric.py` and `compute_constraint_violation_rate.py` take the same
arguments.

## Choice of penalty coefficients

```bash
poetry run python timeseries_synthesis/scripts/generate_constrained_synthetic_test_set.py --config-name=traffic.yaml \
  synthesizer_wrapper_checkpoint_path=$DIFFUSION_CKPT traffic_dataset.log_dir=$DATA \
  discriminator_params.discriminator_wrapper_checkpoint_path=$GAN_CKPT \
  use_projection=True project_during_synthesis=True use_penalty_based_projection=True gamma_choice=lin
```

For the quadratic schedule, use `gamma_choice=quad`. Evaluate as in
[Evaluating generated samples](#evaluating-generated-samples).

## TSTR

TSTR uses the fork of Time-Series-Library in `evaluation/Time-Series-Library/`,
in its own environment:

```bash
conda env create -f evaluation/Time-Series-Library/cps_neurips_tstr.yml
```

If the pinned `torch==2.4.1+cu121` wheels don't resolve, install the yml's pip
packages with `--extra-index-url https://download.pytorch.org/whl/cu121`.

Each method has a script in
`evaluation/Time-Series-Library/scripts/tstr_imputation_scripts/<dataset>_scripts/`.
Set its two paths:

```bash
real_data_dir=$REPO/data/traffic
synthetic_data_dir=$REPO/logs/diffusion_model_logs/traffic_conditional/normal   # must contain train/ val/ test/
```

Then run the script from `evaluation/Time-Series-Library/`. It trains TimesNet
to impute 75% masked values on the synthetic `train`/`val` split and tests on the
real test set. It does this for 3 seeds and prints `mse:..., mae:...` for each.
Report the mean and standard deviation over the seeds.

## Discriminative score

The discriminative score uses the fork of TimeGAN in `evaluation/TimeGAN/`:

```bash
conda env create -f evaluation/TimeGAN/cps_neurips_ds.yml
```

Set `real_timeseries_dir` and `synthetic_timeseries_dir` for your dataset in
`get_real_and_synthetic_paths` in `metrics_computation.py`. Then run it from
`evaluation/TimeGAN/`:

```bash
python metrics_computation.py traffic_conditional
```

It prints the discriminative score for 5 seeds.

## Troubleshooting

- **Poetry rejects the Python version.** `pyproject.toml` pins Python 3.9.12
  exactly. If your conda Python is a different 3.9.x, run the scripts with the
  environment's `python` directly, with `PYTHONPATH=$REPO`.
- **`OSError: [Errno 5]` from the dataloader on a network filesystem.** The
  dataloader workers create Unix sockets in `TMPDIR`, so point `TMPDIR` at a
  local directory such as `/dev/shm/tmp`.
