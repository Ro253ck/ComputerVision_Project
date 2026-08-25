#!/bin/bash
# Script SOLO diagnostico: verifica se "fork" funziona per pushcube su una
# GPU dedicata (invece che sulla GPU condivisa dei primi test), con
# parametri piccoli per avere una risposta rapida. Non e' il run vero -
# quello resta scripts/plan_pushcube.sh (con "spawn").
#SBATCH --job-name=dinowm_plan_pushcube_TEST_FORK
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/cvcs2026/LubiMoRe/logs/plan_pushcube_test_fork_%j.out
#SBATCH --error=/work/cvcs2026/LubiMoRe/logs/plan_pushcube_test_fork_%j.err
#SBATCH --account=cvcs2026

set -eo pipefail
umask 002

MODEL_NAME="dinowm_pushcube_dino_run1000"
MODEL_EPOCH=20
BASE_PATH="/work/cvcs2026/LubiMoRe/dino_wm"
DATASET_SUBDIR="pushcube_noise"
DATA_PATH="$BASE_PATH/data/$DATASET_SUBDIR"

module purge
source /work/cvcs2026/LubiMoRe/dino_environment/bin/activate

cd "$BASE_PATH"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DINO_WM_FORCE_FORK=1   # <-- forza "fork" invece di "spawn", solo per questo test

HYDRA_FILE="$BASE_PATH/outputs/$MODEL_NAME/hydra.yaml"
sed -i "s#/tmp/dino_dataset/$DATASET_SUBDIR#$DATA_PATH#g" "$HYDRA_FILE"

python plan.py \
  --config-name plan_pushcube.yaml \
  ckpt_base_path="$BASE_PATH" \
  model_name="$MODEL_NAME" \
  model_epoch="$MODEL_EPOCH" \
  planner.sub_planner.num_samples=30 \
  planner.sub_planner.opt_steps=30 \
  n_evals=2 \
  run_tag=_test_fork \
  seed=99 \
  planner.max_iter=2
