#!/bin/bash
#SBATCH --job-name=dinowm_plan_pushcube_dino_1000_99
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/cvcs2026/LubiMoRe/logs/plan_pushcube_dino_1000_%j.out
#SBATCH --error=/work/cvcs2026/LubiMoRe/logs/plan_pushcube_dino_1000_%j.err
#SBATCH --account=cvcs2026

set -eo pipefail   # esce subito se un comando fallisce, invece di proseguire alla cieca

# Permessi di gruppo sui file creati da questo job
umask 002

# ============================================================
# Parametri da cambiare per ogni combinazione modello/seed
# ============================================================
MODEL_NAME="dinowm_pushcube_dino_run1000"   # Modello con cui vogliamo fare il planning
MODEL_EPOCH=20                              # Checkpoint del modello da usare
NUM_SAMPLES=300                       # Traiettorie generate da CEM per capire le azioni migliori
OPT_STEPS=30                            # Iterazioni di raffinamento CEM (delle 300, tengo le migliori e rigenero)
N_EVALS=15                              # Numero di task di planning valutati
                                         # (piu' basso che per PushT/Wall: ogni sottoprocesso qui
                                         # usa "spawn" e si crea un proprio contesto CUDA/SAPIEN,
                                         # ~1GB l'uno - con 50 si esaurisce la GPU da 45GB insieme
                                         # al modello caricato nel processo principale)
SEED=99

RUN_TAG="_${MODEL_EPOCH}_seed${SEED}"

BASE_PATH="/work/cvcs2026/LubiMoRe/dino_wm"
CONFIG_NAME="plan_pushcube.yaml"
DATASET_SUBDIR="pushcube_noise"
DATA_PATH="$BASE_PATH/data/$DATASET_SUBDIR"

module purge
source /work/cvcs2026/LubiMoRe/dino_environment/bin/activate

cd "$BASE_PATH"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Correzione automatica del hydra.yaml del modello: il data_path punta al
# TMPDIR morto del training, va corretto puntandolo ai dati persistenti.
HYDRA_FILE="$BASE_PATH/outputs/$MODEL_NAME/hydra.yaml"
if [ -f "$HYDRA_FILE" ]; then
    echo "Correggo $HYDRA_FILE ..."
    sed -i "s#/tmp/dino_dataset/$DATASET_SUBDIR#$DATA_PATH#g" "$HYDRA_FILE"
    echo "data_path corretto."
    grep -E "data_path" "$HYDRA_FILE"
else
    echo "ATTENZIONE: $HYDRA_FILE non trovato!"
    exit 1
fi

# ============================================================
# Planning
# ============================================================
python plan.py \
  --config-name "$CONFIG_NAME" \
  ckpt_base_path="$BASE_PATH" \
  model_name="$MODEL_NAME" \
  model_epoch="$MODEL_EPOCH" \
  planner.sub_planner.num_samples="$NUM_SAMPLES" \
  planner.sub_planner.opt_steps="$OPT_STEPS" \
  n_evals="$N_EVALS" \
  run_tag="$RUN_TAG" \
  seed="$SEED" \
  planner.max_iter=10

#Seed usati finora: 99 (originale), 123, 42, 456, 789
