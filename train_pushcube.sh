#!/bin/bash
#SBATCH --job-name=dinowm_train_dino_pushcube_10000
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/cvcs2026/LubiMoRe/logs/train_dino_pushcube_10000_%j.out
#SBATCH --error=/work/cvcs2026/LubiMoRe/logs/train_dino_pushcube_10000_%j.err
#SBATCH --account=cvcs2026

set -eo pipefail   # esce subito se un comando fallisce (rsync incluso), invece di proseguire alla cieca

# Permessi di gruppo sui file creati da questo job
umask 002

# ============================================================
# Parametri da cambiare per ogni combinazione encoder/task -> ricorda di cambiare le info anche sopra negli #SBATCH
# ============================================================
ENV="pushcube"
ENCODER="dino"               # dino | clip | siglip | mae
N_ROLLOUT=10000
DATASET_SUBDIR="pushcube_noise"
NOME_FILE="dinowm_${ENV}_${ENCODER}_run${N_ROLLOUT}"

module purge
source /work/cvcs2026/LubiMoRe/dino_environment/bin/activate

cd /work/cvcs2026/LubiMoRe/dino_wm
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# wandb setup
export WANDB_API_KEY=$(cat ~/.wandb_key)
export WANDB_PROJECT="dinowm"
export WANDB_ENTITY="lubimore-team"
export WANDB_RUN_ID=$NOME_FILE
export WANDB_RESUME="allow"
# offline: scrive comunque tutti i log in locale, sincronizzabili dopo con
# `wandb sync`, ma non dipende dalla rete durante il job - evita il timeout
# di wandb.init() se il nodo ha problemi di connettivita' verso i server wandb
export WANDB_MODE=offline

echo "Spazio disponibile su $TMPDIR prima della copia:"
df -h "$TMPDIR"

echo "Copia del dataset ($DATASET_SUBDIR) nello spazio temporaneo del nodo ($TMPDIR)..."
rsync -ah --info=progress2 \
  --exclude='dino_feats' \
  /work/cvcs2026/LubiMoRe/dino_wm/data/$DATASET_SUBDIR "$TMPDIR/dino_dataset/"
echo "Copia completata."

echo "Spazio disponibile su $TMPDIR dopo la copia:"
df -h "$TMPDIR"

export DATASET_DIR="$TMPDIR/dino_dataset"

chmod -R u+rwX /work/cvcs2026/LubiMoRe/dino_wm/outputs/$NOME_FILE/ 2>/dev/null || true

# ============================================================
# Rileva quante GPU SLURM ha assegnato a questo job
# ============================================================
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif [ -n "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | grep -c .)
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS=1
fi

echo "GPU rilevate: $NUM_GPUS"

# Argomenti comuni a entrambe le modalità
TRAIN_ARGS=(
  --config-name train.yaml
  env=$ENV
  encoder=$ENCODER
  frameskip=5
  num_hist=3
  env.dataset.n_rollout=$N_ROLLOUT
  env.num_workers=32
  hydra.run.dir=outputs/$NOME_FILE
  +dataset.data_path="$TMPDIR/dino_dataset/$DATASET_SUBDIR"
  training.batch_size=32
  training.epochs=20
)

# ============================================================
# 1 GPU  -> python diretto (niente DDP)
# >1 GPU -> torchrun con un processo per GPU
# ============================================================
if [ "$NUM_GPUS" -le 1 ]; then
    echo "Lancio in modalità singola GPU (python)"
    python train.py "${TRAIN_ARGS[@]}"
else
    echo "Lancio in modalità multi-GPU con torchrun ($NUM_GPUS processi)"
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    torchrun --nproc_per_node="$NUM_GPUS" train.py "${TRAIN_ARGS[@]}"
fi
