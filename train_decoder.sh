#!/bin/bash
#SBATCH --job-name=dinowm_decoder_pushcube_dino_1000
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/cvcs2026/LubiMoRe/logs/decoder_%x_%j.out
#SBATCH --error=/work/cvcs2026/LubiMoRe/logs/decoder_%x_%j.err
#SBATCH --account=cvcs2026

set -eo pipefail
umask 002

# ============================================================
# Parametri da cambiare per ogni combinazione encoder/task/rollout
# ATTENZIONE: env, encoder, frameskip, num_hist, N_ROLLOUT devono
# combaciare ESATTAMENTE con quelli del training originale (predictor)
# di quel checkpoint - controlla outputs/$SRC_NOME_FILE/hydra.yaml
# se hai il minimo dubbio, prima di lanciare.
# ============================================================
ENV="pushcube"                # pushcube | pusht | wall
ENCODER="dino"                 # dino | dino_base | clip | siglip | mae
N_ROLLOUT=1000                 # deve combaciare col checkpoint sorgente
DATASET_SUBDIR="pushcube_noise"
FRAMESKIP=5                    # deve combaciare col checkpoint sorgente
NUM_HIST=3                     # deve combaciare col checkpoint sorgente
DECODER_EPOCHS=20              # epoche di training SOLO del decoder
DECODER_BATCH_SIZE=8           # piu' basso di quello usato per il predictor (32):
                                # il decoder VQVAE fa upsampling fino a immagini
                                # intere 224x224, molto piu' pesante in memoria
                                # GPU del solo forward del predictor - con encoder
                                # grandi (es. dino_base) 32 va facilmente in OOM

SRC_NOME_FILE="dinowm_${ENV}_${ENCODER}_run${N_ROLLOUT}"
NOME_FILE="${SRC_NOME_FILE}_decoder"

BASE_DIR="/work/cvcs2026/LubiMoRe/dino_wm"
SRC_CKPT="$BASE_DIR/outputs/$SRC_NOME_FILE/checkpoints/model_latest.pth"
NEW_CKPT_DIR="$BASE_DIR/outputs/$NOME_FILE/checkpoints"

module purge
source /work/cvcs2026/LubiMoRe/dino_environment/bin/activate

cd "$BASE_DIR"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_API_KEY=$(cat ~/.wandb_key)
export WANDB_PROJECT="dinowm"
export WANDB_ENTITY="lubimore-team"
export WANDB_RUN_ID=$NOME_FILE
export WANDB_RESUME="allow"
export WANDB_MODE=offline

# ============================================================
# Copia il checkpoint del predictor gia' allenato in una cartella
# NUOVA e separata (mai in-place: save_ckpt() sovrascriverebbe
# checkpoints/model_latest.pth dell'originale, e la config di questo
# run - train_predictor=False - non lo include piu' tra le chiavi
# salvate, quindi lo perderesti dal file)
# ============================================================
if [ ! -f "$SRC_CKPT" ]; then
    echo "ERRORE: checkpoint sorgente non trovato: $SRC_CKPT"
    exit 1
fi
mkdir -p "$NEW_CKPT_DIR"
cp "$SRC_CKPT" "$NEW_CKPT_DIR/model_latest.pth"
echo "Checkpoint copiato da $SRC_CKPT a $NEW_CKPT_DIR/model_latest.pth"

echo "Spazio disponibile su $TMPDIR prima della copia:"
df -h "$TMPDIR"

echo "Copia del dataset ($DATASET_SUBDIR) nello spazio temporaneo del nodo ($TMPDIR)..."
rsync -ah --info=progress2 \
  --exclude='dino_feats' \
  "$BASE_DIR/data/$DATASET_SUBDIR" "$TMPDIR/dino_dataset/"
echo "Copia completata."

export DATASET_DIR="$TMPDIR/dino_dataset"

chmod -R u+rwX "$BASE_DIR/outputs/$NOME_FILE/" 2>/dev/null || true

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

# ============================================================
# has_decoder/train_decoder=True, train_predictor=False (e
# train_encoder resta False di default): il predictor gia' allenato
# viene caricato dal checkpoint copiato sopra e resta congelato,
# si allena solo il decoder appena istanziato
# ============================================================
TRAIN_ARGS=(
  --config-name train.yaml
  env=$ENV
  encoder=$ENCODER
  frameskip=$FRAMESKIP
  num_hist=$NUM_HIST
  env.dataset.n_rollout=$N_ROLLOUT
  env.num_workers=32
  hydra.run.dir=outputs/$NOME_FILE
  +dataset.data_path="$TMPDIR/dino_dataset/$DATASET_SUBDIR"
  training.batch_size=$DECODER_BATCH_SIZE
  training.epochs=$DECODER_EPOCHS
  has_decoder=True
  model.train_decoder=True
  model.train_predictor=False
)

if [ "$NUM_GPUS" -le 1 ]; then
    echo "Lancio in modalità singola GPU (python)"
    python train.py "${TRAIN_ARGS[@]}"
else
    echo "Lancio in modalità multi-GPU con torchrun ($NUM_GPUS processi)"
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    torchrun --nproc_per_node="$NUM_GPUS" train.py "${TRAIN_ARGS[@]}"
fi

echo "Training decoder completato. Per usarlo in planning/inferenza, unisci i pesi:"
echo "  python -c \"import torch; old=torch.load('$SRC_CKPT'); new=torch.load('$NEW_CKPT_DIR/model_latest.pth'); old['decoder']=new['decoder']; torch.save(old, '$NEW_CKPT_DIR/model_merged.pth')\""

setfacl -R -b -k "$BASE_DIR/outputs/$NOME_FILE" 2>/dev/null || true
chmod -R g+rwX,o+rX "$BASE_DIR/outputs/$NOME_FILE" 2>/dev/null || true
