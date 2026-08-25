#!/bin/bash
#SBATCH --job-name=pushcube_gen_10000
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/cvcs2026/LubiMoRe/logs/pushcube_gen_%j.out
#SBATCH --error=/work/cvcs2026/LubiMoRe/logs/pushcube_gen_%j.err
#SBATCH --account=cvcs2026

set -eo pipefail
umask 002

module purge
source /work/cvcs2026/LubiMoRe/maniskill_env/bin/activate

cd /work/cvcs2026/LubiMoRe/dino_wm

python scripts/generate_pushcube_dataset.py \
    --demo-path ~/.maniskill/demos/PushCube-v1/motionplanning/trajectory.none.pd_ee_delta_pos.physx_cpu.h5 \
    --out-dir /work/cvcs2026/LubiMoRe/dino_wm/data/pushcube_noise \
    --num-train-rollouts 10000 \
    --num-val-rollouts 50 \
    --max-steps 150 \
    --action-noise-std 0.05
