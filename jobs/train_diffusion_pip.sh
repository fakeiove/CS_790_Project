#!/bin/bash -l

# ============================================================
# Job: Train Diffusion Model for PIP joints
# Submit: qsub jobs/train_diffusion_pip.sh
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=16:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N diff_pip
#$ -j y
#$ -o logs/diff_pip_train.log

module load python3/3.10.12
module load cuda/12.1

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python train_diffusion_v2.py \
    --data_dir data/ \
    --vae_ckpt checkpoints/vae_best.pt \
    --joint_types PIP \
    --img_size 128 \
    --batch_size 64 \
    --epochs 500 \
    --lr 2e-4 \
    --warmup_epochs 10 \
    --base_ch 128 \
    --num_timesteps 1000 \
    --schedule cosine \
    --cfg_dropout 0.15 \
    --cfg_scale 3.0 \
    --ema_decay 0.995 \
    --dropout 0.1 \
    --use_min_snr \
    --snr_gamma 5.0 \
    --num_workers 4 \
    --save_dir checkpoints_pip \
    --log_dir logs/diffusion_pip

echo "Job finished: $(date)"
