#!/bin/bash -l

# ============================================================
# Job 1: Train VAE (Fixed version)
# Submit: qsub jobs/train_vae.sh
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=8:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N vae_train_v3
#$ -j y
#$ -o logs/vae_train_v3.log

module load python3/3.10.12
module load cuda/12.1

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python train_vae.py \
    --data_dir data/ \
    --joint_types DIP \
    --img_size 128 \
    --batch_size 64 \
    --epochs 150 \
    --lr 1e-4 \
    --kl_weight 1e-4 \
    --kl_warmup_epochs 20 \
    --perc_weight 0.1 \
    --use_perceptual \
    --latent_channels 4 \
    --base_ch 64 \
    --num_workers 4

echo "Job finished: $(date)"
