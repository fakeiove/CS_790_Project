#!/bin/bash -l

# ============================================================
# Job: Generate PIP joint images
# Submit: qsub jobs/generate_pip.sh
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=4:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N gen_pip
#$ -j y
#$ -o logs/gen_pip.log

module load python3/3.10.12
module load cuda/12.1

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"

# Sweeps
python generate_v2.py --mode sweep --target_kl 3 --cfg_scale 3.0 --num_steps 100 \
    --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/

python generate_v2.py --mode sweep --target_kl 4 --cfg_scale 3.0 --num_steps 100 \
    --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/

# Unconditional
python generate_v2.py --mode unconditional --target_kl 3 --num_samples 1000 --cfg_scale 3.0 --num_steps 100 \
    --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/

python generate_v2.py --mode unconditional --target_kl 4 --num_samples 500 --cfg_scale 3.0 --num_steps 100 \
    --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/

# Guided
for NS in 0.3 0.4 0.5; do
    python generate_v2.py --mode guided --target_kl 3 --noise_strength $NS --cfg_scale 3.0 --num_steps 100 \
        --num_samples 500 --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/

    python generate_v2.py --mode guided --target_kl 4 --noise_strength $NS --cfg_scale 4.0 --num_steps 100 \
        --num_samples 300 --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/
done

echo "Job finished: $(date)"
