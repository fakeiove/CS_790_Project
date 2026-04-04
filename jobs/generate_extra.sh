#!/bin/bash -l

# ============================================================
# Job: Generate extra ns0.6/0.7 for PIP and MCP
# Submit: qsub jobs/generate_extra.sh
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=4:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N gen_extra
#$ -j y
#$ -o logs/gen_extra.log

module load python3/3.10.12
module load cuda/12.1

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"

# DIP ns0.6 and ns0.7
for NS in 0.6 0.7; do
    python generate_v2.py --mode guided --target_kl 3 --noise_strength $NS --cfg_scale 3.0 --num_steps 100 \
        --num_samples 500 --joint_types DIP --diff_ckpt checkpoints_v2/diffusion_v2_best.pt --output_dir generated_v2/

    python generate_v2.py --mode guided --target_kl 4 --noise_strength $NS --cfg_scale 4.0 --num_steps 100 \
        --num_samples 300 --joint_types DIP --diff_ckpt checkpoints_v2/diffusion_v2_best.pt --output_dir generated_v2/
done

# PIP ns0.6 and ns0.7
for NS in 0.6 0.7; do
    python generate_v2.py --mode guided --target_kl 3 --noise_strength $NS --cfg_scale 3.0 --num_steps 100 \
        --num_samples 500 --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/

    python generate_v2.py --mode guided --target_kl 4 --noise_strength $NS --cfg_scale 4.0 --num_steps 100 \
        --num_samples 300 --joint_types PIP --diff_ckpt checkpoints_pip/diffusion_v2_best.pt --output_dir generated_pip/
done

# MCP ns0.6 and ns0.7
for NS in 0.6 0.7; do
    python generate_v2.py --mode guided --target_kl 3 --noise_strength $NS --cfg_scale 3.0 --num_steps 100 \
        --num_samples 500 --joint_types MCP --diff_ckpt checkpoints_mcp/diffusion_v2_best.pt --output_dir generated_mcp/

    python generate_v2.py --mode guided --target_kl 4 --noise_strength $NS --cfg_scale 4.0 --num_steps 100 \
        --num_samples 300 --joint_types MCP --diff_ckpt checkpoints_mcp/diffusion_v2_best.pt --output_dir generated_mcp/
done

echo "Job finished: $(date)"
