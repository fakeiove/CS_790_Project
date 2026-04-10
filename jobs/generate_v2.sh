#!/bin/bash -l

# ============================================================
# Job 3: Generate with v3 model - sweep + final generation
# Submit: qsub jobs/generate_v2.sh
# Run AFTER diffusion v3 training completes
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=4:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N gen_v3
#$ -j y
#$ -o logs/gen_v3.log

module load python3/3.10.12
module load cuda/12.1

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"

echo "=== Step 1: Parameter sweeps ==="
python generate_v2.py \
    --mode sweep \
    --target_kl 3 \
    --cfg_scale 3.0 \
    --num_steps 100 \
    --output_dir generated_v2/

python generate_v2.py \
    --mode sweep \
    --target_kl 4 \
    --cfg_scale 3.0 \
    --num_steps 100 \
    --output_dir generated_v2/

echo "=== Step 2: Unconditional generation ==="
python generate_v2.py \
    --mode unconditional \
    --target_kl 3 \
    --num_samples 1000 \
    --cfg_scale 3.0 \
    --num_steps 100 \
    --output_dir generated_v2/

python generate_v2.py \
    --mode unconditional \
    --target_kl 4 \
    --num_samples 500 \
    --cfg_scale 3.0 \
    --num_steps 100 \
    --output_dir generated_v2/

echo "=== Step 3: Guided generation (multiple noise strengths) ==="
for NS in 0.3 0.4 0.5 0.6 0.7; do
    echo "--- Guided KL3, noise_strength=$NS ---"
    python generate_v2.py \
        --mode guided \
        --target_kl 3 \
        --noise_strength $NS \
        --cfg_scale 3.0 \
        --num_steps 100 \
        --num_samples 800 \
        --output_dir generated_v2/

    echo "--- Guided KL4, noise_strength=$NS ---"
    python generate_v2.py \
        --mode guided \
        --target_kl 4 \
        --noise_strength $NS \
        --cfg_scale 4.0 \
        --num_steps 100 \
        --num_samples 800 \
        --output_dir generated_v2/
done

echo "Job finished: $(date)"
