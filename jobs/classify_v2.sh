#!/bin/bash -l

# ============================================================
# Job: Optimized classifier - guided only + capped ratio
# Submit: qsub jobs/classify_v2.sh
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=6:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N classify_v2
#$ -j y
#$ -o logs/classify_v2.log

module load python3/3.10.12

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"
echo "Optimization: guided_only + max_gen_ratio=2.0"

# ===== DIP =====
echo ""
echo "============================================"
echo "  DIP Classification (guided only)"
echo "============================================"
python train_classifier.py \
    --joint_types DIP \
    --gen_dir generated_v2/ \
    --guided_ns 0.5 \
    --guided_only \
    --max_gen_ratio 2.0 \
    --experiment all \
    --epochs 30 \
    --batch_size 64 \
    --output_dir results_v2/

# ===== PIP =====
echo ""
echo "============================================"
echo "  PIP Classification (guided only)"
echo "============================================"
python train_classifier.py \
    --joint_types PIP \
    --gen_dir generated_pip/ \
    --guided_ns 0.5 \
    --guided_only \
    --max_gen_ratio 2.0 \
    --experiment all \
    --epochs 30 \
    --batch_size 64 \
    --output_dir results_v2/

# ===== MCP =====
echo ""
echo "============================================"
echo "  MCP Classification (guided only)"
echo "============================================"
python train_classifier.py \
    --joint_types MCP \
    --gen_dir generated_mcp/ \
    --guided_ns 0.5 \
    --guided_only \
    --max_gen_ratio 2.0 \
    --experiment all \
    --epochs 30 \
    --batch_size 64 \
    --output_dir results_v2/

echo ""
echo "Job finished: $(date)"
