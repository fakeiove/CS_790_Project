#!/bin/bash -l

# ============================================================
# Job: Train classifiers for all 3 joints (DIP, PIP, MCP)
# Compare: baseline vs traditional aug vs LDM-generated vs combined
# Submit: qsub jobs/classify_all.sh
# ============================================================
#$ -l gpus=1
#$ -l gpu_type=A40|L40|A100
#$ -l h_rt=6:00:00
#$ -l mem_total=64G
#$ -pe omp 4
#$ -N classify
#$ -j y
#$ -o logs/classify_all.log

module load python3/3.10.12
module load cuda/12.1

cd /projectnb/cs790/students/panke66/ldm_project

echo "Job started: $(date)"

# ===== DIP =====
echo ""
echo "============================================"
echo "  DIP Classification"
echo "============================================"
python train_classifier.py \
    --joint_types DIP \
    --gen_dir generated_v2/ \
    --guided_ns 0.5 \
    --experiment all \
    --epochs 30 \
    --batch_size 64 \
    --output_dir results/

# ===== PIP =====
echo ""
echo "============================================"
echo "  PIP Classification"
echo "============================================"
python train_classifier.py \
    --joint_types PIP \
    --gen_dir generated_pip/ \
    --guided_ns 0.5 \
    --experiment all \
    --epochs 30 \
    --batch_size 64 \
    --output_dir results/

# ===== MCP =====
echo ""
echo "============================================"
echo "  MCP Classification"
echo "============================================"
python train_classifier.py \
    --joint_types MCP \
    --gen_dir generated_mcp/ \
    --guided_ns 0.5 \
    --experiment all \
    --epochs 30 \
    --batch_size 64 \
    --output_dir results/

echo ""
echo "Job finished: $(date)"
echo "Results saved to results/"
