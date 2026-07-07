#!/bin/bash
#SBATCH --job-name=style_full
#SBATCH --output=style_full_%j.out
#SBATCH --error=style_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH -A 


# set -euo pipefail
mkdir -p logs

# Adjust these paths
# PROJECT_DIR=/N/project/prostate_cancer_ai/anshika/dess_pd_stylegan3d
# DESS_DIR=$PROJECT_DIR/data/dess_nifti
# PD_DIR=$PROJECT_DIR/data/pd_nifti
# OUT_DIR=$PROJECT_DIR/runs/sanity_3d_stylegan

# cd $PROJECT_DIR
source venv/bin/activate

python -u dess_pd_stylegan3d.py \
  --mode infer \
  --dess_dir data/skm-tea-dataset/dess-files \
  --pd_dir data/iu-dataset/pd-files\
  --out_dir runs_2/stylegan \
  --checkpoint runs_2/stylegan/checkpoints/final.pth \
  --patch_size 64,64,16

# python -u dess_pd_stylegan3d.py \
#   --mode train \
#   --dess_dir data/skm-tea-dataset/dess-files \
#   --pd_dir data/iu-dataset/pd-files\
#   --out_dir runs/stylegan \
#   --patch_size 64,64,16 \
#   --batch_size 1 \
#   --max_iterations 30000 \
#   --num_workers 4 \
#   --preview_every 200 \
#   --save_every 1000
