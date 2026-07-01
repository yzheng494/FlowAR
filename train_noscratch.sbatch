#!/bin/bash
#SBATCH -J flowar_sb_i2i_small
#SBATCH --account=gts-bdai6
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=A100:1
#SBATCH -t 08:00:00
#SBATCH -q embers
#SBATCH -o /storage/home/hcoda1/7/yzheng494/logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yzheng494@gatech.edu

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

WORKDIR=/storage/scratch1/7/yzheng494/FlowAR
CACHE_DIR=/storage/project/r-bdai6-0/yzheng494/FlowAR/cache_dir
LOG_DIR=/storage/home/hcoda1/7/yzheng494/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_i2i_small_x1_pred_${SLURM_JOB_ID}.log"

cd "$WORKDIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun \
  --nproc_per_node=1 \
  --master_port 7811 \
  main_flowar.py \
  --img_size 256 \
  --vae_path "$WORKDIR/mar/vae/kl16.ckpt" \
  --vae_embed_dim 16 \
  --vae_stride 16 \
  --patch_size 1 \
  --model flowar_sb_i2i_small \
  --diffloss_d 12 \
  --diffloss_w 1024 \
  --epochs 400 \
  --warmup_epochs 100 \
  --batch_size 32 \
  --blr 5e-5 \
  --output_dir ./output_dir/ \
  --resume ./output_dir/ \
  --use_checkpoint \
  --data_path "$WORKDIR" \
  --cached_path "$CACHE_DIR" \
  --use_cached \
  --use_sb \
  --sb_mode i2i \
  --sb_prediction x0 \
  --sb_beta_max 1.0 \
  --online_eval \
  --eval_freq 1 \
  --num_images 200 \
  --eval_bsz 16 \
  --val_eval_freq 1 \
  --num_workers 6 \
  --no_pin_mem \
  --wandb \
  2>&1 | tee "$LOG_FILE"

echo "Training complete: $(date)"
