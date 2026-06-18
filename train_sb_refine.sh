LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_i2i_small_x1_pred$(date +%Y%m%d_%H%M%S).log"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6 uv run torchrun --nproc_per_node=1 --master_port 7811 \
main_flowar.py \
--img_size 256 --vae_path /localscratch/yzheng494/FlowAR/mar/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model flowar_sb_i2i_small --diffloss_d 12 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 32 --blr 5e-5 \
--output_dir ./output_dir/ --resume ./output_dir/ --use_checkpoint \
--data_path /localscratch/yzheng494/FlowAR --cached_path /localscratch/yzheng494/FlowAR/cache_dir --use_cached \
--use_sb --sb_mode i2i_refine --sb_prediction x0 --sb_beta_max 1.0 \
--online_eval --eval_freq 1 --num_images 1000 --eval_bsz 16 --val_eval_freq 1 \
--wandb \
2>&1 | tee "$LOG_FILE"