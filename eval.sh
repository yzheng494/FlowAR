uv run torchrun --nproc_per_node=4 --master_port=4152 \
eval.py \
--model flowar_sb_i2i_small --diffloss_d 12 --diffloss_w 1024 \
--img_size 256 --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--eval_bsz 128 --num_images 50000 --num_step 25 --cfg 4.2 --guidance 0.9 \
--output_dir ./evaluation_dir_freeze_ar \
--resume ./output_dir_freeze_ar/checkpoint-last.pth --vae_path ./mar/vae/kl16.ckpt \
--data_path ./val_dir --val_dir ./val_dir --evaluate
