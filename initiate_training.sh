python3 scripts/train.py \
--dataset_type=ffhq_aging \
--exp_dir='/home/adity/SAM_DEMO_RESULTS_5' \
--workers=2 \
--batch_size=1 \
--test_batch_size=1 \
--test_workers=2 \
--val_interval=800 \
--save_interval=800 \
--image_interval=800 \
--start_from_encoded_w_plus \
--id_lambda=0.1 \
--lpips_lambda=0.1 \
--lpips_lambda_aging=0.1 \
--lpips_lambda_crop=0.6 \
--l2_lambda=0.25 \
--l2_lambda_aging=0.25 \
--l2_lambda_crop=1 \
--w_norm_lambda=0 \
--aging_lambda=5 \
--cycle_lambda=1 \
--input_nc=4 \
--target_age=20 \
--use_weighted_id_loss