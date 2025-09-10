# 确保输出目录存在
mkdir -p output_reason

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 train.py  \
--model_name_or_path ./Llama-3.1-8B \
--bf16 True \
--output_dir ./kdd \
--model_max_length 2048 \
--use_flash_attn True \
--data_path data/NYC/data/train_codebook.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 2    \
--per_device_eval_batch_size 4     \
--gradient_accumulation_steps 1     \
--eval_strategy "no"     \
--save_strategy "steps"     \
--save_steps 1000     \
--save_total_limit 5     \
--learning_rate 2e-5     \
--weight_decay 0.0     \
--warmup_steps 20     \
--lr_scheduler_type "constant_with_warmup"     \
--logging_steps 1     \
--deepspeed "ds_configs/stage2.json" \
--tf32 True  \
> kdd/GNPR-our-729.log 2>&1 & 