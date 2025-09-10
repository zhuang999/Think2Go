#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=10800
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

MAIN_PROCESS_PORT=20138


mkdir -p output_reason

CUDA_VISIBLE_DEVICES=0,1,2,3,4 nohup nice -n -10 \
torchrun --nproc_per_node=4 reasoning.py  \
--model_name_or_path ./Llama-3.1-8B \
--peft_model_path None \
--bf16 True \
--output_dir ./restart \
--model_max_length 2048 \
--use_flash_attn True \
--data_path data/NYC/data/train_codebook_with_difficulty.json \
--low_rank_training True \
--num_train_epochs 10  \
--per_device_train_batch_size 1    \
--per_device_eval_batch_size 2     \
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
--wandb_name "restart" \
> restart/output.log 2>&1 & 