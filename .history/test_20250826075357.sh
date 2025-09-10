# CUDA_VISIBLE_DEVICES=2 nohup python eval_vllm.py --tensor_parallel_size 1 --model_path ./Llama-3.1-8B  --dataset_name nyc --output_dir ./output/checkpoint-500 --test_file "datasets/NYC/test_codebook.json" > output_reason/test-722-500-se5+1.txt 2>&1 &  


CUDA_VISIBLE_DEVICES=0 nohup python eval_vllm.py --model_path ./Llama-3.1-8B  --dataset_name nyc --output_dir ./restart/checkpoint-7000 --test_file "datasets/NYC/data/test_codebook_origin.json" > restart/test-7000-1.txt 2>&1 &  

#2
# generation_config = transformers.GenerationConfig(
#         max_new_tokens=50,  # 增加生成的token数量，确保能完整生成POI序列
#         min_new_tokens=None,
#         # Generation strategy
#         do_sample=True,
#         num_beams=5,  # 使用beam search提高生成质量
#         # num_beam_groups=5,
#         # penalty_alpha=None,
#         use_cache=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,

#         # Hyperparameters for logit manipulation
#         temperature=0.3,  # 降低温度，使输出更确定性
#         top_k=10,         # 减小top_k值，使输出更确定性
#         top_p=0.9,        # 增大top_p值，保留更多可能性
#         typical_p=1.0,
#         # diversity_penalty=4.0,
#         repetition_penalty=1.2,  # 略微增加重复惩罚
#         # length_penalty=1.0,
#         # no_repeat_ngram_size=0,

#         num_return_sequences=1
#     )




#1
# generation_config = transformers.GenerationConfig(
#         max_new_tokens=50,
#         min_new_tokens=None,
#         # Generation strategy
#         do_sample=True,
#         # num_beams=5,
#         # num_beam_groups=5,
#         # penalty_alpha=None,
#         use_cache=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,

#         # Hyperparameters for logit manipulation
#         temperature=0.6,
#         top_k=40,
#         top_p=0.1,
#         typical_p=1.0,
#         # diversity_penalty=4.0,
#         repetition_penalty=1.176,
#         # length_penalty=1.0,
#         # no_repeat_ngram_size=0,

#         num_return_sequences=1
#     )




# generation_config = transformers.GenerationConfig(
#             max_new_tokens=50,  # 增加生成的token数量，确保能完整生成POI序列
#             min_new_tokens=None,
#             # Generation strategy
#             do_sample=True,
#             #num_beams=5,  # 使用beam search提高生成质量
#             # num_beam_groups=5,
#             # penalty_alpha=None,
#             use_cache=True,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,

#             # Hyperparameters for logit manipulation
#             temperature=0.6,
#             top_k=40,
#             top_p=0.1,
#             typical_p=1.0,
#             # diversity_penalty=4.0,
#             repetition_penalty=1.176,
#             # length_penalty=1.0,
#             # no_repeat_ngram_size=0,

#             num_return_sequences=1
#         )
#ACC@1:0.2636986301369863



# generation_config = transformers.GenerationConfig(
#         max_new_tokens=30,  # 增加生成的token数量，确保能完整生成POI序列
#         min_new_tokens=None,
#         # Generation strategy
#         do_sample=True,
#         #num_beams=5,  # 使用beam search提高生成质量
#         # num_beam_groups=5,
#         # penalty_alpha=None,
#         use_cache=True,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,

#         # Hyperparameters for logit manipulation
#         temperature=0.3,
#         top_k=40,
#         top_p=0.1,
#         typical_p=1.0,
#         # diversity_penalty=4.0,
#         repetition_penalty=1.176,
#         # length_penalty=1.0,
#         # no_repeat_ngram_size=0,

#         num_return_sequences=1
#     )
#ACC@1:0.2636986301369863



# pc, opentelemetry-exporter-otlp, vllm
#   Attempting uninstall: triton
#     Found existing installation: triton 3.0.0
#     Uninstalling triton-3.0.0:
#       Successfully uninstalled triton-3.0.0
#   Attempting uninstall: sympy
#     Found existing installation: sympy 1.13.1
#     Uninstalling sympy-1.13.1:
#       Successfully uninstalled sympy-1.13.1
#   Attempting uninstall: nvidia-nvtx-cu12
#     Found existing installation: nvidia-nvtx-cu12 12.4.99
#     Uninstalling nvidia-nvtx-cu12-12.4.99:
#       Successfully uninstalled nvidia-nvtx-cu12-12.4.99
#   Attempting uninstall: nvidia-nvjitlink-cu12
#     Found existing installation: nvidia-nvjitlink-cu12 12.4.99
#     Uninstalling nvidia-nvjitlink-cu12-12.4.99:
#       Successfully uninstalled nvidia-nvjitlink-cu12-12.4.99
#   Attempting uninstall: nvidia-nccl-cu12
#     Found existing installation: nvidia-nccl-cu12 2.20.5
#     Uninstalling nvidia-nccl-cu12-2.20.5:
#       Successfully uninstalled nvidia-nccl-cu12-2.20.5
#   Attempting uninstall: nvidia-curand-cu12
#     Found existing installation: nvidia-curand-cu12 10.3.5.119
#     Uninstalling nvidia-curand-cu12-10.3.5.119:
#       Successfully uninstalled nvidia-curand-cu12-10.3.5.119
#   Attempting uninstall: nvidia-cuda-runtime-cu12
#     Found existing installation: nvidia-cuda-runtime-cu12 12.4.99
#     Uninstalling nvidia-cuda-runtime-cu12-12.4.99:
#       Successfully uninstalled nvidia-cuda-runtime-cu12-12.4.99
#   Attempting uninstall: nvidia-cuda-nvrtc-cu12
#     Found existing installation: nvidia-cuda-nvrtc-cu12 12.4.99
#     Uninstalling nvidia-cuda-nvrtc-cu12-12.4.99:
#       Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.4.99
#   Attempting uninstall: nvidia-cuda-cupti-cu12
#     Found existing installation: nvidia-cuda-cupti-cu12 12.4.99
#     Uninstalling nvidia-cuda-cupti-cu12-12.4.99:
#       Successfully uninstalled nvidia-cuda-cupti-cu12-12.4.99
#   Attempting uninstall: nvidia-cublas-cu12
#     Found existing installation: nvidia-cublas-cu12 12.4.2.65
#     Uninstalling nvidia-cublas-cu12-12.4.2.65:
#       Successfully uninstalled nvidia-cublas-cu12-12.4.2.65
#   Attempting uninstall: jinja2
#     Found existing installation: Jinja2 3.1.3
#     Uninstalling Jinja2-3.1.3:
#       Successfully uninstalled Jinja2-3.1.3
#   Attempting uninstall: filelock
#     Found existing installation: filelock 3.13.1
#     Uninstalling filelock-3.13.1:
#       Successfully uninstalled filelock-3.13.1
#   Attempting uninstall: nvidia-cusparse-cu12
#     Found existing installation: nvidia-cusparse-cu12 12.3.0.142
#     Uninstalling nvidia-cusparse-cu12-12.3.0.142:
#       Successfully uninstalled nvidia-cusparse-cu12-12.3.0.142
#   Attempting uninstall: nvidia-cufft-cu12
#     Found existing installation: nvidia-cufft-cu12 11.2.0.44
#     Uninstalling nvidia-cufft-cu12-11.2.0.44:
#       Successfully uninstalled nvidia-cufft-cu12-11.2.0.44
#   Attempting uninstall: nvidia-cudnn-cu12
#     Found existing installation: nvidia-cudnn-cu12 9.1.0.70
#     Uninstalling nvidia-cudnn-cu12-9.1.0.70:
#       Successfully uninstalled nvidia-cudnn-cu12-9.1.0.70
#   Attempting uninstall: huggingface-hub
#     Found existing installation: huggingface-hub 0.25.1
#     Uninstalling huggingface-hub-0.25.1:
#       Successfully uninstalled huggingface-hub-0.25.1
#   Attempting uninstall: tokenizers
#     Found existing installation: tokenizers 0.19.1
#     Uninstalling tokenizers-0.19.1:
#       Successfully uninstalled tokenizers-0.19.1
#   Attempting uninstall: nvidia-cusolver-cu12
#     Found existing installation: nvidia-cusolver-cu12 11.6.0.99
#     Uninstalling nvidia-cusolver-cu12-11.6.0.99:
#       Successfully uninstalled nvidia-cusolver-cu12-11.6.0.99
#   Attempting uninstall: transformers
#     Found existing installation: transformers 4.44.0
#     Uninstalling transformers-4.44.0:
#       Successfully uninstalled transformers-4.44.0
#   Attempting uninstall: torch
#     Found existing installation: torch 2.4.0+cu124
#     Uninstalling torch-2.4.0+cu124:
#       Successfully uninstalled torch-2.4.0+cu124
#   Attempting uninstall: openai
#     Found existing installation: openai 0.28.1
#     Uninstalling openai-0.28.1:
#       Successfully uninstalled openai-0.28.1
#   Attempting uninstall: torchvision
#     Found existing installation: torchvision 0.19.0+cu124
#     Uninstalling torchvision-0.19.0+cu124:
#       Successfully uninstalled torchvision-0.19.0+cu124
#   Attempting uninstall: torchaudio
#     Found existing installation: torchaudio 2.4.0+cu124
#     Uninstalling torchaudio-2.4.0+cu124:
#       Successfully uninstalled torchaudio-2.4.0+cu124