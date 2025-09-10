import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
# from llama_attn_replace import replace_llama_attn
from llama_attn_replace_sft import replace_llama_attn
from typing import Dict, Optional, Sequence
import sys
from transformers import BitsAndBytesConfig
import json
import re
from vllm import LLM, SamplingParams, TokensPrompt
import rich
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax
from trl.trainer import disable_dropout_in_model
from accelerate.utils.other import is_compiled_module
from accelerate.utils import is_peft_model

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/Llama-2-7b-longlora-32k-ft")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=2048, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--model_path', type=str, default='', help='your model path')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--output_dir', type=str, default="./output", help='')
    parser.add_argument('--dataset_name', type=str, default="nyc",
                        help='')
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_100.txt",
                        help='')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallelism.')
    args = parser.parse_args()
    return args

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def format_chat_conversation(
    system_content: str,
    user_content: str,
    assistant_content: str,
    tokenizer: transformers.PreTrainedTokenizer
    ) -> str:
    """Format a conversation using the tokenizer's chat template."""
    conversation = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    if assistant_content.strip():
        conversation.append({"role": "assistant", "content": assistant_content})
        add_generation_prompt = False
    else:
        add_generation_prompt = True
    
    formatted_conv = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )
    
    return formatted_conv

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx + batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i + seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y



def merge_weights(state_dict):
    new_dict = {}
    layers = set()
    
    for key in state_dict:
        if 'model.layers.' in key:
            layer = key.split('.')[2]
            layers.add(int(layer))
    
    for layer in layers:
        # QKV merge
        q_key = f'model.layers.{layer}.self_attn.q_proj.weight'
        k_key = f'model.layers.{layer}.self_attn.k_proj.weight'
        v_key = f'model.layers.{layer}.self_attn.v_proj.weight'
        
        if all(k in state_dict for k in [q_key, k_key, v_key]):
            qkv = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
            new_dict[f'model.layers.{layer}.self_attn.qkv_proj.weight'] = qkv
        
        # Gate-Up merge
        gate_key = f'model.layers.{layer}.mlp.gate_proj.weight'
        up_key = f'model.layers.{layer}.mlp.up_proj.weight'
        
        if all(k in state_dict for k in [gate_key, up_key]):
            gate_up = torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0)
            new_dict[f'model.layers.{layer}.mlp.gate_up_proj.weight'] = gate_up
    
    skip_keys = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'gate_proj.weight', 'up_proj.weight']
    for key, value in state_dict.items():
        if not any(skip in key for skip in skip_keys):
            new_dict[key] = value
    
    return new_dict

def _move_model_to_vllm(unwrapped_model, vllm_model):
    """
    Move the model to vllm
    """  
    if is_compiled_module(unwrapped_model):
        unwrapped_model = unwrapped_model._orig_mod
    if is_peft_model(unwrapped_model):
        
        param_dict = dict(unwrapped_model.named_parameters())
        state_dict = {}
        merged_count = 0

        lora_config = unwrapped_model.peft_config['default']
        scaling = lora_config.lora_alpha / lora_config.r
        for name, param in unwrapped_model.named_parameters():
            if name.endswith("base_layer.weight"):
                prefix = name[:-len("base_layer.weight")]
                a_name = prefix + "lora_A.default.weight"
                b_name = prefix + "lora_B.default.weight"
                if a_name in param_dict and b_name in param_dict:
                    lora_a = param_dict[a_name]
                    lora_b = param_dict[b_name]
                    quant_state = getattr(param, "quant_state", None)

                    if quant_state is not None:
                        import bitsandbytes.functional as F
                        base_fp = F.dequantize_4bit(param, quant_state)

                        lora_delta = (lora_b @ lora_a) * scaling
                        merged_fp = base_fp + lora_delta.to(base_fp.device).to(base_fp.dtype)

                        new_name = prefix + "weight"
                        state_dict[new_name] = merged_fp
                        merged_count += 1
            else:
                if hasattr(param, "quant_state") and param.quant_state is not None:
                    import bitsandbytes.functional as F
                    state_dict[name] = F.dequantize_4bit(param, param.quant_state)
                else:
                    state_dict[name] = param.clone().detach()
        

        state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in ['.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map','.quant_state',])}
        
        state_dict = {
            k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
        }
        state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
        state_dict = {
            k.replace("modules_to_save.default.", ""): v
            for k, v in state_dict.items()
            if "original_module" not in k
        }
        state_dict = {k: v for k, v in state_dict.items()
                    if "extra_head" not in k and "item_embedding" not in k}
        
        lm_head_keys = [k for k in state_dict.keys() if k.startswith("lm_head")]

        if "lm_head.weight" in state_dict and "lm_head.0.weight" in state_dict:
            del state_dict["lm_head.0.weight"]
        if "lm_head.bias" in state_dict and "lm_head.0.bias" in state_dict:
            del state_dict["lm_head.0.bias"]
    
        state_dict = merge_weights(state_dict)
    else:
        state_dict = unwrapped_model.state_dict()

    llm_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
                enumerate(
                    get_as_batch(
                        data['val'],
                        seq_length,
                        batch_size,
                        device=device,
                        sliding_window=sliding_window
                    )
                ),
                total=iceildiv(
                    iceildiv(len(data['val']), sliding_window),
                    batch_size
                )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]

                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i + seq_length].contiguous(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i + seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt

            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats


def main(args):
    device = "cuda:0"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_path = args.model_path
    output_dir = args.output_dir
    print("data path", args.data_path)
    print("base model", model_path)
    print("peft model", output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        output_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=True,
    )

    def setup_chat_template(tokenizer, model_name_or_path):
        model_name = model_name_or_path.lower()
        tokenizer_class = str(tokenizer.__class__).lower()

        if "qwen" in tokenizer_class or "qwen" in model_name:
            tokenizer.chat_template = """{% for message in messages %}
            <|im_start|>{{ message['role'] }}
            {{ message['content'] }}<|im_end|>
            {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
            {% endif %}"""
            tokenizer.add_special_tokens({
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
            })

        elif "gemma" in tokenizer_class or "gemma" in model_name:
            tokenizer.chat_template = """{% for message in messages %}
            <start_of_turn>{{ message['role'] }}
            {{ message['content'] }}<end_of_turn>
            {% endfor %}{% if add_generation_prompt %}<start_of_turn>model
            {% endif %}"""
            tokenizer.add_special_tokens({
                "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]
            })

        elif "llama" in tokenizer_class or "llama" in model_name:
            tokenizer.chat_template = """{% for message in messages %}
            {% if message['role'] == 'system' %}
            <<SYS>>
            {{ message['content'] }}
            <</SYS>>
            {% elif message['role'] == 'user' %}
            [INST] {{ message['content'] }} [/INST]
            {% elif message['role'] == 'assistant' %}
            {{ message['content'] }}
            {% endif %}
            {% endfor %}{% if add_generation_prompt %}[INST]{% endif %}"""

        else:
            tokenizer.chat_template = """{% for message in messages %}
            ### {{ message['role'] | capitalize }}:
            {{ message['content'] }}
            {% endfor %}{% if add_generation_prompt %}### Assistant:\n{% endif %}"""

        return tokenizer
    
    tokenizer = setup_chat_template(tokenizer, model_path)
    if args.flash_attn:
        replace_llama_attn(inference=False)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        # _flash_attn_2_enabled = True,
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.resize_token_embeddings(128258) #133338   128258
    model.eval()
    if output_dir:
        trainable_params = os.path.join(output_dir, "pytorch_model.bin")
        model = PeftModel.from_pretrained(
            model,
            output_dir,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        print("Loaded trainable params2",trainable_params)


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("tokenizer:", tokenizer.encode("<a_18><b_8><c_17>",add_special_tokens=False))
    generation_config = transformers.GenerationConfig(
        max_new_tokens=1024,
        min_new_tokens=None,
        # Generation strategy
        do_sample=True,
        num_beams=5,
        # num_beam_groups=5,
        # penalty_alpha=None,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,

        # Hyperparameters for logit manipulation
        temperature=0.6,
        top_k=40,
        top_p=0.1,     #0.9=ACC@1:0.2625570776255708  0.3=0.3310502283105023 0.1=0.3344748858447489
        typical_p=1.0,  
        # diversity_penalty=4.0,
        repetition_penalty=1.176,
        # length_penalty=1.0,
        # no_repeat_ngram_size=0,

        num_return_sequences=1
    )


    vllm_model = LLM(
        model='./vllm_model',
        tokenizer='./vllm_tokenizer',
        # quantization="bitsandbytes", 
        # load_format="bitsandbytes",
        dtype='auto',   #torch.bfloat16,
        # trust_remote_code=True,
        gpu_memory_utilization=0.9,
        # dtype=self.args.vllm_dtype,
        # enable_prefix_caching=True,
        device='auto',
        max_model_len=4096,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    vllm_model_runner = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
    original_forward = vllm_model_runner.lm_head.forward

    def float32_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        return output.to(torch.float32)


    vllm_model_runner.lm_head.forward = float32_forward.__get__(vllm_model_runner.lm_head, vllm_model_runner.lm_head.__class__)

    # vllm_sampling_params = SamplingParams(
    #     temperature=0.7,
    #     max_tokens=1024,
    #     top_k=200,
    #     top_p=0.5,
    #     repetition_penalty=1.176,
    #     n=1,
    #     best_of=5,
    #     include_stop_str_in_output=True,
    #     stop=['<|end_of_text|>'],
    # )
    # vllm_sampling_params = SamplingParams(
    #     temperature=0.6,
    #     max_tokens=50,
    #     top_k=40,
    #     top_p=0.1,
    #     repetition_penalty=1.176,
    #     n=1,
    #     best_of=5,
    #     include_stop_str_in_output=True,
    #     stop=['<|end_of_text|>'],
    # )

    vllm_sampling_params = SamplingParams(
        temperature=1.3,
        max_tokens=1024,
        top_k=200,
        top_p=0.5,
        repetition_penalty=1.176,
        n=1,
        best_of=20,
        include_stop_str_in_output=True,
        stop=['<|end_of_text|>'],
    )
    
    vllm_sampling_params1 = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        top_k=200,
        top_p=0.5,
        repetition_penalty=1.176,
        n=1,
        seed=2,
        best_of=5,
        include_stop_str_in_output=True,
        stop=['<|end_of_text|>'],
    )
    _move_model_to_vllm(model, vllm_model)
    print("[vllm_sampling_params]", vllm_sampling_params)

    def evaluate_prediction_accuracy(prediction, ground_truth):

        answer_start = prediction.find('<answer>:')
        if answer_start != -1:
                prediction = prediction[answer_start + len('<answer>:'):].strip()
                
        pred_poi_parts = []
        pred_poi_matches = re.findall(r'<([a-z])_(\d+)>', prediction)
        for type_char, value in pred_poi_matches:
            pred_poi_parts.append(f"<{type_char}_{value}>")
        
        pred_poi_sequence = "".join(pred_poi_parts)
        
        gt_poi_parts = []
        gt_poi_matches = re.findall(r'<([a-z])_(\d+)>', ground_truth)
        for type_char, value in gt_poi_matches:
            gt_poi_parts.append(f"<{type_char}_{value}>")
        
        gt_poi_sequence = "".join(gt_poi_parts)
        

        
        is_correct = (pred_poi_sequence == gt_poi_sequence) and gt_poi_sequence != ""
        if is_correct:
            print(f"预测POI序列: '{prediction}'")
            print(f"真实POI序列: '{gt_poi_sequence}'")
        return int(is_correct)

    #data_path = f'./data/ca/preprocessed/{args.dataset_name}/'
    with open(f"{args.test_file}", "r") as file:
        # 读取JSON格式数据
        lines = json.load(file)
    
    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    device = 'cuda'
    # Iterate over each line and ask the LLM
    correct_list = []
    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        try:
            # 处理JSON格式数据
            # instruction = line.get("instruction", "")
            # source = f" <question> : {instruction}"

            system_content = "You are a helpful assistant specialized in location prediction. You analyze POI access patterns to predict next locations. Respond in the following format: <think>\n[Your step-by-step reasoning here]\n</think>\n<answer>: <a_7><b_25><c_16></answer><|end_of_text|>"

            instruction = f"Here is a record of a user's POI check-in trajectory. Your task is to analyze the user's POI check-in trajectory patterns and predict the next location they will visit at the specified time. You should perform a detailed analysis of temporal patterns, transitions between POIs, and user preferences. Present your reasoning step-by-step inside a <think> and </think> block. After completing your analysis, provide your final POI prediction within <answer>: and </answer> tags. For example: <answer>: <a_7><b_25><c_16></answer>{tokenizer.eos_token}."
            input_text = line.get("input", "")
            user_content = f"{instruction}\n\n{input_text}"
            ground_truth = line.get("output", "")
            
            # # 构建提示词，使用与训练时相同的格式 <question> : instruction
            # prompt = f" <question> : {instruction}"
            # if input_text.strip():
            #     prompt = f"{prompt}\n\n{input_text}"
            
            prompt = format_chat_conversation(
                system_content=system_content,
                user_content=user_content,
                assistant_content="",  # 空的assistant内容
                tokenizer=tokenizer
                )
            
            # # 添加提示词，引导模型直接生成POI序列
            # prompt = f"{prompt}\n\n<answer>:"
            
            # # 检查长度
            # if len(tokenizer.tokenize(prompt)) >= 2048:
            #     print(f"跳过索引 {index}：提示太长")
            #     continue
            
            # 编码
            prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_token_ids = prompt_tokens.input_ids.tolist()
            # 生成
            outputs = vllm_model.generate(prompt_token_ids=prompt_token_ids,
                                          sampling_params=vllm_sampling_params,
                                          use_tqdm=False)
            
            
            # 解码预测结果，保留特殊标记以便于匹配<answer>标记
            generated_token_ids = outputs[0].outputs[0].token_ids
            prediction = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            print("prediction",prediction)
            print("ground_truth",ground_truth)
            # print("ground_truth",ground_truth)
            
            # 清理ground_truth，确保格式一致
            gt = ground_truth.replace('[', '').replace(']', '')
            
            # print(f"\n===== 示例 {index} =====")
            # print(f"提示词: {prompt}")
            # print(f"原始预测: {prediction}")
            # print(f"真实标签: {gt}")
            
            # 评估预测结果
            is_correct = evaluate_prediction_accuracy(prediction, gt)
            
            # # 打印详细的token信息，帮助调试
            # if index < 3:  # 只打印前几个样本的详细信息
            #     print("\n详细的token信息:")
            #     input_tokens = tokenizer.tokenize(prompt)
            #     print(f"输入token数: {len(input_tokens)}")
            #     output_tokens = tokenizer.tokenize(prediction)
            #     print(f"输出token数: {len(output_tokens)}")
            #     print(f"输出token: {output_tokens}")
            if is_correct:
                correct_list.append(index)
                correct_predictions_1 += 1
                
        except Exception as e:
            print(f"处理示例 {index} 时出错: {e}")
            continue

    print(f'ACC@1:{correct_predictions_1 / len(lines)}')
    # print(f'ACC@5:{correct_predictions_5 / len(lines)}')
    # print(f'ACC@10:{correct_predictions_10 / len(lines)}')
    print(f'correct_index:{correct_list}')


if __name__ == "__main__":
    args = parse_config()
    main(args)

