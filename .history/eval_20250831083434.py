# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        # 如果没有assistant内容，添加生成提示
        add_generation_prompt = True
    
    # 使用tokenizer的chat template格式化对话
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
        #model_path,
        output_dir,
        model_max_length=2048,
        padding_side="right",
        use_fast=True,
    )

    
    # 根据模型类型设置适合的chat template
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
    
    # 设置模型特定的chat template
    tokenizer = setup_chat_template(tokenizer, model_path)
    # 打印 chat_template

    # print(tokenizer('6', return_tensors="pt").to(device))
    # print(tokenizer.decode([    29946]))
    # sys.exit()
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
        device_map="auto",
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
        # if os.path.isfile(trainable_params):
        #     params_before = {k: v.clone() for k, v in model.state_dict().items()}
        #     model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        #     # 3. 获取load_state_dict之后的参数
        #     params_after = model.state_dict()

        #     # 4. 对比参数
        #     all_same = True
        #     for k in params_before:
        #         if not torch.equal(params_before[k], params_after[k]):
        #             print(f"参数 {k} 在加载前后不一致")
        #             all_same = False

        #     if all_same:
        #         print("所有参数在加载前后完全一致（说明权重文件和原模型参数一样）")
        #     else:
        #         print("有参数发生了变化（说明权重文件和原模型参数不一样，或者确实被load覆盖了）")

        #     print("trainable_params exists:", os.path.isfile(trainable_params))
        #     print("Loaded trainable params1",trainable_params)
        model = PeftModel.from_pretrained(
            model,
            output_dir,
            device_map="auto",
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

    def evaluate_prediction_accuracy(prediction, ground_truth):
        """评估预测的准确性
        
        Args:
            prediction: 模型预测的输出，可能是数字或带有特殊格式的字符串
            ground_truth: 真实标签，格式可能是 <a_X><b_Y><c_Z><d_W> 或其他格式
            
        Returns:
            int: 如果预测正确返回1，否则返回0
        """
        # print(f"\n预测原始输出: '{prediction}'")
        # print(f"真实标签: '{ground_truth}'")
        answer_start = prediction.find('<answer>:')
        if answer_start != -1:
                prediction = prediction[answer_start + len('<answer>:'):].strip()
                
        # 直接从预测中提取所有POI标签，不关心其他内容
        pred_poi_parts = []
        pred_poi_matches = re.findall(r'<([a-z])_(\d+)>', prediction)
        for type_char, value in pred_poi_matches:
            pred_poi_parts.append(f"<{type_char}_{value}>")
        
        pred_poi_sequence = "".join(pred_poi_parts)
        
        # 从真实标签中提取POI序列
        gt_poi_parts = []
        gt_poi_matches = re.findall(r'<([a-z])_(\d+)>', ground_truth)
        for type_char, value in gt_poi_matches:
            gt_poi_parts.append(f"<{type_char}_{value}>")
        
        gt_poi_sequence = "".join(gt_poi_parts)
        

        
        # 比较纯POI序列
        is_correct = (pred_poi_sequence == gt_poi_sequence) and gt_poi_sequence != ""
        #print(f"POI序列匹配结果: {is_correct}")
        if is_correct:
            # # 打印提取的纯POI序列
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
    model.eval()
    device = 'cuda'
    # Iterate over each line and ask the LLM
    correct_list = []
    for index, line in tqdm(enumerate(lines), desc="Processing lines", total=len(lines)):
        try:
            # 处理JSON格式数据
            #instruction = line.get("instruction", "")

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
            # 生成
            outputs = model.generate(**prompt_tokens, generation_config=generation_config)
            
            
            # 解码预测结果，保留特殊标记以便于匹配<answer>标记
            prediction = tokenizer.decode(outputs[:, prompt_tokens.input_ids.shape[1]:][0], skip_special_tokens=False)
            # print("prediction",prediction)
            # print("ground_truth",ground_truth)
            

            gt = ground_truth.replace('[', '').replace(']', '')
            

            is_correct = evaluate_prediction_accuracy(prediction, gt)
            

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

