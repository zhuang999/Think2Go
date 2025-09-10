import io
import os
import copy
import json
import math
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from llama_attn_replace_sft import replace_llama_attn
#from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
os.environ["WANDB_DISABLED"]="true"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2":(
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Llama-3.1-8B")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    data_path: str = field(default="datasets/NYC/train_codebook.json", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm,lm_head",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit."},
    )
    num_train_epochs: float = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."},
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Save checkpoint every X updates steps."},
    )
    save_total_limit: int = field(
        default=5,
        metadata={"help": "Limit the total amount of checkpoints."},
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    warmup_steps: int = field(
        default=20,
        metadata={"help": "Linear warmup over warmup_steps."},
    )
    lr_scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={"help": "The scheduler type to use."},
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every X updates steps."},
    )
    tf32: bool = field(
        default=True,
        metadata={"help": "Whether to enable tf32 mode."},
    )
    deepspeed: str = field(
        default="ds_configs/stage2.json",
        metadata={"help": "Enable deepspeed and pass the path to deepspeed json config file."},
    )
    
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



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")


        sources = []
        targets = []
        

        for example in list_data_dict:

            instruction = example["instruction"].strip()
            source = f" <question> : {instruction}"
            

            if example.get("input", "").strip():
                input_text = example["input"].strip()
                source = f"{source}\n\n{input_text}"
            
                # source = format_chat_conversation(
                #     system_content=instruction,
                #     user_content=input_text,
                #     assistant_content="",  
                #     tokenizer=tokenizer
                # )
                

            output = example["output"].strip()

            target = f" <answer>: {output}{tokenizer.eos_token}"
            sources.append(source)
            targets.append(target)
        

        # if len(sources) > 0:
        #     print(f"\n=== 训练数据格式示例 ===")
        #     for i in range(min(3, len(sources))):
        #         print(f"样本 {i+1}:")
        #         print(f"  Source: {sources[i]}")
        #         print(f"  Target: {targets[i]}")
        #         print()
        #     print("========================\n")
            
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    else:
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn, inference=False)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )
    

    if not hasattr(config, "tp_parallel") or config.tp_parallel is None:
        config.tp_parallel = "none"
    if not hasattr(config, "model_parallel_style") or config.model_parallel_style is None:
        config.model_parallel_style = "none"

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    else:
        config.rope_scaling = {"type": "linear", "factor": 1.0}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    #     # --- 从数据中动态提取并添加特殊token ---
    # logging.warning("Dynamically extracting special tokens from data...")
    # list_data_dict = jload(data_args.data_path)

    # # 使用正则表达式匹配 <a_x>, <b_x>, <c_x>, <d_x> 格式的ID
    # semantic_id_pattern = re.compile(r"<[a-d]_[0-9]+>")
    # unique_semantic_ids = set()
    # for example in list_data_dict:
    #     # 从 input 和 output 字段中查找
    #     input_ids = semantic_id_pattern.findall(example.get("input", ""))
    #     output_ids = semantic_id_pattern.findall(example.get("output", ""))
    #     unique_semantic_ids.update(input_ids)
    #     unique_semantic_ids.update(output_ids)

    # def extract_semantic_sequences(text):
    #     """
    #     从文本中提取所有可能的语义ID序列
        
    #     Args:
    #         text: 要分析的文本
            
    #     Returns:
    #         set: 包含所有找到的语义ID序列的集合
    #     """
    #     # 使用正则表达式提取连续的语义ID序列
    #     pattern = r'(<[a-z]_\d+>)+'
        
    #     # 找出所有连续的语义ID序列
    #     full_sequences = []
    #     current_pos = 0
    #     while True:
    #         match = re.search(pattern, text[current_pos:])
    #         if not match:
    #             break
            
    #         start, end = match.span()
    #         sequence = text[current_pos + start:current_pos + end]
    #         full_sequences.append(sequence)
    #         current_pos += end
        
    #     return full_sequences

    # unique_semantic_ids = set()
    # for example in list_data_dict:
    #     # 从 input 和 output 字段中查找
    #     input_ids = extract_semantic_sequences(example.get("input", ""))
    #     output_ids = extract_semantic_sequences(example.get("output", ""))
    #     unique_semantic_ids.update(input_ids)
    #     unique_semantic_ids.update(output_ids)
    

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # # 加入自定义的特殊token
    # additional_tokens = ["<think>", "</think>", "<answer>:", "</answer>"] + sorted(list(unique_semantic_ids))
    # special_tokens_dict['additional_special_tokens'] = additional_tokens


    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print('tokenizer length:', len(tokenizer))


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
    tokenizer = setup_chat_template(tokenizer, model_args.model_name_or_path)
    # 打印 chat_template

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # 创建自定义的Trainer类，覆盖compute_loss方法以打印预测正确的情况
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            #print("inputs:", self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
            # 调用原始的compute_loss获取输出
            outputs = model(**inputs)
            loss = outputs.loss
            
            # # 对每个批次都进行检查
            # logits = outputs.logits
            # batch_size = inputs["input_ids"].shape[0]  # 获取实际批次大小
            
            # # 获取labels
            # labels = inputs["labels"]
            
            # # 将logits和labels展平以便计算交叉熵
            # # logits: [batch_size, seq_len, vocab_size]
            # # labels: [batch_size, seq_len]
            # shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置的logits
            # shift_labels = labels[..., 1:].contiguous()      # 去掉第一个位置的labels（左移一位）
            # #print(shift_labels.shape, shift_logits.shape)
            
            # # 展平张量
            # flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch_size * (seq_len-1), vocab_size]
            # flat_labels = shift_labels.view(-1)                         # [batch_size * (seq_len-1)]
            
            # # 创建交叉熵损失函数，ignore_index=-100用于忽略padding和prompt部分
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            
            # # 计算交叉熵损失
            # loss = loss_fct(flat_logits, flat_labels)
                
            #     # 获取相应位置的预测
            #     valid_predictions = sample_predictions[valid_label_indices]
                
            #     # 将token ID转换为文本
            #     tokenizer = self.tokenizer
                
            #     # 将完整输入转换为文本
            #     full_input_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                
            #     # 将真实标签和预测转换为文本
            #     label_text = tokenizer.decode(valid_labels, skip_special_tokens=True)
            #     prediction_text = tokenizer.decode(valid_predictions, skip_special_tokens=True)
                
            #     print(f"预测文本: {prediction_text}")
            #     # 清理文本（删除多余空格和换行符）
            #     label_text = label_text.replace('\n', ' ').strip()
            #     prediction_text = prediction_text.replace('\n', ' ').strip()
                
                
            #     # 从标签中提取纯POI序列
            #     label_poi_parts = []
            #     label_poi_matches = re.findall(r'<([a-z])_(\d+)>', label_text)
            #     for type_char, value in label_poi_matches:
            #         label_poi_parts.append(f"<{type_char}_{value}>")
                
            #     label_poi_sequence = "".join(label_poi_parts)
                
            #     # 从预测中提取纯POI序列
            #     pred_poi_parts = []
            #     pred_poi_matches = re.findall(r'<([a-z])_(\d+)>', prediction_text)
            #     for type_char, value in pred_poi_matches:
            #         pred_poi_parts.append(f"<{type_char}_{value}>")
                
            #     pred_poi_sequence = "".join(pred_poi_parts)
                
            #     # 判断POI序列是否完全匹配
            #     is_exact_match = (label_poi_sequence == pred_poi_sequence) and label_poi_sequence != ""
                
            #     # 只打印完全匹配的样本
            #     if is_exact_match:
            #         # 从输入中提取指令和输入内容
            #         if "<question>" in full_input_text:
            #             question_match = re.search(r'<question>\s*:\s*(.*?)(?:<answer>|$)', full_input_text, re.DOTALL)
            #             instruction = question_match.group(1).strip() if question_match else ""
            #             input_content = ""
            #         else:
            #             # 如果没有找到POI，整个输入就是指令
            #             instruction = full_input_text.strip()
            #             input_content = ""
                    
            #         print("\n=== 预测完全正确的样本 ===")
            #         print(f"批次位置: {i+1}/{batch_size}")
            #         print(f"指令: {instruction}")
            #         if input_content:
            #             print(f"输入: {input_content}")
                    
            #         # 输出详细信息
            #         print(f"标签文本: {label_text}")
            #         print(f"预测文本: {prediction_text}")
            #         print(f"标签POI序列: {label_poi_sequence}")
            #         print(f"预测POI序列: {pred_poi_sequence}")
            #         print("========================\n")
            
            if return_outputs:
                return loss, outputs
            return loss

    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # 使用自定义Trainer
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    # Save trainable parameters (embed and norm layers)
    trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save(trainable_state_dict, os.path.join(training_args.output_dir, "trainable_params.bin"))
    print(f"Trainable parameters saved to {os.path.join(training_args.output_dir, 'trainable_params.bin')}")

if __name__ == "__main__":
    train()