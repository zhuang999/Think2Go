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


os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from llama_attn_replace_sft import replace_llama_attn
#from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from trainers.RecPOTrainer import RecPOTrainer, RecPOTrainingArguments
from trainers.GRecTrainer import GenRecTrainingArguments
from trl.trainer.utils import pad
from peft import PeftModel
import numpy as np
from collections import Counter
from sklearn.cluster import AgglomerativeClustering

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
class TrainingArguments(RecPOTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    peft_model_path: str = field(
        default=None,
        metadata={"help": "Path to the peft model."},
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
        default=1,
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
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove unused columns from the dataset."},
    )
    
    # Generation config parameters 
    max_new_tokens: int = field(default=1024)
    generation_temperature: float = field(default=1.3)  # 1.3
    generation_top_k: int = field(default=200)
    generation_top_p: float = field(default=1.0)
    num_return_sequences: int = field(default=4)
    generation_do_sample: bool = field(default=True)
    repetition_penalty: float = field(default=1.05)
    
    # no_repeat_ngram_size: int = field(default=3)
    # Beam search
    num_beams: int = field(
        default=20,
        metadata={"help": "Number of beams for beam search. Set to 1 to disable beam search."}
    )
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to stop the beam search when at least num_beams sentences are finished per batch or not."}
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for beam search. > 1.0 encourages longer sequences, < 1.0 encourages shorter sequences."}
    )
    

    emb_end_token: Optional[str] = field(
        default='<|end_of_text|>',
        metadata={"help": "Token to stop generation at."}
    )
    
    def to_dict(self):
        """重写to_dict方法，排除generation_config以避免JSON序列化问题"""
        d = super().to_dict()
        if 'generation_config' in d:
            del d['generation_config']
        return d
    

    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    import copy

    old_tokenizer = copy.deepcopy(tokenizer)

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
            add_special_tokens=False,  
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
    # all = sources + targets
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, targets)]
    
    input_ids = examples_tokenized["input_ids"]
    labels = targets_tokenized["labels"]

    
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
    """Dataset for supervised fine-tuning with chat format."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs with chat template...")

        sources = []
        targets = []
        user_contents = []
        time_difficulties = []
        spatial_difficulties = []
        

        for example in list_data_dict:
            system_content = "You are a helpful assistant specialized in location prediction. You analyze POI access patterns to predict next locations. Respond in the following format: <think>\n[Your step-by-step reasoning here]\n</think>\n<answer>: <a_7><b_25><c_16></answer><|end_of_text|>"
    
            # instruction = "Here is a record of a user's POI check-in trajectory. Your task is to analyze the user's POI check-in trajectory patterns and predict the next location they will visit at the specified time. Based on your analysis of temporal patterns, location transitions, and user preferences in the history, predict the next POI using RQ-VAE semantic IDs, such as <a_7><b_25><c_16> or <a_30><b_30><c_28><d_1>. The prediction should be constructed step by step, including <a_7>, <b_25>, <c_16>, and if needed, <d_1> components. Format your response as: <think>Your detailed analysis of the user's patterns, considering factors like time, frequency, transitions between locations, and preferences. Explain your reasoning for each component of the predicted ID.</think> <answer>: <a_7><b_25><c_16></answer>."
            
            instruction = f"Here is a record of a user's POI check-in trajectory. Your task is to analyze the user's POI check-in trajectory patterns and predict the next location they will visit at the specified time.  You should perform a detailed analysis of temporal patterns, transitions between POIs, and user preferences. Present your reasoning step-by-step inside a <think> and </think> block. After completing your analysis, provide your final POI prediction within <answer>: and </answer> tags. For example: <answer>: <a_7><b_25><c_16></answer>{tokenizer.eos_token}."
            
            #In the trajectory data: 'Cat:' is category; 'TimeDiff:' is hours from historical check-in to target time; 'DistToCurrent:' is km distance from historical check-in to current POI.

            # instruction = f"Here is a record of a user's POI check-in trajectory. Your task is to analyze the user's POI check-in trajectory patterns and predict the next location they will visit at the specified time. You should perform a detailed analysis of temporal patterns, transitions between POIs, and user preferences. Based on your reasoning, recommend the next the user might access, and wrap the final recommendation inside <answer>: and </answer>. For example: <answer>: <a_7><b_25><c_16></answer>{tokenizer.eos_token}."

            # instruction = "Here is a record of a user's POI check-in trajectory. Your task is to analyze the user's POI check-in trajectory patterns and predict the next location they will visit at the specified time. Based on your analysis of temporal patterns, location transitions, and user preferences in the history, predict the next POI using RQ-VAE semantic IDs in the format <a_x><b_x><c_x> or <a_x><b_x><c_x><d_x>. The prediction should be constructed step by step, including <a_x>, <b_x>, <c_x>, and if needed, <d_x> components. Format your response using **<think> and </think> as delimiters for your detailed analysis, followed by <answer> and </answer> as delimiters for the final predicted semantic ID (e.g., <answer>:<a_x><b_x><c_x></answer>)**."
            

            instruction_text = example["instruction"].strip()
            input_text = example["input"].strip()
            user_content = f"{instruction}\n\n{input_text}"
            user_contents.append(_tokenize_fn(user_content, tokenizer)["input_ids"])
            time_difficulties.append(example["time_difficulty"])
            spatial_difficulties.append(example["spatial_difficulty"])

            output = example["output"].strip()
            assistant_content = f"<think></think> <answer>:{output}</answer>{tokenizer.eos_token}"
            

            history_part = format_chat_conversation(
                system_content=system_content,
                user_content=user_content,
                assistant_content="",  
                tokenizer=tokenizer
            )
            history_part = history_part + '<think>'
            

            reasoning_part = f"</think>\n<answer>: {output}</answer>{tokenizer.eos_token}"
            
            sources.append(history_part)  
            targets.append(reasoning_part)  
        

            
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.time_difficulties = torch.tensor(time_difficulties, dtype=torch.float)
        self.spatial_difficulties = torch.tensor(spatial_difficulties, dtype=torch.float)
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], time_difficulties=self.time_difficulties[i], spatial_difficulties=self.spatial_difficulties[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Enhanced data collator with RRecDataCollator functionality integrated."""

    tokenizer: transformers.PreTrainedTokenizer
    mlm: bool = field(default=False)
    mlm_probability: float = field(default=0.0)
    pad_to_multiple_of: Optional[int] = field(default=None)
    return_tensors: str = field(default="pt")

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.mlm = kwargs.get("mlm", False)
        self.mlm_probability = kwargs.get("mlm_probability", 0.0)
        self.pad_to_multiple_of = kwargs.get("pad_to_multiple_of", None)
        self.return_tensors = kwargs.get("return_tensors", "pt")
        self.system_content = "You are a helpful assistant specialized in location prediction. You analyze POI access patterns to predict next locations."
            
    def torch_call(self, examples):
        
        batch = {}
        batch["input_ids"] = []
        batch["input_ids_origin"] = []
        batch["token_mask"] = []
        batch["labels"] = []
        batch["pred_ids"] = []
        batch["rewards"] = []
        batch["cluster_semantic_entropy"] = []
        batch["acc"] = []
        batch["time_difficulties"] = []
        batch["spatial_difficulties"] = []
        feature_keys = examples.keys()
        think_token = torch.tensor(self.tokenizer.encode("</think>\n", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        transition_prompt = torch.tensor(self.tokenizer.encode(" Based on the reasoning above, we now evaluate both the format and the accuracy of the prediction.", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        right_miss_token_prompt = torch.tensor(self.tokenizer.encode(" First of all, the response format is correct.", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        error_miss_token_prompt = torch.tensor(self.tokenizer.encode(" First of all, the format is incorrect as it lacks the ", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        error_id_error_prompt = torch.tensor(self.tokenizer.encode(" And, it contains more than one ", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        token_id = torch.tensor(self.tokenizer.encode(" token.", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        right_prompt = torch.tensor(self.tokenizer.encode(" Fortunately, the predicted answer is correct. It is likely that the user will visit:", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        error_prompt = torch.tensor(self.tokenizer.encode(" Unfortunately, the predicted answer is incorrect. It is more likely that the user will visit:", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)



        right_format_error_prompt = torch.tensor(self.tokenizer.encode(" Fortunately, the predicted answer is correct. Final answer:", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)
        error_format_error_prompt = torch.tensor(self.tokenizer.encode("Unfortunately, the predicted answer is incorrect. Final answer:", add_special_tokens=False), dtype=torch.long).to(examples["input_ids"].device)

        for i in range(examples["input_ids"].shape[0]):
            prompt = examples["input_ids_no_pad"][i]
            reasoning = examples["reasoning"][i]
            user_content = examples["user_contents"][i]
            label = examples["labels"][i]
            conversation_list = []
            conversation_list_origin = []
            pred_ids_list = []
            label_list = []
            rewards_list = []
            acc_list = []
            pred_answer_text_list = []
            pred_poi_list = []
            for j in range(len(reasoning)):
                user_content_len = len(user_content)    
                reasoning_len = len(reasoning[j])
                conversation = torch.cat((user_content, torch.tensor(reasoning[j], dtype=torch.long).to(user_content.device)),dim=0)
                pred_ids_list.append(torch.tensor(reasoning[j], dtype=torch.long).to(user_content.device))
                conversation_list_origin.append(conversation)
                reasoning_text = self.tokenizer.decode(reasoning[j], skip_special_tokens=False)

                acc, reward_soft_format, reward_strict_format, pred_answer_text, rep_penalty, pred_poi = self.my_compute_metrics(pred_ids_list[-1], label)
                rewards = acc + reward_soft_format + reward_strict_format - rep_penalty
                rewards_list.append(rewards)
                acc_list.append(acc)
                pred_answer_text_list.append(pred_answer_text)
                pred_poi_list.append(pred_poi)

                if conversation[-1].item() == self.tokenizer.eos_token_id:
                    conversation = conversation[:-1]
                    reasoning_len -= 1
                conversation_reasoning_len = conversation.shape[0]


                if acc == 1:
                    conversation = torch.cat([conversation, right_prompt, label], dim=0)

                    token_label = torch.full_like(conversation, -100)

                    token_label[conversation_reasoning_len+right_prompt.shape[0]:conversation_reasoning_len+right_prompt.shape[0]+len(label)] = label
                else:
                    conversation = torch.cat([conversation, error_prompt, label], dim=0)

                    token_label = torch.full_like(conversation, -100)
                    token_label[conversation_reasoning_len+error_prompt.shape[0]:conversation_reasoning_len+error_prompt.shape[0]+len(label)]= label
                # token_label = label
                label_list.append(token_label)
                token_mask = torch.full_like(conversation, 0)
                token_mask[user_content_len:user_content_len+reasoning_len] = 1
                batch["token_mask"].append(token_mask)
                conversation_list.append(conversation)
                # conversation_char = self.tokenizer.decode(conversation_char.tolist(), skip_special_tokens=False)
                # print("conversation_char", conversation_char)
            #cluster_semantic_entropy = self.cluster_semantic_ids(pred_poi_list)
            #batch["cluster_semantic_entropy"].append(cluster_semantic_entropy)
            batch["input_ids"].append(conversation_list)
            batch["input_ids_origin"].append(conversation_list_origin)
            batch["pred_ids"].append(pred_ids_list)
            batch["labels"].append(label_list)
            batch["rewards"].append(rewards_list)
            batch["acc"].append(acc_list)
            batch["time_difficulties"].append(examples["time_difficulties"][i])
            batch["spatial_difficulties"].append(examples["spatial_difficulties"][i])
        import itertools
        batch["input_ids"] = list(itertools.chain.from_iterable(batch["input_ids"]))
        batch["input_ids_origin"] = list(itertools.chain.from_iterable(batch["input_ids_origin"]))
        batch["pred_ids"] = list(itertools.chain.from_iterable(batch["pred_ids"]))
        batch["labels"] = list(itertools.chain.from_iterable(batch["labels"]))
        batch["rewards"] = torch.tensor(batch["rewards"], dtype=torch.float)
        batch["acc"] = torch.tensor(batch["acc"], dtype=torch.float)
        batch["time_difficulties"] = torch.tensor(batch["time_difficulties"], dtype=torch.float)
        batch["spatial_difficulties"] = torch.tensor(batch["spatial_difficulties"], dtype=torch.float)
        batch["cluster_semantic_entropy"] = torch.tensor(batch["cluster_semantic_entropy"], dtype=torch.float)
        for key in batch.keys():
            if key == 'rewards' or key == 'acc' or key == 'cluster_semantic_entropy' or key == 'time_difficulties' or key == 'spatial_difficulties':
                continue
            if key == 'token_mask':
                batch[key] = pad(
                batch[key],
                0, "right"
            )
            if key == 'labels':
                batch[key] = pad(
                    batch[key],
                    -100, "right"
                )
            batch[key] = pad(
                batch[key],
                self.tokenizer.pad_token_id, "right"
            )
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)
        return batch

    
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, time_difficulties, spatial_difficulties = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "time_difficulties", "spatial_difficulties"))
        input_ids_pad = pad(
            input_ids, self.tokenizer.pad_token_id, "left"
        )


        return dict(
            input_ids=input_ids_pad,
            labels=labels,
            attention_mask=input_ids_pad.ne(self.tokenizer.pad_token_id),
            input_ids_no_pad=input_ids,
            user_contents=input_ids,
            time_difficulties=time_difficulties,
            spatial_difficulties=spatial_difficulties
        )
    
    def find_completion_start_end(self, current_input_ids, search_start_index=0):
        start_pattern = self.tokenizer.encode("<answer>:", add_special_tokens=False)
        end_pattern = self.tokenizer.encode("</answer>", add_special_tokens=False)
        start_len = len(start_pattern)
        end_len = len(end_pattern)
        in_lens = len(current_input_ids)

        batch_size = len(in_lens)
        
        # find the start pattern from the end
        current_in_len = in_lens
        start_index = None
        for idx in range(current_in_len - start_len - end_len, -1, -1):
            if current_input_ids[idx:idx + start_len] == start_pattern or current_input_ids[idx:idx + start_len] == end_pattern:
                start_index = idx
                break
        if start_index is None:
            raise ValueError(
                f"start pattern [{self.tokenizer.decode(start_pattern)}] "
                f"not found in input_ids:\n {self.tokenizer.decode(current_input_ids)}")
        if end_pattern not in current_input_ids[start_index + start_len:]:
            result = (start_index + start_len, current_in_len)
        else:
            for j in range(start_index + start_len, current_in_len - end_len):
                if current_input_ids[j:j + end_len] == end_pattern:
                    result = (start_index + start_len, j + 1)
                    break

        return result
    
    def find_completion_start_end_special_token(self, current_input_ids, search_start_index=0):

        if hasattr(current_input_ids, 'tolist'):
            current_input_ids = current_input_ids.tolist()
            
        # 获取 <answer> 和 </answer> 的token ID
        answer_start_id = self.tokenizer.convert_tokens_to_ids("<answer>:")
        answer_end_id = self.tokenizer.convert_tokens_to_ids("</answer>")
        
        current_in_len = len(current_input_ids)
        

        if answer_start_id not in current_input_ids[search_start_index:]:

            return None, [-1, -1]
        
        start_index = current_input_ids.index(answer_start_id, search_start_index)
        
        if answer_end_id in current_input_ids[start_index + 1:]:
            end_index = current_input_ids.index(answer_end_id, start_index + 1)
            content_start = start_index  
            content_end = end_index + 1  
        else:
            content_start = start_index  
            content_end = current_in_len
        
        prediction_id = current_input_ids[content_start:content_end]

        return prediction_id, [content_start, content_end]
    
    def my_compute_metrics(self, predictions, ground_truths):

        import re
        miss_token = []
        format_error = 0  
        id_error = []
        pred_answer_text = ''
        

        acc = 0  
        reward_soft_format = 0
        reward_strict_format = 0
        pred_poi = ''
        gt = ground_truths
        pred = predictions
        pred_text = self.tokenizer.decode(pred[pred != -100], skip_special_tokens=False)
        # print("pred_text:", pred_text)
        rep_penalty = self.compute_repetition_penalty(pred_text, n=3)
        
        if pred_text.count('<answer>:') == 1 and len(pred_text) > 300:
            
            answer_start = pred_text.find('<answer>:')
            if answer_start != -1:
                answer_end = pred_text.find('</answer>', answer_start)
                if answer_end != -1:
                    answer_content = pred_text[answer_start + len('<answer>:'):answer_end].strip()
                    
                    semantic_id_pattern = re.compile(r'<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?')
                    if semantic_id_pattern.search(answer_content):
                        reward_soft_format += 0.5
                    
                else:
                    reward_soft_format += 0.1

        if reward_soft_format == 0.5:

            gt_text = self.tokenizer.decode(gt, skip_special_tokens=False)
            pred_text = self.tokenizer.decode(pred, skip_special_tokens=False)

            answer_tag = "</think>\n<answer>:"
            gt_answer_pos = gt_text.rfind(answer_tag)  
            pred_answer_pos = pred_text.rfind(answer_tag)  


            strict_pattern1 = re.compile(r'</think>\n<answer>:(.*?)</answer><\|end_of_text\|>', re.DOTALL)
            match = strict_pattern1.search(pred_text[pred_answer_pos:])
            if match:
                answer_content = match.group(1).strip()

                strict_semantic_pattern = re.compile(r'^\s*<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?\s*$')
                if strict_semantic_pattern.match(answer_content):
                    reward_strict_format += 1.0

            if gt_answer_pos == -1 or pred_answer_pos == -1:
                return acc, reward_soft_format, reward_strict_format, pred_answer_text, rep_penalty, pred_poi#, miss_token, id_error, format_error

            if reward_strict_format == 1:

                gt_answer_text = gt_text[gt_answer_pos + len(answer_tag):]
                pred_answer_text = pred_text[pred_answer_pos + len(answer_tag):]
                reasoning_text = pred_text[:pred_answer_pos]

                if "</answer>" in gt_answer_text:
                    gt_answer_text = gt_answer_text[:gt_answer_text.find("</answer>")]
                elif "<|end_of_text|>" in gt_answer_text:
                    gt_answer_text = gt_answer_text[:gt_answer_text.find("<|end_of_text|>")]
                #print("gt_answer_text:*********************", gt_answer_text)
                if "</answer>" in pred_answer_text:
                    pred_answer_text = pred_answer_text[:pred_answer_text.find("</answer>")]
                elif "<|end_of_text|>" in pred_answer_text:
                    pred_answer_text = pred_answer_text[:pred_answer_text.find("<|end_of_text|>")]

                continuous_pattern = re.compile(r'<a_\d+><b_\d+><c_\d+>(?:<d_\d+>)?(?!<[a-z]_\d+>)')
                def extract_ids_from_text(text):

                    poi_matches = continuous_pattern.findall(text)
                    poi = ''.join(poi_matches)
                    is_format_ok = len(poi_matches) > 0
                    return poi, text, is_format_ok
                
                gt_poi, gt_text_extracted, is_format_ok_gt = extract_ids_from_text(gt_answer_text)
                pred_poi, pred_text_extracted, is_format_ok_pred = extract_ids_from_text(pred_answer_text)
                reasoning_poi, reasoning_text_extracted, is_format_ok_reasoning = extract_ids_from_text(reasoning_text)
                

                gt_individual_ids = re.findall(r'<[a-d]_\d+>', gt_poi)
                pred_individual_ids = re.findall(r'<[a-d]_\d+>', pred_poi)
                reasoning_individual_ids = re.findall(r'<[a-d]_\d+>', reasoning_poi)
                

                matched_count = 0
                reasoning_matched_count = 0
                for gt_id in gt_individual_ids:
                    if gt_id in pred_individual_ids:
                        matched_count += 1
                    # if gt_id in reasoning_individual_ids:
                    #     reasoning_matched_count += 1
                

                if len(gt_individual_ids) > 0:
                    acc = (matched_count / len(gt_individual_ids)) * 3
                # if reasoning_matched_count > 0:
                #     reward_strict_format += (reasoning_matched_count / len(reasoning_individual_ids)) * 3
                if acc == 3:
                    print("find the answer*********************", pred_answer_text)

                if is_format_ok_pred:
                    format_error = 1

        acc = torch.tensor(acc, dtype=torch.float32, device=predictions.device)
        reward_soft_format = torch.tensor(reward_soft_format, dtype=torch.float32, device=predictions.device)
        reward_strict_format = torch.tensor(reward_strict_format, dtype=torch.float32, device=predictions.device)
        rep_penalty = torch.tensor(rep_penalty, dtype=torch.float32, device=predictions.device)
        return acc, reward_soft_format, reward_strict_format, pred_answer_text, rep_penalty, pred_poi#, miss_token, id_error, format_error
    
    def cluster_semantic_ids(self, pred_answer_text_list):

        all_units = set()
        parsed_units = []
        for sid in pred_answer_text_list:
            units = re.findall(r'<[a-d]_\d+>', sid)
            parsed_units.append(units)
            all_units.update(units)
        

        unit_to_idx = {unit: i for i, unit in enumerate(sorted(all_units))}
        X = np.zeros((len(pred_answer_text_list), len(all_units)), dtype=np.int8)
        
        for i, units in enumerate(parsed_units):
            for unit in units:
                X[i, unit_to_idx[unit]] = 1
        
        
        labels = AgglomerativeClustering(
            n_clusters=2, metric='jaccard', linkage='average'
        ).fit_predict(X)
        
        # 计算每个聚类的概率
        label_counts = Counter(labels)
        total = len(pred_answer_text_list)
        cluster_probs = {label: count / total for label, count in label_counts.items()}
        
        # 返回每个语义ID对应的聚类概率列表
        return [cluster_probs[label] for label in labels]
    

    def compute_repetition_penalty(self, text, n=3):
        tokens = text.strip().split()
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        ngram_counts = Counter(ngrams)


        num_repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
        total = len(ngrams)


        repetition_ratio = num_repeated / total if total > 0 else 0.0
        return repetition_ratio
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning with chat format."""
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
    


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length+training_args.max_new_tokens,
        padding_side="right",
        use_fast=True,
    )  #model_args.model_name_or_path,

    
    
    special_tokens_dict = dict()
    # 加入基础特殊token
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


    # additional_tokens = ["<think>", "</think>", "<answer>:", "</answer>"] + sorted(list(unique_semantic_ids)) #+ sorted(list(catname_names))
    # special_tokens_dict['additional_special_tokens'] = additional_tokens
    
    # semantic_token_ids = set(tokenizer.convert_tokens_to_ids(t) for t in unique_semantic_ids)
    
    # logging.warning(f"Found {len(unique_semantic_ids)} unique semantic IDs. Total new special tokens: {len(additional_tokens)}")
    # print("unique_semantic_ids: ", sorted(list(unique_semantic_ids)))

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=ref_model,
    # )


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
    

    tokenizer = setup_chat_template(tokenizer, model_args.model_name_or_path)



    from transformers import GenerationConfig
    training_args.generation_config = GenerationConfig(
        max_new_tokens=training_args.max_new_tokens,
        temperature=training_args.generation_temperature,
        top_k=training_args.generation_top_k,
        top_p=training_args.generation_top_p,
        num_return_sequences=training_args.num_return_sequences,
        do_sample=training_args.generation_do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=training_args.repetition_penalty,
        # num_beams=training_args.num_beams,
        early_stopping=training_args.early_stopping,
        length_penalty=training_args.length_penalty,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):

            outputs = model(**inputs)
            loss = outputs.loss
            

            logits = outputs.logits
            batch_size = inputs["input_ids"].shape[0]  
            
            labels = inputs["labels"]
            
            # logits: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len]
            shift_logits = logits[..., :-1, :].contiguous()  
            shift_labels = labels[..., 1:].contiguous()      
            

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch_size * (seq_len-1), vocab_size]
            flat_labels = shift_labels.view(-1)                         # [batch_size * (seq_len-1)]
            

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            

            loss = loss_fct(flat_logits, flat_labels)
                

            
            if return_outputs:
                return loss, outputs
            return loss

    if training_args.low_rank_training:
        if training_args.peft_model_path and os.path.exists(training_args.peft_model_path):
            
            trainable_params = os.path.join(training_args.peft_model_path, "pytorch_model.bin")
            print(f"Loading PEFT model from: {training_args.peft_model_path}")
            model = PeftModel.from_pretrained(
                model,
                training_args.peft_model_path,
                torch_dtype=torch.float16,
                device_map="auto",  
                is_trainable=True,  # 确保模型可训练
            )
            print("Loaded PEFT trainable peft params",trainable_params)
        else:
            # 创建新的LoRA配置
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
        
        # # ====== 使用深拷贝保存vLLM专用float16权重和tokenizer ======
        # print("🔧 开始保存vLLM专用权重...")
        
        # # 使用深拷贝方法：复制模型并合并权重，保持原PEFT结构不变
        # import copy
        # print("🔄 正在创建模型的深拷贝...")
        # model_copy = copy.deepcopy(model)  # 创建模型的深拷贝
        # print("✅ 深拷贝完成")
        
        # # 对复制的模型进行merge操作，生成纯净的基础模型
        # print("🔗 正在合并LoRA权重...")
        # merged_model = model_copy.merge_and_unload()  # 合并权重并获取纯净模型
        # print("✅ 权重合并完成")
        # print("📋 正在过滤量化权重并保存 ...")
        # clean_state_dict = {}
        # for name, param in merged_model.state_dict().items():
        #     if any(x in name for x in ['.absmax', '.quant_state', '.quant_map', '.nested_quant_map', '.nested_absmax', '.quant_state_dict']):
        #         continue
        #     clean_state_dict[name] = param
        # torch.save(clean_state_dict, "./vllm_model/pytorch_model.bin")
        # print("✅ 已保存过滤后的量化权重到 ./vllm_model/pytorch_model.bin")
        # # 保存合并后的纯净模型供vLLM使用
        # print("💾 正在保存合并后的模型...")
        # merged_model.save_pretrained("./vllm_model", state_dict=clean_state_dict)  # 保存合并后的模型
        # torch.save(clean_state_dict, "./vllm_model/pytorch_model.bin")
        # print("✅ 已保存过滤后的量化权重到 ./vllm_model/pytorch_model.bin")
        # # 保存合并后的纯净模型供vLLM使用
        # print("💾 正在保存合并后的模型...")
        # merged_model.save_pretrained("./vllm_model", state_dict=clean_state_dict)  # 保存合并后的模型
        # tokenizer.save_pretrained("./vllm_tokenizer")
        # print("✅ 已保存vLLM专用权重到 ./vllm_model 和 ./vllm_tokenizer")
        
        # # 释放深拷贝模型的内存
        # del model_copy
        # del merged_model
        # print("🧹 已释放深拷贝模型内存")

        # import json

        # config_path = "./vllm_model/config.json"
        # with open(config_path, "r", encoding="utf-8") as f:
        #     config = json.load(f)

        # # 删除量化相关字段
        # if "quantization_config" in config:
        #     del config["quantization_config"]
        # for key in ["load_in_4bit", "load_in_8bit", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant"]:
        #     if key in config:
        #         del config[key]

        # with open(config_path, "w", encoding="utf-8") as f:
        #     json.dump(config, f, indent=2, ensure_ascii=False)

        # print("已自动去除 config.json 中的量化参数！")
            

        # # 打印模型参数到文件
        # with open('model_params.txt', 'w') as f:
        #     f.write('模型参数名称和维度:\n')
        #     for name, param in model.named_parameters():
        #         f.write(f'{name}: {param.shape}\n')
                
        # # ====== 新增：保存bitsandbytes量化权重和tokenizer ======
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=special_tokens_dict,
        #     tokenizer=tokenizer,
        #     model=model,
        # )
        # quant_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4"
        # )
        # # 合并LoRA权重（如有）
        # if hasattr(model, 'merge_and_unload'):
        #     base_model = model.merge_and_unload()
        # else:
        #     base_model = model
        # # 先保存合并后的模型到临时目录
        # tmp_merged_dir = "./temp_merged_model"
        # os.makedirs(tmp_merged_dir, exist_ok=True)
        # base_model.save_pretrained(tmp_merged_dir)
        # # 重新加载为量化模型
        # from transformers import AutoModelForCausalLM
        # quantized_model = AutoModelForCausalLM.from_pretrained(
        #     tmp_merged_dir,
        #     quantization_config=quant_config,
        #     device_map="cpu"
        # )
        # os.makedirs("./vllm_model", exist_ok=True)
        # quantized_model.save_pretrained("./vllm_model")
        # tokenizer.save_pretrained("./vllm_tokenizer")
        # print("已保存bitsandbytes量化权重到 ./vllm_model，tokenizer到 ./vllm_tokenizer")

        



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
    # 创建训练器
    trainer = RecPOTrainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args, 
        **data_module
    )
    # 设置起始步数为500
    # trainer.state.global_step = 1500
    #trainer.state.epoch = 0.0  # 重置epoch计数
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    # Save trainable parameters (embed and norm layers)
    trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save(trainable_state_dict, os.path.join(training_args.output_dir, "trainable_params.bin"))
    print(f"Trainable parameters saved to {os.path.join(training_args.output_dir, 'trainable_params.bin')}")

if __name__ == "__main__":
    train()