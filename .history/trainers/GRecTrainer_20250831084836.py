import os
# os.environ['VLLM_USE_V1'] = '0'
# os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'  # 或 'XFORMERS'
import warnings
from collections import defaultdict
import multiprocessing as mp
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Dict, Callable, Optional, Union, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.amp as amp
import copy

from transformers.utils import logging
from transformers.trainer_utils import seed_worker
from transformers.integrations import deepspeed_init
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback, GenerationConfig
from transformers import Trainer, TrainingArguments
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax
from trl.trainer import disable_dropout_in_model
from accelerate.utils.other import is_compiled_module
from accelerate.utils import is_peft_model
import datasets
from llama_attn_replace_sft import replace_llama_attn
from vllm import LLM, SamplingParams, TokensPrompt
import rich

from unittest.mock import patch
from tqdm import tqdm
import re
import torch.distributed as dist

logger = logging.get_logger(__name__)

@dataclass
class GenRecTrainingArguments(TrainingArguments):
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ["seq_labels"],
        metadata={
            "help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    emb_token: Optional[str] = field(
        default='<answer>:',
        metadata={"help": "The token to indicate the end of the embedding."},
    )
    emb_end_token: Optional[str] = field(
        default='<|end_of_text|>',
        metadata={"help": "The token to indicate the end of the embedding."},
    )   #'</answer>'  '<|end_of_text|>'
    input_max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The max length of user input."},
    )
    generation_config: GenerationConfig = field(
        default=None,
        metadata={"help": "The generation config."},
    )
    test_generation_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The batch size for generation during evaluation."},
    )
    train_generation_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "The batch size for generation during training."},
    )
    print_out_examples: Optional[int] = field(
        default=10,
        metadata={"help": "Print out examples every n steps."},
    )
    mini_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The mini batch size Rec Training."},
    )
    item_emb_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for embedding item profiles for evaluation."},
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Do sampling when generating user and item reasoning."},
    )
    disable_dropout: Optional[bool] = field(
        default=True,
        metadata={"help": "Disable dropout during training."},
    )
    similarity_type: str = field(
        default="dot",
        metadata={
            "help": "The type of similarity function to use. Options: dot, cosine, L2."},
    )
    similarity_temperature: float = field(
        default=0.02,
        metadata={"help": "The temperature for the similarity function."},
    )
    similarity_normalization: bool = field(
        default=True,
        metadata={
            "help": "Whether to normalize the similarity scores before computing similarity."},
    )
    gather_negs_across_processes: bool = field(
        default=False,
        metadata={"help": "Whether to gather negative samples across processes."},
    )

    use_vllm: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use vllm for generation."},
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )

    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    
    # Beam search配置
    num_beams: int = field(
        default=20,
        metadata={"help": "Number of beams for beam search. Set to 1 to disable beam search."}
    )
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to stop the beam search when at least num_beams sentences are finished per batch or not."}
    )


class GenRecTrainer(Trainer):

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module],
            args: GenRecTrainingArguments,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            **kwargs
    ):

        self.args = args
        self.tokenizer = tokenizer


        if getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            assert hasattr(model, "enable_input_require_grads")
            model.enable_input_require_grads()

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.previous_log = {}

        self.is_peft_model = is_peft_model(model)

        if args.disable_dropout:
            disable_dropout_in_model(model)

        
        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            **kwargs
        )

        if args.use_vllm:
            if self.accelerator.is_main_process:

                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"
                    self.accelerator.print(
                        f"Using vLLM on device {vllm_device}")
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                    # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    print("[LOG][vLLM] step1: 准备初始化vLLM LLM 对象")
                    self.llm = LLM(
                        model='./vllm_model',
                        tokenizer='./vllm_tokenizer',
                        # quantization="bitsandbytes",
                        # load_format="bitsandbytes",
                        dtype='auto',   #torch.bfloat16,
                        # trust_remote_code=True,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        # dtype=self.args.vllm_dtype,
                        # enable_prefix_caching=True,
                        device=vllm_device,
                        max_model_len=4096,
                        # When release by vLLM, we would be able to distribute the model on multiple GPUs
                        # See https://github.com/vllm-project/vllm/pull/12071
                        # tensor_parallel_size=torch.cuda.device_count(),
                        # distributed_executor_backend="external_launcher",
                        # tensor_parallel_size=1,
                        # pipeline_parallel_size=1,
                        # # v0引擎的参数
                        # use_v2_block_manager=False,
                        # enforce_eager=True,
                        # disable_custom_all_reduce=True,
                    )
                    
                    
                    vllm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    original_forward = vllm_model.lm_head.forward
                    
                    def float32_forward(self, *args, **kwargs):
                        output = original_forward(*args, **kwargs)
                        return output.to(torch.float32)
                    

                    vllm_model.lm_head.forward = float32_forward.__get__(vllm_model.lm_head, vllm_model.lm_head.__class__)


            self.vllm_sampling_params = SamplingParams(
                temperature=self.args.generation_config.temperature,
                max_tokens=self.args.generation_config.max_new_tokens,
                top_k=self.args.generation_config.top_k,
                top_p=self.args.generation_config.top_p,
                repetition_penalty=self.args.generation_config.repetition_penalty,
                best_of=20,
                # seed=self.args.seed,
                n=self.args.generation_config.num_return_sequences,
                include_stop_str_in_output=True,
                stop=[self.args.emb_end_token] if self.args.emb_end_token else None,
            )

            self.vllm_no_sampling_params = SamplingParams(
                temperature=0,
                max_tokens=self.args.generation_config.max_new_tokens,
                repetition_penalty=self.args.generation_config.repetition_penalty,

                include_stop_str_in_output=True,
                stop=[self.args.emb_end_token] if self.args.emb_end_token else None,
            )
            rich.print("[red]VLLM sampling params[/red]")
            rich.print(self.vllm_sampling_params)

            self._last_loaded_step = 0  # tag to avoid useless loading during grad acc
            self.accelerator.wait_for_everyone()
            print('*'*20 + ' VLLM initialized' + '*'*20)
            self._move_model_to_vllm()
            print('*'*20 + ' VLLM moved to vllm' + '*'*20)

    def get_model_for_eval(self, ):
        assert not self.args.jit_mode_eval, "JIT mode is not supported for evaluation."
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        model = self.accelerator.unwrap_model(self.model)

        # while ``train`` is running, cast it to the right dtype first and then put on device
        args = self.args
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        if args.past_index >= 0:
            self._past = None

        return model

    def _gather_and_cat(self, tensor):
        """
        Gathers tensors and concatenates them on every process.
        This is needed for the multi-processing case.

        Args:
            tensor (torch.Tensor): Tensor to gather and concatenate.

        Returns:
            torch.Tensor: Concatenated tensor.
        https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/bitsandbytes_loader.html#vllm.model_executor.model_loader.bitsandbytes_loader.BitsAndBytesModelLoader.load_weights
        """
        num_processes = self.accelerator.num_processes
        tmp_tensor = tensor.detach()
        gathered_tensor = self.accelerator.gather_for_metrics(
            [tmp_tensor], use_gather_object=True)
        current_process = self.accelerator.process_index
        indices_to_take = [i for i in range(
            num_processes) if i != current_process]
        gathered_tensor = [gathered_tensor[i].to(
            tensor.device) for i in indices_to_take]
        gathered_tensor = torch.cat([tensor, *gathered_tensor], dim=0)

        assert gathered_tensor.shape[0] == num_processes * \
            tensor.shape[0], f"{gathered_tensor.shape[0]} != {num_processes * tensor.shape[0]}"
        return gathered_tensor

    def _move_model_to_vllm(self):
        """
        Move the model to vllm
        """
        
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=True
        ) as unwrapped_model:
            
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
            
                state_dict = self.merge_weights(state_dict)
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

                

                llm_model.load_weights(state_dict.items())
                
    
    def merge_weights(self, state_dict):
        new_dict = {}
        layers = set()
        
        for key in state_dict:
            if 'model.layers.' in key:
                layer = key.split('.')[2]
                layers.add(int(layer))
        
        for layer in layers:
            q_key = f'model.layers.{layer}.self_attn.q_proj.weight'
            k_key = f'model.layers.{layer}.self_attn.k_proj.weight'
            v_key = f'model.layers.{layer}.self_attn.v_proj.weight'
            
            if all(k in state_dict for k in [q_key, k_key, v_key]):
                qkv = torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0)
                new_dict[f'model.layers.{layer}.self_attn.qkv_proj.weight'] = qkv
                # if hasattr(self, 'llm') and self.llm is not None:
                #     try:
                #         vllm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                #         vllm_params_dict = dict(vllm_model.named_parameters())
                        
                #         # 计算清理后的参数名（去掉base_model.model.前缀）
                #         clean_name = f'model.layers.{layer}.self_attn.qkv_proj.weight'
                        
                #         if clean_name in vllm_params_dict:
                #             vllm_weight = vllm_params_dict[clean_name]
                            
                #             # 直接对比merged_fp与vLLM中的权重
                #             are_equal = torch.equal(qkv.to(vllm_weight.device), vllm_weight)
                #             max_diff = torch.abs(qkv.to(vllm_weight.device) - vllm_weight).max().item()
                            
                #             print(f"  🔍 与vLLM权重对比 {clean_name}:")
                #             print(f"      完全相等: {are_equal}")
                #             print(f"      最大差异: {max_diff:.2e}")
                            
                #             if not are_equal:
                #                 print(f"      ⚠️  merged_fp与vLLM权重不匹配！")
                #             else:
                #                 print(f"      ✅ merged_fp与vLLM权重完全匹配")
                #         else:
                #             print(f"  ❌ vLLM中未找到对应权重: {clean_name}")
                #     except Exception as e:
                #         print(f"  ⚠️  对比vLLM权重时出错: {e}")
            
            # Gate-Up合并
            gate_key = f'model.layers.{layer}.mlp.gate_proj.weight'
            up_key = f'model.layers.{layer}.mlp.up_proj.weight'
            
            if all(k in state_dict for k in [gate_key, up_key]):
                gate_up = torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0)
                new_dict[f'model.layers.{layer}.mlp.gate_up_proj.weight'] = gate_up
        
        # 复制其他权重
        skip_keys = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'gate_proj.weight', 'up_proj.weight']
        for key, value in state_dict.items():
            if not any(skip in key for skip in skip_keys):
                new_dict[key] = value
        
        return new_dict
    
    def _batch_generate(self,
                        model,
                        batch: Dict[str, torch.LongTensor],
                        input_ids_key="input_ids",
                        attn_mask_key="attention_mask",
                        do_sample=True,
                        num_return_sequences=None,
                        **kwargs
                        ) -> List[List[str]]:
        """
        Generate sequences for a batch of inputs.

        Args:
            model: The model to use for generation.
            batch: The batch of inputs.
            input_ids_key: The key in the batch dict for the input ids.
            attn_mask_key: The key in the batch dict for the attention mask.
            do_sample: Whether to sample or not, if True, num_return_sequences == 1.
            num_return_sequences: The number of sequences to return, if sampling.

        Returns:
            A list of lists of generated sequences,
            the outer list length is batch size,
            the inner list length is num_return_sequences.
        """

        # 创建自定义停止条件
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class EndTokenStoppingCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_tokens):
                self.tokenizer = tokenizer
                self.stop_token_ids = []
                for token in stop_tokens:
                    if token:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id:
                            self.stop_token_ids.append(token_id)
            
            def __call__(self, input_ids, scores, **kwargs):
                # 检查是否包含停止token
                for stop_id in self.stop_token_ids:
                    if stop_id in input_ids[0][-10:]:  # 检查最后10个token
                        return True
                return False
        
        # 设置停止条件
        stop_tokens = []
        if hasattr(self.args, 'emb_end_token') and self.args.emb_end_token:
            stop_tokens.append(self.args.emb_end_token)
        
        stopping_criteria = StoppingCriteriaList()
        if stop_tokens:
            stopping_criteria.append(EndTokenStoppingCriteria(self.processing_class, stop_tokens))

        ctx = amp.autocast(
            "cuda") if self.args.bf16 else nullcontext()
        
        with ctx:
            kwargs.pop("cache_position", None)
            if do_sample:
                assert num_return_sequences
                output = model.generate(
                    input_ids=batch[input_ids_key],
                    attention_mask=batch[attn_mask_key],
                    do_sample=True,
                    pad_token_id=self.processing_class.pad_token_id,
                    generation_config=self.args.generation_config,
                    num_return_sequences=num_return_sequences,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )
            else:
                output = model.generate(
                    input_ids=batch[input_ids_key],
                    attention_mask=batch[attn_mask_key],
                    do_sample=False,
                    pad_token_id=self.processing_class.pad_token_id,    
                    generation_config=self.args.generation_config,
                    num_return_sequences=1,
                    stopping_criteria=stopping_criteria,
                    **kwargs,
                )
            

        # output = output[:, batch[input_ids_key].shape[1]:]
        # output_decoded = self.processing_class.batch_decode(output[:, batch[input_ids_key].shape[1]:],
        #                                                     skip_special_tokens=True)
        output_decoded = output[:, batch[input_ids_key].shape[1]:]
        n_samples = num_return_sequences if do_sample else 1
        output_list = []
        batch_size = batch[input_ids_key].shape[0]
        for i in range(batch_size):
            output_list.append(
                output_decoded[i * n_samples:(i + 1) * n_samples])
        assert len(output_list) == batch_size
        return output_list
    
    def _generate_in_train(self, model, batch,):
        train_data_ids = batch["input_ids"]
        full_batch_size = batch['input_ids'].shape[0]
        num_processes = self.accelerator.num_processes
        
        generation_samples = self.args.generation_config.num_return_sequences
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            prompts = batch['input_ids']

            # gathered_prompts = self.accelerator.gather_for_metrics(
            #     [prompts], use_gather_object=True)
            gathered_prompts = [None for _ in range(num_processes)]
            dist.all_gather_object(gathered_prompts, prompts)
            # 过滤掉空的
            gathered_prompts = [p for p in gathered_prompts if p is not None and len(p) > 0]

            
            user_results = []
            if self.accelerator.is_main_process:
                # 诊断：检查收集到的prompt信息
                batch_sizes = [len(p) for p in gathered_prompts]
                total_prompts_collected = sum(batch_sizes)
                
                prompts = [prompt for prompts_ in gathered_prompts for prompt in prompts_]
                tokens_prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]
                
                if generation_samples == 1:
                    outputs = self.llm.generate(
                        tokens_prompts, self.vllm_no_sampling_params, use_tqdm=False)
                else:
                    #print(f"[DEBUG] temperature: {self.vllm_sampling_params.temperature}")
                    outputs = self.llm.generate(
                        tokens_prompts, self.vllm_sampling_params, use_tqdm=False)
                
                user_results = [[output.outputs[i].token_ids for i in range(
                    self.args.generation_config.num_return_sequences)] for output in outputs]
            
            # distribute back to each device
            user_results = self.accelerator.gather_for_metrics(
                [user_results], use_gather_object=True)[0]
            local_batch_size = batch['input_ids'].shape[0]
            all_batch_sizes = self.accelerator.gather_for_metrics([local_batch_size], use_gather_object=True)
            assert len(user_results) == full_batch_size * num_processes

            # calculate average lens
            _lens = [len(result) for results in user_results for result in results]
            self.store_metrics({'output_len': sum(_lens) / len(_lens)})

            # calculate average lens
            _lens = [len(result) for results in user_results for result in results]
            self.store_metrics({'output_len': sum(_lens) / len(_lens)})

            current_process = self.accelerator.process_index
            user_results = [
                user_results[full_batch_size * current_process + i] for i in range(full_batch_size)]
            assert len(user_results) == full_batch_size
            

        else:
            def _iterator(mini_batch_size):
                for i in range(0, full_batch_size, mini_batch_size):
                    yield {
                        "input_ids": batch["input_ids"][i:i + mini_batch_size],
                        "attention_mask": batch["attention_mask"][i:i + mini_batch_size], }

            user_results = []
            pbar = _iterator(self.args.train_generation_batch_size)
            for mini_batch in pbar:
                result = self._batch_generate(model, mini_batch,
                                              input_ids_key="input_ids",
                                              attn_mask_key="attention_mask",
                                              do_sample=False if generation_samples == 1 else True,
                                              num_return_sequences=generation_samples,
                                              )
                user_results.extend(result)
        # print(len(user_results), len(user_results[0]))
        # augmented_input = []
        # for i in range(full_batch_size):
        #     augmented_input.append(
        #         self.train_dataset.get_with_profiles(
        #             train_data_ids[i],
        #             user_results[i],
        #         )
        #     )
        
        
        augmented_input = self.data_collator.torch_call(batch | {"reasoning": user_results})
        for k in augmented_input:
            if isinstance(augmented_input[k], torch.Tensor):
                augmented_input[k] = augmented_input[k].to(
                    self.accelerator.device)

        # if self.state.global_step % self.args.print_out_examples == 0:
        #     if self.accelerator.is_main_process:
        #         # rich.print(self.processing_class.decode(
        #         #     batch["user_gen_input_ids"][0], skip_special_tokens=False))
        #         examples = self.processing_class.batch_decode(
        #             augmented_input['input_ids'][0:2], skip_special_tokens=False)
        #         examples = [e.replace(self.processing_class.pad_token, '')
        #                     for e in examples]
        #         rich.print(examples)
        return augmented_input

    def ref_kl(self, pi_logprob, pi_ref_logprob):
        #return pi_ref_logprob.exp() / pi_logprob.exp()- (pi_ref_logprob - pi_logprob) - 1
        # pi_logprob, pi_ref_logprob: [batch, seq, vocab]
        pi_ref_prob = pi_ref_logprob.exp()
        #print(pi_logprob.shape, pi_ref_logprob.shape, (pi_ref_prob * (pi_ref_logprob - pi_logprob)).shape)
        return (pi_ref_prob * (pi_ref_logprob - pi_logprob)).sum(dim=-1)  # [batch, seq]

    def _efficient_forward(self,
                           model: nn.Module,
                           batch: Dict[str, Union[List, torch.LongTensor]],
                           return_with_last_hidden_states: bool = False,
                           ):
        model_kwargs = {"return_dict": True,
                        #"return_with_last_hidden_states": return_with_last_hidden_states,
                        "output_hidden_states": return_with_last_hidden_states,
                        }

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["token_mask"]

        # 获取真正的token labels（用于语言建模的目标）

        # loss_mask = labels != -100
        # assert not (labels == -100).all(dim=-
        #                                 1).any().item(), "All labels are -100 in one data point"

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        # for i in range(attention_mask.size(0)):
        #     # if loss_mask is like [0, 0, 1, 1, 1, 1, 0, 0, 0],
        #     # then set the attn mask accordingly (set the back to 0)
        #     last_one_idx = torch.nonzero(loss_mask[i]).max().item()
        #     attention_mask[i][last_one_idx + 1:] = 0
        #     first_one_idx = torch.nonzero(attention_mask[i])[0].item()
        #     input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
        #     attention_mask[i] = torch.roll(
        #         attention_mask[i], shifts=-first_one_idx)
        #     loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)


        # Get the first column idx that is all zeros and remove every column after that
        empty_cols = torch.sum(attention_mask, dim=0) == 0  # (seq_len, )
        first_empty_col = torch.nonzero(empty_cols)[0].item(
        ) if empty_cols.any() else attention_mask.size(1)
        input_ids = input_ids[:, :first_empty_col]
        attention_mask = attention_mask[:, :first_empty_col]
        loss_mask = loss_mask[:, :first_empty_col]
        labels = labels[:, :first_empty_col]

        # if self.args.use_logits_to_keep:
        # Compute num_logits_to_keep based on loss_mask pattern:
        # [[0, 0, 0, x, x, x, x],
        #  [0, 0, 0, x, x, x, 0]]
        #         ^ start computing logits from here ([:, -(7-3+1):])
        first_compute_index = loss_mask.nonzero(as_tuple=True)
        first_compute_index = first_compute_index[1]
        if not len(first_compute_index):  # no loss_mask
            num_logits_to_keep = loss_mask.shape[1]
        else:
            first_compute_index = first_compute_index.min()
            num_logits_to_keep = loss_mask.shape[1] - first_compute_index
            num_logits_to_keep = num_logits_to_keep.item()
        # + 1 for the first label
        # model_kwargs["logits_to_keep"] = num_logits_to_keep + 1  # 标准LlamaForCausalLM不支持此参数

        # Align labels with logits
        # logits:    -,  -, [x2, x3, x4, x5, x6]
        #                     ^ --------- ^       after logits[:, :-1, :]
        # labels:   [y0, y1, y2, y3, y4, y5, y6]
        #                         ^ --------- ^   with num_logits_to_keep=4, [:, -4:]
        # loss_mask: [0,  0,  0,  1,  1,  1,  1]
        #print("input_ids.text:", self.processing_class.decode(input_ids[0], skip_special_tokens=False))
        # print("labels.text:", self.processing_class.decode(labels[0][labels[0]!=-100], skip_special_tokens=False))
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **model_kwargs,
        )


        # with torch.no_grad():
        #     ref_outputs = self.ref_model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         use_cache=False  # 明确禁用KV缓存
        #     )
        #     ref_logits = ref_outputs.logits

        # pi_logprob = torch.nn.functional.log_softmax(outputs.logits[:, -num_logits_to_keep:], dim = 2)
        # pi_ref_logprob = torch.nn.functional.log_softmax(ref_logits[:, -num_logits_to_keep:], dim = 2)
        # kl_loss = self.ref_kl(pi_logprob, pi_ref_logprob)
        
        # # 诊断代码：打印kl_loss的真实形状
        # print(f"Shape of kl_loss before masking: {kl_loss.shape}")
        
        # # 通过unsqueeze(-1)确保loss_mask能正确广播
        # kl_loss = (kl_loss*loss_mask[:, -num_logits_to_keep:]).sum(dim=-1)
        # kl_loss /= loss_mask[:, -num_logits_to_keep:].sum(dim=-1)
        # kl_loss = kl_loss.sum()
        # kl_loss /= batch['input_ids'].shape[0]

        # loss_mask[~attention_mask] = 0  # set padding tokens to 0


        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -num_logits_to_keep:]


        # labels = labels[:, 1:]
        # labels = labels[:, -num_logits_to_keep:]

        input_ids = input_ids[:, 1:]
        input_ids = input_ids[:, -num_logits_to_keep:]

        labels = labels[:, 1:]
        labels = labels[:, -num_logits_to_keep:]

        loss_mask = loss_mask[:, 1:]
        loss_mask = loss_mask[:, -num_logits_to_keep:]

        probs = torch.softmax(logits, dim=-1)  # [batch, seq, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq]
        per_token_logps = selective_log_softmax(logits, input_ids)  # (B, seq)
        if return_with_last_hidden_states:
            return per_token_logps, loss_mask, logits, labels, entropy

        return per_token_logps, loss_mask, entropy

    def store_metrics(self, metrics, metric_key_prefix="train"):
        for key, value in metrics.items():
            self._stored_metrics[metric_key_prefix][key].append(value)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        for metric_key_prefix in list(self._stored_metrics):
            # Stored metrics are lists of floats, compute the mean
            for key in self._stored_metrics[metric_key_prefix]:
                mean_val = torch.tensor(
                    self._stored_metrics[metric_key_prefix][key]).mean().item()
                logs[f"{metric_key_prefix}_{key}"] = mean_val

            del self._stored_metrics[metric_key_prefix]
        return super().log(logs, *args, **kwargs)


    def evaluate(self,
                 eval_dataset: Optional[Union[Dataset,
                                              Dict[str, Dataset]]] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval",
                 ) -> Dict[str, float]:
        eval_model = self.get_model_for_eval()
        self._generate_item_embeddings(eval_model)

        assert eval_dataset is None
        os.makedirs(os.path.join(
            self.args.output_dir, 'datasets'), exist_ok=True)

        with torch.no_grad():
            for _set_name in self.eval_dataset.keys():
                dset_cache_dir = os.path.join(
                    self.args.output_dir, 'datasets', f'reasoning_{_set_name}_{self.state.global_step}')

                self.eval_dataset[_set_name] = self._update_eval_set(
                    eval_model, self.eval_dataset[_set_name], prefix="user")

                self.eval_dataset[_set_name].save_to_disk(dset_cache_dir)
                self.accelerator.print(
                    f"{_set_name} dataset saved to {dset_cache_dir}")

                self.eval_dataset[_set_name] = self.user_prompter.convert_dataset(
                    dset=self.eval_dataset[_set_name])

            # calculate profile avg length
                _len = [len(x['user_input_ids']) for x in self.eval_dataset]
                self.store_metrics({'output_length': sum(_len) / len(_len)},
                                   metric_key_prefix=f"{metric_key_prefix}_{_set_name}")

        eval_output = super().evaluate(
            eval_dataset=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        return eval_output

    def _update_eval_set(self, model, eval_dataset, prefix='user'):

        if 'profile' in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns('profile')
        id_key_name = 'interaction_id'

        if self.args.use_vllm:
            if self.accelerator.is_main_process:
                prompts = eval_dataset["user_gen_input_ids"]
                tokens_prompts = [TokensPrompt(
                    prompt_token_ids=prompt) for prompt in prompts]
                outputs = self.llm.generate(
                    tokens_prompts, self.vllm_no_sampling_params)
                outputs = [output.outputs[0].text for output in outputs]
                all_result = {_id: output for _id, output in zip(
                    eval_dataset[id_key_name], outputs)}
            else:
                all_result = {}
            all_result = self.accelerator.gather_for_metrics(
                [all_result], use_gather_object=True)[0]

        else:
            batch_size = self.args.test_generation_batch_size

            data_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=False,
                prefetch_factor=self.args.dataloader_prefetch_factor,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(data_loader)
            item_results = {}

            pbar = enumerate(data_loader)
            if self.accelerator.is_main_process:
                pbar = tqdm(pbar, total=len(data_loader),
                            desc=f"Generating {prefix} reasonings")

            for batch_i, batch in pbar:
                result = self._batch_generate(model, batch,
                                              input_ids_key=f"{prefix}_gen_input_ids",
                                              attn_mask_key=f"{prefix}_gen_attention_mask",
                                              do_sample=False,
                                              )
                for i, _id in enumerate(batch[id_key_name]):
                    item_results[_id] = result[i][0]

            all_result_list = self.accelerator.gather_for_metrics(
                [item_results], use_gather_object=True)
            all_result = {}
            for result in all_result_list:
                all_result.update(result)

        assert len(all_result) == len(eval_dataset)

        def map_into_dset(example):
            _id = example[id_key_name]
            example['profile'] = all_result[_id]
            return example

        eval_dataset = eval_dataset.map(map_into_dset)
        return eval_dataset

    def merge_lora_into_quantized_base(self,unwrapped_model):
        param_dict = dict(unwrapped_model.named_parameters())
        state_dict = {}
        merged_count = 0

        # 读取 LoRA 配置
        lora_config = unwrapped_model.peft_config['default']
        scaling = lora_config.lora_alpha / lora_config.r
        for name, param in param_dict.items():
            if name.endswith("base_layer.weight"):
                prefix = name[:-len("base_layer.weight")]
                a_name = prefix + "lora_A.default.weight"
                b_name = prefix + "lora_B.default.weight"
                
                print(f"🔍 检查 LoRA 参数: {name}")
                print(f"🔍 寻找 A: {a_name}")
                print(f"🔍 寻找 B: {b_name}")
                print(f"🔍 A 存在: {a_name in param_dict}")
                print(f"🔍 B 存在: {b_name in param_dict}")

                if a_name in param_dict and b_name in param_dict:
                    lora_a = param_dict[a_name]
                    lora_b = param_dict[b_name]
                    
                    # 检查 LoRA 参数的实际值
                    print(f"🔍 LoRA_A shape: {lora_a.shape}, max: {lora_a.abs().max().item():.8f}, mean: {lora_a.abs().mean().item():.8f}")
                    print(f"🔍 LoRA_B shape: {lora_b.shape}, max: {lora_b.abs().max().item():.8f}, mean: {lora_b.abs().mean().item():.8f}")
                    print(f"🔍 Scaling factor: {scaling}")
                    
                    # 检查 LoRA 参数是否为零
                    if lora_a.abs().max().item() == 0.0:
                        print(f"❌ LoRA_A 全部为零！")
                    if lora_b.abs().max().item() == 0.0:
                        print(f"❌ LoRA_B 全部为零！")
                    
                    quant_state = getattr(param, "quant_state", None)
                    
                    # 检查量化状态
                    if hasattr(quant_state, 'absmax'):
                        absmax = quant_state.absmax.max().item()
                        quant_precision = absmax / 15.0  # 4位量化有15个级别
                        print(f"🔍 量化参数 - absmax: {absmax:.8f}, 量化精度: {quant_precision:.8f}")
                    
                    # 使用 bitsandbytes 的解量化函数
                    import bitsandbytes.functional as F
                    print(f"🔧 原始量化权重 {name}: shape={param.shape}")
                    
                    # 保存原始形状
                    original_shape = param.shape
                    
                    # 解量化
                    q = Quant4bit(scale=quant_state["scale"], zero_point=quant_state["zero_point"])
                    base_fp = q.dequantize(param)
                    # base_fp = F.dequantize_4bit(param, quant_state)
                    print(f"🔧 解量化后权重 {name}: shape={base_fp.shape}")
                    
                    # 检查是否需要reshape回原始形状
                    if base_fp.shape != original_shape:
                        print(f"⚠️  形状不匹配！原始: {original_shape}, 解量化后: {base_fp.shape}")
                        # 如果形状不匹配，尝试reshape
                        if base_fp.numel() == torch.prod(torch.tensor(original_shape)):
                            base_fp = base_fp.reshape(original_shape)
                            print(f"✅ 已reshape回原始形状: {base_fp.shape}")
                        else:
                            print(f"❌ 无法reshape，元素数量不匹配: {base_fp.numel()} vs {torch.prod(torch.tensor(original_shape))}")
                            raise ValueError(f"Cannot reshape weight {name} from {base_fp.shape} to {original_shape}")
                    
                    print(f"🔧 最终解量化权重 {name}: shape={base_fp.shape}")

                    # 合并 LoRA 增量 - 使用更高精度
                    lora_delta = (lora_b @ lora_a) * scaling
                    print(f"🔍 LoRA delta shape: {lora_delta.shape}, max: {lora_delta.abs().max().item():.8f}, mean: {lora_delta.abs().mean().item():.8f}")
                    
                    # 检查基础权重的范围
                    base_max = base_fp.abs().max().item()
                    base_mean = base_fp.abs().mean().item()
                    print(f"🔍 Base weight range: max={base_max:.8f}, mean={base_mean:.8f}")
                    
                    # 检查相对变化大小
                    if base_max > 0:
                        relative_change = lora_delta.abs().max().item() / base_max
                        print(f"🔍 Relative change: {relative_change:.8f} ({relative_change*100:.6f}%)")
                        
                        # 如果相对变化太小，可能被量化抹掉
                        if relative_change < 1e-6:
                            print(f"⚠️  相对变化过小，可能被量化精度损失！")
                    
                    # 检查 LoRA 变化是否小于量化精度
                    if hasattr(quant_state, 'absmax'):
                        lora_max_change = lora_delta.abs().max().item()
                        if lora_max_change < quant_precision:
                            print(f"⚠️  LoRA 变化 ({lora_max_change:.8f}) 小于量化精度 ({quant_precision:.8f})，会被量化抹掉！")
                        else:
                            print(f"✅ LoRA 变化 ({lora_max_change:.8f}) 大于量化精度 ({quant_precision:.8f})，应该能保留")
                    
                    # 保持原始精度，不强制转换为 float16
                    merged_fp = base_fp + lora_delta.to(base_fp.device).to(base_fp.dtype)

                    # 重新量化
                    merged_q = q.requantize(merged_fp)

                    # 调试：检查合并前后的差异
                    diff = torch.abs(merged_fp - base_fp).max().item()
                    mean_diff = torch.abs(merged_fp - base_fp).mean().item()
                    print(f"🔍 {prefix}weight: LoRA合并差异 max={diff:.8f}, mean={mean_diff:.8f}")
                    
                    # 直接保存合并后的 FP16/BF16 权重，不重新量化
                    # 这样可以避免量化精度损失
                    new_name = prefix + "weight"
                    merged_weight = merged_q.to(torch.float16)
                    state_dict[new_name] = merged_weight
                    print(f"🔧 保存权重: {new_name}, shape: {merged_weight.shape}")
                    merged_count += 1
            elif "lora_A" in name or "lora_B" in name:
                # 跳过 LoRA 参数
                continue
            else:
                # 非LoRA、非量化参数直接加入state_dict
                state_dict[name] = param.clone().detach()

        print(f"\n🔧 成功合并 {merged_count} 个 LoRA 模块到量化参数中")
        return state_dict

class Quant4bit:
    """4位量化工具类"""

    def __init__(self, scale: torch.Tensor, zero_point: torch.Tensor):
        self.scale = scale
        self.zero_point = zero_point

    def dequantize(self, weight_int8: torch.Tensor) -> torch.Tensor:
        flattened = weight_int8.view(-1)
        high = (flattened >> 4) & 0x0F
        low = flattened & 0x0F
        values = torch.cat([low.unsqueeze(1), high.unsqueeze(1)], dim=1).reshape(-1)
        values = values.to(self.scale.dtype)
        return (values - self.zero_point) * self.scale

    def requantize(self, weight_fp: torch.Tensor) -> torch.Tensor:
        quantized = torch.clamp((weight_fp / self.scale + self.zero_point).round(), 0, 15).to(torch.uint8)
        if quantized.numel() % 2 != 0:
            quantized = torch.cat([quantized, torch.zeros(1, dtype=quantized.dtype, device=quantized.device)])
        low = quantized[0::2]
        high = quantized[1::2]
        packed = (high << 4) | low
        return packed.view(-1)