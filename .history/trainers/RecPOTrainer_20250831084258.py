from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Union, List, Any
import datasets
import torch
from torch import nn
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback
from transformers.training_args import OptimizerNames
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import is_sagemaker_mp_enabled, logging
from trainers.GRecTrainer import GenRecTrainingArguments, GenRecTrainer
from trl.trainer.utils import selective_log_softmax
import wandb
import weave
logger = logging.get_logger(__name__)

if is_sagemaker_mp_enabled():
    raise ImportError(
        "SageMaker Model Parallelism is not supported by this example.")


@dataclass
class RecPOTrainingArguments(GenRecTrainingArguments):

    epsilon_low: Optional[float] = field(
        default=0.2,
        metadata={"help": " Epsilon value for clipping."},
    )
    epsilon_high: Optional[float] = field(
        default=0.28,
        metadata={"help": " Epsilon value for clipping."},
    )
    reward_type: Optional[str] = field(
        default="mix",
        metadata={"help": "The reward type."},
    )
    advantage_type: Optional[str] = field(
        default="gaussian",
        metadata={
            "help": "The advantage type, either 'leave-one-out' or 'gaussian'."},
    )
    reward_ndcg_k: Optional[int] = field(
        default=1000,
        metadata={"help": "The k value for ndcg@k."},
    )
    reward_softmax_weight: Optional[float] = field(
        default=0.05,
        metadata={"help": "The weight for softmax loss."},
    )
    relabel_topk: Optional[int] = field(
        default=1,
        metadata={"help": "The k value for topk."},
    )
    wandb_name: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run name. If not provided, will use auto_run_rank{rank}."},
    )


class RecPOTrainer(GenRecTrainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        args: RecPOTrainingArguments = None,
        **kwargs
    ):

        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            **kwargs
        )
        self.tokenizer = tokenizer
        self.args.mini_batch_size = min(self.args.per_device_train_batch_size *
                                        self.args.generation_config.num_return_sequences,
                                        self.args.mini_batch_size)
        
        self.original_batch_size = self.args.per_device_train_batch_size
        
        if not wandb.run:
            import json
            import os
            def filter_jsonable(d):
                result = {}
                for k, v in d.items():
                    try:
                        json.dumps(v)
                        result[k] = v
                    except Exception:
                        continue
                return result
            config_dict = vars(self.args) if hasattr(self, 'args') else {}
            config_dict = filter_jsonable(config_dict)
            rank = os.environ.get("RANK", "0")
            
            wandb_name = getattr(self.args, 'wandb_name', None)
            if wandb_name is None:
                wandb_name = f'auto_run_rank{rank}'
            else:
                wandb_name = f'{wandb_name}_rank{rank}'

            wandb.init(project='Think2go', name=wandb_name, config=config_dict)
                
            # wandb.init(project='Think2go', id='ykxyptym', resume='allow', name=wandb_name, config=config_dict)
        

    def compute_rec_score(self,
                          model,
                          inputs,
                          similarity: Optional[torch.FloatTensor] = None,
                          train_eval='train'):
        if train_eval == 'train':
            num_samples = self.args.num_return_sequences
        else:
            num_samples = 1
        seq_labels = inputs['labels']
        seq_input_ids = inputs['pred_ids']
        batch_size = seq_labels.shape[0]//num_samples
        seq_labels = seq_labels.view(
            batch_size, num_samples, -1)
        seq_input_ids = seq_input_ids.view(
            batch_size, num_samples, -1)#.expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)

        rewards = inputs['rewards'] #+ rewards_length
        acc = inputs['acc']

        if self.args.advantage_type == 'gaussian':
            if num_samples == 1:
                advantages = torch.zeros_like(rewards)
            else:
                _mean = rewards.mean(dim=1, keepdim=True)
                _std = rewards.std(dim=1, keepdim=True) + 1e-8
                advantages = (rewards - _mean) #/ _std  # (B, num_samples)
                # advantages = rewards > _mean
                # advantages = advantages.float()
        elif self.args.advantage_type == 'leave-one-out':
            _mean = rewards.mean(dim=1, keepdim=True)
            # advantages = rewards - (_mean - rewards/num_samples)
            advantages = rewards * (1 + 1 / num_samples) - _mean
        else:
            raise NotImplementedError
        advantage = advantages

        row_all_same = ((rewards.max(dim=1).values - rewards.min(dim=1).values) == 0)  # [batch_size]
        if row_all_same.any():
            advantage[row_all_same, :] = 0

        topk = 1   #min(self.args.relabel_topk, advantage.shape[-1])
        _, topk_indices = advantage.topk(topk, dim=-1)
        relabel_mask = torch.zeros_like(advantage, dtype=torch.bool)

        row_all_zero = (advantage == 0).all(dim=-1)  # [batch_size]，True表示该行全为0


        for i, is_all_zero in enumerate(row_all_zero):
            if not is_all_zero:
                relabel_mask[i].scatter_(-1, topk_indices[i], True)
            else:
                relabel_mask[i, 0] = True 
        

        in_batch_labels = torch.arange(batch_size).repeat_interleave(
            num_samples).to(seq_input_ids.device)
        # e.g., num_samples=4, batch_size=8
        # 0,0,0,0,1,1,1,1...7,7,7,7

        result = {
            'acc': acc.view(-1),
            'rewards': rewards.view(-1),
            'advantages': advantage.view(-1),
            'relabel_mask': relabel_mask.view(-1),
            'seq_input_ids': seq_input_ids,
            'in_batch_labels': in_batch_labels,
        }

        self.store_metrics(
            {
                f"acc@1": acc.mean().item(),
                f"acc@1_std": acc.std(dim=1).mean().item(),
                "reward": rewards.mean().item(),
                "reward_std": rewards.std(dim=1).mean().item(),
                "reward_max": rewards.max().item(),
                "reward_min": rewards.min().item(),
                "advantage_min": advantage.min().item(),
                "advantage_max": advantage.max().item(),
                "advantage": advantage.mean().item(),
            },
            metric_key_prefix=train_eval,
        )

        return result


    @staticmethod
    def _rec_mini_batch_iterator(inputs, macro_batch_size, mini_batch_size):
        keys = ['input_ids', 'attention_mask', 'labels', 'token_mask']
        keys += ['acc', 'rewards', 'advantages', 'in_batch_labels', 'relabel_mask', 'time_difficulties', 'spatial_difficulties']

        def _iterator():
            for i in range(0, macro_batch_size, mini_batch_size):
                __start = i
                __end = i + mini_batch_size
                result = {}
                #         {'labels': inputs['labels'],
                #           'input_ids': inputs['input_ids'],
                #           'attention_mask': inputs['attention_mask'],
                #           }

                for key in keys:
                    original_in = inputs[key][__start:__end]
                    result[key] = original_in
                yield result

        return _iterator()

    def training_step(self,
                      model: nn.Module,
                      inputs: Dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None,
                      **kwargs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """
            
        inputs_origin = self._prepare_inputs(inputs)
        eval_model = self.get_model_for_eval()
        oom = False
        try:
            with torch.no_grad():
                # if self.item_hs is None:
                #     self.generate_all_item_embeddings(eval_model)
                inputs = self._generate_in_train(eval_model, inputs_origin)
                current_epoch = self.state.epoch if self.state.epoch is not None else 0
                if current_epoch > 10.0:
                    rewards = inputs['rewards']  # [batch_size, num_samples]
                    batch_size = rewards.shape[0]
                    keep_indices = []
                    should_resample = torch.tensor(0, device='cuda')
                    base_temperature = self.vllm_sampling_params.temperature  
                    temperature_step = 0.5  
                    for i in range(batch_size):
                        for j in range(1):
                            
                            local_need_resample = torch.tensor(
                                bool(torch.all(inputs['rewards'][i] == inputs['rewards'][i][0])),
                                device=inputs['rewards'].device,
                                dtype=torch.int
                            )
                            
                            global_need_resample = self.accelerator.reduce(local_need_resample, reduction='sum')
                            if global_need_resample.item() > 0:
                                new_temp = base_temperature + (j+1) * temperature_step
                                self.vllm_sampling_params.temperature = new_temp

                                torch.cuda.empty_cache()
                                inputs = self._generate_in_train(eval_model, inputs_origin)
                            
                            else:
                                break
                    self.vllm_sampling_params.temperature = base_temperature
                inputs |= self.compute_rec_score(eval_model, inputs)

            del eval_model
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(
                    f"{self.accelerator.device}: OOM when update dset: \n {str(e)}")
                oom = True
                torch.cuda.empty_cache()
            else:
                raise e

        torch.cuda.empty_cache()

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        macro_batch_size = inputs["input_ids"].shape[0]

        user_losses = []

        user_mini_batch_size = self.args.mini_batch_size
        num_user_mini_batches = macro_batch_size // user_mini_batch_size + \
            macro_batch_size % user_mini_batch_size
        user_iterator = self._rec_mini_batch_iterator(
            inputs, macro_batch_size, user_mini_batch_size)

        for i in range(num_user_mini_batches):
            try:
                _inputs = next(user_iterator)
                loss = self.batch_forward(model, _inputs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    _progress = f"{i}/{num_user_mini_batches}"
                    print(
                        f"{self.accelerator.device}: OOM when compute loss on {_progress} mini batches")
                    torch.cuda.empty_cache()
                    loss = torch.tensor(0.0, requires_grad=True).to(
                        self.args.device)
                else:
                    raise e

            kwargs = {}
            torch.cuda.empty_cache()

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                self.accelerator.print("WARNING: ???")
                loss = loss.mean()  # mean() to average on multi-gpu parallel training


            try:
                self.accelerator.backward(loss, **kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    _progress = f"{i}/{num_user_mini_batches}"
                    print(
                        f"{self.accelerator.device}: OOM when backward loss on {_progress} mini batches")
                    model.zero_grad()
                else:
                    raise e
            loss = loss.detach()
            user_losses.append(loss)

        torch.cuda.empty_cache()

        return torch.stack(user_losses).mean()
    
    def grpo_kl(self, pi_logprob, pi_ref_logprob):
        return pi_ref_logprob.exp() / pi_logprob.exp()- (pi_ref_logprob - pi_logprob) - 1
    

    def batch_forward(self, model, batch):
        current_epoch = self.state.epoch if self.state.epoch is not None else 0
        total_epochs = self.args.num_train_epochs
        
        num_samples = self.args.num_return_sequences
        seq_len = batch["input_ids"].shape[-1]
        advantages = batch["advantages"].unsqueeze(-1)  # (B, 1)
        rewards = batch["rewards"].unsqueeze(-1)
        time_difficulties = batch["time_difficulties"].unsqueeze(-1)
        spatial_difficulties = batch["spatial_difficulties"].unsqueeze(-1)

        _batch_size = advantages.shape[0]
        relabel_mask = batch["relabel_mask"]
        per_token_logps, loss_mask, logits, labels, entropy = self._efficient_forward(
            model,
            batch,
            return_with_last_hidden_states=True,
        )

        

        coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
        coef_2 = torch.clamp(coef_1,
                             1 - self.args.epsilon_low,
                             1 + self.args.epsilon_high)
        
        alpha = 0.4
        kappa = 5
        psi_Ht = torch.min(alpha * entropy.detach(), torch.abs(advantages)/kappa)

        
        shaped_advantages = advantages + psi_Ht

        norm_rewards = 1 - ((4.5 - rewards) / 4.5)
        gamma = 1.0

        gamma = 0.5 * (0.5 * time_difficulties + 0.5 * spatial_difficulties)


        shaped_advantages = torch.exp(gamma norm_rewards) * shaped_advantages

        # shift = 1.5
        # gamma = 3.0
        # norm_rewards = (rewards - shift) * gamma
        # shaped_advantages = torch.sigmoid(norm_rewards) * shaped_advantages

        # shaped_advantages_min_mask = (psi_Ht > shaped_advantages)
        # if torch.any(shaped_advantages_min_mask):
        #     print("[DEBUG] psi_Ht大于shaped_advantages的样本:")
        #     print("psi_Ht:", psi_Ht[shaped_advantages_min_mask])
        #     print("shaped_advantages:", shaped_advantages[shaped_advantages_min_mask])


        # norm_rewards = (4.5 - rewards.detach()) / 4.5
        # shaped_advantages = torch.exp(rewards.detach()) * shaped_advantages
        
        # print("alpha * entropy.detach():", alpha * entropy.detach())
        # print("torch.abs(advantages)*kappa:", torch.abs(advantages)*kappa)
        print("rewards:", rewards)
        print("advantages:", advantages)
        #print("shaped_advantages:", shaped_advantages)
        # print("shaped_advantages:", shaped_advantages)
        # print("psi_Ht:", psi_Ht)
        # print("norm_rewards:", norm_rewards)
        # print("entropy:", entropy)

        acc_log = batch["acc"].mean()
        entropy_log = entropy.mean()
        rewards_log = rewards.mean()
        advantages_log = advantages.mean()
        # print(f"[TEST] advantages_log: {advantages_log} (abs={advantages_log.abs()})")
        shaped_advantages_log = shaped_advantages.mean()

        #print("advantages:", advantages)
        per_token_loss1 = coef_1 * shaped_advantages
        per_token_loss2 = coef_2 * shaped_advantages
        mask_seq_len = per_token_logps.shape[-1]
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        tokens_loss = (per_token_loss.reshape(-1, num_samples*mask_seq_len) * loss_mask.reshape(-1, num_samples*mask_seq_len)).sum(dim=-1)
        #print("tokens_loss0:", tokens_loss)
        #print("loss_mask:", loss_mask.reshape(-1, num_samples*mask_seq_len).sum(dim=-1))
        tokens_loss /= loss_mask.reshape(-1, num_samples*mask_seq_len).sum(dim=-1)
        #print("tokens_loss1:", tokens_loss)
        tokens_loss = tokens_loss.sum()
        tokens_loss /= _batch_size
        # print("tokens_loss2:", tokens_loss)
   
        # # ✅ 调试：检查tokens_loss的梯度链
        # print(f"[DEBUG] per_token_logps.requires_grad: {per_token_logps.requires_grad}")
        # print(f"[DEBUG] advantages.requires_grad: {advantages.requires_grad}")
        # print(f"[DEBUG] tokens_loss.requires_grad: {tokens_loss.requires_grad}")
        # print(f"[DEBUG] tokens_loss.grad_fn: {tokens_loss.grad_fn}")


        # labels = batch['in_batch_labels']
        # num_processes = self.accelerator.num_processes
        # if self.args.gather_negs_across_processes and num_processes > 1:
        #     # num_samples=4, batch_size=8, mini_batch_size=5
        #     # {0,0,0,0,1},{1,1,1,2,2}...{6,6,6,7,7},{7,7}
        #     # if gather negs, with 3 cudas, we have 3*8 items in total
        #     # num_labels in a mini batch now is 5*3
        #     # {0,0,0,0,1, 0+8,0+8,0+8,0+8,1+8, 0+16,0+16,0+16,0+16,1+16}
        #     _macro_batch_size = batch['input_ids'].shape[0]
        #     # turn {0,0} -> {0,0,0+8,0+8,0+16,0+16}
        #     labels = torch.cat([
        #         labels + i * _macro_batch_size
        #         for i in range(num_processes)
        #     ], dim=0)

        #     _gathered_advantages = self._gather_and_cat(advantages)
        #     relabel_mask = self._gather_and_cat(relabel_mask)

        #     # assert labels.shape[0] == _batch_size * num_processes
        #     # assert labels.shape[0] == similarity.shape[0]
        # else:
        #     _gathered_advantages = advantages

        # _gathered_advantages = _gathered_advantages * relabel_mask.float().unsqueeze(
        #     -1)
        # assert _gathered_advantages.shape[0] == labels.shape[
        #     0], f"{_gathered_advantages.shape} {labels.shape}"
        # assert _gathered_advantages.shape[1] == 1, f"{_gathered_advantages.shape}"
        

        if current_epoch < 10: 

            # 1. 计算每个 token 的损失，需要设置 reduction='none'
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            item_loss = loss_fct(
                logits.reshape(-1, logits.shape[-1]), 
                labels.reshape(-1)
            )#.reshape(logits.shape[0], logits.shape[1])
            
            # # 2. 应用 token 级别的 loss_mask，并计算出每个样本的平均损失
            # coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
            # coef_2 = torch.clamp(coef_1,
            #                  1 - self.args.epsilon_low,
            #                  1 + self.args.epsilon_high)
            # item_loss1 = coef_1 * shaped_advantages
            # item_loss2 = coef_2 * shaped_advantages
            # item_loss = -torch.min(item_loss1, item_loss2)
            # item_loss = item_loss.reshape(-1, mask_seq_len) * (labels.reshape(-1, mask_seq_len)!=-100)
            # # print('[DEBUG]', (labels.reshape(-1, num_samples)!=-100).sum(dim=-1))
            # item_loss = item_loss.sum(dim=-1)/(labels!=-100).reshape(-1, mask_seq_len).sum(dim=-1)
            # item_loss = item_loss.sum()
            # item_loss /= _batch_size

            # print("item_loss:", item_loss)
            # # item_loss = mean_loss_per_sample.sum() / _batch_size

            # print("relabel_mask:", relabel_mask)
            # # 3. 用每个样本的优势(advantage)来加权其损失
            # weighted_loss = mean_loss_per_sample * relabel_mask.float()#.unsqueeze(-1)
            # #print("weighted_loss", weighted_loss)
            # # 4. 只对被 relabel_mask 选中的样本计算平均损失
            # num_selected_samples = relabel_mask.sum(-1)
            # if num_selected_samples != 0:
            #     item_loss = weighted_loss.sum() / num_selected_samples
            # else:
            #     # ✅ 修复：使用torch.tensor保持梯度图
            #     item_loss = torch.tensor(0.0, device=weighted_loss.device, requires_grad=True)
            # 取logits最后一个维度的最大值索引
            predicted_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            
            # 只取loss_mask==1的token进行解码
            labels_masked = labels[0]!= -100
            predicted_token_ids_masked = predicted_token_ids[0][labels_masked]
            labels_masked = labels[0][labels_masked]
            
            predicted_text = self.processing_class.decode(predicted_token_ids_masked, skip_special_tokens=False)
            print(f"\n样本 - Logits预测文本:\n{predicted_text}")
            
            label_text = self.processing_class.decode(labels_masked, skip_special_tokens=False)
            print(f"样本 - 标签文本:\n{label_text}")
            print("item_loss:", item_loss, "tokens_loss:", tokens_loss)
            # # 计算loss
            total_loss = item_loss + 5 * tokens_loss
        else:
            total_loss = 5 * tokens_loss
            item_loss = torch.tensor(0.0, device=tokens_loss.device, requires_grad=False)

        # # 统计entropy分段分布
        # # 假设entropy是一个tensor，形状为[batch_size, seq_len]或[batch_size, ...]
        # entropy_flat = entropy.detach().cpu().view(-1)
        # # 设定分段区间
        # bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 100]
        # hist = torch.histc(entropy_flat, bins=len(bins)-1, min=bins[0], max=bins[-1])
        # # 构造分布统计dict
        # entropy_hist_dict = {}
        # for i in range(len(bins)-1):
        #     key = f"entropy_{bins[i]}_{bins[i+1]}"
        #     entropy_hist_dict[key] = int(hist[i].item())

        # # 累积统计entropy分布
        # if not hasattr(self, '_entropy_hist_accum'):  # 初始化
        #     self._entropy_hist_accum = {f"entropy_{bins[i]}_{bins[i+1]}": 0 for i in range(len(bins)-1)}
        # for i in range(len(bins)-1):
        #     key = f"entropy_{bins[i]}_{bins[i+1]}"
        #     self._entropy_hist_accum[key] += int(hist[i].item())

        # 获取当前step
        current_step = int(self.state.global_step) if hasattr(self.state, 'global_step') and self.state.global_step is not None else None
        # 每个batch还是log当前batch分布
        gen_len = None
        if hasattr(self, '_stored_metrics') and 'train' in self._stored_metrics and 'output_len' in self._stored_metrics['train']:
            if len(self._stored_metrics['train']['output_len']) > 0:
                gen_len = self._stored_metrics['train']['output_len'][-1]
        
        # 保存熵的均值到_stored_metrics
        if hasattr(self, '_stored_metrics'):
            if 'train' not in self._stored_metrics:
                self._stored_metrics['train'] = {}
            if 'entropy_mean' not in self._stored_metrics['train']:
                self._stored_metrics['train']['entropy_mean'] = []
            self._stored_metrics['train']['entropy_mean'].append(entropy_log.item() if hasattr(entropy_log, 'item') else entropy_log)
        wandb.log({
            "acc": acc_log.item() if hasattr(acc_log, 'item') else acc_log,
            "entropy": entropy_log.item() if hasattr(entropy_log, 'item') else entropy_log,
            "advantages": advantages_log.item() if hasattr(advantages_log, 'item') else advantages_log,
            "shaped_advantages": shaped_advantages_log.item() if hasattr(shaped_advantages_log, 'item') else shaped_advantages_log,
            "tokens_loss": tokens_loss.item() if hasattr(tokens_loss, 'item') else tokens_loss,
            "item_loss": item_loss.item() if hasattr(item_loss, 'item') else item_loss,
            "total_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
            "rewards": rewards_log.item() if hasattr(rewards_log, 'item') else rewards_log,
            "gen_len": gen_len,
            "epoch": current_epoch,
            # **entropy_hist_dict
        }, step=current_step)
        return total_loss

    def on_epoch_end(self):
        # 每个epoch结束时log累积分布
        import matplotlib.pyplot as plt
        keys = list(self._entropy_hist_accum.keys())
        values = list(self._entropy_hist_accum.values())
        plt.figure(figsize=(8,4))
        plt.bar(keys, values)
        plt.xlabel('Entropy区间')
        plt.ylabel('累计数量')
        plt.title('累计Entropy分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # 记录epoch和step
        current_epoch = self.state.epoch if hasattr(self.state, 'epoch') and self.state.epoch is not None else 0
        current_step = int(self.state.global_step) if hasattr(self.state, 'global_step') and self.state.global_step is not None else None
        wandb.log({"epoch_entropy_hist_bar": wandb.Image(plt), "epoch": current_epoch}, step=current_step)
        plt.close()
        # 清空累计
        for k in self._entropy_hist_accum:
            self._entropy_hist_accum[k] = 0


    def compute_loss(self, model, inputs, return_outputs=False, **loss_kwargs):
        print("调用了compute_loss函数******")
        # 1. 前向传播，自动支持自定义forward
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
            time_difficulties=inputs.get("time_difficulties"),
            spatial_difficulties=inputs.get("spatial_difficulties"),
            # **{k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask", "labels", "time_difficulties", "spatial_difficulties"]}
        )

        # 2. 取loss
        loss = outputs.loss if hasattr(outputs, "loss") else None

        # 3. 兼容评估/推理时的输出
        if return_outputs:
            return loss, outputs
        else:
            return loss

    def my_compute_metrics(self, predictions, ground_truths):
        """
        极简实现：只保留一个for循环，直接比对语义ID字符串
        predictions: [batch_size, num_samples, seq_len]
        ground_truths: [batch_size, num_samples, seq_len] 或 [batch_size, 2, seq_len]
        """
        import re
        num_samples = predictions.shape[1]
        predictions_flat = predictions.view(-1, predictions.shape[-1])
        ground_truths_flat = ground_truths.view(-1, ground_truths.shape[-1])
        total_samples = predictions_flat.shape[0]
        seq_len = predictions_flat.shape[1]
        

        answer_token = self.processing_class.convert_tokens_to_ids("<answer>:")
        end_tokens = [
            self.processing_class.convert_tokens_to_ids("</answer>"),
            self.processing_class.eos_token_id,
            self.processing_class.pad_token_id if hasattr(self.processing_class, 'pad_token_id') else -100
        ]
        id_pattern = re.compile(r'(<[a-z]_\d+>)')  # 匹配单个语义ID

        acc = [0] * total_samples  # 先全0，后面有有效样本再赋值为1
        reward_soft_format = [0] * total_samples
        reward_strict_format = [0] * total_samples
        for i in range(total_samples):
            gt = ground_truths_flat[i]
            pred = predictions_flat[i]
            pred_text = self.processing_class.decode(pred[pred != -100], skip_special_tokens=False)
            # print("pred_text:", pred_text)
            # 软奖励：<think>、<answer>:、<|end_of_text|>都只出现一次
            for text in ['</think>', '</answer>', '<|end_of_text|>']:
                if pred_text.count(text) == 1 and len(pred_text) > 40:
                    reward_soft_format[i] += 0.5
            if pred_text.count('<answer>:') == 1 and len(pred_text) > 40:
                reward_soft_format[i] += 1.0
            # 严格奖励：完整正则匹配<think>*</think>*<answer>: *</answer><|end_of_text|>
            strict_pattern = re.compile(r'<think>.*?</think>.*?<answer>:.*?</answer><\|end_of_text\|>', re.DOTALL)
            strict_pattern1 = re.compile(r'<answer>:.*?</answer><\|end_of_text\|>', re.DOTALL)
            if strict_pattern.search(pred_text) or strict_pattern1.search(pred_text):
                reward_strict_format[i] += 1.0

            # 判断是否都能找到<answer>:
            gt_idxs = (gt == answer_token).nonzero(as_tuple=True)
            pred_idxs = (pred == answer_token).nonzero(as_tuple=True)
            if len(gt_idxs[0]) == 0 or len(pred_idxs[0]) == 0:
                continue  # 跳过该样本，acc[i]保持为0

            gt_start = gt_idxs[0][0].item() + 1
            pred_start = pred_idxs[0][0].item() + 1  #只取第一个<answer>位置

            def extract_ids(tokens, start):
                ids = []
                for t in tokens[start:]:
                    if t.item() in end_tokens or t.item() == -100:
                        break
                    ids.append(t.item())
                text = self.processing_class.decode(ids, skip_special_tokens=False)
                # 用正则提取所有连续语义ID
                poi_list = id_pattern.findall(text)
                poi = ''.join(poi_list)
                return poi, text
            gt_poi, gt_text = extract_ids(gt, gt_start)
            pred_poi, pred_text = extract_ids(pred, pred_start)
            # 只比较提取出来的语义ID序列
            if gt_poi == pred_poi:
                acc[i] = 1

        acc = torch.tensor(acc, dtype=torch.float32, device=predictions.device).view(predictions.shape[0], predictions.shape[1])
        reward_soft_format = torch.tensor(reward_soft_format, dtype=torch.float32, device=predictions.device).view(predictions.shape[0], predictions.shape[1])
        reward_strict_format = torch.tensor(reward_strict_format, dtype=torch.float32, device=predictions.device).view(predictions.shape[0], predictions.shape[1])
        return acc, reward_soft_format, reward_strict_format
