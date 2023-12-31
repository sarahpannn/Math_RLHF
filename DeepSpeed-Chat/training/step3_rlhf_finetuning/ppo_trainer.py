# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import re
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.actor_tokenizer = self.rlhf_engine.actor_tokenizer
        self.reward_tokenizer = self.rlhf_engine.reward_tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.actor_tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def _generate_sequence(self, prompts, mask, step):
        
        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        
        with torch.no_grad():
            seq = self.actor_model.module.generate(
                input_ids=prompts,
                attention_mask=mask,
                max_new_tokens=self.max_answer_seq_len,
                min_new_tokens=100,
                # max_length=max_min_length,
                pad_token_id=self.actor_tokenizer.pad_token_id,
                synced_gpus=True,
            )
            
            print('just generated')
            # print(seq)
            
        # with torch.no_grad():
        #     seq = self.actor_model.generate(
        #         inputs=prompts,
        #         attention_mask=mask,
        #         max_new_tokens=self.max_answer_seq_len,
        #         max_length=max_min_length,
        #         pad_token_id=self.actor_tokenizer.pad_token_id,
        #         synced_gpus=self.z3_enabled
        #     )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.actor_tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq
    
    def _split_lines(self, text):
        pattern = r'Step \d+:'
        steps = re.split(pattern, text, re.DOTALL)
        return steps

    def _add_cls(self, list_of_text):
        rew_ret_str, act_ret_str = '', ''
        for i, s in enumerate(list_of_text):
            if len(s) == 0:
                continue
            if i == 0:
                rew_ret_str += f"{s}"
                act_ret_str += f"{s}"
            if i > 0:
                if s[-1] == "\n":
                    s = s[:-1]
                if s[-len(self.actor_tokenizer.eos_token):] == self.actor_tokenizer.eos_token:
                    s = s[:-len(self.actor_tokenizer.eos_token)]
                rew_ret_str += f"Step {i}:{s}{self.reward_tokenizer.cls_token}\n"
                act_ret_str += f"Step {i}:{s}{self.actor_tokenizer.eos_token}\n"
        return rew_ret_str, act_ret_str

    def batched_preprocess(self, text):
        if self.args.reward_delivery_method == 2:
            rew_ret_list, act_ret_list = [], []
            for example in text:
                split = self._split_lines(example)
                rew_clsed, act_clsed = self._add_cls(split)
                # print("IN BATCHED_PREPROCESS: ", act_clsed)
                rew_ret_list.append(rew_clsed)
                act_ret_list.append(act_clsed)
            return rew_ret_list, act_ret_list
        else:
            rew_ret_list = []
            for example in text:
                split = self._split_lines(example)
                clsed, _ = self._add_cls(split)
                # print("IN BATCHED_PREPROCESS: ", clsed)
                rew_ret_list.append(clsed)
            return rew_ret_list

    def tokenize_raw(self, prompts, text, actor_clsed=[]): 
        actor_cls_idxs = []
        if self.args.reward_delivery_method == 2:
            actor_prediction_mask = []
            actor_tokenized = self.actor_tokenizer(actor_clsed, return_tensors='pt')
            # print("actor_tokenized: ", actor_tokenized['input_ids'])
            for j in range(len(actor_tokenized['input_ids'])):
                cls_idxs_j = []
                # print("LEN OF INPUT IDS: ", len(actor_tokenized['input_ids'][j]) - 1)
                for i in range(1, len(actor_tokenized['input_ids'][j]) - 1):
                    if actor_tokenized['input_ids'][j][i] == self.actor_tokenizer.eos_token_id and actor_tokenized['input_ids'][j][i + 1] not in [self.actor_tokenizer.eos_token_id, self.actor_tokenizer.bos_token_id]:
                        cls_idxs_j.append(i - (2 * len(cls_idxs_j)) - 1)
                        # print("i: ", i)
                actor_cls_idxs.append(cls_idxs_j)
                
                prediction_mask_j = [0] * len(actor_tokenized['input_ids'][j])
                for k in range(len(actor_tokenized['input_ids'][j])):
                    if k in cls_idxs_j:
                        prediction_mask_j[k] = 1
                actor_prediction_mask.append(prediction_mask_j)
                
        prediction_mask = []
        cls_idxs = []
        tokenized = self.reward_tokenizer(text, return_tensors='pt')
        for j in range(len(tokenized['input_ids'])):
            cls_idxs_j = []
            for i in range(1, len(tokenized['input_ids'][j])):
                if tokenized['input_ids'][j][i] == self.reward_tokenizer.cls_token_id:
                    cls_idxs_j.append(i)
            cls_idxs.append(cls_idxs_j)
            
            prediction_mask_j = [0] * len(tokenized['input_ids'][j])
            for k in range(len(tokenized['input_ids'][j])):
                if k in cls_idxs_j:
                    prediction_mask_j[k] = 1
            prediction_mask.append(prediction_mask_j)
        
        if self.args.reward_delivery_method == 2:
            return tokenized, cls_idxs, prediction_mask, actor_cls_idxs
        
        return tokenized, cls_idxs, prediction_mask
    
    def get_last_token(self, generated_sequence):
        action_mask = torch.zeros_like(generated_sequence)
        for i in range(generated_sequence.shape[0]):
            for j in reversed(range(generated_sequence.shape[1])):
                if generated_sequence[i][j] != self.reward_tokenizer.pad_token_id:
                    action_mask[i][j] = 1
                    break
        return action_mask
    
    def translate(self, prompts, in_text, tokenizer1, tokenizer2):
        raw_text = tokenizer1.batch_decode(in_text)
        decoded_prompts = tokenizer1.batch_decode(prompts)
        if self.args.reward_delivery_method == 2:
            reward_preprocessed, actor_preprocessed = self.batched_preprocess(raw_text)
            tokenized_text, cls_idxs, prediction_mask, actor_cls = self.tokenize_raw(prompts=decoded_prompts,
                                                                      text=reward_preprocessed, 
                                                                    #   tokenizer=tokenizer2,
                                                                      actor_clsed=actor_preprocessed)
            return tokenized_text, cls_idxs, prediction_mask, actor_cls
        else:
            preprocessed_text = self.batched_preprocess(raw_text)
            tokenized_text, cls_idxs, prediction_mask = self.tokenize_raw(prompts=decoded_prompts, 
                                                                      text=preprocessed_text,)
        
            return tokenized_text, cls_idxs, prediction_mask

    def generate_experience(self, prompts, mask, step):
        self.eval()
        seq = self._generate_sequence(prompts, mask, step)
        self.train()
        
        pad_token_id = self.actor_tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            if not self.args.critic_reward_tokenizer_mismatch:
                action_mask = self.get_last_token(seq)
                reward_score = self.reward_model.forward_value(
                    seq, attention_mask, prediction_mask=action_mask,
                    prompt_length=self.prompt_length).detach()
                    
                values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]
                
            else:
                seq_device = seq.device
                if self.args.reward_delivery_method == 2:
                    translated_tokens, cls_idxs, prediction_mask, actor_cls = self.translate(prompts=prompts, in_text=seq, tokenizer1=self.actor_tokenizer, tokenizer2=self.reward_tokenizer)
                    reward_score = self.reward_model.forward_value(
                        translated_tokens['input_ids'].to(seq_device), translated_tokens['attention_mask'].to(seq_device),
                        prediction_mask=prediction_mask)
                else:
                    translated_tokens, cls_idxs, prediction_mask = self.translate(prompts=prompts, in_text=seq, tokenizer1=self.actor_tokenizer, tokenizer2=self.reward_tokenizer)
                    reward_score = self.reward_model.forward_value(
                        translated_tokens['input_ids'].to(seq_device), translated_tokens['attention_mask'].to(seq_device),
                        prediction_mask=prediction_mask).detach()
                # print('ATTENTION MASK: ', translated_tokens['attention_mask'].sum())

                print(reward_score)
                
                values = self.critic_model.forward_value(
                    seq, attention_mask, return_value_only=True).detach()[:, :-1]
                
            logits = output.logits
            logits_ref = output_ref.logits
            
        if self.args.reward_delivery_method == 2:
            return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask,
            "cls_idxs": cls_idxs,
            "actor_cls_idxs": actor_cls
        }
        else:
            return {
                'prompts': prompts,
                'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
                'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
                'value': values,
                'rewards': reward_score,
                'input_ids': seq,
                "attention_mask": attention_mask,
                "cls_idxs": cls_idxs
            }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask, dump=True, cls_idxs=[]):

        if dump:
            kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
            rewards = kl_divergence_estimate
            start = prompts.shape[1] - 1
            ends = start + action_mask[:, start:].sum(1) + 1
            reward_score = reward_score.to(torch.float32)
            reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                    self.clip_reward_value)
            # print("REWARD CLIP: ", reward_clip)
            batch_size = log_probs.shape[0]
            for j in range(batch_size):
                # print('REWARDS: ', rewards[j, start:ends[j]][-1])
                rewards[j, start:ends[j]][-1] += reward_clip[j]

            return rewards
        
        if not dump:
            kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
            rewards = kl_divergence_estimate
            start = prompts.shape[1] - 1
            ends = start + action_mask[:, start:].sum(1) + 1
            
            for i, reward_i in enumerate(reward_score):
                # print('PROMPTS: ', prompts[i])
                # print('PROMPT LEN: ', start)
                # print('CLS IDX BEFORE: ', cls_idxs[i])
                cls_idxs[i] = [x - (start + 2) for x in cls_idxs[i]]
                if cls_idxs[i][-1] >= rewards[i, start:ends[i]].shape[0]:
                    cls_idxs[i][-1] = cls_idxs[i][-1] - 1
                # print('REWARD I: ', reward_i)
                # print('CLS IDX AFTER : ', cls_idxs[i])
                reward_i = torch.tensor(reward_i)
                clamped_rewards = torch.clamp(reward_i, -self.clip_reward_value, self.clip_reward_value).to(rewards.device)
                
                # print('CLAMPED REWARDS: ', clamped_rewards)
                # print('ORIGINAL REWARDS: ', rewards[i, start:ends[i]][torch.tensor(cls_idxs[i])])
                # print('REWARDS SIZE: ', rewards[i, start:ends[i]].shape)
                rewards[i, start:ends[i]][torch.tensor(cls_idxs[i])] += clamped_rewards

            return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        cls_idxs = inputs['cls_idxs']
        actor_cls_idxs = inputs['actor_cls_idxs']
        
        # print('prompts shape: ',  prompts.shape)
        # print('log_probs shape: ',  log_probs.shape)
        # print('ref_log_probs shape: ',  ref_log_probs.shape)
        # print('reward_score shape: ',  reward_score.shape)
        # print('values shape: ',  values.shape)
        # print('attention_mask shape: ',  attention_mask.shape)
        # print('seq shape: ',  seq.shape)
        

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            if self.args.reward_delivery_method < 2:
                old_rewards = self.compute_rewards(prompts, log_probs,
                                                ref_log_probs, reward_score,
                                                action_mask)
            else:
                old_rewards = self.compute_rewards(prompts, log_probs,
                                                ref_log_probs, reward_score,
                                                action_mask, dump=False, cls_idxs=actor_cls_idxs)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
