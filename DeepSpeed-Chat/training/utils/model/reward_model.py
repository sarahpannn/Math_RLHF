# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn

from utils.utils import to_device

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, args=None):
        super().__init__()
        self.args=args
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Sequential(nn.Linear(self.config.word_embed_proj_dim, 1, bias=False)
                                        )
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None
        
        raise RuntimeError("should not call forward() because we use this for RLHF.")

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }
        
    def get_reward(self, transformer_outputs, prediction_mask, prod=False, avg=False):
        scores_list = []
        logits = transformer_outputs[0] #TODO(self): make this work for larger batch sizes
        prediction_mask = torch.tensor(prediction_mask).to(logits.device) # [bs, seq_len]
        
        logits_of_interest = logits[prediction_mask == 1]
        regularized_relevant_indices = torch.nn.functional.softmax(logits_of_interest, dim=1)
        scores = regularized_relevant_indices[:, 0]
        
        # print("logits size: ", logits.shape)
        # print(prediction_mask)
        
        if prod:
            scores = regularized_relevant_indices[:, 1]
            scores = torch.prod(scores)    
                
        if avg:
            scores = regularized_relevant_indices[:, 1]
            scores = torch.mean(scores)
            
        # if torch.isnan(scores).any():
        #     scores = torch.tensor(0.0)
            
        scores_list.append(scores)
        
        if prod or avg:
            return torch.tensor(scores_list)
                    
        return scores_list
        

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None, 
                      return_value_only=False,
                      dump=True,
                      prompt_length=0,
                      use_cache=False,
                      cls_idx=[],
                      prediction_mask=[]):
        
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            attention_mask=attention_mask,
        )
        
        if return_value_only:
            hidden_states = transformer_outputs.hidden_states   
            
            values = self.v_head(hidden_states[-1]).squeeze(-1)
            return values
        
        if not self.args.prm:
            return self.get_reward(transformer_outputs, prediction_mask)
        
        if self.args.reward_delivery_method == 0:
            return self.get_reward(transformer_outputs, prediction_mask, avg=True)
        
        if self.args.reward_delivery_method == 1:
            return self.get_reward(transformer_outputs, prediction_mask, prod=True)
        
        if self.args.reward_delivery_method == 2:
            return self.get_reward(transformer_outputs, prediction_mask)
        
        # scores = torch.prod(regularized_relevant_indices[:, 1], dim=1)
        
        # logits = transformer_outputs[0] # [bs, seq_len, 3]
        
        # scores = []
        
        # if dump:
        #     for i in range(logits.shape[0]):
        #         idxs_of_interest = cls_idx[i]
        #         logits_of_interest = logits[i][idxs_of_interest]
                
        #         regularized_relevant_indices = torch.nn.functional.softmax(logits_of_interest, dim=1)
        #         correct_likelihoods = regularized_relevant_indices[:, 1]
                
        #         scores.append(torch.prod(correct_likelihoods)) # TODO:(self) implement other ways to do this, too!
            
        #     return torch.tensor(scores)
        
        # if not dump:
        #     for i in range(logits.shape[0]):
        #         idxs_of_interest = cls_idx[i]
        #         logits_of_interest = logits[i][idxs_of_interest]
                
        #         regularized_relevant_indices = torch.nn.functional.softmax(logits_of_interest, dim=1)
        #         correct_likelihoods = regularized_relevant_indices[:, 1]
                
        #         scores.appen(correct_likelihoods)
        #     return scores
            
        
        # hidden_states = transformer_outputs[0]
        # values = self.v_head(hidden_states).squeeze(-1)
        # if return_value_only:
        #     return values
        # else:
        #     # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
        #     # [prompt, answer, 0, 0, 0, 0] this is normal
        #     assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
        #     bs = values.size(0)
        #     seq_len = input_ids.shape[1]
        #     chosen_end_scores = [
        #     ]  # we use this name for consistency with the original forward function
        #     for i in range(bs):
        #         input_id = input_ids[i]
        #         value = values[i]

        #         c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
        #         # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
        #         c_ind = c_inds[0].item() + prompt_length if len(
        #             c_inds) > 0 else seq_len
        #         chosen_end_scores.append(value[c_ind - 1])
        #     return {
        #         "values": values,
        #         "chosen_end_scores": torch.stack(chosen_end_scores),
        #     }
