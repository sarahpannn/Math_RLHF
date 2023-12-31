# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    PreTrainedModel,
    OpenLlamaConfig,
    LlamaForCausalLM,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from .reward_model import RewardModel


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False,
                    is_reward=False,
                    is_ref=False,):
    # model_config = AutoConfig.from_pretrained(model_name_or_path)
    model_config = OpenLlamaConfig.from_pretrained(model_name_or_path)
    model_config.shared_input_output_embedding = False
    model_config.use_memory_efficient_attention = False
    model_config.use_stable_embedding = False
    model_config.output_hidden_states=True
    # print('WHEN INIT MODEL, NAME IS ', model_name_or_path)
    if 'deberta' in model_name_or_path:
        model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
        
    if rlhf_training and not is_reward:
        model = model_class.from_config(model_config)
    
    elif is_reward:
        # the weight loading is handled by create critic model
        model = model_class(model_config)
        # model = model_class.from_config(model_config)
        
    elif is_ref:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            )
    
    elif not rlhf_training:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    if not is_reward:
        model.resize_token_embeddings(int(
            8 *
            math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model

class LlamaForClassification(PreTrainedModel):
    _no_split_modules = ['classification_head']
    supports_gradient_checkpointing = True
    def __init__(self, config):
        super(LlamaForClassification, self).__init__(config)
        self.model = AutoModel.from_config(config=config)
        self.classification_head = nn.Linear(config.hidden_size, 2)  # num_classes is the number of classes in your classification task


    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classification_head(outputs.last_hidden_state)
        return logits, outputs
    

def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False,
                        is_reward=False,
                        args=None):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    
    if "llama" in model_name_or_path and is_reward:
        critic_model = create_hf_model(LlamaForClassification, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout, is_reward=True)
        
    elif "llama" in model_name_or_path and rlhf_training:
        critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                      ds_config, rlhf_training, disable_dropout)
    else:
        critic_model = create_hf_model(AutoModelForTokenClassification, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)

    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        args=args)

    if rlhf_training:
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"
        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'))

    return critic_model
