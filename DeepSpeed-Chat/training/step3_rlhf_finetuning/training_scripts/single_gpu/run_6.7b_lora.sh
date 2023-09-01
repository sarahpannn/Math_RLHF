#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


ACTOR_ZERO_STAGE="--actor_zero_stage 3"
CRITIC_ZERO_STAGE="--critic_zero_stage 3"
ACTOR_MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
CRITIC_MODEL_PATH="/mnt/shared_home/span/lets-reinforce-step-by-step/training/model/deberta-v3-large-800k-3"

OUTPUT="./output"

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=5e-4
Critic_Lr=5e-6

mkdir -p $OUTPUT

deepspeed --num_gpus 2 main.py \
   --data_path sarahpann/PRM800K \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --ppo_epochs 1 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} \
   --offload_reference_model \
   --actor_lora_dim 128 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --inference_tp_size 1 \
   --tp_gather_partition_size 2 \
   --disable_actor_dropout \
   --enable_hybrid_engine \
   --offload \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
