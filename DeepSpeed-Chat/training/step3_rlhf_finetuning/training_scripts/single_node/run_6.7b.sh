#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

#   --inference_tp_size 1 \
#   --tp_gather_partition_size 2 \
# --enable_hybrid_engine \

# DeepSpeed Team
ACTOR_ZERO_STAGE="--actor_zero_stage 2"
CRITIC_ZERO_STAGE="--critic_zero_stage 3"
# ACTOR_MODEL_PATH="gpt2"
ACTOR_MODEL_PATH="/ONLY_MATH_SFT/four_epochs"
# ACTOR_MODEL_PATH="openlm-research/open_llama_3b_v2"
# ACTOR_MODEL_PATH="meta-llama/Llama-2-7b-hf"
# ACTOR_MODEL_PATH="AdamG012/chat-opt-1.3b-rlhf-actor-deepspeed"
CRITIC_MODEL_PATH="/llama-3b-ORM/hf_directory"
# CRITIC_MODEL_PATH="/mnt/shared_home/span/lets-reinforce-step-by-step/training/model/llama-3b/hf_directory"
# CRITIC_MODEL_PATH="/mnt/shared_home/span/lets-reinforce-step-by-step/training/model/deberta-v3-large-800k-6"

OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=3
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=3
fi
mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

deepspeed --master_port 12346 --num_gpus 2 main.py \
   --data_path sarahpann/PRM800K \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 640 \
   --max_prompt_seq_len 128 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 128 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_lora_dim 64 \
   --actor_lora_module_name layers. \
   --critic_lora_dim 32 \
   --critic_lora_module_name layers. \
   --inference_tp_size 1 \
   --tp_gather_partition_size 2 \
   --offload_reference_model \
   --offload_reward_model \
   --enable_hybrid_engine \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
