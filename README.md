# Math_RLHF

# Repository Setup Guide

This guide will walk you through the process of setting up this repository, including cloning it, copying model checkpoints to specific folders, navigating to the correct location, and running a Bash command.

## 1. Clone the Repository

To get started, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/sarahpannn/Math_RLHF.git
```

## 2. Copy Over Model Checkpoints

### Generator Model
Original file path: ```/mnt/shared_home/span/lets-reinforce-step-by-step/training/ONLY_MATH_SFT/four_epochs```

New file path: ```/ONLY_MATH_SFT/four_epochs```

### Outcome-supervised Reward Model (ORM)
Original file path: ```/mnt/shared_home/span/lets-reinforce-step-by-step/training/model/llama-3b-ORM/hf_directory```

New file path: ```/llama-3b-ORM/hf_directory```

### Process-supervised Reward Model (PRM)
Original file path: ```/mnt/shared_home/span/lets-reinforce-step-by-step/training/model/deberta-v3-large-800k-3```

## 3. CD into the right location
```bash
cd DeepSpeed-Chat/training/step3_rlhf_finetuning
```
## 4. Run bash command
If RLHF using ORM, run:
```bash
bash training_scripts/single_node/run_6.7.sh
```
If RLHF using PRM delivery method avg, run:
```bash
bash training_scripts/single_node/critic_reward_mismatch.sh
```
If RLHF using PRM delivery method product, run:
```bash
bash training_scripts/single_node/critic_reward_mismatch.sh
```
If RLHF using PRM delivery method fine-grained, run:
```bash
bash training_scripts/single_node/real_prm.sh
```
## 5. Adjust batch size

I'm not sure how much RAM the A100's will have, but if it's more than 24 gb like Inanna, I assume increasing the per_device_batch_size will maximize usage. 