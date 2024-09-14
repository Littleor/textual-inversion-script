# Textual Inversion Script

<p align="center">
<a href="https://github.com/Littleor/textual-inversion-script/blob/master/LICENSE" target="blank">
<img src="https://img.shields.io/github/license/Littleor/textual-inversion-script?style=flat-square" alt="github-profile-readme-generator license" />
</a>
<a href="https://github.com/Littleor/textual-inversion-script/fork" target="blank">
<img src="https://img.shields.io/github/forks/Littleor/textual-inversion-script?style=flat-square" alt="github-profile-readme-generator forks"/>
</a>
<a href="https://github.com/Littleor/textual-inversion-script/stargazers" target="blank">
<img src="https://img.shields.io/github/stars/Littleor/textual-inversion-script?style=flat-square" alt="github-profile-readme-generator stars"/>
</a>
<a href="https://github.com/Littleor/textual-inversion-script/issues" target="blank">
<img src="https://img.shields.io/github/issues/Littleor/textual-inversion-script?style=flat-square" alt="github-profile-readme-generator issues"/>
</a>
<a href="https://github.com/Littleor/textual-inversion-script/pulls" target="blank">
<img src="https://img.shields.io/github/issues-pr/Littleor/textual-inversion-script?style=flat-square" alt="github-profile-readme-generator pull-requests"/>
</a>
</p>



This repository contains some scripts that can be used to [Textual Inversion](https://github.com/rinongal/textual_inversion) in both FLUX ([FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)) and Stable Diffusion 3 ([SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)) models.

> Updates:
> **24.09.14**: Now we can use 24GB VRAM to train the FLUX.1 Textual Inversion model.

## Introduction
This script need 40G+ VRAM to **directly** train the FLUX.1 textual inversion model. 
But now we support to train `FLUX.1 dev` model with ONLY **~22G** VRAM, any consumer grade graphics card with 24G VRAM can run this script.

## Install

We use the `diffuser` library to run the models, so you need to install it first.

```bash
# 1. Clone
git clone https://github.com/Littleor/textual-inversion-script.git
# 2. cd 
cd textual-inversion-script
# 3. Install
pip install diffusers[torch] transformers[sentencepiece] 
```

## Basic Usage

We reference the [Diffuser script](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion_sdxl.py), so all the arguments are the same, you can read [this](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/README_sdxl.md).

```bash
export DATA_DIR="./cat"

accelerate launch textual_inversion_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./logs" \
  --gradient_checkpointing \
  --cache_latent
```

## Low VRAM Usage
As the FLUX.1 model is too large to fit in a single GPU with 24G VRAM, so we can use `deepspeed` to train the model with only 22G VRAM.

We have write a cofigure file `accelerate_config/deepspeed_zero3_offload_config.yaml` to help you train the model within 24G VRAM, below is the usage:

```bash
accelerate launch --config_file "accelerate_config/deepspeed_zero3_offload_config.yaml"  textual_inversion_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./logs" \
  --gradient_checkpointing \
  --cache_latent
```