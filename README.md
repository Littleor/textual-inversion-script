# Textual Inversion Script

This repository contains some scripts that can be used to [Textual Inversion](https://github.com/rinongal/textual_inversion) in both Stable Diffusion 3 ([SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)) and FLUX ([FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)) models.

> This script need 40G+ VRAM to run, 24G version maybe released soon.

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

## Usage

We reference the [Diffuser script](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion_sdxl.py), so all the arguments are the same, you can read [this](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/README_sdxl.md).

```bash
export DATA_DIR="./cat"

accelerate launch textual_inversion_flux.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
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

