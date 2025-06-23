#!/bin/bash

# Change to project directory
cd /users/scxcw/ilvr_ds

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/users/scxcw/ilvr_ds

# Securely load the API key from the local, untracked file
if [ -f "secrets.sh" ]; then
    source secrets.sh
else
    echo "Warning: secrets.sh not found. WANDB_API_KEY may not be set."
fi

# Create log directory if it doesn't exist
mkdir -p results/log_dm/log_1000_mask

# Run training with a smaller batch size and fewer steps for testing
# Configuration: 1 image channel + 1 mask channel = 2 total channels
# With learn_sigma=True: 2 input channels, 4 output channels (2 * 2)
python scripts/image_train.py \
    --data_dir /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d \
    --log_dir results/log_dm/log_1000_mask \
    --attention_resolutions 16 \
    --class_cond False \
    --diffusion_steps 1000 \
    --dropout 0.0 \
    --image_size 256 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 128 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 False \
    --use_scale_shift_norm True \
    --timestep_respacing 100 \
    --in_channels 1 \
    --out_channels 1 \
    --mask_channels 1 \
    --mask_dir /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d \
    --use_mask True \
    --batch_size 1 \
    --microbatch 1 \
    --lr 1e-4 