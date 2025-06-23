#!/bin/bash
# Training script for diffusion model with mask support

# SLURM configuration
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/users/scxcw/ilvr_ds/logs/1000sample_mask_%j.log
#SBATCH --error=/users/scxcw/ilvr_ds/logs/1000sample_mask_%j.err
#SBATCH --mail-user=scxcw@leeds.ac.uk

# Print start time
start_time=`date +%s`
echo "Job started at: $(date)"

# Change to project directory
cd /users/scxcw/ilvr_ds

# Add current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Securely load the API key from the local, untracked file
if [ -f "secrets.sh" ]; then
    source secrets.sh
else
    echo "Warning: secrets.sh not found. WANDB_API_KEY may not be set."
fi

# Create log directory if it doesn't exist
mkdir -p /scratch/scxcw/results/Jun2025_dm_adjustedLoss/log_1000_mask

# Run training with correct channel configuration
# Configuration: 1 image channel + 1 mask channel = 2 input channels
# With learn_sigma=True: 4 output channels (2 * 2)
python scripts/image_train.py \
    --data_dir /scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d \
    --log_dir /scratch/scxcw/results/Jun2025_dm_adjustedLoss/log_1000_mask \
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
    --lr 1e-4 \
    --batch_size 32 \
    --microbatch 4

# Record command execution time
end_command=`date +%s%N`
command_duration=$((end_command - start_command))

# Print end time and duration
end_time=`date +%s`
runtime=$((end_time - start_time))
echo "Job ended at: $(date)"
echo "Total job duration: $(printf '%02d:%02d:%02d' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))) (hh:mm:ss)"

