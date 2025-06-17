# Diffusion Model Training Work Log

## Project: Cardiac MRI Diffusion Model with Mask Support

### Author: Xinci (Medical AI PhD Student)
### Last Updated: {{DATE}}

---

## Overview
This work log documents the progress, debugging, and improvements made to the diffusion model training pipeline for cardiac MRI segmentation and synthesis. The goal is to maintain a detailed, reproducible record for future reference and collaboration.

---

## 1. Initial Setup
- **Repository Structure:**
  - Main scripts: `image_dataset.py`, `train_diffusion.sh`, `model_metrics.py`, `scripts/image_train.py`
  - Core model code: `guided_diffusion/`
  - Utilities: `resizer.py`, `measure_model_efficiency.py`
  - Data and output folders: `datasets/`, `output/`, `ref_imgs/`, `gif/`
- **Environment:**
  - Python dependencies listed in `requirements.txt`
  - SLURM for job scheduling on HPC cluster
  - Logging and experiment tracking with wandb

---

## 2. Training Pipeline Configuration
- **Training Script:** `train_diffusion.sh`
  - Configured for SLURM with GPU, 128GB RAM, 4 CPUs
  - Sets up environment variables and log directories
  - Runs `scripts/image_train.py` with arguments for:
    - Data directory, log directory
    - Model architecture (channels, blocks, attention)
    - Diffusion steps, learning rate, batch size, microbatch
    - Mask support: `--use_mask True`, `--mask_channels 1`, `--mask_dir ...`
    - Output and checkpointing configuration

---

## 3. Loss Calculation and Model Improvements
- **Losses Implemented:**
  - **Image MSE Loss:** Between original and predicted images
  - **Mask Loss:** Weighted MSE for mask channels, focusing on regions of interest (ROI)
  - **Total Loss:** Sum of image and mask losses
  - **SSIM Metric:** Structural Similarity Index for image quality
- **Key Improvements:**
  - Added explicit type conversions to float32 for all tensors entering the model to avoid type mismatch errors (e.g., double vs. float)
  - Ensured all tensors are on the correct device (CPU/GPU) before operations
  - Improved SSIM calculation to handle both single-channel and multi-channel images, and to avoid CUDA/numpy conversion errors
  - Added detailed error handling and debug logging for SSIM calculation

---

## 4. Logging and Checkpointing
- **Logging System:**
  - Logs all key metrics (loss, image MSE, mask loss, SSIM) to both CSV and wandb
  - Added logic to prevent empty or duplicate log entries
  - Ensured step synchronization between local logs and wandb to avoid step mismatch warnings
  - Logs quartiles of losses for detailed analysis
- **Checkpointing:**
  - Saves model checkpoints every 10,000 steps
  - Keeps last 3 regular checkpoints, best loss, and best SSIM checkpoints
  - Saves optimizer state for resuming training

---

## 5. Debugging and Issue Resolution
- **Type Mismatch Error:**
  - Resolved by converting all input tensors to float32 before model forward pass
- **CUDA/Numpy Conversion Error in SSIM:**
  - Fixed by ensuring all tensors are detached and moved to CPU before numpy conversion
- **Wandb Step Mismatch Warning:**
  - Added logic to track and synchronize step numbers between local logs and wandb
- **Empty Logging Warning:**
  - Prevented by only logging when actual metrics are available

---

## 6. Final Checklist Before Training
- [x] Loss calculations (image, mask, total) are correct and robust
- [x] SSIM metric is calculated and logged properly
- [x] Logging system is synchronized and records all metrics
- [x] Checkpointing is reliable and keeps recent/best models
- [x] All tensor types and devices are consistent
- [x] SLURM script is configured for the correct environment and resources

---

## 7. Next Steps
- Launch training job on SLURM cluster
- Monitor logs and wandb dashboard for progress and potential issues
- Periodically review checkpoints and validation metrics
- Update this log with new findings, issues, or improvements

---

## 8. References
- [Diffusion Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Wandb Documentation](https://docs.wandb.ai/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)

---

*This log is maintained for transparency, reproducibility, and collaboration. Please update regularly as the project progresses.* 