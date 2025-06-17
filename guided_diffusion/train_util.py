import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

# Add SSIM calculation
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim_batch(img1, img2, data_range=2.0):
    """Calculate SSIM between two image batches.
    
    Args:
        img1: First image tensor (B, C, H, W) in range [-1, 1]
        img2: Second image tensor (B, C, H, W) in range [-1, 1]
        data_range: Range of the data (2.0 for [-1, 1] range)
    Returns:
        Mean SSIM value across the batch
    """
    # Ensure tensors are on CPU and detached
    if isinstance(img1, th.Tensor):
        img1 = img1.detach().cpu()
    if isinstance(img2, th.Tensor):
        img2 = img2.detach().cpu()
    
    # Convert to numpy arrays
    img1_np = img1.numpy()
    img2_np = img2.numpy()
    
    batch_size = img1_np.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        # Handle different channel configurations
        if img1_np.shape[1] == 1:  # Grayscale
            img1_sample = img1_np[i, 0]  # Remove channel dimension
            img2_sample = img2_np[i, 0]
            ssim_val = ssim(img1_sample, img2_sample, data_range=data_range)
        else:  # Multi-channel
            # Convert from (C, H, W) to (H, W, C)
            img1_sample = np.transpose(img1_np[i], (1, 2, 0))
            img2_sample = np.transpose(img2_np[i], (1, 2, 0))
            ssim_val = ssim(img1_sample, img2_sample, data_range=data_range, channel_axis=-1)
        
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.samples_per_epoch = 0  # Will be updated in run_loop

        self.sync_cuda = th.cuda.is_available()

        # For SSIM calculation and logging
        self.last_batch = None
        self.last_prediction = None
        self.ssim_values = []

        # Add loss tracking
        self.loss_history = []
        self.last_loss = 0.0

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            checkpoint_data = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            
            # Handle both old and new checkpoint formats
            if 'model_state_dict' in checkpoint_data:
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                self.resume_step = checkpoint_data.get('step', 0)
            else:
                # Old format
                self.model.load_state_dict(checkpoint_data)
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        
        # Store for SSIM calculation
        self.last_batch = batch.clone()
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # Ensure consistent tensor types (float32)
            micro = micro.float()
            t = t.float()
            weights = weights.float()

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            # Store loss for checkpointing
            self.last_loss = loss.item()
            self.loss_history.append(self.last_loss)
            
            # Calculate SSIM for logging (every log_interval steps)
            if self.step % self.log_interval == 0:
                try:
                    with th.no_grad():
                        # Move timestep tensor to CPU for numpy conversion
                        t_cpu = t.cpu()
                        
                        # Get diffusion parameters for the current timestep
                        alpha_bar_t = th.from_numpy(self.diffusion.alphas_cumprod[t_cpu.numpy()]).to(dist_util.dev())
                        sqrt_alpha_bar_t = th.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_bar_t = th.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1)
                        
                        # Create noisy version of x_start using the same process as training
                        noise = th.randn_like(micro)
                        x_t = sqrt_alpha_bar_t * micro + sqrt_one_minus_alpha_bar_t * noise
                        
                        # Get model prediction (noise prediction)
                        model_output = self.ddp_model(x_t, t, **micro_cond)
                        
                        # Reconstruct predicted clean image
                        predicted_x0 = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
                        
                        # Move tensors to CPU before SSIM calculation
                        micro_cpu = micro.detach().cpu()
                        predicted_x0_cpu = predicted_x0.detach().cpu()
                        
                        # Split into image and mask components for SSIM calculation
                        if micro_cpu.shape[1] > 1:  # We have both image and mask
                            original_image = micro_cpu[:, :1]  # First channel is image
                            predicted_image = predicted_x0_cpu[:, :1]  # First channel prediction
                            
                            # Calculate SSIM for image component only
                            ssim_value = calculate_ssim_batch(original_image, predicted_image)
                            
                            # Store and log SSIM
                            self.ssim_values.append(ssim_value)
                            logger.logkv_mean("ssim", ssim_value)
                            logger.log(f"Calculated SSIM: {ssim_value}")  # Debug log
                            
                        else:  # Only image data
                            ssim_value = calculate_ssim_batch(micro_cpu, predicted_x0_cpu)
                            self.ssim_values.append(ssim_value)
                            logger.logkv_mean("ssim", ssim_value)
                            logger.log(f"Calculated SSIM: {ssim_value}")  # Debug log
                            
                except Exception as e:
                    logger.log(f"SSIM calculation failed: {e}")
                    import traceback
                    logger.log(f"Traceback: {traceback.format_exc()}")  # Full traceback for debugging
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            
            # Force log dump after SSIM calculation
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _calculate_and_log_ssim(self, x_start, t, model_kwargs):
        """Calculate SSIM between original and reconstructed images using proper diffusion reconstruction."""
        try:
            with th.no_grad():
                # Get diffusion parameters for the current timestep
                alpha_bar_t = self.diffusion.alphas_cumprod[t]
                sqrt_alpha_bar_t = th.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar_t = th.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1)
                
                # Create noisy version of x_start using the same process as training
                noise = th.randn_like(x_start)
                x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
                
                # Get model prediction (noise prediction)
                model_output = self.ddp_model(x_t, t, **model_kwargs)
                
                # Reconstruct predicted clean image using the formula:
                # x_hat_0 = (1/sqrt(alpha_bar_t)) * (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred)
                predicted_x0 = (x_t - sqrt_one_minus_alpha_bar_t * model_output) / sqrt_alpha_bar_t
                
                # Split into image and mask components for SSIM calculation
                if x_start.shape[1] > 1:  # We have both image and mask
                    original_image = x_start[:, :1]  # First channel is image
                    predicted_image = predicted_x0[:, :1]  # First channel prediction
                    
                    # Calculate SSIM for image component only (move to CPU first)
                    ssim_value = calculate_ssim_batch(original_image.cpu(), predicted_image.cpu())
                    
                    # Store and log SSIM
                    self.ssim_values.append(ssim_value)
                    logger.logkv_mean("ssim", ssim_value)
                        
                else:  # Only image data
                    ssim_value = calculate_ssim_batch(x_start.cpu(), predicted_x0.cpu())
                    self.ssim_values.append(ssim_value)
                    logger.logkv_mean("ssim", ssim_value)
                    
        except Exception as e:
            logger.log(f"SSIM calculation failed: {e}")

    def log_step(self):
        try:
            current_step = self.step + self.resume_step
            
            # Ensure we're on the main process for logging
            if dist.get_rank() == 0:
                # Only log if we have metrics to log
                if hasattr(self, 'ssim_values') and self.ssim_values:
                    avg_ssim = np.mean(self.ssim_values[-10:])  # Average of last 10 values
                    logger.logkv("step", current_step)
                    logger.logkv("samples", (current_step + 1) * self.global_batch)
                    logger.logkv("avg_ssim", avg_ssim)
                    logger.logkv("ssim", self.ssim_values[-1])  # Log the most recent SSIM
                    
                    # Dump all metrics to both CSV and wandb
                    logger.dumpkvs()
                    
                    # Clear SSIM values after logging
                    self.ssim_values = []
            
        except Exception as e:
            logger.log(f"Error in log_step: {str(e)}")
            # Continue training even if logging fails
            pass

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                
                # Calculate current metrics
                current_loss = getattr(self, 'last_loss', 0.0)  # Use tracked loss
                current_ssim = np.mean(self.ssim_values[-10:]) if self.ssim_values else 0.0
                current_step = self.step + self.resume_step
                
                # Save regular checkpoint
                if not rate:
                    filename = f"checkpoint_step_{current_step:06d}.pt"
                else:
                    filename = f"ema_{rate}_step_{current_step:06d}.pt"
                
                checkpoint = {
                    'step': current_step,
                    'model_state_dict': state_dict,
                    'loss': current_loss,
                    'ssim': current_ssim
                }
                
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(checkpoint, f)
                
                # Save best models if applicable
                if not rate:  # Only save best models for non-EMA checkpoints
                    # Save best loss model
                    best_loss_path = bf.join(get_blob_logdir(), 'best_loss_model.pt')
                    if not bf.exists(best_loss_path) or current_loss < th.load(best_loss_path)['loss']:
                        with bf.BlobFile(best_loss_path, "wb") as f:
                            th.save(checkpoint, f)
                    
                    # Save best SSIM model
                    best_ssim_path = bf.join(get_blob_logdir(), 'best_ssim_model.pt')
                    if not bf.exists(best_ssim_path) or current_ssim > th.load(best_ssim_path)['ssim']:
                        with bf.BlobFile(best_ssim_path, "wb") as f:
                            th.save(checkpoint, f)
                
                # Remove old checkpoints (keep only last 3)
                if not rate:  # Only clean up non-EMA checkpoints
                    checkpoints = sorted([f for f in bf.listdir(get_blob_logdir()) 
                                        if f.startswith('checkpoint_step_')])
                    if len(checkpoints) > 3:
                        for checkpoint_to_remove in checkpoints[:-3]:
                            bf.remove(bf.join(get_blob_logdir(), checkpoint_to_remove))

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """Log losses to both console and wandb with proper naming."""
    for key, values in losses.items():
        # Log mean value
        mean_value = values.mean().item()
        logger.logkv_mean(key, mean_value)
        
        # Log quartiles for detailed analysis
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
        
        # Log to logger with proper naming (wandb will get these through dumpkvs)
        if key == "loss":
            logger.logkv_mean("total_loss", mean_value)
        elif key == "image_mse":
            logger.logkv_mean("image_mse", mean_value)
        elif key == "mask_loss":
            logger.logkv_mean("mask_loss", mean_value)
        elif key == "vb":
            logger.logkv_mean("vb_loss", mean_value)
        elif key == "xstart_mse":
            logger.logkv_mean("xstart_mse", mean_value)
