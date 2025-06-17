"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import os
import torch
import wandb

def main():
    args = create_argparser().parse_args()
    
    # Initialize wandb with project configuration
    wandb_config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "microbatch": args.microbatch,
        "image_size": args.image_size,
        "diffusion_steps": args.diffusion_steps,
        "noise_schedule": args.noise_schedule,
        "use_fp16": args.use_fp16,
        "ema_rate": args.ema_rate,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "mask_channels": args.mask_channels,
        "use_mask": args.use_mask,
        "learn_sigma": args.learn_sigma,
        "num_channels": args.num_channels,
        "num_res_blocks": args.num_res_blocks,
        "attention_resolutions": args.attention_resolutions,
        "dropout": args.dropout,
        "use_scale_shift_norm": args.use_scale_shift_norm,
        "resblock_updown": args.resblock_updown,
        "timestep_respacing": args.timestep_respacing,
        "weight_decay": args.weight_decay,
        "lr_anneal_steps": args.lr_anneal_steps,
    }

    dist_util.setup_dist()
    logger.configure(
        dir=args.log_dir,
        format_strs=["stdout", "log", "csv", "wandb"],
        wandb_project="diffusion_model",
        wandb_entity=None,  # Your wandb username/team
        wandb_config=wandb_config
    )

    # Log device information
    device = dist_util.dev()
    logger.log(f"Using device: {device}")
    if device.type == "mps":
        logger.log("MPS (Metal Performance Shaders) is available and will be used")
        # MPS-specific configurations
        if args.use_fp16:
            logger.log("FP16 is not supported on MPS, disabling...")
            args.use_fp16 = False
    elif device.type == "cuda":
        logger.log(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        logger.log("Using CPU for training")

    logger.log("creating model and diffusion...")
    # Remove mask_dir from model creation arguments
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    if 'mask_dir' in model_args:
        del model_args['mask_dir']
    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        mask_dir=args.mask_dir,  # Pass mask_dir to data loader
        use_mask=args.use_mask,  # Pass use_mask to data loader
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",  
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        mask_dir="",  # Add mask_dir parameter
        use_mask=False,  # Add use_mask parameter
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

def save_checkpoint(model, optimizer, ema_params, epoch, loss, ssim_value, output_dir):
    """Save model checkpoint with both regular and best models.
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer state to save
        ema_params: List of EMA parameters
        epoch: Current epoch number
        loss: Current loss value
        ssim_value: Current SSIM value
        output_dir: Directory to save checkpoints
    """
    global best_loss, best_ssim
    
    # Regular checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'ssim': ssim_value
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save optimizer state
    opt_path = os.path.join(output_dir, f'opt_epoch_{epoch}.pt')
    torch.save(optimizer.state_dict(), opt_path)
    
    # Save EMA states
    for i, ema_param in enumerate(ema_params):
        ema_path = os.path.join(output_dir, f'ema_{i}_epoch_{epoch}.pt')
        torch.save(ema_param, ema_path)
    
    # Check for best loss model
    if loss < best_loss:
        best_loss = loss
        # Save best loss model and its EMA
        best_loss_path = os.path.join(output_dir, 'best_loss_model.pt')
        torch.save(checkpoint, best_loss_path)
        
        # Save corresponding EMA
        for i, ema_param in enumerate(ema_params):
            ema_path = os.path.join(output_dir, f'best_loss_ema_{i}.pt')
            torch.save(ema_param, ema_path)
    
    # Check for best SSIM model
    if ssim_value > best_ssim:
        best_ssim = ssim_value
        # Save best SSIM model and its EMA
        best_ssim_path = os.path.join(output_dir, 'best_ssim_model.pt')
        torch.save(checkpoint, best_ssim_path)
        
        # Save corresponding EMA
        for i, ema_param in enumerate(ema_params):
            ema_path = os.path.join(output_dir, f'best_ssim_ema_{i}.pt')
            torch.save(ema_param, ema_path)
    
    # Remove old checkpoints if they exist (keep only last 3)
    checkpoints = sorted([f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')])
    if len(checkpoints) > 3:
        for checkpoint_to_remove in checkpoints[:-3]:
            base_name = checkpoint_to_remove.replace('checkpoint_epoch_', '')
            epoch_num = base_name.replace('.pt', '')
            # Remove corresponding optimizer and EMA checkpoints
            os.remove(os.path.join(output_dir, f'opt_epoch_{epoch_num}.pt'))
            for i in range(len(ema_params)):
                os.remove(os.path.join(output_dir, f'ema_{i}_epoch_{epoch_num}.pt'))
            os.remove(os.path.join(output_dir, checkpoint_to_remove))

def save(self):
    """Save checkpoints for model, optimizer, and EMA parameters."""
    if dist.get_rank() == 0:
        # Calculate current loss and SSIM
        current_loss = self.mp_trainer.master_params[0].item()  # Get current loss
        current_ssim = calculate_ssim(self.last_batch, self.last_pred)  # Get current SSIM
        
        # Save all checkpoints
        logger.save_checkpoint(
            model=self.model,
            optimizer=self.opt,
            ema_params=self.ema_params,
            epoch=self.step + self.resume_step,
            loss=current_loss,
            ssim_value=current_ssim,
            output_dir=logger.get_dir()
        )
    
    dist.barrier()

def log_loss_dict(diffusion, ts, losses):
    """Log losses to both console and wandb."""
    for key, values in losses.items():
        # Log mean value
        mean_value = values.mean().item()
        logger.logkv_mean(key, mean_value)
        
        # Log quartiles for detailed analysis
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
        
        # Log to wandb with proper naming
        if key == "loss":
            logger.logkv_mean("total_loss", mean_value)
        elif key == "image_mse":
            logger.logkv_mean("image_mse", mean_value)
        elif key == "mask_loss":
            logger.logkv_mean("mask_loss", mean_value)

def log_training_metrics(epoch, losses, ssim_value):
    """Log training metrics to wandb with custom visualizations."""
    # Log basic metrics
    wandb.log({
        "epoch": epoch,
        "total_loss": losses["loss"].mean().item(),
        "image_mse": losses["image_mse"].mean().item(),
        "mask_loss": losses["mask_loss"].mean().item(),
        "ssim": ssim_value
    })
    
    # Create custom plots
    wandb.log({
        "loss_components": wandb.plot.line_series(
            xs=[epoch],
            ys=[[losses["image_mse"].mean().item()], 
                [losses["mask_loss"].mean().item()]],
            keys=["Image MSE", "Mask Loss"],
            title="Loss Components",
            xname="Epoch"
        )
    })
