import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
import math
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import nibabel as nib


def load_reference(data_dir, batch_size, image_size, class_cond=False, use_mask=False, mask_dir=None):
    """Load reference data for sampling, supporting both image-only and image+mask data."""
    if use_mask and mask_dir is None:
        # For sampling: we want the model to generate both image and mask
        # So we load only image data but still use 2-channel model
        data = load_data(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            class_cond=class_cond,
            deterministic=True,
            random_flip=False,
            use_mask=False,  # Don't load masks during sampling
            is_sampling=True
        )
    else:
        # For training or when mask_dir is provided
        data = load_data(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            class_cond=class_cond,
            deterministic=True,
            random_flip=False,
            use_mask=use_mask,
            mask_dir=mask_dir,
            is_sampling=True
        )
    
    for large_batch, model_kwargs, filename in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs, filename

def get_data_files(data_dir):
    """Get all supported data files from the directory.
    Beware of _seg.npy which are segmentation"""
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            # Support both .png and .npy files
            if filename.endswith(('.png', '.npy', '.nii.gz')) and not filename.endswith('_seg.nii.gz'):
                file_path = os.path.join(root, filename)
                files.append(file_path)
    return sorted(files)

def main():
    args = create_argparser().parse_args()
    logger.configure(dir=args.save_dir)
    # th.manual_seed(0)
    os.makedirs(args.save_dir, exist_ok=True)  
    logger.log(f"Outputs will be saved to: {args.save_dir}")
    dist_util.setup_dist()

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.D, 2).is_integer()

    logger.log("loading data...")
    data = load_reference(
        args.ref_dir,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        use_mask=args.use_mask,
        mask_dir=None  # No mask_dir needed for sampling
    )

    assert args.num_samples >= args.batch_size * dist_util.get_world_size(), "The number of the generated samples will be larger than the specified number."

    data_files = get_data_files(args.ref_dir)

    logger.log("creating samples...")
    count = 0
    while count < args.num_samples:
        file_path = data_files[count]
        filename = os.path.basename(file_path)
        model_kwargs, _ = next(data)
        # filename = filename_tuple[0]  # Use index 0 for the first filename in the batch
        # ref_img_name = os.path.basename(filename)    #   print(type(model_kwargs), model_kwargs)  # Debugging print
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        
        # Determine output shape based on model configuration
        if args.use_mask:
            # Model outputs both image and mask channels (2 channels total)
            output_channels = 2  # 1 image + 1 mask
        else:
            # Model outputs only image channels
            output_channels = args.in_channels  # Should be 1 (image only)
        
        sample = diffusion.p_sample_loop(
            model,
            (1, output_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=model_kwargs["ref_img"],
            N=args.N,
            D=args.D,
            scale=args.scale
        )
                
        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        
        output = sample[0].cpu().numpy()  # Shape: [C, H, W]
        
        if args.use_mask:
            # Separate image and mask channels
            image_output = output[0]  # First channel is image
            mask_output = output[1]   # Second channel is mask
            
            # Process image
            image_gray = ((image_output + 1) * 127.5).astype(np.uint8)
            image_gray = np.flip(image_gray, axis=0)  # Flip vertically
            image_gray = np.rot90(image_gray, k=1, axes=(1, 0))  # Rotate 90 degrees
            image_gray = image_gray[..., np.newaxis]  # Shape: [H, W, 1]
            
            # Process mask (denormalize from [-1, 1] back to [0, 1, 2, 3])
            mask_denorm = ((mask_output + 1) * 2).astype(np.uint8)  # Convert to [0, 4] range
            mask_denorm = np.clip(mask_denorm, 0, 3)  # Clip to valid range
            mask_denorm = np.flip(mask_denorm, axis=0)
            mask_denorm = np.rot90(mask_denorm, k=1, axes=(1, 0))
            mask_denorm = mask_denorm[..., np.newaxis]
            
            # Save both image and mask
            image_path = os.path.join(args.save_dir, f"{base_name}_image.nii.gz")
            mask_path = os.path.join(args.save_dir, f"{base_name}_mask.nii.gz")
            
            ni_img = nib.Nifti1Image(image_gray, affine=np.eye(4))
            ni_mask = nib.Nifti1Image(mask_denorm, affine=np.eye(4))
            
            nib.save(ni_img, image_path)
            nib.save(ni_mask, mask_path)
            
            logger.log(f"{base_name}_image.nii.gz and {base_name}_mask.nii.gz saved")
        else:
            # Only image output
            output_gray = output.mean(axis=0) if output.shape[0] > 1 else output[0]
            output_gray = ((output_gray + 1) * 127.5).astype(np.uint8)
            output_gray = np.flip(output_gray, axis=0)
            output_gray = np.rot90(output_gray, k=1, axes=(1, 0))
            output_gray = output_gray[..., np.newaxis]

            image_path = os.path.join(args.save_dir, f"{base_name}_image.nii.gz")
            ni_img = nib.Nifti1Image(output_gray, affine=np.eye(4))
            nib.save(ni_img, image_path)
            
            logger.log(f"{base_name}_image.nii.gz saved")
        
        count += 1
        logger.log(f"created {count} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        D=32, # scaling factor
        N=50,
        scale=1,
        use_ddim=False,
        ref_dir="",
        use_mask=False,
        model_path="",
        save_dir="",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
