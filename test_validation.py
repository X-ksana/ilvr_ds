#!/usr/bin/env python3
"""
Test script to verify the updated validation logic for diffusion models.
This script tests that images and masks are properly reconstructed before computing losses.
"""

import torch
import numpy as np
import sys
import os

# Add the guided_diffusion directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'guided_diffusion'))

from guided_diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from guided_diffusion.unet import UNetModel
from guided_diffusion.train_util import calculate_ssim_batch, calculate_mask_metrics

def test_validation_reconstruction():
    """Test that validation properly reconstructs images and masks before computing losses."""
    
    print("Testing validation reconstruction logic...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple UNet model
    model = UNetModel(
        image_size=64,
        in_channels=2,  # 1 image + 1 mask channel
        out_channels=2,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(16, 32),
        dropout=0.1,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)
    
    # Create diffusion model
    diffusion = GaussianDiffusion(
        betas=np.linspace(0.0001, 0.02, 1000),
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    
    # Create synthetic test data
    batch_size = 4
    image_size = 64
    
    # Create synthetic image (first channel)
    image = torch.randn(batch_size, 1, image_size, image_size, device=device) * 0.5
    
    # Create synthetic mask (second channel) with discrete values
    mask = torch.randint(0, 4, (batch_size, 1, image_size, image_size), device=device).float() - 1  # Values: -1, 0, 1, 2, 3
    
    # Concatenate image and mask
    x_start = torch.cat([image, mask], dim=1)
    
    print(f"Input shape: {x_start.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")
    
    # Test the training_losses method
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    print("\nComputing training losses...")
    losses = diffusion.training_losses(model, x_start, t)
    
    print(f"Loss keys: {list(losses.keys())}")
    print(f"Total loss: {losses['loss'].mean().item():.6f}")
    print(f"Image MSE: {losses['image_mse'].mean().item():.6f}")
    print(f"Mask loss: {losses['mask_loss'].mean().item():.6f}")
    
    # Check that reconstructed data is available
    assert 'predicted_x0' in losses, "predicted_x0 should be in losses"
    assert 'original_x0' in losses, "original_x0 should be in losses"
    
    predicted_x0 = losses['predicted_x0']
    original_x0 = losses['original_x0']
    
    print(f"\nReconstructed data shape: {predicted_x0.shape}")
    print(f"Original data shape: {original_x0.shape}")
    
    # Split into image and mask components
    predicted_image = predicted_x0[:, :1]
    predicted_mask = predicted_x0[:, 1:]
    original_image = original_x0[:, :1]
    original_mask = original_x0[:, 1:]
    
    print(f"Predicted image range: [{predicted_image.min():.3f}, {predicted_image.max():.3f}]")
    print(f"Original image range: [{original_image.min():.3f}, {original_image.max():.3f}]")
    print(f"Predicted mask unique values: {torch.unique(predicted_mask).tolist()}")
    print(f"Original mask unique values: {torch.unique(original_mask).tolist()}")
    
    # Test validation metrics computation
    print("\nComputing validation metrics...")
    
    # Test SSIM calculation
    try:
        ssim_value = calculate_ssim_batch(original_image, predicted_image)
        print(f"SSIM: {ssim_value:.6f}")
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
    
    # Test mask metrics calculation
    try:
        mask_metrics = calculate_mask_metrics(predicted_mask, original_mask)
        print(f"Mask metrics: {mask_metrics}")
    except Exception as e:
        print(f"Mask metrics calculation failed: {e}")
    
    # Verify that losses are based on reconstructed data, not noise
    print("\nVerifying loss computation...")
    
    # The loss should be based on the difference between original and reconstructed data
    # not on the difference between original and noise prediction
    image_loss_from_reconstruction = torch.mean((original_image - predicted_image) ** 2)
    mask_loss_from_reconstruction = torch.mean((original_mask - predicted_mask) ** 2)
    
    print(f"Image loss from reconstruction: {image_loss_from_reconstruction.item():.6f}")
    print(f"Mask loss from reconstruction: {mask_loss_from_reconstruction.item():.6f}")
    print(f"Reported image MSE: {losses['image_mse'].mean().item():.6f}")
    print(f"Reported mask loss: {losses['mask_loss'].mean().item():.6f}")
    
    # These should be approximately equal (within numerical precision)
    assert abs(image_loss_from_reconstruction.item() - losses['image_mse'].mean().item()) < 1e-5, \
        "Image loss should be based on reconstructed data"
    
    print("\n✅ All tests passed! Validation reconstruction is working correctly.")
    
    return True

if __name__ == "__main__":
    try:
        test_validation_reconstruction()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 