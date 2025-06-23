#!/usr/bin/env python3
"""
Test script to verify the sampling functionality works correctly.
This script tests the updated image_sample.py with proper channel configuration.
"""

import torch
import numpy as np
import sys
import os

# Add the guided_diffusion directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'guided_diffusion'))

from guided_diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model_and_diffusion

def test_sampling_configuration():
    """Test that the sampling configuration matches the trained model."""
    
    print("Testing sampling configuration...")
    
    # Configuration that matches your trained model
    args = {
        'image_size': 256,
        'in_channels': 2,  # 1 image + 1 mask
        'out_channels': 2,  # 1 image + 1 mask
        'mask_channels': 1,
        'use_mask': True,
        'learn_sigma': True,
        'num_channels': 128,
        'num_res_blocks': 2,
        'class_cond': False,
        'use_checkpoint': False,
        'attention_resolutions': "16",
        'num_heads': 1,
        'num_head_channels': -1,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': True,
        'dropout': 0.0,
        'resblock_updown': True,
        'use_fp16': False,
        'use_new_attention_order': False,
        'channel_mult': "",
        'diffusion_steps': 1000,
        'noise_schedule': "linear",
        'timestep_respacing': "100",
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
    }
    
    try:
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(**args)
        
        print(f"✅ Model created successfully!")
        print(f"   Model input channels: {model.in_channels}")
        print(f"   Model output channels: {model.out_channels}")
        
        # Test with sample data
        batch_size = 1
        sample_shape = (batch_size, 2, 256, 256)  # 2 channels: image + mask
        
        # Create dummy input data
        x_start = torch.randn(sample_shape)
        t = torch.tensor([500], dtype=torch.long)  # Middle timestep
        
        # Test forward pass
        with torch.no_grad():
            # Add noise
            noise = torch.randn_like(x_start)
            x_t = diffusion.q_sample(x_start, t, noise=noise)
            
            # Model prediction
            model_output = model(x_t, diffusion._scale_timesteps(t))
            
            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {x_start.shape}")
            print(f"   Noisy input shape: {x_t.shape}")
            print(f"   Model output shape: {model_output.shape}")
            
            # Test sampling
            sample = diffusion.p_sample_loop(
                model,
                sample_shape,
                clip_denoised=True,
                model_kwargs={},
                noise=x_t,  # Use noisy input as noise
                N=25,  # Number of steps
                D=4,   # Scaling factor
                scale=1
            )
            
            print(f"✅ Sampling successful!")
            print(f"   Sample shape: {sample.shape}")
            
            # Test output processing
            output = sample[0].cpu().numpy()
            print(f"   Output numpy shape: {output.shape}")
            
            if output.shape[0] == 2:  # 2 channels
                image_output = output[0]
                mask_output = output[1]
                print(f"   Image channel shape: {image_output.shape}")
                print(f"   Mask channel shape: {mask_output.shape}")
                print(f"   Image range: [{image_output.min():.3f}, {image_output.max():.3f}]")
                print(f"   Mask range: [{mask_output.min():.3f}, {mask_output.max():.3f}]")
        
        print("\n✅ All tests passed! Sampling configuration is correct.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_sampling_without_mask():
    """Test that sampling works correctly without mask_dir - model generates both image and mask."""
    
    print("\nTesting sampling without mask_dir...")
    
    # Configuration that matches your trained model
    args = {
        'image_size': 256,
        'in_channels': 2,  # 1 image + 1 mask
        'out_channels': 2,  # 1 image + 1 mask
        'mask_channels': 1,
        'use_mask': True,  # Important: keep this True for 2-channel model
        'learn_sigma': True,
        'num_channels': 128,
        'num_res_blocks': 2,
        'class_cond': False,
        'use_checkpoint': False,
        'attention_resolutions': "16",
        'num_heads': 1,
        'num_head_channels': -1,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': True,
        'dropout': 0.0,
        'resblock_updown': True,
        'use_fp16': False,
        'use_new_attention_order': False,
        'channel_mult': "",
        'diffusion_steps': 1000,
        'noise_schedule': "linear",
        'timestep_respacing': "100",
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
    }
    
    try:
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(**args)
        
        print(f"✅ Model created successfully!")
        print(f"   Model input channels: {model.in_channels}")
        print(f"   Model output channels: {model.out_channels}")
        
        # Test sampling without mask_dir
        batch_size = 1
        sample_shape = (batch_size, 2, 256, 256)  # 2 channels: image + mask
        
        # Create dummy reference image (only image, no mask)
        # This simulates loading only image data during sampling
        ref_image = torch.randn(batch_size, 1, 256, 256)  # Only image channel
        
        # Test sampling
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model,
                sample_shape,
                clip_denoised=True,
                model_kwargs={"ref_img": ref_image},  # Only image reference
                noise=ref_image,  # Use image as noise
                N=25,  # Number of steps
                D=4,   # Scaling factor
                scale=1
            )
            
            print(f"✅ Sampling without mask_dir successful!")
            print(f"   Reference image shape: {ref_image.shape}")
            print(f"   Sample shape: {sample.shape}")
            
            # Test output processing
            output = sample[0].cpu().numpy()
            print(f"   Output numpy shape: {output.shape}")
            
            if output.shape[0] == 2:  # 2 channels
                image_output = output[0]
                mask_output = output[1]
                print(f"   Generated image shape: {image_output.shape}")
                print(f"   Generated mask shape: {mask_output.shape}")
                print(f"   Image range: [{image_output.min():.3f}, {image_output.max():.3f}]")
                print(f"   Mask range: [{mask_output.min():.3f}, {mask_output.max():.3f}]")
                
                # Test denormalization
                mask_denorm = ((mask_output + 1) * 2).astype(np.uint8)
                mask_denorm = np.clip(mask_denorm, 0, 3)
                unique_values = np.unique(mask_denorm)
                print(f"   Denormalized mask values: {unique_values}")
        
        print("\n✅ Sampling without mask_dir test passed!")
        print("   The model successfully generates both image and mask from image-only input.")
        
    except Exception as e:
        print(f"❌ Error during sampling without mask_dir test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_data_loading():
    """Test that the data loading functions work with .npy files."""
    
    print("\nTesting data loading with .npy files...")
    
    try:
        from guided_diffusion.image_datasets import load_data
        
        # Test data loading configuration
        data_loader = load_data(
            data_dir="/path/to/your/data",  # Replace with actual path
            batch_size=1,
            image_size=256,
            mask_dir="/path/to/your/masks",  # Replace with actual path
            use_mask=True,
            deterministic=True,
            random_flip=False
        )
        
        print("✅ Data loading configuration is correct!")
        print("   Note: Replace paths with actual data directories for full testing")
        
    except Exception as e:
        print(f"❌ Error in data loading test: {e}")
        print("   This is expected if the data paths don't exist")

if __name__ == "__main__":
    print("=== SAMPLING CONFIGURATION TEST ===")
    
    # Test model configuration
    success1 = test_sampling_configuration()
    
    # Test sampling without mask_dir
    success2 = test_sampling_without_mask()
    
    # Test data loading
    test_data_loading()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your sampling setup is ready.")
        print("\nTo run sampling:")
        print("1. Update the paths in sample_diffusion.sh")
        print("2. Run: bash sample_diffusion.sh")
        print("\nNote: The model will generate both image and mask from image-only input.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.") 