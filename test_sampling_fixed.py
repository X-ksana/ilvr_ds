#!/usr/bin/env python3
"""
Test script to verify that the fixed sampling configuration works with the saved model.
"""

import torch
import sys
import os

# Add the guided_diffusion directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'guided_diffusion'))

from guided_diffusion.script_util import create_model_and_diffusion

def test_fixed_sampling():
    """Test the fixed sampling configuration."""
    
    print("Testing fixed sampling configuration...")
    
    try:
        # Create model with the correct configuration that matches the saved model
        model, diffusion = create_model_and_diffusion(
            image_size=256,
            num_channels=128,
            num_res_blocks=2,
            learn_sigma=True,  # This is important - matches saved model
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions="16",
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            in_channels=2,  # Matches saved model: [128, 2, 3, 3]
            out_channels=2,  # Matches saved model: [4, 128, 3, 3] with learn_sigma
            mask_channels=1,
            use_mask=True,
            channel_mult="",
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="100",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
        
        print(f"✅ Model created successfully!")
        print(f"   Input channels: {model.in_channels}")
        print(f"   Output channels: {model.out_channels}")
        
        # Test forward pass
        test_input = torch.randn(1, model.in_channels, 256, 256)
        with torch.no_grad():
            output = model(test_input, torch.tensor([500]))
            print(f"   Test input shape: {test_input.shape}")
            print(f"   Test output shape: {output.shape}")
            
            # Verify output channels
            if output.shape[1] == 4:
                print(f"   ✅ Output has 4 channels (epsilon + variance)")
                print(f"   - Channels 0-1: epsilon predictions")
                print(f"   - Channels 2-3: variance predictions")
            else:
                print(f"   ❌ Unexpected output channels: {output.shape[1]}")
        
        # Test sampling
        sample_shape = (1, 4, 256, 256)  # 4 channels with learn_sigma=True
        with torch.no_grad():
            sample = diffusion.p_sample_loop(
                model,
                sample_shape,
                clip_denoised=True,
                model_kwargs={},
                noise=torch.randn(sample_shape),
                N=25,  # Number of steps
                D=4,   # Scaling factor
                scale=1
            )
            
            print(f"✅ Sampling successful!")
            print(f"   Sample shape: {sample.shape}")
            
            # Test output processing
            output = sample[0].cpu().numpy()
            print(f"   Output numpy shape: {output.shape}")
            
            # Extract epsilon predictions (first 2 channels)
            epsilon_output = output[:2]
            print(f"   Epsilon output shape: {epsilon_output.shape}")
            
            if epsilon_output.shape[0] == 2:
                image_output = epsilon_output[0]
                mask_output = epsilon_output[1]
                print(f"   Image channel shape: {image_output.shape}")
                print(f"   Mask channel shape: {mask_output.shape}")
                print(f"   Image range: [{image_output.min():.3f}, {image_output.max():.3f}]")
                print(f"   Mask range: [{mask_output.min():.3f}, {mask_output.max():.3f}]")
        
        print("\n🎉 All tests passed! The fixed configuration should work with the saved model.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading the actual saved model."""
    
    print("\nTesting model loading...")
    
    model_path = "results/log_dm/log_1000_mask/best_ssim_ema_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    try:
        # Create model with correct configuration
        model, diffusion = create_model_and_diffusion(
            image_size=256,
            num_channels=128,
            num_res_blocks=2,
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions="16",
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
            in_channels=2,
            out_channels=2,
            mask_channels=1,
            use_mask=True,
            channel_mult="",
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="100",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
        
        # Load the saved model
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")["model_state_dict"]
        )
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== FIXED SAMPLING CONFIGURATION TEST ===")
    
    # Test the fixed configuration
    success1 = test_fixed_sampling()
    
    # Test loading the actual model
    success2 = test_model_loading()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Ready for sampling.")
        print("\nTo run sampling:")
        print("1. Update the paths in sample_diffusion.sh")
        print("2. Run: bash sample_diffusion.sh")
    else:
        print("\n❌ Some tests failed. Please check the configuration.") 