#!/usr/bin/env python3
"""
Debug script to analyze tensor shapes and data flow in the training pipeline.
This will help identify the exact cause of the shape mismatch error.
"""

import torch
import sys
import os
import numpy as np

# Add the guided_diffusion directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'guided_diffusion'))

from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import TrainLoop
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler

def debug_tensor_shapes():
    """Debug tensor shapes throughout the training pipeline."""
    
    print("=== DEBUGGING TENSOR SHAPES ===")
    
    # Create model with the same configuration as training
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
    
    print(f"✅ Model created successfully!")
    print(f"   Model input channels: {model.in_channels}")
    print(f"   Model output channels: {model.out_channels}")
    print(f"   Diffusion learn_sigma: {diffusion.model_var_type}")
    
    # Create a test batch
    batch_size = 2
    test_batch = torch.randn(batch_size, 2, 256, 256)  # 2 channels: image + mask
    test_timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"\n=== TEST BATCH SHAPES ===")
    print(f"   Test batch shape: {test_batch.shape}")
    print(f"   Test timesteps shape: {test_timesteps.shape}")
    print(f"   Test timesteps values: {test_timesteps}")
    
    # Test model forward pass
    print(f"\n=== MODEL FORWARD PASS ===")
    with torch.no_grad():
        model_output = model(test_batch, test_timesteps)
        print(f"   Model output shape: {model_output.shape}")
        
        # Check if learn_sigma is working correctly
        if diffusion.model_var_type in ['LEARNED', 'LEARNED_RANGE']:
            expected_channels = test_batch.shape[1] * 2  # 2 * C channels
            print(f"   Expected output channels (learn_sigma): {expected_channels}")
            print(f"   Actual output channels: {model_output.shape[1]}")
            
            if model_output.shape[1] == expected_channels:
                print(f"   ✅ Model output channels match expected (learn_sigma)")
                # Split into epsilon and variance
                C = test_batch.shape[1]
                epsilon_pred, variance_pred = torch.split(model_output, C, dim=1)
                print(f"   Epsilon prediction shape: {epsilon_pred.shape}")
                print(f"   Variance prediction shape: {variance_pred.shape}")
            else:
                print(f"   ❌ Model output channels mismatch!")
    
    # Test diffusion training_losses method
    print(f"\n=== DIFFUSION TRAINING LOSSES ===")
    try:
        losses = diffusion.training_losses(model, test_batch, test_timesteps)
        print(f"   ✅ Training losses computed successfully!")
        print(f"   Loss keys: {list(losses.keys())}")
        
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key} shape: {value.shape}")
                print(f"   {key} dtype: {value.dtype}")
                print(f"   {key} device: {value.device}")
            else:
                print(f"   {key}: {type(value)}")
                
    except Exception as e:
        print(f"   ❌ Error in training_losses: {e}")
        import traceback
        traceback.print_exc()
    
    # Test _extract_into_tensor function specifically
    print(f"\n=== _EXTRACT_INTO_TENSOR DEBUG ===")
    try:
        # Test with the specific arrays that might cause issues
        arr = diffusion.sqrt_recip_alphas_cumprod
        broadcast_shape = test_batch.shape
        
        print(f"   Array shape: {arr.shape}")
        print(f"   Array dtype: {arr.dtype}")
        print(f"   Broadcast shape: {broadcast_shape}")
        print(f"   Timesteps: {test_timesteps}")
        
        # Test the _extract_into_tensor function
        from guided_diffusion.gaussian_diffusion import _extract_into_tensor
        
        result = _extract_into_tensor(arr, test_timesteps, broadcast_shape)
        print(f"   ✅ _extract_into_tensor result shape: {result.shape}")
        print(f"   ✅ _extract_into_tensor result dtype: {result.dtype}")
        print(f"   ✅ _extract_into_tensor result device: {result.device}")
        
        # Test the other array
        arr2 = diffusion.sqrt_recipm1_alphas_cumprod
        result2 = _extract_into_tensor(arr2, test_timesteps, broadcast_shape)
        print(f"   ✅ Second array result shape: {result2.shape}")
        
    except Exception as e:
        print(f"   ❌ Error in _extract_into_tensor: {e}")
        import traceback
        traceback.print_exc()
    
    # Test data loading
    print(f"\n=== DATA LOADING DEBUG ===")
    try:
        # This would require actual data paths, so we'll just check the data loading function
        print("   Note: Data loading test skipped (requires actual data paths)")
        print("   To test data loading, provide actual data_dir and mask_dir paths")
        
    except Exception as e:
        print(f"   ❌ Error in data loading: {e}")
        import traceback
        traceback.print_exc()

def debug_specific_error():
    """Debug the specific error mentioned in the traceback."""
    
    print("\n=== DEBUGGING SPECIFIC ERROR ===")
    print("Error: 'The size of tensor a (256) must match the size of tensor b (4) at non-singleton dimension 3'")
    
    # This error suggests a broadcasting issue in tensor operations
    # Let's check what could cause this specific shape mismatch
    
    print("\nPossible causes:")
    print("1. Tensor broadcasting issue in _extract_into_tensor")
    print("2. Shape mismatch in model output processing")
    print("3. Incorrect tensor dimensions in loss calculation")
    
    # Create a minimal reproduction
    print("\n=== MINIMAL REPRODUCTION ===")
    
    # Simulate the problematic operation
    try:
        # Create tensors with the problematic shapes
        tensor_a = torch.randn(1, 2, 256, 256)  # Shape with 256
        tensor_b = torch.randn(1, 2, 256, 4)    # Shape with 4
        
        print(f"   Tensor A shape: {tensor_a.shape}")
        print(f"   Tensor B shape: {tensor_b.shape}")
        
        # Try to perform an operation that might cause the error
        result = tensor_a * tensor_b
        print(f"   ✅ Broadcasting successful: {result.shape}")
        
    except Exception as e:
        print(f"   ❌ Broadcasting failed: {e}")
    
    # Test with different shapes
    try:
        tensor_a = torch.randn(1, 2, 256, 256)
        tensor_b = torch.randn(1, 2, 256, 1)  # Singleton dimension
        
        result = tensor_a * tensor_b
        print(f"   ✅ Broadcasting with singleton successful: {result.shape}")
        
    except Exception as e:
        print(f"   ❌ Broadcasting with singleton failed: {e}")

if __name__ == "__main__":
    debug_tensor_shapes()
    debug_specific_error()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Check if the model is outputting the correct number of channels")
    print("2. Verify that learn_sigma=True is working correctly")
    print("3. Ensure tensor broadcasting in _extract_into_tensor is correct")
    print("4. Check if there are any shape mismatches in the loss calculation")
    print("5. Verify that the data loading is providing the expected shapes") 