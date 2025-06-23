#!/usr/bin/env python3
"""
Script to check the actual configuration of a saved model.
This will help us understand the channel configuration mismatch.
"""

import torch
import sys
import os

# Add the guided_diffusion directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'guided_diffusion'))

from guided_diffusion.script_util import create_model_and_diffusion

def check_model_config(model_path):
    """Check the configuration of a saved model."""
    
    print(f"Checking model configuration for: {model_path}")
    
    try:
        # Load the model state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        print("\n=== MODEL STATE DICT INFO ===")
        print(f"Keys in state dict: {list(state_dict.keys())}")
        
        # Check if it's a full checkpoint or just state dict
        if 'model_state_dict' in state_dict:
            model_state = state_dict['model_state_dict']
            print(f"Found 'model_state_dict' key")
        else:
            model_state = state_dict
            print(f"No 'model_state_dict' key, using direct state dict")
        
        # Look for the first convolution layer to determine input channels
        input_conv_key = None
        for key in model_state.keys():
            if 'input_blocks.0.0.weight' in key:
                input_conv_key = key
                break
        
        if input_conv_key:
            input_conv_weight = model_state[input_conv_key]
            print(f"\n=== INPUT CONVOLUTION LAYER ===")
            print(f"Layer: {input_conv_key}")
            print(f"Weight shape: {input_conv_weight.shape}")
            print(f"Input channels: {input_conv_weight.shape[1]}")
            print(f"Output channels: {input_conv_weight.shape[0]}")
            
            # Check output layer
            output_conv_key = None
            for key in model_state.keys():
                if 'out.2.weight' in key:
                    output_conv_key = key
                    break
            
            if output_conv_key:
                output_conv_weight = model_state[output_conv_key]
                print(f"\n=== OUTPUT CONVOLUTION LAYER ===")
                print(f"Layer: {output_conv_key}")
                print(f"Weight shape: {output_conv_weight.shape}")
                print(f"Input channels: {output_conv_weight.shape[1]}")
                print(f"Output channels: {output_conv_weight.shape[0]}")
        
        # Check for other important layers
        print(f"\n=== OTHER LAYERS ===")
        for key in model_state.keys():
            if 'weight' in key and 'conv' in key:
                weight = model_state[key]
                if len(weight.shape) == 4:  # Conv2d layer
                    print(f"{key}: {weight.shape}")
        
        # Check checkpoint info
        if 'step' in state_dict:
            print(f"\n=== CHECKPOINT INFO ===")
            print(f"Training step: {state_dict['step']}")
        
        if 'loss' in state_dict:
            print(f"Loss: {state_dict['loss']}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

def test_model_creation():
    """Test model creation with different configurations."""
    
    print("\n=== TESTING MODEL CREATIONS ===")
    
    # Test different configurations
    configs = [
        {
            'name': '2-channel (image+mask)',
            'in_channels': 2,
            'out_channels': 2,
            'mask_channels': 1,
            'use_mask': True,
            'learn_sigma': True
        },
        {
            'name': '3-channel (RGB)',
            'in_channels': 3,
            'out_channels': 3,
            'mask_channels': 1,
            'use_mask': False,
            'learn_sigma': True
        },
        {
            'name': '1-channel (grayscale)',
            'in_channels': 1,
            'out_channels': 1,
            'mask_channels': 1,
            'use_mask': False,
            'learn_sigma': True
        }
    ]
    
    for config in configs:
        try:
            print(f"\n--- Testing: {config['name']} ---")
            
            model, diffusion = create_model_and_diffusion(
                image_size=256,
                num_channels=128,
                num_res_blocks=2,
                learn_sigma=config['learn_sigma'],
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
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                mask_channels=config['mask_channels'],
                use_mask=config['use_mask']
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
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Check the actual model
    model_path = "results/log_dm/log_1000_mask/best_ssim_ema_model.pt"
    
    if os.path.exists(model_path):
        check_model_config(model_path)
    else:
        print(f"Model path not found: {model_path}")
        print("Please update the model_path variable to point to your actual model file.")
    
    # Test different configurations
    test_model_creation() 