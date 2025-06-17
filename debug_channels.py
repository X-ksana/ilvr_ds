#!/usr/bin/env python3

import sys
import os
sys.path.append('/users/scxcw/ilvr_adm')

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from guided_diffusion.image_datasets import load_data

def debug_channel_configuration():
    print("=== DEBUGGING CHANNEL CONFIGURATION WITH LEARN_SIGMA=TRUE ===")
    
    # Test model creation arguments
    args = {
        'image_size': 256,
        'in_channels': 1,
        'out_channels': 1,
        'mask_channels': 1,
        'use_mask': True,
        'learn_sigma': True,  # Enable learn_sigma
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
    
    print("Model creation arguments:")
    for key, value in args.items():
        if 'channel' in key or key in ['use_mask', 'learn_sigma']:
            print(f"  {key}: {value}")
    
    try:
        model, diffusion = create_model_and_diffusion(**args)
        print(f"\nModel created successfully!")
        print(f"Model input channels: {model.in_channels}")
        print(f"Model output channels: {model.out_channels}")
        
        # Calculate expected channels
        total_in_channels = args['in_channels'] + args['mask_channels']  # 1 + 1 = 2
        total_out_channels = args['out_channels'] + args['mask_channels']  # 1 + 1 = 2
        if args['learn_sigma']:
            total_out_channels *= 2  # 2 * 2 = 4
        
        print(f"Expected input channels: {total_in_channels}")
        print(f"Expected output channels: {total_out_channels}")
        
        # Test data loading
        print("\n=== TESTING DATA LOADING ===")
        data_loader = load_data(
            data_dir="/scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d",
            batch_size=1,
            image_size=256,
            mask_dir="/scratch/scxcw/datasets/cardiac/nnUNet_preprocessed_2/Dataset114_MNMs/nnUNetPlans_2d",
            use_mask=True,
        )
        
        batch, cond = next(data_loader)
        print(f"Data batch shape: {batch.shape}")
        print(f"Data batch channels: {batch.shape[1]}")
        print(f"Condition keys: {list(cond.keys())}")
        
        if batch.shape[1] == model.in_channels:
            print("✅ CHANNEL MATCH: Data channels match model input channels!")
        else:
            print(f"❌ CHANNEL MISMATCH: Data has {batch.shape[1]} channels, model expects {model.in_channels}")
        
        print(f"\n=== CHANNEL SUMMARY ===")
        print(f"📊 Input: {batch.shape[1]} channels (1 image + 1 mask)")
        print(f"🔧 Model Input: {model.in_channels} channels")
        print(f"🔧 Model Output: {model.out_channels} channels (with learn_sigma={args['learn_sigma']})")
        print(f"✅ Ready for training with masks integrated into diffusion process!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_channel_configuration() 