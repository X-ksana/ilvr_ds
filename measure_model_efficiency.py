import torch
from guided_diffusion.script_util import create_model
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from model_metrics import calculate_model_efficiency, print_efficiency_metrics

def main():
    # Model configuration
    image_size = 256
    num_channels = 128
    num_res_blocks = 2
    attention_resolutions = "16,8,4"
    num_heads = 4
    num_head_channels = 64
    use_checkpoint = False
    use_fp16 = False
    
    # Create model
    model = create_model(
        image_size=image_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Input shape for metrics calculation
    batch_size = 1
    input_shape = (batch_size, 3, image_size, image_size)  # 3 channels for RGB
    
    # Number of diffusion timesteps
    num_timesteps = 1000
    
    # Calculate efficiency metrics
    metrics = calculate_model_efficiency(
        model=model,
        input_shape=input_shape,
        num_timesteps=num_timesteps,
        device=device
    )
    
    # Print metrics
    print_efficiency_metrics(metrics)

if __name__ == "__main__":
    main() 