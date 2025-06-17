import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import zoom

class NPYDataset(Dataset):
    """
    Dataset class for loading .npy files for images and masks.
    Assumes the following naming convention:
    - Images: {id}.npy (single channel, shape: 1, depth, height, width)
    - Masks: {id}_seg.npy (class values: -3, 0, 1, 2, 3)
    Both images and masks are handled as float tensors for diffusion model training.
    """
    def __init__(self, data_dir, transform=None, mask_transform=None, target_size=(256, 256)):
        """
        Args:
            data_dir (str): Directory with all the .npy files
            transform (callable, optional): Optional transform to be applied on images
            mask_transform (callable, optional): Optional transform to be applied on masks
            target_size (tuple): Target size for resizing (height, width)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.target_size = target_size
        
        # Get all image files (excluding mask files)
        self.image_files = [f for f in os.listdir(data_dir) 
                          if f.endswith('.npy') and not f.endswith('_seg.npy')]
        
        # Sort files to ensure consistent ordering
        self.image_files.sort()
        
    def __len__(self):
        return len(self.image_files)
    
    def normalize_image(self, image):
        """Normalize image to [-1, 1] range."""
        # Clip outliers
        image = np.clip(image, -1, 1)
        return image
    
    def normalize_mask(self, mask):
        """
        Normalize mask while preserving class values.
        For diffusion model training, we keep the values as floats but ensure they're in the correct range.
        """
        # Ensure mask values are in the correct range
        mask = np.clip(mask, -3, 3)
        # Round to nearest valid value to ensure clean class boundaries
        mask = np.round(mask)
        return mask
    
    def resize_volume(self, volume, target_size):
        """Resize a 3D volume to target size."""
        # Get current dimensions
        c, h, w = volume.shape
        
        # Calculate scale factors
        scale_h = target_size[0] / h
        scale_w = target_size[1] / w
        
        # Resize using cubic interpolation for images, nearest neighbor for masks
        if volume.dtype in [np.int32, np.int64]:  # If it's a mask
            resized = zoom(volume, (1, scale_h, scale_w), order=0)  # order=0 for nearest neighbor
        else:  # If it's an image
            resized = zoom(volume, (1, scale_h, scale_w), order=3)  # order=3 for cubic interpolation
        return resized
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image filename
        img_filename = self.image_files[idx]
        
        # Construct mask filename
        mask_filename = img_filename.replace('.npy', '_seg.npy')
        
        # Load image and mask
        img_path = self.data_dir / img_filename
        mask_path = self.data_dir / mask_filename
        
        # Load numpy arrays
        image = np.load(img_path)  # Shape: (1, depth, height, width)
        mask = np.load(mask_path)  # Shape: (1, depth, height, width)
        
        # Verify shapes
        if image.shape[0] != 1:
            raise ValueError(f"Expected single channel image, got shape {image.shape}")
        
        # Resize to target size
        image = self.resize_volume(image[0], self.target_size)  # Remove batch dimension
        mask = self.resize_volume(mask[0], self.target_size)    # Remove batch dimension
        
        # Add batch dimension back
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        # Normalize
        image = self.normalize_image(image)
        mask = self.normalize_mask(mask)
        
        # Convert to torch tensors - both as float for diffusion model
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()  # Keep as float for diffusion model
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return {
            'image': image,
            'mask': mask,
            'filename': img_filename
        }

def get_npy_dataset(data_dir, batch_size=1, num_workers=4, transform=None, mask_transform=None, target_size=(256, 256)):
    """
    Helper function to create a DataLoader for the NPYDataset.
    
    Args:
        data_dir (str): Directory with all the .npy files
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of workers for the DataLoader
        transform (callable, optional): Optional transform to be applied on images
        mask_transform (callable, optional): Optional transform to be applied on masks
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = NPYDataset(data_dir, transform=transform, mask_transform=mask_transform, target_size=target_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def get_dataset(data_dir, batch_size=1, num_workers=4, use_npy=True, target_size=(256, 256)):
    """
    Get dataset with appropriate transforms.
    
    Args:
        data_dir (str): Directory containing the data
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of workers for the DataLoader
        use_npy (bool): Whether to use .npy files (True) or other formats (False)
        target_size (tuple): Target size for resizing (height, width)
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    if use_npy:
        return get_npy_dataset(
            data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=None,
            mask_transform=None,
            target_size=target_size
        )
    else:
        raise NotImplementedError("Only .npy format is currently supported")

if __name__ == "__main__":
    # Test the dataset
    data_dir = "path/to/your/data"
    
    # Create dataset
    dataset = NPYDataset(
        data_dir,
        transform=None,
        mask_transform=None,
        target_size=(256, 256)
    )
    
    # Get a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Filename: {sample['filename']}")
    
    # Create dataloader
    dataloader = get_dataset(
        data_dir,
        batch_size=4,
        num_workers=4,
        use_npy=True,
        target_size=(256, 256)
    )
    
    # Iterate through batches
    for batch in dataloader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch mask shape: {batch['mask'].shape}")
        break 