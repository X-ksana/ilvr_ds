import math
import random

from PIL import Image, ImageOps
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import nibabel as nib
from torchvision import transforms as T


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    keep_grayscale=True,
    mask_dir=None,
    use_mask=False,
    is_sampling=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param keep_grayscale: if True, keep images in grayscale format (1 channel).
    :param mask_dir: optional directory containing mask files with matching names.
    :param use_mask: if True, load and include mask data.
    :param is_sampling: if True, return image paths along with the data for sampling.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    
    # If mask_dir is provided and use_mask is True, verify matching mask files exist
    mask_files = None
    if mask_dir and use_mask:
        mask_files = {}
        for img_path in all_files:
            img_name = bf.basename(img_path)
            mask_path = bf.join(mask_dir, img_name)
            if not bf.exists(mask_path):
                raise ValueError(f"Missing mask file for {img_name}")
            mask_files[img_path] = mask_path

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        keep_grayscale=keep_grayscale,
        mask_files=mask_files,
        is_sampling=is_sampling,
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    List all image files in the directory recursively.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "nii.gz", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        keep_grayscale=True,
        mask_files=None,
        is_sampling=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = sorted(image_paths[shard:])[::num_shards]
        self.local_classes = None if classes is None else sorted(classes[shard:])[::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.keep_grayscale = keep_grayscale
        self.mask_files = mask_files
        self.is_sampling = is_sampling
        
        # Define valid mask values
        self.valid_mask_values = [-1, 0, 1, 2, 3]

    def normalize_image(self, image):
        """Normalize image to [-1, 1] range."""
        # Clip outliers
        image = np.clip(image, -1, 1)
        return image

    def normalize_mask(self, mask):
        """
        Normalize mask while preserving category values.
        Maps values to continuous range focusing on categories 1, 2, and 3.
        Treats -1 and 0 as background.
        """
        # Create a mapping for continuous values
        value_map = {
            -1: -1.0,  # Unknown -> background
            0: -1.0,   # Background
            1: -0.33,  # Category 1
            2: 0.33,   # Category 2
            3: 1.0     # Category 3
        }
        
        # Create a new array with the same shape
        normalized_mask = np.zeros_like(mask, dtype=np.float32)
        
        # Map each value
        for val, norm_val in value_map.items():
            normalized_mask[mask == val] = norm_val
            
        return normalized_mask

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        
        # Load image
        if path.endswith('.npy'):
            img_data = np.load(path)
            # Handle different array dimensions
            if len(img_data.shape) == 4:  # (1, 1, H, W)
                img_data = img_data[0, 0]  # Take first slice of first channel
            elif len(img_data.shape) == 3:  # (1, H, W)
                img_data = img_data[0]  # Take first slice
            elif len(img_data.shape) == 2:  # (H, W)
                img_data = img_data  # Use as is
            
            # Normalize to [-1, 1] range
            img_data = self.normalize_image(img_data)
            
            # Resize if needed
            if img_data.shape != (self.resolution, self.resolution):
                img_data = np.array(Image.fromarray(img_data).resize(
                    (self.resolution, self.resolution), 
                    Image.BICUBIC
                ))
            
            # Add channel dimension if needed
            if len(img_data.shape) == 2:
                img_data = np.expand_dims(img_data, axis=0)  # (1, H, W)
            
            # Convert to float tensor
            arr = img_data.astype(np.float32)
            
        else:
            # Handle other image formats if needed
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = ImageOps.exif_transpose(pil_image)
            
            if self.keep_grayscale and pil_image.mode in ['L', 'I']:
                pass
            else:
                pil_image = pil_image.convert("RGB")
            
            if self.random_crop:
                arr = random_crop_arr(pil_image, self.resolution)
            else:
                arr = center_crop_arr(pil_image, self.resolution)
            
            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]
            
            # Normalize to [-1, 1] range
            arr = arr.astype(np.float32) / 127.5 - 1
            
            # Handle channel dimension
            if len(arr.shape) == 2:
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = np.transpose(arr, [2, 0, 1])

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # Load and process mask if available
        if self.mask_files is not None:
            mask_path = self.mask_files[path]
            if mask_path.endswith('.npy'):
                mask_data = np.load(mask_path)
                # Handle different array dimensions for mask
                if len(mask_data.shape) == 4:  # (1, 1, H, W)
                    mask_data = mask_data[0, 0]  # Take first slice of first channel
                elif len(mask_data.shape) == 3:  # (1, H, W)
                    mask_data = mask_data[0]  # Take first slice
                elif len(mask_data.shape) == 2:  # (H, W)
                    mask_data = mask_data  # Use as is
                
                # Round mask values to nearest valid category
                rounded_mask = np.round(mask_data)
                # Clip values to valid range
                rounded_mask = np.clip(rounded_mask, -1, 3)
                
                # Verify mask values after rounding
                unique_values = np.unique(rounded_mask)
                invalid_values = set(unique_values) - set(self.valid_mask_values)
                if invalid_values:
                    raise ValueError(f"Invalid mask values found after rounding: {invalid_values}")
                
                # Resize mask if needed
                if rounded_mask.shape != (self.resolution, self.resolution):
                    rounded_mask = np.array(Image.fromarray(rounded_mask).resize(
                        (self.resolution, self.resolution), 
                        Image.NEAREST  # Use nearest neighbor for masks to preserve categories
                    ))
                
                # Normalize mask while preserving categories
                mask_data = self.normalize_mask(rounded_mask)
                
                # Add channel dimension
                mask_data = np.expand_dims(mask_data, axis=0)  # (1, H, W)
                
                # Apply random flip if needed
                if self.random_flip and random.random() < 0.5:
                    mask_data = mask_data[:, :, ::-1]
                
                # Concatenate image and mask along channel dimension
                arr = np.concatenate([arr, mask_data], axis=0)
                
                # Store original mask values for loss calculation (removed from out_dict)
                # out_dict["original_mask"] = rounded_mask

        if self.is_sampling:
            return arr, out_dict, path
        return arr, out_dict


def center_crop_arr(pil_image, image_size):
    """
    Center crop an image to the specified size.
    """
    # Resize the image while preserving the aspect ratio
    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=Image.LANCZOS)

    # Center crop to the target size
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    Randomly crop an image to the specified size.
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # Resize the image while preserving the aspect ratio
    scale = smaller_dim_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=Image.LANCZOS)

    # Random crop to the target size
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
