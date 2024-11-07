from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

class ResizeStrategy(Enum):
    """
    Enumeration of different resize strategies
    """
    RESIZE = "resize"  # Simple resize
    PAD = "pad"       # Pad to square
    CROP = "crop"     # Center crop
    SCALE = "scale"   # Scale to fit

class IrregularImagePreprocessor:
    """
    Preprocessor for handling irregular image sizes
    """
    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = 224,
        resize_strategy: ResizeStrategy = ResizeStrategy.SCALE,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        pad_value: int = 0,
        keep_aspect_ratio: bool = True
    ):
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.resize_strategy = resize_strategy
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]
        self.pad_value = pad_value
        self.keep_aspect_ratio = keep_aspect_ratio
        
        self._setup_transforms()
    
    def _setup_transforms(self):
        """
        Set up basic transformation pipeline
        """
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        # Training augmentations
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(15),
            transforms.RandomGrayscale(p=0.2)
        ])
    
    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """
        Resize image and pad to maintain aspect ratio
        """
        original_size = image.size
        ratio = min(self.target_size[0] / original_size[0], 
                   self.target_size[1] / original_size[1])
        
        new_size = tuple([int(dim * ratio) for dim in original_size])
        image = image.resize(new_size, Image.BILINEAR)
        
        # Calculate padding
        delta_w = self.target_size[0] - new_size[0]
        delta_h = self.target_size[1] - new_size[1]
        padding = (delta_w//2, delta_h//2, 
                  delta_w-(delta_w//2), delta_h-(delta_h//2))
        
        return transforms.Pad(padding, fill=self.pad_value)(image)
    
    def _resize_with_crop(self, image: Image.Image) -> Image.Image:
        """
        Resize image and center crop
        """
        # Resize to larger dimension
        ratio = max(self.target_size[0] / image.size[0], 
                   self.target_size[1] / image.size[1])
        new_size = tuple([int(dim * ratio) for dim in image.size])
        image = image.resize(new_size, Image.BILINEAR)
        
        # Center crop
        return transforms.CenterCrop(self.target_size)(image)
    
    def _scale_to_fit(self, image: Image.Image) -> Image.Image:
        """
        Scale image to fit target size while maintaining aspect ratio
        """
        if self.keep_aspect_ratio:
            ratio = min(self.target_size[0] / image.size[0], 
                       self.target_size[1] / image.size[1])
            new_size = tuple([int(dim * ratio) for dim in image.size])
            return image.resize(new_size, Image.BILINEAR)
        else:
            return image.resize(self.target_size, Image.BILINEAR)
    
    def __call__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        is_training: bool = False, 
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess a single image using the specified strategy
        """
        # Load image if necessary
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply resize strategy
        if self.resize_strategy == ResizeStrategy.RESIZE:
            image = image.resize(self.target_size, Image.BILINEAR)
        elif self.resize_strategy == ResizeStrategy.PAD:
            image = self._resize_with_padding(image)
        elif self.resize_strategy == ResizeStrategy.CROP:
            image = self._resize_with_crop(image)
        elif self.resize_strategy == ResizeStrategy.SCALE:
            image = self._scale_to_fit(image)
        
        # Apply augmentations during training
        if is_training:
            image = self.train_transforms(image)
        
        # Convert to tensor and normalize
        image = transforms.ToTensor()(image)
        if normalize:
            image = self.normalize(image)
        
        return image
