import torch
import torchvision
from torch import nn
from typing import Tensor, Optional, Dict, Tuple


@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:] # just gain H and W
    min_size = torch.min(im_shape).to(dtype=torch.float32) # convert to float32
    max_size = torch.max(im_shape).to(dtype=torch.float32) # convert to float32
    # calculate scale factor
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)
    
    

def _resize_image(image, min_size, max_size):
    pass

class GeneralizedRCNNTransform(nn.Module):
    """ 
    This class is used to transform the input image to the format expected by the model.
    The transform consists of resizing the image to the given size and normalizing it with
    the given mean and standard deviation.
    """
    def __init__(self, min_size, max_size, image_mean, image_std):
        """
        Args:
            min_size (int): minimum size of the image to be rescaled
            max_size (int): maximum size of the image to be rescaled
            image_mean (tuple[float]): mean values used for input normalization
            image_std (tuple[float]): standard deviation values used for input normalization
        """
        super(GeneralizedRCNNTransform, self).__init__()
        # judge whether min_size is tuple or list, if not, convert it to tuple
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        """
        Normalize an image with mean and standard deviation.
        Args:
            image (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # [:, None, None] is used to expand the dimension of mean and std
        # first expand the dimension, then broadcast
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def torch_choice(self, k):
        # type: (int) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with TorchScript.
        """
        index = torch.randint(len(k), (1,)).item()
        return k[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        Resize the input image to the given size.
        Args:
            image (Tensor): Tensor image of size (C, H, W) to be resized.
            target (Dict or None): Target to be resized.
        Returns:
            Tensor: Resized image.
            Dict or None: Resized target.
        """
        # image shape is [channel, height, width]
        w, h = image.shape[-2:]
        # if in training mode, randomly choose a size from min_size
        # if in testing mode, choose the smallest size from min_size
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        # resize the image to the given size
        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image, size, float(self.max_size))