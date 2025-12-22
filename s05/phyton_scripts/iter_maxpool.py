import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from helper_maxpool import run_experiment


def maxpool_iter(x: torch.Tensor, k: int = 3, iters: int = 1) -> torch.Tensor:
    """
    Apply 2D max pooling with kernel size k, stride 1, and padding k//2
    repeatedly.

    Args:
        x: input tensor of shape [B, C, H, W]
        k: pooling kernel size
        iters: how many times to apply pooling

    Returns:
        Tensor of same shape as x
    """
    # TODO: repeat max pooling 'iters' times.
    raise NotImplementedError("maxpool_iter is not implemented yet.")


def gauss_kernel_2d(k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Create a 2D isotropic Gaussian kernel of size k x k.

    The kernel should:
    - have shape [k, k],
    - and be normalised so that its entries sum to 1.
    """
    # TODO:
    # 1) create a 1D axis: ax = torch.arange(k) - (k - 1) / 2.0
    # 2) use torch.meshgrid to get coordinates xx, yy
    # 3) apply the Gaussian formula: exp(-(xx^2 + yy^2) / (2*sigma^2))
    # 4) normalise so that kernel.sum() == 1
    raise NotImplementedError("gauss_kernel_2d is not implemented yet.")


def gaussian_blur_iter(x: torch.Tensor,
                       k: int = 5,
                       sigma: float = 1.0,
                       iters: int = 1) -> torch.Tensor:
    """
    Apply Gaussian blur 'iters' times using a fixed 2D kernel.

    Args:
        x: input tensor of shape [B, C, H, W]
        k: kernel size (odd)
        sigma: standard deviation of the Gaussian
        iters: how many times to apply blur

    Returns:
        Tensor of same shape as x
    """
    # TODO:
    # 1) build kernel = gauss_kernel_2d(k, sigma)
    # 2) reshape to [out_channels, in_channels, kH, kW] with out_channels=in_channels=1
    # 3) loop 'iters' times and apply F.conv2d with appropriate padding
    raise NotImplementedError("gaussian_blur_iter is not implemented yet.")


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Load an image with matplotlib, convert to grayscale float32
    in [0, 1], and wrap as [1, 1, H, W] tensor.
    """
    img = mpimg.imread(image_path)        # shape [H, W] or [H, W, C]
    img = np.asarray(img)

    # If RGB, average channels to get grayscale
    if img.ndim == 3:
        img = img.mean(axis=2)            # shape [H, W]

    img = img.astype(np.float32)

    # TODO: normalise to [0, 1]
    # TODO: convert to a torch tensor 
    # and reshape to [1, 1, H, W]
    raise NotImplementedError("load_image_as_tensor is not implemented yet.")


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert [1, 1, H, W] tensor to 2D numpy image [H, W].
    """
    raise NotImplementedError("tensor_to_numpy_image is not implemented yet.")




if __name__ == "__main__":
    IMAGE_PATH = "sample_image.png"

    run_experiment(IMAGE_PATH)
