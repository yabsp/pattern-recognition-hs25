import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

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
    b = x
    for i in range(iters):
        b = F.max_pool2d(b, kernel_size=k, stride=1, padding=k // 2)
    return b


def gauss_kernel_2d(k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Create a 2D isotropic Gaussian kernel of size k x k.

    The kernel should:
    - have shape [k, k],
    - and be normalised so that its entries sum to 1.
    """
    # 1) create a 1D axis: ax = torch.arange(k) - (k - 1) / 2.0
    # 2) use torch.meshgrid to get coordinates xx, yy
    # 3) apply the Gaussian formula: exp(-(xx^2 + yy^2) / (2*sigma^2))
    # 4) normalise so that kernel.sum() == 1

    ax = torch.arange(k) - (k - 1) / 2.0
    xx,yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))

    return kernel / kernel.sum()

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
    # 1) build kernel = gauss_kernel_2d(k, sigma)
    # 2) reshape to [out_channels, in_channels, kH, kW] with out_channels=in_channels=1
    # 3) loop 'iters' times and apply F.conv2d with appropriate padding
    _, _ , h, w = x.shape
    kernel = gauss_kernel_2d(k, sigma)
    kernel = kernel.view(1, 1, k, k)
    blurred = x
    for i in range(iters):
        blurred = F.conv2d(blurred, kernel, stride=1, padding=(k-1)//2)

    return blurred

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

    # and reshape to [1, 1, H, W]
    img = img
    img = torch.from_numpy(img)
    h, w = img.shape
    img = img.view(1, 1, h, w)
    return img

def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert [1, 1, H, W] tensor to 2D numpy image [H, W].
    """
    _, _, h, w = x.shape
    x = x.view(h, w)
    return x.numpy()




if __name__ == "__main__":
    from helper_maxpool import run_experiment
    IMAGE_PATH = "sample_image.png"

    run_experiment(IMAGE_PATH)
