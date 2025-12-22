import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

def run_experiment(image_path: str,
                   pool_k: int = 3,
                   blur_k: int = 5,
                   blur_sigma: float = 1.0) -> None:
    """
    Runs the experiment:
      - loads the image,
      - applies iterated max pooling and Gaussian blur
        for different numbers of iterations,
      - shows the results in a 2 x 5 grid.
    """
    

    x0 = load_image_as_tensor(image_path)  # [1, 1, H, W]

    iteration_list = [0, 1, 2, 4, 8]

    fig, axes = plt.subplots(
        2, len(iteration_list), figsize=(3 * len(iteration_list), 6)
    )

    for col, iters in enumerate(iteration_list):
        if iters == 0:
            x_mp = x0
            x_gb = x0
        else:
            x_mp = maxpool_iter(x0, k=pool_k, iters=iters)
            x_gb = gaussian_blur_iter(x0, k=blur_k, sigma=blur_sigma, iters=iters)

        # Maxpool row
        axes[0, col].imshow(tensor_to_numpy_image(x_mp),
                            cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"Maxpool, iters={iters}")
        axes[0, col].axis("off")

        # Gaussian blur row
        axes[1, col].imshow(tensor_to_numpy_image(x_gb),
                            cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"Gaussian, iters={iters}")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Max pooling", fontsize=12)
    axes[1, 0].set_ylabel("Gaussian blur", fontsize=12)

    plt.tight_layout()
    plt.show()