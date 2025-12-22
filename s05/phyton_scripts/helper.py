import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Toy dataset
# -----------------------------

def make_toy_dataset(n_per_class: int = 200, seed: int = 0):
    """
    Returns:
        X: torch.Tensor of shape [N, 2]
        y: torch.Tensor of shape [N] with values in {0, 1}
    """
    g = torch.Generator().manual_seed(seed)

    # Class 0: single Gaussian near (0, 0)
    mean0 = torch.tensor([0.0, 0.0])
    cov0 = torch.tensor([[1.0, 0.0],
                         [0.0, 1.0]])
    L0 = torch.linalg.cholesky(cov0)
    z0 = torch.randn(n_per_class, 2, generator=g)
    X0 = z0 @ L0.T + mean0
    y0 = torch.zeros(n_per_class, dtype=torch.long)

    # Class 1: mixture of two Gaussians, together n_per_class points
    n1a = n_per_class // 2
    n1b = n_per_class - n1a

    mean1a = torch.tensor([2.0, 2.0])
    mean1b = torch.tensor([-2.0, 2.0])
    cov1 = torch.tensor([[0.5, 0.0],
                         [0.0, 0.5]])
    L1 = torch.linalg.cholesky(cov1)

    z1a = torch.randn(n1a, 2, generator=g)
    z1b = torch.randn(n1b, 2, generator=g)
    X1a = z1a @ L1.T + mean1a
    X1b = z1b @ L1.T + mean1b
    X1 = torch.cat([X1a, X1b], dim=0)
    y1 = torch.ones(n_per_class, dtype=torch.long)

    # Stack and shuffle
    X = torch.cat([X0, X1], dim=0)          # [2*n_per_class, 2]
    y = torch.cat([y0, y1], dim=0)          # [2*n_per_class]

    N = X.shape[0]
    perm = torch.randperm(N, generator=g)
    X = X[perm]
    y = y[perm]

    return X, y

# -----------------------------
# Plotting
# -----------------------------

def plot_data_boundary_and_gradients(X, y, XX, YY, P, GX, GY, step=3):
    """
    Produces a figure with:
      - training data as colored scatter points
      - decision boundary p(y=1|x)=0.5 as a contour line
      - gradient field as arrows at a subset of grid points
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter plot of data
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
               s=20, alpha=0.7, label="Class 0")
    ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
               s=20, alpha=0.7, label="Class 1")

    # Decision boundary contour at p=0.5
    cs = ax.contour(XX, YY, P, levels=[0.5],
                    linewidths=2.0, linestyles="solid", colors="k")
    cs.collections[0].set_label("Decision boundary p(y=1|x)=0.5")

    # Gradient quiver plot, subsampled
    XX_sub = XX[::step, ::step]
    YY_sub = YY[::step, ::step]
    GX_sub = GX[::step, ::step]
    GY_sub = GY[::step, ::step]

    ax.quiver(XX_sub, YY_sub, GX_sub, GY_sub,
              angles="xy", scale_units="xy", scale=5.0, alpha=0.7)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", "box")
    ax.legend(loc="lower right")
    ax.set_title("Toy data, decision boundary, and input gradients")

    plt.tight_layout()
    plt.show()

