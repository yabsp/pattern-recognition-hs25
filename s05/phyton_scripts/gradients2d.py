import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from helper import make_toy_dataset, plot_data_boundary_and_gradients


class MLP2d(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # two class scores
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [B, 2]
        returns: tensor of shape [B, 2] with class scores (logits)
        """
        # TODO: implement forward pass
        raise NotImplementedError


def train_classifier(model, X, y, epochs=200, lr=1e-2, batch_size=64):
    """
    Trains model on (X, y) using cross entropy loss.
    """
    model.train()
    # TODO: implement training loop

    # Compute training accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    print(f"Training accuracy: {acc * 100:.1f}%")

    return acc


def evaluate_on_grid(model,
                     x_min=-4.0, x_max=4.0,
                     y_min=-4.0, y_max=4.0,
                     steps=50):
    """
    Evaluates p_theta(y=1 | x) on a regular grid.

    Returns:
        XX, YY: numpy arrays of shape [steps, steps] with grid coordinates
        P: numpy array of shape [steps, steps] with probabilities p(y=1 | x)
        xs, ys: 1D torch tensors of length steps (useful for gradients)
    """
    model.eval()

    xs = 
    ys = 

    XX, YY =    # shape [steps, steps]
    grid =      # [steps*steps, 2]
    
    #compute class scores for all grid points
    with torch.no_grad():
        logits =                      # [steps*steps, 2]
        probs =                       # [steps*steps, 2]
        p1 =                          # [steps*steps]
    #reshape 
    P =                 # [steps, steps]
    P = P.cpu().numpy() #convert to numpy 
    XX_np = XX.cpu().numpy()
    YY_np = YY.cpu().numpy()

    return XX_np, YY_np, P, xs, ys


def compute_input_gradients(model, xs, ys):
    """
    xs, ys: 1D torch tensors of length "steps" defining the grid.

    Returns:
        GX, GY: numpy arrays of shape [steps, steps]
            GX[i, j] = d p(y=1|x) / d x1 at grid point (i, j)
            GY[i, j] = d p(y=1|x) / d x2 at grid point (i, j)
    """
    model.eval()

    steps = xs.numel()
    XX, YY =      # [steps, steps]
    grid =        # [steps*steps, 2]
    
    #enable gradient tracking here:

    #pass grid through the model to obtain scores
    logits =                            # [steps*steps, 2]
    #apply softmax
    probs =                             # [steps*steps, 2]
    #extract the probability for class 1
    p1 =                                # [steps*steps]

    #compute the gradient of the sum of these probabilites wrt grid
    #i) call model.zero_grad()
    #ii) call p1.sum().backward()
    #iii) 

    grads = grid.grad                    # [steps*steps, 2]
    grads = grads.reshape(steps, steps, 2) # [steps, steps, 2]

    GX =                                        # d p / d x1
    GY =                                        # d p / d x2

    GX = GX.detach().cpu().numpy()
    GY = GY.detach().cpu().numpy()
    
    return GX, GY


if __name__ == "__main__":
    # Generate data
    X, y = make_toy_dataset(n_per_class=200, seed=0)

    # Train classifier
    model = MLP2d(hidden_dim=16)
    train_classifier(model, X, y, epochs=200, lr=1e-2, batch_size=64)

    # Grid evaluation
    XX, YY, P, xs, ys = evaluate_on_grid(model,
                                         x_min=-4, x_max=4,
                                         y_min=-4, y_max=4,
                                         steps=50)

    # Gradients wrt input
    GX, GY = compute_input_gradients(model, xs, ys)

    # Plot all together
    # Pass the required inputs into the provided plotting function 
    #\verb|plot_data_boundary_and_gradients()| that you can find in \verb|helper.py|
