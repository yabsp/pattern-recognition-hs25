"""
This script is used to train a 1-layer neural network.
"""
import numpy as np
import matplotlib.pyplot as plt

class OLNN():
    """
    The class which contains the functions a 1-layer neural network
    """

    @staticmethod
    def update(theta: list, x: np.ndarray, y_true: np.ndarray, gamma: float) -> list:
        """
        Updates the parameters using one step of gradient descent.
        Inputs:
            - theta: list containing [W_1, W_2, b_1, b_2]
            - x: Input data (N, K)
            - y_true: True labels (N, 1)
            - gamma: Learning rate
        Outputs:
            - updated_theta: list containing updated parameters
        """
        W_1, W_2, b_1, b_2 = theta[0], theta[1], theta[2], theta[3]
        N = x.shape[0]

        y_pred, h_1, z_1 = OLNN.nnf(theta, x)

        # dL/dy_pred
        grad_L_y = (1 / N) * (y_pred - y_true)

        # dL/db_2 = sum(dL/dy * dy/db_2) = sum(grad_L_y * 1)
        grad_b2 = np.sum(grad_L_y)

        # dL/dW_2 = (dL/dy)^T @ (dy/dW_2) = grad_L_y.T @ h_1
        grad_W2 = np.dot(grad_L_y.T, h_1)

        # dL/dh_1 = dL/dy @ dy/dh_1 = grad_L_y @ W_2
        grad_L_h1 = np.dot(grad_L_y, W_2)

        # dL/dz_1 = dL/dh_1 * dh_1/dz_1 = grad_L_h1 * sigma'(z_1) (element-wise)
        grad_L_z1 = grad_L_h1 * OLNN.sigma_derivative(z_1)

        # dL/db_1 = sum(dL/dz_1 * dz_1/db_1) = sum(grad_L_z1 * 1) column-wise
        grad_b1 = np.sum(grad_L_z1, axis=0).reshape(-1, 1)

        grad_W1 = np.dot(grad_L_z1.T, x)

        # --- 3. Parameter Update ---
        W_1_new = W_1 - gamma * grad_W1
        W_2_new = W_2 - gamma * grad_W2
        b_1_new = b_1 - gamma * grad_b1
        b_2_new = b_2 - gamma * grad_b2

        updated_theta = [W_1_new, W_2_new, b_1_new, b_2_new]
        return updated_theta


    @staticmethod
    def nnf(theta: list, x: np.ndarray) -> tuple:
        """
        Represents the function of the one hidden layer nn.
        Handles a batch of N data points. Also returns intermediate values.
        Inputs:
            - theta containing the function parameters:
                theta[0] = W_1 (F, K)
                theta[1] = W_2 (1, F)
                theta[2] = b_1 (F, 1)
                theta[3] = b_2 (scalar)
            - x: Input data (N, K)
        Outputs:
            - y_pred: Predicted output (N, 1)
            - h_1: Hidden layer activation (N, F)
            - z_1: Hidden layer pre-activation (N, F)
        """
        W_1, W_2, b_1, b_2 = theta[0], theta[1], theta[2], theta[3]

        xw = np.dot(x, W_1.T)
        print(f'shape xW1\': {xw.shape}')
        z_1 =  xw + b_1.T
        h_1 = OLNN.sigma(z_1)
        y_pred = np.dot(h_1, W_2.T) + b_2

        return y_pred, h_1, z_1


    @staticmethod
    def loss(y: np.ndarray, x:np.ndarray, theta: list) -> float:
        """
        Calulates the loss according to the following loss function:
        L = 1 / (2 * N) * (nnf(x) - y)^2
        Inputs:
            - true labels y
        Outputs:
            - loss
        """
        y_pred = OLNN.nnf(theta, x)[0]
        return 1 / (2 * y.shape[0]) * np.sum(np.pow(y_pred - y, 2))

    @staticmethod
    def sigma(x: np.ndarray) -> np.ndarray:
        """ Non-linear activation function.
        In our case ReLU.
        Inputs:
            - x of size K
        Outputs:
            - ReLU(x)
        """
        return np.where(x >= 0, x, 0)

    @staticmethod
    def sigma_derivative(x: np.ndarray) -> np.ndarray:
        """ Derivative of the ReLU activation function. """
        return np.where(x > 0, 1, 0)

if __name__ == '__main__':
    x_train = np.array([-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2]).reshape((9, 1))
    y_true = np.array([1.44, 0.81, 0.36, 0.09, 0., 0.09, 0.36, 0.81, 1.44]).reshape((9, 1))
    K = x_train.shape[1]
    dimensions = [2, 5, 10, 50, 100, 500]

    gamma = 0.1
    #print(OLNN.nnf(theta, x_train))

    for F in dimensions:
        np.random.seed(42)
        W_1 = np.random.randn(F, K) * 0.1
        W_2 = np.random.randn(1, F) * 0.1
        b_1 = np.zeros((F, 1))
        b_2 = 0.0
        theta = [W_1, W_2, b_1, b_2]
        for i in range(0, 2000):
            theta = OLNN.update(theta, x_train, y_true, gamma)
            print(f'loss: {OLNN.loss(y_true, x_train, theta)}')

        x_plot = np.linspace(-1.5, 1.5, 100).reshape((-1, 1))
        y_plot = np.power(x_plot, 2)
        y_pred, _, _ = OLNN.nnf(theta, x_plot)

        plt.plot(x_plot, y_pred, 'r-', label=f'NN prediction (F={F})', linewidth=2)
        plt.plot(x_plot, y_plot, 'k--', label='Target function: $y=x^2$')
        plt.plot(x_train, y_true, 'bo', label='Training points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Neural Network Fit (Hidden Size F={F})')
        plt.legend()
        plt.grid(True)
        plt.ylim(-0.1, 2.2)
        plt.show()

