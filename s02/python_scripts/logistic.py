"""
This script contains the functions for logistic regression assignment
"""

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    The class which contains the functions for logistic regression
    """


    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        """
        Calculates the sigmoid of the input
        Inputs:
            - z : vector of size N
        Outputs:
            - y : vector of size N containing the sigmoid of x
        """
        # Complete your code here

        return 1 / (1 + np.exp(-z))

    
    @staticmethod
    def predict_score(X: np.array, w: np.array) -> np.array:
        """
        predicts the score for the data
        Inputs:
            - X : N x D data matrix
            - w : D size weight vector
        Outputs:
            - y_pred : Vector of size N containing the predicted scores
        """

        # Complete your code here

        return LogisticRegression.sigmoid(np.dot(X,w))
    

    @staticmethod
    def predict(X: np.array, w: np.array, threshold: float = 0.5) -> np.array:
        """
        predicts the labels for the data
        Inputs: 
            - X : N x D data matrix
            - w : D size weight vector
            - threshold : threshold for the sigmoid function
        Outputs:
            - y_pred : Vector of size N containing the predicted labels as 1 or 0
        """

        # Complete your code here
        score = LogisticRegression.predict_score(X,w)
        y_pred = np.where(score >= threshold, 1, 0)

        return y_pred
    

    @staticmethod
    def cross_entropy_loss(y_pred_score: np.array, y_true: np.array) -> float:
        """
        Calculates the cross entropy loss for the data
        Inputs:
            - y_pred_score : vector of size N containing the predicted scores
            - y_true : vector of size N containing the true labels as 1 or 0
        Outputs:
            - loss : cross entropy loss
        """

        # Complete your code here
        e_n = np.ones(y_true.shape[0]) # Info: Optional numpy handles operations between scalar and vector
        return  (-1) * (np.dot(y_true, np.log(y_pred_score)) + np.dot((e_n - y_true), np.log(e_n - y_pred_score))) / y_true.shape[0]
    
    @staticmethod
    def gradient(X: np.array, w: np.array, y_true: np.array) -> np.array:
        """
        Calculates the gradient of the loss function
        Inputs:
            - X : N x D data matrix
            - w : D size weight vector
            - y_true : vector of size N containing the true labels as 1 or 0
        Outputs:
            - grad : D size vector containing the gradient of the loss function
        """

        # Complete your code here
        y_pred = LogisticRegression.predict_score(X, w)

   
        return np.dot(np.transpose(X), y_pred - y_true) / X.shape[0]
    
    @staticmethod
    def train(X: np.array, w_init: np.array,y_true: np.array,
               epochs: int, lr: float) -> Tuple[np.array, List[float]]:
        """
        Use gradient descent to train the logistic regression model
        Inputs:
            - X : N x D data matrix
            - w_init : D size initial weight vector
            - y_true : vector of size N containing the true labels as 1 or 0
            - epochs : number of epochs to train
            - lr : learning rate
        Outputs:
            - w : D size weight vector
            - losses : list containing the loss at each epoch 
        """

        #Complete your code here
        w = w_init
        lossesl = []
        for i in range(1, epochs+1):
            y_pred = LogisticRegression.predict_score(X,w)
            loss = LogisticRegression.cross_entropy_loss(y_pred, y_true)
            lossesl.append(loss)
            w = w - lr * LogisticRegression.gradient(X, w, y_true)

        return w, lossesl
    


if __name__ == "__main__":
    # Test the logistic regression class on a toy dataset
    # Create a toy dataset

    np.random.seed(0) # Setting the seed to get consistent results
    N = 100
    D = 2
    X_data = np.random.randn(N, D)
    w_true = np.random.randn(D)
    y_data = np.where(X_data @ w_true > 0, 1, 0)


    # Check outputs with the toy dataset
    y_est = LogisticRegression.predict(X_data, w_true)


    if np.all(y_est == y_data):
        print("Passed the predict function test")

    # Testing Cross Entropy

    y1 = np.array([0.5,0.5])
    y2 = np.array([1,0])

    loss = LogisticRegression.cross_entropy_loss(y1, y2)

    if np.isclose(loss, 0.6931471805599453):
        print("Passed the cross entropy test")


    # Testing the gradient function
    w_init = np.zeros(D)
    grad = LogisticRegression.gradient(X_data, w_init, y_data)

    # Note that this for the case when the seed is set to 0
    if np.all(np.isclose(grad, np.array([0.35061729,0.2135946 ]))):
        print("Passed the gradient test")

    # Testing the train function

    w_init = np.zeros(D)


    X_train = np.random.randn(N, D)
    y_train = np.where(X_train @ w_true > 0, 1, 0)


    X_test = np.random.randn(N, D)
    y_test = np.where(X_test @ w_true > 0, 1, 0)



    y_pred_init = LogisticRegression.predict(X_test, w_init)
    accuracy_init = np.mean(y_pred_init == y_test)


    w_train, losses = LogisticRegression.train(X_train, w_init,
                                               y_train, epochs=100, lr=0.1)
    y_test_predict = LogisticRegression.predict(X_test, w_train)
    accuracy_final = np.mean(y_test_predict == y_test)

    print("Initial accuracy is ", accuracy_init)
    print("Final accuracy is ", accuracy_final)

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


    X = np.random.randn(100, 2)
    W = np.random.randn(2)
    x = X[:, 1]
    #print(LogisticRegression.sigmoid(x))
    #print(LogisticRegression.predict_score(X, W))
    threshold = 0.5
    #print(f'From {X.shape[0]} possible, {np.count_nonzero(LogisticRegression.predict(X,W, threshold))} where labeled (prediction) with 1 (threshold = {threshold}).')
    



    