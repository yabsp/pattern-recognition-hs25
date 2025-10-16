'''
This contains the skeleton code for the random classifier
exercise.
'''

import numpy as np
import matplotlib.pyplot as plt
from perceptron import MutiClassPerceptron




if __name__ == '__main__':

    # Generate random data

    N = 100
    p = 0.5
    d_init = 10

    # Let #X_train = N x d abd #y_train = N x 1 be the training data
    # Using random.randn() generate X_train and random.binomial() generate y_train
    # make sure to one-hot-encode y_train
    # similarly generate X_test and y_test
    # make sure to one-hot-encode y_test
    # Initialize the perceptron with W_int = np.zeros((d,2)) and train it
    # Repeat the above steps for d = np.arange(10,200,10)
    # Plot the accuracy on the training and test set as a function of d

    train_accuracies = []
    test_accuracies = []
    d_range = np.arange(d_init, 201, 10)

    # Initialize the perceptron model
    model = MutiClassPerceptron()

    for D in d_range:
        print(f"Running experiment for dimension D = {D}")

        # a) Generate new random data for this dimension
        x_train = np.random.randn(N, D)
        y_train = np.random.randint(0, 2, size=N)
        x_test = np.random.randn(N, D)
        y_test = np.random.randint(0, 2, size=N)

        W_init = np.zeros((2, D))

        y_train = model.one_hot_encode(y_train)
        y_test = model.one_hot_encode(y_test)

        W_trained, _ = model.train(x_train, y_train, W_init, n_iter=20)

        y_pred_train = model.predict(W_trained, x_train)
        train_acc = np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(y_train, axis=1))
        train_accuracies.append(train_acc)

        y_pred_test = model.predict(W_trained, x_test)
        test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
        test_accuracies.append(test_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(d_range, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(d_range, test_accuracies, marker='s', label='Test Accuracy')
    plt.title('Perceptron Accuracy vs. Input Dimension (N=100)')
    plt.xlabel('Dimension (D)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()







