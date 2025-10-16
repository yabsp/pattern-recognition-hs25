"""
This script is used to train the logistic regression classifier on the rice dataset
Note that gradescope will not run this script. However you need to upload this script
or a similar script to gradescope for us to evaluate your report and plots
"""

import numpy as np
import matplotlib.pyplot as plt
from logistic import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


if __name__== "__main__":
    # Load the Rice dataset
    data = pd.read_csv('Rice.csv')


    y = data['Class']
    print(data)
    X = data.drop(['Class'], axis=1)


    # Complete the code here:
    # Convert y to binary values
    y_bin = (y == 'Cammeo').astype(int)
    X_st = ((X - X.mean()) / X.std())


    # Convert to numpy arrays
    # preprocess dataset

    # DataFrama -> ndarray + split, split = 20 % train, random_state (=seed)
    X_train, X_test, y_train, y_test = train_test_split(X_st, y_bin, test_size=0.2, random_state=1)
 
    # Split the data into train and test sets

    # Train the logistic regression classifier
    # Note this is just a sample code for training the classifier
    epochs = 100000
    w = np.zeros(X_train.shape[1])
    W_train, loss = LogisticRegression.train(X_train,w,y_train, epochs, 0.1)
    y_test_pred = LogisticRegression.predict(X_test, W_train)
    y_score = LogisticRegression.predict_score(X_test, W_train)
    print('Accuracy: ', np.mean(y_test_pred == y_test))

    plt.figure()
    plt.plot(range(epochs), loss)
    plt.xlabel('Epochs')
    plt.ylabel('CE loss')
    plt.grid(True)
    plt.show()
