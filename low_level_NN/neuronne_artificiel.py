import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=10000, centers=2, n_features=2, random_state=42, cluster_std=1.5)
y = y.reshape(-1, 1)

def initialize_parameters(n_features):
    """
    Initialize weights W and bias b to zeros.
    """
    W = np.random.randn(n_features, 1)
    b = np.random.randn(1)
    return W, b


def model(X, W, b):
    """
    Compute the output of the neural network model.
    """
    Z = X.dot(W) + b
    Y = 1 / (1 + np.exp(-Z)) # Sigmoid activation
    return Y

def NLL(Y_true, Y_pred):
    """
    Compute the Negative Log-Likelihood loss.
    """
    m = Y_true.shape[0]
    loss = - (1/m) * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    return loss

def gradients(X, Y_true, Y_pred):
    """
    Compute the gradients of the loss with respect to W and b.
    """
    m = Y_true.shape[0]
    dW = (1/m)*np.dot(X.T, (Y_pred - Y_true))
    db = (1/m)*np.sum(Y_pred - Y_true)
    return dW, db

def gradients_descent(W, b, dW, db, learning_rate):
    """
    Update weights W and bias b using gradient descent.
    """
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def artificial_neuron(X, y, learning_rate=0.01, n_iterations=100000):
    """
    Train the artificial neuron using gradient descent.
    """
    n_features = X.shape[1]
    W, b = initialize_parameters(n_features)
    
    losses = []
    accuracies = []
    
    for i in range(n_iterations):
        Y_pred = model(X, W, b)
        loss = NLL(y, Y_pred)
        losses.append(loss)

        # Compute accuracy
        y_pred_labels = (Y_pred >= 0.5).astype(int)
        acc = accuracy_score(y, y_pred_labels)
        accuracies.append(acc)
        
        dW, db = gradients(X, y, Y_pred)
        W, b = gradients_descent(W, b, dW, db, learning_rate)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    
    return W, b, losses, accuracies




W, b, losses, accuracies = artificial_neuron(X, y, learning_rate=0.1, n_iterations=10000)

def show_graph(losses, accuracies, X, y, W, b):
    plt.figure(figsize=(18, 5))

    # ---- 1. Loss ----
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss over Iterations')

    # ---- 2. Accuracy ----
    plt.subplot(1, 3, 2)
    plt.plot(accuracies, color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Accuracy over Iterations')

    # ---- 3. Decision boundary ----
    plt.subplot(1, 3, 3)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    Z = model(np.c_[xx.ravel(), yy.ravel()], W, b)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, colors=['blue', 'orange'])
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

    plt.show()



show_graph(losses, accuracies, X, y, W, b)