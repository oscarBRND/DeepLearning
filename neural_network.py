import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
from copy import deepcopy

# Neural network implementation with two layers, the first layer containing 3 neurons and the second layer containing 1 neuron. 
# The activation function used is Sigmoid.
# Our dataset is a binary classification problem generated using make_blobs from sklearn. With 10,000 samples, 2 features, and 2 centers.

# X, y = make_blobs(n_samples=10000, centers=2, n_features=2, random_state=42, cluster_std=1.5)
# y = y.reshape(-1, 1)
# X = X.T

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=42)
y = y.reshape(-1, 1)
X = X.T

def initialize_parameters(n0, n1, n2):
    """
    
    """
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    parametres = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parametres


def forward_propagation(X, parametres):
    """
    Compute the output of the neural network model.
    """
    w1 = parametres["W1"]
    b1 = parametres["b1"]
    w2 = parametres["W2"]
    b2 = parametres["b2"]

    Z1 = w1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1)) # Sigmoid activation
    Z2 = w2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2)) # Sigmoid activation

    activations = {
        "A1": A1,
        "A2": A2
    }

    return activations

def backpropagation(X, Y_true, activations, parameters):
    """
    Compute the gradients of the loss with respect to W and b.
    """
    
    m = Y_true.shape[0]
    A1 = activations["A1"]
    A2 = activations["A2"]
    w2 = parameters["W2"]

    dZ2 = A2 - Y_true.T
    dW2 = (1/m)*dZ2.dot(A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.T, dZ2) * A1 * (1 - A1)
    dW1 = (1/m)*dZ1.dot(X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return gradients

def gradients_descent(parameters, gradients, learning_rate):
    """
    Update weights W and bias b using gradient descent.
    """
    parameters["W1"] -= learning_rate * gradients["dW1"]
    parameters["b1"] -= learning_rate * gradients["db1"]
    parameters["W2"] -= learning_rate * gradients["dW2"]
    parameters["b2"] -= learning_rate * gradients["db2"]
    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    Y_pred = activations["A2"]
    return Y_pred >=0.5

def NLL(Y_true, Y_pred):
    m = Y_true.shape[0]
    eps = 1e-15
    Y_pred = Y_pred.T
    Y_pred = np.clip(Y_pred, eps, 1-eps)
    loss = - (1/m) * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
    return loss


def artificial_neuron(X, y, learning_rate=0.01, n_iterations=10000, n1=3, record_every=10):
    n_features = X.shape[0]
    parameters = initialize_parameters(n_features, n1, 1)
    
    losses = []
    accuracies = []
    params_history = []   # pour l'animation
    iters_history = []    # pour afficher le n° d'itération sur l'animation
    
    for i in range(n_iterations):
        activations = forward_propagation(X, parameters)
        gradients = backpropagation(X, y, activations, parameters)
        parameters = gradients_descent(parameters, gradients, learning_rate)

        loss = NLL(y, activations["A2"])
        y_pred_labels = predict(X, parameters)
        acc = accuracy_score(y.flatten(), y_pred_labels.flatten())
        
        # On enregistre à chaque itération, ou bien toutes les record_every itérations
        if i % record_every == 0:
            losses.append(loss)
            accuracies.append(acc)
            params_history.append({k: v.copy() for k, v in parameters.items()})
            iters_history.append(i)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    return parameters, losses, accuracies, params_history, iters_history