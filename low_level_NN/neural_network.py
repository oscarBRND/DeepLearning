import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
from copy import deepcopy
from abc import ABC, abstractmethod

# -------- Layers class -------
class Layer(ABC):
    """
    Classe abstraite pour toute couche du réseau.
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Passage avant.
        x : entrée (batch_size, ...)
        retourne : sortie
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Rétropropagation.
        grad_output : ∂L/∂(sortie)
        retourne : ∂L/∂(entrée)
        """
        pass

    def parameters(self) -> list[np.ndarray]:
        """
        Paramètres entraînables de la couche.
        Par défaut : aucun (ReLU, Sigmoid, etc.).
        """
        return []

    def gradients(self) -> list[np.ndarray]:
        """
        Gradients associés aux paramètres.
        Même ordre que `parameters()`.
        """
        return []




# =========================
# Model
# =========================
def initialize_parameters(list_dimensions):
    """
    list_dimensions = [n0, n1, ..., nL]
      n0: number of features
      nL: output size
    returns parameters dict with W1..WL and b1..bL
    """
    L = len(list_dimensions) - 1  # number of weight layers
    parameters = {}
    for l in range(1, L + 1):
        parameters["W" + str(l)] = np.random.randn(list_dimensions[l], list_dimensions[l - 1])
        parameters["b" + str(l)] = np.random.randn(list_dimensions[l], 1)
    return parameters


def forward_propagation(X, parameters, list_activation_function):
    """
    X shape: (n0, m)
    returns:
      activations: {"A1":..., ..., "AL":...}
      caches: {"Z1":..., ..., "ZL":...}
    """
    L = len(parameters) // 2
    activations = {}
    caches = {}

    A_prev = X
    for l in range(1, L + 1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        layer_l = layer(W.shape[1], W.shape[0], list_activation_function[l - 1])
        A, Z = layer_l.forward(A_prev, W, b)
        caches["Z" + str(l)] = Z
        activations["A" + str(l)] = A
        A_prev = A

    return activations, caches

def backpropagation(X, Y_true, activations, parameters, list_activation_function):
    """
    Binary cross-entropy with sigmoid output:
      dZ_L = A_L - Y
    Shapes:
      Y_true: (m, 1)
      A_L: (1, m) if output layer size is 1
    """
    gradients = {}
    m = X.shape[1]
    L = len(parameters) // 2

    # Output layer
    A_L = activations["A" + str(L)]          # (nL, m)
    dZ = A_L - Y_true.T                      # (nL, m)

    for l in range(L, 0, -1):
        A_prev = X if l == 1 else activations["A" + str(l - 1)]  # (n_{l-1}, m)

        gradients["dW" + str(l)] = (1 / m) * dZ.dot(A_prev.T)
        gradients["db" + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if l > 1:
            A_prev_act = activations["A" + str(l - 1)]
            dZ = parameters["W" + str(l)].T.dot(dZ) * A_prev_act * (1 - A_prev_act)

    return gradients

def gradient_descent(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * gradients["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * gradients["db" + str(l)]
    return parameters

def predict_proba(X, parameters):
    L = len(parameters) // 2
    activations, _ = forward_propagation(X, parameters)
    return activations["A" + str(L)]  # (1, m) if last layer is 1

def predict(X, parameters, threshold=0.5):
    Y_pred = predict_proba(X, parameters)
    return (Y_pred >= threshold)

def NLL(Y_true, Y_pred):
    """
    Binary cross-entropy.
    Y_true: (m, 1)
    Y_pred: (1, m) or (m, 1)
    """
    eps = 1e-15
    if Y_pred.shape[0] == 1:
        Yp = Y_pred.T  # (m, 1)
    else:
        Yp = Y_pred    # (m, 1)

    Yp = np.clip(Yp, eps, 1 - eps)
    m = Y_true.shape[0]
    return - (1 / m) * np.sum(Y_true * np.log(Yp) + (1 - Y_true) * np.log(1 - Yp))


def neural_network(X, y, list_dimensions, learning_rate=0.01, n_iterations=10000, record_every=10):

    # Architecture summary
    print("Neural network architecture:")
    for i in range(len(list_dimensions)):
        if i == 0:
            print(f"  Input layer      : {list_dimensions[i]} features")
        elif i == len(list_dimensions) - 1:
            print(f"  Output layer     : {list_dimensions[i]} neuron(s)")
        else:
            print(f"  Hidden layer {i}    : {list_dimensions[i]} neuron(s)")
    parameters = initialize_parameters(list_dimensions)
    L = len(parameters) // 2

    losses = []
    accuracies = []
    params_history = []
    iters_history = []

    for i in range(n_iterations):
        activations, _ = forward_propagation(X, parameters)
        loss = NLL(y, activations["A" + str(L)])

        gradients = backpropagation(X, y, activations, parameters)
        parameters = gradient_descent(parameters, gradients, learning_rate)

        y_pred_labels = predict(X, parameters)
        acc = accuracy_score(y.flatten(), y_pred_labels.flatten())

        if i % record_every == 0:
            losses.append(loss)
            accuracies.append(acc)
            params_history.append({k: v.copy() for k, v in parameters.items()})
            iters_history.append(i)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    return parameters, losses, accuracies, params_history, iters_history
