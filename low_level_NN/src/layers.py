from abc import ABC, abstractmethod
import numpy as np

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
    
class Dense(Layer):

    def __init__(self, input_size: int, output_size: int, initializer) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer

        self.W = self.initializer.init_weight((output_size, input_size), input_size, output_size)
        self.b = self.initializer.init_bias((output_size, 1))

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        self.input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        z = np.dot(self.W, x) + self.b
        return z
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        m = self.input_cache.shape[1]

        self.grad_W = np.dot(grad_output, self.input_cache.T) / m
        self.grad_b = np.sum(grad_output, axis=1, keepdims=True) / m

        grad_input = np.dot(self.W.T, grad_output)
        return grad_input
    
    def parameters(self) -> list[np.ndarray]:
        return {
            "W": self.W,
            "b": self.b
        }
    def gradients(self) -> list[np.ndarray]:
        return {
            "dW": self.grad_W,
            "db": self.grad_b
        }

class Activation(Layer, ABC):
    pass

class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output * (self.input_cache > 0)
        return grad_input
    
class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output_cache = 1 / (1 + np.exp(-x))
        return self.output_cache
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output * self.output_cache * (1 - self.output_cache)
        return grad_input