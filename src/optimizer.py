from abc import ABC, abstractmethod
import numpy as np 

class Optimizer(ABC):
    @abstractmethod
    def update(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pass

class SGD(Optimizer):
    """
    SGD stands for Stochastic Gradient Descent. It is an optimization algorithm used to minimize the loss function by iteratively updating the model parameters in the direction of the negative gradient of the loss function with respect to the parameters.
    """
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def update(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        updated_parameters = {}
        for key in parameters.keys():
            updated_parameters[key] = parameters[key] - self.learning_rate * gradients[key]
        return updated_parameters
    

#TODO
class Adam(Optimizer): ...
