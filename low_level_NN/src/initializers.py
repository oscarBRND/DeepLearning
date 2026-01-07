from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    @abstractmethod
    def init_weight(self, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.array:
        pass
    @abstractmethod
    def init_bias(self, shape: tuple[int, ...]) -> np.array:
        pass

class XavierUniform(Initializer):
    '''
    Understanding the difficulty of training deep feedforward neural networks : https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''
    def init_weight(self, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.array:
        np.random.seed(self.seed)
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    
    def init_bias(self, shape: tuple[int, ...]) -> np.array:
        np.random.seed(self.seed)
        return np.zeros(shape)

class XavierNormal(Initializer): 
    def init_weight(self, shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.array:
        np.random.seed(self.seed)
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)
    
    def init_bias(self, shape: tuple[int, ...]) -> np.array:
        np.random.seed(self.seed)
        return np.zeros(shape)
class HeNormal(Initializer): ...
class HeUniform(Initializer): ...