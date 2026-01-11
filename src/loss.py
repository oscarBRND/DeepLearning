from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class NLL(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[0]
        grad = y_pred.copy()
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad