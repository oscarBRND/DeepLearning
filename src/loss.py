from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class BCE(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15

        # Force shape (1, m)
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        assert y_true.shape == y_pred.shape
        m = y_true.shape[1]  # batch size

        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # (1, m)
        return float(np.mean(loss))  # moyenne sur tous les éléments

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-15

        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        assert y_true.shape == y_pred.shape
        m = y_true.shape[1]

        y_pred = np.clip(y_pred, eps, 1 - eps)

        grad = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)  # (1, m)
        return grad / m

    
class CE(Loss):
    '''
    Cross Entropy Loss for multi-class classification tasks.
    '''
    def forward(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return loss.mean()
    
    def backward(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        grad = - (y_true / y_pred)
        return grad / y_true.shape[0]
    

#TODO
class BinaryCrossEntropy(Loss):...
class CategoricalCrossEntropy(Loss):...

