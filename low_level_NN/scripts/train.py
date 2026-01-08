import numpy as np
from src.model import Sequential
from src.loss import NLL
from src.optimizer import SGD
from src.layers import Layer, Dense
from src.initializers import XavierInitializer

def train_step(model, loss_fn, optimizer, x, y):
    y_pred = model.forward(x)
    loss = loss_fn.forward(y, y_pred)
    grad_loss = loss_fn.backward(y, y_pred)
    model.backward(grad_loss)
    parameters, gradients = model.parameters_and_gradients()
    updated_parameters = optimizer.update(parameters, gradients)

    # Update model parameters
    model.set_parameters(updated_parameters)
    return loss

def train(model, loss_fn, optimizer, x_train, y_train, epochs: int = 100):
    for epoch in range(epochs):
        loss = train_step(model, loss_fn, optimizer, x_train, y_train)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")