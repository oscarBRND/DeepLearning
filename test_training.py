from sklearn.datasets import make_circles
from src.initializers import XavierUniform
from src.layers import Dense, ReLU, Sigmoid
from src.model import Sequential
from src.loss import BCE
from src.optimizer import SGD
from src.train import train

if __name__ == "__main__":
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    y_train = y
    x_train = X.T


    model = Sequential([Dense(input_size=2, output_size=5, initializer=XavierUniform()), ReLU(), Dense(input_size=5, output_size=1, initializer=XavierUniform()), Sigmoid()])
    loss_fn = BCE()
    optimizer = SGD(learning_rate=0.1)
    epochs = 10000

    train(model, loss_fn, optimizer, x_train, y_train, epochs)