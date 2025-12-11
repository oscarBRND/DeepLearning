import neural_network as nn
import graphs_n_animations as ga
from sklearn.datasets import make_circles


X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
y = y.reshape(-1, 1)
X = X.T

parameters, losses, accuracies, params_history, iters_history = nn.artificial_neuron(X, y, learning_rate=0.1, n_iterations=10000, n1=5, record_every=10)

ga.show_graph(losses, accuracies, X, y, parameters)

anim = ga.animate_training_contour(X, y, losses, accuracies, params_history, iters_history)
# anim.save('nn_training_contour.gif', fps=30)