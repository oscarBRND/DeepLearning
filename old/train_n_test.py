import old.neural_network_2_layers as nn2
from old.neural_network_2_layers import forward_propagation
import low_level_NN.old.graph_n_animation as ga
from sklearn.datasets import make_circles


X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
y = y.reshape(-1, 1)
X = X.T

list_dimension = [2, 5, 5, 1]

# parameters, losses, accuracies, params_history, iters_history = nn2.artificial_neuron(X, y, learning_rate=0.1, n_iterations=10000, n1=5, record_every=10)
parameters, losses, accuracies, params_history, iters_history = nn2.neural_network(X, y, learning_rate=0.1, n_iterations=10000, record_every=10)

# Graphiques de l'entraînement
ga.show_graph(losses, accuracies, X, y, parameters, forward_propargation=nn2.forward_propagation)

# Animation de la frontière de décision pendant l'entraînement
anim = ga.animate_training_contour(X, y, losses, accuracies, params_history, iters_history, forward_propargation=nn2.forward_propagation)
anim.save('nn_training_contour.gif', fps=30)