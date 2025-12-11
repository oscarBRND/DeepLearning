import neural_network as nn
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np


def animate_training_contour(X, y, losses, accuracies, params_history, iters_history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_loss, ax_acc, ax_db = axes

    # -----------------------------
    # 1) Courbe de loss
    ax_loss.set_title("Loss over iterations")
    ax_loss.set_xlabel("Recorded steps")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, len(losses))
    ax_loss.set_ylim(min(losses) * 0.9, max(losses) * 1.1)
    line_loss, = ax_loss.plot([], [])

    # -----------------------------
    # 2) Courbe d'accuracy
    ax_acc.set_title("Accuracy over iterations")
    ax_acc.set_xlabel("Recorded steps")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlim(0, len(accuracies))
    ax_acc.set_ylim(0, 1.05)
    line_acc, = ax_acc.plot([], [])

    # -----------------------------
    # 3) Grille pour la frontière de décision
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # (2, N_points)

    def init():
        line_loss.set_data([], [])
        line_acc.set_data([], [])
        return line_loss, line_acc

    def update(frame):
        params = params_history[frame]

        # ----- Loss & accuracy -----
        x_axis = np.arange(frame + 1)
        line_loss.set_data(x_axis, losses[:frame+1])
        line_acc.set_data(x_axis, accuracies[:frame+1])

        # ----- Frontière de décision -----
        ax_db.clear()  # on nettoie tout l'axe

        # Recalcule des prédictions sur la grille
        activations = nn.forward_propagation(grid_points, params)
        Z = activations["A2"].reshape(xx.shape)

        # Fond coloré (0 / 1)
        cf = ax_db.contourf(
            xx, yy, Z,
            levels=[0.0, 0.5, 1.0],
            alpha=0.5
        )

        # Frontière nette A2 = 0.5
        cs = ax_db.contour(
            xx, yy, Z,
            levels=[0.5],
            linewidths=2
        )

        # Données originales
        ax_db.scatter(X[0], X[1], c=y.flatten(), edgecolors="k")

        ax_db.set_xlim(x_min, x_max)
        ax_db.set_ylim(y_min, y_max)
        ax_db.set_xlabel("Feature 1")
        ax_db.set_ylabel("Feature 2")
        ax_db.set_title(f"Decision boundary - iter {iters_history[frame]}")

        return line_loss, line_acc

    anim = FuncAnimation(
        fig,
        update,
        frames=len(params_history),
        init_func=init,
        interval=10,
        blit=False
    )

    plt.tight_layout()
    plt.show()
    return anim

def show_graph(losses, accuracies, X, y, parameters):
    plt.figure(figsize=(18, 5))

    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss over Iterations')

    # 2. Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(accuracies)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Accuracy over Iterations')

    # 3. Decision boundary
    plt.subplot(1, 3, 3)
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # (2, N_points)
    activations = nn.forward_propagation(grid_points, parameters)
    Z = activations["A2"].reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.5, colors=['blue', 'orange'])
    plt.scatter(X[0, :], X[1, :], c=y.flatten(), edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

    plt.show()

