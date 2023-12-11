from NN import *
from Data import *
from PINN import *
import torch
import numpy as np
import matplotlib.pyplot as plt


def display(pinn, save_path=None, title=None, show=True):
    # PINN finial predict
    pinn.predict()
    # Define variables for plot
    p_in, pred_in, p_bo, pred_bo = pinn.get_all()
    x, t = p_in[:, 0], p_in[:, 1]
    u = pred_in[:, 0]
    x_bo, t_bo = p_bo[:, 0], p_bo[:, 1]
    u_bo = pred_bo[:, 0]
    # Visualization
    plt.figure(figsize=(5, 5), dpi=300)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if title is not None:
        plt.title(title)
    ax.plot_trisurf(x, t, u)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def pinn_test(hidden, lr):
    # Net
    net = MPL(2, hidden, 1)

    # variable name
    var_name = ['x', 't']

    # variables
    variables = list()
    variables.append(torch.arange(-1, 1, 0.05))
    variables.append(torch.arange(0, 2, 0.05))

    # Boundary
    boundaries = list()
    boundaries.append((0, 1.0))  # x = 1.0
    boundaries.append((0, -1.0))  # x = -1.0
    boundaries.append((1, 0.0))  # t = 0.0

    # equations
    pass

    # bound_cond
    bound_cond = list()
    bound_cond.append(-0.5)
    bound_cond.append(0.5)
    bound_cond.append(lambda x1, x2: (-0.5 * x1,))

    pinn = PINN(net, variables, boundaries, bound_cond, lr=lr)

    # PINN Train
    pinn.train(500)

    return pinn


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pinn = pinn_test([16, 8, 4, ], 0.1)

    display(pinn)

