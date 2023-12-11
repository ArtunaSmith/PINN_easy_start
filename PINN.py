from NN import *
import torch
from Equations import *
import numpy as np


def _dim_var_to_point_list(var_list: list):
    """
    transfer the vars to a tensor of points list with shape (Batch, dimension)
    where the dimension is the length of var_list
    :param var_list: a list of torch.tensor, the value in tensor varify within its range.
    :return: a tensor, means point list with shape (Batch, dimension)
    """
    return torch.stack(torch.meshgrid(*var_list)).reshape(len(var_list), -1).T


def batch_apply_function(input_tensor, function):
    test_tensor = input_tensor[0, :]
    output_size = len(function(*test_tensor))

    input_size = input_tensor.shape[0]
    output_value = torch.zeros(input_size, output_size)
    for i in range(input_size):
        for dim, dim_value in enumerate(function(*input_tensor[i, :])):
            output_value[i, dim] = dim_value

    return output_value


class PINN:
    def __init__(self, net, variables: list, boundaries: list, boundary_conditions: list, equations=None, lr=0.01):
        # Get the running device.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Define the neutral network.
        self.model = net.to(device)

        # p_inside
        # Generate the input data points.
        # Points' shape is (PointSet, Dimension), Dimension here has the sense of Physical meaning.
        self.p_inside = _dim_var_to_point_list(variables)

        # p_boundary
        # Generate the boundary data points.
        #   Modify each of the boundaries
        bound_vars_list = list()
        for tup in boundaries:
            index, value = tup
            vars = variables.copy()
            vars[index] = torch.tensor(value)
            bound_vars_list.append(vars)
        #   get point list of each boundary
        bound_points_list = list()
        for bound_vars_list in bound_vars_list:
            bound_points_list.append(_dim_var_to_point_list(bound_vars_list))
        #   concatenate the points list together
        self.p_boundary = torch.concatenate(bound_points_list, dim=0)

        # v_boundary
        # Generate the value on boundary points
        v_boundary = list()
        for bound_points, cond_value in zip(bound_points_list, boundary_conditions):
            # bound_points shape: (B, D)
            if callable(cond_value):
                v_boundary.append(batch_apply_function(bound_points, cond_value))
            else:
                v_boundary.append(torch.ones(bound_points.shape[0], self.model.output_size) * cond_value)
        self.v_boundary = torch.concatenate(v_boundary, dim=0)

        # prediction variables
        #   pred_inside
        self.pred_inside = torch.zeros(self.p_inside.shape[0], self.model.output_size)
        #   pred_boundary
        self.pred_boundary = torch.zeros(self.p_boundary.shape[0], self.model.output_size)


        # Copy variables to GPU if available
        self.p_inside = self.p_inside.to(device)
        self.p_boundary = self.p_boundary.to(device)
        self.v_boundary = self.v_boundary.to(device)
        self.pred_inside = self.pred_inside.to(device)
        self.pred_boundary = self.pred_boundary.to(device)

        # Grad on
        self.p_inside.requires_grad = True

        # Set Optimizer
        self.adam = torch.optim.Adam(self.model.parameters())
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=1000,
            max_eval=1000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

    def loss(self):
        # Define the mse
        mse = torch.nn.MSELoss()

        # Clear the gradient
        self.adam.zero_grad()

        # loss on boundary
        boundary_pred_value = self.model(self.p_boundary)
        loss_boundary = mse(self.v_boundary, boundary_pred_value)

        # loss on inside physical equation
        inside_equation_value = Equation.get_burge_equation(self.model, self.p_inside)
        loss_equation = mse(inside_equation_value, torch.zeros_like(inside_equation_value))

        loss_sum = loss_boundary + loss_equation

        loss_sum.backward()

        return loss_sum

    def train(self, epoch):
        print("Training...")

        # Train mode on
        self.model.train()

        for i in range(epoch):
            self.adam.step(self.loss)
        self.lbfgs.step(self.loss)

    def predict(self):
        self.model.eval()
        self.pred_inside[:] = self.model(self.p_inside)
        self.pred_boundary[:] = self.model(self.p_boundary)

    def get_p_inside(self):
        return self.p_inside.cpu().detach().numpy()

    def get_p_boundary(self):
        return self.p_boundary.cpu().detach().numpy()

    def get_pred_inside(self):
        return self.pred_inside.cpu().detach().numpy()

    def get_pred_boundary(self):
        return self.pred_boundary.cpu().detach().numpy()

    def get_all(self):
        return self.get_p_inside(), self.get_pred_inside(), self.get_p_boundary(), self.get_pred_boundary()



