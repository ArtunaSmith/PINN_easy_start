import torch


class Equation:
    def __init__(self, function, variables):
        self.fun = function
        self.vars = variables

    @staticmethod
    def get_burge_equation(function, variables):
        # Burger's Equation
        #   U_t + U * U_x - omiga * U_xx = 0

        # Define omiga
        omiga = 0.001

        # function return (B, 1)
        # variable shape (B, 2)

        out = function(variables)[:, 0]  # shape: (B)
        do_dv = torch.autograd.grad(
            outputs=out,
            inputs=variables,
            grad_outputs=torch.ones_like(out),
            retain_graph=True,
            create_graph=True
        )[0]  # shape: (B)

        do_dx, do_dt = do_dv[:, 0], do_dv[:, 1]  # shape: (B)
        do_dxx = torch.autograd.grad(
            outputs=do_dv,
            inputs=variables,
            grad_outputs=torch.ones_like(do_dv),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]  # shape: (B)

        return do_dt + out * do_dx - omiga * do_dxx


    @staticmethod
    def get_ploy_euqtion(function, variables, partial_list):
            pass

