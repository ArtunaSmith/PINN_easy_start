import torch.nn as nn
from collections import OrderedDict


class MPL(nn.Module):
    def __init__(self, input_size: int, hidden_size_list: list, output_size: int, act=nn.Tanh):
        """
        Define the MPL neutral network
        :param input_size: the input size
        :param hidden_size_list: the output size list, each element controls the hidden layer's size
        """
        super(MPL, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        last_size = input_size
        for size in hidden_size_list:
            self.layers.append(nn.Linear(last_size, size))
            last_size = size
            self.layers.append(act())
        self.layers.append(nn.Linear(last_size, output_size))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, input_x):
        return self.layers(input_x)

