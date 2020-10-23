from abc import ABC
import torch
import torch.nn as nn
import numpy as np
from utils.math_functions import identity


class neuralNetwork(nn.Module, ABC):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes,
                 init_w=3e-3,
                 init_b=0.1,
                 batch_norm=False,
                 hidden_init=nn.init.xavier_uniform_,
                 hidden_activation=nn.ReLU(),
                 output_activation=identity
                 ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        input_size = self.input_size
        for idx, hidden_size in enumerate(hidden_sizes):
            layer = nn.Linear(input_size, hidden_size)
            norm_layer = nn.BatchNorm1d(hidden_size)
            hidden_init(layer.weight)
            layer.bias.data.fill_(init_b)
            self.layers.append(layer)
            self.norm_layers.append(norm_layer)
            input_size = hidden_size
        self.last_layer = nn.Linear(input_size, output_size)
        self.last_layer.weight.data.uniform_(-init_w, init_w)
        self.last_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, data):
        h = data
        h = self.apply_forward(h, self.layers, self.norm_layers, use_batch_norm=self.batch_norm)
        output = self.output_activation(self.last_layer(h))
        return output

    def apply_forward(self, data, hidden_layers, norm_layers, use_batch_norm=False):
        h = data
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


class neuralNetworkMultiInput(neuralNetwork, ABC):
    def forward(self, *inputs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs)




