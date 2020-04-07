import torch
import torch.nn as nn
import sys


def generator_layer(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.BatchNorm1d(output_size),
        nn.ReLU()
    )


def discriminator_layer(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LayerNorm(output_size),
        nn.LeakyReLU()
    )


class Generator(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, output_lsn=None):
        super(Generator, self).__init__()

        self.latent_dim = input_size
        self.output_lsn = output_lsn
        self.layer_sizes = [input_size] + [layer_size for _,
                                           layer_size in enumerate(hidden_layers)]

        hidden_layers = [generator_layer(input_s, output_s) for input_s, output_s in zip(
            self.layer_sizes, self.layer_sizes[1:])]

        self.hidden = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Sequential(nn.Linear(self.layer_sizes[-1], output_size, False), nn.ReLU())

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

    def sample_latent(self, num_samples):
        return torch.rand(num_samples, self.latent_dim) * 2 - 1


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size):
        super(Discriminator, self).__init__()

        self.layer_sizes = [input_size] + [layer_size for _,
                                           layer_size in enumerate(hidden_layers)]

        hidden_layers = [discriminator_layer(input_s, output_s) for input_s, output_s in zip(
            self.layer_sizes, self.layer_sizes[1:])]

        self.hidden = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(self.layer_sizes[-1], output_size, False)

    def forward(self, x):
        x = self.hidden(x)
        output = self.output_layer(x)
        return output, x
