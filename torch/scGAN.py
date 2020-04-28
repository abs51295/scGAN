import torch
import torch.nn as nn
import sys


def generator_layer(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.LeakyReLU()
    )


def discriminator_layer(input_size, output_size):
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(input_size, output_size)),
        nn.LeakyReLU()
    )


class Generator(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, num_classes, output_lsn=None):
        super(Generator, self).__init__()

        self.latent_dim = input_size
        self.output_lsn = output_lsn
        self.num_classes = num_classes

        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=input_size)

        self.layer_sizes = [input_size * 2] + [layer_size for _,
                                           layer_size in enumerate(hidden_layers)]

        hidden_layers = [generator_layer(input_s, output_s) for input_s, output_s in zip(
            self.layer_sizes, self.layer_sizes[1:])]

        self.hidden = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Sequential(nn.Linear(self.layer_sizes[-1], output_size), nn.LeakyReLU())

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat([x, self.embedding(y)], dim=1)
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

    def sample_latent(self, num_samples):
        return torch.randn(num_samples, self.latent_dim)


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_layers, num_classes, output_size):
        super(Discriminator, self).__init__()

        self.layer_sizes = [input_size] + [layer_size for _,
                                           layer_size in enumerate(hidden_layers)]

        hidden_layers = [discriminator_layer(input_s, output_s) for input_s, output_s in zip(
            self.layer_sizes, self.layer_sizes[1:])]

        self.hidden = nn.Sequential(*hidden_layers)

        self.aux_linear = nn.utils.spectral_norm(nn.Linear(self.layer_sizes[-1], num_classes))
        self.mi_linear = nn.utils.spectral_norm(nn.Linear(self.layer_sizes[-1], num_classes))

        self.num_classes = num_classes

        self.output_layer = nn.utils.spectral_norm(nn.Linear(self.layer_sizes[-1], output_size))

    def forward(self, x):
        h = self.hidden(x)
        c = self.aux_linear(h)
        mi = self.mi_linear(h)
        x = self.output_layer(h)
        return x.squeeze(1), c.squeeze(1), mi.squeeze(1)
