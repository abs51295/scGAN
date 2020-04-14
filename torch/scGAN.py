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

    def __init__(self, input_size, hidden_layers, output_size, output_lsn=None):
        super(Generator, self).__init__()

        self.latent_dim = input_size
        self.output_lsn = output_lsn
        self.layer_sizes = [input_size] + [layer_size for _,
                                           layer_size in enumerate(hidden_layers)]

        hidden_layers = [generator_layer(input_s, output_s) for input_s, output_s in zip(
            self.layer_sizes, self.layer_sizes[1:])]

        self.projection_layer = generator_layer(self.layer_sizes[0], self.layer_sizes[0])

        self.hidden = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Sequential(nn.Linear(self.layer_sizes[-1], output_size), nn.LeakyReLU())

    def forward(self, x):
        x = self.projection_layer(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        # if self.output_lsn:
        #     gammas_output = torch.ones(
        #         x.size(0), dtype=torch.float32, device=torch.device('cuda')) * self.output_lsn
        #     sigmas = torch.sum(x, 1)
        #     scale_ls = gammas_output / (sigmas + sys.float_info.epsilon)
        #     x = x * scale_ls[:, None]
        #     x = torch.sqrt(torch.add(x, sys.float_info.epsilon))
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

        self.projection_layer = discriminator_layer(self.layer_sizes[-1], self.layer_sizes[-1])

        self.num_classes = num_classes

        self.output_layer = nn.utils.spectral_norm(nn.Linear(self.layer_sizes[-1], output_size))

        self.embedding_layer = nn.utils.spectral_norm(nn.Embedding(num_classes, self.layer_sizes[-1]))

    def forward(self, x, y=None):
        h = self.hidden(x)
        h = self.projection_layer(h)
        x = self.output_layer(h)
        if y is not None:
          x += torch.sum(self.embedding_layer(y) * h, dim=1, keepdim=True)
        return x
