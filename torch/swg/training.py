"""Modified from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from validation import Validator
import numpy as np
import os


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                exp_dir, num_projections, valid_cells, critic_iterations=5, print_every=50, validation_every=1000, 
                use_cuda=True):
        self.G = generator
        self.G_opt = gen_optimizer
        # self.G_scheduler = gen_scheduler
        self.D = discriminator
        self.D_loss = nn.BCEWithLogitsLoss()
        self.D_opt = dis_optimizer
        # self.D_scheduler = dis_scheduler
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.projections = num_projections
        self.num_steps = 0
        self.use_cuda = torch.cuda.is_available()
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.validation_every = validation_every
        self.exp_dir = exp_dir
        self.valid_cells = valid_cells

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def wasserstein1d(self, x, y):
        x1, _ = torch.sort(x, dim=0)
        y1, _ = torch.sort(y, dim=0)
        z = (x1-y1).view(-1)
        return (z ** 2).mean()


    def _critic_train_iteration(self, data):
        
        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        # data = Variable(data)
        if self.use_cuda:
            data = data.cuda()

        # noise = torch.normal(mean=0, std=0.1, size=data.size(), device=torch.device('cuda'))

        # data += noise
        # generated_data += noise


        d_real, _ = self.D(data)
        d_generated, _ = self.D(generated_data)

        dloss_real = self.D_loss(d_real, torch.ones_like(d_real))
        dloss_fake = self.D_loss(d_generated, torch.zeros_like(d_generated))


        d_loss = dloss_real.mean() + dloss_fake.mean() 
        d_loss.backward()

        self.D_opt.step()
        # self.D_scheduler.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        if self.use_cuda:
            data = data.cuda()

        generated_data = self.sample_generator(batch_size)

        # Generate projections
        theta = torch.randn(self.D.layer_sizes[-1], self.projections)
        if self.use_cuda:
            theta = theta.cuda()

        theta = theta / torch.norm(theta, dim=0)[None, :]

        _, fake_features = self.D(generated_data)
        projected_distribution = fake_features @ theta

        _, true_features = self.D(data)
        true_distribution = true_features.detach() @ theta

        g_loss = self.wasserstein1d(projected_distribution, true_distribution)
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _train_epoch(self, train_data_loader, valid_data_loader):
        for i, data in enumerate(train_data_loader):
            self.num_steps += 1
            self._generator_train_iteration(data)
            self._critic_train_iteration(data)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

            if self.num_steps % self.validation_every == 0:
                print("Validation started")
                self.G.eval()
                validator = Validator(self.G, self.valid_cells, valid_data_loader, self.exp_dir, self.num_steps, self.use_cuda)
                validator.run_validation()
                self.G.train()


    def train(self, train_data_loader, valid_data_loader, epochs):
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(train_data_loader, valid_data_loader)

        print("Saving the models")
        torch.save(self.G.state_dict(), os.path.join(self.exp_dir, 'generator.pth'))
        torch.save(self.D.state_dict(), os.path.join(self.exp_dir, 'discriminator.pth'))
        np.save(os.path.join(self.exp_dir, 'G.npy'), np.array(self.losses['G']))
        np.save(os.path.join(self.exp_dir, 'D.npy'), np.array(self.losses['D']))

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data
