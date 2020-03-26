"""Modified from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from validation import Validator


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                exp_dir, valid_cells, gp_weight=10, 
                critic_iterations=5, print_every=50, validation_every=1000, 
                use_cuda=True):
        self.G = generator
        self.G_opt = gen_optimizer
        # self.G_scheduler = gen_scheduler
        self.D = discriminator
        self.D_opt = dis_optimizer
        # self.D_scheduler = dis_scheduler
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.validation_every = validation_every
        self.exp_dir = exp_dir
        self.valid_cells = valid_cells

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        
        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        # data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.data)

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()
        # self.D_scheduler.step()

        # Record loss
        self.losses['D'].append(d_loss.data)

    def _generator_train_iteration(self, data):
        
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()
        # self.G_scheduler.step()

        # Record loss
        self.losses['G'].append(g_loss.data)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, train_data_loader, valid_data_loader):
        for i, data in enumerate(train_data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

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

    def sample_generator(self, num_samples):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data
