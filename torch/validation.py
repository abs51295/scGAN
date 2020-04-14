"""Validation scripts for the scRNA data"""

import torch
import umap
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import MMD as mmd
import numpy as np


class Validator:

    def __init__(self, generator, cells_no, valid_dataloader, exp_folder, train_step, use_cuda=True):
        self.G = generator
        self.no = cells_no
        self.dataloader = valid_dataloader
        self.use_cuda = use_cuda
        self.exp_folder = exp_folder
        self.train_step = train_step
        if os.path.exists(os.path.join(exp_folder, 'mmd.npy')):
          self.mmd_loss = np.load(os.path.join(exp_folder, 'mmd.npy'))
        else:
          self.mmd_loss = np.array([])

    def generate_samples(self, number):
        latent_samples = self.G.sample_latent(number)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_cells = self.G(latent_samples)
        generated_cells = generated_cells.cpu().detach().numpy()
        generated_cells = generated_cells.reshape(
            (-1, generated_cells.shape[1]))
        # generated_cells = np.square(generated_cells) * float(20000)
        # print("Generated cells shape: " + str(generated_cells.shape))
        return generated_cells
    
    def read_valid_samples(self):
        cells, labels = next(iter(self.dataloader))
        cells = cells.cpu().detach().numpy()
        cells = cells.reshape((-1, cells.shape[1]))
        # cells = np.square(cells) * float(20000)
        # print("Valid cells shape: " + str(cells.shape))
        return cells

    def run_validation(self):
        valid_cells = self.read_valid_samples()
        generated_cells = self.generate_samples(len(valid_cells))
        self.generate_UMAP_image(
            valid_cells, generated_cells, self.exp_folder, self.train_step)
        self.generate_PCA_image(valid_cells, generated_cells, self.exp_folder, self.train_step)
        self.calculate_MMD(valid_cells, generated_cells)

    def generate_UMAP_image(self, valid_cells, fake_cells, exp_folder, train_step):
        """
        Generates and saves a UMAP plot with real and simulated cells

        Parameters
        ----------
        sess : Session
            The TF Session in use.
        cells_no : int
            Number of cells to use for the real and simulated cells (each) used
             for the plot.
        exp_folder : str
            Path to the job folder in which the outputs will be saved.
        train_step : int
            Index of the current training step.

        Returns
        -------

        """

        tnse_logdir = os.path.join(exp_folder, 'UMAP')
        if not os.path.isdir(tnse_logdir):
            os.makedirs(tnse_logdir)

        reducer = umap.UMAP(n_components=20, n_neighbors=20)

        embedded_cells = reducer.fit_transform(
            np.concatenate((valid_cells, fake_cells), axis=0))
        
        embedded_cells_real = embedded_cells[0:valid_cells.shape[0], :]
        embedded_cells_fake = embedded_cells[valid_cells.shape[0]:, :]

        plt.clf()
        plt.figure(figsize=(16, 12))

        plt.scatter(embedded_cells_real[:, 0], embedded_cells_real[:, 1],
                    c='blue',
                    marker='*',
                    label='real')

        plt.scatter(embedded_cells_fake[:, 0], embedded_cells_fake[:, 1],
                    c='red',
                    marker='o',
                    label='fake')

        plt.grid(True)
        plt.legend(loc='lower left', numpoints=1, ncol=2,
                   fontsize=8, bbox_to_anchor=(0, 0))
        plt.savefig(tnse_logdir + '/step_' + str(train_step) + '.jpg')
        plt.close()

    def generate_PCA_image(self, valid_cells, fake_cells, exp_folder, train_step):

        tnse_logdir = os.path.join(exp_folder, 'PCA')
        if not os.path.isdir(tnse_logdir):
            os.makedirs(tnse_logdir)

        reducer = PCA(n_components=5)

        embedded_cells = reducer.fit_transform(
            np.concatenate((valid_cells, fake_cells), axis=0))
        
        embedded_cells_real = embedded_cells[0:valid_cells.shape[0], :]
        embedded_cells_fake = embedded_cells[valid_cells.shape[0]:, :]

        plt.clf()
        plt.figure(figsize=(16, 12))

        plt.scatter(embedded_cells_real[:, 0], embedded_cells_real[:, 1],
                    c='blue',
                    marker='*',
                    label='real')

        plt.scatter(embedded_cells_fake[:, 0], embedded_cells_fake[:, 1],
                    c='red',
                    marker='o',
                    label='fake')

        plt.grid(True)
        plt.legend(loc='lower left', numpoints=1, ncol=2,
                   fontsize=8, bbox_to_anchor=(0, 0))
        plt.savefig(tnse_logdir + '/step_' + str(train_step) + '.jpg')
        plt.close()

    def calculate_MMD(self, valid_cells, fake_cells):
        sigma_list = np.logspace(-3, 2, 10)
        valid_cells = torch.from_numpy(valid_cells)
        fake_cells = torch.from_numpy(fake_cells)
        loss = mmd.mix_rbf_mmd2(valid_cells, fake_cells, sigma_list=sigma_list)
        loss = np.append(self.mmd_loss, loss)
        np.save(os.path.join(self.exp_folder, 'mmd.npy'), loss)
