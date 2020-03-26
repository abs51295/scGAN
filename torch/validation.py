"""Validation scripts for the scRNA data"""

import torch
import umap
import os
import numpy as np
import matplotlib.pyplot as plt


class Validator:

    def __init__(self, generator, cells_no, valid_dataloader, exp_folder, train_step, use_cuda=True):
        self.G = generator
        self.no = cells_no
        self.dataloader = valid_dataloader
        self.use_cuda = use_cuda
        self.exp_folder = exp_folder
        self.train_step = train_step

    def generate_samples(self):
        latent_samples = self.G.sample_latent(self.no)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_cells = self.G(latent_samples)
        generated_cells = generated_cells.cpu().detach().numpy()
        generated_cells = generated_cells.reshape(
            (-1, generated_cells.shape[1]))

        return generated_cells * float(20000)

    def read_valid_samples(self):
        cells = next(iter(self.dataloader))
        return cells * float(20000)

    def run_validation(self):
        generated_cells = self.generate_samples()
        valid_cells = self.read_valid_samples()
        self.generate_UMAP_image(
            valid_cells, generated_cells, self.exp_folder, self.train_step)

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

        reducer = umap.UMAP()

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
