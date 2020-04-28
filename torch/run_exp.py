#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json
from scGAN import Generator, Discriminator
from training import Trainer
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler
import scanpy
import torch
import numpy as np

class GeneDataset(Dataset):
    """Filtered dataset as above"""

    def __init__(self, h5ad_file, transform=None):
        """
        Args:
            pickle_file (string): path to the fitered data pickle file
            root_dir (string): Directory with the pickle file
            transform: Transforms to be applied to the data
        """
        sc = scanpy.read_h5ad(filename=h5ad_file)
        self.gene_dataset = sc.to_df()
        self.gene_dataset['cluster'] = sc.obs['cluster']
        self.transform = transform

    def __len__(self):
        return len(self.gene_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gene = self.gene_dataset.iloc[idx]
        label = int(gene.pop('cluster'))
        gene = gene.to_numpy(dtype=np.float32)
        return torch.from_numpy(gene), torch.tensor(label, dtype=torch.int64)


def get_optimizer(model, optimizer, alpha_0, alpha_final, beta1, beta2):
    optim = Adam(params=model.parameters(), lr=alpha_0, betas=(0., 0.9))
    # scheduler = lr_scheduler.ExponentialLR(optimizer=optim, gamma=alpha_final / alpha_0)
    return optim


def run_exp(exp_gpu, mode='train', cells_no=None, save_cells_path=None):
    """
    Function that runs the experiment.
    It loads the json parameter file, instantiates the correct model, runs the
     training or the generation of the cells.

    Parameters
    ----------
    exp_gpu : tuple
        Tuple containing first the path (string) to the experiment folder
         and second a list of available GPU indexes.
    mode : string
        If "train" is passed, the training will be started, else, it will
        generate cells using the model whose checkpoint is in the experiment
         folder (in a job sub-folder).
    cells_no : int or list
        Number of cells to generate.
        Should be a list with number per cluster for a cscGAN model.
        Default is None.
    save_cells_path : str
        Path in which the simulated cells should be saved.
        Default is None.

    Returns
    -------

    """

    # # read the available GPU for training
    # avail_gpus = exp_gpu[1]
    # gpu_id = avail_gpus.pop(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

    # read the parameters
    exp_folder = exp_gpu[0]
    with open(os.path.join(exp_folder, 'parameters.json')) as fp:
        hparams = json.load(fp)

    # find training and validation TF records
    input_tfr = os.path.join(exp_folder, 'h5ad_records')

    # log directory
    log_dir = os.path.join(exp_folder, 'job')

    if save_cells_path is None:
        save_cells_path = os.path.join(exp_folder, 'generated_cells.h5ad')

    if hparams['model']['type'] == 'scGAN':

        G = Generator(
            input_size=hparams['model']['latent_dim'],
            hidden_layers=hparams['model']['gen_layers'],
            output_size=hparams['preprocessed']['genes_no'],
            output_lsn=hparams['model']['output_LSN']
        )

        D = Discriminator(
            input_size=hparams['preprocessed']['genes_no'],
            hidden_layers=hparams['model']['critic_layers'],
            num_classes = hparams['preprocessed']['clusters_no'],
            output_size=1
        )

        optimizer = hparams['training']['optimizer']['algorithm']
        beta1 = hparams['training']['optimizer']['beta1']
        beta2 = hparams['training']['optimizer']['beta2']
        alpha_0 = hparams['training']['learning_rate']['alpha_0']
        alpha_final = hparams['training']['learning_rate']['alpha_final']
        lambd = hparams['model']['lambd']
        critic_iter = hparams['training']['critic_iters']
        batch_size = hparams['training']['batch_size']
        valid_cells_no = hparams["preprocessed"]["valid_count"]
        progress_freq = hparams['training']['progress_freq']
        validation_freq = hparams['training']['validation_freq']
        valid_cells_no = hparams["preprocessed"]["valid_count"]
        max_steps=hparams['training']['max_steps']
        cluster_ratios = hparams['preprocessed']['cluster_ratios']

        G_optim = get_optimizer(
            G, optimizer, alpha_0, alpha_final, beta1, beta2)
        D_optim = get_optimizer(
            D, optimizer, alpha_0, alpha_final, beta1, beta2)

        trainer = Trainer(generator=G, discriminator=D, gen_optimizer=G_optim,
                          dis_optimizer=D_optim,
                          exp_dir=log_dir, valid_cells=valid_cells_no, print_every=progress_freq, 
                          validation_every=validation_freq,
                          cluster_ratios = cluster_ratios,
                          gp_weight=lambd, critic_iterations=critic_iter)

        train_dataset = GeneDataset(
            h5ad_file=os.path.join(input_tfr, 'train.h5ad'))
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = GeneDataset(
            h5ad_file=os.path.join(input_tfr, 'valid.h5ad'))
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=valid_cells_no, shuffle=True)

        trainer.train(train_dataloader, valid_dataloader, max_steps)
