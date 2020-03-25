#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Functions that convert the expression data to h5ad.
"""
import multiprocessing as mp
import os

from preprocessing.process_raw import GeneMatrix


def read_and_serialize(job_path):
    """
    Reads a raw file, and runs all the pipeline until creation of the TFRecords.

    Parameters
    ----------
    job_path : str
        Path to the job directory.

    Returns
    -------
    A string to be printed in the main process
    """

    sc_data = GeneMatrix(job_path)

    sc_data.apply_preprocessing()

    worker_path = join(job_path, 'h5ad_records')
    os.makedirs(worker_path, exist_ok=True)

    train_data = sc_data.sc_raw[
            sc_data.sc_raw.obs['split'] == 'train']
    
    valid_data = sc_data.sc_raw[
      sc_data.sc_raw.obs['split'] == 'valid'
    ]

    train_data.write_h5ad(filename=os.path.join(worker_path, 'train.h5ad'))
    valid_data.write_h5ad(filename=os.path.join(worker_path, 'valid.h5ad'))

    # print("No of observations with zero are: " + str(len(temp)))
    return 'done with writing for: ' + job_path


def process_files(exp_folders):
    """
    Parallel pre-processing of the different experiments.

    Parameters
    ----------
    exp_folders : list of strings
        The path to the folder containing all the experiments.

    Returns
    -------

    """
    pool = mp.Pool()
    results = pool.imap_unordered(read_and_serialize, exp_folders)

    stat = []
    for res in results:
        print(res)
        stat.append(res)

    pool.close()
    pool.join()
