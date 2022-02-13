import numpy as np

import os

from srt import data


def get_dataset(mode, cfg, max_len=None, full_scale=False):
    ''' Returns a dataset.
    Args:
        mode: Dataset split, 'train', 'val', or 'test'
        cfg (dict): data config dictionary
        max_len (int): Limit to number of items in the dataset
        full_scale: Provide full images as targets, instead of a sampled set of pixels
    '''
    dataset_type = cfg['dataset']

    dataset_folder = f'data/{dataset_type}'
    if 'path' in cfg:
        dataset_folder = cfg['path']

    points_per_item = cfg['num_points'] if 'num_points' in cfg else 2048

    if 'kwargs' in cfg:
        kwargs = cfg['kwargs']
    else:
        kwargs = dict()

    # Create dataset
    if dataset_type == 'nmr':
        dataset = data.NMRDataset(dataset_folder, mode, points_per_item=points_per_item,
                                  max_len=max_len, full_scale=full_scale, **kwargs)
    elif dataset_type == 'msn':
        dataset = data.MultishapenetDataset(dataset_folder, mode, points_per_item=points_per_item,
                                            full_scale=full_scale, **kwargs)
    elif dataset_type == 'clevr3d':
        dataset = data.Clevr3dDataset(dataset_folder, mode, points_per_item=points_per_item,
                                      shapenet=False, max_len=max_len, full_scale=full_scale, **kwargs)
    elif dataset_type == 'obsurf_msn':
        dataset = data.Clevr3dDataset(dataset_folder, mode, points_per_item=points_per_item,
                                      shapenet=True, max_len=max_len, full_scale=full_scale, **kwargs)
    else:
        raise ValueError('Invalid dataset "{}"'.format(cfg['dataset']))

    return dataset


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

